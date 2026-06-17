#!/usr/bin/env python3
"""v8/mrms/mrms_extract.py - build the MRMS training set (RESUMABLE).
Same output (npz key 'x'=(3,64,64); manifest path,label,event_id,dt,lat,lon,
conv_day,split,ref_max,az02_max,az36_max). Resume (skip existing _pos.npz),
disk listing cache, chunked memory, manifest rebuilt from disk."""
import argparse, csv, gzip, hashlib, os, sys, tempfile, random, re, glob, pickle
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from scipy.interpolate import RegularGridInterpolator

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from common import paths
from mrms_read import BUCKET, PRODUCTS

KEYRE = re.compile(r"(\d{8})-(\d{6})")


def make_client(workers):
    import boto3
    from botocore import UNSIGNED
    from botocore.client import Config
    return boto3.client("s3", config=Config(
        signature_version=UNSIGNED, max_pool_connections=max(24, workers + 8),
        connect_timeout=10, read_timeout=60,
        retries={"max_attempts": 5, "mode": "standard"}))


def list_day(cli, product, day):
    prefix = f"CONUS/{product}/{day:%Y%m%d}/"
    keys = []
    for page in cli.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=prefix):
        for o in page.get("Contents", []):
            keys.append(o["Key"])
    return keys


def build_or_load_cache(cli, products, days, cache_path):
    need = {(p, d.strftime("%Y%m%d")) for p in products for d in days}
    if os.path.exists(cache_path):
        cache = pickle.load(open(cache_path, "rb"))
        if need.issubset(cache):
            print(f"loaded listing cache from disk ({len(cache)} dirs) -- no S3 listing",
                  file=sys.stderr)
            return cache
    cache = {}; total = len(products) * len(days); done = 0
    for product in products:
        for d in sorted(days):
            cache[(product, d.strftime("%Y%m%d"))] = list_day(cli, product, datetime(d.year, d.month, d.day))
            done += 1
            if done % 50 == 0:
                print(f"  listed {done}/{total} (product,day) dirs", file=sys.stderr)
    pickle.dump(cache, open(cache_path, "wb"))
    print(f"saved listing cache -> {cache_path}", file=sys.stderr)
    return cache


def nearest_key_cached(cache, product, dt):
    days = {dt.date()}
    if dt.hour == 0:
        days.add((dt - timedelta(days=1)).date())
    if dt.hour == 23:
        days.add((dt + timedelta(days=1)).date())
    keys = []
    for d in days:
        keys += cache.get((product, d.strftime("%Y%m%d")), [])
    best, bd = None, 1e18
    for k in keys:
        m = KEYRE.search(k)
        if not m:
            continue
        t = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        dd = abs((t - dt).total_seconds())
        if dd < bd:
            bd, best = dd, k
    return (best, bd) if best else (None, None)


def read_full(cli, cache, product, dt):
    import pygrib
    key, gap = nearest_key_cached(cache, product, dt)
    if key is None:
        return None
    data = gzip.decompress(cli.get_object(Bucket=BUCKET, Key=key)["Body"].read())
    path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as f:
            f.write(data); path = f.name
        grbs = pygrib.open(path); g = grbs[1]
        vals = np.array(g.values, float)
        try:
            lat1 = np.array(g.distinctLatitudes, float)
            lon1 = np.array(g.distinctLongitudes, float)
        except Exception:
            lats, lons = g.latlons()
            lat1 = lats[:, 0].astype(float); lon1 = lons[0, :].astype(float)
        grbs.close()
    finally:
        if path and os.path.exists(path):
            os.unlink(path)
    lon1 = np.where(lon1 > 180, lon1 - 360, lon1)
    if lat1[0] > lat1[-1]:
        lat1 = lat1[::-1]; vals = vals[::-1, :]
    return {"lat": lat1, "lon": lon1, "vals": vals}


def interp_of(full):
    return RegularGridInterpolator((full["lat"], full["lon"]), full["vals"],
                                   method="nearest", bounds_error=False, fill_value=-999.0)


def box_from(itp, clat, clon, half_deg, H, W):
    tlat = np.linspace(clat - half_deg, clat + half_deg, H)
    tlon = np.linspace(clon - half_deg, clon + half_deg, W)
    gy, gx = np.meshgrid(tlat, tlon, indexing="ij")
    return itp(np.column_stack([gy.ravel(), gx.ravel()])).reshape(H, W).astype(np.float32)


def boxmax(a):
    v = a[a > -900]
    return float(v.max()) if v.size else float("nan")


def mine_negatives(ref_full, az_itp, clat, clon, region_deg, min_dbz, min_az,
                   tor_pts, excl_km, k, rng):
    la, lo = ref_full["lat"], ref_full["lon"]
    ia = np.where((la >= clat - region_deg) & (la <= clat + region_deg))[0]
    io = np.where((lo >= clon - region_deg) & (lo <= clon + region_deg))[0]
    if len(ia) < 2 or len(io) < 2:
        return []
    sub = ref_full["vals"][np.ix_(ia, io)]
    yy, xx = np.where(sub >= min_dbz)
    if len(yy) == 0:
        return []
    cand = list(zip(la[ia[yy]], lo[io[xx]]))
    rng.shuffle(cand)
    KM = 111.0; chosen = []

    def far(plat, plon, pts):
        for qlat, qlon in pts:
            dx = (plon - qlon) * np.cos(np.radians(plat)) * KM
            dy = (plat - qlat) * KM
            if dx * dx + dy * dy < excl_km * excl_km:
                return False
        return True

    for plat, plon in cand:
        if len(chosen) >= k:
            break
        if not far(plat, plon, tor_pts) or not far(plat, plon, chosen):
            continue
        if min_az > 0 and float(az_itp([[plat, plon]])[0]) < min_az:
            continue
        chosen.append((plat, plon))
    return chosen


def split_of(conv_day, val_frac=0.2):
    h = int(hashlib.md5(conv_day.encode()).hexdigest(), 16)
    return "val" if (h % 100) < int(val_frac * 100) else "train"


def roc_auc(scores, labels):
    s = np.asarray(scores, float); y = np.asarray(labels, int)
    k = np.isfinite(s); s, y = s[k], y[k]
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    cs = np.cumsum(cnt); ranks = ((cs - cnt + cs + 1) / 2.0)[inv]
    n1 = int(y.sum()); n0 = len(y) - n1
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def ptime(s):
    s = s.strip().replace("T", " ").replace("Z", "")
    for f in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(s, f)
        except Exception:
            continue
    return None


def process_event(e, cli, cache, args):
    pos_fp = os.path.join(args.out_dir, f"{e['event_id']}_pos.npz")
    if e["event_id"] and os.path.exists(pos_fp):
        return ("skip_exists", 0)
    ref = read_full(cli, cache, PRODUCTS["ref"], e["dt"])
    a02 = read_full(cli, cache, PRODUCTS["az02"], e["dt"])
    a36 = read_full(cli, cache, PRODUCTS["az36"], e["dt"])
    if ref is None or a02 is None or a36 is None:
        return ("skip_nodata", 0)
    iref, i02, i36 = interp_of(ref), interp_of(a02), interp_of(a36)
    rng = random.Random((hash(str(e["event_id"])) ^ args.seed) & 0xFFFFFFFF)
    n = 0

    def emit(lat, lon, label, tag):
        nonlocal n
        bx = np.stack([box_from(iref, lat, lon, args.half_deg, args.H, args.H),
                       box_from(i02, lat, lon, args.half_deg, args.H, args.H),
                       box_from(i36, lat, lon, args.half_deg, args.H, args.H)])
        fp = os.path.join(args.out_dir, f"{e['event_id'] or tag}_{tag}.npz")
        np.savez(fp, x=bx, label=label, lat=lat, lon=lon,
                 dt=e["dt"].isoformat(), event_id=str(e["event_id"]), conv_day=e["day"])
        n += 1

    emit(e["lat"], e["lon"], 1, "pos")
    for j, (nlat, nlon) in enumerate(
            mine_negatives(ref, i02, e["lat"], e["lon"], args.region_deg,
                           args.min_dbz, args.min_az, args.tor_by_day[e["day"]],
                           args.excl_km, args.neg_per_pos, rng)):
        emit(nlat, nlon, 0, f"neg{j}")
    del ref, a02, a36, iref, i02, i36
    return ("done", n)


def rebuild_manifest(out_dir, manifest):
    files = sorted(glob.glob(os.path.join(out_dir, "*.npz")))
    fh = open(manifest, "w")
    fh.write("path,label,event_id,dt,lat,lon,conv_day,split,ref_max,az02_max,az36_max\n")
    coll = []
    for fp in files:
        try:
            d = np.load(fp, allow_pickle=True)
            x = d["x"]; label = int(d["label"])
            lat = float(d["lat"]); lon = float(d["lon"])
            dt = str(d["dt"]); eid = str(d["event_id"]); cday = str(d["conv_day"])
            rmax, a2max, a3max = boxmax(x[0]), boxmax(x[1]), boxmax(x[2])
            spl = split_of(cday)
            fh.write(f"{fp},{label},{eid},{dt},{lat:.4f},{lon:.4f},{cday},{spl},"
                     f"{rmax:.1f},{a2max:.2f},{a3max:.2f}\n")
            coll.append((label, a2max))
        except Exception as ex:
            print(f"  manifest skip {os.path.basename(fp)}: {type(ex).__name__}", file=sys.stderr)
    fh.close()
    return coll


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=os.path.join(paths.TEST, "tornado_training_index.csv"))
    ap.add_argument("--out-dir", default=os.path.join(paths.DATA, "mrms_samples"))
    ap.add_argument("--manifest", default=os.path.join(paths.DATA, "mrms_manifest.csv"))
    ap.add_argument("--cache", default=os.path.join(paths.DATA, "mrms_listing_cache.pkl"))
    ap.add_argument("--since", default="2020-10-01")
    ap.add_argument("--ef-min", type=int, default=0)
    ap.add_argument("--half-deg", type=float, default=0.25)
    ap.add_argument("--H", type=int, default=64)
    ap.add_argument("--neg-per-pos", type=int, default=2)
    ap.add_argument("--min-dbz", type=float, default=45.0)
    ap.add_argument("--min-az", type=float, default=0.0)
    ap.add_argument("--region-deg", type=float, default=2.5)
    ap.add_argument("--excl-km", type=float, default=50.0)
    ap.add_argument("--limit-events", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.manifest), exist_ok=True)
    since = datetime.strptime(args.since, "%Y-%m-%d")

    pos = []
    for r in csv.DictReader(open(args.index)):
        t = ptime(r.get("begin_datetime_utc", ""))
        if not t or t < since:
            continue
        ef = r.get("ef_scale", ""); efn = next((int(c) for c in ef if c.isdigit()), 0)
        if efn < args.ef_min:
            continue
        try:
            lat = float(r["begin_lat"]); lon = float(r["begin_lon"])
        except Exception:
            continue
        pos.append({"event_id": r.get("event_id", ""), "dt": t, "lat": lat, "lon": lon,
                    "ef": efn, "day": t.strftime("%Y-%m-%d")})
    pos.sort(key=lambda e: e["dt"])
    if args.limit_events:
        pos = pos[:args.limit_events]
    print(f"{len(pos)} positive events since {args.since}", file=sys.stderr)

    args.tor_by_day = {}
    for e in pos:
        args.tor_by_day.setdefault(e["day"], []).append((e["lat"], e["lon"]))

    cli = make_client(args.workers)
    days_needed = set()
    for e in pos:
        days_needed.add(e["dt"].date())
        if e["dt"].hour == 0:
            days_needed.add((e["dt"] - timedelta(days=1)).date())
        if e["dt"].hour == 23:
            days_needed.add((e["dt"] + timedelta(days=1)).date())
    cache = build_or_load_cache(cli, list(PRODUCTS.values()), days_needed, args.cache)

    already = sum(1 for e in pos if e["event_id"] and
                  os.path.exists(os.path.join(args.out_dir, f"{e['event_id']}_pos.npz")))
    print(f"resume: {already}/{len(pos)} events already on disk, doing the rest", file=sys.stderr)

    done = exists = nodata = 0
    for i in range(0, len(pos), args.chunk):
        batch = pos[i:i + args.chunk]
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(process_event, e, cli, cache, args): e for e in batch}
            for fut in as_completed(futs):
                e = futs[fut]
                try:
                    status, _ = fut.result()
                except Exception as ex2:
                    print(f"  skip {e['event_id']}: {type(ex2).__name__}: {ex2}", file=sys.stderr)
                    nodata += 1; continue
                if status == "done":
                    done += 1
                elif status == "skip_exists":
                    exists += 1
                else:
                    nodata += 1
        print(f"  chunk {i//args.chunk + 1}: done={done} resumed={exists} nodata/err={nodata} "
              f"({i + len(batch)}/{len(pos)})", file=sys.stderr)

    print("rebuilding manifest from disk...", file=sys.stderr)
    coll = rebuild_manifest(args.out_dir, args.manifest)
    if not coll:
        print("no samples on disk"); return
    labs = np.array([c[0] for c in coll]); az = np.array([c[1] for c in coll])
    p = az[labs == 1]; nq = az[labs == 0]
    print("\n" + "=" * 60)
    print(f"MRMS EXTRACT  ({len(coll)} samples: {int(labs.sum())} pos, {int((labs==0).sum())} neg)")
    print(f"  this run: done={done} resumed={exists} nodata/err={nodata}")
    print("=" * 60)
    print(f"  az02_max  tornadic median: {np.nanmedian(p):.2f}")
    print(f"  az02_max  non-torn median: {np.nanmedian(nq):.2f}")
    print(f"  SEPARATION AUC (az02_max, pos vs neg): {roc_auc(az, labs):.3f}")
    print(f"  manifest -> {args.manifest}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
