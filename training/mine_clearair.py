#!/usr/bin/env python3
"""v8/mrms/mine_clearair.py -- add clear-air + non-rotating-precip negatives.

Reuses mrms_extract's grid-load / interp / box / npz / manifest code VERBATIM,
so the new negatives share the EXACT convention of the storm samples (no new
extraction artifact). Mines from the SAME scans as the positives -> clear air on
real severe-weather days, >=excl_km from any tornado report that day.

Two negative subtypes (both label 0):
  clear-air      box ref_max < --clear-hi          (incl. totally-empty boxes)
  precip-no-rot  --precip-lo <= ref_max <= --precip-hi  AND  az02_max < --az-hi

After mining it rebuilds the manifest from disk (mrms_extract.rebuild_manifest),
so the new npz get conv_day-based leak-safe splits automatically.

Run:
  /workspace/test/venv/bin/python mrms/mine_clearair.py \
      --target-clear 800 --target-precip 500
"""
import argparse, os, sys, csv, random
from datetime import datetime
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from common import paths
import mrms_extract as ex


def ptime(s):
    s = str(s).strip().replace("T", " ").replace("Z", "")
    for f in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(s, f)
        except Exception:
            continue
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(paths.DATA, "mrms_manifest.csv"))
    ap.add_argument("--out-dir", default=os.path.join(paths.DATA, "mrms_samples"))
    ap.add_argument("--cache", default=os.path.join(paths.DATA, "mrms_listing_cache.pkl"))
    ap.add_argument("--half-deg", type=float, default=0.25)
    ap.add_argument("--H", type=int, default=64)
    ap.add_argument("--region-deg", type=float, default=3.5)
    ap.add_argument("--excl-km", type=float, default=60.0)
    ap.add_argument("--clear-hi", type=float, default=25.0)
    ap.add_argument("--precip-lo", type=float, default=30.0)
    ap.add_argument("--precip-hi", type=float, default=60.0)
    ap.add_argument("--az-hi", type=float, default=4.0)
    ap.add_argument("--candidates-per-scan", type=int, default=80)
    ap.add_argument("--clear-per-scan", type=int, default=8)
    ap.add_argument("--precip-per-scan", type=int, default=4)
    ap.add_argument("--target-clear", type=int, default=800)
    ap.add_argument("--target-precip", type=int, default=500)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    KM = 111.0

    rows = list(csv.DictReader(open(args.manifest)))
    pos = [r for r in rows if r["label"] == "1"]
    by_day = {}
    for r in pos:
        by_day.setdefault(r["conv_day"], []).append(r)
    tor_by_day = {d: [(float(r["lat"]), float(r["lon"])) for r in rs] for d, rs in by_day.items()}
    days = list(by_day.keys()); rng.shuffle(days)
    print(f"{len(days)} conv_days available to mine from", file=sys.stderr)

    cli = ex.make_client(args.workers)
    day_dates = {datetime.strptime(d, "%Y-%m-%d").date() for d in days}
    cache = ex.build_or_load_cache(cli, list(ex.PRODUCTS.values()), day_dates, args.cache)

    def far(plat, plon, pts):
        for qlat, qlon in pts:
            dx = (plon - qlon) * np.cos(np.radians(plat)) * KM
            dy = (plat - qlat) * KM
            if dx * dx + dy * dy < args.excl_km ** 2:
                return False
        return True

    n_clear = n_precip = 0
    for d in days:
        if n_clear >= args.target_clear and n_precip >= args.target_precip:
            break
        anchor = rng.choice(by_day[d])
        dt = ptime(anchor["dt"]); alat = float(anchor["lat"]); alon = float(anchor["lon"])
        ref = ex.read_full(cli, cache, ex.PRODUCTS["ref"], dt)
        a02 = ex.read_full(cli, cache, ex.PRODUCTS["az02"], dt)
        a36 = ex.read_full(cli, cache, ex.PRODUCTS["az36"], dt)
        if ref is None or a02 is None or a36 is None:
            continue
        iref, i02, i36 = ex.interp_of(ref), ex.interp_of(a02), ex.interp_of(a36)
        la, lo = ref["lat"], ref["lon"]
        la_min = max(alat - args.region_deg, la.min()); la_max = min(alat + args.region_deg, la.max())
        lo_min = max(alon - args.region_deg, lo.min()); lo_max = min(alon + args.region_deg, lo.max())
        tor = tor_by_day[d]
        got_c = got_p = 0
        for _ in range(args.candidates_per_scan):
            if got_c >= args.clear_per_scan and got_p >= args.precip_per_scan:
                break
            plat = rng.uniform(la_min, la_max); plon = rng.uniform(lo_min, lo_max)
            if not far(plat, plon, tor):
                continue
            rbox = ex.box_from(iref, plat, plon, args.half_deg, args.H, args.H)
            rmax = ex.boxmax(rbox)
            if not np.isfinite(rmax):
                rmax = -999.0
            tag = None
            if rmax < args.clear_hi and got_c < args.clear_per_scan and n_clear < args.target_clear:
                tag = "clr"
            elif args.precip_lo <= rmax <= args.precip_hi and got_p < args.precip_per_scan and n_precip < args.target_precip:
                a2box = ex.box_from(i02, plat, plon, args.half_deg, args.H, args.H)
                if ex.boxmax(a2box) < args.az_hi:
                    tag = "prc"
            if tag is None:
                continue
            bx = np.stack([rbox,
                           ex.box_from(i02, plat, plon, args.half_deg, args.H, args.H),
                           ex.box_from(i36, plat, plon, args.half_deg, args.H, args.H)])
            idx = n_clear if tag == "clr" else n_precip
            eid = f"{tag}{idx:05d}"
            fp = os.path.join(args.out_dir, f"{eid}_{tag}.npz")
            np.savez(fp, x=bx, label=0, lat=plat, lon=plon,
                     dt=dt.isoformat(), event_id=eid, conv_day=d)
            if tag == "clr":
                n_clear += 1; got_c += 1
            else:
                n_precip += 1; got_p += 1
        del ref, a02, a36, iref, i02, i36
        print(f"  {d}: +{got_c} clear +{got_p} precip  (totals {n_clear}/{args.target_clear}, "
              f"{n_precip}/{args.target_precip})", file=sys.stderr)

    print(f"\nmined {n_clear} clear-air + {n_precip} precip-no-rot negatives", file=sys.stderr)
    print("rebuilding manifest (folds new npz in with conv_day splits)...", file=sys.stderr)
    coll = ex.rebuild_manifest(args.out_dir, args.manifest)
    labs = np.array([c[0] for c in coll])
    print(f"manifest now: {len(coll)} samples, {int(labs.sum())} pos, {int((labs==0).sum())} neg")


if __name__ == "__main__":
    main()
