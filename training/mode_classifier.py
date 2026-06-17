#!/usr/bin/env python3
"""
mode_classifier.py - storm MODE (supercell vs QLCS) from low-level reflectivity
morphology, measured on the FULL sweep around each storm (not the tight crop, so
a squall line reads as linear instead of being clipped).

Per event: open the raw V06 volume, grid the lowest-tilt reflectivity to
Cartesian over an R_km neighborhood, take the connected >=THRESH region
containing the storm point, and measure its shape via PCA:
  elong     = sqrt(lambda1/lambda2)   axis ratio (high = linear)
  major_len = ~full extent of the major axis, km
  area      = component area, km^2

  QLCS      if elong >= --q-elong and major_len >= --q-major
  supercell if elong <  --s-elong
  ambiguous otherwise   -> downstream runs BOTH experts

Writes per-event morphology to CSV and prints the separation, split by label and
tor_f_scale (sanity: EF2+ tornadoes should skew supercell).
"""
import argparse, csv, glob, math, os, sys, collections
import numpy as np
from scipy import ndimage


def build_vol_index(root):
    idx = {}
    for p in glob.iglob(os.path.join(root, "**", "*_V0*"), recursive=True):
        b = os.path.basename(p)
        if not b.endswith("_MDM"):
            idx[b] = p
    return idx


def lowest_ref_sweep(radar):
    best, bel = None, 999.0
    for s in range(radar.nsweeps):
        try:
            e = float(np.mean(radar.get_elevation(s)))
            d = radar.get_field(s, "reflectivity")
            if np.isfinite(d).any() and e < bel:
                bel, best = e, s
        except Exception:
            continue
    return best if best is not None else 0


def morphology(radar, sx, sy, R_km=80.0, res_km=1.0, thresh=40.0):
    """sx, sy = storm position in km relative to radar. Returns dict or None."""
    sw = lowest_ref_sweep(radar)
    x, y, _ = radar.get_gate_x_y_z(sw)
    x = x / 1000.0; y = y / 1000.0
    ref = np.ma.filled(radar.get_field(sw, "reflectivity"), np.nan)
    sel = (np.abs(x - sx) <= R_km) & (np.abs(y - sy) <= R_km) & np.isfinite(ref) & (ref >= thresh)
    if sel.sum() < 5:
        return None
    gx, gy = x[sel], y[sel]
    nb = int(2 * R_km / res_km)
    xi = ((gx - (sx - R_km)) / res_km).astype(int).clip(0, nb - 1)
    yi = ((gy - (sy - R_km)) / res_km).astype(int).clip(0, nb - 1)
    grid = np.zeros((nb, nb), bool)
    grid[yi, xi] = True
    grid = ndimage.binary_closing(grid, iterations=1)  # bridge small gaps in a line
    lab, n = ndimage.label(grid)
    if n == 0:
        return None
    cxi = int(R_km / res_km); cyi = int(R_km / res_km)  # storm sits at neighborhood center
    comp = lab[cyi, cxi]
    if comp == 0:  # storm cell below thresh -> nearest component
        ys, xs = np.where(grid)
        d = (xs - cxi) ** 2 + (ys - cyi) ** 2
        comp = lab[ys[d.argmin()], xs[d.argmin()]]
    ys, xs = np.where(lab == comp)
    if len(xs) < 5:
        return None
    pts = np.column_stack([xs * res_km, ys * res_km]).astype(float)
    pts -= pts.mean(0)
    ev = np.linalg.eigvalsh(np.cov(pts.T))      # ascending [l2, l1]
    l2, l1 = max(ev[0], 1e-6), max(ev[1], 1e-6)
    return {
        "elong": float(np.sqrt(l1 / l2)),
        "major_len": float(4 * np.sqrt(l1)),
        "area": float(len(xs) * res_km * res_km),
    }


def classify(m, q_elong, q_major, s_elong):
    if m is None:
        return "nodata"
    if m["elong"] >= q_elong and m["major_len"] >= q_major:
        return "qlcs"
    if m["elong"] < s_elong:
        return "supercell"
    return "ambiguous"


def storm_xy_from_npz(d):
    try:
        az = math.radians(float(d["storm_az_deg"])); rng = float(d["storm_range_m"]) / 1000.0
        return rng * math.sin(az), rng * math.cos(az)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="dataset_manifest.csv")
    ap.add_argument("--vol-root", default="radar_volumes")
    ap.add_argument("--out", default="mode_morphology.csv")
    ap.add_argument("--R-km", type=float, default=80.0)
    ap.add_argument("--thresh", type=float, default=40.0)
    ap.add_argument("--q-elong", type=float, default=3.0)
    ap.add_argument("--q-major", type=float, default=60.0)
    ap.add_argument("--s-elong", type=float, default=2.0)
    ap.add_argument("--limit-events", type=int, default=0, help="quick look on N events")
    ap.add_argument("--split", default=None, help="restrict to a split (train/val)")
    args = ap.parse_args()

    import pyart

    # group manifest rows by event, pick the middle scan as representative
    by_event = collections.OrderedDict()
    for r in csv.DictReader(open(args.manifest)):
        if args.split and r["split"] != args.split:
            continue
        by_event.setdefault(r["event_id"], []).append(r)
    events = list(by_event.items())
    if args.limit_events:
        events = events[:args.limit_events]
    print(f"{len(events)} events", file=sys.stderr)

    print("indexing volumes...", file=sys.stderr)
    vidx = build_vol_index(args.vol_root)

    fh = open(args.out, "w")
    fh.write("event_id,label,tor_f_scale,station,elong,major_len,area,mode\n")
    rows_out = []
    done = 0
    for eid, rows in events:
        rows = sorted(rows, key=lambda r: r["path"])
        r = rows[len(rows) // 2]                       # middle scan
        base = os.path.basename(r["path"])[:-4]        # strip .npz -> volume name
        vpath = vidx.get(base)
        if not vpath:
            continue
        try:
            d = np.load(r["path"])
            sxy = storm_xy_from_npz(d)
            if sxy is None:
                continue
            radar = pyart.io.read_nexrad_archive(vpath)
            m = morphology(radar, sxy[0], sxy[1], args.R_km, 1.0, args.thresh)
            mode = classify(m, args.q_elong, args.q_major, args.s_elong)
            tfs = str(d["tor_f_scale"]) if "tor_f_scale" in d.files else ""
            lab = int(r["label"])
            if m:
                fh.write(f"{eid},{lab},{tfs},{r['station']},{m['elong']:.2f},"
                         f"{m['major_len']:.1f},{m['area']:.0f},{mode}\n")
                rows_out.append((lab, tfs, m["elong"], m["major_len"], mode))
            done += 1
            if done % 200 == 0:
                print(f"  {done}", file=sys.stderr); fh.flush()
        except Exception as e:
            print(f"  skip {base}: {type(e).__name__}", file=sys.stderr)
    fh.close()

    if not rows_out:
        print("no events processed"); return
    el = np.array([x[2] for x in rows_out]); ml = np.array([x[3] for x in rows_out])
    modes = [x[4] for x in rows_out]; labs = np.array([x[0] for x in rows_out])
    print("\n" + "=" * 60)
    print(f"MODE MORPHOLOGY  ({len(rows_out)} events)")
    print("=" * 60)
    print(f"  elong     p25/50/75: {np.percentile(el,[25,50,75]).round(2)}")
    print(f"  major_len p25/50/75: {np.percentile(ml,[25,50,75]).round(0)} km")
    cnt = collections.Counter(modes)
    for k in ("supercell", "qlcs", "ambiguous"):
        print(f"  {k:10}: {cnt.get(k,0)}  ({100*cnt.get(k,0)/len(modes):.0f}%)")
    print("  --- by class (sanity: tornadic EF2+ should skew supercell) ---")
    for lv, name in ((1, "tornadic"), (0, "non-torn")):
        sub = [m for l, m in zip(labs, modes) if l == lv]
        c = collections.Counter(sub)
        tot = max(len(sub), 1)
        print(f"  {name:9}: super {100*c.get('supercell',0)/tot:.0f}%  "
              f"qlcs {100*c.get('qlcs',0)/tot:.0f}%  amb {100*c.get('ambiguous',0)/tot:.0f}%")
    print("=" * 60)
    print(f"  full per-event metrics -> {args.out}")
    print("  set thresholds from the elong/major_len distribution, then spot-check")
    print("  a few qlcs/supercell event_ids against events you know.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
