#!/usr/bin/env python3
"""v8/mrms/mrms_sweep.py - build MRMS boxes for ONE fixed point across several
times, score each with the v8 model, render the progression. Shows the empty-box
false alarm: early time (storm not arrived, near-empty box) vs storm centered."""
import argparse, os, sys
from datetime import datetime, timedelta
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from mrms_extract import make_client, read_full, interp_of, box_from, list_day, PRODUCTS
from mrms_predict import load_model, score


def masked(a):
    return np.ma.masked_less(np.asarray(a, float), -900)


def boxmax(a):
    v = np.asarray(a, float); v = v[v > -900]
    return float(v.max()) if v.size else float("nan")


def build_box(cli, cache, dt, lat, lon, half_deg=0.25, H=64):
    chans = []
    for k in ("ref", "az02", "az36"):
        full = read_full(cli, cache, PRODUCTS[k], dt)
        if full is None:
            return None
        chans.append(box_from(interp_of(full), lat, lon, half_deg, H, H))
    return np.stack(chans)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--date", required=True, help="YYYY-MM-DD (UTC)")
    ap.add_argument("--times", required=True, help="comma list, e.g. 20:00,20:10,20:30")
    ap.add_argument("--ckpt", default="data/mrms_v8.pt")
    ap.add_argument("--out", default="sweep.png")
    ap.add_argument("--save-dir", default=None, help="also write each box as npz")
    args = ap.parse_args()

    d0 = datetime.strptime(args.date, "%Y-%m-%d")
    dts = []
    for tt in args.times.split(","):
        hh, mm = tt.strip().split(":")
        dts.append(d0.replace(hour=int(hh), minute=int(mm)))

    cli = make_client(4)
    days = set()
    for dt in dts:
        days.add(dt.date())
        if dt.hour == 0:
            days.add((dt - timedelta(days=1)).date())
        if dt.hour == 23:
            days.add((dt + timedelta(days=1)).date())
    cache = {}
    for prod in PRODUCTS.values():
        for dd in days:
            cache[(prod, dd.strftime("%Y%m%d"))] = list_day(cli, prod, datetime(dd.year, dd.month, dd.day))

    model, mean, std = load_model(args.ckpt)
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    rows = []
    for dt in dts:
        box = build_box(cli, cache, dt, args.lat, args.lon)
        if box is None:
            print(f"  {dt}  no MRMS data", file=sys.stderr)
            continue
        p = score(model, box, mean, std)
        rows.append((dt, box, p))
        print(f"  {dt:%H:%M}Z  P(tor)={p:.3f}  ref_max={boxmax(box[0]):.1f}  az02_max={boxmax(box[1]):.1f}")
        if args.save_dir:
            np.savez(os.path.join(args.save_dir, f"sweep_{dt:%H%M}.npz"),
                     x=box, label=-1, lat=args.lat, lon=args.lon, dt=dt.isoformat(),
                     event_id=f"sweep_{dt:%H%M}", conv_day=args.date)

    if not rows:
        sys.exit("no boxes built")

    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(10.5, 3.4 * n), squeeze=False)
    titles = ["reflectivity (dBZ)", "0-2 km AzShear", "3-6 km AzShear"]
    cmaps = ["turbo", "RdBu_r", "RdBu_r"]
    vl = [(-10, 70), (-30, 30), (-30, 30)]
    for i, (dt, box, p) in enumerate(rows):
        cx = box.shape[2] // 2; cy = box.shape[1] // 2
        for c in range(3):
            ax = axes[i][c]
            im = ax.imshow(masked(box[c]), cmap=cmaps[c], vmin=vl[c][0], vmax=vl[c][1], origin="lower")
            ax.plot(cx, cy, "k+", markersize=11, markeredgewidth=1.6)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(titles[c], fontsize=10)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        axes[i][0].set_ylabel(f"{dt:%H:%M}Z\nP(tor)={p:.3f}\nrefmax={boxmax(box[0]):.0f}  "
                              f"az02={boxmax(box[1]):.0f}",
                              fontsize=8.5, rotation=0, ha="right", va="center", labelpad=46)
    fig.suptitle(f"Same point ({args.lat},{args.lon}) over time  -  storm moving into a fixed box",
                 fontsize=11, y=0.997)
    fig.tight_layout(rect=[0.07, 0, 1, 0.99])
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"\nwrote {args.out}  ({n} times)")


if __name__ == "__main__":
    main()
