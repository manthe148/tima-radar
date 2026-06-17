#!/usr/bin/env python3
"""v8/mrms/mrms_view.py - render MRMS sample npz as PNG panels:
reflectivity | 0-2km AzShear | 3-6km AzShear. '+' = box center (labeled point).
Optional per-sample P(tor) with --ckpt."""
import argparse, csv, os, random, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)


def masked(a):
    return np.ma.masked_less(np.asarray(a, float), -900)


def collect_paths(args):
    if args.npz:
        return list(args.npz)
    if args.manifest:
        rows = [r for r in csv.DictReader(open(args.manifest))
                if r["split"] == args.split and (args.label is None or int(r["label"]) == args.label)]
        random.Random(args.seed).shuffle(rows)
        return [r["path"] for r in rows[:args.n]]
    sys.exit("give --npz PATH... or --manifest (+--split/--label/--n)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", nargs="*")
    ap.add_argument("--manifest")
    ap.add_argument("--split", default="val")
    ap.add_argument("--label", type=int)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--ckpt")
    ap.add_argument("--out", default="mrms_view.png")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    paths = collect_paths(args)
    if not paths:
        sys.exit("no samples to render")

    model = mean = std = None
    if args.ckpt:
        from mrms_predict import load_model
        model, mean, std = load_model(args.ckpt)

    n = len(paths)
    fig, axes = plt.subplots(n, 3, figsize=(10.5, 3.4 * n), squeeze=False)
    chan_titles = ["reflectivity (dBZ)", "0-2 km AzShear", "3-6 km AzShear"]
    cmaps = ["turbo", "RdBu_r", "RdBu_r"]
    vlims = [(-10, 70), (-30, 30), (-30, 30)]

    for i, p in enumerate(paths):
        d = np.load(p, allow_pickle=True)
        x = d["x"]
        lab = int(d["label"]) if "label" in d.files else -1
        dt = str(d["dt"]) if "dt" in d.files else "?"
        cx = x.shape[2] // 2; cy = x.shape[1] // 2

        ptxt = ""
        if model is not None:
            from mrms_predict import score
            ptxt = f"  P(tor)={score(model, x, mean, std):.3f}"

        for c in range(3):
            ax = axes[i][c]
            im = ax.imshow(masked(x[c]), cmap=cmaps[c], vmin=vlims[c][0], vmax=vlims[c][1],
                           origin="lower")
            ax.plot(cx, cy, "k+", markersize=11, markeredgewidth=1.6)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(chan_titles[c], fontsize=10)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

        tag = "TORNADIC" if lab == 1 else ("non-torn" if lab == 0 else "?")
        axes[i][0].set_ylabel(f"{os.path.basename(p)}\n{tag}{ptxt}\n{dt}",
                              fontsize=8, rotation=0, ha="right", va="center", labelpad=42)

    fig.suptitle("MRMS boxes  (+ = box center / labeled point;  north up, east right)",
                 fontsize=11, y=0.995)
    fig.tight_layout(rect=[0.06, 0, 1, 0.99])
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"wrote {args.out}  ({n} samples)")


if __name__ == "__main__":
    main()
