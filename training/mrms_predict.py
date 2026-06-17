#!/usr/bin/env python3
"""v8/mrms/mrms_predict.py - score the trained v8 model on storms.
Modes: --manifest+--split (held-out check, zero S3), --npz PATH..., or
--lat/--lon/--time (fresh MRMS box). Loads data/mrms_v8.pt + saved mean/std."""
import argparse, csv, os, sys
from datetime import datetime, timedelta
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from common import paths

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FILL = -900.0


class SmallCNN(nn.Module):
    def __init__(self, ch=3, w=32):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
                                 nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.net = nn.Sequential(blk(ch, w), blk(w, w * 2), blk(w * 2, w * 4),
                                 nn.Conv2d(w * 4, w * 4, 3, padding=1),
                                 nn.BatchNorm2d(w * 4), nn.ReLU(inplace=True),
                                 nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Linear(w * 4, 1)
    def forward(self, x):
        return self.fc(self.net(x).flatten(1)).squeeze(1)


def normalize(x, mean, std):
    x = x.astype(np.float32).copy()
    for c in range(3):
        valid = x[c] > FILL
        x[c] = np.where(valid, (x[c] - mean[c]) / std[c], 0.0)
    return x


def boxmax(a):
    v = a[a > -900]
    return float(v.max()) if v.size else float("nan")


def load_model(ckpt_path):
    ck = torch.load(ckpt_path, map_location=DEV, weights_only=False)
    m = SmallCNN().to(DEV)
    m.load_state_dict(ck["model_state"]); m.eval()
    return m, np.asarray(ck["mean"], np.float32), np.asarray(ck["std"], np.float32)


@torch.no_grad()
def score(model, box, mean, std):
    x = normalize(np.asarray(box, np.float32), mean, std)
    t = torch.from_numpy(x[None]).to(DEV)
    return float(torch.sigmoid(model(t)).item())


def box_from_mrms(lat, lon, dt, half_deg=0.25, H=64):
    from mrms_extract import make_client, read_full, interp_of, box_from, list_day, PRODUCTS
    cli = make_client(4)
    days = {dt.date()}
    if dt.hour == 0:
        days.add((dt - timedelta(days=1)).date())
    if dt.hour == 23:
        days.add((dt + timedelta(days=1)).date())
    cache = {}
    for prod in PRODUCTS.values():
        for d in days:
            cache[(prod, d.strftime("%Y%m%d"))] = list_day(cli, prod, datetime(d.year, d.month, d.day))
    chans = []
    for k in ("ref", "az02", "az36"):
        full = read_full(cli, cache, PRODUCTS[k], dt)
        if full is None:
            raise SystemExit(f"no MRMS {k} near {dt}")
        chans.append(box_from(interp_of(full), lat, lon, half_deg, H, H))
    return np.stack(chans)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(paths.DATA, "mrms_v8.pt"))
    ap.add_argument("--manifest")
    ap.add_argument("--split", default="val")
    ap.add_argument("--npz", nargs="*")
    ap.add_argument("--lat", type=float)
    ap.add_argument("--lon", type=float)
    ap.add_argument("--time")
    args = ap.parse_args()

    model, mean, std = load_model(args.ckpt)
    print(f"loaded {args.ckpt}  dev={DEV}  means={mean.round(2)}")

    if args.manifest:
        rows = [r for r in csv.DictReader(open(args.manifest)) if r["split"] == args.split]
        P, Y, info = [], [], []
        for r in rows:
            try:
                box = np.load(r["path"], allow_pickle=True)["x"]
            except Exception:
                continue
            p = score(model, box, mean, std)
            P.append(p); Y.append(int(r["label"]))
            info.append((p, int(r["label"]), os.path.basename(r["path"]), float(r["az02_max"])))
        P = np.array(P); Y = np.array(Y)
        pos = P[Y == 1]; neg = P[Y == 0]
        print(f"\n{args.split}: {len(pos)} tornadic, {len(neg)} non-tornadic")
        print(f"  P(tor) median   tornadic {np.median(pos):.3f}   non-torn {np.median(neg):.3f}")
        print(f"  P(tor)>=0.5      tornadic {(pos>=.5).mean():.2f} (recall)   non-torn {(neg>=.5).mean():.2f} (false-alarm)")
        info.sort(reverse=True)
        print("\n  most-confident calls (top 8):")
        for p, lab, fn, a2 in info[:8]:
            tag = "TOR " if lab == 1 else "neg "
            print(f"    {tag} P={p:.3f}  az02={a2:5.1f}  {fn}")
        miss = sorted([t for t in info if t[1] == 1 and t[0] < 0.5])
        print(f"\n  missed tornadoes (P<0.5): {len(miss)}")
        for p, lab, fn, a2 in miss[:6]:
            print(f"    TOR  P={p:.3f}  az02={a2:5.1f}  {fn}")
        return

    if args.npz:
        for p in args.npz:
            d = np.load(p, allow_pickle=True); box = d["x"]
            lab = int(d["label"]) if "label" in d.files else -1
            print(f"  {os.path.basename(p)}  P(tor)={score(model, box, mean, std):.3f}  "
                  f"label={lab}  az02_max={boxmax(box[1]):.1f}")
        return

    if args.lat is not None and args.lon is not None and args.time:
        dt = datetime.strptime(args.time, "%Y-%m-%d %H:%M")
        box = box_from_mrms(args.lat, args.lon, dt)
        print(f"\n  ({args.lat},{args.lon}) {dt}Z  P(tor)={score(model, box, mean, std):.3f}  "
              f"ref_max={boxmax(box[0]):.1f}  az02_max={boxmax(box[1]):.1f}  az36_max={boxmax(box[2]):.1f}")
        return

    ap.error("give --manifest, --npz, or --lat/--lon/--time")


if __name__ == "__main__":
    main()
