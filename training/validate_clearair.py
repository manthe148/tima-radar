#!/usr/bin/env python3
"""Validate a checkpoint on the val split, broken down by sample subtype, and
confirm the empty-box false alarm is fixed without denting real couplets.

Run:
  cd /workspace/v8
  /workspace/test/venv/bin/python mrms/validate_clearair.py --ckpt data/mrms_v8_jitter_ca.pt
"""
import os, sys, csv, argparse
import numpy as np, torch, torch.nn as nn

FILL = -900.0

class SmallCNN(nn.Module):
    def __init__(self, ch=3, w=32):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
                                 nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.net = nn.Sequential(blk(ch, w), blk(w, w*2), blk(w*2, w*4),
                                 nn.Conv2d(w*4, w*4, 3, padding=1),
                                 nn.BatchNorm2d(w*4), nn.ReLU(inplace=True),
                                 nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Linear(w*4, 1)
    def forward(self, x):
        return self.fc(self.net(x).flatten(1)).squeeze(1)

def normalize(x, mean, std):
    x = x.astype(np.float32).copy()
    for c in range(x.shape[0]):
        v = x[c] > FILL
        x[c] = np.where(v, (x[c]-mean[c])/std[c], 0.0)
    return x

def subtype(path):
    b = os.path.basename(path)
    if "_pos" in b: return "pos"
    if "_clr" in b: return "clear"
    if "_prc" in b: return "precip"
    return "hard_neg"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", default="data/mrms_manifest.csv")
    ap.add_argument("--thresh", type=float, default=0.5)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    m = SmallCNN().to(dev); m.load_state_dict(ck["model_state"]); m.eval()
    mean = np.asarray(ck["mean"], np.float32); std = np.asarray(ck["std"], np.float32)
    print(f"ckpt {a.ckpt}  saved val_roc={ck.get('val_roc')}  mean={list(np.round(mean,3))}")

    @torch.no_grad()
    def score(x):
        t = torch.from_numpy(normalize(x, mean, std)[None]).to(dev)
        return float(torch.sigmoid(m(t)).item())

    groups = {}
    for r in csv.DictReader(open(a.manifest)):
        if r["split"] != "val": continue
        try: x = np.load(r["path"])["x"]
        except Exception: continue
        groups.setdefault(subtype(r["path"]), []).append(score(x))

    print("\nval scores by subtype (n, median P, frac>=thresh):")
    for g in ["pos", "hard_neg", "clear", "precip"]:
        v = np.array(groups.get(g, []))
        if len(v) == 0:
            print(f"  {g:9s}: (none yet)"); continue
        tag = "recall" if g == "pos" else "FAR  "
        print(f"  {g:9s}: n={len(v):4d}  median={np.median(v):.3f}  {tag}={(v>=a.thresh).mean():.3f}")

    empty = np.full((3, 64, 64), -999.0, np.float32)
    print(f"\nEMPTY BOX (off-storm click):     P(tor)={score(empty):.3f}   <- want LOW (was ~0.9)")
    strong = np.zeros((3, 64, 64), np.float32)
    strong[0, 26:38, 26:38] = 60; strong[1, 30:34, 30:34] = 25; strong[2, 30:34, 30:34] = 18
    print(f"STRONG CENTERED COUPLET:         P(tor)={score(strong):.3f}   <- want HIGH (stay >0.8)")

if __name__ == "__main__":
    main()
