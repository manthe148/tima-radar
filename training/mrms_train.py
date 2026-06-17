#!/usr/bin/env python3
"""v8 trainer: small CNN on MRMS 3-channel boxes [ref, az02, az36], 64x64.
Leak-safe split from the manifest. Reports val ROC/PR/CSI vs the 0.881 az02
scalar floor and v7's 0.930."""
import argparse, csv, os, sys, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from common import paths

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FILL = -900.0


def compute_stats(rows):
    s = np.zeros(3); sq = np.zeros(3); n = np.zeros(3)
    for r in rows:
        x = np.load(r["path"])["x"].astype(np.float64)
        for c in range(3):
            v = x[c][x[c] > FILL]
            s[c] += v.sum(); sq[c] += (v * v).sum(); n[c] += v.size
    mean = s / np.maximum(n, 1)
    std = np.sqrt(np.maximum(sq / np.maximum(n, 1) - mean ** 2, 1e-6))
    return mean.astype(np.float32), std.astype(np.float32)


def normalize(x, mean, std):
    x = x.astype(np.float32).copy()
    for c in range(3):
        valid = x[c] > FILL
        x[c] = np.where(valid, (x[c] - mean[c]) / std[c], 0.0)
    return x


def jitter(x, J=12):
    """Random translation +/- J px; exposed edges -> 0 (the normalized no-data
    value). Breaks the centering shortcut so the couplet is no longer reliably
    at box-center -- forces the model to learn structure, not position."""
    dy = random.randint(-J, J); dx = random.randint(-J, J)
    if dy == 0 and dx == 0:
        return x
    H, W = x.shape[1], x.shape[2]
    out = np.zeros_like(x)
    sy0, sy1 = max(0, -dy), min(H, H - dy)
    sx0, sx1 = max(0, -dx), min(W, W - dx)
    ty0, ty1 = max(0, dy), min(H, H + dy)
    tx0, tx1 = max(0, dx), min(W, W + dx)
    out[:, ty0:ty1, tx0:tx1] = x[:, sy0:sy1, sx0:sx1]
    return out


def augment(x):
    x = jitter(x, 12)
    x = np.rot90(x, random.randint(0, 3), axes=(1, 2))
    if random.random() < 0.5:
        x = x[:, :, ::-1]
    if random.random() < 0.5:
        x = x[:, ::-1, :]
    return np.ascontiguousarray(x)


class Boxes(Dataset):
    def __init__(self, rows, mean, std, augment_on=False):
        self.rows = rows; self.mean = mean; self.std = std; self.aug = augment_on
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        x = normalize(np.load(r["path"])["x"], self.mean, self.std)
        if self.aug:
            x = augment(x)
        return torch.from_numpy(x.copy()), float(r["label"])


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


def roc_auc(s, y):
    s = np.asarray(s, float); y = np.asarray(y, int)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    cs = np.cumsum(cnt); ranks = ((cs - cnt + cs + 1) / 2.0)[inv]
    n1 = int(y.sum()); n0 = len(y) - n1
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def pr_auc(s, y):
    s = np.asarray(s, float); y = np.asarray(y, int)
    o = np.argsort(-s, kind="mergesort"); y = y[o]
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    P = tp / np.maximum(tp + fp, 1); R = tp / max(int(y.sum()), 1)
    ap = 0.0; pr = 0.0
    for p, r in zip(P, R):
        ap += p * (r - pr); pr = r
    return ap


def best_csi(s, y):
    s = np.asarray(s, float); y = np.asarray(y, int)
    best = 0.0; bt = 0.5
    for t in np.linspace(0.05, 0.95, 19):
        tp = int(((s >= t) & (y == 1)).sum()); fp = int(((s >= t) & (y == 0)).sum())
        fn = int(((s < t) & (y == 1)).sum())
        csi = tp / max(tp + fp + fn, 1)
        if csi > best:
            best, bt = csi, t
    return best, bt


@torch.no_grad()
def evaluate(model, dl):
    model.eval(); S = []; Y = []
    for x, y in dl:
        S += torch.sigmoid(model(x.to(DEV))).cpu().numpy().tolist()
        Y += y.numpy().tolist()
    return np.array(S), np.array(Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(paths.DATA, "mrms_manifest.csv"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--out", default=os.path.join(paths.DATA, "mrms_v8.pt"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    rows = [r for r in csv.DictReader(open(args.manifest)) if os.path.exists(r["path"])]
    tr = [r for r in rows if r["split"] == "train"]
    va = [r for r in rows if r["split"] == "val"]
    ntp = sum(int(r["label"]) for r in tr); nvp = sum(int(r["label"]) for r in va)
    print(f"train {len(tr)} ({ntp} pos)  val {len(va)} ({nvp} pos)  dev={DEV}")
    if len(tr) < 20 or len(va) < 10:
        print("too few samples -- let the extract finish first."); return

    mean, std = compute_stats(tr)
    print(f"channel means {mean.round(2)}  stds {std.round(2)}")
    tr_dl = DataLoader(Boxes(tr, mean, std, True), batch_size=args.bs, shuffle=True)
    va_dl = DataLoader(Boxes(va, mean, std, False), batch_size=args.bs, shuffle=False)

    model = SmallCNN().to(DEV)
    npar = sum(p.numel() for p in model.parameters())
    pw = torch.tensor([max((len(tr) - ntp) / max(ntp, 1), 1.0)], device=DEV)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    print(f"SmallCNN params={npar/1e3:.0f}k  pos_weight={pw.item():.2f}")

    best_pr = -1; best_state = None; wait = 0
    for ep in range(1, args.epochs + 1):
        model.train(); tl = 0
        for x, y in tr_dl:
            opt.zero_grad()
            loss = crit(model(x.to(DEV)), y.to(DEV))
            loss.backward(); opt.step(); tl += loss.item() * len(y)
        S, Y = evaluate(model, va_dl)
        roc = roc_auc(S, Y); pr = pr_auc(S, Y); csi, ct = best_csi(S, Y)
        print(f"ep{ep:3d}  loss {tl/len(tr):.3f}  val ROC {roc:.3f}  PR {pr:.3f}  CSI {csi:.3f}@{ct:.2f}")
        if pr > best_pr:
            best_pr = pr; best_state = {k: v.cpu() for k, v in model.state_dict().items()}; wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                print(f"early stop (val PR-AUC flat {args.patience} epochs)"); break

    model.load_state_dict(best_state)
    S, Y = evaluate(model, va_dl)
    roc = roc_auc(S, Y); pr = pr_auc(S, Y); csi, ct = best_csi(S, Y)
    torch.save({"model_state": best_state, "mean": mean, "std": std,
                "val_roc": roc, "val_pr": pr, "val_csi": csi}, args.out)
    print("\n" + "=" * 60)
    print("MRMS v8 (small CNN, held-out val)")
    print("=" * 60)
    print(f"  ROC-AUC : {roc:.3f}   (az02 scalar floor 0.881 | v7 0.930)")
    print(f"  PR-AUC  : {pr:.3f}   (v7 0.883)")
    print(f"  best CSI: {csi:.3f} @ {ct:.2f}   (v7 0.672)")
    print("=" * 60)
    print(f"  saved -> {args.out}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
