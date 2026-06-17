#!/usr/bin/env python3
"""TIMA v8 AI -- GAMING PC. Only the model. torch + numpy, no web framework."""
import os, json, base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np
import torch
import torch.nn as nn

FILL = -900.0
CKPT = os.environ.get("TIMA_V8_CKPT", "mrms_v8_jitter.pt")
PORT = int(os.environ.get("PORT", "8008"))
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


_MODEL = _MEAN = _STD = None
def get_model():
    global _MODEL, _MEAN, _STD
    if _MODEL is None:
        ck = torch.load(CKPT, map_location=DEV, weights_only=False)
        m = SmallCNN().to(DEV); m.load_state_dict(ck["model_state"]); m.eval()
        _MODEL = m
        _MEAN = np.asarray(ck["mean"], np.float32)
        _STD = np.asarray(ck["std"], np.float32)
    return _MODEL, _MEAN, _STD


def normalize(x, mean, std):
    x = x.astype(np.float32).copy()
    for c in range(x.shape[0]):
        valid = x[c] > FILL
        x[c] = np.where(valid, (x[c] - mean[c]) / std[c], 0.0)
    return x


@torch.no_grad()
def score_box(x):
    model, mean, std = get_model()
    xn = normalize(x, mean, std)
    t = torch.from_numpy(xn[None]).to(DEV)
    return float(torch.sigmoid(model(t)).item())


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            try:
                get_model()
                self._send(200, {"status": "ok", "role": "ai", "device": str(DEV)})
            except Exception as e:
                self._send(500, {"status": "error", "error": str(e)})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/infer":
            self._send(404, {"error": "not found"}); return
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n))
            x = np.frombuffer(base64.b64decode(req["data"]), dtype=np.float32).reshape(req["shape"])
        except Exception as e:
            self._send(400, {"error": f"bad input: {e}"}); return
        if tuple(x.shape) != (3, 64, 64):
            self._send(400, {"error": f"expected (3,64,64), got {tuple(x.shape)}"}); return
        try:
            self._send(200, {"prob": score_box(x)})
        except Exception as e:
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    get_model()
    print(f"TIMA v8 AI listening on 0.0.0.0:{PORT}  (device={DEV})", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
