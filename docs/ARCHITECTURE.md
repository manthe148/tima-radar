# Architecture

TIMA Radar is three cooperating services. They can run on three machines (the
reference deployment) or all on one host for development.

## Data flow
             browser
                |  click lat/lon/time
                v
    +-------------------------+
    |  backend (FastAPI)      |   pulls MRMS, builds the box,
    |  tima_v8_api.py  :8010  |   renders the 3-panel, serves the UI
    +-------------------------+
        |  POST box (3,64,64)        ^  prob + render
        v                            |
    +-------------------------+      |
    |  inference (GPU)        |  ----+
    |  tima_infer.py   :8008  |   normalizes, runs the CNN, returns P(tor)
    +-------------------------+
                ^
                |  loads
        models/mrms_v8_jitter_ca.pt
MRMS on AWS (unsigned S3)  -->  backend & training read it directly

NCEI Storm Events (CSV)    -->  training only (tornado report labels)

## Services

### inference/ — GPU model server
- Single-purpose: holds the checkpoint on GPU and scores boxes it is POSTed.
- Pure stdlib HTTP server + torch + numpy. No web framework, no outbound network.
- Reads `mean`/`std` from the checkpoint, so its preprocessing is guaranteed to
  match training. Listens on `:8008`. Checkpoint path via `TIMA_V8_CKPT`.

### backend/ — web service
- Resolves a query (lat, lon, time, timezone), pulls the nearest MRMS scan for
  each of the three products, interpolates a storm-centered box, and POSTs it to
  the inference server (`TIMA_INFER_URL`).
- Renders the reflectivity + 0–2 km + 3–6 km rotation 3-panel and returns it with
  the score. Serves `frontend/index.html` same-origin. Listens on `:8010`.
- A reflectivity guard flags storm-free boxes so the score is never presented as
  authoritative over clear air.
- Disk cache: clear it whenever the inference checkpoint changes, or it will serve
  the previous model's scores.

### training/ — dataset + model
- `mrms_extract.py` — joins NCEI Storm Events tornado reports with MRMS scans,
  emits storm-centered positives and nearby strong-cell negatives, with a
  leak-safe split by convective day. Resumable.
- `mine_clearair.py` — adds clear-air and non-rotating-precip negatives using the
  same box convention (kills the storm-free false alarm).
- `mrms_train.py` — trains the `SmallCNN` with jitter/rotation/flip augmentation.
- `validate_clearair.py` — reports val metrics broken down by subtype (positives,
  hard negatives, clear-air, precip) plus synthetic empty-box / strong-couplet
  checks.

## Box / preprocessing contract (must stay identical everywhere)

- Box: ±0.25° around the query point, 64×64, nearest-neighbor interpolation,
  out-of-grid fill = −999.
- Channels: `[reflectivity, az_0-2km, az_3-6km]`.
- Normalize: per channel, no-data (`< −900`) → 0, then `(x − mean) / std` with
  `mean`/`std` taken from the checkpoint.

If you change the box build, change it in `training/mrms_extract.py` and
`backend/tima_v8_api.py` together — divergence between them silently corrupts
scores.

## Single-machine setup

Run all three with the inference URL pointed at localhost:

```bash
TIMA_V8_CKPT=models/mrms_v8_jitter_ca.pt python inference/tima_infer.py &
TIMA_INFER_URL=http://127.0.0.1:8008/infer \
  uvicorn tima_v8_api:app --host 0.0.0.0 --port 8010   # from backend/
```
