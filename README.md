# TIMA Radar — open MRMS tornado-signature detection

> **Research prototype. Not a warning system.** This model estimates how strongly
> a patch of radar matches the rotational signature it was trained on. Most
> rotating storms do **not** produce tornadoes. Always follow official National
> Weather Service warnings — never this tool.

TIMA Radar is an end-to-end, open pipeline that pulls live NOAA MRMS radar,
crops storm-centered boxes, and scores them with a small CNN trained on real
tornado reports. It runs as a three-service stack (training, GPU inference, web
backend) and is usable interactively from a browser, including live map clicks.

## Why this exists

There is no open-source, real-time NEXRAD/MRMS + machine-learning tornado
pipeline with an honest, leak-safe validation story. Most public examples either
stop at offline notebooks or quietly inflate their numbers with leaky splits and
centering artifacts. The goal here is the opposite: a reproducible pipeline where
the **validation discipline is part of the contribution**, not an afterthought.

## The model

A ~242k-parameter `SmallCNN` (conv/BN/ReLU/maxpool → global pool → linear) over
**3-channel, 64×64 storm-centered boxes**, ±0.25° around the query point:

| channel | MRMS product |
|---|---|
| reflectivity | `MergedReflectivityQCComposite_00.50` |
| 0–2 km rotation | `MergedAzShear_0-2kmAGL_00.50` |
| 3–6 km rotation | `MergedAzShear_3-6kmAGL_00.50` |

Per-channel normalization `(x − mean) / std` with no-data (`< −900`) mapped to 0;
`mean`/`std` are stored **inside the checkpoint** so inference always matches
training. Released weights: `models/mrms_v8_jitter_ca.pt`.

## Validation — the honest numbers

Held-out validation uses a **leak-safe split**: samples are partitioned by
*convective day* (and storms kept ≥150 km apart), so no storm appears in both
train and val. Three deliberate disciplines shape the result:

1. **Jitter augmentation** (±12 px random translation) breaks the box-center
   shortcut — the model can't assume the couplet sits at the exact center, so it
   has to learn structure instead of position. (This dropped the apparent ROC from
   an inflated 0.985 to an honest 0.919 — the lower number is the real one.)
2. **Clear-air negatives** teach the model that storm-free space is not tornadic,
   killing a false alarm where empty boxes scored ~90%.
3. **Non-rotating precip negatives** span the reflectivity range so the model
   keys on *rotation*, not raw intensity.

On the held-out set (leak-safe conv_day split):

- ROC-AUC **0.944** (reference: az-shear scalar floor 0.881)
- Tornadic recall ≈ **0.90**
- Clear-air / non-rotating-precip boxes score ≈ **0.01–0.02** (the false-alarm fix)
- An empty, storm-free box scores ≈ **0.09** (was ~0.90 before the clear-air work)

A backend guard additionally flags low-reflectivity boxes so the score is never
shown as authoritative over storm-free areas.

## Architecture

Three services (see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full
data flow):

- **`training/`** — builds the dataset from NCEI Storm Events + MRMS on AWS and
  trains the CNN. GPU recommended.
- **`inference/`** — a tiny stdlib HTTP server that holds the model on GPU and
  scores boxes it receives. No web framework, no outbound calls.
- **`backend/`** — FastAPI service that pulls MRMS, builds the box, calls the
  inference server, renders the 3-panel image, and serves the frontend.

The split exists so the GPU box only ever runs the model, and the data/web work
lives elsewhere; for a single-machine setup, point `TIMA_INFER_URL` at localhost
and run all three on one host.

## Quickstart (inference only)

```bash
# 1. GPU inference server (holds the released model)
cd inference && pip install -r requirements.txt
TIMA_V8_CKPT=../models/mrms_v8_jitter_ca.pt python tima_infer.py   # :8008

# 2. web backend (pulls MRMS, builds boxes, serves the UI)
cd ../backend && pip install -r requirements.txt
TIMA_INFER_URL=http://127.0.0.1:8008/infer uvicorn tima_v8_api:app --host 0.0.0.0 --port 8010
# open http://localhost:8010
```

## Reproduce training

Data is **not** committed (it's gigabytes and fully regenerable). Sources are
public and need no credentials — NCEI Storm Events for tornado reports and NOAA
MRMS on AWS (unsigned S3) for radar.

```bash
cd training && pip install -r requirements.txt
python mrms_extract.py            # NCEI + MRMS  -> data/mrms_samples/*.npz + manifest
python mine_clearair.py --target-clear 800 --target-precip 500   # add clear-air negatives
python mrms_train.py --out ../models/mrms_v8_jitter_ca.pt
python validate_clearair.py --ckpt ../models/mrms_v8_jitter_ca.pt
```

## Repo layout
training/    dataset build + train + validate (RunPod / any GPU box)

inference/   GPU model server (tima_infer.py)

backend/     FastAPI: MRMS pull, box build, render, serve UI

frontend/    Leaflet workbench (index.html)

models/      released checkpoint

docs/        architecture + data flow

## License

AGPL-3.0. If you run a modified version as a network service, you must make your
source available under the same license. Commercial licensing is available
separately — see the contact below.

> Add the canonical license text with the GitHub license picker (GNU AGPLv3) when
> creating the repo, or: `curl -L https://www.gnu.org/licenses/agpl-3.0.txt -o LICENSE`
