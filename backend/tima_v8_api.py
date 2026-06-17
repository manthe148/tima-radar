#!/usr/bin/env python3
"""
TIMA v8 MRMS backend (FastAPI)  --  runs on the WEB SERVER (.136 / StormData).

Matches the v8 frontend exactly: GET /presets (keyed object), POST /query
({lat, lon, timestamp, tz}) -> {cnn_signature_match, ...}. Does the MRMS pull,
box build, render, guard, timezone handling, caching. The model itself runs on
the GAMING PC (127.0.0.1); this calls it for every score.

Run (.136):
  TIMA_INFER_URL=http://127.0.0.1:8008/infer \
  uvicorn tima_v8_api:app --host 0.0.0.0 --port 8010

Deps:  fastapi uvicorn numpy scipy boto3 pygrib matplotlib requests
       (timezonefinder optional -- falls back to a CONUS longitude band)
Env:
  TIMA_INFER_URL  AI endpoint   (default http://127.0.0.1:8008/infer)
  TIMA_V8_CORS    origins       (default *)
  TIMA_V8_CACHE   cache dir      (default ./v8_cache)
"""
import os, io, re, gzip, json, base64, hashlib, tempfile, pathlib
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

import requests
from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

BUCKET = "noaa-mrms-pds"
PRODUCTS = {"ref": "MergedReflectivityQCComposite_00.50",
            "az02": "MergedAzShear_0-2kmAGL_00.50",
            "az36": "MergedAzShear_3-6kmAGL_00.50"}
KEYRE = re.compile(r"(\d{8})-(\d{6})")
FILL = -900.0
ARCHIVE_START = datetime(2020, 10, 1)
CONUS = (20.0, 55.0, -130.0, -60.0)
STORM_REF_MIN = 35.0
HALF_DEG = 0.25
H = 64

INFER_URL = os.environ.get("TIMA_INFER_URL", "http://127.0.0.1:8008/infer")
CORS = [o.strip() for o in os.environ.get("TIMA_V8_CORS", "*").split(",")]
CACHE_DIR = pathlib.Path(os.environ.get("TIMA_V8_CACHE", "./v8_cache"))
CACHE_DIR.mkdir(exist_ok=True)

# keyed by id, LOCAL wall-clock time (frontend sends tz="auto"), datetime-local format
PRESETS = [
    {"id": "greenfield_2024",  "name": "Greenfield, IA EF4 (2024)",  "lat": 41.31, "lon": -94.46, "time": "2024-05-21 15:30"},
    {"id": "mayfield_2021",    "name": "Mayfield, KY EF4 (2021)",    "lat": 36.74, "lon": -88.64, "time": "2021-12-10 21:26"},
    {"id": "rollingfork_2023", "name": "Rolling Fork, MS EF4 (2023)", "lat": 32.91, "lon": -90.88, "time": "2023-03-24 20:07"},
    {"id": "bassfield_2020",   "name": "Bassfield, MS EF4 (2020)",   "lat": 31.50, "lon": -89.75, "time": "2020-12-25 12:30"},
]


# ---------------- MRMS reader ----------------
def s3():
    import boto3
    from botocore import UNSIGNED
    from botocore.client import Config
    return boto3.client("s3", config=Config(
        signature_version=UNSIGNED, max_pool_connections=8,
        connect_timeout=10, read_timeout=60,
        retries={"max_attempts": 5, "mode": "standard"}))


def list_day(cli, product, day):
    keys = []
    for page in cli.get_paginator("list_objects_v2").paginate(
            Bucket=BUCKET, Prefix=f"CONUS/{product}/{day:%Y%m%d}/"):
        for o in page.get("Contents", []):
            keys.append(o["Key"])
    return keys


def nearest_key(cli, product, dt):
    days = {dt.date()}
    if dt.hour == 0:
        days.add((dt - timedelta(days=1)).date())
    if dt.hour == 23:
        days.add((dt + timedelta(days=1)).date())
    keys = []
    for d in days:
        keys += list_day(cli, product, datetime(d.year, d.month, d.day))
    best, bd = None, 1e18
    for k in keys:
        m = KEYRE.search(k)
        if not m:
            continue
        t = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        dd = abs((t - dt).total_seconds())
        if dd < bd:
            bd, best = dd, k
    return (best, bd) if best else (None, None)


def read_full(cli, product, dt):
    import pygrib
    key, gap = nearest_key(cli, product, dt)
    if key is None:
        return None
    data = gzip.decompress(cli.get_object(Bucket=BUCKET, Key=key)["Body"].read())
    path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as f:
            f.write(data); path = f.name
        grbs = pygrib.open(path); g = grbs[1]
        vals = np.array(g.values, float)
        try:
            lat1 = np.array(g.distinctLatitudes, float)
            lon1 = np.array(g.distinctLongitudes, float)
        except Exception:
            lats, lons = g.latlons(); lat1 = lats[:, 0]; lon1 = lons[0, :]
        grbs.close()
    finally:
        if path and os.path.exists(path):
            os.unlink(path)
    lon1 = np.where(lon1 > 180, lon1 - 360, lon1)
    if lat1[0] > lat1[-1]:
        lat1 = lat1[::-1]; vals = vals[::-1, :]
    m = KEYRE.search(key)
    stime = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S") if m else dt
    return {"lat": lat1, "lon": lon1, "vals": vals, "gap": gap, "scan_time": stime}


def box_from(full, clat, clon):
    itp = RegularGridInterpolator((full["lat"], full["lon"]), full["vals"],
                                  method="nearest", bounds_error=False, fill_value=-999.0)
    tlat = np.linspace(clat - HALF_DEG, clat + HALF_DEG, H)
    tlon = np.linspace(clon - HALF_DEG, clon + HALF_DEG, H)
    gy, gx = np.meshgrid(tlat, tlon, indexing="ij")
    return itp(np.column_stack([gy.ravel(), gx.ravel()])).reshape(H, H).astype(np.float32)


def boxmax(a):
    v = a[a > -900]
    return float(v.max()) if v.size else float("nan")


# ---------------- call the AI on the gaming PC ----------------
def remote_infer(x):
    x = np.ascontiguousarray(x, dtype=np.float32)
    payload = {"shape": list(x.shape), "data": base64.b64encode(x.tobytes()).decode("ascii")}
    r = requests.post(INFER_URL, json=payload, timeout=30)
    r.raise_for_status()
    j = r.json()
    if "prob" not in j:
        raise RuntimeError(j.get("error", "inference service returned no prob"))
    return float(j["prob"])


# ---------------- render ----------------
def render_png(x, lat, lon, scan_time, prob):
    titles = ["Reflectivity (dBZ)", "0-2 km Rotation", "3-6 km Rotation"]
    cmaps = ["turbo", "RdBu_r", "RdBu_r"]
    vl = [(-10, 70), (-15, 15), (-15, 15)]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    for c in range(3):
        ax = axes[c]
        im = ax.imshow(np.ma.masked_less(x[c], -900), cmap=cmaps[c],
                       vmin=vl[c][0], vmax=vl[c][1], origin="lower")
        ax.plot(H // 2, H // 2, "k+", markersize=12, markeredgewidth=1.8)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(titles[c], fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(f"P(tornado) = {prob:.3f}   |   {scan_time:%Y-%m-%d %H:%M} UTC   |   "
                 f"({lat:.3f}, {lon:.3f})   |   + = query point", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=95, bbox_inches="tight", facecolor="white")
    plt.close(fig); buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


# ---------------- time helpers ----------------
def parse_time(s):
    s = s.strip().replace("T", " ").replace("Z", "")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _tz_from_lon(lon):
    if lon >= -82.5:  return "America/New_York"
    if lon >= -97.5:  return "America/Chicago"
    if lon >= -114.0: return "America/Denver"
    return "America/Los_Angeles"


def resolve_tz(tz, lat, lon):
    if not tz or tz.upper() == "UTC":
        return None
    if tz.lower() == "auto":
        try:
            from timezonefinder import TimezoneFinder
            name = TimezoneFinder().timezone_at(lat=lat, lng=lon)
            return name or _tz_from_lon(lon)
        except Exception:
            return _tz_from_lon(lon)        # robust fallback, no hard dependency
    return tz


def local_to_utc(dt, tz):
    return dt.replace(tzinfo=ZoneInfo(tz)).astimezone(ZoneInfo("UTC")).replace(tzinfo=None)


def utc_to_local(dt, tz):
    return dt.replace(tzinfo=ZoneInfo("UTC")).astimezone(ZoneInfo(tz)).replace(tzinfo=None)


def in_conus(lat, lon):
    return CONUS[0] <= lat <= CONUS[1] and CONUS[2] <= lon <= CONUS[3]


# ---------------- core ----------------
def build_and_score(lat, lon, dt, include_image=True):
    cli = s3()
    fulls = {}
    for k in ("ref", "az02", "az36"):
        full = read_full(cli, PRODUCTS[k], dt)
        if full is None:
            return None, f"no MRMS {k} data near {dt:%Y-%m-%d %H:%M} UTC"
        fulls[k] = full
    x = np.stack([box_from(fulls["ref"], lat, lon),
                  box_from(fulls["az02"], lat, lon),
                  box_from(fulls["az36"], lat, lon)])
    prob = remote_infer(x)
    ref_max = boxmax(x[0]); az02_max = boxmax(x[1]); az36_max = boxmax(x[2])
    scan_time = fulls["ref"]["scan_time"]
    storm_present = bool(ref_max >= STORM_REF_MIN)
    warning = None if storm_present else (
        "Low reflectivity at this location -- likely no storm here. The model's "
        "score is unreliable over storm-free areas; ignore it.")
    out = {
        "prob": round(prob, 4),
        "prob_pct": round(prob * 100, 1),
        "cnn_signature_match": round(prob * 100, 1),   # <-- the field the frontend reads
        "lat": lat, "lon": lon,
        "requested_time": dt.strftime("%Y-%m-%d %H:%M:%S"),
        "scan_time": scan_time.strftime("%Y-%m-%d %H:%M:%S"),
        "scan_gap_min": round(fulls["ref"]["gap"] / 60.0, 1),
        "ref_max": round(ref_max, 1), "az02_max": round(az02_max, 1), "az36_max": round(az36_max, 1),
        "storm_present": storm_present, "warning": warning, "cached": False,
    }
    if include_image:
        out["image"] = render_png(x, lat, lon, scan_time, prob)
    return out, None


def cache_key(lat, lon, dt):
    raw = f"{round(lat,2)}_{round(lon,2)}_{dt:%Y%m%d%H%M}"
    return CACHE_DIR / (hashlib.md5(raw.encode()).hexdigest() + ".json")


def run_query(lat, lon, timestamp, tz):
    if not timestamp:
        return JSONResponse({"error": "missing time"}, status_code=400)
    dt_in = parse_time(timestamp)
    if dt_in is None:
        return JSONResponse({"error": "bad time format"}, status_code=400)
    if not in_conus(lat, lon):
        return JSONResponse({"error": "location outside CONUS MRMS domain"}, status_code=422)
    tzname = resolve_tz(tz, lat, lon)
    try:
        dt = local_to_utc(dt_in, tzname) if tzname else dt_in
    except Exception as e:
        return JSONResponse({"error": f"bad timezone: {e}"}, status_code=400)
    if dt < ARCHIVE_START:
        return JSONResponse({"error": f"MRMS archive starts {ARCHIVE_START:%Y-%m-%d}"}, status_code=422)
    now_utc = datetime.utcnow()
    if dt > now_utc:
        dt = now_utc    # no future radar -- clamp now/future clicks to the latest scan

    ck = cache_key(lat, lon, dt)
    if ck.exists():
        try:
            r = json.loads(ck.read_text()); r["cached"] = True
            return r
        except Exception:
            pass
    try:
        out, err = build_and_score(lat, lon, dt)
    except requests.RequestException as e:
        return JSONResponse({"error": f"AI inference service unreachable: {e}"}, status_code=502)
    except Exception as e:
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)
    if err:
        return JSONResponse({"error": err}, status_code=404)

    out["tz"] = tzname or "UTC"
    if tzname:
        scan_utc = datetime.strptime(out["scan_time"], "%Y-%m-%d %H:%M:%S")
        out["scan_time_local"] = utc_to_local(scan_utc, tzname).strftime("%Y-%m-%d %H:%M:%S")
        out["requested_local"] = dt_in.strftime("%Y-%m-%d %H:%M:%S")
    else:
        out["scan_time_local"] = out["scan_time"]
        out["requested_local"] = out["requested_time"]
    try:
        ck.write_text(json.dumps(out))
    except Exception:
        pass
    return out


# ---------------- API ----------------
app = FastAPI(title="TIMA v8 MRMS backend (web)", version="1.1")
app.add_middleware(CORSMiddleware, allow_origins=CORS, allow_methods=["*"], allow_headers=["*"])


@app.get("/")
def index():
    return FileResponse("index.html")   # save the frontend as index.html next to this file


class QueryReq(BaseModel):
    lat: float
    lon: float
    time: str | None = None
    timestamp: str | None = None
    tz: str = "auto"

    def ts(self):
        return self.time or self.timestamp


@app.get("/health")
def health():
    ai = "unknown"
    try:
        r = requests.get(INFER_URL.rsplit("/", 1)[0] + "/health", timeout=5)
        ai = "ok" if r.ok else f"error {r.status_code}"
    except Exception as e:
        ai = f"unreachable: {type(e).__name__}"
    return {"status": "ok", "role": "web-api", "ai_infer_url": INFER_URL, "ai_status": ai}


@app.get("/presets")
def presets():
    return {"presets": PRESETS}                       # keyed object, exactly what the frontend parses


@app.post("/query")
def query(req: QueryReq):
    return run_query(req.lat, req.lon, req.ts(), req.tz)


# alias so /score also works (same body shape)
@app.post("/score")
def score(req: QueryReq):
    return run_query(req.lat, req.lon, req.ts(), req.tz)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8010")))
