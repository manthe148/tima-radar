#!/usr/bin/env python3
"""
mrms/mrms_read.py - MRMS reader foundation for v8.
Pulls from the public noaa-mrms-pds S3 bucket (unsigned), finds the file
nearest an event time, reads the grib2, flips 0-360 lon, drops -999 fill,
and extracts a box around (lat, lon).

Validates against Mayfield EF4 (2021-12-11 03:26 UTC, 36.74,-88.64):
  reflectivity tight-box max ~57.5 dBZ, 0-2km AzShear ~32.

  python3 mrms/mrms_read.py        # runs the Mayfield self-test
"""
import gzip, os, re, sys, tempfile
from datetime import datetime, timedelta
import numpy as np

BUCKET = "noaa-mrms-pds"

# minimal product set (dir names under CONUS/). adjust if a listing comes back empty.
PRODUCTS = {
    "ref":   "MergedReflectivityQCComposite_00.50",
    "az02":  "MergedAzShear_0-2kmAGL_00.50",
    "az36":  "MergedAzShear_3-6kmAGL_00.50",
}


def s3():
    import boto3
    from botocore import UNSIGNED
    from botocore.client import Config
    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def _list(cli, product, day):
    prefix = f"CONUS/{product}/{day:%Y%m%d}/"
    keys = []
    for page in cli.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=prefix):
        for o in page.get("Contents", []):
            keys.append(o["Key"])
    return keys


def nearest_key(cli, product, dt):
    """File nearest dt; checks the day and the adjacent day near a boundary."""
    days = {dt.date()}
    if dt.hour == 0:
        days.add((dt - timedelta(days=1)).date())
    if dt.hour == 23:
        days.add((dt + timedelta(days=1)).date())
    keys = []
    for d in days:
        keys += _list(cli, product, datetime(d.year, d.month, d.day))
    best, bd = None, 1e18
    for k in keys:
        m = re.search(r"(\d{8})-(\d{6})", k)
        if not m:
            continue
        t = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        d = abs((t - dt).total_seconds())
        if d < bd:
            bd, best = d, k
    return (best, bd) if best else (None, None)


def read_box(cli, product, dt, lat, lon, half_deg=0.25):
    key, gap = nearest_key(cli, product, dt)
    if key is None:
        return {"key": None, "error": "no file in window"}
    raw = cli.get_object(Bucket=BUCKET, Key=key)["Body"].read()
    data = gzip.decompress(raw)
    path = None
    try:
        import pygrib
        with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as f:
            f.write(data); path = f.name
        grbs = pygrib.open(path); g = grbs[1]
        vals = np.array(g.values, dtype=float)
        lats, lons = g.latlons()
        grbs.close()
    finally:
        if path and os.path.exists(path):
            os.unlink(path)
    lons = np.where(lons > 180, lons - 360, lons)
    sel = (np.abs(lats - lat) <= half_deg) & (np.abs(lons - lon) <= half_deg)
    box = vals[sel]
    box = box[box > -900]
    return {
        "key": os.path.basename(key),
        "gap_s": round(gap, 0),
        "n_valid": int(box.size),
        "max": float(box.max()) if box.size else float("nan"),
        "mean": float(box.mean()) if box.size else float("nan"),
    }


def _mayfield():
    cli = s3()
    dt = datetime(2021, 12, 11, 3, 26, 0)
    lat, lon = 36.74, -88.64
    print(f"Mayfield EF4  {dt} UTC  ({lat},{lon})  half_deg=0.25\n")
    expect = {"ref": "~57.5 dBZ", "az02": "~32 (strong rotation)", "az36": "mid-level"}
    for k, prod in PRODUCTS.items():
        r = read_box(cli, prod, dt, lat, lon, 0.25)
        print(f"  {k:5} {prod}")
        print(f"        -> {r}   expect {expect[k]}\n")
    print("if ref ~57 and az02 ~30+, the reader is faithful. then we build the extractor.")


if __name__ == "__main__":
    _mayfield()
