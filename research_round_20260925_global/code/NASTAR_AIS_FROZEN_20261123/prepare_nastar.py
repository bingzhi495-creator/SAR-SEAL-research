"""Pin the public NASTaR image roster, then download its 8-bit TIFFs without AIS access."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from urllib.parse import quote
import zipfile
import zlib

from PIL import Image
import requests


ACCESSION = "2tfa6x37oerz2lyiw6hp47058"
ROOT = f"https://data.bris.ac.uk/datasets/{ACCESSION}/"
ARCHIVE = f"{ROOT}{ACCESSION}.zip"
ARCHIVE_SIZE = 6_895_666_628
DIRECTORY_START = 6_890_736_085
DIRECTORY_SIZE = 4_930_445
DIRECTORY_SHA = "f84628515acca6c258ef50e5d5a80f398ab63934166aa49f3d6df4a3fedf27cf"
TAIL_SHA = "95dd9f3323eb6931caacfb3f032868d6159076899b1fbfc7a48fc73444888326"
SALT = "NASTAR-AIS-20261123/v1"
EXCLUDED_PASS = "51025-20231226"
PRODUCT = re.compile(r"NovaSAR_01_(\d+)_grd_13_(\d{6})_\d{6}_(HH|VV)(?:_\d+)?")
PATCH = re.compile(r"(.+)_patch_([1-9]\d*)_([^/]+)_uint8\.tif")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def ranking(*fields):
    return sha("|".join((SALT, *fields)).encode("ascii"))


def store_exact(path, data):
    """Create an immutable file or verify the existing one, without partial writes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"Existing file differs: {path}")
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        tmp = Path(handle.name)
        try:
            handle.write(data)
            handle.flush()
        except BaseException:
            handle.close()
            tmp.unlink(missing_ok=True)
            raise
    try:
        # link() creates path exclusively. Both names are on the same filesystem.
        try:
            os.link(tmp, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise ValueError(f"Concurrent file differs: {path}")
    finally:
        tmp.unlink(missing_ok=True)


def get_bytes(url, size, start=None):
    headers = {"Accept-Encoding": "identity"}
    if start is not None:
        end = start + size - 1
        headers["Range"] = f"bytes={start}-{end}"
    elif not url.startswith(ROOT + "Data/") or "/ship_patches_uint8/" not in url or not url.endswith("_uint8.tif"):
        raise ValueError("Only selected display TIFFs may be retrieved")
    for attempt in range(3):
        try:
            with requests.get(url, headers=headers, stream=True, allow_redirects=False, timeout=(12, 35)) as response:
                response.raise_for_status()
                expected_status = 206 if start is not None else 200
                if response.status_code != expected_status:
                    raise ValueError(f"Expected status {expected_status}; got {response.status_code}")
                if start is not None and response.headers.get("Content-Range") != f"bytes {start}-{end}/{ARCHIVE_SIZE}":
                    raise ValueError("Archive Content-Range mismatch")
                declared = response.headers.get("Content-Length")
                if declared is not None and int(declared) != size:
                    raise ValueError("Unexpected Content-Length")
                body = response.raw.read(size + 1, decode_content=False)
                if len(body) != size:
                    raise ValueError("Response length mismatch")
                return body
        except (requests.RequestException, OSError, ValueError) as error:
            if attempt == 2 or (isinstance(error, requests.HTTPError) and error.response.status_code in (401, 403, 404, 410)):
                raise
            time.sleep(1 + attempt)


def pinned_directory():
    # Both byte ranges contain ZIP indexing structures only.
    directory = get_bytes(ARCHIVE, DIRECTORY_SIZE, DIRECTORY_START)
    tail_start = DIRECTORY_START + DIRECTORY_SIZE
    tail = get_bytes(ARCHIVE, ARCHIVE_SIZE - tail_start, tail_start)
    if sha(directory) != DIRECTORY_SHA or sha(tail) != TAIL_SHA:
        raise ValueError("Published archive directory differs from the pinned release")
    with zipfile.ZipFile(io.BytesIO(directory + tail)) as archive:
        records = archive.infolist()
    if len(records) != 24_265:
        raise ValueError("Unexpected count of archive entries")
    return records


def build_lock():
    by_pass = defaultdict(list)
    identities = set()
    products = set()
    for member in pinned_directory():
        parts = member.filename.split("/")
        if len(parts) != 5 or parts[:2] != [ACCESSION, "Data"] or parts[3] != "ship_patches_uint8":
            continue
        product, filename = parts[2], parts[4]
        if not filename.lower().endswith(".tif"):
            continue
        product_match, patch_match = PRODUCT.fullmatch(product), PATCH.fullmatch(filename)
        if not product_match or not patch_match:
            raise ValueError(f"Unexpected display TIFF naming: {member.filename}")
        orbit, date, pol = product_match.groups()
        prefix, number, _ = patch_match.groups()
        if prefix not in (product, f"image_{pol}.tif"):
            raise ValueError(f"Unexpected patch prefix: {member.filename}")
        pass_id = f"{orbit}-20{date}"
        key = product, int(number)
        if key in identities or not 0 < member.file_size < 4 * 1024**2:
            raise ValueError("Duplicate product/patch ID or implausible TIFF size")
        identities.add(key)
        products.add(product)
        relative = "/".join(parts[1:])
        by_pass[pass_id].append({
            "pass_id": pass_id, "product": product, "patch_number": int(number),
            "member": member.filename, "url": ROOT + quote(relative, safe="/"),
            "ais_url": f"{ROOT}Data/{product}/AIS.csv",
            "ais_patch_stem": filename.removesuffix("_uint8.tif"),
            "size_bytes": member.file_size, "crc32": f"{member.CRC:08x}",
        })
    if (len(identities), len(products), len(by_pass)) != (3415, 624, 87) or EXCLUDED_PASS not in by_pass:
        raise ValueError("Published image-only census mismatch")
    passes = sorted(set(by_pass) - {EXCLUDED_PASS}, key=lambda p: (ranking("pass", p), p))[:24]
    items = []
    for pass_id in passes:
        population = by_pass[pass_id]
        ordered = sorted(population, key=lambda x: (ranking("chip", x["product"], str(x["patch_number"])), x["product"], x["patch_number"]))
        for item in ordered[:6]:
            items.append({"image_id": len(items) + 1, **item})
    return {
        "source": ARCHIVE, "archive_bytes": ARCHIVE_SIZE,
        "central_directory_sha256": DIRECTORY_SHA, "zip64_trailer_sha256": TAIL_SHA,
        "archive_entries": 24265, "display_images": len(identities), "products": len(products),
        "all_passes": len(by_pass), "excluded_pass": EXCLUDED_PASS,
        "excluded_pass_chips": len(by_pass[EXCLUDED_PASS]), "ranking_salt": SALT,
        "selected_passes": [
            {"pass_id": p, "available": len(by_pass[p]), "selected": min(6, len(by_pass[p]))} for p in passes
        ],
        "items": items,
    }


def plain_tag(value):
    if value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, bytes):
        return value.decode("ascii").rstrip("\x00")
    if isinstance(value, (list, tuple)):
        return [plain_tag(x) for x in value]
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Nonfinite GeoTIFF metadata")
    return result


def prepare_one(item, out):
    name = f"{item['image_id']:06d}"
    tiff = Path("tiffs") / (name + ".tif")
    png = Path("images") / (name + ".png")
    row = {k: item[k] for k in ("image_id", "pass_id", "product", "patch_number")}
    try:
        target = out / tiff
        data = target.read_bytes() if target.exists() else get_bytes(item["url"], item["size_bytes"])
        if len(data) != item["size_bytes"] or f"{zlib.crc32(data):08x}" != item["crc32"]:
            raise ValueError("TIFF size or CRC32 differs from pinned archive")
        with Image.open(io.BytesIO(data)) as im:
            if (im.format, im.mode, im.size, im.n_frames) != ("TIFF", "L", (512, 512), 1):
                raise ValueError("Expected exactly one native 512x512 uint8 grayscale image")
            if im.tag_v2.get(274, 1) != 1 or im.tag_v2.get(262) != 1:
                raise ValueError("Unsupported orientation or photometric interpretation")
            tags = {
                "model_transformation": 34264, "model_pixel_scale": 33550,
                "model_tiepoint": 33922, "geo_key_directory": 34735,
                "geo_double_params": 34736, "geo_ascii_params": 34737, "gdal_nodata": 42113,
            }
            geometry = {k: plain_tag(im.tag_v2.get(v)) for k, v in tags.items()}
            if not geometry["geo_key_directory"] or not (
                geometry["model_transformation"] or (geometry["model_pixel_scale"] and geometry["model_tiepoint"])
            ):
                raise ValueError("Missing GeoTIFF transform or CRS keys")
            pixels = im.tobytes()
        if len(pixels) != 512 * 512:
            raise ValueError("Unexpected native raster size")
        buf = io.BytesIO()
        Image.frombytes("L", (512, 512), pixels).save(buf, format="PNG")
        png_bytes = buf.getvalue()
        with Image.open(io.BytesIO(png_bytes)) as verification:
            if verification.tobytes() != pixels:
                raise ValueError("Local PNG readback differs from TIFF")
        store_exact(target, data)
        store_exact(out / png, png_bytes)
        return {**row, "status": "ready", "tiff_path": tiff.as_posix(), "image_path": png.as_posix(),
                "tiff_sha256": sha(data), "png_sha256": sha(png_bytes), "decoded_sha256": sha(pixels),
                "geometry": geometry}
    except Exception as error:
        return {**row, "status": "failed", "error": f"{type(error).__name__}: {error}"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for command in ("lock", "prepare"):
        sub = commands.add_parser(command)
        sub.add_argument("--output", required=True, type=Path)
        if command == "prepare":
            sub.add_argument("--lock-sha256", required=True)
            sub.add_argument("--workers", type=int, choices=range(1, 13), default=6)
    args = parser.parse_args()
    out = args.output.resolve()
    lock_path = out / "cohort_lock.json"
    if args.action == "prepare":
        saved = lock_path.read_bytes()
        if sha(saved) != args.lock_sha256:
            raise ValueError("Saved roster differs from supplied lock fingerprint")
    lock = build_lock()
    lock_data = encoded(lock)
    if args.action == "lock":
        store_exact(lock_path, lock_data)
        print(json.dumps({"lock_sha256": sha(lock_data), "selected_passes": len(lock["selected_passes"]),
                          "selected_images": len(lock["items"])}))
        return 0
    if saved != lock_data:
        raise ValueError("Saved roster differs from fresh pinned official catalog reconstruction")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(lambda item: prepare_one(item, out), lock["items"]))
    first_by_pixels = {}
    counts = Counter(row["status"] for row in rows)
    for row in rows:
        if row["status"] == "ready":
            first = first_by_pixels.setdefault(row["decoded_sha256"], row["image_id"])
            row["duplicate_of_image_id"] = None if first == row["image_id"] else first
    ready = [r for r in rows if r["status"] == "ready"]
    intake = {"lock_sha256": sha(saved), "expected": len(rows), "ready": len(ready),
              "failed": counts["failed"], "unique_decoded_images": len(first_by_pixels),
              "exact_decoded_duplicates": len(ready) - len(first_by_pixels),
              "ready_passes": len({r["pass_id"] for r in ready}), "items": rows}
    if counts["failed"]:
        # Keep every partial roster; a retry only attempts the same pinned items.
        prefix = sha(encoded(intake))[:16]
        store_exact(out / f"intake_failed_{prefix}.json", encoded(intake))
    else:
        store_exact(out / "intake.json", encoded(intake))
    print(json.dumps({k: v for k, v in intake.items() if k != "items"}))
    return 2 if counts["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
