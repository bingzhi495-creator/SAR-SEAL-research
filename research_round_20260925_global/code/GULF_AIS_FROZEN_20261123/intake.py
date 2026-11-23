"""Admit pinned Gulf VV chips and source scene geometry before frozen inference."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from http.client import HTTPException
from io import BytesIO
import json
import math
from pathlib import Path
import re
from time import perf_counter, sleep
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET
from zipfile import BadZipFile, ZipFile

import numpy as np
from PIL import Image


REPO = "mshaya/sar-ship-gulf"
REVISION = "820ae84a00677d535c4db315adffb129b0d4a835"
LOCK = "6001a3d9be9cf5cccb71e4156603d91c2dec1858ba46816a8380f2d315272cbb"
CENSUS_WORKER = "d70c3a569c931b4d2a28b572a3a63a4226577ac0dc0ba2d4ea1bd37669d0351d"
RESOLVE = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/"
S3 = "https://sentinel-s1-l1c.s3.eu-central-1.amazonaws.com/GRD"
SCENE = re.compile(r"(S1A)_IW_GRDH_1SDV_(\d{8}T\d{6})_(\d{8}T\d{6})_(\d{6})_[A-F0-9]{6}_[A-F0-9]{4}")
HEX = re.compile(r"[a-f0-9]{64}")
ATTEMPTS = 4


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(data):
    return sha256(data).hexdigest()


def encode(value):
    return (json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def store(path, data):
    with path.open("xb") as f:
        require(f.write(data) == len(data), "Incomplete local write")
    require(path.stat().st_size == len(data) and digest(path.read_bytes()) == digest(data), "Local readback differs")


def admit_census(args):
    protocol = digest(args.protocol.read_bytes())
    require(protocol == args.protocol_sha256 == LOCK, "Protocol digest differs")
    terminal_bytes = args.census_terminal.read_bytes()
    terminal = json.loads(terminal_bytes)
    receipt_bytes = (args.census / "RESULT_SAFE.json").read_bytes()
    receipt = json.loads(receipt_bytes)
    require(terminal["phase"] == receipt["phase"] == "census" and type(terminal["actual_exit"]) is int
            and terminal["actual_exit"] == 0 and terminal["output_footer_verified"] is True
            and terminal["result_sha256"] == digest(receipt_bytes)
            and terminal["worker_sha256"] == receipt["worker_sha256"] == CENSUS_WORKER
            and receipt["status"] == "GULF_AIS_FROZEN_CENSUS_COMPLETE" and receipt["protocol_sha256"] == protocol
            and receipt["repo"] == REPO and receipt["revision"] == REVISION
            and receipt["target_labels_opened"] is False and receipt["npz_opened"] is False
            and receipt["model_forwards"] == receipt["optimizer_updates"] == 0, "Census closure differs")
    roster_bytes = (args.census / "ROSTER_PRIVATE.json").read_bytes()
    roster = json.loads(roster_bytes)
    require(digest(roster_bytes) == receipt["roster_sha256"] and roster["protocol_sha256"] == protocol
            and roster["repo"] == REPO and roster["revision"] == REVISION
            and roster["catalog_sha256"] == receipt["catalog_sha256"]
            and roster["pilot_pass_excluded"] == "055158-20240811", "Census roster linkage differs")
    items, passes, counts = roster["items"], roster["passes"], receipt["counts"]
    by_pass = Counter(v["pass_id"] for v in items)
    require(0 < len(items) == counts["selected_chips"] and len(passes) == counts["selected_passes"] == 64
            and len({v["pass_id"] for v in passes}) == len(passes)
            and all(0 <= p["selected"] == by_pass[p["pass_id"]] <= 4 for p in passes)
            and set(by_pass) <= {p["pass_id"] for p in passes}
            and "055158-20240811" not in {p["pass_id"] for p in passes}
            and sum(v["size_bytes"] for v in items) == counts["selected_npz_bytes"]
            and len({v["mmsi"] for v in items}) == len({v["path"] for v in items}) == len(items),
            "Census population differs")
    for i, v in enumerate(items, 1):
        m = SCENE.fullmatch(v["scene_id"])
        name = v["mmsi"] + "_" + v["acquisition_time"] + ".npz"
        require(m is not None and v["image_id"] == i and v["pass_id"] == m[4] + "-" + m[2][:8]
                and re.fullmatch(r"\d{9}_\d{8}T\d{6}\.npz", name)
                and v["path"] == f"chips/{v['scene_id']}/{name}" and v["url"] == RESOLVE + v["path"]
                and type(v["size_bytes"]) is int and v["size_bytes"] > 0 and HEX.fullmatch(v["file_sha256"]),
                "Invalid immutable member identity")
    lineage = dict(protocol_sha256=protocol, roster_sha256=digest(roster_bytes),
                   census_result_sha256=digest(receipt_bytes), census_terminal_sha256=digest(terminal_bytes))
    return items, lineage


class FetchFailure(Exception):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


def fetch(url, check, max_bytes):
    # Retry the same immutable member for network, length, digest or server-body failures.
    last = "download_failure"
    for attempt in range(1, ATTEMPTS + 1):
        try:
            req = Request(url, headers={"User-Agent": "SAR-SEAL-Gulf-intake/1", "Accept-Encoding": "identity",
                                        "Cache-Control": "no-cache"})
            with urlopen(req, timeout=40) as response:
                body = response.read(max_bytes + 1)
            if len(body) > max_bytes:
                raise FetchFailure("oversized_response")
            return body, check(body), attempt
        except FetchFailure as exc:
            last = exc.reason
        except (OSError, EOFError, HTTPException) as exc:
            last = "http_" + str(exc.code) if hasattr(exc, "code") else "download_failure"
        if attempt < ATTEMPTS:
            sleep(0.5 * 2**(attempt - 1))
    raise FetchFailure(last)


def validate_xml(body, scene):
    m = SCENE.fullmatch(scene)
    try:
        product = ET.fromstring(body)
        header = product.find("adsHeader")
        image = product.find("imageAnnotation/imageInformation")
        points = product.find("geolocationGrid/geolocationGridPointList")
        require(product.tag == "product" and header is not None and image is not None and points is not None,
                "Missing VV geolocation XML")
        stamp = lambda field: header.findtext(field).split(".")[0].replace("-", "").replace(":", "")
        require(header.findtext("missionId") == m[1] and header.findtext("mode") == "IW"
                and header.findtext("polarisation") == "VV"
                and int(header.findtext("absoluteOrbitNumber")) == int(m[4])
                and stamp("startTime") == m[2] and stamp("stopTime") == m[3], "Wrong source scene")
        width, height = (int(image.findtext(k)) for k in ("numberOfSamples", "numberOfLines"))
        dx, dy = (float(image.findtext(k)) for k in ("rangePixelSpacing", "azimuthPixelSpacing"))
        require(width >= 512 and height >= 512 and dx == dy == 10.0, "Unknown sampling grid")
        nodes = points.findall("geolocationGridPoint")
        coords = set()
        for node in nodes:
            col, row = (int(node.findtext(k)) for k in ("pixel", "line"))
            lon, lat = (float(node.findtext(k)) for k in ("longitude", "latitude"))
            require(0 <= col < width and 0 <= row < height and math.isfinite(lon) and math.isfinite(lat)
                    and -180 <= lon <= 180 and -90 < lat < 90, "Invalid scene geolocation node")
            coords.add((col, row))
        columns, lines = {c for c,r in coords}, {r for c,r in coords}
        require(len(coords) == len(nodes) == int(points.attrib["count"]) == len(columns) * len(lines)
                and len(columns) >= 2 and len(lines) >= 2, "Nonrectangular native sample geolocation")
        return dict(width=width, height=height, range_pixel_spacing_m=dx, azimuth_pixel_spacing_m=dy,
                    geolocation_grid_points=len(coords), geolocation_grid_columns=len(columns),
                    geolocation_grid_lines=len(lines))
    except (ET.ParseError, ValueError, AttributeError, TypeError, KeyError):
        raise FetchFailure("geometry_schema_or_identity") from None


def scene_geometry(scene, output):
    date = SCENE.fullmatch(scene)[2][:8]
    url = f"{S3}/{date[:4]}/{int(date[4:6])}/{int(date[6:])}/IW/DV/{scene}/annotation/iw-vv.xml"
    try:
        body, fields, attempts = fetch(url, lambda raw: validate_xml(raw, scene), 32 * 1024 * 1024)
    except FetchFailure as exc:
        return dict(status="failed", source_url=url, failure_category=exc.reason, attempts=ATTEMPTS)
    rel = f"geometry/{scene}.xml"
    store(output / rel, body)
    return dict(status="ready", source_url=url, xml_path=rel, xml_sha256=digest(body), attempts=attempts, **fields)


def decode_vv(raw):
    # No metadata array is materialized. NumPy reads only this one .npy stream.
    with ZipFile(BytesIO(raw)) as z:
        require(z.namelist().count("chip_vv.npy") == 1, "Ambiguous VV member")
        with z.open("chip_vv.npy") as f:
            vv = np.lib.format.read_array(f, allow_pickle=False)
            require(not f.read(1), "Trailing VV bytes")
    require(vv.dtype == np.dtype("float32") and vv.shape == (512, 512) and np.isfinite(vv).all()
            and np.min(vv) >= 0, "Invalid VV source array")
    positive = vv > 0
    require(positive.any(), "Empty VV response")
    log_vv = np.log1p(vv)
    lo, hi = np.percentile(log_vv[positive], [1, 99.9], method="linear")
    require(hi > lo, "Degenerate VV radiometric range")
    normalized = (log_vv.astype(np.float64) - lo) / (hi - lo)
    gray = np.rint(255.0 * np.clip(normalized, 0, 1)).astype(np.uint8)
    gray[~positive] = 0
    raw_gray = gray.tobytes(order="C")
    img = Image.fromarray(gray)
    data = BytesIO()
    img.save(data, format="PNG", compress_level=6)
    body = data.getvalue()
    with Image.open(BytesIO(body)) as decoded:
        require(decoded.mode == "L" and decoded.size == (512, 512) and decoded.tobytes() == raw_gray,
                "Lossless grayscale readback differs")
        rgb_sha = digest(decoded.convert("RGB").tobytes())
    return body, digest(raw_gray), rgb_sha


def prepare_chip(row, geometry, output):
    base = {k: row[k] for k in ("image_id", "pass_id", "scene_id", "path")}
    base.update(status="failed", modeled=False, failure_category=None, archive_path=None, archive_sha256=None,
                image_path=None, png_sha256=None, gray_sha256=None, rgb_sha256=None)
    def check(body):
        if len(body) != row["size_bytes"]:
            raise FetchFailure("archive_byte_count")
        if digest(body) != row["file_sha256"]:
            raise FetchFailure("archive_sha256")
    try:
        raw, _, attempts = fetch(row["url"], check, row["size_bytes"])
    except FetchFailure as exc:
        return dict(base, failure_category=exc.reason, archive_attempts=ATTEMPTS)
    name = f"{row['image_id']:06d}"
    archive = f"archives/{name}.npz"
    store(output / archive, raw)
    base.update(archive_path=archive, archive_sha256=row["file_sha256"], archive_attempts=attempts)
    try:
        png, gray_sha, rgb_sha = decode_vv(raw)
    except BadZipFile:
        return dict(base, failure_category="invalid_npz")
    except (ValueError, OSError, EOFError, KeyError, RuntimeError):
        return dict(base, failure_category="vv_decode_or_render")
    image = f"images/{name}.png"
    store(output / image, png)
    base.update(image_path=image, png_sha256=digest(png), gray_sha256=gray_sha, rgb_sha256=rgb_sha)
    if geometry[row["scene_id"]]["status"] != "ready":
        return dict(base, failure_category="source_geometry_unavailable")
    return dict(base, status="ready", modeled=True)


def run(args):
    start = perf_counter()
    items, lineage = admit_census(args)
    args.output.mkdir(exist_ok=False)
    for sub in ("images", "archives", "geometry"):
        (args.output / sub).mkdir()
    scenes = sorted({row["scene_id"] for row in items})
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        geometry = dict(zip(scenes, pool.map(lambda s: scene_geometry(s, args.output), scenes), strict=True))
        print(json.dumps(dict(stage="geometry", scenes=len(geometry), ready=sum(
            v["status"] == "ready" for v in geometry.values()))), flush=True)
        rows = []
        for row in pool.map(lambda s: prepare_chip(s, geometry, args.output), items):
            rows.append(row)
            if len(rows) == 1 or len(rows) % 20 == 0 or len(rows) == len(items):
                print(json.dumps(dict(stage="vv", processed=len(rows), selected=len(items))), flush=True)
    require([v["image_id"] for v in rows] == [v["image_id"] for v in items], "Changed locked ordering")
    manifest = encode(dict(**lineage, items=rows, scene_geometry=geometry))
    store(args.output / "INTAKE_PRIVATE.json", manifest)
    ready = [v for v in rows if v["modeled"]]
    counts = dict(selected=len(rows), ready=len(ready), failed=len(rows)-len(ready), selected_scenes=len(geometry),
                  geometry_ready_scenes=sum(v["status"] == "ready" for v in geometry.values()),
                  represented_passes=len({v["pass_id"] for v in ready}),
                  verified_archives=sum(v["archive_sha256"] is not None for v in rows),
                  failure_categories=dict(Counter(v["failure_category"] for v in rows if not v["modeled"])))
    summary = dict(status="GULF_AIS_FROZEN_INTAKE_COMPLETE", phase="intake", **lineage, counts=counts,
                   intake_sha256=digest(manifest), worker_sha256=digest(Path(__file__).read_bytes()),
                   target_labels_opened=False, ais_fields_opened=False, model_forwards=0,
                   detector_forwards=0, optimizer_updates=0, wall_seconds=perf_counter()-start)
    result = encode(summary)
    store(args.output / "RESULT_SAFE.json", result)
    print(json.dumps(dict(status=summary["status"], result_sha256=digest(result))), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for key in ("census", "census-terminal", "protocol", "output"):
        p.add_argument("--" + key, required=True, type=Path)
    p.add_argument("--protocol-sha256", required=True)
    p.add_argument("--workers", type=int, default=10, choices=range(1, 17))
    run(p.parse_args())


if __name__ == "__main__":
    main()
