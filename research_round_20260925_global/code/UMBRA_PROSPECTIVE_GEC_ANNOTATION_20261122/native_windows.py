"""Read complete, centered Umbra GEC windows from a frozen public cohort."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from http.client import IncompleteRead
from io import BytesIO
import json
from math import hypot
from pathlib import Path
import re
from threading import Lock
from time import sleep
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

import numpy as np
from PIL import Image
import tifffile

HOST = "umbra-open-data-catalog.s3.us-west-2.amazonaws.com"
SOURCE_PREFIX = "/sar-data/tasks/ship_detection_testdata/"
HEADER_BYTES = 65536
TILE = 512
WINDOW = 2 * TILE
CAP = 64 * 1024**2


def require(ok, message):
    if not ok:
        raise ValueError(message)


class ByteBudget:
    def __init__(self, maximum):
        self.maximum = maximum
        self.used = 0
        self.lock = Lock()

    def reserve(self, amount):
        with self.lock:
            require(0 < amount <= self.maximum - self.used, "Run range budget exhausted")
            self.used += amount


def origin(length):
    # Round (length - WINDOW) / (2*TILE) to a tile index; ties round down.
    require(length >= WINDOW, "Image smaller than required complete native window")
    nearest = (length - WINDOW + TILE - 1) // (2 * TILE)
    return TILE * min(nearest, length // TILE - 2)


def source_affine(page, window_origin):
    tags = page.tags
    keys = page.geotiff_tags or {}
    epsg = int(keys.get("ProjectedCSTypeGeoKey", 0))
    require(1024 <= epsg <= 32766, "Registered projected EPSG CRS is required")
    raster_type = int(keys.get("GTRasterTypeGeoKey", 1))
    require(raster_type in (1, 2), "Defined GeoTIFF raster-space convention required")
    require(int(keys.get("ProjLinearUnitsGeoKey", 9001)) == 9001,
            "Meter-projected GEC raster required")
    if 34264 in tags:
        matrix = np.asarray(tags[34264].value, dtype=float).reshape(4, 4)
        require(np.isfinite(matrix).all() and tuple(matrix[3]) == (0, 0, 0, 1),
                "Finite affine GeoTIFF transformation required")
        a, b, _, c = matrix[0]
        d, e, _, f = matrix[1]
    else:
        require(33550 in tags and 33922 in tags, "GeoTIFF scale and tiepoint required")
        scales = tags[33550].value
        points = tags[33922].value
        require(len(scales) == 3 and len(points) == 6, "Unambiguous single GeoTIFF tiepoint required")
        a, e = scales[0], -scales[1]
        i, j, _, x, y, _ = points
        b = d = 0.0
        c, f = x - a*i, y - e*j
    require(np.isfinite([a, b, c, d, e, f]).all() and a*e != b*d,
            "Invertible horizontal transformation required")
    y0, x0 = window_origin
    return {
        "crs_epsg": epsg,
        "raster_space": "PixelIsArea" if raster_type == 1 else "PixelIsPoint",
        "absolute_geotiff_tags": {
            str(k): tags[k].value for k in (33550, 33922, 34264, 34735, 34736, 34737, 42113) if k in tags
        },
        "source_affine_ij_to_xy": [float(v) for v in (a, b, c, d, e, f)],
        "window_affine_ij_to_xy": [float(v) for v in
                                   (a, b, c + a*x0 + b*y0, d, e, f + d*x0 + e*y0)],
        "projected_pixel_step_m_xy": [hypot(a, d), hypot(b, e)],
    }


def download_window(item, output, budget, opener=None, events=None):
    url = item["author_url"]
    location = urlsplit(url)
    require(location.scheme == "https" and location.netloc == HOST
            and location.path.startswith(SOURCE_PREFIX) and location.path.endswith("_GEC.tif")
            and not location.query and not location.fragment, "Official public GEC object required")
    if opener is None:
        opener = build_opener(ProxyHandler({}))
    if events is None:
        events = []

    def request(headers, offset=None, length=None):
        operation = "HEAD" if offset is None else "GET"
        for attempt in (1, 2):
            req = Request(url, method=operation, headers=headers)
            if length is not None:
                budget.reserve(length)
            try:
                with opener.open(req, timeout=25) as response:
                    require(response.url == url, "Exact author object URL required")
                    if operation == "HEAD":
                        require(response.status == 200, "Author HEAD must return 200")
                        return int(response.headers["Content-Length"]), response.headers["ETag"]
                    end = offset + length - 1
                    require(response.status == 206 and response.headers["ETag"] == etag
                            and response.headers["Content-Range"] == f"bytes {offset}-{end}/{size}",
                            "Exact same-object byte-range receipt required")
                    chunk = response.read(length + 1)
                    if len(chunk) < length:
                        raise IncompleteRead(chunk, length - len(chunk))
                    require(len(chunk) == length, "Exact byte-range payload required")
                    return chunk
            except (HTTPError, URLError, TimeoutError, ConnectionError, IncompleteRead) as exc:
                status = exc.code if isinstance(exc, HTTPError) else None
                events.append({"operation": operation, "offset": offset, "length": length,
                               "attempt": attempt, "error_type": type(exc).__name__, "http_status": status})
                if attempt == 2 or (status is not None and status not in (408, 429, 500, 502, 503, 504)):
                    raise
                sleep(0.5)

    size, etag = request({})
    require(size >= HEADER_BYTES and etag, "Complete stable author HEAD required")
    require(item.get("etag", etag) == etag and item.get("object_bytes", size) == size,
            "Author object differs from manifest")

    def fetch(offset, length):
        require(0 <= offset < offset + length <= size, "Native range exceeds object extent")
        return request({"Range": f"bytes={offset}-{offset + length - 1}", "If-Match": etag}, offset, length)

    head = fetch(0, HEADER_BYTES)
    native = np.empty((WINDOW, WINDOW), dtype=np.uint8)
    source_ranges = []
    with tifffile.TiffFile(BytesIO(head)) as tiff:
        page = tiff.pages[0]
        h, w = page.imagelength, page.imagewidth
        orientation = int(page.tags[274].value) if 274 in page.tags else 1
        require(page.is_tiled and page.samplesperpixel == 1 and page.dtype == np.dtype("uint8")
                and (page.tilelength, page.tilewidth) == (TILE, TILE)
                and page.photometric.name == "MINISBLACK" and orientation == 1,
                "Native single-band top-left uint8 tiled GEC required")
        y0, x0 = origin(h), origin(w)
        referencing = source_affine(page, (y0, x0))
        nx, ny = (w + TILE - 1) // TILE, (h + TILE - 1) // TILE
        require(len(page.dataoffsets) == len(page.databytecounts) == nx*ny,
                "Complete native tile offset and count tables required")
        for dy, dx in ((0, 0), (0, TILE), (TILE, 0), (TILE, TILE)):
            y, x = y0 + dy, x0 + dx
            index = (y // TILE) * nx + x // TILE
            offset, length = page.dataoffsets[index], page.databytecounts[index]
            raw = fetch(offset, length)
            decoded, indices, _ = page.decode(raw, index, jpegtables=page.jpegtables)
            tile = np.squeeze(decoded)
            require(tile.shape == (TILE, TILE) and tile.dtype == np.uint8
                    and tuple(indices[-3:-1]) == (y, x), "Exact native decoded tile geometry required")
            if page.compression.name == "NONE":
                require(tile.tobytes() == raw, "Independent native unsigned-byte tile equality")
            native[dy:dy + TILE, dx:dx + TILE] = tile
            source_ranges.append({"offset": offset, "length": length, "row_col": [y, x],
                                  "sha256": sha256(raw).hexdigest()})
        compression = page.compression.name

    encoded = BytesIO()
    Image.fromarray(native).save(encoded, format="PNG")
    png = encoded.getvalue()
    with output.open("xb") as stream:
        stream.write(png)
    pixel_sha = sha256(native.tobytes()).hexdigest()
    with Image.open(output) as reference:
        verify = np.asarray(reference)
        require(reference.mode == "L" and verify.shape == native.shape and verify.dtype == native.dtype
                and sha256(verify.tobytes()).hexdigest() == pixel_sha, "Independent saved PNG pixel readback")
    return {
        **referencing,
        "window_id": item["window_id"], "author_url": url, "object_bytes": size, "etag": etag,
        "full_raster_shape_yx": [h, w], "native_origin_yx": [y0, x0], "native_shape_yx": [WINDOW, WINDOW],
        "native_dtype": "|u1", "compression": compression, "source_ranges": source_ranges,
        "native_array_sha256": pixel_sha, "png_sha256": sha256(png).hexdigest(),
        "source_header_sha256": sha256(head).hexdigest(), "source_header_bytes": len(head),
    }


def save_document(path, payload):
    data = (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with path.open("xb") as out:
        out.write(data)
    return sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--cohort-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    digest = args.cohort_sha256.lower()
    roster_bytes = args.cohort.read_bytes()
    require(re.fullmatch(r"[0-9a-f]{64}", digest) and sha256(roster_bytes).hexdigest() == digest,
            "Frozen cohort manifest SHA256 required before native reads")
    roster = json.loads(roster_bytes)
    require(roster["schema"] == "umbra_gec_cohort_v1", "Cohort schema")
    items = roster["items"]
    ids = [item["window_id"] for item in items]
    require(8 <= len(ids) <= 12 and ids == [f"UMB-{i:03d}" for i in range(1, len(ids)+1)]
            and len({item["author_url"] for item in items}) == len(ids), "Frozen ordered unique cohort")
    args.output.mkdir()
    images = args.output / "images"
    images.mkdir()
    budget = ByteBudget(CAP)

    def capture(item):
        ident = item["window_id"]
        events = []
        try:
            value = download_window(item, images / (ident + ".png"), budget, events=events)
        except Exception as exc:
            return {"status": "failed", "window_id": ident, "error_type": type(exc).__name__,
                    "error": str(exc), "network_events": events}
        return {"status": "complete", **value, "network_events": events}

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(capture, items))
    failures = [r for r in results if r["status"] == "failed"]
    complete = [r for r in results if r["status"] == "complete"]
    provenance_sha = save_document(args.output / "PROVENANCE_PRIVATE.json",
                                   {"manifest_sha256": digest, "results": results})
    safe = {
        "status": "UMBRA_NATIVE_WINDOWS_COMPLETE" if not failures else "UMBRA_NATIVE_WINDOWS_PARTIAL",
        "manifest_sha256": digest, "provenance_sha256": provenance_sha,
        "requested_windows": len(results), "verified_native_windows": len(complete),
        "native_window_shape_yx": [WINDOW, WINDOW],
        "failed_windows": [{"window_id": r["window_id"], "error_type": r["error_type"]} for r in failures],
        "windows_with_transport_errors": sum(bool(r["network_events"]) for r in results),
        "http_payload_bytes_reserved": budget.used, "http_payload_cap_bytes": budget.maximum,
        "decoded_unique_windows": len({r["native_array_sha256"] for r in complete}),
        "annotation_files": 0, "detector_forwards": 0, "optimizer_updates": 0,
    }
    print(json.dumps({"status": safe["status"], "result_sha256":
                      save_document(args.output / "RESULT_SAFE.json", safe)}), flush=True)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
