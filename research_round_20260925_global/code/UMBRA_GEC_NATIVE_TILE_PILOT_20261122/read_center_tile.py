"""Bounded read of one deterministic native centre block per official Umbra GEC product."""
import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

import numpy as np
import tifffile

HOST = "umbra-open-data-catalog.s3.us-west-2.amazonaws.com"
PREFIX = "/sar-data/tasks/ship_detection_testdata/"
CAP = 2 * 1024**2


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_product(url, output_tile, budget):
    loc = urlsplit(url)
    require(loc.scheme == "https" and loc.netloc == HOST and loc.path.startswith(PREFIX)
            and loc.path.endswith("_GEC.tif") and not loc.query and not loc.fragment, "Official public GEC path")
    opener = build_opener(ProxyHandler({}))
    with opener.open(Request(url, method="HEAD"), timeout=25) as response:
        size = int(response.headers["Content-Length"])
        etag = response.headers["ETag"]
        require(response.status == 200 and response.url == url and etag, "Stable author object HEAD")

    def fetch(offset, length):
        require(length > 0 and offset >= 0 and offset + length <= size and length <= budget[0], "Bounded source range")
        budget[0] -= length
        end = offset + length - 1
        request = Request(url, headers={"Range": f"bytes={offset}-{end}", "If-Match": etag})
        with opener.open(request, timeout=25) as r:
            require(r.status == 206 and r.url == url and r.headers["ETag"] == etag and
                    r.headers["Content-Range"] == f"bytes {offset}-{end}/{size}", "Exact same-object partial response")
            data = r.read(length + 1)
        require(len(data) == length, "Whole pinned interval received")
        return data

    head = fetch(0, 65536)
    with tifffile.TiffFile(BytesIO(head)) as tiff:
        page = tiff.pages[0]
        h, w = page.imagelength, page.imagewidth
        th, tw = page.tilelength, page.tilewidth
        require(page.is_tiled and page.samplesperpixel == 1 and (th, tw) == (512, 512), "Single-band tiled GEC page")
        x, y = (w // 2) // tw, (h // 2) // th
        require((x + 1) * tw <= w and (y + 1) * th <= h, "Complete center tile")
        index = y * ((w + tw - 1) // tw) + x
        offset, length = page.dataoffsets[index], page.databytecounts[index]
        pixels = fetch(offset, length)
        repeat = fetch(offset, length)
        require(pixels == repeat, "Independent repeat request must agree byte-for-byte")
        decoded, indices, _ = page.decode(pixels, index, jpegtables=page.jpegtables)
        native = np.ascontiguousarray(np.squeeze(decoded))
        require(native.shape == (th, tw) and native.dtype == page.dtype and
                tuple(indices[-3:-1]) == (y * th, x * tw), "Native tile geometry and dtype")
        require(np.isfinite(native).all() and int(np.count_nonzero(native)) > 0
                and int(native.min()) < int(native.max()), "Nonconstant finite native sample")
        with output_tile.open("xb") as storage:
            np.save(storage, native, allow_pickle=False)
        read_back = np.load(output_tile, allow_pickle=False)
        require(np.array_equal(native, read_back) and native.dtype == read_back.dtype, "Decoded array readback")
        geotiff = page.geotiff_tags or {}
        return {"author_url": url, "object_bytes": size, "etag": etag, "shape": [h, w],
                "native_tile_shape": [th, tw], "native_tile_origin_yx": [y * th, x * tw],
                "dtype": native.dtype.str, "compression": page.compression.name,
                "projected_crs": (int(geotiff["ProjectedCSTypeGeoKey"]) if "ProjectedCSTypeGeoKey" in geotiff else None),
                "pixel_scale": (list(page.tags[33550].value) if 33550 in page.tags else None),
                "range_offset": offset, "range_length": length,
                "source_range_sha256": sha256(pixels).hexdigest(),
                "decoded_native_sha256": sha256(native.tobytes()).hexdigest(),
                "min_value": int(native.min()), "max_value": int(native.max())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--author-url", action="append", required=True)
    p.add_argument("--output", type=Path, required=True)
    options = p.parse_args()
    require(1 <= len(options.author_url) <= 2, "Technical pilot of at most two preselected author products")
    options.output.mkdir()
    available = [CAP]
    proven = [read_product(url, options.output / f"tile_{n}.npy", available)
              for n, url in enumerate(options.author_url, start=1)]
    data = (json.dumps({"status": "NATIVE_TECHNICAL_PILOT_COMPLETE", "products": proven,
                        "bounded_http_bytes": CAP - available[0], "label_files": 0, "detector_forwards": 0,
                        "role": "input qualification only"}, sort_keys=True, indent=2) + "\n").encode()
    with (options.output / "RESULT_SAFE.json").open("xb") as stream:
        stream.write(data)
    print(sha256(data).hexdigest(), flush=True)


if __name__ == "__main__":
    main()

