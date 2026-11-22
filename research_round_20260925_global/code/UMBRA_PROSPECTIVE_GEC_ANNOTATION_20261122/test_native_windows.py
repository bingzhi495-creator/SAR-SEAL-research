from hashlib import sha256
from io import BytesIO
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import tifffile

from native_windows import ByteBudget, download_window, save_document


class Response(BytesIO):
    def __init__(self, data, url, status, headers):
        super().__init__(data)
        self.url = url
        self.status = status
        self.headers = headers


class SyntheticObject:
    etag = '"synthetic"'

    def __init__(self, blob):
        self.blob = blob
        self.requests = []

    def open(self, request, timeout):
        url = request.full_url
        headers = {"ETag": self.etag, "Content-Length": str(len(self.blob))}
        if request.get_method() == "HEAD":
            return Response(b"", url, 200, headers)
        self.assert_if_match(request.get_header("If-match"))
        first, last = (int(v) for v in request.get_header("Range")[6:].split("-"))
        self.requests.append((first, last))
        headers["Content-Range"] = f"bytes {first}-{last}/{len(self.blob)}"
        return Response(self.blob[first:last + 1], url, 206, headers)

    def assert_if_match(self, value):
        if value != self.etag:
            raise AssertionError("Version-independent range request")


class WindowRoundtrip(unittest.TestCase):
    def test_nearest_complete_1024_window_preserves_pixels_and_affine(self):
        h, w = 2560, 3072
        data = np.random.default_rng(17).integers(0, 256, size=(h, w), dtype=np.uint8)
        transform = (0.35, 0.02, 0.0, 400000.0,
                     0.01, -0.35, 0.0, 2000000.0,
                     0.0, 0.0, 1.0, 0.0,
                     0.0, 0.0, 0.0, 1.0)
        geokeys = (1, 1, 0, 4, 1024, 0, 1, 1, 1025, 0, 1, 1,
                   3072, 0, 1, 32651, 3076, 0, 1, 9001)
        f = BytesIO()
        tifffile.imwrite(f, data, photometric="minisblack", tile=(512, 512),
                         extratags=[(34264, "d", 16, transform, False),
                                    (34735, "H", len(geokeys), geokeys, False)])
        remote = SyntheticObject(f.getvalue())
        item = {"window_id": "UMB-001", "author_url":
                "https://umbra-open-data-catalog.s3.us-west-2.amazonaws.com/"
                "sar-data/tasks/ship_detection_testdata/synthetic/fake_GEC.tif"}
        with tempfile.TemporaryDirectory() as storage:
            png = Path(storage) / "UMB-001.png"
            record = download_window(item, png, ByteBudget(2**21), opener=remote)
            independently_expected = np.ascontiguousarray(data[512:1536, 1024:2048])
            with Image.open(png) as image:
                self.assertEqual(image.mode, "L")
                self.assertTrue(np.array_equal(np.array(image), independently_expected))
            self.assertEqual(record["native_origin_yx"], [512, 1024])
            self.assertEqual(record["native_array_sha256"],
                             sha256(independently_expected.tobytes()).hexdigest())
            self.assertEqual(record["crs_epsg"], 32651)
            self.assertEqual(record["raster_space"], "PixelIsArea")
            self.assertEqual(len(save_document(Path(storage) / "provenance.json", record)), 64)
            np.testing.assert_allclose(
                np.array(record["window_affine_ij_to_xy"]),
                [0.35, 0.02, 400000.0 + 0.35*1024 + 0.02*512,
                 0.01, -0.35, 2000000.0 + 0.01*1024 - 0.35*512],
            )
        self.assertEqual(len(remote.requests), 5)
        self.assertEqual(sum(end-start+1 for start, end in remote.requests), 1114112)


if __name__ == "__main__":
    unittest.main()
