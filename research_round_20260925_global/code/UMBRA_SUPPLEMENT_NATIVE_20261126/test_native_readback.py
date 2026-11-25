"""Exercise the exact ranged GeoTIFF core and independent local verification with no network."""
from argparse import ArgumentParser
from contextlib import redirect_stdout
from io import BytesIO, StringIO
import json
from pathlib import Path
import tempfile
from urllib.error import HTTPError

import numpy as np
import tifffile

from admission import NATIVE_CORE, READBACK_CORE, READBACK, PARTIAL, load_worker, read_admission
from extract import capture_cohort
from test_admission import mock_case, write
from verify import run_readback


class Response(BytesIO):
    def __init__(self, data, url, status, headers):
        super().__init__(data)
        self.url = url
        self.status = status
        self.headers = headers


class FakeS3:
    def __init__(self, rows):
        self.assets = {}
        geometry = (0.35, 0., 0., 400000., 0., -0.35, 0., 2000000.,
                    0., 0., 1., 0., 0., 0., 0., 1.)
        geokeys = (1, 1, 0, 4, 1024, 0, 1, 1, 1025, 0, 1, 1,
                   3072, 0, 1, 32651, 3076, 0, 1, 9001)
        for i, row in enumerate(rows):
            image = np.full((1024, 1024), i, dtype=np.uint8)
            raw = BytesIO()
            tifffile.imwrite(raw, image, photometric="minisblack", tile=(512, 512),
                             extratags=[(34264, "d", 16, geometry, False),
                                        (34735, "H", len(geokeys), geokeys, False)])
            self.assets[row["author_url"]] = raw.getvalue()
            row["object_bytes"] = len(raw.getvalue())
        self.missing = rows[3]["author_url"]
        self.assets[rows[-1]["author_url"]] = self.assets[rows[-2]["author_url"]]
        self.hdr = rows[0]["etag"]

    def open(self, request, timeout):
        url = request.full_url
        if url == self.missing:
            raise HTTPError(url, 404, "fixture retained unavailable", {}, None)
        blob = self.assets[url]
        headers = {"ETag": self.hdr, "Content-Length": str(len(blob))}
        if request.get_method() == "HEAD":
            return Response(b"", url, 200, headers)
        assert request.get_header("If-match") == self.hdr
        first, last = (int(v) for v in request.get_header("Range")[6:].split("-"))
        headers["Content-Range"] = f"bytes {first}-{last}/{len(blob)}"
        return Response(blob[first:last+1], url, 206, headers)


def main():
    parser = ArgumentParser()
    parser.add_argument("--original-core-dir", required=True, type=Path)
    inputs = parser.parse_args()
    native = load_worker(inputs.original_core_dir/"native_windows.py", NATIVE_CORE)
    verifier = load_worker(inputs.original_core_dir/"verify_native_windows.py", READBACK_CORE)
    with tempfile.TemporaryDirectory() as path:
        here = Path(path)
        args = mock_case(here, ids=range(101, 121))
        manifest = json.loads(args.manifest.read_bytes())
        transport = FakeS3(manifest["items"])
        args.manifest_sha256 = write(args.manifest, manifest)
        metadata = json.loads(args.metadata_result.read_bytes())
        metadata["cohort_private_sha256"] = args.manifest_sha256
        args.metadata_result_sha256 = write(args.metadata_result, metadata)
        terminal = json.loads(args.metadata_terminal.read_bytes())
        terminal["result_sha256"] = args.metadata_result_sha256
        args.metadata_terminal_sha256 = write(args.metadata_terminal, terminal)
        args.published_lock_sha256 = write(args.published_lock, dict(frozen=args.manifest_sha256))
        admitted = read_admission(args)

        args.intake = here/"intake"
        stdout = StringIO()
        with redirect_stdout(stdout):
            assert capture_cohort(admitted, native, args.intake, opener=transport) == 1
        summary = json.loads((args.intake/"RESULT_SAFE.json").read_bytes())
        footer = json.loads(stdout.getvalue())
        assert summary["status"] == PARTIAL
        assert summary["verified_native_windows"] == summary["decoded_unique_windows"] == 18
        assert summary["failed_window_count"] == 2
        assert summary["failure_types"] == {"DuplicateDecodedRaster": 1, "HTTPError": 1}
        assert "UMB-" not in json.dumps(summary)
        records = json.loads((args.intake/"PROVENANCE_PRIVATE.json").read_bytes())["results"]
        assert [x["window_id"] for x in records if x["status"] == "failed"] == ["UMB-104", "UMB-120"]
        assert (args.intake/"images"/"UMB-119.png").is_file()

        args.extraction_result_sha256 = footer["result_sha256"]
        args.extraction_terminal = here/"intake_terminal.json"
        args.extraction_terminal_sha256 = write(args.extraction_terminal, dict(phase="native-window-intake",
            actual_exit=1, output_footer_verified=True, result_sha256=args.extraction_result_sha256,
            worker_sha256=summary["worker_sha256"]))
        args.output = here/"readback"
        with redirect_stdout(StringIO()):
            run_readback(args, admitted, verifier)
        readback = json.loads((args.output/"RESULT_SAFE.json").read_bytes())
        assert readback["status"] == READBACK
        assert readback["eligible_for_detector"] is True
        assert readback["matching_uncompressed_ranges"] == 72
        assert readback["failed_window_count"] == 2 and readback["requested_windows"] == 20
        assert "UMB-" not in json.dumps(readback)
    print("PASS: 20 frozen positions, 18 distinct PNGs, 72 exact original tile hashes, two explicit withheld slots")


if __name__ == "__main__":
    main()
