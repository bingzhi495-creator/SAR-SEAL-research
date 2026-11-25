"""Extract original Umbra windows only after a published supplemental acquisition lock."""
from argparse import ArgumentParser
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from admission import COMPLETE, PARTIAL, NATIVE_CORE, admission_flags, digest, load_worker, read_admission, write_json, write_result


class UnsupportedCompression(ValueError):
    """This source cannot support independent original uncompressed bytes."""


def capture_cohort(admitted, native, output, opener=None):
    output.mkdir()
    images = output / "images"
    images.mkdir()
    budget = native.ByteBudget(native.CAP)

    def capture(item):
        ident = item["window_id"]
        pending = images / (ident + ".pending")
        events = []
        try:
            result = native.download_window(item, pending, budget, opener=opener, events=events)
            if result["compression"] != "NONE":
                raise UnsupportedCompression("Original raw tile equality requires uncompressed data")
        except Exception as exc:
            pending.unlink(missing_ok=True)
            return dict(status="failed", window_id=ident, error_type=type(exc).__name__,
                        error=str(exc), network_events=events)
        return dict(status="complete", **result, network_events=events)

    with ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(capture, admitted.items))

    seen = set()
    for row in records:
        if row["status"] != "complete":
            continue
        ident = row["window_id"]
        digest_pixels = row["native_array_sha256"]
        pending = images / (ident + ".pending")
        if digest_pixels in seen:
            pending.unlink()
            row.update(status="failed", error_type="DuplicateDecodedRaster",
                       error="Exact same decoded array as an earlier original window")
        else:
            pending.rename(images / (ident + ".png"))
            seen.add(digest_pixels)

    manifest = admitted.binding["manifest_sha256"]
    failures = Counter(row["error_type"] for row in records if row["status"] == "failed")
    failed = sum(failures.values())
    provenance_sha = write_json(output / "PROVENANCE_PRIVATE.json",
                                dict(manifest_sha256=manifest, results=records))
    summary = dict(status=PARTIAL if failed else COMPLETE, **admitted.binding,
                   worker_sha256=digest(Path(__file__).read_bytes()), native_core_sha256=NATIVE_CORE,
                   provenance_sha256=provenance_sha, requested_windows=len(records),
                   verified_native_windows=len(seen), decoded_unique_windows=len(seen),
                   failed_window_count=failed, failure_types=dict(failures),
                   native_window_shape_yx=[native.WINDOW, native.WINDOW],
                   windows_with_transport_errors=sum(bool(r["network_events"]) for r in records),
                   http_payload_bytes_reserved=budget.used, http_payload_cap_bytes=budget.maximum,
                   annotation_files=0, detector_forwards=0, optimizer_updates=0)
    write_result(output / "RESULT_SAFE.json", summary)
    return int(bool(failed))


def main():
    parser = ArgumentParser(description=__doc__, allow_abbrev=False)
    admission_flags(parser)
    parser.add_argument("--native-core", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    admission = read_admission(args)
    native = load_worker(args.native_core, NATIVE_CORE)
    return capture_cohort(admission, native, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
