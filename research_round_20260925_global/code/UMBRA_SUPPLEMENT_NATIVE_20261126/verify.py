"""Independently verify admitted PNGs and original raw tiles from a closed Umbra intake."""
from argparse import ArgumentParser
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from admission import COMPLETE, PARTIAL, READBACK, NATIVE_CORE, READBACK_CORE
from admission import admission_flags, digest, load_worker, pin, read_admission, require, write_result


def run_readback(args, admission, verifier):
    terminal = json.loads(pin(args.extraction_terminal, args.extraction_terminal_sha256))
    summary = json.loads(pin(args.intake / "RESULT_SAFE.json", args.extraction_result_sha256))
    exit_code = terminal["actual_exit"]
    require(terminal["phase"] == "native-window-intake" and exit_code in (0, 1) and
            terminal["output_footer_verified"] is True and terminal["result_sha256"] == args.extraction_result_sha256 and
            terminal["worker_sha256"] == summary["worker_sha256"], "Observed exact extraction closure")
    require(all(summary.get(key) == value for key, value in admission.binding.items()) and
            summary["native_core_sha256"] == NATIVE_CORE and summary["requested_windows"] == len(admission.items) and
            summary["native_window_shape_yx"] == [1024, 1024] and
            summary["annotation_files"] == summary["detector_forwards"] == summary["optimizer_updates"] == 0,
            "Frozen metadata chain and exact native shape")
    admitted = summary["verified_native_windows"]
    failed = summary["failed_window_count"]
    require(admitted == summary["decoded_unique_windows"] and admitted + failed == len(admission.items) and
            failed == sum(summary["failure_types"].values()) and exit_code == int(bool(failed)) and
            summary["status"] == (PARTIAL if failed else COMPLETE), "Original request and explicit unavailable accounting")

    args.output.mkdir()
    report = args.output / "READBACK_CORE_SAFE.json"
    original_argv = sys.argv
    capture = io.StringIO()
    try:
        sys.argv = [verifier.__file__, "--terminal", str(args.extraction_terminal), "--output", str(args.intake),
                    "--cohort", str(args.manifest), "--result", str(report)]
        with redirect_stdout(capture):
            verifier.main()
    finally:
        sys.argv = original_argv
    independent = json.loads(pin(report, capture.getvalue().strip()))
    require(independent["status"] == "INDEPENDENT_NATIVE_WINDOW_READBACK_COMPLETE" and
            independent["extraction_result_sha256"] == args.extraction_result_sha256 and
            independent["manifest_sha256"] == admission.binding["manifest_sha256"] and
            independent["requested_windows"] == len(admission.items) and
            independent["extraction_actual_exit"] == exit_code and
            independent["readback_windows"] == independent["unique_pixel_arrays"] == admitted and
            independent["matching_uncompressed_ranges"] == 4*admitted,
            "Every original admitted native window has four exact raw source tile hashes")
    value = {**independent, "status": READBACK, "extraction_terminal_sha256": args.extraction_terminal_sha256,
             "worker_sha256": digest(Path(__file__).read_bytes()), "readback_core_sha256": READBACK_CORE,
             "core_readback_sha256": capture.getvalue().strip(), "metadata_terminal_sha256": args.metadata_terminal_sha256,
             "metadata_result_sha256": args.metadata_result_sha256, "published_lock_sha256": args.published_lock_sha256,
             "published_commit": args.published_commit, "failed_window_count": failed, "eligible_for_detector": admitted >= 18,
             "native_window_shape_yx": [1024, 1024], "optimizer_updates": 0}
    write_result(args.output / "RESULT_SAFE.json", value)


def main():
    parser = ArgumentParser(description=__doc__, allow_abbrev=False)
    admission_flags(parser)
    for name in ("extraction-terminal", "intake", "readback-core", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("extraction-terminal-sha256", "extraction-result-sha256"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    admission = read_admission(args)
    verifier = load_worker(args.readback_core, READBACK_CORE)
    run_readback(args, admission, verifier)


if __name__ == "__main__":
    main()
