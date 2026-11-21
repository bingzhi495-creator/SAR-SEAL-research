"""Independently verify the completed native official HRSID resumed export after its owner records exit."""

import argparse
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import re
import sys
from types import ModuleType


OLD_EXPORT = "0f9aa1f4be24d604d38db965b8c038031278c669c1a0f10f91174c925b6e4423"
OLD_ADMISSION = "2fa42819d62f80dfe49eef9ee5fb2dfe087aa298cebdb26537df3379c3d28de3"
RESUME = "a5d87562e2858a64194cc204e2f220c073773730407bb010b9865066f25b80cd"
VERIFY = "27c3f2685c8aa2c6e6443b5eeb003a7da20f2cdf6391167bc386e3e097b11784"
WITNESS = "e0f68b0a066a0fe1373d369cfb98db406275a015aeaf2b11a0f97380c9566240"
MANIFEST = "dd42cbf5ef415dc152bab9160c5eb385ea33255d697fba2dddf49917eda3ee1b"
ARCHIVE = "f089fe3417a6c58f3e71f4bc35f8895ac6b136f29c58191b4a90442ab630d159"
ENDPOINT = "011bd30ed056c871be1d61e14d71b3b2b258330a9d20cf2f164ca4a7c4f23c11"
INPUT = "15aabe26b7580c55a5b5f3e4ae1558cac84294ae950f283c826de5b0720e53ee"
PREFIX = "0cc3af480888e50da9d01e1a854a68f22e06d9e9456fc6ddc36975bb7e6441d8"
N_PREFIX, PREFIX_BYTES, N_IMAGES = 1134, 18195008, 1962
PHASE = "HRSID_OFFICIAL_TEST_NATIVE_INTERRUPTED_EXPORT_RESUME_V1"
COMPLETE = "HRSID_OFFICIAL_TEST_NATIVE_RESUMED_EXPORT_COMPLETE"
OWNER_STATUS = "HRSID_OFFICIAL_TEST_NATIVE_EXPORT_RESUME_TERMINAL"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def valid_hash(value):
    return type(value) is str and re.fullmatch("[0-9a-f]{64}", value) is not None


def pinned(path, expected):
    require(valid_hash(expected), "Expected exact lowercase SHA-256")
    with path.open("rb") as handle:
        raw = handle.read(1024**2 + 1)
    require(0 < len(raw) <= 1024**2 and sha256(raw).hexdigest() == expected, "Closed input digest for " + path.name)
    return raw


def check_terminal(receipt, launch_digest):
    require(receipt.get("status") == OWNER_STATUS and receipt.get("actual_process_terminal") is True
            and type(receipt.get("actual_exit")) is int and receipt["actual_exit"] == 0
            and "original_interrupted_exit" in receipt and receipt["original_interrupted_exit"] is None
            and receipt.get("footer_status") == COMPLETE and receipt.get("resume_source_sha256") == RESUME
            and valid_hash(launch_digest) and receipt.get("launch_request_sha256") == launch_digest
            and valid_hash(receipt.get("result_sha256")), "Successful observed resumed worker and pinned owner request")
    require(all(type(receipt.get(k)) is int and receipt[k] > 0 for k in ("worker_pid", "owner_pid"))
            and receipt["owner_pid"] != receipt["worker_pid"], "Distinct recorded worker and owner PIDs")


def check_result(result):
    fixed = dict(status=COMPLETE, phase=PHASE, device="cpu", precision="float32",
                 original_export_source_sha256=OLD_EXPORT, original_admission_sha256=OLD_ADMISSION,
                 resume_source_sha256=RESUME, prefix_verifier_source_sha256=VERIFY,
                 missing_witness_sha256=WITNESS, manifest_sha256=MANIFEST, archive_sha256=ARCHIVE,
                 endpoint_sha256=ENDPOINT, input_readback_sha256=INPUT, prefix_sha256=PREFIX)
    counts = dict(images=N_IMAGES, model_forward_calls=831, suffix_model_forward_calls=828,
                  replay_model_forward_calls=3, prefix_images=N_PREFIX, prefix_bytes=PREFIX_BYTES,
                  discarded_tail_bytes=0, original_session_handle=88756, native_joint_rows=196200, optimizer_updates=0)
    require(all(result.get(k) == v for k, v in fixed.items())
            and all(type(result.get(k)) is int and result[k] == v for k, v in counts.items())
            and result.get("TEST") is True and all(result.get(k) is False for k in ("FIT", "DEV", "GT_opened", "AP"))
            and "original_exit" in result and result["original_exit"] is None
            and result.get("replay_indices_zero_based") == [0, 567, 1133]
            and all(type(i) is int for i in result["replay_indices_zero_based"]),
            "Fixed export identity, prefix, replay and suffix accounting")
    require(type(result.get("mapped_rows")) is int and 0 <= result["mapped_rows"] <= 196200
            and type(result.get("prefix_mapped_rows")) is int and 0 <= result["prefix_mapped_rows"] <= N_PREFIX * 100
            and result["prefix_mapped_rows"] <= result["mapped_rows"] and valid_hash(result.get("predictions_sha256"))
            and type(result.get("prediction_bytes")) is int and result["prediction_bytes"] > PREFIX_BYTES,
            "Reported canonical prediction bounds")


def verify_stream(stream, manifest, result, verify, prefix_images, prefix_bytes, prefix_digest, duplicates):
    prefix_hash = sha256()
    count_bytes = count_mapped = seen = 0

    class AuditedStream:
        def readline(self):
            nonlocal count_bytes, count_mapped, seen
            line = stream.readline()
            require(line.endswith(b"\n"), "Complete canonical image block")
            block = json.loads(line)
            require(list(block) == ["image_id", "raw_query_count", "native_count", "unmapped_slot1_count", "native_ranks", "coco"]
                    and (json.dumps(block, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode() == line,
                    "Original native block key order and canonical bytes")
            for row in block["coco"]:
                require(list(row) == ["image_id", "category_id", "score", "segmentation"]
                        and list(row["segmentation"]) == ["size", "counts"]
                        and type(row["score"]) is float and all(type(i) is int for i in row["segmentation"]["size"]),
                        "Original native COCO row representation")
            if seen < prefix_images:
                prefix_hash.update(line)
                count_bytes += len(line)
                count_mapped += len(block["coco"])
            seen += 1
            return line

        def read(self, size):
            return stream.read(size)

    metrics = verify.check_stream(AuditedStream(), manifest, result)
    require(seen == len(manifest) and 0 < prefix_images <= seen
            and (count_bytes, count_mapped, prefix_hash.hexdigest()) ==
            (prefix_bytes, result["prefix_mapped_rows"], prefix_digest),
            "Independent recovered prefix SHA, complete line boundary and mapped row count")
    require(metrics["identical_duplicate_prediction_pairs"] == duplicates
            and metrics["mapped_rows"] + metrics["unmapped_slots"] == 100 * seen,
            "Independent duplicate-image predictions and native slot census")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("terminal", "launch-request", "result", "manifest", "predictions", "stream-verifier", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    for key in ("terminal-sha256", "launch-request-sha256"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--expected-owner-pid", type=int, default=32008)
    args = parser.parse_args()
    receipt = json.loads(pinned(args.terminal, args.terminal_sha256))
    check_terminal(receipt, args.launch_request_sha256)
    require(receipt["owner_pid"] == args.expected_owner_pid, "Independently observed owner PID")
    pinned(args.launch_request, args.launch_request_sha256)
    result = json.loads(pinned(args.result, receipt["result_sha256"]))
    check_result(result)
    require(sys.flags.isolated and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1",
            "Isolated CPU readback")
    require(not args.output.exists() and args.output.parent.is_dir(), "Fresh independent closure")
    verify = ModuleType("_frozen_native_stream_verifier")
    verify.__file__ = str(args.stream_verifier)
    exec(compile(pinned(args.stream_verifier, VERIFY), str(args.stream_verifier), "exec"), verify.__dict__)
    runtime = dict(python=platform.python_version(), **{name:version(name) for name in verify.RUNTIME if name != "python"})
    require(runtime == verify.RUNTIME, "Exact original native runtime distributions")
    manifest = json.loads(pinned(args.manifest, MANIFEST))
    require(len(manifest) == len({r["image_id"] for r in manifest}) == N_IMAGES
            and all(type(r["image_id"]) is int and r["width"] == r["height"] == 800
                    and valid_hash(r["decoded_rgb_sha256"]) for r in manifest)
            and len({r["decoded_rgb_sha256"] for r in manifest}) == N_IMAGES - 1, "Complete official image manifest")
    with args.predictions.open("rb") as handle:
        before = os.fstat(handle.fileno())
        require(before.st_size == result["prediction_bytes"], "Closed prediction size")
        metrics = verify_stream(handle, manifest, result, verify, N_PREFIX, PREFIX_BYTES, PREFIX, 1)
        after = os.fstat(handle.fileno())
        require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
                "Stable closed predictions during independent full readback")
    closure = dict(status="HRSID_OFFICIAL_TEST_NATIVE_EXPORT_INDEPENDENT_READBACK_ACCEPTED", phase=PHASE, resume_mode=True,
                   original_process_terminal=receipt["actual_process_terminal"], original_exit=receipt["actual_exit"],
                   original_exit_applies_to="resumed_export_worker_pid",
                   original_process=dict(role="resumed_export_worker", pid=receipt["worker_pid"], exit_code=receipt["actual_exit"]),
                   original_interrupted_process=dict(tool_session_handle=88756, exit_code=None, terminal_observed=False),
                   original_interrupted_exit=None, actual_resume_exit=receipt["actual_exit"], actual_resume_pid=receipt["worker_pid"],
                   owner_pid=receipt["owner_pid"], terminal_sha256=args.terminal_sha256,
                   launch_request_sha256=args.launch_request_sha256, result_sha256=receipt["result_sha256"],
                   resume_source_sha256=RESUME, original_export_source_sha256=OLD_EXPORT, original_admission_sha256=OLD_ADMISSION,
                   missing_witness_sha256=WITNESS, prefix_sha256=PREFIX, prefix_images=N_PREFIX, prefix_bytes=PREFIX_BYTES,
                   prefix_mapped_rows=result["prefix_mapped_rows"], discarded_tail_bytes=0,
                   replay_indices_zero_based=[0, 567, 1133], suffix_model_forward_calls=828, replay_model_forward_calls=3,
                   endpoint_sha256=ENDPOINT, archive_sha256=ARCHIVE, input_readback_sha256=INPUT, manifest_sha256=MANIFEST,
                   stream_verifier_source_sha256=VERIFY, independent_source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                   images=N_IMAGES, GT_opened=False, AP=False, independent_model_forwards=0, independent_optimizer_updates=0,
                   runtime_versions=runtime, **metrics)
    raw = (json.dumps(closure, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with args.output.open("xb") as dest:
        require(dest.write(raw) == len(raw), "Complete independent closure")
        dest.flush()
        os.fsync(dest.fileno())
    print(json.dumps(dict(status=closure["status"], readback_sha256=sha256(raw).hexdigest())), flush=True)


if __name__ == "__main__":
    main()
