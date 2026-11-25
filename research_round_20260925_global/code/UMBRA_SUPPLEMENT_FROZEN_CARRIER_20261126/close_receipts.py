"""Bind observed frozen supplemental carrier phases without opening author labels."""

from argparse import ArgumentParser
from hashlib import sha256
import json
import os
from pathlib import Path
import sys


PHASES = {
    "capture": "UMBRA_SUPPLEMENT_FROZEN_CAPTURE_COMPLETE",
    "verify-capture": "UMBRA_SUPPLEMENT_FROZEN_CAPTURE_READBACK_COMPLETE",
    "prepare-edits": "UMBRA_SUPPLEMENT_FROZEN_EDITS_PREPARED",
    "verify-edits": "UMBRA_SUPPLEMENT_FROZEN_EDITS_READBACK_COMPLETE",
}
STREAMS = {
    "predictions_sha256": "capture/PREDICTIONS_PRIVATE.jsonl",
    "audit_sha256": "capture/CANDIDATE_AUDIT_PRIVATE.jsonl",
    "edits_sha256": "prepare-edits/EDITS_PRIVATE.jsonl",
}


def need(condition, description):
    if not condition:
        raise ValueError(description)


def digest(raw):
    return sha256(raw).hexdigest()


def file_digest(path):
    value = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def close(root, admission_sha):
    lock_bytes = (root / "ADMISSION_PRIVATE.json").read_bytes()
    need(digest(lock_bytes) == admission_sha, "Exact observed admission required")
    lock = json.loads(lock_bytes)
    worker = file_digest(root / "supplemental.py")
    need(worker == lock["worker_sha256"] and file_digest(root / "PRELOCK.md") == lock["protocol_sha256"],
         "Same frozen worker and protocol")
    bound = dict(worker_sha256=worker, protocol_sha256=lock["protocol_sha256"],
                 admission_sha256=admission_sha, intake_sha256=lock["manifest_sha256"],
                 metadata_result_sha256=lock["metadata_result_sha256"],
                 metadata_terminal_sha256=lock["metadata_terminal_sha256"],
                 intake_result_sha256=lock["intake_result"], intake_terminal_sha256=lock["intake_terminal"],
                 intake_readback_sha256=lock["readback_result"], intake_readback_terminal_sha256=lock["readback_terminal"],
                 anonymous_roster_sha256=lock["anonymous_roster"], reviewer_receipt_sha256=lock["packet_result"],
                 reviewer_terminal_sha256=lock["packet_terminal"], optimizer_updates=0, target_labels_opened=False)
    phases = {}
    results = {}
    for phase, status in PHASES.items():
        terminal_bytes = (root / ("observed_" + phase) / "TERMINAL.json").read_bytes()
        terminal = json.loads(terminal_bytes)
        result_bytes = (root / phase / "RESULT_SAFE.json").read_bytes()
        result = json.loads(result_bytes)
        result_sha = digest(result_bytes)
        need(terminal["phase"] == result["phase"] == phase and result["status"] == status
             and terminal["worker_sha256"] == worker and terminal["actual_exit"] == 0
             and terminal["output_footer_verified"] is True and terminal["result_sha256"] == result_sha
             and all(result[key] == expected for key, expected in bound.items()), "Observed and exact phase: " + phase)
        results[phase] = result
        phases[phase] = dict(result_sha256=result_sha, terminal_sha256=digest(terminal_bytes), actual_exit=0,
                             output_footer_verified=True)

    capture, readback, prepare, verified = [results[phase] for phase in PHASES]
    checkpoint = capture["checkpoint_sha256"]
    n = capture["images"]
    cap = phases["capture"]
    cpu = phases["verify-capture"]
    prior = phases["prepare-edits"]
    stream_hashes = {key: verified[key] for key in STREAMS}
    need(18 <= n <= 24 and all(r["images"] == n and r["checkpoint_sha256"] == checkpoint for r in results.values())
         and capture["model_forward_calls"] == n and readback["model_forward_calls"] == 0
         and readback["exact_native_parity"] is True and readback["counts"] == capture["counts"]
         and readback["capture_sha256"] == cap["result_sha256"],
         "Complete original capture and exact independent readback")
    for result in (prepare, verified):
        need(result["capture_result_sha256"] == cap["result_sha256"]
             and result["capture_terminal_sha256"] == cap["terminal_sha256"]
             and result["capture_readback_result_sha256"] == cpu["result_sha256"]
             and result["capture_readback_terminal_sha256"] == cpu["terminal_sha256"]
             and result["detector_forwards"] == 0 and result["exact_support_and_xor"] is True
             and result["point_head_sha256"] == verified["point_head_sha256"]
             and all(result[k] == v for k, v in stream_hashes.items()),
             "Both action ledgers depend on the exact observed capture")
    need(prepare["exact_independent_replay"] is False and verified["exact_independent_replay"] is True
         and verified["prepared_result_sha256"] == prior["result_sha256"]
         and verified["prepared_terminal_sha256"] == prior["terminal_sha256"], "Exact independent action replay")
    for key in ("predictions_sha256", "audit_sha256"):
        need(capture[key] == readback[key] == stream_hashes[key], "Capture stream identities survive independent replay")
    need(all(file_digest(root / STREAMS[key]) == value for key, value in stream_hashes.items()),
         "Three sealed private streams match the observed result")

    return dict(status="UMBRA_SUPPLEMENT_FROZEN_CARRIER_AND_EDITS_VERIFIED",
                manifest_sha256=bound.pop("intake_sha256"), images=n, phases=phases,
                checkpoint_sha256=checkpoint, point_head_sha256=verified["point_head_sha256"],
                counts=capture["counts"], model_forward_calls=n, exact_native_parity=True,
                exact_support_and_xor=True, exact_independent_replay=True,
                aggregation_worker_sha256=file_digest(Path(__file__).resolve()), **stream_hashes, **bound)


def main():
    parser = ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--admission-sha256", required=True)
    args = parser.parse_args()
    need(sys.flags.isolated and sys.dont_write_bytecode, "Isolated CPU receipt audit required")
    run = Path(__file__).resolve().parent
    output = run / "RESULT_SAFE.json"
    need(not output.exists(), "Fresh closed aggregate receipt")
    summary = close(run, args.admission_sha256)
    raw = (json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with output.open("xb") as handle:
        need(handle.write(raw) == len(raw), "Complete aggregate receipt")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(dict(status=summary["status"], result_sha256=digest(raw))), flush=True)


if __name__ == "__main__":
    main()
