"""Bind naturally closed spatial cue acquisition for the FIT worker."""
from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path


def read(path, expected=None):
    raw = path.read_bytes()
    digest = sha256(raw).hexdigest()
    if expected is not None and digest != expected:
        raise ValueError("Actual approved result digest differs for " + path.name)
    return json.loads(raw), digest


def terminal(path, phase, result):
    observed, digest = read(path)
    if (observed["phase"] != phase or observed["actual_exit"] != 0
            or observed["output_footer_verified"] is not True or observed["result_sha256"] != result):
        raise ValueError("Actual terminal closure is absent or unverified for " + phase)
    return observed, digest


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument("--capture", type=Path, required=True)
    p.add_argument("--capture-result", required=True)
    p.add_argument("--readback-result", required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    cap, cap_end = terminal(a.capture/"observed_capture_active/TERMINAL.json", "capture", a.capture_result)
    rb, rb_end = terminal(a.capture/"observed_readback/TERMINAL.json", "verify", a.readback_result)
    cap_doc, result_sha = read(a.capture/"actual/RESULT_SAFE.json", a.capture_result)
    rb_doc, readback_sha = read(a.capture/"readback/RESULT_SAFE.json", a.readback_result)
    point_result = cap_doc["point_result_sha256"]
    if (cap_doc["worker_sha256"] != cap["worker_sha256"] or rb_doc["worker_sha256"] != rb["worker_sha256"]
            or rb_doc["spatial_capture_result_sha256"] != result_sha
            or rb_doc["spatial_capture_terminal_sha256"] != cap_end or rb_doc["point_result_sha256"] != point_result
            or cap_doc["partitions"] != rb_doc["partitions"]
            or any(doc[k] for doc in (cap_doc, rb_doc)
                   for k in ("AP", "train_labels_opened", "test_labels_opened", "test_images_opened", "optimizer_updates"))):
        raise ValueError("Complete original scope and independent cue readback required")
    value = dict(status="FIT_SPATIAL_CUE_ACQUISITION_LOCKED", cue_capture_result_sha256=result_sha,
                 cue_capture_terminal_sha256=cap_end, cue_capture_worker_sha256=cap["worker_sha256"],
                 cue_capture_readback_sha256=readback_sha, cue_capture_readback_terminal_sha256=rb_end,
                 cue_capture_readback_worker_sha256=rb["worker_sha256"], point_result_sha256=point_result,
                 cue_fit_index_sha256=cap_doc["partitions"]["fit"]["cue_index_sha256"],
                 source_gate_labels_opened=False, historical_test_opened=False)
    payload = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with a.out.open("xb") as stream:
        stream.write(payload)
    print(json.dumps(dict(status=value["status"], acquisition_lock_sha256=sha256(payload).hexdigest())),flush=True)


if __name__ == "__main__":
    main()
