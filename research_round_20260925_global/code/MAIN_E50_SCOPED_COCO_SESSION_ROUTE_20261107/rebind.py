"""Rebind the original shard-0 spec metadata in one fresh Linux directory."""

import base64
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import sys


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _canonical_path(value):
    _require(type(value) is str and "\\" not in value and "\x00" not in value, "canonical Linux path")
    path = Path(value)
    _require(path.anchor == "/" and str(path) == value and path.resolve() == path, "canonical Linux path")
    return path


def _encode(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")


def rebind(original_ref, fresh_parent, phase):
    """Read one pinned original JSON, then freeze its four-field rebind."""
    _require(sys.platform == "linux", "Linux metadata stage required")
    _require(type(original_ref) is dict and set(original_ref) == {"path", "bytes", "sha256"}
             and type(original_ref["bytes"]) is int and 0 < original_ref["bytes"] <= 2 * 1024**2
             and type(original_ref["sha256"]) is str
             and re.fullmatch(r"[0-9a-f]{64}", original_ref["sha256"]) is not None, "exact original spec reference")
    original_path = _canonical_path(original_ref["path"])
    _require(original_path.is_file() and not original_path.is_symlink(), "ordinary original spec file")
    before = original_path.stat()
    with original_path.open("rb") as stream:
        original_raw = stream.read(original_ref["bytes"] + 1)
    after = original_path.stat()
    identity = lambda st: (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
    _require(identity(before) == identity(after) and len(original_raw) == original_ref["bytes"]
             and hashlib.sha256(original_raw).hexdigest() == original_ref["sha256"], "stable original spec bytes")
    original = json.loads(original_raw)
    _require(type(original) is dict and set(original) == {"schema", "operation", "phase", "entry_source",
             "owner_source", "request", "limits", "sources", "output_directory", "owner_output_directory"}
             and original["schema"] == "ROOT-U74-DGX-fixed50-single-shard-supervision-spec-v1"
             and original["operation"] == "GPU_U74_DGX_FIXED50_SINGLE_SHARD_PREDICTION", "fixed U74 spec envelope")
    request = original["request"]
    _require(type(request) is dict and set(request) == {"worker_source", "prediction_core_source", "source_refs",
             "library_sources", "endpoint_refs", "checkpoint_refs", "manifest_ref", "shard_index", "dataset_range",
             "output_directory"} and type(request["shard_index"]) is int and request["shard_index"] == 0
             and _encode(request["dataset_range"]) == b"[0,47]\n", "original ordered DEV shard 0")
    _require(type(phase) is str and re.fullmatch(r"[A-Za-z0-9_.-]{1,160}", phase) is not None
             and phase != original["phase"], "fresh explicit attempt phase")

    parent = _canonical_path(fresh_parent)
    _require(parent.parent.is_dir() and not parent.exists() and not parent.is_symlink(), "fresh output parent")
    old_outputs = (original["output_directory"], original["owner_output_directory"], request["output_directory"])
    for value in (*old_outputs, original_ref["path"]):
        path = _canonical_path(value)
        _require(parent != path and parent not in path.parents and path not in parent.parents,
                 "new parent is independent of the original paths")

    spec = deepcopy(original)
    outputs = [str(parent / leaf) for leaf in ("ENTRY", "OWNER", "WORKER")]
    spec["phase"] = phase
    spec["output_directory"], spec["owner_output_directory"], spec["request"]["output_directory"] = outputs
    raw = _encode(spec)

    parent.mkdir(mode=0o700)
    _require(all(not Path(value).exists() and not Path(value).is_symlink() for value in outputs), "absent output leaves")
    spec_path = parent / "SPEC_PRIVATE.json"
    with spec_path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    _require(spec_path.read_bytes() == raw, "exact saved corrected spec bytes")
    return dict(status="CORRECTED_SHARD00_SPEC_SAVED_PENDING_ROOT", phase=phase, original_ref=deepcopy(original_ref),
                spec_ref=dict(path=str(spec_path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()),
                base64=base64.b64encode(raw).decode("ascii"), output_directories=outputs,
                changed_fields=["phase", "output_directory", "owner_output_directory", "request.output_directory"],
                launched=False, qualified=False)
