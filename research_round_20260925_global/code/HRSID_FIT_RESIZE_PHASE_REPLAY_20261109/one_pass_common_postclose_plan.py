"""Build a P260 plan after a separately waited B/C DEV705 owner has closed."""
from hashlib import sha256
import json
from math import isfinite
import os
from pathlib import Path
import re
import stat
from types import ModuleType


P260_PATH = "HRSID_FIT_RESIZE_PHASE_REPLAY_20261109/one_pass_common_score_pair.py"
P260_SHA = "46bcb96d6b4b4fd34184fe5976588a46427c51d149a71ca05ac9d794fce66707"
LOCK_STATUS = "HRSID_ONE_PASS_COMMON_FIT_DEV705_OWNER_LOCK_V1"
COMPLETION_STATUS = "HRSID_ONE_PASS_COMMON_FIT_DEV705_OWNER_WORKERS_CLOSED_PENDING_OWNER_WAIT_V1"
OWNER_EVENT = "HRSID_ONE_PASS_COMMON_FIT_DEV705_WORKERS_CLOSED"
MODES = ("stepwise_center", "one_pass")
OPERATIONS = ("export", "score")
MIB = 1024**2
PAIR_FIELDS = ("training_protocol", "implementation_commit", "qualification_pair_sha256", "training_pair",
               "stepwise_center_arm_receipt", "one_pass_arm_receipt", "gt")


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def exact(value, fixed, extras=(), *, reason):
    require(type(value) is dict and set(value) == set(fixed) | set(extras)
            and all(type(value[key]) is type(expected) and value[key] == expected for key, expected in fixed.items()),
            reason)


def unique(items):
    value = dict(items)
    require(len(value) == len(items), "Duplicate closed evidence JSON key")
    return value


def reject_constant(token):
    raise ValueError("Nonfinite closed evidence JSON value " + token)


def document(raw):
    value = json.loads(raw, object_pairs_hook=unique, parse_constant=reject_constant)
    require(type(value) is dict, "Expected closed evidence JSON object")
    return value


def digest(value, length=64):
    return type(value) is str and re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is not None


def read_ref(ref, limit, *, path=None, allow_empty=False, capture=True):
    """Check original closed bytes and file identity without trusting mutable metadata."""
    exact(ref, {}, ("path", "bytes", "sha256"), reason="Exact closed path, size and digest required")
    require(type(ref["path"]) is str and type(ref["bytes"]) is int and digest(ref["sha256"])
            and (0 if allow_empty else 1) <= ref["bytes"] <= limit, "Bounded original artifact reference required")
    target = Path(ref["path"])
    require(target.is_absolute() and target.resolve(strict=True) == target
            and ref["path"] in (str(target), target.as_posix()) and (path is None or target == path),
            "Exact canonical original artifact path required")

    def signature(info):
        return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns

    hasher = sha256()
    chunks = [] if capture else None
    total = 0
    with target.open("rb") as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode) and before.st_size == ref["bytes"], "Original artifact size differs")
        for chunk in iter(lambda: stream.read(min(MIB, ref["bytes"] - total + 1)), b""):
            total += len(chunk)
            require(total <= ref["bytes"], "Original artifact grew during readback")
            hasher.update(chunk)
            if capture:
                chunks.append(chunk)
        require(signature(before) == signature(os.fstat(stream.fileno())) == signature(target.stat()),
                "Original artifact identity changed during readback")
    require(total == ref["bytes"] and hasher.hexdigest() == ref["sha256"], "Original artifact digest or size differs")
    return b"".join(chunks) if capture else None


def birth(value):
    exact(value, {}, ("pid", "creation_time"), reason="Exact original process birth required")
    require(type(value["pid"]) is int and value["pid"] > 0 and type(value["creation_time"]) is float
            and isfinite(value["creation_time"]) and value["creation_time"] > 0, "Invalid original process birth")
    return value


def elapsed(value):
    return type(value) in (float, int) and isfinite(value) and value > 0


def load_p260(code):
    require(code.is_absolute() and code.resolve(strict=True) == code, "Canonical ordinary code root required")
    path = code / P260_PATH
    require(path.resolve(strict=True) == path, "Canonical original P260 source required")
    with path.open("rb") as stream:
        raw = stream.read(64 * 1024 + 1)
    require(0 < len(raw) <= 64 * 1024 and sha256(raw).hexdigest() == P260_SHA,
            "Exact immutable P260 bridge required")
    module = ModuleType("_bc_postclose_p260")
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def _owner_closed(terminal_ref, owner, is_live):
    require(is_live(owner) is False, "Original B/C export/score owner is still live")
    control = Path(terminal_ref["path"]).parent
    terminal = document(read_ref(terminal_ref, MIB, path=control / "OWNER_TERMINAL_PRIVATE.json"))
    exact(terminal, dict(owner=owner, original_os_wait=True, original_os_exit=0, job_assigned=True,
                         job_active_before_drain=0, job_active_after_wait=0, stop_reason=None),
          ("wall_seconds", "stdout_ref", "stderr_ref"), reason="Exact original owner OS wait and drained JobObject required")
    birth(terminal["owner"])
    require(elapsed(terminal["wall_seconds"]), "Exact original owner wall duration required")
    stdout = read_ref(terminal["stdout_ref"], MIB, path=control / "OWNER.stdout")
    read_ref(terminal["stderr_ref"], MIB, path=control / "OWNER.stderr", allow_empty=True, capture=False)
    return control, stdout


def _worker_closed(work, stage, ref, owner, previous, is_live):
    directory = work / stage
    terminal = document(read_ref(ref, MIB, path=directory / "TERMINAL_PRIVATE.json"))
    exact(terminal, dict(stage=stage, owner=owner, exact_worker_closed=True, original_os_wait=True,
                         original_os_exit=0, resource_stop=False, stop_reason=None),
          ("worker", "worker_wall_seconds", "peak_worker_rss_bytes", "stdout_ref", "stderr_ref"),
          reason="Foreign or unclosed original B/C worker terminal")
    birth(terminal["owner"])
    worker = birth(terminal["worker"])
    require(worker["creation_time"] > previous["creation_time"] and is_live(worker) is False
            and elapsed(terminal["worker_wall_seconds"]) and type(terminal["peak_worker_rss_bytes"]) is int
            and terminal["peak_worker_rss_bytes"] >= 0, "Live or out-of-order original B/C worker birth")
    read_ref(terminal["stdout_ref"], MIB, path=directory / "WORKER.stdout", allow_empty=True, capture=False)
    read_ref(terminal["stderr_ref"], MIB, path=directory / "WORKER.stderr", allow_empty=True, capture=False)
    return worker, terminal["stdout_ref"]["sha256"]


def build_plan(code, lock_ref, owner_terminal_ref, completion_ref, *, owner_birth, coordinator_birth, is_live):
    """Trust only a caller's original launch birth and independently waited owner terminal."""
    owner, coordinator = birth(owner_birth), birth(coordinator_birth)
    require(callable(is_live) and coordinator["creation_time"] < owner["creation_time"],
            "Original coordinator must precede the original B/C owner")
    control, stdout = _owner_closed(owner_terminal_ref, owner, is_live)
    lock = document(read_ref(lock_ref, MIB, path=control / "EVALUATION_LOCK_PRIVATE.json"))
    exact(lock, dict(status=LOCK_STATUS), (*PAIR_FIELDS, "evaluation_protocol", "owner_source", "work"),
          reason="Exact independently prelocked B/C evaluation owner request required")
    require(all(digest(lock[name], 40) for name in ("training_protocol", "implementation_commit", "evaluation_protocol"))
            and lock["evaluation_protocol"] != lock["training_protocol"] and digest(lock["qualification_pair_sha256"])
            and type(lock["work"]) is str, "Separate prospective training and evaluation identities required")
    work = Path(lock["work"])
    require(work.is_absolute() and work.resolve(strict=True) == work and work.is_dir() and work != control,
            "Exact distinct original B/C owner output directory required")
    read_ref(lock["owner_source"], MIB, path=control / "PINNED_BC_OWNER.py", capture=False)
    completed = document(read_ref(completion_ref, MIB, path=work / "OWNER_COMPLETION_PRIVATE.json"))
    exact(completed, dict(status=COMPLETION_STATUS, owner=owner, coordinator=coordinator, lock_ref=lock_ref,
                          owner_source_sha256=lock["owner_source"]["sha256"]), ("stages",),
          reason="Original B/C owner completion must bind this launch, source and prelock")
    birth(completed["owner"])
    birth(completed["coordinator"])
    expected_event = dict(event=OWNER_EVENT, lock_sha256=lock_ref["sha256"], completion_sha256=completion_ref["sha256"],
                          owner_source_sha256=lock["owner_source"]["sha256"])
    lines = stdout.splitlines()
    require(stdout.endswith(b"\n") and lines and document(lines[-1]) == expected_event
            and sum(OWNER_EVENT.encode("ascii") in line for line in lines) == 1,
            "Exact final original owner stdout must bind this closed completion")

    p260 = load_p260(code)
    support = p260.load_support(code)
    driver = support.driver
    seen = {control, work}
    pair_path, pair_pin = p260._pin(driver, lock["training_pair"], seen, 8192)
    arm_paths = {mode: p260._pin(driver, lock[f"{mode}_arm_receipt"], seen, 8192)[0] for mode in MODES}
    pair, trained = support.admission.read_pair(
        support.original, support.phase, pair_path, pair_pin["sha256"],
        stepwise_center_path=arm_paths["stepwise_center"], one_pass_path=arm_paths["one_pass"],
        training_protocol=lock["training_protocol"], implementation_commit=lock["implementation_commit"],
        qualification_pair_sha256=lock["qualification_pair_sha256"],
    )
    require(all(lock[f"{mode}_arm_receipt"]["sha256"] == pair[f"{mode}_arm_receipt_sha256"] for mode in MODES),
            "Exact independently accepted B/C arm receipt digests required")
    _, gt_pin = p260._pin(driver, lock["gt"], seen, 4 * MIB)
    require(gt_pin["sha256"] == driver.GT_SHA, "Exact original fixed DEV705 GT digest required")

    ordering = tuple((operation, mode) for operation in OPERATIONS for mode in MODES)
    entries = completed["stages"]
    require(type(entries) is list and len(entries) == len(ordering) and all(type(entry) is dict for entry in entries)
            and [entry.get("stage") for entry in entries] == [f"{operation}_{mode}".upper() for operation, mode in ordering],
            "Unexpected original B/C stage order or population")
    plan = dict(
        status=p260.PLAN_STATUS, stage="e24", training_seed=support.original.TRAINING_SEED,
        protocol=lock["evaluation_protocol"], **{key: lock[key] for key in PAIR_FIELDS},
        arms={mode: {} for mode in MODES},
    )
    previous = owner
    for entry, (operation, mode) in zip(entries, ordering, strict=True):
        stage = f"{operation}_{mode}".upper()
        inputs = ("predictions",) if operation == "export" else (
            "export_terminal_sha256", "export_report_sha256", "predictions_sha256",
        )
        exact(entry, dict(stage=stage), ("terminal", "report", *inputs), reason="Unexpected original B/C stage order or fields")
        previous, stdout_sha = _worker_closed(work, stage, entry["terminal"], owner, previous, is_live)
        common = dict(
            status=p260.STAGE_STATUS[operation], mode=mode, evaluation_protocol=plan["protocol"],
            training_protocol=plan["training_protocol"], training_pair_sha256=pair_pin["sha256"],
            checkpoint_sha256=trained[mode]["checkpoint_sha256"], original_os_wait=True, original_os_exit=0,
            resource_stop=False, owner_closed=True, worker_closed=True, terminal_sha256=entry["terminal"]["sha256"],
            stdout_sha256=stdout_sha, report=entry["report"],
        )
        if operation == "export":
            directory = work / (stage + "_ACTUAL_PRIVATE")
            read_ref(entry["report"], MIB, path=directory / "DEV_NATIVE_EXPORT_SAFE.json", capture=False)
            read_ref(entry["predictions"], 512 * MIB, path=directory / "DEV_NATIVE_PREDICTIONS_PRIVATE.jsonl", capture=False)
            result = dict(common, predictions=entry["predictions"])
        else:
            read_ref(entry["report"], MIB, path=work / (stage + "_SAFE.json"), capture=False)
            result = dict(common, original_dev_gt_sha256=driver.GT_SHA, **{key: entry[key] for key in inputs})
        plan["arms"][mode][operation] = result

    # The pinned P260 preflight independently checks artifact report semantics,
    # score-to-export linkage and all four distinct terminal digests.
    p260._closed_arms(support, plan, trained, seen, pair_pin["sha256"])
    read_ref(lock["gt"], 4 * MIB, capture=False)
    require(is_live(owner) is False, "Original owner liveness changed during closed readback")
    return plan
