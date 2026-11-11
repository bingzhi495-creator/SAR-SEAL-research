"""Reuse the exact Windows floating/common E24 owner for the sole prelocked repeat."""
from hashlib import sha256
import json
from pathlib import Path
from types import ModuleType


SELF = Path(__file__).resolve()
ROOT = SELF.parent.parent
ORIGINAL = ROOT / "SSDD_WINDOWS_FLOAT_COMMON_E24_DRAFT_PRIVATE_20261112_02/float_e24_bridge.py"
ORIGINAL_SHA = "eb3351588f27c985f777d8d620ddce097402a07aaf32281d692f6e217686d467"
SHORT = ROOT / "SSDD_WINDOWS_FLOAT_COMMON_REPEAT5202_SHORT_DRAFT_PRIVATE_20261112_01/repeat5202_float_short.py"
SHORT_SHA = "320d7a5dba0f55c888761314519c94a008bbaa657559cde832c1de48e5f474d7"
SHORT_REQUEST = ROOT / "SSDD_WINDOWS_FLOAT_COMMON_REPEAT5202_SHORT_ORIGINAL_WAIT_PRIVATE_20261112_01/REQUEST_PRIVATE.json"
SHORT_REQUEST_SHA = "f4ba04a50a7d4bb8647ed3b5cb764c351ac779698efe2d5cbd7514f4c32d66a3"
SHORT_PAIR = ROOT / "SSDD_WINDOWS_Q100_FLOAT_COMMON_REPEAT5202_SHORT_ACTUAL_PRIVATE_20261112_01/FLOAT_COMMON_REPEAT5202_SHORT_PAIR_ACCEPTED_SAFE.json"
SHORT_PAIR_SHA = "6bef2983c978f3614fb610089ce173713dfc3892641a1890139fe560517692f8"
SHORT_MANIFEST = "ac34720d21833e81ee98eda9a8f4a1bc21a4d1cd71668cfd39803f71ae375239"
RESULT_ROOT = ROOT / "SSDD_WINDOWS_Q100_FLOAT_COMMON_REPEAT5202_E24_ACTUAL_PRIVATE_20261112_01"
ARM_READY = "SSDD_WINDOWS_FLOAT_COMMON_REPEAT5202_E24_ORIGINAL_ARM_CLOSED_PENDING_ACCEPTANCE_V1"
WORKER_STATUS = "SSDD_WINDOWS_FLOAT_COMMON_REPEAT5202_E24_ARM_CLOSED_PENDING_INDEPENDENT_ACCEPTANCE_V1"


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def module(path, expected, name):
    raw = path.read_bytes()
    require(path.resolve(strict=True) == path and sha256(raw).hexdigest() == expected,
            "Exact unchanged original source " + path.name)
    loaded = ModuleType(name)
    loaded.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), loaded.__dict__)
    return loaded


def bridge():
    original = module(ORIGINAL, ORIGINAL_SHA, "_exact_float_e24_native_owner")
    repeat = module(SHORT, SHORT_SHA, "_exact_qualified_repeat5202_short")
    native_inputs = original.admit_worker_inputs
    native_launch = original.launch_manifest
    seed_binding = None

    def context():
        nonlocal seed_binding
        stock_short, ctx, adapter, programs = repeat.context(SHORT_REQUEST, SHORT_REQUEST_SHA)
        pair = json.loads(repeat.pinned(SHORT_PAIR, SHORT_PAIR_SHA))
        fixed = dict(status=repeat.PAIR_STATUS, authority=repeat.AUTHORITY, initialization_seed=5201,
                     seed=5202, training_protocol=original.PROTOCOL, source_manifest_sha256=SHORT_MANIFEST,
                     first_float_score_sha256=repeat.FIRST_SCORE_SHA, anchor_AP=repeat.ANCHOR,
                     visits_per_arm=32, optimizer_updates_per_arm=8, ema_updates_per_arm=8,
                     common_targets_total=69, common_empty_visits=3, changed_mask_visits=29, **repeat.SCOPE)
        require(all(type(pair.get(key)) is type(value) and pair[key] == value for key, value in fixed.items())
                and ctx.worker.manifest(ctx.entry, dict(ctx.cfg, stage="short")) == SHORT_MANIFEST
                and original.ref(Path(ctx.cfg["coco"]["path"])) == ctx.cfg["coco"],
                "Exact accepted seed5202 float/common qualification and seed5201 COCO input")
        repeat.select_layout = stock_short.select_layout
        seed_binding = adapter, programs
        return repeat, ctx, pair

    def admit_worker_inputs(short, ctx, scope):
        require(short is repeat and seed_binding is not None, "Qualification precedes worker admission")
        torch, worker, deps, layout, ids = native_inputs(short, ctx, scope)
        adapter, programs = seed_binding
        torch, _, worker, deps = repeat.bind_repeat_runtime((torch, None, worker, deps), programs, adapter)
        return torch, worker, deps, layout, ids

    def launch_manifest(mode, stock_scope, worker, operation_commit):
        return dict(native_launch(mode, stock_scope, worker, operation_commit),
                    initialization_seed=5201, training_seed=5202, anchor_AP=repeat.ANCHOR,
                    first_float_score_sha256=repeat.FIRST_SCORE_SHA)

    def key_for(mode):
        require(mode in repeat.MODES, "Exact two frozen arms")
        return "SSDD_WINDOWS_Q100_FLOAT_COMMON_E24_" + mode.upper() + "_5202"

    original.SELF = SELF
    original.RESULT_ROOT = RESULT_ROOT
    original.SOURCE_MANIFEST = SHORT_MANIFEST
    original.SHORT_REQUEST_SHA = SHORT_REQUEST_SHA
    original.SHORT_ACCEPTANCE = SHORT_PAIR
    original.SHORT_ACCEPTANCE_SHA = SHORT_PAIR_SHA
    original.SHORT_STATUS = repeat.PAIR_STATUS
    original.ARM_READY = ARM_READY
    original.WORKER_STATUS = WORKER_STATUS
    original.context = context
    original.admit_worker_inputs = admit_worker_inputs
    original.launch_manifest = launch_manifest
    original.key_for = key_for
    return original


if __name__ == "__main__":
    raise SystemExit(bridge().main())
