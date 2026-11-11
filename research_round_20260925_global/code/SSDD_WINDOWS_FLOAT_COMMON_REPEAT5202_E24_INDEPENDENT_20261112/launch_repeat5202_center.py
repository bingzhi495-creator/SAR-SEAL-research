"""Launch original floating/common center E24 after independently accepted native E24."""
import argparse
import ast
from hashlib import sha256
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace


ROOT = Path(__file__).resolve().parent.parent
BRIDGE = ROOT / "SSDD_WINDOWS_FLOAT_COMMON_REPEAT5202_E24_DRAFT_PRIVATE_20261112_01/repeat5202_float_e24.py"
BRIDGE_SHA = "e469f36f2d57017ee2a75d618e5fbc077343eee7c9ec0a480bc3b8d729ef5cdf"
SHORT_PAIR_SHA = "6bef2983c978f3614fb610089ce173713dfc3892641a1890139fe560517692f8"
ARM_STATUS = "SSDD_WINDOWS_FLOAT_COMMON_REPEAT5202_E24_ARM_INDEPENDENTLY_ACCEPTED_V1"
ARM_READY = "SSDD_WINDOWS_FLOAT_COMMON_REPEAT5202_E24_ORIGINAL_ARM_CLOSED_PENDING_ACCEPTANCE_V1"
PROTOCOL = "194e16f289a644430015f310dd63a3c1c1618cfd"
MANIFEST = "ac34720d21833e81ee98eda9a8f4a1bc21a4d1cd71668cfd39803f71ae375239"


def require(ok, why):
    if not ok:
        raise ValueError(why)


def same_fields(value, fields):
    return all(key in value and type(value[key]) is type(item) and value[key] == item for key, item in fields.items())


def verify_native(receipt, authority):
    require(same_fields(receipt, dict(status=ARM_STATUS, qualification_pair_sha256=SHORT_PAIR_SHA,
                                    initialization_seed=5201, training_seed=5202,
                                    completed_epochs=24, visits=17856, optimizer_updates=4464, ema_updates=4464,
                                    original_os_wait=True, original_os_exit=0, resource_stop=False, worker_closed=True,
                                    FIT=True, DEV=False, TEST=False, AP=False))
            and same_fields(receipt["phase"], dict(stage="e24", mask_resize_mode="native", seed=5202,
                         initialization_seed=5201, training_protocol=PROTOCOL, source_manifest_sha256=MANIFEST)),
            "Exact independently accepted native E24 receipt at downstream seed 5202")
    require(same_fields(authority, dict(status=ARM_READY, mode="native", resource_closed=True,
                                       active_attempt=None, failure=None))
            and authority["terminal_ref"]["sha256"] == receipt["terminal_sha256"]
            and authority["launch_ref"]["sha256"] == receipt["launch_sha256"],
            "Exact original closed native terminal and launch must match the independent receipt")


def serial_owner(m, source):
    before = '    need(mode == "native", "This original launch admits only native A; center is after independent A acceptance")'
    require(source.count(before) == 1, "Exact prospective coordinator-only serial admission seam")
    source = source.replace(before, "    admit_center(mode, legacy)", 1)
    owner = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == "run_owner"]
    require(len(owner) == 1, "Exact original owner definition")
    local = {}
    exec(compile(ast.Module(body=owner, type_ignores=[]), str(m.__file__), "exec"), vars(m), local)
    return local["run_owner"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--native-receipt", type=Path, required=True)
    parser.add_argument("--native-receipt-sha256", required=True)
    parser.add_argument("--operation-commit", required=True)
    args = parser.parse_args()
    raw = BRIDGE.read_bytes()
    require(BRIDGE.resolve(strict=True) == BRIDGE and sha256(raw).hexdigest() == BRIDGE_SHA,
            "Exact originally executed repeat5202 worker")
    training = ModuleType("_same_repeat5202_training_worker_serial_coordinator")
    training.__file__ = str(BRIDGE)
    exec(compile(raw, str(BRIDGE), "exec"), training.__dict__)
    m = training.bridge()

    def admit_center(mode, legacy):
        require(mode == "center" and args.native_receipt.is_absolute()
                and args.native_receipt.resolve(strict=True) == args.native_receipt,
                "Exact original serial center and canonical native receipt")
        receipt = json.loads(m.raw_pinned(args.native_receipt, args.native_receipt_sha256))
        state = json.loads(legacy.STATE.read_bytes())
        authority = state[m.key_for("native")]
        verify_native(receipt, authority)
        require(not legacy.live(authority["owner_birth"]) and m.key_for("center") not in state,
                "Native process closed and fresh serial center")
        checkpoint = m.ref(m.RESULT_ROOT / "native/ACTUAL/checkpoint_23.ckpt")
        require(checkpoint["sha256"] == receipt["checkpoint_sha256"] and checkpoint["bytes"] == receipt["checkpoint_bytes"],
                "Exact independently verified original native E24 EMA archive")

    m.admit_center = admit_center
    owner = serial_owner(m, m.raw_pinned(training.ORIGINAL, training.ORIGINAL_SHA).decode("utf8"))
    return owner(SimpleNamespace(arm="center", preflight_cpu=False, operation_commit=args.operation_commit))


if __name__ == "__main__":
    raise SystemExit(main())
