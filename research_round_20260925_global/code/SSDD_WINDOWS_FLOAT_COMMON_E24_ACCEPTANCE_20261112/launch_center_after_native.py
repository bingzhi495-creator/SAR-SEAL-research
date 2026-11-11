"""Admit the serial center coordinator only after independent native E24 closure."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from types import ModuleType


ROOT = Path(__file__).resolve().parent.parent
BRIDGE = ROOT / "SSDD_WINDOWS_FLOAT_COMMON_E24_DRAFT_PRIVATE_20261112_02/float_e24_bridge.py"
BRIDGE_SHA = "eb3351588f27c985f777d8d620ddce097402a07aaf32281d692f6e217686d467"
SHORT_PAIR = "78d2c6d1b99a1b05755324e061c1ad1bd4c5a460578e06e73720fa8c748ea8e0"
ARM_STATUS = "SSDD_WINDOWS_FLOAT_COMMON_E24_ARM_INDEPENDENTLY_ACCEPTED_V1"
PROTOCOL = "194e16f289a644430015f310dd63a3c1c1618cfd"
MANIFEST = "56fdd4be8d19b4e2e42ff736e89775f32b1e6ce2a9c91d2a90afad509640bc14"


def require(ok, why):
    if not ok:
        raise ValueError(why)


def same_fields(data, expected):
    return all(k in data and type(data[k]) is type(v) and data[k] == v for k,v in expected.items())


def verify_independent_native(receipt, authority):
    require(same_fields(receipt,dict(status=ARM_STATUS,qualification_pair_sha256=SHORT_PAIR,
                                    completed_epochs=24,visits=17856,optimizer_updates=4464,ema_updates=4464,
                                    original_os_wait=True,original_os_exit=0,resource_stop=False,worker_closed=True,
                                    FIT=True,DEV=False,TEST=False,AP=False))
            and same_fields(receipt["phase"],dict(stage="e24",mask_resize_mode="native",seed=5201,
                 initialization_seed=5201,training_protocol=PROTOCOL,source_manifest_sha256=MANIFEST)),
            "Exact independently accepted native E24 receipt required")
    require(same_fields(authority,dict(mode="native",resource_closed=True,active_attempt=None,failure=None))
            and authority["terminal_ref"]["sha256"] == receipt["terminal_sha256"]
            and authority["launch_ref"]["sha256"] == receipt["launch_sha256"],
            "Exact original native terminal and launch authority must agree")


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument("--native-receipt",type=Path,required=True)
    parser.add_argument("--native-receipt-sha256",required=True)
    parser.add_argument("--operation-commit",required=True)
    args=parser.parse_args()
    raw=BRIDGE.read_bytes()
    require(sha256(raw).hexdigest()==BRIDGE_SHA,"Exact original executed worker source")
    source=raw.decode()
    original='    need(mode == "native", "This original launch admits only native A; center is after independent A acceptance")'
    replacement='    admit_center(mode)'
    require(source.count(original)==1,"Exact closed-only coordinator admission seam")
    m=ModuleType("_same_closed_e24_worker_center_coordinator")
    m.__file__=str(BRIDGE)
    exec(compile(source.replace(original,replacement,1),str(BRIDGE),"exec"),m.__dict__)

    def admit_center(mode):
        require(mode=="center","This closed-only coordinator admits only center")
        raw=args.native_receipt.read_bytes()
        require(sha256(raw).hexdigest()==args.native_receipt_sha256,"Exact independently accepted native receipt digest")
        receipt=json.loads(raw)
        state=json.loads((ROOT/"authoritative_operations_20261021_01/ACTIVE_RESEARCH_STATE.json").read_bytes())
        authority=state[m.key_for("native")]
        verify_independent_native(receipt,authority)
        archive=m.ref(m.RESULT_ROOT/"native"/"ACTUAL"/"checkpoint_23.ckpt")
        require(archive["sha256"]==receipt["checkpoint_sha256"] and archive["bytes"]==receipt["checkpoint_bytes"],
                "Exact independently verified closed native E24 archive")

    m.admit_center=admit_center
    sys.argv=[str(BRIDGE),"--run","--arm","center","--operation-commit",args.operation_commit]
    return m.main()


if __name__=="__main__":
    raise SystemExit(main())
