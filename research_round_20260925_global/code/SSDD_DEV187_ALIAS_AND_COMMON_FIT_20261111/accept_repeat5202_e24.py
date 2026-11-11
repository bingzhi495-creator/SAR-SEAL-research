"""Independently accept the closed Windows PSeg-SSDD seed-5202 E24 repeat."""

import argparse
from hashlib import sha256
import json
from math import isfinite
import os
from pathlib import Path
import sys
from types import ModuleType
from unittest.mock import patch

MIB = 1024**2
GIB = 1024**3
MODES = ("native", "center")
FIT_ONLY = dict(FIT=True, DEV=False, TEST=False, AP=False)
TRAINING_PROTOCOL = "1832f1e052c5fae15a5b2007b523579a73b40f1d"
SOURCE_MANIFEST_SHA = "530eca90ee7a74414f68a0a9bc189406f96921654f64a853548e85da1c99656e"
IMPLEMENTATION_COMMIT = "48587900e587a07f24fe3ee632e1f2fd0aca88e2"
SHORT_SHA = "14a6c9be480f9c6567a23d8a0abcaa86d2d5f4d6921db5595224d67129723425"
SHORT_STATUS = "SSDD_MASK_PHASE_REPEAT_5202_QUALIFICATION_PAIR_ACCEPTED_V1"
TRAINING_SEED, INITIALIZATION_SEED = 5202, 5201
STOCK_OWNER_SHA = "7afe0a47d1f9177976320f5699504181bcfb32dd24177ab0e98548e7ca55ea8e"
STOCK_LOCK_SHA = "2cf720dd23db6fcde37b24ca644e006dd22fcdb9288edd1d232e7986fb0415ab"
WINDOWS_ENTRY_SHA = "e29edeb0a3aff7c871123deee7c0c198ef5913fd33e7c016434a671d186796d3"
TRAIN_SHA = "6785b1f9ffe23b6b2c1da5939d10d8b5877b93127316f535af5890ca05f4653c"
QUALIFIER_SHA = "8b76b42d59fff857dfb0c0fc9b67664817c72a494966f5f0ec808156776cdaec"
ACCEPTANCE_SHA = "5a22d2dc7a9dd67427fcedd3c6b45278c8abe8c9023ced33357206bc2d275604"
ADAPTER_SHA = "2c008f48cf8c0b09327279c78d977ad02a0795e12054fb848c19bc12deb7ac10"
ORIGINAL_MANIFEST_SHA = "046ddaa300e428140952a09931269f6914c789a63fabdc94f98484c3f2efaa91"
EXPORT_SHA = "7bf88b0880a360f52de9263b215bfda30d34ab14d91183dc33a13a13b8aa26f5"
PHASE_DIR = "SSDD_MASK_PHASE_TRANSFER_20261110"
REQUEST = "P350_WINDOWS_SSDD_REPEAT5202_E24_REQUEST_V1"
RESERVED = "P350_WINDOWS_REPEAT5202_E24_OWNER_RESERVED"
ARM_START = "P350_WINDOWS_REPEAT5202_E24_ORIGINAL_ARM_START"
CLOSED = "P350_WINDOWS_REPEAT5202_E24_ORIGINALS_CLOSED_PENDING_INDEPENDENT_ACCEPTANCE"
WAIT = "P350_WINDOWS_REPEAT5202_E24_EXTERNAL_ORIGINAL_WAIT_V1"
GPU_ADMITTED = "SSDD_WINDOWS_Q100_REPEAT_5202_GPU_ADMITTED_V1"
OUTPUT = "P351_SSDD_WINDOWS_REPEAT5202_E24_INDEPENDENTLY_ACCEPTED_V1"
PAIR_NAME = "E24_PAIR_ACCEPTED_SAFE.json"
EXTRAS = ("initialization_seed","authority_identity","implementation_commit",
          "adapter_sha256","original_manifest_sha256")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return sha256(data).hexdigest()


def exact_fields(value, fixed, extras=()):
    return (type(value) is dict and set(value)==set(fixed)|set(extras)
            and all(type(value[k]) is type(v) and value[k]==v for k,v in fixed.items()))


def birth_key(birth):
    require(type(birth) is dict and set(birth)=={"pid","creation_time"}
            and type(birth["pid"]) is int and birth["pid"]>0 and type(birth["creation_time"]) is float
            and isfinite(birth["creation_time"]) and birth["creation_time"]>0,
            "Exact Windows process birth required")
    return birth["pid"],birth["creation_time"]


def verify_wait(wait, birth, request_sha, summary_sha, live):
    birth_key(birth)
    require(exact_fields(wait,dict(status=WAIT,owner_birth=birth,request_sha256=request_sha,
                                  summary_sha256=summary_sha,original_os_wait=True,original_os_exit=0),
                         ("combined_stdout",))
            and type(wait["combined_stdout"]) is str and 0<len(wait["combined_stdout"])<=16*1024
            and live(birth) is False,"Exact successful original closed P350 coordinator wait required")
    expected=[dict(status=RESERVED,owner_birth=birth,request_sha256=request_sha),
              *[dict(status=ARM_START,mode=mode,request_sha256=request_sha) for mode in MODES],
              dict(status=CLOSED,request_sha256=request_sha,summary_sha256=summary_sha)]
    observed=[json.loads(line) for line in wait["combined_stdout"].splitlines() if line.strip()]
    require(observed==expected,"Original owner stdout must bind ordered native/center starts and closure")


def stable(path, limit, expected=None, *, empty=False):
    require(path.is_absolute() and path.resolve(strict=True)==path and path.is_file(),
            "Canonical original regular file required")
    with path.open("rb") as stream:
        before=os.fstat(stream.fileno())
        raw=stream.read(limit+1)
        after=os.fstat(stream.fileno())
    identify=lambda stat:(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns)
    require(identify(before)==identify(after)==identify(path.stat()) and len(raw)==before.st_size
            and len(raw)<=limit and (empty or bool(raw)) and (expected is None or digest(raw)==expected),
            "Original file bytes, size, or immutable digest differ")
    return raw


def ref(path, raw):
    return dict(path=path.as_posix(),bytes=len(raw),sha256=digest(raw))


def from_ref(path, reference, limit, *, empty=False):
    require(type(reference) is dict and set(reference)=={"path","bytes","sha256"}
            and type(reference["path"]) is str and Path(reference["path"])==path
            and type(reference["bytes"]) is int and type(reference["sha256"]) is str,
            "Exact original path and byte reference required")
    raw=stable(path,limit,reference["sha256"],empty=empty)
    require(len(raw)==reference["bytes"],"Exact original reference byte length required")
    return raw


def module(path, source, name):
    result=ModuleType(name)
    result.__file__=str(path)
    exec(compile(source,str(path),"exec"),result.__dict__)
    return result


def load_acceptance(code):
    folder=code/PHASE_DIR
    qualifier_path=folder/"ssdd_phase_qualification.py"
    qualifier=module(qualifier_path,stable(qualifier_path,64*1024,QUALIFIER_SHA),"_p351_original_qualifier")
    path=folder/"ssdd_phase_e24_acceptance.py"
    source=stable(path,64*1024,ACCEPTANCE_SHA).decode("utf-8")
    replacements=(
        ("initialization_seed=SEED","initialization_seed=INITIALIZATION_SEED",2),
        ('phase.get("initialization_seed") == SEED','phase.get("initialization_seed") == INITIALIZATION_SEED',1),
        ('extra=("classifier_initial_sha256", "ordered_inputs_sha256", "changes", "arms"))',
         'extra=("classifier_initial_sha256", "ordered_inputs_sha256", "changes", "arms", *REPEAT_EXTRAS))',1),
    )
    for old,new,count in replacements:
        require(source.count(old)==count,"Exact original seed and qualification binding sites required")
        source=source.replace(old,new)
    with patch.dict(sys.modules,{"ssdd_phase_qualification":qualifier}):
        api=module(path,source,"_p351_repeat_e24_acceptance")
    require(api.SEED==INITIALIZATION_SEED,"Immutable original seed constant required")
    api.SEED=TRAINING_SEED
    api.INITIALIZATION_SEED=INITIALIZATION_SEED
    api.SHORT_STATUS=SHORT_STATUS
    api.REPEAT_EXTRAS=EXTRAS
    original=api.verify_scope

    def verify_scope(protocol,source_manifest_sha256,expected_fit_ids,qualification_bytes,qualification_sha256):
        require(protocol==TRAINING_PROTOCOL and source_manifest_sha256==SOURCE_MANIFEST_SHA
                and qualification_sha256==SHORT_SHA and len(qualification_bytes)==3356,
                "Exact admitted original 5202 short lineage required")
        scope,population,short=original(protocol,source_manifest_sha256,expected_fit_ids,
                                        qualification_bytes,qualification_sha256)
        require(short["initialization_seed"]==INITIALIZATION_SEED
                and short["authority_identity"]=="SSDD_MASK_PHASE_REPEAT_5202_SHORT_PROSPECTIVE_V1"
                and short["implementation_commit"]==IMPLEMENTATION_COMMIT
                and short["adapter_sha256"]==ADAPTER_SHA and short["original_manifest_sha256"]==ORIGINAL_MANIFEST_SHA,
                "Original short qualification must preserve initialization and source provenance")
        return scope,population,short

    api.verify_scope=verify_scope
    return api


P350_SHA = "298c8f9d09c603bc0b6078ff770417e82400da69ce8e6d4903cb03ecca5766cb"


def verify_gpu_event(entry, stdout, mode):
    candidates=[]
    for line in stdout.splitlines():
        if line.lstrip().startswith(b"{"):
            item=entry.document(line)
            if item.get("event")==GPU_ADMITTED:
                candidates.append(item)
    fixed=dict(event=GPU_ADMITTED,protocol=TRAINING_PROTOCOL,mode=mode,stage="e24",
               qualification_pair_sha256=SHORT_SHA)
    require(len(candidates)==1 and exact_fields(candidates[0],fixed,
            ("cuda_free_bytes","cuda_total_bytes","cuda_allocator_fraction")),
            "Exact one repeat E24 GPU admission event required")
    item=candidates[0]
    free,total,fraction=(item[k] for k in ("cuda_free_bytes","cuda_total_bytes","cuda_allocator_fraction"))
    require(type(free) is type(total) is int and total>=free>=12*GIB
            and type(fraction) is float and isfinite(fraction) and 0<fraction<=min(.35,6*GIB/total),
            "Original SSDD E24 GPU admission resource bounds required")


def read_closed_arm(plan, short_guard, legacy, reservation, summary, mode):
    arm=plan.old.controls(plan.work,mode)
    scope_raw=from_ref(arm.lock,reservation["scopes"][mode],64*1024)
    scope=plan.entry.document(scope_raw)
    terminal_path=arm.control/"OWNER_TERMINAL_PRIVATE.json"
    terminal_ref=summary["terminal_refs"][mode]
    terminal=plan.entry.document(from_ref(terminal_path,terminal_ref,64*1024))
    original=dict(original_os_wait=True,original_os_exit=0,job_assigned=True,
                  job_active_before_drain=0,job_active_after_wait=0,stop_reason=None)
    require(exact_fields(terminal,original,("owner","stdout_ref","stderr_ref","wall_seconds"))
            and type(terminal["wall_seconds"]) in (int,float) and isfinite(terminal["wall_seconds"])
            and 0<terminal["wall_seconds"]<=36000
            and short_guard.closed_arm(legacy,arm,terminal,reservation["owner_birth"])==terminal_ref,
            "Exact closed E24 original worker JobObject, process birth and logs required")
    guard=plan.entry.document(stable(arm.control/"RESOURCE_GUARD_PRIVATE.json",64*1024))
    require(exact_fields(guard,dict(guard=None,wall_limit_seconds=36000),("peak_worker_RSS_bytes",))
            and type(guard["peak_worker_RSS_bytes"]) is int and 0<guard["peak_worker_RSS_bytes"]<=10*GIB
            and stable(arm.control/"JOB_ATTACHED.txt",64)==P350_SHA.encode("ascii"),
            "Exact original worker source attachment and RSS/time evidence required")
    stdout=from_ref(arm.control/"OWNER.stdout",terminal["stdout_ref"],8*MIB)
    from_ref(arm.control/"OWNER.stderr",terminal["stderr_ref"],8*MIB,empty=True)
    verify_gpu_event(plan.entry,stdout,mode)
    return arm,scope,terminal,stdout


def checkpoint_reader(api, export_api, deps, torch, model, init_witness, short):
    """Use one stock CPU load, verify both raw and strictly reloaded EMA tensors."""
    eager=model.model
    classifier=eager.class_embed.weight.detach().contiguous().clone()
    checks=[]

    class StrictCPU:
        float32=torch.float32
        isfinite=staticmethod(torch.isfinite)

        def load(self, stream, *, map_location, weights_only):
            require(map_location=="cpu" and weights_only is True and len(checks)<len(MODES),
                    "One original CPU weights-only load per E24 arm required")
            checkpoint=torch.load(stream,map_location="cpu",weights_only=True)
            require(type(checkpoint) is dict and type(checkpoint.get("args")) is dict
                    and type(checkpoint["args"].get("seed")) is int
                    and checkpoint["args"]["seed"]==TRAINING_SEED,
                    "Saved RF-DETR downstream seed must be exactly 5202")
            model_config=checkpoint.get("model_config")
            expected_config=dict(num_queries=100,num_select=100,group_detr=13,resolution=576,
                                 num_classes=1,segmentation_head=True)
            require(type(model_config) is dict and all(type(model_config.get(k)) is type(v) and model_config[k]==v
                    for k,v in expected_config.items()),"Exact saved SSDD Nano Q100 model config required")
            raw=checkpoint["state_dict"]
            current=eager.state_dict()
            require(len(current)==545 and set(current)=={key[6:] for key in raw if key.startswith("model.")},
                    "Exact original raw E24 model tensor roster required")
            for name,target in current.items():
                value=raw["model."+name]
                require(torch.is_tensor(value) and value.device.type=="cpu" and value.shape==target.shape
                        and value.dtype==target.dtype
                        and (not value.is_floating_point() or bool(torch.isfinite(value).all().item())),
                        "Exact finite raw E24 model tensor shape/dtype required")
            witness=export_api.strict_ema_load(eager,checkpoint,torch)
            require(witness["tensors"]==545 and witness["ema_updates"]==4464
                    and witness["strict_readback_exact"] is True,"Exact original E24 full strict EMA readback required")
            checks.append(witness)
            return checkpoint

    inspect=api.make_checkpoint_reader(torch=StrictCPU(),native_inspector=deps.e100.inspect_archive,
        e1_api=deps.e1,boundary_api=deps.boundary,initial_classifier=classifier,
        initialization=init_witness,classifier_initial_sha256=short["classifier_initial_sha256"])
    return inspect,checks


def write(path, raw):
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def run(a):
    owner=module(a.owner_source,stable(a.owner_source,64*1024,P350_SHA),"_p351_p350_exact_owner")
    require(owner.TRAINING_PROTOCOL==TRAINING_PROTOCOL and owner.TRAINING_MANIFEST==SOURCE_MANIFEST_SHA
            and owner.PAIR_SHA==SHORT_SHA and owner.LONG_SECONDS==36000 and owner.REQUEST==REQUEST,
            "Exact original E24 owner provenance required")
    _,base=owner.load_helpers()
    require(base.OLD_OWNER_SHA==STOCK_OWNER_SHA and base.OLD_LOCK_SHA==STOCK_LOCK_SHA
            and base.ADAPTER_SHA==ADAPTER_SHA,"Exact original Windows closure and stock scope sources required")
    plan=owner.prepare(base,a.request,a.request_sha256)
    require(plan.worker_sha==P350_SHA and plan.identity["implementation_commit"]==IMPLEMENTATION_COMMIT
            and plan.adapter.TRAIN[1]==TRAIN_SHA and plan.adapter.WINDOWS[1]==WINDOWS_ENTRY_SHA,
            "Exact closed original 5202 repeat owner required")
    require(a.output.is_absolute() and a.output.parent.resolve(strict=True)==a.output.parent
            and not a.output.exists() and not a.output.is_symlink() and not a.output.is_relative_to(plan.work),
            "Fresh separate independent CPU acceptance output required")
    short_guard,legacy,runtime=plan.old.load_guards("short")
    short_guard.SHORT_SECONDS=36000
    reservation,reservation_raw,reservation_ref=base.read_reservation(plan,legacy)
    owner_birth=reservation["owner_birth"]
    summary_path=plan.work/base.SUMMARY_NAME
    summary_raw=stable(summary_path,64*1024)
    summary_ref=ref(summary_path,summary_raw)
    summary=plan.entry.document(summary_raw)
    require(exact_fields(summary,dict(status=CLOSED,owner_birth=owner_birth,request_ref=plan.request_ref,
                                       reservation_ref=reservation_ref,resource_closed=True,failure=None,**FIT_ONLY),
                         ("terminal_refs",)) and set(summary["terminal_refs"])==set(MODES),
            "Exact successful closed original two-arm owner summary required")
    wait_raw=stable(a.wait,64*1024,a.wait_sha256)
    verify_wait(plan.entry.document(wait_raw),owner_birth,plan.request_sha,summary_ref["sha256"],legacy.live)
    records={mode:read_closed_arm(plan,short_guard,legacy,reservation,summary,mode) for mode in MODES}
    births=[birth_key(owner_birth),*(birth_key(records[mode][2]["owner"]) for mode in MODES)]
    require(len({pid for pid,_ in births})==len(births)==3
            and births[0][1]<=births[1][1]<births[2][1]
            and not any(legacy.live(records[mode][2]["owner"]) for mode in MODES),
            "Distinct serial closed original coordinator and arm worker births required")

    entry=plan.entry
    first_scope=records["native"][1]
    require(Path(first_scope["code"])==owner.CODE
            and Path(first_scope["python"]["path"]).resolve(strict=True)==Path(sys.executable).resolve(strict=True),
            "Exact admitted Windows source root and isolated interpreter required")
    torch,_,_,deps=entry.admit_windows_runtime(dict(first_scope,operation="replay"))
    require(not torch.cuda.is_available() and not torch.cuda.is_initialized(),"Independent readback is CPU only")
    original_scope,replay,fit_ids=plan.old.inputs(entry,legacy,runtime,plan.old_lock)
    for mode,(arm,scope,_,_) in records.items():
        require(scope==plan.old.stage_scope(original_scope,mode,arm.output,replay,plan.old_lock["coco"]),
                "Exact admitted stock FIT scope for each new original E24 arm required")

    api=load_acceptance(owner.CODE)
    qualification=stable(owner.PAIR,64*1024,SHORT_SHA)
    args=dict(protocol=TRAINING_PROTOCOL,source_manifest_sha256=SOURCE_MANIFEST_SHA,expected_fit_ids=fit_ids,
              qualification_bytes=qualification,qualification_sha256=SHORT_SHA)
    _,_,qualified=api.verify_scope(**args)
    require(qualified==plan.qualification,"Original owner and independent qualification readback differ")
    export_path=owner.CODE/"RFDETR_NATIVE_E24_DEV_EXPORT_WORKER_CANDIDATE_20261107/e24_dev_export_worker.py"
    export_api=module(export_path,stable(export_path,64*1024,EXPORT_SHA),"_p351_original_ema_reader")
    layout=Path(first_scope["fit_layout"])
    _,_,model,init=deps.initialize_q100(layout,a.output,Path(first_scope["coco"]["path"]))
    inspect,checks=checkpoint_reader(api,export_api,deps,torch,model,init,qualified)
    arm_receipts,visits={},{}
    for mode,(arm,scope,terminal,stdout) in records.items():
        census=sorted(path.name for path in arm.output.iterdir())
        closure=dict(**api.CLOSURE,terminal_sha256=summary["terminal_refs"][mode]["sha256"],
                     stdout_sha256=terminal["stdout_ref"]["sha256"],output_files=census)
        report=stable(arm.output/api.REPORT_NAME,64*1024)
        peak=entry.document(report)["peak_allocated_bytes"]
        require(type(peak) is int and 0<peak<=6*GIB,"Exact original E24 CUDA allocator cap required")
        epochs=stable(arm.output/api.EPOCH_NAME,2*MIB)
        visits[mode]=stable(arm.output/api.VISIT_NAME,2*MIB)
        evidence=api.ClosedE24Arm(closure,report,epochs,visits[mode],stdout)
        arm_receipts[mode],_=api.validate_arm(evidence,mode=mode,
            expected_checkpoint_path=arm.output/api.CHECKPOINT_NAME,checkpoint_reader=inspect,**args)
    pair,pair_sha,stats=api.create_pair_receipt(*(arm_receipts[mode] for mode in MODES),
                                               *(visits[mode] for mode in MODES),**args)
    require(len(checks)==len(MODES) and not torch.cuda.is_initialized()
            and stable(plan.work/base.RESERVATION_NAME,64*1024)==reservation_raw
            and stable(summary_path,64*1024)==summary_raw,
            "Closed original authority and CPU scope must stay exact during independent readback")
    a.output.mkdir()
    for mode,raw in arm_receipts.items():
        write(a.output/(mode.upper()+"_E24_ACCEPTED_SAFE.json"),raw)
    write(a.output/PAIR_NAME,pair)
    safe=dict(status=OUTPUT,training_protocol=TRAINING_PROTOCOL,source_manifest_sha256=SOURCE_MANIFEST_SHA,
              training_seed=TRAINING_SEED,initialization_seed=INITIALIZATION_SEED,
              qualification_pair_sha256=SHORT_SHA,e24_pair_sha256=pair_sha,
              arm_receipt_sha256={mode:digest(raw) for mode,raw in arm_receipts.items()},
              original_owner_source_sha256=P350_SHA,original_owner_wait_sha256=digest(wait_raw),
              original_summary_sha256=summary_ref["sha256"],original_request_sha256=plan.request_sha,
              original_terminal_sha256={mode:summary["terminal_refs"][mode]["sha256"] for mode in MODES},
              independently_validated_raw_tensors_per_arm=545,
              independently_strict_ema_load={mode:check for mode,check in zip(MODES,checks)},
              full_fit_visit_pair_statistics=stats,**FIT_ONLY)
    write(a.output/"INDEPENDENT_ACCEPTANCE_SAFE.json",api.encode_receipt(safe))
    print(json.dumps(dict(status=OUTPUT,pair_sha256=pair_sha,
                          safe_sha256=digest(api.encode_receipt(safe)),**stats),sort_keys=True),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for key in ("owner-source","request","wait","output"):
        parser.add_argument("--"+key,type=Path,required=True)
    for key in ("request-sha256","wait-sha256"):
        parser.add_argument("--"+key,required=True)
    args=parser.parse_args()
    require(os.name=="nt" and sys.version_info[:3]==(3,10,13)
            and sys.flags.isolated==sys.flags.utf8_mode==1 and sys.dont_write_bytecode
            and os.environ.get("CUDA_VISIBLE_DEVICES")== "-1"
            and os.environ.get("HF_HUB_OFFLINE")==os.environ.get("TRANSFORMERS_OFFLINE")=="1",
            "Original isolated offline Windows CPU readback required")
    run(args)


if __name__=="__main__":
    main()
