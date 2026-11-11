"""Independently verify closed float/common SSDD E24 original arms and matched pair."""
import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from types import ModuleType


ROOT = Path(__file__).resolve().parent.parent
BRIDGE = ROOT / "SSDD_WINDOWS_FLOAT_COMMON_E24_DRAFT_PRIVATE_20261112_02/float_e24_bridge.py"
BRIDGE_SHA = "eb3351588f27c985f777d8d620ddce097402a07aaf32281d692f6e217686d467"
ACCEPT_API = "5a22d2dc7a9dd67427fcedd3c6b45278c8abe8c9023ced33357206bc2d275604"
ARM_STATUS = "SSDD_WINDOWS_FLOAT_COMMON_E24_ARM_INDEPENDENTLY_ACCEPTED_V1"
PAIR_STATUS = "SSDD_WINDOWS_FLOAT_COMMON_E24_PAIR_INDEPENDENTLY_ACCEPTED_V1"
SCOPE = dict(FIT=True,DEV=False,TEST=False,AP=False)
CLOSURE = dict(original_os_wait=True,original_os_exit=0,resource_stop=False,worker_closed=True)
MODES = ("native","center")


def require(ok, why):
    if not ok:
        raise ValueError(why)


def canonical(data):
    return (json.dumps(data,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()


def raw_reference(path):
    raw = path.read_bytes()
    return raw,dict(sha256=sha256(raw).hexdigest(),bytes=len(raw))


def write(path, value):
    raw = canonical(value)
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(status=value["status"],result_sha256=sha256(raw).hexdigest()),sort_keys=True),flush=True)


def bridge():
    raw,_ = raw_reference(BRIDGE)
    require(sha256(raw).hexdigest() == BRIDGE_SHA, "Exact actual isolated-start bridge source")
    m = ModuleType("_original_ssdd_float_e24_bridge")
    m.__file__ = str(BRIDGE)
    exec(compile(raw,str(BRIDGE),"exec"),m.__dict__)
    return m


def same_fields(data, expected):
    return all(k in data and type(data[k]) is type(v) and data[k] == v for k,v in expected.items())


def accept_arm(args, m):
    short,ctx,qualification=m.context()
    entry=ctx.entry
    args.output.parent.mkdir(parents=True,exist_ok=True)
    require(not args.output.exists() and not args.output.is_symlink(), "Fresh independent arm acceptance")
    require(args.wait is not None and args.wait_sha256 is not None and args.original_stdout is not None,
            "Exact original outer wait and stdout required")
    receipt=json.loads(m.raw_pinned(args.wait,args.wait_sha256))
    outer=receipt["exact_tool_wait"]
    require(same_fields(receipt,dict(arm=args.arm,bridge_sha256=BRIDGE_SHA,original_os_wait=True))
            and outer["exit_code"] == 0 and outer.get("session_id") is None,
            "Captured exact successful original outer tool wait")
    original_bytes,_=raw_reference(args.original_stdout)
    require(original_bytes.decode("utf-8").splitlines() == outer["output"].splitlines(),
            "Original owner stdout must exactly agree with captured original wait")
    original_events=[json.loads(line) for line in original_bytes.splitlines()]
    require(len(original_events)==2 and original_events[0]==dict(
                status="SSDD_WINDOWS_FLOAT_COMMON_E24_ORIGINAL_ARM_START_V1",arm=args.arm)
            and same_fields(original_events[-1],dict(status=m.ARM_READY,arm=args.arm,original_exit=0,failure=None)),
            "Exact successful original arm progress and closure")

    root=m.RESULT_ROOT/args.arm
    control=root/"CONTROL"
    actual=root/"ACTUAL"
    audit=root/"COMMON_AUDIT"
    state_path=ROOT/"authoritative_operations_20261021_01/ACTIVE_RESEARCH_STATE.json"
    state=json.loads(state_path.read_bytes())
    record=state[m.key_for(args.arm)]
    require(same_fields(record,dict(status=m.ARM_READY,mode=args.arm,resource_closed=True,
                                    active_attempt=None,failure=None,**SCOPE)),
            "Exact independently closed original authority")
    terminal_path=control/"OWNER_TERMINAL_PRIVATE.json"
    terminal_bytes,terminal_ref=raw_reference(terminal_path)
    terminal=json.loads(terminal_bytes)
    birth=json.loads((control/"OWNER_BIRTH_PRIVATE.json").read_bytes())
    guard=json.loads((control/"RESOURCE_GUARD_PRIVATE.json").read_bytes())
    require(terminal["owner"] == birth and terminal["owner"]["creation_time"] >= record["owner_birth"]["creation_time"]
            and same_fields(terminal,dict(original_os_wait=True,original_os_exit=0,job_assigned=True,
                                          job_active_before_drain=0,job_active_after_wait=0,stop_reason=None))
            and terminal_ref["sha256"] == record["terminal_ref"]["sha256"] == original_events[-1]["terminal_sha256"]
            and terminal["wall_seconds"] == original_events[-1]["worker_wall_seconds"]
            and same_fields(guard,dict(guard=None,wall_limit_seconds=36000))
            and 0 < guard["peak_worker_RSS_bytes"] <= 10*1024**3,
            "Actual successful child wait, empty original job and exact resource evidence")
    for leaf,field in (("OWNER.stdout","stdout_ref"),("OWNER.stderr","stderr_ref")):
        require(m.ref(control/leaf) == terminal[field], "Exact original terminal log bytes")
    stdout=(control/"OWNER.stdout").read_bytes()
    require((control/"JOB_ATTACHED.txt").read_text("ascii") == BRIDGE_SHA,
            "Exact contained original bridge bytes")
    launch_path=root/"E24_FLOAT_LAUNCH_PRIVATE.json"
    require(m.ref(launch_path) == record["launch_ref"], "Exact original launch reference")
    launch=json.loads(launch_path.read_bytes())
    scope_path=root/"WINDOWS_SCOPE_LOCK_PRIVATE.json"
    require(launch == m.launch_manifest(args.arm,m.ref(scope_path),m.ref(BRIDGE),record["operation_commit"]),
            "Original arm operation, source, short qualification and stock scope unchanged")

    scope=entry.document(m.raw_pinned(scope_path,launch["stock_scope"]["sha256"]))
    require(same_fields(scope,dict(operation="short",mode=args.arm,coco=ctx.cfg["coco"]))
            and Path(scope["output"])==actual,
            "Exact immutable closed stock scope, archive, mode and original output")
    entry.verify_replay_receipt(scope)
    torch,_,deps,layout,ids=m.admit_worker_inputs(short,ctx,dict(scope,operation="replay"))
    require(not torch.cuda.is_initialized(), "Independent CPU archive validation only")
    code=Path(scope["code"])/"SSDD_MASK_PHASE_TRANSFER_20261110"
    api=m.module(code/"ssdd_phase_e24_acceptance.py",ACCEPT_API,"_exact_stock_ssdd_e24_acceptance")
    require(set(p.name for p in actual.iterdir()) == {api.REPORT_NAME,api.VISIT_NAME,api.EPOCH_NAME,api.CHECKPOINT_NAME},
            "Exact native FIT-only output roster")
    report_raw,report_ref=raw_reference(actual/api.REPORT_NAME)
    visit_raw,visit_ref=raw_reference(actual/api.VISIT_NAME)
    epoch_raw,epoch_ref=raw_reference(actual/api.EPOCH_NAME)
    sorted_ids=sorted(ids)
    population=Counter(sorted_ids)
    population.update(sorted_ids[i] for i in api.PAD_INDICES)
    phase_scope=dict(training_protocol=m.PROTOCOL,source_manifest_sha256=m.SOURCE_MANIFEST,seed=api.SEED,
                     fit_ids_sha256=api.digest(api.packed(sorted_ids)),initial_online_sha256=api.Q100_SHA,
                     coco_asset_sha256=api.COCO_SHA,recipe_sha256=api.RECIPE_SHA)
    require(same_fields(qualification,phase_scope),"Exact same official float short scientific scope")
    phase=api.expected_phase(phase_scope,args.arm)
    visits=api.read_visit_epochs(visit_raw,population)
    epochs=api.validate_epoch_audits(epoch_raw,visits)
    report=api.validate_report(report_raw,phase,epoch_raw,visit_raw,epochs)
    api.validate_events(stdout,report,visits,epochs)
    require(report["removed_empty_rows"] == 0 and 0 < report["peak_allocated_bytes"] <= 6*1024**3,
            "Exact common support and original CUDA allocator cap")

    _,_,initial_model,initial=deps.initialize_q100(layout,args.output.parent,Path(scope["coco"]["path"]))
    classifier=initial_model.model.class_embed.weight.detach().contiguous().clone()
    del initial_model
    checkpoint_reader=api.make_checkpoint_reader(torch=torch,native_inspector=deps.e100.inspect_archive,
        e1_api=deps.e1,boundary_api=deps.boundary,initial_classifier=classifier,initialization=initial,
        classifier_initial_sha256=qualification["classifier_initial_sha256"])
    archive=api.validate_archive(report["archive"],actual/api.CHECKPOINT_NAME,phase,checkpoint_reader)
    require(m.ref(actual/api.CHECKPOINT_NAME) == archive and not torch.cuda.is_initialized(),
            "Independent exact original E24 EMA archive and no CUDA initialization")

    common_file=audit/"CONSUMED_COMMON_PRIVATE.jsonl"
    stock_file=audit/"ACTUAL_STOCK_CONSUMPTION_PRIVATE.jsonl"
    common_raw,common_ref=raw_reference(common_file)
    _,stock_ref=raw_reference(stock_file)
    common=[json.loads(line) for line in common_raw.splitlines()]
    audit_raw,audit_ref=raw_reference(audit/"COMMON_E24_SAFE.json")
    audit_report=json.loads(audit_raw)
    nonmask,selected=m.verified_actual(common_file,stock_file,actual/api.VISIT_NAME,args.arm)
    require(json.loads(stdout.splitlines()[-1]) == dict(status=m.WORKER_STATUS,result_sha256=audit_ref["sha256"])
            and same_fields(audit_report,dict(status=m.WORKER_STATUS,mode=args.arm,
                 training_protocol=m.PROTOCOL,source_manifest_sha256=m.SOURCE_MANIFEST,
                 launch_sha256=record["launch_ref"]["sha256"],original_short_sha256=m.SHORT_ACCEPTANCE_SHA,
                 report_sha256=report_ref["sha256"],stock_audit_sha256=stock_ref["sha256"],
                 common_ledger_sha256=common_ref["sha256"],identical_fields_sha256=nonmask,selected_mask_stream_sha256=selected,
                 complete_epochs=24,optimizer_updates=4464,ema_updates=4464,removed_empty_rows=0,consumed_visits=17856,
                 common_targets_total=sum(row["common_nonempty"] for row in common),
                 common_empty_visits=sum(row["common_nonempty"] == 0 for row in common),
                 changed_mask_visits=sum(row["common_changed_masks"] > 0 for row in common),**SCOPE)),
            "Exact independently reconciled actual selected masks, native support, and full ordered common audit")

    result=dict(status=ARM_STATUS,phase=phase,qualification_pair_sha256=m.SHORT_ACCEPTANCE_SHA,
        operation_commit=record["operation_commit"],launch_sha256=record["launch_ref"]["sha256"],
        completed_epochs=24,visits=17856,optimizer_updates=4464,ema_updates=4464,
        checkpoint_sha256=archive["sha256"],checkpoint_bytes=archive["bytes"],result_sha256=report_ref["sha256"],
        epoch_audit_sha256=epoch_ref["sha256"],visit_ledger_sha256=visit_ref["sha256"],
        stdout_sha256=terminal["stdout_ref"]["sha256"],terminal_sha256=terminal_ref["sha256"],
        original_outer_wait_sha256=args.wait_sha256,common_ledger_sha256=common_ref["sha256"],
        stock_audit_sha256=stock_ref["sha256"],identical_fields_sha256=nonmask,selected_mask_stream_sha256=selected,
        common_targets_total=audit_report["common_targets_total"],common_empty_visits=audit_report["common_empty_visits"],
        changed_mask_visits=audit_report["changed_mask_visits"],filtered_empty_rows_total=0,**SCOPE,**CLOSURE)
    write(args.output,result)


def accept_pair(args,m):
    require(args.native_receipt is not None and args.center_receipt is not None, "Both independent original arms")
    raw_refs=[raw_reference(path) for path in (args.native_receipt,args.center_receipt)]
    accepted=[json.loads(raw) for raw,_ in raw_refs]
    a,b=accepted
    for mode,arm in zip(MODES,accepted,strict=True):
        require(same_fields(arm,dict(status=ARM_STATUS,qualification_pair_sha256=m.SHORT_ACCEPTANCE_SHA,
                       completed_epochs=24,visits=17856,optimizer_updates=4464,ema_updates=4464,
                       filtered_empty_rows_total=0,**CLOSURE,**SCOPE))
                and same_fields(arm["phase"],dict(stage="e24",mask_resize_mode=mode,seed=5201,
                       initialization_seed=5201,training_protocol=m.PROTOCOL,source_manifest_sha256=m.SOURCE_MANIFEST)),
                "Exact independently accepted official float E24 arm")
        root=m.RESULT_ROOT/mode
        require(m.ref(root/"ACTUAL"/"checkpoint_23.ckpt")["sha256"] == arm["checkpoint_sha256"]
                and m.ref(root/"COMMON_AUDIT"/"CONSUMED_COMMON_PRIVATE.jsonl")["sha256"] == arm["common_ledger_sha256"],
                "Exact independently closed native archive and common evidence")
    common=("common_ledger_sha256","identical_fields_sha256","common_targets_total","common_empty_visits","changed_mask_visits",
            "visit_ledger_sha256")
    require(all(a[k]==b[k] for k in common) and a["changed_mask_visits"]>0
            and a["selected_mask_stream_sha256"] != b["selected_mask_stream_sha256"]
            and a["checkpoint_sha256"] != b["checkpoint_sha256"]
            and a["terminal_sha256"] != b["terminal_sha256"],
            "Exact full-run common support, identical actual nonmask/RNG, distinct selected masks and original archives")
    result=dict(status=PAIR_STATUS,training_protocol=m.PROTOCOL,source_manifest_sha256=m.SOURCE_MANIFEST,
                qualification_pair_sha256=m.SHORT_ACCEPTANCE_SHA,seed=5201,fit_images=741,
                completed_epochs_per_arm=24,visits_per_arm=17856,optimizer_updates_per_arm=4464,ema_updates_per_arm=4464,
                arms={mode:ref for mode,(_,ref) in zip(MODES,raw_refs,strict=True)},
                common_ledger_sha256=a["common_ledger_sha256"],identical_fields_sha256=a["identical_fields_sha256"],
                selected_mask_stream_sha256={mode:arm["selected_mask_stream_sha256"] for mode,arm in zip(MODES,accepted,strict=True)},
                common_targets_total=a["common_targets_total"],common_empty_visits=a["common_empty_visits"],
                changed_mask_visits=a["changed_mask_visits"],identical_nonmask_inputs_and_rng=True,
                identical_common_support=True,**SCOPE)
    write(args.output,result)


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    actions=parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--arm",choices=MODES)
    actions.add_argument("--pair",action="store_true")
    parser.add_argument("--wait",type=Path)
    parser.add_argument("--wait-sha256")
    parser.add_argument("--original-stdout",type=Path)
    parser.add_argument("--native-receipt",type=Path)
    parser.add_argument("--center-receipt",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    m=bridge()
    if args.pair:
        accept_pair(args,m)
    else:
        accept_arm(args,m)


if __name__=="__main__":
    main()
