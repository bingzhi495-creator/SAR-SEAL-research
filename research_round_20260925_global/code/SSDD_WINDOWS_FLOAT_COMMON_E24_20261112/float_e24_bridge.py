"""Run a fresh author-native RF-DETR SSDD floating/common FIT-only E24 arm."""
import argparse
from contextlib import ExitStack
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sys
from types import ModuleType
from unittest.mock import patch


SELF = Path(__file__).resolve()
ROOT = SELF.parent.parent
SHORT_STAGE = ROOT / "SSDD_WINDOWS_FLOAT_COMMON_SHORT_PUBLISHED_STAGE_PRIVATE_20261112_01"
SHORT_OWNER = SHORT_STAGE / "windows_float_common_short.py"
SHORT_SOURCE_SHA = "7b0b7b3f12d802f888240b1b979366d9df0304213b5eaceee5faa9d9711b3c27"
SHORT_REQUEST = SHORT_STAGE / "REQUEST_PRIVATE.json"
SHORT_REQUEST_SHA = "b4f4617b110267f4b56cf5c2c6c917c7821058e7213a83b07c1fa3477b8bc84c"
SHORT_OUTPUT = ROOT / "SSDD_WINDOWS_Q100_FLOAT_COMMON_PUBLISHED_ACTUAL_PRIVATE_20261112_01"
SHORT_ACCEPTANCE = SHORT_OUTPUT / "FLOAT_COMMON_SHORT_PAIR_ACCEPTED_SAFE.json"
SHORT_ACCEPTANCE_SHA = "78d2c6d1b99a1b05755324e061c1ad1bd4c5a460578e06e73720fa8c748ea8e0"
SHORT_STATUS = "SSDD_WINDOWS_FLOAT_COMMON_SHORT_PAIR_INDEPENDENTLY_ACCEPTED_V1"
RESULT_ROOT = ROOT / "SSDD_WINDOWS_Q100_FLOAT_COMMON_E24_ACTUAL_PRIVATE_20261112_01"
FIT_VISITS, EPOCH_VISITS, UPDATES = 17856, 744, 4464
SOURCE_MANIFEST = "56fdd4be8d19b4e2e42ff736e89775f32b1e6ce2a9c91d2a90afad509640bc14"
PROTOCOL = "194e16f289a644430015f310dd63a3c1c1618cfd"
WORKER_STATUS = "SSDD_WINDOWS_FLOAT_COMMON_E24_ARM_CLOSED_PENDING_INDEPENDENT_ACCEPTANCE_V1"
ARM_READY = "SSDD_WINDOWS_FLOAT_COMMON_E24_ORIGINAL_ARM_CLOSED_PENDING_ACCEPTANCE_V1"
FIT_SCOPE = dict(FIT=True, DEV=False, TEST=False, AP=False)


def need(ok, why):
    if not ok:
        raise ValueError(why)


def visit_position(cursor):
    need(type(cursor) is int and 0 <= cursor < FIT_VISITS, "Expected exact E24 visit cursor")
    epoch, batch = divmod(cursor, EPOCH_VISITS)
    return epoch + 1, batch + 1


def ref(path):
    hasher = sha256()
    with path.open("rb") as stream:
        first = os.fstat(stream.fileno())
        for chunk in iter(lambda: stream.read(1024**2), b""):
            hasher.update(chunk)
        last = os.fstat(stream.fileno())
    need((first.st_size,first.st_mtime_ns) == (last.st_size,last.st_mtime_ns), "Stable exact file")
    return dict(path=path.as_posix(), bytes=last.st_size, sha256=hasher.hexdigest())


def raw_pinned(path, expected):
    data = path.read_bytes()
    need(sha256(data).hexdigest() == expected, "Expected exact original source " + path.name)
    return data


def module(path, expected, name):
    obj = ModuleType(name)
    obj.__file__ = str(path)
    exec(compile(raw_pinned(path, expected), str(path), "exec"), obj.__dict__)
    return obj


def context():
    short = module(SHORT_OWNER, SHORT_SOURCE_SHA, "_qualified_ssdd_float_short")
    ctx = short.context(SHORT_REQUEST, SHORT_REQUEST_SHA)
    pair = json.loads(raw_pinned(SHORT_ACCEPTANCE, SHORT_ACCEPTANCE_SHA))
    need(pair["status"] == SHORT_STATUS and pair["source_manifest_sha256"] == SOURCE_MANIFEST
         and pair["training_protocol"] == PROTOCOL and pair["seed"] == 5201
         and pair["visits_per_arm"] == 32 and pair["optimizer_updates_per_arm"] == pair["ema_updates_per_arm"] == 8
         and pair["common_targets_total"] > 0 and pair["changed_mask_visits"] > 0
         and pair["FIT"] is True and pair["DEV"] is pair["TEST"] is pair["AP"] is False
         and ctx.worker.manifest(ctx.entry, dict(ctx.cfg,stage="short")) == SOURCE_MANIFEST
         and ref(Path(ctx.cfg["coco"]["path"])) == ctx.cfg["coco"],
         "Exact independently accepted float/common short with original COCO Q100 reference")
    return short, ctx, pair


def key_for(mode):
    return "SSDD_WINDOWS_Q100_FLOAT_COMMON_E24_" + mode.upper() + "_5201"


def launch_manifest(mode, stock_scope, worker, operation_commit):
    return dict(status="SSDD_WINDOWS_FLOAT_COMMON_E24_ARM_LAUNCH_V1",mode=mode,
                original_short_sha256=SHORT_ACCEPTANCE_SHA,original_short_request_sha256=SHORT_REQUEST_SHA,
                operation_commit=operation_commit,training_protocol=PROTOCOL,source_manifest_sha256=SOURCE_MANIFEST,
                stock_scope=stock_scope,worker=worker,complete_epochs=24,fit_visits=FIT_VISITS,
                optimizer_updates=UPDATES,optimizer_plan_updates=18600,warmup_updates=558,**FIT_SCOPE)


def check_arm_scope(entry, mode, scope_path, scope_sha, launch_path, launch_sha, cfg, operation_commit):
    arm_root = RESULT_ROOT / mode
    launch = entry.document(raw_pinned(launch_path, launch_sha))
    expected = launch_manifest(mode,ref(scope_path),ref(SELF),operation_commit)
    need(launch == expected and scope_path == arm_root/"WINDOWS_SCOPE_LOCK_PRIVATE.json"
         and launch_path == arm_root/"E24_FLOAT_LAUNCH_PRIVATE.json"
         and scope_sha == launch["stock_scope"]["sha256"] and Path.cwd() == arm_root/"CONTROL"
         and (Path.cwd()/"JOB_ATTACHED.txt").read_text("ascii") == launch["worker"]["sha256"],
         "Exact contained original E24 worker and immutable launch/scope")
    scope = entry.read_lock(scope_path, scope_sha, "short")
    need(Path(scope["output"]) == arm_root/"ACTUAL" and Path(scope["output"]).parent == arm_root
         and scope["mode"] == mode and scope["coco"] == cfg["coco"],
         "Exact fresh per-arm author E24 output and immutable COCO input")
    return scope, arm_root


def verified_actual(common_file, stock_file, native_visits, mode):
    native = [json.loads(line) for line in native_visits.read_bytes().splitlines()]
    need(len(native) == 24 and all(row["epoch"] == i and len(row["visits"]) == EPOCH_VISITS
                                  for i,row in enumerate(native,1)), "Actual native visit epoch boundaries")
    native_rows = [visit for row in native for visit in row["visits"]]
    identical_nonmask = sha256()
    selected_masks = sha256()
    total = 0
    with common_file.open("rb") as common, stock_file.open("rb") as stock:
        for index,(common_raw,stock_raw) in enumerate(zip(common,stock,strict=True)):
            c,s = json.loads(common_raw),json.loads(stock_raw)
            epoch,batch = visit_position(index)
            selected = c[mode+"_common_masks_sha256"]
            need(c["epoch"] == s["epoch"] == epoch and c["batch"] == s["batch"] == batch
                 and c["ordinal"] == s["ordinal"] == index+1
                 and c["image_id"] == s["image_id"] == native_rows[index][0]
                 and c["common_nonempty"] == s["labels"] == s["postfilter_labels"] == native_rows[index][1]
                 and s["boxes_sha256"] == s["postfilter_boxes_sha256"]
                 and selected == s["masks_sha256"] == s["postfilter_masks_sha256"],
                 "Exact native actual IDs, ordinal, selected masks, and common support")
            for field in ("masks_sha256","postfilter_masks_sha256"):
                del s[field]
            identical_nonmask.update(json.dumps(s,sort_keys=True,separators=(",",":")).encode() + b"\n")
            selected_masks.update(bytes.fromhex(selected))
            total += 1
    need(total == len(native_rows) == FIT_VISITS, "All actual E24 visits must be witnessed")
    return identical_nonmask.hexdigest(),selected_masks.hexdigest()


def run_worker(args):
    short,ctx,_ = context()
    entry = ctx.entry
    scope, root = check_arm_scope(entry,args.arm,args.scope,args.scope_sha256,args.launch,args.launch_sha256,ctx.cfg,args.operation_commit)
    base_layout,ids = entry.read_fit_population(scope)
    layout = short.select_layout(ctx,scope,base_layout,ids)
    torch,_,worker,deps = entry.admit_windows_runtime(scope)
    entry.admit_host()
    free,total,fraction = entry.admit_cuda_short(torch.cuda)
    entry.emit(dict(event="SSDD_WINDOWS_FLOAT_COMMON_E24_GPU_ADMITTED_V1",mode=args.arm,protocol=PROTOCOL,
                    cuda_free_bytes=free,cuda_total_bytes=total,cuda_allocator_fraction=fraction))
    audit = root/"COMMON_AUDIT"
    audit.mkdir()
    common_file = audit/"CONSUMED_COMMON_PRIVATE.jsonl"
    stock_file = audit/"ACTUAL_STOCK_CONSUMPTION_PRIVATE.jsonl"
    code = Path(scope["code"])
    p330 = ctx.worker.module(code/ctx.worker.P330_SOURCE[0],ctx.worker.P330_SOURCE[1],"p330")
    helpers = {name:p330.pinned(code,name) for name in ("trace_resize","common_mask_supervision")}
    with ExitStack() as stack:
        stack.enter_context(patch.dict(sys.modules,helpers))
        paired = p330.pinned(code,"paired_fit_transform")
        binding = p330.pinned(code,"common_fit_binding")
        stack.enter_context(patch.dict(sys.modules,{"paired_fit_transform":paired}))
        common = stack.enter_context(common_file.open("x",encoding="utf-8",newline="\n"))
        stock = stack.enter_context(stock_file.open("xb"))
        ledger = binding.ConsumedFitLedger(common)
        consume = ledger.consume

        def consume_full(*, image_id, common_count, epoch, batch):
            need(epoch == 1 and batch == ledger.consumed_visits+1, "Exact inherited short witness call seam")
            actual_epoch,actual_batch = visit_position(ledger.consumed_visits)
            return consume(image_id=image_id,common_count=common_count,epoch=actual_epoch,batch=actual_batch)

        ledger.consume = consume_full
        wrapped,chosen = ctx.worker.bind_dependencies(deps,torch,p330,paired,binding,layout,args.arm,ledger)
        native_retained = worker.__globals__["retained_target"]

        def observe_retained(active_torch,image,targets,allowed_ids,empty_guard,phase):
            need(active_torch is torch and phase is None and empty_guard is wrapped.empty_guard,
                 "Exact E24 native retained-target witness seam")
            ordinal = ledger.consumed_visits+1
            epoch,batch = visit_position(ordinal-1)
            iid,retained,removed,record = native_retained(torch,image,targets,allowed_ids,empty_guard,wrapped.phase)
            need(ledger.consumed_visits == ordinal and removed == 0, "Exact stock empty-mask guard, no removals")
            gpu_rng = sha256(torch.cuda.get_rng_state(0).cpu().numpy().tobytes()).hexdigest()
            row = dict(record,ordinal=ordinal,epoch=epoch,batch=batch,cuda_rng_sha256=gpu_rng)
            raw = (json.dumps(row,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
            need(stock.write(raw) == len(raw), "Exact consumed stock evidence write")
            return iid,retained,removed,None

        stack.enter_context(patch.dict(worker.__globals__,{"retained_target":observe_retained}))
        report = worker(stage="e24",mode=args.arm,output=root/"ACTUAL",fit_layout=layout,
                        coco_path=Path(scope["coco"]["path"]),expected_fit_ids=ids,
                        expected_initial_sha256=entry.Q100_SHA,protocol=PROTOCOL,recipe_sha256=entry.RECIPE_SHA,
                        source_manifest_sha256=SOURCE_MANIFEST,dependencies=wrapped,event_sink=entry.emit)
        stats = ledger.finish(FIT_VISITS)
        stock.flush()
        os.fsync(stock.fileno())
        common.flush()
        os.fsync(common.fileno())
    need(report["completed_epochs"] == 24 and report["visits"] == FIT_VISITS
         and report["optimizer_updates"] == report["ema_updates"] == UPDATES
         and report["planned_updates"] == 18600 and report["warmup_updates"] == 558
         and report["removed_empty_rows"] == 0 and report["checkpoint_saved"] is True
         and stats["common_targets_total"] > 0 and stats["changed_mask_visits"] > 0,
         "Exact native E24 progress, schedule, original empty guard and selected archive")
    nonmask, selected = verified_actual(common_file,stock_file,root/"ACTUAL"/"FIT_VISITS_PRIVATE.jsonl",args.arm)
    need(selected == chosen.hexdigest(), "Exact actual selected mask stream")
    safe = dict(status=WORKER_STATUS,mode=args.arm,training_protocol=PROTOCOL,source_manifest_sha256=SOURCE_MANIFEST,
                launch_sha256=args.launch_sha256,original_short_sha256=SHORT_ACCEPTANCE_SHA,
                report_sha256=ref(root/"ACTUAL"/"SSDD_PHASE_TRAIN_SAFE.json")["sha256"],
                stock_audit_sha256=ref(stock_file)["sha256"],common_ledger_sha256=ref(common_file)["sha256"],
                identical_fields_sha256=nonmask,selected_mask_stream_sha256=selected,
                complete_epochs=24,optimizer_updates=UPDATES,ema_updates=UPDATES,removed_empty_rows=0,
                **stats,**FIT_SCOPE)
    raw = (json.dumps(safe,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    with (audit/"COMMON_E24_SAFE.json").open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    entry.emit(dict(status=WORKER_STATUS,result_sha256=sha256(raw).hexdigest()))


def run_owner(args):
    short_module,ctx,pair = context()
    old,entry = ctx.old,ctx.entry
    short,legacy,runtime = old.load_guards("e24")
    qualification_ref = legacy.ref(SHORT_ACCEPTANCE)
    lock = dict(ctx.cfg,stage="e24")
    base,replay,ids = old.inputs(entry,legacy,runtime,lock)
    short_module.select_layout(ctx,base,Path(base["fit_layout"]),ids)
    mode = args.arm
    arm = old.controls(RESULT_ROOT,mode)
    arm.audit = RESULT_ROOT/mode/"COMMON_AUDIT"
    need(not (RESULT_ROOT/mode).exists() and len(ids) == 741 and "torch" not in sys.modules,
         "Exact fresh FIT-only CPU admission")
    if args.preflight_cpu:
        short.emit("SSDD_WINDOWS_FLOAT_COMMON_E24_CPU_PREFLIGHT_PASSED_V1", arm=mode, fit_images=len(ids),
                   visits_per_epoch=EPOCH_VISITS,planned_epochs=24,planned_fit_visits=FIT_VISITS,planned_optimizer_updates=UPDATES,
                   optimizer_plan_updates=18600,warmup_updates=558,source_manifest_sha256=SOURCE_MANIFEST,
                   torch_imported=False,model_forward_calls=0,optimizer_updates=0,**FIT_SCOPE)
        return 0
    need(mode == "native", "This original launch admits only native A; center is after independent A acceptance")
    need(re.fullmatch("[0-9a-f]{40}",args.operation_commit or "") is not None and args.operation_commit != PROTOCOL,
         "Exact distinct prospective E24 operation commit required")
    raw = legacy.P189["stable"](legacy.STATE,short.STATE_LIMIT)
    state = json.loads(raw)
    old.check_idle(state)
    old.verify_main_handoff(state,entry,legacy)
    accepted = state.get(short_module.AUTHORITY,{})
    need(accepted.get("status") == SHORT_STATUS and accepted.get("resource_closed") is True
         and accepted.get("active_attempt") is None and accepted.get("accepted_pair_ref") == qualification_ref
         and not legacy.live(accepted["owner_birth"]), "Original independently accepted float short is fully closed")
    key = key_for(mode)
    need(key not in state and not (RESULT_ROOT/mode).exists(), "Fresh exact original native E24")
    old.check_resources(runtime)
    arm.control.mkdir(parents=True)
    (arm.control/"scratch").mkdir()
    legacy.write_json(arm.lock,old.stage_scope(base,mode,arm.output,replay,lock["coco"]))
    scope_ref = legacy.ref(arm.lock)
    launch = RESULT_ROOT/mode/"E24_FLOAT_LAUNCH_PRIVATE.json"
    legacy.write_json(launch,launch_manifest(mode,scope_ref,ref(SELF),args.operation_commit))
    launch_ref = legacy.ref(launch)
    arm.arguments = ["--worker","--arm",mode,"--scope",str(arm.lock),"--scope-sha256",scope_ref["sha256"],
                     "--launch",str(launch),"--launch-sha256",launch_ref["sha256"],"--operation-commit",args.operation_commit]
    coordinator = legacy.birth(os.getpid())
    state[key] = dict(status="SSDD_WINDOWS_FLOAT_COMMON_E24_ORIGINAL_ARM_RESERVED_V1",active_attempt=key,
                      mode=mode,owner_birth=coordinator,resource_closed=False,operation_commit=args.operation_commit,
                      original_short_ref=qualification_ref,launch_ref=launch_ref,**FIT_SCOPE)
    state.update(GPU_owner_or_worker_active=True,gpu_training_active=True)
    legacy.cas(raw,state,"SSDD_WINDOWS_FLOAT_COMMON_E24_"+mode.upper()+"_RESERVE")
    reserved = legacy.P189["stable"](legacy.STATE,short.STATE_LIMIT)
    need(json.loads(reserved) == state, "Exact retained exclusive original reservation")
    short.emit("SSDD_WINDOWS_FLOAT_COMMON_E24_ORIGINAL_ARM_START_V1",arm=mode)
    terminal,guard = short.original(legacy,runtime,arm,SELF,reserved)
    terminal_ref = short.closed_arm(legacy,arm,terminal,coordinator)
    failure = terminal["stop_reason"] or guard
    if terminal["original_os_exit"] != 0 or terminal["job_active_before_drain"] != 0:
        failure = failure or ("Original Windows worker exit " + str(terminal["original_os_exit"]))
    state[key].update(status=ARM_READY if failure is None else "SSDD_WINDOWS_FLOAT_COMMON_E24_ORIGINAL_ARM_FAILED_V1",
                      active_attempt=None,resource_closed=True,terminal_ref=terminal_ref,failure=failure)
    state.update(GPU_owner_or_worker_active=False,gpu_training_active=False)
    legacy.cas(reserved,state,"SSDD_WINDOWS_FLOAT_COMMON_E24_"+mode.upper()+"_CLOSE")
    short.emit(state[key]["status"],arm=mode,original_exit=terminal["original_os_exit"],
               worker_wall_seconds=terminal["wall_seconds"],terminal_sha256=terminal_ref["sha256"],failure=failure)
    return 0 if failure is None else 5


def main():
    parser = argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight-cpu",action="store_true")
    action.add_argument("--run",action="store_true")
    action.add_argument("--worker",action="store_true")
    parser.add_argument("--arm",required=True,choices=("native","center"))
    parser.add_argument("--scope",type=Path)
    parser.add_argument("--scope-sha256")
    parser.add_argument("--launch",type=Path)
    parser.add_argument("--launch-sha256")
    parser.add_argument("--operation-commit")
    args = parser.parse_args()
    if args.worker:
        need(all(value is not None for value in (args.scope,args.scope_sha256,args.launch,args.launch_sha256)),
             "Exact original worker scope and launch references required")
        run_worker(args)
        return 0
    return run_owner(args)


if __name__ == "__main__":
    raise SystemExit(main())
