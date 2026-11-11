"""Independently accept only the original naturally closed D-FINE-seg-S E24 FIT replay."""
import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT/"HRSID_DFINE_E24_TRAIN_ACTUAL_20261112_01"
DRAFT = ROOT/"HRSID_DFINE_E24_TRAIN_DRAFT_20261112_01"
REPLAY = ROOT/"HRSID_DFINE_E24_VISIT_REPLAY_DRAFT_20261112_01/ACTUAL"
WAIT = ROOT/"HRSID_DFINE_E24_TRAIN_ORIGINAL_WAIT_PRIVATE_20261112_01"
KEY = "DFINE_SEG_S_HRSID_E24_WINDOWS_CUDA_20261112_01"
PROTOCOL_SHA = "684ab3a18dddc3a9887819050dc545c28736bbb3ccc664e83be9c67ef2b1d0a3"
WORKER_SHA = "a3d3f9ae300f209ca4251b9e564ee262c3637429c208ccbd4e390d6b2b628d1d"
OWNER_SHA = "34b704cda895fdeeb25755a0d6a2553d9feb126ec02c2672f36fda3bd0949f38"
ORDER_SHA = "c6bd988e2728f0f50e44dcd52134a970ca1a135c300ff15d828bc9950771dc0f"
MAP_SHA = "5731bf5b6422171f9c90108b32a0607f131164f6e785e1b2a476c0ce82391939"
INIT_SHA = "261c6b73525d29bee9e0d1a095b21ea67ca3d3fbbe2149e077996c0c85fd34a2"
CLOSED = "DFINE_E24_GPU_ORIGINAL_CLOSED_PENDING_ACCEPTANCE"
STATUS = "DFINE_SEG_S_HRSID_E24_CUDA_INDEPENDENT_READBACK_PASSED"
SCOPE = dict(FIT=True, DEV=False, TEST=False, AP=False)
EPOCHS = 24
STEPS = 735
UPDATES = EPOCHS*STEPS
VISITS = 4*UPDATES
GIB = 1024**3


def require(ok, why):
    if not ok:
        raise ValueError(why)


def equal_fields(value, expected):
    return all(key in value and type(value[key]) is type(item) and value[key]==item for key,item in expected.items())


def digest(raw):
    return sha256(raw).hexdigest()


def read(path, expected=None):
    require(path.is_file() and path.resolve(strict=True)==path, "Exact closed regular file")
    raw=path.read_bytes()
    require(expected is None or digest(raw)==expected, "Exact frozen file digest")
    return raw


def stable_ref(path):
    require(path.is_file() and path.resolve(strict=True)==path, "Exact closed regular file")
    h=sha256()
    with path.open("rb") as stream:
        first=os.fstat(stream.fileno())
        for chunk in iter(lambda:stream.read(1024**2),b""):
            h.update(chunk)
        last=os.fstat(stream.fileno())
    require((first.st_size,first.st_mtime_ns)==(last.st_size,last.st_mtime_ns), "Closed original file changed")
    return dict(bytes=last.st_size,sha256=h.hexdigest(),mtime_ns=last.st_mtime_ns)


def tensor_sha(tensor):
    return digest(tensor.detach().contiguous().cpu().numpy().tobytes())


def state_sha(state):
    h=sha256()
    for name in sorted(state):
        value=state[name]
        meta=json.dumps([name,list(value.shape),str(value.dtype)],separators=(",",":")).encode()
        h.update(meta+b"\n"+bytes.fromhex(tensor_sha(value)))
    return h.hexdigest()


def original_closure(wait_path, wait_sha):
    state=json.loads(read(ROOT/"authoritative_operations_20261021_01/ACTIVE_RESEARCH_STATE.json"))
    authority=state[KEY]
    require(state["GPU_owner_or_worker_active"] is state["gpu_training_active"] is False
            and equal_fields(authority,dict(status=CLOSED,active_attempt=None,resource_closed=True,
                                          failure=None,prelock_sha256=PROTOCOL_SHA,worker_source_sha256=WORKER_SHA,**SCOPE)),
            "Exact independently closed original authority before any training payload")
    require(wait_path==WAIT/"EXACT_ORIGINAL_TOOL_WAIT_PRIVATE.json", "Exact retained original outer wait path")
    receipt=json.loads(read(wait_path,wait_sha))
    outer=receipt["exact_tool_wait"]
    require(equal_fields(receipt,dict(original_os_wait=True,owner_source_sha256=OWNER_SHA,worker_source_sha256=WORKER_SHA))
            and outer["exit_code"]==0 and outer.get("session_id") is None, "Actual successful retained original outer process")
    coordinator=read(WAIT/"ORIGINAL_COORDINATOR.stdout")
    require(not read(WAIT/"ORIGINAL_COORDINATOR.stderr") and coordinator.decode().splitlines()==outer["output"].splitlines(),
            "Exact retained original outer stdout and zero stderr")
    launch,end=[json.loads(line) for line in coordinator.splitlines()]
    require(equal_fields(launch,dict(status="DFINE_E24_GPU_ORIGINAL_RESERVED",prelock_sha256=PROTOCOL_SHA,**SCOPE))
            and launch["owner_pid"]==authority["owner_birth"]["pid"]
            and equal_fields(end,dict(status=CLOSED,resource_closed=True,original_exit=0,failure=None,**SCOPE)),
            "Exact original reservation and terminal stdout")
    control=RUN/"CONTROL"
    terminal_raw=read(control/"OWNER_TERMINAL_PRIVATE.json",end["terminal_sha256"])
    terminal=json.loads(terminal_raw)
    require(digest(terminal_raw)==authority["terminal_ref"]["sha256"]
            and equal_fields(terminal,dict(original_os_wait=True,original_os_exit=0,job_assigned=True,
                                          job_active_before_drain=0,job_active_after_wait=0,stop_reason=None))
            and terminal["wall_seconds"]==end["original_wall_seconds"]
            and terminal["owner"]==json.loads(read(control/"OWNER_BIRTH_PRIVATE.json"))
            and read(control/"JOB_ATTACHED.txt").decode("ascii")==WORKER_SHA,
            "Exact original child OS exit and empty JobObject")
    guard=json.loads(read(control/"RESOURCE_GUARD_PRIVATE.json"))
    require(equal_fields(guard,dict(reason=None,wall_limit_seconds=108000))
            and guard["minimum_host_bytes"]>=16*GIB and guard["minimum_gpu_free_mib"]>=12288
            and 0<guard["peak_worker_tree_rss_bytes"]<=10*GIB, "Closed original resource guard limits")
    stdout=read(control/"OWNER.stdout",terminal["stdout_ref"]["sha256"])
    read(control/"OWNER.stderr",terminal["stderr_ref"]["sha256"])
    read(DRAFT/"dfine_e24_worker.py",WORKER_SHA)
    read(DRAFT/"dfine_e24_owner.py",OWNER_SHA)
    read(DRAFT/"PROTOCOL.md",PROTOCOL_SHA)
    return terminal,guard,stdout,end


def verify_ledgers(actual,report,stdout):
    order=[json.loads(line) for line in read(REPLAY/"FIT_ORDER_PRIVATE.jsonl",ORDER_SHA).splitlines()]
    names=json.loads(read(REPLAY/"FIT_ID_BASENAME_PRIVATE.json",MAP_SHA))
    require(len(order)==24 and len(names)==2937, "Exact sealed replay scope")
    batch_file=actual/"ACTUAL_BATCH_LEDGER_PRIVATE.jsonl"
    epoch_file=actual/"ACTUAL_EPOCH_LEDGER_PRIVATE.jsonl"
    require(stable_ref(batch_file)==report["batch_ledger"] and stable_ref(epoch_file)==report["epoch_ledger"],
            "Exact independently closed worker ledgers")
    epoch_rows=[json.loads(line) for line in read(epoch_file).splitlines()]
    events=[json.loads(line) for line in stdout.splitlines()]
    epochs=[v for v in events if v["status"]=="DFINE_SEG_S_HRSID_E24_ACTUAL_EPOCH_COMPLETE"]
    first8=[v for v in events if v["status"]=="DFINE_SEG_S_HRSID_E24_FIRST8_ACTUAL_RETURN"]
    require(len(epoch_rows)==len(epochs)==24 and len(first8)==8
            and events[0]["initial_state_sha256"]==INIT_SHA and events[-1]["result_sha256"]==digest(read(actual/"RESULT_SAFE.json")),
            "Exact original initialization, first-eight, all epoch and completion stdout")
    total_retained,total_nonempty,update=0,0,0
    previous_mask=None
    with batch_file.open("rb") as stream:
        for i,(fixed,epoch_row,event) in enumerate(zip(order,epoch_rows,epochs,strict=True),1):
            expected_batches=fixed["batches"]
            require(fixed["epoch"]==i and len(expected_batches)==735 and event==dict(status=event["status"],**epoch_row,**SCOPE),
                    "Exact original epoch marker and ledger")
            retained,nonempty=0,0
            for batch,ids in enumerate(expected_batches,1):
                update+=1
                row=json.loads(stream.readline())
                require(equal_fields(row,dict(epoch=i,batch=batch,update=update,image_ids=ids,optimizer_returns=update,
                                               scheduler_returns=update,ema_returns=update,score_parameter_clock=update))
                        and len(ids)==len(row["retained_targets"])==len(row["nonempty_targets"])==4
                        and all(str(key) in names for key in ids)
                        and all(type(a) is int and type(b) is int and 0<=b<=a
                                for a,b in zip(row["retained_targets"],row["nonempty_targets"],strict=True))
                        and row["seconds"]>0 and row["grad_before_clip"]>0
                        and all(math.isfinite(row[k]) for k in ("loss","mask_bce","mask_dice","grad_before_clip","seconds")),
                        "Exact actual replay order, retained support and no-skip update clocks")
                if update<=8:
                    require(first8[update-1]==dict(status=first8[update-1]["status"],**row)
                            and row["mask_optimizer_clock"]==update and row["mask_loss_dtype"]==row["total_loss_dtype"]=="torch.float32"
                            and row["mask_before_sha256"]!=row["mask_after_sha256"]
                            and (previous_mask is None or previous_mask==row["mask_before_sha256"]),
                            "Actual first-eight BF16-context/FP32-loss mask and EMA witness")
                    previous_mask=row["mask_after_sha256"]
                retained+=sum(row["retained_targets"])
                nonempty+=sum(row["nonempty_targets"])
            require(equal_fields(epoch_row,dict(epoch=i,optimizer_updates=update,ema_updates=update,
                                               scheduler_updates=update,fit_visits=update*4,
                                               retained_targets=retained,nonempty_targets=nonempty))
                    and epoch_row["wall_seconds"]>0, "Exact epoch boundary and original target counters")
            total_retained+=retained
            total_nonempty+=nonempty
        require(not stream.read(1) and update==17640 and report["retained_targets"]==total_retained
                and report["nonempty_targets"]==total_nonempty,"Exact all E24 actual rows")
    return epoch_rows


def verify_archives(actual,report,epoch_rows):
    import torch

    require(torch.version.cuda is None and not torch.cuda.is_initialized(), "Independent CPU archive inspection only")
    archives=report["ema_archives"]
    require(set(archives)=={"1","8","24"},"Exact locked E1, E8 and E24 archive roster")
    meta=dict(dfine_seg_version="0.7.0",model_name="s",task="segment",num_classes=1,
              in_channels=3,label_to_name={0:"ship"},img_size=[640,640],keep_ratio=False)
    for i in (1,8,24):
        value=archives[str(i)]
        path=actual/f"E{i:02d}_EMA_PRIVATE.pt"
        ref=stable_ref(path)
        require(equal_fields(value,dict(name=path.name,bytes=ref["bytes"],sha256=ref["sha256"],optimizer_updates=i*735))
                and value==epoch_rows[i-1]["ema_archive"],"Exact original EMA archive boundary reference")
        raw=torch.load(path,map_location="cpu",weights_only=True)
        require(raw["meta"]==meta and len(raw["model"])==815 and state_sha(raw["model"])==value["ema_state_sha256"],
                "Exact independently decoded original native EMA tensors and author metadata")
        del raw
    require(len({row["ema_state_sha256"] for row in archives.values()})==3, "Distinct original EMA integrity milestones")
    final=actual/"E24_TRAIN_STATE_PRIVATE.pt"
    require(stable_ref(final)==report["final_training_state"],"Exact original E24 online/optimizer/scheduler state")
    native=torch.load(final,map_location="cpu",weights_only=True)
    require(equal_fields(native,dict(epoch=24,optimizer_updates=17640,ema_updates=17640,
                                   ema_checkpoint_sha256=archives["24"]["sha256"]))
            and len(native["model"])==815 and state_sha(native["model"])!=archives["24"]["ema_state_sha256"]
            and native["scheduler"]["last_epoch"]==17640 and native["scheduler"]["total_steps"]==40425
            and native["torch_rng_state"].numel()>0 and native["cuda_rng_state"].numel()>0,
            "Exact independent online, scheduler, RNG and matched native EMA archive")
    groups=native["optimizer"]["param_groups"]
    states=native["optimizer"]["state"]
    require(len(groups)==5 and all(g.get("aux_optimizer")=="adan" for g in groups[:4]) and groups[4]["use_muon"]
            and any(row.get("step")==17640 for row in states.values()) and not torch.cuda.is_initialized(),
            "Original author optimizer group state and actual terminal Adan step")
    return archives["24"]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--wait",type=Path,required=True)
    parser.add_argument("--wait-sha256",required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    require(not args.output.exists() and args.output.parent.is_dir()
            and sys.flags.isolated==sys.flags.utf8_mode==1 and sys.dont_write_bytecode
            and os.environ.get("CUDA_VISIBLE_DEVICES")=="-1", "Fresh CPU-only acceptance output")
    term,guard,stdout,end=original_closure(args.wait,args.wait_sha256)
    actual=RUN/"ACTUAL"
    expected={"RESULT_SAFE.json","ACTUAL_BATCH_LEDGER_PRIVATE.jsonl","ACTUAL_EPOCH_LEDGER_PRIVATE.jsonl",
              "E01_EMA_PRIVATE.pt","E08_EMA_PRIVATE.pt","E24_EMA_PRIVATE.pt","E24_TRAIN_STATE_PRIVATE.pt"}
    require({p.name for p in actual.iterdir()}==expected,"Exact closed E24 output roster")
    result_raw=read(actual/"RESULT_SAFE.json",end["result_sha256"])
    report=json.loads(result_raw)
    require(equal_fields(report,dict(status="DFINE_SEG_S_HRSID_E24_CUDA_WORKER_COMPLETE_V1",protocol_sha256=PROTOCOL_SHA,
                                    worker_source_sha256=WORKER_SHA,fit_order_sha256=ORDER_SHA,fit_id_map_sha256=MAP_SHA,
                                    initial_state_sha256=INIT_SHA,initialization_seed=42,initialized_coco_tensors=806,
                                    seeded_new_class_tensors=9,completed_epochs=24,fit_visits=70560,batch_size=4,
                                    actual_forwards=17640,optimizer_updates=17640,ema_updates=17640,scheduler_updates=17640,
                                    planned_scheduler_updates=40425,first8_actual_return_checks=True,**SCOPE))
            and 0<report["peak_cuda_allocated_bytes"]<=report["peak_cuda_reserved_bytes"]<=10*GIB,
            "Exact matched E24 published protocol and actual clock report")
    epochs=verify_ledgers(actual,report,stdout)
    final=verify_archives(actual,report,epochs)
    safe=dict(status=STATUS,protocol_sha256=PROTOCOL_SHA,worker_source_sha256=WORKER_SHA,result_sha256=digest(result_raw),
              original_tool_wait_sha256=args.wait_sha256,worker_terminal_sha256=end["terminal_sha256"],original_exit=0,job_drained=True,
              completed_epochs=24,fit_visits=70560,optimizer_updates=17640,ema_updates=17640,scheduler_updates=17640,
              ema_checkpoint_sha256=final["sha256"],ema_checkpoint_bytes=final["bytes"],fit_order_sha256=ORDER_SHA,
              fit_id_map_sha256=MAP_SHA,seed=42,initial_state_sha256=INIT_SHA,stock_source_commit=report["source_commit"],
              retained_targets=report["retained_targets"],nonempty_targets=report["nonempty_targets"],
              original_wall_seconds=term["wall_seconds"],epoch_wall_seconds=sum(row["wall_seconds"] for row in epochs),
              peak_cuda_allocated_bytes=report["peak_cuda_allocated_bytes"],peak_cuda_reserved_bytes=report["peak_cuda_reserved_bytes"],
              peak_host_worker_rss_bytes=guard["peak_worker_tree_rss_bytes"],**SCOPE)
    raw=(json.dumps(safe,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    with args.output.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(status=STATUS,result_sha256=digest(raw))),flush=True)


if __name__=="__main__":
    main()
