"""Run the exact bounded E24 D-FINE-seg-S HRSID FIT replay through author training cores."""
import argparse
from collections import Counter
from hashlib import sha256
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import sys
from sysconfig import get_path
from time import perf_counter
from types import ModuleType


SELF = Path(__file__).resolve()
DRAFT = SELF.parent
ROOT = DRAFT.parent
INTAKE = ROOT / "DFINE_SEG_EXTERNAL_BASELINE_INTAKE_20261112_01"
AUTHOR = INTAKE / "AUTHOR_SOURCE_VERIFIED"
FIT = ROOT / "HRSID_FIT_NATIVE_COCO_LAYOUT_ACTUAL_PRIVATE_20261109_01"
INDEX = FIT / "train/_annotations.coco.json"
REPLAY = ROOT / "HRSID_DFINE_E24_VISIT_REPLAY_DRAFT_20261112_01/ACTUAL"
OUT = ROOT / "HRSID_DFINE_E24_TRAIN_ACTUAL_20261112_01"
SHORT = INTAKE / "HRSID_FIT32_CUDA_TRAIN_QUAL_PRIVATE_20261112_01"
SHORT_SOURCE_SHA = "18745199ec0b2cff7e1e47ed6a0939f7e9d956583d34008dac4631a87598a214"
PROTOCOL_SHA = "684ab3a18dddc3a9887819050dc545c28736bbb3ccc664e83be9c67ef2b1d0a3"
FIT_ORDER_SHA = "c6bd988e2728f0f50e44dcd52134a970ca1a135c300ff15d828bc9950771dc0f"
FIT_ID_MAP_SHA = "5731bf5b6422171f9c90108b32a0607f131164f6e785e1b2a476c0ce82391939"
COCO_SHA = "07500e13824b82c9ee7422fd98736259e3bafcad65bf63f48d0c386a4ba987c1"
AUTHOR_CONFIG_SHA = "a94606169aa20d097b9d88290c50a054f2d7ba5b0a5a6df3728191c3f9f13fdb"
INDEX_SHA = "e95cf26cc60a2c6fb651e907541380a256f2df7b1e153244a6c0f803b0b19f8f"
FIT_MANIFEST_SHA = "a7e4d2a7ff9e8ff238adbf94a1d0397553d8746e2558c316db4e417f86f8fec9"
COMPLETE = "DFINE_SEG_S_HRSID_E24_CUDA_WORKER_COMPLETE_V1"
SCOPE = dict(FIT=True, DEV=False, TEST=False, AP=False)
EPOCHS, STEPS_PER_EPOCH, BATCH_SIZE = 24, 735, 4
ALLOCATOR_BYTES = 10*1024**3


def need(ok, why):
    if not ok:
        raise ValueError(why)


def pin(path, expected, size=None):
    need(path.resolve(strict=True) == path and path.is_file(), "Exact canonical file")
    data = path.read_bytes()
    need(sha256(data).hexdigest() == expected and (size is None or len(data) == size), "Exact frozen file " + path.name)
    return data


def encode(data):
    return (json.dumps(data, sort_keys=True, separators=(",",":"), allow_nan=False)+"\n").encode()


def write(path, raw):
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def emit(value):
    print(encode(value).decode("ascii").strip(),flush=True)


def file_ref(path):
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        digest = sha256()
        for chunk in iter(lambda: stream.read(1024**2),b""):
            digest.update(chunk)
        after = os.fstat(stream.fileno())
    need((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns), "Stable exact source file " + path.name)
    return dict(bytes=after.st_size,sha256=digest.hexdigest(),mtime_ns=after.st_mtime_ns)


def short_helpers():
    path = SHORT/"run_author_fit32_gpu.py"
    prior = ModuleType("_exact_closed_dfine_fit32_helpers")
    prior.__file__ = str(path)
    exec(compile(pin(path, SHORT_SOURCE_SHA),str(path),"exec"),prior.__dict__)
    return prior


def prepare():
    pin(DRAFT/"PROTOCOL.md",PROTOCOL_SHA)
    cfg = AUTHOR/"config.yaml"
    pin(cfg,AUTHOR_CONFIG_SHA)
    weights = INTAKE/"dfine_seg_s_coco.pt"
    pin(weights,COCO_SHA,48119167)
    prior = json.loads(pin(SHORT/"RESULT_SAFE.json","414b2198fb79aa54716f175f890aa9e4a9a6ff562a27f670c101a2df5dc9c20a"))
    pin(SHORT/"INDEPENDENT_READBACK_SAFE.json","0090abe01b10546b73e93230ad8d64015b739a3f78d740be1fc82cbc4011b3a3")
    pin(ROOT/"HRSID_DFINE_E24_READER_ACTUAL_20261112_02/RESULT_SAFE.json",
        "9abd4d5afcc90c483ca4baafcefa034b991202a70440d3abf8c2779177b89758")
    pin(REPLAY/"INDEPENDENT_READBACK_SAFE.json","70c41da7dc4bfe10bedfbbd0915ef3d950cdab4e3b73dfa162e79aa746d1a855")
    index = json.loads(pin(INDEX,INDEX_SHA,30076663))
    manifest = json.loads(pin(FIT/"NATIVE_FIT_LAYOUT_MANIFEST_PRIVATE.json",FIT_MANIFEST_SHA))
    images = {row["id"]:row for row in index["images"]}
    provenance = {row["image_id"]:row for row in manifest["images"]}
    mapping = {int(key):val for key,val in json.loads(pin(REPLAY/"FIT_ID_BASENAME_PRIVATE.json",FIT_ID_MAP_SHA)).items()}
    need(len(images)==len(provenance)==len(mapping)==2937 and images.keys()==provenance.keys()==mapping.keys()
         and len(index["annotations"])==9260 and index["categories"][0]["id"]==1 and len(index["categories"])==1,
         "Exact full FIT image and native polygon population")
    paths = {}
    for key,row in images.items():
        basename = mapping[key]
        need(Path(basename).name==basename==row["file_name"]==provenance[key]["native_basename"]
             and row["width"]==row["height"]==800, "Exact fixed FIT ID/name/geometry mapping")
        path = FIT/"train"/basename
        need(path.stat().st_size==provenance[key]["encoded_bytes"], "Exact FIT encoded byte length")
        paths[key] = path
    epochs = [json.loads(line) for line in pin(REPLAY/"FIT_ORDER_PRIVATE.jsonl",FIT_ORDER_SHA).splitlines()]
    need(len(epochs)==EPOCHS,"Exact 24 accepted RF visit epochs")
    for epoch,row in enumerate(epochs,1):
        batches = row["batches"]
        visits = [item for batch in batches for item in batch]
        counts = Counter(visits)
        need(set(row)=={"epoch","batches"} and row["epoch"]==epoch and len(batches)==STEPS_PER_EPOCH
             and all(len(batch)==BATCH_SIZE for batch in batches) and len(visits)==2940
             and counts.keys()==paths.keys() and sum(n-1 for n in counts.values())==3,
             "Exact accepted per-epoch FIT coverage, padding and batch grouping")
    install = Path(get_path("purelib"))/"dfine_seg"
    exact = 0
    for path in (AUTHOR/"dfine_seg").rglob("*"):
        if path.is_file() and path.suffix in {".py",".yaml",".yml"}:
            need((install/path.relative_to(AUTHOR/"dfine_seg")).read_bytes()==path.read_bytes(),"Exact installed author files")
            exact += 1
    versions = {name:version(name) for name in ("dfine-seg","torch","torchvision","albumentations")}
    need(exact==70 and tuple(versions[k] for k in ("dfine-seg","torch","torchvision"))==
         ("0.7.0","2.13.0+cu130","0.28.0+cu130"),"Exact author and CUDA version scope")
    return cfg,weights,prior,provenance,paths,epochs,versions


def validate_pngs(paths, provenance):
    snapshots = {}
    for key,path in paths.items():
        identity = file_ref(path)
        expected = provenance[key]
        need(identity["sha256"]==expected["encoded_sha256"] and identity["bytes"]==expected["encoded_bytes"],
             "Exact FIT PNG bytes from closed native layout")
        snapshots[key] = identity["mtime_ns"]
    return snapshots


def training(config, weights, prior, provenance, paths, epochs, versions):
    need(Path.cwd()==OUT/"CONTROL" and (Path.cwd()/"JOB_ATTACHED.txt").read_text("ascii")==file_ref(SELF)["sha256"],
         "Actual E24 requires exact original Windows JobObject worker")
    out = OUT/"ACTUAL"
    out.mkdir()
    started = perf_counter()
    snapshots = validate_pngs(paths, provenance)
    permitted = set(paths.values()) | {INDEX}
    def audit(event, args):
        if event in {"socket.connect","socket.getaddrinfo","urllib.Request"}:
            raise PermissionError("Exact offline FIT-only E24")
        if event=="open" and args and isinstance(args[0],(str,bytes)):
            path=Path(os.fsdecode(args[0]))
            if path.is_absolute() and path.is_relative_to(FIT) and path not in permitted:
                raise PermissionError("Only exact native FIT train index and admitted PNGs")
    sys.addaudithook(audit)
    import cv2
    from dfine_seg.dl.dataset import CustomDataset, Loader, load_coco_split
    from dfine_seg.dl.train import ModelEMA, ckpt_meta, training_forward, training_optimizer
    from dfine_seg.dl.utils import set_seeds
    from dfine_seg.model.dfine import build_loss, build_model
    from dfine_seg.model.utils import save_checkpoint, unwrap_checkpoint
    from omegaconf import OmegaConf
    import pandas as pd
    import torch

    helpers=short_helpers()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    device=torch.device("cuda:0")
    need(torch.version.cuda=="13.0" and torch.cuda.is_available() and torch.cuda.device_count()==1
         and torch.cuda.is_bf16_supported() and torch.cuda.get_device_capability(device)==(8,6)
         and "RTX 3090 Ti" in torch.cuda.get_device_name(device),"Exact local CUDA BF16 device")
    free,total=torch.cuda.mem_get_info(device)
    need(free>=12*1024**3,"Exact 12 GiB free CUDA admission")
    torch.cuda.set_per_process_memory_fraction(ALLOCATOR_BYTES/total,device)
    torch.backends.fp32_precision="ieee"
    torch.backends.cuda.matmul.fp32_precision="ieee"
    torch.backends.cudnn.fp32_precision="ieee"
    torch.cuda.reset_peak_memory_stats(device)

    cfg=OmegaConf.load(config)
    cfg.model_name,cfg.task="s","segment"
    cfg.train.label_to_name={0:"ship"}
    cfg.train.device,cfg.train.batch_size,cfg.train.num_workers="cuda",4,0
    cfg.train.use_one_class,cfg.train.coco_dataset=True,True
    cfg.train.mosaic_augs.mosaic_prob=0.0
    need(list(cfg.train.img_size)==[640,640] and not cfg.train.keep_ratio
         and cfg.train.augs.multiscale_prob==0 and cfg.train.seed==42 and cfg.train.epochs==55
         and cfg.train.amp_enabled and cfg.train.amp_dtype=="bfloat16"
         and cfg.train.use_muon and cfg.train.aux_optimizer=="adan" and cfg.train.use_scheduler
         and cfg.train.clip_max_norm==.1 and cfg.train.use_ema and cfg.train.ema_momentum==.9998,
         "Exact author CUDA BF16, native geometry, optimizer, clipping and EMA recipe")
    raw_entries,_=load_coco_split(INDEX,use_one_class=True)
    by_name={entry["file_name"]:entry for entry in raw_entries}
    positions={key:position for position,key in enumerate(paths)}
    entries=[dict(by_name[path.name],file_name=str(path)) for path in paths.values()]
    need(len(entries)==2937 and sum(len(e["targets"]) for e in entries)==9260,"Exact native author FIT parser")
    split=pd.DataFrame([entry["file_name"] for entry in entries])
    dataset=CustomDataset((640,640),OUT,split,False,"train",cfg,coco_annotations=entries)
    dataset.transform.set_random_seed(cfg.train.seed)
    collate=Loader.__new__(Loader)
    collate.multiscale_prob=0.0

    set_seeds(cfg.train.seed,cfg.train.cudnn_fixed)
    fresh=build_model("s",1,True,"cpu",img_size=(640,640),task="segment")
    fresh_state=fresh.state_dict()
    set_seeds(cfg.train.seed,cfg.train.cudnn_fixed)
    model=build_model("s",1,True,"cpu",img_size=(640,640),task="segment",pretrained_model_path=str(weights))
    initial=model.state_dict()
    official,_=unwrap_checkpoint(torch.load(weights,map_location="cpu",weights_only=True))
    reused,new=[],[]
    for name,value in initial.items():
        copied=name in official and official[name].shape==value.shape
        need(helpers.same_tensor(value,official[name] if copied else fresh_state[name]),"Exact original copied or seed42 class tensor")
        (reused if copied else new).append(name)
    initial_sha=helpers.state_sha(initial)
    need(len(reused)==806 and len(new)==9 and initial_sha==prior["initial_state_sha256"]
         and new==prior["seeded_new_tensor_names"],"Exact independently qualified fresh official COCO initialization")
    del initial,official,fresh,fresh_state,value
    model.to(device).train()
    criterion=build_loss("s",1,cfg.train.label_smoothing,True,task="segment",img_size=(640,640))
    optimizer=training_optimizer(model,cfg)
    ema=ModelEMA(model,cfg.train.ema_momentum)
    need(helpers.state_sha(ema.model.state_dict())==initial_sha and len(optimizer.param_groups)==5
         and all(g.get("aux_optimizer")=="adan" for g in optimizer.param_groups[:4])
         and optimizer.param_groups[4]["use_muon"],"Exact stock author five-group optimizer and EMA initialization")
    peak=cfg.train.base_lr*2*cfg.train.adan_lr_mult
    peaks=[peak]*4+[cfg.train.base_lr*20]
    need(peaks==[.0025]*4+[.005],"Exact published author OneCycle LR peaks")
    scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=peaks,epochs=55,steps_per_epoch=735,
                                                    pct_start=.1,cycle_momentum=False)
    need(scheduler.total_steps==40425,"Exact frozen author OneCycle horizon")
    parameters=dict(model.named_parameters())
    score_names=[n for n in parameters if n.startswith("decoder.dec_score_head.") and n.endswith(".weight")]
    clock_name=score_names[-1]
    score_clock=parameters[clock_name]
    mask_name=next(n for n,p in parameters.items() if "mask_decoder" in n and p.requires_grad and p.ndim>=2)
    tracked=parameters[mask_name]
    ema_weight=dict(ema.model.named_parameters())[mask_name]
    counts=Counter()
    def observed_forward(*_):
        counts["forward"]+=1
    def observed_step(*_):
        counts["optimizer"]+=1
    model.register_forward_hook(observed_forward)
    optimizer.register_step_post_hook(observed_step)
    emit(dict(status="DFINE_SEG_S_HRSID_E24_REAL_TRAIN_STARTED", initial_state_sha256=initial_sha,
              initialized_from_coco=806,seeded_class_tensors=9,verified_pngs=2937,original_polygon_targets=9260,
              author_dtype="bfloat16",loss_autocast=False,scheduler_total=40425,**SCOPE))
    optimizer.zero_grad()

    archives={}
    batches_path=out/"ACTUAL_BATCH_LEDGER_PRIVATE.jsonl"
    epochs_path=out/"ACTUAL_EPOCH_LEDGER_PRIVATE.jsonl"
    epoch_rows=[]
    global_step=0
    actual_targets=0
    nonempty_targets=0
    with batches_path.open("xb") as batches, epochs_path.open("xb") as epoch_log:
        for epoch_row in epochs:
            epoch=epoch_row["epoch"]
            epoch_start=perf_counter()
            retained_epoch,nonempty_epoch=0,0
            for batch_index,ids in enumerate(epoch_row["batches"],1):
                tick=perf_counter()
                global_step+=1
                need(all(paths[key].stat().st_mtime_ns==snapshots[key] for key in ids),"Admitted PNGs have not changed")
                inputs,targets,selected=collate.train_collate_fn([dataset[positions[key]] for key in ids])
                need(inputs.shape==(4,3,640,640) and inputs.dtype==torch.float32
                     and selected==[paths[key] for key in ids],"Actual ordered native source and physical batch4")
                retained,nonempty=[],[]
                for t in targets:
                    n=t["labels"].numel()
                    need(t["labels"].eq(0).all().item() and t["masks"].shape==(n,640,640) and t["boxes"].shape==(n,4),
                         "Exact native label, mask and box row alignment")
                    retained.append(n)
                    nonempty.append(int(t["masks"].flatten(1).any(1).sum().item()))
                retained_epoch+=sum(retained)
                nonempty_epoch+=sum(nonempty)
                inputs=inputs.to(device)
                targets=[{k:(v.to(device) if hasattr(v,"to") else v) for k,v in t.items()} for t in targets]
                lrs=[g["lr"] for g in optimizer.param_groups]
                if global_step<=8:
                    mask_before=helpers.tensor_sha(tracked)
                    ema_before=ema_weight.detach().clone()
                losses=training_forward(model,criterion,None,inputs,targets,enabled=True,dtype=torch.bfloat16)
                need("loss_mask_bce" in losses and "loss_mask_dice" in losses and all(v.ndim==0 and v.device.type=="cuda" for v in losses.values()),
                     "Stock native scalar CUDA criterion losses with autocast disabled")
                loss_values=torch.stack(tuple(losses.values()))
                need(torch.isfinite(loss_values).all().item(),"All native criterion terms finite")
                loss=sum(losses.values())
                loss.backward()
                if global_step<=8:
                    need(sum(nonempty)>0 and tracked.grad is not None and torch.isfinite(tracked.grad).all().item()
                         and bool(tracked.grad.abs().max().item()),"First-eight actual nonzero native mask gradients")
                grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.train.clip_max_norm).item())
                need(math.isfinite(grad) and grad>0,"Finite actual gradient clip before stock optimizer")
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                need(counts["optimizer"]==counts["forward"]==scheduler.last_epoch==global_step
                     and optimizer.state[score_clock]["step"]==global_step,"Exact no-skip independent optimizer clock")
                ema.update(global_step,model)
                counts["ema"]+=1
                torch.cuda.synchronize(device)
                row=dict(epoch=epoch,batch=batch_index,update=global_step,image_ids=ids,
                         retained_targets=retained,nonempty_targets=nonempty,optimizer_returns=counts["optimizer"],
                         scheduler_returns=scheduler.last_epoch,ema_returns=counts["ema"],
                         score_parameter_clock=optimizer.state[score_clock]["step"],loss=float(loss.item()),
                         mask_bce=float(losses["loss_mask_bce"].item()),mask_dice=float(losses["loss_mask_dice"].item()),
                         grad_before_clip=grad,learning_rates=lrs,seconds=perf_counter()-tick)
                if global_step<=8:
                    mask_after=helpers.tensor_sha(tracked)
                    momentum=ema.ema_scheduler(global_step)
                    expected=ema_before.mul_(momentum).add_((1-momentum)*tracked.detach())
                    need(mask_before!=mask_after and optimizer.state[tracked]["step"]==global_step
                         and helpers.same_tensor(expected,ema_weight),"First eight actual stock mask optimizer and EMA weights")
                    row.update(mask_before_sha256=mask_before,mask_after_sha256=mask_after,
                               mask_ema_sha256=helpers.tensor_sha(ema_weight),mask_optimizer_clock=optimizer.state[tracked]["step"],
                               mask_loss_dtype=str(losses["loss_mask_bce"].dtype),total_loss_dtype=str(loss.dtype))
                    emit(dict(status="DFINE_SEG_S_HRSID_E24_FIRST8_ACTUAL_RETURN",**row))
                batches.write(encode(row))
                if global_step<=8 or batch_index%100==0:
                    batches.flush()
                    if global_step>8:
                        emit(dict(status="DFINE_SEG_S_HRSID_E24_ACTUAL_PROGRESS",epoch=epoch,batch=batch_index,
                                  optimizer_updates=global_step,ema_updates=counts["ema"],**SCOPE))
                del losses,loss,loss_values,inputs,targets
            need(global_step==epoch*735 and counts["ema"]==global_step,"Exact actual native epoch boundary")
            actual_targets+=retained_epoch
            nonempty_targets+=nonempty_epoch
            summary=dict(epoch=epoch,optimizer_updates=global_step,ema_updates=counts["ema"],
                         scheduler_updates=scheduler.last_epoch,fit_visits=global_step*4,
                         retained_targets=retained_epoch,nonempty_targets=nonempty_epoch,wall_seconds=perf_counter()-epoch_start,
                         peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device),
                         peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device))
            if epoch in (1,8,24):
                leaf=f"E{epoch:02d}_EMA_PRIVATE.pt"
                archive=out/leaf
                state={k:v.detach().cpu() for k,v in ema.model.state_dict().items()}
                save_checkpoint(archive,state,ckpt_meta(cfg))
                ref=file_ref(archive)
                archives[str(epoch)]=dict(name=leaf,bytes=ref["bytes"],sha256=ref["sha256"],
                                          ema_state_sha256=helpers.state_sha(state),optimizer_updates=global_step)
                summary["ema_archive"]=archives[str(epoch)]
                del state
            batches.flush()
            os.fsync(batches.fileno())
            epoch_log.write(encode(summary))
            epoch_log.flush()
            os.fsync(epoch_log.fileno())
            epoch_rows.append(summary)
            emit(dict(status="DFINE_SEG_S_HRSID_E24_ACTUAL_EPOCH_COMPLETE",**summary,**SCOPE))

    need(global_step==17640 and counts==Counter(forward=17640,optimizer=17640,ema=17640)
         and scheduler.last_epoch==17640 and len(epoch_rows)==24,"Exact final E24 author clocks")
    state_path=out/"E24_TRAIN_STATE_PRIVATE.pt"
    torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(),torch_rng_state=torch.get_rng_state(),cuda_rng_state=torch.cuda.get_rng_state(device),
                    epoch=24,optimizer_updates=17640,ema_updates=17640,
                    ema_checkpoint_sha256=archives["24"]["sha256"],clock_name=clock_name),state_path)
    for key,path in paths.items():
        need(path.stat().st_mtime_ns==snapshots[key] and path.stat().st_size==provenance[key]["encoded_bytes"],
             "Exact admitted FIT source unchanged through closure")
    result=dict(status=COMPLETE,protocol_sha256=PROTOCOL_SHA,worker_source_sha256=file_ref(SELF)["sha256"],
                source_commit="b782ece206487792f36c165ca65548154cd12094",fit_order_sha256=FIT_ORDER_SHA,fit_id_map_sha256=FIT_ID_MAP_SHA,
                fit_index_sha256=INDEX_SHA,initial_state_sha256=initial_sha,initialization_seed=42,
                initialized_coco_tensors=806,seeded_new_class_tensors=9,versions=versions,
                completed_epochs=24,fit_visits=70560,batch_size=4,actual_forwards=counts["forward"],
                optimizer_updates=17640,ema_updates=17640,scheduler_updates=17640,retained_targets=actual_targets,
                nonempty_targets=nonempty_targets,planned_scheduler_updates=40425,precision="author_bfloat16_forward_FP32_criterion_context",
                learning_rate_final=scheduler.get_last_lr(),ema_archives=archives,final_training_state=file_ref(state_path),
                batch_ledger=file_ref(batches_path),epoch_ledger=file_ref(epochs_path),first8_actual_return_checks=True,
                worker_wall_seconds=perf_counter()-started,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device),
                peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device),**SCOPE)
    raw=encode(result)
    write(out/"RESULT_SAFE.json",raw)
    emit(dict(status=COMPLETE,result_sha256=sha256(raw).hexdigest(),optimizer_updates=17640,fit_visits=70560,**SCOPE))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--preflight",action="store_true")
    args=parser.parse_args()
    need(sys.version_info[:3]==(3,11,15) and sys.flags.isolated==sys.flags.utf8_mode==1 and sys.dont_write_bytecode
         and os.environ.get("CUDA_VISIBLE_DEVICES")=="0","Exact isolated author CUDA interpreter")
    context=prepare()
    if args.preflight:
        need(not OUT.exists() and "torch" not in sys.modules,"Fresh CPU source/data preflight only")
        emit(dict(status="DFINE_SEG_S_HRSID_E24_PREFLIGHT_PASS",protocol_sha256=PROTOCOL_SHA,
                  fit_images=2937,polygon_targets=9260,planned_visits=70560,planned_updates=17640,
                  initialized_cuda=False,model_forward_calls=0,**SCOPE))
    else:
        training(*context)


if __name__=="__main__":
    main()
