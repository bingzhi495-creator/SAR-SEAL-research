"""SRNet author runner on the original fixed SSDD FIT741, at matched batch/update budget."""

import argparse
from collections import Counter
from contextlib import ExitStack
from hashlib import sha256
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType
from unittest.mock import patch
from zipfile import ZipFile

P344_SOURCE_SHA = "1c4216c1716aee97669ff49041f06bedfb23b49748eb024c0a9eefb9be2e7d04"
P344_REPORT_SHA = "29d049853c9ab50f785516263b9d83eea1818de7ecc6c99a6dc96bae8b3aed72"
P344_WAIT_SHA = "e319ce4dc89035e9d092633b2fb0b9cc596b4d4904643dbf3fbb51857da18c67"
EXTRA_AUTHOR_PINS = {
    "mmrotate/apis/train.py": "e6d2acbd47574f3aa66120214c169ed10dba9ee0abb6a7a5e0d7de61fa9faccc",
    "tools/train.py": "421fa328207bf0581173455773f7c0ac27c69e8c81efbf88269c57bb23292d90",
    "mmrotate/datasets/pipelines/RTransforms.py": "d5063607de87058099aba3fda60694f273419da9f44aafddfe963dae9fa0d821",
    "configs/_base_/default_runtime.py": "c65c120d8543f4493a3e33aaf6f8fda5220c412ab20097224f11ab9391ee772c",
}
BATCH = 4
EPOCHS = 24
FIT_IMAGES = 741
BATCHES_PER_EPOCH = 186
SHORT_UPDATES = 2
SEED = 5201
STATUS = "P346_SRNET_AUTHOR_BATCH4_MATCHED_TRAIN_COMPLETE_V1"
RECIPE = dict(epochs=EPOCHS, samples_per_gpu=BATCH, workers_per_gpu=0, precision="fp32",
              repeat_times=1, optimizer="SGD", learning_rate=.0025, momentum=.9, weight_decay=.0001,
              gradient_clip_norm=35, gradient_clip_norm_type=2, warmup_iterations=112,
              warmup_ratio=1/3, epoch_decay_milestones=[16,22], anomaly_detection=False,
              sampler="stock_non_distributed_GroupSampler", init_seed=SEED, training_seed=SEED)


def require(condition, message):
    if not condition:
        raise ValueError(message)


class VisitAudit:
    """Pair loader attempts with model-bound metadata for each actual batch."""

    def __init__(self, fit_ids, batches_per_epoch=BATCHES_PER_EPOCH):
        self.fit_ids = set(fit_ids)
        self.batches_per_epoch = batches_per_epoch
        self.attempts = []
        self.scheduled = Counter()
        self.consumed = Counter()
        self.rejected = 0
        self.batches = 0

    def attempt(self, image_id, accepted):
        require(image_id in self.fit_ids and type(accepted) is bool, "FIT loader attempted an unadmitted image")
        self.attempts.append((image_id,accepted))

    def consume(self, image_ids):
        scheduled, consumed, rejected = [], [], []
        needs_requested_id = True
        for image_id, accepted in self.attempts:
            if needs_requested_id:
                scheduled.append(image_id)
                needs_requested_id = False
            if accepted:
                consumed.append(image_id)
                needs_requested_id = True
            else:
                rejected.append(image_id)
        require(needs_requested_id and len(scheduled) == len(consumed) == BATCH and consumed == list(image_ids),
                "Original sampler, augmentation attempts and actual model batch do not reconcile")
        self.attempts.clear()
        self.scheduled.update(scheduled)
        self.consumed.update(consumed)
        self.rejected += len(rejected)
        self.batches += 1
        return dict(scheduled_ids=scheduled,consumed_ids=consumed,rejected_attempt_ids=rejected)

    def close_epoch(self):
        visits = self.batches_per_epoch * BATCH
        require(not self.attempts and self.batches == self.batches_per_epoch
                and set(self.scheduled) == self.fit_ids and sum(self.scheduled.values()) == visits
                and sum(self.consumed.values()) == visits,
                "Complete native scheduled FIT roster and actual model visit count required")
        value = dict(scheduled_visits=visits,consumed_visits=visits,scheduled_unique=len(self.scheduled),
                     scheduled_padding_visits=visits-len(self.scheduled),consumed_unique=len(self.consumed),
                     rejected_attempts=self.rejected,optimizer_updates=self.batches)
        self.scheduled.clear()
        self.consumed.clear()
        self.rejected = self.batches = 0
        return value


class ShortBoundary(Exception):
    """Exit the stock training loop at the exact two-update qualification boundary."""


def encode(value):
    return (json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode("utf-8")


def exact(path,digest,maximum=1024**2):
    require(path.is_absolute() and path.resolve(strict=True)==path
            and path.is_file() and 0<path.stat().st_size<=maximum,"Canonical bounded input required")
    raw=path.read_bytes()
    require(sha256(raw).hexdigest()==digest,"Exact admitted file digest required")
    return raw


def verify_short(a, own_sha):
    if a.stage=="short":
        require(all(getattr(a,k) is None for k in
                    ("short_report","short_report_sha256","short_terminal","short_terminal_sha256","short_stdout")),
                "Short qualification must have its own fresh original output")
        return None
    short=json.loads(exact(a.short_report,a.short_report_sha256))
    terminal=json.loads(exact(a.short_terminal,a.short_terminal_sha256))
    stdout=exact(a.short_stdout,terminal["log_sha256"]["ORIGINAL.stdout"],4*1024**2)
    require(terminal["original_os_wait"] is True and type(terminal["exit_code"]) is int
            and terminal["exit_code"]==0 and terminal["job_assigned"] is True
            and terminal["active_before_drain"]==terminal["active_after_wait"]==0
            and terminal["resource_stop"] is None and terminal["source_sha256"]==own_sha,
            "Closed original batch-four Windows qualification required")
    require(short["status"]==STATUS and short["stage"]=="short" and short["worker_source_sha256"]==own_sha
            and short["recipe"]==RECIPE and short["optimizer_updates"]==SHORT_UPDATES
            and short["consumed_visits"]==SHORT_UPDATES*BATCH and short["completed_epochs"]==0
            and short["checkpoint"] is None and short["FIT"] is True
            and short["DEV"] is short["TEST"] is short["AP"] is False,
            "Same-source, same-recipe successful batch-four original qualification required")
    require(json.loads(stdout.splitlines()[-1])==dict(status=STATUS,result_sha256=a.short_report_sha256),
            "Original short worker stdout must bind exact successful report")
    return dict(report_sha256=a.short_report_sha256,terminal_sha256=a.short_terminal_sha256,
                stdout_sha256=terminal["log_sha256"]["ORIGINAL.stdout"])


def admit_inputs(a):
    raw=exact(a.p344_source,P344_SOURCE_SHA)
    p344=ModuleType("_p346_p344")
    p344.__file__=str(a.p344_source)
    exec(compile(raw,str(a.p344_source),"exec"),p344.__dict__)
    closed=json.loads(exact(a.p344_wait,P344_WAIT_SHA))
    prior=json.loads(exact(a.p344_result,P344_REPORT_SHA))
    require(closed["source_sha256"]==P344_SOURCE_SHA and closed["original_os_wait"] is True
            and closed["exit_code"]==closed["active_after_wait"]==0 and closed["resource_stop"] is None
            and prior["status"]==p344.STATUS and prior["optimizer_updates"]==2
            and prior["actual_batch_size"]==1 and prior["FIT"] is True
            and prior["DEV"] is prior["TEST"] is prior["AP"] is False,
            "Exact original two-step CUDA readiness witness required")
    p336=p344.load_source(a.p336_source,p344.P336_SHA)
    p335=p336.load_prior(a.p335_source)
    fit_prior=json.loads(p336.read_exact(a.prior_report,p344.P335_REPORT_SHA,64*1024))
    require(fit_prior["status"]==p335.STATUS and fit_prior["author_parsed_images"]==FIT_IMAGES
            and fit_prior["author_parsed_targets"]==1661,"Exact original FIT741 admission required")
    pins={**p335.SOURCE_PINS,**p344.TRAIN_PINS,**EXTRA_AUTHOR_PINS}
    for name,digest in pins.items():
        p336.read_exact(a.source/name,digest,1024**2)
    index_path=a.fit/"_annotations.coco.json"
    index=json.loads(p336.read_exact(index_path,p335.FIT_INDEX_SHA,2*1024**2))
    require(len(index["images"])==FIT_IMAGES and len(index["annotations"])==1661,
            "Exact fixed FIT741 image and annotation counts required")
    p336.read_exact(a.archive,p335.FIT_SHA,34*1024**2)
    with ZipFile(a.archive) as archive:
        manifest={row["image_id"]:row for row in json.loads(archive.read("manifest.json"))["rows"]}
    roster={}
    for row in index["images"]:
        name=row["file_name"]
        item=manifest[row["id"]]
        require(name==Path(name).name==item["original_basename"] and name not in roster,
                "Exact unique original FIT-only basename required")
        p336.read_exact(a.fit/name,item["original_image_ref"]["sha256"],4*1024**2)
        roster[name]=row["id"]
    require(len(roster)==len(set(roster.values()))==len(manifest)==FIT_IMAGES,
            "Full original FIT741 image byte census required")
    require(len(p336.read_exact(a.r50,p336.R50_SHA,p336.R50_BYTES))==p336.R50_BYTES,
            "Exact supplied author ImageNet initializer required")
    return p344,p335,p336,pins,index_path,roster


def resource_guard(p344, started, limits):
    observed=p344.resources()
    require(observed["available_bytes"]>=limits["host_available_floor_bytes"]
            and observed["peak_rss_bytes"]<=limits["process_rss_cap_bytes"]
            and perf_counter()-started<=limits["worker_seconds"],"SRNet resource limit reached")
    return observed


def runtime(a,p344,limits):
    import cv2
    import mmcv
    import mmrotate
    import torch

    versions={name:version(name) for name in p344.PACKAGES}
    require(versions==p344.PACKAGES and Path(mmrotate.__file__).resolve()==a.source/"mmrotate/__init__.py",
            "Exact author CUDA packages and source registry required")
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    require(torch.cuda.is_available() and torch.cuda.device_count()==1,"One CUDA device required")
    torch.cuda.init()
    free,total=torch.cuda.mem_get_info(0)
    require(free>=limits["cuda_free_floor_bytes"],"CUDA memory below original six-GiB admission floor")
    torch.cuda.set_per_process_memory_fraction(min(.35,limits["cuda_allocator_cap_bytes"]/total),0)
    torch.cuda.reset_peak_memory_stats(0)
    return mmcv,torch,versions,free


def configure(cfg,a):
    require(cfg.data.train.type=="RepeatDataset" and cfg.data.train.times==3
            and cfg.data.samples_per_gpu==BATCH and cfg.runner.type=="EpochBasedRunner"
            and cfg.runner.max_epochs==36 and list(cfg.lr_config.step)==[24,33]
            and cfg.lr_config.warmup_iters==500 and cfg.lr_config.warmup=="linear" and cfg.lr_config.warmup_ratio==1/3
            and dict(cfg.optimizer)==dict(type="SGD",lr=.0025,momentum=.9,weight_decay=.0001)
            and dict(cfg.optimizer_config.grad_clip)==dict(max_norm=35,norm_type=2)
            and cfg.get("fp16") is None,"Exact source author SGD, batch, precision, and schedule required")
    cfg.data.train=cfg.data.train.dataset.copy()
    cfg.data.workers_per_gpu=0
    cfg.data.train_dataloader=dict(shuffle=True,drop_last=False,persistent_workers=False)
    cfg.gpu_ids=[0]
    cfg.seed=SEED
    cfg.runner.max_epochs=EPOCHS
    cfg.lr_config.step=list(RECIPE["epoch_decay_milestones"])
    cfg.lr_config.warmup_iters=RECIPE["warmup_iterations"]
    cfg.work_dir=str(a.output)
    cfg.resume_from=cfg.load_from=None
    cfg.auto_resume=False
    cfg.workflow=[("train",1)]
    cfg.checkpoint_config=(None if a.stage=="short" else dict(
        interval=EPOCHS,save_last=True,save_optimizer=True,create_symlink=False,
        max_keep_ckpts=1,meta=dict(CLASSES=("ship",))))
    cfg.log_config.interval=50
    cfg.model.backbone.init_cfg.checkpoint=str(a.r50)


def train_with_audit(a,p344,torch,model,dataset,cfg,roster,limits,started):
    from mmcv.parallel import MMDataParallel
    from mmcv.runner import EpochBasedRunner,HOOKS,Hook
    from mmdet.datasets.samplers import GroupSampler
    from mmrotate.apis import train_detector

    audit=VisitAudit(roster.values())
    progress=dict(updates=0,epochs=0,rejected=0,rows=0,completed=False)
    original_prepare=dataset.prepare_train_img

    def prepare(index):
        source_id=dataset.data_infos[index]["id"]
        item=original_prepare(index)
        audit.attempt(source_id,item is not None)
        return item

    short=a.stage=="short"
    visits_path=a.output/"FIT_VISITS_PRIVATE.jsonl"
    epoch_path=a.output/"EPOCH_AUDIT_SAFE.jsonl"
    with ExitStack() as resources:
        visits=resources.enter_context(visits_path.open("xb"))
        epochs=resources.enter_context(epoch_path.open("xb")) if not short else None
        resources.enter_context(patch.object(dataset,"prepare_train_img",prepare))

        class BudgetObserver(Hook):
            def before_run(self,runner):
                require(type(runner) is EpochBasedRunner and type(runner.model) is MMDataParallel
                        and runner.model.module is model and type(runner.optimizer) is torch.optim.SGD
                        and runner.epoch==runner.iter==0 and len(runner.optimizer.state)==0,
                        "Fresh author EpochBasedRunner, single CUDA MMDataParallel and SGD required")
                original_step=runner.optimizer.step

                def observe_step(*args,**kwargs):
                    result=original_step(*args,**kwargs)
                    progress["updates"]+=1
                    return result

                resources.enter_context(patch.object(runner.optimizer,"step",observe_step))
                resource_guard(p344,started,limits)

            def before_train_epoch(self,runner):
                loader=runner.data_loader
                require(loader.dataset is dataset and type(loader.sampler) is GroupSampler
                        and loader.batch_size==BATCH and loader.num_workers==0 and len(loader)==BATCHES_PER_EPOCH
                        and len(loader.sampler)==BATCH*BATCHES_PER_EPOCH,
                        "Exact stock single-group 741->744 sampler and batch-four loader required")

            def before_train_iter(self,runner):
                batch=runner.data_batch
                images=batch["img"].data
                metadata=batch["img_metas"].data
                require(len(images)==len(metadata)==1 and tuple(images[0].shape)==(BATCH,3,608,608)
                        and len(metadata[0])==BATCH and runner.iter==progress["updates"]==progress["rows"],
                        "Exact collated four-image 608 square native batch and step cursor required")
                self.consumed=[roster[item["ori_filename"]] for item in metadata[0]]

            def after_train_iter(self,runner):
                loss=runner.outputs["log_vars"]
                require(runner.outputs["num_samples"]==BATCH
                        and progress["updates"]==runner.iter+1==progress["rows"]+1
                        and set(loss)==set(p344.LOSS_KEYS)|{"loss"}
                        and all(math.isfinite(float(value)) for value in loss.values()),
                        "Four consumed images, five finite original losses and one true SGD step required")
                clip=float(runner.log_buffer.val_history["grad_norm"][-1])
                require(math.isfinite(clip),"Finite stock gradient clipping norm required")
                row=audit.consume(self.consumed)
                progress["rows"]+=1
                progress["rejected"]+=len(row["rejected_attempt_ids"])
                row.update(epoch=runner.epoch+1,batch=runner.inner_iter+1,optimizer_update=progress["updates"])
                visits.write(encode(row))
                if short:
                    torch.cuda.synchronize()
                    resource_guard(p344,started,limits)
                    print(encode(dict(event="P346_SRNET_BATCH4_SGD_UPDATE",update=progress["updates"],
                                      batch_size=BATCH,losses=loss,gradient_norm=clip,
                                      rejected_attempts=len(row["rejected_attempt_ids"]))).decode().strip(),flush=True)
                    if progress["updates"]==SHORT_UPDATES:
                        raise ShortBoundary

            def after_train_epoch(self,runner):
                require(not short and runner.epoch==progress["epochs"]
                        and progress["updates"]==(runner.epoch+1)*BATCHES_PER_EPOCH,
                        "Exact full original epoch/update clock required")
                record=audit.close_epoch()
                torch.cuda.synchronize()
                resource_guard(p344,started,limits)
                progress["epochs"]+=1
                record.update(epoch=progress["epochs"],cumulative_updates=progress["updates"])
                epochs.write(encode(record))
                visits.flush()
                epochs.flush()
                print(encode(dict(event="P346_SRNET_BATCH4_EPOCH",**record)).decode().strip(),flush=True)

            def after_run(self,runner):
                require(not short and runner.epoch==progress["epochs"]==EPOCHS
                        and runner.iter==progress["updates"]==EPOCHS*BATCHES_PER_EPOCH,
                        "Full stock E24 runner completion required")
                progress["completed"]=True

        hook_name="SARSealSRNetBatch4BudgetObserver"
        HOOKS.register_module(name=hook_name,module=BudgetObserver)
        cfg.custom_hooks=[dict(type=hook_name,priority="NORMAL")]
        try:
            with torch.autograd.set_detect_anomaly(RECIPE["anomaly_detection"]):
                train_detector(model,dataset,cfg,distributed=False,validate=False,timestamp="srnet_batch4",
                               meta=dict(seed=SEED,exp_name="SRNet_R50_FIT741_batch4"))
        except ShortBoundary:
            require(short and progress["updates"]==progress["rows"]==SHORT_UPDATES and progress["epochs"]==0
                    and not progress["completed"] and not audit.attempts,
                    "Only the exact completed two-update qualification may stop the short runner")
        else:
            require(not short and progress["completed"],"Expected stock E24 run completion")
        for stream in (visits,epochs):
            if stream is not None:
                stream.flush()
                os.fsync(stream.fileno())
    return progress,visits_path,epoch_path if not short else None


def file_ref(path):
    digest=sha256()
    with path.open("rb") as stream:
        size=os.fstat(stream.fileno()).st_size
        require(size>0,"Nonempty completed artifact required")
        for chunk in iter(lambda:stream.read(1024**2),b""):
            digest.update(chunk)
        require(os.fstat(stream.fileno()).st_size==size,"Completed artifact size changed during readback")
    return dict(name=path.name,bytes=size,sha256=digest.hexdigest())


def run(a,own_sha,started):
    short_proof=verify_short(a,own_sha)
    p344,p335,p336,pins,index,roster=admit_inputs(a)
    limits=dict(p344.LIMITS)
    if a.stage=="e24":
        limits["worker_seconds"]=36000
    resource_guard(p344,started,limits)
    sys.path.insert(0,str(a.source))
    mmcv,torch,versions,free=runtime(a,p344,limits)
    from mmdet.apis import set_random_seed
    from mmrotate.datasets import build_dataset
    from mmrotate.models import build_detector

    cfg=mmcv.Config.fromfile(str(a.source/p335.AUTHOR_CONFIG))
    configure(cfg,a)
    cfg.data.train.ann_file,cfg.data.train.img_prefix=str(index),str(a.fit)+os.sep
    set_random_seed(SEED,deterministic=False)
    model=build_detector(cfg.model,train_cfg=cfg.get("train_cfg"),test_cfg=cfg.get("test_cfg"))
    model.init_weights()
    expected=torch.load(str(a.r50),map_location="cpu")
    require(torch.equal(model.backbone.conv1.weight.detach(),expected["conv1.weight"]),
            "Exact official author ImageNet backbone initialization required")
    del expected
    dataset=build_dataset(cfg.data.train)
    require(type(dataset).__module__=="mmrotate.datasets.SSDD" and dataset.CLASSES==("ship",)
            and dataset.version=="le90" and len(dataset)==FIT_IMAGES and dataset.flag.shape==(FIT_IMAGES,)
            and not dataset.flag.any() and {r["filename"]:r["id"] for r in dataset.data_infos}==roster,
            "Unmodified author dataset registry, single group and original FIT741 roster required")
    model.CLASSES=dataset.CLASSES
    watched={name:next(p for p in getattr(model,name).parameters() if p.requires_grad) for name in p344.HEADS}
    initial={name:p.detach().cpu().clone() for name,p in watched.items()}
    progress,visits,epochs=train_with_audit(a,p344,torch,model,dataset,cfg,roster,limits,started)
    torch.cuda.synchronize()
    movement={name:float((param.detach().cpu()-initial[name]).abs().max()) for name,param in watched.items()}
    require(all(math.isfinite(value) for value in movement.values()) and any(value>0 for value in movement.values()),
            "Observed trained head parameters must move")
    resources=resource_guard(p344,started,limits)
    allocated,reserved=torch.cuda.max_memory_allocated(0),torch.cuda.max_memory_reserved(0)
    require(0<allocated<=reserved<=limits["cuda_allocator_cap_bytes"],"CUDA peak outside original allocator cap")
    archives=list(a.output.glob("*.pth"))
    if a.stage=="short":
        require(not archives,"Short qualification may not write a trained archive")
        checkpoint=None
    else:
        final=a.output/"epoch_24.pth"
        require(archives==[final] and final.is_file(),"Exactly one stock E24 snapshot required")
        checkpoint=file_ref(final)
    for name,digest in pins.items():
        p336.read_exact(a.source/name,digest,1024**2)
    return dict(status=STATUS,stage=a.stage,worker_source_sha256=own_sha,recipe=RECIPE,short_proof=short_proof,
                fit_images=FIT_IMAGES,visits_per_full_epoch=BATCH*BATCHES_PER_EPOCH,
                actual_batch_size=BATCH,optimizer_updates=progress["updates"],consumed_visits=progress["rows"]*BATCH,
                rejected_augmentation_attempts=progress["rejected"],completed_epochs=progress["epochs"],
                model_forward_calls=progress["rows"],stock_author_source=True,package_versions=versions,
                author_commit=p335.AUTHOR_COMMIT,author_source_pins=pins,
                source_initializer_sha256=p336.R50_SHA,fit_archive_sha256=p335.FIT_SHA,fit_index_sha256=p335.FIT_INDEX_SHA,
                previous_gpu_short_result_sha256=P344_REPORT_SHA,previous_gpu_short_wait_sha256=P344_WAIT_SHA,
                head_parameter_max_abs_changes=movement,limits=limits,resource_end=resources,
                cuda_free_before_bytes=free,cuda_peak_allocated_bytes=allocated,cuda_peak_reserved_bytes=reserved,
                cuda_matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,
                complete_loss_keys=list(p344.LOSS_KEYS),visit_ledger=file_ref(visits),
                epoch_ledger=file_ref(epochs) if epochs is not None else None,checkpoint=checkpoint,
                FIT=True,DEV=False,TEST=False,AP=False,elapsed_seconds=perf_counter()-started)


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument("--stage",choices=("short","e24"),required=True)
    inputs=("p344-source","p344-result","p344-wait","p336-source","p335-source","prior-report",
            "source","fit","archive","r50","expected-python")
    for name in (*inputs,"output"):
        parser.add_argument("--"+name,required=True,type=Path)
    for name in ("short-report","short-terminal","short-stdout"):
        parser.add_argument("--"+name,type=Path)
    for name in ("short-report-sha256","short-terminal-sha256"):
        parser.add_argument("--"+name)
    a=parser.parse_args()
    require(all(path.is_absolute() and path.resolve(strict=True)==path
                for path in (getattr(a,name.replace("-","_")) for name in inputs))
            and a.output.is_absolute() and a.output.parent.resolve(strict=True)==a.output.parent
            and not a.output.exists() and not a.output.is_symlink()
            and not a.output.is_relative_to(a.fit) and not a.fit.is_relative_to(a.output),
            "Canonical pinned inputs and fresh separate output required")
    require(os.name=="nt" and sys.version_info[:2]==(3,10)
            and Path(sys.executable).resolve(strict=True)==a.expected_python
            and sys.flags.isolated==sys.flags.utf8_mode==1 and sys.dont_write_bytecode
            and os.environ.get("CUDA_VISIBLE_DEVICES")=="0","Exact isolated Windows CUDA worker required")
    if a.stage=="e24":
        require(all(getattr(a,k) is not None for k in
                    ("short_report","short_report_sha256","short_terminal","short_terminal_sha256","short_stdout")),
                "Full E24 requires successful original batch-four qualification")
    def offline(event,_):
        if event.startswith("socket.") and event!="socket.gethostname":
            raise PermissionError("Offline fixed FIT741 training")
    sys.addaudithook(offline)
    started=perf_counter()
    own_sha=sha256(Path(__file__).read_bytes()).hexdigest()
    a.output.mkdir()
    report=run(a,own_sha,started)
    raw=encode(report)
    with (a.output/"RESULT_SAFE.json").open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    print(encode(dict(status=STATUS,result_sha256=sha256(raw).hexdigest())).decode().strip(),flush=True)


if __name__=="__main__":
    main()
