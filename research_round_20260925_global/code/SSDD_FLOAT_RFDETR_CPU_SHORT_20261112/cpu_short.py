"""Exercise the exact float FIT model input through native CPU RF-DETR training."""
import argparse
from contextlib import contextmanager
from hashlib import sha256
from io import StringIO
import json
import math
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType
from unittest.mock import patch


ENTRY = ("SSDD_MASK_PHASE_TRANSFER_20261110/ssdd_windows_phase_entry.py",
         "e29edeb0a3aff7c871123deee7c0c198ef5913fd33e7c016434a671d186796d3")
P330_SOURCE = ("SSDD_DEV187_ALIAS_AND_COMMON_FIT_20261111/p330_common_preflight.py",
               "5c1aa7441375692c0ca724fdab7b54e74659af481e42d82b268deb1dd50ceb51")
MODES = ("native", "center")
SCOPE = dict(FIT=True, DEV=False, TEST=False, AP=False, cuda_initialized=False)
VISITS, UPDATES = 32, 8


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return sha256(path.read_bytes()).hexdigest()


def event(status, **values):
    print(json.dumps(dict(status=status, **values), sort_keys=True, allow_nan=False), flush=True)


def save(path, raw):
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def encoded(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode("ascii")


def verify_fit(entry, layout, receipt):
    index = layout / "train/_annotations.coco.json"
    require(sha(receipt) == "96f80f9d3a0662f93189665d2161ada7078cd33353d6e768ef0d34925f76af15",
            "Exact closed float layout receipt required")
    checked = json.loads(receipt.read_bytes())
    require(checked["images"] == 741 and checked["annotations"] == 1661 and checked["fit"] is True
            and checked["dev"] is checked["test"] is checked["model_executed"] is False
            and sha(index) == checked["float_index_sha256"]
            == "97162d23a33b9387fe0b47a117f7e5b1f9dde7471033f244ab336cee1ca9039b",
            "Exact float FIT741 annotation digest and scope required")
    data = entry.document(index.read_bytes())
    ids = sorted(row["id"] for row in data["images"])
    require(len(ids) == len(set(ids)) == 741 and len(data["annotations"]) == 1661
            and sha256(entry.compact(ids)).hexdigest() == entry.FIT_ORDER_SHA,
            "Exact source FIT image identity and row count required")
    return ids


# Dataset and ledger functions extracted unchanged from the verified P333 worker.
def need(ok, reason):
    if not ok:
        raise ValueError(reason)

def digest(raw):
    return sha256(raw).hexdigest()

def load_pinned_module(path, expected, label):
    need(path.is_absolute() and path.resolve(strict=True) == path and 0 < path.stat().st_size <= 64*1024,
         "Exact canonical original ordinary source required")
    raw = path.read_bytes()
    need(digest(raw) == expected, "Exact original source digest differs: "+label)
    mod = ModuleType("_p333_"+label)
    mod.__file__ = str(path)
    exec(compile(raw,str(path),"exec"),mod.__dict__)
    return mod

def verify_consumption(common,stock,mode,expected_visits=32):
    need(mode in MODES and len(common) == len(stock) == expected_visits,
         "Exact original stock and common visit counts required")
    selected = sha256()
    for ordinal,(a,b) in enumerate(zip(common,stock,strict=True),1):
        need(a["image_id"] == b["image_id"] and a["ordinal"] == a["batch"] == b["batch"] == ordinal
             and a["epoch"] == 1 and a["common_nonempty"] == b["labels"] == b["postfilter_labels"],
             "Actual common batch must match original stock visit and remain nonempty per row")
        expected = a[mode+"_common_masks_sha256"]
        need(expected == b["masks_sha256"] == b["postfilter_masks_sha256"],
             "Actual selected mask digest before and after native guard must match paired audit")
        selected.update(bytes.fromhex(expected))
    return selected.hexdigest()

def bind_dependencies(deps,torch,p330,paired,binding,layout,mode,ledger):
    """Keep native optimizer/model callbacks and replace only the data patch and evidence hook."""
    from rfdetr.datasets import coco
    from rfdetr.training import module_data

    pending = None
    selected = sha256()

    @contextmanager
    def dataset_context(native_transforms,chosen):
        need(chosen == mode and native_transforms is deps.native_transforms, "Exact one-arm native data wrapper")
        with binding.common_fit_dataset(module_data,coco.CocoDetection,layout/"train",741,native_transforms,mode,ledger):
            yield

    def observed_identity(image,target):
        nonlocal pending
        need(pending is None, "The preceding common batch must have completed its original guard")
        count, iid = int(target["labels"].numel()), int(target["image_id"].item())
        ledger.require_next(iid,count)
        audit = ledger.pending[0]
        cpu_image = image.tensors[0].detach().cpu()
        cpu_target = {name:tensor.detach().cpu() for name,tensor in target.items()}
        expected = p330.validate_selected(torch,paired,cpu_image,cpu_target,audit,mode)
        record = deps.phase.batch_identity(image,target)
        need(record["image_id"] == iid and record["labels"] == count and record["masks_sha256"] == expected,
             "Stock original selected mask digest must equal actual common mask")
        pending = (target,iid,count,expected)
        return record

    def guarded_empty(target):
        nonlocal pending
        need(pending is not None and pending[0] is target, "Exact selected batch must reach original stock guard")
        _,iid,count,expected = pending
        removed = deps.empty_guard(target)
        need(removed == 0 and int(target["labels"].numel()) == count
             and deps.phase.tensor_digest(target["masks"]) == expected,
             "Original guard must leave actual selected common masks and counts unchanged")
        ledger.consume(image_id=iid,common_count=count,epoch=1,batch=ledger.consumed_visits+1)
        selected.update(bytes.fromhex(expected))
        pending = None
        return removed

    phase = ModuleType("_p333_original_phase_with_common_binding")
    phase.__dict__.update(vars(deps.phase))
    phase.mask_resize_phase = dataset_context
    phase.batch_identity = observed_identity
    return deps._replace(phase=phase,empty_guard=guarded_empty),selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--mode", choices=("native", "center"))
    for name in ("code-root", "fit-layout", "layout-receipt", "dependency-overlay"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--coco", type=Path)
    parser.add_argument("--arm-dir", type=Path)
    args = parser.parse_args()
    for name in ("code_root", "fit_layout", "layout_receipt", "dependency_overlay"):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    if not args.preflight:
        if args.coco is None or args.arm_dir is None:
            parser.error("--mode also requires --coco and --arm-dir")
        args.coco = args.coco.resolve(strict=True)
        args.arm_dir = args.arm_dir.resolve(strict=True)
    entry = load_pinned_module(args.code_root / ENTRY[0], ENTRY[1], "windows_entry")
    p330 = load_pinned_module(args.code_root / P330_SOURCE[0], P330_SOURCE[1], "p330")
    ids = verify_fit(entry, args.fit_layout, args.layout_receipt)
    torch, _, _, deps = entry.admit_windows_runtime(dict(
        operation="replay", code=str(args.code_root), overlay=str(args.dependency_overlay),
        source_manifest_sha256=entry.WORKER_MANIFEST_SHA))
    require(not torch.cuda.is_available() and not torch.cuda.is_initialized()
            and torch.get_num_threads() == torch.get_num_interop_threads() == 1,
            "A genuine one-thread CPU process with CUDA hidden is required")
    torch.set_float32_matmul_precision("highest")
    event("FIT_FLOAT_CPU_PREFLIGHT", fit_images=len(ids), fit_annotations=1661, **SCOPE)
    if args.preflight:
        return
    require(args.arm_dir.is_dir() and args.arm_dir.name == args.mode
            and args.fit_layout not in args.arm_dir.parents and args.arm_dir not in args.fit_layout.parents,
            "A separate per-arm output directory is required")
    from pytorch_lightning import Callback, seed_everything
    from pytorch_lightning.callbacks import ModelCheckpoint
    from rfdetr.training.callbacks.ema import RFDETREMACallback
    from rfdetr.training.trainer import build_trainer

    output = args.arm_dir / "MODEL_OUTPUT_UNUSED"
    model, config, module, init = deps.initialize_q100(args.fit_layout, output, args.coco)
    require(init["online_tensor_sha256"] == entry.Q100_SHA and init["num_queries"] == 100,
            "Exact fresh COCO Q100 initialization required")
    event("FIT_FLOAT_CPU_Q100_READY", mode=args.mode, init_sha256=entry.Q100_SHA)
    data = deps.fit_only_data(model, config, args.fit_layout)
    stream = StringIO()
    helpers = {name: p330.pinned(args.code_root, name) for name in ("trace_resize", "common_mask_supervision")}
    with patch.dict(sys.modules, helpers):
        paired = p330.pinned(args.code_root, "paired_fit_transform")
        binding = p330.pinned(args.code_root, "common_fit_binding")
        with patch.dict(sys.modules, {"paired_fit_transform": paired}):
            ledger = binding.ConsumedFitLedger(stream)
            wrapped, chosen = bind_dependencies(deps, torch, p330, paired, binding,
                                                args.fit_layout, args.mode, ledger)
            train(args, torch, Callback, seed_everything, ModelCheckpoint, RFDETREMACallback, build_trainer,
                  model, config, module, init, data, wrapped, stream, ledger, chosen, ids)


def train(args, torch, Callback, seed_everything, ModelCheckpoint, EMACallback, build_trainer,
          model, config, module, init, data, deps, stream, ledger, chosen, ids):
    class Observe(Callback):
        def __init__(self):
            self.records, self.losses, self.gradients = [], [], []
            self.updates = 0
            self.step_hook = None

        def on_train_start(self, trainer, model_module):
            require(trainer.precision == "32-true" and model_module.device.type == "cpu"
                    and trainer.accumulate_grad_batches == 4 and trainer.world_size == 1
                    and trainer.max_epochs == 100 and trainer.max_steps == -1
                    and trainer.num_training_batches == 744 and trainer.estimated_stepping_batches == 18600
                    and trainer.global_step == 0 and not trainer.enable_validation
                    and list(data._dataset_train.ids) == ids and torch.initial_seed() == 5201,
                    "Exact CPU FP32, seed and stock E100 loader/clock required")
            self.optimizer, = trainer.optimizers
            lr_config, = trainer.lr_scheduler_configs
            self.scheduler = lr_config.scheduler
            require(type(self.optimizer) is torch.optim.AdamW and not self.optimizer.state
                    and type(self.scheduler) is torch.optim.lr_scheduler.LambdaLR
                    and lr_config.interval == "step" and lr_config.frequency == 1
                    and all(math.isclose(fn(step), deps.e100.schedule_factor(step), rel_tol=1e-9, abs_tol=1e-12)
                            for fn in self.scheduler.lr_lambdas for step in (0, 1, 186, 558, 4464, 18600)),
                    "Stock AdamW and E100 schedule anchors required")
            require(all(model_module.criterion.weight_dict.get(k) == v for k, v in deps.e1.TERMS.items())
                    and trainer.gradient_clip_val == config.clip_max_norm,
                    "Stock losses and gradient clipping required")
            self.classifier = model_module.model.class_embed.weight
            self.initial = deps.phase.tensor_digest(self.classifier)
            self.step_hook = self.optimizer.register_step_post_hook(self.after_adam)
            event("FIT_FLOAT_CPU_TRAIN_STARTED", mode=args.mode, planned_updates=18600, warmup_updates=558)

        def after_adam(self, optimizer, _args, _kwargs):
            require(optimizer is self.optimizer, "Unexpected optimizer step source")
            self.updates += 1

        def on_train_batch_start(self, trainer, _module, batch, batch_idx):
            image, targets = batch
            target, = targets
            box = target["boxes"]
            half = box[:, 2:] / 2
            require(trainer.current_epoch == 0 and batch_idx == len(self.records)
                    and int(target["image_id"]) in ids and len(box) <= 100
                    and bool((target["labels"] == 0).all()) and bool(torch.isfinite(box).all())
                    and bool((half > 0).all()) and bool((box[:, :2] - half >= -1e-6).all())
                    and bool((box[:, :2] + half <= 1 + 1e-6).all()),
                    "Stock SSDD target identity, category and normalized box required")
            identity = deps.phase.batch_identity(image, target)
            require(deps.empty_guard(target) == 0, "Common row must survive unchanged stock empty-mask guard")
            self.records.append(dict(batch=batch_idx + 1, **identity, postfilter_labels=len(target["labels"]),
                                     postfilter_masks_sha256=deps.phase.tensor_digest(target["masks"]),
                                     postfilter_boxes_sha256=deps.phase.tensor_digest(target["boxes"])))

        def on_before_backward(self, _trainer, _module, loss):
            require(bool(torch.isfinite(loss)), "Actual model loss must be finite before backward")
            if len(self.records) == 1:
                event("FIT_FLOAT_CPU_FIRST_FORWARD_COMPLETE", mode=args.mode, loss=float(loss.detach()))

        def on_before_optimizer_step(self, trainer, _module, optimizer):
            gradient = self.classifier.grad
            require(optimizer is self.optimizer and self.updates == trainer.global_step == self.scheduler.last_epoch
                    and gradient is not None and gradient.dtype == torch.float32,
                    "Expected actual finite CPU classifier gradients and optimizer clock")
            norm = float(torch.linalg.vector_norm(gradient.detach()))
            require(math.isfinite(norm), "Non-finite classifier gradient")
            self.gradients.append(norm)

        def on_train_batch_end(self, trainer, _module, outputs, _batch, batch_idx):
            visits = batch_idx + 1
            loss = float(outputs["loss"].detach())
            require(math.isfinite(loss) and self.updates == trainer.global_step == visits // 4,
                    "Exact native CPU loss and AdamW return clock required")
            self.losses.append(loss)
            if visits in (1, 4, 8, 16, 24, VISITS):
                event("FIT_FLOAT_CPU_PROGRESS", mode=args.mode, visits=visits, updates=self.updates)
            if visits == VISITS:
                self.ending = deps.phase.tensor_digest(self.classifier)
                require(self.initial != self.ending, "Online classifier did not change before the EMA swap")
                trainer.should_stop = True

    observe = Observe()
    trainer = build_trainer(config, model, accelerator="cpu", devices=1, precision="32-true",
                            extra_callbacks=[observe], limit_val_batches=0, num_sanity_val_steps=0,
                            min_epochs=0, min_steps=0, logger=False, enable_model_summary=False,
                            enable_progress_bar=False)
    trainer.callbacks[:] = [c for c in trainer.callbacks if not isinstance(c, ModelCheckpoint)
                            and type(c).__name__ not in {"BestModelCallback", "COCOEvalCallback",
                                                       "_ForceLastEpochValidationCallback", "GPUMemoryTQDMProgressBar"}]
    ema, = (c for c in trainer.callbacks if type(c) is EMACallback)
    seed_everything(5201, workers=True, verbose=False)
    began = perf_counter()
    try:
        with deps.phase.mask_resize_phase(deps.native_transforms, args.mode):
            trainer.fit(module, datamodule=data, ckpt_path=None)
    finally:
        if observe.step_hook is not None:
            observe.step_hook.remove()
    stats = ledger.finish(VISITS)
    common = [json.loads(line) for line in stream.getvalue().splitlines()]
    require(verify_consumption(common, observe.records, args.mode) == chosen.hexdigest()
            and len(observe.losses) == VISITS and len(observe.gradients) == UPDATES and max(observe.gradients) > 0
            and trainer.global_step == observe.updates == UPDATES and trainer.current_epoch == 1
            and trainer.should_stop and not torch.cuda.is_initialized()
            and data._dataset_val is None and data._dataset_test is None
            and not list(args.arm_dir.rglob("*.ckpt")),
            "Exact post-guard lineage, no archive and 32/8 short boundary required")
    state = observe.optimizer.state_dict()
    deps.e100.validate_schedule(state, observe.scheduler.state_dict(), UPDATES)
    deps.boundary.adam_progress(state, UPDATES)
    deps.boundary.ema_progress({deps.boundary.EMA_KEY: ema.state_dict()}, UPDATES)
    common_path = args.arm_dir / "COMMON_VISITS_PRIVATE.jsonl"
    stock_path = args.arm_dir / "STOCK_VISITS_PRIVATE.json"
    save(common_path, stream.getvalue().encode("utf-8"))
    save(stock_path, encoded(observe.records))
    report = dict(status="SSDD_FLOAT_RFDETR_CPU_SHORT_ARM_COMPLETE", mode=args.mode, precision="32-true",
                  visits=VISITS, optimizer_updates=UPDATES, ema_updates=UPDATES, warmup_updates=558,
                  planned_updates=18600, checkpoint_saved=False, loss_min=min(observe.losses),
                  loss_max=max(observe.losses), gradient_max=max(observe.gradients),
                  classifier_initial_sha256=observe.initial, classifier_ending_sha256=observe.ending,
                  q100_initial_sha256=init["online_tensor_sha256"], common_ledger_sha256=sha(common_path),
                  stock_ledger_sha256=sha(stock_path), selected_mask_stream_sha256=chosen.hexdigest(),
                  seconds=perf_counter()-began, **stats, **SCOPE)
    result = args.arm_dir / "RESULT_SAFE.json"
    save(result, encoded(report))
    event(report["status"], mode=args.mode, report_sha256=sha(result), visits=VISITS, updates=UPDATES)


if __name__ == "__main__":
    main()
