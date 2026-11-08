"""Train the COCO-only RF-DETR HRSID FIT baseline to E24 with the native E100 schedule."""
import argparse
from collections import Counter
from hashlib import md5, sha256
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import re
import sys
from time import perf_counter
from types import ModuleType


PRIVATE = Path(__file__).resolve().parent.parent
E1 = PRIVATE / "HRSID_FIT_NATIVE_E1_WORKER_CANDIDATE_20261109_01/train_hrsid_native_e1.py"
E1_SHA = "23bf7b432763f86f57322b3c1e09cf7966fa57c80e0ce74e9185b8977a414362"
COCO = PRIVATE / "RFDETR_COCO_SEG_NANO_OFFICIAL_INTAKE_20261109_01/rf-detr-seg-nano.pt"
COCO_SHA = "a44613a4ecd6b5ba61a62002c600b0b6cb7a9da2936a45317ec4b62c635fb99b"
COCO_MD5 = "9995497791d0ff1664a1d9ddee9cfd20"
COCO_BYTES = 134545398
RECIPE = PRIVATE / "github_release_20261011/research_round_20260925_global/configs/RFDETR_SEG_NANO_BASELINE_576_E100_20261107.yaml"
RECIPE_SHA = "ba9508d9acab5ff68b10db8a1aff5d9001868d6dda00e39f299d6f5cc1b8011b"
CONFIG = PRIVATE / "RFDETR_REAL_SSDD_DATA_QUALIFICATION_DRAFT_20261107_01/entry.py"
CONFIG_SHA = "f0642e8f33e75a2440eac0968dd76f270d445498ba88e812543902e7d0ad8453"
RNG = PRIVATE / "RFDETR_NATIVE_E1_WORKER_CHECKPOINT_API_RECOVERY_20261107_02/e1_resume.py"
RNG_SHA = "014e087379ffe6e8d32329c337d6b625b98dfc2d235cb321becebfdf3832b013"
VISITS, FIT, ACCUM, STEPS = 2940, 2937, 4, 735
EPOCHS, WARMUP, TOTAL = 100, 2205, 73500
ENDPOINTS = (1, 8, 24)
WIDTH, SEED = 164, 5201
CAP = 6 * 1024**3
MARKER = "sar_seal_hrsid_coco_fit_native_endpoint_v1"


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def pinned_module(path, digest, name):
    source = path.read_bytes()
    need(sha256(source).hexdigest() == digest, f"Exact closed source required for {name}")
    module = ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


def event(name, **values):
    print(json.dumps(dict(event=name, **values), sort_keys=True, allow_nan=False), flush=True)


def offline(event, args):
    if event in ("socket.connect", "socket.getaddrinfo"):
        raise RuntimeError("The FIT-only baseline worker is offline")


def archive_due(epoch):
    return epoch in ENDPOINTS


def schedule_factor(step):
    if step < WARMUP:
        return step / WARMUP
    return .05 + .95 * .5 * (1 + math.cos(math.pi * (step - WARMUP) / (TOTAL - WARMUP)))


def optimizer_state(optimizer, scheduler, step):
    need(scheduler["last_epoch"] == step, "Native scheduler cursor differs from the actual AdamW step")
    base, next_lrs, groups = scheduler["base_lrs"], scheduler["_last_lr"], optimizer["param_groups"]
    need(len(base) == len(next_lrs) == len(groups) > 0 and all(
        math.isfinite(a) and a >= 0 and math.isclose(a * schedule_factor(step), b, rel_tol=1e-9, abs_tol=1e-12)
        and b == group["lr"] for a, b, group in zip(base, next_lrs, groups, strict=True)), "Native E100 scheduler LRs")
    states = list(optimizer["state"].values())
    need(states and all({"step", "exp_avg", "exp_avg_sq"} <= state.keys()
                        and 0 < float(state["step"]) == int(state["step"]) <= step for state in states),
         "Full AdamW moment slots and actual parameter update clocks")
    counts = Counter(int(state["step"]) for state in states)
    need(max(counts) == step, "AdamW moment state reached the actual endpoint")
    return dict(adam_parameter_clocks=dict(counts), native_base_lrs=base, native_next_lrs=next_lrs)


def verify_loop(checkpoint, epoch):
    need(checkpoint["pytorch-lightning_version"] == "2.6.6" and checkpoint["epoch"] == epoch - 1
         and checkpoint["global_step"] == epoch * STEPS, "Exact native endpoint archive clock")
    fit = checkpoint["loops"]["fit_loop"]
    batches = fit["epoch_loop.batch_progress"]
    updates = fit["epoch_loop.automatic_optimization.optim_progress"]["optimizer"]["step"]
    need(batches["is_last_batch"] is True, "Archive contains the full final FIT batch")
    for scope, multiple in (("current", 1), ("total", epoch)):
        ep = fit["epoch_progress"][scope]
        need(ep["ready"] == ep["started"] == ep["processed"] == epoch and ep["completed"] == epoch - 1,
             "Unexpected native epoch checkpoint cursor")
        need(all(batches[scope][key] == multiple * VISITS for key in ("ready", "started", "processed", "completed"))
             and all(updates[scope][key] == multiple * STEPS for key in ("ready", "completed")),
             "Native batch or step cursor differs at checkpoint boundary")


def file_identity(path):
    st = path.stat()
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns


def verified_coco():
    start = file_identity(COCO)
    need(COCO.resolve(strict=True) == COCO and start[2] == COCO_BYTES, "Exact original COCO SegNano file size and path")
    strong, upstream = sha256(), md5()
    with COCO.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            strong.update(block)
            upstream.update(block)
    need(strong.hexdigest() == COCO_SHA and upstream.hexdigest() == COCO_MD5
         and file_identity(COCO) == start, "Exact verified official Roboflow COCO SegNano bytes")
    return start


def initialize(base, output, module_type):
    probe = base.closed_probe()
    state_api = pinned_module(probe.STOCK / "state_preservation.py", probe.STATE_SHA, "_closed_coco_fit_digest")
    configs = pinned_module(CONFIG, CONFIG_SHA, "_closed_coco_fit_recipe_loader")
    base.pinned_bytes(RECIPE, RECIPE_SHA)
    stock_model, train = configs._load_configs(RECIPE, base.LAYOUT)
    train = train.model_copy(update={"output_dir": str(output), "compute_train_metrics": False})
    need(stock_model.num_classes == 1 and stock_model.num_queries == stock_model.num_select == 100
         and stock_model.group_detr == 13 and stock_model.resolution == 576 and stock_model.patch_size == 12
         and train.epochs == EPOCHS and train.batch_size == 1 and train.grad_accum_steps == ACCUM
         and train.num_workers == 0 and train.warmup_epochs == 3 and train.optimizer == "adamw"
         and train.use_ema and train.ema_decay == .993 and train.ema_tau == 100
         and train.checkpoint_interval == 1 and train.clip_max_norm == .1 and train.scale_jitter
         and train.augmentation_backend == "torchvision" and train.dataset_file == "roboflow"
         and Path(train.dataset_dir) == base.LAYOUT and not train.compute_train_metrics and not train.run_test,
         "Exact native E100 training and single-class 576 architecture recipe")
    file_token = verified_coco()
    load_config = stock_model.model_copy(update={"pretrain_weights": str(COCO)})
    original = module_type(load_config, train, load_encoder_weights=False)
    need(file_identity(COCO) == file_token and original.model.class_embed.weight.shape[0] == 2
         and original.model.num_queries == 100 and load_config.num_classes == 1,
         "Native official COCO import aligned the class head at the original query width")
    original_digest = state_api.tensor_dict_sha256(original.model.state_dict())
    probe.expand_groups(original.model, WIDTH)
    expanded_digest = state_api.tensor_dict_sha256(original.model.state_dict())
    model = stock_model.model_copy(update={"pretrain_weights": None, "num_queries": WIDTH})
    module = module_type(model, train, load_encoder_weights=False)
    module.model.load_state_dict(original.model.state_dict(), strict=True)
    need(state_api.tensor_dict_sha256(module.model.state_dict()) == expanded_digest
         and module.model.num_queries == module.model_config.num_queries == WIDTH
         and module.model.class_embed.weight.shape[0] == 2, "Strict native expanded width and model metadata")
    info = dict(coco_asset_sha256=COCO_SHA, coco_asset_bytes=COCO_BYTES, native_original_width=100,
                width100_loaded_sha256=original_digest, expanded_online_sha256=expanded_digest,
                query_width=WIDTH, query_groups=13, class_head_slots=2, initialized_from_SSDD=False)
    return model, train, module, state_api, info


def append_line(stream, value):
    row = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    stream.write(row)
    stream.flush()
    os.fsync(stream.fileno())
    return sha256(row).hexdigest()


def run(base, output, protocol, expected_initial):
    import numpy as np
    import torch
    from pytorch_lightning import Callback, seed_everything
    from pytorch_lightning.callbacks import ModelCheckpoint
    from rfdetr.training.callbacks.ema import RFDETREMACallback
    from rfdetr.training.module_data import GradAccumAlignedDataset, RFDETRDataModule
    from rfdetr.training.module_model import RFDETRModelModule
    from rfdetr.training.trainer import build_trainer

    need(torch.__version__ == "2.10.0+cu130" and version("rfdetr") == "1.12.0"
         and version("pytorch-lightning") == "2.6.6", "Exact previously qualified numeric runtime")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True
    cpu_only = output is None
    if cpu_only:
        need(not torch.cuda.is_available(), "Initialization check is CPU-only")
    else:
        need(torch.cuda.is_available() and torch.cuda.device_count() == 1
             and torch.cuda.is_bf16_supported(including_emulation=False), "One visible native BF16 GPU")
        free, total = torch.cuda.mem_get_info(0)
        need(free >= CAP, "GPU free memory floor")
        torch.cuda.set_per_process_memory_fraction(min(.35, CAP / total), 0)
    seed_everything(SEED, workers=True)
    model, train, module, state_api, init = initialize(base, output or base.LAYOUT, RFDETRModelModule)
    event("HRSID_COCO_NATIVE_INITIALIZATION", **init, training=False)
    if cpu_only:
        return
    need(init["expanded_online_sha256"] == expected_initial, "Exact independent CPU COCO width-164 initialization pin")
    ids = base.fit_ids()
    rng_api = pinned_module(RNG, RNG_SHA, "_closed_native_endpoint_rng")
    initial_classifier = module.model.class_embed.weight.detach().cpu().clone()
    started = perf_counter()
    output.mkdir()
    visits_path = output / "FIT_VISITS_PRIVATE.jsonl"
    audits_path = output / "EPOCH_AUDIT_SAFE.jsonl"

    class FitOnlyData(RFDETRDataModule):
        def setup(self, stage):
            need(stage == "fit" and self._dataset_val is None and self._dataset_test is None, "Only FIT setup")
            # Preserve the admitted native DataModule seam while excluding VAL dataset construction.
            self._dataset_val = object()
            try:
                super().setup(stage)
            finally:
                self._dataset_val = None
            need(len(self._dataset_train) == FIT, "Exact original native FIT count")

        def val_dataloader(self):
            raise RuntimeError("Non-FIT loaders are forbidden")

        test_dataloader = predict_dataloader = val_dataloader

    class Observe(Callback):
        def __init__(self, visit_stream, audit_stream):
            self.visit_stream, self.audit_stream = visit_stream, audit_stream
            self.updates = 0
            self.handle = None
            self.completed = 0
            self.armed = None
            self.saved = []

        def on_train_start(self, trainer, pl_module):
            need(trainer.current_epoch == trainer.global_step == 0 and not trainer.enable_validation
                 and trainer.max_epochs == EPOCHS and trainer.max_steps == -1
                 and trainer.accumulate_grad_batches == ACCUM and trainer.num_training_batches == VISITS
                 and trainer.estimated_stepping_batches == TOTAL and trainer.gradient_clip_val == .1,
                 "Fresh native E100 schedule and aligned FIT loader clock")
            loader = trainer.train_dataloader
            need(type(loader.dataset) is GradAccumAlignedDataset and len(loader) == VISITS
                 and loader.batch_size == 1 and loader.num_workers == 0, "Exact native aligned training loader")
            native_ids = data._dataset_train.ids
            repeats = [native_ids[i] for i in loader.dataset._pad_indices]
            need(len(repeats) == VISITS - FIT and len(native_ids) == FIT and set(native_ids) == ids,
                 "Exact native aligned duplicate access population")
            self.population = Counter(native_ids) + Counter(repeats)
            self.optim, = trainer.optimizers
            cfg, = trainer.lr_scheduler_configs
            self.scheduler = cfg.scheduler
            need(type(self.optim) is torch.optim.AdamW and len(self.optim.state) == 0
                 and type(self.scheduler) is torch.optim.lr_scheduler.LambdaLR and self.scheduler.last_epoch == 0
                 and cfg.interval == "step" and cfg.frequency == 1 and all(
                     math.isclose(fn(step), schedule_factor(step), rel_tol=1e-9, abs_tol=1e-12)
                     for fn in self.scheduler.lr_lambdas for step in (0, 1, STEPS, WARMUP, 24 * STEPS, TOTAL)),
                 "Fresh native AdamW and exact 73500-step LR schedule")
            base.ema_clock(ema.state_dict(), 0)
            self.handle = self.optim.register_step_post_hook(self.adam_returned)
            event("HRSID_COCO_NATIVE_E24_STARTED", visits_per_epoch=VISITS, updates_per_epoch=STEPS,
                  planned_epochs=EPOCHS, stop_after_epoch=ENDPOINTS[-1])

        def on_train_epoch_start(self, trainer, pl_module):
            need(trainer.current_epoch == self.completed and trainer.global_step == self.updates == self.completed * STEPS
                 and self.armed is None and not trainer.should_stop, "Native epoch entered out of order")
            if archive_due(self.completed):
                need((output / f"checkpoint_{self.completed - 1}.ckpt").is_file(), "Prior expected endpoint archive missing")
            self.visits = []
            self.losses = []
            self.counts = Counter()
            self.gradients = []
            self.epoch_start = perf_counter()

        def adam_returned(self, optimizer, args, kwargs):
            need(optimizer is self.optim, "Unexpected optimizer post-hook source")
            self.updates += 1

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            image, targets = batch
            need(trainer.current_epoch == self.completed and batch_idx == len(self.visits) < VISITS
                 and image.tensors.shape == (1, 3, 576, 576) and image.tensors.dtype == torch.float32
                 and image.tensors.device.type == "cuda" and len(targets) == 1, "Exact actual native FIT batch")
            target, = targets
            labels, masks, boxes = target["labels"], target["masks"], target["boxes"]
            count = labels.numel()
            need(count <= WIDTH and labels.shape == (count,) and labels.dtype == torch.int64
                 and masks.dtype == torch.bool and masks.shape == (count, 576, 576)
                 and boxes.shape == (count, 4) and boxes.dtype == torch.float32, "Native target shapes and query width")
            half_wh = boxes[:, 2:] / 2
            low, high = boxes[:, :2] - half_wh, boxes[:, :2] + half_wh
            need(bool((labels == 0).all() & masks.flatten(1).any(1).all() & torch.isfinite(boxes).all()
                      & (half_wh > 0).all() & (low >= -1e-6).all() & (high <= 1 + 1e-6).all()),
                 "Retained target masks are nonempty and boxes are finite, normalized and positive")
            iid = int(target["image_id"])
            need(iid in ids, "Only the admitted FIT image population")
            self.visits.append((iid, count))
            self.counts[count] += 1

        def on_before_backward(self, trainer, pl_module, loss):
            need(bool(torch.isfinite(loss)), "Finite native loss before backward")

        def on_before_optimizer_step(self, trainer, pl_module, optimizer):
            need(optimizer is self.optim and trainer.global_step == self.updates == self.scheduler.last_epoch,
                 "Native optimizer, scheduler and Lightning clocks must agree before actual update")
            grad = pl_module.model.class_embed.weight.grad
            need(grad is not None and grad.dtype == torch.float32, "Native classifier gradient witness")
            norm = float(torch.linalg.vector_norm(grad.detach()))
            need(math.isfinite(norm), "Finite classifier gradient before update")
            self.gradients.append(norm)

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            need(trainer.global_step == self.updates == self.completed * STEPS + (batch_idx + 1) // ACCUM,
                 "Exact native microbatch and AdamW post-step clocks")
            loss = float(outputs["loss"].detach())
            need(math.isfinite(loss), "Finite Lightning accumulation-scaled loss")
            self.losses.append(loss)
            if (batch_idx + 1) % ACCUM == 0 and self.updates % 100 == 0:
                event("HRSID_COCO_NATIVE_PROGRESS", epoch=self.completed + 1, visits_in_epoch=batch_idx + 1,
                      optimizer_updates=self.updates, peak_allocated_bytes=torch.cuda.max_memory_allocated())

        def on_train_epoch_end(self, trainer, pl_module):
            epoch = self.completed + 1
            step = epoch * STEPS
            need(trainer.current_epoch == self.completed and not trainer.should_stop
                 and len(self.visits) == len(self.losses) == VISITS and len(self.gradients) == STEPS
                 and trainer.global_step == self.updates == step and max(self.gradients) > 0
                 and Counter(iid for iid, _ in self.visits) == self.population, "Only complete full native FIT epochs")
            base.ema_clock(ema.state_dict(), step)
            lr = optimizer_state(self.optim.state_dict(), self.scheduler.state_dict(), step)
            ledger_sha = append_line(self.visit_stream, dict(epoch=epoch, visits=self.visits))
            record = dict(epoch=epoch, visits=VISITS, unique_fit_images=FIT, duplicate_fit_visits=VISITS - FIT,
                          zero_target_visits=self.counts[0], maximum_augmented_targets=max(self.counts),
                          total_augmented_targets=sum(n * count for n, count in self.counts.items()),
                          optimizer_updates=step, ema_updates=step, visit_row_sha256=ledger_sha,
                          lightning_accumulation_scaled_loss_mean=math.fsum(self.losses) / VISITS,
                          lightning_accumulation_scaled_loss_min=min(self.losses),
                          lightning_accumulation_scaled_loss_max=max(self.losses),
                          classifier_gradient_norm_max=max(self.gradients), epoch_seconds=perf_counter() - self.epoch_start,
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(), **lr)
            append_line(self.audit_stream, record)
            self.completed = epoch
            self.armed = epoch if archive_due(epoch) else None
            checkpoint_callback.save_top_k = -1 if self.armed else 0
            if epoch == ENDPOINTS[-1]:
                trainer.should_stop = True
            event("HRSID_COCO_NATIVE_EPOCH_OBSERVED", **record)

        def on_save_checkpoint(self, trainer, pl_module, checkpoint):
            epoch = self.armed
            need(epoch is not None and archive_due(epoch) and MARKER not in checkpoint,
                 "Only prospectively selected full epoch checkpoints")
            verify_loop(checkpoint, epoch)
            checkpoint[MARKER] = dict(protocol=protocol, epoch=epoch, actual_updates=epoch * STEPS,
                                      native_index_sha256=base.INDEX_SHA, coco_sha256=COCO_SHA,
                                      original_recipe_sha256=RECIPE_SHA, initial_expanded_sha256=expected_initial,
                                      FIT_only=True, exact_restart_trajectory_verified=False,
                                      rng_state=rng_api.capture_rng_state(torch, np))
            self.saved.append(epoch)
            self.armed = None

        def on_validation_start(self, trainer, pl_module):
            raise RuntimeError("Non-FIT evaluation is forbidden")

        on_test_start = on_predict_start = on_validation_start

    with visits_path.open("xb") as visits_stream, audits_path.open("xb") as audit_stream:
        observe = Observe(visits_stream, audit_stream)
        data = FitOnlyData(model, train)
        trainer = build_trainer(train, model, accelerator="gpu", devices=1, extra_callbacks=[observe],
                                limit_val_batches=0, num_sanity_val_steps=0, min_epochs=0, min_steps=0,
                                logger=False, enable_model_summary=False, enable_progress_bar=False)
        trainer.callbacks[:] = [cb for cb in trainer.callbacks if type(cb).__name__ not in {
            "BestModelCallback", "COCOEvalCallback", "_ForceLastEpochValidationCallback", "GPUMemoryTQDMProgressBar"}]
        ema, = (cb for cb in trainer.callbacks if type(cb) is RFDETREMACallback)
        checkpoint_callback, = (cb for cb in trainer.checkpoint_callbacks if type(cb) is ModelCheckpoint)
        need(trainer.precision == "bf16-mixed" and trainer.scaler is None and not trainer.enable_validation
             and checkpoint_callback.filename == "checkpoint_{epoch}" and checkpoint_callback.monitor is None
             and checkpoint_callback.save_weights_only is False and checkpoint_callback.save_on_exception is False,
             "Native BF16, native EMA and full checkpoint callbacks")
        checkpoint_callback.save_top_k = 0
        checkpoint_callback.save_last = False
        torch.cuda.reset_peak_memory_stats()
        seed_everything(SEED, workers=True)
        try:
            trainer.fit(module, datamodule=data, ckpt_path=None)
        finally:
            if observe.handle is not None:
                observe.handle.remove()
    epoch = ENDPOINTS[-1]
    archives = [output / f"checkpoint_{n - 1}.ckpt" for n in ENDPOINTS]
    need(trainer.current_epoch == observe.completed == epoch and trainer.global_step == observe.updates == epoch * STEPS
         and trainer.should_stop and tuple(observe.saved) == ENDPOINTS and observe.armed is None
         and data._dataset_val is data._dataset_test is None
         and set(output.iterdir()) == {visits_path, audits_path, *archives}, "Complete uninterrupted FIT E24 and only planned files")
    torch.cuda.synchronize()
    refs = []
    for n, path in zip(ENDPOINTS, archives, strict=True):
        start = file_identity(path)
        need(0 < start[2] <= 1024**3, "Bounded native checkpoint size")
        with path.open("rb") as stream:
            checkpoint = torch.load(stream, map_location="cpu", weights_only=True)
        verify_loop(checkpoint, n)
        marker = checkpoint[MARKER]
        need(marker["protocol"] == protocol and marker["epoch"] == n and marker["actual_updates"] == n * STEPS
             and marker["initial_expanded_sha256"] == expected_initial and marker["exact_restart_trajectory_verified"] is False
             and checkpoint["model_config"]["num_queries"] == WIDTH and checkpoint["model_config"]["num_select"] == 100,
             "Exact archive protocol, native width and saved RNG qualification boundary")
        rng_api.rng_envelope(marker["rng_state"])
        saved_ema = checkpoint["callbacks"][ema.state_key]
        base.ema_clock(saved_ema, n * STEPS)
        optim, = checkpoint["optimizer_states"]
        schedule, = checkpoint["lr_schedulers"]
        lrs = optimizer_state(optim, schedule, n * STEPS)
        raw = {key.removeprefix("model."): value for key, value in checkpoint["state_dict"].items()
               if key.startswith("model.")}
        averaged = {key.removeprefix("module.model."): value for key, value in
                    saved_ema["average_model_state_dict"].items() if key.startswith("module.model.")}
        need(raw.keys() == averaged.keys() and raw["refpoint_embed.weight"].shape[0] == WIDTH * 13
             and not torch.equal(initial_classifier, raw["class_embed.weight"])
             and all(not x.is_floating_point() or bool(torch.isfinite(x).all())
                     for state in (raw, averaged) for x in state.values()), "Finite updated online and EMA model states")
        raw_sha = state_api.tensor_dict_sha256(raw)
        ema_sha = state_api.tensor_dict_sha256(averaged)
        need(raw_sha != ema_sha, "Native online and EMA endpoint states are distinct")
        ref = state_api.file_ref(path, 1024**3)
        need(ref["bytes"] == start[2] and file_identity(path) == start, "Native archive stable during readback")
        refs.append(dict(epoch=n, archive_name=path.name, archive_sha256=ref["sha256"],
                         archive_bytes=ref["bytes"], ending_online_sha256=raw_sha, ending_ema_sha256=ema_sha,
                         rng_saved=True, **lrs))
        del checkpoint, marker, saved_ema, optim, schedule, raw, averaged
    result = dict(status="NATIVE_HRSID_COCO_FIT_E24_COMPLETE_PENDING_INDEPENDENT_ACCEPTANCE",
                  protocol=protocol, native_index_sha256=base.INDEX_SHA, original_recipe_sha256=RECIPE_SHA,
                  visited_fit_images=FIT, total_visits=VISITS * epoch, total_updates=STEPS * epoch,
                  starting_optimizer_updates=0, starting_ema_updates=0, ema_updates=STEPS * epoch,
                  completed_epochs=epoch, scheduled_epochs=EPOCHS, scheduled_updates=TOTAL, warmup_updates=WARMUP,
                  precision="bf16-mixed", batch_size=1, grad_accumulation=ACCUM,
                  gpu_allocator_limit_bytes=CAP, gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(), elapsed_seconds=perf_counter() - started,
                  epoch_audit_sha256=sha256(audits_path.read_bytes()).hexdigest(),
                  fit_visit_ledger_sha256=sha256(visits_path.read_bytes()).hexdigest(),
                  archive_refs=refs, FIT_only=True, DEV=False, TEST=False, AP=False,
                  exact_restart_trajectory_verified=False, **init)
    with (output / "NATIVE_HRSID_COCO_FIT_E24_SAFE.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initialization-check", action="store_true")
    parser.add_argument("--protocol")
    parser.add_argument("--expected-initial-sha256")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    base = pinned_module(E1, E1_SHA, "_closed_hrsid_e1_environment")
    visibility = "-1" if args.initialization_check else "0"
    need(tuple(sys.version_info[:3]) == (3, 10, 13) and os.environ.get("CUDA_VISIBLE_DEVICES") == visibility
         and os.environ.get("HF_HUB_OFFLINE") == os.environ.get("TRANSFORMERS_OFFLINE") == "1",
         "Pinned isolated offline runtime and GPU visibility")
    if args.initialization_check:
        need(args.output is args.protocol is args.expected_initial_sha256 is None, "CPU initialization-only route")
    else:
        need(bool(re.fullmatch(r"[0-9a-f]{40}", args.protocol or ""))
             and bool(re.fullmatch(r"[0-9a-f]{64}", args.expected_initial_sha256 or ""))
             and args.output is not None and args.output.is_absolute() and not args.output.exists()
             and not args.output.is_symlink() and args.output.parent.resolve(strict=True) == args.output.parent
             and args.output not in base.LAYOUT.parents and base.LAYOUT not in args.output.parents,
             "Committed protocol, independent expanded initializer digest and fresh absolute private output")
    sys.addaudithook(offline)
    base.admit_dependency_overlay()
    run(base, args.output, args.protocol, args.expected_initial_sha256)


if __name__ == "__main__":
    main()
