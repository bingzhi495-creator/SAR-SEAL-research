"""Run an owner-admitted SSDD native or center-mask phase arm through stock RF-DETR."""

from collections import Counter
from contextlib import ExitStack
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
from time import perf_counter
from typing import NamedTuple


FIT_IMAGES = 741
VISITS_PER_EPOCH = 744
ACCUMULATION = 4
UPDATES_PER_EPOCH = 186
PLANNED_EPOCHS = 100
PLANNED_UPDATES = 18600
WARMUP_UPDATES = 558
STOP_EPOCH = 24
SHORT_VISITS = 32
SHORT_UPDATES = 8
SEED = 5201
QUERY_WIDTH = 100
Q100_SHA = "40524fc63797d2fcab7b50b6ee1f041f9429df73e505cb9e647a5a8cdf6f2e2f"
COCO_SHA = "a44613a4ecd6b5ba61a62002c600b0b6cb7a9da2936a45317ec4b62c635fb99b"
PHASE_MARKER = "sar_seal_ssdd_mask_resize_phase_v1"
VISITS_NAME = "FIT_VISITS_PRIVATE.jsonl"
AUDIT_NAME = "EPOCH_AUDIT_SAFE.jsonl"
SHORT_NAME = "QUALIFICATION_VISITS_PRIVATE.json"
RESULT_NAME = "SSDD_PHASE_TRAIN_SAFE.json"


class TrainingDependencies(NamedTuple):
    initialize_q100: object
    fit_only_data: object
    e100: object
    e1: object
    boundary: object
    phase: object
    empty_guard: object
    native_transforms: object


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def exact_hash(value, length):
    return type(value) is str and re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is not None


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest_file(path):
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def append_jsonl(stream, value):
    raw = encode(value) + b"\n"
    require(stream.write(raw) == len(raw), "Incomplete SSDD FIT audit write")
    stream.flush()
    os.fsync(stream.fileno())
    return sha256(raw).hexdigest()


def fit_id_set(expected_fit_ids):
    values = tuple(expected_fit_ids)
    require(len(values) == FIT_IMAGES and all(type(value) is int and value >= 0 for value in values)
            and len(set(values)) == FIT_IMAGES, "Exact independently admitted SSDD FIT741 IDs required")
    return set(values)


class FitProgress:
    """Check actual per-batch order, padded population, and stage boundaries."""

    def __init__(self, stage, expected_fit_ids, padded_ids):
        require(stage in ("qualification", "e24"), "Expected a fixed SSDD phase stage")
        ids = fit_id_set(expected_fit_ids)
        require(len(padded_ids) == VISITS_PER_EPOCH - FIT_IMAGES
                and all(type(value) is int and value in ids for value in padded_ids),
                "Exact native three-entry SSDD alignment padding required")
        self.short = stage == "qualification"
        self.population = Counter(ids) + Counter(padded_ids)
        self.completed = 0
        self.visits = []
        self.total_visits = 0
        self.removed_empty_rows = 0

    def consume(self, *, epoch, batch, image_id, retained, removed):
        require(epoch == self.completed + 1 and batch == len(self.visits)
                and batch < (SHORT_VISITS if self.short else VISITS_PER_EPOCH)
                and image_id in self.population, "Unexpected SSDD FIT epoch, batch cursor or image")
        require(type(retained) is type(removed) is int and min(retained, removed) >= 0
                and retained + removed <= QUERY_WIDTH, "Invalid SSDD augmented target counts")
        self.visits.append((image_id, retained))
        self.total_visits += 1
        self.removed_empty_rows += removed

    def after_batch(self, batch, updates):
        require(batch + 1 == len(self.visits) and updates == self.completed * UPDATES_PER_EPOCH
                + (batch + 1) // ACCUMULATION, "Exact native SSDD AdamW and microbatch cursor required")
        return self.short and batch + 1 == SHORT_VISITS

    def finish_epoch(self, epoch, updates, *, should_stop):
        require(epoch == self.completed + 1, "Native SSDD epoch cursor differs")
        if self.short:
            require(epoch == 1 and updates == SHORT_UPDATES and len(self.visits) == SHORT_VISITS
                    and should_stop, "Short SSDD FIT must exit without completing an epoch")
            return None
        require(not should_stop and updates == epoch * UPDATES_PER_EPOCH
                and len(self.visits) == VISITS_PER_EPOCH
                and Counter(iid for iid, _ in self.visits) == self.population,
                "Complete SSDD epoch requires the exact original padded FIT population")
        visits, self.visits = self.visits, []
        self.completed = epoch
        return visits


def mark_e24_archive(gate, checkpoint, phase_marker):
    require(PHASE_MARKER not in checkpoint and gate.armed_epoch == STOP_EPOCH
            and phase_marker["stage"] == "e24", "Only the fresh fixed SSDD E24 phase archive is allowed")
    gate.mark_archive(checkpoint)
    checkpoint[PHASE_MARKER] = phase_marker.copy()


def check_start_witness(witness, expected_initial_sha256, recipe_sha256):
    expected = dict(online_tensor_sha256=expected_initial_sha256, state_tensors=545, num_queries=100,
                    num_select=100, query_groups=13, training_queries_per_image=1300, class_head_slots=2,
                    initialization_seed=SEED, coco_asset_sha256=COCO_SHA, coco_asset_bytes=134545398,
                    recipe_sha256=recipe_sha256)
    require(type(witness) is dict and witness.keys() == expected.keys()
            and all(type(witness[key]) is type(value) and witness[key] == value for key, value in expected.items()),
            "Exact fresh official SSDD Q100 start witness required")


def retained_target(torch, image, targets, allowed_ids, empty_guard, phase):
    require(image.tensors.shape == (1, 3, 576, 576) and image.tensors.dtype == torch.float32
            and image.tensors.device.type == "cuda" and len(targets) == 1, "Exact native SSDD FIT batch")
    target, = targets
    labels, masks, boxes = (target[name] for name in ("labels", "masks", "boxes"))
    count = labels.numel()
    iid = int(target["image_id"])
    require(iid in allowed_ids and count <= QUERY_WIDTH and labels.shape == (count,) and labels.dtype == torch.int64
            and masks.dtype == torch.bool and masks.shape == (count, 576, 576)
            and boxes.dtype == torch.float32 and boxes.shape == (count, 4),
            "Exact native SSDD image ID, labels, masks, boxes and Q100 capacity")
    half_wh = boxes[:, 2:] / 2
    low, high = boxes[:, :2] - half_wh, boxes[:, :2] + half_wh
    valid = torch.stack(((labels == 0).all(), torch.isfinite(boxes).all(), (half_wh > 0).all(),
                         (low >= -1e-6).all(), (high <= 1 + 1e-6).all())).tolist()
    require(all(valid), "Native SSDD label or box hard guard failed")
    record = phase.batch_identity(image, target) if phase is not None else None
    removed = empty_guard(target)
    retained = target["labels"].numel()
    require(type(removed) is int and 0 <= removed <= count and retained == count - removed
            and target["masks"].shape == (retained, 576, 576) and target["boxes"].shape == (retained, 4),
            "Native empty-mask guard must retain aligned SSDD target rows")
    if record is not None:
        record.update(postfilter_labels=retained, postfilter_masks_sha256=phase.tensor_digest(target["masks"]),
                      postfilter_boxes_sha256=phase.tensor_digest(target["boxes"]))
    return iid, retained, removed, record


def run_fit_arm(*, stage, mode, output, fit_layout, coco_path, expected_fit_ids, expected_initial_sha256,
                protocol, recipe_sha256, source_manifest_sha256, dependencies, event_sink=None):
    """Run one fresh FIT-only arm inside the already admitted owner process."""
    allowed_ids = fit_id_set(expected_fit_ids)
    require(stage in ("qualification", "e24") and mode in ("native", "center")
            and expected_initial_sha256 == Q100_SHA and exact_hash(protocol, 40)
            and exact_hash(recipe_sha256, 64) and exact_hash(source_manifest_sha256, 64),
            "Exact owner-admitted SSDD stage, mode, recipe, source and Q100 initialization required")
    output, fit_layout, coco_path = (Path(path) for path in (output, fit_layout, coco_path))
    require(output.is_absolute() and output.parent.resolve(strict=True) == output.parent
            and not output.exists() and not output.is_symlink() and output != fit_layout
            and not output.is_relative_to(fit_layout) and not fit_layout.is_relative_to(output),
            "Fresh canonical SSDD arm output must be isolated from FIT")
    require(event_sink is None or callable(event_sink), "Expected an optional owner event sink")

    import torch
    from pytorch_lightning import Callback, seed_everything
    from pytorch_lightning.callbacks import ModelCheckpoint
    from rfdetr.training.callbacks.ema import RFDETREMACallback
    from rfdetr.training.module_data import GradAccumAlignedDataset
    from rfdetr.training.trainer import build_trainer

    def event(name, **values):
        if event_sink is not None:
            event_sink(dict(event=name, **values))

    deps = dependencies
    short = stage == "qualification"
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1
            and torch.cuda.is_bf16_supported(including_emulation=False), "One native BF16 SSDD GPU required")
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True
    model, train, module, init = deps.initialize_q100(fit_layout, output, coco_path)
    check_start_witness(init, expected_initial_sha256, recipe_sha256)
    require(train.seed == SEED and Path(train.dataset_dir) == fit_layout and Path(train.output_dir) == output,
            "Exact admitted effective SSDD FIT training configuration required")
    phase_marker = dict(training_protocol=protocol, stage=stage, mask_resize_mode=mode,
                        seed=SEED, initialization_seed=SEED, num_queries=QUERY_WIDTH, num_select=QUERY_WIDTH,
                        query_groups=13, class_head_slots=2, coco_asset_sha256=COCO_SHA,
                        initial_online_sha256=expected_initial_sha256, recipe_sha256=recipe_sha256,
                        fit_ids_sha256=sha256(encode(sorted(allowed_ids))).hexdigest(),
                        source_manifest_sha256=source_manifest_sha256)
    event("SSDD_MASK_PHASE_INITIALIZED", phase=phase_marker, initialization=init)
    data = deps.fit_only_data(model, train, fit_layout)
    output.mkdir()

    class Observe(Callback):
        def __init__(self, visit_stream, audit_stream):
            super().__init__()
            self.visit_stream, self.audit_stream = visit_stream, audit_stream
            self.handle = None
            self.updates = 0
            self.records = []

        def clocks(self, step):
            optimizer = self.optimizer.state_dict()
            scheduler = self.scheduler.state_dict()
            deps.e100.validate_schedule(optimizer, scheduler, step)
            deps.boundary.adam_progress(optimizer, step)
            deps.boundary.ema_progress({deps.boundary.EMA_KEY: ema.state_dict()}, step)
            counts = Counter(str(int(deps.boundary.scalar(row["step"]))) for row in optimizer["state"].values())
            return dict(adam_parameter_clocks=dict(counts), native_base_lrs=scheduler["base_lrs"],
                        native_next_lrs=scheduler["_last_lr"])

        def on_train_start(self, trainer, pl_module):
            deps.boundary.trainer_recipe(trainer, pl_module)
            require(trainer.current_epoch == trainer.global_step == 0 and not trainer.should_stop
                    and not trainer.enable_validation and trainer.min_epochs == trainer.min_steps == 0,
                    "Fresh native SSDD FIT-only trainer must start at update zero")
            loader = trainer.train_dataloader
            native_ids = data._dataset_train.ids
            padded = loader.dataset
            generator = torch.Generator().manual_seed(0)
            expected_pad = torch.randint(0, FIT_IMAGES, (VISITS_PER_EPOCH - FIT_IMAGES,), generator=generator).tolist()
            require(type(padded) is GradAccumAlignedDataset and padded._dataset is data._dataset_train
                    and len(loader) == VISITS_PER_EPOCH and loader.batch_size == 1 and loader.num_workers == 0
                    and padded._pad_indices == expected_pad and fit_id_set(native_ids) == allowed_ids,
                    "Exact stock SSDD FIT741 loader and independent alignment indices required")
            self.progress = FitProgress(stage, allowed_ids, [native_ids[i] for i in expected_pad])
            self.optimizer, = trainer.optimizers
            configuration, = trainer.lr_scheduler_configs
            self.scheduler = configuration.scheduler
            anchors = (0, 1, UPDATES_PER_EPOCH, WARMUP_UPDATES, STOP_EPOCH * UPDATES_PER_EPOCH, PLANNED_UPDATES)
            require(type(self.optimizer) is torch.optim.AdamW and not self.optimizer.state
                    and type(self.scheduler) is torch.optim.lr_scheduler.LambdaLR and self.scheduler.last_epoch == 0
                    and configuration.interval == "step" and configuration.frequency == 1
                    and bool(self.scheduler.lr_lambdas) and all(
                        math.isclose(fn(step), deps.e100.schedule_factor(step), rel_tol=1e-9, abs_tol=1e-12)
                        for fn in self.scheduler.lr_lambdas for step in anchors),
                    "Fresh native SSDD AdamW and exact E100 scheduler anchors required")
            require(all(pl_module.criterion.weight_dict.get(key) == value for key, value in deps.e1.TERMS.items())
                    and trainer.gradient_clip_val == train.clip_max_norm,
                    "Exact native SSDD loss and clipping recipe required")
            self.classifier = pl_module.model.class_embed.weight
            require(self.classifier.shape[0] == 2 and self.classifier.dtype == torch.float32
                    and self.classifier.device.type == "cuda", "Native two-slot FP32 classifier witness required")
            self.initial = self.classifier.detach().cpu().clone()
            self.initial_classifier_sha256 = deps.phase.tensor_digest(self.initial)
            self.seed_witness = dict(train_config_seed=train.seed, torch_cpu_initial_seed=torch.initial_seed(),
                                     torch_cuda_initial_seed=torch.cuda.initial_seed(),
                                     lightning_global_seed=os.environ.get("PL_GLOBAL_SEED"),
                                     lightning_seed_workers=os.environ.get("PL_SEED_WORKERS"))
            require(self.seed_witness == dict(train_config_seed=SEED, torch_cpu_initial_seed=SEED,
                                              torch_cuda_initial_seed=SEED, lightning_global_seed=str(SEED),
                                              lightning_seed_workers="1"),
                    "Actual effective SSDD seed after native fit initialization required")
            self.handle = self.optimizer.register_step_post_hook(self.adam_returned)
            event("SSDD_MASK_PHASE_TRAIN_STARTED", seed_witness=self.seed_witness,
                  visits_per_epoch=VISITS_PER_EPOCH, planned_updates=PLANNED_UPDATES,
                  warmup_updates=WARMUP_UPDATES, updates_per_epoch=UPDATES_PER_EPOCH)

        def on_train_epoch_start(self, trainer, pl_module):
            require(trainer.current_epoch == self.progress.completed and not self.progress.visits
                    and trainer.global_step == self.updates == self.progress.completed * UPDATES_PER_EPOCH
                    and not trainer.should_stop, "Native SSDD epoch start clocks differ")
            self.losses = []
            self.gradients = []
            self.removed_at_epoch_start = self.progress.removed_empty_rows
            self.epoch_start = perf_counter()

        def adam_returned(self, optimizer, args, kwargs):
            require(optimizer is self.optimizer, "Unexpected SSDD AdamW post-hook source")
            self.updates += 1

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            image, targets = batch
            iid, retained, removed, record = retained_target(
                torch, image, targets, allowed_ids, deps.empty_guard, deps.phase if short else None)
            epoch = trainer.current_epoch + 1
            self.progress.consume(epoch=epoch, batch=batch_idx, image_id=iid, retained=retained, removed=removed)
            if record is not None:
                self.records.append(dict(batch=batch_idx + 1, **record))
            if removed:
                event("SSDD_MASK_PHASE_EMPTY_FILTER_APPLIED", epoch=epoch, batch_idx=batch_idx,
                      image_id=iid, removed_rows=removed)

        def on_before_backward(self, trainer, pl_module, loss):
            require(bool(torch.isfinite(loss)), "Finite native SSDD FIT loss before backward required")

        def on_before_optimizer_step(self, trainer, pl_module, optimizer):
            gradient = self.classifier.grad
            require(optimizer is self.optimizer and trainer.global_step == self.updates == self.scheduler.last_epoch
                    and gradient is not None and gradient.dtype == torch.float32,
                    "Actual native SSDD optimizer and classifier gradient clocks required")
            norm = float(torch.linalg.vector_norm(gradient.detach()))
            require(math.isfinite(norm), "Finite SSDD classifier gradient norm required")
            self.gradients.append(norm)

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            require(trainer.global_step == self.updates, "Actual native SSDD AdamW return count differs")
            end_short = self.progress.after_batch(batch_idx, self.updates)
            loss = float(outputs["loss"].detach())
            require(math.isfinite(loss), "Finite actual native SSDD FIT loss required")
            self.losses.append(loss)
            if end_short:
                self.ending_classifier_sha256 = deps.phase.tensor_digest(self.classifier)
                require(self.ending_classifier_sha256 != self.initial_classifier_sha256,
                        "SSDD online classifier must update before the EMA switch")
                trainer.should_stop = True
            if (batch_idx + 1) % ACCUMULATION == 0 and self.updates % 100 == 0:
                event("SSDD_MASK_PHASE_PROGRESS", epoch=trainer.current_epoch + 1,
                      visits_in_epoch=batch_idx + 1, optimizer_updates=self.updates)

        def on_train_epoch_end(self, trainer, pl_module):
            epoch = trainer.current_epoch + 1
            visits = self.progress.finish_epoch(epoch, self.updates, should_stop=trainer.should_stop)
            expected_visits, expected_updates = (SHORT_VISITS, SHORT_UPDATES) if short else (VISITS_PER_EPOCH, UPDATES_PER_EPOCH)
            require(len(self.losses) == expected_visits and len(self.gradients) == expected_updates
                    and max(self.gradients) > 0, "Complete finite SSDD loss and gradient observations required")
            self.last_clock = self.clocks(self.updates)
            if short:
                return
            retained = [count for _, count in visits]
            visit_sha = append_jsonl(self.visit_stream, dict(epoch=epoch, visits=visits))
            audit = dict(epoch=epoch, visits=VISITS_PER_EPOCH, unique_fit_images=FIT_IMAGES,
                         duplicate_fit_visits=VISITS_PER_EPOCH - FIT_IMAGES, zero_target_visits=retained.count(0),
                         total_augmented_targets=sum(retained), maximum_augmented_targets=max(retained),
                         discarded_empty_mask_rows=self.progress.removed_empty_rows - self.removed_at_epoch_start,
                         optimizer_updates=self.updates, ema_updates=self.updates, visit_row_sha256=visit_sha,
                         loss_min=min(self.losses), loss_max=max(self.losses), classifier_gradient_norm_max=max(self.gradients),
                         epoch_seconds=perf_counter() - self.epoch_start, **self.last_clock)
            append_jsonl(self.audit_stream, audit)
            gate.complete_epoch(trainer, epoch)
            event("SSDD_MASK_PHASE_EPOCH_OBSERVED", **audit)

        def on_save_checkpoint(self, trainer, pl_module, checkpoint):
            mark_e24_archive(gate, checkpoint, phase_marker)

        def on_validation_start(self, trainer, pl_module):
            raise RuntimeError("SSDD DEV, TEST and prediction are forbidden inside the FIT arm")

        on_test_start = on_predict_start = on_validation_start

    with ExitStack() as files:
        visits = files.enter_context((output / VISITS_NAME).open("xb")) if not short else None
        audits = files.enter_context((output / AUDIT_NAME).open("xb")) if not short else None
        observe = Observe(visits, audits)
        trainer = build_trainer(train, model, accelerator="gpu", devices=1, extra_callbacks=[observe],
                                limit_val_batches=0, num_sanity_val_steps=0, min_epochs=0, min_steps=0,
                                logger=False, enable_model_summary=False, enable_progress_bar=False)
        trainer.callbacks[:] = [callback for callback in trainer.callbacks if type(callback).__name__ not in {
            "BestModelCallback", "COCOEvalCallback", "_ForceLastEpochValidationCallback", "GPUMemoryTQDMProgressBar"}]
        ema, = (callback for callback in trainer.callbacks if type(callback) is RFDETREMACallback)
        archive, = (callback for callback in trainer.checkpoint_callbacks if type(callback) is ModelCheckpoint)
        require(archive.filename == "checkpoint_{epoch}" and archive.monitor is None
                and archive.save_weights_only is False and archive.save_on_exception is False
                and Path(archive.dirpath) == output, "Exact stock SSDD full archive callback required")
        gate = deps.e100.ArchiveEndpoints(archive, stop_epoch=STOP_EPOCH, recipe_sha256=recipe_sha256)
        seed_everything(SEED, workers=True, verbose=False)
        torch.cuda.reset_peak_memory_stats()
        started = perf_counter()
        try:
            with deps.phase.mask_resize_phase(deps.native_transforms, mode):
                trainer.fit(module, datamodule=data, ckpt_path=None)
        finally:
            if observe.handle is not None:
                observe.handle.remove()

    torch.cuda.synchronize()
    completed = 0 if short else STOP_EPOCH
    visits = SHORT_VISITS if short else STOP_EPOCH * VISITS_PER_EPOCH
    steps = SHORT_UPDATES if short else STOP_EPOCH * UPDATES_PER_EPOCH
    require(observe.progress.completed == completed and observe.progress.total_visits == visits
            and trainer.current_epoch == (1 if short else STOP_EPOCH)
            and trainer.global_step == observe.updates == steps and trainer.should_stop
            and data._dataset_val is None and data._dataset_test is None,
            "Exact uninterrupted SSDD FIT arm closure and stage clocks required")
    clock = observe.clocks(steps)
    if short:
        require(gate.completed == 0 and gate.armed_epoch is None and not gate.saved and len(observe.records) == visits
                and not any(output.iterdir()), "SSDD qualification may not create an archive or completed-epoch ledger")
        visit_path = output / SHORT_NAME
        deps.phase.exclusive_json(visit_path, observe.records)
        summary = dict(visit_ledger_sha256=digest_file(visit_path), checkpoint_saved=False,
                       input_pair_sha256=sha256(encode([(row["image_id"], row["image_sha256"], row["cpu_rng_sha256"])
                                                       for row in observe.records])).hexdigest(),
                       classifier_initial_sha256=observe.initial_classifier_sha256,
                       classifier_online_ending_sha256=observe.ending_classifier_sha256,
                       targets_after_filter=sum(row["postfilter_labels"] for row in observe.records),
                       loss_min=min(observe.losses), loss_max=max(observe.losses),
                       classifier_gradient_norm_max=max(observe.gradients))
    else:
        require(gate.verify_completion() == (STOP_EPOCH,), "Only the original fixed SSDD E24 archive is required")
        path = output / f"checkpoint_{STOP_EPOCH - 1}.ckpt"
        require(set(output.iterdir()) == {path, output / VISITS_NAME, output / AUDIT_NAME},
                "Unexpected SSDD E24 output roster")
        summary = dict(fit_visit_ledger_sha256=digest_file(output / VISITS_NAME),
                       epoch_audit_sha256=digest_file(output / AUDIT_NAME), checkpoint_saved=True,
                       archive=deps.e100.inspect_archive(path, epoch=STOP_EPOCH, torch=torch, e1_api=deps.e1,
                                                         boundary_api=deps.boundary, recipe_sha256=recipe_sha256,
                                                         initial_witness=observe.initial))
    report = dict(status="SSDD_MASK_PHASE_ARM_CLOSED_PENDING_INDEPENDENT_ACCEPTANCE_V1", phase=phase_marker,
                  initialization=init, seed_witness=observe.seed_witness, completed_epochs=completed,
                  actual_trainer_epoch_cursor=trainer.current_epoch, visits=visits, optimizer_updates=steps,
                  ema_updates=steps, planned_epochs=PLANNED_EPOCHS, planned_updates=PLANNED_UPDATES,
                  warmup_updates=WARMUP_UPDATES, fit_images=FIT_IMAGES, visits_per_epoch=VISITS_PER_EPOCH,
                  removed_empty_rows=observe.progress.removed_empty_rows, precision=trainer.precision,
                  float32_matmul_precision=torch.get_float32_matmul_precision(),
                  cudnn_tf32=torch.backends.cudnn.allow_tf32, elapsed_seconds=perf_counter() - started,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                  FIT=True, DEV=False, TEST=False, AP=False, **clock, **summary)
    deps.phase.exclusive_json(output / RESULT_NAME, report)
    event("SSDD_MASK_PHASE_ARM_RETURNED", **report)
    return report

