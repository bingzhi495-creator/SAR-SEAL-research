"""Run one complete native HRSID FIT epoch under the frozen E100 schedule."""
import argparse
from collections import Counter
from hashlib import sha256
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
PROBE = PRIVATE / "HRSID_FIT_REAL_NATIVE_MATCH_CANDIDATE_20261109_02/probe_native_match.py"
PROBE_SHA = "fbc793f8df253c9f340ffa671cb502c0a9a770facce126f15ad7d15de5b416f0"
LAYOUT = PRIVATE / "HRSID_FIT_NATIVE_COCO_LAYOUT_ACTUAL_PRIVATE_20261109_01"
LAYOUT_SHA = "aa4e985e20647b6ea88ca428239da5eebc8380aee8974e97c89980c129c265b1"
LAYOUT_ROOT_SHA = "0ac416d96575b9fe5b9784d2252e0bc3430277e5e88d1437d13830c2039bf7f7"
OVERLAY = PRIVATE / "RFDETR_WINDOWS_HRSID_E1_IMPORT_OVERLAY_20261109_01"
WHEEL_SHA = "3eba19bf8e2c0365e698bcc51a88797fef8b13320d333f3accfd7684de59051d"
INDEX_SHA = "e95cf26cc60a2c6fb651e907541380a256f2df7b1e153244a6c0f803b0b19f8f"
E24_EXPANDED_SHA = "24fbfbf3783d6cffbebccb4dc0cb32b33cd14db41ad72252f17e30983a5128df"
E24_RAW_SHA = "5da981aade9d676a601c1b6da5c362c2855438e45d8d7deb5c41368d7008f4de"
VISITS, FIT, ACCUM, UPDATES = 2940, 2937, 4, 735
EPOCHS, WARMUP, TOTAL = 100, 2205, 73500
CAP = 6 * 1024**3
WIDTH = 164
SEED = 5201
ARCHIVE_NAME = "checkpoint_0.ckpt"
MARKER = "sar_seal_hrsid_fit_native_e1_v1"


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def schedule_factor(step):
    if step < WARMUP:
        return step / WARMUP
    return .05 + .95 * .5 * (1 + math.cos(math.pi * (step - WARMUP) / (TOTAL - WARMUP)))


def pinned_bytes(path, digest):
    data = path.read_bytes()
    need(sha256(data).hexdigest() == digest, f"Exact pinned source required for {path.name}")
    return data


def closed_probe():
    module = ModuleType("_closed_hrsid_query_expansion")
    module.__file__ = str(PROBE)
    exec(compile(pinned_bytes(PROBE, PROBE_SHA), str(PROBE), "exec"), module.__dict__)
    return module


def admit_dependency_overlay():
    from io import BytesIO
    from zipfile import ZipFile

    wheel = OVERLAY / "faster_coco_eval-1.8.0-cp310-cp310-win_amd64.whl"
    site = OVERLAY / "site-packages"
    blob = pinned_bytes(wheel, WHEEL_SHA)
    need(len(blob) == 558663 and site.resolve(strict=True) == site, "Exact private Windows wheel overlay")
    with ZipFile(BytesIO(blob)) as archive:
        members = [item for item in archive.infolist() if not item.is_dir()]
        expected = {item.filename for item in members}
        actual = {p.relative_to(site).as_posix() for p in site.rglob("*") if p.is_file()}
        need(len(expected) == len(members) == 30 and actual == expected, "Exact isolated wheel file census")
        for member in members:
            path = site / member.filename
            need(path.resolve(strict=True) == path and site in path.parents and path.read_bytes() == archive.read(member),
                 "Installed private wheel member must equal the official source bytes")
    sys.path.append(str(site))
    import faster_coco_eval

    need(version("faster-coco-eval") == "1.8.0" and site in Path(faster_coco_eval.__file__).resolve().parents,
         "Exact isolated faster-coco-eval package is importable")


def fit_ids():
    receipt = json.loads(pinned_bytes(LAYOUT / "NATIVE_FIT_LAYOUT_SAFE.json", LAYOUT_SHA))
    independent = json.loads(pinned_bytes(LAYOUT / "ROOT_INDEPENDENT_LAYOUT_SAFE.json", LAYOUT_ROOT_SHA))
    need(receipt["native_index_sha256"] == independent["native_index_sha256"] == INDEX_SHA
         and receipt["private_manifest_sha256"] == independent["private_manifest_sha256"]
         and receipt["fit_images"] == independent["images"] == independent["files_sha256_verified"] == FIT
         and receipt["retained_annotations"] == independent["annotations"] == 9260
         and receipt["FIT_only"] is independent["no_DEV_TEST_directories"] is True
         and not (LAYOUT / "valid").exists() and not (LAYOUT / "test").exists(), "Exact accepted FIT-only P115 layout")
    view = json.loads(pinned_bytes(LAYOUT / "train/_annotations.coco.json", INDEX_SHA))
    ids = {row["id"] for row in view["images"]}
    need(len(ids) == len(view["images"]) == FIT and len(view["annotations"]) == 9260,
         "Exact admitted FIT image and annotation counts")
    return ids


def ema_clock(ema_state, step):
    need(int(ema_state["latest_update_step"]) == int(ema_state["average_model_state_dict"]["n_averaged"]) == step,
         "Native EMA cursor differs from actual AdamW post-step count")


def verify_loop(checkpoint):
    need(checkpoint["pytorch-lightning_version"] == "2.6.6"
         and checkpoint["epoch"] == 0 and checkpoint["global_step"] == UPDATES,
         "Native full archive is not on the first exact epoch boundary")
    fit_loop = checkpoint["loops"]["fit_loop"]
    epoch = fit_loop["epoch_progress"]
    batches = fit_loop["epoch_loop.batch_progress"]
    steps = fit_loop["epoch_loop.automatic_optimization.optim_progress"]["optimizer"]["step"]
    for scope in ("current", "total"):
        need(all(epoch[scope][key] == count
                 for key, count in (("ready", 1), ("started", 1), ("processed", 1), ("completed", 0)))
             and all(batches[scope][key] == VISITS for key in ("ready", "started", "processed", "completed"))
             and all(steps[scope][key] == UPDATES for key in ("ready", "completed")),
             "Native E1 loop epoch, microbatch or actual update cursors differ")
    need(batches["is_last_batch"] is True, "Native checkpoint must contain the final E1 batch")


def check_optimizer(optimizer, scheduler):
    need(scheduler["last_epoch"] == UPDATES, "Native scheduler cursor differs from optimizer clock")
    base, next_lrs, groups = scheduler["base_lrs"], scheduler["_last_lr"], optimizer["param_groups"]
    need(len(base) == len(next_lrs) == len(groups) > 0 and all(
        math.isclose(a / 3, b, rel_tol=1e-9) and b == group["lr"]
        for a, b, group in zip(base, next_lrs, groups, strict=True)), "Native E100 schedule E1 LR must be base/3")
    counts = Counter(int(value["step"]) for value in optimizer["state"].values())
    need(bool(counts) and max(counts) == UPDATES and all(0 < clock <= UPDATES for clock in counts)
         and all({"step", "exp_avg", "exp_avg_sq"} <= value.keys() for value in optimizer["state"].values()),
         "Native AdamW archive requires complete moment slots and observed update count")
    return dict(adam_parameter_clocks=dict(counts), native_base_lrs=base, native_next_lrs=next_lrs)


def event(name, **values):
    print(json.dumps(dict(event=name, **values), sort_keys=True, allow_nan=False), flush=True)


def run(output, protocol):
    import torch
    from pytorch_lightning import Callback, seed_everything
    from pytorch_lightning.callbacks import ModelCheckpoint
    from rfdetr.training.callbacks.ema import RFDETREMACallback
    from rfdetr.training.module_data import GradAccumAlignedDataset, RFDETRDataModule
    from rfdetr.training.module_model import RFDETRModelModule
    from rfdetr.training.trainer import build_trainer

    need(torch.__version__ == "2.10.0+cu130" and version("rfdetr") == "1.12.0"
         and version("pytorch-lightning") == "2.6.6", "Exact original numeric runtime")
    need(torch.cuda.is_available() and torch.cuda.device_count() == 1
         and torch.cuda.is_bf16_supported(including_emulation=False), "One visible native BF16 CUDA GPU")
    free, total = torch.cuda.mem_get_info(0)
    need(free >= CAP, "GPU free memory floor")
    torch.cuda.set_per_process_memory_fraction(min(.35, CAP / total), 0)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    started = perf_counter()
    ids = fit_ids()
    probe = closed_probe()
    need(probe.file_digest(probe.STOCK / "state_preservation.py") == probe.STATE_SHA, "Pinned E24 state helper")
    stock = probe.read_pinned_module(probe.STOCK / "run_state_arm.py", probe.STOCK_SHA, "_closed_hrsid_training_e24")
    sys.addaudithook(stock.offline)
    seed_everything(SEED, workers=True)
    original = stock.p77_source()
    native, warm, unused_criterion, unused_adam, unused_ema, _, initialization = stock.initialize(
        "F", torch.device("cpu"), original)
    need(initialization["archive_start_sha256"] == probe.E24_SHA
         and initialization["initial_online_sha256"] == E24_RAW_SHA, "Exact independently closed E24 online initialization")
    del unused_criterion, unused_adam, unused_ema
    probe.expand_groups(warm, WIDTH)
    need(stock.tensor_dict_sha256(warm.state_dict()) == E24_EXPANDED_SHA, "Exact previously verified width-164 expansion")
    configuration = original.module(native["sources"]["configuration"], "_hrsid_e100_configuration")
    recipe = original.exact_file(native["recipe"], 1024**2)
    model, train = configuration._load_configs(recipe, LAYOUT)
    model = model.model_copy(update={"pretrain_weights": None, "num_queries": WIDTH})
    train = train.model_copy(update={"output_dir": str(output), "compute_train_metrics": False})
    need(model.resolution == 576 and model.num_queries == WIDTH and model.num_select == 100 and model.group_detr == 13
         and train.epochs == EPOCHS and train.batch_size == 1 and train.grad_accum_steps == ACCUM
         and train.num_workers == 0 and train.warmup_epochs == 3 and train.optimizer == "adamw"
         and train.use_ema and train.ema_decay == .993 and train.ema_tau == 100
         and train.checkpoint_interval == 1 and train.clip_max_norm == .1 and train.scale_jitter
         and train.augmentation_backend == "torchvision" and train.dataset_file == "roboflow"
         and Path(train.dataset_dir) == LAYOUT and not train.compute_train_metrics and not train.run_test,
         "Exact native FIT schedule, model, loss, sampler and augmentation recipe")
    module = RFDETRModelModule(model, train, load_encoder_weights=False)
    module.model.load_state_dict(warm.state_dict(), strict=True)
    need(stock.tensor_dict_sha256(module.model.state_dict()) == E24_EXPANDED_SHA,
         "Strict width-164 model readback after warm transfer")
    del warm
    initial_classifier = module.model.class_embed.weight.detach().cpu().clone()

    class FitOnlyData(RFDETRDataModule):
        def setup(self, stage):
            need(stage == "fit" and self._dataset_val is None and self._dataset_test is None,
                 "Non-FIT setup is forbidden")
            # The stock FIT setup builds val when its slot is unset. Occupy it only during setup;
            # native FIT dataset construction, checking, augmentation and loader stay unchanged.
            self._dataset_val = object()
            try:
                super().setup(stage)
            finally:
                self._dataset_val = None
            need(len(self._dataset_train) == FIT, "Exact stock FIT dataset length")

        def val_dataloader(self):
            raise RuntimeError("Non-FIT loaders are forbidden")

        test_dataloader = predict_dataloader = val_dataloader

    class Observe(Callback):
        def __init__(self):
            super().__init__()
            self.visits = []
            self.losses = []
            self.counts = Counter()
            self.gradients = []
            self.updates = 0
            self.handle = None
            self.completed = self.saved = False

        def on_train_start(self, trainer, pl_module):
            need(trainer.current_epoch == trainer.global_step == 0 and not trainer.enable_validation
                 and trainer.max_epochs == EPOCHS and trainer.max_steps == -1
                 and trainer.accumulate_grad_batches == ACCUM and trainer.num_training_batches == VISITS
                 and trainer.estimated_stepping_batches == TOTAL and trainer.gradient_clip_val == .1,
                 "Exact initial native trainer clock, FIT-only and full schedule")
            loader = trainer.train_dataloader
            need(type(loader.dataset) is GradAccumAlignedDataset and len(loader) == VISITS
                 and loader.batch_size == 1 and loader.num_workers == 0,
                 "Stock aligned single-worker training loader")
            self.optim, = trainer.optimizers
            config, = trainer.lr_scheduler_configs
            self.scheduler = config.scheduler
            need(type(self.optim) is torch.optim.AdamW and len(self.optim.state) == 0
                 and type(self.scheduler) is torch.optim.lr_scheduler.LambdaLR and self.scheduler.last_epoch == 0
                 and config.interval == "step" and config.frequency == 1 and all(
                     math.isclose(fn(step), schedule_factor(step), rel_tol=1e-9, abs_tol=1e-12)
                     for fn in self.scheduler.lr_lambdas for step in (0, 1, UPDATES, WARMUP, TOTAL)),
                 "Fresh AdamW and exact native E100 warmup/cosine schedule")
            ema_clock(ema.state_dict(), 0)
            self.handle = self.optim.register_step_post_hook(self.adam_returned)
            event("HRSID_NATIVE_E1_STARTED", expected_visits=VISITS, expected_optimizer_updates=UPDATES,
                  full_schedule_updates=TOTAL)

        def adam_returned(self, optimizer, args, kwargs):
            need(optimizer is self.optim, "Unexpected optimizer source")
            self.updates += 1

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            image, targets = batch
            need(trainer.current_epoch == 0 and batch_idx == len(self.visits) < VISITS
                 and image.tensors.shape == (1, 3, 576, 576) and image.tensors.dtype == torch.float32
                 and image.tensors.device.type == "cuda" and len(targets) == 1, "Exact actual native FIT batch")
            target, = targets
            labels, masks, boxes = target["labels"], target["masks"], target["boxes"]
            count = labels.numel()
            need(count <= WIDTH and labels.shape == (count,) and labels.dtype == torch.int64
                 and masks.dtype == torch.bool and masks.shape == (count, 576, 576)
                 and boxes.shape == (count, 4) and boxes.dtype == torch.float32,
                 "Exact native target shapes and per-image query capacity")
            half_wh = boxes[:, 2:] / 2
            low, high = boxes[:, :2] - half_wh, boxes[:, :2] + half_wh
            need(bool((labels == 0).all() & masks.flatten(1).any(1).all()
                      & torch.isfinite(boxes).all() & (half_wh > 0).all()
                      & (low >= -1e-6).all() & (high <= 1 + 1e-6).all()),
                 "Every retained native target has a nonempty mask and a finite normalized positive box")
            iid = int(target["image_id"])
            need(iid in ids, "Only the admitted native FIT image population")
            self.visits.append((iid, count))
            self.counts[count] += 1

        def on_before_backward(self, trainer, pl_module, loss):
            need(bool(torch.isfinite(loss)), "Native loss must be finite before backward")

        def on_before_optimizer_step(self, trainer, pl_module, optimizer):
            need(optimizer is self.optim and trainer.global_step == self.updates == self.scheduler.last_epoch,
                 "Optimizer, Lightning and scheduler pre-step cursors must agree")
            witness = pl_module.model.class_embed.weight.grad
            need(witness is not None and witness.dtype == torch.float32, "Native BF16 classifier gradient witness")
            gradient = float(torch.linalg.vector_norm(witness.detach()))
            need(math.isfinite(gradient), "Finite classifier gradient before actual update")
            self.gradients.append(gradient)

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            need(trainer.global_step == self.updates == (batch_idx + 1) // ACCUM,
                 "Native batch and actual AdamW post-step cursors must agree")
            loss = float(outputs["loss"].detach())
            need(math.isfinite(loss), "Finite native returned training loss")
            self.losses.append(loss)
            if (batch_idx + 1) % (ACCUM * 100) == 0:
                event("HRSID_NATIVE_E1_PROGRESS", visits=batch_idx + 1, optimizer_updates=self.updates,
                      zero_target_visits=self.counts[0], max_targets=max(self.counts),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated())

        def on_train_epoch_end(self, trainer, pl_module):
            need(not self.completed and trainer.current_epoch == 0
                 and len(self.visits) == len(self.losses) == VISITS and len(self.gradients) == UPDATES
                 and trainer.global_step == self.updates == UPDATES and max(self.gradients) > 0
                 and {iid for iid, _ in self.visits} == ids, "Only stop after complete native FIT and AdamW epoch")
            ema_clock(ema.state_dict(), UPDATES)
            check_optimizer(self.optim.state_dict(), self.scheduler.state_dict())
            self.completed = True
            trainer.should_stop = True
            event("HRSID_NATIVE_E1_EPOCH_OBSERVED", visits=VISITS, optimizer_updates=UPDATES)

        def on_save_checkpoint(self, trainer, pl_module, checkpoint):
            need(self.completed and not self.saved and checkpoint["global_step"] == UPDATES
                 and checkpoint["epoch"] == 0 and MARKER not in checkpoint, "Only one full completed E1 archive")
            checkpoint[MARKER] = dict(protocol=protocol, FIT_only=True, visits=VISITS, planned_epochs=EPOCHS,
                                      actual_adam_updates=UPDATES, native_index_sha256=INDEX_SHA,
                                      e24_expanded_sha256=E24_EXPANDED_SHA)
            self.saved = True

        def on_validation_start(self, trainer, pl_module):
            raise RuntimeError("Non-FIT evaluation is forbidden")

        on_test_start = on_predict_start = on_validation_start

    output.mkdir()
    observe = Observe()
    data = FitOnlyData(model, train)
    trainer = build_trainer(train, model, accelerator="gpu", devices=1, extra_callbacks=[observe],
                            limit_val_batches=0, num_sanity_val_steps=0, min_epochs=0, min_steps=0,
                            logger=False, enable_model_summary=False, enable_progress_bar=False)
    # Native callback construction remains authoritative for EMA and full archive. No evaluation callbacks are used.
    trainer.callbacks[:] = [cb for cb in trainer.callbacks if type(cb).__name__ not in {
        "BestModelCallback", "COCOEvalCallback", "_ForceLastEpochValidationCallback", "GPUMemoryTQDMProgressBar"}]
    ema, = (cb for cb in trainer.callbacks if type(cb) is RFDETREMACallback)
    checkpoint_callback, = (cb for cb in trainer.checkpoint_callbacks if type(cb) is ModelCheckpoint)
    need(trainer.precision == "bf16-mixed" and trainer.scaler is None and not trainer.enable_validation
         and checkpoint_callback.filename == "checkpoint_{epoch}" and checkpoint_callback.monitor is None
         and checkpoint_callback.save_weights_only is False and checkpoint_callback.save_on_exception is False,
         "Stock fresh BF16 FIT, EMA and full epoch archive")
    torch.cuda.reset_peak_memory_stats()
    seed_everything(SEED, workers=True)
    try:
        trainer.fit(module, datamodule=data, ckpt_path=None)
    finally:
        if observe.handle is not None:
            observe.handle.remove()
    archive = output / ARCHIVE_NAME
    need(trainer.current_epoch == 1 and trainer.global_step == observe.updates == UPDATES and trainer.should_stop
         and observe.completed and observe.saved and {p.name for p in output.iterdir()} == {ARCHIVE_NAME}
         and data._dataset_val is data._dataset_test is None, "Exact FIT completion and sole native archive")
    torch.cuda.synchronize()
    with archive.open("rb") as stream:
        checkpoint = torch.load(stream, map_location="cpu", weights_only=True)
    verify_loop(checkpoint)
    need(checkpoint[MARKER]["protocol"] == protocol and checkpoint["model_config"]["num_queries"] == WIDTH
         and checkpoint["model_config"]["num_select"] == 100, "Native checkpoint model metadata and protocol")
    optimizer, = checkpoint["optimizer_states"]
    scheduler, = checkpoint["lr_schedulers"]
    optimizer_summary = check_optimizer(optimizer, scheduler)
    saved_ema = checkpoint["callbacks"][ema.state_key]
    ema_clock(saved_ema, UPDATES)
    raw_state = {k.removeprefix("model."): v for k, v in checkpoint["state_dict"].items() if k.startswith("model.")}
    ema_state = {k.removeprefix("module.model."): v for k, v in saved_ema["average_model_state_dict"].items()
                 if k.startswith("module.model.")}
    need(raw_state.keys() == ema_state.keys() and raw_state["refpoint_embed.weight"].shape[0] == WIDTH * 13
         and not torch.equal(initial_classifier, raw_state["class_embed.weight"]), "Native online weights changed")
    raw_digest = stock.tensor_dict_sha256(raw_state)
    ema_digest = stock.tensor_dict_sha256(ema_state)
    need(raw_digest != ema_digest, "Distinct archived online and new EMA weights")
    ledger = json.dumps(observe.visits, separators=(",", ":")).encode("utf-8") + b"\n"
    with (output / "FIT_VISITS_PRIVATE.json").open("xb") as stream:
        stream.write(ledger)
    result = dict(status="NATIVE_HRSID_FIT_E1_COMPLETE_PENDING_INDEPENDENT_ACCEPTANCE", protocol=protocol,
                  archive_name=ARCHIVE_NAME, archive_sha256=probe.file_digest(archive), archive_bytes=archive.stat().st_size,
                  native_index_sha256=INDEX_SHA, starting_e24_online_sha256=E24_RAW_SHA,
                  starting_expanded_online_sha256=E24_EXPANDED_SHA, ending_online_sha256=raw_digest,
                  ending_ema_sha256=ema_digest, query_width=WIDTH, query_groups=13, resolution=576,
                  visited_fit_images=FIT, visits=VISITS, duplicate_fit_visits=VISITS - FIT,
                  fit_visit_ledger_sha256=sha256(ledger).hexdigest(), zero_target_visits=observe.counts[0],
                  maximum_augmented_targets=max(observe.counts), total_augmented_targets=sum(
                      count * number for count, number in observe.counts.items()),
                  total_updates=UPDATES, starting_optimizer_updates=0, starting_ema_updates=0, ema_updates=UPDATES,
                  completed_epochs=1, scheduled_epochs=EPOCHS, scheduled_updates=TOTAL, warmup_updates=WARMUP,
                  lightning_accumulation_scaled_loss_mean=math.fsum(observe.losses) / VISITS,
                  lightning_accumulation_scaled_loss_min=min(observe.losses), lightning_accumulation_scaled_loss_max=max(observe.losses),
                  classifier_gradient_norm_max=max(observe.gradients), precision="bf16-mixed",
                  gpu_allocator_limit_bytes=CAP, gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(), elapsed_seconds=perf_counter() - started,
                  faster_coco_eval_wheel_sha256=WHEEL_SHA, evaluator_executed=False,
                  FIT_only=True, DEV=False, TEST=False, AP=False, **optimizer_summary)
    with (output / "NATIVE_HRSID_FIT_E1_SAFE.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    need(bool(re.fullmatch(r"[0-9a-f]{40}", args.protocol)) and args.output.is_absolute()
         and args.output.parent.resolve(strict=True) == args.output.parent
         and not args.output.exists() and not args.output.is_symlink() and args.output not in LAYOUT.parents
         and LAYOUT not in args.output.parents, "Committed protocol and fresh isolated absolute output")
    need(tuple(sys.version_info[:3]) == (3, 10, 13) and os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
         and os.environ.get("HF_HUB_OFFLINE") == os.environ.get("TRANSFORMERS_OFFLINE") == "1",
         "Pinned isolated offline GPU environment")
    admit_dependency_overlay()
    run(args.output, args.protocol)


if __name__ == "__main__":
    main()
