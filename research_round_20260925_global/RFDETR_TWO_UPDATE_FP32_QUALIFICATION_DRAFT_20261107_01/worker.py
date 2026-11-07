"""Observe two native optimizer updates of the admitted RF-DETR SSDD recipe."""

import math
from pathlib import Path
import time


TERMS = {"loss_ce": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0, "loss_mask_ce": 5.0, "loss_mask_dice": 5.0}
PROBES = ("class_embed.weight", "segmentation_head.bias")
PLANNED_STEPS = 18_600
WARMUP_STEPS = 558


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def run_probe(checkpoint_path, model_config, train_config):
    """Run the FP32 qualification once in the caller's admitted single-GPU process."""
    import torch
    from pytorch_lightning import Callback, seed_everything
    from rfdetr.config import AugmentationBackend, MultiScale, RFDETRSegNanoConfig, SegmentationTrainConfig
    from rfdetr.training.callbacks.ema import RFDETREMACallback
    from rfdetr.training.module_data import RFDETRDataModule
    from rfdetr.training.module_model import RFDETRModelModule
    from rfdetr.training.trainer import build_trainer

    checkpoint = Path(checkpoint_path)
    _require(checkpoint.is_absolute() and checkpoint.resolve(strict=True) == checkpoint and checkpoint.is_file(),
             "expected the admitted local COCO checkpoint")
    mc = model_config.model_copy(update={"pretrain_weights": str(checkpoint)})
    tc = train_config.model_copy(update={"amp_dtype": None})
    _require(type(mc) is RFDETRSegNanoConfig and mc.num_classes == 1 and mc.resolution == 576
             and mc.num_channels == 3 and mc.positional_encoding_size == 48 and mc.num_queries == 100
             and mc.group_detr == 13 and mc.mask_downsample_ratio == 4 and mc.segmentation_head
             and not any((mc.compile, mc.cuda_graphs, mc.fused_optimizer, mc.use_grouppose_keypoints, mc.backbone_lora)),
             "expected the frozen Nano architecture")
    _require(type(tc) is SegmentationTrainConfig and tc.epochs == 100 and tc.seed == 5201
             and tc.dataset_file == "roboflow" and tc.batch_size == 1 and tc.grad_accum_steps == 4
             and tc.num_workers == 0 and tc.square_resize_div_64 and tc.multi_scale is MultiScale.OFF
             and tc.augmentation_backend == AugmentationBackend.TV and tc.scale_jitter and tc.aug_config is None
             and tc.optimizer == "adamw" and tc.lr_scheduler == "cosine" and tc.warmup_epochs == 3
             and tc.lr_scheduler_kwargs == {"min_factor": 0.05} and tc.lr == 0.0001 and tc.lr_encoder == 0.00015
             and tc.weight_decay == 0.0001 and tc.clip_max_norm == 0.1 and tc.mask_point_sample_ratio == 16
             and tc.use_ema and tc.ema_update_interval == 1 and tc.progress_bar is None
             and not any((tc.save_dataset_grids, tc.compute_train_metrics, tc.run_test)),
             "expected the frozen training and accumulation recipe")
    dataset = Path(tc.dataset_dir)
    _require(dataset.is_absolute() and dataset.resolve(strict=True) == dataset and dataset.is_dir()
             and dataset.name != "REQUIRES_QUALIFIED_SAR_COCO_SPLITS", "expected the admitted native SSDD root")
    _require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "expected one visible CUDA device")

    ema = RFDETREMACallback(decay=tc.ema_decay, tau=tc.ema_tau, update_interval_steps=tc.ema_update_interval)

    class Observe(Callback):
        def __init__(self):
            self.batches = []
            self.updates = []
            self.pending = None
            self.before = None
            self.handle = None

        def on_train_start(self, trainer, pl_module):
            _require(pl_module.automatic_optimization and trainer.world_size == 1 and trainer.precision == "32-true"
                     and trainer.accumulate_grad_batches == 4 and trainer.max_steps == -1
                     and trainer.max_epochs == 100 and trainer.num_training_batches == 744
                     and trainer.estimated_stepping_batches == PLANNED_STEPS and not trainer.enable_validation
                     and trainer.gradient_clip_val == tc.clip_max_norm
                     and torch.get_float32_matmul_precision() == "highest" and not torch.backends.cudnn.allow_tf32,
                     "native full schedule or FP32 trainer differs")
            _require(len(trainer.optimizers) == len(trainer.lr_scheduler_configs) == 1,
                     "one native optimizer and scheduler")
            self.optimizer = trainer.optimizers[0]
            lr_config = trainer.lr_scheduler_configs[0]
            self.scheduler = lr_config.scheduler
            _require(type(self.optimizer) is torch.optim.AdamW and self.scheduler.last_epoch == 0
                     and type(self.scheduler) is torch.optim.lr_scheduler.LambdaLR
                     and lr_config.interval == "step" and lr_config.frequency == 1,
                     "native AdamW and step-wise LambdaLR")
            for schedule in self.scheduler.lr_lambdas:
                _require(all(math.isclose(schedule(step), factor, rel_tol=1e-9, abs_tol=1e-12) for step, factor in (
                    (0, 0.0), (1, 1 / WARMUP_STEPS), (WARMUP_STEPS, 1.0),
                    ((PLANNED_STEPS + WARMUP_STEPS) // 2, 0.525), (PLANNED_STEPS, 0.05))),
                    "native warmup and cosine schedule horizon")
            self.params = {name: pl_module.model.get_parameter(name) for name in PROBES}
            _require(all(p.dtype == torch.float32 and p.device.type == "cuda" for p in self.params.values()),
                     "FP32 CUDA witness parameters")
            _require(all(pl_module.criterion.weight_dict.get(k) == v for k, v in TERMS.items()),
                     "frozen native loss weights")
            self.handle = pl_module.criterion.register_forward_hook(self.criterion_output)

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            _require(self.pending is None and batch_idx == len(self.batches) < 8
                     and trainer.global_step == batch_idx // 4, "unexpected microbatch or pre-step count")
            images, targets = batch
            _require(tuple(images.tensors.shape) == (1, 3, 576, 576) and images.tensors.dtype == torch.float32
                     and images.tensors.device.type == "cuda" and len(targets) == 1, "native CUDA FIT batch at 576")
            target = targets[0]
            labels = target["labels"]
            count = labels.numel()
            _require(labels.dtype == torch.int64 and labels.device.type == "cuda" and bool((labels == 0).all())
                     and target["masks"].dtype == torch.bool and tuple(target["masks"].shape) == (count, 576, 576),
                     "native FIT ship labels and masks")
            self.pending = {"batch_index": batch_idx, "image_id": int(target["image_id"].item()),
                            "instances": count}

        def criterion_output(self, criterion, inputs, losses):
            _require(self.pending is not None and "weighted_loss" not in self.pending, "one native criterion call")
            prediction = inputs[0]
            masks = prediction["pred_masks"]
            _require(not torch.is_autocast_enabled("cuda") and prediction["pred_logits"].dtype == torch.float32
                     and tuple(prediction["pred_logits"].shape) == (1, 1300, 2)
                     and type(masks) is dict and tuple(masks["spatial_features"].shape[-2:]) == (144, 144)
                     and masks["spatial_features"].dtype == torch.float32, "FP32 native sparse-mask training path")
            names = [name for name in losses if name in criterion.weight_dict]
            _require(set(TERMS).issubset(names) and len(names) > len(TERMS), "base and auxiliary weighted losses")
            values = dict(zip(names, torch.stack([losses[name].detach() for name in names]).cpu().tolist()))
            _require(all(math.isfinite(value) for value in values.values()), "finite native component losses")
            self.pending.update(components={name: values[name] for name in TERMS}, weighted_terms=len(names),
                                weighted_loss=sum(value * criterion.weight_dict[name] for name, value in values.items()))

        def on_before_backward(self, trainer, pl_module, loss):
            _require("backward_loss" not in self.pending and loss.dtype == torch.float32,
                     "one FP32 backward per microbatch")
            observed = loss.detach().item()
            _require(math.isclose(observed, self.pending["weighted_loss"] / 4, rel_tol=2e-6, abs_tol=1e-7),
                     "native automatic accumulation divides the weighted loss by four")
            self.pending["backward_loss"] = observed

        def on_after_backward(self, trainer, pl_module):
            _require("backward_loss" in self.pending and "backward_completed" not in self.pending,
                     "exact native backward completion")
            self.pending["backward_completed"] = True

        def _adam_steps(self):
            steps = {}
            for name, parameter in self.params.items():
                step = self.optimizer.state.get(parameter, {}).get("step")
                steps[name] = 0 if step is None else int(step.item())
            return steps

        def on_before_optimizer_step(self, trainer, pl_module, optimizer):
            step = len(self.updates)
            _require(optimizer is self.optimizer and self.before is None and step < 2
                     and self.pending.get("backward_completed") is True
                     and len(self.batches) == 4 * step + 3 and trainer.global_step == step,
                     "optimizer runs only after a completed four-microbatch window")
            window = [*self.batches[-3:], self.pending]
            _require(any(row["instances"] and row["components"]["loss_mask_ce"] > 0
                         and row["components"]["loss_mask_dice"] > 0 for row in window),
                     "each optimizer window exercises actual positive mask losses")
            grads = [p.grad.detach() for group in optimizer.param_groups for p in group["params"] if p.grad is not None]
            _require(bool(grads), "native backward produced gradients")
            grad_norm = torch.stack([torch.linalg.vector_norm(grad) for grad in grads]).norm().item()
            probe_norms = {name: torch.linalg.vector_norm(p.grad.detach()).item() for name, p in self.params.items()}
            _require(all(math.isfinite(norm) and norm > 0 for norm in (grad_norm, *probe_norms.values())),
                     "finite nonzero pre-clip optimized and witness gradients")
            adam_steps = self._adam_steps()
            _require(all(value == step for value in adam_steps.values()), "native AdamW witness counters before step")
            rates = [float(group["lr"]) for group in optimizer.param_groups]
            _require(all(math.isclose(lr, base * step / WARMUP_STEPS, rel_tol=1e-9, abs_tol=1e-15)
                         for lr, base in zip(rates, self.scheduler.base_lrs))
                     and (step == 0 or any(lr > 0 for lr in rates)), "native warmup learning rates used by AdamW")
            self.before = {"lr_used": rates, "gradient_norm_before_clip": grad_norm,
                           "gradient_tensors": len(grads), "witness_gradient_norms": probe_norms,
                           "adam_steps_before": adam_steps,
                           "parameters": {name: p.detach().clone() for name, p in self.params.items()}}

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            _require(self.pending.get("backward_completed") is True, "completed backward before batch end")
            self.batches.append(self.pending)
            self.pending = None
            step = len(self.batches) // 4
            _require(trainer.global_step == step, "native global step matches the accumulation windows")
            if len(self.batches) % 4:
                _require(self.before is None, "no intermediate optimizer update")
                return
            _require(self.before is not None and self.scheduler.last_epoch == step,
                     "optimizer and scheduler updates complete before observation")
            previous = self.before.pop("parameters")
            deltas = {name: (p.detach() - previous[name]).abs().max().item() for name, p in self.params.items()}
            _require(all(math.isfinite(delta) and (delta == 0 if step == 1 else delta > 0)
                         for delta in deltas.values()), "zero-LR first step and actual FP32 witness movement on second")
            adam_steps = self._adam_steps()
            _require(all(value == step for value in adam_steps.values()), "native AdamW witness counters after step")
            average = ema._average_model
            _require(average is not None and int(average.n_averaged.item()) == ema._latest_update_step == step,
                     "actual native EMA update count")
            self.updates.append({**self.before, "global_step": step, "adam_steps_after": adam_steps,
                                 "max_abs_parameter_change": deltas, "ema_updates": step,
                                 "lr_after_scheduler": self.scheduler.get_last_lr()})
            self.before = None
            if step == 2:
                trainer.should_stop = True

    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    seed_everything(5201, workers=True)
    module = RFDETRModelModule(mc, tc, load_encoder_weights=False)
    _require(mc.num_classes == 1 and tuple(module.model.class_embed.weight.shape)[0] == 2,
             "native transfer retained the configured two-slot classifier")
    model_seconds = time.monotonic() - started
    data = RFDETRDataModule(mc, tc)
    observe = Observe()
    prior_matmul = torch.get_float32_matmul_precision()
    prior_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        trainer = build_trainer(tc, mc, accelerator="gpu", devices=1, include_training_callbacks=False,
                                extra_callbacks=[ema, observe], limit_val_batches=0, num_sanity_val_steps=0,
                                min_epochs=0, min_steps=0, enable_model_summary=False)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cudnn.allow_tf32 = False
        trainer.fit(module, datamodule=data)
        _require(trainer.global_step == 2 and trainer.should_stop and len(observe.batches) == 8
                 and len(observe.updates) == 2 and observe.pending is observe.before is None,
                 "exactly eight completed microbatches and two native optimizer updates")
        _require(len(data._dataset_train) == 741 and len(data._dataset_val) == 187, "admitted FIT and DEV sizes")
        torch.cuda.synchronize()
        return dict(status="NATIVE_TWO_UPDATE_FP32_COMPLETE_PENDING_ROOT", qualified=False,
                    epochs_planned=100, train_loader_batches=744, optimizer_steps_planned=PLANNED_STEPS,
                    warmup_steps=WARMUP_STEPS, precision=trainer.precision,
                    float32_matmul_precision=torch.get_float32_matmul_precision(),
                    cudnn_tf32=torch.backends.cudnn.allow_tf32, gradient_clip=trainer.gradient_clip_val,
                    seed=5201, model_setup_seconds=model_seconds,
                    elapsed_seconds=time.monotonic() - started, cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(), microbatches=observe.batches,
                    updates=observe.updates, validation=False, checkpoints_written=False)
    finally:
        if observe.handle is not None:
            observe.handle.remove()
        torch.set_float32_matmul_precision(prior_matmul)
        torch.backends.cudnn.allow_tf32 = prior_cudnn_tf32
