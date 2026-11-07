"""Run the admitted native RF-DETR BF16 E100 recipe through one full epoch."""
import hashlib
from importlib.metadata import version
import math
import os
from pathlib import Path
import time


EPOCH_BATCHES = 744
ACCUM = 4
EPOCH_STEPS = EPOCH_BATCHES // ACCUM
PLANNED_STEPS = 18_600
WARMUP_STEPS = 558
TERMS = dict(loss_ce=1.0, loss_bbox=5.0, loss_giou=2.0, loss_mask_ce=5.0, loss_mask_dice=5.0)
WITNESSES = ('class_embed.weight', 'segmentation_head.bias')
ARCHIVE_NAME = 'checkpoint_0.ckpt'
MAX_ARCHIVE_BYTES = 1024**3


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def native_recipe(model, train, config):
    require(type(model) is config.RFDETRSegNanoConfig and model.num_classes == 1
            and model.resolution == 576 and model.num_channels == 3 and model.positional_encoding_size == 48
            and model.num_queries == model.num_select == 100 and model.group_detr == 13
            and model.mask_downsample_ratio == 4 and model.segmentation_head
            and not any((model.compile, model.cuda_graphs, model.fused_optimizer,
                         model.use_grouppose_keypoints, model.backbone_lora)),
            'Expected frozen native Seg Nano architecture')
    require(type(train) is config.SegmentationTrainConfig and train.epochs == 100 and train.seed == 5201
            and train.amp_dtype == 'bf16' and train.dataset_file == 'roboflow'
            and train.batch_size == 1 and train.grad_accum_steps == ACCUM and train.num_workers == 0
            and train.square_resize_div_64 and train.multi_scale is config.MultiScale.OFF
            and train.augmentation_backend == config.AugmentationBackend.TV and train.scale_jitter
            and train.aug_config is None and train.optimizer == 'adamw' and train.lr_scheduler == 'cosine'
            and train.warmup_epochs == 3 and train.lr_scheduler_kwargs == {'min_factor': 0.05}
            and train.lr == 0.0001 and train.lr_encoder == 0.00015 and train.weight_decay == 0.0001
            and train.clip_max_norm == 0.1 and train.mask_point_sample_ratio == 16
            and train.use_ema and train.ema_update_interval == 1 and train.checkpoint_interval == 1
            and train.progress_bar is None and train.eval_max_dets == 100
            and train.eval_backend == 'faster_coco_eval' and not train.resume
            and not any((train.early_stopping, train.save_dataset_grids, train.compute_train_metrics,
                         train.run_test, train.tensorboard, train.wandb, train.mlflow)),
            'Expected frozen native E100 training and output recipe')


def canonical(value, *, directory):
    path = Path(value)
    require(path.is_absolute() and path.resolve(strict=True) == path
            and (path.is_dir() if directory else path.is_file()),
            'Expected canonical existing admitted path')
    return path


def state_identity(value):
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


def inspect_archive(path, *, torch, boundary_api, recipe_sha256, initial_witness):
    """Reopen exactly the stock E1 archive with the CPU weights-only loader."""
    canonical(path, directory=False)
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno())
        require(0 < before.st_size <= MAX_ARCHIVE_BYTES, 'Unexpected native E1 archive size')
        sha = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            sha.update(block)
        stream.seek(0)
        payload = torch.load(stream, map_location='cpu', weights_only=True)
        boundary_api.validate_e1_checkpoint(payload, recipe_sha256=recipe_sha256)
        raw = payload['state_dict']['model.class_embed.weight']
        average = payload['callbacks']['RFDETREMACallback']['average_model_state_dict']
        ema = average['module.model.class_embed.weight']
        require(raw.dtype == ema.dtype == initial_witness.dtype == torch.float32
                and raw.shape == ema.shape == initial_witness.shape and raw.shape[0] == 2
                and bool(torch.isfinite(raw).all()) and bool(torch.isfinite(ema).all()),
                'Native archive has invalid raw/EMA classifier witnesses')
        move = (raw - initial_witness).abs().max().item()
        ema_difference = (raw - ema).abs().max().item()
        require(math.isfinite(move) and move > 0 and math.isfinite(ema_difference),
                'Missing actual E1 FP32 classifier movement')
        after = os.fstat(stream.fileno())
        require(state_identity(before) == state_identity(after) == state_identity(path.stat())
                and path.resolve(strict=True) == path, 'Native archive changed during independent readback')
    return (dict(path=path.as_posix(), bytes=before.st_size, sha256=sha.hexdigest()),
            dict(classifier_raw_max_abs_change=move, classifier_raw_ema_max_abs_difference=ema_difference))


def run_e1(checkpoint_path, model_config, train_config, *, boundary_api, recipe_sha256, event_sink=None):
    """Run once inside a separately admitted single-device worker. No validation or test loop."""
    import numpy as np
    import torch
    from pytorch_lightning import Callback, seed_everything
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.plugins.precision import MixedPrecision
    from rfdetr import config
    from rfdetr.training.callbacks.best_model import BestModelCallback
    from rfdetr.training.callbacks.ema import RFDETREMACallback
    from rfdetr.training.module_data import RFDETRDataModule
    from rfdetr.training.module_model import RFDETRModelModule
    from rfdetr.training.trainer import build_trainer

    def event(name, **values):
        if event_sink is not None:
            event_sink(dict(event=name, **values))

    require(version('rfdetr') == '1.12.0' and version('pytorch-lightning') == '2.6.6',
            'Unexpected RF-DETR or Lightning version')
    source_checkpoint = canonical(checkpoint_path, directory=False)
    model = model_config.model_copy(update={'pretrain_weights': str(source_checkpoint)})
    train = train_config
    native_recipe(model, train, config)
    dataset = canonical(train.dataset_dir, directory=True)
    output = canonical(train.output_dir, directory=True)
    require(dataset.name != 'REQUIRES_QUALIFIED_SAR_COCO_SPLITS' and output != dataset
            and dataset not in output.parents and output not in dataset.parents
            and source_checkpoint not in output.parents and output not in source_checkpoint.parents
            and not any(output.iterdir()), 'Expected fresh dedicated output separate from inputs')
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1
            and torch.cuda.is_bf16_supported(including_emulation=False), 'Expected one native BF16 CUDA device')
    capability = list(torch.cuda.get_device_capability())
    boundary = boundary_api.make_e1_boundary_callback(
        Callback, recipe_sha256=recipe_sha256,
        capture_rng=lambda: boundary_api.capture_rng_state(torch, np),
    )

    class ObserveE1(Callback):
        def __init__(self):
            self.ids = []
            self.instance_accesses = 0
            self.label_errors = []
            self.losses = []
            self.grad_norms = {name: [] for name in WITNESSES}
            self.optimizer_steps = 0
            self.handle = None
            self.initial_witness = None
            self.train_seconds = None
            self.summary = None

        def on_train_start(self, trainer, pl_module):
            require(not trainer.enable_validation and trainer.gradient_clip_val == train.clip_max_norm
                    and len(trainer.optimizers) == len(trainer.lr_scheduler_configs) == 1,
                    'Expected FIT-only native optimizer and schedule')
            self.optimizer = trainer.optimizers[0]
            lr = trainer.lr_scheduler_configs[0]
            self.scheduler = lr.scheduler
            require(type(self.optimizer) is torch.optim.AdamW and self.scheduler.last_epoch == 0
                    and type(self.scheduler) is torch.optim.lr_scheduler.LambdaLR
                    and lr.interval == 'step' and lr.frequency == 1, 'Expected native AdamW and step LambdaLR')
            anchors = ((0, 0.0), (1, 1 / WARMUP_STEPS), (WARMUP_STEPS, 1.0),
                       ((PLANNED_STEPS + WARMUP_STEPS) // 2, 0.525), (PLANNED_STEPS, 0.05))
            require(bool(self.scheduler.lr_lambdas) and all(
                math.isclose(schedule(step), factor, rel_tol=1e-9, abs_tol=1e-12)
                for schedule in self.scheduler.lr_lambdas for step, factor in anchors
            ), 'Native E100 warmup/cosine schedule differs')
            require(all(pl_module.criterion.weight_dict.get(key) == val for key, val in TERMS.items()),
                    'Native loss coefficients differ')
            self.params = {name: pl_module.model.get_parameter(name) for name in WITNESSES}
            require(all(p.dtype == torch.float32 and p.device.type == 'cuda' for p in self.params.values()),
                    'Expected FP32 CUDA training witnesses')
            self.initial_witness = self.params[WITNESSES[0]].detach().cpu().clone()
            self.handle = self.optimizer.register_step_post_hook(self.adam_returned)
            self.started = time.monotonic()
            event('native_e1_train_started', planned_updates=PLANNED_STEPS,
                  train_batches=trainer.num_training_batches, actual_precision=trainer.precision)

        def adam_returned(self, optimizer, args, kwargs):
            require(optimizer is self.optimizer, 'Unexpected AdamW post-hook source')
            self.optimizer_steps += 1

        def on_before_optimizer_step(self, trainer, pl_module, optimizer):
            require(optimizer is self.optimizer and trainer.global_step == self.optimizer_steps
                    and self.scheduler.last_epoch == self.optimizer_steps,
                    'Native optimizer and scheduler were out of sequence')
            for name, parameter in self.params.items():
                grad = parameter.grad
                require(grad is not None or name != WITNESSES[0], 'Missing classifier gradient')
                if grad is not None:
                    require(grad.dtype == torch.float32, 'Expected FP32 mixed-precision gradient')
                    self.grad_norms[name].append(torch.linalg.vector_norm(grad.detach()))

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            images, targets = batch
            require(batch_idx == len(self.ids) and tuple(images.tensors.shape) == (1, 3, 576, 576)
                    and images.tensors.dtype == torch.float32 and images.tensors.device.type == 'cuda'
                    and len(targets) == 1 and trainer.global_step == self.optimizer_steps,
                    'Unexpected native 576 FIT batch or optimizer cursor')
            target = targets[0]
            labels = target['labels']
            count = labels.numel()
            require(labels.dtype == torch.int64 and target['masks'].dtype == torch.bool
                    and tuple(target['masks'].shape) == (count, 576, 576),
                    'Unexpected original single-class FIT mask layout')
            loss = outputs['loss']
            require(loss.dtype == torch.float32 and loss.numel() == 1,
                    'Missing scalar FP32 native training loss')
            self.ids.append(target['image_id'].detach().reshape(()))
            self.label_errors.append((labels != 0).any())
            self.instance_accesses += count
            self.losses.append(loss.detach().reshape(()))
            if (batch_idx + 1) % (64 * ACCUM) == 0:
                event('native_e1_progress', completed_microbatches=batch_idx + 1,
                      completed_updates=self.optimizer_steps)

        def on_train_epoch_end(self, trainer, pl_module):
            require(len(self.ids) == EPOCH_BATCHES and self.optimizer_steps == EPOCH_STEPS
                    and trainer.global_step == self.scheduler.last_epoch == EPOCH_STEPS,
                    'Expected all E1 samples, AdamW steps and scheduler updates')
            sample_ids = torch.stack(self.ids).cpu().tolist()
            losses = torch.stack(self.losses).cpu().tolist()
            require(len(set(sample_ids)) == 741 and self.instance_accesses > 0
                    and not bool(torch.stack(self.label_errors).any().item())
                    and all(math.isfinite(value) for value in losses),
                    'E1 unique FIT count, supervision or losses are invalid')
            gradients = {}
            for name, values in self.grad_norms.items():
                require(bool(values), 'No observed gradient for a training witness')
                norms = torch.stack(values).cpu().tolist()
                require(all(math.isfinite(value) for value in norms) and max(norms) > 0,
                        'Nonfinite or entirely zero witness gradients')
                gradients[name] = dict(observations=len(norms), maximum=max(norms))
            require(gradients[WITNESSES[0]]['observations'] == EPOCH_STEPS,
                    'Classifier witness did not see all AdamW windows')
            self.train_seconds = time.monotonic() - self.started
            self.summary = dict(unique_fit_images=len(set(sample_ids)),
                                duplicate_fit_accesses=len(sample_ids) - len(set(sample_ids)),
                                augmented_instance_accesses=self.instance_accesses,
                                native_returned_loss_mean=math.fsum(losses) / len(losses),
                                native_returned_loss_min=min(losses), native_returned_loss_max=max(losses),
                                witness_gradient_norms=gradients)
            event('native_e1_epoch_end_observed', optimizer_updates=self.optimizer_steps,
                  microbatches=len(self.ids), train_seconds=self.train_seconds)

    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    seed_everything(5201, workers=True)
    event('native_e1_model_constructor_entered')
    module = RFDETRModelModule(model, train, load_encoder_weights=False)
    require(module.model.class_embed.weight.shape[0] == 2, 'Native single-class head must retain two slots')
    model_seconds = time.monotonic() - started
    data = RFDETRDataModule(model, train)
    observe = ObserveE1()
    trainer = build_trainer(train, model, accelerator='gpu', devices=1, include_training_callbacks=True,
                            extra_callbacks=[observe, boundary], limit_val_batches=0, num_sanity_val_steps=0,
                            min_epochs=0, min_steps=0, enable_model_summary=False, logger=False)
    plugin = trainer.precision_plugin
    require(trainer.precision == 'bf16-mixed' and type(plugin) is MixedPrecision
            and plugin.device == 'cuda' and trainer.scaler is None, 'Expected native scaler-free BF16 precision')
    best = [cb for cb in trainer.callbacks if type(cb) is BestModelCallback]
    archives = [cb for cb in trainer.callbacks if type(cb) is ModelCheckpoint]
    ema = [cb for cb in trainer.callbacks if type(cb) is RFDETREMACallback]
    require(len(best) == len(archives) == len(ema) == 1
            and archives[0].filename == 'checkpoint_{epoch}' and not archives[0].save_weights_only
            and Path(archives[0].dirpath) == output,
            'Expected stock full archive and EMA before FIT-only callback filtering')
    trainer.callbacks.remove(best[0])

    try:
        trainer.fit(module, datamodule=data)
        require(trainer.global_step == EPOCH_STEPS and trainer.current_epoch == 1 and trainer.should_stop
                and boundary.microbatches == EPOCH_BATCHES and boundary.archive_payloads == 1
                and observe.summary is not None and len(data._dataset_train) == 741 and len(data._dataset_val) == 187,
                'Native E1 training or stock archive hook did not close as expected')
        torch.cuda.synchronize()
        event('native_e1_fit_returned', completed_epochs=1, completed_updates=EPOCH_STEPS)
        archive, payload = inspect_archive(output / ARCHIVE_NAME, torch=torch, boundary_api=boundary_api,
                                           recipe_sha256=recipe_sha256, initial_witness=observe.initial_witness)
        return dict(status='NATIVE_E1_BF16_ARCHIVE_COMPLETE_PENDING_ROOT', qualified=False,
                    terminal_self_certified=False, recipe_sha256=recipe_sha256,
                    checkpoint_ref=archive, weights_only_cpu_readback=True, payload=payload,
                    fixed_seed=5201, initialization='native_COCO_two_slot_prefix', fit_images=741, dev_images=187,
                    completed_epochs=1, completed_microbatches=EPOCH_BATCHES, completed_updates=EPOCH_STEPS,
                    planned_epochs=100, planned_updates=PLANNED_STEPS, warmup_updates=WARMUP_STEPS,
                    precision=trainer.precision, cuda_device_capability=capability,
                    float32_matmul_precision=torch.get_float32_matmul_precision(),
                    cudnn_tf32=torch.backends.cudnn.allow_tf32, model_setup_seconds=model_seconds,
                    native_train_seconds=observe.train_seconds, elapsed_seconds=time.monotonic() - started,
                    cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    observations=observe.summary, validation=False, AP=False, TEST=False,
                    resumed=False, exact_restart_trajectory_verified=False)
    finally:
        if observe.handle is not None:
            observe.handle.remove()
