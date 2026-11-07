"""Fresh native RF-DETR E100 recipe with fixed E24/E50/E100 archives."""
import hashlib
from importlib.metadata import version
import math
import os
from pathlib import Path
import time


EPOCH_BATCHES = 744
ACCUM = 4
EPOCH_STEPS = EPOCH_BATCHES // ACCUM
PLANNED_EPOCHS = 100
PLANNED_STEPS = PLANNED_EPOCHS * EPOCH_STEPS
WARMUP_STEPS = 3 * EPOCH_STEPS
ENDPOINTS = (24, 50, 100)
STOP_EPOCHS = (24, 100)
BOUNDARY_KEY = 'sar_seal_rf_detr_uninterrupted_endpoint_v1'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def schedule_factor(step):
    if step < WARMUP_STEPS:
        return step / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / (PLANNED_STEPS - WARMUP_STEPS)
    return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * progress))


class ArchiveEndpoints:
    """Gate the stock monitoring callback after an independently checked full epoch."""

    def __init__(self, archive, *, stop_epoch, recipe_sha256):
        require(type(stop_epoch) is int and stop_epoch in STOP_EPOCHS, 'Expected explicit stop_epoch 24 or 100')
        self.archive = archive
        self.stop_epoch = stop_epoch
        self.recipe_sha256 = recipe_sha256
        self.expected = tuple(epoch for epoch in ENDPOINTS if epoch <= stop_epoch)
        self.completed = 0
        self.armed_epoch = None
        self.saved = []
        self.archive.save_top_k = 0
        self.archive.save_last = False

    def complete_epoch(self, trainer, epoch):
        require(not trainer.should_stop and self.armed_epoch is None
                and epoch == self.completed + 1 <= self.stop_epoch, 'Unexpected completed native epoch')
        self.completed = epoch
        self.armed_epoch = epoch if epoch in self.expected else None
        self.archive.save_top_k = -1 if self.armed_epoch is not None else 0
        if epoch == self.stop_epoch < PLANNED_EPOCHS:
            trainer.should_stop = True

    def mark_archive(self, checkpoint):
        epoch = self.armed_epoch
        require(epoch is not None and BOUNDARY_KEY not in checkpoint
                and checkpoint['epoch'] == epoch - 1 and checkpoint['global_step'] == epoch * EPOCH_STEPS,
                'Unexpected or duplicate native endpoint archive')
        checkpoint[BOUNDARY_KEY] = endpoint_marker(epoch, self.recipe_sha256)
        self.saved.append(epoch)
        self.armed_epoch = None

    def verify_completion(self):
        require(self.completed == self.stop_epoch and self.armed_epoch is None
                and tuple(self.saved) == self.expected, 'Native endpoint archive sequence is incomplete')
        return self.expected


def endpoint_marker(epoch, recipe_sha256):
    return dict(recipe_sha256=recipe_sha256, completed_epochs=epoch,
                microbatches=epoch * EPOCH_BATCHES, optimizer_updates=epoch * EPOCH_STEPS)


def validate_schedule(optimizer, scheduler, step):
    require(scheduler['last_epoch'] == step, 'Native LambdaLR cursor differs')
    base = scheduler['base_lrs']
    last = scheduler['_last_lr']
    groups = optimizer['param_groups']
    require(len(base) == len(last) == len(groups) and bool(base), 'Native optimizer LR groups differ')
    factor = schedule_factor(step)
    require(all(math.isfinite(a) and a >= 0
                and math.isclose(b, a * factor, rel_tol=1e-9, abs_tol=1e-12)
                and math.isclose(c['lr'], b, rel_tol=1e-9, abs_tol=1e-12)
                for a, b, c in zip(base, last, groups)), 'Native E100 learning rate differs')


def validate_archive_progress(checkpoint, epoch, *, boundary_api, recipe_sha256):
    step = epoch * EPOCH_STEPS
    require(checkpoint['pytorch-lightning_version'] == '2.6.6'
            and checkpoint['epoch'] == epoch - 1 and checkpoint['global_step'] == step
            and checkpoint[BOUNDARY_KEY] == endpoint_marker(epoch, recipe_sha256),
            'Native endpoint archive version, cursor or recipe differs')
    fit = checkpoint['loops']['fit_loop']
    batch = fit['epoch_loop.batch_progress']
    require(batch['is_last_batch'] is True, 'Endpoint was not archived at the last batch')
    updates = fit['epoch_loop.automatic_optimization.optim_progress']['optimizer']['step']
    for kind, multiplier in (('current', 1), ('total', epoch)):
        epoch_counter = fit['epoch_progress'][kind]
        require(all(epoch_counter[key] == epoch for key in ('ready', 'started', 'processed'))
                and epoch_counter['completed'] == epoch - 1,
                'Native endpoint epoch counters differ')
        require(all(batch[kind][key] == EPOCH_BATCHES * multiplier
                    for key in ('ready', 'started', 'processed', 'completed'))
                and all(updates[kind][key] == EPOCH_STEPS * multiplier for key in ('ready', 'completed')),
                'Native endpoint batch or update counters differ')
    optimizers = checkpoint['optimizer_states']
    schedules = checkpoint['lr_schedulers']
    require(len(optimizers) == len(schedules) == 1, 'Expected one native optimizer and scheduler')
    boundary_api.adam_progress(optimizers[0], step)
    boundary_api.ema_progress(checkpoint['callbacks'], step)
    validate_schedule(optimizers[0], schedules[0], step)


def inspect_archive(path, *, epoch, torch, e1_api, boundary_api, recipe_sha256, initial_witness):
    e1_api.canonical(path, directory=False)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno())
        require(0 < before.st_size <= e1_api.MAX_ARCHIVE_BYTES, 'Native endpoint archive size differs')
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
        stream.seek(0)
        checkpoint = torch.load(stream, map_location='cpu', weights_only=True)
        validate_archive_progress(checkpoint, epoch, boundary_api=boundary_api, recipe_sha256=recipe_sha256)
        key = boundary_api.RAW_WITNESS
        raw = checkpoint['state_dict'][key]
        average = checkpoint['callbacks'][boundary_api.EMA_KEY]['average_model_state_dict']['module.' + key]
        require(raw.dtype == average.dtype == initial_witness.dtype == torch.float32
                and raw.shape == average.shape == initial_witness.shape and raw.shape[0] == 2
                and bool(torch.isfinite(raw).all()) and bool(torch.isfinite(average).all()),
                'Native endpoint has invalid raw/EMA classifier witnesses')
        movement = (raw - initial_witness).abs().max().item()
        ema_difference = (raw - average).abs().max().item()
        require(math.isfinite(movement) and movement > 0 and math.isfinite(ema_difference),
                'Native endpoint classifier movement or EMA difference is invalid')
        identity = e1_api.state_identity(before)
        require(identity == e1_api.state_identity(os.fstat(stream.fileno()))
                == e1_api.state_identity(path.stat()) and path.resolve(strict=True) == path,
                'Native endpoint archive changed during readback')
    return dict(checkpoint_ref=dict(path=path.as_posix(), bytes=before.st_size, sha256=digest.hexdigest()),
                weights_only_cpu_readback=True, classifier_raw_max_abs_change=movement,
                classifier_raw_ema_max_abs_difference=ema_difference)


def run_uninterrupted(source_checkpoint_path, model_config, train_config, *,
                      stop_epoch, e1_api, boundary_api, recipe_sha256, event_sink=None):
    """Single fresh native fit. Admission, resource closure and endpoint evaluation are external."""
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

    boundary_api.recipe_digest(recipe_sha256)
    require(type(stop_epoch) is int and stop_epoch in STOP_EPOCHS, 'Expected explicit stop_epoch 24 or 100')
    require(version('rfdetr') == '1.12.0' and version('pytorch-lightning') == '2.6.6',
            'Unexpected RF-DETR or Lightning version')
    source = e1_api.canonical(source_checkpoint_path, directory=False)
    model = model_config.model_copy(update={'pretrain_weights': str(source)})
    train = train_config
    e1_api.native_recipe(model, train, config)
    dataset = e1_api.canonical(train.dataset_dir, directory=True)
    output = e1_api.canonical(train.output_dir, directory=True)
    require(dataset.name != 'REQUIRES_QUALIFIED_SAR_COCO_SPLITS' and output != dataset
            and dataset not in output.parents and output not in dataset.parents
            and output not in source.parents and not any(output.iterdir()),
            'Expected fresh dedicated output separate from admitted inputs')
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1
            and torch.cuda.is_bf16_supported(including_emulation=False),
            'Expected one native BF16 CUDA device')

    class Observe(Callback):
        def __init__(self):
            super().__init__()
            self.handle = None
            self.updates = 0
            self.stats = []
            self.ids = []

        def on_train_start(self, trainer, pl_module):
            boundary_api.trainer_recipe(trainer, pl_module)
            require(trainer.current_epoch == trainer.global_step == 0 and not trainer.should_stop
                    and trainer.min_epochs == trainer.min_steps == 0 and not trainer.enable_validation
                    and trainer.gradient_clip_val == train.clip_max_norm
                    and len(trainer.optimizers) == len(trainer.lr_scheduler_configs) == 1,
                    'Expected fresh FIT-only native E100 trainer')
            self.optimizer = trainer.optimizers[0]
            lr = trainer.lr_scheduler_configs[0]
            self.scheduler = lr.scheduler
            require(type(self.optimizer) is torch.optim.AdamW and self.scheduler.last_epoch == 0
                    and type(self.scheduler) is torch.optim.lr_scheduler.LambdaLR
                    and lr.interval == 'step' and lr.frequency == 1,
                    'Expected native AdamW and step LambdaLR')
            anchors = (0, 1, WARMUP_STEPS, *(epoch * EPOCH_STEPS for epoch in ENDPOINTS))
            require(self.scheduler.lr_lambdas and all(
                math.isclose(schedule(step), schedule_factor(step), rel_tol=1e-9, abs_tol=1e-12)
                for schedule in self.scheduler.lr_lambdas for step in anchors
            ), 'Native E100 warmup/cosine schedule differs')
            require(all(pl_module.criterion.weight_dict.get(key) == value
                        for key, value in e1_api.TERMS.items()), 'Native loss coefficients differ')
            self.params = {name: pl_module.model.get_parameter(name) for name in e1_api.WITNESSES}
            require(all(p.dtype == torch.float32 and p.device.type == 'cuda' for p in self.params.values()),
                    'Expected FP32 CUDA training witnesses')
            self.initial = self.params[e1_api.WITNESSES[0]].detach().cpu().clone()
            self.handle = self.optimizer.register_step_post_hook(self.adam_returned)
            self.started = time.monotonic()
            event('native_uninterrupted_train_started', planned_updates=PLANNED_STEPS, stop_epoch=stop_epoch)

        def on_train_epoch_start(self, trainer, pl_module):
            require(trainer.current_epoch == gate.completed == len(self.stats) < stop_epoch
                    and gate.armed_epoch is None and not trainer.should_stop
                    and trainer.global_step == self.updates == gate.completed * EPOCH_STEPS,
                    'Native epoch started out of sequence')
            if gate.completed in gate.expected:
                require((output / f'checkpoint_{gate.completed - 1}.ckpt').is_file(),
                        'Previous native endpoint archive was not written')
            self.ids = []
            self.losses = []
            self.label_errors = []
            self.instances = 0
            self.grad_norms = {name: [] for name in self.params}

        def adam_returned(self, optimizer, args, kwargs):
            require(optimizer is self.optimizer, 'Unexpected AdamW post-hook source')
            self.updates += 1

        def on_before_optimizer_step(self, trainer, pl_module, optimizer):
            require(optimizer is self.optimizer and trainer.global_step == self.updates
                    and self.scheduler.last_epoch == self.updates, 'Native update cursors differ')
            for name, parameter in self.params.items():
                grad = parameter.grad
                require(grad is not None or name != e1_api.WITNESSES[0], 'Missing classifier gradient')
                if grad is not None:
                    require(grad.dtype == torch.float32, 'Expected FP32 mixed-precision gradient')
                    self.grad_norms[name].append(torch.linalg.vector_norm(grad.detach()))

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            images, targets = batch
            require(batch_idx == len(self.ids) < EPOCH_BATCHES
                    and trainer.global_step == self.updates == gate.completed * EPOCH_STEPS + (batch_idx + 1) // ACCUM
                    and tuple(images.tensors.shape) == (1, 3, 576, 576)
                    and images.tensors.dtype == torch.float32 and images.tensors.device.type == 'cuda'
                    and len(targets) == 1, 'Unexpected native FIT batch or update cursor')
            target = targets[0]
            labels = target['labels']
            count = labels.numel()
            require(labels.dtype == torch.int64 and target['masks'].dtype == torch.bool
                    and tuple(target['masks'].shape) == (count, 576, 576),
                    'Unexpected native single-class FIT mask layout')
            loss = outputs['loss']
            require(loss.dtype == torch.float32 and loss.numel() == 1, 'Expected scalar FP32 training loss')
            self.ids.append(target['image_id'].detach().reshape(()))
            self.label_errors.append((labels != 0).any())
            self.losses.append(loss.detach().reshape(()))
            self.instances += count

        def on_train_epoch_end(self, trainer, pl_module):
            epoch = trainer.current_epoch + 1
            step = epoch * EPOCH_STEPS
            require(len(self.ids) == EPOCH_BATCHES and trainer.global_step == self.updates == step,
                    'Native full epoch batches or AdamW updates are incomplete')
            validate_schedule(self.optimizer.state_dict(), self.scheduler.state_dict(), step)
            boundary_api.ema_progress({boundary_api.EMA_KEY: ema.state_dict()}, step)
            ids = torch.stack(self.ids).cpu().tolist()
            losses = torch.stack(self.losses).cpu().tolist()
            require(len(set(ids)) == 741 and self.instances > 0
                    and not bool(torch.stack(self.label_errors).any().item())
                    and all(math.isfinite(loss) for loss in losses),
                    'Native FIT population, supervision or losses are invalid')
            gradients = {}
            for name, values in self.grad_norms.items():
                require(bool(values), 'No observed gradient for a training witness')
                norms = torch.stack(values).cpu().tolist()
                require(all(math.isfinite(value) for value in norms) and max(norms) > 0,
                        'Nonfinite or entirely zero witness gradients')
                gradients[name] = dict(observations=len(norms), maximum=max(norms))
            require(gradients[e1_api.WITNESSES[0]]['observations'] == EPOCH_STEPS,
                    'Classifier witness did not see all native update windows')
            self.stats.append(dict(epoch=epoch, unique_fit_images=741, duplicate_fit_accesses=3,
                                   augmented_instance_accesses=self.instances,
                                   native_returned_loss_mean=math.fsum(losses) / len(losses),
                                   native_returned_loss_min=min(losses), native_returned_loss_max=max(losses),
                                   witness_gradient_norms=gradients))
            gate.complete_epoch(trainer, epoch)
            event('native_uninterrupted_epoch_observed', completed_epochs=epoch, completed_updates=step,
                  train_seconds=time.monotonic() - self.started)

        def on_save_checkpoint(self, trainer, pl_module, checkpoint):
            gate.mark_archive(checkpoint)

        def on_validation_start(self, trainer, pl_module):
            raise RuntimeError('Native FIT-only baseline refuses validation or test')

        on_test_start = on_validation_start

    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    seed_everything(5201, workers=True)
    module = RFDETRModelModule(model, train, load_encoder_weights=False)
    require(module.model.class_embed.weight.shape[0] == 2, 'Native single-class head must retain two slots')
    data = RFDETRDataModule(model, train)
    observe = Observe()
    trainer = build_trainer(train, model, accelerator='gpu', devices=1, include_training_callbacks=True,
                            extra_callbacks=[observe], limit_val_batches=0, num_sanity_val_steps=0,
                            min_epochs=0, min_steps=0, enable_model_summary=False, logger=False)
    require(trainer.precision == 'bf16-mixed' and type(trainer.precision_plugin) is MixedPrecision
            and trainer.precision_plugin.device == 'cuda' and trainer.scaler is None,
            'Expected native scaler-free BF16 precision')
    best = [cb for cb in trainer.callbacks if type(cb) is BestModelCallback]
    archives = [cb for cb in trainer.checkpoint_callbacks if type(cb) is ModelCheckpoint]
    averages = [cb for cb in trainer.callbacks if type(cb) is RFDETREMACallback]
    require(len(best) == len(archives) == len(averages) == 1
            and archives[0].filename == 'checkpoint_{epoch}' and archives[0].monitor is None
            and archives[0].save_weights_only is False and archives[0].save_on_exception is False
            and Path(archives[0].dirpath) == output, 'Expected stock full archive and EMA callbacks')
    trainer.callbacks.remove(best[0])
    ema = averages[0]
    gate = ArchiveEndpoints(archives[0], stop_epoch=stop_epoch, recipe_sha256=recipe_sha256)

    try:
        trainer.fit(module, datamodule=data)
        train_seconds = time.monotonic() - observe.started
        epochs = gate.verify_completion()
        require(trainer.current_epoch == len(observe.stats) == stop_epoch
                and trainer.global_step == observe.updates == stop_epoch * EPOCH_STEPS
                and trainer.should_stop == (stop_epoch < PLANNED_EPOCHS)
                and len(data._dataset_train) == 741 and len(data._dataset_val) == 187,
                'Native FIT did not close at the exact uninterrupted boundary')
        paths = {epoch: output / f'checkpoint_{epoch - 1}.ckpt' for epoch in epochs}
        require(set(output.iterdir()) == set(paths.values()), 'Native output has missing or unexpected files')
        torch.cuda.synchronize()
        refs = [dict(epoch=epoch, **inspect_archive(
                    path, epoch=epoch, torch=torch, e1_api=e1_api, boundary_api=boundary_api,
                    recipe_sha256=recipe_sha256, initial_witness=observe.initial))
                for epoch, path in paths.items()]
        event('native_uninterrupted_fit_returned', completed_epochs=stop_epoch, completed_updates=observe.updates)
        return dict(status='NATIVE_UNINTERRUPTED_BF16_ENDPOINTS_COMPLETE_PENDING_ROOT',
                    qualified=False, terminal_self_certified=False, recipe_sha256=recipe_sha256,
                    resumed=False, fresh_fit=True, fixed_seed=5201, initialization='native_COCO_two_slot_prefix',
                    fit_images=741, dev_images=187, planned_epochs=PLANNED_EPOCHS, planned_updates=PLANNED_STEPS,
                    warmup_updates=WARMUP_STEPS, completed_epochs=stop_epoch, completed_updates=observe.updates,
                    completed_microbatches=stop_epoch * EPOCH_BATCHES, precision=trainer.precision,
                    cuda_device_capability=list(torch.cuda.get_device_capability()),
                    float32_matmul_precision=torch.get_float32_matmul_precision(), cudnn_tf32=torch.backends.cudnn.allow_tf32,
                    endpoint_archives=refs, epoch_observations=observe.stats, native_train_seconds=train_seconds,
                    elapsed_seconds=time.monotonic() - started, cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(), validation=False, AP=False, TEST=False)
    finally:
        if observe.handle is not None:
            observe.handle.remove()
