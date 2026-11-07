"""Resume the admitted native E1 archive for one BF16 accumulation window."""
import hashlib
from importlib.metadata import version
import os
from pathlib import Path
import time


E1_STEP = 186
WINDOW = 4
LOADER_BATCHES = 744


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def make_first_update_stop(callback_type, *, observer, optimizer_type):
    """Stop only after the native restore observer and a real AdamW post-hook agree."""
    class FirstUpdateStop(callback_type):
        def __init__(self):
            super().__init__()
            self.started_batches = 0
            self.completed_batches = 0
            self.adam_returns = 0
            self.handle = None

        def on_train_start(self, trainer, pl_module):
            loader = trainer.train_dataloader
            require(observer.checkpoint_seen and trainer.current_epoch == 1
                    and trainer.global_step == E1_STEP and not trainer.should_stop
                    and trainer.min_epochs == trainer.min_steps == 0 and not trainer.enable_validation
                    and loader.num_workers == 0 and len(loader) == LOADER_BATCHES,
                    'Native E2 cursor, stop settings or sized FIT loader differs')
            require(self.handle is None and len(trainer.optimizers) == 1
                    and type(trainer.optimizers[0]) is optimizer_type,
                    'Expected one restored native AdamW optimizer')
            self.optimizer = trainer.optimizers[0]
            self.handle = self.optimizer.register_step_post_hook(self.adam_returned)

        def adam_returned(self, optimizer, args, kwargs):
            require(optimizer is self.optimizer and self.started_batches == WINDOW
                    and self.completed_batches == WINDOW - 1 and self.adam_returns == 0,
                    'Unexpected AdamW return outside the single E2 accumulation window')
            self.adam_returns += 1

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            require(batch_idx == self.started_batches == self.completed_batches < WINDOW
                    and trainer.current_epoch == 1 and trainer.global_step == E1_STEP
                    and not trainer.should_stop, 'unexpected resumed batch start')
            self.started_batches += 1

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            require(batch_idx == self.completed_batches < WINDOW
                    and self.started_batches == self.completed_batches + 1,
                    'Unexpected resumed batch completion')
            final = self.started_batches == WINDOW
            require(trainer.current_epoch == 1 and trainer.global_step == E1_STEP + int(final)
                    and self.adam_returns == int(final),
                    'E2 batch completion lacks its expected actual AdamW return')
            self.completed_batches += 1
            if final:
                require(observer.first_new_update_observed,
                        'Native E2 observer did not verify restored optimizer, scheduler and EMA')
                trainer.should_stop = True

        def verify_completion(self, trainer):
            require(trainer.should_stop and trainer.global_step == E1_STEP + 1
                    and self.started_batches == self.completed_batches == WINDOW
                    and self.adam_returns == 1 and observer.first_new_update_observed,
                    'Native E2 single-window fit did not stop at its exact boundary')

        def on_save_checkpoint(self, trainer, pl_module, checkpoint):
            raise RuntimeError('E2 probe refuses checkpoint writes')

        def on_validation_start(self, trainer, pl_module):
            raise RuntimeError('E2 probe refuses validation and test loops')

        on_test_start = on_validation_start

        def close(self):
            if self.handle is not None:
                self.handle.remove()
                self.handle = None

    return FirstUpdateStop()


def verify_archive(ref, *, e1_api):
    require(type(ref) is dict and set(ref) == {'path', 'bytes', 'sha256'}
            and type(ref['path']) is str and type(ref['bytes']) is int
            and 0 < ref['bytes'] <= e1_api.MAX_ARCHIVE_BYTES,
            'Expected one admitted native E1 archive reference')
    path = e1_api.canonical(ref['path'], directory=False)
    require(path.as_posix() == ref['path'] and path.name == e1_api.ARCHIVE_NAME,
            'Expected canonical native E1 archive name')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno())
        require(before.st_size == ref['bytes'], 'Native E1 archive length differs')
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
        identity = e1_api.state_identity(before)
        require(digest.hexdigest() == ref['sha256'] and path.resolve(strict=True) == path
                and identity == e1_api.state_identity(os.fstat(stream.fileno()))
                == e1_api.state_identity(path.stat()), 'Native E1 archive hash or identity differs')
    return path, identity


def run_e2_resume(source_checkpoint_path, resume_checkpoint_ref, model_config, train_config, *,
                  e1_api, boundary_api, recipe_sha256, event_sink=None):
    """Run once in a separately admitted worker, leaving the E1 archive untouched."""
    import numpy as np
    import torch
    from pytorch_lightning import Callback, seed_everything
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.plugins.precision import MixedPrecision
    from rfdetr import config
    from rfdetr.training.callbacks.best_model import BestModelCallback
    from rfdetr.training.module_data import RFDETRDataModule
    from rfdetr.training.module_model import RFDETRModelModule
    from rfdetr.training.trainer import build_trainer

    require(version('rfdetr') == '1.12.0' and version('pytorch-lightning') == '2.6.6',
            'Unexpected RF-DETR or Lightning version')
    boundary_api.recipe_digest(recipe_sha256)
    resume_path, identity = verify_archive(resume_checkpoint_ref, e1_api=e1_api)
    source = e1_api.canonical(source_checkpoint_path, directory=False)
    model = model_config.model_copy(update={'pretrain_weights': str(source)})
    train = train_config
    e1_api.native_recipe(model, train, config)
    dataset = e1_api.canonical(train.dataset_dir, directory=True)
    output = e1_api.canonical(train.output_dir, directory=True)
    require(dataset.name != 'REQUIRES_QUALIFIED_SAR_COCO_SPLITS' and source != resume_path
            and output != dataset and dataset not in output.parents and output not in dataset.parents
            and all(output not in path.parents for path in (source, resume_path))
            and not any(output.iterdir()), 'Expected dedicated empty output separate from admitted inputs')
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1
            and torch.cuda.is_bf16_supported(including_emulation=False),
            'Expected one native BF16 CUDA device')

    start = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    seed_everything(5201, workers=True)
    module = RFDETRModelModule(model, train, load_encoder_weights=False)
    require(module.model.class_embed.weight.shape[0] == 2, 'Native single-class head must retain two slots')
    data = RFDETRDataModule(model, train)
    observer = boundary_api.make_e1_resume_observer(
        Callback, recipe_sha256=recipe_sha256, restore_rng=lambda state: boundary_api.restore_rng_state(state, torch, np),
    )
    stop = make_first_update_stop(Callback, observer=observer, optimizer_type=torch.optim.AdamW)
    trainer = build_trainer(train, model, accelerator='gpu', devices=1, include_training_callbacks=True,
                            extra_callbacks=[observer, stop], limit_val_batches=0, num_sanity_val_steps=0,
                            min_epochs=0, min_steps=0, enable_model_summary=False, logger=False)
    require(trainer.precision == 'bf16-mixed' and type(trainer.precision_plugin) is MixedPrecision
            and trainer.precision_plugin.device == 'cuda' and trainer.scaler is None,
            'Expected native scaler-free BF16 precision')
    best = [cb for cb in trainer.callbacks if type(cb) is BestModelCallback]
    archives = [cb for cb in trainer.checkpoint_callbacks if type(cb) is ModelCheckpoint]
    require(len(best) == len(archives) == 1 and archives[0].filename == 'checkpoint_{epoch}'
            and archives[0].save_weights_only is False and archives[0].save_on_exception is False
            and Path(archives[0].dirpath) == output, 'Expected the stock full-archive callback')
    trainer.callbacks.remove(best[0])
    archives[0].save_top_k = 0
    archives[0].save_last = False

    try:
        if event_sink is not None:
            event_sink(dict(event='native_e2_fit_entered', checkpoint_step=E1_STEP))
        trainer.fit(module, datamodule=data, ckpt_path=str(resume_path), weights_only=True)
        stop.verify_completion(trainer)
        require(len(data._dataset_train) == 741 and len(data._dataset_val) == 187 and not any(output.iterdir()),
                'Native E2 FIT/DEV lengths or empty output invariant differs')
        torch.cuda.synchronize()
        require(verify_archive(resume_checkpoint_ref, e1_api=e1_api) == (resume_path, identity),
                'Native E1 archive identity differs after resume')
        if event_sink is not None:
            event_sink(dict(event='native_e2_fit_returned', completed_updates=E1_STEP + 1))
        return dict(status='NATIVE_E2_BF16_FIRST_UPDATE_COMPLETE_PENDING_ROOT',
                    qualified=False, terminal_self_certified=False, recipe_sha256=recipe_sha256,
                    input_checkpoint_ref=dict(resume_checkpoint_ref), input_checkpoint_bytes_unchanged=True,
                    weights_only_native_resume=True, resumed=True, resume_epoch=1, resume_global_step=E1_STEP,
                    completed_microbatches=stop.completed_batches, completed_new_updates=stop.adam_returns,
                    completed_updates=trainer.global_step, first_new_update_observed=True,
                    planned_epochs=e1_api.PLANNED_STEPS // E1_STEP, planned_updates=e1_api.PLANNED_STEPS,
                    warmup_updates=e1_api.WARMUP_STEPS, precision=trainer.precision, fixed_seed=5201,
                    elapsed_seconds=time.monotonic() - start, cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    checkpoint_written=False, validation=False, AP=False, TEST=False,
                    exact_restart_trajectory_verified=False)
    finally:
        stop.close()
