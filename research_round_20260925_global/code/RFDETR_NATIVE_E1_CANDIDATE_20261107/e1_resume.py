"""RF-DETR 1.12 / Lightning 2.6.6 E1 checkpoint and continuation candidate.

The native ModelCheckpoint owns the file write. Callers own exact checkpoint
bytes, source/data admission, device setup, and terminal/process observations.
Importing this module only imports the standard library.
"""
import math
import random
import re


BOUNDARY_KEY = 'sar_seal_rf_detr_e1_boundary_v1'
EMA_KEY = 'RFDETREMACallback'
RAW_WITNESS = 'model.class_embed.weight'
BATCHES = 744
ACCUM = 4
STEPS = BATCHES // ACCUM
MAX_EPOCHS = 100
TOTAL_STEPS = STEPS * MAX_EPOCHS
WARMUP = 3 * STEPS
RNG_KEYS = {'python', 'numpy', 'torch_cpu', 'torch_cuda'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, expected):
    return type(value) is int and value == expected


def scalar(value):
    return value.item() if callable(getattr(value, 'item', None)) else value


def recipe_digest(value):
    require(type(value) is str and re.fullmatch(r'[0-9a-f]{64}', value),
            'Expected exact effective-recipe SHA256')


def rng_envelope(state):
    require(type(state) is dict and set(state) == RNG_KEYS and
            type(state['torch_cuda']) is tuple and len(state['torch_cuda']) == 1,
            'Expected Python, NumPy, Torch CPU and one CUDA RNG state')
    return state


def capture_rng_state(torch, numpy):
    """Weights-only-compatible container for the stock global RNG sources."""
    kind, keys, pos, has_gauss, cached = numpy.random.get_state()
    return rng_envelope(dict(python=random.getstate(), numpy=(kind, keys.tolist(), pos, has_gauss, cached),
                             torch_cpu=torch.get_rng_state(), torch_cuda=tuple(torch.cuda.get_rng_state_all())))


def restore_rng_state(state, torch, numpy):
    """Explicit handoff; actual restart ordering and trajectory require a real check."""
    state = rng_envelope(state)
    kind, keys, pos, has_gauss, cached = state['numpy']
    random.setstate(state['python'])
    numpy.random.set_state((kind, numpy.asarray(keys, dtype=numpy.uint32), pos, has_gauss, cached))
    torch.set_rng_state(state['torch_cpu'])
    torch.cuda.set_rng_state_all(list(state['torch_cuda']))


def ema_progress(callbacks, step):
    ema = callbacks[EMA_KEY]
    average = ema['average_model_state_dict']
    require(integer(ema['latest_update_step'], step) and
            integer(scalar(average['n_averaged']), step) and 'module.' + RAW_WITNESS in average,
            'Native EMA did not reach the checkpoint optimizer cursor')


def adam_progress(optimizer, step):
    state = optimizer['state']
    require(isinstance(state, dict) and bool(state), 'Missing AdamW moment state')
    observed = []
    for param in state.values():
        require({'step', 'exp_avg', 'exp_avg_sq'} <= param.keys(), 'Incomplete AdamW parameter state')
        counter = scalar(param['step'])
        require(type(counter) in (int, float) and 0 < counter <= step and counter == int(counter),
                'Invalid AdamW parameter update counter')
        observed.append(counter)
    require(max(observed) == step, 'AdamW state did not reach the checkpoint optimizer cursor')


def warmup_lr(optimizer, scheduler, step):
    require(integer(scheduler['last_epoch'], step), 'LambdaLR cursor differs from optimizer cursor')
    base_lrs = scheduler['base_lrs']
    actual = scheduler['_last_lr']
    group_lrs = [group['lr'] for group in optimizer['param_groups']]
    require(len(base_lrs) == len(actual) == len(group_lrs) and bool(base_lrs),
            'Scheduler and optimizer parameter groups differ')
    for base, last, group in zip(base_lrs, actual, group_lrs):
        require(all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in (base, last, group))
                and math.isclose(last, base * step / WARMUP, rel_tol=1e-9, abs_tol=1e-12)
                and math.isclose(last, group, rel_tol=1e-9, abs_tol=1e-12),
                'Optimizer LR is not at the native E100 warmup cursor')


def loop_boundary(checkpoint):
    fit = checkpoint['loops']['fit_loop']
    epoch = fit['epoch_progress']
    batch = fit['epoch_loop.batch_progress']
    for kind in ('current', 'total'):
        require(all(integer(epoch[kind][k], v) for k, v in
                    (('ready', 1), ('started', 1), ('processed', 1), ('completed', 0))),
                'Not the native processed E1 archive boundary')
        require(all(integer(batch[kind][k], BATCHES) for k in ('ready', 'started', 'processed', 'completed')),
                'Checkpoint did not complete all E1 batches')
    require(batch['is_last_batch'] is True, 'Checkpoint is not on the final E1 batch')
    step = fit['epoch_loop.automatic_optimization.optim_progress']['optimizer']['step']
    require(all(integer(step[kind][key], STEPS) for kind in ('current', 'total')
                for key in ('ready', 'completed')), 'Native optimizer loop cursor differs')


def validate_e1_checkpoint(checkpoint, *, recipe_sha256):
    """Validate a trusted, already loaded native full checkpoint. No loading or I/O."""
    recipe_digest(recipe_sha256)
    require(isinstance(checkpoint, dict), 'Expected checkpoint dictionary')
    try:
        require(checkpoint['pytorch-lightning_version'] == '2.6.6'
                and integer(checkpoint['epoch'], 0) and integer(checkpoint['global_step'], STEPS),
                'Wrong Lightning version or E1 archive cursor')
        boundary = checkpoint[BOUNDARY_KEY]
        require(type(boundary) is dict and set(boundary) == {'recipe_sha256', 'microbatches', 'rng_state'}
                and boundary['recipe_sha256'] == recipe_sha256
                and integer(boundary['microbatches'], BATCHES), 'Wrong effective recipe or E1 boundary marker')
        rng_envelope(boundary['rng_state'])
        loop_boundary(checkpoint)
        require(RAW_WITNESS in checkpoint['state_dict'], 'Raw native model weights missing')
        ema_progress(checkpoint['callbacks'], STEPS)
        optimizers, schedulers = checkpoint['optimizer_states'], checkpoint['lr_schedulers']
        require(len(optimizers) == len(schedulers) == 1, 'Expected one full optimizer and scheduler')
        adam_progress(optimizers[0], STEPS)
        warmup_lr(optimizers[0], schedulers[0], STEPS)
    except (KeyError, IndexError, TypeError, AttributeError) as error:
        raise ValueError('Incomplete native E1 continuation checkpoint') from error
    return boundary


def trainer_recipe(trainer, module):
    require(module.automatic_optimization is True and trainer.precision == 'bf16-mixed'
            and trainer.scaler is None and trainer.enable_checkpointing is True
            and all(integer(getattr(trainer, k), v) for k, v in (
                ('world_size', 1), ('max_epochs', MAX_EPOCHS), ('max_steps', -1),
                ('num_training_batches', BATCHES), ('estimated_stepping_batches', TOTAL_STEPS),
                ('accumulate_grad_batches', ACCUM))),
            'Expected single-device native BF16 E100 schedule with full checkpointing')


def make_e1_boundary_callback(callback_type, *, recipe_sha256, capture_rng):
    """Pass pytorch_lightning.Callback and an explicit four-RNG capture callable."""
    recipe_digest(recipe_sha256)

    class E1Boundary(callback_type):
        def __init__(self):
            super().__init__()
            self.microbatches = 0
            self.archive_payloads = 0
            self.armed = False

        def on_train_start(self, trainer, pl_module):
            trainer_recipe(trainer, pl_module)
            require(integer(trainer.current_epoch, 0) and integer(trainer.global_step, 0)
                    and trainer.min_epochs in (None, 0, 1)
                    and (trainer.min_steps is None or 0 <= trainer.min_steps <= STEPS)
                    and not trainer.should_stop, 'Expected a fresh E1 prefix')

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            require(not self.armed and integer(trainer.current_epoch, 0)
                    and self.microbatches < BATCHES and integer(batch_idx, self.microbatches)
                    and integer(trainer.global_step, (self.microbatches + 1) // ACCUM),
                    'E1 batch or optimizer progress is not the fixed full prefix')
            self.microbatches += 1

        def on_train_epoch_end(self, trainer, pl_module):
            require(not self.armed and self.microbatches == BATCHES
                    and integer(trainer.current_epoch, 0) and integer(trainer.global_step, STEPS),
                    'Refusing stop before the complete E1 boundary')
            self.armed = True
            trainer.should_stop = True

        def on_save_checkpoint(self, trainer, pl_module, checkpoint):
            if not self.armed:
                return
            require(BOUNDARY_KEY not in checkpoint and self.archive_payloads == 0,
                    'Expected one stock E1 archive payload')
            checkpoint[BOUNDARY_KEY] = dict(recipe_sha256=recipe_sha256, microbatches=self.microbatches,
                                            rng_state=rng_envelope(capture_rng()))
            validate_e1_checkpoint(checkpoint, recipe_sha256=recipe_sha256)
            self.archive_payloads += 1

        def on_train_end(self, trainer, pl_module):
            require(self.armed and self.archive_payloads == 1
                    and integer(trainer.current_epoch, 1) and integer(trainer.global_step, STEPS),
                    'Native E1 stop and archive payload were not observed')
            self.armed = False

    return E1Boundary()


def make_e1_resume_observer(callback_type, *, recipe_sha256, restore_rng):
    """Observe the first new accumulation window when Root explicitly resumes fit."""
    recipe_digest(recipe_sha256)

    class E1ResumeObserver(callback_type):
        def __init__(self):
            super().__init__()
            self.checkpoint_seen = False
            self.microbatches = 0
            self.first_new_update_observed = False

        def on_load_checkpoint(self, trainer, pl_module, checkpoint):
            require(not self.checkpoint_seen, 'Multiple E1 checkpoint load hooks')
            boundary = validate_e1_checkpoint(checkpoint, recipe_sha256=recipe_sha256)
            restore_rng(boundary['rng_state'])
            self.checkpoint_seen = True

        def on_train_start(self, trainer, pl_module):
            trainer_recipe(trainer, pl_module)
            require(self.checkpoint_seen and integer(trainer.current_epoch, 1)
                    and integer(trainer.global_step, STEPS), 'Native resume missed the E2 starting cursor')
            self.check_optimizer_and_ema(trainer, STEPS)

        def check_optimizer_and_ema(self, trainer, step):
            require(len(trainer.optimizers) == len(trainer.lr_scheduler_configs) == 1,
                    'Expected one restored optimizer and scheduler')
            lr = trainer.lr_scheduler_configs[0]
            require(lr.interval == 'step' and integer(lr.frequency, 1), 'Expected native per-update schedule')
            optimizer = trainer.optimizers[0].state_dict()
            adam_progress(optimizer, step)
            warmup_lr(optimizer, lr.scheduler.state_dict(), step)
            ema = [cb for cb in trainer.callbacks if cb.state_key == EMA_KEY]
            require(len(ema) == 1, 'Missing or duplicate native EMA callback')
            ema_progress({EMA_KEY: ema[0].state_dict()}, step)

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            if self.first_new_update_observed:
                return
            next_count = self.microbatches + 1
            require(self.checkpoint_seen and integer(trainer.current_epoch, 1)
                    and integer(batch_idx, self.microbatches)
                    and integer(trainer.global_step, STEPS + next_count // ACCUM),
                    'Native resumed batch or update cursor differs')
            self.microbatches = next_count
            if next_count == ACCUM:
                self.check_optimizer_and_ema(trainer, STEPS + 1)
                self.first_new_update_observed = True

    return E1ResumeObserver()
