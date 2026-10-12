"""Inert ROOT-only COCO-to-ship native-loss four-image resource preflight.

This entry is not admitted merely by existing. Scientific imports and inputs
are consumed only when ROOT later executes a pinned spec under its GPU owner.
No inference, DEV, AP, Sobel addition, retries, or random missing-key fallback.
Training model.loss intentionally receives native GT supervision, including
Cascade's native GT-proposal sampling; no GT-free training claim is made.
"""
import argparse
import copy
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import time
import traceback

SCHEMA = 'ROOT-native-coco-comparator-micro-preflight-spec-v1'
KINDS = ('mask2former', 'condinst', 'cascade_maskrcnn')
REPORT_SCHEMA = 'native-coco-comparator-micro-preflight-report-v1'
CP_SCHEMA = 'native-coco-comparator-micro-preflight-checkpoint-v1'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def reference(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b''):
            digest.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def bound(row):
    require(Path(row['path']).is_absolute(), 'Absolute bound path required')
    actual = reference(row['path'])
    require(actual == {k: row[k] for k in ('path', 'bytes', 'sha256')}, 'Pinned input differs')
    return Path(actual['path'])


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def load_source(row, name):
    path = bound(row)
    module_spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def digest_tensor(value):
    array = value.detach().cpu().contiguous().numpy()
    return dict(shape=list(array.shape), dtype=str(array.dtype),
                sha256=hashlib.sha256(array.tobytes(order='C')).hexdigest())


def state_proof(state):
    return {name: digest_tensor(value) for name, value in state.items()}


def cpu_tree(torch, value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(torch, item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(torch, item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(torch, item) for item in value)
    return value


def tree_equal(torch, left, right):
    if isinstance(left, torch.Tensor):
        return (isinstance(right, torch.Tensor) and left.shape == right.shape and
                left.dtype == right.dtype and torch.equal(left, right))
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            tree_equal(torch, left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(
            tree_equal(torch, a, b) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


def check_deadline(started, seconds):
    require(time.monotonic() - started < seconds, 'TECHNICAL_STOP: immutable owner deadline exceeded')


def prepare_cpu(torch, initializer, registry, revert, Config, classes, spec, kind, accepted):
    """Reuse qualified canonical80/boat mapping; no CUDA has been initialized."""
    entry = spec['initializers'][kind]
    require(entry['checkpoint'] == accepted['checkpoint'] and entry['config'] == accepted['official_config'] and
            accepted['strict_official80_and_ship1_compatible'] is True,
            'Actual ROOT initializer qualification/config/checkpoint mismatch')
    config = Config.fromfile(str(bound(entry['config'])))
    require(config.model.backbone.style == 'pytorch' and config.model.backbone.depth == 50,
            'Declared native PyTorch R50 required')
    closure = initializer.config_closure(bound(entry['config']))
    supplied = {str(Path(row['path']).resolve()): row for row in spec['sources']}
    require(all(supplied.get(row['path']) == row for row in closure), 'Config inheritance is not source-bound')
    checkpoint = torch.load(str(bound(entry['checkpoint'])), map_location='cpu')
    state80, prefix = initializer.canonical_state(torch, checkpoint)
    original = initializer.build_cpu(torch, registry, revert, config.model)
    strict80 = initializer.strict_load(torch, original, state80)
    ship_config, mapped, rows = initializer.ship_state(torch, kind, config.model, original, state80)
    del original, checkpoint, state80
    # These are declared common preflight overrides, not a published recipe.
    ship_config.backbone.frozen_stages = -1
    ship_config.backbone.norm_eval = True
    ship_config.backbone.norm_cfg.requires_grad = False
    ship_config.backbone.with_cp = spec['backbone_with_cp']
    preprocessor = ship_config.data_preprocessor
    preprocessor.pad_mask = True
    preprocessor.pad_size_divisor = 32
    preprocessor.batch_augments = [dict(type='BatchFixedSizePad', size=(800, 800),
        img_pad_value=0, pad_mask=True, mask_pad_value=0, pad_seg=False)]
    model = initializer.build_cpu(torch, registry, revert, ship_config)
    strict1 = initializer.strict_load(torch, model, mapped)
    initial = {name: value.detach().cpu().clone() for name, value in mapped.items()}
    proof = dict(kind=kind, checkpoint=entry['checkpoint'], official_config=entry['config'],
        official_config_closure=closure, canonical_module_prefix_removed=prefix,
        official80=strict80, ship1=strict1, allowed_changed_keys=rows,
        initial_state=state_proof(initial), initial_ship_tensors_exact=True,
        only_boat_classifier_rows_changed=True, random_missing_keys=False,
        public_recipe_equivalence_claimed=False, backbone_with_cp=spec['backbone_with_cp'],
        declared_preprocessing_overrides=dict(resize=[800, 800], keep_ratio=True,
            fixed_canvas=[800, 800], augmentation=False, mask_resize='nearest'))
    return model, initial, proof


def trainability(torch, model):
    model.requires_grad_(True).train()
    bn_types = (torch.nn.modules.batchnorm._BatchNorm,)
    frozen_names, bn_buffers, bn_modules = set(), set(), []
    for module_name, module in model.named_modules():
        if isinstance(module, bn_types):
            module.eval()
            bn_modules.append(module_name)
            for local, value in module.named_parameters(recurse=False):
                value.requires_grad_(False)
                frozen_names.add(module_name + '.' + local if module_name else local)
            for local, value in module.named_buffers(recurse=False):
                bn_buffers.add(module_name + '.' + local if module_name else local)
    named = dict(model.named_parameters())
    active = [name for name, value in named.items() if value.requires_grad]
    require(set(named) - set(active) == frozen_names and bool(active), 'Only typed BN affine may be frozen')
    conv_names = [prefix + '.' + local for prefix, module in model.backbone.named_modules()
        if isinstance(module, torch.nn.Conv2d) for local, _ in module.named_parameters(recurse=False)]
    conv_names = ['backbone.' + name for name in conv_names]
    require(bool(conv_names) and all(name in active for name in conv_names), 'All actual R50 Conv2d parameters must train')
    require(all(not module.training for module in model.modules() if isinstance(module, bn_types)), 'BN stats not frozen')
    return dict(active_names=active, frozen_names=sorted(frozen_names), BN_modules=bn_modules,
        BN_buffer_names=sorted(bn_buffers), backbone_conv_names=conv_names,
        trainable_tensors=len(active), trainable_scalars=sum(named[name].numel() for name in active),
        frozen_tensors=len(frozen_names), frozen_scalars=sum(named[name].numel() for name in frozen_names),
        state_entries=len(model.state_dict()), parameter_tensors=len(named))


def make_optimizer(torch, model, entry, policy):
    recipe = copy.deepcopy(entry['optimizer'])
    kind = recipe.pop('type')
    require(kind == ('AdamW' if policy['kind'] == 'mask2former' else 'SGD'), 'Native optimizer family required')
    allowed = {'lr', 'weight_decay', 'betas', 'eps'} if kind == 'AdamW' else {'lr', 'weight_decay', 'momentum', 'nesterov', 'dampening'}
    require(set(recipe) <= allowed and recipe['lr'] > 0 and recipe['weight_decay'] >= 0, 'Explicit native optimizer recipe required')
    if 'betas' in recipe:
        recipe['betas'] = tuple(recipe['betas'])
    options = entry['paramwise_cfg']
    require(set(options) <= {'custom_keys', 'bias_lr_mult', 'bias_decay_mult', 'norm_decay_mult'},
            'Unsupported parameter policy must be reviewed, never silently approximated')
    norms = (torch.nn.modules.batchnorm._BatchNorm, torch.nn.modules.instancenorm._InstanceNorm,
             torch.nn.GroupNorm, torch.nn.LayerNorm)
    owners = dict(model.named_modules())
    custom = options.get('custom_keys', {})
    keys = sorted(sorted(custom), key=len, reverse=True)
    groups, audit = [], []
    for name, value in model.named_parameters():
        if not value.requires_grad:
            continue
        owner_name, local = name.rsplit('.', 1) if '.' in name else ('', name)
        norm, bias = isinstance(owners[owner_name], norms), local == 'bias'
        lr, wd, selected = recipe['lr'], recipe['weight_decay'], None
        for key in keys:
            if key in name:
                require(set(custom[key]) <= {'lr_mult', 'decay_mult'}, 'Unsupported custom optimizer option')
                selected = key
                lr *= custom[key].get('lr_mult', 1.)
                wd *= custom[key].get('decay_mult', 1.)
                break
        if selected is None:
            if bias and not norm:
                lr *= options.get('bias_lr_mult', 1.)
            if norm:
                wd *= options.get('norm_decay_mult', 1.)
            elif bias:
                wd *= options.get('bias_decay_mult', 1.)
        require(math.isfinite(lr) and math.isfinite(wd) and lr > 0 and wd >= 0, 'Invalid group policy')
        groups.append(dict(params=[value], lr=lr, weight_decay=wd))
        audit.append(dict(name=name, shape=list(value.shape), lr=lr, weight_decay=wd,
            typed_norm_owner=norm, native_local_parameter_name_is_bias=bias, custom_key=selected))
    optimizer = (torch.optim.AdamW if kind == 'AdamW' else torch.optim.SGD)(groups, **recipe)
    require([row['name'] for row in audit] == policy['active_names'], 'Optimizer membership differs from all active parameters')
    return optimizer, dict(type=kind, recipe=recipe, paramwise_cfg=options, parameter_groups=audit,
        clip_grad=copy.deepcopy(entry['clip_grad']))


def loss_terms(torch, losses):
    require(isinstance(losses, dict) and bool(losses), 'Native model.loss returned no fields')
    values, reduced = {}, {}
    for name, value in losses.items():
        items = list(value) if isinstance(value, (tuple, list)) else [value]
        require(bool(items) and all(isinstance(item, torch.Tensor) for item in items), 'Unsupported native loss field: ' + name)
        require(all(bool(torch.isfinite(item).all()) for item in items), 'Nonfinite native loss field')
        reduced[name] = sum(item.mean() for item in items)
        values[name] = [float(item.detach().mean().cpu()) for item in items] if isinstance(value, (tuple, list)) else float(reduced[name].detach().cpu())
    chosen = [value for name, value in reduced.items() if 'loss' in name]
    require(bool(chosen), 'Native loss keys missing')
    total = sum(chosen)
    require(total.requires_grad and bool(torch.isfinite(total)), 'Native total is disconnected/nonfinite')
    return total, values


def rng_snapshot(torch):
    return [torch.get_rng_state().clone(), torch.cuda.get_rng_state(0).clone()]


def check_targets(torch, data, identity):
    sample = data['data_samples'][0]
    gt = sample.gt_instances
    boxes = gt.bboxes.tensor if hasattr(gt.bboxes, 'tensor') else gt.bboxes
    count = len(identity['ann_ids'])
    require(len(gt) == count and tuple(boxes.shape) == (count, 4) and
            tuple(gt.labels.shape) == (count,) and bool((gt.labels == 0).all()) and
            bool(torch.isfinite(boxes).all()) and len(sample.ignored_instances) == 0,
            'Complete actual raw GT boxes/labels must reach native loss')
    masks = gt.masks.to_tensor(dtype=torch.uint8, device='cpu')
    h, w = identity['actual_img_shape']
    require(tuple(masks.shape) == (count, 800, 800) and not bool(masks[:, h:, :].any()) and
            not bool(masks[:, :, w:].any()), 'Native BitmapMasks canvas/padding differs')
    return dict(ann_ids=identity['ann_ids'], boxes=boxes.detach().cpu().tolist(),
        bbox=digest_tensor(boxes), labels=digest_tensor(gt.labels), masks=digest_tensor(masks),
        masks_per_GT=[digest_tensor(mask) for mask in masks], labels_values=gt.labels.detach().cpu().tolist(),
        complete_all_raw_annotations=True, bboxes_are_original_annotations_not_mask_inference=True)


def qualify_model(torch, old, spec, spec_ref, kind, prepared, dataset, coco, output, started):
    model, initial, init_proof = prepared
    directory = output / kind
    directory.mkdir()
    write(directory / 'INITIALIZATION_PROOF.json', init_proof)
    model.to('cuda:0')
    policy = trainability(torch, model)
    policy['kind'] = kind
    optimizer, optimizer_audit = make_optimizer(torch, model, spec['initializers'][kind], policy)
    active = dict(model.named_parameters())
    activity = {name: 0 for name in policy['active_names']}
    last_none = {name: False for name in policy['active_names']}
    traces, consumed_ann_ids, updates = [], [], 0
    optimizer.zero_grad(set_to_none=True)
    try:
        for micro, image_id in enumerate(spec['image_ids']):
            check_deadline(started, spec['deadline_seconds'])
            index = next(i for i in range(len(dataset)) if int(dataset.get_data_info(i)['img_id']) == image_id)
            data, stripped_samples, gray800, raw_gray, identity = old.load_image(model, dataset, coco, dataset.get_data_info(index))
            del stripped_samples, gray800, raw_gray
            identity['dataset_index'] = index
            target = check_targets(torch, data, identity)
            external = rng_snapshot(torch)
            seed = spec['loss_seed_base'] + micro
            with torch.random.fork_rng(devices=[0]):
                torch.random.default_generator.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                private_before = rng_snapshot(torch)
                # Exact native detector supervision: no extracted/fabricated loss.
                losses = model.loss(data['inputs'], data['data_samples'])
                total, scalars = loss_terms(torch, losses)
                (total / 2).backward()
                private_after = rng_snapshot(torch)
            external_after = rng_snapshot(torch)
            require(all(torch.equal(a, b) for a, b in zip(external, external_after)), 'External RNG restoration failed')
            norms, grad_names, none = [], [], []
            for name in policy['active_names']:
                gradient = active[name].grad
                last_none[name] = gradient is None
                if gradient is None:
                    none.append(name)
                else:
                    require(bool(torch.isfinite(gradient).all()), 'Nonfinite actual gradient: ' + name)
                    grad_names.append(name)
                    norms.append(gradient.detach().float().norm())
            magnitudes = torch.stack(norms).cpu().tolist() if norms else []
            norm_map = dict(zip(grad_names, magnitudes))
            gradient_rows = {name: dict(is_None=name in none, finite=True if name in norm_map else None,
                L2=norm_map.get(name)) for name in policy['active_names']}
            positive = [name for name, norm in zip(grad_names, magnitudes) if norm > 0]
            for name in positive:
                activity[name] += 1
            require(all(active[name].grad is None for name in policy['frozen_names']), 'Frozen BN affine received gradient')
            stepped, preclip, clipping_applied = False, None, False
            if (micro + 1) % 2 == 0:
                settings = spec['initializers'][kind]['clip_grad']
                if settings is None:
                    require(kind != 'mask2former', 'Native M2F requires its positive finite clipping policy')
                    # Observe the accumulated global norm without touching gradients.
                    norm = torch.stack(norms).norm() if norms else total.detach().new_zeros(())
                else:
                    require(set(settings) == {'max_norm', 'norm_type'} and
                            math.isfinite(settings['max_norm']) and settings['max_norm'] > 0 and settings['norm_type'] == 2,
                            'Explicit finite global clip policy required')
                    norm = torch.nn.utils.clip_grad_norm_([active[name] for name in policy['active_names']], **settings)
                    clipping_applied = True
                require(bool(torch.isfinite(norm)), 'Nonfinite global preclip norm')
                preclip = float(norm.detach().cpu())
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
                stepped = True
            consumed_ann_ids.extend(identity['ann_ids'])
            row = dict(kind=kind, microstep=micro, image_id=image_id, identity=identity, GT=target,
                input=digest_tensor(data['inputs']), loss_fields=scalars, native_total=float(total.detach().cpu()),
                loss_is_native_model_loss=True, Sobel_added=False, accumulation_divisor=2,
                optimizer_step=stepped, optimizer_updates=updates, preclip_norm=preclip,
                gradient_clipping_applied=clipping_applied,
                accumulated_preclip_gradients=True, positive_gradient_names=positive,
                None_gradient_names=none, finite_gradient_tensors=len(grad_names),
                gradient_rows=gradient_rows, frozen_BN_parameter_grads_None=True,
                BN_module_training={name: module.training for name, module in model.named_modules()
                    if name in policy['BN_modules']},
                private_seed=seed, private_RNG_before=[digest_tensor(x) for x in private_before],
                private_RNG_after=[digest_tensor(x) for x in private_after],
                external_RNG_before=[digest_tensor(x) for x in external],
                external_RNG_after=[digest_tensor(x) for x in external_after], external_RNG_restored=True,
                actual_input_shape=list(data['inputs'].shape), BN_stats_and_affine_frozen=True)
            traces.append(row)
            write(directory / ('MICRO_%02d.json' % micro), row)
            print(json.dumps(dict(kind=kind, microsteps=micro + 1, updates=updates, status='TECHNICAL_PROGRESS')), flush=True)
            del data, losses, total, external, external_after, private_before, private_after, norms
            if grad_names:
                del gradient
            if stepped:
                del norm
        require(updates == 2 and len(traces) == 4, 'Exactly four images/two updates required')
        final = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        require(final.keys() == initial.keys() and all(bool(torch.isfinite(v).all()) for v in final.values()), 'Final native state nonfinite/schema changed')
        frozen_state = policy['frozen_names'] + policy['BN_buffer_names']
        require(all(torch.equal(initial[name], final[name]) for name in frozen_state), 'BN affine/persistent statistics changed')
        changed = {name: not torch.equal(initial[name], value) for name, value in final.items()}
        require(any(changed[name] for name in policy['active_names']), 'No actual active parameter update')
        meta = dict(schema=CP_SCHEMA, kind=kind, phase=spec['phase'], spec=spec_ref, source=reference(__file__),
            initial_checkpoint=spec['initializers'][kind]['checkpoint'], initializer_source=spec['initializer_source'],
            image_ids=spec['image_ids'], consumed_ann_ids=consumed_ann_ids, microsteps=4, optimizer_updates=2,
            batch_size=1, accumulation=2, full_R50_conv_trainable=True, BN_stats_and_affine_frozen=True,
            optimizer=optimizer_audit, optimizer_param_names=[[row['name']] for row in optimizer_audit['parameter_groups']],
            trainable=policy, native_loss_only=True, not_formal_training=True)
        checkpoint = dict(state_dict=final, optimizer_state_dict=cpu_tree(torch, optimizer.state_dict()),
            external_rng_state=cpu_tree(torch, rng_snapshot(torch)), GT_trace_counters=dict(
                image_ids=spec['image_ids'], microsteps=4, optimizer_updates=2, ann_ids=consumed_ann_ids), meta=meta)
        cp_path = directory / 'technical_final.pth'
        with cp_path.open('xb') as stream:
            torch.save(checkpoint, stream)
        readback = torch.load(cp_path, map_location='cpu')
        require(tree_equal(torch, checkpoint, readback), 'Entire native/optimizer/RNG CPU readback differs')
        proof = dict(schema='native-coco-comparator-micro-state-proof-v1', kind=kind,
            before=state_proof(initial), after=state_proof(final), changed=changed, policy=policy,
            positive_gradient_microcounts=activity, last_preclip_gradient_None=last_none,
            gradient_activity_is_accumulated=True, frozen_BN_all_exact=True,
            all_final_state_finite=True, optimizer=optimizer_audit,
            CPU_checkpoint_readback_exact=True, changed_active_tensors=sum(changed[n] for n in policy['active_names']),
            changed_with_positive_activity=sum(changed[n] and activity[n] > 0 for n in policy['active_names']))
        write(directory / 'MODEL_STATE_PROOF.json', proof)
        result = dict(kind=kind, microsteps=4, optimizer_updates=2, image_ids=spec['image_ids'],
            consumed_ann_ids=consumed_ann_ids, checkpoint=reference(cp_path),
            initialization_proof=reference(directory / 'INITIALIZATION_PROOF.json'),
            state_proof=reference(directory / 'MODEL_STATE_PROOF.json'),
            trace=[reference(directory / ('MICRO_%02d.json' % i)) for i in range(4)],
            native_loss_only=True, inference_executed=False, AP_computed=False,
            no_GT_inference_route_added=True, GT_free_inference_runtime_verified=False,
            CPU_checkpoint_readback_exact=True, trainable_tensors=policy['trainable_tensors'],
            clip_grad=copy.deepcopy(spec['initializers'][kind]['clip_grad']))
        return result
    finally:
        # Parent may still own this object; return it to CPU before its frame dies.
        optimizer.zero_grad(set_to_none=True)
        model.to('cpu')


def clear_gpu(torch):
    before = torch.cuda.memory_allocated(0)
    gc.collect()
    torch.cuda.synchronize(0)
    torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache()
    after = torch.cuda.memory_allocated(0)
    require(after == 0, 'Live GPU aliases survived the model function frame')
    return dict(allocated_before=before, allocated_after=after)


def run(spec_file):
    started = time.monotonic()
    require(Path(spec_file).is_absolute(), 'Absolute --spec required')
    spec_path = Path(spec_file).resolve(strict=True)
    spec = json.loads(spec_path.read_text(encoding='utf-8-sig'))
    require(spec['schema'] == SCHEMA and tuple(spec['models']) == KINDS and
            len(spec['image_ids']) == 4 and len(set(spec['image_ids'])) == 4 and
            all(type(i) is int for i in spec['image_ids']) and type(spec['backbone_with_cp']) is bool and
            spec['loss_seed_base'] == 1000000 and spec['deadline_seconds'] > 0,
            'Exact three-model/four-distinct-image native preflight contract required')
    require(Path(spec['output_directory']).is_absolute(), 'Absolute fresh output required')
    output = Path(spec['output_directory'])
    require(not output.exists(), 'Exclusive fresh output required; no resume/retry')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '0', 'ROOT owner must expose only CUDA0')
    require(spec['CPU_affinity'] == [0, 1] and spec['allocator_limit_MiB'] == 4096 and
            spec['whole_GPU_owner_limit_MiB'] == 8192, 'ROOT CPU01/4GiB allocator/8GiB owner guard required')
    for row in spec['sources']:
        bound(row)
    qualification = json.loads(bound(spec['initializer_qualification']).read_text(encoding='utf-8-sig'))
    require(qualification['schema'] == 'ROOT-native-coco-comparator-initializer-CPU-inspection-v1' and
            qualification['all_initializers_strictly_compatible'] is True and
            qualification['source'] == spec['initializer_source'], 'ROOT exact initializer qualification required')
    output.mkdir(parents=True)
    write(output / 'START.json', dict(schema='native-coco-comparator-micro-preflight-start-v1',
        phase=spec['phase'], spec=reference(spec_path), source=reference(__file__), models=list(KINDS),
        image_ids=spec['image_ids'], four_microsteps_two_updates=True,
        formal_training=False, no_inference=True, no_DEV=True, no_AP=True, no_retries=True))
    # No scientific import or input was performed by the source author.
    import psutil
    import torch
    from mmengine import Config
    from mmengine.model import revert_sync_batchnorm
    from mmengine.registry import init_default_scope
    from mmdet.registry import MODELS
    from mmdet.utils import register_all_modules
    from mmdet.datasets import CocoDataset
    from mmdet.datasets.api_wrappers import COCO
    require(psutil.Process().cpu_affinity() == [0, 1] and not torch.cuda.is_initialized(), 'ROOT CPU affinity / cold CUDA mismatch')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    register_all_modules(init_default_scope=False)
    init_default_scope('mmdet')
    initializer = load_source(spec['initializer_source'], 'comparator_initializer_v2')
    old = load_source(spec['native_helpers'], 'comparator_native_helper')
    old.torch = torch
    classes = tuple(CocoDataset.METAINFO['classes'])
    require(len(classes) == 80 and classes[8] == 'boat', 'Actual official COCO80 class order mismatch')
    prepared = {kind: prepare_cpu(torch, initializer, MODELS, revert_sync_batchnorm, Config, classes,
        spec, kind, qualification['initializers'][kind]) for kind in KINDS}
    fit_path = bound(spec['FIT'])
    dataset = CocoDataset(ann_file=str(fit_path), data_prefix=dict(img=spec['image_root']),
        metainfo=dict(classes=('ship',)), filter_cfg=None, pipeline=[], test_mode=True, lazy_init=False)
    coco = COCO(str(fit_path))
    require(len(dataset) == len(coco.imgs) == 741 and len(coco.anns) == 1661 and
            set(spec['image_ids']) <= set(coco.imgs), 'Actual complete741/1661 FIT/four fixed IDs mismatch')
    torch.cuda.init()
    capacity = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1., 4096 * 1024 ** 2 / capacity), 0)
    torch.cuda.reset_peak_memory_stats(0)
    results, cleanups, error_record = {}, {}, None
    try:
        for kind in KINDS:
            check_deadline(started, spec['deadline_seconds'])
            item = prepared.pop(kind)
            results[kind] = qualify_model(torch, old, spec, reference(spec_path), kind, item, dataset, coco, output, started)
            del item
            cleanups[kind] = clear_gpu(torch)
        require(all(results[k]['image_ids'] == spec['image_ids'] for k in KINDS), 'Four-image order differs')
        require(all(results[k]['consumed_ann_ids'] == results[KINDS[0]]['consumed_ann_ids'] for k in KINDS), 'Actual raw-GT population differs across models')
        for row in spec['sources']:
            bound(row)
        for kind in KINDS:
            bound(spec['initializers'][kind]['checkpoint'])
        check_deadline(started, spec['deadline_seconds'])
    except Exception as error:
        error_record = dict(type=type(error).__name__, message=str(error),
            OOM_technical_stop=isinstance(error, torch.cuda.OutOfMemoryError), no_fallback=True, no_retry=True)
        traceback.clear_frames(error.__traceback__)
        if 'item' in locals():
            item[0].to('cpu')
            del item
        prepared.clear()
        gc.collect()
        try:
            cleanups['error_path'] = clear_gpu(torch)
        except Exception as cleanup_error:
            error_record['cleanup_error'] = str(cleanup_error)
    peak = torch.cuda.max_memory_allocated(0)
    report = dict(schema=REPORT_SCHEMA, status=('TECHNICAL_NATIVE_LOSS_MICRO_PREFLIGHT_COMPLETE_NO_AP' if error_record is None else
        'TECHNICAL_STOP_NO_CHECKPOINT_FALLBACK_NO_METHOD_RESULT'), phase=spec['phase'],
        spec=reference(spec_path), source=reference(__file__), initializer_source=spec['initializer_source'],
        initializer_qualification=spec['initializer_qualification'], native_helpers=spec['native_helpers'],
        models=list(KINDS), model_reports=results, image_ids=spec['image_ids'], microsteps_per_model=4,
        optimizer_updates_per_model=2, accumulation=2, all_raw_GT_retained=error_record is None,
        error=error_record, cleanup=cleanups, torch_peak_allocated_bytes=peak,
        elapsed_seconds=time.monotonic() - started, no_inference=True, no_DEV=True, no_AP=True,
        scientific_method_decision_computed=False, published_recipe_reproduction_claimed=False,
        warm610_adapter_comparison_track=False, GT_free_inference_runtime_verified=False)
    write(output / 'REPORT.json', report)
    if error_record is not None:
        raise RuntimeError('TECHNICAL_STOP: ' + error_record['type'] + ': ' + error_record['message'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', required=True)
    run(parser.parse_args().spec)


if __name__ == '__main__':
    main()
