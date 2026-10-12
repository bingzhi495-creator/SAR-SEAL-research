"""Inert, manual ROOT CPU consumer of the three-model COCO micro preflight.

No scientific module is imported at file import. The actual owned terminal
gate precedes all payload access and payload hashing. This consumer does not
run inference, CUDA loss, AP, a scientific screen, or a training trajectory.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path

KINDS = ('mask2former', 'condinst', 'cascade_maskrcnn')
SPEC_SCHEMA = 'ROOT-native-coco-comparator-micro-preflight-spec-v1'
REPORT_SCHEMA = 'native-coco-comparator-micro-preflight-report-v1'
CP_SCHEMA = 'native-coco-comparator-micro-preflight-checkpoint-v1'
DRIVER_BYTES = 30547
DRIVER_SHA = '7f7e5ce0eea8f0dae6e7fc01a4b54baafdc52fb3339b039ae50ef26b60870201'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def reference(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            digest.update(chunk)
    return dict(path=str(path), bytes=path.stat().st_size,
                sha256=digest.hexdigest())


def bound(row):
    require(isinstance(row, dict) and
            set(('path', 'bytes', 'sha256')) <= set(row), 'Incomplete reference')
    require(Path(row['path']).is_absolute(), 'Absolute artifact reference required')
    require(reference(row['path']) == {key: row[key] for key in
                                     ('path', 'bytes', 'sha256')},
            'Artifact reference differs: ' + row['path'])
    return Path(row['path'])


def bind_tree(value):
    if isinstance(value, dict):
        if set(('path', 'bytes', 'sha256')) <= set(value):
            bound(value)
        else:
            for item in value.values():
                bind_tree(item)
    elif isinstance(value, list):
        for item in value:
            bind_tree(item)


def canonical_sha(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'),
                     allow_nan=False).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def terminal_guard(receipt, phase):
    keys = ('status', 'returncode', 'root_wait_reaped', 'held_handle_exit', 'numeric_birth_absence',
            'abort', 'error', 'wait_errors', 'GPU_observation_errors', 'original_identity',
            'admission_sha256')
    require(all(key in receipt for key in keys) and
            receipt['status'] == 'ACTUAL_OWNED_TERMINAL' and receipt['original_identity'] is not None and
            type(receipt.get('returncode')) is int and receipt['returncode'] == 0 and
            receipt.get('root_wait_reaped') is True and
            receipt.get('held_handle_exit') ==
            {'wait_state': 'WAIT_OBJECT_0', 'actual_exit_code': 0} and
            receipt.get('numeric_birth_absence', {}).get('original_birth_absent') is True and
            receipt.get('abort') is None and receipt.get('error') is None and
            receipt.get('wait_errors') is False and receipt.get('GPU_observation_errors') == [],
            'STOP: actual owned exit0, held reap, birth absence and no errors required')
    if 'phase' in receipt:
        require(receipt['phase'] == phase, 'Owned receipt phase differs')


def admission_guard(spec_path, spec, receipt):
    """Join the already reaped receipt to this spec/entry before scientific access."""
    spec_path = spec_path.resolve(strict=True)
    admission_name = spec['GPU_admission_file']
    require(isinstance(admission_name, str) and Path(admission_name).is_absolute(),
            'SPEC GPU_admission_file must be an absolute path string')
    admission_path = Path(admission_name).resolve(strict=True)
    admission_ref = reference(admission_path)
    require(receipt['admission_sha256'] == admission_ref['sha256'],
            'Owned receipt does not bind this actual GPU admission')
    admission = read_json(admission_path)
    spec_ref = reference(spec_path)
    entry = Path(__file__).resolve().with_name('run.py')
    entry_ref = reference(entry)
    owner = spec_path.parent / 'finite_point_cache_execution_20261005_v2' / 'run_owned_gpu_limited.py'
    owner_ref = reference(owner)
    require(owner_ref['sha256'] ==
            'ef9a9f17d4fb1bc3464936095ef799021dd880f8b15488d558a871c08b6e1924' and
            admission['driver_sha256'] == owner_ref['sha256'], 'Actual owned driver source differs')
    require(admission['status'] == 'ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE' and
            admission['automatic_retry'] is False and admission['phase'] == spec['phase'] and
            admission['allowed_cpus'] == spec['CPU_affinity'] == [0, 1] and
            admission['cuda_visible_devices'] == '0' and
            type(admission['qualified_leaf_deadline_seconds']) is int and
            admission['qualified_leaf_deadline_seconds'] == spec['deadline_seconds'] and
            admission['whole_GPU_used_limit_MiB'] == spec['whole_GPU_owner_limit_MiB'] == 8192,
            'Admission phase/CPU/CUDA/deadline/single-attempt resource contract differs')
    require(Path(admission['entry']).is_absolute() and
            Path(admission['entry']).resolve(strict=True) == entry and
            admission['entry_args'] == ['--spec', str(spec_path)] and
            entry_ref['bytes'] == DRIVER_BYTES and entry_ref['sha256'] == DRIVER_SHA,
            'Admission does not execute the exact frozen sibling run.py with this SPEC')
    hashes = admission['bound_source_sha256']
    require(isinstance(hashes, dict) and hashes.get(str(spec_path)) == spec_ref['sha256'] and
            hashes.get(str(entry)) == entry_ref['sha256'], 'Admission spec/entry hashes differ')
    for row in spec['sources']:
        require(hashes.get(row['path']) == row['sha256'], 'SPEC source is absent from actual admission')
        bound(row)
    require(entry_ref in [{key: row[key] for key in ('path', 'bytes', 'sha256')}
                          for row in spec['sources']], 'Frozen entry absent from SPEC source list')
    if 'scientific_spec' in admission:
        require(admission['scientific_spec'] == spec_ref, 'Admission scientific_spec differs')
    peak = receipt['whole_GPU_observed_peak_MiB']
    require(type(peak) in (int, float) and math.isfinite(peak) and 0 <= peak <= 8192 and
            receipt['whole_GPU_used_limit_MiB'] == 8192 and
            receipt['whole_leaf_VRAM_certified'] is False and
            receipt['GPU_scope'] == 'whole GPU observed, not per-process WDDM allocation',
            'Actual whole-GPU observation/resource scope differs')
    return admission_ref, spec_ref, entry_ref, owner_ref


def finite(value, label, nonnegative=False):
    require(type(value) in (int, float) and math.isfinite(value), label + ' is nonfinite')
    require(not nonnegative or value >= 0, label + ' is negative')


def tensor_digest(torch, value, dtype=None):
    require(isinstance(value, torch.Tensor) and value.device.type == 'cpu',
            'CPU tensor required')
    array = value.detach().contiguous().numpy()
    allowed = (str(value.dtype), str(array.dtype))
    require(dtype is None or dtype in allowed, 'Tensor digest dtype differs')
    return dict(shape=list(value.shape), dtype=dtype or str(value.dtype),
                sha256=hashlib.sha256(array.tobytes()).hexdigest())


def check_state(torch, state, proof, label):
    require(isinstance(state, dict) and set(state) == set(proof),
            label + ' state keys differ')
    for name, value in state.items():
        require(isinstance(value, torch.Tensor) and value.device.type == 'cpu' and
                bool(torch.isfinite(value).all()), label + ' tensor invalid: ' + name)
        require(tensor_digest(torch, value, proof[name]['dtype']) == proof[name],
                label + ' tensor proof differs: ' + name)


def canonical_state(torch, checkpoint):
    require(isinstance(checkpoint, dict) and
            isinstance(checkpoint.get('state_dict'), dict),
            'Canonical checkpoint.state_dict required')
    state = checkpoint['state_dict']
    require(state and all(isinstance(key, str) and isinstance(value, torch.Tensor)
                          for key, value in state.items()), 'Invalid state container')
    prefix = [key.startswith('module.') for key in state]
    require(all(prefix) or not any(prefix), 'Mixed module. state prefixes')
    result = {key[7:] if all(prefix) else key: value for key, value in state.items()}
    require(len(result) == len(state), 'Duplicate normalized state names')
    return result


def independent_ship_rows(torch, kind, official, official_model):
    """Select exactly official COCO boat8/background80, with no random fallback."""
    result, selected = dict(official), {}

    def take(name, original_rows, rows):
        require(name in official and official[name].shape[0] == original_rows,
                'Official class tensor shape differs: ' + name)
        result[name] = official[name][rows].clone()
        selected[name] = rows

    if kind == 'mask2former':
        require(official_model.panoptic_head.num_classes == 80, 'COCO80 M2F required')
        for suffix in ('weight', 'bias'):
            take('panoptic_head.cls_embed.' + suffix, 81, [8, 80])
    elif kind == 'condinst':
        require(official_model.bbox_head.num_classes == 80 and
                official_model.bbox_head.loss_cls.use_sigmoid, 'COCO80 sigmoid CondInst required')
        for suffix in ('weight', 'bias'):
            take('bbox_head.conv_cls.' + suffix, 80, [8])
    else:
        require(len(official_model.roi_head.bbox_head) == 3 and
                len(official_model.roi_head.mask_head) == 3, 'Three Cascade stages required')
        for stage, head in enumerate(official_model.roi_head.bbox_head):
            require(head.num_classes == 80 and
                    not official_model.roi_head.mask_head[stage].class_agnostic,
                    'COCO80 class-specific Cascade masks required')
            for suffix in ('weight', 'bias'):
                take('roi_head.bbox_head.%d.fc_cls.%s' % (stage, suffix), 81, [8, 80])
                take('roi_head.mask_head.%d.conv_logits.%s' % (stage, suffix), 80, [8])
                key = 'roi_head.bbox_head.%d.fc_reg.%s' % (stage, suffix)
                if head.reg_class_agnostic:
                    require(official[key].shape[0] == 4, 'Class-agnostic box rows must stay4')
                else:
                    take(key, 320, [32, 33, 34, 35])
    require(set(result) == set(official) and all(torch.equal(result[key], official[key])
                for key in set(official) - set(selected)), 'Unexpected initializer changes')
    return result, selected


def load_source(row, name):
    definition = importlib.util.spec_from_file_location(name, str(bound(row)))
    module = importlib.util.module_from_spec(definition)
    definition.loader.exec_module(module)
    return module


def independent_policy(torch, model, kind):
    parameters = dict(model.named_parameters())
    bn_parameters, bn_buffers, bn_modules = set(), set(), []
    conv_parameters = []
    for prefix, module in model.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            bn_modules.append(prefix)
            bn_parameters.update((prefix + '.' if prefix else '') + key
                                 for key, _ in module.named_parameters(recurse=False))
            bn_buffers.update((prefix + '.' if prefix else '') + key
                              for key, _ in module.named_buffers(recurse=False))
        if prefix.startswith('backbone.') and isinstance(module, torch.nn.Conv2d):
            conv_parameters.extend(prefix + '.' + key
                                   for key, _ in module.named_parameters(recurse=False))
    active = [name for name in parameters if name not in bn_parameters]
    require(conv_parameters and set(conv_parameters) <= set(active),
            'Every actual R50 Conv2d parameter must be active')
    return dict(active_names=active, frozen_names=sorted(bn_parameters), BN_modules=bn_modules,
                BN_buffer_names=sorted(bn_buffers), backbone_conv_names=conv_parameters,
                trainable_tensors=len(active),
                trainable_scalars=sum(parameters[name].numel() for name in active),
                frozen_tensors=len(bn_parameters),
                frozen_scalars=sum(parameters[name].numel() for name in bn_parameters),
                state_entries=len(model.state_dict()), parameter_tensors=len(parameters), kind=kind)


def independent_optimizer(torch, model, entry, policy):
    recipe = dict(entry['optimizer'])
    family = recipe.pop('type')
    expected_family = 'AdamW' if policy['kind'] == 'mask2former' else 'SGD'
    require(family == expected_family, 'Native optimizer family differs')
    options = entry['paramwise_cfg']
    require(set(options) <= {'custom_keys', 'bias_lr_mult', 'bias_decay_mult', 'norm_decay_mult'},
            'Unsupported parameter policy')
    custom = options.get('custom_keys', {})
    owners = dict(model.named_modules())
    parameters = dict(model.named_parameters())
    norms = (torch.nn.modules.batchnorm._BatchNorm, torch.nn.modules.instancenorm._InstanceNorm,
             torch.nn.GroupNorm, torch.nn.LayerNorm)
    groups, rows = [], []
    for name in policy['active_names']:
        value = parameters[name]
        owner, local = name.rsplit('.', 1) if '.' in name else ('', name)
        is_norm, is_bias = isinstance(owners[owner], norms), local == 'bias'
        candidates = [key for key in custom if key in name]
        selected = min(candidates, key=lambda key: (-len(key), key)) if candidates else None
        lr, wd = recipe['lr'], recipe['weight_decay']
        if selected is not None:
            require(set(custom[selected]) <= {'lr_mult', 'decay_mult'}, 'Unsupported custom option')
            lr *= custom[selected].get('lr_mult', 1.)
            wd *= custom[selected].get('decay_mult', 1.)
        else:
            if is_bias and not is_norm:
                lr *= options.get('bias_lr_mult', 1.)
            wd *= options.get('norm_decay_mult', 1.) if is_norm else (
                options.get('bias_decay_mult', 1.) if is_bias else 1.)
        finite(lr, 'Group LR', True)
        finite(wd, 'Group decay', True)
        require(lr > 0, 'Positive active group LR required')
        groups.append(dict(params=[value], lr=lr, weight_decay=wd))
        rows.append(dict(name=name, shape=list(value.shape), lr=lr, weight_decay=wd,
                         typed_norm_owner=is_norm, native_local_parameter_name_is_bias=is_bias,
                         custom_key=selected))
    if 'betas' in recipe:
        recipe['betas'] = tuple(recipe['betas'])
    # Construction only: no optimizer step, backward, or CUDA trajectory replay.
    optimizer = (torch.optim.AdamW if family == 'AdamW' else torch.optim.SGD)(groups, **recipe)
    settings = entry['clip_grad']
    if settings is not None:
        require(isinstance(settings, dict) and set(settings) == {'max_norm', 'norm_type'} and
                settings['norm_type'] == 2, 'Explicit finite global clipping dictionary required')
        finite(settings['max_norm'], 'Clip max norm', True)
        require(settings['max_norm'] > 0, 'Positive clip max norm required')
    audit = dict(type=family, recipe=recipe, paramwise_cfg=options, parameter_groups=rows,
                 clip_grad=settings)
    return optimizer.state_dict()['param_groups'], audit


def validate_optimizer(torch, checkpoint, expected_groups, audit, update_counts):
    saved = checkpoint['optimizer_state_dict']
    require(set(saved) == {'state', 'param_groups'} and
            canonical_sha(saved['param_groups']) == canonical_sha(expected_groups),
            'Independent optimizer group defaults/membership/hyperparameters differ')
    names = [row['name'] for row in audit['parameter_groups']]
    require(checkpoint['meta']['optimizer_param_names'] == [[name] for name in names],
            'Optimizer parameter ID/name mapping differs')
    ids = [group['params'][0] for group in expected_groups]
    require(len(ids) == len(set(ids)) and all(len(group['params']) == 1 for group in expected_groups),
            'One unique active parameter per group required')
    expected_ids = {parameter_id for name, parameter_id in zip(names, ids)
                    if update_counts[name] and (audit['type'] == 'AdamW' or
                                               audit['recipe'].get('momentum', 0) != 0)}
    require(set(saved['state']) == expected_ids, 'Actual optimizer moment membership differs')
    for name, parameter_id in zip(names, ids):
        if parameter_id not in expected_ids:
            continue
        moments = saved['state'][parameter_id]
        value = checkpoint['state_dict'][name]
        if audit['type'] == 'AdamW':
            require(set(moments) == {'step', 'exp_avg', 'exp_avg_sq'}, 'AdamW moment fields differ')
            step = moments['step']
            require(isinstance(step, torch.Tensor) and step.device.type == 'cpu' and
                    step.numel() == 1 and bool(torch.isfinite(step).all()) and
                    float(step.item()) == update_counts[name], 'AdamW actual per-parameter steps differ')
            keys = ('exp_avg', 'exp_avg_sq')
        else:
            require(set(moments) == {'momentum_buffer'}, 'Native SGD has no invented step field')
            keys = ('momentum_buffer',)
        for key in keys:
            moment = moments[key]
            require(isinstance(moment, torch.Tensor) and moment.device.type == 'cpu' and
                    moment.shape == value.shape and moment.dtype == value.dtype and
                    bool(torch.isfinite(moment).all()), 'Optimizer moment shape/dtype/finite differs')
            if key == 'exp_avg_sq':
                require(bool((moment >= 0).all()), 'AdamW squared moment is negative')
    return dict(family=audit['type'], unique_active_groups=len(names),
                actual_moment_entries=len(saved['state']), per_parameter_updates=update_counts,
                SGD_step_field_not_fabricated=True, CUDA_trajectory_replayed=False)


def validate_rng(torch, row):
    for key in ('private_RNG_before', 'private_RNG_after', 'external_RNG_before', 'external_RNG_after'):
        values = row[key]
        require(isinstance(values, list) and len(values) == 2, 'CPU/CUDA RNG pair required')
        for item in values:
            require(item['dtype'] == 'uint8' and len(item['shape']) == 1 and
                    type(item['shape'][0]) is int and item['shape'][0] > 0 and
                    isinstance(item['sha256'], str) and len(item['sha256']) == 64 and
                    all(char in '0123456789abcdef' for char in item['sha256']), 'RNG digest invalid')
    generator = torch.Generator(device='cpu')
    generator.manual_seed(row['private_seed'])
    require(tensor_digest(torch, generator.get_state(), 'uint8') == row['private_RNG_before'][0],
            'Independent private CPU loss seed differs')
    require(row['external_RNG_before'] == row['external_RNG_after'] and
            row['external_RNG_restored'] is True, 'External RNG was not restored')


def validate_GT(torch, np, cv2, coco, image, ann_ids, row):
    identity, target = row['identity'], row['GT']
    height, width = int(image['height']), int(image['width'])
    scale = 800. / max(height, width)
    actual = [int(height * scale + .5), int(width * scale + .5)]
    factors = [actual[1] / width, actual[0] / height]
    require(identity['image_id'] == image['id'] and identity['original_shape'] == [height, width] and
            identity['actual_img_shape'] == actual and identity['batch_input_shape'] == [800, 800] and
            identity['scale_factor'] == factors and identity['native_input_gamma'] == 1 and
            identity['target_gray_resize'] == 'nearest' and identity['ann_ids'] == ann_ids,
            'Independent keep-ratio800 identity/scale/order differs')
    require(identity['parser_rows'] == [dict(annotation_id=key, native_parser_retained=True)
                                        for key in ann_ids], 'Raw annotation was dropped by parser')
    require(target['ann_ids'] == ann_ids and target['complete_all_raw_annotations'] is True and
            target['bboxes_are_original_annotations_not_mask_inference'] is True and
            target['labels_values'] == [0] * len(ann_ids), 'All original GT/category mapping differs')
    require(row['actual_input_shape'] == [1, 3, 800, 800] and
            row['input']['shape'] == [1, 3, 800, 800] and row['input']['dtype'] == 'float32',
            'Actual native normalized input canvas differs')
    masks = np.zeros((len(ann_ids), 800, 800), dtype=np.uint8)
    boxes = []
    for index, key in enumerate(ann_ids):
        ann = coco.anns[key]
        require(ann['image_id'] == image['id'] and ann['category_id'] == 1 and
                not ann.get('ignore', False) and not ann.get('iscrowd', 0),
                'Expected complete nonignored ship annotations')
        x, y, w, h = ann['bbox']
        require(w > 0 and h > 0 and ann.get('area', 0) > 0, 'Native parser-incompatible bbox/area')
        boxes.append([x, y, x + w, y + h])
        original = coco.annToMask(ann).astype(np.uint8, copy=False)
        require(original.shape == (height, width), 'Original COCO GT raster shape differs')
        masks[index, :actual[0], :actual[1]] = cv2.resize(
            original, (actual[1], actual[0]), interpolation=cv2.INTER_NEAREST)
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    boxes *= np.asarray([factors[0], factors[1], factors[0], factors[1]], dtype=np.float32)
    boxes[:, (0, 2)] = boxes[:, (0, 2)].clip(0, actual[1])
    boxes[:, (1, 3)] = boxes[:, (1, 3)].clip(0, actual[0])
    box_tensor = torch.from_numpy(np.ascontiguousarray(boxes))
    label_tensor = torch.zeros(len(ann_ids), dtype=torch.int64)
    mask_tensor = torch.from_numpy(masks)
    require(target['boxes'] == boxes.tolist() and tensor_digest(torch, box_tensor, 'float32') == target['bbox'] and
            tensor_digest(torch, label_tensor, 'int64') == target['labels'] and
            tensor_digest(torch, mask_tensor, 'uint8') == target['masks'] and
            [tensor_digest(torch, value, 'uint8') for value in mask_tensor] == target['masks_per_GT'],
            'Independent raw boxes/labels/nearest800 padded masks differ')
    bound(identity['image'])


def validate_losses(row, kind):
    values = row['loss_fields']
    require(isinstance(values, dict) and values, 'Native loss fields missing')
    atoms = []
    for name, value in values.items():
        parts = value if isinstance(value, list) else [value]
        require(parts, 'Empty loss/metric list')
        for item in parts:
            finite(item, 'Native loss/metric ' + name)
        if 'loss' in name:
            atoms.extend(parts)
    require(atoms and row['loss_is_native_model_loss'] is True and row['Sobel_added'] is False,
            'Only actual native model.loss is permitted')
    if kind == 'mask2former':
        expected = {'loss_' + term for term in ('cls', 'mask', 'dice')}
        expected |= {'d%d.loss_%s' % (layer, term) for layer in range(9)
                     for term in ('cls', 'mask', 'dice')}
        require(set(values) == expected and len(atoms) == 30, 'M2F native30 fields differ')
    elif kind == 'condinst':
        require({'loss_cls', 'loss_bbox', 'loss_centerness', 'loss_mask'} <= set(values),
                'CondInst original losses missing')
    else:
        expected = {'loss_rpn_cls', 'loss_rpn_bbox'}
        expected |= {'s%d.loss_%s' % (stage, term) for stage in range(3)
                     for term in ('cls', 'bbox', 'mask')}
        require(expected <= set(values), 'Cascade original losses missing')
    finite(row['native_total'], 'Native total')
    # Each field mean is exported; list reductions and total use FP32 adds.
    count = len(atoms) + sum('loss' in key for key in values)
    unit = 2. ** -24
    gamma = count * unit / (1. - count * unit)
    require(abs(row['native_total'] - math.fsum(atoms)) <=
            gamma * math.fsum(abs(value) for value in atoms) + count * 2. ** -126,
            'Native total outside legal FP32 sum-of-exported-means bound')


def validate_trace(torch, np, cv2, coco, spec, kind, result, policy, images, image_order):
    require(len(result['trace']) == 4, 'Exactly four actual micro traces required')
    active = policy['active_names']
    positive_counts = {name: 0 for name in active}
    update_counts = {name: 0 for name in active}
    consumed, traces, last_none = [], [], {}
    for micro, artifact in enumerate(result['trace']):
        row = read_json(bound(artifact))
        image_id = spec['image_ids'][micro]
        require(row['kind'] == kind and row['microstep'] == micro and row['image_id'] == image_id and
                row['identity']['dataset_index'] == image_order.index(image_id) and
                row['private_seed'] == spec['loss_seed_base'] + micro and
                row['accumulation_divisor'] == 2 and row['accumulated_preclip_gradients'] is True and
                row['optimizer_step'] is (micro % 2 == 1) and row['optimizer_updates'] == (micro + 1) // 2,
                'Actual image/seed/accumulation/update counters differ')
        require(row['gradient_clipping_applied'] is
                (row['optimizer_step'] and spec['initializers'][kind]['clip_grad'] is not None),
                'Actual nullable native clipping policy differs')
        ann_ids = [int(ann['id']) for ann in coco.imgToAnns.get(image_id, [])]
        validate_GT(torch, np, cv2, coco, images[image_id], ann_ids, row)
        consumed.extend(ann_ids)
        validate_losses(row, kind)
        validate_rng(torch, row)
        if traces:
            require(traces[-1]['external_RNG_after'] == row['external_RNG_before'],
                    'External RNG continuity differs')
        gradient_rows = row['gradient_rows']
        require(set(gradient_rows) == set(active), 'Every active gradient row must be retained')
        positive, none = [], []
        for name in active:
            value = gradient_rows[name]
            require(set(value) == {'is_None', 'finite', 'L2'} and type(value['is_None']) is bool,
                    'Actual gradient row schema differs')
            last_none[name] = value['is_None']
            if value['is_None']:
                require(value['finite'] is None and value['L2'] is None, 'Missing gradient must stay None')
                none.append(name)
            else:
                require(value['finite'] is True, 'Observed nonfinite active gradient')
                finite(value['L2'], 'Actual accumulated gradient L2', True)
                if value['L2'] > 0:
                    positive.append(name)
                    positive_counts[name] += 1
                if row['optimizer_step']:
                    update_counts[name] += 1
        require(positive == row['positive_gradient_names'] and none == row['None_gradient_names'] and
                row['finite_gradient_tensors'] == len(active) - len(none), 'Gradient activity/counts differ')
        require(row['BN_stats_and_affine_frozen'] is True and
                row['frozen_BN_parameter_grads_None'] is True and
                row['BN_module_training'] == {name: False for name in policy['BN_modules']},
                'BN train mode or affine gradients differ')
        if row['optimizer_step']:
            finite(row['preclip_norm'], 'Actual global preclip norm', True)
        else:
            require(row['preclip_norm'] is None, 'No clip/optimizer step on odd accumulation micro')
        traces.append(row)
    return consumed, traces, positive_counts, update_counts, last_none


def qualify_model(torch, np, cv2, registry, revert, Config, initializer, qualification,
                  spec, kind, result, coco, images, image_order, spec_ref, driver_ref):
    entry = spec['initializers'][kind]
    accepted = qualification['initializers'][kind]
    require(entry['checkpoint'] == accepted['checkpoint'] and entry['config'] == accepted['official_config'] and
            accepted['strict_official80_and_ship1_compatible'] is True,
            'Previously qualified official initializer lineage differs')
    config = Config.fromfile(str(bound(entry['config'])))
    require(config.model.backbone.depth == 50 and config.model.backbone.style == 'pytorch',
            'Declared PyTorch-style R50 required')
    closure = initializer.config_closure(bound(entry['config']))
    sources = {str(Path(row['path']).resolve()): {key: row[key] for key in ('path', 'bytes', 'sha256')}
               for row in spec['sources']}
    require(all(sources.get(row['path']) == row for row in closure), 'Unbound config inheritance')
    original_cp = torch.load(str(bound(entry['checkpoint'])), map_location='cpu')
    official = canonical_state(torch, original_cp)
    original_model = initializer.build_cpu(torch, registry, revert, config.model)
    strict80 = initializer.strict_load(torch, original_model, official)
    initial, selections = independent_ship_rows(torch, kind, official, original_model)
    ship_config, crosscheck, declared_selections = initializer.ship_state(
        torch, kind, config.model, original_model, official)
    require(all(torch.equal(initial[name], crosscheck[name]) for name in initial),
            'Independent boat mapping differs from qualified source mapping')
    require(set(selections) == set(declared_selections) and all(
        selections[name] == declared_selections[name]['selected_rows'] for name in selections),
        'Allowed class/box row selections differ')
    del original_model, original_cp, official, crosscheck
    ship_config.backbone.frozen_stages = -1
    ship_config.backbone.norm_eval = True
    ship_config.backbone.norm_cfg.requires_grad = False
    ship_config.backbone.with_cp = spec['backbone_with_cp']
    ship_config.data_preprocessor.pad_mask = True
    ship_config.data_preprocessor.pad_size_divisor = 32
    ship_config.data_preprocessor.batch_augments = [dict(type='BatchFixedSizePad', size=(800, 800),
        img_pad_value=0, pad_mask=True, mask_pad_value=0, pad_seg=False)]
    model = initializer.build_cpu(torch, registry, revert, ship_config)
    strict1 = initializer.strict_load(torch, model, initial)
    init_proof = read_json(bound(result['initialization_proof']))
    require(init_proof['kind'] == kind and init_proof['checkpoint'] == entry['checkpoint'] and
            init_proof['official_config'] == entry['config'] and
            init_proof['official_config_closure'] == closure and init_proof['official80'] == strict80 and
            init_proof['ship1'] == strict1 and init_proof['allowed_changed_keys'] == declared_selections and
            init_proof['initial_ship_tensors_exact'] is True and
            init_proof['only_boat_classifier_rows_changed'] is True and
            init_proof['random_missing_keys'] is False and
            init_proof['public_recipe_equivalence_claimed'] is False and
            init_proof['backbone_with_cp'] == spec['backbone_with_cp'], 'Initialization proof differs')
    check_state(torch, initial, init_proof['initial_state'], kind + ' independently remapped initializer')
    policy = independent_policy(torch, model, kind)
    expected_groups, optimizer_audit = independent_optimizer(torch, model, entry, policy)
    checkpoint = torch.load(str(bound(result['checkpoint'])), map_location='cpu')
    require(set(checkpoint) == {'state_dict', 'optimizer_state_dict', 'external_rng_state',
                                'GT_trace_counters', 'meta'}, 'Final checkpoint container differs')
    final, meta = checkpoint['state_dict'], checkpoint['meta']
    expected = dict(schema=CP_SCHEMA, kind=kind, phase=spec['phase'], spec=spec_ref, source=driver_ref,
        initial_checkpoint=entry['checkpoint'], initializer_source=spec['initializer_source'],
        image_ids=spec['image_ids'], microsteps=4, optimizer_updates=2, batch_size=1,
        accumulation=2, full_R50_conv_trainable=True, BN_stats_and_affine_frozen=True,
        trainable=policy, native_loss_only=True, not_formal_training=True)
    for key, value in expected.items():
        require(meta[key] == value, 'Final checkpoint metadata differs: ' + key)
    require(canonical_sha(meta['optimizer']) == canonical_sha(optimizer_audit), 'CP optimizer policy differs')
    proof = read_json(bound(result['state_proof']))
    require(proof['schema'] == 'native-coco-comparator-micro-state-proof-v1' and
            proof['kind'] == kind and proof['policy'] == policy and
            canonical_sha(proof['optimizer']) == canonical_sha(optimizer_audit), 'State/policy proof differs')
    check_state(torch, initial, proof['before'], kind + ' initial')
    check_state(torch, final, proof['after'], kind + ' final')
    require(set(final) == set(initial), 'Final native state keys differ')
    for name, value in final.items():
        require(value.shape == initial[name].shape and value.dtype == initial[name].dtype,
                'Final state shape/dtype differs: ' + name)
    changed = {name: not torch.equal(initial[name], value) for name, value in final.items()}
    require(changed == proof['changed'] and all(not changed[name] for name in
            set(initial) - set(policy['active_names'])), 'A frozen/nonparameter state changed')
    consumed, traces, positive_counts, update_counts, last_none = validate_trace(
        torch, np, cv2, coco, spec, kind, result, policy, images, image_order)
    require(meta['consumed_ann_ids'] == consumed and result['consumed_ann_ids'] == consumed and
            checkpoint['GT_trace_counters'] == dict(image_ids=spec['image_ids'], microsteps=4,
                                                   optimizer_updates=2, ann_ids=consumed),
            'Actual four-image GT/update counters differ')
    require(proof['positive_gradient_microcounts'] == positive_counts and
            proof['last_preclip_gradient_None'] == last_none and
            proof['gradient_activity_is_accumulated'] is True and
            proof['frozen_BN_all_exact'] is True and proof['all_final_state_finite'] is True and
            proof['CPU_checkpoint_readback_exact'] is True, 'Actual gradient/state evidence differs')
    changed_active = sum(changed[name] for name in policy['active_names'])
    changed_positive = sum(changed[name] and positive_counts[name] > 0 for name in policy['active_names'])
    require(changed_active == proof['changed_active_tensors'] and
            changed_positive == proof['changed_with_positive_activity'] and changed_positive > 0,
            'No independently changed active tensor with observed genuine gradient')
    optimizer = validate_optimizer(torch, checkpoint, expected_groups, optimizer_audit, update_counts)
    rng = checkpoint['external_rng_state']
    require(isinstance(rng, list) and len(rng) == 2 and all(isinstance(value, torch.Tensor) and
            value.device.type == 'cpu' and value.dtype == torch.uint8 and value.ndim == 1 and
            value.numel() > 0 for value in rng), 'Actual CPU/CUDA external RNG checkpoint invalid')
    require([tensor_digest(torch, value, 'uint8') for value in rng] == traces[-1]['external_RNG_after'],
            'Checkpoint RNG differs from actual last restored trace')
    for key, value in dict(kind=kind, microsteps=4, optimizer_updates=2, image_ids=spec['image_ids'],
                          native_loss_only=True, inference_executed=False, AP_computed=False,
                          no_GT_inference_route_added=True, GT_free_inference_runtime_verified=False,
                          CPU_checkpoint_readback_exact=True,
                          trainable_tensors=policy['trainable_tensors'],
                          clip_grad=spec['initializers'][kind]['clip_grad']).items():
        require(result[key] == value, 'Model report differs: ' + key)
    result_proof = dict(checkpoint=result['checkpoint'], initialization_proof=result['initialization_proof'],
        state_proof=result['state_proof'], trace=result['trace'], initial_rows=selections,
        native_state_entries=len(initial), trainability=policy, actual_microsteps=4,
        actual_optimizer_updates=2, all_raw_ann_ids=consumed,
        changed_active_tensors=changed_active, changed_with_positive_gradient=changed_positive,
        positive_gradient_microcounts=positive_counts, zero_positive_gradient_names=[name for name in
            policy['active_names'] if not positive_counts[name]], optimizer=optimizer,
        initial_and_final_state_independently_CPU_verified=True,
        all_nonactive_state_exact=True, all_GT_nearest800_independently_reconstructed=True)
    del model, initial, final, checkpoint
    gc.collect()
    require(not torch.cuda.is_initialized(), 'CPU qualification initialized CUDA')
    return result_proof


def qualify(spec_file, acceptance_file):
    spec_path, target = Path(spec_file), Path(acceptance_file)
    require(spec_path.is_absolute() and target.is_absolute() and not target.exists() and
            target.parent.is_dir(), 'Absolute spec and fresh acceptance with existing parent required')
    spec = read_json(spec_path)
    require(spec['schema'] == SPEC_SCHEMA and tuple(spec['models']) == KINDS and
            len(spec['image_ids']) == len(set(spec['image_ids'])) == 4 and
            all(type(value) is int for value in spec['image_ids']) and
            type(spec['backbone_with_cp']) is bool and spec['loss_seed_base'] == 1000000 and
            set(spec['initializers']) == set(KINDS), 'Locked three-model/four-micro spec differs')
    receipt_path = spec_path.resolve().parent / 'finite_point_cache_execution_20261005_v2' / 'runs' / spec['phase'] / 'PARENT_RECEIPT.json'
    receipt = read_json(receipt_path)
    terminal_guard(receipt, spec['phase'])
    # No scientific import, payload read, or any artifact hash occurs above.
    admission_ref, spec_ref, admitted_entry_ref, owner_ref = admission_guard(spec_path, spec, receipt)
    receipt_ref = reference(receipt_path)
    bind_tree(spec)
    qualification = read_json(bound(spec['initializer_qualification']))
    require(qualification['schema'] == 'ROOT-native-coco-comparator-initializer-CPU-inspection-v1' and
            qualification['all_initializers_strictly_compatible'] is True and
            qualification['source'] == spec['initializer_source'] and
            set(qualification['initializers']) == set(KINDS), 'Previous independent CPU initializer qualification differs')
    bind_tree(qualification)
    report_path = Path(spec['output_directory']) / 'REPORT.json'
    report = read_json(report_path)
    report_ref, driver_ref = reference(report_path), report['source']
    require(driver_ref == admitted_entry_ref and
            driver_ref['bytes'] == DRIVER_BYTES and driver_ref['sha256'] == DRIVER_SHA and
            Path(driver_ref['path']).resolve() == Path(__file__).resolve().with_name('run.py') and
            driver_ref in [{key: row[key] for key in ('path', 'bytes', 'sha256')} for row in spec['sources']],
            'Producer exact frozen source pin differs')
    bound(driver_ref)
    for key, value in dict(schema=REPORT_SCHEMA, status='TECHNICAL_NATIVE_LOSS_MICRO_PREFLIGHT_COMPLETE_NO_AP',
        phase=spec['phase'], spec=spec_ref, models=list(KINDS), image_ids=spec['image_ids'],
        microsteps_per_model=4, optimizer_updates_per_model=2, accumulation=2,
        initializer_source=spec['initializer_source'], initializer_qualification=spec['initializer_qualification'],
        native_helpers=spec['native_helpers'], all_raw_GT_retained=True, error=None, no_inference=True,
        no_DEV=True, no_AP=True, scientific_method_decision_computed=False,
        published_recipe_reproduction_claimed=False, warm610_adapter_comparison_track=False,
        GT_free_inference_runtime_verified=False).items():
        require(report[key] == value, 'Actual completed report differs: ' + key)
    require(set(report['model_reports']) == set(KINDS) and set(report['cleanup']) == set(KINDS),
            'All three actual models/cleanups required')
    for row in report['cleanup'].values():
        require(row['allocated_after'] == 0, 'GPU cleanup was not exact0')
        finite(row['allocated_before'], 'Cleanup allocation', True)
    finite(report['torch_peak_allocated_bytes'], 'Allocator peak', True)
    finite(report['elapsed_seconds'], 'Actual wall seconds', True)
    require(report['torch_peak_allocated_bytes'] <= 4096 * 1024 ** 2 and
            report['elapsed_seconds'] < spec['deadline_seconds'], 'Prelocked allocator/deadline differs')
    start_path = Path(spec['output_directory']) / 'START.json'
    start = read_json(start_path)
    require(start == dict(schema='native-coco-comparator-micro-preflight-start-v1', phase=spec['phase'],
        spec=spec_ref, source=driver_ref, models=list(KINDS), image_ids=spec['image_ids'],
        four_microsteps_two_updates=True, formal_training=False, no_inference=True,
        no_DEV=True, no_AP=True, no_retries=True), 'Actual START differs')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '2'
    os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
    import psutil
    psutil.Process().cpu_affinity([0, 1])
    import numpy as np
    import cv2
    import torch
    from pycocotools.coco import COCO
    from mmengine import Config
    from mmengine.model import revert_sync_batchnorm
    from mmengine.registry import init_default_scope
    from mmdet.registry import MODELS
    from mmdet.utils import register_all_modules
    from mmdet.datasets import CocoDataset
    require(psutil.Process().cpu_affinity() == [0, 1] and not torch.cuda.is_initialized(),
            'CPU01 and no initialized CUDA required')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    cv2.setNumThreads(2)
    register_all_modules(init_default_scope=False)
    init_default_scope('mmdet')
    classes = tuple(CocoDataset.METAINFO['classes'])
    require(len(classes) == 80 and classes[8] == 'boat', 'Installed official COCO80 class order differs')
    initializer = load_source(spec['initializer_source'], 'independently_qualified_initializer_CPU')
    fit = read_json(bound(spec['FIT']))
    images = {int(row['id']): row for row in fit['images']}
    image_order = [int(row['id']) for row in fit['images']]
    require(len(images) == len(image_order) == 741 and len(fit['annotations']) == 1661 and
            set(spec['image_ids']) <= set(images) and len({ann['id'] for ann in fit['annotations']}) == 1661,
            'Original complete FIT population/unique raw annotations differ')
    coco = COCO(str(bound(spec['FIT'])))
    before_rng = torch.get_rng_state().clone()
    proofs = {kind: qualify_model(torch, np, cv2, MODELS, revert_sync_batchnorm, Config,
        initializer, qualification, spec, kind, report['model_reports'][kind], coco, images,
        image_order, spec_ref, driver_ref) for kind in KINDS}
    require(torch.equal(before_rng, torch.get_rng_state()) and not torch.cuda.is_initialized(),
            'CPU constructor/qualification consumed external RNG or initialized CUDA')
    require(all(proofs[kind]['all_raw_ann_ids'] == proofs[KINDS[0]]['all_raw_ann_ids'] for kind in KINDS),
            'The three models did not consume the same complete GT/order')
    acceptance = dict(schema='ROOT-native-coco-comparator-micro-preflight-CPU-qualification-v1',
        status='ROOT_NATIVE_COCO_COMPARATOR_MICRO_PREFLIGHT_TECHNICALLY_QUALIFIED',
        phase=spec['phase'], spec=spec_ref, source=driver_ref, reader_source=reference(__file__),
        report=report_ref, start=reference(start_path), parent_receipt=receipt_ref,
        GPU_admission=admission_ref, execution_source_binding=dict(
            admission=admission_ref, spec=spec_ref, entry=admitted_entry_ref, owner_driver=owner_ref,
            phase=spec['phase'], entry_args=['--spec', str(spec_path.resolve())],
            CPU_affinity=[0, 1], CUDA_visible_devices='0', automatic_retry=False,
            deadline_seconds=spec['deadline_seconds'], whole_GPU_used_limit_MiB=8192,
            receipt_admission_SHA_join_verified_before_scientific_access=True),
        initializer_qualification=spec['initializer_qualification'], initializer_source=spec['initializer_source'],
        model_proofs=proofs, checkpointmap={kind: proofs[kind]['checkpoint'] for kind in KINDS},
        terminal_gate_passed_before_scientific_access=True, CPU_affinity=[0, 1], torch_threads=2,
        CUDA_initialized=False, external_CPU_RNG_restored=True, all_raw_GT_preserved=True,
        BN_affine_and_buffers_independently_exact=True, full_R50_conv_active=True,
        training_loss_used_original_GT=True, inference_executed=False, AP_computed=False,
        GT_free_inference_runtime_verified=False, scientific_model_verdict=None,
        formal_training_qualified=False, public_checkpoint_recipe_provenance_verified=False,
        evidence_limits=[
            'Only four native supervised microsteps/two updates are technically qualified; no convergence, AP or superiority claim.',
            'CPU independently verifies initializer mapping, full endpoint tensors, GT rasters and optimizer membership/moments; it does not replay the CUDA training trajectory.',
            'Gradient norms and finite/None observations are source-observed trace evidence, not independently repeated CPU backward.',
            'All raw GT reach native model.loss; this does not imply every GT is selected as a positive by every architecture.',
            'No per-stage positive assignments, native cost matrices, or inference outputs were exported by this producer.',
            'COCO boat8/background80 selection follows the installed official class order and qualified compatibility; full public pretraining provenance is not established.'])
    with target.open('x', encoding='utf-8') as stream:
        json.dump(acceptance, stream, indent=2, sort_keys=True, allow_nan=False)
    print(json.dumps(dict(status=acceptance['status'], acceptance=reference(target)), sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', required=True)
    parser.add_argument('--acceptance', required=True)
    args = parser.parse_args()
    qualify(args.spec, args.acceptance)


if __name__ == '__main__':
    main()
