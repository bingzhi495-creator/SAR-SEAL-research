"""Inert SOURCE for a bounded shared frozen-teacher objective control.

ROOT alone executes after five prior raw terminal gates and actual CPU fixture.
One GT-free native forward per image visit feeds three independent readouts.
Only fixed teacher assignment is cached; features are recomputed each visit.
Uniform integer-pixel BCE and full-raster BCE have the same expectation but
different realized gradients, compute, clipping and AdamW trajectories.
Gain acts after original-size rendering and is an ancillary calibration arm.
The teacher trained on all FIT; heldout means held out from new readout fitting.
No resume, convergence, novelty, SOTA, or causal verdict is provided here.
"""
import argparse
from contextlib import contextmanager
import gc
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import time
import types

SCHEMA = 'ROOT-shared-readout-objective-control-FIT128-spec-v1'
REPORT_SCHEMA = 'shared-readout-objective-control-FIT128-report-v1'
REPORT_STATUS = 'SHARED_READOUT_OBJECTIVE_CONTROL_TRAIN_HOLDOUT_GPU_COMPLETE_NO_SOTA'
ARMS = ('raw_uniform_BCE', 'raw_full_BCE', 'positive_global_gain')
ENDPOINTS = ('native0',) + ARMS
DETAIL = (0, 1, 510, 511)
OWNER_SHA = 'ef9a9f17d4fb1bc3464936095ef799021dd880f8b15488d558a871c08b6e1924'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def reference(path):
    path = Path(path).resolve(strict=True)
    data = path.read_bytes()
    return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def bound(row):
    require(isinstance(row, dict) and Path(row['path']).is_absolute(), 'Absolute pinned reference required')
    require(reference(row['path']) == row, 'Pinned bytes differ: ' + row['path'])
    return Path(row['path']).resolve(strict=True)


def write(path, value):
    path = Path(path)
    require(not path.exists(), 'Fresh JSON artifact required')
    temp = path.with_name(path.name + '.tmp')
    require(not temp.exists(), 'Fresh temporary JSON required')
    temp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def load_source(row, name):
    path = bound(row)
    require(path.suffix == '.py', 'Explicit Python SOURCE required')
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    return module


def per_image_BCE(torch, original_logits, targets, indices=None):
    """Pure per-GT pixel mean then image GT mean; empty G stays connected.

    Indices are actual flattened integer original pixels, with duplicates kept.
    No class, Dice, balancing, ignore, unmatched or earlier-stage loss occurs.
    """
    require(torch.is_tensor(original_logits) and original_logits.ndim == 3 and
        original_logits.dtype == torch.float32 and min(original_logits.shape[1:]) > 0 and
        bool(torch.isfinite(original_logits).all()), 'Finite FP32 GxHxW logits required')
    require(torch.is_tensor(targets) and targets.shape == original_logits.shape and
        targets.device == original_logits.device and bool(torch.isfinite(targets).all()) and
        bool(((targets == 0) | (targets == 1)).all()), 'Same-device original binary targets required')
    G, H, W = original_logits.shape
    if indices is not None:
        require(torch.is_tensor(indices) and indices.dtype == torch.int64 and indices.device == original_logits.device and
            indices.ndim == 2 and indices.shape[0] == G and indices.shape[1] > 0 and
            bool(((indices >= 0) & (indices < H * W)).all()), 'Same-device GxK integer pixel indices required')
    if G == 0:
        return original_logits.sum() * 0.0
    logits = original_logits.reshape(G, H * W)
    target = targets.to(torch.float32).reshape(G, H * W)
    if indices is not None:
        logits, target = logits.gather(1, indices), target.gather(1, indices)
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction='none').mean(1).mean()


def fixed_assignment_rows(raw_ann_ids, pos_inds, assigned_gt_indices):
    """Pure GT-row ordered map from actual native positive/assigned pairs."""
    require(isinstance(raw_ann_ids, (list, tuple)) and isinstance(pos_inds, (list, tuple)) and
        isinstance(assigned_gt_indices, (list, tuple)), 'Explicit native/raw integer lists required')
    require(all(type(x) is int for x in list(raw_ann_ids) + list(pos_inds) + list(assigned_gt_indices)) and
        len(set(raw_ann_ids)) == len(raw_ann_ids) and len(pos_inds) == len(assigned_gt_indices) == len(raw_ann_ids) and
        len(set(pos_inds)) == len(pos_inds) and all(0 <= q < 100 for q in pos_inds) and
        sorted(assigned_gt_indices) == list(range(len(raw_ann_ids))), 'Complete unique native raw-GT assignment required')
    by_gt = {j: q for q, j in zip(pos_inds, assigned_gt_indices)}
    return [dict(annotation_id=a, GT_index=j, query_index=by_gt[j]) for j, a in enumerate(raw_ann_ids)]


def score_gain(torch, native_original, log_gain):
    """Pure scalar exp gain AFTER native rendering, preserving autograd."""
    require(torch.is_tensor(native_original) and native_original.dtype == torch.float32 and
        (native_original.numel() > 0 or (native_original.ndim == 3 and min(native_original.shape[1:]) > 0)) and
        bool(torch.isfinite(native_original).all()) and
        torch.is_tensor(log_gain) and log_gain.shape == torch.Size([]) and log_gain.dtype == torch.float32 and
        log_gain.device == native_original.device and bool(torch.isfinite(log_gain)), 'Finite original logits and scalar log_gain required')
    gain = log_gain.exp()
    require(bool(torch.isfinite(gain)) and bool(gain > 0), 'Actual exp gain must be finite and positive')
    result = native_original * gain
    require(bool(torch.isfinite(result).all()), 'Scaled original logits must remain finite')
    return result


def cpu_copy(torch, value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_copy(torch, item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_copy(torch, item) for item in value)
    return value


def packet(torch, path, value):
    require(not Path(path).exists(), 'Fresh CPU audit/checkpoint packet required')
    torch.save(cpu_copy(torch, value), str(path))
    return reference(path)


def records(pref, state):
    return {name: pref.digest_tensor(value) for name, value in state.items()}


def raw_terminal(receipt, phase):
    require(receipt['status'] == 'ACTUAL_OWNED_TERMINAL' and receipt['original_identity'] is not None and
        receipt['returncode'] == 0 and type(receipt['returncode']) is int and receipt['root_wait_reaped'] is True and
        receipt['held_handle_exit'] == dict(wait_state='WAIT_OBJECT_0', actual_exit_code=0) and
        receipt['numeric_birth_absence']['original_birth_absent'] is True and receipt['abort'] is None and
        receipt['error'] is None and receipt['wait_errors'] is False and receipt['GPU_observation_errors'] == [],
        'STOP before hashes/imports: actual raw held0/reap/birth absence/noerror required')
    require(receipt.get('phase', phase) == phase and receipt['whole_GPU_used_limit_MiB'] == 8192 and
        receipt['whole_leaf_VRAM_certified'] is False and
        receipt['GPU_scope'] == 'whole GPU observed, not per-process WDDM allocation' and
        type(receipt['whole_GPU_observed_peak_MiB']) in (int, float) and
        math.isfinite(receipt['whole_GPU_observed_peak_MiB']) and 0 <= receipt['whole_GPU_observed_peak_MiB'] <= 8192,
        'Raw phase/observed whole GPU scope differs')


def owner_gate(spec_path, spec):
    admission = read(spec['GPU_admission_file'])
    source, spec_ref = reference(__file__), reference(spec_path)
    require(admission['status'] == 'ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE' and
        admission['automatic_retry'] is False and admission['phase'] == spec['phase'] and
        admission['driver_sha256'] == OWNER_SHA and admission['allowed_cpus'] == spec['CPU_affinity'] == [0, 1] and
        admission['cuda_visible_devices'] == '0' and admission['qualified_leaf_deadline_seconds'] == spec['deadline_seconds'] == 5400 and
        admission['whole_GPU_used_limit_MiB'] == spec['whole_GPU_owner_limit_MiB'] == 8192 and
        Path(admission['entry']).resolve(strict=True) == Path(source['path']) and
        admission['entry_args'] == ['--spec', str(spec_path), '--owner-execute'], 'Current actual owner/entry/resources differ')
    hashes = admission['bound_source_sha256']
    require(spec['entry_source'] == source and source in spec['sources'] and spec['CPU_reader'] in spec['sources'] and
        hashes.get(str(spec_path)) == spec_ref['sha256'], 'Current SPEC/entry/reader pins absent')
    for row in spec['sources']:
        require(hashes.get(row['path']) == row['sha256'], 'Complete SOURCE/input closure must be owner-bound')
        bound(row)
    require(admission.get('scientific_spec', spec_ref) == spec_ref, 'Owner scientific SPEC differs')
    return spec_ref


def prior_gate(spec, contract, raw):
    for role in ('prior_diag_spec', 'prior_diag_CPU_qualification', 'prior_diag_receipt', 'prior_diag_phase',
                 'prior_diag_entry_source', 'prior_diag_CPU_reader_source', 'prior_diag_contract'):
        require(spec[role] == contract[role], 'Canonical actual diagnostic lineage differs: ' + role)
    prior = read(bound(spec['prior_diag_spec']))
    q = read(bound(spec['prior_diag_CPU_qualification']))
    require(q['schema'] == 'ROOT-frozen-feature-linear-readout-FIT-CPU-qualification-v1' and
        q['status'] == 'ROOT_FROZEN_FEATURE_LINEAR_READOUT_FIT_CPU_QUALIFIED' and
        q['source'] == prior['entry_source'] == spec['prior_diag_entry_source'] and
        q['reader_source'] == prior['CPU_reader'] == spec['prior_diag_CPU_reader_source'] and
        q['spec'] == spec['prior_diag_spec'] and q['parent_receipt'] == spec['prior_diag_receipt'] and
        q['SOURCE_contract'] == prior['SOURCE_contract'] == spec['prior_diag_contract'] and
        q['phase'] == prior['phase'] == spec['prior_diag_phase'] and
        q['exact_full_teacher610_and_typed_BN106_affine159_buffers_independently_CPU_checked'] is True and
        q['all_raw_selected_GT_including_empty_crowd_ignore_retained'] is True and
        q['native100_actual_GPU_replay_TorchEqual_source_observed'] is True and q['AP_computed'] is False,
        'Actual independently CPU qualified diagnostic required')
    admission = read(bound(q['GPU_admission']))
    require(raw['admission_sha256'] == q['GPU_admission']['sha256'] and admission['driver_sha256'] == OWNER_SHA and
        admission['phase'] == q['phase'] and admission['automatic_retry'] is False and
        Path(prior['GPU_admission_file']).resolve(strict=True) == Path(q['GPU_admission']['path']).resolve(strict=True) and
        admission['bound_source_sha256'].get(q['spec']['path']) == q['spec']['sha256'] and
        all(admission['bound_source_sha256'].get(row['path']) == row['sha256'] for row in prior['sources']),
        'Original diagnostic raw/admission/SPEC join differs')
    require(len(prior['sources']) == 83 and all(row in spec['sources'] for row in prior['sources']), 'Entire qualified diagnostic83 closure required')
    for role, row in contract['base_role_SOURCE_refs'].items():
        if role not in ('entry_source', 'CPU_reader'):
            require(prior[role] == spec[role] == row, 'Preserved original helper role differs: ' + role)
    for role in ('checkpointmap', 'qualificationmap', 'teacher_checkpoint', 'teacher_qualification', 'teacher_training_spec',
                 'teacher_receipt', 'teacher_phase', 'preflight_CPU_qualification', 'preflight_training_spec', 'preflight_receipt',
                 'preflight_phase', 'short2_CPU_qualification', 'short2_training_spec', 'short2_receipt', 'short2_phase',
                 'prior_DEV_spec', 'prior_DEV_CPU_qualification', 'prior_DEV_receipt', 'prior_DEV_phase', 'FIT_annotation', 'image_root'):
        require(prior[role] == spec[role], 'Preserved qualified teacher lineage/input differs: ' + role)
    for role in ('source', 'reader_source', 'report', 'start', 'GPU_qualification', 'owner_driver'):
        bound(q[role])
    return load_source(spec['prior_diag_entry_source'], 'sealed_original_renderer_and_ancestry'), prior


def fixture_gate(spec, contract):
    for role in ('control_CPU_fixture', 'control_CPU_fixture_spec', 'control_CPU_fixture_source'):
        require(spec[role] in spec['sources'], 'Actual CPU fixture must be owner-bound')
    manifest = read(bound(spec['control_CPU_fixture_spec']))
    fixture = read(bound(spec['control_CPU_fixture']))
    sources = dict(producer=spec['entry_source'], fixture=spec['control_CPU_fixture_source'], contract=spec['SOURCE_contract'])
    execution = dict(device='cpu', cpu_affinity=[0, 1], torch_threads=2, torch_interop_threads=2, cuda_visible_devices='')
    helpers = dict(readout_source=spec['readout_source'], geometry_source=spec['geometry_source'], renderer_source=spec['prior_diag_entry_source'])
    require(manifest == dict(schema=contract['fixture_schema'], inputs='SOURCE_ONLY', sources=sources, execution=execution) and
        fixture['schema'] == contract['fixture_schema'] and fixture['status'] == contract['fixture_status'] and fixture['sources'] == sources and fixture['execution'] == execution and
        fixture['synthetic_execution_performed'] is True and fixture['torch_version'] == '2.0.1+cu118' and
        Path(fixture['spec_path']).resolve(strict=True) == Path(spec['control_CPU_fixture_spec']['path']).resolve(strict=True) and
        fixture['helper_SOURCE_refs'] == helpers and fixture['library_SOURCE_refs'] == contract['library_SOURCE_refs'],
        'Actual SOURCE-only CPU fixture PASS/manifest/helper joins differ')
    for flag in ('independent_scalar_BCE_and_coefficient_gradients', 'duplicate_12544_indices_preserved',
            'same_expected_full_BCE_enumeration', 'per_GT_then_image_then_accum2_normalization',
            'private_CPU_sampling_RNG_restored', 'connected_empty_GT', 'connected_empty_gain', 'fixed_assignment_raw_ann_mapping_and_rejections',
            'raw_cold_warm_two_weight_gradients', 'frozen_F_e0_unchanged_and_no_grad',
            'gain_after_original_renderer_gradient', 'positive_gain_binary_examples',
            'quality_score_ranking_change', 'empty_prediction_quality_zero', 'external_CPU_RNG_unchanged'):
        require(fixture[flag] is True, 'Actual synthetic fixture check absent: ' + flag)
    require(fixture['CUDA_initialized'] is False and fixture['native_model_integration_qualified'] is False and
        fixture['Hungarian_optimality_qualified'] is False and fixture['AP_computed'] is False,
        'Synthetic fixture scope differs')
    return fixture


def source_file(obj, row):
    require(Path(inspect.getsourcefile(obj)).resolve(strict=True) == bound(row).resolve(strict=True), 'Imported implementation SOURCE differs')


def original_targets(torch, np, coco, identity, expected, data, head):
    import cv2
    ids = identity['ann_ids']
    require(ids == expected['annotation_ids'] and identity['dataset_index'] == expected['dataset_index'], 'Raw parser ID/order differs')
    H, W = identity['original_shape']
    rasters = []
    for ann in coco.load_anns(ids):
        mask = np.zeros((H, W), dtype=np.uint8) if ann.get('segmentation') == [] else coco.annToMask(ann)
        require(mask.shape == (H, W) and bool(((mask == 0) | (mask == 1)).all()), 'Original raw binary GT required')
        rasters.append(mask)
    array = np.stack(rasters) if rasters else np.zeros((0, H, W), dtype=np.uint8)
    processed = head.preprocess_gt([data['data_samples'][0].gt_instances], [None])[0]
    h, w = identity['actual_img_shape']
    padded = np.zeros((len(ids), 800, 800), dtype=np.int64)
    for j, mask in enumerate(rasters):
        padded[j, :h, :w] = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    require(len(processed) == len(ids) and bool((processed.labels == 0).all()) and
        torch.equal(processed.masks, torch.from_numpy(padded).to(processed.masks.device)), 'Actual preprocessed native GT row order/raster differs from all raw annotations')
    return torch.from_numpy(array).to(device=data['inputs'].device, dtype=torch.float32), processed


def teacher_forward(torch, pref, core, model, data, samples, image_id, detailed=False):
    head = model.panoptic_head
    outside, inventory = pref.rng_snapshot(torch), core.hook_inventory(model)
    calls, captured = dict(feature=0, embedding=0), {}
    def feature(module, args, output):
        calls['feature'] += 1
        captured['F'] = output[0].detach()
    def embedding(module, args, output):
        calls['embedding'] += 1
        if calls['embedding'] == 10:
            captured.update(h=args[0].detach(), e0=output.detach())
    handles = [head.pixel_decoder.register_forward_hook(feature), head.mask_embed.register_forward_hook(embedding)]
    try:
        with torch.random.fork_rng(devices=[0]), torch.no_grad():
            seed = 100000 + image_id
            torch.random.default_generator.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            before = pref.rng_snapshot(torch)
            features = model.extract_feat(data['inputs'])
            classes, masks = head(features, samples)
            captured.update(classes=classes[-1].detach(), M200=masks[-1].detach())
            after = pref.rng_snapshot(torch)
        require(calls == dict(feature=1, embedding=10) and len(classes) == len(masks) == 10 and
            tuple(captured['F'].shape) == (1, 256, 200, 200) and tuple(captured['h'].shape) == tuple(captured['e0'].shape) == (1, 100, 256) and
            tuple(captured['classes'].shape) == (1, 100, 2) and tuple(captured['M200'].shape) == (1, 100, 200, 200) and
            all(value.dtype == torch.float32 and not value.requires_grad and bool(torch.isfinite(value).all()) for value in captured.values()) and
            torch.equal(torch.einsum('bqc,bchw->bqhw', captured['e0'], captured['F']), captured['M200']) and
            core.equal_rng(torch, outside, pref.rng_snapshot(torch)), 'One GT-free native final100 shared capture/replay/RNG required')
        require(all(not v.requires_grad and v.dtype == torch.float32 for v in classes + masks), 'All native stages must be detached FP32')
    finally:
        for handle in handles:
            handle.remove()
        handles.clear()
    require(core.hook_inventory(model) == inventory, 'Passive teacher hooks leaked')
    protected = dict(tensors=classes + masks, versions=[value._version for value in classes + masks])
    return captured, dict(head_calls=1, backbone_calls=1, pixel_decoder_calls=1, mask_embed_calls=10,
        seed=seed, private_RNG_before=[pref.digest_tensor(v) for v in before], private_RNG_after=[pref.digest_tensor(v) for v in after],
        external_RNG_restored=True, hooks_removed=True, all4_GT_fields_absent=True, classes_all10_detached=True,
        first9_masks_detached=True, native_all10_tensor_versions_unchanged=None,
        classes10=[pref.digest_tensor(v) for v in classes] if detailed else None,
        native_masks10=[pref.digest_tensor(v) for v in masks] if detailed else None), protected


@contextmanager
def observed_RAW_inputs(module):
    """Observe actual hidden inputs without replacing any readout computation."""
    chunks = []
    before = list(module.hidden._forward_pre_hooks)
    def capture(current, args):
        chunks.append(args[0].detach())
    handle = module.hidden.register_forward_pre_hook(capture)
    try:
        yield chunks
    finally:
        handle.remove()
        require(list(module.hidden._forward_pre_hooks) == before, 'RAW input observer leaked')


def fixed_matching(torch, pref, core, old, head, fields, gt, identity):
    outside = pref.rng_snapshot(torch)
    with torch.random.fork_rng(devices=[0]), torch.no_grad():
        seed = 2800000 + identity['image_id']
        torch.random.default_generator.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        before = pref.rng_snapshot(torch)
        with old.observe_targets(head) as matches, old.observe_native_costs(head) as costs:
            targets = head.get_targets([fields['classes'][0]], [fields['M200'][0]], [gt], [identity['metainfo']])
        after = pref.rng_snapshot(torch)
    require(len(targets) == 5 and len(matches) == 1 and core.equal_rng(torch, outside, pref.rng_snapshot(torch)), 'One defaultFalse teacher target call/RNG restoration required')
    observed = matches[0]
    rows = fixed_assignment_rows(identity['ann_ids'], observed['positive_query_indices'], observed['assigned_GT_indices'])
    require(torch.equal(targets[2][0], gt.masks[observed['assigned_GT_indices']]) and
        [c['cost_index'] for c in costs['components']] == [0, 1, 2] and len(costs['totals']) == 1,
        'Native complete targets/three cost parts/actual SciPy sum required')
    parts = [c['tensor'] for c in costs['components']]
    total = costs['totals'][0]
    require(all(tuple(v.shape) == (100, len(rows)) and v.dtype == torch.float32 and bool(torch.isfinite(v).all()) for v in parts + [total]), 'Native finite weighted cost matrix differs')
    return dict(image_id=identity['image_id'], raw_ann_ids=identity['ann_ids'], rows=rows,
        positive_query_indices=observed['positive_query_indices'], assigned_GT_indices=observed['assigned_GT_indices'],
        query_by_GT=[r['query_index'] for r in rows], cost_parts=[v.tolist() for v in parts], actual_total=total.tolist(),
        cost_digests=[pref.digest_tensor(v) for v in parts], actual_total_digest=pref.digest_tensor(total), weights=[2.0, 5.0, 5.0],
        actual_total_is_native_GPU_stack_sum_then_CPU=True, CPU_cost_derivation_claimed=False,
        seed=seed, points=12544, native_get_targets_calls=1, external_RNG_restored=True,
        private_RNG_before=[pref.digest_tensor(v) for v in before], private_RNG_after=[pref.digest_tensor(v) for v in after])


def arm_states(torch, modules):
    return {arm: cpu_copy(torch, module.state_dict()) for arm, module in modules.items()}


def gradient_records(torch, pref, module):
    result = {}
    for name, parameter in module.named_parameters():
        require(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()), 'Finite nonNone readout gradients required')
        result[name] = dict(digest=pref.digest_tensor(parameter.grad), norm=float(torch.linalg.vector_norm(parameter.grad).item()))
    return result


def export_original(torch, np, masks_api, fusion, classes, original, image_id, pref):
    require(fusion.num_classes == fusion.num_things_classes == 1 and fusion.num_stuff_classes == 0 and
        fusion.test_cfg['max_per_image'] == 100, 'Official single-class all100 instance fusion required')
    instances = fusion.instance_postprocess(classes[0], original)
    probabilities = torch.nn.functional.softmax(classes[0], dim=-1)[:, :-1]
    top, indices = probabilities.flatten(0, 1).topk(100, sorted=False)
    qids = indices // fusion.num_classes
    ordered = original[qids]
    binary = ordered > 0
    quality = (ordered.sigmoid() * binary).flatten(1).sum(1) / (binary.flatten(1).sum(1) + 1e-6)
    scores = top * quality
    require(torch.equal(qids.sort().values, torch.arange(100, device=qids.device)) and len(instances) == 100 and
        torch.equal(instances.masks, binary) and torch.equal(instances.scores, scores) and bool((instances.labels == 0).all()) and
        bool(torch.isfinite(scores).all()) and bool(((scores >= 0) & (scores <= 1)).all()), 'Actual official original-space fusion TorchEqual failed')
    predictions, factors = [], []
    for q, mask, p, m, score in zip(qids.cpu().tolist(), instances.masks.cpu().numpy().astype(np.uint8), top.cpu().tolist(), quality.cpu().tolist(), scores.cpu().tolist()):
        rle = masks_api.encode(np.asfortranarray(mask))
        counts = rle['counts'].decode('ascii') if isinstance(rle['counts'], bytes) else rle['counts']
        predictions.append(dict(image_id=image_id, query_id=q, category_id=1, score=float(score), segmentation=dict(size=list(rle['size']), counts=counts)))
        factors.append(dict(image_id=image_id, query_id=q, class_probability=float(p), mask_quality=float(m), official_score=float(score)))
    return predictions, factors, qids.cpu().tolist(), dict(pref.digest_tensor(classes[0]), values=classes[0].cpu().tolist())


def execute(torch, np, masks_api, core, pref, old, native, diag, base, hooks, model, anchor, loading, modules, optimizers,
            spec, spec_ref, contract, dataset, coco, output, started):
    deadline = lambda: require(time.monotonic() - started < 5400, 'Qualified fixed deadline exceeded; no retry')
    teacher_names = list(dict(model.named_parameters()))
    bn = [(name, module) for name, module in model.named_modules() if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)]
    bn_parameters = [name + '.' + key for name, module in bn for key, value in module.named_parameters(recurse=False)]
    bn_buffers = [name + '.' + key for name, module in bn for key, value in module.named_buffers(recurse=False)]
    require(len(teacher_names) == 451 and len(bn) == 53 and len(bn_parameters) == 106 and len(bn_buffers) == 159, 'Typed native parameter/BN counts differ')
    initial_states = arm_states(torch, modules)
    require(all(torch.equal(initial_states[ARMS[0]][k], initial_states[ARMS[1]][k]) for k in initial_states[ARMS[0]]) and
        all(len(optimizer.state) == 0 for optimizer in optimizers.values()), 'Fresh bitexact RAW initialization/empty independent AdamW required')
    model.to('cuda:0').eval()
    for module in modules.values():
        module.to('cuda:0')
    order_generator = torch.Generator(device='cpu').manual_seed(2026101021)
    orders, trace, audit_refs, matches, update_counts = [], [], {}, {}, {arm: 0 for arm in ARMS}
    step_grad_counts = {arm: {name: 0 for name, p in module.named_parameters()} for arm, module in modules.items()}
    train_rows = {row['image_id']: row for row in contract['split']['train_rows']}
    for optimizer in optimizers.values():
        optimizer.zero_grad(set_to_none=True)
    for epoch in range(4):
        before_order = order_generator.get_state().clone()
        order = [contract['split']['train_ids'][i] for i in torch.randperm(128, generator=order_generator).tolist()]
        orders.append(dict(epoch=epoch, image_ids=order, RNG_before=pref.digest_tensor(before_order), RNG_after=pref.digest_tensor(order_generator.get_state())))
        for offset, image_id in enumerate(order):
            deadline()
            micro = epoch * 128 + offset
            data, samples, identity = core.image_data(native, old, model, dataset, coco, image_id)
            require(all(not any(key in sample for key in core.GT_FIELDS) for sample in samples), 'GT entered teacher neural forward')
            fields, forward, protected = teacher_forward(torch, pref, core, model, data, samples, image_id, micro in DETAIL)
            identity['metainfo'] = dict(img_shape=list(identity['actual_img_shape']), batch_input_shape=[800, 800], ori_shape=list(identity['original_shape']))
            targets, processed = original_targets(torch, np, coco, identity, train_rows[image_id], data, model.panoptic_head)
            if image_id not in matches:
                matches[image_id] = fixed_matching(torch, pref, core, old, model.panoptic_head, fields, processed, identity)
            match = matches[image_id]
            positive = torch.tensor(match['query_by_GT'], dtype=torch.long, device='cuda:0')
            valid = hooks.metadata_valid_fraction(torch, [samples[0].metainfo], fields['F'])
            outside_CPU = torch.get_rng_state().clone()
            generator = torch.Generator(device='cpu').manual_seed(3800000 + micro)
            indices_CPU = torch.randint(targets.shape[1] * targets.shape[2], (len(identity['ann_ids']), 12544), generator=generator, dtype=torch.int64)
            indices = indices_CPU.to('cuda:0')
            require(torch.equal(outside_CPU, torch.get_rng_state()), 'Private integer sampling altered ambient CPU RNG')
            detailed = micro in DETAIL
            audit = dict(schema='shared-readout-objective-control-audit-v1', global_microstep=micro, epoch=epoch, image_id=image_id,
                identity=identity, matching=match, teacher=fields, valid_fraction=valid, original_targets=targets, indices=indices_CPU,
                RAWstates_before=arm_states(torch, modules),
                optimizer_states_before={arm: cpu_copy(torch, value.state_dict()) for arm, value in optimizers.items()}, arms={}) if detailed else None
            row = dict(global_microstep=micro, epoch=epoch, offset=offset, image_id=image_id, identity=identity,
                raw_GT=len(identity['ann_ids']), input_digest=pref.digest_tensor(data['inputs']), forward=forward,
                matching_image_id=image_id, matching_reused=(epoch > 0), indices_digest=pref.digest_tensor(indices_CPU),
                integer_pixel_seed=3800000 + micro, integer_pixel_draws=len(identity['ann_ids']) * 12544, sampled_with_replacement=True,
                shared_native_fields=records(pref, fields) if detailed else None, teacher_all451_grad_None=True, all_teacher_modules_eval=True,
                pending_accumulation_before=micro % 2, pending_accumulation_after=(micro + 1) % 2, arms={})
            with torch.no_grad():
                native_original = (diag.render_original(torch, fields['M200'], identity).index_select(0, positive)
                    if positive.numel() == 0 else diag.render_original(torch, fields['M200'].index_select(1, positive), identity))
            for arm in ARMS:
                module, optimizer = modules[arm], optimizers[arm]
                parameter_before = records(pref, module.state_dict())
                pending_before = {name: None if p.grad is None else cpu_copy(torch, p.grad) for name, p in module.named_parameters()} if detailed else None
                if arm == 'positive_global_gain':
                    original = score_gain(torch, native_original, module.log_gain)
                    result = None
                else:
                    if detailed:
                        with observed_RAW_inputs(module) as chunks:
                            result = module(fields['F'], fields['e0'], valid, native_logits=fields['M200'])
                        h = torch.cat(chunks, dim=1)
                        eligible = result['diagnostics']['eligible']
                        require(tuple(h.shape) == (1, 100, 16, 512) and not h.requires_grad and bool(torch.isfinite(h).all()) and
                            tuple(eligible.shape) == (1, 100, 16) and eligible.dtype == torch.bool,
                            'Actual RAW SiLU input/eligibility packet differs')
                        if 'RAW_context' not in audit:
                            audit['RAW_context'] = cpu_copy(torch, dict(h=h, eligible=eligible, activation='silu', actual_hidden_pre_hook_observed=True))
                        else:
                            require(torch.equal(h.cpu(), audit['RAW_context']['h']) and torch.equal(eligible.cpu(), audit['RAW_context']['eligible']),
                                'Shared native RAW context differs between arms')
                        del chunks, h, eligible
                    else:
                        result = module(fields['F'], fields['e0'], valid, native_logits=fields['M200'])
                    original = (diag.render_original(torch, result['mask_logits'], identity).index_select(0, positive)
                        if positive.numel() == 0 else diag.render_original(torch, result['mask_logits'].index_select(1, positive), identity))
                used_indices = indices if arm == 'raw_uniform_BCE' else None
                loss = per_image_BCE(torch, original, targets, used_indices)
                require(loss.requires_grad and bool(torch.isfinite(loss)), 'Finite final-only BCE live readout graph required')
                micro_grad = torch.autograd.grad(loss / 2.0, list(module.parameters()), retain_graph=True) if detailed else None
                (loss / 2.0).backward()
                gradients = gradient_records(torch, pref, module)
                accumulated = {name: cpu_copy(torch, p.grad) for name, p in module.named_parameters()} if detailed else None
                step = (micro + 1) % 2 == 0
                norm = None
                clipped = None
                if step:
                    norm = float(torch.nn.utils.clip_grad_norm_(list(module.parameters()), 0.01, error_if_nonfinite=True).item())
                    require(math.isfinite(norm) and norm >= 0, 'Finite nonnegative clipping norm required')
                    clipped = gradient_records(torch, pref, module)
                    for name, p in module.named_parameters():
                        step_grad_counts[arm][name] += int(p.grad is not None)
                    optimizer.step()
                    update_counts[arm] += 1
                after = records(pref, module.state_dict())
                per_GT = targets.new_empty((0,)) if len(targets) == 0 else torch.nn.functional.binary_cross_entropy_with_logits(
                    original.reshape(len(targets), -1).gather(1, indices) if used_indices is not None else original.reshape(len(targets), -1),
                    targets.reshape(len(targets), -1).gather(1, indices) if used_indices is not None else targets.reshape(len(targets), -1), reduction='none').mean(1)
                row['arms'][arm] = dict(loss=float(loss.item()), per_GT_loss=per_GT.detach().cpu().tolist(),
                    supervision='uniform_original_integer_pixels' if used_indices is not None else 'full_original_raster',
                    pixels_per_GT=12544 if used_indices is not None else int(targets.shape[1] * targets.shape[2]),
                    draw_count=int(len(targets) * (12544 if used_indices is not None else targets.shape[1] * targets.shape[2])),
                    parameter_before=parameter_before, accumulated_gradients_before_clip=gradients,
                    clipped_gradients=clipped, clip_norm_before=norm, optimizer_update=step, updates_completed=update_counts[arm], parameter_after=after,
                    cold_hidden_gradient_exact_zero=(bool(torch.count_nonzero(module.hidden.weight.grad) == 0) if micro < 2 and arm != 'positive_global_gain' else None))
                if detailed:
                    audit['arms'][arm] = dict(original_logits=original, pending_gradients_before=pending_before,
                        micro_gradients={name: value for (name, p), value in zip(module.named_parameters(), micro_grad)},
                        accumulated_gradients_before_clip=accumulated,
                        clipped_gradients=None if not step else {name: p.grad for name, p in module.named_parameters()},
                        readout_result=result, parameters_after=cpu_copy(torch, module.state_dict()),
                        optimizer_state_after=cpu_copy(torch, optimizer.state_dict()))
                    audit['arms'][arm] = cpu_copy(torch, audit['arms'][arm])
                if step:
                    optimizer.zero_grad(set_to_none=True)
                del original, loss, per_GT, result, micro_grad, pending_before, accumulated
            require(all(p.grad is None for p in model.parameters()) and all(not p.requires_grad for p in model.parameters()) and
                all(not m.training for m in model.modules()) and [v._version for v in protected['tensors']] == protected['versions'],
                'Native parameter gradients/trainability/module eval/stage versions changed')
            forward['native_all10_tensor_versions_unchanged'] = True
            if detailed:
                audit_refs[str(micro)] = packet(torch, output / ('AUDIT_%04d.pth' % micro), audit)
            trace.append(row)
            del data, samples, targets, processed, fields, valid, native_original, indices, indices_CPU, audit, protected
            if (micro + 1) % 64 == 0:
                print(json.dumps(dict(completed_microsteps=micro + 1, updates_per_arm=update_counts)), flush=True)
    require(update_counts == {arm: 256 for arm in ARMS} and len(trace) == 512 and
        all(p.grad is None for module in modules.values() for p in module.parameters()), 'Exact E4 counters/no pending accumulation required')
    checkpoints = {}
    for arm in ARMS:
        module, optimizer = modules[arm], optimizers[arm]
        names = [[name] for name, p in module.named_parameters()]
        require(len(optimizer.state) == len(names) and all(float(value['step'].item()) == 256 for value in optimizer.state.values()), 'Fresh AdamW readout-only moments must have step256')
        checkpoints[arm] = packet(torch, output / (arm + '_E4.pth'), dict(schema='shared-readout-objective-control-checkpoint-v1',
            readout_state_dict=module.state_dict(), optimizer_state_dict=optimizer.state_dict(), meta=dict(arm=arm, completed_epochs=4,
                microsteps=512, optimizer_updates=256, raw_GT_visits=1168, optimizer_param_names=names,
                phase=spec['phase'], spec=spec_ref, teacher_qualification=spec['teacher_qualification'],
                teacher_checkpoint=spec['teacher_checkpoint'], source=spec['entry_source'], SOURCE_contract=spec['SOURCE_contract'],
                teacher_mode='native0', teacher_epoch=24, frozen_native=True, final_only_BCE=True, formal_resumable=False,
                teacher_optimizer_loaded=False, teacher_order_RNG_loaded=False, preflight_or_SHORT2_CP_loaded=False,
                gain_after_original_renderer=(arm == 'positive_global_gain'), capacity_or_optimizer_matched=(arm != 'positive_global_gain')),
            order_generator_state=order_generator.get_state(), fixed_assignment_rows=matches))
    write(output / 'TRACE.json', trace)
    write(output / 'ORDER.json', orders)
    write(output / 'MATCHING.json', matches)
    endpoints = {mode: dict(predictions=[], score_factors=[], image_rows=[]) for mode in ENDPOINTS}
    heldout_matches = {}
    for expected in contract['split']['readout_heldout_rows']:
        deadline()
        image_id = expected['image_id']
        data, samples, identity = core.image_data(native, old, model, dataset, coco, image_id)
        fields, forward, protected = teacher_forward(torch, pref, core, model, data, samples, image_id)
        identity['metainfo'] = dict(img_shape=list(identity['actual_img_shape']), batch_input_shape=[800, 800], ori_shape=list(identity['original_shape']))
        targets, processed = original_targets(torch, np, coco, identity, expected, data, model.panoptic_head)
        heldout_matches[image_id] = fixed_matching(torch, pref, core, old, model.panoptic_head, fields, processed, identity)
        valid = hooks.metadata_valid_fraction(torch, [samples[0].metainfo], fields['F'])
        with torch.no_grad():
            native_original = diag.render_original(torch, fields['M200'], identity)
            native_preds, native_factors, native_qids, _, _ = base.official_predictions(torch, np, masks_api, model,
                fields['classes'], fields['M200'], samples, identity, pref)
            for mode in ENDPOINTS:
                result = None
                if mode == 'native0':
                    original = native_original
                elif mode == 'positive_global_gain':
                    original = score_gain(torch, native_original, modules[mode].log_gain)
                    require(torch.equal(original > 0, native_original > 0), 'Actual positive gain binary-mask agreement required; no universal FP32 claim')
                else:
                    result = modules[mode](fields['F'], fields['e0'], valid, native_logits=fields['M200'])
                    original = diag.render_original(torch, result['mask_logits'], identity)
                preds, factors, qids, class_packet = export_original(torch, np, masks_api, model.panoptic_fusion_head, fields['classes'], original, image_id, pref)
                if mode == 'native0':
                    require(preds == native_preds and factors == native_factors and qids == native_qids, 'Direct original-space export differs from actual official native fusion')
                endpoints[mode]['predictions'].extend(preds)
                endpoints[mode]['score_factors'].extend(factors)
                endpoints[mode]['image_rows'].append(dict(mode=mode, image_id=image_id, identity=identity, forward=forward,
                    GT=identity['native_GT'] if 'native_GT' in identity else pref.digest_tensor(processed.masks),
                    input_digest=pref.digest_tensor(data['inputs']), native_fields=records(pref, fields), native_class_packet=class_packet,
                    query_ids=qids, original_logits=pref.digest_tensor(original), masks200=pref.digest_tensor(fields['M200']) if result is None else pref.digest_tensor(result['mask_logits']),
                    official_fusion_mask_score_query_TorchEqual=True, all100_queries_retained=True, GT_free_forward=True,
                    gain=None if mode != 'positive_global_gain' else float(modules[mode].log_gain.exp().item()),
                    gain_binary_mask_agreement=True if mode == 'positive_global_gain' else None,
                    readout_diagnostics=None if result is None else {key: (pref.digest_tensor(value) if torch.is_tensor(value) else value) for key, value in result['diagnostics'].items()}))
                del original, result
        require([v._version for v in protected['tensors']] == protected['versions'], 'Heldout native stage tensors changed')
        forward['native_all10_tensor_versions_unchanged'] = True
        del data, samples, targets, processed, fields, native_original, valid, protected
    write(output / 'HELDOUT_MATCHING.json', heldout_matches)
    mode_rows = {}
    for mode, values in endpoints.items():
        require(len(values['image_rows']) == 64 and len(values['predictions']) == len(values['score_factors']) == 6400, 'All64/all100 endpoint population differs')
        row = {}
        for key, value in values.items():
            path = output / (mode + '_' + key.upper() + '.json')
            write(path, value)
            row[key] = reference(path)
        row['checkpoint'] = spec['teacher_checkpoint'] if mode == 'native0' else checkpoints[mode]
        mode_rows[mode] = row
    model.cpu()
    final_native = model.state_dict()
    require(set(final_native) == set(anchor) and all(torch.equal(final_native[k], anchor[k]) for k in anchor), 'Every native610 state must remain CPU exact')
    proof = dict(schema='shared-readout-objective-control-state-proof-v1', native_teacher610=records(pref, anchor),
        before=records(pref, anchor), after=records(pref, final_native), teacher_loading=loading, native_parameter_names=teacher_names,
        BN_module_names=[name for name, module in bn], BN_parameter_names=bn_parameters, BN_buffer_names=bn_buffers,
        full_native610_CPU_exact=True, frozen_BN_all_exact=True, all_native451_frozen=True, all_modules_eval=True,
        native_parameter_grads_None=True, hook_cache_clear=True, GT_free_forward=True, teacher_forward_calls=576,
        teacher_optimizer_loaded=False, teacher_order_RNG_loaded=False, feature_cache_used=False,
        initial_readout_states=records(pref, initial_states[ARMS[0]]), initial_gain_state=records(pref, initial_states['positive_global_gain']),
        final_readout_states={arm: records(pref, module.state_dict()) for arm, module in modules.items()},
        fresh_optimizer_state_entries={arm: 0 for arm in ARMS}, optimizer_step_nonNone_gradient_counts=step_grad_counts,
        completed_epochs=4, microsteps=512, optimizer_updates=update_counts, no_pending_accumulation_gradients=True,
        audit_microsteps=list(DETAIL), audit_refs=audit_refs, training_raw_GT=292, training_raw_GT_visits_per_arm=1168,
        readout_heldout_raw_GT=106, teacher_trained_all_selected_FIT=True, blind_holdout_claimed=False)
    write(output / 'STATE_PROOF.json', proof)
    for module in modules.values():
        module.cpu()
    return dict(mode_rows=mode_rows, checkpoints=checkpoints, trace=reference(output / 'TRACE.json'), order=reference(output / 'ORDER.json'),
        matching=reference(output / 'MATCHING.json'), heldout_matching=reference(output / 'HELDOUT_MATCHING.json'),
        state_proof=reference(output / 'STATE_PROOF.json'), audits=audit_refs)


def run(spec_file):
    started = time.monotonic()
    require(Path(spec_file).is_absolute(), 'Absolute --spec required')
    spec_path = Path(spec_file).resolve(strict=True)
    spec = read(spec_path)  # Metadata only until all five raw gates complete.
    raw = {}
    for receipt_role, phase_role in (('teacher_receipt', 'teacher_phase'), ('preflight_receipt', 'preflight_phase'),
            ('short2_receipt', 'short2_phase'), ('prior_DEV_receipt', 'prior_DEV_phase'), ('prior_diag_receipt', 'prior_diag_phase')):
        require(Path(spec[receipt_role]['path']).is_absolute(), 'Absolute original raw receipt required')
        raw[receipt_role] = read(spec[receipt_role]['path'])
        raw_terminal(raw[receipt_role], spec[phase_role])
    require(spec['schema'] == SCHEMA and spec['phase'].startswith('shared_readout_objective_control_FIT128_') and
        Path(spec['phase']).name == spec['phase'] and spec['arms'] == list(ARMS) and spec['expected_epoch'] == 24 and
        spec['deadline_seconds'] == 5400 and spec['CPU_affinity'] == [0, 1] and spec['allocator_limit_MiB'] == 4096 and
        os.environ.get('CUDA_VISIBLE_DEVICES') == '0', 'Exact new arm/resource scope required')
    output = Path(spec['output_directory'])
    require(output.is_absolute() and not output.exists() and Path(spec['GPU_admission_file']).is_absolute(), 'Fresh absolute output/admission required')
    spec_ref = owner_gate(spec_path, spec)
    contract = read(bound(spec['SOURCE_contract']))
    require(spec['SOURCE_contract'] in spec['sources'] and contract['kind'] == 'INERT_SOURCE_AUTHORING_CONTRACT_NOT_SCIENTIFIC_ADMISSION' and
        contract['schema'] == SCHEMA and contract['arms'] == list(ARMS) and contract['epochs'] == 4 and contract['microsteps'] == 512 and
        contract['optimizer_updates'] == 256 and contract['detail_microsteps'] == list(DETAIL), 'Locked SOURCE authoring contract differs')
    require(spec['FIT_annotation'] == contract['split']['annotation_reference'] and len(contract['split']['train_ids']) == 128 and
        len(contract['split']['readout_heldout_ids']) == 64 and contract['split']['train_raw_GT'] == 292 and
        contract['split']['readout_heldout_raw_GT'] == 106, 'Prelocked FIT split/all raw GT differs')
    for role, row in contract['library_SOURCE_refs'].items():
        require(spec[role] == row and row in spec['sources'], 'Exact imported library SOURCE binding differs')
    fixture_gate(spec, contract)
    diag, prior = prior_gate(spec, contract, raw['prior_diag_receipt'])
    old_diag_contract = read(bound(spec['prior_diag_contract']))
    base, prior_DEV_q = diag.prior_DEV_gate(spec, raw['prior_DEV_receipt'], old_diag_contract)
    short = load_source(spec['short2_source'], 'sealed_SHORT2_ancestry')
    short.preflight_gate(spec, raw['preflight_receipt'])
    base.qualified_short2(spec, raw['short2_receipt'])
    core = load_source(spec['preflight_source'], 'sealed_frozen_native_teacher_ancestry')
    teacher_row = core.qualified_teacher(spec, raw['teacher_receipt'])
    core.fixture_gate(spec)
    output.mkdir(parents=True)
    joins = {role: spec[role] for role in ('entry_source', 'CPU_reader', 'SOURCE_contract', 'teacher_checkpoint', 'teacher_qualification',
        'prior_diag_spec', 'prior_diag_CPU_qualification', 'prior_diag_receipt', 'control_CPU_fixture', 'control_CPU_fixture_spec', 'control_CPU_fixture_source')}
    write(output / 'START.json', dict(schema='shared-readout-objective-control-start-v1', phase=spec['phase'], spec=spec_ref,
        arms=list(ARMS), endpoints=list(ENDPOINTS), joins=joins, original_five_raw_gates_before_any_hash_or_import=True,
        shared_teacher_forward=True, feature_cache_used=False, fixed_matching_cache_only=True, epochs=4, microsteps=512, updates=256,
        teacher_trained_all_selected_FIT=True, blind_holdout_claimed=False, no_DEV_TEST_access=True, no_automatic_retry=True))
    import psutil
    import numpy as np
    import torch
    from pycocotools import mask as masks_api
    from mmengine import Config
    from mmengine.model import revert_sync_batchnorm
    from mmengine.registry import init_default_scope
    from mmdet.registry import MODELS
    from mmdet.utils import register_all_modules
    from mmdet.datasets import CocoDataset
    from mmdet.datasets.api_wrappers import COCO
    require(psutil.Process().cpu_affinity() == [0, 1] and not torch.cuda.is_initialized(), 'Actual CPU01/cold CUDA required')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    require(torch.__version__ == '2.0.1+cu118' and torch.get_num_threads() == torch.get_num_interop_threads() == 2, 'Qualified Torch/threads differ')
    source_file(torch.nn.functional.interpolate, spec['interpolation_SOURCE'])
    source_file(torch.optim.AdamW, spec['adamw_SOURCE'])
    source_file(torch.nn.utils.clip_grad_norm_, spec['clip_grad_SOURCE'])
    source_file(torch.version, spec['torch_version_source'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    register_all_modules(init_default_scope=False)
    init_default_scope('mmdet')
    pref = load_source(spec['micro_preflight_source'], 'shared_control_tensor_helpers')
    initializer = load_source(spec['initializer_source'], 'shared_control_original_initializer')
    old = load_source(spec['native_helpers'], 'shared_control_image_target_observers'); old.torch = torch
    native = load_source(spec['native_preflight_helpers'], 'shared_control_GT_free_image_loader')
    official = load_source(spec['official_evaluation_source'], 'shared_control_native610_teacher_loader')
    readout = load_source(spec['readout_source'], 'shared_control_exact_RAW_operator')
    geometry = load_source(spec['geometry_source'], 'shared_control_exact_prediction_geometry')
    hooks = load_source(spec['hooks_source'], 'shared_control_metadata_valid_fraction')
    classes = tuple(CocoDataset.METAINFO['classes'])
    model, state, meta, loading = official.prepare_model(torch, pref, initializer, None, None, None, MODELS,
        revert_sync_batchnorm, Config, classes, spec, teacher_row, 'native0')
    model.requires_grad_(False).eval()
    anchor = cpu_copy(torch, model.state_dict())
    source_file(type(model.panoptic_head), spec['native_mask_head_SOURCE'])
    source_file(type(model.panoptic_fusion_head), spec['native_fusion_SOURCE'])
    head = model.panoptic_head
    source_file(head.preprocess_gt, spec['native_parent_head_SOURCE'])
    source_file(head.get_targets, spec['native_parent_head_SOURCE'])
    source_file(head._get_targets_single, spec['native_mask_head_SOURCE'])
    source_file(type(head.assigner), spec['hungarian_SOURCE'])
    source_file(type(head.sampler), spec['mask_sampler_SOURCE'])
    source_file(inspect.getmodule(type(head.sampler)).MaskSamplingResult, spec['mask_sampling_result_SOURCE'])
    source_file(inspect.getmodule(type(head)).point_sample, spec['point_sample_SOURCE'])
    source_file(inspect.getmodule(head.preprocess_gt).preprocess_panoptic_gt, spec['panoptic_GT_SOURCE'])
    require(head.num_points == 12544 and
        [type(value).__name__ for value in head.assigner.match_costs] == ['ClassificationCost', 'CrossEntropyLossCost', 'DiceCost'] and
        [float(value.weight) for value in head.assigner.match_costs] == [2.0, 5.0, 5.0],
        'Actual native final matcher point budget/cost order/weights differ')
    for cost in head.assigner.match_costs:
        source_file(type(cost), spec['match_cost_SOURCE'])
    require(len(anchor) == 610 and len(list(model.parameters())) == 451 and not torch.cuda.is_initialized(), 'Strict CPU native610 before CUDA required')
    modules = {arm: readout.build_adapter(torch, geometry, mode='raw', hidden_seed=2026100911, permutation_seed=2026101013) for arm in ARMS[:2]}
    gain = torch.nn.Module()
    gain.register_parameter('log_gain', torch.nn.Parameter(torch.zeros((), dtype=torch.float32, device='cpu')))
    modules[ARMS[2]] = gain
    # Optimizers are created after moving fresh CPU-built readouts to GPU below.
    dataset = CocoDataset(ann_file=str(bound(spec['FIT_annotation'])), data_prefix=dict(img=spec['image_root']),
        metainfo=dict(classes=('ship',)), pipeline=[], filter_cfg=None, test_mode=True)
    coco = COCO(str(bound(spec['FIT_annotation'])))
    require(len(dataset) == len(coco.imgs) == 741 and len(coco.anns) == 1661 and not torch.cuda.is_initialized(), 'Complete original FIT CPU population differs')
    for expected in contract['split']['train_rows'] + contract['split']['readout_heldout_rows']:
        require(int(dataset.get_data_info(expected['dataset_index'])['img_id']) == expected['image_id'] and
            coco.get_ann_ids(img_ids=[expected['image_id']]) == expected['annotation_ids'], 'Prelocked actual FIT ID/ann order differs')
    torch.cuda.init()
    torch.cuda.set_per_process_memory_fraction(4096 * 1024 ** 2 / torch.cuda.get_device_properties(0).total_memory, 0)
    for module in modules.values():
        module.to('cuda:0')
    optimizers = {arm: torch.optim.AdamW([dict(params=[p]) for name, p in module.named_parameters()],
        lr=0.01 if arm == ARMS[2] else 1e-4, weight_decay=0.05, betas=(0.9, 0.999), eps=1e-8) for arm, module in modules.items()}
    result = execute(torch, np, masks_api, core, pref, old, native, diag, base, hooks, model, anchor, loading,
        modules, optimizers, spec, spec_ref, contract, dataset, coco, output, started)
    del model, head, modules, optimizers, state, anchor
    gc.collect()
    cleanup = pref.clear_gpu(torch)
    require(cleanup['allocated_after'] == 0 and time.monotonic() - started < 5400, 'Actual endpoint cleanup/deadline differs')
    report = dict(schema=REPORT_SCHEMA, status=REPORT_STATUS, phase=spec['phase'], source=spec['entry_source'], spec=spec_ref,
        SOURCE_contract=spec['SOURCE_contract'], joins=joins, arms=list(ARMS), endpoints=list(ENDPOINTS), result=result, error=None,
        epochs=4, microsteps=512, optimizer_updates_per_arm=256, train_images=128, train_raw_GT=292, raw_GT_visits_per_arm=1168,
        readout_heldout_images=64, readout_heldout_raw_GT=106, teacher_forward_calls=576, shared_teacher_forward=True,
        full_native610_and_BN_exact=True, all_native451_frozen=True, GT_free_forward=True, feature_cache_used=False,
        fixed_assignment_cache_only=True, final_only_BCE=True, no_Dice_training_loss=True, no_unmatched_mask_loss=True,
        no_DEV_TEST_access=True, teacher_trained_all_selected_FIT=True, blind_holdout_claimed=False,
        official_all100_endpoint_exports=True, AP_computed_by_GPU_producer=False, AP_pending_independent_CPU_reader=True,
        gain_location='after_original_renderer_before_instance_postprocess', gain_capacity_or_optimizer_matched=False,
        GPU_optimizer_bitwise_determinism_claimed=False, efficacy_claimed=False, convergence_claimed=False, novelty_claimed=False,
        SOTA_claimed=False, previous_SHORT2_causal_explanation_claimed=False, full_empty_image_model_integration_qualified=False, cleanup=cleanup,
        torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(0), elapsed_seconds=time.monotonic() - started,
        evidence_limits=contract['nonclaims'] + ['Fixed E4 only; heldout is from new readout fitting, not frozen teacher training.',
            'Only four prelocked microsteps permit independent floating loss/gradient replay; all512 are not CPU optimizer replayed.',
            'RAW geometry recomputes teacher e0.F in chunks8; no bitexact native100 geometry claim.',
            'Uniform pixel BCE estimates full BCE; clipping, AdamW and realized compute remain different.',
            'Gain acts after rendering; actual binary agreement is checked without universal FP32 threshold invariance.'])
    write(output / 'REPORT.json', report)
    write(output / 'QUALIFIED.json', dict(schema='shared-readout-objective-control-aggregate-v1',
        status='ROOT_SHARED_READOUT_OBJECTIVE_CONTROL_PENDING_CPU_READER', phase=spec['phase'], source=spec['entry_source'],
        spec=spec_ref, report=reference(output / 'REPORT.json'), result=result, independently_CPU_qualified=False, AP_computed=False))
    print(json.dumps(dict(status=REPORT_STATUS, CPU_qualification_pending=True)), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', required=True)
    parser.add_argument('--owner-execute', action='store_true')
    args = parser.parse_args()
    require(args.owner_execute, 'Only ROOT-owned admitted execution is permitted')
    run(args.spec)


if __name__ == '__main__':
    main()
