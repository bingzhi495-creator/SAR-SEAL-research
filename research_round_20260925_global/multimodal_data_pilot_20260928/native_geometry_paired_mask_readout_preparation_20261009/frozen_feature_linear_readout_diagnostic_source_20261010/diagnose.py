"""Inert ROOT-owned FIT feasibility diagnostic, never a deployable readout.

Four original RAW terminal gates precede every hash and scientific import.
The exact full610 E24 native teacher is frozen/eval and forwarded without GT.
For each of the prelocked12 FIT images, capture F/e0/nativeM200 once and verify
native100 einsum replay plus unchanged official all100 instance fusion.
Each raw GT gets an independent free256 coefficient initialized from its
actual native GT-best100 query (lowest actual ordinal on ties), without bias,
MLP, boxes, point sampling, regularization, threshold tuning or model changes.
Full original-size unweighted mean BCE drives one fresh FP32 LBFGS40/80 solve.
max_eval80 is soft under strong-Wolfe; actual optimizer counters are audited.
Only actual native baseline and finite final candidate are retained as masks.

CPU tensor packets are audit artifacts, not a reusable feature-cache training
recipe. Oracle1 vs native100 FP32 GEMM grouping need not be bitwise identical.
CUDA bilinear backward can be nondeterministic: fixed full-raster supervision
and no solver RNG do not certify a bitwise deterministic optimization path.
Success is a finite FIT witness; failure unresolved. No AP, optimal IoU bound,
convergence, generator-versus-budget causal verdict, TEST or efficacy claim.
SOURCE authoring executes no science or scientific payload access.
"""

import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import time
import types


SCHEMA = 'ROOT-frozen-feature-linear-readout-FIT-diagnostic-spec-v1'
REPORT_SCHEMA = 'frozen-feature-linear-readout-FIT-report-v1'
STATE_SCHEMA = 'frozen-feature-linear-readout-FIT-state-proof-v1'
IMAGE_SCHEMA = 'frozen-feature-linear-readout-image-audit-v1'
GT_SCHEMA = 'frozen-feature-linear-readout-GT-audit-v1'
SUBSET = [1, 2, 3, 5, 6, 8, 9, 10, 11, 12, 13, 14]
GT_COUNTS = [1, 1, 1, 3, 1, 1, 1, 1, 1, 3, 2, 2]
SOLVER = dict(name='torch.optim.LBFGS', lr=1, max_iter=40, max_eval=80,
    history_size=10, tolerance_grad=1e-7, tolerance_change=1e-9, line_search_fn='strong_wolfe')
LIBRARY_ROLES = ('solver_source', 'torch_version_source', 'native_mask_head_SOURCE',
    'native_fusion_SOURCE', 'interpolation_SOURCE')
OWNER_SHA = 'ef9a9f17d4fb1bc3464936095ef799021dd880f8b15488d558a871c08b6e1924'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def reference(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for value in iter(lambda: stream.read(1048576), b''):
            digest.update(value)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def bound(row):
    require(isinstance(row, dict) and set(row) == {'path', 'bytes', 'sha256'} and
        Path(row['path']).is_absolute() and reference(row['path']) == row, 'Pinned absolute reference differs')
    return Path(row['path'])


def load_source(row, name):
    path = bound(row)
    payload = path.read_bytes()
    require(len(payload) == row['bytes'] and hashlib.sha256(payload).hexdigest() == row['sha256'], 'SOURCE changed')
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(payload, str(path), 'exec'), module.__dict__)
    return module


def raw_terminal(receipt, phase):
    require(receipt['status'] == 'ACTUAL_OWNED_TERMINAL' and receipt['original_identity'] is not None and
        type(receipt['returncode']) is int and receipt['returncode'] == 0 and receipt['root_wait_reaped'] is True and
        receipt['held_handle_exit'] == dict(wait_state='WAIT_OBJECT_0', actual_exit_code=0) and
        receipt['numeric_birth_absence']['original_birth_absent'] is True and receipt['abort'] is None and
        receipt['error'] is None and receipt['wait_errors'] is False and receipt['GPU_observation_errors'] == [],
        'STOP before scientific access: raw held0/reap/original birth absence/noerror required')
    if 'phase' in receipt:
        require(receipt['phase'] == phase, 'Original raw terminal phase differs')
    peak = receipt['whole_GPU_observed_peak_MiB']
    require(type(peak) in (int, float) and math.isfinite(peak) and 0 <= peak <= 8192 and
        receipt['whole_GPU_used_limit_MiB'] == 8192 and receipt['whole_leaf_VRAM_certified'] is False and
        receipt['GPU_scope'] == 'whole GPU observed, not per-process WDDM allocation', 'Observed whole-GPU scope differs')


def owner_gate(spec_path, spec):
    admission = read(Path(spec['GPU_admission_file']).resolve(strict=True))
    entry, spec_ref = reference(__file__), reference(spec_path)
    require(admission['status'] == 'ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE' and
        admission['automatic_retry'] is False and admission['phase'] == spec['phase'] and
        admission['allowed_cpus'] == spec['CPU_affinity'] == [0, 1] and admission['cuda_visible_devices'] == '0' and
        admission['qualified_leaf_deadline_seconds'] == spec['deadline_seconds'] == 1800 and
        admission['whole_GPU_used_limit_MiB'] == spec['whole_GPU_owner_limit_MiB'] == 8192 and
        Path(admission['entry']).resolve(strict=True) == Path(entry['path']) and
        admission['entry_args'] == ['--spec', str(spec_path), '--owner-execute'] and admission['driver_sha256'] == OWNER_SHA,
        'Current actual owner/entry/resource contract differs')
    hashes = admission['bound_source_sha256']
    require(spec['entry_source'] == entry and entry in spec['sources'] and spec['CPU_reader'] in spec['sources'] and
        hashes.get(str(spec_path)) == spec_ref['sha256'], 'Current entry/reader/SPEC owner pins absent')
    for row in spec['sources']:
        require(hashes.get(row['path']) == row['sha256'], 'Complete current closure not owner-bound')
        bound(row)
    if 'scientific_spec' in admission:
        require(admission['scientific_spec'] == spec_ref, 'Current scientific SPEC differs')
    return spec_ref


def prior_DEV_gate(spec, raw, contract):
    """Require actual qualified prior DEV, preserving its complete admitted closure."""
    require(spec['prior_DEV_spec'] == contract['source_baseline_evaluation'], 'Canonical prior DEV SPEC differs')
    prior = read(bound(spec['prior_DEV_spec']))
    q = read(bound(spec['prior_DEV_CPU_qualification']))
    bound(spec['prior_DEV_receipt'])
    require(prior['schema'] == 'ROOT-query-conditioned-frozen-native-short2-evaluation-spec-v1' and
        prior['entry_source'] == q['source'] == contract['source_baseline_evaluation_entry'] and
        prior['CPU_reader'] == q['reader_source'] == contract['base_role_SOURCE_refs']['CPU_reader'] and
        q['schema'] == 'ROOT-query-conditioned-frozen-native-short2-evaluation-CPU-qualification-v1' and
        q['status'] == 'ROOT_QUERY_CONDITIONED_FROZEN_NATIVE_SHORT2_DEV_CPU_EVALUATED' and
        q['spec'] == spec['prior_DEV_spec'] and q['parent_receipt'] == spec['prior_DEV_receipt'] and
        q['phase'] == prior['phase'] == spec['prior_DEV_phase'] and
        q['complete_SOURCE_closure_verified_before_scientific_import'] is True and
        q['full_native610_teacher_and_typed_BN106_affine159_buffers_CPU_exact'] is True and
        q['official_segmentation_COCO12_independently_computed'] is True and
        q['all100_predictions_per_image_and_empty_masks_preserved'] is True,
        'Actual complete independently qualified prior DEV lineage differs')
    for key in ('checkpointmap', 'qualificationmap', 'teacher_checkpoint', 'teacher_qualification',
                'teacher_training_spec', 'teacher_receipt', 'teacher_phase', 'preflight_CPU_qualification',
                'preflight_training_spec', 'preflight_receipt', 'preflight_phase', 'short2_CPU_qualification',
                'short2_training_spec', 'short2_receipt', 'short2_phase', 'image_root', 'FIT_annotation', 'DEV_annotation'):
        require(prior[key] == spec[key], 'Preserved prior native construction/input role differs: ' + key)
    for role, row in contract['base_role_SOURCE_refs'].items():
        if role not in ('entry_source', 'CPU_reader'):
            require(prior[role] == spec[role] == row, 'Immutable qualified helper SOURCE differs: ' + role)
    require(isinstance(prior['sources'], list) and len(prior['sources']) > 0 and
        all(row in spec['sources'] for row in prior['sources']), 'Complete prior DEV closure must be preserved')
    admission = read(bound(q['GPU_admission']))
    hashes = admission['bound_source_sha256']
    require(admission['status'] == 'ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE' and
        raw['admission_sha256'] == q['GPU_admission']['sha256'] and q['owner_driver']['sha256'] == OWNER_SHA and
        Path(prior['GPU_admission_file']).resolve(strict=True) == Path(q['GPU_admission']['path']).resolve(strict=True) and
        admission['phase'] == q['phase'] and admission['automatic_retry'] is False and
        admission['driver_sha256'] == OWNER_SHA and admission['allowed_cpus'] == [0, 1] and
        admission['cuda_visible_devices'] == '0' and admission['whole_GPU_used_limit_MiB'] == 8192 and
        admission['qualified_leaf_deadline_seconds'] == 1800 and
        Path(admission['entry']).resolve(strict=True) == Path(q['source']['path']).resolve(strict=True) and
        admission['entry_args'] == ['--spec', q['spec']['path'], '--owner-execute'] and
        hashes.get(q['spec']['path']) == q['spec']['sha256'] and
        all(hashes.get(row['path']) == row['sha256'] for row in prior['sources']), 'Original DEV admission/raw/SPEC join differs')
    if 'scientific_spec' in admission:
        require(admission['scientific_spec'] == q['spec'], 'Original DEV owner scientific SPEC differs')
    for key in ('source', 'reader_source', 'report', 'start', 'GPU_qualification', 'owner_driver'):
        bound(q[key])
    return load_source(prior['entry_source'], 'sealed_prior_DEV_native_fusion_helpers'), q


def diagnostic_fixture_gate(spec):
    for role in ('diagnostic_CPU_fixture', 'diagnostic_CPU_fixture_spec', 'diagnostic_CPU_fixture_source'):
        require(spec[role] in spec['sources'], 'Actual diagnostic CPU fixture must be owner-bound: ' + role)
    manifest = read(bound(spec['diagnostic_CPU_fixture_spec']))
    fixture = read(bound(spec['diagnostic_CPU_fixture']))
    sources = dict(producer=spec['entry_source'], fixture=spec['diagnostic_CPU_fixture_source'], contract=spec['SOURCE_contract'])
    execution = dict(device='cpu', cpu_affinity=[0, 1], torch_threads=2, torch_interop_threads=2, cuda_visible_devices='')
    require(manifest == dict(schema='ROOT-frozen-feature-linear-readout-synthetic-SOURCE-spec-v1',
        inputs='SOURCE_ONLY', sources=sources, execution=execution) and
        fixture['status'] == 'FROZEN_FEATURE_LINEAR_READOUT_SYNTHETIC_CPU_PASS' and
        fixture['sources'] == sources and fixture['execution'] == execution and fixture['synthetic_execution_performed'] is True and
        Path(fixture['spec_path']).resolve(strict=True) == Path(spec['diagnostic_CPU_fixture_spec']['path']).resolve(strict=True) and
        fixture['torch_version'] == '2.0.1+cu118' and fixture['solver'] == SOLVER and
        fixture['real_data_feasibility_qualified'] is False and fixture['native_model_integration_qualified'] is False and
        fixture['AP_computed'] is False and fixture['max_eval_is_soft_limit'] is True and
        fixture['GPU_optimizer_bitwise_determinism_claimed'] is False,
        'Actual independently checked SOURCE-only CPU synthetic fixture differs')
    for role in LIBRARY_ROLES:
        require(fixture[role] == spec[role], 'Synthetic fixture implementation binding differs: ' + role)
    # ROOT separately gates the actual CPU tool exit0; a JSON file cannot prove that receipt.
    return fixture


def project_coefficients(torch, feature, coefficients):
    """Pure injected-torch helper; grouping differs between Q1 and Q100."""
    require(feature.ndim == 4 and feature.shape[0] == 1 and feature.shape[1] == 256 and
        coefficients.ndim == 3 and coefficients.shape[0] == 1 and coefficients.shape[2] == 256 and
        feature.dtype == coefficients.dtype == torch.float32 and feature.device == coefficients.device and
        not feature.requires_grad and bool(torch.isfinite(feature).all()) and bool(torch.isfinite(coefficients).all()),
        'Frozen F and coefficients must be finite FP32 with native256 channels')
    return torch.einsum('bqc,bchw->bqhw', coefficients, feature)


def render_original(torch, masks200, identity):
    """Pure native FP32 interpolation; GT does not define crop geometry."""
    require(masks200.ndim == 4 and masks200.shape[0] == 1 and masks200.dtype == torch.float32,
        'Native batch1 FP32 mask logits required')
    ih, iw = map(int, identity['actual_img_shape'])
    oh, ow = map(int, identity['original_shape'])
    require(0 < ih <= 800 and 0 < iw <= 800 and oh > 0 and ow > 0, 'Native metadata-only rendering dimensions invalid')
    padded = torch.nn.functional.interpolate(masks200, size=(800, 800), mode='bilinear', align_corners=False)
    return torch.nn.functional.interpolate(padded[0, :, :ih, :iw][:, None], size=(oh, ow),
        mode='bilinear', align_corners=False)[:, 0]


def binary_metrics(torch, binary, target):
    require(binary.shape == target.shape and binary.dtype == target.dtype == torch.bool, 'Same-shape binary masks required')
    pa, ga = int(binary.sum().item()), int(target.sum().item())
    intersection = int((binary & target).sum().item())
    union = pa + ga - intersection
    return dict(intersection=intersection, union=union, prediction_area=pa, GT_area=ga,
        IoU=intersection / union if union else 1.0, Dice=2 * intersection / (pa + ga) if pa + ga else 1.0,
        precision=intersection / pa if pa else float(ga == 0), recall=intersection / ga if ga else float(pa == 0),
        prediction_empty=pa == 0, GT_empty=ga == 0)


def lowest_best_query(scores):
    require(len(scores) > 0 and all(math.isfinite(float(value)) and 0 <= value <= 1 for value in scores), 'Finite query IoU scores required')
    best = max(scores)
    return next(index for index, score in enumerate(scores) if score == best)


def fit_coefficient(torch, feature, initial_coefficient, target, identity, solver, deadline_check=None):
    """One fresh independent256 parameter; no SOURCE/data/model/RNG I/O."""
    require(solver == SOLVER and initial_coefficient.shape == (256,) and initial_coefficient.dtype == torch.float32 and
        initial_coefficient.device == feature.device == target.device and target.dtype == torch.bool and
        tuple(target.shape) == tuple(identity['original_shape']), 'Fixed native coefficient/GT/solver interface differs')
    coefficient = torch.nn.Parameter(initial_coefficient.detach().clone())
    config = {key: value for key, value in solver.items() if key != 'name'}
    optimizer = torch.optim.LBFGS([coefficient], **config)
    require(len(optimizer.state) == 0 and len(optimizer.param_groups) == 1 and
        optimizer.param_groups[0]['params'] == [coefficient], 'Fresh single-coefficient LBFGS only')
    callbacks = []
    with torch.no_grad():
        initial_logits = render_original(torch, project_coefficients(torch, feature, coefficient[None, None]), identity)[0]
        initial_loss = float(torch.nn.functional.binary_cross_entropy_with_logits(initial_logits, target.float(), reduction='mean').item())
        initial_norm = float(torch.linalg.vector_norm(coefficient).item())
    del initial_logits

    def closure():
        if deadline_check is not None:
            deadline_check()
        optimizer.zero_grad(set_to_none=True)
        logits = render_original(torch, project_coefficients(torch, feature, coefficient[None, None]), identity)[0]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target.float(), reduction='mean')
        require(loss.requires_grad and bool(torch.isfinite(loss)), 'Finite connected full-raster BCE required')
        loss.backward()
        require(coefficient.grad is not None and bool(torch.isfinite(coefficient.grad).all()), 'Finite connected coefficient gradient required')
        callbacks.append(dict(closure_index=len(callbacks) + 1, loss=float(loss.detach().item()),
            coefficient_norm=float(torch.linalg.vector_norm(coefficient.detach()).item()),
            gradient_norm=float(torch.linalg.vector_norm(coefficient.grad.detach()).item())))
        return loss

    optimizer.step(closure)
    actual = optimizer.state[coefficient]
    require(actual['func_evals'] == len(callbacks) and 0 <= actual['n_iter'] <= solver['max_iter'], 'Actual LBFGS callback counters differ')
    if deadline_check is not None:
        deadline_check()
    # This endpoint evaluation is separate from optimizer closure accounting.
    final_logits = render_original(torch, project_coefficients(torch, feature, coefficient[None, None]), identity)[0]
    final_loss = torch.nn.functional.binary_cross_entropy_with_logits(final_logits, target.float(), reduction='mean')
    final_gradient, = torch.autograd.grad(final_loss, coefficient)
    require(bool(torch.isfinite(final_logits).all()) and bool(torch.isfinite(final_loss)) and
        bool(torch.isfinite(final_gradient).all()) and bool(torch.isfinite(coefficient).all()), 'Finite final candidate required')
    result = dict(final_coefficient=coefficient.detach().clone(), final_logits=final_logits.detach(),
        config=dict(solver), initial_loss=initial_loss, final_loss=float(final_loss.detach().item()),
        initial_norm=initial_norm, final_norm=float(torch.linalg.vector_norm(coefficient.detach()).item()),
        final_gradient_norm=float(torch.linalg.vector_norm(final_gradient).item()), solver_callbacks=callbacks,
        closure_calls=len(callbacks), optimizer_func_evals=int(actual['func_evals']), optimizer_n_iter=int(actual['n_iter']),
        max_eval_is_soft_limit=True, nonclosure_initial_evaluations=1, nonclosure_final_evaluations=1,
        actual_coefficient_grad_connected=True)
    del coefficient, optimizer, final_logits, final_loss, final_gradient
    return result


def encode_mask(np, masks_api, binary):
    rle = masks_api.encode(np.asfortranarray(binary.detach().cpu().numpy().astype(np.uint8)))
    counts = rle['counts'].decode('ascii') if isinstance(rle['counts'], bytes) else rle['counts']
    return dict(size=list(rle['size']), counts=counts)


def write_packet(torch, path, packet):
    require(not path.exists(), 'Fresh CPU audit artifact required')
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('xb') as stream:
        torch.save(packet, stream)
    require(not path.exists(), 'Audit output unexpectedly appeared')
    os.replace(str(temporary), str(path))


def source_file(obj, row):
    actual = inspect.getsourcefile(obj)
    require(actual is not None and Path(actual).resolve(strict=True) == bound(row).resolve(strict=True), 'Imported implementation SOURCE differs')


def diagnose_images(torch, np, masks_api, base, core, pref, old, native, model, state, loading,
                    spec, dataset, coco, output, started):
    model.to('cuda:0').requires_grad_(False).eval()
    head = model.panoptic_head
    before = pref.state_proof(model.state_dict())
    require(before == pref.state_proof(state) and len(before) == 610 and len(list(model.parameters())) == 451 and
        all(not parameter.requires_grad and parameter.grad is None for parameter in model.parameters()) and
        all(not module.training for module in model.modules()), 'Exact fully frozen native610/451 eval model required')
    source_file(type(head), spec['native_mask_head_SOURCE'])
    source_file(type(model.panoptic_fusion_head), spec['native_fusion_SOURCE'])
    inventory = core.hook_inventory(model)
    images, gt_rows, predictions, score_factors = [], [], [], []
    try:
        for position, image_id in enumerate(SUBSET):
            pref.check_deadline(started, spec['deadline_seconds'])
            data, samples, identity = core.image_data(native, old, model, dataset, coco, image_id)
            require(identity['dataset_index'] == position and identity['image_id'] == image_id and
                len(identity['ann_ids']) == GT_COUNTS[position] and len(samples) == 1 and
                all(not any(key in sample for key in core.GT_FIELDS) for sample in samples), 'Prelocked FIT population/all-four GT-free forward differs')
            external = pref.rng_snapshot(torch)
            counts, capture = dict(feature=0, coefficient=0, head=0), {}

            def feature_hook(module, inputs, value):
                counts['feature'] += 1
                feature = value[0]
                require(tuple(feature.shape) == (1, 256, 200, 200) and feature.dtype == torch.float32 and
                    not feature.requires_grad and bool(torch.isfinite(feature).all()), 'Actual frozen finite FP32 F200 required')
                capture['feature'] = feature.detach()

            def coefficient_hook(module, inputs, value):
                counts['coefficient'] += 1
                require(tuple(value.shape) == (1, 100, 256) and value.dtype == torch.float32 and
                    not value.requires_grad and bool(torch.isfinite(value).all()), 'Actual frozen native100 coefficient required')
                if counts['coefficient'] == 10:
                    capture['native_coeff'] = value.detach()

            def head_hook(module, inputs, value):
                counts['head'] += 1
                require(len(value) == 2 and len(value[0]) == len(value[1]) == 10 and
                    all(tuple(v.shape) == (1, 100, 2) and v.dtype == torch.float32 and not v.requires_grad for v in value[0]) and
                    all(tuple(v.shape) == (1, 100, 200, 200) and v.dtype == torch.float32 and not v.requires_grad for v in value[1]),
                    'Native ten-stage interfaces differ')
                capture['classes'] = value[0][-1]
                capture['native_masks200'] = value[1][-1]

            handles = [head.pixel_decoder.register_forward_hook(feature_hook), head.mask_embed.register_forward_hook(coefficient_hook),
                head.register_forward_hook(head_hook)]
            seed = 100000 + image_id
            try:
                with torch.random.fork_rng(devices=[0]):
                    torch.random.default_generator.manual_seed(seed)
                    torch.cuda.manual_seed(seed)
                    private_before = pref.rng_snapshot(torch)
                    with torch.no_grad():
                        features = model.extract_feat(data['inputs'])
                        classes, masks = head(features, samples)
                        require(counts == dict(feature=1, coefficient=10, head=1) and
                            classes[-1] is capture['classes'] and masks[-1] is capture['native_masks200'], 'One actual native forward/capture required')
                        replay200 = project_coefficients(torch, capture['feature'], capture['native_coeff'])
                        require(torch.equal(replay200, masks[-1]), 'Native100 actual GPU einsum replay not TorchEqual to native200')
                        preds, factors, query_ids, class_packet, fields = base.official_predictions(torch, np, masks_api,
                            model, classes[-1], masks[-1], samples, identity, pref)
                        native_original = render_original(torch, masks[-1], identity)
                        replay_original = render_original(torch, replay200, identity)
                        require(torch.equal(native_original, replay_original) and
                            torch.equal(native_original > 0, replay_original > 0), 'Native100 official-rendered masks must replay TorchEqual on GPU')
                    private_after = pref.rng_snapshot(torch)
                require(core.equal_rng(torch, external, pref.rng_snapshot(torch)), 'Private forward changed ambient CPU/CUDA RNG')
            finally:
                for handle in handles:
                    handle.remove()
                handles.clear()
            require(core.hook_inventory(model) == inventory, 'Passive forward hooks not removed')
            require(query_ids and sorted(query_ids) == list(range(100)) and len(preds) == len(factors) == 100, 'Official all100 actual query IDs required')
            by_query = {row['query_id']: row['segmentation'] for row in preds}
            image_packet_path = output / ('AUDIT_IMAGE_%d.pth' % image_id)
            packet = dict(schema=IMAGE_SCHEMA, image_id=image_id, identity=identity,
                feature=capture['feature'].detach().cpu().clone(), native_coeff=capture['native_coeff'].detach().cpu().clone(),
                native_masks200=capture['native_masks200'].detach().cpu().clone())
            write_packet(torch, image_packet_path, packet)
            image_ref = reference(image_packet_path)
            predictions.extend(preds)
            score_factors.extend(factors)
            native_binary = native_original > 0
            ann_ids = sorted(int(value) for value in coco.get_ann_ids(img_ids=[image_id]))
            require(ann_ids == sorted(identity['ann_ids']) and len(ann_ids) == GT_COUNTS[position], 'All raw original annotation IDs must be retained')
            for ann_id in ann_ids:
                pref.check_deadline(started, spec['deadline_seconds'])
                ann = coco.anns[ann_id]
                raw_mask = np.zeros(tuple(identity['original_shape']), dtype=np.uint8) \
                    if ann.get('segmentation') == [] else coco.annToMask(ann)
                require(raw_mask.shape == tuple(identity['original_shape']) and
                    np.isin(raw_mask, [0, 1]).all(), 'Original raw binary GT raster shape differs')
                target = torch.from_numpy(np.ascontiguousarray(raw_mask)).to(device='cuda:0', dtype=torch.bool)
                intersections = (native_binary & target[None]).flatten(1).sum(1).cpu().tolist()
                prediction_areas = native_binary.flatten(1).sum(1).cpu().tolist()
                gt_area = int(target.sum().item())
                ious = [int(hit) / (int(area) + gt_area - int(hit)) if int(area) + gt_area - int(hit) else 1.0
                    for hit, area in zip(intersections, prediction_areas)]
                query = lowest_best_query(ious)
                initial = capture['native_coeff'][0, query].detach()
                baseline_logits = native_original[query].detach()
                baseline = dict(query_id=query, metrics=binary_metrics(torch, native_binary[query], target),
                    full_raster_BCE=float(torch.nn.functional.binary_cross_entropy_with_logits(baseline_logits, target.float()).item()),
                    coefficient_norm=float(torch.linalg.vector_norm(initial).item()), mask_RLE=by_query[query])
                fit = fit_coefficient(torch, capture['feature'], initial, target, identity, spec['solver'],
                    deadline_check=lambda: pref.check_deadline(started, spec['deadline_seconds']))
                final = dict(metrics=binary_metrics(torch, fit['final_logits'] > 0, target), full_raster_BCE=fit['final_loss'],
                    coefficient_norm=fit['final_norm'], mask_RLE=encode_mask(np, masks_api, fit['final_logits'] > 0))
                best_name, best_row = ('final', final) if final['metrics']['IoU'] > baseline['metrics']['IoU'] else ('native_best100', baseline)
                gt_packet_path = output / ('AUDIT_GT_%d.pth' % ann_id)
                write_packet(torch, gt_packet_path, dict(schema=GT_SCHEMA, image_id=image_id, annotation_id=ann_id,
                    baseline_query_id=query, baseline_original_logits=baseline_logits.cpu().clone(),
                    final_coefficient=fit['final_coefficient'].cpu().clone(), final_original_logits=fit['final_logits'].cpu().clone()))
                solver_audit = {key: value for key, value in fit.items() if key not in ('final_coefficient', 'final_logits')}
                gt_rows.append(dict(image_id=image_id, annotation_id=ann_id, raw_annotation=ann,
                    original_shape=list(identity['original_shape']), baseline_query_id=query, native100_IoU=ious,
                    candidates=['native_best100', 'final'], baseline=baseline, final=final,
                    best_of_two=dict(candidate=best_name, IoU=best_row['metrics']['IoU'], Dice=best_row['metrics']['Dice'],
                        tie_rule='native_best100_first', not_optimal_upper_bound=True), solver_audit=solver_audit,
                    audit_packet=reference(gt_packet_path), GT_digest=pref.digest_tensor(target.to(dtype=torch.uint8))))
                require(all(parameter.grad is None and not parameter.requires_grad for parameter in model.parameters()) and
                    all(not module.training for module in model.modules()), 'Oracle changed native model gradients/mode')
                del target, fit, initial, baseline_logits, raw_mask
            require(core.equal_rng(torch, external, pref.rng_snapshot(torch)) and core.hook_inventory(model) == inventory,
                'Oracle solve changed ambient RNG or native hook inventory')
            images.append(dict(image_id=image_id, dataset_index=position, identity=identity,
                annotation_ids=ann_ids, original_shape=list(identity['original_shape']), actual_img_shape=list(identity['actual_img_shape']),
                input=pref.digest_tensor(data['inputs']), audit_packet=image_ref,
                tensor_digests={key: pref.digest_tensor(packet[key]) for key in ('feature', 'native_coeff', 'native_masks200')},
                native_class_packet=class_packet, final_fields=fields, head_seed=seed,
                RNG_external_before=[pref.digest_tensor(value) for value in external],
                RNG_private_before=[pref.digest_tensor(value) for value in private_before],
                RNG_private_after=[pref.digest_tensor(value) for value in private_after],
                RNG_external_after=[pref.digest_tensor(value) for value in pref.rng_snapshot(torch)],
                external_RNG_restored=True, forward_capture_counts=counts, GT_removed_from_forward=True,
                loader_accesses_annotations=True, matching_or_loss_executed=False, native100_GPU_mask200_TorchEqual=True,
                native100_GPU_official_masks_TorchEqual=True, native_official_fusion=True, prediction_query_ids=query_ids,
                predictions=100, native_parameter_grads_None=True, all_modules_eval=True, hooks_removed=True, hook_cache_clear=True,
                oracle1_vs_native100_bitwise_algebra_claimed=False, GPU_optimizer_bitwise_determinism_claimed=False))
            del data, samples, features, classes, masks, replay200, replay_original, native_original, native_binary
            capture.clear()
            del packet, preds, factors, by_query
        after = pref.state_proof(model.state_dict())
        require(after == before == pref.state_proof(state) and core.hook_inventory(model) == inventory and
            all(not module.training for module in model.modules()) and
            all(parameter.grad is None and not parameter.requires_grad for parameter in model.parameters()),
            'All610 native tensors/BN/451 frozen gradients/modules must remain exact at endpoint')
        cpu_state = pref.cpu_tree(torch, model.state_dict())
        require(pref.tree_equal(torch, cpu_state, state), 'Strict actual GPU-to-CPU native610 readback differs')
        proof = dict(schema=STATE_SCHEMA, mode='native0', checkpoint=spec['teacher_checkpoint'],
            qualification=spec['teacher_qualification'], loading_proof=loading, native_teacher610=pref.state_proof(state),
            before=before, after=after, full_native_state_exact=True, frozen_BN_all_exact=True,
            BN_parameter_names=loading['BN_parameter_names'], BN_buffer_names=loading['BN_buffer_names'],
            native_parameter_grads_None=True, all_parameters_frozen=True, all_modules_eval=True,
            hooks_removed=True, hook_cache_clear=True, CPU_readback_exact=True, native_state_entries=610,
            native_parameter_tensors=451, oracle_parameters_registered_in_model=False, optimizer_or_RNG_restored=False)
        require(len(images) == 12 and len(gt_rows) == 18 and sorted(row['annotation_id'] for row in gt_rows) == list(range(1, 19)) and
            len(predictions) == len(score_factors) == 1200, 'Complete prelocked12/all18 GT/all1200 native predictions required')
        core.write(output / 'IMAGE_ROWS.json', images)
        core.write(output / 'GT_ROWS.json', gt_rows)
        core.write(output / 'NATIVE_PREDICTIONS.json', predictions)
        core.write(output / 'QUERY_SCORE_FACTORS.json', score_factors)
        core.write(output / 'STATE_PROOF.json', proof)
        return dict(image_rows=reference(output / 'IMAGE_ROWS.json'), GT_rows=reference(output / 'GT_ROWS.json'),
            native_predictions=reference(output / 'NATIVE_PREDICTIONS.json'), query_score_factors=reference(output / 'QUERY_SCORE_FACTORS.json'),
            state_proof=reference(output / 'STATE_PROOF.json'), images=12, raw_GT=18, native_predictions_count=1200,
            fitted_coefficients=18, fullstate_tensors=610, native_parameters=451, full_native_state_exact=True,
            frozen_BN_all_exact=True, native_parameter_grads_None=True, peak_allocated_bytes=torch.cuda.max_memory_allocated(0))
    finally:
        model.to('cpu')


def run(spec_file):
    started = time.monotonic()
    require(Path(spec_file).is_absolute(), 'Absolute --spec required')
    spec_path = Path(spec_file).resolve(strict=True)
    spec = read(spec_path)  # Inert metadata access only until four RAW gates pass.
    receipts = {}
    for role, phase in (('teacher_receipt', 'teacher_phase'), ('preflight_receipt', 'preflight_phase'),
                        ('short2_receipt', 'short2_phase'), ('prior_DEV_receipt', 'prior_DEV_phase')):
        require(Path(spec[role]['path']).is_absolute(), 'Absolute prior RAW receipt required')
        receipts[role] = read(spec[role]['path'])
        raw_terminal(receipts[role], spec[phase])
    # Four original owner held0/reaped/original-birth-absent gates precede ALL hashes/imports.
    require(spec['schema'] == SCHEMA and spec['phase'].startswith('frozen_feature_linear_readout_FIT_') and
        Path(spec['phase']).name == spec['phase'] and spec['subset_image_ids'] == SUBSET and
        spec['subset_images'] == 12 and spec['subset_raw_GT'] == 18 and spec['solver'] == SOLVER and
        spec['diagnostic_mode'] == 'native0' and spec['no_feature_cache'] is False and spec['audit_packet_only'] is True and
        spec['GPU_optimizer_bitwise_determinism_claimed'] is False and
        spec['expected_epoch'] == 24 and spec['CPU_affinity'] == [0, 1] and spec['deadline_seconds'] == 1800 and
        spec['allocator_limit_MiB'] == 4096 and spec['whole_GPU_owner_limit_MiB'] == 8192 and
        os.environ.get('CUDA_VISIBLE_DEVICES') == '0', 'Exact prelocked FIT/solver/current resource scope required')
    output = Path(spec['output_directory'])
    require(output.is_absolute() and not output.exists() and Path(spec['GPU_admission_file']).is_absolute(), 'Fresh absolute output/admission required')
    spec_ref = owner_gate(spec_path, spec)
    require(spec['SOURCE_contract'] in spec['sources'], 'Inert authoring contract must be owner-bound')
    contract = read(bound(spec['SOURCE_contract']))
    require(contract['kind'] == 'INERT_SOURCE_AUTHORING_CONTRACT_NOT_SCIENTIFIC_ADMISSION' and
        contract['schema'] == SCHEMA and contract['subset_ids'] == SUBSET and contract['solver'] == SOLVER and
        contract['candidates'] == ['native_best100', 'final'] and contract['entry_name'] == 'diagnose.py' and
        contract['reader_name'] == 'qualify_cpu.py', 'Locked authoring contract differs')
    for role in LIBRARY_ROLES:
        require(spec[role] == contract[role] and spec[role] in spec['sources'], 'Actual implementation SOURCE pin differs: ' + role)
    diagnostic_fixture_gate(spec)
    base, prior_q = prior_DEV_gate(spec, receipts['prior_DEV_receipt'], contract)
    short = load_source(spec['short2_source'], 'sealed_SHORT2_preflight_lineage')
    short.preflight_gate(spec, receipts['preflight_receipt'])
    base.qualified_short2(spec, receipts['short2_receipt'])
    core = load_source(spec['preflight_source'], 'sealed_native_teacher_lineage')
    teacher_row = core.qualified_teacher(spec, receipts['teacher_receipt'])
    core.fixture_gate(spec)
    output.mkdir(parents=True)
    core.write(output / 'START.json', dict(schema='frozen-feature-linear-readout-FIT-start-v1', phase=spec['phase'],
        source=spec['entry_source'], spec=spec_ref, mode='native0', SOURCE_contract=spec['SOURCE_contract'], subset_image_ids=SUBSET,
        subset_images=12, subset_raw_GT=18, solver=SOLVER, teacher_checkpoint=spec['teacher_checkpoint'],
        teacher_qualification=spec['teacher_qualification'], prior_DEV_spec=spec['prior_DEV_spec'],
        prior_DEV_CPU_qualification=spec['prior_DEV_CPU_qualification'], prior_DEV_receipt=spec['prior_DEV_receipt'],
        diagnostic_CPU_fixture=spec['diagnostic_CPU_fixture'], diagnostic_CPU_fixture_spec=spec['diagnostic_CPU_fixture_spec'],
        diagnostic_CPU_fixture_source=spec['diagnostic_CPU_fixture_source'],
        original_four_raw_terminal_gates_before_scientific_access=True, full610_CPU_before_CUDA_required=True,
        no_AP=True, no_TEST=True, no_retry=True, no_bias=True, no_regularizer=True, full_raster_supervision=True,
        optimizer_or_RNG_restored=False, no_feature_cache=False, audit_packet_only=True,
        GPU_optimizer_bitwise_determinism_claimed=False))
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
    require(torch.get_num_threads() == torch.get_num_interop_threads() == 2 and torch.__version__ == '2.0.1+cu118' and
        torch.optim.LBFGS.__module__ == 'torch.optim.lbfgs', 'Qualified Torch/threads/solver version differs')
    source_file(torch.optim.LBFGS, spec['solver_source'])
    require(Path(torch.version.__file__).resolve(strict=True) == bound(spec['torch_version_source']).resolve(strict=True), 'Actual Torch version SOURCE differs')
    source_file(torch.nn.functional.interpolate, spec['interpolation_SOURCE'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    register_all_modules(init_default_scope=False)
    init_default_scope('mmdet')
    pref = load_source(spec['micro_preflight_source'], 'FIT_native_tensor_helpers')
    initializer = load_source(spec['initializer_source'], 'FIT_original_COCO_ship_initializer')
    old = load_source(spec['native_helpers'], 'FIT_original_image_loader'); old.torch = torch
    native = load_source(spec['native_preflight_helpers'], 'FIT_GT_free_forward_loader')
    official = load_source(spec['official_evaluation_source'], 'FIT_official_teacher_state_loader')
    classes = tuple(CocoDataset.METAINFO['classes'])
    require(len(classes) == 80 and classes[8] == 'boat', 'Original COCO80 boat order differs')
    model, state, meta, loading = official.prepare_model(torch, pref, initializer, None, None, None,
        MODELS, revert_sync_batchnorm, Config, classes, spec, teacher_row, 'native0')
    model.requires_grad_(False).eval()
    require(len(state) == 610 and len(list(model.parameters())) == 451 and not torch.cuda.is_initialized(), 'Full native610 CPU reconstruction required before CUDA')
    dataset = CocoDataset(ann_file=str(bound(spec['FIT_annotation'])), data_prefix=dict(img=spec['image_root']),
        metainfo=dict(classes=('ship',)), pipeline=[], filter_cfg=None, test_mode=True)
    coco = COCO(str(bound(spec['FIT_annotation'])))
    require(len(dataset) == len(coco.imgs) == 741 and len(coco.anns) == 1661 and
        [int(dataset.get_data_info(index)['img_id']) for index in range(12)] == SUBSET and
        [len(coco.get_ann_ids(img_ids=[image_id])) for image_id in SUBSET] == GT_COUNTS and not torch.cuda.is_initialized(),
        'Prelocked actual first12 FIT IDs/all18 annotations differ')
    torch.cuda.init()
    torch.cuda.set_per_process_memory_fraction(4096 * 1024 ** 2 / torch.cuda.get_device_properties(0).total_memory, 0)
    result = diagnose_images(torch, np, masks_api, base, core, pref, old, native, model, state, loading,
        spec, dataset, coco, output, started)
    del model
    cleanup = pref.clear_gpu(torch)
    report = dict(schema=REPORT_SCHEMA, status='FROZEN_FEATURE_LINEAR_READOUT_FIT_DIAGNOSTIC_GPU_COMPLETE_NO_AP',
        phase=spec['phase'], source=spec['entry_source'], spec=spec_ref, mode='native0', error=None, SOURCE_contract=spec['SOURCE_contract'],
        teacher_checkpoint=spec['teacher_checkpoint'], teacher_qualification=spec['teacher_qualification'],
        prior_DEV_spec=spec['prior_DEV_spec'], prior_DEV_CPU_qualification=spec['prior_DEV_CPU_qualification'],
        prior_DEV_receipt=spec['prior_DEV_receipt'], diagnostic_CPU_fixture=spec['diagnostic_CPU_fixture'],
        diagnostic_CPU_fixture_spec=spec['diagnostic_CPU_fixture_spec'], diagnostic_CPU_fixture_source=spec['diagnostic_CPU_fixture_source'],
        subset_image_ids=SUBSET, subset_images=12, subset_raw_GT=18,
        solver=SOLVER, result=result, cleanup=cleanup, full610_CPU_before_CUDA=True,
        full_native610_and_BN_exact=True, native451_frozen=True, GT_removed_from_forward=True,
        native100_GPU_mask200_and_official_masks_TorchEqual=True, retained_candidates=['native_best100', 'final'],
        full_raster_BCE_unweighted_mean=True, no_bias=True, no_MLP=True, no_GT_box_crop=True,
        no_sampled_points=True, no_regularizer=True, no_threshold_tuning=True, optimizer_or_RNG_restored=False,
        no_feature_cache=False, audit_packet_only=True, no_AP=True, AP_computed=False, no_TEST=True,
        efficacy_claimed=False, convergence_claimed=False, optimal_IoU_upper_bound_claimed=False,
        generator_versus_budget_causal_identification_claimed=False, GPU_optimizer_bitwise_determinism_claimed=False,
        oracle1_vs_native100_bitwise_algebra_claimed=False, max_eval_is_soft_limit=True,
        torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(0), elapsed_seconds=time.monotonic() - started,
        allocator_limit_MiB=4096, GPU_scope='whole GPU must be verified from actual owner receipt',
        evidence_limits=['Finite fits on the prelocked12 FIT images/all18 raw GT only, not a representative population.',
            'GT-best100 and each GT-specific free coefficient use GT and are diagnostic, not deployable.',
            'A successful candidate witnesses restricted frozen F plus native rendering feasibility; failure remains unresolved.',
            'Full-raster BCE is not IoU optimization; no optimum, convergence or generator-versus-budget causal conclusion is certified.',
            'Configured max_eval80 is soft under strong-Wolfe; actual closure/function/iteration counts are retained.',
            'CUDA bilinear backward can be nondeterministic despite fixed supervision/no RNG; no bitwise optimizer path claim.',
            'Oracle1 and native100 FP32 GEMM grouping can differ; only actual native100 GPU replay is required TorchEqual.',
            'Audit CPU tensor packets enable tolerance-based independent rendering; they are not a reusable feature-cache training recipe.'])
    require(report['elapsed_seconds'] < 1800 and cleanup['allocated_after'] == 0, 'Actual diagnostic deadline/cleanup differs')
    core.write(output / 'REPORT.json', report)
    core.write(output / 'QUALIFIED.json', dict(schema='frozen-feature-linear-readout-FIT-aggregate-v1',
        status='ROOT_FROZEN_FEATURE_LINEAR_READOUT_FIT_PENDING_CPU_READER', phase=spec['phase'], source=spec['entry_source'],
        spec=spec_ref, report=reference(output / 'REPORT.json'), result=result, independently_CPU_qualified=False,
        AP_computed=False, optimal_IoU_upper_bound_claimed=False, efficacy_claimed=False))
    print(json.dumps(dict(status='FIT_FEASIBILITY_EXPORT_COMPLETE_CPU_QUALIFICATION_PENDING', images=12, raw_GT=18)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--owner-execute', action='store_true')
    args = parser.parse_args()
    require(args.owner_execute, 'Inert SOURCE requires explicit ROOT-owned execution after four original RAW gates')
    run(args.spec)


if __name__ == '__main__':
    main()
