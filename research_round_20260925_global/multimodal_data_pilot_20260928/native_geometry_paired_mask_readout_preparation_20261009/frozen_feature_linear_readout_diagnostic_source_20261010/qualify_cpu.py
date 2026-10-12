"""Independent CPU inspection of a bounded frozen-feature FIT witness.

SOURCE authoring is inert. ROOT alone executes after current and all four
original owned raw terminal gates. This reader independently decodes every
selected raw annotation, checks native100 and final RLEs, projects FP32
coefficients and renders the original raster, and recomputes full-raster BCE,
norms and the two-candidate witness. CPU/GPU binary differences are counted
and permitted only in a stated near-zero envelope. No CPU bit-exact native
replay, optimizer replay, AP, optimum, causal or general efficacy is claimed.
"""
import argparse
import gc
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import time
import types

SPEC_SCHEMA = 'ROOT-frozen-feature-linear-readout-FIT-diagnostic-spec-v1'
REPORT_SCHEMA = 'frozen-feature-linear-readout-FIT-report-v1'
REPORT_STATUS = 'FROZEN_FEATURE_LINEAR_READOUT_FIT_DIAGNOSTIC_GPU_COMPLETE_NO_AP'
QUAL_SCHEMA = 'ROOT-frozen-feature-linear-readout-FIT-CPU-qualification-v1'
QUAL_STATUS = 'ROOT_FROZEN_FEATURE_LINEAR_READOUT_FIT_CPU_QUALIFIED'
SUBSET = [1, 2, 3, 5, 6, 8, 9, 10, 11, 12, 13, 14]
COUNTS = [1, 1, 1, 3, 1, 1, 1, 1, 1, 3, 2, 2]
SOLVER = dict(name='torch.optim.LBFGS', lr=1, max_iter=40, max_eval=80,
    history_size=10, tolerance_grad=1e-7, tolerance_change=1e-9, line_search_fn='strong_wolfe')
OWNER_SHA = 'ef9a9f17d4fb1bc3464936095ef799021dd880f8b15488d558a871c08b6e1924'
ATOL, RTOL = 1e-4, 1e-4


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def expect(row, expected, label):
    require(isinstance(row, dict), label + ' must be an object')
    for key, value in expected.items():
        actual = row[key]
        require(actual == value and (type(value) not in (bool, int) or type(actual) is type(value)),
            label + ' differs: ' + key)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def reference(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def bound(row):
    require(isinstance(row, dict) and set(row) == {'path', 'bytes', 'sha256'} and
        Path(row['path']).is_absolute() and type(row['bytes']) is int and row['bytes'] > 0 and
        isinstance(row['sha256'], str) and len(row['sha256']) == 64 and
        set(row['sha256']) <= set('0123456789abcdef'), 'Complete absolute pinned reference required')
    require(reference(row['path']) == row, 'Pinned bytes differ: ' + row['path'])
    return Path(row['path'])


def read_ref(row):
    return read(bound(row))


def finite(value, label, nonnegative=False):
    require(type(value) in (float, int) and math.isfinite(value) and (not nonnegative or value >= 0),
        label + ' must be finite' + (' and nonnegative' if nonnegative else ''))


def close_number(actual, expected, label):
    finite(actual, label)
    finite(expected, label + ' independent value')
    require(math.isclose(actual, expected, rel_tol=RTOL, abs_tol=ATOL), label + ' independently differs')


def terminal(receipt, phase):
    require(receipt.get('status') == 'ACTUAL_OWNED_TERMINAL' and receipt.get('original_identity') is not None and
        type(receipt.get('returncode')) is int and receipt['returncode'] == 0 and receipt.get('root_wait_reaped') is True and
        receipt.get('held_handle_exit') == dict(wait_state='WAIT_OBJECT_0', actual_exit_code=0) and
        receipt.get('numeric_birth_absence', {}).get('original_birth_absent') is True and
        all(key in receipt and receipt[key] is None for key in ('abort', 'error')) and
        receipt.get('wait_errors') is False and receipt.get('GPU_observation_errors') == [],
        'STOP before hashes/imports/science: actual raw held0/reap/birth absence/noerrors required')
    if 'phase' in receipt:
        require(receipt['phase'] == phase, 'Raw terminal phase differs')
    peak = receipt.get('whole_GPU_observed_peak_MiB')
    finite(peak, 'Actual whole-GPU peak', True)
    require(peak <= 8192 and receipt.get('whole_GPU_used_limit_MiB') == 8192 and
        receipt.get('whole_leaf_VRAM_certified') is False and
        receipt.get('GPU_scope') == 'whole GPU observed, not per-process WDDM allocation',
        'Actual whole-GPU resource scope differs')


def load_source(row, name):
    path = bound(row)
    value = path.read_bytes()
    require(len(value) == row['bytes'] and hashlib.sha256(value).hexdigest() == row['sha256'], 'SOURCE changed before import')
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(value, str(path), 'exec'), module.__dict__)
    return module


def source_gate(spec_path, spec, raw):
    source = reference(Path(__file__).resolve().with_name('diagnose.py'))
    reader, spec_ref = reference(__file__), reference(spec_path)
    admission_ref = reference(spec['GPU_admission_file'])
    admission = read_ref(admission_ref)
    owner = reference(spec_path.parent / 'finite_point_cache_execution_20261005_v2' / 'run_owned_gpu_limited.py')
    expect(admission, dict(status='ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE', automatic_retry=False,
        phase=spec['phase'], allowed_cpus=[0, 1], cuda_visible_devices='0', qualified_leaf_deadline_seconds=1800,
        whole_GPU_used_limit_MiB=8192, driver_sha256=OWNER_SHA,
        entry_args=['--spec', str(spec_path), '--owner-execute']), 'Current diagnostic admission')
    require(raw['admission_sha256'] == admission_ref['sha256'] and owner['sha256'] == OWNER_SHA and
        Path(admission['entry']).resolve() == Path(source['path']) and spec['entry_source'] == source and
        spec['CPU_reader'] == reader and source in spec['sources'] and reader in spec['sources'] and
        len({row['path'] for row in spec['sources']}) == len(spec['sources']), 'Current owner/source/SPEC bindings differ')
    hashes = admission['bound_source_sha256']
    require(hashes.get(str(spec_path)) == spec_ref['sha256'], 'Current SPEC is not owner-bound')
    for row in spec['sources']:
        bound(row)
        require(hashes.get(row['path']) == row['sha256'], 'Complete inherited/new SOURCE closure is not owner-bound')
    contract = read_ref(spec['SOURCE_contract'])
    require(Path(spec['SOURCE_contract']['path']).resolve() == Path(__file__).resolve().with_name('SOURCE_CONTRACT.json') and
        spec['SOURCE_contract'] in spec['sources'], 'Canonical inert SOURCE contract absent from owner closure')
    expect(contract, dict(kind='INERT_SOURCE_AUTHORING_CONTRACT_NOT_SCIENTIFIC_ADMISSION', schema=SPEC_SCHEMA,
        entry_name='diagnose.py', reader_name='qualify_cpu.py', report_schema=REPORT_SCHEMA, report_status=REPORT_STATUS,
        qualification_schema=QUAL_SCHEMA, qualification_status=QUAL_STATUS, subset_ids=SUBSET,
        solver=SOLVER, candidates=['native_best100', 'final']), 'Inert prelocked authoring contract')
    for role, row in contract['base_role_SOURCE_refs'].items():
        require(row in spec['sources'], 'Original admitted DEV SOURCE absent: ' + role)
        if role not in ('entry_source', 'CPU_reader'):
            require(spec[role] == row, 'Inherited canonical SOURCE role differs: ' + role)
    for role in ('solver_source', 'torch_version_source', 'native_mask_head_SOURCE', 'native_fusion_SOURCE', 'interpolation_SOURCE'):
        require(spec[role] == contract[role] and spec[role] in spec['sources'], 'Canonical library SOURCE role differs: ' + role)
    require(spec['prior_DEV_spec'] == contract['source_baseline_evaluation'] and
        contract['source_baseline_evaluation_entry'] == contract['base_role_SOURCE_refs']['entry_source'],
        'Canonical actual prior DEV SPEC/entry lineage differs')
    if 'scientific_spec' in admission:
        require(admission['scientific_spec'] == spec_ref, 'Admitted current scientific SPEC differs')
    return dict(source=source, reader=reader, spec=spec_ref, GPU_admission=admission_ref, owner_driver=owner), contract


def prior_DEV_gate(spec, raw, prior, contract):
    q = read_ref(spec['prior_DEV_CPU_qualification'])
    old_source, old_reader = contract['base_role_SOURCE_refs']['entry_source'], contract['base_role_SOURCE_refs']['CPU_reader']
    expect(q, dict(schema=prior.QUAL_SCHEMA, status=prior.QUAL_STATUS, phase=spec['prior_DEV_phase'],
        source=old_source, reader_source=old_reader, spec=spec['prior_DEV_spec'], parent_receipt=spec['prior_DEV_receipt'],
        expected_epoch=24, short2_epoch=2, modes=list(prior.MODES), endpoints=list(prior.ENDPOINTS),
        raw_current_teacher_preflight_SHORT2_terminals_before_scientific_access=True,
        complete_SOURCE_closure_verified_before_scientific_import=True, all187_images_all380_original_GT_preserved=True,
        all100_predictions_per_image_and_empty_masks_preserved=True, no_added_NMS_score_cutoff_or_query_filter=True,
        full_native610_teacher_and_typed_BN106_affine159_buffers_CPU_exact=True,
        strict_full610_or612_endpoint_CPU_loaded=True, zero_native_state_mutants=True,
        official_COCO_maxDets=[1, 10, 100], official_segmentation_COCO12_independently_computed=True,
        official_fusion_TorchEqual_masks_scores_source_observed=True, pixel_fusion_threshold_or_quality_mean_CPU_recomputed=False,
        no_TEST=True, blind_TEST_claimed=False, convergence_claimed=False, SOTA_claimed=False, scientific_method_verdict=None),
        'Actual completed independent prior DEV qualification')
    old_spec, admission = read_ref(q['spec']), read_ref(q['GPU_admission'])
    expect(old_spec, dict(schema=prior.SPEC_SCHEMA, phase=spec['prior_DEV_phase'], entry_source=old_source,
        CPU_reader=old_reader, endpoints=list(prior.ENDPOINTS), modes=list(prior.MODES), evaluation_images=187,
        evaluation_GT=380, expected_epoch=24, short2_epoch=2, head_seed_base=100000,
        CPU_affinity=[0, 1], deadline_seconds=1800, allocator_limit_MiB=4096, whole_GPU_owner_limit_MiB=8192),
        'Canonical actual prior DEV SPEC')
    expect(admission, dict(status='ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE', automatic_retry=False,
        phase=spec['prior_DEV_phase'], allowed_cpus=[0, 1], cuda_visible_devices='0', qualified_leaf_deadline_seconds=1800,
        whole_GPU_used_limit_MiB=8192, driver_sha256=OWNER_SHA,
        entry_args=['--spec', q['spec']['path'], '--owner-execute']), 'Original prior DEV owned admission')
    require(len(old_spec['sources']) == 69 and raw['admission_sha256'] == q['GPU_admission']['sha256'] and
        Path(admission['entry']).resolve() == Path(old_source['path']).resolve() and
        Path(old_spec['GPU_admission_file']).resolve() == Path(q['GPU_admission']['path']).resolve() and
        admission['bound_source_sha256'].get(q['spec']['path']) == q['spec']['sha256'], 'Original69 DEV closure/admission join differs')
    for row in old_spec['sources']:
        bound(row)
        require(row in spec['sources'] and admission['bound_source_sha256'].get(row['path']) == row['sha256'],
            'Original actual DEV closure not completely retained and owner-bound')
    for role in contract['base_role_SOURCE_refs']:
        require(old_spec[role] == contract['base_role_SOURCE_refs'][role], 'Original DEV source role differs: ' + role)
    for role in ('teacher_checkpoint', 'teacher_qualification', 'teacher_training_spec', 'teacher_receipt', 'teacher_phase',
                 'preflight_CPU_qualification', 'preflight_training_spec', 'preflight_receipt', 'preflight_phase',
                 'short2_CPU_qualification', 'short2_training_spec', 'short2_receipt', 'short2_phase',
                 'checkpointmap', 'qualificationmap', 'FIT_annotation', 'DEV_annotation', 'image_root'):
        require(old_spec[role] == spec[role], 'Original DEV/current diagnostic lineage differs: ' + role)
    for role in ('teacher_checkpoint', 'teacher_qualification', 'teacher_training_spec', 'teacher_receipt',
                 'preflight_CPU_qualification', 'preflight_receipt', 'short2_CPU_qualification', 'short2_training_spec',
                 'short2_receipt', 'checkpointmap', 'qualificationmap', 'DEV_annotation'):
        require(q[role] == spec[role], 'Actual prior DEV qualification lineage differs: ' + role)
    for role in ('source', 'reader_source', 'spec', 'report', 'start', 'GPU_qualification', 'parent_receipt', 'GPU_admission', 'owner_driver'):
        bound(q[role])
    require(q['owner_driver']['sha256'] == OWNER_SHA and set(q['mode_results']) == set(q['endpoint_states']) == set(prior.ENDPOINTS),
        'Prior actual allfive endpoint inspection/owner identity differs')
    return q


def cpu_tensor(torch, value, shape, label):
    require(isinstance(value, torch.Tensor) and value.device.type == 'cpu' and value.dtype == torch.float32 and
        list(value.shape) == list(shape) and not value.requires_grad and bool(torch.isfinite(value).all()),
        label + ' must be finite detached CPU FP32 with exact shape')


def project(torch, feature, coefficient):
    """Independent actual native linear coefficient projection; no bias."""
    return torch.einsum('bqc,bchw->bqhw', coefficient, feature)


def render(torch, logits200, img_shape, original_shape):
    """Independent fixed bilinear800, actual-image crop, original-raster resize."""
    height, width = [int(v) for v in img_shape[:2]]
    original_h, original_w = [int(v) for v in original_shape[:2]]
    require(0 < height <= 800 and 0 < width <= 800 and original_h > 0 and original_w > 0,
        'Renderer actual/original shapes are invalid')
    value = torch.nn.functional.interpolate(logits200, size=(800, 800), mode='bilinear', align_corners=False)
    value = value[:, :, :height, :width]
    return torch.nn.functional.interpolate(value, size=(original_h, original_w), mode='bilinear', align_corners=False)


def logit_check(torch, actual, expected, label):
    require(actual.shape == expected.shape and actual.dtype == expected.dtype == torch.float32 and
        bool(torch.isfinite(actual).all()) and bool(torch.isfinite(expected).all()), label + ' finite FP32 shape differs')
    absolute = (actual - expected).abs()
    require(torch.allclose(actual, expected, rtol=RTOL, atol=ATOL), label + ' exceeds independent FP32 logit tolerance')
    return dict(max_absolute_error=float(absolute.max().item()), mean_absolute_error=float(absolute.mean().item()),
        rtol=RTOL, atol=ATOL, bitwise_equality_claimed=False)


def binary_check(torch, cpu_logits, GPU_binary, label, GPU_logits=None):
    require(cpu_logits.shape == GPU_binary.shape and GPU_binary.dtype == torch.bool, label + ' raster shape differs')
    flips = (cpu_logits > 0) != GPU_binary
    anchor = cpu_logits if GPU_logits is None else GPU_logits
    # For omitted all100 GPU original logits this is a consistency envelope,
    # not a measured cross-device original-logit error or exact CPU replay.
    envelope = ATOL + RTOL * anchor.abs()
    require(not bool((flips & (anchor.abs() > envelope)).any()), label + ' binary flip lies outside near-zero envelope')
    return dict(binary_flip_count=int(flips.sum().item()), raster_pixels=int(flips.numel()),
        near_zero_envelope_atol=ATOL, near_zero_envelope_rtol=RTOL,
        envelope_anchor='CPU_reconstructed_logit' if GPU_logits is None else 'exported_GPU_logit',
        measured_GPU_original_logit_comparison=GPU_logits is not None, CPU_bit_exact_replay_claimed=False)


def decode_RLE(np, masks, row, shape, label):
    require(isinstance(row, dict) and set(row) == {'size', 'counts'} and row['size'] == list(shape) and
        isinstance(row['counts'], str), label + ' exact original-size compressed RLE required')
    encoded = dict(size=row['size'], counts=row['counts'].encode('ascii'))
    decoded = masks.decode(encoded)
    require(decoded.shape == tuple(shape) and bool(np.isin(decoded, [0, 1]).all()) and
        int(masks.area(encoded)) == int(decoded.sum()), label + ' decoded binary area differs')
    canonical = masks.encode(np.asfortranarray(decoded.astype(np.uint8)))
    require(canonical['counts'].decode('ascii') == row['counts'], label + ' RLE is not canonical')
    return decoded.astype(np.bool_)


def binary_metrics(torch, prediction, target):
    intersection = int((prediction & target).sum().item())
    p_area, g_area = int(prediction.sum().item()), int(target.sum().item())
    union = p_area + g_area - intersection
    return dict(intersection=intersection, union=union, prediction_area=p_area, GT_area=g_area,
        IoU=intersection / union if union else 1.0,
        Dice=2 * intersection / (p_area + g_area) if p_area + g_area else 1.0,
        precision=intersection / p_area if p_area else float(g_area == 0),
        recall=intersection / g_area if g_area else float(p_area == 0),
        prediction_empty=p_area == 0, GT_empty=g_area == 0)


def metric_check(actual, expected, label):
    require(set(actual) == set(expected), label + ' exact metric field population differs')
    for key, value in expected.items():
        if type(value) in (int, bool):
            require(type(actual[key]) is type(value) and actual[key] == value, label + ' integer/empty metric differs: ' + key)
        else:
            require(type(actual[key]) in (int, float) and math.isclose(actual[key], value, abs_tol=1e-12, rel_tol=1e-12),
                label + ' binary metric differs: ' + key)


def solver_check(audit, baseline_BCE, final_BCE, baseline_norm, final_norm):
    expect(audit, dict(config=SOLVER, max_eval_is_soft_limit=True, nonclosure_initial_evaluations=1,
        nonclosure_final_evaluations=1, actual_coefficient_grad_connected=True), 'Exact source-observed solver audit')
    calls, evaluations, iterations = audit['closure_calls'], audit['optimizer_func_evals'], audit['optimizer_n_iter']
    require(type(calls) is int and calls >= 1 and type(evaluations) is int and calls == evaluations and
        type(iterations) is int and 0 <= iterations <= 40 and len(audit['solver_callbacks']) == calls,
        'Actual LBFGS closure/func-eval/n-iter counts differ; max_eval80 is not a hard closure ceiling')
    for index, callback in enumerate(audit['solver_callbacks']):
        require(callback['closure_index'] == index + 1, 'Actual closure sequence differs')
        for name in ('loss', 'coefficient_norm', 'gradient_norm'):
            finite(callback[name], 'Source-observed closure ' + name, True)
    close_number(audit['initial_loss'], baseline_BCE, 'Initial full-raster loss')
    close_number(audit['final_loss'], final_BCE, 'Final full-raster loss')
    close_number(audit['initial_norm'], baseline_norm, 'Initial coefficient norm')
    close_number(audit['final_norm'], final_norm, 'Final coefficient norm')
    finite(audit['final_gradient_norm'], 'Source-observed final coefficient gradient norm', True)
    return dict(config=SOLVER, closure_calls=calls, optimizer_func_evals=evaluations, optimizer_n_iter=iterations,
        max_eval_is_soft_limit=True, hard_max_eval80_claimed=False, CPU_optimizer_or_gradient_trajectory_replayed=False)


def fixture_gate(spec):
    require(all(spec[role] in spec['sources'] for role in ('diagnostic_CPU_fixture',
        'diagnostic_CPU_fixture_spec', 'diagnostic_CPU_fixture_source')), 'Actual fixture qualification/manifest/source not owner-bound')
    fixture = read_ref(spec['diagnostic_CPU_fixture'])
    manifest = read_ref(spec['diagnostic_CPU_fixture_spec'])
    execution = dict(device='cpu', cpu_affinity=[0, 1], torch_threads=2, torch_interop_threads=2, cuda_visible_devices='')
    sources = dict(producer=spec['entry_source'], fixture=spec['diagnostic_CPU_fixture_source'], contract=spec['SOURCE_contract'])
    expect(manifest, dict(schema='ROOT-frozen-feature-linear-readout-synthetic-SOURCE-spec-v1', inputs='SOURCE_ONLY',
        sources=sources, execution=execution), 'Actual independently executed SOURCE-only fixture manifest')
    require(set(manifest) == {'schema', 'inputs', 'sources', 'execution'} and
        all(row in spec['sources'] for row in sources.values()), 'Complete fixture SOURCE closure differs')
    expect(fixture, dict(status='FROZEN_FEATURE_LINEAR_READOUT_SYNTHETIC_CPU_PASS', sources=sources,
        execution=execution, spec_path=spec['diagnostic_CPU_fixture_spec']['path'], torch_version='2.0.1+cu118', solver=SOLVER),
        'Actual SOURCE-only synthetic CPU PASS reference')
    expect(fixture, dict(synthetic_execution_performed=True, coefficient_only_no_bias=True,
        zero_F_connected_zero_and_stationary=True, frozen_F_unchanged=True, external_CPU_RNG_unchanged=True,
        native100_query_axis_and_lowest_ordinal_ties=True, max_eval_is_soft_limit=True,
        real_data_feasibility_qualified=False, native_model_integration_qualified=False, AP_computed=False,
        efficacy_claimed=False, convergence_claimed=False, optimal_IoU_upper_bound_claimed=False,
        GPU_optimizer_bitwise_determinism_claimed=False), 'Actual synthetic fixture assertion scope')
    fit = fixture['finite_fit']
    for key in ('initial_BCE', 'final_BCE', 'initial_IoU', 'final_IoU', 'final_coefficient_norm', 'final_gradient_norm'):
        finite(fit[key], 'Actual synthetic finite fit ' + key, True)
    require(type(fit['closure_calls']) is int and fit['closure_calls'] >= 1 and
        fit['closure_calls'] == fit['optimizer_func_evals'] and
        type(fit['optimizer_n_iter']) is int and 0 <= fit['optimizer_n_iter'] <= 40 and
        fit['final_BCE'] < fit['initial_BCE'] - 1e-4 and fit['final_IoU'] > fit['initial_IoU'],
        'Actual synthetic LBFGS counters differ')
    for role in ('solver_source', 'torch_version_source', 'native_mask_head_SOURCE', 'native_fusion_SOURCE', 'interpolation_SOURCE'):
        require(fixture[role] == spec[role], 'Actual synthetic fixture library SOURCE join differs: ' + role)
    # ROOT supplies the actual tool-exit/admission evidence. A status in this
    # pinned file alone is not claimed to prove that an external tool ran.
    return dict(qualification=spec['diagnostic_CPU_fixture'], manifest=spec['diagnostic_CPU_fixture_spec'],
        sources=sources, status=fixture['status'], actual_tool_exit_is_ROOT_admission_evidence=True)


def library_paths(torch, model, spec):
    require(torch.__version__ == '2.0.1+cu118', 'Actual pinned Torch2.0.1cu118 required')
    actual = dict(solver_source=inspect.getsourcefile(torch.optim.LBFGS),
        torch_version_source=torch.version.__file__, interpolation_SOURCE=torch.nn.functional.__file__,
        native_mask_head_SOURCE=inspect.getsourcefile(type(model.panoptic_head)),
        native_fusion_SOURCE=inspect.getsourcefile(type(model.panoptic_fusion_head)))
    for role, path in actual.items():
        require(Path(path).resolve(strict=True) == Path(spec[role]['path']).resolve(strict=True),
            'Actual imported library SOURCE path differs: ' + role)
        bound(spec[role])
    require(model.panoptic_head.num_queries == 100 and model.panoptic_head.num_classes == 1 and
        model.panoptic_fusion_head.num_classes == model.panoptic_fusion_head.num_things_classes == 1 and
        model.panoptic_fusion_head.num_stuff_classes == 0, 'Canonical single-class100-query native architecture differs')


def full_state_check(torch, old, pre, model, teacher, spec, record):
    native, modules, affine, buffers = pre.typed_native(torch, model)
    require(len(model.state_dict()) == len(teacher) == 610 and len(list(model.parameters())) == 451 and
        set(model.state_dict()) == set(teacher) and all(torch.equal(value, teacher[name])
        for name, value in model.state_dict().items()), 'Independent strict full610 teacher CPU endpoint differs')
    proof = read_ref(record['state_proof'])
    expect(proof, dict(schema='frozen-feature-linear-readout-FIT-state-proof-v1', mode='native0',
        checkpoint=spec['teacher_checkpoint'], qualification=spec['teacher_qualification'],
        full_native_state_exact=True, frozen_BN_all_exact=True, native_parameter_grads_None=True,
        all_parameters_frozen=True, all_modules_eval=True, hooks_removed=True, hook_cache_clear=True,
        CPU_readback_exact=True, native_state_entries=610, native_parameter_tensors=451,
        oracle_parameters_registered_in_model=False, optimizer_or_RNG_restored=False), 'Actual full frozen GPU endpoint proof')
    require(proof['BN_parameter_names'] == affine and proof['BN_buffer_names'] == buffers,
        'Actual typed BN name population differs from independent architecture')
    loading = proof['loading_proof']
    expect(loading, dict(mode='native0', checkpoint=spec['teacher_checkpoint'], qualification=spec['teacher_qualification'],
        CP_qualified=True, strict_fullstate_loaded=True, native_state_tensors=610, adapter_state_tensors=0,
        registered_parameters=451, original_frozen_BN_exact=True, inference_backbone_with_cp=False,
        no_random_missing_keys=True), 'Actual complete original teacher loading proof')
    require(loading['BN_parameter_names'] == affine and loading['BN_buffer_names'] == buffers,
        'Original loading typed BN names differ')
    old.check_state(torch, teacher, loading['trained_fullstate'], 'Actual strictly loaded E24 native610')
    for role in ('before', 'after', 'native_teacher610'):
        old.check_state(torch, teacher, proof[role], 'Frozen diagnostic actual ' + role + '610')
    require(all(torch.equal(model.state_dict()[name], teacher[name]) for name in affine + buffers),
        'Independent typed BN106+159 state identity differs')
    return dict(state_proof=record['state_proof'], checkpoint=spec['teacher_checkpoint'],
        native_state_tensors=610, native_parameter_tensors=len(native), typed_BN_modules=len(modules),
        typed_BN_affine_tensors=len(affine), typed_BN_buffer_tensors=len(buffers),
        strict_full610_teacher_CPU_load=True, all_before_after_and_teacher610_digest_identities_exact=True,
        zero_native_state_mutants=True, typed_BN106_affine159_buffers_CPU_exact=True,
        GPU_native_forward_or_optimizer_CPU_replayed=False)


def GT_check(torch, np, masks, coco, old, row, identity, feature, coefficients, native_original, native_predictions):
    image_id, annotation_id = row['image_id'], row['annotation_id']
    annotation = coco.anns[annotation_id]
    shape = identity['original_shape']
    require(annotation['image_id'] == image_id == identity['image_id'] and annotation['category_id'] == 1 and
        row['raw_annotation'] == annotation and row['original_shape'] == shape and
        row['candidates'] == ['native_best100', 'final'], 'Complete original raw GT/candidate population differs')
    # Empty polygon populations remain empty raw rasters. No crowd, ignore,
    # area, bbox or parser filter is applied to this independent annotation set.
    decoded = np.zeros(tuple(shape), dtype=np.uint8) if annotation.get('segmentation') == [] else coco.annToMask(annotation)
    require(decoded.shape == tuple(shape) and bool(np.isin(decoded, [0, 1]).all()), 'Raw GT raster shape/binary differs')
    GT_uint8 = torch.from_numpy(np.ascontiguousarray(decoded.astype(np.uint8)))
    target = GT_uint8.bool()
    require(row['GT_digest'] == old.tensor_digest(torch, GT_uint8, 'uint8'), 'Independent complete original GT raster digest differs')
    metrics = {query: binary_metrics(torch, native_predictions[query], target) for query in range(100)}
    require(row['native100_IoU'] == [metrics[query]['IoU'] for query in range(100)],
        'Every actual native query IoU differs from independent RLE metric')
    # Query ordinal is the actual native tensor row, independent of unsorted
    # official topk export order. It is not physical object identity.
    query = min(range(100), key=lambda value: (-metrics[value]['IoU'], value))
    require(type(row['baseline_query_id']) is int and row['baseline_query_id'] == query and
        row['baseline']['query_id'] == query, 'Actual native GT-best100 lowest-query-ordinal tie differs')
    baseline = row['baseline']
    metric_check(baseline['metrics'], metrics[query], 'Independent native best100 baseline')
    baseline_binary = torch.from_numpy(decode_RLE(np, masks, baseline['mask_RLE'], shape, 'Native baseline RLE'))
    require(torch.equal(baseline_binary, native_predictions[query]), 'Baseline RLE does not select actual native best100 mask')
    packet = torch.load(str(bound(row['audit_packet'])), map_location='cpu')
    require(set(packet) == {'schema', 'image_id', 'annotation_id', 'baseline_query_id', 'baseline_original_logits',
        'final_coefficient', 'final_original_logits'}, 'Complete per-GT audit packet field population differs')
    expect(packet, dict(schema='frozen-feature-linear-readout-GT-audit-v1', image_id=image_id,
        annotation_id=annotation_id, baseline_query_id=query), 'Per-GT actual CPU audit identity')
    cpu_tensor(torch, packet['baseline_original_logits'], shape, 'Actual baseline original logits')
    cpu_tensor(torch, packet['final_original_logits'], shape, 'Actual final original logits')
    cpu_tensor(torch, packet['final_coefficient'], [256], 'Actual final coefficient without bias')
    require(torch.equal(packet['baseline_original_logits'] > 0, baseline_binary),
        'Exported actual GPU baseline logit>0 is not exact native baseline RLE')
    baseline_logit = logit_check(torch, native_original[query], packet['baseline_original_logits'],
        'Direct selected native baseline original-raster CPU/GPU logit')
    baseline_flips = binary_check(torch, native_original[query], baseline_binary, 'Selected baseline CPU binary',
        packet['baseline_original_logits'])
    final_cpu = render(torch, project(torch, feature, packet['final_coefficient'].reshape(1, 1, 256)),
        identity['actual_img_shape'], shape)[0, 0]
    final_logit = logit_check(torch, final_cpu, packet['final_original_logits'], 'Direct final original-raster CPU/GPU logit')
    final_binary = torch.from_numpy(decode_RLE(np, masks, row['final']['mask_RLE'], shape, 'Final actual GPU RLE'))
    require(torch.equal(final_binary, packet['final_original_logits'] > 0), 'Final GPU exported logit>0 RLE is not exact')
    final_flips = binary_check(torch, final_cpu, final_binary, 'Final CPU binary', packet['final_original_logits'])
    final_metrics = binary_metrics(torch, final_binary, target)
    metric_check(row['final']['metrics'], final_metrics, 'Independent final RLE metrics')
    # These full-raster losses use the exported actual GPU logits on CPU, not
    # a sampled target, point objective, regularized loss or solver trajectory.
    target_float = target.float()
    baseline_BCE = float(torch.nn.functional.binary_cross_entropy_with_logits(
        packet['baseline_original_logits'], target_float, reduction='mean').item())
    final_BCE = float(torch.nn.functional.binary_cross_entropy_with_logits(
        packet['final_original_logits'], target_float, reduction='mean').item())
    baseline_norm = float(coefficients[0, query].norm().item())
    final_norm = float(packet['final_coefficient'].norm().item())
    for name, actual, expected in (('baseline full-raster BCE', baseline['full_raster_BCE'], baseline_BCE),
        ('final full-raster BCE', row['final']['full_raster_BCE'], final_BCE),
        ('baseline coefficient norm', baseline['coefficient_norm'], baseline_norm),
        ('final coefficient norm', row['final']['coefficient_norm'], final_norm)):
        close_number(actual, expected, name)
    candidate = 'final' if final_metrics['IoU'] > metrics[query]['IoU'] else 'native_best100'
    best = final_metrics if candidate == 'final' else metrics[query]
    expect(row['best_of_two'], dict(candidate=candidate, IoU=best['IoU'], Dice=best['Dice'],
        tie_rule='native_best100_first', not_optimal_upper_bound=True), 'Independent two-endpoint witness selection')
    solver = solver_check(row['solver_audit'], baseline_BCE, final_BCE, baseline_norm, final_norm)
    return dict(image_id=image_id, annotation_id=annotation_id, raw_annotation=annotation,
        GT_digest=row['GT_digest'], raw_GT_area=int(target.sum().item()), crowd_retained=bool(annotation.get('iscrowd', 0)),
        ignore_retained=bool(annotation.get('ignore', False)), empty_GT_retained=not bool(target.any()),
        baseline_query_id=query, native_best100_metrics=metrics[query], final_metrics=final_metrics,
        baseline_full_raster_BCE=baseline_BCE, final_full_raster_BCE=final_BCE,
        baseline_coefficient_norm=baseline_norm, final_coefficient_norm=final_norm,
        baseline_original_logit_comparison=baseline_logit, final_original_logit_comparison=final_logit,
        baseline_binary_comparison=baseline_flips, final_binary_comparison=final_flips,
        final_exported_GPU_logit_positive_RLE_exact=True, raw_GT_and_binary_metrics_independently_recomputed=True,
        best_of_two=row['best_of_two'], solver=solver, audit_packet=row['audit_packet'],
        GT_conditioned_finite_witness_only=True, optimal_IoU_upper_bound_claimed=False,
        optimizer_convergence_or_bitwise_repeatability_claimed=False)


def image_checks(torch, np, masks, coco, old, pre, spec, record, output, started):
    rows, GT_rows = read_ref(record['image_rows']), read_ref(record['GT_rows'])
    predictions, factors = read_ref(record['native_predictions']), read_ref(record['query_score_factors'])
    require(len(rows) == 12 and [row['image_id'] for row in rows] == SUBSET and len(GT_rows) == 18 and
        len(predictions) == len(factors) == 1200 and sorted(row['annotation_id'] for row in GT_rows) == list(range(1, 19)),
        'Complete prelocked12/all18 raw GT/all1200 native predictions differs')
    grouped = {image_id: {} for image_id in SUBSET}
    grouped_GT = {image_id: [] for image_id in SUBSET}
    for row in GT_rows:
        require(row['image_id'] in grouped_GT, 'GT outside prelocked FIT subset')
        require(Path(row['audit_packet']['path']).resolve() == (output / ('AUDIT_GT_%d.pth' % row['annotation_id'])).resolve(),
            'Actual per-GT audit path differs')
        grouped_GT[row['image_id']].append(row)
    for prediction, factor in zip(predictions, factors):
        require(set(prediction) == {'image_id', 'query_id', 'category_id', 'score', 'segmentation'} and
            type(prediction['image_id']) is int and prediction['image_id'] in grouped and
            type(prediction['query_id']) is int and 0 <= prediction['query_id'] < 100 and prediction['category_id'] == 1 and
            prediction['query_id'] not in grouped[prediction['image_id']], 'Native all100 unique query population differs')
        grouped[prediction['image_id']][prediction['query_id']] = prediction
        require(set(factor) == {'image_id', 'query_id', 'class_probability', 'mask_quality', 'official_score'} and
            (factor['image_id'], factor['query_id']) == (prediction['image_id'], prediction['query_id']) and
            factor['official_score'] == prediction['score'], 'Actual native prediction/factor identity differs')
        for key in ('class_probability', 'mask_quality', 'official_score'):
            finite(factor[key], 'Native factor ' + key, True)
            require(factor[key] <= 1 and float(np.float32(factor[key])) == factor[key], 'Native factor is not bounded exact FP32')
        exact = factor['class_probability'] * factor['mask_quality']
        error = (2. ** -24 / (1. - 2. ** -24)) * abs(exact) + 2. ** -126
        require(abs(factor['official_score'] - exact) <= error and
            abs(float(np.float32(factor['class_probability']) * np.float32(factor['mask_quality'])) - exact) <= error,
            'Official native FP32 class-probability/quality score arithmetic differs')
    factor_map = {(row['image_id'], row['query_id']): row for row in factors}
    image_proofs, GT_proofs = [], []
    for position, row in enumerate(rows):
        require(time.monotonic() - started < 1800, 'Independent CPU diagnostic deadline exceeded')
        image_id, identity = row['image_id'], row['identity']
        image = coco.imgs[image_id]
        height, width = int(image['height']), int(image['width'])
        scale = 800. / max(height, width)
        actual = [int(height * scale + .5), int(width * scale + .5)]
        ann_ids = sorted(int(value) for value in coco.getAnnIds(imgIds=[image_id]))
        require(len(ann_ids) == COUNTS[position] and [r['annotation_id'] for r in grouped_GT[image_id]] == ann_ids and
            row['annotation_ids'] == ann_ids and identity['ann_ids'] == ann_ids,
            'Every original annotation ID must be retained without filtering')
        expect(identity, dict(image_id=image_id, dataset_index=position, original_shape=[height, width],
            actual_img_shape=actual, batch_input_shape=[800, 800], scale_factor=[actual[1] / width, actual[0] / height],
            native_input_gamma=1.0, target_gray_resize='nearest', parser_rows=[dict(annotation_id=value,
                native_parser_retained=True) for value in ann_ids]), 'Actual fixed FIT image identity')
        require(Path(identity['image']['path']).resolve() == (Path(spec['image_root']) / image['file_name']).resolve(),
            'Actual source image path differs from original FIT annotation')
        bound(identity['image'])
        expect(row, dict(dataset_index=position, original_shape=[height, width], actual_img_shape=actual,
            head_seed=100000 + image_id, external_RNG_restored=True, forward_capture_counts=dict(feature=1, coefficient=10, head=1),
            GT_removed_from_forward=True, loader_accesses_annotations=True, matching_or_loss_executed=False,
            native100_GPU_mask200_TorchEqual=True, native100_GPU_official_masks_TorchEqual=True,
            native_official_fusion=True, predictions=100, native_parameter_grads_None=True, all_modules_eval=True,
            hooks_removed=True, hook_cache_clear=True, oracle1_vs_native100_bitwise_algebra_claimed=False,
            GPU_optimizer_bitwise_determinism_claimed=False), 'Actual native forward/frozen/hook scope')
        pre.digest_contract(row['input'], [1, 3, 800, 800], 'float32')
        for key in ('RNG_external_before', 'RNG_external_after', 'RNG_private_before', 'RNG_private_after'):
            require(len(row[key]) == 2, 'Actual CPU/CUDA RNG pair missing')
            for value in row[key]:
                require(value['dtype'] == 'uint8' and len(value['shape']) == 1 and
                    type(value['shape'][0]) is int and value['shape'][0] > 0, 'Actual RNG digest dtype/shape differs')
                pre.digest_contract(value, value['shape'], 'uint8')
        require(row['RNG_external_before'] == row['RNG_external_after'], 'Actual external CPU/CUDA RNG did not restore')
        generator = torch.Generator(device='cpu').manual_seed(row['head_seed'])
        require(old.tensor_digest(torch, generator.get_state(), 'uint8') == row['RNG_private_before'][0],
            'Independent private CPU head seed identity differs')
        require(Path(row['audit_packet']['path']).resolve() == (output / ('AUDIT_IMAGE_%d.pth' % image_id)).resolve(),
            'Actual per-image audit path differs')
        packet = torch.load(str(bound(row['audit_packet'])), map_location='cpu')
        require(set(packet) == {'schema', 'image_id', 'identity', 'feature', 'native_coeff', 'native_masks200'},
            'Complete per-image audit field population differs')
        expect(packet, dict(schema='frozen-feature-linear-readout-image-audit-v1', image_id=image_id, identity=identity),
            'Actual per-image packet identity')
        feature, coefficients, native200 = packet['feature'], packet['native_coeff'], packet['native_masks200']
        for name, shape in (('feature', [1, 256, 200, 200]), ('native_coeff', [1, 100, 256]), ('native_masks200', [1, 100, 200, 200])):
            cpu_tensor(torch, packet[name], shape, 'Actual native audit ' + name)
            require(old.tensor_digest(torch, packet[name], 'float32') == row['tensor_digests'][name],
                'Actual CPU audit tensor digest differs: ' + name)
        require(set(row['tensor_digests']) == {'feature', 'native_coeff', 'native_masks200'}, 'Native packet digest roles differ')
        native_projection = logit_check(torch, project(torch, feature, coefficients), native200,
            'Independent native100 FP32 feature/coefficient projection at200')
        require(row['final_fields']['masks'] == row['tensor_digests']['native_masks200'], 'Captured final native mask identity differs')
        class_packet = row['native_class_packet']
        pre.digest_contract({key: class_packet[key] for key in ('shape', 'dtype', 'sha256')}, [100, 2], 'float32')
        classes = torch.tensor(class_packet['values'], dtype=torch.float32, device='cpu')
        cpu_tensor(torch, classes, [100, 2], 'Actual native class packet')
        require(classes.tolist() == class_packet['values'] and old.tensor_digest(torch, classes, 'float32') ==
            {key: class_packet[key] for key in ('shape', 'dtype', 'sha256')} and
            old.tensor_digest(torch, classes[None], 'float32') == row['final_fields']['classes'], 'Actual native class packet digest differs')
        probabilities = torch.softmax(classes, dim=-1)[:, 0]
        require(set(grouped[image_id]) == set(range(100)) and len(row['prediction_query_ids']) == 100 and
            row['prediction_query_ids'] == [p['query_id'] for p in predictions if p['image_id'] == image_id] and
            sorted(row['prediction_query_ids']) == list(range(100)), 'All100 original native query ordinals/order must survive')
        native_original = render(torch, native200, actual, [height, width])[0]
        native_masks = {query: torch.from_numpy(decode_RLE(np, masks, grouped[image_id][query]['segmentation'],
            [height, width], 'Actual native query%d RLE' % query)) for query in range(100)}
        binary_proofs = [binary_check(torch, native_original[query], native_masks[query], 'Native100 CPU raster query%d' % query)
            for query in range(100)]
        for query in range(100):
            require(abs(float(probabilities[query].item()) - factor_map[(image_id, query)]['class_probability']) <= 1e-6,
                'Native class probability independently differs from exported class packet')
        for GT_row in grouped_GT[image_id]:
            GT_proofs.append(GT_check(torch, np, masks, coco, old, GT_row, identity, feature,
                coefficients, native_original, native_masks))
        image_proofs.append(dict(image_id=image_id, annotation_ids=ann_ids, identity=identity, audit_packet=row['audit_packet'],
            native100_feature_coefficient_projection=native_projection,
            native100_original_binary_comparisons=binary_proofs,
            native100_binary_flip_count=sum(value['binary_flip_count'] for value in binary_proofs),
            native100_original_GPU_logits_exported=False, native100_original_logit_allclose_measured=False,
            actual_GPU_native100_mask200_and_official_masks_TorchEqual_source_observed=True,
            all100_native_query_ordinals_and_empty_predictions_retained=True,
            class_packet_and_FP32_score_arithmetic_checked=True, omitted_mask_quality_CPU_recomputed=False))
        del packet, feature, coefficients, native200, native_original, native_masks, classes
        gc.collect()
    return image_proofs, GT_proofs


def run(spec_file, acceptance_file):
    started = time.monotonic()
    spec_path, target = Path(spec_file), Path(acceptance_file)
    require(spec_path.is_absolute() and target.is_absolute() and target.parent.is_dir() and not target.exists(),
        'Absolute SPEC and fresh absolute acceptance with existing parent required')
    spec_path = spec_path.resolve(strict=True)
    spec = read(spec_path)  # Inert execution metadata only before raw gates.
    expect(spec, dict(schema=SPEC_SCHEMA, subset_image_ids=SUBSET, subset_images=12, subset_raw_GT=18,
        solver=SOLVER, diagnostic_mode='native0', expected_epoch=24, CPU_affinity=[0, 1], deadline_seconds=1800,
        allocator_limit_MiB=4096, whole_GPU_owner_limit_MiB=8192, no_feature_cache=False, audit_packet_only=True,
        GPU_optimizer_bitwise_determinism_claimed=False), 'Exact frozen native FIT diagnostic SPEC')
    require(isinstance(spec['phase'], str) and spec['phase'].startswith('frozen_feature_linear_readout_FIT_') and
        Path(spec['phase']).name == spec['phase'], 'Safe fixed diagnostic owner phase required')
    receipt_path = spec_path.parent / 'finite_point_cache_execution_20261005_v2' / 'runs' / spec['phase'] / 'PARENT_RECEIPT.json'
    raw = read(receipt_path)
    terminal(raw, spec['phase'])
    receipts = {}
    for role, phase_role in (('teacher_receipt', 'teacher_phase'), ('preflight_receipt', 'preflight_phase'),
                            ('short2_receipt', 'short2_phase'), ('prior_DEV_receipt', 'prior_DEV_phase')):
        require(Path(spec[role]['path']).is_absolute(), 'Absolute original raw receipt required')
        receipts[role] = read(spec[role]['path'])
        terminal(receipts[role], spec[phase_role])
    # Current RAW first, then four original RAW held0/reap/birth-absence/noerror
    # gates. ALL hashes, Q/CP/GT/audit accesses and scientific imports follow.
    execution, contract = source_gate(spec_path, spec, raw)
    fixture_proof = fixture_gate(spec)
    prior = load_source(contract['base_role_SOURCE_refs']['CPU_reader'], 'sealed_actual_prior_DEV_CPU_helpers')
    short = load_source(spec['short2_CPU_reader'], 'sealed_actual_SHORT2_CPU_helpers')
    pre = load_source(spec['preflight_CPU_reader'], 'sealed_actual_preflight_CPU_helpers')
    for role in pre.ARTIFACT_PINS:
        pre.pinned_role(spec, role)
    for role, pin in pre.SOURCE_PINS.items():
        require((spec[role]['bytes'], spec[role]['sha256']) == pin and spec[role] in spec['sources'],
            'Canonical reused source pin differs: ' + role)
    for role, path in pre.SOURCE_PATHS.items():
        require(Path(spec[role]['path']).resolve() == path.resolve(), 'Canonical reused source path differs: ' + role)
    for role, pin in short.PREFLIGHT_PINS.items():
        require(spec[role] == pin, 'Canonical actual preflight lineage differs: ' + role)
    short.preflight_gate(pre, spec, receipts['preflight_receipt'])
    training, teacher_q = pre.teacher_lineage(spec, receipts['teacher_receipt'])
    pre.fixture_gate(spec)
    prior.qualified_short2(spec, receipts['short2_receipt'], short, pre)
    prior_DEV_gate(spec, receipts['prior_DEV_receipt'], prior, contract)
    require(Path(spec['image_root']).resolve() == Path('E:/PSeg_SSDD/voc_style/JPEGImages_train').resolve(),
        'Canonical original FIT image root differs')
    output = Path(spec['output_directory'])
    require(output.is_absolute(), 'Absolute actual diagnostic output directory required')
    report_ref, start_ref = reference(output / 'REPORT.json'), reference(output / 'START.json')
    aggregate_ref = reference(output / 'QUALIFIED.json')
    report, start, aggregate = read_ref(report_ref), read_ref(start_ref), read_ref(aggregate_ref)
    expect(report, dict(schema=REPORT_SCHEMA, status=REPORT_STATUS, phase=spec['phase'], mode='native0',
        source=execution['source'], spec=execution['spec'], error=None, SOURCE_contract=spec['SOURCE_contract'],
        teacher_checkpoint=spec['teacher_checkpoint'], teacher_qualification=spec['teacher_qualification'],
        prior_DEV_spec=spec['prior_DEV_spec'], prior_DEV_CPU_qualification=spec['prior_DEV_CPU_qualification'],
        prior_DEV_receipt=spec['prior_DEV_receipt'], diagnostic_CPU_fixture=spec['diagnostic_CPU_fixture'],
        diagnostic_CPU_fixture_spec=spec['diagnostic_CPU_fixture_spec'], diagnostic_CPU_fixture_source=spec['diagnostic_CPU_fixture_source'],
        subset_image_ids=SUBSET, subset_images=12, subset_raw_GT=18,
        solver=SOLVER, full610_CPU_before_CUDA=True, full_native610_and_BN_exact=True, native451_frozen=True,
        GT_removed_from_forward=True, native100_GPU_mask200_and_official_masks_TorchEqual=True,
        retained_candidates=['native_best100', 'final'], full_raster_BCE_unweighted_mean=True, no_bias=True,
        no_MLP=True, no_GT_box_crop=True, no_sampled_points=True, no_regularizer=True, no_threshold_tuning=True,
        optimizer_or_RNG_restored=False, no_feature_cache=False, audit_packet_only=True, no_AP=True,
        AP_computed=False, no_TEST=True, efficacy_claimed=False, convergence_claimed=False,
        optimal_IoU_upper_bound_claimed=False, generator_versus_budget_causal_identification_claimed=False,
        GPU_optimizer_bitwise_determinism_claimed=False, oracle1_vs_native100_bitwise_algebra_claimed=False,
        max_eval_is_soft_limit=True, allocator_limit_MiB=4096,
        GPU_scope='whole GPU must be verified from actual owner receipt'), 'Actual completed bounded diagnostic report')
    expect(start, dict(schema='frozen-feature-linear-readout-FIT-start-v1', phase=spec['phase'], mode='native0',
        source=execution['source'], spec=execution['spec'], SOURCE_contract=spec['SOURCE_contract'],
        subset_image_ids=SUBSET, subset_images=12, subset_raw_GT=18, solver=SOLVER,
        teacher_checkpoint=spec['teacher_checkpoint'], teacher_qualification=spec['teacher_qualification'],
        prior_DEV_spec=spec['prior_DEV_spec'], prior_DEV_CPU_qualification=spec['prior_DEV_CPU_qualification'],
        prior_DEV_receipt=spec['prior_DEV_receipt'], diagnostic_CPU_fixture=spec['diagnostic_CPU_fixture'],
        diagnostic_CPU_fixture_spec=spec['diagnostic_CPU_fixture_spec'], diagnostic_CPU_fixture_source=spec['diagnostic_CPU_fixture_source'],
        original_four_raw_terminal_gates_before_scientific_access=True,
        full610_CPU_before_CUDA_required=True, no_AP=True, no_TEST=True, no_retry=True, no_bias=True,
        no_regularizer=True, full_raster_supervision=True, optimizer_or_RNG_restored=False,
        no_feature_cache=False, audit_packet_only=True, GPU_optimizer_bitwise_determinism_claimed=False),
        'Actual prelocked START/provenance/scope joins')
    record = report['result']
    expect(aggregate, dict(schema='frozen-feature-linear-readout-FIT-aggregate-v1',
        status='ROOT_FROZEN_FEATURE_LINEAR_READOUT_FIT_PENDING_CPU_READER', phase=spec['phase'],
        source=execution['source'], spec=execution['spec'], report=report_ref, result=record,
        independently_CPU_qualified=False, AP_computed=False, optimal_IoU_upper_bound_claimed=False,
        efficacy_claimed=False), 'Actual pending aggregate/report/result join')
    expect(record, dict(images=12, raw_GT=18, native_predictions_count=1200, fitted_coefficients=18,
        fullstate_tensors=610, native_parameters=451, full_native_state_exact=True,
        frozen_BN_all_exact=True, native_parameter_grads_None=True), 'Actual complete diagnostic result')
    for role, name in (('image_rows', 'IMAGE_ROWS.json'), ('GT_rows', 'GT_ROWS.json'),
                       ('native_predictions', 'NATIVE_PREDICTIONS.json'), ('query_score_factors', 'QUERY_SCORE_FACTORS.json'),
                       ('state_proof', 'STATE_PROOF.json')):
        require(Path(record[role]['path']).resolve() == (output / name).resolve(), 'Actual diagnostic artifact path differs: ' + role)
    for value in (report['torch_peak_allocated_bytes'], record['peak_allocated_bytes']):
        require(type(value) is int and 0 <= value <= 4096 * 1024 ** 2, 'Actual allocator4096 bound differs')
    finite(report['elapsed_seconds'], 'Actual GPU elapsed seconds', True)
    cleanup = report['cleanup']
    require(report['elapsed_seconds'] < 1800 and type(cleanup['allocated_before']) is int and
        cleanup['allocated_before'] >= 0 and type(cleanup['allocated_after']) is int and cleanup['allocated_after'] == 0,
        'Actual diagnostic deadline/cleanup differs')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '2'
    os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
    import psutil
    psutil.Process().cpu_affinity([0, 1])
    import numpy as np
    import torch
    from pycocotools import mask as masks
    from pycocotools.coco import COCO
    from mmengine import Config
    from mmengine.model import revert_sync_batchnorm
    from mmengine.registry import init_default_scope
    from mmdet.registry import MODELS
    from mmdet.utils import register_all_modules
    from mmdet.datasets import CocoDataset
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    require(psutil.Process().cpu_affinity() == [0, 1] and not torch.cuda.is_initialized() and
        os.environ['CUDA_VISIBLE_DEVICES'] == '', 'Actual CPU01/CUDAempty/noCUDAinitialization required')
    register_all_modules(init_default_scope=False)
    init_default_scope('mmdet')
    require(len(CocoDataset.METAINFO['classes']) == 80 and CocoDataset.METAINFO['classes'][8] == 'boat',
        'Original official COCO80 boat order differs')
    old = load_source(spec['reused_CPU_reader'], 'sealed_original_native_CPU_state_helpers')
    teacher_reader = load_source(spec['training_CPU_reader'], 'sealed_original_E24_CPU_helpers')
    initializer = load_source(spec['initializer_source'], 'sealed_original_COCO_ship_initializer')
    rng_before = torch.get_rng_state().clone()
    model, teacher = pre.load_teacher(torch, old, teacher_reader, initializer, MODELS,
        revert_sync_batchnorm, Config, spec, training, teacher_q)
    model.requires_grad_(False).eval()
    library_paths(torch, model, spec)
    state_proof = full_state_check(torch, old, pre, model, teacher, spec, record)
    coco = COCO(str(bound(spec['FIT_annotation'])))
    fit = read_ref(spec['FIT_annotation'])
    require(len(coco.imgs) == len(fit['images']) == 741 and len(coco.anns) == len(fit['annotations']) == 1661 and
        set(coco.cats) == {1} and [int(row['id']) for row in fit['images'][:12]] == SUBSET and
        [len(coco.getAnnIds(imgIds=[image_id])) for image_id in SUBSET] == COUNTS and
        sorted(value for image_id in SUBSET for value in coco.getAnnIds(imgIds=[image_id])) == list(range(1, 19)),
        'Independent pinned original741FIT/1661GT/prelocked12/all18 annotation population differs')
    image_proofs, GT_proofs = image_checks(torch, np, masks, coco, old, pre, spec, record, output, started)
    require(torch.equal(rng_before, torch.get_rng_state()) and not torch.cuda.is_initialized() and
        torch.get_num_threads() == torch.get_num_interop_threads() == 2 and psutil.Process().cpu_affinity() == [0, 1] and
        os.environ['CUDA_VISIBLE_DEVICES'] == '' and time.monotonic() - started < 1800,
        'CPU reader changed external RNG/resources, initialized CUDA or exceeded deadline')
    acceptance = dict(schema=QUAL_SCHEMA, status=QUAL_STATUS, phase=spec['phase'], mode='native0',
        source=execution['source'], reader_source=execution['reader'], spec=execution['spec'],
        report=report_ref, start=start_ref, GPU_qualification=aggregate_ref, parent_receipt=reference(receipt_path),
        GPU_admission=execution['GPU_admission'], owner_driver=execution['owner_driver'], SOURCE_contract=spec['SOURCE_contract'],
        teacher_checkpoint=spec['teacher_checkpoint'], teacher_qualification=spec['teacher_qualification'],
        teacher_training_spec=spec['teacher_training_spec'], teacher_receipt=spec['teacher_receipt'],
        preflight_CPU_qualification=spec['preflight_CPU_qualification'], preflight_receipt=spec['preflight_receipt'],
        short2_CPU_qualification=spec['short2_CPU_qualification'], short2_receipt=spec['short2_receipt'],
        prior_DEV_spec=spec['prior_DEV_spec'], prior_DEV_CPU_qualification=spec['prior_DEV_CPU_qualification'],
        prior_DEV_receipt=spec['prior_DEV_receipt'], FIT_annotation=spec['FIT_annotation'], fixture=fixture_proof,
        subset_image_ids=SUBSET, subset_images=12, subset_raw_GT=18, raw_annotation_ids=list(range(1, 19)),
        state_proof=state_proof, image_proofs=image_proofs, GT_proofs=GT_proofs,
        raw_current_and_four_original_terminal_gates_before_scientific_access=True,
        complete_inherited69_and_new_SOURCE_closure_owner_bound_before_imports=True,
        prior_actual_DEV_CPU_qualification_and_original_admission_joined=True,
        exact_full_teacher610_and_typed_BN106_affine159_buffers_independently_CPU_checked=True,
        all_raw_selected_GT_including_empty_crowd_ignore_retained=True,
        all_native100_and_final_original_RLE_independently_decoded=True,
        actual_native_GT_best100_lowest_query_ordinal_ties_independently_checked=True,
        final_actual_GPU_logit_positive_RLE_exact=True,
        native_feature_coefficient_projection_and_final_original_logits_CPU_tolerance_checked=True,
        full_original_raster_BCE_coefficient_norm_and_binary_metrics_independently_recomputed=True,
        only_native_best100_and_final_two_candidate_witness_checked=True,
        native100_actual_GPU_replay_TorchEqual_source_observed=True,
        native100_original_GPU_logits_all_exported=False, native100_original_logit_allclose_measured=False,
        CPU_native_binary_replay_bit_exact_claimed=False,
        native100_binary_flip_count=sum(row['native100_binary_flip_count'] for row in image_proofs),
        final_binary_flip_count=sum(row['final_binary_comparison']['binary_flip_count'] for row in GT_proofs),
        selected_baseline_binary_flip_count=sum(row['baseline_binary_comparison']['binary_flip_count'] for row in GT_proofs),
        solver=SOLVER, max_eval_is_soft_limit=True, closure_counters_source_observed_and_consistent=True,
        CPU_optimizer_or_GPU_forward_replayed=False, GPU_optimizer_bitwise_determinism_claimed=False,
        oracle1_vs_native100_bitwise_algebra_claimed=False, CPU_affinity=[0, 1], torch_threads=2,
        torch_interop_threads=2, CUDA_initialized=False, external_CPU_RNG_restored=True,
        AP_computed=False, no_TEST=True, efficacy_claimed=False, convergence_claimed=False,
        optimal_IoU_upper_bound_claimed=False, generator_versus_budget_causal_identification_claimed=False,
        scientific_method_verdict=None, evidence_limits=[
            'Only the prelocked first12 FIT images/all18 original annotations; this feasibility sample is not representative.',
            'GT-best100 and one free256 coefficient per GT are diagnostic witnesses using GT, not deployable selectors or trained generators.',
            'Actual raw0 terminal gates and pinned SOURCE prove the runtime scope; frozen GPU forward, capture and gradient-none facts are source-observed, while full610 CPU state digests are checked independently.',
            'CPU independently projects FP32 F/e0, renders bilinear800/crop/original, decodes rawGT and RLE, and recomputes binary metrics/full-raster BCE/norms with atol=rtol=1e-4 for floating checks.',
            'Native100 GPU original logits are not all exported: all100 CPU binary replay is a counted near-zero-envelope consistency check, not measured original-logit allclose or bit-exact CPU replay. Final and selected-baseline GPU original logits permit direct tolerance comparisons.',
            'Only actual exported GPU logit>0 final masks define final metrics. Any CPU/GPU binary flips must lie within the stated near-zero envelope and are reported.',
            'Native100 and single-query FP32 GEMM grouping may differ. CUDA bilinear backward may be nondeterministic; fixed full-raster/no RNG is not a bitwise optimization repeatability certificate.',
            'LBFGS max_eval80 is soft after strong-Wolfe line search; actual closure/function counts are joined and n_iter<=40, without a hard80 evaluation ceiling or CPU solver replay.',
            'Finite native-best100/final endpoints provide no convergence, optimal-IoU upper bound, AP, blind TEST, efficacy or generator-versus-budget causal verdict; an unsuccessful fit leaves feasibility unresolved.',
            'CPU tensor packets are bounded audit artifacts, not an admitted feature-cache training recipe.'])
    with target.open('x', encoding='utf-8') as stream:
        json.dump(acceptance, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=QUAL_STATUS, acceptance=reference(target)), sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--acceptance', required=True)
    parser.add_argument('--owner-execute', action='store_true')
    args = parser.parse_args()
    require(args.owner_execute, 'Inert SOURCE requires explicit ROOT CPU execution after five actual raw terminal gates')
    run(args.spec, args.acceptance)


if __name__ == '__main__':
    main()
