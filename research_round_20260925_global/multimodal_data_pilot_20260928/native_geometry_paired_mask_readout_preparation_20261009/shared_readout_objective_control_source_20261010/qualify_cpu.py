"""Independent CPU qualification of the prelocked three-arm objective control.

Authoring this SOURCE is inert. ROOT executes it only after current and five
prior actual raw terminal gates. Independent floating replay is restricted to
the four prelocked saved audit steps; all other training facts are inspected
from the pinned source-observed trace. Exported Hungarian matrices permit a
minimum-cost check, not an independent derivation of native sampled costs.
Readout-heldout FIT was used to train the teacher and is not blind evaluation.
"""
import argparse
import gc
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import time
import types

SPEC_SCHEMA = 'ROOT-shared-readout-objective-control-FIT128-spec-v1'
REPORT_SCHEMA = 'shared-readout-objective-control-FIT128-report-v1'
REPORT_STATUS = 'SHARED_READOUT_OBJECTIVE_CONTROL_TRAIN_HOLDOUT_GPU_COMPLETE_NO_SOTA'
QUAL_SCHEMA = 'ROOT-shared-readout-objective-control-FIT128-CPU-qualification-v1'
QUAL_STATUS = 'ROOT_SHARED_READOUT_OBJECTIVE_CONTROL_FIT128_CPU_QUALIFIED'
ARMS = ['raw_uniform_BCE', 'raw_full_BCE', 'positive_global_gain']
ENDPOINTS = ['native0'] + ARMS
DETAILS = [0, 1, 510, 511]
RAW_NAMES = ['hidden.weight', 'final.weight']
OWNER_SHA = 'ef9a9f17d4fb1bc3464936095ef799021dd880f8b15488d558a871c08b6e1924'
RTOL, ATOL = 1e-4, 1e-4


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def expect(row, expected, label):
    require(isinstance(row, dict), label + ' must be an object')
    for key, value in expected.items():
        actual = row[key]
        require(actual == value and (type(value) not in (int, bool) or type(actual) is type(value)),
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


def close(actual, expected, label, rtol=RTOL, atol=ATOL):
    finite(actual, label)
    finite(expected, label + ' independent value')
    require(math.isclose(actual, expected, rel_tol=rtol, abs_tol=atol), label + ' independently differs')


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
    finite(receipt.get('whole_GPU_observed_peak_MiB'), 'Actual whole-GPU peak', True)
    require(receipt['whole_GPU_observed_peak_MiB'] <= 8192 and receipt.get('whole_GPU_used_limit_MiB') == 8192 and
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
    source, reader = reference(Path(__file__).with_name('train.py')), reference(__file__)
    spec_ref = reference(spec_path)
    admission_ref = reference(spec['GPU_admission_file'])
    admission = read_ref(admission_ref)
    owner = reference(spec_path.parent / 'finite_point_cache_execution_20261005_v2' / 'run_owned_gpu_limited.py')
    expect(admission, dict(status='ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE', automatic_retry=False,
        phase=spec['phase'], allowed_cpus=[0, 1], cuda_visible_devices='0', qualified_leaf_deadline_seconds=5400,
        whole_GPU_used_limit_MiB=8192, driver_sha256=OWNER_SHA,
        entry_args=['--spec', str(spec_path), '--owner-execute']), 'Current owned objective-control admission')
    require(raw['admission_sha256'] == admission_ref['sha256'] and owner['sha256'] == OWNER_SHA and
        Path(admission['entry']).resolve() == Path(source['path']) and spec['entry_source'] == source and
        spec['CPU_reader'] == reader and source in spec['sources'] and reader in spec['sources'] and
        len({row['path'] for row in spec['sources']}) == len(spec['sources']), 'Current owner/source/SPEC bindings differ')
    hashes = admission['bound_source_sha256']
    require(hashes.get(str(spec_path)) == spec_ref['sha256'], 'Current SPEC is not owner-bound')
    for row in spec['sources']:
        bound(row)
        require(hashes.get(row['path']) == row['sha256'], 'Complete inherited/new SOURCE closure not owner-bound')
    contract = read_ref(spec['SOURCE_contract'])
    require(Path(spec['SOURCE_contract']['path']).resolve() == Path(__file__).with_name('SOURCE_CONTRACT.json').resolve() and
        spec['SOURCE_contract'] in spec['sources'], 'Canonical inert SOURCE contract absent from owner closure')
    expect(contract, dict(kind='INERT_SOURCE_AUTHORING_CONTRACT_NOT_SCIENTIFIC_ADMISSION', schema=SPEC_SCHEMA,
        entry_name='train.py', reader_name='qualify_cpu.py', report_schema=REPORT_SCHEMA, report_status=REPORT_STATUS,
        qualification_schema=QUAL_SCHEMA, qualification_status=QUAL_STATUS, arms=ARMS,
        epochs=4, microsteps=512, optimizer_updates=256, raw_GT_visits_per_arm=1168,
        accumulation=2, detail_microsteps=DETAILS, sampled_pixels_per_GT=12544, sampling_with_replacement=True,
        hidden_seed=2026100911, permutation_seed=2026101013, order_seed=2026101021, head_seed_base=100000,
        matching_seed_base=2800000, integer_pixel_seed_base=3800000,
        raw_optimizer=dict(name='AdamW', lr=.0001, weight_decay=.05, betas=[.9, .999], eps=1e-8, clip_norm=.01),
        gain_optimizer=dict(name='AdamW', lr=.01, weight_decay=.05, betas=[.9, .999], eps=1e-8, clip_norm=.01,
            capacity_or_optimizer_matched=False)),
        'Canonical prelocked SOURCE contract')
    expect(contract['split'], dict(split_seed=2026101019, train_images=128, readout_heldout_images=64,
        train_raw_GT=292, readout_heldout_raw_GT=106, teacher_was_trained_on_all_selected_FIT=True,
        blind_holdout_claimed=False), 'Canonical prelocked readout split scope')
    for role, row in contract['base_role_SOURCE_refs'].items():
        require(row in spec['sources'], 'Original admitted SOURCE absent: ' + role)
        if role not in ('entry_source', 'CPU_reader'):
            require(spec[role] == row, 'Inherited canonical SOURCE role differs: ' + role)
    for role, row in contract['library_SOURCE_refs'].items():
        require(spec[role] == row and row in spec['sources'], 'Canonical library SOURCE differs: ' + role)
    for role in ('prior_diag_spec', 'prior_diag_CPU_qualification', 'prior_diag_receipt', 'prior_diag_phase',
                 'prior_diag_entry_source', 'prior_diag_CPU_reader_source', 'prior_diag_contract'):
        require(spec[role] == contract[role], 'Canonical actual prior diagnostic lineage differs: ' + role)
    require(spec['FIT_annotation'] == contract['split']['annotation_reference'], 'Canonical FIT reference differs')
    if 'scientific_spec' in admission:
        require(admission['scientific_spec'] == spec_ref, 'Admitted current SPEC differs')
    return dict(source=source, reader=reader, spec=spec_ref, GPU_admission=admission_ref, owner_driver=owner), contract


def prior_diagnostic_gate(spec, raw, diag, contract):
    q = read_ref(spec['prior_diag_CPU_qualification'])
    expect(q, dict(schema=diag.QUAL_SCHEMA, status=diag.QUAL_STATUS, phase=spec['prior_diag_phase'], mode='native0',
        source=spec['prior_diag_entry_source'], reader_source=spec['prior_diag_CPU_reader_source'],
        spec=spec['prior_diag_spec'], parent_receipt=spec['prior_diag_receipt'],
        raw_current_and_four_original_terminal_gates_before_scientific_access=True,
        exact_full_teacher610_and_typed_BN106_affine159_buffers_independently_CPU_checked=True,
        all_raw_selected_GT_including_empty_crowd_ignore_retained=True, AP_computed=False,
        efficacy_claimed=False, convergence_claimed=False, optimal_IoU_upper_bound_claimed=False),
        'Actual independently completed prior diagnostic qualification')
    original = read_ref(q['spec'])
    admission = read_ref(q['GPU_admission'])
    require(len(original['sources']) == 83 and original['SOURCE_contract'] == spec['prior_diag_contract'] and
        original['entry_source'] == spec['prior_diag_entry_source'] and original['CPU_reader'] == spec['prior_diag_CPU_reader_source'] and
        original['phase'] == spec['prior_diag_phase'] and raw['admission_sha256'] == q['GPU_admission']['sha256'] and
        Path(original['GPU_admission_file']).resolve() == Path(q['GPU_admission']['path']).resolve() and
        admission['bound_source_sha256'].get(q['spec']['path']) == q['spec']['sha256'], 'Original actual83 closure/admission joins differ')
    expect(admission, dict(status='ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE', automatic_retry=False,
        phase=spec['prior_diag_phase'], allowed_cpus=[0, 1], cuda_visible_devices='0', qualified_leaf_deadline_seconds=1800,
        whole_GPU_used_limit_MiB=8192, driver_sha256=OWNER_SHA), 'Original diagnostic owned admission')
    require(Path(admission['entry']).resolve() == Path(spec['prior_diag_entry_source']['path']).resolve() and
        q['owner_driver']['sha256'] == OWNER_SHA, 'Original diagnostic entry/owner differ')
    for row in original['sources']:
        bound(row)
        require(row in spec['sources'] and admission['bound_source_sha256'].get(row['path']) == row['sha256'],
            'Original actual83 SOURCE closure not completely retained and owner-bound')
    for role in ('teacher_checkpoint', 'teacher_qualification', 'teacher_training_spec', 'teacher_receipt', 'teacher_phase',
                 'preflight_CPU_qualification', 'preflight_training_spec', 'preflight_receipt', 'preflight_phase',
                 'short2_CPU_qualification', 'short2_training_spec', 'short2_receipt', 'short2_phase',
                 'prior_DEV_spec', 'prior_DEV_CPU_qualification', 'prior_DEV_receipt', 'prior_DEV_phase',
                 'checkpointmap', 'qualificationmap', 'FIT_annotation', 'DEV_annotation', 'image_root'):
        require(original[role] == spec[role], 'Prior diagnostic/current ancestry differs: ' + role)
    for role in ('source', 'reader_source', 'spec', 'report', 'start', 'GPU_qualification', 'parent_receipt',
                 'GPU_admission', 'owner_driver', 'SOURCE_contract'):
        bound(q[role])
    require(q['SOURCE_contract'] == spec['prior_diag_contract'], 'Actual prior diagnostic contract differs')
    return q, read_ref(spec['prior_diag_contract'])


def fixture_gate(spec, contract):
    roles = ['control_CPU_fixture', 'control_CPU_fixture_spec', 'control_CPU_fixture_source']
    require(all(spec[role] in spec['sources'] for role in roles), 'Actual control fixture refs absent from owner closure')
    manifest, q = read_ref(spec[roles[1]]), read_ref(spec[roles[0]])
    sources = dict(producer=spec['entry_source'], fixture=spec[roles[2]], contract=spec['SOURCE_contract'])
    execution = dict(device='cpu', cpu_affinity=[0, 1], torch_threads=2, torch_interop_threads=2, cuda_visible_devices='')
    expect(manifest, dict(schema=contract['fixture_schema'], inputs='SOURCE_ONLY', sources=sources, execution=execution),
        'Actual owner-bound control fixture manifest')
    require(set(manifest) == {'schema', 'inputs', 'sources', 'execution'}, 'Fixture manifest field population differs')
    expect(q, dict(status=contract['fixture_status'], sources=sources, execution=execution,
        spec_path=spec[roles[1]]['path'], synthetic_execution_performed=True, torch_version='2.0.1+cu118',
        AP_computed=False, efficacy_claimed=False), 'Actual executed synthetic control fixture')
    require(q['helper_SOURCE_refs'] == contract['helper_SOURCE_refs'] and
        q['library_SOURCE_refs'] == contract['library_SOURCE_refs'], 'Actual fixture helper/library SOURCE closure differs')
    expect(q, dict(independent_scalar_BCE_and_coefficient_gradients=True, duplicate_12544_indices_preserved=True,
        same_expected_full_BCE_enumeration=True, per_GT_then_image_then_accum2_normalization=True,
        private_CPU_sampling_RNG_restored=True, connected_empty_GT=True, fixed_assignment_raw_ann_mapping_and_rejections=True,
        connected_empty_gain=True,
        raw_cold_warm_two_weight_gradients=True, frozen_F_e0_unchanged_and_no_grad=True,
        gain_after_original_renderer_gradient=True, positive_gain_binary_examples=True,
        quality_score_ranking_change=True, empty_prediction_quality_zero=True, external_CPU_RNG_unchanged=True,
        CUDA_initialized=False, real_data_feasibility_qualified=False, native_model_integration_qualified=False,
        Hungarian_optimality_qualified=False, optimizer_trajectory_qualified=False,
        convergence_claimed=False, novelty_claimed=False, real_FP32_binary_invariance_claimed=False),
        'Actual independent SOURCE-only fixture proof and limits')
    return dict(qualification=spec[roles[0]], manifest=spec[roles[1]], source=spec[roles[2]],
        synthetic_execution_performed=True, actual_ROOT_tool_exit_is_external_owner_evidence=True)


def cpu_tensor(torch, value, shape, label, dtype=None):
    require(isinstance(value, torch.Tensor) and value.device.type == 'cpu' and
        value.dtype == (torch.float32 if dtype is None else dtype) and list(value.shape) == list(shape) and
        not value.requires_grad and bool(torch.isfinite(value).all()), label + ' exact finite detached CPU tensor required')


def tensor_close(torch, actual, expected, label, rtol=RTOL, atol=ATOL):
    require(actual.shape == expected.shape and actual.dtype == expected.dtype == torch.float32 and
        bool(torch.isfinite(actual).all()) and bool(torch.isfinite(expected).all()) and
        torch.allclose(actual, expected, rtol=rtol, atol=atol), label + ' exceeds independent FP32 tolerance')
    error = (actual - expected).abs()
    return dict(max_absolute_error=float(error.max().item()) if error.numel() else 0., rtol=rtol, atol=atol,
        bitwise_equality_claimed=False)


def tensor_record(value):
    array = value.detach().cpu().contiguous().numpy()
    return dict(shape=list(array.shape), dtype=str(array.dtype), sha256=hashlib.sha256(array.tobytes(order='C')).hexdigest())


def digest_row(row, shape, dtype='float32'):
    require(isinstance(row, dict) and set(row) == {'shape', 'dtype', 'sha256'} and row['shape'] == list(shape) and
        row['dtype'] == dtype and isinstance(row['sha256'], str) and len(row['sha256']) == 64 and
        set(row['sha256']) <= set('0123456789abcdef'), 'Exact SOURCE-observed tensor digest population differs')


def independent_BCE(torch, logits, targets, indices=None):
    require(logits.shape == targets.shape and logits.ndim == 3, 'Per-GT original raster shape differs')
    if logits.shape[0] == 0:
        return logits.sum() * 0., logits.new_empty((0,))
    value, target = logits.flatten(1), targets.flatten(1)
    if indices is not None:
        require(indices.shape == (logits.shape[0], 12544) and indices.dtype == torch.long,
            'Uniform sampled12544 integer indices per GT required')
        value, target = value.gather(1, indices), target.gather(1, indices)
    # Stable scalar BCE formula, independently expressed from the producer API.
    losses = (torch.logaddexp(torch.zeros_like(value), value) - value * target).mean(1)
    return losses.mean(), losses


def raw_forward(torch, feature, h, eligible, native200, weights, query_ids):
    require(h.shape == (1, 100, 16, 512) and eligible.shape == (1, 100, 16) and eligible.dtype == torch.bool,
        'SOURCE-observed RAW context/eligible packet axes differ')
    hidden = torch.nn.functional.silu(torch.nn.functional.linear(h, weights['hidden.weight']))
    values = torch.nn.functional.linear(hidden, weights['final.weight'])
    counts = eligible.sum(-1).clamp_min(1).to(torch.float32)
    coefficient = (values * (eligible.to(torch.float32) / counts[..., None])[..., None]).sum(2)
    # E-empty rows retain the connected zero residual; supplied M200 is anchor.
    residual = torch.einsum('bqc,bchw->bqhw', coefficient[:, query_ids], feature)
    return native200[:, query_ids] + residual


def sampled_indices(torch, GT_count, height, width, microstep):
    generator = torch.Generator(device='cpu')
    generator.manual_seed(3800000 + microstep)
    return torch.randint(height * width, (GT_count, 12544), generator=generator, device='cpu', dtype=torch.long)


def identity_check(torch, np, cv2, coco, dataset, spec, target, identity):
    image_id = identity['image_id']
    expect(identity, dict(ann_ids=target['ann_ids'], original_shape=target['shape'], dataset_index=target['dataset_index'],
        batch_input_shape=[800, 800], native_input_gamma=1.0, gray_decode='cv2.IMREAD_GRAYSCALE', target_gray_resize='nearest'),
        'Canonical original grayscale/native loader identity')
    if 'identity' in target:
        require(identity == target['identity'], 'Repeated image identity changed across epoch or endpoint')
        return
    image = coco.imgs[image_id]
    require(Path(identity['image']['path']).resolve() == (Path(spec['image_root']) / image['file_name']).resolve(),
        'Canonical original FIT image path differs')
    gray = cv2.imread(str(bound(identity['image'])), cv2.IMREAD_GRAYSCALE)
    require(gray is not None and gray.dtype == np.uint8 and list(gray.shape) == target['shape'], 'Independent original grayscale decode differs')
    h, w = identity['actual_img_shape']
    require(type(h) is int and type(w) is int and 0 < h <= 800 and 0 < w <= 800 and
        len(identity['scale_factor']) == 2 and all(type(value) is float and math.isfinite(value) and value > 0 for value in identity['scale_factor']),
        'Native keep-ratio actual geometry differs')
    close(identity['scale_factor'][0], w / target['shape'][1], 'Native horizontal resize scale', 1e-7, 1e-7)
    close(identity['scale_factor'][1], h / target['shape'][0], 'Native vertical resize scale', 1e-7, 1e-7)
    gray800 = np.zeros((800, 800), dtype=np.uint8)
    gray800[:h, :w] = cv2.resize(gray, (w, h), interpolation=cv2.INTER_NEAREST)
    require(hashlib.sha256(gray800.tobytes()).hexdigest() == identity['gray800_sha256'], 'Independent padded original grayscale descriptor differs')
    raw_image = dict(image, img_id=image_id)
    parser_rows = []
    for ann_id in target['ann_ids']:
        parsed = dataset.parse_data_info(dict(raw_img_info=raw_image, raw_ann_info=[coco.anns[ann_id]]))['instances']
        require(len(parsed) in (0, 1), 'Actual native parser annotation multiplicity differs')
        parser_rows.append(dict(annotation_id=ann_id, native_parser_retained=bool(parsed)))
    require(identity['parser_rows'] == parser_rows and all(row['native_parser_retained'] for row in parser_rows),
        'All original raw annotation identities must survive the actual native parser')
    padded = np.zeros((len(target['ann_ids']), 800, 800), dtype=np.int64)
    for index, mask in enumerate(target['targets'].numpy()):
        padded[index, :h, :w] = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    target['native_GT_digest'] = tensor_record(torch.from_numpy(padded))
    target['identity'] = identity


def imported_libraries(torch, model, spec):
    head = model.panoptic_head
    actual = dict(native_parent_head_SOURCE=type(head).__mro__[1],
        panoptic_GT_SOURCE=head.preprocess_gt.__func__.__globals__['preprocess_panoptic_gt'],
        hungarian_SOURCE=type(head.assigner), mask_sampler_SOURCE=type(head.sampler),
        mask_sampling_result_SOURCE=head.sampler.sample.__func__.__globals__['MaskSamplingResult'],
        point_sample_SOURCE=head._get_targets_single.__func__.__globals__['point_sample'],
        adamw_SOURCE=torch.optim.AdamW, clip_grad_SOURCE=torch.nn.utils.clip_grad_norm_)
    for role, obj in actual.items():
        require(Path(inspect.getsourcefile(obj)).resolve() == bound(spec[role]).resolve(), 'Actual imported library SOURCE differs: ' + role)
    require(head.num_points == 12544 and len(head.assigner.match_costs) == 3 and
        [type(cost).__name__ for cost in head.assigner.match_costs] == ['ClassificationCost', 'CrossEntropyLossCost', 'DiceCost'] and
        [cost.weight for cost in head.assigner.match_costs] == [2., 5., 5.], 'Original final-stage native Hungarian point/cost recipe differs')
    for cost in head.assigner.match_costs:
        require(Path(inspect.getsourcefile(type(cost))).resolve() == bound(spec['match_cost_SOURCE']).resolve(),
            'Actual imported native weighted cost SOURCE differs')


def raw_population(torch, np, coco, fit, contract):
    require(len(fit['images']) == len(coco.imgs) == 741 and len(fit['annotations']) == len(coco.anns) == 1661 and
        set(coco.cats) == {1}, 'Pinned original741FIT1661GT population differs')
    original_ids = [int(row['id']) for row in fit['images']]
    selected = list(original_ids)
    random.Random(2026101019).shuffle(selected)
    split = contract['split']
    require(sorted(selected[:128]) == split['train_ids'] and sorted(selected[128:192]) == split['readout_heldout_ids'] and
        len(set(selected[:192])) == 192, 'Independent prelocked128/64 disjoint split differs')
    cache = {}
    for partition, rows, total in (('train', split['train_rows'], 292), ('heldout', split['readout_heldout_rows'], 106)):
        require(len(rows) == (128 if partition == 'train' else 64), 'Prelocked partition count differs')
        count = 0
        for row in rows:
            image_id = row['image_id']
            ids = [ann['id'] for ann in coco.imgToAnns[image_id]]
            require(ids == row['annotation_ids'] and original_ids[row['dataset_index']] == image_id and
                len(ids) == row['raw_GT'], 'Every raw original annotation/order must survive: ' + str(image_id))
            image = coco.imgs[image_id]
            shape = [image['height'], image['width']]
            raster = [np.zeros(shape, dtype=np.uint8) if coco.anns[ann_id].get('segmentation') == [] else
                coco.annToMask(coco.anns[ann_id]) for ann_id in ids]
            require(all(list(mask.shape) == shape and bool(np.isin(mask, [0, 1]).all()) for mask in raster),
                'Independent raw crowd/ignore/empty raster decode differs')
            targets = torch.from_numpy(np.stack(raster).astype(np.float32)) if raster else torch.empty((0,) + tuple(shape))
            cache[image_id] = dict(ann_ids=ids, targets=targets, shape=shape, dataset_index=row['dataset_index'])
            count += len(ids)
        require(count == total, 'Every raw partition GT must survive')
    generator = torch.Generator(device='cpu')
    generator.manual_seed(2026101021)
    orders = [[split['train_ids'][int(index)] for index in torch.randperm(128, generator=generator).tolist()] for _ in range(4)]
    return cache, orders


def matching_check(torch, np, old, assignment, cache, linear_sum_assignment):
    checked = {}
    for image_key, row in assignment.items():
        image_id = int(image_key)
        target = cache[image_id]
        ids = target['ann_ids']
        expect(row, dict(image_id=image_id, weights=[2.0, 5.0, 5.0], seed=2800000 + image_id,
            points=12544, native_get_targets_calls=1, external_RNG_restored=True,
            actual_total_is_native_GPU_stack_sum_then_CPU=True, CPU_cost_derivation_claimed=False),
            'Actual source-observed native final-stage matching')
        require(row['raw_ann_ids'] == ids and len(row['query_by_GT']) == len(ids) and
            len(set(row['query_by_GT'])) == len(ids) and all(type(q) is int and 0 <= q < 100 for q in row['query_by_GT']),
            'Fixed native assignment must cover each raw annotation once with unique native query')
        require(len(row['positive_query_indices']) == len(ids) and len(set(row['positive_query_indices'])) == len(ids) and
            sorted(row['assigned_GT_indices']) == list(range(len(ids))) and
            all(row['query_by_GT'][j] == q for q, j in zip(row['positive_query_indices'], row['assigned_GT_indices'])) and
            row['rows'] == [dict(annotation_id=ann_id, GT_index=j, query_index=row['query_by_GT'][j]) for j, ann_id in enumerate(ids)],
            'Actual positive query/assigned rawGT row map differs')
        parts = [torch.tensor(value, dtype=torch.float32) for value in row['cost_parts']]
        total = torch.tensor(row['actual_total'], dtype=torch.float32)
        require(len(parts) == 3 and all(part.shape == total.shape == (100, len(ids)) and bool(torch.isfinite(part).all())
            for part in parts) and bool(torch.isfinite(total).all()), 'Three finite weighted native100xGT cost parts required')
        tensor_close(torch, torch.stack(parts).sum(0), total, 'Actual native cost-parts sum')
        require([tensor_record(part) for part in parts] == row['cost_digests'] and
            tensor_record(total) == row['actual_total_digest'], 'Actual exported matching FP32 cost digests differ')
        require(len(row['private_RNG_before']) == len(row['private_RNG_after']) == 2,
            'Actual native CPU+CUDA matching RNG packet required')
        rows, columns = linear_sum_assignment(total.numpy())
        independent = math.fsum(float(total[r, col]) for r, col in zip(rows, columns))
        actual = math.fsum(float(total[q, col]) for col, q in enumerate(row['query_by_GT']))
        close(actual, independent, 'Exported native Hungarian minimum', rtol=1e-6, atol=1e-6)
        checked[image_id] = dict(query_by_GT=row['query_by_GT'], raw_ann_ids=ids,
            actual_exported_matrix_minimum_cost=actual, native_cost_derivation_CPU_recomputed=False)
    return checked


def tree_equal(torch, left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and set(left) == set(right) and all(tree_equal(torch, left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(tree_equal(torch, a, b) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


def AdamW_audit(torch, arm, g, weights, saved, optimizer_before):
    after = saved['optimizer_state_after']
    if (g + 1) % 2 != 0:
        require(tree_equal(torch, optimizer_before, after) and
            all(torch.equal(weights[name].detach(), saved['parameters_after'][name]) for name in weights),
            'Pending accumulation microstep cannot mutate parameters or AdamW moments')
        return dict(optimizer_update=False, actual_optimizer_and_parameter_identity_preserved=True)
    names = ['log_gain'] if arm == ARMS[2] else RAW_NAMES
    require(len(optimizer_before['param_groups']) == len(after['param_groups']) == len(names) and
        optimizer_before['param_groups'] == after['param_groups'], 'Bounded optimizer groups changed during one update')
    proof = {}
    for name, group in zip(names, optimizer_before['param_groups']):
        lr = .01 if arm == ARMS[2] else .0001
        expect(group, dict(lr=lr, weight_decay=.05, betas=(.9, .999), eps=1e-8, amsgrad=False,
            maximize=False, capturable=False, differentiable=False), 'Exact actual bounded AdamW group')
        require(len(group['params']) == 1, 'One active parameter per bounded AdamW group required')
        key = group['params'][0]
        original = weights[name].detach()
        if key in optimizer_before['state']:
            previous = optimizer_before['state'][key]
            m, v, step = previous['exp_avg'], previous['exp_avg_sq'], float(previous['step'])
        else:
            require(g == 1, 'Only first actual update may lack previous AdamW moments')
            m, v, step = torch.zeros_like(original), torch.zeros_like(original), 0.
        require(step == (g + 1) // 2 - 1, 'Actual before-update moment step differs')
        gradient = saved['clipped_gradients'][name]
        # Independent AdamW equations on exported actual clipped gradients.
        mean = m * .9 + gradient * .1
        variance = v * .999 + gradient.square() * .001
        new_step = step + 1
        denominator = variance.sqrt() / math.sqrt(1 - .999 ** new_step) + 1e-8
        expected = original * (1 - lr * .05) - (lr / (1 - .9 ** new_step)) * mean / denominator
        actual_state = after['state'][key]
        require(set(actual_state) == {'step', 'exp_avg', 'exp_avg_sq'} and float(actual_state['step']) == new_step,
            'Actual bounded AdamW update step/moment population differs')
        moment = tensor_close(torch, actual_state['exp_avg'], mean, 'Independent bounded AdamW first moment', 1e-4, 1e-8)
        variance_proof = tensor_close(torch, actual_state['exp_avg_sq'], variance, 'Independent bounded AdamW second moment', 1e-4, 1e-8)
        delta = tensor_close(torch, saved['parameters_after'][name] - original, expected - original,
            'Independent bounded AdamW parameter delta', 1e-3, 1e-7)
        proof[name] = dict(first_moment=moment, second_moment=variance_proof, parameter_delta=delta, step=int(new_step))
    require(set(after['state']) == {group['params'][0] for group in after['param_groups']},
        'No native or foreign state may enter bounded AdamW update')
    return dict(optimizer_update=True, independently_recomputed_actual_clipped_gradient_update=proof,
        full512_optimizer_trajectory_replayed=False)


def training_trace(torch, old, contract, trace, orders, cache, matches):
    require(isinstance(trace, list) and len(trace) == 512, 'Complete shared512 micro trace required')
    visit_GT, positive = 0, {arm: [0] * (1 if arm == ARMS[2] else 2) for arm in ARMS}
    previous = {arm: None for arm in ARMS}
    for g, row in enumerate(trace):
        image_id = orders[g // 128][g % 128]
        target = cache[image_id]
        expect(row, dict(global_microstep=g, epoch=g // 128, offset=g % 128, image_id=image_id,
            raw_GT=len(target['ann_ids']), matching_image_id=image_id, matching_reused=g >= 128,
            integer_pixel_seed=3800000 + g, integer_pixel_draws=len(target['ann_ids']) * 12544,
            sampled_with_replacement=True, pending_accumulation_before=g % 2, pending_accumulation_after=(g + 1) % 2),
            'Actual shared teacher training trace')
        require(row['identity']['ann_ids'] == target['ann_ids'] and
            set(row['arms']) == set(ARMS), 'Shared image/allGT/fixed assignment/three independent arms differ')
        expect(row, dict(teacher_all451_grad_None=True, all_teacher_modules_eval=True), 'Frozen single teacher trace')
        expect(row['forward'], dict(head_calls=1, backbone_calls=1, pixel_decoder_calls=1, mask_embed_calls=10,
            seed=100000 + image_id, external_RNG_restored=True, hooks_removed=True, all4_GT_fields_absent=True,
            classes_all10_detached=True, first9_masks_detached=True, native_all10_tensor_versions_unchanged=True),
            'One shared GT-free teacher forward per microstep')
        require(len(row['forward']['private_RNG_before']) == len(row['forward']['private_RNG_after']) == 2,
            'Actual private CPU/CUDA head RNG identities required')
        indices = sampled_indices(torch, len(target['ann_ids']), *target['shape'], g)
        require(tensor_record(indices) == row['indices_digest'], 'Every micro private integer12544 sample sequence differs')
        require((row['shared_native_fields'] is not None) == (g in DETAILS), 'Detailed SOURCE-observed input scope differs')
        for role, shape in (('classes10', [1, 100, 2]), ('native_masks10', [1, 100, 200, 200])):
            if g in DETAILS:
                require(len(row['forward'][role]) == 10, 'All ten actual native stages must be represented in detailed source proof')
                for value in row['forward'][role]:
                    digest_row(value, shape)
            else:
                require(row['forward'][role] is None, 'No detailed-stage hash scope may be extrapolated beyond four audits')
        visit_GT += len(target['ann_ids'])
        for arm in ARMS:
            data = row['arms'][arm]
            finite(data['loss'], 'Actual per-image BCE', True)
            require(len(data['per_GT_loss']) == len(target['ann_ids']), 'Every matched raw GT must contribute once')
            for value in data['per_GT_loss']:
                finite(value, 'Actual per-GT mean BCE', True)
            close(data['loss'], math.fsum(data['per_GT_loss']) / max(len(target['ann_ids']), 1), 'GT-mean image loss')
            step = (g + 1) % 2 == 0
            expect(data, dict(optimizer_update=step, updates_completed=(g + 1) // 2,
                supervision='uniform_original_integer_pixels' if arm == ARMS[0] else 'full_original_raster',
                pixels_per_GT=12544 if arm == ARMS[0] else math.prod(target['shape'])),
                'Global accumulation2/update counters')
            require(data['draw_count'] == len(target['ann_ids']) * (12544 if arm == ARMS[0] else math.prod(target['shape'])),
                'Actual uniform/full original raster draw count differs')
            names = ['log_gain'] if arm == ARMS[2] else RAW_NAMES
            gradients = data['accumulated_gradients_before_clip']
            require(set(gradients) == set(data['parameter_before']) == set(data['parameter_after']) == set(names),
                'Only expected standalone active parameter/gradient population required')
            for index, name in enumerate(names):
                value = gradients[name]['norm']
                finite(value, 'Actual finite active accumulated gradient norm', True)
                positive[arm][index] += int(value > 0)
                shape = [] if name == 'log_gain' else ([512, 512] if name == 'hidden.weight' else [256, 512])
                digest_row(gradients[name]['digest'], shape)
                digest_row(data['parameter_before'][name], shape)
                digest_row(data['parameter_after'][name], shape)
            if step:
                finite(data['clip_norm_before'], 'Actual preclip accumulated norm', True)
                require(set(data['clipped_gradients']) == set(names), 'Every active clipped gradient must be present')
                for name in names:
                    finite(data['clipped_gradients'][name]['norm'], 'Actual clipped active norm', True)
                    require(data['clipped_gradients'][name]['norm'] <= .01001,
                        'Actual active clipped gradient norm exceeds fixed clip.01')
            else:
                require(data['clip_norm_before'] is None and data['clipped_gradients'] is None and
                    data['parameter_before'] == data['parameter_after'], 'No clipping or parameter update on pending microstep')
            require(data['cold_hidden_gradient_exact_zero'] is (True if g < 2 and arm != ARMS[2] else None),
                'Actual cold zero-final RAW hidden gradient proof differs')
            if previous[arm] is not None:
                require(data['parameter_before'] == previous[arm], 'Independent arm parameter trajectory continuity differs')
            previous[arm] = data['parameter_after']
    require(visit_GT == 1168, 'Four complete128/292 epochs must visit every rawGT once per epoch')
    return dict(microsteps=512, updates_per_arm=256, raw_GT_visits_per_arm=visit_GT,
        positive_gradient_microcounts=positive, independently_floating_replayed_microsteps=DETAILS,
        all512_floating_losses_or_gradients_CPU_replayed=False)


def audit_check(torch, old, diag, packet, trace_row, cache, match):
    g, image_id = trace_row['global_microstep'], trace_row['image_id']
    target = cache[image_id]
    expect(packet, dict(schema='shared-readout-objective-control-audit-v1', global_microstep=g,
        epoch=g // 128, image_id=image_id), 'Prelocked detailed micro audit identity')
    require(packet['matching']['query_by_GT'] == match['query_by_GT'], 'Detailed audit fixed native mapping differs')
    context, teacher_fields = packet['RAW_context'], packet['teacher']
    expect(context, dict(activation='silu', actual_hidden_pre_hook_observed=True), 'Actual existing RAW SiLU context observation')
    feature, h, E, native200 = teacher_fields['F'], context['h'], context['eligible'], teacher_fields['M200']
    for value, shape, name in ((feature, [1, 256, 200, 200], 'F'), (h, [1, 100, 16, 512], 'RAW h'),
                               (native200, [1, 100, 200, 200], 'Actual native additive anchor')):
        cpu_tensor(torch, value, shape, name)
    cpu_tensor(torch, E, [1, 100, 16], 'SOURCE-observed eligible', torch.bool)
    cpu_tensor(torch, teacher_fields['e0'], [1, 100, 256], 'Native frozen coefficient')
    cpu_tensor(torch, teacher_fields['h'], [1, 100, 256], 'Native frozen query embedding')
    cpu_tensor(torch, teacher_fields['classes'], [1, 100, 2], 'Native frozen final-stage classes')
    cpu_tensor(torch, packet['valid_fraction'], [1, 1, 200, 200], 'Metadata valid fraction')
    require({name: tensor_record(value) for name, value in teacher_fields.items()} == trace_row['shared_native_fields'],
        'Detailed native packet must match actual shared forward trace digests')
    require(torch.equal(packet['original_targets'], target['targets']), 'Detailed audit targets differ from independent original rawGT')
    expected_indices = sampled_indices(torch, len(target['ann_ids']), *target['shape'], g)
    require(torch.equal(packet['indices'], expected_indices), 'Independent private12544 integer draws differ')
    identity = packet['identity']
    require(json.loads(json.dumps(identity, allow_nan=False)) == trace_row['identity'] and identity['ann_ids'] == target['ann_ids'] and
        identity['original_shape'] == target['shape'], 'Detailed independent rawGT/data identity differs')
    height, width = identity['actual_img_shape']
    coordinates = torch.arange(200, dtype=torch.float32)
    yf = ((float(height) - 4 * coordinates).clamp(0, 4) / 4)[None, None, :, None]
    xf = ((float(width) - 4 * coordinates).clamp(0, 4) / 4)[None, None, None, :]
    require(torch.equal(packet['valid_fraction'], yf * xf), 'Independent metadata-only valid fraction differs')
    tensor_close(torch, native200, torch.einsum('bqc,bchw->bqhw', teacher_fields['e0'], feature),
        'Independent native coefficient/feature projection audit')
    require(trace_row['forward']['classes10'][-1] == tensor_record(teacher_fields['classes']) and
        trace_row['forward']['native_masks10'][-1] == tensor_record(native200), 'Detailed final-stage packet/ten-stage source identity differs')
    results = {}
    for arm in ARMS:
        saved = packet['arms'][arm]
        weights = {name: value.detach().clone().requires_grad_(True) for name, value in packet['RAWstates_before'][arm].items()}
        names = ['log_gain'] if arm == ARMS[2] else RAW_NAMES
        require(list(weights) == names, 'Only standalone expected active weights may train')
        require({name: tensor_record(value) for name, value in weights.items()} == trace_row['arms'][arm]['parameter_before'] and
            {name: tensor_record(value) for name, value in saved['parameters_after'].items()} == trace_row['arms'][arm]['parameter_after'],
            'Detailed active parameter before/after must match complete actual trace')
        if arm == ARMS[2]:
            original = diag.render(torch, native200[:, match['query_by_GT']], identity['actual_img_shape'], target['shape'])[0]
            require(weights['log_gain'].numel() == 1, 'Gain has exactly one scalar and no added capacity')
            original = original * weights['log_gain'].exp()
        else:
            logits200 = raw_forward(torch, feature, h, E, native200, weights, match['query_by_GT'])
            original = diag.render(torch, logits200, identity['actual_img_shape'], target['shape'])[0]
        logit_proof = tensor_close(torch, saved['original_logits'], original.detach(), arm + ' CPU original logits')
        loss, losses = independent_BCE(torch, original, target['targets'], expected_indices if arm == ARMS[0] else None)
        close(trace_row['arms'][arm]['loss'], float(loss.detach()), arm + ' independently derived image BCE')
        for actual, expected in zip(trace_row['arms'][arm]['per_GT_loss'], losses.detach().tolist()):
            close(actual, expected, arm + ' independent per-GT BCE')
        (loss / 2).backward()
        gradients = {}
        for name in names:
            require(weights[name].grad is not None and bool(torch.isfinite(weights[name].grad).all()),
                'Actual CPU audit gradient must remain connected and finite')
            gradients[name] = tensor_close(torch, saved['micro_gradients'][name], weights[name].grad,
                arm + ' independent micro BCE/2 gradient: ' + name)
            close(float(saved['micro_gradients'][name].norm()), float(weights[name].grad.norm()),
                arm + ' independent micro gradient norm: ' + name, 1e-3, 1e-7)
            pending = saved['pending_gradients_before'][name]
            require((pending is None) == (g % 2 == 0), 'Actual pending gradients must be absent after every preceding update')
            if pending is None:
                pending = torch.zeros_like(weights[name])
            else:
                cpu_tensor(torch, pending, list(weights[name].shape), 'Actual pending accumulation gradient')
            tensor_close(torch, saved['accumulated_gradients_before_clip'][name], pending + weights[name].grad,
                arm + ' independently added accumulated gradient')
            require(tensor_record(saved['accumulated_gradients_before_clip'][name]) ==
                trace_row['arms'][arm]['accumulated_gradients_before_clip'][name]['digest'],
                'Actual detailed accumulated gradient must match complete trace')
            close(float(saved['accumulated_gradients_before_clip'][name].norm()),
                trace_row['arms'][arm]['accumulated_gradients_before_clip'][name]['norm'], 'Actual trace gradient norm')
            if g < 2 and name == 'hidden.weight':
                require(torch.count_nonzero(weights[name].grad) == 0 and torch.count_nonzero(saved['micro_gradients'][name]) == 0,
                    'Independent cold zero-final hidden gradient must be exactly zero')
        if (g + 1) % 2 == 0:
            norms = [saved['accumulated_gradients_before_clip'][name].norm() for name in names]
            norm = torch.stack(norms).norm()
            close(trace_row['arms'][arm]['clip_norm_before'], float(norm), arm + ' independent preclip norm')
            scale = (torch.tensor(.01) / (norm + 1e-6)).clamp(max=1.)
            for name in names:
                tensor_close(torch, saved['clipped_gradients'][name], saved['accumulated_gradients_before_clip'][name] * scale,
                    arm + ' independent clipping transform')
                require(tensor_record(saved['clipped_gradients'][name]) == trace_row['arms'][arm]['clipped_gradients'][name]['digest'],
                    'Detailed clipped gradient must match complete actual trace')
        else:
            require(saved['clipped_gradients'] is None, 'No clipping on first pending accumulation microstep')
        update = AdamW_audit(torch, arm, g, weights, saved, packet['optimizer_states_before'][arm])
        results[arm] = dict(logits=logit_proof, gradient=gradients, optimizer=update, image_BCE=float(loss.detach()),
            geometry_eligibility_or_native_matching_cost_CPU_rederived=False)
    return dict(global_microstep=g, image_id=image_id, arms=results, sampling_with_replacement=True,
        independent_integer_draw_count=len(target['ann_ids']) * 12544)


def checkpoint_check(torch, old, pre, teacher, spec, record, trace):
    cold, permutation = pre.independent_adapter(torch)
    results = {}
    for arm in ARMS:
        checkpoint_ref = record['checkpoints'][arm]
        checkpoint = torch.load(str(bound(checkpoint_ref)), map_location='cpu')
        require(set(checkpoint) == {'schema', 'meta', 'readout_state_dict', 'optimizer_state_dict', 'order_generator_state', 'fixed_assignment_rows'} and
            checkpoint['schema'] == 'shared-readout-objective-control-checkpoint-v1',
            'Readout-only endpoint checkpoint population differs')
        meta, weights = checkpoint['meta'], checkpoint['readout_state_dict']
        names = ['log_gain'] if arm == ARMS[2] else RAW_NAMES
        expect(meta, dict(arm=arm, phase=spec['phase'], source=spec['entry_source'], spec=record['spec'],
            teacher_checkpoint=spec['teacher_checkpoint'], teacher_qualification=spec['teacher_qualification'],
            SOURCE_contract=spec['SOURCE_contract'], completed_epochs=4, microsteps=512, optimizer_updates=256,
            raw_GT_visits=1168, optimizer_param_names=[[name] for name in names], teacher_mode='native0', teacher_epoch=24,
            frozen_native=True, final_only_BCE=True, formal_resumable=False, teacher_optimizer_loaded=False,
            teacher_order_RNG_loaded=False, preflight_or_SHORT2_CP_loaded=False,
            gain_after_original_renderer=arm == ARMS[2], capacity_or_optimizer_matched=arm != ARMS[2]),
            'Readout-only fixedE4 endpoint metadata')
        require(list(weights) == names and sum(value.numel() for value in weights.values()) == (1 if arm == ARMS[2] else 393216),
            'Only expected two weights or positive gain scalar can be saved')
        for name, value in weights.items():
            cpu_tensor(torch, value, [] if name == 'log_gain' else list(cold[name].shape), 'Final active ' + name)
            require(tensor_record(value) == trace[-1]['arms'][arm]['parameter_after'][name], 'CP final trace tensor identity differs')
        initial = {'log_gain': torch.zeros((), dtype=torch.float32)} if arm == ARMS[2] else cold
        require({name: tensor_record(value) for name, value in initial.items()} == trace[0]['arms'][arm]['parameter_before'],
            'Independent fresh RAW seed/zero scalar does not match first actual trace')
        changed = [name for name in names if not torch.equal(weights[name], initial[name])]
        optimizer = checkpoint['optimizer_state_dict']
        require(len(optimizer['param_groups']) == len(optimizer['state']) == len(names), 'Only singleton active AdamW groups/state required')
        seen = []
        for name, group in zip(names, optimizer['param_groups']):
            expect(group, dict(lr=.01 if arm == ARMS[2] else .0001, weight_decay=.05, eps=1e-8,
                betas=(.9, .999), amsgrad=False), 'Exact active AdamW endpoint group')
            require(len(group['params']) == 1 and group['params'][0] not in seen, 'Distinct singleton active optimizer identities required')
            index = group['params'][0]
            seen.append(index)
            state = optimizer['state'][index]
            require(set(state) == {'step', 'exp_avg', 'exp_avg_sq'}, 'Every AdamW moment/step must be present')
            require(float(state['step']) == 256, 'Every present active AdamW state must reach exactly step256')
            for key in ('exp_avg', 'exp_avg_sq'):
                cpu_tensor(torch, state[key], list(weights[name].shape), 'Final actual AdamW ' + key)
            require(bool((state['exp_avg_sq'] >= 0).all()), 'AdamW second moment must be nonnegative')
        require(set(seen) == set(optimizer['state']), 'Optimizer state identities contain native or foreign entries')
        generator = torch.Generator(device='cpu').manual_seed(2026101021)
        for _ in range(4):
            torch.randperm(128, generator=generator)
        require(torch.equal(generator.get_state(), checkpoint['order_generator_state']) and
            checkpoint['fixed_assignment_rows'] == read_ref(record['matching']), 'Endpoint private order state/fixed assignment identities differ')
        logical = dict(teacher)
        logical.update({'standalone_readout.' + name: value for name, value in weights.items()})
        require(len(logical) == (611 if arm == ARMS[2] else 612) and
            all(torch.equal(logical[name], value) for name, value in teacher.items()), 'Strict logical native610+standalone active endpoint reconstruction differs')
        results[arm] = dict(checkpoint=checkpoint_ref, active_names=names, AdamW_step=256,
            changed_active_names=changed,
            logical_state_tensors=len(teacher) + len(names), strict_readout_only_CPU_loaded=True,
            native610_bound_to_exact_immutable_teacher=True, eligible_for_formal_resume=False)
    return results


def heldout_check(torch, np, masks, coco, COCOeval, old, diag, official, contract, cache, record):
    ids = contract['split']['readout_heldout_ids']
    results, common = {}, {}
    require(set(record['mode_rows']) == set(ENDPOINTS), 'Exactly native0 and three fixedE4 heldout endpoints required')
    native_binary, native_query_order = {}, {}
    gain_checkpoint = torch.load(str(bound(record['checkpoints'][ARMS[2]])), map_location='cpu')
    expected_gain = float(gain_checkpoint['readout_state_dict']['log_gain'].exp())
    finite(expected_gain, 'Exact final checkpoint positive gain')
    require(expected_gain > 0, 'Final gain checkpoint must define finite positive exp(log_gain)')
    for endpoint in ENDPOINTS:
        value = record['mode_rows'][endpoint]
        require(value['checkpoint'] == (record['teacher_checkpoint'] if endpoint == 'native0' else record['checkpoints'][endpoint]),
            'Official heldout endpoint must join its exact teacher/readout checkpoint')
        predictions, rows, factors = read_ref(value['predictions']), read_ref(value['image_rows']), read_ref(value['score_factors'])
        require(len(rows) == 64 and len(predictions) == len(factors) == 6400 and
            [row['image_id'] for row in rows] == ids, 'Every heldout image/all100 predictions and factors required')
        grouped, seen = {image_id: [] for image_id in ids}, set()
        for prediction, factor in zip(predictions, factors):
            image_id, q = prediction['image_id'], prediction['query_id']
            require(image_id in grouped and type(q) is int and 0 <= q < 100 and (image_id, q) not in seen and
                prediction['category_id'] == 1, 'All100 original native query ordinals must be retained once')
            seen.add((image_id, q))
            diag.decode_RLE(np, masks, prediction['segmentation'], cache[image_id]['shape'], 'Official heldout RLE')
            if endpoint == 'native0':
                native_binary[image_id, q] = prediction['segmentation']
            if endpoint == ARMS[2]:
                require(prediction['segmentation'] == native_binary[image_id, q],
                    'Actual positive original-space gain must preserve every native RLE in this run')
            expect(factor, dict(image_id=image_id, query_id=q, official_score=prediction['score']), 'Original query score-factor join')
            for name in ('class_probability', 'mask_quality', 'official_score'):
                finite(factor[name], 'Official FP32 factor')
                require(0 <= factor[name] <= 1 and float(np.float32(factor[name])) == factor[name], 'Bounded exact FP32 factors required')
            product = factor['class_probability'] * factor['mask_quality']
            envelope = 2. ** -24 / (1 - 2. ** -24) * abs(product) + 2. ** -126
            require(abs(factor['official_score'] - product) <= envelope, 'Official score violates FP32 factor multiplication bound')
            grouped[image_id].append(prediction)
        require(seen == {(image_id, q) for image_id in ids for q in range(100)}, 'Heldout native100 query coverage differs')
        for row in rows:
            image_id = row['image_id']
            require(row['identity']['ann_ids'] == cache[image_id]['ann_ids'], 'All raw heldout annotations must survive loader identity')
            expect(row, dict(mode=endpoint, official_fusion_mask_score_query_TorchEqual=True,
                all100_queries_retained=True, GT_free_forward=True), 'Official frozen heldout inference scope')
            expect(row['forward'], dict(head_calls=1, backbone_calls=1, pixel_decoder_calls=1, mask_embed_calls=10,
                seed=100000 + image_id, external_RNG_restored=True, hooks_removed=True, all4_GT_fields_absent=True,
                classes_all10_detached=True, first9_masks_detached=True, native_all10_tensor_versions_unchanged=True),
                'One shared GT-free official heldout forward')
            require(row['forward']['classes10'] is None and row['forward']['native_masks10'] is None and
                row['GT'] == cache[image_id]['native_GT_digest'], 'Independent complete padded native GT or bounded-stage scope differs')
            require(sorted(row['query_ids']) == list(range(100)) and row['query_ids'] ==
                [prediction['query_id'] for prediction in grouped[image_id]], 'Actual official query ordering differs from export')
            if image_id in native_query_order:
                require(row['query_ids'] == native_query_order[image_id], 'Official class/query ordering changed across endpoints')
            else:
                native_query_order[image_id] = row['query_ids']
            packet = row['native_class_packet']
            classes = torch.tensor(packet['values'], dtype=torch.float32)
            cpu_tensor(torch, classes, [100, 2], 'Native final-stage class packet')
            require(packet['dtype'] == 'float32' and packet['shape'] == [100, 2] and
                tensor_record(classes)['sha256'] == packet['sha256'], 'Native class packet FP32 digest differs')
            if image_id in common:
                require(common[image_id] == (row['native_fields'], packet), 'Native classes/stage tensors changed across heldout endpoints')
            else:
                common[image_id] = row['native_fields'], packet
            if endpoint == ARMS[2]:
                finite(row['gain'], 'Actual positive original-space gain')
                require(row['gain'] > 0 and row['gain_binary_mask_agreement'] is True and row['readout_diagnostics'] is None,
                    'Gain must be positive after original renderer and have actual binary agreement')
                close(row['gain'], expected_gain, 'Actual heldout gain/final scalar checkpoint', 1e-6, 1e-6)
            else:
                require(row['gain'] is None and row['gain_binary_mask_agreement'] is None, 'Only ancillary gain endpoint may scale original logits')
            probabilities = classes.softmax(-1)[:, 0]
            for factor in factors:
                if factor['image_id'] == image_id:
                    close(factor['class_probability'], float(probabilities[factor['query_id']]), 'Independent class probability', 1e-6, 1e-6)
        diagnostic = []
        empty_predictions = 0
        for image_id in ids:
            ordered = sorted(grouped[image_id], key=lambda row: row['query_id'])
            binary = torch.from_numpy(np.stack([diag.decode_RLE(np, masks, p['segmentation'], cache[image_id]['shape'], 'GT-best100 RLE')
                for p in ordered]))
            empty_predictions += int((binary.sum((1, 2)) == 0).sum())
            for index, ann_id in enumerate(cache[image_id]['ann_ids']):
                target = cache[image_id]['targets'][index].bool()
                metrics = [diag.binary_metrics(torch, mask, target) for mask in binary]
                best = max(range(100), key=lambda q: (metrics[q]['IoU'], -q))
                diagnostic.append(dict(image_id=image_id, annotation_id=ann_id, best_query_id=best,
                    category_id=coco.anns[ann_id]['category_id'], iscrowd=int(coco.anns[ann_id].get('iscrowd', 0)),
                    ignore=int(coco.anns[ann_id].get('ignore', 0)), metrics=metrics[best],
                    tie_rule='lowest_actual_query_id', diagnostic_only_not_AP=True))
        require(len(diagnostic) == 106, 'All106 original heldout GT must remain in score-free diagnostic')
        metrics = official.coco_metrics(coco, predictions, ids, COCOeval)
        results[endpoint] = dict(predictions=value['predictions'], image_rows=value['image_rows'],
            query_score_factors=value['score_factors'], official_COCO=metrics,
            GT_best100=diagnostic, empty_predictions=empty_predictions,
            pixel_fusion_threshold_or_quality_mean_CPU_recomputed=False)
    return results


def run(spec_file, acceptance_file):
    started = time.monotonic()
    spec_path, target = Path(spec_file), Path(acceptance_file)
    require(spec_path.is_absolute() and target.is_absolute() and target.parent.is_dir() and not target.exists(),
        'Absolute SPEC and fresh absolute acceptance in existing parent required')
    spec_path = spec_path.resolve(strict=True)
    spec = read(spec_path)  # Execution metadata only; no hash before raw gates.
    expect(spec, dict(schema=SPEC_SCHEMA, arms=ARMS, expected_epoch=24, CPU_affinity=[0, 1], deadline_seconds=5400,
        allocator_limit_MiB=4096, whole_GPU_owner_limit_MiB=8192), 'Prelocked objective-control SPEC')
    require(spec['phase'].startswith('shared_readout_objective_control_FIT128_') and Path(spec['phase']).name == spec['phase'],
        'Safe exclusive objective-control phase required')
    receipt_path = spec_path.parent / 'finite_point_cache_execution_20261005_v2' / 'runs' / spec['phase'] / 'PARENT_RECEIPT.json'
    raw = read(receipt_path)
    terminal(raw, spec['phase'])
    receipts = {}
    for role, phase_role in (('teacher_receipt', 'teacher_phase'), ('preflight_receipt', 'preflight_phase'),
                            ('short2_receipt', 'short2_phase'), ('prior_DEV_receipt', 'prior_DEV_phase'),
                            ('prior_diag_receipt', 'prior_diag_phase')):
        require(Path(spec[role]['path']).is_absolute(), 'Absolute prior raw receipt required')
        receipts[role] = read(spec[role]['path'])
        terminal(receipts[role], spec[phase_role])
    # ALL six actual held0/reap/birth-absence/noerror terminals precede hashes,
    # qualifications, checkpoint/GT/audit reads and scientific imports.
    execution, contract = source_gate(spec_path, spec, raw)
    for name in ('epochs', 'microsteps', 'optimizer_updates', 'raw_GT_visits_per_arm', 'accumulation',
                 'detail_microsteps', 'hidden_seed', 'permutation_seed', 'order_seed', 'head_seed_base',
                 'matching_seed_base', 'integer_pixel_seed_base', 'sampled_pixels_per_GT', 'sampling_with_replacement',
                 'raw_optimizer', 'gain_optimizer'):
        if name in spec:
            require(spec[name] == contract[name], 'Optional SPEC protocol duplicate differs from authoritative contract: ' + name)
    fixture = fixture_gate(spec, contract)
    diag = load_source(spec['prior_diag_CPU_reader_source'], 'sealed_prior_diagnostic_CPU_helpers')
    prior_diag, diag_contract = prior_diagnostic_gate(spec, receipts['prior_diag_receipt'], diag, contract)
    # The original diagnostic fixture binds the original producer, so validate
    # it against the pinned original SPEC rather than this fresh entry.
    diag.fixture_gate(read_ref(spec['prior_diag_spec']))
    prior = load_source(contract['base_role_SOURCE_refs']['CPU_reader'], 'sealed_prior_DEV_CPU_helpers')
    short = load_source(spec['short2_CPU_reader'], 'sealed_SHORT2_CPU_helpers')
    pre = load_source(spec['preflight_CPU_reader'], 'sealed_preflight_CPU_helpers')
    for role in pre.ARTIFACT_PINS:
        pre.pinned_role(spec, role)
    short.preflight_gate(pre, spec, receipts['preflight_receipt'])
    training, teacher_q = pre.teacher_lineage(spec, receipts['teacher_receipt'])
    pre.fixture_gate(spec)
    prior.qualified_short2(spec, receipts['short2_receipt'], short, pre)
    diag.prior_DEV_gate(spec, receipts['prior_DEV_receipt'], prior, diag_contract)
    output = Path(spec['output_directory'])
    require(output.is_absolute(), 'Absolute current output directory required')
    report_ref, start_ref, aggregate_ref = [reference(output / name) for name in ('REPORT.json', 'START.json', 'QUALIFIED.json')]
    report, start, aggregate = read_ref(report_ref), read_ref(start_ref), read_ref(aggregate_ref)
    joins = {role: spec[role] for role in ('entry_source', 'CPU_reader', 'SOURCE_contract', 'teacher_checkpoint', 'teacher_qualification',
        'prior_diag_spec', 'prior_diag_CPU_qualification', 'prior_diag_receipt', 'control_CPU_fixture',
        'control_CPU_fixture_spec', 'control_CPU_fixture_source')}
    expect(report, dict(schema=REPORT_SCHEMA, status=REPORT_STATUS, phase=spec['phase'], source=execution['source'],
        spec=execution['spec'], SOURCE_contract=spec['SOURCE_contract'], joins=joins, arms=ARMS, endpoints=ENDPOINTS, error=None,
        epochs=4, microsteps=512, optimizer_updates_per_arm=256, train_images=128, train_raw_GT=292,
        raw_GT_visits_per_arm=1168, readout_heldout_images=64, readout_heldout_raw_GT=106, teacher_forward_calls=576,
        shared_teacher_forward=True, full_native610_and_BN_exact=True, all_native451_frozen=True, GT_free_forward=True,
        feature_cache_used=False, fixed_assignment_cache_only=True, final_only_BCE=True, no_Dice_training_loss=True,
        no_unmatched_mask_loss=True, no_DEV_TEST_access=True, teacher_trained_all_selected_FIT=True,
        blind_holdout_claimed=False, official_all100_endpoint_exports=True, AP_computed_by_GPU_producer=False,
        AP_pending_independent_CPU_reader=True, gain_location='after_original_renderer_before_instance_postprocess',
        gain_capacity_or_optimizer_matched=False, GPU_optimizer_bitwise_determinism_claimed=False,
        efficacy_claimed=False, convergence_claimed=False, novelty_claimed=False, SOTA_claimed=False,
        previous_SHORT2_causal_explanation_claimed=False), 'Actual completed control report')
    require(report['full_empty_image_model_integration_qualified'] is False,
        'SOURCE-only empty helper proofs cannot qualify an unexecuted empty-image model path')
    expect(start, dict(schema='shared-readout-objective-control-start-v1', phase=spec['phase'], spec=execution['spec'],
        arms=ARMS, endpoints=ENDPOINTS, joins=joins, original_five_raw_gates_before_any_hash_or_import=True,
        shared_teacher_forward=True, feature_cache_used=False, fixed_matching_cache_only=True, epochs=4, microsteps=512,
        updates=256, teacher_trained_all_selected_FIT=True, blind_holdout_claimed=False,
        no_DEV_TEST_access=True, no_automatic_retry=True), 'Prelocked START identity')
    record = report['result']
    expect(aggregate, dict(schema='shared-readout-objective-control-aggregate-v1',
        status='ROOT_SHARED_READOUT_OBJECTIVE_CONTROL_PENDING_CPU_READER', phase=spec['phase'], source=execution['source'],
        spec=execution['spec'], report=report_ref, result=record, independently_CPU_qualified=False, AP_computed=False),
        'Pending aggregate/report/result join')
    require(set(record) == {'mode_rows', 'checkpoints', 'trace', 'order', 'matching', 'heldout_matching', 'state_proof', 'audits'} and
        set(record['checkpoints']) == set(ARMS), 'Complete current result artifact population differs')
    record = dict(record, spec=execution['spec'], teacher_checkpoint=spec['teacher_checkpoint'])
    finite(report['elapsed_seconds'], 'Actual GPU elapsed', True)
    require(report['elapsed_seconds'] < 5400 and type(report['torch_peak_allocated_bytes']) is int and
        0 <= report['torch_peak_allocated_bytes'] <= 4096 * 1024 ** 2 and report['cleanup']['allocated_after'] == 0,
        'Actual deadline/allocator/cleanup resource proof differs')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '2'
    os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
    import psutil
    psutil.Process().cpu_affinity([0, 1])
    import numpy as np
    import cv2
    import torch
    from pycocotools import mask as masks
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from scipy.optimize import linear_sum_assignment
    from mmengine import Config
    from mmengine.model import revert_sync_batchnorm
    from mmengine.registry import init_default_scope
    from mmdet.registry import MODELS
    from mmdet.utils import register_all_modules
    from mmdet.datasets import CocoDataset
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    require(psutil.Process().cpu_affinity() == [0, 1] and not torch.cuda.is_initialized(), 'CPU01/CUDAempty/noCUDAinitialization required')
    register_all_modules(init_default_scope=False)
    init_default_scope('mmdet')
    old = load_source(spec['reused_CPU_reader'], 'sealed_original_CPU_state_helpers')
    teacher_reader = load_source(spec['training_CPU_reader'], 'sealed_E24_CPU_state_helpers')
    initializer = load_source(spec['initializer_source'], 'sealed_original_initializer')
    official = load_source(spec['official_evaluation_CPU_reader'], 'sealed_official_COCO_CPU_helpers')
    rng_before = torch.get_rng_state().clone()
    model, teacher = pre.load_teacher(torch, old, teacher_reader, initializer, MODELS,
        revert_sync_batchnorm, Config, spec, training, teacher_q)
    model.requires_grad_(False).eval()
    diag.library_paths(torch, model, spec)
    imported_libraries(torch, model, spec)
    state = read_ref(record['state_proof'])
    native, BN_modules, BN_affine, BN_buffers = pre.typed_native(torch, model)
    require(len(teacher) == len(model.state_dict()) == 610 and len(native) == 451 and
        state['BN_parameter_names'] == BN_affine and state['BN_buffer_names'] == BN_buffers,
        'Independent native610/451/typed BN106+159 population differs')
    for role in ('before', 'after', 'native_teacher610'):
        old.check_state(torch, teacher, state[role], 'Actual immutable frozen native ' + role)
    expect(state, dict(schema='shared-readout-objective-control-state-proof-v1', full_native610_CPU_exact=True,
        frozen_BN_all_exact=True, all_native451_frozen=True, all_modules_eval=True, native_parameter_grads_None=True,
        hook_cache_clear=True, GT_free_forward=True, teacher_forward_calls=576, teacher_optimizer_loaded=False,
        teacher_order_RNG_loaded=False, feature_cache_used=False, fresh_optimizer_state_entries={arm: 0 for arm in ARMS},
        optimizer_step_nonNone_gradient_counts={arm: {name: 256 for name in (['log_gain'] if arm == ARMS[2] else RAW_NAMES)} for arm in ARMS},
        completed_epochs=4, microsteps=512, optimizer_updates={arm: 256 for arm in ARMS},
        no_pending_accumulation_gradients=True, audit_microsteps=DETAILS, audit_refs=record['audits'],
        training_raw_GT=292, training_raw_GT_visits_per_arm=1168, readout_heldout_raw_GT=106,
        teacher_trained_all_selected_FIT=True, blind_holdout_claimed=False), 'Actual frozen native SOURCE-observed endpoint')
    require(state['native_parameter_names'] == native and state['BN_module_names'] == BN_modules and
        all(torch.equal(model.state_dict()[name], teacher[name]) for name in teacher), 'Independent full native architecture/teacher tensors differ')
    old.check_state(torch, teacher, state['teacher_loading']['trained_fullstate'], 'Actual strictly loaded teacher610')
    expect(state['teacher_loading'], dict(mode='native0', checkpoint=spec['teacher_checkpoint'], qualification=spec['teacher_qualification'],
        CP_qualified=True, strict_fullstate_loaded=True, native_state_tensors=610, adapter_state_tensors=0,
        registered_parameters=451, original_frozen_BN_exact=True, inference_backbone_with_cp=False,
        no_random_missing_keys=True), 'Actual qualified complete teacher loading')
    cold, permutation = pre.independent_adapter(torch)
    old.check_state(torch, cold, state['initial_readout_states'], 'Actual fresh common RAW393216 initialization')
    old.check_state(torch, {'log_gain': torch.zeros((), dtype=torch.float32)}, state['initial_gain_state'], 'Actual gain initial1')
    coco = COCO(str(bound(spec['FIT_annotation'])))
    cache, orders = raw_population(torch, np, coco, read_ref(spec['FIT_annotation']), contract)
    require(Path(spec['image_root']).resolve() == Path('E:/PSeg_SSDD/voc_style/JPEGImages_train').resolve(),
        'Canonical original FIT image root differs')
    dataset = CocoDataset(ann_file=str(bound(spec['FIT_annotation'])), data_prefix=dict(img=spec['image_root']),
        metainfo=dict(classes=('ship',)), pipeline=[], filter_cfg=None, test_mode=True)
    plan = read_ref(record['order'])
    require(len(plan) == 4 and [row['image_ids'] for row in plan] == orders and
        [row['epoch'] for row in plan] == list(range(4)), 'Independent private common4epoch image orders differ')
    order_generator = torch.Generator(device='cpu').manual_seed(2026101021)
    for row in plan:
        require(row['RNG_before'] == tensor_record(order_generator.get_state()), 'Actual private order before-state differs')
        torch.randperm(128, generator=order_generator)
        require(row['RNG_after'] == tensor_record(order_generator.get_state()), 'Actual private order after-state differs')
    matching = read_ref(record['matching'])
    require(set(matching) == {str(image_id) for image_id in contract['split']['train_ids']}, 'Every128 training image needs one fixed mapping')
    matches = matching_check(torch, np, old, matching, cache, linear_sum_assignment)
    heldout_matching = read_ref(record['heldout_matching'])
    require(set(heldout_matching) == {str(image_id) for image_id in contract['split']['readout_heldout_ids']},
        'Every64 heldout image must preserve all rawGT matching metadata')
    heldout_matches = matching_check(torch, np, old, heldout_matching, cache, linear_sum_assignment)
    trace = read_ref(record['trace'])
    for row in trace:
        identity_check(torch, np, cv2, coco, dataset, spec, cache[row['image_id']], row['identity'])
    for row in read_ref(record['mode_rows']['native0']['image_rows']):
        identity_check(torch, np, cv2, coco, dataset, spec, cache[row['image_id']], row['identity'])
    trace_proof = training_trace(torch, old, contract, trace, orders, cache, matches)
    audits = []
    require(set(record['audits']) == {str(g) for g in DETAILS}, 'Exactly four prelocked detailed audit packets required')
    previous_audit = None
    for g in DETAILS:
        packet = torch.load(str(bound(record['audits'][str(g)])), map_location='cpu')
        if g % 2:
            require(previous_audit is not None and previous_audit['global_microstep'] == g - 1,
                'Detailed accumulation audits must be two consecutive prelocked pairs')
            for arm in ARMS:
                require(tree_equal(torch, packet['arms'][arm]['pending_gradients_before'],
                    previous_audit['arms'][arm]['accumulated_gradients_before_clip']) and
                    tree_equal(torch, packet['RAWstates_before'][arm], previous_audit['arms'][arm]['parameters_after']) and
                    tree_equal(torch, packet['optimizer_states_before'][arm], previous_audit['arms'][arm]['optimizer_state_after']),
                    'Actual detailed pair pending-gradient/parameter/moment identities differ')
        audits.append(audit_check(torch, old, diag, packet, trace[g], cache, matches[trace[g]['image_id']]))
        previous_audit = dict(global_microstep=g, arms={arm: {key: packet['arms'][arm][key] for key in
            ('accumulated_gradients_before_clip', 'parameters_after', 'optimizer_state_after')} for arm in ARMS}) if g % 2 == 0 else None
        del packet
        gc.collect()
    checkpoints = checkpoint_check(torch, old, pre, teacher, spec, record, trace)
    for arm in ARMS:
        cp = torch.load(str(bound(record['checkpoints'][arm])), map_location='cpu')
        old.check_state(torch, cp['readout_state_dict'], state['final_readout_states'][arm], 'Actual final readout SOURCE proof joined to CP')
    heldout = heldout_check(torch, np, masks, coco, COCOeval, old, diag, official, contract, cache, record)
    require(torch.equal(rng_before, torch.get_rng_state()) and not torch.cuda.is_initialized() and
        torch.get_num_threads() == torch.get_num_interop_threads() == 2 and psutil.Process().cpu_affinity() == [0, 1] and
        os.environ['CUDA_VISIBLE_DEVICES'] == '' and time.monotonic() - started < 5400,
        'CPU reader changed external RNG/resources, initialized CUDA or exceeded deadline')
    acceptance = dict(schema=QUAL_SCHEMA, status=QUAL_STATUS, phase=spec['phase'], arms=ARMS, endpoints=ENDPOINTS,
        source=execution['source'], reader_source=execution['reader'], spec=execution['spec'], report=report_ref,
        start=start_ref, GPU_qualification=aggregate_ref, parent_receipt=reference(receipt_path),
        GPU_admission=execution['GPU_admission'], owner_driver=execution['owner_driver'], SOURCE_contract=spec['SOURCE_contract'],
        teacher_checkpoint=spec['teacher_checkpoint'], teacher_qualification=spec['teacher_qualification'],
        prior_diag_CPU_qualification=spec['prior_diag_CPU_qualification'], fixture=fixture,
        lineage={role: spec[role] for role in ('teacher_training_spec', 'teacher_receipt', 'teacher_phase',
            'preflight_CPU_qualification', 'preflight_training_spec', 'preflight_receipt', 'preflight_phase',
            'short2_CPU_qualification', 'short2_training_spec', 'short2_receipt', 'short2_phase',
            'prior_DEV_spec', 'prior_DEV_CPU_qualification', 'prior_DEV_receipt', 'prior_DEV_phase',
            'prior_diag_spec', 'prior_diag_CPU_qualification', 'prior_diag_receipt', 'prior_diag_phase',
            'FIT_annotation', 'control_CPU_fixture', 'control_CPU_fixture_spec', 'control_CPU_fixture_source')},
        artifacts={key: record[key] for key in ('trace', 'order', 'matching', 'heldout_matching', 'state_proof', 'audits', 'checkpoints', 'mode_rows')},
        full_state_proof=dict(source=record['state_proof'], native_state_tensors=610, native_parameter_tensors=451,
            typed_BN_modules=len(BN_modules), typed_BN_affine_tensors=len(BN_affine), typed_BN_buffer_tensors=len(BN_buffers),
            before_after_teacher_and_strict_CPU_architecture_exact=True),
        fixed_matching=matches, training=trace_proof, audit_results=audits, checkpoint_results=checkpoints,
        endpoint_results=heldout, independent_floating_audit_microsteps=DETAILS,
        raw_current_and_five_original_terminal_gates_before_scientific_access=True,
        complete_original83_and_new_SOURCE_closure_owner_bound_before_imports=True,
        native610_and_typed_BN106_affine159_buffers_CPU_exact=True, zero_native_state_mutants=True,
        all_selected_raw_GT_including_empty_crowd_ignore_preserved=True,
        exported_Hungarian_matrix_minimum_CPU_checked=True, native_matching_cost_derivation_CPU_recomputed=False,
        all512_floating_losses_or_gradients_CPU_replayed=False, geometry_eligibility_CPU_rederived=False,
        official_COCO12_maxDets=[1, 10, 100], all64_readout_heldout106_GT_and_all100_queries_preserved=True,
        pixel_fusion_threshold_or_quality_mean_CPU_recomputed=False, teacher_trained_all_selected_FIT=True,
        blind_holdout_claimed=False, capacity_or_optimizer_matched_gain_claimed=False,
        eligible_for_formal_resume=False, GPU_forward_or_full_optimizer_trajectory_CPU_replayed=False,
        CPU_affinity=[0, 1], torch_threads=2, torch_interop_threads=2, CUDA_initialized=False,
        AP_computed=True, AP_scope='prelocked64_FIT_readout_heldout_at_fixedE4', no_DEV_TEST_access=True,
        efficacy_claimed=False, novelty_claimed=False, convergence_claimed=False, SOTA_claimed=False, global_method_failure_claimed=False,
        previous_SHORT2_causal_explanation_claimed=False, scientific_method_verdict=None,
        evidence_limits=[
            'Only prelocked FIT128 readout training and64 FIT readout-heldout images at fixedE4; teacher trained all FIT.',
            'Uniform with-replacement integer sampling estimates the full-raster expected BCE; realized cost, clipping and AdamW trajectories can differ.',
            'Floating CPU loss/gradient/clip replay covers only the four saved audit steps. Other512 trace facts are SOURCE-observed and checked for identity/count/state consistency.',
            'CPU minimum-cost assignment verifies exported actual100xGT matrices, not native sampled cost derivation, GT-independent matching or alternate seeds.',
            'SOURCE-observed h/E and native additive anchors condition independent RAW Phi replay; eligibility geometry is not independently rederived.',
            'Readout-only checkpoints bind immutable teacher610 and permit logical612/611 reconstruction, without formal resume eligibility.',
            'Official native fusion tensor equality is SOURCE-observed; CPU checks all100 RLE, class packets, score-factor arithmetic and official COCO12. Omitted pixel logits prevent threshold/quality-mean replay.',
            'Positive gain follows original renderer and is an ancillary amplitude/score control with different capacity and learning rate; no universal FP32 binary invariance claim.',
            'No convergence, blind TEST, SOTA, global method failure, previous SHORT2 causal explanation or GPU optimizer bitwise-repeatability verdict.'])
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
    require(args.owner_execute, 'Inert SOURCE requires explicit ROOT execution after six actual raw terminal gates')
    run(args.spec, args.acceptance)


if __name__ == '__main__':
    main()
