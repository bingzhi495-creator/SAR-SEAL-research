"""Inert independent CPU consumer of four fresh frozen-native SHORT2 arms.

ROOT alone executes this consumer after actual current, teacher and preflight
owned raw terminal gates. SOURCE preparation does no scientific import or
artifact access. Every arm must cover all741 FIT images and1661 raw GT twice,
using1482 microsteps and741 accumulation-two updates, including the odd741
epoch boundary without a gradient flush. Native610 stays exactly the same E24
teacher. This bounded two-epoch audit establishes no AP, efficacy, convergence,
novelty or superiority. Eight detailed observations do not prove all-micro
feature/geometry equality. CPU reconstructs GT descriptors once per raw image;
it never caches features, replays GPU forward/backward or recomputes geometry.
"""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import types

SCHEMA = 'ROOT-query-conditioned-frozen-native-short2-spec-v1'
REPORT_SCHEMA = 'query-conditioned-frozen-native-short2-report-v1'
CP_SCHEMA = 'query-conditioned-frozen-native-short2-checkpoint-v1'
PROOF_SCHEMA = 'query-conditioned-frozen-native-short2-state-proof-v1'
QUAL_SCHEMA = 'ROOT-query-conditioned-frozen-native-short2-CPU-qualification-v1'
QUAL_STATUS = 'ROOT_QUERY_CONDITIONED_FROZEN_NATIVE_SHORT2_CPU_QUALIFIED'
MODES = ('raw', 'raw_rms', 'query_rms', 'permuted_query_rms')
DETAILS = (0, 1, 2, 3, 1478, 1479, 1480, 1481)
FIT_IMAGES, FIT_GT, EPOCHS, MICROS, UPDATES, GT_VISITS = 741, 1661, 2, 1482, 741, 3322
PREFIX = 'panoptic_head.query_pair_readout.'
ACTIVE = [PREFIX + 'hidden.weight', PREFIX + 'final.weight']
OWNER_SHA = 'ef9a9f17d4fb1bc3464936095ef799021dd880f8b15488d558a871c08b6e1924'
C_ROOT = Path(__file__).resolve().parent.parent
H_ROOT = C_ROOT.parent / 'goal_continuation_20261005'
PREFLIGHT_PHASE = 'query_conditioned_frozen_native_preflight_20261010_01'
PREFLIGHT_PINS = {
    'preflight_training_spec': dict(path=str(H_ROOT / 'ROOT_QUERY_PAIR_FROZEN_NATIVE_PREFLIGHT_SPEC_20261010_01.json'),
        bytes=24062, sha256='6ca7bf3202afa8ce34de7627aac26558a7d5b84611d6da1521046fe9a0f76a3c'),
    'preflight_receipt': dict(path=str(H_ROOT / 'finite_point_cache_execution_20261005_v2' / 'runs' / PREFLIGHT_PHASE / 'PARENT_RECEIPT.json'),
        bytes=1073, sha256='8311ac8017b4fb4c01cd46065eaa4e3b7787480815801dd0769d701ba607a185'),
    'preflight_CPU_qualification': dict(path=str(H_ROOT / 'ROOT_QUERY_PAIR_FROZEN_NATIVE_PREFLIGHT_CPU_QUALIFICATION_20261010_01.json'),
        bytes=338510, sha256='4471892c213ff097f2e6a55a33dba47333c29debabd3adb854fe0598ba507fc9'),
    'preflight_source': dict(path=str(C_ROOT / 'query_conditioned_frozen_native_preflight_source_20261010' / 'native_preflight.py'),
        bytes=44920, sha256='e9abe88fae01a4d7f80d374ad2e0bafccb37cd655e3e15dc39e7a64535a1a428'),
    'preflight_CPU_reader': dict(path=str(C_ROOT / 'query_conditioned_frozen_native_preflight_source_20261010' / 'qualify_native_preflight_cpu.py'),
        bytes=65719, sha256='38ad577aa674c7cf4e5fa740b3345b744bf7d920efabbbde9a8ff144ea8b1ddd')}
PREFLIGHT_ADMISSION = dict(path=str(H_ROOT / 'ROOT_QUERY_PAIR_FROZEN_NATIVE_PREFLIGHT_ADMISSION_20261010_01.json'),
    bytes=14217, sha256='dfa4652ba092da811f9ba4621236860e55db8a6a3ec3f1dfee9ceb15e9f1247d')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def expect(row, values, scope):
    """Join emitted metadata; booleans and integer counters keep their types."""
    require(isinstance(row, dict), scope + ' must be an object')
    for key, value in values.items():
        actual = row[key]
        same = actual == value
        if type(value) in (bool, int):
            same = same and type(actual) is type(value)
        require(same, scope + ' differs: ' + key)


def epoch_summaries(plan):
    return [dict(epoch=epoch + 1, microsteps=FIT_IMAGES, updates_this_epoch=(370, 371)[epoch],
        cumulative_updates=(370, 741)[epoch], GT_visits=FIT_GT, seen_dataset_indices=plan['orders'][epoch],
        pending_accumulation=(1, 0)[epoch]) for epoch in range(EPOCHS)]


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
    require(reference(row['path']) == row, 'Pinned reference changed: ' + row['path'])
    return Path(row['path'])


def read_ref(row):
    return read(bound(row))


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode('utf-8')).hexdigest()


def terminal(receipt, phase):
    require(receipt.get('status') == 'ACTUAL_OWNED_TERMINAL' and receipt.get('original_identity') is not None and
        type(receipt.get('returncode')) is int and receipt['returncode'] == 0 and
        receipt.get('root_wait_reaped') is True and receipt.get('held_handle_exit') ==
        dict(wait_state='WAIT_OBJECT_0', actual_exit_code=0) and
        receipt.get('numeric_birth_absence', {}).get('original_birth_absent') is True and
        all(key in receipt and receipt[key] is None for key in ('abort', 'error')) and
        receipt.get('wait_errors') is False and receipt.get('GPU_observation_errors') == [],
        'STOP: actual owned held raw0/reap/original birth absence/noerrors required')
    if 'phase' in receipt:
        require(receipt['phase'] == phase, 'Raw owner receipt phase differs')
    peak = receipt.get('whole_GPU_observed_peak_MiB')
    require(type(peak) in (int, float) and math.isfinite(peak) and 0 <= peak <= 8192 and
        receipt.get('whole_GPU_used_limit_MiB') == 8192 and receipt.get('whole_leaf_VRAM_certified') is False and
        receipt.get('GPU_scope') == 'whole GPU observed, not per-process WDDM allocation',
        'STOP: actual observed wholeGPU <=8192MiB and truthful resource scope required')


def load_source(row, name):
    path = bound(row)
    data = path.read_bytes()
    require(len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256'],
        'Loaded SOURCE bytes differ')
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(data, str(path), 'exec'), module.__dict__)
    return module


def source_gate(spec_path, spec, raw_receipt):
    spec_ref, reader = reference(spec_path), reference(__file__)
    source = reference(Path(__file__).resolve().with_name('train.py'))
    require(Path(spec['GPU_admission_file']).is_absolute(), 'Absolute current admission required')
    admission_ref = reference(spec['GPU_admission_file'])
    admission = read(spec['GPU_admission_file'])
    owner = reference(spec_path.parent / 'finite_point_cache_execution_20261005_v2' / 'run_owned_gpu_limited.py')
    require(raw_receipt['admission_sha256'] == admission_ref['sha256'] and
        owner['sha256'] == admission['driver_sha256'] == OWNER_SHA and
        admission['status'] == 'ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE' and
        admission['automatic_retry'] is False and admission['phase'] == spec['phase'] and
        admission['allowed_cpus'] == spec['CPU_affinity'] == [0, 1] and admission['cuda_visible_devices'] == '0' and
        type(admission['qualified_leaf_deadline_seconds']) is int and
        admission['qualified_leaf_deadline_seconds'] == spec['deadline_seconds'] and
        admission['whole_GPU_used_limit_MiB'] == spec['whole_GPU_owner_limit_MiB'] == 8192 and
        Path(admission['entry']).resolve() == Path(source['path']) and
        admission['entry_args'] == ['--spec', str(spec_path), '--owner-execute'],
        'Current raw terminal/admission/SOURCE/resources join differs')
    require(spec['entry_source'] == source and spec['CPU_reader'] == reader and
        source in spec['sources'] and reader in spec['sources'] and
        admission['bound_source_sha256'].get(str(spec_path)) == spec_ref['sha256'] and
        len({row['path'] for row in spec['sources']}) == len(spec['sources']),
        'New entry/consumer/SPEC and unique complete SOURCE closure must be owner-bound')
    for row in spec['sources']:
        bound(row)
        require(admission['bound_source_sha256'].get(row['path']) == row['sha256'], 'Current admitted SOURCE differs')
    for role in ('preflight_source', 'preflight_CPU_reader', 'readout_source', 'geometry_source', 'hooks_source',
                 'initializer_source', 'micro_preflight_source', 'native_helpers', 'native_preflight_helpers',
                 'reused_CPU_reader', 'evaluation_source', 'training_source', 'training_CPU_reader', 'operator_CPU_fixture_source'):
        require(spec[role] in spec['sources'], 'Pinned runtime SOURCE role absent: ' + role)
    for role, pin in PREFLIGHT_PINS.items():
        require(spec[role] == pin, 'Immutable canonical preflight role differs: ' + role)
    if 'scientific_spec' in admission:
        require(admission['scientific_spec'] == spec_ref, 'Current admitted scientific SPEC differs')
    return dict(spec=spec_ref, source=source, reader=reader, GPU_admission=admission_ref, owner=owner)


def preflight_gate(helper, spec, raw_receipt):
    require(raw_receipt['admission_sha256'] == PREFLIGHT_ADMISSION['sha256'], 'Raw preflight/original admission SHA differs')
    q = read_ref(spec['preflight_CPU_qualification'])
    require(q['schema'] == helper.QUAL_SCHEMA and q['status'] == helper.QUAL_STATUS and
        q['phase'] == spec['preflight_phase'] == PREFLIGHT_PHASE and
        q['spec'] == spec['preflight_training_spec'] and q['source'] == spec['preflight_source'] and
        q['reader_source'] == spec['preflight_CPU_reader'] and q['parent_receipt'] == spec['preflight_receipt'] and
        q['GPU_admission'] == PREFLIGHT_ADMISSION and q['modes'] == list(MODES) and
        q['image_ids'] == [1, 2, 3, 5] and q['actual_trace_records'] == 16 and
        q['actual_native_runtime_technically_qualified'] is True and q['formal_training_qualified'] is False and
        q['eligible_for_formal_resume'] is False and q['AP_computed'] is False and q['efficacy_claimed'] is False and
        q['scientific_method_verdict'] is None,
        'Actual pinned four-mode CPU preflight technical qualification differs; its CP cannot resume training')
    for role in ('source', 'reader_source', 'report', 'start', 'parent_receipt', 'GPU_admission', 'owner_driver'):
        bound(q[role])
    training = read_ref(spec['preflight_training_spec'])
    admission = read_ref(PREFLIGHT_ADMISSION)
    require(training['schema'] == helper.SPEC_SCHEMA and training['phase'] == PREFLIGHT_PHASE and
        training['entry_source'] == q['source'] and training['CPU_reader'] == q['reader_source'] and
        len(training['sources']) == 52 and
        Path(training['GPU_admission_file']).resolve() == Path(PREFLIGHT_ADMISSION['path']).resolve() and
        admission['phase'] == PREFLIGHT_PHASE and admission['driver_sha256'] == OWNER_SHA and
        admission['bound_source_sha256'].get(q['spec']['path']) == q['spec']['sha256'] and
        admission['entry_args'] == ['--spec', q['spec']['path'], '--owner-execute'], 'Original preflight admission/SPEC binding differs')
    for row in training['sources']:
        require(row in spec['sources'] and admission['bound_source_sha256'].get(row['path']) == row['sha256'],
            'Original qualified preflight SOURCE closure not preserved')
        bound(row)
    for role in ('teacher_training_spec', 'teacher_qualification', 'teacher_checkpoint', 'teacher_receipt',
                 'operator_CPU_fixture', 'FIT_annotation'):
        require(training[role] == spec[role] == q[role], 'Preflight/new SHORT2 canonical lineage differs: ' + role)
    for role in ('initializer_source', 'micro_preflight_source', 'native_helpers', 'native_preflight_helpers',
                 'reused_CPU_reader', 'evaluation_source', 'training_source', 'training_CPU_reader',
                 'readout_source', 'geometry_source', 'hooks_source', 'operator_CPU_fixture_source'):
        require(training[role] == spec[role], 'Immutable validated preflight SOURCE role differs: ' + role)
    require(training['image_root'] == spec['image_root'] and q['teacher_full610_exact_before_after_every_arm'] is True and
        q['native451_frozen_and_grads_None'] is True and q['only_two_weights_changed'] is True and
        q['every_AdamW_moment_and_actual_step2_verified'] is True,
        'Original technical native frozen/readout seam qualification incomplete')
    for mode in MODES:
        proof = q['training_proofs'][mode]
        require(proof['actual_microsteps'] == 4 and proof['actual_optimizer_updates'] == 2 and
            proof['full_teacher610_and_all_frozen_states_independently_exact'] is True and
            proof['eligible_for_formal_resume'] is False,
            'Qualified preflight arm counters/scope differ')
    return q


def order_plan(torch, helper, image_ids):
    before = torch.get_rng_state().clone()
    generator = torch.Generator(device='cpu')
    generator.manual_seed(2026100917)
    orders = [torch.randperm(FIT_IMAGES, generator=generator, device='cpu').tolist() for _ in range(EPOCHS)]
    require(torch.equal(before, torch.get_rng_state()) and all(sorted(order) == list(range(FIT_IMAGES)) for order in orders),
        'Independent private CPU Generator order or external RNG differs')
    return dict(schema='query-conditioned-frozen-native-short2-order-plan-v1', seed=2026100917, epochs=2,
        FIT_images=741, FIT_GT=1661, orders=orders, dataset_image_ids=image_ids,
        private_CPU_generator=True, teacher_order_RNG_loaded=False, external_CPU_RNG_restored=True)


def GT_descriptor(torch, np, cv2, coco, old, image, dataset_index, image_root):
    """One raw rasterization per unique image; cache only small digest descriptors."""
    image_id, height, width = int(image['id']), int(image['height']), int(image['width'])
    ann_ids = [int(ann['id']) for ann in coco.imgToAnns[image_id]]
    scale = 800. / max(height, width)
    actual = [int(height * scale + .5), int(width * scale + .5)]
    factors = [actual[1] / width, actual[0] / height]
    masks, boxes = np.zeros((len(ann_ids), 800, 800), dtype=np.uint8), []
    for index, ann_id in enumerate(ann_ids):
        ann = coco.anns[ann_id]
        require(ann['image_id'] == image_id and ann['category_id'] == 1 and not ann.get('ignore', False) and
            not ann.get('iscrowd', 0), 'Complete nonignored original ship GT required')
        x, y, w, h = ann['bbox']
        require(w > 0 and h > 0 and ann.get('area', 0) > 0, 'Original raw GT must survive native parser')
        boxes.append([x, y, x + w, y + h])
        raster = coco.annToMask(ann).astype(np.uint8, copy=False)
        require(raster.shape == (height, width), 'Original COCO raster shape differs')
        masks[index, :actual[0], :actual[1]] = cv2.resize(raster, (actual[1], actual[0]), interpolation=cv2.INTER_NEAREST)
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    boxes *= np.asarray([factors[0], factors[1], factors[0], factors[1]], dtype=np.float32)
    boxes[:, (0, 2)] = boxes[:, (0, 2)].clip(0, actual[1])
    boxes[:, (1, 3)] = boxes[:, (1, 3)].clip(0, actual[0])
    labels = torch.zeros(len(ann_ids), dtype=torch.int64, device='cpu')
    mask = torch.from_numpy(masks)
    native_mask = old.tensor_digest(torch, mask.to(dtype=torch.int64), 'int64')
    target = dict(ann_ids=ann_ids, boxes=boxes.tolist(), bbox=old.tensor_digest(torch,
        torch.from_numpy(np.ascontiguousarray(boxes)), 'float32'), labels=old.tensor_digest(torch, labels, 'int64'),
        masks=old.tensor_digest(torch, mask, 'uint8'), masks_per_GT=[old.tensor_digest(torch, value, 'uint8') for value in mask],
        labels_values=[0] * len(ann_ids), complete_all_raw_annotations=True,
        bboxes_are_original_annotations_not_mask_inference=True)
    return dict(identity=dict(image_id=image_id, dataset_index=dataset_index, original_shape=[height, width],
        actual_img_shape=actual, batch_input_shape=[800, 800], scale_factor=factors, native_input_gamma=1,
        target_gray_resize='nearest', ann_ids=ann_ids, parser_rows=[dict(annotation_id=key, native_parser_retained=True) for key in ann_ids],
        image=reference(Path(image_root) / image['file_name']), native_GT=native_mask), GT=target,
        posterior_GT=dict(mask=native_mask, mask_uint8=target['masks'], labels=target['labels'], GT_count=len(ann_ids),
            complete_all_raw_annotations=True, preprocess_after_head_forward=True))


def compact_hook(helper, row, spec, mode, micro):
    hook = row['native_hook']
    for key, value in dict(forward_calls=1, completed_forwards=1, pixel_decoder_calls=1, coefficient_calls=10,
        passive_coefficient_captures=1, unchanged_early_readouts=9, final_mask_replacements=1, adapter_calls=1,
        apply=True, mode=mode, adapter_prefix=PREFIX, readout_source=spec['readout_source'], geometry_source=spec['geometry_source'],
        hidden_seed=2026100911, permutation_seed=2026101013, metadata_fields_used=['img_shape', 'batch_input_shape'],
        class_list_object_preserved=True, first_nine_mask_objects_preserved=True, native_coefficient_outputs_untouched=True,
        earlier_gate_paths_untouched=True, all_queries_retained=True, adapter_geometry_phi_skipped=False,
        cache_cleared_after_forward=True, registered_state_not_pruned=True, hooks_removed=True,
        hook_handles_installed=4, hook_handles_removed=4, grad_mode_enabled_at_start=True,
        feature_shape=[1, 256, 200, 200], coefficient_shape=[1, 100, 256], final_native_mask_shape=[1, 100, 200, 200]).items():
        require(hook[key] == value, 'Source-observed actual native hook contract differs: ' + key)
    cold = hook['cold_zero_final_weight']
    require(hook['native_graph_properties'] == dict(feature_requires_grad=False, coefficient_requires_grad=False,
        native_final_mask_requires_grad=False, captured_tensors_detached_by_bridge=False, forward_gradient_proof=False) and
        type(cold) is bool and (micro >= 2 or cold is True) and
        hook['cold_native_value_equal'] is (True if cold else None),
        'Actual frozen-native graph or cold/warm hook scope differs')
    checks = row['readout_checks']
    for key, value in dict(canonical_count_equal=True, E_counts_checked=True, global_coefficient_fields_equal=True,
        zero_count_basis='reported_FP32_input_RMS_equals_zero', auxiliary_query_detached=True,
        all_queries_retained=True, input_RMS_normalized=mode != 'raw', global_pool=True,
        dense_readout='same_unnormalized_F', hidden_seed=2026100911, permutation_seed=2026101013,
        permutation_kind='seeded_256_cycle_not_all_derangements_uniform', parameters=393216, trainable_tensors=2,
        empty_E_connected_zero=True, equal_compute_claimed=False, RMS_zero_is_exact_zero_vector_claimed=False,
        input_RMS_finite_nonnegative_checked=True, normalized_nonzero_unit_or_zero_checked=(True if mode != 'raw' else None),
        raw_input_RMS_before_after_exact_checked=(True if mode == 'raw' else None)).items():
        require(checks[key] == value, 'Source-observed compact readout checks differ: ' + key)
    total, eligible, supported = (checks[key] for key in ('eligible_pairs_total', 'eligible_queries', 'supported_queries'))
    require(all(type(v) is int for v in (total, eligible, supported)) and
        0 <= eligible <= supported <= 100 and eligible <= total <= 16 * eligible and
        ((total == 0) == (eligible == 0)), 'Compact E/support/count scalar bounds differ')


def trace_rows(torch, np, cv2, coco, helper, old, spec, result, plan, images, image_ids, GT_cache, mode, P):
    path = bound(result['traces'])
    require(result['trace_records'] == MICROS and result['detailed_microsteps'] == list(DETAILS),
        'Exactly1482 compact rows and first4+last4 detailed scope required')
    activity, visited, consumed, epoch_GT = {name: 0 for name in ACTIVE}, [[], []], 0, [0, 0]
    details, inputs, external, boundary_after, boundary_before, warm_positive = {}, {}, None, None, None, False
    with path.open('r', encoding='utf-8-sig') as stream:
        count = 0
        for micro, line in enumerate(stream):
            require(micro < MICROS and bool(line.strip()), 'Extra/empty actual trace record')
            row = json.loads(line)
            epoch, local = divmod(micro, FIT_IMAGES)
            index = plan['orders'][epoch][local]
            image_id = image_ids[index]
            require(row['mode'] == mode and row['microstep'] == row['global_microstep'] == micro and
                row['epoch'] == epoch + 1 and row['epoch_microstep'] == local and row['dataset_index'] == index and
                row['image_id'] == image_id and row['identity']['dataset_index'] == index and
                row['private_seed'] == 1000000 + micro and row['accumulation_divisor'] == 2 and
                row['optimizer_step'] is bool((micro + 1) % 2 == 0) and row['optimizer_updates'] == (micro + 1) // 2 and
                row['pending_accumulation_before'] == micro % 2 and row['pending_accumulation_after'] == (micro + 1) % 2 and
                row['gradient_clipping_applied'] is bool((micro + 1) % 2 == 0) and
                row['gradients_are_accumulated_preclip'] is True and row['native_parameter_grads_None'] is True and
                row['all_native_modules_eval'] is True and row['hook_cache_clear'] is True and
                row['GT_removed_from_forward'] is True and row['common_realized_matching_or_points_claimed'] is False,
                'Continuous2epoch/allFIT/globalacc2/private seed/native451 frozen trace contract differs')
            if image_id not in GT_cache:
                GT_cache[image_id] = GT_descriptor(torch, np, cv2, coco, old, images[image_id], index, spec['image_root'])
            expected = GT_cache[image_id]
            for key, value in expected['identity'].items():
                require(row['identity'][key] == value, 'Independent raw identity/preprocessing differs: ' + key)
            require(row['GT'] == expected['GT'] and row['posterior_GT'] == expected['posterior_GT'],
                'Independent raw boxes/labels/nearest800 uint8+int64 padded GT descriptor differs')
            helper.digest_contract(row['input'], [1, 3, 800, 800], 'float32')
            if image_id in inputs:
                require(inputs[image_id] == row['input'], 'Repeated no-augmentation input digest changed')
            inputs[image_id] = row['input']
            require(row['native_loss_only'] is True and row['Sobel_added'] is False and row['auxiliary_loss_added'] is False,
                'Only original30 native loss atoms are permitted')
            old.validate_losses(dict(row, loss_is_native_model_loss=True), 'mask2former')
            helper.check_matches(torch, old, row['matches'], expected['GT']['ann_ids'])
            compact_hook(helper, row, spec, mode, micro)
            detailed = micro in DETAILS
            require(row['detailed_observation'] is detailed, 'Detailed observation scope must be exactlyfirst4+last4')
            if detailed:
                require(row['readout_observation'] == row['native_hook']['readout_observation'], 'Detailed readout/hook packet differs')
                # Reuse the validated cold/warm helper only with its actual
                # zero-final predicate; an empty-E update may leave finalW0.
                phase_micro = 0 if row['native_hook']['cold_zero_final_weight'] else 2
                helper.hook_contract(row['native_hook'], spec, mode, phase_micro, P, torch)
                helper.check_fields(dict(row, microstep=phase_micro))
                packet = row['readout_observation']
                counts = torch.tensor(packet['eligible_counts'], dtype=torch.int64, device='cpu')
                mass = torch.tensor(packet['rho_mass'], dtype=torch.float32, device='cpu')
                require(row['readout_checks']['eligible_pairs_total'] == int(counts.sum()) and
                    row['readout_checks']['eligible_queries'] == int((counts > 0).sum()) and
                    row['readout_checks']['supported_queries'] == int((mass > 1e-12).sum()),
                    'Detailed packet independently reproduced compact scalar counts')
                h, w = expected['identity']['actual_img_shape']
                axes = torch.arange(200, dtype=torch.float32, device='cpu')
                valid = ((h - 4 * axes).clamp(0, 4) / 4).reshape(1, 1, 200, 1) * \
                    ((w - 4 * axes).clamp(0, 4) / 4).reshape(1, 1, 1, 200)
                require(old.tensor_digest(torch, valid, 'float32') == row['bridge_inputs']['valid_fraction'],
                    'Detailed metadata-only valid fraction differs')
                details[micro] = {key: row[key] for key in ('image_id', 'input', 'readout_observation', 'head_fields', 'bridge_inputs')}
            else:
                require(row['readout_observation'] is None and row['native_hook']['readout_observation'] is None and
                    row['head_fields'] is None and row['bridge_inputs'] is None,
                    'Dense native field digests/full E-RMS packets must not imply observations outside eight micros')
            norms = row['gradient_norms_aligned_active_names']
            require(isinstance(norms, list) and len(norms) == 2 and all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in norms),
                'Actual finite accumulated2W gradient norms required')
            require(row['cold_hidden_gradient_exact_zero'] is (True if micro < 2 else None) and
                (micro >= 2 or norms[0] == 0), 'Actual cold count_nonzero/gradient proof differs')
            warm_positive = warm_positive or (micro >= 2 and all(v > 0 for v in norms))
            for name, norm in zip(ACTIVE, norms):
                activity[name] += int(norm > 0)
            if micro % 2:
                old.finite(row['preclip_global_norm'], 'Actual2W global preclip norm', True)
            else:
                require(row['preclip_global_norm'] is None, 'Non-update microstep reported clipping')
            old.validate_rng(torch, row)
            if external is None:
                external = row['external_RNG_before']
            require(row['external_RNG_before'] == external, 'Private loss seed leaked into restored external RNG')
            before, after = row['accumulation_gradient_before'], row['accumulation_gradient_after']
            if micro == 740:
                require(before is None and isinstance(after, list) and len(after) == 2, 'Odd epoch-end pending2W gradient digests required')
                boundary_after = after
            elif micro == 741:
                require(after is None and isinstance(before, list) and len(before) == 2 and before == boundary_after,
                    'Odd741 pending gradient tensors changed/flushed before first next-epoch forward')
                boundary_before = before
            else:
                require(before is None and after is None, 'Unexpected accumulation boundary digest packet')
            if micro in (740, 741):
                for digest, shape in zip(after if micro == 740 else before, ([512, 512], [256, 512])):
                    helper.digest_contract(digest, shape, 'float32')
            GT_count = len(expected['GT']['ann_ids'])
            consumed += GT_count
            epoch_GT[epoch] += GT_count
            visited[epoch].append(index)
            count += 1
    require(count == MICROS and consumed == GT_VISITS and epoch_GT == [FIT_GT, FIT_GT] and
        visited == plan['orders'] and boundary_before == boundary_after and boundary_after is not None and
        set(details) == set(DETAILS) and len(inputs) == FIT_IMAGES and warm_positive and all(v > 0 for v in activity.values()),
        'Complete2epoch/741each/3322rawGT/odd-boundary/warm2W/detailed-scope proof differs')
    return dict(rows=count, raw_GT_visits=consumed, epoch_raw_GT=epoch_GT, positive=activity,
        last_external_RNG=external, boundary_gradients=boundary_after, details=details, inputs=inputs,
        epoch_boundary_accumulation_preserved=True, all_micro_native_features_or_geometry_equality_claimed=False)


def qualify_arm(torch, np, cv2, coco, helper, old, teacher_reader, spec, result, model, teacher,
                readout, geometry, hooks, direct, P, plan, plan_ref, images, image_ids, GT_cache, mode, execution):
    summaries = epoch_summaries(plan)
    expect(result, dict(mode=mode, plan=plan_ref, epochs=2, completed_epoch=2, completed_epochs=2,
        global_microsteps=MICROS, global_updates=UPDATES, consumed_raw_GT=GT_VISITS,
        FIT_images=FIT_IMAGES, FIT_GT=FIT_GT, detailed_microsteps=list(DETAILS), epoch_summaries=summaries,
        native610_exact=True, frozen_native=True, native_loss_only=True, AP_computed=False,
        eligible_for_formal_resume=False, no_pending_accumulation_gradients=True), 'Completed fresh arm result')
    directory = Path(spec['output_directory']) / (mode + '_training')
    for role, name in (('checkpoint', 'NODE_E2.pth'), ('state_proof', 'TRAIN_STATE_PROOF.json'), ('traces', 'TRACE.jsonl')):
        require(Path(result[role]['path']).resolve() == (directory / name).resolve(), 'Actual fresh arm artifact path differs: ' + role)
    adapter = hooks.register_query_pair_readout(torch, readout, geometry, model, mode,
        spec['readout_source'], spec['geometry_source'], hidden_seed=2026100911, permutation_seed=2026101013)
    require(all(torch.equal(adapter.state_dict()[name], value) for name, value in direct.items()) and
        set(adapter.state_dict()) == set(direct) and not list(adapter.buffers()) and list(adapter.permutation_indices) == P,
        'Fresh independent direct2W/393216/seeded one256cycle factory differs')
    policy = helper.independent_frozen_policy(torch, model)
    initial = {name: value.detach().clone() for name, value in model.state_dict().items()}
    require(set(initial) == set(teacher) | set(ACTIVE) and all(torch.equal(initial[name], value) for name, value in teacher.items()),
        'Registration altered same immutable E24 native610 teacher')
    training = read_ref(spec['teacher_training_spec'])
    groups, audit = old.independent_optimizer(torch, model, training['initializers']['mask2former'], policy)
    recipe = audit['recipe']
    require(audit['type'] == 'AdamW' and recipe['lr'] == 0.0001 and recipe['weight_decay'] == 0.05 and
        tuple(recipe.get('betas', (0.9, 0.999))) == (0.9, 0.999) and recipe.get('eps', 1e-8) == 1e-8 and
        audit['clip_grad'] == dict(max_norm=0.01, norm_type=2) and
        [row['name'] for row in audit['parameter_groups']] == ACTIVE and
        all(row['lr'] == 0.0001 and row['weight_decay'] == 0.05 for row in audit['parameter_groups']) and
        len(groups) == 2 and all(group['lr'] == 0.0001 and group['weight_decay'] == 0.05 and
            tuple(group['betas']) == (0.9, 0.999) and group['eps'] == 1e-8 for group in groups),
        'Fresh2W constant native AdamW defaults/groups/.01clip differ')
    trace = trace_rows(torch, np, cv2, coco, helper, old, spec, result, plan, images, image_ids, GT_cache, mode, P)
    cp = torch.load(str(bound(result['checkpoint'])), map_location='cpu')
    require(set(cp) == {'state_dict', 'adapter_state_dict', 'optimizer_state_dict', 'external_rng_state', 'meta'},
        'Complete fresh612-state/2W/optimizer/RNG endpoint required')
    final, meta = cp['state_dict'], cp['meta']
    proof = read_ref(result['state_proof'])
    require(proof['schema'] == PROOF_SCHEMA and proof['mode'] == mode and proof['policy'] == policy and
        json_sha(proof['optimizer']) == json_sha(audit),
        'Independent frozen native451/typed BN106+159/two-weight optimizer proof differs')
    old.check_state(torch, teacher, proof['native_teacher610'], 'immutable full E24 teacher610 baseline')
    old.check_state(torch, direct, proof['cold_adapter'], 'fresh direct cold2W seed2026100911')
    expect(proof['teacher_loading'], dict(mode='native0', checkpoint=spec['teacher_checkpoint'],
        qualification=spec['teacher_qualification'], CP_qualified=True, strict_fullstate_loaded=True,
        native_state_tensors=610, adapter_state_tensors=0, registered_parameters=451, original_frozen_BN_exact=True,
        inference_backbone_with_cp=False, training_backbone_with_cp=training['backbone_with_cp'], no_random_missing_keys=True,
        BN_parameter_names=policy['BN_parameter_names'], BN_buffer_names=policy['BN_buffer_names']),
        'Actual strict immutable teacher loading proof')
    old.check_state(torch, teacher, proof['teacher_loading']['trained_fullstate'], 'actual strict loader teacher identity')
    old.check_state(torch, initial, proof['before'], 'same teacher610 and fresh cold2W before SHORT2')
    old.check_state(torch, final, proof['after'], 'actual full612 SHORT2 CPU endpoint')
    require(set(final) == set(initial) and all(isinstance(value, torch.Tensor) and value.device.type == 'cpu' and
        value.shape == initial[name].shape and value.dtype == initial[name].dtype and bool(torch.isfinite(value).all())
        for name, value in final.items()), 'Final endpoint keys/shape/dtype/CPU/finite differ')
    changed = {name: not torch.equal(value, initial[name]) for name, value in final.items()}
    require(proof['changed'] == changed and {name for name, value in changed.items() if value} == set(ACTIVE) and
        all(torch.equal(final[name], value) for name, value in teacher.items()) and
        all(torch.equal(final[name], initial[name]) for name in policy['BN_parameter_names'] + policy['BN_buffer_names']),
        'Only new2W may change; allteacher610/typedBN106+159 must remain exact')
    require(set(cp['adapter_state_dict']) == set(direct) and all(torch.equal(value, final[PREFIX + name])
        for name, value in cp['adapter_state_dict'].items()), 'Compact2W alias differs from full612 endpoint')
    model.load_state_dict(final, strict=True)
    require(all(torch.equal(value, final[name]) for name, value in model.state_dict().items()), 'Strict full612 CPU endpoint readback differs')
    counts = {name: UPDATES for name in ACTIVE}
    optimizer = old.validate_optimizer(torch, cp, groups, audit, counts)
    actual_optimizer = teacher_reader.optimizer_snapshot(torch, old, cp['optimizer_state_dict'], [[name] for name in ACTIVE])
    require(proof['actual_final_optimizer'] == actual_optimizer and proof['positive_gradient_microcounts'] == trace['positive'] and
        proof['adapter_names'] == ACTIVE and proof['optimizer_step_nonNone_gradient_counts'] == counts,
        'Every actual AdamW moment tensor/group/step741/activity identity differs')
    for key, value in dict(native610_exact=True, native_grad_None=True, all_native_modules_eval=True,
        hook_cache_clear=True, frozen_BN_all_exact=True, CPU_checkpoint_readback_exact=True,
        GPU_to_CPU_readback_state_exact=True, all_adapter_weights_positive_and_changed=True,
        warm_both_weights_positive=True, fresh_optimizer_state_entries=0, teacher_optimizer_loaded=False,
        teacher_order_RNG_loaded=False, teacher_external_RNG_loaded=False, preflight_checkpoint_loaded=False,
        epoch_boundary_accumulation_preserved=True, no_pending_accumulation_gradients=True,
        completed_epochs=2, global_microsteps=MICROS, global_updates=UPDATES, consumed_raw_GT=GT_VISITS,
        native_grad_None_microchecks=MICROS, cold_constructor_repeated_exact=True, cold_constructor_RNG_restored=True,
        fresh_identical_adapter_initialization_across_modes=True, boundary_gradient_digest=trace['boundary_gradients'],
        epoch_summaries=summaries, detailed_microsteps=list(DETAILS), common_realized_matching_or_points_claimed=False).items():
        require(proof[key] == value, 'Actual final state/freshness/odd-boundary source proof differs: ' + key)
    for key, value in dict(schema=CP_SCHEMA, kind='mask2former', mode=mode, phase=spec['phase'], source=execution['source'],
        spec=execution['spec'], policy=policy, teacher_checkpoint=spec['teacher_checkpoint'], teacher_qualification=spec['teacher_qualification'],
        teacher_training_spec=spec['teacher_training_spec'], teacher_receipt=spec['teacher_receipt'], teacher_mode='native0', teacher_epoch=24,
        readout_source=spec['readout_source'], geometry_source=spec['geometry_source'], hooks_source=spec['hooks_source'],
        plan=plan_ref, completed_epoch=2, completed_epochs=2, initial_epoch=0, epochs_this_run=2, total_epochs=2,
        global_microsteps=MICROS, global_updates=UPDATES, consumed_raw_GT=GT_VISITS, FIT_images=FIT_IMAGES, FIT_GT=FIT_GT,
        batch_size=1, accumulation=2, frozen_native=True, native610_exact=True, native_loss_only=True, native_loss_atoms=30,
        no_Sobel_or_auxiliary_loss=True, no_AP=True, eligible_for_formal_resume=False, no_pending_accumulation_gradients=True,
        teacher_optimizer_loaded=False, teacher_order_RNG_loaded=False, teacher_external_RNG_loaded=False,
        preflight_checkpoint_loaded=False, LR_decay=False, preflight_CPU_qualification=spec['preflight_CPU_qualification'],
        preflight_training_spec=spec['preflight_training_spec'], fresh_optimizer_state_entries=0, resume_checkpoint=None,
        resume_qualification=None, no_scheduler=True, no_warmup=True, no_feature_cache=True,
        short_budget_hypothesis_probe=True, AP_computed=False).items():
        require(meta[key] == value, 'Actual SHORT2 complete endpoint metadata differs: ' + key)
    require(json_sha(meta['optimizer']) == json_sha(audit), 'Endpoint optimizer audit differs')
    rng = teacher_reader.check_rng_tensors(torch, old, cp['external_rng_state'])
    require(rng == trace['last_external_RNG'], 'Endpoint stored external CPU/CUDA RNG identity differs')
    out = dict(mode=mode, checkpoint=result['checkpoint'], state_proof=result['state_proof'], traces=result['traces'],
        actual_epochs=2, actual_microsteps=MICROS, actual_optimizer_updates=UPDATES, raw_GT_visits=GT_VISITS,
        full_teacher610_and_BN106_affine159_buffers_exact=True, full612_strict_CPU_readback=True,
        only_fresh_two_weights_changed=True, parameter_tensors=453, active_parameter_tensors=2, active_scalars=393216,
        trainability=policy, positive_gradient_microcounts=trace['positive'], optimizer=optimizer, actual_final_optimizer=actual_optimizer,
        odd741_epoch_boundary_pending_gradient_digests=trace['boundary_gradients'], epoch_boundary_accumulation_preserved=True,
        all741_raw_GT_descriptors_independently_reconstructed=True, raw_GT_unique_per_epoch=1661,
        native30_total_consistent=True, ten_native_assignments_retain_all_raw_GT=True, details_microsteps=list(DETAILS),
        eight_detailed_E_RMS_native_observations_independently_checked=True, all_micro_FE_or_realized_points_equality_claimed=False,
        native_grad_None_and_hooks_source_observed=True, GPU_forward_backward_or_PCA_CPU_replayed=False,
        formal_training_qualified=False, eligible_for_formal_resume=False, AP_computed=False, efficacy_claimed=False)
    del cp, final, initial, adapter
    gc.collect()
    require(not torch.cuda.is_initialized(), 'CPU arm audit initialized CUDA')
    return out, trace


def qualify(spec_file, acceptance_file):
    spec_path, target = Path(spec_file), Path(acceptance_file)
    require(spec_path.is_absolute() and target.is_absolute() and target.parent.is_dir() and not target.exists(),
        'Absolute SPEC and fresh acceptance with existing parent required')
    spec_path = spec_path.resolve(strict=True)
    spec = read(spec_path)
    expect(spec, dict(schema=SCHEMA, modes=list(MODES), epochs=2, training_all_FIT=True,
        preflight_image_ids=[1, 2, 3, 5], FIT_images=FIT_IMAGES, FIT_GT=FIT_GT,
        total_microsteps=MICROS, total_updates=UPDATES, order_seed=2026100917, loss_seed_base=1000000,
        hidden_seed=2026100911, permutation_seed=2026101013, expected_epoch=24,
        accumulation=2, learning_rate=0.0001, weight_decay=0.05, clip_grad=dict(max_norm=0.01, norm_type=2),
        checkpoint_epoch=2, detail_microsteps=list(DETAILS), CPU_affinity=[0, 1], allocator_limit_MiB=4096,
        whole_GPU_owner_limit_MiB=8192, deadline_seconds=10800, preflight_phase=PREFLIGHT_PHASE),
        'Exact fresh4arm/allFIT SHORT2 resources/order/seeds/scope')
    require(isinstance(spec['phase'], str) and spec['phase'].startswith('query_conditioned_frozen_native_short2_') and
        Path(spec['phase']).name == spec['phase'] and all(Path(spec[role]['path']).is_absolute()
        for role in ('teacher_receipt', 'preflight_receipt')), 'Safe current phase and absolute prior raw receipts required')
    receipt_path = spec_path.parent / 'finite_point_cache_execution_20261005_v2' / 'runs' / spec['phase'] / 'PARENT_RECEIPT.json'
    raw = read(receipt_path)
    terminal(raw, spec['phase'])
    teacher_raw = read(spec['teacher_receipt']['path'])
    terminal(teacher_raw, spec['teacher_phase'])
    preflight_raw = read(spec['preflight_receipt']['path'])
    terminal(preflight_raw, spec['preflight_phase'])
    # Raw current first, then teacher/preflight terminals above precede ALL
    # SOURCE/SPEC/scientific artifact hashes, Q/CP access and scientific imports.
    execution = source_gate(spec_path, spec, raw)
    helper = load_source(spec['preflight_CPU_reader'], 'scientifically_validated_native_CPU_helpers')
    for role in helper.ARTIFACT_PINS:
        helper.pinned_role(spec, role)
    for role, pin in helper.SOURCE_PINS.items():
        require((spec[role]['bytes'], spec[role]['sha256']) == pin, 'Exact original validated SOURCE role differs: ' + role)
    for role, path in helper.SOURCE_PATHS.items():
        require(Path(spec[role]['path']).resolve() == path.resolve(), 'Canonical validated SOURCE path differs: ' + role)
    preflight_gate(helper, spec, preflight_raw)
    training, teacher_q = helper.teacher_lineage(spec, teacher_raw)
    helper.fixture_gate(spec)
    output = Path(spec['output_directory'])
    require(output.is_absolute(), 'Absolute actual SHORT2 output directory required')
    report_path, start_path, plan_path, aggregate_path = (output / name for name in
        ('REPORT.json', 'START.json', 'PLAN.json', 'QUALIFIED.json'))
    report = read(report_path)
    plan_ref, report_ref = reference(plan_path), reference(report_path)
    expect(report, dict(schema=REPORT_SCHEMA, phase=spec['phase'], source=execution['source'], spec=execution['spec'],
        status='SHORT2_QUERY_CONDITIONED_FROZEN_NATIVE_COMPLETE_NO_METHOD_RESULT', error=None, modes=list(MODES), plan=plan_ref,
        teacher_checkpoint=spec['teacher_checkpoint'], teacher_qualification=spec['teacher_qualification'],
        teacher_receipt=spec['teacher_receipt'], preflight_CPU_qualification=spec['preflight_CPU_qualification'],
        preflight_training_spec=spec['preflight_training_spec'], training_all_FIT=True, epochs=2,
        FIT_images=FIT_IMAGES, FIT_GT=FIT_GT, global_microsteps_per_arm=MICROS, global_updates_per_arm=UPDATES,
        consumed_raw_GT_per_arm=GT_VISITS, detailed_microsteps=list(DETAILS), native610_exact=True, frozen_native=True,
        native_loss_only=True, native_loss_atoms=30, all_four_CPU_models_before_CUDA=True, only_one_model_on_CUDA_at_a_time=True,
        teacher_optimizer_loaded=False, teacher_order_RNG_loaded=False, teacher_external_RNG_loaded=False,
        preflight_checkpoint_loaded=False, no_scheduler=True, no_warmup=True, no_feature_cache=True,
        epoch_boundary_accumulation_preserved=True, no_AP=True, no_DEV=True, no_TEST=True, AP_computed=False,
        efficacy_claimed=False, formal_resume_eligible=False, scientific_method_decision_computed=False,
        common_realized_matching_or_points_claimed=False, native_field_equality_checked_only_at_detailed_microsteps=True,
        allocator_limit_MiB=4096, GPU_scope='whole GPU must be verified from actual owner receipt'),
        'Actual completed four-arm report/source/scope')
    start = read(start_path)
    expect(start, dict(schema='query-conditioned-frozen-native-short2-start-v1', phase=spec['phase'], source=execution['source'],
        spec=execution['spec'], modes=list(MODES), teacher_checkpoint=spec['teacher_checkpoint'],
        teacher_qualification=spec['teacher_qualification'], teacher_receipt=spec['teacher_receipt'],
        preflight_CPU_qualification=spec['preflight_CPU_qualification'], preflight_training_spec=spec['preflight_training_spec'],
        training_all_FIT=True, epochs=2, FIT_images=FIT_IMAGES, FIT_GT=FIT_GT, total_microsteps=MICROS, total_updates=UPDATES,
        accumulation=2, order_seed=2026100917, loss_seed_base=1000000, detail_microsteps=list(DETAILS),
        teacher_terminal_gate_before_scientific_access=True, preflight_terminal_gate_before_scientific_access=True,
        all_four_CPU_models_before_CUDA_required=True, fresh_optimizer_state_entries=0, teacher_optimizer_loaded=False,
        teacher_order_RNG_loaded=False, teacher_external_RNG_loaded=False, preflight_checkpoint_loaded=False,
        no_scheduler=True, no_warmup=True, no_feature_cache=True, no_formal_resume=True,
        no_DEV=True, no_TEST=True, no_AP=True, no_retry=True), 'Actual planned START lineage/scope/freshness')
    mode_rows = report['mode_rows']
    expect(read(aggregate_path), dict(schema='query-conditioned-frozen-native-short2-aggregate-v1',
        status='ROOT_QUERY_CONDITIONED_FROZEN_NATIVE_SHORT2_PENDING_CPU_READER', source=execution['source'], spec=execution['spec'],
        report=report_ref, plan=plan_ref, mode_rows=mode_rows, teacher_checkpoint=spec['teacher_checkpoint'],
        preflight_CPU_qualification=spec['preflight_CPU_qualification'], independently_CPU_qualified=False,
        efficacy_claimed=False, AP_computed=False, formal_resume_eligible=False), 'Actual pending aggregate/report/arm joins')
    require(set(mode_rows) == set(report['cleanup']) == set(MODES), 'Allfour fresh completed arms and actual cleanup required')
    for row in report['cleanup'].values():
        require(type(row['allocated_before']) is int and row['allocated_before'] >= 0 and row['allocated_after'] == 0,
            'Actual per-arm GPU allocated cleanup must be exact0')
    require(type(report['torch_peak_allocated_bytes']) is int and 0 <= report['torch_peak_allocated_bytes'] <= 4096 * 1024 ** 2 and
        type(report['elapsed_seconds']) in (int, float) and math.isfinite(report['elapsed_seconds']) and
        0 <= report['elapsed_seconds'] < spec['deadline_seconds'], 'Actual allocator/wall resource limit differs')
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
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    cv2.setNumThreads(2)
    require(psutil.Process().cpu_affinity() == [0, 1] and not torch.cuda.is_initialized(), 'CPU01/no initialized CUDA required')
    register_all_modules(init_default_scope=False)
    init_default_scope('mmdet')
    require(len(CocoDataset.METAINFO['classes']) == 80 and CocoDataset.METAINFO['classes'][8] == 'boat', 'Official COCO80 boat8 order differs')
    old = helper.load_source(spec['reused_CPU_reader'], 'qualified_original_native_GT_loss_optimizer_CPU_helpers')
    teacher_reader = helper.load_source(spec['training_CPU_reader'], 'qualified_full24_native_CPU_helpers')
    initializer = helper.load_source(spec['initializer_source'], 'qualified_official_initializer_CPU_helpers')
    geometry = helper.load_source(spec['geometry_source'], 'sealed_original_geometry_CPU_constructor')
    readout = helper.load_source(spec['readout_source'], 'sealed_query_pair_operator_CPU_constructor')
    hooks = helper.load_source(spec['hooks_source'], 'sealed_query_pair_whole_head_CPU_registration')
    fit = helper.read_ref(spec['FIT_annotation'])
    images = {int(row['id']): row for row in fit['images']}
    image_ids = [int(row['id']) for row in fit['images']]
    require(len(images) == len(image_ids) == FIT_IMAGES and len(fit['annotations']) == FIT_GT and
        len({int(ann['id']) for ann in fit['annotations']}) == FIT_GT, 'Complete pinned native FIT741/1661 population differs')
    coco = COCO(str(helper.bound(spec['FIT_annotation'])))
    rng_before = torch.get_rng_state().clone()
    plan = order_plan(torch, helper, image_ids)
    require(read_ref(plan_ref) == plan, 'Actual2epoch private TorchGenerator order/complete FIT input mapping differs')
    model, teacher = helper.load_teacher(torch, old, teacher_reader, initializer, MODELS,
        revert_sync_batchnorm, Config, spec, training, teacher_q)
    direct, P = helper.independent_adapter(torch)
    proofs, traces, GT_cache = {}, {}, {}
    import copy
    for mode in MODES:
        arm = copy.deepcopy(model)
        proofs[mode], traces[mode] = qualify_arm(torch, np, cv2, coco, helper, old, teacher_reader, spec, mode_rows[mode],
            arm, teacher, readout, geometry, hooks, direct, P, plan, plan_ref, images, image_ids, GT_cache, mode, execution)
        del arm
        gc.collect()
    require(len(GT_cache) == FIT_IMAGES and sum(len(row['GT']['ann_ids']) for row in GT_cache.values()) == FIT_GT,
        'Exactly741 independently reconstructed GT descriptors and1661 unique raw GT required')
    first = traces[MODES[0]]
    for mode in MODES[1:]:
        require(traces[mode]['inputs'] == first['inputs'], 'Same original no-augmentation inputs differ across arms')
        for micro in DETAILS:
            a, b = first['details'][micro], traces[mode]['details'][micro]
            require(a['image_id'] == b['image_id'] and a['input'] == b['input'] and
                a['head_fields']['classes'] == b['head_fields']['classes'] and
                a['head_fields']['native_masks'] == b['head_fields']['native_masks'] and
                a['bridge_inputs'] == b['bridge_inputs'], 'Eight detailed frozen-native field/F/e0/valid digests differ across arms')
            for key in ('eligible', 'eligible_counts', 'canonical_eligible_counts', 'rho_mass', 'support_nonempty'):
                require(a['readout_observation'][key] == b['readout_observation'][key],
                    'Eight detailed held-fixed detached support packets differ: ' + key)
    require(torch.equal(rng_before, torch.get_rng_state()) and not torch.cuda.is_initialized() and
        torch.get_num_threads() == torch.get_num_interop_threads() == 2 and psutil.Process().cpu_affinity() == [0, 1],
        'CPU qualification changed external RNG, resources or initialized CUDA')
    acceptance = dict(schema=QUAL_SCHEMA, status=QUAL_STATUS, phase=spec['phase'], source=execution['source'],
        reader_source=execution['reader'], spec=execution['spec'], report=report_ref, start=reference(start_path),
        GPU_qualification=reference(aggregate_path), plan=plan_ref, parent_receipt=reference(receipt_path),
        GPU_admission=execution['GPU_admission'], owner_driver=execution['owner'],
        preflight_CPU_qualification=spec['preflight_CPU_qualification'], preflight_source=spec['preflight_source'],
        preflight_CPU_reader=spec['preflight_CPU_reader'], preflight_receipt=spec['preflight_receipt'],
        preflight_training_spec=spec['preflight_training_spec'],
        teacher_checkpoint=spec['teacher_checkpoint'], teacher_qualification=spec['teacher_qualification'],
        teacher_training_spec=spec['teacher_training_spec'], teacher_receipt=spec['teacher_receipt'], FIT_annotation=spec['FIT_annotation'],
        modes=list(MODES), training_proofs=proofs, checkpointmap={mode: proof['checkpoint'] for mode, proof in proofs.items()},
        actual_epochs_per_arm=2, actual_microsteps_per_arm=MICROS, actual_optimizer_updates_per_arm=UPDATES,
        actual_raw_GT_visits_per_arm=GT_VISITS, total_trace_records=4 * MICROS, GT_descriptor_cache_unique_images=FIT_IMAGES,
        GT_descriptors_independently_reconstructed_once_per_unique_image=True, disk_feature_cache_used=False,
        private_CPU_generator_order_independently_reproduced=True, common_input_and_private_loss_seed_sequence_verified=True,
        first4_last4_detailed_microsteps=list(DETAILS), detailed_native_fields_support_packets_equal_across_modes=True,
        all_micro_FE_geometry_equality_or_common_matching_points_claimed=False, all_teacher610_states_and_typed_BN_exact=True,
        only_fresh2W_changed_and_all_AdamW_moments_step741_verified=True, native30_and_all10_assignment_GT_coverage_verified=True,
        odd741_epoch_boundary_actual_pending_gradient_digest_identity_verified=True, no_epoch_gradient_flush=True,
        teacher_optimizer_order_RNG_or_preflight_CP_restored=False, raw_current_teacher_preflight_gates_before_scientific_access=True,
        complete_SOURCE_closure_verified_before_scientific_import=True, CPU_affinity=[0, 1], torch_threads=2,
        torch_interop_threads=2, CUDA_initialized=False, external_CPU_RNG_restored=True,
        formal_training_qualified=False, eligible_for_formal_resume=False, AP_computed=False, efficacy_claimed=False,
        scientific_method_verdict=None, GPU_forward_backward_or_PCA_CPU_replayed=False,
        evidence_limits=[
            'Four fresh frozen-native two-epoch fixed-budget arms only; no AP, efficacy, convergence, novelty or external superiority conclusion.',
            'CPU reconstructs raw GT/order/cold weights and verifies full native/endpoints plus all AdamW storage; it does not replay CUDA forward/backward, gradient elements or Hungarian costs.',
            'Actual native gradNone, safe hooks and scalar readout checks outside the eight detailed micros are bound source-observed runtime evidence, not independently recomputed feature or geometry packets.',
            'Only first4+last4 actual native F/e0/class/mask and E/RMS packets are independently joined; this does not assert all-micro realized feature/geometry equality or loss-sampled point/matching equality.',
            'FP32 reported zero input RMS does not certify exactly zero channel vectors.',
            'The odd741 boundary is checked through global counters and equal actual pending gradient digests before the next forward; the optimization trajectory is not reproduced.',
            'GT descriptor caching retains independently checked small digests only and never weakens raw annotation coverage or creates a disk feature cache.',
            'Preflight checkpoints, teacher optimizer/order/RNG states and automatic/formal resume remain ineligible.'])
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
    require(args.owner_execute, 'SOURCE_ONLY: explicit ROOT CPU execution after actual owned terminal gates required')
    qualify(args.spec, args.acceptance)


if __name__ == '__main__':
    main()
