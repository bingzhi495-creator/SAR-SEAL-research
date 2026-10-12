"""Inert independent CPU reader of fixed teacher plus four SHORT2 DEV endpoints.

SOURCE preparation performs no scientific imports/execution or scientific
payload access. ROOT executes only after actual current evaluation, teacher,
preflight and SHORT2 owned raw held0/reap/birth-absence/whole-GPU gates. The
reader verifies canonical qualified checkpoints, original frozen610/typed BN,
all187 DEV images/380 raw GT, all100 RLE/query populations, class packets and
score arithmetic. Official COCO12 and reusable best100 GT geometry are then
independently recomputed. Omitted pixel logits prevent independent recreation
of fusion thresholds and mask-quality means; official fusion tensor equality
is source-observed runtime evidence. DEV is historically exposed development
data. No blind TEST, convergence, external SOTA or overall efficacy is claimed.
"""
import argparse
import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import types

SPEC_SCHEMA = 'ROOT-query-conditioned-frozen-native-short2-evaluation-spec-v1'
REPORT_SCHEMA = 'query-conditioned-frozen-native-short2-evaluation-report-v1'
STATE_SCHEMA = 'query-conditioned-frozen-native-short2-evaluation-state-proof-v1'
QUAL_SCHEMA = 'ROOT-query-conditioned-frozen-native-short2-evaluation-CPU-qualification-v1'
QUAL_STATUS = 'ROOT_QUERY_CONDITIONED_FROZEN_NATIVE_SHORT2_DEV_CPU_EVALUATED'
MODES = ('raw', 'raw_rms', 'query_rms', 'permuted_query_rms')
ENDPOINTS = ('native0',) + MODES
PREFIX = 'panoptic_head.query_pair_readout.'
ACTIVE = [PREFIX + 'hidden.weight', PREFIX + 'final.weight']
OWNER_SHA = 'ef9a9f17d4fb1bc3464936095ef799021dd880f8b15488d558a871c08b6e1924'
C_ROOT = Path(__file__).resolve().parent.parent
H_ROOT = C_ROOT.parent / 'goal_continuation_20261005'
SHORT2_PHASE = 'query_conditioned_frozen_native_short2_20261010_01'
SHORT2_SPEC_PIN = dict(path=str(H_ROOT / 'ROOT_QUERY_PAIR_FROZEN_NATIVE_SHORT2_SPEC_20261010_01.json'),
    bytes=27994, sha256='b72a91894d4ee9e82fbb49bc6cfbcb91a27eaf053dac7aeeae97cac4c7b20393')
SHORT2_ADMISSION_PIN = dict(path=str(H_ROOT / 'ROOT_QUERY_PAIR_FROZEN_NATIVE_SHORT2_ADMISSION_20261010_01.json'),
    bytes=15411, sha256='a57cc973eb7ed09e670589e74a57f52d0110654eb6c0e3356369583f457eea9e')
SOURCE_PINS = {
    'preflight_source': (44920, 'e9abe88fae01a4d7f80d374ad2e0bafccb37cd655e3e15dc39e7a64535a1a428',
        C_ROOT / 'query_conditioned_frozen_native_preflight_source_20261010' / 'native_preflight.py'),
    'preflight_CPU_reader': (65719, '38ad577aa674c7cf4e5fa740b3345b744bf7d920efabbbde9a8ff144ea8b1ddd',
        C_ROOT / 'query_conditioned_frozen_native_preflight_source_20261010' / 'qualify_native_preflight_cpu.py'),
    'short2_source': (48366, '16f838776edf3a6d24730dc283a7c98239b2c365e4a513d731ec4ead05d68548',
        C_ROOT / 'query_conditioned_frozen_native_short2_source_20261010' / 'train.py'),
    'short2_CPU_reader': (56265, '1d2c2c06c358db731556094f68a35ee7d70a92bb43464350200b396412bb0cfc',
        C_ROOT / 'query_conditioned_frozen_native_short2_source_20261010' / 'qualify_cpu.py'),
    'official_evaluation_source': (40816, 'b4ca717bd2018ea8404808c03510f830081a6660c893eda04fce72d7adb43a48',
        C_ROOT / 'spatial_position_preserving_common24_evaluation_source_20261009' / 'evaluate.py'),
    'official_evaluation_CPU_reader': (49923, '2357929fcd38ffc3020c5b8d204ee371da8da3972ff2e2e03398c8225c55e916',
        C_ROOT / 'spatial_position_preserving_common24_evaluation_source_20261009' / 'qualify_evaluation_cpu.py')}
DEV_PIN = dict(path=str(C_ROOT.parent.parent / 'experiments' / 'public_baseline_20260925' / 'protocol' / 'pseg_ssdd' / 'dev.json'),
    bytes=118319, sha256='5657600d6ea198f79bd5d557d2a1aadce5bec031d2ecf72a98716248d1183a55')


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
    require(isinstance(row, dict) and set(row) == {'path', 'bytes', 'sha256'} and Path(row['path']).is_absolute() and
        type(row['bytes']) is int and row['bytes'] > 0 and isinstance(row['sha256'], str) and
        len(row['sha256']) == 64 and set(row['sha256']) <= set('0123456789abcdef'), 'Complete absolute pinned ref required')
    require(reference(row['path']) == row, 'Pinned bytes differ: ' + row['path'])
    return Path(row['path'])


def read_ref(row):
    return read(bound(row))


def finite(value, label, nonnegative=False):
    require(type(value) in (int, float) and math.isfinite(value) and (not nonnegative or value >= 0),
        label + ' must be finite')


def terminal(receipt, phase):
    require(receipt.get('status') == 'ACTUAL_OWNED_TERMINAL' and receipt.get('original_identity') is not None and
        type(receipt.get('returncode')) is int and receipt['returncode'] == 0 and receipt.get('root_wait_reaped') is True and
        receipt.get('held_handle_exit') == dict(wait_state='WAIT_OBJECT_0', actual_exit_code=0) and
        receipt.get('numeric_birth_absence', {}).get('original_birth_absent') is True and
        all(key in receipt and receipt[key] is None for key in ('abort', 'error')) and
        receipt.get('wait_errors') is False and receipt.get('GPU_observation_errors') == [],
        'STOP before hashes/scientific access: actual raw held0/reap/original-birth absence/noerrors required')
    if 'phase' in receipt:
        require(receipt['phase'] == phase, 'Actual raw terminal phase differs')
    peak = receipt.get('whole_GPU_observed_peak_MiB')
    finite(peak, 'Actual whole-GPU observed peak', True)
    require(peak <= 8192 and receipt.get('whole_GPU_used_limit_MiB') == 8192 and
        receipt.get('whole_leaf_VRAM_certified') is False and
        receipt.get('GPU_scope') == 'whole GPU observed, not per-process WDDM allocation',
        'Actual observed whole-GPU scope/8192 limit differs')


def load_source(row, name):
    path = bound(row)
    value = path.read_bytes()
    require(len(value) == row['bytes'] and hashlib.sha256(value).hexdigest() == row['sha256'], 'SOURCE changed before compile')
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(value, str(path), 'exec'), module.__dict__)
    return module


def source_gate(spec_path, spec, raw):
    producer = reference(Path(__file__).resolve().with_name('evaluate.py'))
    reader, spec_ref = reference(__file__), reference(spec_path)
    require(Path(spec['GPU_admission_file']).is_absolute(), 'Absolute actual evaluation admission required')
    admission_ref = reference(spec['GPU_admission_file'])
    admission = read(spec['GPU_admission_file'])
    owner = reference(spec_path.parent / 'finite_point_cache_execution_20261005_v2' / 'run_owned_gpu_limited.py')
    expect(admission, dict(status='ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE', automatic_retry=False,
        phase=spec['phase'], allowed_cpus=[0, 1], cuda_visible_devices='0', qualified_leaf_deadline_seconds=1800,
        whole_GPU_used_limit_MiB=8192, driver_sha256=OWNER_SHA,
        entry_args=['--spec', str(spec_path), '--owner-execute']), 'Current evaluation owned admission')
    require(raw['admission_sha256'] == admission_ref['sha256'] and owner['sha256'] == OWNER_SHA and
        Path(admission['entry']).resolve() == Path(producer['path']) and
        spec['entry_source'] == producer and spec['CPU_reader'] == reader and
        producer in spec['sources'] and reader in spec['sources'] and
        len({row['path'] for row in spec['sources']}) == len(spec['sources']), 'Current source/SPEC/receipt/owner bindings differ')
    hashes = admission['bound_source_sha256']
    require(hashes.get(str(spec_path)) == spec_ref['sha256'], 'Current SPEC not admission-bound')
    for row in spec['sources']:
        bound(row)
        require(hashes.get(row['path']) == row['sha256'], 'Complete evaluation SOURCE closure not owner-bound')
    for role, (size, sha, path) in SOURCE_PINS.items():
        row = spec[role]
        require(row in spec['sources'] and row['bytes'] == size and row['sha256'] == sha and
            Path(row['path']).resolve() == path.resolve(), 'Canonical reused evaluator/SHORT2 SOURCE role differs: ' + role)
    require(spec['official_evaluation_source'] == spec['evaluation_source'], 'Official evaluation SOURCE alias differs')
    for role in ('preflight_source', 'preflight_CPU_reader', 'readout_source', 'geometry_source', 'hooks_source',
                 'training_source', 'training_CPU_reader', 'initializer_source', 'micro_preflight_source', 'native_helpers',
                 'native_preflight_helpers', 'reused_CPU_reader', 'operator_CPU_fixture_source'):
        require(spec[role] in spec['sources'], 'Reused canonical SOURCE role absent: ' + role)
    if 'scientific_spec' in admission:
        require(admission['scientific_spec'] == spec_ref, 'Admitted current scientific SPEC differs')
    return dict(source=producer, reader=reader, spec=spec_ref, GPU_admission=admission_ref, owner_driver=owner)


def qualified_short2(spec, raw, short, pre):
    """Consume actual SHORT2 CPU qualification; no preflight checkpoint is loaded."""
    require(spec['short2_phase'] == SHORT2_PHASE and spec['short2_training_spec'] == SHORT2_SPEC_PIN,
        'Canonical already-admitted original SHORT2 phase/SPEC differs')
    q = read_ref(spec['short2_CPU_qualification'])
    expect(q, dict(schema=short.QUAL_SCHEMA, status=short.QUAL_STATUS, phase=spec['short2_phase'],
        source=spec['short2_source'], reader_source=spec['short2_CPU_reader'], spec=spec['short2_training_spec'],
        parent_receipt=spec['short2_receipt'], modes=list(MODES), actual_epochs_per_arm=2,
        actual_microsteps_per_arm=1482, actual_optimizer_updates_per_arm=741, actual_raw_GT_visits_per_arm=3322,
        total_trace_records=5928, GT_descriptor_cache_unique_images=741, formal_training_qualified=False,
        eligible_for_formal_resume=False, AP_computed=False, efficacy_claimed=False, scientific_method_verdict=None,
        all_teacher610_states_and_typed_BN_exact=True, only_fresh2W_changed_and_all_AdamW_moments_step741_verified=True,
        native30_and_all10_assignment_GT_coverage_verified=True,
        odd741_epoch_boundary_actual_pending_gradient_digest_identity_verified=True, no_epoch_gradient_flush=True,
        teacher_optimizer_order_RNG_or_preflight_CP_restored=False,
        complete_SOURCE_closure_verified_before_scientific_import=True), 'Actual independently qualified SHORT2 scope')
    require(q['GPU_admission'] == SHORT2_ADMISSION_PIN, 'Canonical original SHORT2 admission differs')
    training, admission = read_ref(q['spec']), read_ref(q['GPU_admission'])
    expect(training, dict(schema=short.SCHEMA, phase=spec['short2_phase'], entry_source=spec['short2_source'],
        CPU_reader=spec['short2_CPU_reader'], modes=list(MODES), training_all_FIT=True, epochs=2,
        FIT_images=741, FIT_GT=1661, total_microsteps=1482, total_updates=741, order_seed=2026100917,
        loss_seed_base=1000000, expected_epoch=24, hidden_seed=2026100911, permutation_seed=2026101013,
        accumulation=2, learning_rate=0.0001, weight_decay=0.05, clip_grad=dict(max_norm=0.01, norm_type=2),
        checkpoint_epoch=2, detail_microsteps=list(short.DETAILS), CPU_affinity=[0, 1],
        allocator_limit_MiB=4096, whole_GPU_owner_limit_MiB=8192, deadline_seconds=10800), 'Actual SHORT2 training SPEC')
    expect(admission, dict(status='ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE', automatic_retry=False,
        phase=spec['short2_phase'], allowed_cpus=[0, 1], cuda_visible_devices='0', qualified_leaf_deadline_seconds=10800,
        whole_GPU_used_limit_MiB=8192, driver_sha256=OWNER_SHA,
        entry_args=['--spec', q['spec']['path'], '--owner-execute']), 'Actual SHORT2 admission')
    require(raw['admission_sha256'] == q['GPU_admission']['sha256'] and
        Path(admission['entry']).resolve() == Path(spec['short2_source']['path']).resolve() and
        Path(training['GPU_admission_file']).resolve() == Path(q['GPU_admission']['path']).resolve() and
        admission['bound_source_sha256'].get(q['spec']['path']) == q['spec']['sha256'] and
        len(training['sources']) == 57, 'Actual SHORT2 raw terminal/SPEC/source/admission closure differs')
    for row in training['sources']:
        require(row in spec['sources'] and admission['bound_source_sha256'].get(row['path']) == row['sha256'],
            'Evaluation does not preserve actual qualified SHORT2 SOURCE closure')
        bound(row)
    for role in ('teacher_training_spec', 'teacher_qualification', 'teacher_checkpoint', 'teacher_receipt',
                 'preflight_training_spec', 'preflight_CPU_qualification', 'preflight_source', 'preflight_CPU_reader',
                 'preflight_receipt', 'readout_source', 'geometry_source', 'hooks_source', 'initializer_source',
                 'micro_preflight_source', 'native_helpers', 'native_preflight_helpers', 'reused_CPU_reader',
                 'evaluation_source', 'training_source', 'training_CPU_reader', 'operator_CPU_fixture',
                 'operator_CPU_fixture_source', 'FIT_annotation', 'image_root'):
        require(training[role] == spec[role], 'Actual SHORT2/evaluation lineage differs: ' + role)
    for role in ('teacher_phase', 'preflight_phase'):
        require(training[role] == spec[role], 'Original owner phase differs: ' + role)
    for role in ('teacher_checkpoint', 'teacher_qualification', 'teacher_training_spec', 'teacher_receipt',
                 'preflight_CPU_qualification', 'preflight_source', 'preflight_CPU_reader',
                 'preflight_training_spec', 'preflight_receipt', 'FIT_annotation'):
        require(q[role] == training[role] == spec[role], 'Actual independently qualified SHORT2 lineage differs: ' + role)
    for role in ('source', 'reader_source', 'report', 'start', 'GPU_qualification', 'plan', 'parent_receipt',
                 'GPU_admission', 'owner_driver'):
        bound(q[role])
    require(read_ref(q['plan']) == read_ref(read_ref(q['report'])['plan']), 'Actual qualified SHORT2 plan/report join differs')
    require(set(q['training_proofs']) == set(q['checkpointmap']) == set(MODES), 'Allfour qualified SHORT2 endpoints required')
    for mode in MODES:
        proof = q['training_proofs'][mode]
        expect(proof, dict(mode=mode, checkpoint=spec['checkpointmap'][mode], actual_epochs=2, actual_microsteps=1482,
            actual_optimizer_updates=741, raw_GT_visits=3322, full_teacher610_and_BN106_affine159_buffers_exact=True,
            full612_strict_CPU_readback=True, only_fresh_two_weights_changed=True, parameter_tensors=453,
            active_parameter_tensors=2, active_scalars=393216, epoch_boundary_accumulation_preserved=True,
            native30_total_consistent=True, ten_native_assignments_retain_all_raw_GT=True,
            formal_training_qualified=False, eligible_for_formal_resume=False, AP_computed=False, efficacy_claimed=False),
            'Actual SHORT2 endpoint qualification/' + mode)
        require(q['checkpointmap'][mode] == spec['checkpointmap'][mode] and
            spec['qualificationmap'][mode] == spec['short2_CPU_qualification'], 'SHORT2 endpoint CP/CPUQ map differs')
    require(spec['checkpointmap']['native0'] == spec['teacher_checkpoint'] and
        spec['qualificationmap']['native0'] == spec['teacher_qualification'], 'Native baseline must use immutable E24 teacher')
    return q, training


def digest_contract(pre, value, shape, dtype='float32'):
    pre.digest_contract(value, shape, dtype)


def checkpoint_state(torch, old, pre, short, spec, q, teacher_model, teacher, readout, geometry, hooks, mode):
    """Independent strict CPU architecture/load; never inference or optimizer replay."""
    model = copy.deepcopy(teacher_model)
    if mode != 'native0':
        hooks.register_query_pair_readout(torch, readout, geometry, model, mode, spec['readout_source'], spec['geometry_source'],
            hidden_seed=2026100911, permutation_seed=2026101013)
    native, modules, BN_params, BN_buffers = pre.typed_native(torch, model)
    current = model.state_dict()
    if mode == 'native0':
        state = teacher
        training_proof = read_ref(read_ref(spec['teacher_qualification'])['model_proof']['proof_refs']['state_proof'])
    else:
        cp = torch.load(str(bound(spec['checkpointmap'][mode])), map_location='cpu')
        require(set(cp) == {'state_dict', 'adapter_state_dict', 'optimizer_state_dict', 'external_rng_state', 'meta'},
            'Only actual complete612 SHORT2 endpoint CP may be evaluated')
        state = cp['state_dict']
        meta = cp['meta']
        expect(meta, dict(schema=short.CP_SCHEMA, kind='mask2former', mode=mode, phase=spec['short2_phase'],
            source=spec['short2_source'], spec=spec['short2_training_spec'], completed_epoch=2, completed_epochs=2,
            initial_epoch=0, epochs_this_run=2, total_epochs=2, global_microsteps=1482, global_updates=741,
            consumed_raw_GT=3322, FIT_images=741, FIT_GT=1661, batch_size=1, accumulation=2,
            teacher_checkpoint=spec['teacher_checkpoint'], teacher_qualification=spec['teacher_qualification'],
            teacher_training_spec=spec['teacher_training_spec'], teacher_receipt=spec['teacher_receipt'],
            preflight_CPU_qualification=spec['preflight_CPU_qualification'], preflight_training_spec=spec['preflight_training_spec'],
            teacher_mode='native0', teacher_epoch=24, readout_source=spec['readout_source'], geometry_source=spec['geometry_source'],
            hooks_source=spec['hooks_source'], native610_exact=True, frozen_native=True, native_loss_only=True,
            native_loss_atoms=30, no_Sobel_or_auxiliary_loss=True, no_AP=True, LR_decay=False,
            fresh_optimizer_state_entries=0, teacher_optimizer_loaded=False, teacher_order_RNG_loaded=False,
            teacher_external_RNG_loaded=False, preflight_checkpoint_loaded=False, resume_checkpoint=None,
            resume_qualification=None, no_pending_accumulation_gradients=True, no_scheduler=True, no_warmup=True,
            no_feature_cache=True, short_budget_hypothesis_probe=True, eligible_for_formal_resume=False, AP_computed=False),
            'Actual full SHORT2 endpoint metadata/' + mode)
        training_proof = read_ref(q['training_proofs'][mode]['state_proof'])
        expect(training_proof, dict(schema=short.PROOF_SCHEMA, mode=mode, native610_exact=True, frozen_BN_all_exact=True,
            CPU_checkpoint_readback_exact=True, GPU_to_CPU_readback_state_exact=True, global_microsteps=1482,
            global_updates=741, consumed_raw_GT=3322, completed_epochs=2), 'Actual qualified SHORT2 state proof')
        old.check_state(torch, teacher, training_proof['native_teacher610'], 'qualified immutable native610 teacher')
        require(training_proof['policy'] == q['training_proofs'][mode]['trainability'] == meta['policy'],
            'Qualified endpoint trainability/typed native policy differs')
    require(set(state) == set(current) and len(state) == (610 if mode == 'native0' else 612) and
        all(isinstance(value, torch.Tensor) and value.device.type == 'cpu' and value.shape == current[name].shape and
            value.dtype == current[name].dtype and bool(torch.isfinite(value).all()) for name, value in state.items()),
        'Full endpoint architecture keys/shape/dtype/finite differ')
    old.check_state(torch, state, training_proof['after'], 'independently qualified full endpoint state')
    require(set(teacher) == {name for name in state if not name.startswith(PREFIX)} and
        all(torch.equal(state[name], value) for name, value in teacher.items()) and
        all(torch.equal(state[name], teacher[name]) for name in BN_params + BN_buffers),
        'All frozen native610 and typed BN106+159 must match E24 teacher exactly')
    if mode != 'native0':
        require(set(cp['adapter_state_dict']) == {'hidden.weight', 'final.weight'} and
            all(torch.equal(value, state[PREFIX + name]) for name, value in cp['adapter_state_dict'].items()) and
            list(state[ACTIVE[0]].shape) == [512, 512] and list(state[ACTIVE[1]].shape) == [256, 512] and
            sum(state[name].numel() for name in ACTIVE) == 393216 and bool(torch.count_nonzero(state[ACTIVE[1]])),
            'Actual full-state two-weight alias/shape/nonzero final differs')
        require(training_proof['policy']['native_names'] == native and training_proof['policy']['BN_module_names'] == modules and
            training_proof['policy']['BN_parameter_names'] == BN_params and training_proof['policy']['BN_buffer_names'] == BN_buffers,
            'Independent typed frozen native/BN membership differs')
    result = model.load_state_dict(state, strict=True)
    require(not result.missing_keys and not result.unexpected_keys and
        all(torch.equal(value, state[name]) for name, value in model.state_dict().items()), 'Independent strict full CPU readback differs')
    model.requires_grad_(False).eval()
    require(all(not value.requires_grad and value.grad is None for value in model.parameters()) and
        all(not value.training for value in model.modules()) and len(list(model.parameters())) == (451 if mode == 'native0' else 453),
        'Independent inference architecture must be frozen/eval/gradNone')
    fusion = model.panoptic_fusion_head
    require(model.panoptic_head.num_queries == 100 and model.panoptic_head.num_classes == 1 and
        fusion.num_classes == fusion.num_things_classes == 1 and fusion.num_stuff_classes == 0 and
        fusion.test_cfg.get('max_per_image', 100) == 100 and fusion.test_cfg.get('instance_on', False) is True and
        fusion.test_cfg.get('panoptic_on', True) is False and fusion.test_cfg.get('semantic_on', False) is False,
        'Original single-class/all100 official instance-fusion configuration differs')
    out = dict(mode=mode, checkpoint=spec['checkpointmap'][mode], qualification=spec['qualificationmap'][mode],
        fullstate={name: old.tensor_digest(torch, value, str(value.detach().numpy().dtype)) for name, value in state.items()},
        state_tensors=len(state), native_state_tensors=610, adapter_state_tensors=0 if mode == 'native0' else 2,
        native_parameter_names=native, BN_module_names=modules, BN_parameter_names=BN_params, BN_buffer_names=BN_buffers,
        registered_parameters=451 if mode == 'native0' else 453, full_native610_exact=True, typed_BN106_affine159_buffers_exact=True,
        complete_endpoint_strict_CPU_loaded=True, all_parameters_frozen=True, all_modules_eval=True, parameters_gradNone=True,
        no_CPU_forward_or_geometry_replay=True)
    del model, current, training_proof
    if mode != 'native0':
        del cp, state
    gc.collect()
    require(not torch.cuda.is_initialized(), 'CPU endpoint audit initialized CUDA')
    return out


def evaluation_state(record, state, spec, mode):
    proof = read_ref(record['state_proof'])
    expect(proof, dict(schema=STATE_SCHEMA, mode=mode, checkpoint_mode=mode, before=state['fullstate'], after=state['fullstate'],
        checkpoint=spec['checkpointmap'][mode], qualification=spec['qualificationmap'][mode],
        full_native_and_adapter_state_exact=True, params_gradNone=True, all_parameters_frozen=True, all_modules_eval=True,
        hook_cache_clear=True, frozen_teacher610_exact=True, registered_state_not_pruned=mode != 'native0',
        optimizer_or_RNG_restored=False, same_CP_inference_ablation=False),
        'Actual evaluation endpoint state proof/' + mode)
    loading = proof['loading_proof']
    expect(loading, dict(mode=mode, checkpoint=spec['checkpointmap'][mode], qualification=spec['qualificationmap'][mode],
        CP_qualified=True, strict_fullstate_loaded=True, native_state_tensors=610, adapter_state_tensors=state['adapter_state_tensors'],
        registered_parameters=state['registered_parameters'], trained_fullstate=state['fullstate'], original_frozen_BN_exact=True,
        inference_backbone_with_cp=False, no_random_missing_keys=True), 'Actual endpoint strict loading proof/' + mode)
    require(set(loading['BN_parameter_names']) == set(state['BN_parameter_names']) and
        set(loading['BN_buffer_names']) == set(state['BN_buffer_names']), 'Actual typed BN loading membership differs')
    require(proof['native_teacher610'] == {name: value for name, value in state['fullstate'].items()
        if not name.startswith(PREFIX)}, 'Actual inference native610 teacher proof differs')
    return dict(source=record['state_proof'], full_endpoint_CPU_checkpoint_exact=True, full_native610_teacher_exact=True,
        typed_BN106_affine159_buffers_exact=True, before_after_inference_exact=True, zero_native_state_mutants=True,
        parameters_frozen_eval_gradNone_source_observed=True)


def image_population(torch, np, old, pre, spec, record, coco, mode, P):
    ids, rows = list(coco.imgs), read_ref(record['image_rows'])
    require(len(rows) == 187 and [row['image_id'] for row in rows] == ids, 'Complete ordered original DEV187 population required')
    for index, row in enumerate(rows):
        image = coco.imgs[ids[index]]
        h, w = int(image['height']), int(image['width'])
        scale = 800. / max(h, w)
        actual = [int(h * scale + .5), int(w * scale + .5)]
        expect(row, dict(mode=mode, checkpoint_mode=mode, dataset_index=index, image_id=ids[index],
            ann_ids=[ann['id'] for ann in coco.imgToAnns[ids[index]]], annotation_count=len(coco.imgToAnns[ids[index]]),
            original_shape=[h, w], actual_img_shape=actual, batch_input_shape=[800, 800],
            scale_factor=[actual[1] / w, actual[0] / h], checkpoint=spec['checkpointmap'][mode], CP_qualified=True,
            head_seed=100000 + ids[index], external_RNG_restored=True, GT_removed_from_forward=True,
            loader_accesses_annotations=True, official_instance_postprocess=True, source_native_query_ID_mapping_verified=True,
            official_masks_query_ID_bitwise_equal=True, official_scores_query_ID_bitwise_equal=True,
            query_score_factors_verified=True, class_probability_mask_quality_product_TorchEqual_official_score=True,
            matching_or_loss_executed=False, added_NMS=False, added_score_cutoff=False, query_filter=False, predictions=100,
            all_native_state610_exact=True, native_gradNone=True, all_modules_eval=True, hook_cache_clear=True,
            inference_ablation_not_training_arm=False, class_or_query_identity_does_not_imply_equal_mask_quality_or_scores=True),
            'Actual GT-free all100 DEV image/' + mode)
        require(Path(row['image']['path']).resolve() == (Path(spec['image_root']) / image['file_name']).resolve(), 'Original DEV image path differs')
        bound(row['image'])
        digest_contract(pre, row['input'], [1, 3, 800, 800])
        expect(row['identity'], dict(image_id=ids[index], dataset_index=index,
            ann_ids=[ann['id'] for ann in coco.imgToAnns[ids[index]]], original_shape=[h, w], actual_img_shape=actual,
            batch_input_shape=[800, 800], scale_factor=[actual[1] / w, actual[0] / h],
            image=row['image']), 'Actual original input/annotation identity')
        old.validate_rng(torch, dict(row, private_seed=row['head_seed']))
        require(row['external_RNG_before'] == row['external_RNG_after'], 'Actual private head RNG not restored')
        for values in (row['external_RNG_before'], row['external_RNG_after']):
            require(isinstance(values, list) and len(values) == 2, 'Actual external CPU/CUDA RNG digest pair required')
            for value in values:
                require(isinstance(value, dict) and set(value) == {'shape', 'dtype', 'sha256'} and
                    len(value['shape']) == 1 and type(value['shape'][0]) is int and value['shape'][0] > 0,
                    'Actual RNG digest shape differs')
                digest_contract(pre, value, value['shape'], 'uint8')
        query_ids = row['prediction_query_ids']
        require(isinstance(query_ids, list) and len(query_ids) == 100 and all(type(v) is int for v in query_ids) and
            set(query_ids) == set(range(100)), 'Every original native query ordinal must appear exactly once')
        native = row['native_fields']
        require(len(native['classes']) == len(native['native_masks']) == 10, 'All10 native frozen fields required')
        for value in native['classes']:
            digest_contract(pre, value, [1, 100, 2])
        for value in native['native_masks']:
            digest_contract(pre, value, [1, 100, 200, 200])
        for key, shape in (('feature', [1, 256, 200, 200]), ('coefficient', [1, 100, 256]), ('valid_fraction', [1, 1, 200, 200])):
            digest_contract(pre, native[key], shape)
        axes = torch.arange(200, dtype=torch.float32, device='cpu')
        valid = ((actual[0] - 4 * axes).clamp(0, 4) / 4).reshape(1, 1, 200, 1) * \
            ((actual[1] - 4 * axes).clamp(0, 4) / 4).reshape(1, 1, 1, 200)
        require(old.tensor_digest(torch, valid, 'float32') == native['valid_fraction'], 'Independent metadata-only valid fraction differs')
        digest_contract(pre, row['final_fields']['classes'], [1, 100, 2])
        digest_contract(pre, row['final_fields']['masks'], [1, 100, 200, 200])
        require(row['final_fields']['classes'] == native['classes'][-1], 'Adapter changed actual final native classes')
        if mode == 'native0':
            require(row['native_hook'] is None and row['final_fields']['masks'] == native['native_masks'][-1],
                'Native baseline cannot invoke an adapter or mutate its final mask')
        else:
            hook = row['native_hook']
            expect(hook, dict(forward_calls=1, completed_forwards=1, pixel_decoder_calls=1, coefficient_calls=10,
                passive_coefficient_captures=1, unchanged_early_readouts=9, final_mask_replacements=1, adapter_calls=1,
                apply=True, mode=mode, adapter_prefix=PREFIX, readout_source=spec['readout_source'], geometry_source=spec['geometry_source'],
                hidden_seed=2026100911, permutation_seed=2026101013, metadata_fields_used=['img_shape', 'batch_input_shape'],
                class_list_object_preserved=True, first_nine_mask_objects_preserved=True, native_coefficient_outputs_untouched=True,
                earlier_gate_paths_untouched=True, all_queries_retained=True, adapter_geometry_phi_skipped=False,
                cache_cleared_after_forward=True, registered_state_not_pruned=True, hooks_removed=True,
                hook_handles_installed=4, hook_handles_removed=4, grad_mode_enabled_at_start=False,
                feature_shape=[1, 256, 200, 200], coefficient_shape=[1, 100, 256], final_native_mask_shape=[1, 100, 200, 200],
                cold_zero_final_weight=False, cold_native_value_equal=None,
                native_graph_properties=dict(feature_requires_grad=False, coefficient_requires_grad=False,
                    native_final_mask_requires_grad=False, captured_tensors_detached_by_bridge=False, forward_gradient_proof=False)),
                'Actual frozen all100 query bridge/' + mode)
            pre.check_readout(torch, hook['readout_observation'], mode, P)
        packet = row['native_class_packet']
        require(set(packet) == {'shape', 'dtype', 'values', 'sha256'} and packet['shape'] == [100, 2] and packet['dtype'] == 'float32',
            'Complete actual native class packet required')
        values = np.asarray(packet['values'], dtype=np.float32)
        require(values.shape == (100, 2) and bool(np.isfinite(values).all()) and
            all(type(v) in (int, float) and math.isfinite(v) and float(np.float32(v)) == v for row_values in packet['values'] for v in row_values) and
            hashlib.sha256(values.tobytes(order='C')).hexdigest() == packet['sha256'] == native['classes'][-1]['sha256'],
            'Independent raw FP32 class packet/hash identity differs')
    return ids, rows


def class_probability_check(torch, np, rows, factors):
    by_image = {row['image_id']: row for row in rows}
    probabilities = {}
    for row in rows:
        values = torch.from_numpy(np.asarray(row['native_class_packet']['values'], dtype=np.float32))
        ship = values.softmax(-1)[:, 0]
        require(bool(torch.isfinite(ship).all()) and bool(((ship >= 0) & (ship <= 1)).all()), 'Independent ship softmax nonfinite')
        probabilities[row['image_id']] = ship.tolist()
    max_difference = 0.
    for factor in factors:
        require(factor['image_id'] in by_image and type(factor['query_id']) is int and 0 <= factor['query_id'] < 100,
            'Actual class factor population identity differs')
        difference = abs(probabilities[factor['image_id']][factor['query_id']] - factor['class_probability'])
        require(difference <= 1e-6, 'Independent CPU class softmax differs beyond explicit FP32 device tolerance')
        max_difference = max(max_difference, difference)
    return dict(actual_native_class_packet_FP32_hash_verified=True, CPU_ship_softmax_checked=True,
        CPU_GPU_softmax_absolute_tolerance=1e-6, max_absolute_difference=max_difference, exact_CPU_GPU_softmax_claimed=False)


def class_identity(spec, image_rows, factor_rows):
    first = image_rows['native0']
    reference_factors = {(v['image_id'], v['query_id']): v['class_probability'] for v in factor_rows['native0']}
    require(len(reference_factors) == 18700, 'Complete baseline class/query population required')
    for mode in MODES:
        require(len(image_rows[mode]) == len(first) == 187 and all(a['dataset_index'] == b['dataset_index'] and
            a['image_id'] == b['image_id'] and a['native_fields'] == b['native_fields'] and
            a['native_class_packet'] == b['native_class_packet'] and a['final_fields']['classes'] == b['final_fields']['classes'] and
            a['input'] == b['input'] and a['head_seed'] == b['head_seed'] and a['prediction_query_ids'] == b['prediction_query_ids']
            for a, b in zip(first, image_rows[mode])), 'Frozen native classes/F/e0/valid/order differ across endpoints')
        require({(v['image_id'], v['query_id']): v['class_probability'] for v in factor_rows[mode]} == reference_factors,
            'Actual frozen class probability by native query ordinal differs')
    return dict(full_native610_shared=True, native10_class_mask_and_Fe0_valid_packets_equal=True,
        final_native_classes_and_class_packet_equal=True, actual_class_probability_by_query_exact=True,
        all18700_native_queries_retained_and_actual_order_equal=True,
        mask_quality_or_official_score_equality_required=False, quality_dependent_scores_allowed_to_change=True,
        predicted_query_ID_is_object_identity_claimed=False, CPU_forward_or_geometry_replayed=False)


def run(spec_file, acceptance_file):
    spec_path, target = Path(spec_file), Path(acceptance_file)
    require(spec_path.is_absolute() and target.is_absolute() and target.parent.is_dir() and not target.exists(),
        'Absolute SPEC and fresh absolute acceptance with existing parent required')
    spec_path = spec_path.resolve(strict=True)
    spec = read(spec_path)
    expect(spec, dict(schema=SPEC_SCHEMA, modes=list(MODES), endpoints=list(ENDPOINTS), expected_epoch=24, short2_epoch=2,
        head_seed_base=100000, evaluation_images=187, evaluation_GT=380, CPU_affinity=[0, 1], deadline_seconds=1800,
        allocator_limit_MiB=4096, whole_GPU_owner_limit_MiB=8192), 'Exact fixed five-endpoint official DEV SPEC')
    require(isinstance(spec['phase'], str) and spec['phase'].startswith('query_conditioned_frozen_native_short2_evaluation_') and
        Path(spec['phase']).name == spec['phase'] and set(spec['checkpointmap']) == set(spec['qualificationmap']) == set(ENDPOINTS),
        'Safe fixed phase and exactly five qualified endpoints required')
    receipt_path = spec_path.parent / 'finite_point_cache_execution_20261005_v2' / 'runs' / spec['phase'] / 'PARENT_RECEIPT.json'
    raw = read(receipt_path)
    terminal(raw, spec['phase'])
    receipts = {}
    for role, phase_role in (('teacher_receipt', 'teacher_phase'), ('preflight_receipt', 'preflight_phase'), ('short2_receipt', 'short2_phase')):
        require(Path(spec[role]['path']).is_absolute(), 'Absolute prior raw terminal required')
        receipts[role] = read(spec[role]['path'])
        terminal(receipts[role], spec[phase_role])
    # Actual current first, then teacher/preflight/SHORT2 RAW terminal gates
    # precede ALL SOURCE/SPEC/scientific hashes, Q/CP/predictions and imports.
    execution = source_gate(spec_path, spec, raw)
    short = load_source(spec['short2_CPU_reader'], 'sealed_actual_SHORT2_CPU_helpers')
    pre = load_source(spec['preflight_CPU_reader'], 'sealed_actual_native_preflight_CPU_helpers')
    for role in pre.ARTIFACT_PINS:
        pre.pinned_role(spec, role)
    for role, pin in pre.SOURCE_PINS.items():
        require((spec[role]['bytes'], spec[role]['sha256']) == pin and spec[role] in spec['sources'], 'Canonical old SOURCE pin differs: ' + role)
    for role, path in pre.SOURCE_PATHS.items():
        require(Path(spec[role]['path']).resolve() == path.resolve(), 'Canonical old SOURCE path differs: ' + role)
    for role, pin in short.PREFLIGHT_PINS.items():
        require(spec[role] == pin, 'Canonical actual preflight artifact/SOURCE differs: ' + role)
    short.preflight_gate(pre, spec, receipts['preflight_receipt'])
    training, teacher_q = pre.teacher_lineage(spec, receipts['teacher_receipt'])
    pre.fixture_gate(spec)
    short_q, short_training = qualified_short2(spec, receipts['short2_receipt'], short, pre)
    require(spec['DEV_annotation'] == DEV_PIN and Path(spec['image_root']).resolve() ==
        Path('E:/PSeg_SSDD/voc_style/JPEGImages_train').resolve(), 'Canonical original historically exposed DEV/image root differs')
    bound(spec['DEV_annotation'])
    official = load_source(spec['official_evaluation_CPU_reader'], 'sealed_official_COCO_RLE_scorefree_CPU_helpers')
    output = Path(spec['output_directory'])
    require(output.is_absolute(), 'Absolute actual evaluation output directory required')
    report_ref, start_ref = reference(output / 'REPORT.json'), reference(output / 'START.json')
    aggregate_ref = reference(output / 'QUALIFIED.json')
    report, start = read(output / 'REPORT.json'), read(output / 'START.json')
    expect(report, dict(schema=REPORT_SCHEMA, phase=spec['phase'], source=execution['source'], spec=execution['spec'],
        status='QUERY_CONDITIONED_FROZEN_NATIVE_SHORT2_DEV_EXPORT_COMPLETE_NO_AP', error=None, modes=list(MODES), endpoints=list(ENDPOINTS),
        checkpointmap=spec['checkpointmap'], qualificationmap=spec['qualificationmap'], teacher_qualification=spec['teacher_qualification'],
        short2_CPU_qualification=spec['short2_CPU_qualification'], short2_training_spec=spec['short2_training_spec'],
        evaluation_images=187, evaluation_GT=380, full_teacher610_exact_all_endpoints=True,
        all_five_CPU_models_before_CUDA=True, only_one_model_on_CUDA_at_a_time=True, native_official_fusion=True,
        all100_predictions_per_image=True, GT_removed_from_forward=True, no_TEST=True, no_AP=True, AP_computed=False,
        official_metrics_pending_independent_CPU_reader=True, mask_quality_not_independently_CPU_recomputed_from_omitted_logits=True,
        class_or_query_identity_does_not_imply_equal_mask_quality_or_scores=True, no_added_NMS_or_score_cutoff=True,
        optimizer_or_RNG_restored=False, efficacy_claimed=False, scientific_method_decision_computed=False,
        allocator_limit_MiB=4096, GPU_scope='whole GPU must be verified from actual owner receipt'),
        'Actual completed five-endpoint export report')
    expect(start, dict(schema='query-conditioned-frozen-native-short2-evaluation-start-v1', phase=spec['phase'],
        source=execution['source'], spec=execution['spec'], modes=list(MODES), endpoints=list(ENDPOINTS),
        expected_epoch=24, short2_epoch=2, evaluation_images=187, evaluation_GT=380,
        checkpointmap=spec['checkpointmap'], qualificationmap=spec['qualificationmap'], teacher_qualification=spec['teacher_qualification'],
        short2_CPU_qualification=spec['short2_CPU_qualification'], short2_training_spec=spec['short2_training_spec'],
        original_three_terminal_gates_before_scientific_access=True, all_five_CPU_models_before_CUDA_required=True,
        no_AP=True, no_TEST=True, no_retry=True, optimizer_or_RNG_restored=False, official_metrics_pending_independent_CPU_reader=True),
        'Actual planned five-endpoint START')
    expect(read_ref(aggregate_ref), dict(schema='query-conditioned-frozen-native-short2-evaluation-aggregate-v1',
        status='ROOT_QUERY_CONDITIONED_FROZEN_NATIVE_SHORT2_EVALUATION_PENDING_CPU_READER', source=execution['source'],
        spec=execution['spec'], report=report_ref, mode_rows=report['mode_rows'], short2_CPU_qualification=spec['short2_CPU_qualification'],
        independently_CPU_qualified=False, AP_computed=False, efficacy_claimed=False), 'Actual pending aggregate/report/endpoint joins')
    finite(report['elapsed_seconds'], 'Actual completed evaluation elapsed seconds', True)
    require(report['elapsed_seconds'] < 1800 and type(report['torch_peak_allocated_bytes']) is int and
        0 <= report['torch_peak_allocated_bytes'] <= 4096 * 1024 ** 2 and
        set(report['mode_rows']) == set(report['cleanup']) == set(ENDPOINTS),
        'Actual evaluation resources/endpoints differ')
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
    from pycocotools.cocoeval import COCOeval
    from mmengine import Config
    from mmengine.model import revert_sync_batchnorm
    from mmengine.registry import init_default_scope
    from mmdet.registry import MODELS
    from mmdet.utils import register_all_modules
    from mmdet.datasets import CocoDataset
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    require(psutil.Process().cpu_affinity() == [0, 1] and not torch.cuda.is_initialized(), 'Actual CPU01/CUDAempty consumer required')
    register_all_modules(init_default_scope=False)
    init_default_scope('mmdet')
    require(len(CocoDataset.METAINFO['classes']) == 80 and CocoDataset.METAINFO['classes'][8] == 'boat', 'Official COCO80 boat order differs')
    coco = COCO(str(bound(spec['DEV_annotation'])))
    require(len(coco.imgs) == 187 and len(coco.anns) == 380 and set(coco.cats) == {1}, 'Complete original DEV187/380/ship population required')
    old = load_source(spec['reused_CPU_reader'], 'sealed_original_native_CPU_state_helpers')
    teacher_reader = load_source(spec['training_CPU_reader'], 'sealed_teacher_full24_CPU_helpers')
    initializer = load_source(spec['initializer_source'], 'sealed_original_COCO_boat_CPU_initializer')
    readout = load_source(spec['readout_source'], 'sealed_query_pair_readout_CPU_constructor')
    geometry = load_source(spec['geometry_source'], 'sealed_query_geometry_CPU_constructor')
    hooks = load_source(spec['hooks_source'], 'sealed_query_pair_registration_CPU_constructor')
    rng_before = torch.get_rng_state().clone()
    teacher_model, teacher = pre.load_teacher(torch, old, teacher_reader, initializer, MODELS,
        revert_sync_batchnorm, Config, spec, training, teacher_q)
    unused_cold, P = pre.independent_adapter(torch)
    states, results, image_rows, factor_rows = {}, {}, {}, {}
    for mode in ENDPOINTS:
        states[mode] = checkpoint_state(torch, old, pre, short, spec, short_q,
            teacher_model, teacher, readout, geometry, hooks, mode)
        record = report['mode_rows'][mode]
        expect(record, dict(mode=mode, checkpoint_mode=mode, checkpoint=spec['checkpointmap'][mode], qualification=spec['qualificationmap'][mode],
            completed_epoch=24 if mode == 'native0' else 2, CP_qualified=True, images=187, raw_GT=380,
            predictions_count=18700, query_score_factors_count=18700, fullstate_tensors=610 if mode == 'native0' else 612,
            full_native_state_exact=True, adapter_state_exact=True, params_gradNone=True, loader_accesses_annotations=True,
            GT_removed_from_forward=True, native_official_fusion=True, query_score_factors_verified=True,
            source_native_query_ID_mapping_verified=True, class_probability_mask_quality_product_TorchEqual_official_score=True,
            frozen_teacher610_exact=True, cross_model_query_ID_object_identity_claimed=False, cross_model_factor_exchange=False,
            inference_ablation_not_training_arm=False, registered_state_not_pruned=mode != 'native0',
            added_NMS=False, added_score_cutoff=False), 'Actual endpoint record/' + mode)
        directory = output / mode
        for role, name in (('image_rows', 'IMAGE_ROWS.json'), ('predictions', 'ALL_PREDICTIONS.json'),
                           ('query_score_factors', 'QUERY_SCORE_FACTORS.json'), ('state_proof', 'STATE_PROOF.json')):
            require(Path(record[role]['path']).resolve() == (directory / name).resolve(), 'Actual endpoint artifact path differs: ' + role)
        cleanup = report['cleanup'][mode]
        require(type(record['peak_allocated_bytes']) is int and 0 <= record['peak_allocated_bytes'] <= 4096 * 1024 ** 2 and
            type(cleanup['allocated_before']) is int and cleanup['allocated_before'] >= 0 and
            type(cleanup['allocated_after']) is int and cleanup['allocated_after'] == 0, 'Actual endpoint allocator/cleanup guard differs')
        state_proof = evaluation_state(record, states[mode], spec, mode)
        ids, rows = image_population(torch, np, old, pre, spec, record, coco, mode, P)
        predictions, grouped = official.prediction_population(record, ids, coco, np, masks, rows)
        factors = read_ref(record['query_score_factors'])
        factor_proof = official.score_factors(record, predictions, rows, np)
        class_proof = class_probability_check(torch, np, rows, factors)
        diagnostics, empty_predictions = official.geometry_rows(ids, grouped, coco, np, masks)
        require(len(diagnostics) == 380, 'All original380 raw DEV GT must remain in score-free diagnostics')
        results[mode] = dict(checkpoint=spec['checkpointmap'][mode], qualification=spec['qualificationmap'][mode],
            completed_epoch=24 if mode == 'native0' else 2, predictions=record['predictions'], image_rows=record['image_rows'],
            state_proof=state_proof, score_factor_proof=factor_proof, class_probability_proof=class_proof,
            COCO=official.coco_metrics(coco, predictions, ids, COCOeval),
            score_free_all100_best_geometry=official.geometry_summary(diagnostics), score_free_geometry_rows=diagnostics,
            empty_predictions_retained=empty_predictions, fixed_short_budget_probe=mode != 'native0',
            best100_is_diagnostic_not_AP_or_one_to_one_recall=True)
        image_rows[mode], factor_rows[mode] = rows, factors
        del predictions, grouped, diagnostics
        gc.collect()
    identity = class_identity(spec, image_rows, factor_rows)
    require(torch.equal(rng_before, torch.get_rng_state()) and not torch.cuda.is_initialized() and
        torch.get_num_threads() == torch.get_num_interop_threads() == 2 and psutil.Process().cpu_affinity() == [0, 1] and
        os.environ['CUDA_VISIBLE_DEVICES'] == '', 'CPU evaluation reader changed RNG/resources or initialized CUDA')
    acceptance = dict(schema=QUAL_SCHEMA, status=QUAL_STATUS, phase=spec['phase'], source=execution['source'], reader_source=execution['reader'],
        spec=execution['spec'], report=report_ref, start=start_ref, GPU_qualification=aggregate_ref, parent_receipt=reference(receipt_path),
        GPU_admission=execution['GPU_admission'], owner_driver=execution['owner_driver'],
        teacher_checkpoint=spec['teacher_checkpoint'], teacher_qualification=spec['teacher_qualification'],
        teacher_receipt=spec['teacher_receipt'], teacher_training_spec=spec['teacher_training_spec'],
        preflight_CPU_qualification=spec['preflight_CPU_qualification'], preflight_receipt=spec['preflight_receipt'],
        short2_CPU_qualification=spec['short2_CPU_qualification'], short2_training_spec=spec['short2_training_spec'],
        short2_receipt=spec['short2_receipt'], short2_source=spec['short2_source'], short2_CPU_reader=spec['short2_CPU_reader'],
        official_evaluation_source=spec['official_evaluation_source'], official_evaluation_CPU_reader=spec['official_evaluation_CPU_reader'],
        DEV_annotation=spec['DEV_annotation'], expected_epoch=24, short2_epoch=2, modes=list(MODES), endpoints=list(ENDPOINTS),
        checkpointmap=spec['checkpointmap'], qualificationmap=spec['qualificationmap'], mode_results=results,
        frozen_native_class_query_identity=identity, endpoint_states=states,
        raw_current_teacher_preflight_SHORT2_terminals_before_scientific_access=True,
        complete_SOURCE_closure_verified_before_scientific_import=True, all187_images_all380_original_GT_preserved=True,
        all100_predictions_per_image_and_empty_masks_preserved=True, no_added_NMS_score_cutoff_or_query_filter=True,
        full_native610_teacher_and_typed_BN106_affine159_buffers_CPU_exact=True, strict_full610_or612_endpoint_CPU_loaded=True,
        zero_native_state_mutants=True, official_COCO_maxDets=[1, 10, 100], official_segmentation_COCO12_independently_computed=True,
        score_free_best100_diagnostics_independently_computed=True, official_fusion_TorchEqual_masks_scores_source_observed=True,
        pixel_fusion_threshold_or_quality_mean_CPU_recomputed=False, CPU_affinity=[0, 1], torch_threads=2,
        torch_interop_threads=2, CUDA_initialized=False, external_CPU_RNG_restored=True, DEV_historically_exposed=True,
        no_TEST=True, blind_TEST_claimed=False, convergence_claimed=False, SOTA_claimed=False, scientific_method_verdict=None,
        evidence_limits=[
            'Four fresh frozen-native two-epoch hypothesis probes and the fixed E24 native teacher only; DEV results do not establish convergence or overall efficacy.',
            'Actual current/teacher/preflight/SHORT2 owned raw0 terminal gates precede all scientific hashes and accesses; qualified checkpoint provenance and full CPU states are checked independently.',
            'CPU does not replay CUDA inference, PCA, native eligibility, or optimizer trajectories; E/RMS and safe hook evidence is source-observed and checked for packet consistency.',
            'CPU validates all emitted RLE/binary masks/areas, class packets and factor arithmetic; omitted pixel logits prevent independent recreation of fusion thresholds or predicted-mask quality means.',
            'Actual frozen native class/query ordinal equality permits class probability joins; mask quality and quality-dependent official scores may change, and query ordinal is not physical object identity.',
            'FP32 reported zero RMS is not a certificate of an exactly zero channel vector.',
            'Official COCO segmentation metrics retain original crowd/ignore semantics and maxDets1,10,100; all380 raw GT including empty rasters remain in separate best100 diagnostics.',
            'Reusable best100 per-GT pairing is a score-free diagnostic, not AP, deployable routing, one-to-one recall or an independently trained selector.',
            'Public DEV is historically exposed development data, not blind TEST; no external method/SOTA comparison or general method verdict is established.'])
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
    require(args.owner_execute, 'Inert SOURCE requires explicit ROOT CPU execution after actual raw terminal gates')
    run(args.spec, args.acceptance)


if __name__ == '__main__':
    main()
