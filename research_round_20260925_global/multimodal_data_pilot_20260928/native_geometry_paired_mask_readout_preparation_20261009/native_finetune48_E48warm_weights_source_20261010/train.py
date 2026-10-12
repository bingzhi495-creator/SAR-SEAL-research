"""Fresh native E48-to-E96 weight-warm-start phase; inert until owned admission.

The qualified E48 predecessor supplies all 610 native weights. Its original
E24 ancestor supplies construction, native paramwise recipe and typed frozen
BN identity. Fresh AdamW moments, private order/loss RNG and a new48 scheduler
are created; no predecessor optimizer, scheduler or RNG trajectory is resumed.
Native345 active parameters, typedBN106 affine/159 buffers and native30 loss
with ten dynamic assignments are retained. Fixed phase4/8/16/24/36/48 nodes
have effective native epochs52/56/64/72/84/96. No DEV/AP or automatic retry.
ROOT executes only after separate E48 restoration/wiring CPU qualification.
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
import traceback
import types

SCHEMA = 'ROOT-native-finetune48-E48warm-weights-fresh-optimizer-spec-v1'
REPORT_SCHEMA = 'native-finetune48-E48warm-weights-fresh-optimizer-report-v1'
CP_SCHEMA = 'native-finetune48-E48warm-weights-fresh-optimizer-checkpoint-v1'
PROPOSAL_KIND = 'ROOT_INERT_NATIVE_FINETUNE48_E48WARM_WEIGHTS_SOURCE_PROPOSAL_NOT_SCIENCE_ADMISSION'
LEGACY_MODES = ('native0', 'global', 'spatial_halves', 'spatial_reversed')
EXECUTION_KINDS = ('technical_restore_four_microsteps', 'formal_additional48')
NODES = (4, 8, 16, 24, 36, 48)
FIT_IMAGES, FIT_GT = 741, 1661
PHASE_EPOCHS, PHASE_MICROSTEPS, PHASE_UPDATES, PHASE_GT = 48, 35568, 17784, 79728
INITIAL_NATIVE_EPOCH = 48
ORDER_SEED = 2026101021
LOSS_SEED_BASE = 1000000
SCHEDULE = dict(gamma=.1, milestones=[15808, 17126], warmup_start_factor=1, warmup_updates=0)
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
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def bound(row):
    require(isinstance(row, dict) and set(row) == {'path', 'bytes', 'sha256'} and
        Path(row['path']).is_absolute() and type(row['bytes']) is int and row['bytes'] > 0 and
        isinstance(row['sha256'], str) and len(row['sha256']) == 64 and
        set(row['sha256']) <= set('0123456789abcdef'), 'Exact absolute pinned reference required')
    require(reference(row['path']) == row, 'Pinned input differs: ' + row['path'])
    return Path(row['path'])


def read_ref(row):
    return read(bound(row))


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode('utf-8')).hexdigest()


def load_source(row, name):
    path = bound(row)
    require(path.suffix == '.py', 'Python SOURCE required')
    value = path.read_bytes()
    require(len(value) == row['bytes'] and hashlib.sha256(value).hexdigest() == row['sha256'],
        'SOURCE changed before import')
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(value, str(path), 'exec'), module.__dict__)
    return module


def terminal(raw, phase):
    require(raw.get('status') == 'ACTUAL_OWNED_TERMINAL' and raw.get('original_identity') is not None and
        type(raw.get('returncode')) is int and raw['returncode'] == 0 and raw.get('root_wait_reaped') is True and
        raw.get('held_handle_exit') == dict(wait_state='WAIT_OBJECT_0', actual_exit_code=0) and
        raw.get('numeric_birth_absence', {}).get('original_birth_absent') is True and
        all(key in raw and raw[key] is None for key in ('abort', 'error')) and raw.get('wait_errors') is False and
        raw.get('GPU_observation_errors') == [], 'STOP before hashes/imports: actual held0/reap/birth absence/noerrors')
    peak = raw.get('whole_GPU_observed_peak_MiB')
    require(raw.get('phase', phase) == phase and type(peak) in (int, float) and math.isfinite(peak) and
        0 <= peak <= 8192 and raw.get('whole_GPU_used_limit_MiB') == 8192 and
        raw.get('whole_leaf_VRAM_certified') is False and
        raw.get('GPU_scope') == 'whole GPU observed, not per-process WDDM allocation', 'Original owner scope differs')


def source_gate(spec_path, spec):
    # Only raw metadata reads before the first payload/SOURCE hash or import.
    raw = {}
    require(set(spec['training_receiptmap']) == set(spec['training_phasemap']) == set(LEGACY_MODES),
        'Original four teacher-training raw identities required')
    for mode in LEGACY_MODES:
        ref = spec['training_receiptmap'][mode]
        require(Path(ref['path']).is_absolute(), 'Absolute original raw receipt required')
        raw[mode] = read(ref['path'])
        terminal(raw[mode], spec['training_phasemap'][mode])
    for role, phase_role, key in (('native48_restoration_receipt', 'native48_restoration_phase', 'native48_restoration'),
            ('native48_receipt', 'native48_phase', 'native48')):
        require(Path(spec[role]['path']).is_absolute(), 'Absolute E48 ancestral raw required')
        raw[key] = read(spec[role]['path'])
        terminal(raw[key], spec[phase_role])
    if spec['execution_kind'] == 'formal_additional48':
        require(Path(spec['restoration_receipt']['path']).is_absolute(), 'Absolute fresh technical raw required')
        raw['restoration'] = read(spec['restoration_receipt']['path'])
        terminal(raw['restoration'], spec['restoration_phase'])
    require(Path(spec['GPU_admission_file']).is_absolute(), 'Absolute fresh owned admission required')
    admission = read(spec['GPU_admission_file'])
    source, spec_ref = reference(__file__), reference(spec_path)
    require(admission['status'] == 'ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE' and
        admission['automatic_retry'] is False and admission['phase'] == spec['phase'] and
        admission['allowed_cpus'] == spec['CPU_affinity'] == [0, 1] and admission['cuda_visible_devices'] == '0' and
        admission['qualified_leaf_deadline_seconds'] == spec['deadline_seconds'] and
        admission['whole_GPU_used_limit_MiB'] == spec['whole_GPU_owner_limit_MiB'] == 8192 and
        Path(admission['entry']).resolve(strict=True) == Path(source['path']) and
        admission['entry_args'] == ['--spec', str(spec_path), '--owner-execute'] and
        admission['driver_sha256'] == OWNER_SHA, 'Fresh current owner/admission/entry/resources differ')
    require(spec['entry_source'] == source and source in spec['sources'] and spec['CPU_reader'] in spec['sources'] and
        len({row['path'] for row in spec['sources']}) == len(spec['sources']), 'Fresh source/reader closure differs')
    hashes = admission['bound_source_sha256']
    require(hashes.get(str(spec_path)) == spec_ref['sha256'], 'Current SPEC is not owner-bound')
    for ref in spec['sources']:
        bound(ref)
        require(hashes.get(ref['path']) == ref['sha256'], 'Current complete closure is not owner-bound')
    if 'scientific_spec' in admission:
        require(admission['scientific_spec'] == spec_ref, 'Current admitted scientific SPEC differs')
    proposal = read_ref(spec['SOURCE_proposal'])
    require(proposal['kind'] == PROPOSAL_KIND and proposal['schema'] == SCHEMA and
        proposal['producer_SOURCE'] == source and
        spec['SOURCE_proposal'] in spec['sources'] and proposal['phase_epochs'] == PHASE_EPOCHS and proposal['teacher_epoch'] == 48 and
        proposal['phase_microsteps'] == PHASE_MICROSTEPS and proposal['phase_updates'] == PHASE_UPDATES and
        proposal['phase_raw_GT_visits'] == PHASE_GT and proposal['phase_checkpoint_nodes'] == list(NODES) and
        proposal['phase_loss_seed_base'] == LOSS_SEED_BASE and spec['CPU_reader'] == proposal['CPU_reader_SOURCE'] and
        proposal['order_seed'] == ORDER_SEED and proposal['old_optimizer_restored'] is False and
        proposal['old_RNG_restored'] is False, 'Canonical inert SOURCE proposal differs')
    for role, ref in proposal['immutable_SOURCE_refs'].items():
        require(spec[role] == ref and ref in spec['sources'], 'Original immutable SOURCE role differs: ' + role)
    for role, ref in proposal['library_SOURCE_refs'].items():
        require(spec[role] == ref and ref in spec['sources'], 'Bound native library SOURCE differs: ' + role)
    return raw, source, spec_ref, reference(spec['GPU_admission_file'])


def validate_phase_recipe(spec, training):
    require(spec['initializers'] == training['initializers'], 'Original qualified native paramwise recipe required')
    require(spec['schedule'] == SCHEDULE,
        'Explicit new phase48 no-warmup gamma.1 native schedule required')
    entry = spec['initializers']['mask2former']
    require(entry['optimizer']['type'] == 'AdamW' and entry['optimizer']['lr'] == .0001 and
        entry['clip_grad'] == dict(max_norm=.01, norm_type=2), 'Original native AdamW baseLR1e-4/clip.01 required')
    require(spec['backbone_with_cp'] is training['backbone_with_cp'] and
        spec['order_seed'] == ORDER_SEED and type(spec['phase_loss_seed_base']) is int and
        spec['phase_loss_seed_base'] == LOSS_SEED_BASE,
        'Original checkpointing and explicit new phase RNG policy required')
    for key in ('image_root', 'native_helpers', 'initializer_source', 'micro_preflight_source'):
        require(spec[key] == training[key], 'Original native construction/data differs: ' + key)
    require(spec['FIT'] == training['FIT'], 'Exactly original FIT741/1661 required')


def prepare_training_model(torch, pref, initializer, common, registry, revert, Config, classes, spec, teacher_row, predecessor_row):
    """Original E24 construction/BN proof, then strict qualified E48 weights."""
    train, q = teacher_row['training_spec'], teacher_row['qualification']
    validate_phase_recipe(spec, train)
    initialized = read_ref(train['initializer_qualification'])
    model, initial, initialization = pref.prepare_cpu(torch, initializer, registry, revert, Config, classes,
        train, 'mask2former', initialized['initializers']['mask2former'])
    cp = torch.load(str(bound(spec['checkpointmap']['native0'])), map_location='cpu')
    require(set(cp) == {'state_dict', 'optimizer_state_dict', 'external_rng_state', 'private_order_state', 'meta'},
        'Complete original qualified checkpoint container required')
    state, meta = cp['state_dict'], cp['meta']
    require(meta['schema'] == common.CP_SCHEMA and meta['kind'] == 'mask2former' and meta['mode'] == 'native0' and
        meta['source'] == spec['training_source'] and meta['spec'] == q['spec'] and meta['phase'] == q['phase'] and
        meta['training_protocol'] == q['training_protocol'] and meta['initial_epoch'] == 0 and meta['completed_epoch'] == 24 and
        meta['global_microsteps'] == 17784 and meta['global_updates'] == 8892 and meta['consumed_raw_GT'] == 39864 and
        meta['no_pending_accumulation_gradients'] is True and meta['not_technical_preflight_checkpoint'] is True and
        meta['fresh_single_owned_24'] is True and meta['eligible_for_automatic_resume'] is False and
        meta['resume_checkpoint'] is None and meta['resume_qualification'] is None, 'Exact qualified native0 E24 metadata required')
    current = model.state_dict()
    require(len(state) == len(current) == 610 and len(list(model.parameters())) == 451 and state.keys() == current.keys() and
        all(isinstance(v, torch.Tensor) and v.device.type == 'cpu' and v.shape == current[n].shape and
            v.dtype == current[n].dtype and bool(torch.isfinite(v).all()) for n, v in state.items()),
        'Original native610 keys/shapes/dtypes/finite differ')
    proof = read_ref(q['model_proof']['proof_refs']['state_proof'])
    require(pref.state_proof(state) == proof['after'], 'E24 native weights differ from independent qualified proof')
    loaded = model.load_state_dict(state, strict=True)
    require(not loaded.missing_keys and not loaded.unexpected_keys and
        all(torch.equal(v.detach().cpu(), state[n]) for n, v in model.state_dict().items()), 'Strict E24 load differs')
    anchor = pref.cpu_tree(torch, state)
    policy = pref.trainability(torch, model)
    policy['kind'] = 'mask2former'
    require(policy['trainable_tensors'] == 345 and policy['parameter_tensors'] == 451 and policy['state_entries'] == 610 and
        policy['frozen_tensors'] == 106 and len(policy['BN_buffer_names']) == 159 and len(policy['backbone_conv_names']) == 53,
        'Only typedBN106 affine frozen, all native345 active required')
    require(all(torch.equal(anchor[n], initial[n]) for n in policy['frozen_names'] + policy['BN_buffer_names']),
        'Qualified E24 typedBN must remain original initializer-exact')
    require(all(v.dtype == torch.float32 for v in model.parameters()) and model.backbone.with_cp is spec['backbone_with_cp'],
        'Original FP32/checkpointing policy differs')

    predecessor = predecessor_row
    cp48 = torch.load(str(bound(spec['native48_checkpoint'])), map_location='cpu')
    require(set(cp48) == {'state_dict', 'optimizer_state_dict', 'external_rng_state', 'private_order_state', 'meta'},
        'Complete qualified native E48 checkpoint container required')
    state48, meta48 = cp48['state_dict'], cp48['meta']
    previous = predecessor['training_spec']
    require(meta48['schema'] == 'native-finetune24-fresh-optimizer-checkpoint-v1' and
        meta48['kind'] == 'mask2former' and meta48['mode'] == 'native0' and
        meta48['source'] == spec['native48_source'] and meta48['spec'] == spec['native48_spec'] and
        meta48['phase'] == spec['native48_phase'] and meta48['execution_kind'] == 'formal_additional24' and
        meta48['teacher_checkpoint'] == spec['checkpointmap']['native0'] and meta48['teacher_epoch'] == 24 and
        meta48['phase_initial_epoch'] == 0 and meta48['completed_phase_epoch'] == 24 and
        meta48['effective_native_epoch'] == 48 and meta48['phase_microsteps'] == 17784 and
        meta48['phase_updates'] == 8892 and meta48['phase_raw_GT_visits'] == 39864 and
        meta48['effective_native_microsteps'] == 35568 and meta48['effective_native_updates'] == 17784 and
        meta48['effective_native_raw_GT_visits'] == 79728 and
        meta48['phase_recipe'] == spec['initializers']['mask2former'] and meta48['schedule'] == previous['schedule'] and
        meta48['fresh_optimizer_at_phase_start'] is True and meta48['old_optimizer_restored'] is False and
        meta48['old_RNG_restored'] is False and meta48['no_pending_accumulation_gradients'] is True and
        meta48['formal_resume_eligible'] is False,
        'Strict qualified E48 metadata required; this is weights warm start, not resume')
    original_anchor = anchor
    require(state48.keys() == original_anchor.keys() and len(state48) == 610 and
        all(isinstance(v, torch.Tensor) and v.device.type == 'cpu' and
            v.shape == original_anchor[n].shape and v.dtype == original_anchor[n].dtype and
            bool(torch.isfinite(v).all()) for n, v in state48.items()) and
        all(torch.equal(state48[n], original_anchor[n]) for n in policy['frozen_names'] + policy['BN_buffer_names']),
        'E48 native610 keys/types/finite or original E24 typedBN differs')
    proof48 = predecessor['state_proof']
    require(proof48['schema'] == 'native-finetune24-fresh-optimizer-node-state-proof-v1' and
        proof48['source'] == spec['native48_source'] and proof48['spec'] == spec['native48_spec'] and
        proof48['completed_phase_epoch'] == 24 and proof48['effective_native_epoch'] == 48 and
        proof48['phase_microsteps'] == 17784 and proof48['phase_updates'] == 8892 and
        proof48['phase_raw_GT_visits'] == 39864 and proof48['frozen_BN_all_exact'] is True and
        proof48['no_pending_accumulation_gradients'] is True and proof48['formal_resume_eligible'] is False,
        'Qualified E48 node proof differs')

    require(pref.state_proof(state48) == proof48['after'], 'Qualified E48 full610 state proof differs')
    loaded48 = model.load_state_dict(state48, strict=True)
    require(not loaded48.missing_keys and not loaded48.unexpected_keys and
        all(torch.equal(v.detach().cpu(), state48[n]) for n, v in model.state_dict().items()),
        'Strict E48 full610 warm load differs')
    anchor = pref.cpu_tree(torch, state48)
    loading = dict(schema='native-finetune48-E48warm-weights-fresh-optimizer-loading-proof-v1', teacher_checkpoint=spec['native48_checkpoint'],
        teacher_qualification=spec['native48_CPU_qualification'], teacher_training_source=spec['native48_source'],
        teacher_epoch=48, original_teacher_checkpoint=spec['checkpointmap']['native0'],
        original_teacher_qualification=spec['qualificationmap']['native0'], original_teacher_epoch=24,
        original_teacher_state=pref.state_proof(original_anchor), predecessor_spec=spec['native48_spec'],
        predecessor_state_proof=predecessor['node']['state_proof'], weights_warm_start_only=True, strict_full_native610_loaded=True, original_initialization=initialization,
        teacher_state=pref.state_proof(anchor), typed_BN_initializer_exact=True, native_policy=policy,
        old_checkpoint_optimizer_and_RNG_containers_deserialized=True, old_optimizer_restored=False, old_RNG_restored=False,
        old_automatic_resume_eligible=False, fresh_phase_not_formal_resume=True, registered_adapter_tensors=0)
    del cp, cp48, state, state48, current, initial, proof, proof48, original_anchor, predecessor
    return model, anchor, policy, loading


def qualified_native48(spec, raw):
    """Validate the actual qualified E48 predecessor; no optimizer/RNG restore."""
    roles = ('native48_source', 'native48_CPU_reader', 'native48_SOURCE_proposal',
        'native48_spec', 'native48_CPU_qualification', 'native48_receipt',
        'native48_GPU_admission', 'native48_checkpoint', 'native48_restoration_receipt')
    for role in roles:
        require(spec[role] in spec['sources'], 'E48 lineage absent from current closure: ' + role)
        bound(spec[role])
    q = read_ref(spec['native48_CPU_qualification'])
    previous = read_ref(spec['native48_spec'])
    admission = read_ref(spec['native48_GPU_admission'])
    require(q['schema'] == 'ROOT-native-finetune24-fresh-optimizer-CPU-qualification-v1' and
        q['status'] == 'ROOT_NATIVE_FINETUNE24_FRESH_OPTIMIZER_CPU_QUALIFIED' and
        q['source'] == spec['native48_source'] and q['reader_source'] == spec['native48_CPU_reader'] and
        q['SOURCE_proposal'] == spec['native48_SOURCE_proposal'] and q['spec'] == spec['native48_spec'] and
        q['phase'] == spec['native48_phase'] and q['parent_receipt'] == spec['native48_receipt'] and
        q['GPU_admission'] == spec['native48_GPU_admission'] and
        q['checkpoint'] == spec['native48_checkpoint'] == q['nodes']['24']['checkpoint'] and
        q['teacher_checkpoint'] == spec['checkpointmap']['native0'] and
        q['teacher_qualification'] == spec['qualificationmap']['native0'] and
        q['formal_additional24_completed'] is True and
        q['formal_training_completion_CPU_state_qualified'] is True and q['formal_resume_qualified'] is False and
        q['initial_native610_CPU_exact'] is True and q['typed_BN106_affine159_buffers_exact'] is True and
        q['native345_active'] is True and q['old_optimizer_restored'] is False and q['old_RNG_restored'] is False and
        q['completed_phase_epochs'] == 24 and q['effective_native_epoch'] == 48 and
        q['phase_microsteps'] == 17784 and q['phase_updates'] == 8892 and q['phase_raw_GT_visits'] == 39864,
        'Exact completed original native additional24 / effectiveE48 qualification required')
    require(previous['schema'] == 'ROOT-native-finetune24-fresh-optimizer-spec-v1' and
        previous['execution_kind'] == 'formal_additional24' and previous['phase'] == spec['native48_phase'] and
        previous['entry_source'] == spec['native48_source'] and previous['CPU_reader'] == spec['native48_CPU_reader'] and
        previous['SOURCE_proposal'] == spec['native48_SOURCE_proposal'] and previous['teacher_epoch'] == 24 and
        previous['phase_epochs'] == previous['expected_epoch'] == 24 and previous['phase_microsteps'] == 17784 and
        previous['phase_updates'] == 8892 and previous['phase_raw_GT_visits'] == 39864 and
        previous['schedule'] == dict(gamma=.1, milestones=[7904,8563], warmup_start_factor=1, warmup_updates=0) and
        all(previous[k] == spec[k] for k in ('initializers', 'backbone_with_cp', 'FIT', 'image_root',
            'checkpointmap', 'qualificationmap', 'training_receiptmap', 'training_phasemap', 'training_source',
            'training_CPU_reader', 'native_helpers', 'initializer_source', 'micro_preflight_source')) and
        previous['restoration_receipt'] == spec['native48_restoration_receipt'] and
        previous['restoration_phase'] == spec['native48_restoration_phase'] and
        q['technical_restoration_qualification'] == previous['restoration_CPU_qualification'],
        'Actual E48 predecessor SPEC/original recipe/technical ancestry differs')
    require(raw['native48']['admission_sha256'] == spec['native48_GPU_admission']['sha256'] and
        admission['status'] == 'ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE' and
        admission['automatic_retry'] is False and admission['phase'] == spec['native48_phase'] and
        admission['allowed_cpus'] == previous['CPU_affinity'] == [0,1] and admission['cuda_visible_devices'] == '0' and
        admission['qualified_leaf_deadline_seconds'] == previous['deadline_seconds'] and
        admission['whole_GPU_used_limit_MiB'] == previous['whole_GPU_owner_limit_MiB'] == 8192 and
        Path(admission['entry']).resolve(strict=True) == Path(spec['native48_source']['path']) and
        admission['entry_args'] == ['--spec', spec['native48_spec']['path'], '--owner-execute'] and
        admission['driver_sha256'] == q['owner_driver']['sha256'] == OWNER_SHA,
        'Actual E48 predecessor raw/admission/entry/resources differ')
    hashes = admission['bound_source_sha256']
    require(hashes.get(spec['native48_spec']['path']) == spec['native48_spec']['sha256'],
        'Original E48 SPEC is not owner-bound')
    for ref in previous['sources']:
        bound(ref)
        require(ref in spec['sources'] and hashes.get(ref['path']) == ref['sha256'],
            'Preserve the complete admitted E48 predecessor closure')
    if 'scientific_spec' in admission:
        require(admission['scientific_spec'] == spec['native48_spec'], 'E48 admitted SPEC differs')
    for ref in [q[k] for k in ('report', 'start', 'owner_driver')] + list(q['proof_refs'].values()) + [
            q['nodes']['24']['state_proof'], previous['restoration_CPU_qualification']]:
        bound(ref)
        require(ref in spec['sources'], 'Qualified E48 proof absent from current closure')
    technical = read_ref(previous['restoration_CPU_qualification'])
    require(technical['schema'] == 'ROOT-native-finetune24-fresh-optimizer-restoration-CPU-qualification-v1' and
        technical['status'] == 'ROOT_NATIVE_FINETUNE24_FRESH_OPTIMIZER_RESTORATION_CPU_QUALIFIED' and
        technical['source'] == spec['native48_source'] and technical['reader_source'] == spec['native48_CPU_reader'] and
        technical['parent_receipt'] == spec['native48_restoration_receipt'] and
        technical['phase'] == spec['native48_restoration_phase'] and
        technical['teacher_checkpoint'] == spec['checkpointmap']['native0'] and
        technical['teacher_qualification'] == spec['qualificationmap']['native0'] and
        raw['native48_restoration']['admission_sha256'] == technical['GPU_admission']['sha256'],
        'E48 predecessor technical raw/qualification ancestry differs')
    for role in ('spec', 'report', 'start', 'GPU_admission', 'owner_driver'):
        bound(technical[role])
        require(technical[role] in spec['sources'], 'E48 technical ancestry absent from current closure')
    node = q['nodes']['24']
    require(node['full_native610_CPU_verified'] is True and node['typed_BN106_affine159_buffers_exact'] is True and
        node['full_AdamW_moments_and_step_counts_verified'] is True and node['effective_native_epoch'] == 48 and
        node['completed_phase_epoch'] == 24 and node['phase_microsteps'] == 17784 and
        node['phase_updates'] == 8892 and node['phase_raw_GT_visits'] == 39864,
        'Qualified final E48 native610 node differs')
    return dict(training_spec=previous, qualification=q, node=node, state_proof=read_ref(node['state_proof']))


def restoration_gate(spec, source):
    if spec['execution_kind'] != 'formal_additional48':
        return None
    for role in ('restoration_CPU_qualification', 'restoration_CPU_reader', 'restoration_spec', 'restoration_receipt'):
        require(spec[role] in spec['sources'], 'Technical lineage role absent from current owner closure: ' + role)
    q = read_ref(spec['restoration_CPU_qualification'])
    require(q['schema'] == 'ROOT-native-finetune48-E48warm-weights-fresh-optimizer-restoration-CPU-qualification-v1' and
        q['status'] == 'ROOT_NATIVE_FINETUNE48_E48WARM_WEIGHTS_FRESH_OPTIMIZER_RESTORATION_CPU_QUALIFIED' and
        q['source'] == source and q['reader_source'] == spec['restoration_CPU_reader'] and
        q['spec'] == spec['restoration_spec'] and q['parent_receipt'] == spec['restoration_receipt'] and
        q['phase'] == spec['restoration_phase'] and q['teacher_checkpoint'] == spec['native48_checkpoint'] and
        q['teacher_qualification'] == spec['native48_CPU_qualification'] and
        q['initial_native610_CPU_exact'] is True and q['typed_BN106_affine159_buffers_exact'] is True and
        q['native345_active'] is True and q['fresh_optimizer_state_entries'] == 0 and
        q['old_optimizer_restored'] is False and q['old_RNG_restored'] is False and
        q['native_loss_atoms'] == 30 and q['technical_microsteps'] == 4 and q['technical_updates'] == 2 and
        q['formal_training_or_resume_qualified'] is False, 'Separate fresh full-native restoration CPU qualification required')
    for role in ('source', 'reader_source', 'spec', 'report', 'start', 'parent_receipt', 'GPU_admission', 'owner_driver'):
        bound(q[role])
    technical = read_ref(spec['restoration_spec'])
    raw = read_ref(q['parent_receipt'])
    admission = read_ref(q['GPU_admission'])
    require(technical['execution_kind'] == 'technical_restore_four_microsteps' and
        technical['entry_source'] == source and technical['CPU_reader'] == q['reader_source'] and
        technical['phase'] == q['phase'] and
        all(technical[k] == spec[k] for k in ('initializers', 'schedule', 'backbone_with_cp', 'order_seed',
            'phase_loss_seed_base', 'checkpointmap', 'qualificationmap', 'FIT', 'image_root',
            'native48_checkpoint', 'native48_CPU_qualification', 'native48_spec', 'native48_receipt',
            'native48_GPU_admission', 'native48_phase', 'native48_restoration_receipt', 'native48_restoration_phase')),
        'Technical restoration and formal fresh phase recipes differ')
    require(raw['admission_sha256'] == q['GPU_admission']['sha256'] and
        admission['status'] == 'ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE' and
        admission['automatic_retry'] is False and admission['phase'] == q['phase'] and
        admission['allowed_cpus'] == technical['CPU_affinity'] == [0, 1] and admission['cuda_visible_devices'] == '0' and
        admission['qualified_leaf_deadline_seconds'] == technical['deadline_seconds'] and
        admission['whole_GPU_used_limit_MiB'] == technical['whole_GPU_owner_limit_MiB'] == 8192 and
        Path(admission['entry']).resolve(strict=True) == Path(source['path']) and
        admission['entry_args'] == ['--spec', q['spec']['path'], '--owner-execute'] and
        admission['driver_sha256'] == q['owner_driver']['sha256'] == OWNER_SHA,
        'Actual technical raw/admission/source/SPEC/owner join differs')
    hashes = admission['bound_source_sha256']
    require(hashes.get(q['spec']['path']) == q['spec']['sha256'], 'Technical SPEC is not owner-bound')
    for ref in technical['sources']:
        bound(ref)
        require(ref in spec['sources'] and hashes.get(ref['path']) == ref['sha256'],
            'Technical complete closure must be preserved and owner-bound')
    if 'scientific_spec' in admission:
        require(admission['scientific_spec'] == q['spec'], 'Technical admitted SPEC differs')
    return q


def imported_library_gate(torch, model, spec):
    head = model.panoptic_head
    actual = dict(native_mask_head_SOURCE=type(head), native_parent_head_SOURCE=type(head).__mro__[1],
        panoptic_GT_SOURCE=head.preprocess_gt.__func__.__globals__['preprocess_panoptic_gt'],
        hungarian_SOURCE=type(head.assigner), mask_sampler_SOURCE=type(head.sampler),
        mask_sampling_result_SOURCE=head.sampler.sample.__func__.__globals__['MaskSamplingResult'],
        point_sample_SOURCE=head._get_targets_single.__func__.__globals__['point_sample'],
        adamw_SOURCE=torch.optim.AdamW, clip_grad_SOURCE=torch.nn.utils.clip_grad_norm_)
    for role, obj in actual.items():
        require(Path(inspect.getsourcefile(obj)).resolve() == bound(spec[role]).resolve(),
            'Actual native library SOURCE differs: ' + role)
    require(torch.__version__ == '2.0.1+cu118' and
        (Path(torch.__file__).resolve().parent / 'version.py').resolve() == bound(spec['torch_version_source']).resolve(),
        'Bound original Torch version differs')
    require(head.num_queries == 100 and head.num_classes == 1 and head.num_points == 12544 and
        len(head.assigner.match_costs) == 3 and
        [type(c).__name__ for c in head.assigner.match_costs] == ['ClassificationCost', 'CrossEntropyLossCost', 'DiceCost'] and
        [c.weight for c in head.assigner.match_costs] == [2., 5., 5.], 'Original native sampled matching recipe differs')
    for cost in head.assigner.match_costs:
        require(Path(inspect.getsourcefile(type(cost))).resolve() == bound(spec['match_cost_SOURCE']).resolve(),
            'Actual weighted native cost SOURCE differs')


def pending_gradient_proof(pref, named, active_names):
    """Bounded odd-epoch audit of the live accumulated gradients, not replay."""
    return {name: None if named[name].grad is None else pref.digest_tensor(named[name].grad)
        for name in active_names}


def save_phase_node(torch, pref, common, model, optimizer, names, base_lrs, spec, spec_ref, output,
                    anchor, policy, audit, loading, epoch, micro, updates, GT_visits, order_state, activity, counts):
    require(epoch in NODES and micro == epoch * 741 and updates == micro // 2 and GT_visits == epoch * 1661 and
        all(p.grad is None for p in model.parameters()), 'Even-node counters/no pending gradients differ')
    state = pref.cpu_tree(torch, model.state_dict())
    require(state.keys() == anchor.keys() and all(bool(torch.isfinite(v).all()) for v in state.values()) and
        all(torch.equal(state[n], anchor[n]) for n in policy['frozen_names'] + policy['BN_buffer_names']),
        'Native finite state or frozen BN changed')
    factor = common.factor(spec['schedule'], updates - 1)
    require(all(g['lr'] == base * factor for g, base in zip(optimizer.param_groups, base_lrs)), 'Stored phase LR differs')
    changed = {n: not torch.equal(state[n], anchor[n]) for n in state}
    meta = dict(schema=CP_SCHEMA, kind='mask2former', mode='native0', phase=spec['phase'], source=reference(__file__), spec=spec_ref,
        execution_kind=spec['execution_kind'], teacher_checkpoint=spec['native48_checkpoint'], teacher_epoch=48,
        phase_initial_epoch=0, completed_phase_epoch=epoch, effective_native_epoch=48 + epoch,
        phase_microsteps=micro, phase_updates=updates, phase_raw_GT_visits=GT_visits,
        effective_native_microsteps=35568 + micro, effective_native_updates=17784 + updates,
        effective_native_raw_GT_visits=79728 + GT_visits, phase_recipe=spec['initializers']['mask2former'],
        schedule=spec['schedule'], base_lrs=base_lrs, stored_lr_factor=factor, optimizer_param_names=names,
        fresh_optimizer_at_phase_start=True, old_optimizer_restored=False, old_RNG_restored=False,
        no_pending_accumulation_gradients=True, formal_resume_eligible=False)
    checkpoint = dict(state_dict=state, optimizer_state_dict=pref.cpu_tree(torch, optimizer.state_dict()),
        external_rng_state=pref.cpu_tree(torch, pref.rng_snapshot(torch)),
        private_order_state=dict(seed=ORDER_SEED, next_phase_epoch=epoch, torch_CPU_generator_state=order_state.cpu().clone()), meta=meta)
    path = output / ('NODE_PHASE_E%02d_NATIVE_E%02d.pth' % (epoch, 48 + epoch))
    with path.open('xb') as stream:
        torch.save(checkpoint, stream)
    loaded = torch.load(path, map_location='cpu')
    require(pref.tree_equal(torch, checkpoint, loaded), 'Native full model/optimizer/RNG node readback differs')
    proof = dict(schema='native-finetune48-E48warm-weights-fresh-optimizer-node-state-proof-v1', source=reference(__file__), spec=spec_ref,
        teacher_loading=loading, before=pref.state_proof(anchor), after=pref.state_proof(state), changed=changed,
        policy=policy, optimizer=audit, actual_optimizer=common.optimizer_summary(torch, pref, optimizer, names),
        positive_accumulated_gradient_microcounts=dict(activity), optimizer_step_nonNone_gradient_counts=dict(counts),
        completed_phase_epoch=epoch, effective_native_epoch=48 + epoch, phase_microsteps=micro, phase_updates=updates,
        phase_raw_GT_visits=GT_visits, frozen_BN_all_exact=True, CPU_checkpoint_readback_exact=True,
        no_pending_accumulation_gradients=True, formal_resume_eligible=False)
    proof_path = output / ('NODE_PHASE_E%02d_STATE_PROOF.json' % epoch)
    write(proof_path, proof)
    return dict(checkpoint=reference(path), state_proof=reference(proof_path), completed_phase_epoch=epoch,
        effective_native_epoch=48 + epoch, phase_microsteps=micro, phase_updates=updates, phase_raw_GT_visits=GT_visits,
        stored_lr_factor=factor, frozen_BN_all_exact=True, CPU_checkpoint_readback_exact=True)


def train_phase(torch, pref, old, common, model, anchor, policy, loading, spec, spec_ref, output, dataset, coco, started):
    model.to('cuda:0')
    policy_after = pref.trainability(torch, model)
    policy_after['kind'] = 'mask2former'
    require(policy_after == policy, 'Training policy changed after CUDA placement')
    optimizer, audit = pref.make_optimizer(torch, model, spec['initializers']['mask2former'], policy)
    require(len(optimizer.state) == 0 and len(optimizer.param_groups) == 345, 'Fresh native345-group empty AdamW required')
    names = [[row['name']] for row in audit['parameter_groups']]
    base_lrs = [g['lr'] for g in optimizer.param_groups]
    named = dict(model.named_parameters())
    write(output / 'START_STATE_PROOF.json', dict(schema='native-finetune48-E48warm-weights-fresh-optimizer-start-state-v1', source=reference(__file__),
        spec=spec_ref, teacher_loading=loading, actual_native_state=pref.state_proof(model.state_dict()), policy=policy,
        optimizer=audit, actual_optimizer=common.optimizer_summary(torch, pref, optimizer, names), base_lrs=base_lrs,
        external_RNG=[pref.digest_tensor(v) for v in pref.rng_snapshot(torch)], fresh_optimizer_state_entries=0,
        old_optimizer_restored=False, old_RNG_restored=False, phase_microsteps=0, phase_updates=0,
        initial_native_epoch=48, phase_initial_epoch=0, fresh_phase_not_formal_resume=True))
    technical = spec['execution_kind'] == 'technical_restore_four_microsteps'
    limit = 4 if technical else PHASE_MICROSTEPS
    order_rng = torch.Generator(device='cpu').manual_seed(ORDER_SEED)
    orders, epochs, nodes = [], [], {}
    activity, counts = ({n: 0 for n in policy['active_names']} for _ in range(2))
    updates, micro_count, GT_visits = 0, 0, 0
    pending_boundary = None
    optimizer.zero_grad(set_to_none=True)
    try:
        with (output / 'TRACE.jsonl').open('x', encoding='utf-8', newline='\n') as stream:
            for epoch in range(PHASE_EPOCHS):
                before_rng = order_rng.get_state().clone()
                order = torch.randperm(741, generator=order_rng).tolist()
                require(sorted(order) == list(range(741)), 'Complete private phase permutation required')
                after_rng = order_rng.get_state().clone()
                orders.append(dict(phase_epoch=epoch, order_seed=ORDER_SEED, dataset_indices=order,
                    image_ids=[int(dataset.get_data_info(i)['img_id']) for i in order],
                    private_CPU_RNG_before=pref.digest_tensor(before_rng), private_CPU_RNG_after=pref.digest_tensor(after_rng)))
                epoch_GT, initial_updates = 0, updates
                for within, index in enumerate(order):
                    pref.check_deadline(started, spec['deadline_seconds'])
                    micro = epoch * 741 + within
                    require(micro == micro_count and micro < limit, 'Phase micro counter differs')
                    boundary_join = None
                    if within == 0 and micro % 2 == 1:
                        require(pending_boundary is not None and pending_boundary['phase_microstep'] == micro - 1,
                            'Missing odd741 pending-gradient boundary')
                        before_forward = pending_gradient_proof(pref, named, policy['active_names'])
                        require(before_forward == pending_boundary['gradients'], 'Odd741 pending gradients changed before next forward')
                        boundary_join = dict(previous_phase_microstep=micro-1,
                            after_previous_odd_epoch=pending_boundary['gradients'], before_next_forward=before_forward,
                            exact=True, SOURCE_observed_not_CPU_gradient_replay=True)
                        pending_boundary = None
                    factor = common.factor(spec['schedule'], updates)
                    for g, base in zip(optimizer.param_groups, base_lrs):
                        g['lr'] = base * factor
                    data, stripped, gray, raw_gray, identity = old.load_image(model, dataset, coco, dataset.get_data_info(index))
                    del stripped, gray, raw_gray
                    identity['dataset_index'] = index
                    targets = pref.check_targets(torch, data, identity)
                    external = pref.rng_snapshot(torch)
                    seed = spec['phase_loss_seed_base'] + micro
                    with torch.random.fork_rng(devices=[0]):
                        torch.random.default_generator.manual_seed(seed)
                        torch.cuda.manual_seed(seed)
                        private_before = pref.rng_snapshot(torch)
                        with common.observe_native_loss(torch, pref, old, None, None, model.panoptic_head, data, 'native0', micro) as observation:
                            losses = model.loss(data['inputs'], data['data_samples'])
                        total, atoms = pref.loss_terms(torch, losses)
                        require(len(atoms) == 30 and len(observation['matches']) == 10, 'Original native30/ten dynamic assignments required')
                        for match in observation['matches']:
                            qs, gs = match['positive_query_indices'], match['assigned_GT_indices']
                            require(len(qs) == len(gs) == len(set(qs)) == len(identity['ann_ids']) and
                                sorted(gs) == list(range(len(identity['ann_ids']))) and
                                all(0 <= q < 100 for q in qs), 'Every raw GT exactly once in each native assignment')
                        (total / 2).backward()
                        private_after = pref.rng_snapshot(torch)
                    external_after = pref.rng_snapshot(torch)
                    require(all(torch.equal(a, b) for a, b in zip(external, external_after)), 'Private native-loss RNG not restored')
                    grad_names = [n for n in policy['active_names'] if named[n].grad is not None]
                    gradients = [named[n].grad.detach().float().norm() for n in grad_names]
                    values = torch.stack(gradients).cpu().tolist() if gradients else []
                    norm_map = dict(zip(grad_names, values))
                    norms = {n: norm_map.get(n) for n in policy['active_names']}
                    require(all(v is None or math.isfinite(v) for v in norms.values()), 'Nonfinite accumulated native gradients')
                    for n, v in norms.items():
                        activity[n] += int(v is not None and v > 0)
                    require(all(named[n].grad is None for n in policy['frozen_names']), 'Frozen typedBN affine received gradients')
                    BN_modes = {n: m.training for n, m in model.named_modules() if n in policy['BN_modules']}
                    require(not any(BN_modes.values()), 'TypedBN left eval mode')
                    stepped, global_norm = (micro + 1) % 2 == 0, None
                    if stepped:
                        norm = torch.nn.utils.clip_grad_norm_([named[n] for n in policy['active_names']], max_norm=.01, norm_type=2)
                        require(bool(torch.isfinite(norm)), 'Nonfinite native global norm')
                        global_norm = float(norm.detach().cpu())
                        for n, v in norms.items():
                            counts[n] += int(v is not None)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        updates += 1
                    GT_visits += len(identity['ann_ids'])
                    epoch_GT += len(identity['ann_ids'])
                    micro_count += 1
                    odd_epoch_end = None
                    if within == 740 and micro_count % 2 == 1:
                        require(not stepped and pending_boundary is None, 'Odd741 epoch must retain one accumulation microstep')
                        odd_epoch_end = pending_gradient_proof(pref, named, policy['active_names'])
                        pending_boundary = dict(phase_microstep=micro, gradients=odd_epoch_end)
                    row = dict(mode='native0', execution_kind=spec['execution_kind'], phase_microstep=micro, phase_epoch=epoch,
                        effective_native_epoch_index=48 + epoch, within_epoch=within, identity=identity, GT=targets,
                        input=pref.digest_tensor(data['inputs']), loss_fields=atoms, native_total=float(total.detach().cpu()),
                        native_model_loss_only=True, original_native_loss_atoms=30, native_matches=observation['matches'],
                        native_head_fields=observation['head_fields'], first_forward_digest_scope='phase_micro0_from_qualified_nativeE48',
                        accumulation_divisor=2, optimizer_step=stepped, phase_updates=updates,
                        scheduler_u_before_step=updates - int(stepped), lr_factor=factor, gradient_clipping_applied=stepped,
                        preclip_global_norm=global_norm, gradient_norms_aligned_active_names=[norms[n] for n in policy['active_names']],
                        gradients_are_accumulated_preclip=True, frozen_BN_parameter_grads_None=True, BN_module_training=BN_modes,
                        odd_epoch_end_pending_gradients=odd_epoch_end, pending_boundary_gradient_join=boundary_join,
                        private_seed=seed, private_RNG_before=[pref.digest_tensor(v) for v in private_before],
                        private_RNG_after=[pref.digest_tensor(v) for v in private_after],
                        external_RNG_before=[pref.digest_tensor(v) for v in external],
                        external_RNG_after=[pref.digest_tensor(v) for v in external_after], external_RNG_restored=True)
                    stream.write(json.dumps(row, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n')
                    if micro_count % 100 == 0 or micro_count == limit:
                        stream.flush()
                        print(json.dumps(dict(phase=spec['phase'], phase_microsteps=micro_count, phase_updates=updates,
                            phase_raw_GT_visits=GT_visits, elapsed_seconds=time.monotonic() - started), sort_keys=True), flush=True)
                    del data, losses, total, observation, gradients, private_before, private_after, external, external_after
                    if stepped:
                        del norm
                    if micro_count == limit:
                        break
                if not technical:
                    require(epoch_GT == 1661 and within == 740, 'Complete741/1661 native phase epoch required')
                    epochs.append(dict(phase_epoch=epoch, images=741, raw_GT=epoch_GT, phase_microsteps=micro_count,
                        epoch_updates=updates-initial_updates, phase_updates=updates, pending_accumulation_microsteps=micro_count % 2))
                    if epoch + 1 in NODES:
                        stream.flush()
                        nodes[str(epoch+1)] = save_phase_node(torch, pref, common, model, optimizer, names, base_lrs, spec, spec_ref,
                            output, anchor, policy, audit, loading, epoch+1, micro_count, updates, GT_visits, after_rng, activity, counts)
                if micro_count == limit:
                    break
        require(micro_count == limit and updates == limit // 2 and pending_boundary is None and
            all(p.grad is None for p in model.parameters()),
            'Phase endpoint counters/pending gradients differ')
        final = pref.cpu_tree(torch, model.state_dict())
        require(all(torch.equal(final[n], anchor[n]) for n in policy['frozen_names'] + policy['BN_buffer_names']),
            'Endpoint typedBN changed')
        require(all(bool(torch.isfinite(v).all()) for v in final.values()), 'Nonfinite native endpoint state')
        if not technical:
            require(GT_visits == PHASE_GT and len(epochs) == PHASE_EPOCHS and set(nodes) == {str(e) for e in NODES},
                'Complete additional48/79728GT/six phase nodes required')
        write(output / 'ORDER.json', orders)
        write(output / 'EPOCH_ROWS.json', epochs)
        endpoint_proof = dict(schema='native-finetune48-E48warm-weights-fresh-optimizer-endpoint-proof-v1', before=pref.state_proof(anchor),
            after=pref.state_proof(final), changed={n: not torch.equal(v, anchor[n]) for n, v in final.items()}, policy=policy,
            teacher_loading=loading, optimizer=audit, actual_optimizer=common.optimizer_summary(torch, pref, optimizer, names),
            positive_accumulated_gradient_microcounts=activity, optimizer_step_nonNone_gradient_counts=counts,
            phase_microsteps=micro_count, phase_updates=updates, phase_raw_GT_visits=GT_visits, frozen_BN_all_exact=True,
            no_pending_accumulation_gradients=True, formal_resume_eligible=False, native_loss_atoms=30,
            completed_phase_epochs=0 if technical else PHASE_EPOCHS, effective_native_epoch=48 if technical else 96,
            technical_weights_are_not_formal_initialization=True, old_optimizer_restored=False, old_RNG_restored=False)
        write(output / 'ENDPOINT_PROOF.json', endpoint_proof)
        technical_snapshot = None
        if technical:
            snapshot = dict(schema='native-finetune48-E48warm-weights-fresh-optimizer-technical-CPU-snapshot-v1',
                state_dict=final, optimizer_state_dict=pref.cpu_tree(torch, optimizer.state_dict()),
                external_rng_state=pref.cpu_tree(torch, pref.rng_snapshot(torch)),
                private_order_state=dict(seed=ORDER_SEED, generated_phase_epochs=1,
                    torch_CPU_generator_state=order_rng.get_state().cpu().clone()),
                meta=dict(source=reference(__file__), spec=spec_ref, phase=spec['phase'],
                    teacher_checkpoint=spec['native48_checkpoint'], phase_microsteps=4, phase_updates=2,
                    phase_raw_GT_visits=GT_visits, optimizer_param_names=names, base_lrs=base_lrs,
                    schedule=spec['schedule'], policy=policy, optimizer=audit,
                    technical_only=True, formal_resume_eligible=False, technical_weights_are_not_formal_initialization=True))
            snapshot_path = output / 'TECHNICAL_ONLY_CPU_SNAPSHOT.pth'
            with snapshot_path.open('xb') as stream:
                torch.save(snapshot, stream)
            readback = torch.load(snapshot_path, map_location='cpu')
            require(pref.tree_equal(torch, snapshot, readback), 'Technical CPU model/optimizer/RNG readback differs')
            technical_snapshot = reference(snapshot_path)
            del snapshot, readback
        return dict(execution_kind=spec['execution_kind'], nodes=nodes, endpoint_proof=reference(output / 'ENDPOINT_PROOF.json'),
            start_state_proof=reference(output / 'START_STATE_PROOF.json'), trace=reference(output / 'TRACE.jsonl'),
            order=reference(output / 'ORDER.json'), epoch_rows=reference(output / 'EPOCH_ROWS.json'),
            phase_microsteps=micro_count, phase_updates=updates, phase_raw_GT_visits=GT_visits,
            completed_phase_epochs=0 if technical else PHASE_EPOCHS, effective_native_epoch=48 if technical else 96,
            checkpoint=None if technical else nodes['48']['checkpoint'], technical_CPU_snapshot=technical_snapshot,
            frozen_BN_all_exact=True,
            fresh_optimizer_at_phase_start=True, old_optimizer_restored=False, old_RNG_restored=False, formal_resume_eligible=False)
    finally:
        optimizer.zero_grad(set_to_none=True)
        model.cpu()


def run(spec_file):
    started = time.monotonic()
    require(Path(spec_file).is_absolute(), 'Absolute --spec required')
    spec_path = Path(spec_file).resolve(strict=True)
    spec = read(spec_path)
    require(spec['schema'] == SCHEMA and spec['kind'] == 'mask2former' and spec['mode'] == 'native0' and
        spec['execution_kind'] in EXECUTION_KINDS and spec['phase_epochs'] == PHASE_EPOCHS and spec['teacher_epoch'] == 48 and
        spec['phase_initial_epoch'] == 0 and spec['order_seed'] == ORDER_SEED and spec['expected_epoch'] == 48 and
        spec['phase_microsteps'] == PHASE_MICROSTEPS and spec['phase_updates'] == PHASE_UPDATES and spec['phase_raw_GT_visits'] == PHASE_GT,
        'Fresh native0 E48 + additional48 proposal required')
    raw, source, spec_ref, admission_ref = source_gate(spec_path, spec)
    predecessor = qualified_native48(spec, raw)
    restoration = restoration_gate(spec, source)
    require(spec['CPU_affinity'] == [0, 1] and spec['allocator_limit_MiB'] == 4096 and
        spec['whole_GPU_owner_limit_MiB'] == 8192 and type(spec['deadline_seconds']) is int and
        0 < spec['deadline_seconds'] <= 21600 and os.environ.get('CUDA_VISIBLE_DEVICES') == '0',
        'CPU01/CUDA0/4GiB allocator/8GiB whole-owner/fresh phase deadline required')
    output = Path(spec['output_directory'])
    require(output.is_absolute() and not output.exists(), 'Fresh absolute output required')
    legacy = load_source(spec['evaluation_source'], 'sealed_original_native_teacher_loading')
    teacher_row = legacy.qualified_training(spec)['native0']
    validate_phase_recipe(spec, teacher_row['training_spec'])
    output.mkdir(parents=True)
    write(output / 'START.json', dict(schema='native-finetune48-E48warm-weights-fresh-optimizer-start-v1', source=source, spec=spec_ref,
        phase=spec['phase'], execution_kind=spec['execution_kind'], teacher_checkpoint=spec['native48_checkpoint'],
        teacher_qualification=spec['native48_CPU_qualification'], teacher_epoch=48, phase_epochs=48,
        old_optimizer_restored=False, old_RNG_restored=False, order_seed=ORDER_SEED, phase_loss_seed_base=spec['phase_loss_seed_base'],
        no_DEV=True, no_AP=True, no_retry=True, fresh_phase_not_formal_resume=True,
        technical_CPU_qualification=None if restoration is None else spec['restoration_CPU_qualification']))
    import psutil
    import torch
    from mmengine import Config
    from mmengine.model import revert_sync_batchnorm
    from mmengine.registry import init_default_scope
    from mmdet.registry import MODELS
    from mmdet.utils import register_all_modules
    from mmdet.datasets import CocoDataset
    from mmdet.datasets.api_wrappers import COCO
    require(psutil.Process().cpu_affinity() == [0, 1] and not torch.cuda.is_initialized(), 'ROOT CPU01/coldCUDA required')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    register_all_modules(init_default_scope=False)
    init_default_scope('mmdet')
    require(len(CocoDataset.METAINFO['classes']) == 80 and CocoDataset.METAINFO['classes'][8] == 'boat',
        'Original boat8/null80 class mapping required')
    pref = load_source(spec['micro_preflight_source'], 'sealed_native_optimizer_and_preprocessing')
    initializer = load_source(spec['initializer_source'], 'sealed_official_native_initializer')
    common = load_source(spec['training_source'], 'sealed_common24_native_loss_observer')
    old = load_source(spec['native_helpers'], 'sealed_original_native_image_helpers')
    old.torch = torch
    prepared = prepare_training_model(torch, pref, initializer, common, MODELS, revert_sync_batchnorm, Config,
        tuple(CocoDataset.METAINFO['classes']), spec, teacher_row, predecessor)
    imported_library_gate(torch, prepared[0], spec)
    dataset = CocoDataset(ann_file=str(bound(spec['FIT'])), data_prefix=dict(img=spec['image_root']),
        metainfo=dict(classes=('ship',)), pipeline=[], filter_cfg=None, test_mode=True)
    coco = COCO(str(bound(spec['FIT'])))
    require(len(dataset) == len(coco.imgs) == 741 and len(coco.anns) == 1661, 'Complete original FIT741/1661 required')
    roster = [int(dataset.get_data_info(i)['img_id']) for i in range(741)]
    require(roster == spec['FIT_image_ids'] and len(set(roster)) == 741, 'Original ordered FIT roster differs')
    torch.cuda.init()
    capacity = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1., 4096 * 1024 ** 2 / capacity), 0)
    torch.cuda.reset_peak_memory_stats(0)
    error, result, cleanup = None, None, None
    try:
        result = train_phase(torch, pref, old, common, *prepared, spec, spec_ref, output, dataset, coco, started)
        del prepared
        cleanup = pref.clear_gpu(torch)
        for ref in spec['sources']:
            bound(ref)
        pref.check_deadline(started, spec['deadline_seconds'])
    except Exception as exception:
        error = dict(type=type(exception).__name__, message=str(exception), no_retry=True, no_fallback=True,
            OOM_technical_stop=isinstance(exception, torch.cuda.OutOfMemoryError))
        traceback.clear_frames(exception.__traceback__)
        if 'prepared' in locals():
            prepared[0].cpu()
            del prepared
        gc.collect()
        try:
            cleanup = pref.clear_gpu(torch)
        except Exception as cleanup_error:
            error['cleanup_error'] = str(cleanup_error)
    report = dict(schema=REPORT_SCHEMA, status='NATIVE_FINETUNE48_E48WARM_WEIGHTS_FRESH_OPTIMIZER_OWNED_COMPLETE_NO_AP' if error is None else
        'NATIVE_FINETUNE48_E48WARM_WEIGHTS_FRESH_OPTIMIZER_STOP_NO_METHOD_RESULT', phase=spec['phase'], source=source, spec=spec_ref,
        GPU_admission=admission_ref, execution_kind=spec['execution_kind'], result=result, error=error, cleanup=cleanup,
        SOURCE_proposal=spec['SOURCE_proposal'], teacher_checkpoint=spec['native48_checkpoint'],
        teacher_qualification=spec['native48_CPU_qualification'], phase_recipe=spec['initializers']['mask2former'],
        schedule=spec['schedule'], order_seed=ORDER_SEED, phase_loss_seed_base=spec['phase_loss_seed_base'],
        old_optimizer_restored=False, old_RNG_restored=False, fresh_phase_not_formal_resume=True,
        native345_trainable=True, native451_parameters_610_states=True, typed_BN106_affine159_buffers_frozen=True,
        native_loss_atoms=30, dynamic_ten_stage_native_matching=True, feature_cache_used=False,
        elapsed_seconds=time.monotonic()-started, torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
        no_DEV=True, no_AP=True, no_TEST=True, no_automatic_retry=True, formal_resume_eligible=False,
        additional_imagevisits_and_updates_comparator=True, compute_matched_claimed=False, objective_matched_claimed=False,
        efficacy_claimed=False, convergence_claimed=False, novelty_claimed=False, SOTA_claimed=False)
    write(output / 'REPORT.json', report)
    if error is not None:
        raise RuntimeError('TECHNICAL_STOP: ' + error['type'] + ': ' + error['message'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--owner-execute', action='store_true')
    args = parser.parse_args()
    require(args.owner_execute, 'SOURCE_ONLY: explicit ROOT-owned execution required')
    run(args.spec)


if __name__ == '__main__':
    main()
