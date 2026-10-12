"""Inert independent CPU consumer of the frozen-native four-arm technical seam.

SOURCE preparation is limited to source inspection and static compilation.
ROOT alone executes this reader after raw current and canonical teacher owned
terminal receipts. No scientific import, artifact hash or checkpoint access
occurs before those gates. CPU reconstruction verifies states, targets and
optimizer storage, never replays native forward/backward, geometry or matching.
Four fresh four-micro/two-update cases provide no AP or efficacy conclusion.
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

SPEC_SCHEMA = 'ROOT-query-conditioned-frozen-native-preflight-spec-v1'
REPORT_SCHEMA = 'query-conditioned-frozen-native-preflight-report-v1'
CP_SCHEMA = 'query-conditioned-frozen-native-technical-checkpoint-v1'
PROOF_SCHEMA = 'query-conditioned-frozen-native-technical-state-proof-v1'
QUAL_SCHEMA = 'ROOT-query-conditioned-frozen-native-preflight-CPU-qualification-v1'
QUAL_STATUS = 'ROOT_QUERY_CONDITIONED_FROZEN_NATIVE_PREFLIGHT_TECHNICALLY_QUALIFIED'
MODES = ('raw', 'raw_rms', 'query_rms', 'permuted_query_rms')
IMAGE_IDS = (1, 2, 3, 5)
PREFIX = 'panoptic_head.query_pair_readout.'
ACTIVE = [PREFIX + 'hidden.weight', PREFIX + 'final.weight']
OWNER_SHA = 'ef9a9f17d4fb1bc3464936095ef799021dd880f8b15488d558a871c08b6e1924'
C_ROOT = Path(__file__).resolve().parent.parent
M_ROOT = C_ROOT.parent
H_ROOT = M_ROOT / 'goal_continuation_20261005'
TEACHER_PHASE = 'native_spatial_position_preserving_common24_native0_20261009_01'
SOURCE_PINS = {
    'readout_source': (10481, '5199354074939e335e0d0b5c19940ad56d637b750843cb38cc9c7213e81b4b2a'),
    'hooks_source': (27567, 'd043e86d5699193533d4a8279ff7961e10adef916028b8250ae38ebe58a42dcc'),
    'geometry_source': (17500, '7ad19e356342afc15a2a41f1358634b23486c19021e63ccd94cfed03f6841d15'),
    'training_source': (43363, 'a50b7c95691af7884dcf0bf23ad331d0c1a4055c11caa3da255d0efafb25cc09'),
    'training_CPU_reader': (60137, '51443701bf31273a1df771a597b05b7540b75366c2e09d4919bf49587062d97d'),
    'evaluation_source': (40816, 'b4ca717bd2018ea8404808c03510f830081a6660c893eda04fce72d7adb43a48'),
    'native_preflight_helpers': (38723, '840bdf376ee0d9eb291e24cb605c110d310d32338ccba6d3f231c827c0e6e1ba'),
    'reused_CPU_reader': (46683, 'ae810fb940fed62588b1a4e0bd7ee199647ff656dbbef5d0bb1247864662d960')}
SOURCE_PATHS = {
    'readout_source': C_ROOT / 'query_conditioned_global_pair_probe_source_20261010' / 'readout.py',
    'hooks_source': C_ROOT / 'query_conditioned_global_pair_probe_source_20261010' / 'native_hooks.py',
    'geometry_source': C_ROOT / 'paired_context.py',
    'training_source': C_ROOT / 'spatial_position_preserving_common24_source_20261009' / 'train_common.py',
    'training_CPU_reader': C_ROOT / 'spatial_position_preserving_common24_source_20261009' / 'qualify_common_cpu.py',
    'evaluation_source': C_ROOT / 'spatial_position_preserving_common24_evaluation_source_20261009' / 'evaluate.py',
    'native_preflight_helpers': C_ROOT / 'spatial_position_preserving_native_preflight_source_20261009' / 'native_preflight.py',
    'reused_CPU_reader': M_ROOT / 'native_coco_comparator_micro_preflight_preparation_20261008' / 'qualify_cpu.py',
    'operator_CPU_fixture_source': C_ROOT / 'query_conditioned_global_pair_probe_source_20261010' / 'synthetic_fixture.py'}
ARTIFACT_PINS = {
    'FIT_annotation': (M_ROOT.parent / 'experiments' / 'public_baseline_20260925' / 'protocol' / 'pseg_ssdd' / 'fit.json',
        498760, '62966857c78cea1e3b060700c62a7b896942686c2e674d3eb09f516a92250bc1'),
    'teacher_training_spec': (H_ROOT / 'ROOT_SPATIAL_COMMON24_NATIVE0_SPEC_20261009_01.json',
        20305, '28b39444eb535fdb95e9fbceb1b83018f02ee587c9afebfc1ff1d5e4393907aa'),
    'teacher_qualification': (H_ROOT / 'ROOT_SPATIAL_COMMON24_NATIVE0_CPU_QUALIFICATION_20261009_01.json',
        903455, '3b661bc7155f75554e56a57e03ba3669419e360b16e579180aa41b3d365e9f19'),
    'teacher_checkpoint': (H_ROOT / 'SPATIAL24_NATIVE0_01' / 'NODE_E24.pth',
        528421034, '376f725c42a0d522700add19548943328064426835061282d95d510f27d1ec4e'),
    'teacher_receipt': (H_ROOT / 'finite_point_cache_execution_20261005_v2' / 'runs' / TEACHER_PHASE / 'PARENT_RECEIPT.json',
        1073, 'ba0c82cbef5e19572657b534738aecff4da3d792b17759107abb57b0eae60b33'),
    'operator_CPU_fixture': (H_ROOT / 'ROOT_QUERY_PAIR_GENERATOR_SYNTHETIC_CPU_QUALIFICATION_20261010_01.json',
        3719, 'e6458035b8924eb8e1b4e5f8a9dffd1bc61030515fd96b3e72ff4b81bf4c3dd6')}
TEACHER_ADMISSION = dict(path=str(H_ROOT / 'ROOT_SPATIAL_COMMON24_NATIVE0_ADMISSION_20261009_01.json'),
    bytes=10544, sha256='0865b683f8ec437f69ece854db602e77272bea458a0f9bb741b78a64f886ae06')


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
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def ref_shape(row):
    require(isinstance(row, dict) and set(row) == {'path', 'bytes', 'sha256'} and
        isinstance(row['path'], str) and Path(row['path']).is_absolute() and
        type(row['bytes']) is int and row['bytes'] > 0 and
        isinstance(row['sha256'], str) and len(row['sha256']) == 64 and
        set(row['sha256']) <= set('0123456789abcdef'), 'Malformed absolute pinned reference')


def bound(row):
    ref_shape(row)
    require(reference(row['path']) == row, 'Pinned reference changed: ' + row['path'])
    return Path(row['path'])


def read_ref(row):
    return read_json(bound(row))


def pinned_role(spec, role):
    ref_shape(spec[role])
    path, size, digest = ARTIFACT_PINS[role]
    require(Path(spec[role]['path']).resolve() == path.resolve() and
        spec[role]['bytes'] == size and spec[role]['sha256'] == digest,
        'Canonical pinned input differs: ' + role)


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode('utf-8')).hexdigest()


def terminal(receipt, phase):
    require(receipt.get('status') == 'ACTUAL_OWNED_TERMINAL' and
        receipt.get('original_identity') is not None and
        type(receipt.get('returncode')) is int and receipt['returncode'] == 0 and
        receipt.get('root_wait_reaped') is True and
        receipt.get('held_handle_exit') == dict(wait_state='WAIT_OBJECT_0', actual_exit_code=0) and
        receipt.get('numeric_birth_absence', {}).get('original_birth_absent') is True and
        all(key in receipt and receipt[key] is None for key in ('abort', 'error')) and
        receipt.get('wait_errors') is False and receipt.get('GPU_observation_errors') == [],
        'STOP: actual owned held raw exit0/reap/original birth absence/noerrors required')
    if 'phase' in receipt:
        require(receipt['phase'] == phase, 'Raw owner receipt phase differs')
    peak = receipt.get('whole_GPU_observed_peak_MiB')
    require(type(peak) in (int, float) and math.isfinite(peak) and 0 <= peak <= 8192 and
        receipt.get('whole_GPU_used_limit_MiB') == 8192 and
        receipt.get('whole_leaf_VRAM_certified') is False and
        receipt.get('GPU_scope') == 'whole GPU observed, not per-process WDDM allocation',
        'STOP: actual whole-GPU observation or <=8192MiB resource scope differs')


def source_gate(spec_path, spec, receipt):
    """Called only after current and teacher raw terminal gates; no SCI import."""
    spec_ref, consumer = reference(spec_path), reference(__file__)
    producer = reference(Path(__file__).resolve().with_name('native_preflight.py'))
    admission_path = Path(spec['GPU_admission_file'])
    require(admission_path.is_absolute(), 'Absolute current GPU admission filename required')
    admission_ref, admission = reference(admission_path), read_json(admission_path)
    owner_ref = reference(spec_path.parent / 'finite_point_cache_execution_20261005_v2' / 'run_owned_gpu_limited.py')
    require(receipt['admission_sha256'] == admission_ref['sha256'] and
        owner_ref['sha256'] == admission['driver_sha256'] == OWNER_SHA and
        admission['status'] == 'ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE' and
        admission['automatic_retry'] is False and admission['phase'] == spec['phase'] and
        admission['allowed_cpus'] == spec['CPU_affinity'] == [0, 1] and
        admission['cuda_visible_devices'] == '0' and
        type(admission['qualified_leaf_deadline_seconds']) is int and
        admission['qualified_leaf_deadline_seconds'] == spec['deadline_seconds'] and
        admission['whole_GPU_used_limit_MiB'] == spec['whole_GPU_owner_limit_MiB'] == 8192 and
        Path(admission['entry']).is_absolute() and Path(admission['entry']).resolve() == Path(producer['path']) and
        admission['entry_args'] == ['--spec', str(spec_path), '--owner-execute'],
        'Current held receipt/admission/new entry/resources/source join differs')
    sources = spec['sources']
    require(isinstance(sources, list) and len({row['path'] for row in sources}) == len(sources) and
        spec['entry_source'] == producer and spec['CPU_reader'] == consumer and
        producer in sources and consumer in sources and
        admission['bound_source_sha256'].get(str(spec_path)) == spec_ref['sha256'],
        'Current entry/consumer/SPEC/unique source closure is not owner-bound')
    for row in sources:
        bound(row)
        require(admission['bound_source_sha256'].get(row['path']) == row['sha256'],
            'Current source differs from actual owner admission')
    for role in ('readout_source', 'geometry_source', 'hooks_source', 'initializer_source',
                 'micro_preflight_source', 'native_helpers', 'native_preflight_helpers',
                 'reused_CPU_reader', 'evaluation_source', 'training_source', 'training_CPU_reader', 'operator_CPU_fixture_source'):
        require(spec[role] in sources and Path(spec[role]['path']).suffix == '.py',
            'Pinned runtime SOURCE role absent from full closure: ' + role)
        if role in SOURCE_PINS:
            require((spec[role]['bytes'], spec[role]['sha256']) == SOURCE_PINS[role],
                'Immutable exact SOURCE pin differs: ' + role)
        if role in SOURCE_PATHS:
            require(Path(spec[role]['path']).resolve() == SOURCE_PATHS[role].resolve(),
                'Canonical runtime SOURCE path differs: ' + role)
    if 'scientific_spec' in admission:
        require(admission['scientific_spec'] == spec_ref, 'Current admitted scientific SPEC differs')
    return dict(spec=spec_ref, producer=producer, consumer=consumer,
        GPU_admission=admission_ref, owner=owner_ref)


def load_source(row, name):
    """Execute verified SOURCE bytes without writing pyc; caller has gated all roles."""
    ref_shape(row)
    path = Path(row['path']).resolve(strict=True)
    data = path.read_bytes()
    require(str(path) == row['path'] and len(data) == row['bytes'] and
        hashlib.sha256(data).hexdigest() == row['sha256'], 'Actual loaded SOURCE bytes differ')
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(data, str(path), 'exec'), module.__dict__)
    return module


def teacher_lineage(spec, raw_receipt):
    training = read_ref(spec['teacher_training_spec'])
    require(training['schema'] == 'ROOT-native-spatial-position-preserving-common24-spec-v1' and
        training['kind'] == 'mask2former' and training['mode'] == 'native0' and
        training['phase'] == spec['teacher_phase'] == TEACHER_PHASE and
        training['initial_epoch'] == 0 and training['epochs_this_run'] == training['total_epochs'] == 24 and
        training['resume_checkpoint'] is None and training['resume_qualification'] is None and
        training['checkpoint_nodes'] == [2, 4, 8, 12, 18, 24] and
        training['FIT'] == spec['FIT_annotation'] and training['image_root'] == spec['image_root'] and
        training['CPU_affinity'] == [0, 1] and training['whole_GPU_owner_limit_MiB'] == 8192,
        'Canonical fresh native0 full24 training SPEC differs')
    require(raw_receipt['admission_sha256'] == TEACHER_ADMISSION['sha256'] and
        Path(training['GPU_admission_file']).resolve() == Path(TEACHER_ADMISSION['path']).resolve(),
        'Teacher raw terminal/original admission join differs before teacher Q/CP')
    admission = read_ref(TEACHER_ADMISSION)
    require(admission['phase'] == TEACHER_PHASE and
        admission['status'] == 'ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE' and
        admission['automatic_retry'] is False and admission['allowed_cpus'] == [0, 1] and
        admission['cuda_visible_devices'] == '0' and
        admission['qualified_leaf_deadline_seconds'] == training['deadline_seconds'] and
        admission['whole_GPU_used_limit_MiB'] == 8192 and admission['driver_sha256'] == OWNER_SHA and
        admission['entry_args'] == ['--spec', spec['teacher_training_spec']['path'], '--owner-execute'] and
        Path(admission['entry']).resolve() == Path(spec['training_source']['path']).resolve() and
        training['entry_source'] == spec['training_source'] and
        training['CPU_reader'] == spec['training_CPU_reader'] and
        admission['bound_source_sha256'].get(spec['teacher_training_spec']['path']) == spec['teacher_training_spec']['sha256'],
        'Canonical teacher actual admission/source/resources differ')
    if 'scientific_spec' in admission:
        require(admission['scientific_spec'] == spec['teacher_training_spec'], 'Original teacher admitted scientific SPEC differs')
    for row in training['sources']:
        require(row in spec['sources'] and admission['bound_source_sha256'].get(row['path']) == row['sha256'],
            'Teacher original source closure not preserved in current owner closure')
        bound(row)
    for role in ('initializer_source', 'micro_preflight_source', 'native_helpers'):
        require(training[role] == spec[role], 'Teacher/current native SOURCE lineage differs: ' + role)
    q = read_ref(spec['teacher_qualification'])
    require(q['schema'] == 'ROOT-native-spatial-position-preserving-common24-CPU-qualification-v1' and
        q['status'] == 'ROOT_NATIVE_SPATIAL_POSITION_PRESERVING_COMMON24_CPU_QUALIFIED' and
        q['kind'] == 'mask2former' and q['mode'] == 'native0' and q['phase'] == TEACHER_PHASE and
        q['spec'] == spec['teacher_training_spec'] and q['source'] == spec['training_source'] and
        q['reader_source'] == spec['training_CPU_reader'] and q['parent_receipt'] == spec['teacher_receipt'] and
        q['GPU_admission'] == TEACHER_ADMISSION and
        q['checkpoint'] == spec['teacher_checkpoint'] == q['node_checkpointmap']['24'] and
        q['completed_epoch'] == 24 and q['global_microsteps'] == 17784 and
        q['global_updates'] == 8892 and q['consumed_raw_GT'] == 39864 and
        q['all_six_nodes_CPU_qualified'] is True and q['completed_24_epoch_schedule'] is True and
        q['technical_training_checkpoint_eligible'] is True and q['automatic_resume_eligible'] is False and
        q['AP_computed'] is False and q['inference_executed'] is False,
        'Actual independently qualified native0 E24 full checkpoint lineage differs')
    for role in ('source', 'reader_source', 'report', 'start', 'owner_driver'):
        bound(q[role])
    require(q['owner_driver']['sha256'] == OWNER_SHA, 'Qualified teacher original owner SOURCE differs')
    protocol = q['training_protocol']
    require(protocol['mode'] == 'native0' and protocol['training_source'] == spec['training_source'] and
        protocol['FIT'] == spec['FIT_annotation'] and protocol['total_epochs'] == 24 and
        protocol['total_microsteps'] == 17784 and protocol['total_updates'] == 8892 and
        protocol['total_raw_GT_visits'] == 39864 and protocol['native_loss_only'] is True and
        protocol['native_loss_atoms'] == 30 and protocol['Sobel'] is False and
        protocol['fresh_single_owned_24'] is True and protocol['resume_allowed'] is False,
        'Teacher canonical native full24 protocol differs')
    proof = q['model_proof']
    require(proof['checkpoint'] == spec['teacher_checkpoint'] == proof['nodes']['24']['checkpoint'] and
        proof['proof_refs']['state_proof'] == proof['nodes']['24']['state_proof'] and
        proof['training_protocol'] == protocol and proof['completed_epoch'] == 24 and
        proof['technical_proof']['full_native_state_CPU_finite_verified'] is True and
        proof['technical_proof']['original_typed_BN_affine_and_buffers_exact'] is True and
        proof['technical_proof']['all_six_nodes_verified'] is True,
        'Teacher independent CPU full24 state proof is incomplete')
    bound(spec['teacher_receipt'])
    return training, q


def fixture_gate(spec):
    q = read_ref(spec['operator_CPU_fixture'])
    require(q['status'] == 'QUERY_CONDITIONED_GLOBAL_PAIR_SYNTHETIC_CPU_PASS' and
        q['synthetic_execution_performed'] is True and q['native_integration_qualified'] is False and
        q['efficacy_claimed'] is False and q['execution'] == dict(device='cpu', cpu_affinity=[0, 1],
            threads=2, cuda_visible_devices='') and
        q['sources']['fixture'] == spec['operator_CPU_fixture_source'] and
        q['sources']['operator'] == spec['readout_source'] and
        q['sources']['geometry'] == spec['geometry_source'],
        'Actual independently pinned SOURCE-only CPU fixture differs')
    for name in ('real_original_rectangle_E', 'injected_E_is_not_original_predicate_evidence',
                 'all_four_cold_native_bit_identity', 'seeded_state_and_rng_restoration',
                 'permutation_cycle_derangement_multiset_norm', 'normalization_scale_sign_zero_gradients',
                 'auxiliary_e0_detached_native_e0_live', 'all_E0_connected_zero',
                 'raw_original_spatial_global_bit_equal'):
        require(q[name] is True, 'Actual CPU fixture assertion failed: ' + name)
    for row in q['sources'].values():
        require(row in spec['sources'], 'Fixture SOURCE absent from current admitted closure')
        bound(row)
    fixture_spec = read_json(q['spec_path'])
    require(fixture_spec['schema'] == 'query_conditioned_global_pair_probe_source_only_spec_v1' and
        fixture_spec['inputs'] == 'SOURCE_ONLY' and fixture_spec['sources'] == q['sources'] and
        fixture_spec['execution'] == q['execution'], 'Actual fixture SOURCE-only SPEC/report join differs')
    return q


def digest_contract(row, shape, dtype):
    require(isinstance(row, dict) and set(row) == {'shape', 'dtype', 'sha256'} and
        row['shape'] == shape and row['dtype'] == dtype and
        isinstance(row['sha256'], str) and len(row['sha256']) == 64 and
        set(row['sha256']) <= set('0123456789abcdef'), 'Observed tensor digest contract differs')


def scalar_leaves(value):
    if isinstance(value, list):
        for child in value:
            for item in scalar_leaves(child):
                yield item
    else:
        yield value


def independent_adapter(torch):
    before = torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(2026100911)
        hidden = torch.nn.Linear(512, 512, bias=False, device='cpu', dtype=torch.float32)
        final = torch.nn.Linear(512, 256, bias=False, device='cpu', dtype=torch.float32)
        with torch.no_grad():
            final.weight.zero_()
        state = {'hidden.weight': hidden.weight.detach().clone(), 'final.weight': final.weight.detach().clone()}
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(2026101013)
        order = torch.randperm(256, device='cpu')
        permutation = torch.empty(256, dtype=torch.long, device='cpu')
        permutation[order] = order.roll(-1)
    P = permutation.tolist()
    require(torch.equal(before, torch.get_rng_state()) and sum(v.numel() for v in state.values()) == 393216 and
        bool((state['hidden.weight'] != 0).any()) and bool((state['final.weight'] == 0).all()) and
        sorted(P) == list(range(256)) and all(i != value for i, value in enumerate(P)),
        'Independent cold2W/393216/seeded permutation/RNG differs')
    seen, at = set(), 0
    for _ in range(256):
        require(at not in seen, 'Fixed channel permutation is not one 256-cycle')
        seen.add(at)
        at = P[at]
    require(at == 0 and len(seen) == 256, 'Fixed seeded channel cycle differs')
    return state, P


def typed_native(torch, model):
    parameters = dict(model.named_parameters())
    bn_modules, affine, buffers = [], [], []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            bn_modules.append(name)
            for local, value in module.named_parameters(recurse=False):
                affine.append(name + '.' + local)
            for local, value in module.named_buffers(recurse=False):
                buffers.append(name + '.' + local)
    native = [name for name in parameters if not name.startswith(PREFIX)]
    require(len(native) == 451 and len(bn_modules) == 53 and len(affine) == 106 and len(buffers) == 159 and
        len(set(affine + buffers)) == 265 and set(affine) <= set(native),
        'Independent typed native451/BN53/affine106/buffers159 differs')
    return native, bn_modules, affine, buffers


def load_teacher(torch, old, teacher_reader, initializer, registry, revert, Config, spec, training, q):
    initialization = read_ref(q['model_proof']['proof_refs']['initialization_proof'])
    accepted = read_ref(training['initializer_qualification'])
    require(accepted['schema'] == 'ROOT-native-coco-comparator-initializer-CPU-inspection-v1' and
        accepted['all_initializers_strictly_compatible'] is True and accepted['source'] == spec['initializer_source'],
        'Actual canonical strict COCO80-to-boat initializer qualification differs')
    model, original = teacher_reader.build_initial(torch, old, initializer, registry, revert, Config,
        training, accepted, initialization)
    require(len(original) == len(model.state_dict()) == 610 and len(list(model.parameters())) == 451,
        'Independent canonical bare teacher architecture differs')
    cp = torch.load(str(bound(spec['teacher_checkpoint'])), map_location='cpu')
    require(set(cp) == {'state_dict', 'optimizer_state_dict', 'external_rng_state', 'private_order_state', 'meta'},
        'Full native0 E24 teacher CP required; compact adapter checkpoint is forbidden')
    state, meta = cp['state_dict'], cp['meta']
    for key, value in dict(schema='native-spatial-position-preserving-common24-checkpoint-v1', kind='mask2former',
        mode='native0', phase=TEACHER_PHASE, spec=spec['teacher_training_spec'], source=spec['training_source'],
        training_protocol=q['training_protocol'], initial_epoch=0, completed_epoch=24, global_microsteps=17784,
        global_updates=8892, consumed_raw_GT=39864, no_pending_accumulation_gradients=True,
        not_technical_preflight_checkpoint=True, resume_checkpoint=None, resume_qualification=None,
        fresh_single_owned_24=True, eligible_for_automatic_resume=False).items():
        require(meta[key] == value, 'Actual canonical E24 teacher metadata differs: ' + key)
    proof = read_ref(q['model_proof']['proof_refs']['state_proof'])
    require(proof['schema'] == 'native-spatial-position-preserving-common24-node-state-proof-v1' and
        proof['kind'] == 'mask2former' and proof['mode'] == 'native0' and proof['completed_epoch'] == 24 and
        proof['initialization_proof'] == q['model_proof']['proof_refs']['initialization_proof'] and
        proof['frozen_BN_all_exact'] is True and proof['all_final_state_finite'] is True and
        set(state) == set(original) and len(state) == 610 and
        not any(name.startswith('panoptic_head.spatial_mask_readout.') or name.startswith(PREFIX) for name in state),
        'Actual bare native teacher state boundary/proof differs')
    old.check_state(torch, original, proof['original_native_cold'], 'independent original native610')
    old.check_state(torch, state, proof['after'], 'actual canonical trained E24 teacher610')
    native, modules, affine, buffers = typed_native(torch, model)
    require(set(affine) == set(proof['policy']['frozen_names']) and
        set(buffers) == set(proof['policy']['BN_buffer_names']) and
        all(torch.equal(state[name], original[name]) for name in affine + buffers),
        'Canonical E24 teacher original typed BN106+159 changed')
    for name, value in state.items():
        require(isinstance(value, torch.Tensor) and value.device.type == 'cpu' and
            value.dtype == original[name].dtype and value.shape == original[name].shape and
            bool(torch.isfinite(value).all()), 'Teacher shape/dtype/finite differs: ' + name)
    changed = {name: not torch.equal(value, original[name]) for name, value in state.items()}
    require(changed == proof['changed'] == q['model_proof']['changed'], 'Teacher native training state identity differs')
    model.load_state_dict(state, strict=True)
    require(all(torch.equal(value, state[name]) for name, value in model.state_dict().items()),
        'Independent CPU strict full610 teacher load differs')
    teacher = {name: value.detach().clone() for name, value in state.items()}
    del cp, original
    gc.collect()
    return model, teacher


def check_readout(torch, observation, mode, P):
    require(observation['auxiliary_query_detached'] is True and observation['global_pool'] is True and
        observation['dense_readout'] == 'same_unnormalized_F' and
        observation['input_RMS_normalized'] is (mode != 'raw') and
        observation['input_conditioning'] == ('real_query' if mode == 'query_rms' else
            'permuted_query' if mode == 'permuted_query_rms' else 'raw_features') and
        observation['hidden_seed'] == 2026100911 and observation['permutation_seed'] == 2026101013 and
        observation['permutation_kind'] == 'seeded_256_cycle_not_all_derangements_uniform' and
        observation['permutation_indices'] == P and observation['trainable_tensors'] == 2 and
        observation['parameter_scalars'] == 393216 and
        observation['native_freeze_policy_checked'] is False and observation['forward_gradient_proof'] is False and
        observation['zero_count_basis'] == 'reported_FP32_input_RMS_equals_zero',
        'Actual mode/detached query/global dense F/fixed cycle/input scope differs')
    for key in ('eligible', 'support_nonempty'):
        require(all(type(v) is bool for v in scalar_leaves(observation[key])), 'Boolean packet scalar type differs: ' + key)
    for key in ('eligible_counts', 'canonical_eligible_counts'):
        require(all(type(v) is int for v in scalar_leaves(observation[key])), 'Integer packet scalar type differs: ' + key)
    require(all(type(v) in (int, float) and math.isfinite(v) for v in scalar_leaves(observation['rho_mass'])),
        'Finite rho-mass packet scalar type differs')
    E = torch.tensor(observation['eligible'], dtype=torch.bool, device='cpu')
    counts = torch.tensor(observation['eligible_counts'], dtype=torch.int64, device='cpu')
    canonical = torch.tensor(observation['canonical_eligible_counts'], dtype=torch.int64, device='cpu')
    mass = torch.tensor(observation['rho_mass'], dtype=torch.float32, device='cpu')
    support = torch.tensor(observation['support_nonempty'], dtype=torch.bool, device='cpu')
    require(list(E.shape) == [1, 100, 16] and list(counts.shape) == list(canonical.shape) ==
        list(mass.shape) == list(support.shape) == [1, 100] and
        torch.equal(counts, E.sum(-1)) and torch.equal(counts, canonical) and
        bool(((counts >= 0) & (counts <= 16)).all()) and bool(torch.isfinite(mass).all()) and
        bool(((mass >= 0) & (mass <= 40000)).all()) and
        torch.equal(support, mass > 1e-12) and bool(((counts == 0) | support).all()),
        'Actual E/all100/count/support/rho-mass observations differ')
    rms = {}
    for side in ('inside', 'outside'):
        for stage in ('before', 'after'):
            key = side + '_' + stage
            require(all(type(v) in (int, float) and math.isfinite(v) for v in
                scalar_leaves(observation['input_rms'][key])) and
                all(type(v) is int for v in scalar_leaves(observation['zero_input_rms_counts'][key])),
                'Actual RMS/count scalar type differs: ' + key)
            value = torch.tensor(observation['input_rms'][key], dtype=torch.float32, device='cpu')
            zero = torch.tensor(observation['zero_input_rms_counts'][key], dtype=torch.int64, device='cpu')
            require(list(value.shape) == [1, 100, 16] and bool(torch.isfinite(value).all()) and
                bool((value >= 0).all()) and list(zero.shape) == [1, 100] and
                torch.equal(zero, (value == 0).sum(-1)), 'Actual FP32 input RMS packet/count differs: ' + key)
            if stage == 'after' and mode != 'raw':
                require(bool(((value == 0) | ((value - 1).abs() <= 3e-5)).all()),
                    'Normalized reported input RMS must be zero or near unit')
            rms[key] = value
        if mode == 'raw':
            require(torch.equal(rms[side + '_before'], rms[side + '_after']), 'Raw sampled input RMS changed')
        else:
            # A positive pre-RMS proves a nonzero vector; a zero pre-RMS can
            # underflow and does NOT certify an exactly zero channel vector.
            positive = rms[side + '_before'] > 0
            require(bool(((rms[side + '_after'][positive] - 1).abs() <= 3e-5).all()),
                'A positive reported pre-RMS did not normalize to unit RMS')
    return dict(eligible_pairs=int(counts.sum()), supported_queries=int(support.sum()),
        RMS_zero_is_exact_zero_vector_claimed=False, PCA_or_M0_sign_eligibility_CPU_replayed=False)


def hook_contract(row, spec, mode, micro, P, torch):
    for key, value in dict(forward_calls=1, completed_forwards=1, pixel_decoder_calls=1,
        coefficient_calls=10, passive_coefficient_captures=1, unchanged_early_readouts=9,
        final_mask_replacements=1, adapter_calls=1, apply=True, mode=mode, adapter_prefix=PREFIX,
        readout_source=spec['readout_source'], geometry_source=spec['geometry_source'], hidden_seed=2026100911,
        permutation_seed=2026101013, metadata_fields_used=['img_shape', 'batch_input_shape'],
        class_list_object_preserved=True, first_nine_mask_objects_preserved=True,
        native_coefficient_outputs_untouched=True, earlier_gate_paths_untouched=True, all_queries_retained=True,
        adapter_geometry_phi_skipped=False, cache_cleared_after_forward=True, registered_state_not_pruned=True,
        hooks_removed=True, hook_handles_installed=4, hook_handles_removed=4, grad_mode_enabled_at_start=True,
        feature_shape=[1, 256, 200, 200], coefficient_shape=[1, 100, 256],
        final_native_mask_shape=[1, 100, 200, 200]).items():
        require(row[key] == value, 'Actual safe whole-head/final-only hook differs: ' + key)
    require(row['native_graph_properties'] == dict(feature_requires_grad=False, coefficient_requires_grad=False,
        native_final_mask_requires_grad=False, captured_tensors_detached_by_bridge=False, forward_gradient_proof=False),
        'Frozen native feature/coefficient/mask graph properties differ')
    require(row['cold_zero_final_weight'] is (micro < 2) and
        row['cold_native_value_equal'] is (True if micro < 2 else None),
        'Actual cold zero-final/native-mask identity or warm scope differs')
    return check_readout(torch, row['readout_observation'], mode, P)


def check_matches(torch, old, matches, ann_ids):
    require(isinstance(matches, list) and len(matches) == 10, 'Exactly ten original native assignments required')
    for match in matches:
        q, assigned = match['positive_query_indices'], match['assigned_GT_indices']
        require(type(match['positives']) is int and match['positives'] == len(ann_ids) == len(q) == len(assigned) and
            len(set(q)) == len(q) and all(type(v) is int and 0 <= v < 100 for v in q) and
            all(type(v) is int for v in assigned) and sorted(assigned) == list(range(len(ann_ids))),
            'Original native one-to-one assignment dropped/duplicated GT or invalid query')
        labels = torch.ones(100, dtype=torch.int64, device='cpu')
        labels[q] = 0
        require(old.tensor_digest(torch, labels, 'int64') == match['target_labels'],
            'Actual native Q100 assignment classification target differs')


def check_fields(row):
    fields = row['head_fields']
    require(len(fields['classes']) == len(fields['native_masks']) == 10 and
        fields['classes10_objects_preserved'] is True and fields['first9_mask_objects_preserved'] is True and
        fields['native_classes_require_grad'] == [False] * 10 and
        fields['native_masks_require_grad'] == [False] * 10 and fields['adapted_final_requires_grad'] is True,
        'Actual native ten frozen class/mask fields and live adapted final differ')
    for value in fields['classes']:
        digest_contract(value, [1, 100, 2], 'float32')
    for value in fields['native_masks']:
        digest_contract(value, [1, 100, 200, 200], 'float32')
    digest_contract(fields['adapted_final_mask'], [1, 100, 200, 200], 'float32')
    if row['microstep'] < 2:
        require(fields['adapted_final_mask'] == fields['native_masks'][-1],
            'Cold zero-final actual adapted/native final value digest differs')
    bridge = row['bridge_inputs']
    for key, shape in (('feature', [1, 256, 200, 200]), ('coefficient', [1, 100, 256]),
                       ('valid_fraction', [1, 1, 200, 200])):
        digest_contract(bridge[key], shape, 'float32')
    require(bridge['feature_requires_grad'] is False and bridge['coefficient_requires_grad'] is False and
        bridge['residual_requires_grad'] is True, 'Actual frozen native/live2W bridge graph differs')


def check_posterior_GT(torch, np, cv2, coco, old, row, ann_ids):
    """Independently reproduce native preprocess_gt's int64 padded targets."""
    h, w = row['identity']['actual_img_shape']
    masks = np.zeros((len(ann_ids), 800, 800), dtype=np.int64)
    for index, ann_id in enumerate(ann_ids):
        raster = coco.annToMask(coco.anns[ann_id]).astype(np.uint8, copy=False)
        masks[index, :h, :w] = cv2.resize(raster, (w, h), interpolation=cv2.INTER_NEAREST)
    native_mask = old.tensor_digest(torch, torch.from_numpy(masks), 'int64')
    native_uint8 = old.tensor_digest(torch, torch.from_numpy(masks.astype(np.uint8)), 'uint8')
    require(native_uint8 == row['GT']['masks'] and row['posterior_GT'] ==
        dict(mask=native_mask, mask_uint8=native_uint8, labels=row['GT']['labels'], GT_count=len(ann_ids),
        complete_all_raw_annotations=True, preprocess_after_head_forward=True) and
        row['identity']['native_GT'] == native_mask,
        'Independently reconstructed native int64 posterior padded masks/labels differ')


def check_traces(torch, np, cv2, coco, old, spec, result, policy, images, order, mode, P):
    require(len(result['traces']) == 4, 'Exactly four traces per fresh arm required')
    rows, consumed, positive = [], [], {name: 0 for name in ACTIVE}
    for micro, (ref, image_id) in enumerate(zip(result['traces'], IMAGE_IDS)):
        row = read_ref(ref)
        ann_ids = [int(ann['id']) for ann in coco.imgToAnns[image_id]]
        require(row['mode'] == mode and row['microstep'] == micro and row['image_id'] == image_id and
            row['identity']['dataset_index'] == order.index(image_id) and
            row['private_seed'] == 1000000 + micro and row['accumulation_divisor'] == 2 and
            row['optimizer_step'] is bool(micro % 2) and row['optimizer_updates'] == (micro + 1) // 2 and
            row['gradient_clipping_applied'] is bool(micro % 2) and
            row['gradients_are_accumulated_preclip'] is True and row['native_parameter_grads_None'] is True and
            row['all_native_modules_eval'] is True and row['hook_cache_clear'] is True,
            'Fresh four-micro/acc2/two-update/native451-frozen protocol differs')
        old.validate_GT(torch, np, cv2, coco, images[image_id], ann_ids,
            dict(row, actual_input_shape=row['input']['shape']))
        require(Path(row['identity']['image']['path']).resolve() ==
            (Path(spec['image_root']) / images[image_id]['file_name']).resolve(), 'Raw image root/trace identity differs')
        require(row['native_loss_only'] is True and row['Sobel_added'] is False and
            row['auxiliary_loss_added'] is False, 'Only original native30 supervised loss is permitted')
        old.validate_losses(dict(row, loss_is_native_model_loss=True), 'mask2former')
        check_matches(torch, old, row['matches'], ann_ids)
        require(row['readout_observation'] == row['native_hook']['readout_observation'],
            'Copied actual readout observation differs from safe hook record')
        hook_contract(row['native_hook'], spec, mode, micro, P, torch)
        check_fields(row)
        h, w = row['identity']['actual_img_shape']
        axes = torch.arange(200, dtype=torch.float32, device='cpu')
        valid = ((h - 4 * axes).clamp(0, 4) / 4).reshape(1, 1, 200, 1) * \
            ((w - 4 * axes).clamp(0, 4) / 4).reshape(1, 1, 1, 200)
        require(old.tensor_digest(torch, valid, 'float32') == row['bridge_inputs']['valid_fraction'],
            'Independently reconstructed metadata-only valid fraction differs')
        check_posterior_GT(torch, np, cv2, coco, old, row, ann_ids)
        require(row['GT_removed_from_forward'] is True and row['native_all10_fields_equal_across_modes'] is True and
            row['common_matching_or_points_after_divergence_required'] is False,
            'GT-free native forward/supervised posterior/common native field source scope differs')
        norms = row['gradient_norms_aligned_active_names']
        require(isinstance(norms, list) and len(norms) == 2 and all(type(v) in (int, float) and
            math.isfinite(v) and v >= 0 for v in norms), 'Two actual finite accumulated2W gradient norms required')
        require((micro != 0 or norms[1] > 0) and (micro >= 2 or norms[0] == 0),
            'Cold first hidden zero/final positive actual gradient proof failed')
        require(row['cold_hidden_gradient_exact_zero'] is (True if micro < 2 else None),
            'Cold hidden exact all-element-zero proof must use actual count_nonzero; warm scope must be null')
        for name, norm in zip(ACTIVE, norms):
            positive[name] += int(norm > 0)
        if micro % 2:
            old.finite(row['preclip_global_norm'], 'Actual2W global preclip norm', True)
            require(row['preclip_global_norm'] > 0, 'Actual2W preclip norm must be positive')
        else:
            require(row['preclip_global_norm'] is None, 'Non-update microstep reported clipping')
        old.validate_rng(torch, row)
        if rows:
            require(row['external_RNG_before'] == rows[0]['external_RNG_before'],
                'External RNG restoration sequence differs within fresh arm')
        consumed.extend(ann_ids)
        rows.append(row)
    require(any(all(v > 0 for v in row['gradient_norms_aligned_active_names']) for row in rows[2:]),
        'At least one warm microstep must have both actual2W gradients positive')
    return rows, consumed, positive


def independent_frozen_policy(torch, model):
    native, modules, affine, buffers = typed_native(torch, model)
    parameters = dict(model.named_parameters())
    for name, value in parameters.items():
        value.requires_grad_(name in ACTIVE)
        value.grad = None
    model.eval()
    require(len(model.state_dict()) == 612 and len(parameters) == 453 and
        [name for name, value in parameters.items() if value.requires_grad] == ACTIVE and
        all(not module.training for name, module in model.named_modules()
            if name != 'panoptic_head.query_pair_readout' and not name.startswith('panoptic_head.query_pair_readout.')),
        'Independent native451 frozen/eval and exactly2W trainable boundary differs')
    return dict(kind='mask2former', state_entries=612, parameter_tensors=453, trainable_tensors=2,
        frozen_tensors=451, active_names=ACTIVE, native_names=native, frozen_names=native,
        BN_module_names=modules, BN_parameter_names=affine, BN_buffer_names=buffers,
        all_native_modules_eval=True)


def check_training(torch, np, cv2, coco, old, spec, result, model, teacher, readout, geometry,
                   hooks, direct, P, images, order, mode, execution):
    adapter = hooks.register_query_pair_readout(torch, readout, geometry, model, mode,
        spec['readout_source'], spec['geometry_source'], hidden_seed=2026100911, permutation_seed=2026101013)
    require(set(adapter.state_dict()) == set(direct) and not list(adapter.buffers()) and
        all(torch.equal(adapter.state_dict()[name], value) for name, value in direct.items()) and
        list(adapter.permutation_indices) == P, 'Actual operator factory differs from independent direct2W/cycle')
    policy = independent_frozen_policy(torch, model)
    initial = {name: value.detach().clone() for name, value in model.state_dict().items()}
    require(set(initial) == set(teacher) | set(ACTIVE) and
        all(torch.equal(initial[name], value) for name, value in teacher.items()),
        'Fresh registration changed actual canonical trained teacher610')
    training = read_ref(spec['teacher_training_spec'])
    groups, audit = old.independent_optimizer(torch, model, training['initializers']['mask2former'], policy)
    require(audit['type'] == 'AdamW' and audit['recipe']['lr'] == 0.0001 and audit['recipe']['weight_decay'] == 0.05 and
        audit['clip_grad'] == dict(max_norm=0.01, norm_type=2) and
        [row['name'] for row in audit['parameter_groups']] == ACTIVE and
        all(row['lr'] == 0.0001 and row['weight_decay'] == 0.05 for row in audit['parameter_groups']),
        'Independent fresh native AdamW2W/1e-4/.01clip membership differs')
    cp = torch.load(str(bound(result['checkpoint'])), map_location='cpu')
    require(isinstance(cp, dict) and set(cp) == {'state_dict', 'adapter_state_dict', 'optimizer_state_dict',
        'external_rng_state', 'meta'}, 'Fresh complete612-state/2W/optimizer/RNG technical checkpoint required')
    final, meta = cp['state_dict'], cp['meta']
    proof = read_ref(result['state_proof'])
    require(proof['schema'] == PROOF_SCHEMA and proof['policy'] == policy and
        canonical_sha(proof['optimizer']) == canonical_sha(audit), 'Independent actual frozen policy/optimizer audit differs')
    old.check_state(torch, teacher, proof['native_teacher610'], 'canonical E24 teacher610 immutable baseline')
    old.check_state(torch, direct, proof['cold_adapter'], 'fresh independently reconstructed exact cold2W')
    loading = proof['teacher_loading']
    for key, value in dict(mode='native0', checkpoint=spec['teacher_checkpoint'], qualification=spec['teacher_qualification'],
        CP_qualified=True, strict_fullstate_loaded=True, native_state_tensors=610, adapter_state_tensors=0,
        registered_parameters=451, original_frozen_BN_exact=True, inference_backbone_with_cp=False,
        training_backbone_with_cp=training['backbone_with_cp'], no_random_missing_keys=True,
        BN_parameter_names=policy['BN_parameter_names'], BN_buffer_names=policy['BN_buffer_names']).items():
        require(loading[key] == value, 'Actual source-observed strict fullteacher loading proof differs: ' + key)
    old.check_state(torch, teacher, loading['trained_fullstate'], 'actual strict teacher loader state identity')
    old.check_state(torch, initial, proof['before'], 'fresh frozen teacher612+direct cold2W')
    old.check_state(torch, final, proof['after'], 'actual full612 final technical CPU checkpoint')
    require(set(final) == set(initial), 'Final technical full-state key boundary differs')
    for name, value in final.items():
        require(isinstance(value, torch.Tensor) and value.device.type == 'cpu' and
            value.shape == initial[name].shape and value.dtype == initial[name].dtype and
            bool(torch.isfinite(value).all()), 'Final CPU tensor shape/dtype/finite differs: ' + name)
    changed = {name: not torch.equal(value, initial[name]) for name, value in final.items()}
    require(proof['changed'] == changed and {name for name, value in changed.items() if value} == set(ACTIVE) and
        all(torch.equal(final[name], value) for name, value in teacher.items()) and
        all(torch.equal(final[name], initial[name]) for name in policy['BN_parameter_names'] + policy['BN_buffer_names']),
        'Exactly fresh2W must change; fullteacher610/typedBN106+159 must remain exact')
    model.load_state_dict(final, strict=True)
    require(all(torch.equal(value, final[name]) for name, value in model.state_dict().items()),
        'Strict CPU full612 readback differs from actual technical checkpoint')
    require(set(cp['adapter_state_dict']) == set(direct) and all(torch.equal(value, final[PREFIX + name])
        for name, value in cp['adapter_state_dict'].items()), 'Compact2W alias differs from complete technical checkpoint')
    rows, consumed, positive = check_traces(torch, np, cv2, coco, old, spec, result, policy,
        images, order, mode, P)
    require(proof['adapter_names'] == ACTIVE and proof['positive_gradient_microcounts'] == positive and
        proof['native610_exact'] is True and proof['native_grad_None'] is True and
        proof['all_native_modules_eval'] is True and proof['hook_cache_clear'] is True and
        proof['fresh_optimizer_state_entries'] == 0 and proof['teacher_optimizer_loaded'] is False and
        proof['teacher_order_RNG_loaded'] is False and proof['teacher_external_RNG_loaded'] is False and
        proof['CPU_checkpoint_readback_exact'] is True and proof['GPU_to_CPU_readback_state_exact'] is True and
        proof['frozen_BN_all_exact'] is True and proof['all_adapter_weights_positive_and_changed'] is True and
        proof['warm_both_weights_positive'] is True and proof['initial_hidden_gradient_zero_is_expected'] is True and
        proof['cold_constructor_repeated_exact'] is True and proof['cold_constructor_RNG_restored'] is True and
        proof['fresh_identical_adapter_initialization_across_modes'] is True,
        'Actual two-weight activity/full native frozen state/safe hook source proof differs')
    for key, value in dict(schema=CP_SCHEMA, kind='mask2former', mode=mode, phase=spec['phase'], spec=execution['spec'],
        source=execution['producer'], policy=policy, teacher_checkpoint=spec['teacher_checkpoint'],
        teacher_qualification=spec['teacher_qualification'], teacher_training_spec=spec['teacher_training_spec'],
        teacher_receipt=spec['teacher_receipt'], teacher_mode='native0', teacher_epoch=24,
        readout_source=spec['readout_source'], geometry_source=spec['geometry_source'], hooks_source=spec['hooks_source'],
        image_ids=list(IMAGE_IDS), consumed_ann_ids=consumed, microsteps=4, optimizer_updates=2,
        batch_size=1, accumulation=2, frozen_native=True, native610_exact=True, native_loss_only=True,
        native_loss_atoms=30, no_Sobel_or_auxiliary_loss=True, not_formal_training=True, eligible_for_formal_resume=False,
        no_AP=True, no_pending_accumulation_gradients=True, teacher_optimizer_loaded=False,
        teacher_order_RNG_loaded=False, teacher_external_RNG_loaded=False).items():
        require(meta[key] == value, 'Actual complete technical CP metadata differs: ' + key)
    require(canonical_sha(meta['optimizer']) == canonical_sha(audit), 'CP actual optimizer audit differs')
    counts = {name: 2 for name in ACTIVE}
    optimizer = old.validate_optimizer(torch, cp, groups, audit, counts)
    teacher_reader = load_source(spec['training_CPU_reader'], 'independent_final2W_optimizer_CPU_snapshot')
    actual_snapshot = teacher_reader.optimizer_snapshot(torch, old, cp['optimizer_state_dict'], [[name] for name in ACTIVE])
    require(proof['actual_final_optimizer'] == actual_snapshot,
        'Every actual AdamW moment/step/group tensor identity differs from GPU-to-CPU state proof')
    rng = cp['external_rng_state']
    require(isinstance(rng, list) and len(rng) == 2 and all(isinstance(value, torch.Tensor) and
        value.device.type == 'cpu' and value.dtype == torch.uint8 and value.ndim == 1 and value.numel() > 0
        for value in rng) and [old.tensor_digest(torch, value, 'uint8') for value in rng] == rows[-1]['external_RNG_after'],
        'Actual final technical CP external CPU/CUDA RNG CPU-storage identity differs')
    for key, value in dict(mode=mode, checkpoint=result['checkpoint'], state_proof=result['state_proof'],
        microsteps=4, optimizer_updates=2, image_ids=list(IMAGE_IDS), consumed_ann_ids=consumed,
        native_loss_only=True, AP_computed=False, eligible_for_formal_resume=False).items():
        require(result[key] == value, 'Actual fresh arm result contract differs: ' + key)
    result_proof = dict(mode=mode, checkpoint=result['checkpoint'], state_proof=result['state_proof'],
        traces=result['traces'], native_state_entries=610, full_state_entries=612, parameter_tensors=453,
        frozen_native_parameter_tensors=451, active_parameter_tensors=2, active_parameter_scalars=393216,
        typed_BN_modules=53, typed_BN_affine_tensors=106, typed_BN_buffer_tensors=159,
        trainability=policy, optimizer=optimizer, actual_final_optimizer=actual_snapshot,
        positive_gradient_microcounts=positive, changed_adapter_names=ACTIVE, actual_microsteps=4,
        actual_optimizer_updates=2, consumed_raw_annotation_ids=consumed,
        full_teacher610_and_all_frozen_states_independently_exact=True,
        full_final_GPU_to_CPU_state_digest_proof_exact=True, all_raw_GT_nearest800_independently_reconstructed=True,
        ten_original_assignments_retain_all_raw_GT=True, native30_only=True, CPU_forward_or_backward_replayed=False,
        actual_native_grad_None_source_observed=True, safe_hook_cache_and_handle_cleanup_source_observed=True,
        cold_first_hidden_gradient_zero_and_final_positive_source_observed=True,
        warm_both_weights_positive_at_least_once_source_observed=True, eligible_for_formal_resume=False)
    del cp, initial, final, adapter
    gc.collect()
    require(not torch.cuda.is_initialized(), 'Independent CPU arm check initialized CUDA')
    return result_proof, rows


def qualify(spec_file, acceptance_file):
    spec_path, target = Path(spec_file), Path(acceptance_file)
    require(spec_path.is_absolute() and target.is_absolute() and target.parent.is_dir() and not target.exists(),
        'Absolute SPEC and fresh acceptance with existing parent required')
    spec_path = spec_path.resolve(strict=True)
    spec = read_json(spec_path)
    require(spec['schema'] == SPEC_SCHEMA and tuple(spec['modes']) == MODES and tuple(spec['image_ids']) == IMAGE_IDS and
        spec['expected_epoch'] == 24 and spec['hidden_seed'] == 2026100911 and spec['permutation_seed'] == 2026101013 and
        spec['loss_seed_base'] == 1000000 and spec['CPU_affinity'] == [0, 1] and
        spec['allocator_limit_MiB'] == 4096 and spec['whole_GPU_owner_limit_MiB'] == 8192 and
        type(spec['deadline_seconds']) is int and 0 < spec['deadline_seconds'] <= 3600 and
        isinstance(spec['phase'], str) and spec['phase'] and Path(spec['phase']).name == spec['phase'],
        'Exact four-arm frozen native technical SPEC/resource/seed scope differs')
    for role in ARTIFACT_PINS:
        pinned_role(spec, role)
    require(spec['teacher_phase'] == TEACHER_PHASE, 'Canonical teacher owner phase differs')
    receipt_path = spec_path.parent / 'finite_point_cache_execution_20261005_v2' / 'runs' / spec['phase'] / 'PARENT_RECEIPT.json'
    receipt = read_json(receipt_path)
    terminal(receipt, spec['phase'])
    teacher_receipt = read_json(spec['teacher_receipt']['path'])
    terminal(teacher_receipt, spec['teacher_phase'])
    # Raw owned receipts above precede ALL SOURCE/artifact hashes, Q/CP/trace
    # reads and scientific imports. The teacher receipt itself is then pinned.
    execution = source_gate(spec_path, spec, receipt)
    training, teacher_q = teacher_lineage(spec, teacher_receipt)
    fixture_gate(spec)
    output = Path(spec['output_directory'])
    require(output.is_absolute(), 'Absolute actual preflight output directory required')
    report_path, qualified_path = output / 'REPORT.json', output / 'QUALIFIED.json'
    report, qualified = read_json(report_path), read_json(qualified_path)
    require(report['schema'] == REPORT_SCHEMA and report['phase'] == spec['phase'] and report['error'] is None and
        report['status'] == 'TECHNICAL_QUERY_CONDITIONED_FROZEN_NATIVE_PREFLIGHT_COMPLETE_NO_METHOD_RESULT' and
        report['source'] == execution['producer'] and report['spec'] == execution['spec'] and
        report['no_AP'] is True and report['no_DEV'] is True and report['formal_resume_eligible'] is False and
        report['scientific_method_decision_computed'] is False,
        'Actual completed frozen-native technical report/source/scope differs')
    mode_rows = qualified['mode_rows']
    require(set(mode_rows) == set(report['cleanup']) == set(MODES) and report['mode_rows'] == mode_rows,
        'All four fresh report/aggregate/case-cleanup rows must match')
    require(qualified['schema'] == 'query-conditioned-frozen-native-preflight-technical-aggregate-v1' and
        qualified['status'] == 'ROOT_QUERY_CONDITIONED_FROZEN_NATIVE_PREFLIGHT_TECHNICAL_PENDING_CPU_READER' and
        qualified['independently_CPU_qualified'] is False and qualified['efficacy_claimed'] is False and
        qualified['AP_computed'] is False and qualified['formal_resume_eligible'] is False and
        qualified['report'] == reference(report_path) and qualified['source'] == execution['producer'] and
        qualified['spec'] == execution['spec'] and qualified['teacher_checkpoint'] == spec['teacher_checkpoint'] and
        qualified['teacher_qualification'] == spec['teacher_qualification'] and qualified['teacher_receipt'] == spec['teacher_receipt'],
        'GPU qualification report/new source/SPEC/teacher full CP joins differ')
    for key, value in dict(modes=list(MODES), image_ids=list(IMAGE_IDS), microsteps_per_mode=4,
        optimizer_updates_per_mode=2, native_loss_only=True, native_loss_atoms=30, frozen_native=True,
        native610_exact=True, all_four_CPU_models_before_CUDA=True, only_one_model_on_CUDA_at_a_time=True,
        teacher_checkpoint=spec['teacher_checkpoint'], teacher_qualification=spec['teacher_qualification'],
        teacher_training_spec=spec['teacher_training_spec'], teacher_receipt=spec['teacher_receipt'],
        operator_CPU_fixture=spec['operator_CPU_fixture'], AP_computed=False, efficacy_claimed=False,
        teacher_optimizer_loaded=False, teacher_order_RNG_loaded=False, teacher_external_RNG_loaded=False,
        allocator_limit_MiB=4096, GPU_scope='whole GPU must be verified from actual owner receipt').items():
        require(report[key] == value, 'Actual completed four-arm report contract differs: ' + key)
    start_path = output / 'START.json'
    require(read_json(start_path) == dict(schema='query-conditioned-frozen-native-preflight-start-v1',
        phase=spec['phase'], source=execution['producer'], spec=execution['spec'], modes=list(MODES), image_ids=list(IMAGE_IDS),
        teacher_checkpoint=spec['teacher_checkpoint'], teacher_qualification=spec['teacher_qualification'],
        teacher_receipt=spec['teacher_receipt'], operator_CPU_fixture=spec['operator_CPU_fixture'],
        teacher_terminal_gate_before_scientific_access=True, all_four_CPU_models_before_CUDA=True,
        teacher_optimizer_loaded=False, teacher_order_RNG_loaded=False, teacher_external_RNG_loaded=False,
        fresh_optimizer_state_entries=0, no_formal_resume=True, no_DEV=True, no_AP=True, no_retry=True),
        'Actual pre-import technical START/source/SPEC/teacher/state-only/fresh optimizer scope differs')
    for row in report['cleanup'].values():
        require(type(row['allocated_before']) is int and row['allocated_before'] >= 0 and
            row['allocated_after'] == 0, 'Actual per-arm GPU cleanup must report exact0 allocated')
    require(type(report['torch_peak_allocated_bytes']) is int and
        0 <= report['torch_peak_allocated_bytes'] <= 4096 * 1024 ** 2 and
        type(report['elapsed_seconds']) in (int, float) and math.isfinite(report['elapsed_seconds']) and
        0 <= report['elapsed_seconds'] < spec['deadline_seconds'], 'Actual allocator/wall limit differs')
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
        'Independent CPU01 and noCUDA-init gate failed')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    cv2.setNumThreads(2)
    register_all_modules(init_default_scope=False)
    init_default_scope('mmdet')
    require(len(CocoDataset.METAINFO['classes']) == 80 and CocoDataset.METAINFO['classes'][8] == 'boat',
        'Actual official COCO80 boat8 metadata order differs')
    old = load_source(spec['reused_CPU_reader'], 'independent_original_native_loss_CPU_helpers')
    teacher_reader = load_source(spec['training_CPU_reader'], 'independent_qualified_native24_CPU_helpers')
    initializer = load_source(spec['initializer_source'], 'independent_qualified_initializer_CPU_helpers')
    geometry = load_source(spec['geometry_source'], 'sealed_original_geometry_CPU_constructor')
    readout = load_source(spec['readout_source'], 'query_pair_readout_CPU_constructor')
    hooks = load_source(spec['hooks_source'], 'query_pair_whole_head_CPU_registration')
    require(tuple(readout.MODES) == tuple(hooks.MODES) == MODES and hooks.REGISTRATION_NAME == 'query_pair_readout' and
        readout.INITIALIZATION_SEED == 2026100911 and readout.PERMUTATION_SEED == 2026101013,
        'Actual pinned four-mode operator/bridge seed/prefix differs')
    fit = read_ref(spec['FIT_annotation'])
    images = {int(row['id']): row for row in fit['images']}
    order = [int(row['id']) for row in fit['images']]
    require(len(images) == len(order) == 741 and len(fit['annotations']) == 1661 and
        len({int(ann['id']) for ann in fit['annotations']}) == 1661 and set(IMAGE_IDS) <= set(images),
        'Pinned native FIT741/1661 population differs')
    coco = COCO(str(bound(spec['FIT_annotation'])))
    rng_before = torch.get_rng_state().clone()
    model, teacher = load_teacher(torch, old, teacher_reader, initializer, MODELS,
        revert_sync_batchnorm, Config, spec, training, teacher_q)
    direct, P = independent_adapter(torch)
    proofs, all_rows = {}, {}
    for mode in MODES:
        arm_model = copy.deepcopy(model)
        proofs[mode], all_rows[mode] = check_training(torch, np, cv2, coco, old, spec, mode_rows[mode],
            arm_model, teacher, readout, geometry, hooks, direct, P, images, order, mode, execution)
        del arm_model
        gc.collect()
    for micro in range(4):
        first = all_rows[MODES[0]][micro]
        for mode in MODES[1:]:
            row = all_rows[mode][micro]
            for key in ('image_id', 'identity', 'GT', 'input', 'private_seed', 'private_RNG_before', 'external_RNG_before'):
                require(row[key] == first[key], 'Actual common input/GT/private seed sequence differs: ' + key)
            require(row['bridge_inputs']['feature'] == first['bridge_inputs']['feature'] and
                row['bridge_inputs']['coefficient'] == first['bridge_inputs']['coefficient'] and
                row['bridge_inputs']['valid_fraction'] == first['bridge_inputs']['valid_fraction'] and
                row['head_fields']['classes'] == first['head_fields']['classes'] and
                row['head_fields']['native_masks'] == first['head_fields']['native_masks'],
                'Same immutable native teacher feature/coefficient/class/native-mask digests differ')
            for key in ('eligible', 'eligible_counts', 'canonical_eligible_counts', 'rho_mass', 'support_nonempty'):
                require(row['readout_observation'][key] == first['readout_observation'][key],
                    'Actual held-fixed detached support observation differs across input modes: ' + key)
    require(torch.equal(rng_before, torch.get_rng_state()) and not torch.cuda.is_initialized() and
        torch.get_num_threads() == torch.get_num_interop_threads() == 2 and psutil.Process().cpu_affinity() == [0, 1],
        'CPU qualification changed external RNG, affinity/threads or initialized CUDA')
    acceptance = dict(schema=QUAL_SCHEMA, status=QUAL_STATUS, phase=spec['phase'], spec=execution['spec'],
        source=execution['producer'], reader_source=execution['consumer'], report=reference(report_path), start=reference(start_path),
        GPU_qualification=reference(qualified_path), parent_receipt=reference(receipt_path),
        GPU_admission=execution['GPU_admission'], owner_driver=execution['owner'],
        teacher_training_spec=spec['teacher_training_spec'], teacher_qualification=spec['teacher_qualification'],
        teacher_checkpoint=spec['teacher_checkpoint'], teacher_receipt=spec['teacher_receipt'],
        teacher_original_admission=TEACHER_ADMISSION, operator_CPU_fixture=spec['operator_CPU_fixture'],
        FIT_annotation=spec['FIT_annotation'], modes=list(MODES), image_ids=list(IMAGE_IDS),
        training_proofs=proofs, checkpointmap={mode: proof['checkpoint'] for mode, proof in proofs.items()},
        actual_trace_records=16, source_closure_verified_before_scientific_import=True,
        raw_current_and_teacher_terminal_gates_before_scientific_access=True,
        teacher_full610_exact_before_after_every_arm=True, native451_frozen_and_grads_None=True,
        typed_BN106_affine159_buffers_independently_exact=True, only_two_weights_changed=True,
        every_AdamW_moment_and_actual_step2_verified=True, all_raw_GT_independently_reconstructed=True,
        ten_native_assignments_cover_all_raw_GT=True, native30_loss_total_consistent=True,
        common_input_and_loss_seed_sequence_verified=True, common_realized_matching_or_points_claimed=False,
        cold_hidden_zero_final_positive_and_warm_two_weight_positive_source_observed=True,
        safe_whole_head_final_only_hooks_source_observed=True, actual_native_runtime_technically_qualified=True,
        CPU_affinity=[0, 1], torch_threads=2, CUDA_initialized=False, external_CPU_RNG_restored=True,
        formal_training_qualified=False, eligible_for_formal_resume=False, AP_computed=False,
        scientific_method_verdict=None, efficacy_claimed=False, CPU_forward_or_backward_replayed=False,
        PCA_or_M0_sign_eligibility_CPU_replayed=False, RMS_zero_is_exact_zero_vector_claimed=False,
        evidence_limits=[
            'Four independent fresh four-micro/two-update technical cases only; no AP, efficacy, convergence or novelty claim.',
            'CPU verifies exact full teacher/native/readout states, native GT rasters, tensor identities and AdamW storage; it does not replay GPU forward/backward.',
            'Cold identity, gradients, safe hook removal and native gradNone are bound source-observed runtime evidence checked for internal consistency.',
            'Small E/support/RMS packets are independently checked, without recomputing native geometry, PCA or hard-sign eligibility.',
            'FP32 reported zero input RMS does not certify an exactly zero channel vector; positive pre-RMS and normalized near-unit observations are checked.',
            'Common original input/native teacher/loss seed sequence does not assert common realized point sampling or matching after residual divergence.',
            'Technical checkpoints are ineligible for formal resume and cannot establish comparison against external methods.'])
    with target.open('x', encoding='utf-8') as stream:
        json.dump(acceptance, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=acceptance['status'], acceptance=reference(target)), sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--acceptance', required=True)
    parser.add_argument('--owner-execute', action='store_true')
    args = parser.parse_args()
    require(args.owner_execute, 'SOURCE_ONLY: explicit ROOT CPU execution after raw owned terminals required')
    qualify(args.spec, args.acceptance)


if __name__ == '__main__':
    main()
