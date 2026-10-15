"""UNBOUND_NOT_EXECUTED: repeat202 GPU metadata front, no numerical imports/run/CLI.

Assembly prerequisite: unchanged typed RAW1..26 / inherited_source_gate prefix
from the completed F/P entry. Its CURRENT GPU dispatch uses N20_DEV_SCHEMA below;
historical SCHEMA/CPU_SCHEMA/N20_TRAIN_SCHEMA and failure checks stay unchanged.
Required globals from that prefix: require/ref_shape/read/closed/load, and
n20_failed22..26_raw_only/closed. Never import a historical entry and monkeypatch.

New spec fields: training_owner_index (explicit current owner, >=181),
N20_first201_ROOT_qualified_result + expected_N20_first201_ROOT_qualified_result,
N20_repeat202_initialization_qualified_result,
N20_repeat202_initialization_record (exact8, output key metadata),
N20_repeat202_cleanup_probe_qualified_result. Existing N20_initialization_lock,
expected_N20_initialization_digest, four model refs and N20_training_tail_source
describe NEW202; old201 fields are not overwritten in any historical artifact.

Tail ABI is the original common/parent/lineage/context/closure/source/spec_ref/
admission_ref plus parent_cache_report/_ref, neighbor_report/_ref, protocol,
training_binding_refs. Added decoded first201_root_permission and
initialization_metadata are metadata only. All binding_refs values are strictrefs.
Root alone assembles/freezes/admit/executes, after original terminals and pure
prefix actual0. This draft performs no reads on import or future qualification.
"""

import ast
import json
import math
from pathlib import Path

N20_REPEAT202_TRAIN_SCHEMA = 'ROOT-N20-repeat202-paired-local-center-feature-error-training-GPU-spec-v1'
N20_DEV_SCHEMA = N20_REPEAT202_TRAIN_SCHEMA  # Only CURRENT inherited GPU dispatch.
N20_REPEAT202_SEED = 202610150202
N20_REPEAT202_MODEL_KEYS = ('scout_module_source', 'base_scout_source',
    'paired_training_body_source', 'original_paired_training_body_source')
N20_REPEAT202_ROOT_PINS = {
    'N20_first201_ROOT_qualified_result': (537072, 'eca7fff506119bb87b7e1b32ead8ff143ef89de0fcd6ac9faa08cae887e7f90f'),
    'N20_repeat202_initialization_qualified_result': (410671, 'e0314dfd1a2d5761bfc5ea98f5d8260e79d5f7a4fb929e4b45d3f704edf18975'),
    'N20_repeat202_cleanup_probe_qualified_result': (407613, '52042656353c1c43ff488ee3d22062d88e5f5b03778d5dfdcf3c2c1e58297b9b'),
}


def n20_repeat202_exact(a, b):
    return json.dumps(a, sort_keys=True, separators=(',', ':'), allow_nan=False) == json.dumps(
        b, sort_keys=True, separators=(',', ':'), allow_nan=False)


def n20_repeat202_root(spec, gate, key, count, schema):
    ref = spec[key]; ref_shape(ref)
    require((ref['bytes'], ref['sha256']) == N20_REPEAT202_ROOT_PINS[key],
            'Exact independently completed ROOT metadata artifact: ' + key)
    value = read(closed(gate['closure'], ref))
    require(value['schema'] == schema and value['qualified'] is True
        and value['normal_RAW_count'] == count and value['total_failed_RAW_count'] == 26
        and value['total_terminal_identity_pin_count'] == count + 26
        and value['RAW_inputs'] == spec['RAW_inputs'][:count]
        and len(value['terminal_identity_pins']) == count + 8
        and all(spec['terminal_identity_pins'].get(k) == v
                for k, v in value['terminal_identity_pins'].items()),
        'ROOT actual completion joins the current full RAW ancestry, never a live/pending output')
    return value


def n20_repeat202_completed(record, output_key, index, cuda, gate, spec):
    require(type(record) is dict and set(record) == {'source','spec',output_key,'RAW',
        'admission','phase_binding','bootstrap','owner_index'}
        and type(record['owner_index']) is int and record['owner_index'] == index,
        'Exact8 completed owner record, with its declared output role')
    closure = gate['closure']
    for key in set(record) - {'owner_index'}:
        closed(closure, record[key])
    row = spec['RAW_inputs'][index]
    require(all(row[k] == record[v] for k, v in (('receipt','RAW'),('admission','admission'),
        ('phase_binding','phase_binding'),('bootstrap','bootstrap'))),
        'Completed record equals original RAW row')
    joined = gate['lineage'].owner_ref_join(gate['context'], gate['owners'],
        record['RAW'], record['admission'], record['source'], record['spec'], cuda)
    parent = read(closed(closure, record['spec']))
    if index == 179 and output_key == 'metadata':
        require(parent['schema'] == 'ROOT-N20-source-only-repeat202-initialization-spec-v1'
            and 'expected_normal_RAW_count' not in parent and parent['normal_RAW_count'] == 179,
            'Pinned init179 source-only spec has normal179 and no expected counter field')
    elif index == 180 and output_key == 'report':
        require(parent['schema'] == 'ROOT-N20-repeat202-synthetic-GPU-cleanup-probe-spec-v1'
            and 'expected_normal_RAW_count' not in parent and parent['normal_RAW_count'] == 180,
            'Pinned probe180 synthetic spec has normal180 and no expected counter field')
    else:
        require(parent['expected_normal_RAW_count'] == index,
                'All original F/P/cache parent expected counters remain strictly required')
    require(type(joined) is int and joined == index == parent['normal_RAW_count']
        and parent['entry_source'] == record['source']
        and parent['phase'] == row['phase'] and parent['RAW_inputs'] == spec['RAW_inputs'][:index]
        and all(closure.get(r['path']) == r for r in parent['sources'])
        and all(spec['terminal_identity_pins'].get(k) == v
                for k, v in parent['terminal_identity_pins'].items()),
        'Original source/spec/ad/phase/bootstrap/actual owner/prefix stays exact')
    admission_key = ('GPU' if cuda == '0' else 'CPU') + '_admission'
    require(Path(parent[admission_key + '_file']).resolve(strict=True) ==
            Path(record['admission']['path']).resolve(strict=True)
        and Path(record[output_key]['path']).resolve(strict=True).parent ==
            Path(parent['output_directory']).resolve(strict=True), 'Original output/admission locator')
    output = read(closed(closure, record[output_key]))
    require(output['source'] == record['source'] and output['spec'] == record['spec']
        and output[admission_key] == record['admission'] and output['phase'] == row['phase'],
        'Original output binds its completed owner, not the new training owner')
    return parent, output


def n20_repeat202_first_permission(spec, gate):
    root = n20_repeat202_root(spec, gate, 'N20_first201_ROOT_qualified_result', 179,
        'ROOT-N20-post178-externally-qualified-first201-result-v1')
    require(n20_repeat202_exact(root, spec['expected_N20_first201_ROOT_qualified_result'])
        and root['status'] == 'ROOT_FINAL177_AND_POST178_ACTUAL0_FIRST201_MAIN_RESULT_QUALIFIED'
        and type(root['original_terminal_chunk']) is str
        and root['original_terminal_chunk'] not in ('', 'TO_BE_FILLED_BY_ROOT')
        and root['pure179_actual_exit_code'] == 0 and root['all_original_pending_outputs_preserved'] is True,
        'Full prelocked first201 external qualification, not a preview')
    q = root['final_qualified_result']; decision = root['primary_interpretation']
    require(q['qualified'] is True and q['original_final_CPU_actual0_verified'] is True
        and q['images'] == 187 and q['GT'] == 380 and q['natural_queries'] == 18700
        and q['initialization_seed'] == 202610150201 and len(q['numeric_CPU_shards']) == 32
        and len(q['completed_GPU_shards']) == 4 and decision == q['primary_interpretation']
        and decision['seeds'] == [202610150201] and decision['main_epoch'] == 24
        and decision['main_mode'] == 'atmost' and decision['AP_unit'] == 'percent',
        'Only complete first201 fixed E24-atmost population authorizes repeat202')
    results = q['control_results']
    for field, name in (('LOCAL_AP_percent','E24_LOCAL_ERROR_atmost'),
                        ('CENTER_AP_percent','E24_CENTER_ERROR_atmost'),
                        ('strong_AP_percent','strong_fixed_E24_Omega')):
        raw = results[name]['stats'][0]
        require(type(raw) in (int,float) and math.isfinite(raw) and 0 <= raw <= 1
            and decision[field] == raw * 100, 'Permission AP binds actual full COCO stats: ' + field)
    delta = decision['LOCAL_AP_percent'] - max(decision['CENTER_AP_percent'], decision['strong_AP_percent'])
    require(decision['strong_AP_percent'] == 68.09938494745545
        and delta == decision['Delta_AP_points'] and 0 < delta < 1
        and decision['decision'] == 'ONE_EXTRA_SEED'
        and decision['allowed_next_seeds'] == [N20_REPEAT202_SEED]
        and decision['additional_budget_or_fusion_allowed'] is False
        and decision['qualification_performed'] is False,
        'Actual first201 allows only prelocked202, no203/budget/fusion or efficacy conclusion')
    for key in ('formal_TEST_evaluated','cross_dataset_evaluated',
                'task_efficacy_established','innovation_established','TGRS_readiness_established'):
        require(root[key] is False and q[key] is False, 'Preserve first201 evidence limit: ' + key)
    post_record = dict(source=root['post_source'], spec=root['post_spec'],
        result=root['post_pending_result'], RAW=root['post_RAW'], admission=root['post_admission'],
        phase_binding=root['post_phase_binding'], bootstrap=root['post_bootstrap'], owner_index=178)
    post_spec, post = n20_repeat202_completed(post_record, 'result', 178, '', gate, spec)
    require(post['final_record'] == q['final_owner'] and post['final_qualified_result'] == q
        and post['qualified'] is False and post['ROOT_post_actual_CPU0_verified'] is False
        and post['current_post_owned_terminal_self_certified'] is False
        and post_spec['initialization_seed'] == 202610150201
        and q['original_training_GPU_actual0_verified'] is False
        and q['training_artifacts_recovery_qualified'] is True,
        'P178 actual0 evidence retains pending output and original failed training/recovery')
    final_spec, final = n20_repeat202_completed(q['final_owner'], 'Q', 177, '', gate, spec)
    require(final['qualified'] is False and final['ROOT_actual_CPU0_verified'] is False
        and final['control_results'] == results
        and final['primary_interpretation_preview_pending_ROOT_actual0'] == decision
        and final_spec['initialization_seed'] == 202610150201,
        'Original F177 pending result links full first201 results without parent own_gate/run')
    for key in ('pure179_source','held_actual_file'):
        closed(gate['closure'], root[key])
    return root, post_spec


def n20_repeat202_history_view(spec, gate, first201_spec):
    """Only historical closed checks see201 science; own gate already saw202."""
    stable = ('N20_root_cache_qualified_result','expected_N20_root_cache_qualified_result',
        'N20_locked_protocol','expected_N20_locked_protocol','locked_protocol',
        'cache_GPU_report','region_common_source','N20_failed_training_record',
        'N20_failed_training_stderr','N20_failed_DEV_capture_record','N20_failed_DEV_capture_stderr',
        'N20_failed_CPU_numeric_record','N20_failed26_helper_source','failed26_partials_reused')
    require(all(n20_repeat202_exact(spec[k], first201_spec[k]) for k in stable)
        and all(n20_repeat202_exact(spec[k], v) for k, v in first201_spec.items()
                if k.startswith('additional_preserved_')),
        'Same cache/protocol/all exact historical failures; only new202 initialization/sources are separate')
    view = dict(first201_spec)
    for key in ('sources','RAW_inputs','normal_RAW_count','expected_normal_RAW_count',
                'total_failed_RAW_count','total_terminal_identity_pin_count','terminal_identity_pins'):
        view[key] = spec[key]
    require(view['initialization_seed'] == 202610150201
        and all(gate['closure'].get(r['path']) == r for r in first201_spec['sources']),
        'Authentic completed first201 history remains completely closed, never rewritten to202')
    return view


def n20_repeat202_cache(spec, gate):
    c = gate['closure']; protocol = read(closed(c, spec['N20_locked_protocol']))
    require(n20_repeat202_exact(protocol, spec['expected_N20_locked_protocol'])
        and protocol['schema'] == 'N20-true-feature-neighborhood-fixed24-protocol-v1'
        and protocol['first_initialization_seed'] == 202610150201
        and protocol['repeat_initialization_seeds'] == [202610150201,202610150202,202610150203],
        'Original fixed protocol, repeat list and complete expectedJSON remain unchanged')
    root = read(closed(c, spec['N20_root_cache_qualified_result']))
    require(n20_repeat202_exact(root, spec['expected_N20_root_cache_qualified_result'])
        and root['schema'] == 'Root-N20-fullFIT-true-feature-neighbor-cache-post-actual-qualified-v1'
        and root['status'] == 'FULLFIT_TRUE_FEATURE_NEIGHBOR_CACHE_QUALIFIED'
        and all(root[k] is True for k in ('qualified','original_GPU_CPU_actual0_verified','full_proofs_ROOT_verified'))
        and root['images'] == 741 and root['sampled_points'] == 3035136 and root['unique_centers'] == 1786820
        and root['parent_cache'] == spec['cache_GPU_report'] and root['protocol'] == spec['N20_locked_protocol']
        and root['GPU_report'] == root['gpu_record']['report'] and root['CPU_Q'] == root['CPU_record']['Q'],
        'Only original fullFIT741 independently qualified same-input neighborhood cache')
    gpu_spec, report = n20_repeat202_completed(root['gpu_record'], 'report', 138, '0', gate, spec)
    cpu_spec, cpu_q = n20_repeat202_completed(root['CPU_record'], 'Q', 139, '', gate, spec)
    require(all(p['total_failed_RAW_count'] == 21 and p['total_terminal_identity_pin_count'] == i + 21
                for p, i in ((gpu_spec,138),(cpu_spec,139)))
        and Path(root['GPU_report']['path']).name == 'REPORT.json' and Path(root['CPU_Q']['path']).name == 'Q.json'
        and report['schema'] == 'N20-true-feature-neighbor-cache-GPU-report-v1'
        and report['status'] == 'N20_FULLFIT_TRUE_FEATURE_NEIGHBORS_COMPLETE_PENDING_INDEPENDENT_CPU'
        and report['parent_cache'] == spec['cache_GPU_report'] and report['protocol'] == spec['N20_locked_protocol']
        and cpu_q['schema'] == 'N20-fullFIT-feature-neighbor-cache-CPU-qualification-v1'
        and cpu_q['status'] == 'N20_FULLFIT_FEATURE_NEIGHBOR_CACHE_CPU_COMPLETE_PENDING_ROOT_ACTUAL0'
        and cpu_q['capture_GPU_report'] == root['GPU_report'] and cpu_q['protocol'] == spec['N20_locked_protocol']
        and all(cpu_q[k] is False for k in ('qualified','ROOT_actual_CPU0_verified','original_owned_terminal_self_certified'))
        and cpu_q['images'] == 741 and cpu_q['sampled_points'] == 3035136 and cpu_q['unique_centers'] == 1786820
        and cpu_q['extra_feature_bytes'] == report['extra_feature_bytes']
        and gpu_spec['N20_locked_protocol'] == cpu_spec['N20_locked_protocol'] == spec['N20_locked_protocol'],
        'GPU138/CPU139 original history21 and immutable pending cache outputs beneath ROOT qualification')
    closed(c, cpu_q['image_proofs']); old = read(closed(c, spec['cache_GPU_report']))
    require(old['images'] == report['images'] == 741
        and old['sampled_points'] == report['sampled_points'] == 3035136
        and len(old['rows']) == len(old['packet_refs']) == len(report['rows']) == len(report['packet_refs']) == 741,
        'Complete original and new cache rows, not a subset')
    for index, (before, row) in enumerate(zip(old['rows'], report['rows'])):
        require(index == before['dataset_index'] == row['dataset_index']
            and before['packet'] == old['packet_refs'][index] and row['packet'] == report['packet_refs'][index]
            and row['parent_packet'] == before['packet'] and row['identity'] == before['identity']
            and row['image_id'] == before['image_id'] and row['points'] == before['points'] == 4096
            and row['unique_centers'] == before['unique_sampled_F_points'], 'Exact FIT cache identity/order/sample binding')
        closed(c, before['packet']); closed(c, row['packet'])
    return protocol, root, old, report


def n20_repeat202_digest_algorithm(helper_ref, tail_ref, closure):
    """Compare pure digest algorithm AST, never import/execute either source."""
    def evidence(ref):
        source = Path(closed(closure, ref)).read_text(encoding='utf-8-sig')
        tree = ast.parse(source); functions = [n for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == 'n20_training_state_digest']
        require(len(functions) == 1, 'One original N20 digest function')
        constants = {}
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in ('N20_TRAIN_SEED','N20_TRAIN_ORDER_SEED','N20_TRAIN_ARMS'):
                        constants[target.id] = ast.literal_eval(node.value)
        require(constants.get('N20_TRAIN_SEED') == N20_REPEAT202_SEED,
                'Both source-only digest helper and actual training tail are explicitly202')
        return ast.dump(functions[0], include_attributes=False), constants
    a, ca = evidence(helper_ref); b, cb = evidence(tail_ref)
    require(a == b and ca == cb, 'Exact digest algorithm/order/arms; whole different tails need not share SHA')


def n20_repeat202_initialization(spec, gate, first201_spec):
    c = gate['closure']
    root = n20_repeat202_root(spec, gate, 'N20_repeat202_initialization_qualified_result', 180,
        'ROOT-N20-repeat202-source-only-initialization-qualified-result-v1')
    record = spec['N20_repeat202_initialization_record']
    require(root['original_CPU179_actual0_verified'] is True and root['initialization_record'] == record
        and root['initialization_lock'] == record['metadata'] == spec['N20_initialization_lock']
        and root['initialization_seed'] == N20_REPEAT202_SEED
        and root['initialization_digest'] == spec['expected_N20_initialization_digest']
        and root['source_only'] is True and root['metadata_original_pending_preserved'] is True
        and all(root[k] is False for k in ('training_executed','SAR_data_read','CUDA_initialized','AP_evaluated',
            'task_efficacy_established','innovation_established','TGRS_ready')),
        'Independent completed CPU179 source-only initialization, never a training result')
    parent, metadata = n20_repeat202_completed(record, 'metadata', 179, '', gate, spec)
    require(parent['schema'] == 'ROOT-N20-source-only-repeat202-initialization-spec-v1'
        and parent['initialization_seed'] == N20_REPEAT202_SEED and parent['total_failed_RAW_count'] == 26
        and parent['total_terminal_identity_pin_count'] == 205
        and parent['first201_external_qualification_sha256'] ==
            spec['N20_first201_ROOT_qualified_result']['sha256']
        and parent['first201_initialization_digest'] == first201_spec['expected_N20_initialization_digest']
        and parent['first201_initialization_digest'] != spec['expected_N20_initialization_digest']
        and metadata == root['expected_initialization_lock']
        and metadata['schema'] == 'Root-N20-source-only-private-initialization-lock-v1'
        and metadata['seed'] == N20_REPEAT202_SEED and metadata['combined_digest'] == spec['expected_N20_initialization_digest']
        and all(metadata[k] is True for k in ('paired_state_exact','paired_storage_independent',
            'spatial_weights_zero','external_CPU_RNG_unchanged','source_only'))
        and all(metadata[k] is False for k in ('CUDA_initialized','training_executed','qualified',
            'ROOT_actual_CPU0_verified','current_owned_terminal_self_certified'))
        and metadata['torch_version'] == '2.0.1+cu118' and metadata['CPU_affinity'] == [0,1] and metadata['threads'] == 2,
        'Exact new202 initialization metadata/digest, preserving original pending flags')
    sources = metadata['source_sha256']; require(type(sources) is dict, 'Initialization basename SHA map')
    model_refs = [spec[k] for k in N20_REPEAT202_MODEL_KEYS]
    require(len({Path(r['path']).name for r in model_refs}) == 4
        and all(sources.get(Path(r['path']).name) == r['sha256'] for r in model_refs),
        'All four actual model-source basenames and hashes equal the independently initialized mathematics')
    model_names = {Path(r['path']).name for r in model_refs}
    helper_names = set(sources) - model_names
    require(len(helper_names) == 1, 'Exactly one distinct source-only digest-helper basename')
    name = next(iter(helper_names))
    helpers = [r for r in parent['sources'] if Path(r['path']).name == name and r['sha256'] == sources[name]]
    require(len(helpers) == 1, 'Original digest-helper strictref from completed init Source closure')
    n20_repeat202_digest_algorithm(helpers[0], spec['N20_training_tail_source'], c)
    return root, metadata


def n20_repeat202_cleanup_probe(spec, gate):
    root = n20_repeat202_root(spec, gate, 'N20_repeat202_cleanup_probe_qualified_result', 181,
        'ROOT-N20-repeat202-synthetic-cleanup-probe-qualified-result-v1')
    require(root['original_GPU180_actual0_verified'] is True and root['synthetic_only'] is True
        and root['metadata_original_pending_preserved'] is True
        and all(root[k] is False for k in ('SAR_training_executed','original_failure24_cause_established',
            'AP_evaluated','task_efficacy_established','innovation_established','TGRS_ready')),
        'Completed synthetic cleanup probe does not qualify SAR training or uniquely explain failure24')
    parent, report = n20_repeat202_completed(root['probe_record'], 'report', 180, '0', gate, spec)
    require(parent['total_failed_RAW_count'] == 26 and parent['total_terminal_identity_pin_count'] == 206
        and report['cleanup'] == root['cleanup'] and root['cleanup']['allocated_after'] == 0
        and root['cleanup']['reserved_after'] == 0
        and root['cleanup']['preserved_failure24_unique_root_cause_established'] is False
        and report['synthetic_steps_per_arm'] == 8 and report['points_per_step'] == 4096
        and report['actual_head_parameters'] == 18851 and report['actual_head_tensors'] == 5
        and report['history_and_orders_preserved'] is True
        and all(report[k] is False for k in ('SAR_data_read','SAR_training_executed','AP_evaluated',
            'original_failure24_cause_established','qualified','ROOT_actual_GPU0_verified',
            'current_owned_terminal_self_certified')),
        'Original GPU180 synthetic report/zero final allocator counters; no failed partial reuse')
    tested_tail = report['new_tail_source']; closed(gate['closure'], tested_tail)
    require((tested_tail['bytes'], tested_tail['sha256']) ==
            (spec['N20_training_tail_source']['bytes'], spec['N20_training_tail_source']['sha256']),
        'New training tail bytes equal the independently exercised cleanup tail; paths remain original')
    return root


def source_gate(path, spec):
    owner = spec['training_owner_index']
    require(spec['schema'] == N20_REPEAT202_TRAIN_SCHEMA and type(owner) is int and owner >= 181
        and spec['normal_RAW_count'] == spec['expected_normal_RAW_count'] == owner
        and spec['total_failed_RAW_count'] == 26 and spec['total_terminal_identity_pin_count'] == owner + 26
        and spec['initialization_seed'] == N20_REPEAT202_SEED and spec['deadline_seconds'] == 1800
        and spec['automatic_retry'] is False and spec['maximum_process_RSS_bytes'] == 16*1024**3
        and spec['allocator_limit_MiB'] == 4096 and spec['whole_GPU_limit_MiB'] == 8192
        and spec['failed26_partials_reused'] is False and Path(path).is_absolute()
        and Path(__file__).resolve(strict=True) == Path(spec['entry_source']['path']).resolve(strict=True),
        'Explicit new202 GPU own source/owner/resources; current owner is not historical140 or fixed181')
    # Mandatory original RAW-only checks happen before inherited first SOURCE SHA.
    raw22 = n20_failed22_raw_only(spec); raw23 = n20_failed23_raw_only(spec)
    raw24 = n20_failed24_raw_only(spec); raw25 = n20_failed25_raw_only(spec)
    raw26 = n20_failed26_raw_only(spec)
    gate = inherited_source_gate(path, spec, cpu=False)
    require(gate['source'] == spec['entry_source'] and gate['context']['normal_RAW_count'] == owner
        and gate['context']['RAW_inputs'] == spec['RAW_inputs'] and 'parent' in gate,
        'Current GPU own_gate/context must remain current202, never a completed parent own_gate')
    root, history_spec = n20_repeat202_first_permission(spec, gate)
    historical = n20_repeat202_history_view(spec, gate, history_spec)
    n20_failed22_closed(historical, gate, raw22); n20_failed23_closed(historical, gate, raw23)
    n20_failed24_closed(historical, gate, raw24); n20_failed25_closed(historical, gate, raw25)
    n20_failed26_closed(historical, gate, raw26)
    for key in N20_REPEAT202_MODEL_KEYS + ('N20_training_tail_source','N20_initialization_lock','region_common_source'):
        closed(gate['closure'], spec[key])
    protocol, cache_root, old_report, report = n20_repeat202_cache(spec, gate)
    init_root, metadata = n20_repeat202_initialization(spec, gate, history_spec)
    probe_root = n20_repeat202_cleanup_probe(spec, gate)
    common = load(spec['region_common_source'], '_N20_repeat202_original_library_gate')
    require(callable(getattr(common, 'imported_files', None)), 'Original imported_files library ABI')
    gate.update(common=common, parent_cache_report=old_report, parent_cache_report_ref=spec['cache_GPU_report'],
        neighbor_report=report, neighbor_report_ref=cache_root['GPU_report'], protocol=protocol,
        first201_root_permission=root, initialization_metadata=metadata,
        repeat202_initialization_qualification=init_root, repeat202_cleanup_probe_qualification=probe_root,
        training_binding_refs=dict(N20_root_cache_qualified_result=spec['N20_root_cache_qualified_result'],
            N20_initialization_lock=spec['N20_initialization_lock'], N20_cache_CPU_Q=cache_root['CPU_Q'],
            N20_first201_ROOT_qualified_result=spec['N20_first201_ROOT_qualified_result'],
            N20_repeat202_initialization_qualified_result=spec['N20_repeat202_initialization_qualified_result'],
            N20_repeat202_cleanup_probe_qualified_result=spec['N20_repeat202_cleanup_probe_qualified_result']))
    return gate
