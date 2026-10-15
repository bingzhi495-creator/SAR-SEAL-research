"""UNBOUND N20 eight-scope DEV CPU metadata front; no RAW prefix, run or CLI.

ROOT appends the existing typed24 RAW-first prefix and changes only its current
CPU_SCHEMA branch to N20_DEV_CPU_SCHEMA. Historical SCHEMA/recovery schemas and
failed-parent specs stay unchanged. No GPU source_gate is called or monkeypatched.
Four original GPU141..144 actual0 records must precede CPU145+cid. This draft does
not certify any future execution. CPU source/admission/phase/context remain own.

The shared recovery/legacy/prior metadata function below is copied from ordinary
GPU front02, without its current-owner GPU gate. It takes a CLOSED ORIGINAL GPU
spec: that spec's CPU_adapter_source is the old common-PR decoder. The own CPU
spec's CPU_adapter_source is the new F9 decoder and is never substituted into the
old hook. This is an explicit parent argument, not an edited scientific view.

ROOT supplies inherited_source_gate, typed22/23/24 helpers, legacy_gate and
n20_prior_DEV_carrier_gate exactly as for the original GPU front02. The prefix
continues to require all normal/failed RAW and pin checks before first Source SHA.
Only stdlib is imported here. Runtime tensor/model/numeric/RLE proof is in tail.
"""
import json
from pathlib import Path

N20_DEV_CPU_SCHEMA = 'ROOT-N20-fixedDEV-local-center-feature-error-CPU-numeric-shard-spec-v1'
N20_CPU_RANGES = ((0,24),(24,47),(47,71),(71,94),(94,118),(118,141),(141,164),(164,187))
N20_CPU_PARENT_GPU_IDS = (0,0,1,1,2,2,3,3)
N20_COMPLETED_GPU_KEYS = {'shard_id','dataset_ordinal_range','source','spec','report','RAW',
    'admission','phase_binding','bootstrap','shard_owner_index'}
N20_CPU_SOURCE_KEYS = ('GPU_outer_source','CPU_adapter_source','selector_cpu_proof_source',
    'legacy_selector_proof_source','endpoint_body_source','PR_formula_source','labels_source')
N20_GPU_BINDING_KEYS = ('source','spec','GPU_admission','phase','N20_root_training_qualified_result',
    'N20_locked_protocol','legacy_qualified_result','new_eval_body_source','capture_body_source',
    'coordinates_source','checkpoints','shard_id','dataset_ordinal_range')

N20_DEV_SCHEMA = 'ROOT-N20-fixedDEV-local-center-feature-error-GPU-shard-spec-v1'
N20_TRAIN_SCHEMA = 'ROOT-N20-paired-local-center-feature-error-training-GPU-spec-v1'
N20_TRAIN_CPU_SCHEMA = 'ROOT-N20-paired-feature-error-saved-artifact-recovery-CPU-spec-v1'
N20_DEV_TRAINING_BINDING_KEYS = (
    'source', 'spec', 'GPU_admission', 'phase', 'N20_locked_protocol', 'locked_protocol',
    'parent_cache_report', 'neighbor_report', 'scout_module_source', 'base_scout_source',
    'paired_training_body_source', 'original_paired_training_body_source',
    'initialization_seed', 'order_seed', 'expected_N20_initialization_digest',
    'frozen_teacher_checkpoint', 'frozen_PR_checkpoint', 'N20_root_cache_qualified_result',
    'N20_initialization_lock', 'N20_cache_CPU_Q')
N20_DEV_RANGES = ((0, 47), (47, 94), (94, 141), (141, 187))


def _n20_dev_exact(left, right):
    return json.dumps(left, sort_keys=True, separators=(',', ':'), allow_nan=False) == json.dumps(
        right, sort_keys=True, separators=(',', ':'), allow_nan=False)

def n20_completed_cache_record(record, output_key, owner_index, cuda, gate, spec):
    """Close cache GPU138/CPU139 or recovery CPU140; never completed GPU140."""
    require(type(record) is dict and set(record) == {
        'source', 'spec', output_key, 'RAW', 'admission', 'phase_binding', 'bootstrap', 'owner_index'}
        and type(record['owner_index']) is int and record['owner_index'] == owner_index
        and (owner_index, output_key, cuda) in ((138, 'report', '0'), (139, 'Q', ''), (140, 'Q', '')),
        'Exact eight-field original completion record required')
    closure = gate['closure']
    for key in set(record) - {'owner_index'}:
        closed(closure, record[key])
    row = spec['RAW_inputs'][owner_index]
    require(all(row[key] == record[record_key] for key, record_key in (
        ('receipt', 'RAW'), ('admission', 'admission'), ('phase_binding', 'phase_binding'),
        ('bootstrap', 'bootstrap'))), 'Completed record must equal its original RAW prefix row')
    joined = gate['lineage'].owner_ref_join(gate['context'], gate['owners'],
        record['RAW'], record['admission'], record['source'], record['spec'], cuda)
    parent = read(closed(closure, record['spec']))
    output = read(closed(closure, record[output_key]))
    expected_failed = {138: 21, 139: 21, 140: 24}[owner_index]
    admission_key = 'GPU_admission' if cuda == '0' else 'CPU_admission'
    filename = 'REPORT.json' if output_key == 'report' else 'Q.json'
    require(type(joined) is int and joined == owner_index == parent['normal_RAW_count']
        and parent['expected_normal_RAW_count'] == owner_index
        and parent['RAW_inputs'] == spec['RAW_inputs'][:owner_index]
        and parent['entry_source'] == record['source'] and parent['phase'] == row['phase']
        and parent['total_failed_RAW_count'] == expected_failed
        and parent['total_terminal_identity_pin_count'] == owner_index + expected_failed
        and all(closure.get(ref['path']) == ref for ref in parent['sources'])
        and all(spec['terminal_identity_pins'].get(key) == value
                for key, value in parent['terminal_identity_pins'].items())
        and Path(parent[admission_key + '_file']).resolve(strict=True) ==
            Path(record['admission']['path']).resolve(strict=True)
        and Path(record[output_key]['path']).resolve(strict=True) ==
            (Path(parent['output_directory']) / filename).resolve(strict=True),
        'Completed original owner/source/spec/admission/pins/prefix/output locator must be exact')
    require(output['source'] == record['source'] and output['spec'] == record['spec']
        and output[admission_key] == record['admission'] and output['phase'] == row['phase'],
        'Completed pending output must bind the actual original owner')
    return parent, output


def n20_recovered_training_metadata_gate(spec, gate):
    """Read-only parent metadata; spec is the original closed GPU spec, not own."""
    closure = gate['closure']
    model_keys = ('scout_module_source', 'base_scout_source', 'paired_training_body_source',
                  'original_paired_training_body_source')
    for key in model_keys + ('N20_root_training_qualified_result', 'N20_root_cache_qualified_result',
            'N20_initialization_lock', 'N20_locked_protocol', 'locked_protocol', 'cache_GPU_report',
            'region_common_source', 'new_eval_body_source', 'legacy_qualified_result',
            'N20_training_tail_source', 'N20_failed_training_stderr'):
        closed(closure, spec[key])
    protocol = read(closed(closure, spec['N20_locked_protocol']))
    require(_n20_dev_exact(protocol, spec['expected_N20_locked_protocol'])
        and protocol['schema'] == 'N20-true-feature-neighborhood-fixed24-protocol-v1',
        'Complete original N20 protocol literal, separate from inherited N16 protocol')
    metadata = read(closed(closure, spec['N20_initialization_lock']))
    digest = spec['expected_N20_initialization_digest']
    require(type(digest) is str and len(digest) == 64 and all(c in '0123456789abcdef' for c in digest)
        and digest == 'a13802fc0bf4f23a8ab368b11b15a24b60552baadf1ab8264daf242173854a68'
        and metadata['schema'] == 'Root-N20-source-only-private-initialization-lock-v1'
        and metadata['seed'] == 202610150201 and metadata['combined_digest'] == digest
        and metadata['CUDA_initialized'] is False and metadata['training_executed'] is False
        and all(metadata[key] is True for key in ('paired_state_exact', 'paired_storage_independent',
            'spatial_weights_zero', 'external_CPU_RNG_unchanged')),
        'Use original prelocked source-only initialization; never synthesize unknown digest')
    initialization_sources = [spec[key] for key in model_keys]
    if 'N20_training_tail_source' in spec:
        closed(closure, spec['N20_training_tail_source'])
        initialization_sources.append(spec['N20_training_tail_source'])
    require(type(metadata['source_sha256']) is dict and all(
        metadata['source_sha256'].get(Path(ref['path']).name) == ref['sha256']
        for ref in initialization_sources), 'All original initialization source basenames and hashes')
    cache = read(closed(closure, spec['N20_root_cache_qualified_result']))
    require(_n20_dev_exact(cache, spec['expected_N20_root_cache_qualified_result'])
        and cache['schema'] == 'Root-N20-fullFIT-true-feature-neighbor-cache-post-actual-qualified-v1'
        and cache['status'] == 'FULLFIT_TRUE_FEATURE_NEIGHBOR_CACHE_QUALIFIED'
        and cache['qualified'] is True and cache['original_GPU_CPU_actual0_verified'] is True
        and cache['full_proofs_ROOT_verified'] is True and cache['images'] == 741
        and cache['sampled_points'] == 3035136 and cache['unique_centers'] == 1786820
        and cache['parent_cache'] == spec['cache_GPU_report'] and cache['protocol'] == spec['N20_locked_protocol']
        and cache['GPU_report'] == cache['gpu_record']['report'] and cache['CPU_Q'] == cache['CPU_record']['Q'],
        'Original qualified cache envelope and GPU138/CPU139 refs remain unchanged')
    cache_gpu, cache_report = n20_completed_cache_record(cache['gpu_record'], 'report', 138, '0', gate, spec)
    cache_cpu, cache_q = n20_completed_cache_record(cache['CPU_record'], 'Q', 139, '', gate, spec)
    require(cache_gpu['N20_locked_protocol'] == cache_cpu['N20_locked_protocol'] == spec['N20_locked_protocol']
        and cache_report['protocol'] == cache_q['protocol'] == spec['N20_locked_protocol']
        and cache_q['capture_GPU_report'] == cache['GPU_report'] and cache_q['qualified'] is False
        and cache_q['ROOT_actual_CPU0_verified'] is False,
        'Do not rewrite old cache pending outputs as current qualified records')
    root = read(closed(closure, spec['N20_root_training_qualified_result']))
    require(_n20_dev_exact(root, spec['expected_N20_root_training_qualified_result'])
        and root['schema'] == 'Root-N20-fullFIT-paired-feature-error-fixed24-saved-artifact-recovery-qualified-v1'
        and root['status'] == 'N20_PAIRED_E8_E24_SAVED_ARTIFACTS_CPU_RECOVERY_QUALIFIED'
        and all(root[key] is True for key in ('qualified', 'original_GPU_exit1_preserved',
            'recovery_CPU_actual0_verified', 'full_saved_endpoint_ROOT_verified'))
        and root['original_GPU_actual0_verified'] is False
        and root['images'] == 741 and root['epochs'] == 24
        and root['sampled_points_per_epoch'] == 3035136 and root['updates_each_arm'] == 17784
        and root['parameters_per_arm'] == 18851 and root['tensors_per_arm'] == 5
        and root['cache_qualified_result'] == spec['N20_root_cache_qualified_result']
        and root['protocol'] == spec['N20_locked_protocol']
        and root['initialization_lock'] == spec['N20_initialization_lock']
        and root['initialization_digest'] == digest
        and root['CPU_Q'] == root['CPU_record']['Q']
        and root['failed_training_record'] == spec['N20_failed_training_record']
        and not any(key in root for key in ('GPU_report', 'gpu_record', 'training_GPU_report')),
        'Mandatory saved-artifact ROOT qualification preserves GPU actual1; no GPU report/completion')
    cpu, q = n20_completed_cache_record(root['CPU_record'], 'Q', 140, '', gate, spec)
    record = root['failed_training_record']
    require(type(record) is dict and set(record) == {
        'RAW_ref', 'phase', 'pin', 'returncode', 'admission', 'phase_binding', 'bootstrap',
        'entry_source', 'spec', 'original_report_absent'}
        and type(record['returncode']) is int and record['returncode'] == 1
        and record['original_report_absent'] is True,
        'Preserved failed training record exact10; never completed GPU140')
    for key in ('RAW_ref', 'admission', 'phase_binding', 'bootstrap', 'entry_source', 'spec'):
        closed(closure, record[key])
    gpu = read(closed(closure, record['spec']))
    require(gpu['schema'] == N20_TRAIN_SCHEMA and cpu['schema'] == N20_TRAIN_CPU_SCHEMA
        and gpu['normal_RAW_count'] == gpu['expected_normal_RAW_count'] == 140
        and gpu['total_failed_RAW_count'] == 23 and gpu['total_terminal_identity_pin_count'] == 163
        and gpu['RAW_inputs'] == spec['RAW_inputs'][:140]
        and gpu['entry_source'] == record['entry_source'] and gpu['phase'] == record['phase']
        and all(closure.get(ref['path']) == ref for ref in gpu['sources'])
        and all(spec['terminal_identity_pins'].get(key) == value
                for key, value in gpu['terminal_identity_pins'].items())
        and Path(gpu['GPU_admission_file']).resolve(strict=True) ==
            Path(record['admission']['path']).resolve(strict=True)
        and cpu['N20_failed_training_record'] == record,
        'Original failed GPU140/23/163 and independent recovery CPU140/24/164 identities stay separate')
    shared_keys = model_keys + ('N20_root_cache_qualified_result', 'expected_N20_root_cache_qualified_result',
        'N20_initialization_lock', 'expected_N20_initialization_digest', 'N20_locked_protocol',
        'expected_N20_locked_protocol', 'locked_protocol', 'cache_GPU_report', 'region_common_source',
        'initialization_seed', 'N20_training_tail_source')
    require(all(_n20_dev_exact(parent[key], spec[key]) for parent in (gpu, cpu) for key in shared_keys),
        'Preserved failed/recovery parents retain identical original cache/model/init/protocol refs')
    require(q['schema'] == 'N20-paired-local-center-feature-error-training-recovery-CPU-qualification-v1'
        and q['status'] == 'N20_PAIRED_FEATURE_ERROR_E8_E24_RECOVERY_CPU_COMPLETE_PENDING_ROOT_ACTUAL0'
        and q['normal_RAW_count'] == 140 and q['total_failed_RAW_count'] == 24
        and q['total_terminal_identity_pin_count'] == 164
        and q['preserved_failed_training_GPU'] == record
        and q['recovered_from_preserved_failed_GPU'] is True
        and q['original_GPU_actual0_verified'] is False and q['original_GPU_REPORT_created'] is False
        and q['images'] == 741 and q['fixed_endpoints'] == [8, 24]
        and q['natural_queries'] == 74100 and q['sampled_points_per_endpoint'] == 3035136
        and q['CPU_packet_pair_reads'] == 1482 and q['total_CPU_endpoint_sampled_points'] == 6070272
        and q['parameters_per_arm'] == 18851 and q['tensors_per_arm'] == 5
        and q['E8_updates_per_arm'] == 5928 and q['E24_updates_per_arm'] == 17784
        and q['arms'] == ['LOCAL_ERROR', 'CENTER_ERROR']
        and q['qualified'] is False and q['ROOT_actual_CPU0_verified'] is False
        and q['original_owned_terminal_self_certified'] is False
        and all(q[key] is False for key in ('GPU_pointwise_logits_qualified', 'optimizer_trajectory_replayed',
            'selected_indices_qualified', 'mask_RLE_qualified', 'AP_evaluated',
            'teacher_or_PR_loaded', 'full_native_L_loaded'))
        and type(q['checks']) is dict and bool(q['checks']) and all(v is True for v in q['checks'].values()),
        'Immutable recovery CPU Q is endpoint-only; preserved GPU actual1 is not upgraded')
    closed(closure, q['details'])
    closed(closure, cpu['qualifier_body_source'])
    require(q['qualifier_body_source'] == cpu['qualifier_body_source'], 'Original CPU numerical body exact')
    for key in model_keys + ('N20_locked_protocol', 'locked_protocol', 'N20_root_cache_qualified_result',
            'N20_initialization_lock', 'initialization_seed'):
        require(q[key] == spec[key], 'Original recovery CPU training binding differs: ' + key)
    require(q['neighbor_report'] == cache['GPU_report']
        and q['parent_cache_report'] == spec['cache_GPU_report']
        and q['N20_cache_CPU_Q'] == cache['CPU_Q']
        and q['initialization_seed'] == 202610150201 and q['order_seed'] == 202610140160001
        and q['expected_N20_initialization_digest'] == digest,
        'Actual training uses the same independently qualified extra-F and original center caches')
    artifacts = root['saved_artifacts']
    require(type(artifacts) is dict and set(artifacts) == {'START', 'checkpoints', 'FIT_endpoints', 'UPDATE_TRACE'}
        and type(artifacts['checkpoints']) is dict and type(artifacts['FIT_endpoints']) is dict
        and set(artifacts['checkpoints']) == set(artifacts['FIT_endpoints']) == {'8', '24'}
        and cpu['N20_saved_training_artifacts'] == q['recovery_artifact_refs'] == artifacts,
        'Exactly original saved E8/E24 references; no DEV-selected replacement')
    output = Path(gpu['output_directory'])
    require(output.is_absolute() and output.is_dir() and not output.is_symlink()
        and not (output / 'REPORT.json').exists() and not (output / 'REPORT.json').is_symlink(),
        'Preserved failed-training output exists while original REPORT remains absent')
    named_refs = [
        (artifacts['START'], 'START.json'), (artifacts['UPDATE_TRACE'], 'UPDATE_TRACE.jsonl'),
        (artifacts['checkpoints']['8'], 'E08_COMPLETE_PAIRED_SNAPSHOT.pth'),
        (artifacts['checkpoints']['24'], 'E24_COMPLETE_PAIRED_SNAPSHOT.pth'),
        (artifacts['FIT_endpoints']['8'], 'E08_FIT_ENDPOINT.json'),
        (artifacts['FIT_endpoints']['24'], 'E24_FIT_ENDPOINT.json')]
    require(len({ref['path'] for ref, _ in named_refs}) == 6, 'Six distinct original saved artifact refs')
    for ref, filename in named_refs:
        require(Path(closed(closure, ref)).resolve(strict=True) == (output / filename).resolve(strict=True),
            'Saved artifact exact original failed-output filename: ' + filename)
    old_report = read(closed(closure, spec['cache_GPU_report']))
    binding = dict(source=record['entry_source'], spec=record['spec'],
        GPU_admission=record['admission'], phase=record['phase'],
        N20_locked_protocol=spec['N20_locked_protocol'], locked_protocol=spec['locked_protocol'],
        parent_cache_report=spec['cache_GPU_report'], neighbor_report=cache['GPU_report'],
        initialization_seed=202610150201, order_seed=202610140160001,
        expected_N20_initialization_digest=digest,
        frozen_teacher_checkpoint=old_report['teacher_checkpoint'],
        frozen_PR_checkpoint=old_report['fixed_PR_checkpoint'],
        N20_root_cache_qualified_result=spec['N20_root_cache_qualified_result'],
        N20_initialization_lock=spec['N20_initialization_lock'], N20_cache_CPU_Q=cache['CPU_Q'])
    binding.update({key: spec[key] for key in model_keys})
    start = read(closed(closure, artifacts['START']))
    require(set(binding) == set(N20_DEV_TRAINING_BINDING_KEYS)
        and q['original_training_binding'] == binding
        and type(start) is dict and all(start[key] == value for key, value in binding.items())
        and start['schema'] == 'N20-paired-feature-error-training-start-v1',
        'Original saved START and recovery Q have exact20 failed-owner/cache/model/init bindings')
    require(callable(globals().get('legacy_gate')), 'ROOT must append the known ordinary N17 legacy metadata gate')
    require(legacy_gate(spec, gate) is gate, 'Preserve original fully qualified DEV shared carrier metadata gate')
    require(len(gate['legacy_rows']) == 187 and len(gate['legacy_reports']) == 4
        and len(gate['legacy_manifests']) == 4 and set(gate['populations']) == {
            'native_official', 'strong_fixed_E24_Omega'}, 'Original full187 native/R and physical carrier ABI')
    common = load(spec['region_common_source'], '_N20_DEV_original_region_library_gate')
    require(callable(getattr(common, 'imported_files', None)), 'Original library gate ABI remains exact')
    gate.update(common=common, N20_protocol=protocol, initialization_metadata=metadata,
        N20_cache_root=cache, N20_training_root=root,
        training=dict(root=root, failed_gpu_spec=gpu, failed_gpu_record=record, cpu=cpu, q=q,
            saved_artifacts=artifacts, start=start, original_binding=binding,
            checkpoints=artifacts['checkpoints'], FIT_endpoints=artifacts['FIT_endpoints']),
        trained_checkpoints=artifacts['checkpoints'], N20_training_CPU_Q_ref=root['CPU_Q'])
    require('parent' in gate and 'lineage' in gate and 'evaluation' in gate,
            'Inherited original endpoint parent/lineage/fixed PR loader must remain available')
    require(callable(globals().get('n20_prior_DEV_carrier_gate')),
        'UNBOUND: ROOT must supply exact metadata-only native/shared/common-PR/decoder reference gate')
    require(n20_prior_DEV_carrier_gate(spec, gate) is gate,
        'ROOT prior DEV carrier hook must preserve this verified gate; no default true or guessed fields')
    return gate

def n20_cpu_policy_names():
    return ('native_official', 'strong_fixed_E24_Omega') + tuple(
        'E%d_%s_%s' % (epoch, arm, mode)
        for epoch in (24, 8) for arm in ('LOCAL_ERROR', 'CENTER_ERROR')
        for mode in ('atmost', 'exact'))


def n20_validate_cpu_header(spec):
    require(type(spec) is dict and spec['schema'] == N20_DEV_CPU_SCHEMA
        and type(spec['cpu_shard_id']) is int and 0 <= spec['cpu_shard_id'] < 8
        and type(spec['CPU_parent_GPU_id']) is int
        and spec['CPU_parent_GPU_id'] == N20_CPU_PARENT_GPU_IDS[spec['cpu_shard_id']]
        and spec['cpu_dataset_ordinal_range'] == list(N20_CPU_RANGES[spec['cpu_shard_id']])
        and all(type(spec[k]) is int for k in ('normal_RAW_count', 'expected_normal_RAW_count',
            'cpu_shard_owner_index', 'total_failed_RAW_count', 'total_terminal_identity_pin_count',
            'initialization_seed', 'deadline_seconds', 'maximum_process_RSS_bytes'))
        and spec['normal_RAW_count'] == spec['expected_normal_RAW_count'] ==
            spec['cpu_shard_owner_index'] == 145 + spec['cpu_shard_id']
        and spec['total_failed_RAW_count'] == 24
        and spec['total_terminal_identity_pin_count'] == spec['normal_RAW_count'] + 24
        and spec['initialization_seed'] == 202610150201 and spec['deadline_seconds'] == 1800
        and spec['automatic_retry'] is False and spec['maximum_process_RSS_bytes'] == 16 * 1024**3
        and spec['allocator_limit_MiB'] == 4096 and spec['whole_GPU_limit_MiB'] == 8192
        and Path(__file__).resolve(strict=True) == Path(spec['entry_source']['path']).resolve(strict=True),
        'Independent CPU145+cid/failed24 own entry/schema/resources and fixed eight whole-image scopes')
    records = spec['completed_GPU_shards']
    require(type(records) is list and len(records) == 4, 'Four original completed GPU descriptors required before any CPU scope')
    for sid, item in enumerate(records):
        require(type(item) is dict and set(item) == N20_COMPLETED_GPU_KEYS
            and type(item['shard_id']) is int and item['shard_id'] == sid
            and item['dataset_ordinal_range'] == list(N20_DEV_RANGES[sid])
            and type(item['shard_owner_index']) is int and item['shard_owner_index'] == 141 + sid
            and item['shard_owner_index'] < spec['normal_RAW_count'],
            'Exact ten-key original GPU141..144 descriptors in global ordinal order')
        for key in ('source', 'spec', 'report', 'RAW', 'admission', 'phase_binding', 'bootstrap'):
            ref_shape(item[key])
    require(spec['GPU_shard'] == records[spec['CPU_parent_GPU_id']]
        and spec['GPU_outer_source'] == spec['GPU_shard']['source'],
        'Mapped complete original GPU descriptor/source; never normal-minus-one ownership guesses')
    return True


def n20_completed_DEV_GPU_record(spec, gate, item):
    c = gate['closure']
    for key in ('source', 'spec', 'report', 'RAW', 'admission', 'phase_binding', 'bootstrap'):
        closed(c, item[key])
    parent = read(closed(c, item['spec']))
    report = read(closed(c, item['report']))
    owner = item['shard_owner_index']; sid = item['shard_id']; row = spec['RAW_inputs'][owner]
    joined = gate['lineage'].owner_ref_join(gate['context'], gate['owners'],
        item['RAW'], item['admission'], item['source'], item['spec'], '0')
    require(type(joined) is int and joined == owner
        and parent['schema'] == N20_DEV_SCHEMA and parent['entry_source'] == item['source']
        and parent['normal_RAW_count'] == parent['expected_normal_RAW_count'] == owner
        and parent['total_failed_RAW_count'] == 24 and parent['total_terminal_identity_pin_count'] == owner + 24
        and parent['RAW_inputs'] == spec['RAW_inputs'][:owner]
        and parent['shard_id'] == sid and parent['dataset_ordinal_range'] == list(N20_DEV_RANGES[sid])
        and all(row[k] == item[v] for k, v in (('receipt', 'RAW'), ('admission', 'admission'),
            ('phase_binding', 'phase_binding'), ('bootstrap', 'bootstrap')))
        and row['phase'] == parent['phase']
        and all(c.get(ref['path']) == ref for ref in parent['sources'])
        and all(spec['terminal_identity_pins'].get(k) == v for k, v in parent['terminal_identity_pins'].items())
        and Path(parent['GPU_admission_file']).resolve(strict=True) == Path(item['admission']['path']).resolve(strict=True)
        and Path(item['report']['path']).resolve(strict=True) == (Path(parent['output_directory']) / 'REPORT.json').resolve(strict=True),
        'Original owned GPU actual0 CUDA0/source/spec/ad/bootstrap/prefix/full Sources/pins/report locator')
    common_keys = ('N20_root_training_qualified_result', 'expected_N20_root_training_qualified_result',
        'N20_root_cache_qualified_result', 'expected_N20_root_cache_qualified_result',
        'N20_locked_protocol', 'expected_N20_locked_protocol', 'N20_initialization_lock',
        'expected_N20_initialization_digest', 'N20_failed_training_record', 'N20_failed_training_stderr',
        'N20_training_tail_source', 'locked_protocol', 'cache_GPU_report', 'region_common_source',
        'scout_module_source', 'base_scout_source', 'paired_training_body_source',
        'original_paired_training_body_source', 'initialization_seed', 'legacy_qualified_result',
        'legacy_shards', 'n17_root_qualified_result_ref', 'expected_n17_root_qualified_result',
        'new_eval_body_source', 'legacy_eval_body_source', 'capture_body_source', 'coordinates_source', 'labels_source')
    require(all(_n20_dev_exact(parent[k], spec[k]) for k in common_keys)
        and all(_n20_dev_exact(v, spec[k]) for k, v in parent.items() if k.startswith('additional_preserved_')),
        'CPU retains all original frozen model/cache/lock/native/prior and failed-record semantics')
    binding = dict(source=item['source'], spec=item['spec'], GPU_admission=item['admission'], phase=parent['phase'],
        N20_root_training_qualified_result=parent['N20_root_training_qualified_result'], N20_locked_protocol=parent['N20_locked_protocol'],
        legacy_qualified_result=parent['legacy_qualified_result'], new_eval_body_source=parent['new_eval_body_source'],
        capture_body_source=parent['capture_body_source'], coordinates_source=parent['coordinates_source'],
        checkpoints=gate['trained_checkpoints'], shard_id=sid, dataset_ordinal_range=list(N20_DEV_RANGES[sid]))
    names = n20_cpu_policy_names(); first, last = N20_DEV_RANGES[sid]
    require(set(binding) == set(N20_GPU_BINDING_KEYS)
        and report['schema'] == 'N20-fixedDEV-feature-error-GPU-shard-report-v1'
        and report['status'] == 'N20_FIXEDDEV_FEATURE_ERROR_SHARD_COMPLETE_PENDING_ROOT_ACTUAL0_AND_CPU'
        and all(report[k] == v for k, v in binding.items())
        and report['normal_RAW_count'] == owner and report['total_failed_RAW_count'] == 24
        and report['total_terminal_identity_pin_count'] == owner + 24
        and report['qualified'] is False and report['ROOT_actual_GPU0_verified'] is False
        and report['original_training_GPU_actual0_verified'] is False
        and report['training_artifacts_recovery_qualified'] is True
        and report['CPU_numerical_qualification_pending'] is True and report['models_frozen'] is True
        and report['teacher_forward_calls'] == report['new_fullC_PR_forward_calls'] == 0
        and report['fresh_selected_PR'] is True and report['reused_common_C_reference'] is True
        and report['AP_evaluated'] is False and report['local_AP_computed'] is False
        and report['full187_qualification_claimed'] is False and report['standalone_FPS_measured'] is False
        and report['cleanup'] == {'allocated_after': 0, 'reserved_after': 0}
        and report['controls'] == list(names) and set(report['control_results']) == set(names),
        'Immutable pending GPU report; terminal0 comes from original owner, never output self-certification')
    manifest = read(closed(c, report['capture_manifest'])); start = read(closed(c, report['START']))
    output = Path(parent['output_directory'])
    require(Path(report['capture_manifest']['path']).resolve(strict=True) == (output / 'CAPTURE_MANIFEST.json').resolve(strict=True)
        and Path(report['START']['path']).resolve(strict=True) == (output / 'START.json').resolve(strict=True)
        and manifest['schema'] == 'N20-fixedDEV-feature-neighbors-GPU-shard-capture-manifest-v1'
        and start['schema'] == 'N20-fixedDEV-feature-GPU-shard-start-v1'
        and all(manifest[k] == start[k] == v for k, v in binding.items())
        and start['original_GPU_training_actual0'] is False and start['recovery_CPU_qualified'] is True
        and start['policies'] == list(names)
        and manifest['images'] == report['images'] == last - first
        and manifest['queries'] == report['queries'] == (last - first) * 100
        and manifest['global_images'] == 187 and manifest['global_queries'] == 18700
        and manifest['internal_member_locators_are_strings'] is True
        and report['scope_dataset_indices'] == list(range(first, last))
        and len(manifest['rows']) == last - first
        and manifest['physical_capture_refs'] == report['capture_refs'] == [r['bundle'] for r in manifest['rows']],
        'Complete original GPU ordinal scope, START, manifest and actual physical captures')
    for i, captured in enumerate(manifest['rows'], start=first):
        origin = gate['legacy_rows'][i]; old = origin['row']; common = gate['prior_common_rows'][i]
        carrier = dict(bundle=old['bundle'], shared_member=old['shared_actual_image_inputs'], original_GPU_binding=origin['binding'],
            common_PR_bundle=common['row']['bundle'], common_PR_original_GPU_binding=common['binding'])
        require(captured['dataset_index'] == i and captured['image_id'] == old['image_id']
            and captured['identity'] == old['identity'] and captured['qids'] == old['qids']
            and captured['input_carrier'] == carrier and captured['natural_queries'] == 100
            and captured['neighbor_input_member'] == 'i%03d/FEATURE_NEIGHBORS_INPUTS.pth' % i
            and type(captured['member_locators']) is list
            and all(type(v) is str for v in captured['member_locators'])
            and len(captured['member_locators']) == len(set(captured['member_locators']))
            and type(captured['query_profiles']) is list and len(captured['query_profiles']) == 100
            and [v['query_id'] for v in captured['query_profiles']] == list(range(100)),
            'Full natural100 new-F9 rows with original input/native/prior-common provenance')
        closed(c, captured['bundle']); closed(c, carrier['bundle']); closed(c, carrier['common_PR_bundle'])
    image_ids = [r['image_id'] for r in manifest['rows']]
    require(report['scope_image_ids'] == image_ids and len(set(image_ids)) == last - first, 'Original canonical image namespace/order')
    expected = {}; grouped = {}
    for name in names:
        value = report['control_results'][name]
        require(type(value) is dict and set(value) == {'predictions'}, 'Actual predictions are strict physical refs, not inline COCO')
        values = read(closed(c, value['predictions']))
        require(type(values) is list and len(values) == (last - first) * 100, 'Complete original GPU local prediction list')
        groups = {}
        for prediction in values:
            groups.setdefault(prediction['image_id'], []).append(prediction)
        require(list(groups) == image_ids and all(
            [p['query_id'] for p in groups[r['image_id']]] == r['qids'] for r in manifest['rows']),
            'All100 query export order; never force natural order onto exported predictions')
        for captured in manifest['rows']:
            baseline = gate['populations']['native_official'][captured['image_id']]
            observed = groups[captured['image_id']]
            require(all({k: v for k, v in p.items() if k != 'segmentation'} ==
                        {k: v for k, v in q.items() if k != 'segmentation'} for p, q in zip(observed, baseline)),
                'Original class/S0/query identity and all nonsegmentation fields stay unchanged')
            if name in ('native_official', 'strong_fixed_E24_Omega'):
                require(observed == gate['populations'][name][captured['image_id']], 'Original native/strong R predictions exact')
        expected[name] = values; grouped[name] = groups
    return dict(descriptor=item, spec=parent, report=report, manifest=manifest, START=start,
        binding=binding, expected_predictions=expected, expected_prediction_groups=grouped)


def n20_scope_parent_GPU(spec, gate):
    cid = spec['cpu_shard_id']; pid = spec['CPU_parent_GPU_id']
    first, last = N20_CPU_RANGES[cid]; pa, pz = N20_DEV_RANGES[pid]
    rows = gate['parent_GPU_manifest']['rows']
    require(pa <= first < last <= pz and spec['cpu_dataset_ordinal_range'] == [first, last]
        and [r['dataset_index'] for r in rows] == list(range(pa, pz)), 'Immutable whole parent before CPU child slicing')
    selected = rows[first-pa:last-pa]; ids = [r['image_id'] for r in selected]
    groups = {name: {image_id: parent[image_id] for image_id in ids}
              for name, parent in gate['expected_prediction_groups'].items()}
    expected = {name: [p for r in selected for p in parent[r['image_id']]] for name, parent in groups.items()}
    require(len(selected) == last-first and len(set(ids)) == last-first
        and all(len(v) == (last-first)*100 for v in expected.values()), 'Complete child images/all100; no failed partials')
    gate.update(scope_rows=selected, scope_capture_refs=[r['bundle'] for r in selected], scope_image_ids=ids,
        scope_expected_predictions=expected, scope_expected_prediction_groups=groups)
    return gate


def source_gate(path, spec):
    n20_validate_cpu_header(spec)
    failed22 = n20_failed22_raw_only(spec)
    failed23 = n20_failed23_raw_only(spec)
    failed24 = n20_failed24_raw_only(spec)
    gate = inherited_source_gate(path, spec, cpu=True)
    n20_failed22_closed(spec, gate, failed22)
    n20_failed23_closed(spec, gate, failed23)
    n20_failed24_closed(spec, gate, failed24)
    c = gate['closure']
    for key in N20_CPU_SOURCE_KEYS:
        closed(c, spec[key])
    partition = read(closed(c, spec['CPU_execution_partition_supplement']))
    require(type(partition) is dict and type(spec['expected_CPU_execution_partition_supplement']) is dict
        and _n20_dev_exact(partition, spec['expected_CPU_execution_partition_supplement']),
        'Exact complete ROOT-prelocked eight-scope CPU execution supplement, not a derived scientific spec')
    gate['CPU_execution_partition_supplement'] = partition
    for item in spec['completed_GPU_shards']:
        for key in ('source', 'spec', 'report', 'RAW', 'admission', 'phase_binding', 'bootstrap'):
            closed(c, item[key])
    original_parent_spec = read(closed(c, spec['GPU_shard']['spec']))
    # This pure parent function never calls original GPU source_gate/own_gate.
    n20_recovered_training_metadata_gate(original_parent_spec, gate)
    parents = [n20_completed_DEV_GPU_record(spec, gate, item) for item in spec['completed_GPU_shards']]
    parent = parents[spec['CPU_parent_GPU_id']]
    gpu = load(spec['GPU_outer_source'], '_N20_completed_DEV_GPU_metadata_helpers')
    require(Path(gpu.__file__).resolve(strict=True) == Path(spec['GPU_outer_source']['path']).resolve(strict=True)
        and callable(getattr(gpu, 'n20_model_load_cpu', None))
        and callable(getattr(gpu, 'n20_policy_names', None))
        and tuple(gpu.n20_policy_names()) == n20_cpu_policy_names(), 'Closed original GPU pure CPU model loader/policy ABI')
    gate.update(gpu_helpers=gpu, completed_GPU_shards=spec['completed_GPU_shards'],
        parent_GPU=parent['spec'], parent_GPU_report=parent['report'], parent_GPU_manifest=parent['manifest'],
        parent_GPU_START=parent['START'], parent_GPU_binding=parent['binding'],
        expected_predictions=parent['expected_predictions'], expected_prediction_groups=parent['expected_prediction_groups'])
    n20_scope_parent_GPU(spec, gate)
    return gate
