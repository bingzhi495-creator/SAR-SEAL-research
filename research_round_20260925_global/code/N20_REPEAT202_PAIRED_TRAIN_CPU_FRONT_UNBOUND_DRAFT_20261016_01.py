"""UNBOUND_NOT_EXECUTED: independent repeat202 endpoint CPU metadata front.

Assembly: completed F typed prefix L1..1190 (historical schema/counts unchanged),
then UNBOUND repeat202 GPU front0886 pure metadata helpers, then THIS front,
then repeat202 CPU run-tail03 and one --spec/--owner-execute CLI. The GPU front's
source_gate definition is overwritten below; it MUST NEVER be called for CPU.
N20_FINAL_CPU_SCHEMA below is ONLY the inherited current cpu=True dispatch.

Requires original prefix helpers and n20_repeat202_exact/completed/first_permission/
history_view/cache/initialization/cleanup_probe from the GPU front. No scientific
imports, reads, model loading or execution occur on import. Root binds/freeze/runs
only AFTER original GPU181 actual0 and a new original pure-prefix actual0.

New spec: qualification_owner_index (explicit int>=182),
N20_repeat202_completed_training_GPU_record exact8 with report/owner_index181.
Same first201/init179/probe180 Root refs, expected permission, four model refs,
new202 initialization lock/digest and frozen GPU N20_training_tail_source remain
unchanged. qualifier_body_source refers to the new independent seed202 body01.

Tail ABI retains current CPU closure/source/spec_ref/admission/admission_ref/
context/owners/parent/lineage/common; both qualified cache reports/_ref;
training_report/_ref, training_GPU_record, training_start, training_endpoints
(integer8/24, decoded inner endpoint), training_binding_refs. Extra metadata:
training_binding (exact23), training_GPU_spec, training_endpoint_refs.
All current qualified/terminal claims remain pending until separate Root actual0.
"""

from pathlib import Path

N20_REPEAT202_TRAIN_CPU_SCHEMA = 'ROOT-N20-repeat202-paired-local-center-feature-error-training-endpoint-CPU-spec-v1'
N20_FINAL_CPU_SCHEMA = N20_REPEAT202_TRAIN_CPU_SCHEMA
N20_REPEAT202_COMPLETED_GPU_SOURCE_BYTES = 137497
N20_REPEAT202_COMPLETED_GPU_SOURCE_SHA = '24e1a55acf4cc6a85d29e767329a53857aed1b0f9ec8ab1e07694d65a9b5aafb'
N20_REPEAT202_TRAINING_REPORT_SCHEMA = 'N20-paired-local-center-feature-error-training-GPU-report-v1'
N20_REPEAT202_TRAINING_REPORT_STATUS = 'N20_FULLFIT_PAIRED_FEATURE_ERROR_E8_E24_COMPLETE_PENDING_INDEPENDENT_CPU'
N20_REPEAT202_CPU_SHARED_KEYS = N20_REPEAT202_MODEL_KEYS + (
    'N20_training_tail_source','N20_root_cache_qualified_result','expected_N20_root_cache_qualified_result',
    'N20_locked_protocol','expected_N20_locked_protocol','locked_protocol','cache_GPU_report','region_common_source',
    'N20_initialization_lock','expected_N20_initialization_digest','initialization_seed',
    'N20_first201_ROOT_qualified_result','expected_N20_first201_ROOT_qualified_result',
    'N20_repeat202_initialization_qualified_result','N20_repeat202_initialization_record',
    'N20_repeat202_cleanup_probe_qualified_result','N20_failed_training_record','N20_failed_training_stderr',
    'N20_failed_DEV_capture_record','N20_failed_DEV_capture_stderr','N20_failed_CPU_numeric_record',
    'N20_failed26_helper_source','failed26_partials_reused')


def n20_repeat202_CPU_header(spec):
    owner = spec['qualification_owner_index']
    require(type(owner) is int and owner >= 182 and spec['schema'] == N20_REPEAT202_TRAIN_CPU_SCHEMA
        and spec['normal_RAW_count'] == spec['expected_normal_RAW_count'] == owner
        and spec['total_failed_RAW_count'] == 26 and spec['total_terminal_identity_pin_count'] == owner + 26
        and spec['initialization_seed'] == 202610150202 and spec['deadline_seconds'] == 1800
        and spec['maximum_process_RSS_bytes'] == 16*1024**3 and spec['automatic_retry'] is False
        and spec['allocator_limit_MiB'] == 4096 and spec['whole_GPU_limit_MiB'] == 8192
        and spec['failed26_partials_reused'] is False
        and Path(spec['CPU_admission_file']).is_absolute()
        and Path(__file__).resolve(strict=True) == Path(spec['entry_source']['path']).resolve(strict=True),
        'Independent current CPU owner/source/schema/resources, never inherited GPU-own identity')
    record = spec['N20_repeat202_completed_training_GPU_record']
    require(type(record) is dict and set(record) == {'source','spec','report','RAW','admission',
        'phase_binding','bootstrap','owner_index'} and type(record['owner_index']) is int
        and record['owner_index'] == 181 and record['source']['bytes'] == N20_REPEAT202_COMPLETED_GPU_SOURCE_BYTES
        and record['source']['sha256'] == N20_REPEAT202_COMPLETED_GPU_SOURCE_SHA,
        'Only original completed GPU181 exact8 record and Root-pinned assembled source; future completion not assumed')
    for key in set(record) - {'owner_index'}:
        ref_shape(record[key])
    require(record['source'] != spec['entry_source'] and record['RAW'] == spec['RAW_inputs'][181]['receipt'],
        'GPU parent is an earlier actual original owner, separate from current CPU')
    return owner, record


def n20_repeat202_CPU_training_binding(spec, gate, record, parent):
    binding = dict(source=record['source'], spec=record['spec'], GPU_admission=record['admission'],
        phase=parent['phase'], N20_locked_protocol=spec['N20_locked_protocol'], locked_protocol=spec['locked_protocol'],
        parent_cache_report=gate['parent_cache_report_ref'], neighbor_report=gate['neighbor_report_ref'],
        scout_module_source=spec['scout_module_source'], base_scout_source=spec['base_scout_source'],
        paired_training_body_source=spec['paired_training_body_source'],
        original_paired_training_body_source=spec['original_paired_training_body_source'],
        initialization_seed=202610150202, order_seed=202610140160001,
        expected_N20_initialization_digest=spec['expected_N20_initialization_digest'],
        frozen_teacher_checkpoint=gate['parent_cache_report']['teacher_checkpoint'],
        frozen_PR_checkpoint=gate['parent_cache_report']['fixed_PR_checkpoint'])
    refs = gate['training_binding_refs']
    require(set(refs) == {'N20_root_cache_qualified_result','N20_initialization_lock','N20_cache_CPU_Q',
        'N20_first201_ROOT_qualified_result','N20_repeat202_initialization_qualified_result',
        'N20_repeat202_cleanup_probe_qualified_result'} and not set(binding).intersection(refs),
        'Original six cache/permission/init/probe strictrefs cannot replace actual GPU own binding')
    for ref in refs.values():
        closed(gate['closure'], ref)
    binding.update(refs)
    require(len(binding) == 23, 'Exact17 original own/scientific plus six completed qualification refs')
    return binding


def n20_repeat202_CPU_completed_training(spec, gate, record):
    c = gate['closure']
    parent, report = n20_repeat202_completed(record, 'report', 181, '0', gate, spec)
    require(parent['schema'] == N20_REPEAT202_TRAIN_SCHEMA and parent['training_owner_index'] == 181
        and parent['initialization_seed'] == 202610150202
        and parent['total_failed_RAW_count'] == 26 and parent['total_terminal_identity_pin_count'] == 207
        and parent['deadline_seconds'] == 1800 and parent['automatic_retry'] is False
        and parent['maximum_process_RSS_bytes'] == 16*1024**3
        and parent['allocator_limit_MiB'] == 4096 and parent['whole_GPU_limit_MiB'] == 8192
        and all(n20_repeat202_exact(parent[k], spec[k]) for k in N20_REPEAT202_CPU_SHARED_KEYS)
        and all(n20_repeat202_exact(v, spec[k]) for k, v in parent.items()
                if k.startswith('additional_preserved_'))
        and Path(record['report']['path']).resolve(strict=True) ==
            (Path(parent['output_directory'])/'REPORT.json').resolve(strict=True),
        'Original GPU181 normal actual0/complete prefix/Source and same202 cache/model/init/protocol, no own gate rerun')
    binding = n20_repeat202_CPU_training_binding(spec, gate, record, parent)
    require(report['schema'] == N20_REPEAT202_TRAINING_REPORT_SCHEMA
        and report['status'] == N20_REPEAT202_TRAINING_REPORT_STATUS
        and all(report[k] == v for k, v in binding.items())
        and report['normal_RAW_count'] == 181 and report['total_failed_RAW_count'] == 26
        and report['total_terminal_identity_pin_count'] == 207
        and report['images'] == 741 and report['natural_queries'] == 74100
        and report['sampled_points_per_epoch'] == 3035136 and report['epochs'] == 24
        and report['paired_updates_per_arm'] == 17784 and report['point_exposure_per_arm'] == 72843264
        and report['parameters_per_arm'] == 18851 and report['tensors_per_arm'] == 5
        and report['arms'] == ['LOCAL_ERROR','CENTER_ERROR']
        and len(report['history']) == len(report['epoch_orders']) == 24
        and report['CPU_endpoint_qualification_pending'] is True
        and report['old_point_query_label_weight_order_preserved'] is True
        and all(report[k] is False for k in ('qualified','ROOT_actual_GPU0_verified',
            'current_owned_terminal_self_certified','teacher_or_PR_loaded','full_native_L_loaded',
            'DEV_AP_evaluated','best_endpoint_selected','old_cache_Source_or_protocol_modified',
            'additional_FIT_annotations','new_primitive_claimed')),
        'Actual completed same-budget GPU report stays pending for independent CPU qualification; no AP/rank claim')
    cleanup = report['cleanup']
    require(cleanup['allocated_after'] == cleanup['reserved_after'] == 0
        and cleanup['preserved_failure24_unique_root_cause_established'] is False,
        'Original GPU final cleanup zero; preserved failure24 is not reclassified as successful')
    start_ref = report['START']; start = read(closed(c, start_ref))
    require(Path(start_ref['path']).resolve(strict=True) ==
            (Path(parent['output_directory'])/'START.json').resolve(strict=True)
        and start['schema'] == 'N20-paired-feature-error-training-start-v1'
        and all(start[k] == v for k, v in binding.items())
        and start['initial_states'] == report['initial_states']
        and start['parameters_per_arm'] == 18851 and start['tensors_per_arm'] == 5
        and start['arms'] == ['LOCAL_ERROR','CENTER_ERROR'] and start['fixed_epochs'] == 24
        and start['updates_each_arm'] == 17784 and start['point_exposure_each_arm'] == 72843264
        and all(start[k] is False for k in ('teacher_or_PR_loaded','full_native_L_loaded','AMP','DEV_AP_evaluated','qualified')),
        'Actual original GPU START/initial evidence/23 bindings, not a newly synthesized training report')
    require(type(report['checkpoints']) is dict and set(report['checkpoints']) == {'8','24'}
        and type(report['FIT_endpoints']) is dict and set(report['FIT_endpoints']) == {'8','24'},
        'Only complete prelocked E8/E24 checkpoints and FIT endpoints')
    endpoints = {}
    for epoch in (8,24):
        snapshot_ref = report['checkpoints'][str(epoch)]; closed(c, snapshot_ref)
        endpoint_ref = report['FIT_endpoints'][str(epoch)]; wrapper = read(closed(c, endpoint_ref))
        require(Path(snapshot_ref['path']).resolve(strict=True) ==
                (Path(parent['output_directory'])/('E%02d_COMPLETE_PAIRED_SNAPSHOT.pth' % epoch)).resolve(strict=True)
            and Path(endpoint_ref['path']).resolve(strict=True) ==
                (Path(parent['output_directory'])/('E%02d_FIT_ENDPOINT.json' % epoch)).resolve(strict=True)
            and type(wrapper) is dict and set(wrapper) == set(binding) | {'endpoint'}
            and all(wrapper[k] == v for k, v in binding.items()) and type(wrapper['endpoint']) is dict,
            'Original complete snapshot ref and actual FIT wrapper with exact23 metadata; no tensor load in front')
        endpoints[epoch] = wrapper['endpoint']
    closed(c, report['UPDATE_TRACE'])
    require(Path(report['UPDATE_TRACE']['path']).resolve(strict=True) ==
        (Path(parent['output_directory'])/'UPDATE_TRACE.jsonl').resolve(strict=True),
        'Original saved update substream reference; numerical trace proof belongs to independent CPU body')
    return parent, report, start, endpoints, binding


def source_gate(path, spec):
    owner, record = n20_repeat202_CPU_header(spec)
    # Mandatory raw checks precede inherited first SHA; no parent GPU source_gate call.
    raw22 = n20_failed22_raw_only(spec); raw23 = n20_failed23_raw_only(spec)
    raw24 = n20_failed24_raw_only(spec); raw25 = n20_failed25_raw_only(spec)
    raw26 = n20_failed26_raw_only(spec)
    gate = inherited_source_gate(path, spec, cpu=True)
    require(gate['source'] == spec['entry_source'] and gate['context']['normal_RAW_count'] == owner
        and gate['context']['RAW_inputs'] == spec['RAW_inputs'] and 'parent' in gate,
        'Independent current CPU own context, original endpoint parent retained')
    root, history_spec = n20_repeat202_first_permission(spec, gate)
    historical = n20_repeat202_history_view(spec, gate, history_spec)
    n20_failed22_closed(historical, gate, raw22); n20_failed23_closed(historical, gate, raw23)
    n20_failed24_closed(historical, gate, raw24); n20_failed25_closed(historical, gate, raw25)
    n20_failed26_closed(historical, gate, raw26)
    for key in N20_REPEAT202_MODEL_KEYS + ('N20_training_tail_source','N20_initialization_lock',
                                         'region_common_source','qualifier_body_source'):
        closed(gate['closure'], spec[key])
    protocol, cache_root, old_report, neighbor_report = n20_repeat202_cache(spec, gate)
    init_root, metadata = n20_repeat202_initialization(spec, gate, history_spec)
    probe_root = n20_repeat202_cleanup_probe(spec, gate)
    common = load(spec['region_common_source'], '_N20_repeat202_CPU_original_library_gate')
    require(callable(getattr(common, 'imported_files', None)), 'Original imported_files independent CPU library ABI')
    gate.update(common=common, parent_cache_report=old_report, parent_cache_report_ref=spec['cache_GPU_report'],
        neighbor_report=neighbor_report, neighbor_report_ref=cache_root['GPU_report'], protocol=protocol,
        first201_root_permission=root, initialization_metadata=metadata,
        repeat202_initialization_qualification=init_root, repeat202_cleanup_probe_qualification=probe_root,
        training_binding_refs=dict(N20_root_cache_qualified_result=spec['N20_root_cache_qualified_result'],
            N20_initialization_lock=spec['N20_initialization_lock'], N20_cache_CPU_Q=cache_root['CPU_Q'],
            N20_first201_ROOT_qualified_result=spec['N20_first201_ROOT_qualified_result'],
            N20_repeat202_initialization_qualified_result=spec['N20_repeat202_initialization_qualified_result'],
            N20_repeat202_cleanup_probe_qualified_result=spec['N20_repeat202_cleanup_probe_qualified_result']))
    parent, report, start, endpoints, binding = n20_repeat202_CPU_completed_training(spec, gate, record)
    gate.update(training_GPU_record=record, training_GPU_spec=parent,
        training_report=report, training_report_ref=record['report'], training_start=start,
        training_endpoints=endpoints, training_endpoint_refs=report['FIT_endpoints'], training_binding=binding)
    return gate
