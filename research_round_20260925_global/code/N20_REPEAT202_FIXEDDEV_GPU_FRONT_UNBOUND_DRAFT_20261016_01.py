"""UNBOUND_NOT_EXECUTED: repeat202 fixed-DEV GPU metadata front, no run/CLI.

Assemble unchanged F typed prefix L1..1190 + original legacy_gate/COCO helpers,
repeat202 GPUfront0886 pure metadata + repeat202 CPUfrontf2eb pure metadata,
the original metadata-only n20_prior_DEV_carrier_gate, THIS front, DEV tail.
This source_gate overwrites all parent own entry definitions; none are called.
N20_DEV_SCHEMA controls only inherited CURRENT GPU dispatch, never historical
SCHEMA/CPU_SCHEMA/N20_TRAIN_SCHEMA or first201 failures/saved checkpoints.

New spec: DEV_owner_index>=183 and N20_repeat202_ROOT_training_qualification
+ expected_N20_repeat202_ROOT_training_qualification (complete future RootJSON).
Four logical ordinal scopes stay0:47/47:94/94:141/141:187. Root supplies actual
owner counts; an absent/future qualification never authorizes execution.

Mandatory original legacy_gate retains N16 FOUR logical shards. Mandatory prior
carrier hook retains original N17 FIVE physical shards/native packet/e/class/
native query order/common PR. Do not replace spec.legacy_shards by spec.shards,
call an old training/own gate, invent unknown native/shared fields, or import an
old entry and monkeypatch it. This draft does not read any payload on import.
"""

from pathlib import Path

N20_REPEAT202_DEV_SCHEMA = 'ROOT-N20-repeat202-fixedDEV-local-center-feature-error-GPU-shard-spec-v1'
N20_DEV_SCHEMA = N20_REPEAT202_DEV_SCHEMA
N20_REPEAT202_DEV_RANGES = ((0,47),(47,94),(94,141),(141,187))
N20_REPEAT202_ENDPOINT_ROOT_SCHEMA = 'ROOT-N20-repeat202-paired-training-endpoints-qualified-result-v1'
N20_REPEAT202_CPU_SOURCE_BYTES = 147873
N20_REPEAT202_CPU_SOURCE_SHA = '3f4bb9df63bcff7dcc95b3f69a9c5e20cfe65da890c5debf7a8705564f35b7ae'
N20_REPEAT202_CPU_CHECKS = (
    'initialization_digest_and_all_five_tensors_exact',
    'both_E8_E24_full_snapshot_and_report_bindings_exact',
    'all_five_AdamW_states_moments_and_exact_steps_5928_17784',
    'private_orders_rng_history_and_saved_paired_update_substream_exact',
    'complete_same_input_FIT741_CPU_two_formulas_absolute1e4',
    'recorded_GPU_endpoint_mean_weighted_CE_absolute1e4',
    'original_source_spec_admission_imports_rng_and_cpu_resources_unchanged')


def n20_repeat202_DEV_header(spec):
    owner = spec['DEV_owner_index']; sid = spec['shard_id']
    require(spec['schema'] == N20_REPEAT202_DEV_SCHEMA and type(owner) is int and owner >= 183
        and type(sid) is int and 0 <= sid < 4
        and spec['dataset_ordinal_range'] == list(N20_REPEAT202_DEV_RANGES[sid])
        and spec['normal_RAW_count'] == spec['expected_normal_RAW_count'] == owner
        and spec['total_failed_RAW_count'] == 26 and spec['total_terminal_identity_pin_count'] == owner + 26
        and spec['initialization_seed'] == 202610150202 and spec['deadline_seconds'] == 1800
        and spec['maximum_process_RSS_bytes'] == 16*1024**3 and spec['automatic_retry'] is False
        and spec['allocator_limit_MiB'] == 4096 and spec['whole_GPU_limit_MiB'] == 8192
        and spec['failed26_partials_reused'] is False
        and Path(__file__).resolve(strict=True) == Path(spec['entry_source']['path']).resolve(strict=True),
        'New202 DEV GPU own source/explicit owner/four fixed ranges/resources; no prior CPU/GPU own identity')
    return owner


def n20_repeat202_DEV_training_metadata_gate(spec, gate):
    """Completed GPU181/CPU182 endpoint ancestry only, not a current own gate."""
    c = gate['closure']; ref = spec['N20_repeat202_ROOT_training_qualification']
    root = read(closed(c, ref))
    require(n20_repeat202_exact(root, spec['expected_N20_repeat202_ROOT_training_qualification'])
        and root['schema'] == N20_REPEAT202_ENDPOINT_ROOT_SCHEMA and root['qualified'] is True
        and root['normal_RAW_count'] == 183 and root['total_failed_RAW_count'] == 26
        and root['total_terminal_identity_pin_count'] == 209
        and root['RAW_inputs'] == spec['RAW_inputs'][:183]
        and len(root['terminal_identity_pins']) == 191
        and all(spec['terminal_identity_pins'].get(k) == v for k,v in root['terminal_identity_pins'].items())
        and root['initialization_seed'] == 202610150202
        and root['initialization_digest'] == spec['expected_N20_initialization_digest'],
        'Complete future Root183/26/209 endpoint qualification required; never a pending/fabricated success')
    GPU = root['GPU_record']; CPU = root['CPU_record']
    require(GPU == spec['N20_repeat202_completed_training_GPU_record']
        and root['training_report'] == GPU['report'] and root['CPU_Q'] == CPU['Q']
        and CPU['source']['bytes'] == N20_REPEAT202_CPU_SOURCE_BYTES
        and CPU['source']['sha256'] == N20_REPEAT202_CPU_SOURCE_SHA,
        'Actual normal GPU181/CPU182 original completed record/source identities')
    gpu_spec, report, start, endpoints, binding = n20_repeat202_CPU_completed_training(spec, gate, GPU)
    cpu_spec, q = n20_repeat202_completed(CPU, 'Q', 182, '', gate, spec)
    require(cpu_spec['schema'] == N20_REPEAT202_TRAIN_CPU_SCHEMA
        and cpu_spec['qualification_owner_index'] == 182
        and cpu_spec['total_failed_RAW_count'] == 26 and cpu_spec['total_terminal_identity_pin_count'] == 208
        and cpu_spec['N20_repeat202_completed_training_GPU_record'] == GPU
        and all(n20_repeat202_exact(cpu_spec[k], spec[k]) for k in N20_REPEAT202_CPU_SHARED_KEYS)
        and all(n20_repeat202_exact(v, spec[k]) for k,v in cpu_spec.items() if k.startswith('additional_preserved_')),
        'Independent completed CPU182 original Source/cache/model/protocol/init and full raw ancestry')
    cpu_binding = dict(source=CPU['source'], spec=CPU['spec'], CPU_admission=CPU['admission'],
        phase=cpu_spec['phase'], training_GPU_record=GPU, training_GPU_report=GPU['report'],
        parent_cache_report=gate['parent_cache_report_ref'], neighbor_report=gate['neighbor_report_ref'],
        N20_locked_protocol=spec['N20_locked_protocol'], locked_protocol=spec['locked_protocol'],
        qualifier_body_source=cpu_spec['qualifier_body_source'],
        scout_module_source=spec['scout_module_source'], base_scout_source=spec['base_scout_source'],
        paired_training_body_source=spec['paired_training_body_source'],
        original_paired_training_body_source=spec['original_paired_training_body_source'],
        initialization_seed=202610150202, order_seed=202610140160001,
        expected_N20_initialization_digest=spec['expected_N20_initialization_digest'])
    cpu_binding.update(gate['training_binding_refs'])
    require(q['schema'] == 'N20-paired-local-center-feature-error-training-endpoint-CPU-qualification-v1'
        and q['status'] == 'N20_PAIRED_FEATURE_ERROR_E8_E24_ENDPOINT_CPU_COMPLETE_PENDING_ROOT_ACTUAL0'
        and all(q[k] == v for k,v in cpu_binding.items())
        and q['normal_RAW_count'] == 182 and q['total_failed_RAW_count'] == 26
        and q['total_terminal_identity_pin_count'] == 208
        and q['images'] == 741 and q['natural_queries'] == 74100
        and q['sampled_points_per_endpoint'] == 3035136 and q['CPU_packet_pair_reads'] == 1482
        and q['total_CPU_endpoint_sampled_points'] == 6070272 and q['fixed_endpoints'] == [8,24]
        and q['parameters_per_arm'] == 18851 and q['tensors_per_arm'] == 5
        and q['E8_updates_per_arm'] == 5928 and q['E24_updates_per_arm'] == 17784
        and q['arms'] == ['LOCAL_ERROR','CENTER_ERROR']
        and set(q['checks']) == set(N20_REPEAT202_CPU_CHECKS)
        and all(q['checks'][k] is True for k in N20_REPEAT202_CPU_CHECKS)
        and all(q[k] is False for k in ('qualified','ROOT_actual_CPU0_verified',
            'original_owned_terminal_self_certified','GPU_pointwise_logits_qualified',
            'optimizer_trajectory_replayed','selected_indices_qualified','mask_RLE_qualified',
            'AP_evaluated','teacher_or_PR_loaded','full_native_L_loaded')),
        'Original CPU Q proves fixed741 endpoints/two CPU formulas/aggregate CE only and remains pending')
    require(root['CPU_details'] == q['details'], 'Root points to the original complete endpoint CPU DETAILS')
    details = read(closed(c, q['details']))
    require(Path(q['details']['path']).resolve(strict=True) ==
            (Path(cpu_spec['output_directory'])/'DETAILS.json').resolve(strict=True)
        and details['schema'] == 'N20-paired-feature-error-endpoint-CPU-details-v1'
        and all(details[k] == v for k,v in cpu_binding.items())
        and details['checks'] == q['checks'] and details['endpoints'] == q['endpoints']
        and details['checkpoints'] == root['checkpoints'] == report['checkpoints']
        and details['FIT_endpoints'] == root['FIT_endpoints'] == report['FIT_endpoints']
        and details['UPDATE_TRACE'] == report['UPDATE_TRACE']
        and details['saved_paired_updates'] == 17784 and details['E8_saved_trace_prefix_updates'] == 5928
        and details['same_actual_GPU_input_F_reused'] is True
        and all(details[k] is False for k in ('qualified','ROOT_actual_CPU0_verified',
            'GPU_pointwise_logits_qualified','optimizer_trajectory_replayed','selected_indices_qualified',
            'mask_RLE_qualified','AP_evaluated','extra_F_independently_rerendered','teacher_or_PR_loaded','full_native_L_loaded')),
        'Immutable CPU details/snapshot and endpoint refs; no numeric rerun or CPU qualification upgraded to AP')
    closed(c, details['CPU_input_reads']); closed(c, q['qualifier_body_source'])
    require(set(root['checkpoints']) == set(root['FIT_endpoints']) == {'8','24'},
        'Exactly both prelocked endpoints; no DEV winner chosen')
    gate.update(N20_training_root=root, N20_training_CPU_Q_ref=CPU['Q'],
        trained_checkpoints=root['checkpoints'],
        training=dict(root=root, gpu=gpu_spec, GPU_record=GPU, report=report, cpu=cpu_spec,
            CPU_record=CPU, q=q, details=details, start=start, original_binding=binding,
            checkpoints=root['checkpoints'], FIT_endpoints=root['FIT_endpoints']),
        original_training_GPU_actual0_verified=True, training_saved_artifact_recovery=False,
        independent_endpoint_CPU_qualified=True)
    return gate


def source_gate(path, spec):
    owner = n20_repeat202_DEV_header(spec)
    raw22 = n20_failed22_raw_only(spec); raw23 = n20_failed23_raw_only(spec)
    raw24 = n20_failed24_raw_only(spec); raw25 = n20_failed25_raw_only(spec)
    raw26 = n20_failed26_raw_only(spec)
    gate = inherited_source_gate(path, spec, cpu=False)
    require(gate['source'] == spec['entry_source'] and gate['context']['normal_RAW_count'] == owner
        and gate['context']['RAW_inputs'] == spec['RAW_inputs'] and 'parent' in gate,
        'Current DEV GPU own context/parent retained, never a completed training own_gate')
    first201, history_spec = n20_repeat202_first_permission(spec, gate)
    historical = n20_repeat202_history_view(spec, gate, history_spec)
    n20_failed22_closed(historical, gate, raw22); n20_failed23_closed(historical, gate, raw23)
    n20_failed24_closed(historical, gate, raw24); n20_failed25_closed(historical, gate, raw25)
    n20_failed26_closed(historical, gate, raw26)
    for key in N20_REPEAT202_MODEL_KEYS + ('N20_training_tail_source','region_common_source','new_eval_body_source',
            'N20_repeat202_ROOT_training_qualification','legacy_qualified_result'):
        closed(gate['closure'], spec[key])
    protocol, cache_root, old_report, report = n20_repeat202_cache(spec, gate)
    init_root, metadata = n20_repeat202_initialization(spec, gate, history_spec)
    probe_root = n20_repeat202_cleanup_probe(spec, gate)
    common = load(spec['region_common_source'], '_N20_repeat202_DEV_original_library_gate')
    require(callable(getattr(common, 'imported_files', None)), 'Original same-input library ABI')
    # Retain inherited gate.protocol (old N16); the N20 protocol is a separate role.
    gate.update(common=common, N20_protocol=protocol, initialization_metadata=metadata,
        N20_cache_root=cache_root, parent_cache_report=old_report, parent_cache_report_ref=spec['cache_GPU_report'],
        neighbor_report=report, neighbor_report_ref=cache_root['GPU_report'],
        first201_root_permission=first201, repeat202_initialization_qualification=init_root,
        repeat202_cleanup_probe_qualification=probe_root,
        training_binding_refs=dict(N20_root_cache_qualified_result=spec['N20_root_cache_qualified_result'],
            N20_initialization_lock=spec['N20_initialization_lock'], N20_cache_CPU_Q=cache_root['CPU_Q'],
            N20_first201_ROOT_qualified_result=spec['N20_first201_ROOT_qualified_result'],
            N20_repeat202_initialization_qualified_result=spec['N20_repeat202_initialization_qualified_result'],
            N20_repeat202_cleanup_probe_qualified_result=spec['N20_repeat202_cleanup_probe_qualified_result']))
    n20_repeat202_DEV_training_metadata_gate(spec, gate)
    require(callable(globals().get('legacy_gate')) and legacy_gate(spec, gate) is gate
        and len(gate['legacy_rows']) == 187 and len(gate['legacy_reports']) == len(gate['legacy_manifests']) == 4
        and set(gate['populations']) == {'native_official','strong_fixed_E24_Omega'},
        'Original N16 FOUR logical shards/full187 native and strong R references remain exact')
    require('evaluation' in gate and 'lineage' in gate and 'parent' in gate
        and callable(globals().get('n20_prior_DEV_carrier_gate')),
        'Required original prior carrier metadata hook/PR loader; no default true or invented physical shard refs')
    require(n20_prior_DEV_carrier_gate(spec, gate) is gate,
        'Original N17 FIVE physical shared/common-PR carriers are distinct from N16 logical partition')
    return gate
