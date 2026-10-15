"""UNBOUND full187 N20 final CPU metadata front; no typed prefix/run/CLI.

ROOT supplies typed22..26 and inherited_source_gate, whose current CPU dispatch
uses N20_FINAL_CPU_SCHEMA. Historical SCHEMA/CPU_SCHEMA literals stay unchanged.
numeric_CPU_shard_gate_source is a completed standalone numeric CPU ENTRY, not
the isolated front fragment: pure functions retain their original globals and
__file__. No parent's source_gate/run/model/numerical/COCO function is invoked.
All future original actual0 joins are mandatory; this source asserts no result.
"""
import json
import math
from pathlib import Path

N20_FINAL_CPU_SCHEMA = 'ROOT-N20-fixedDEV-full187-CPU-COCO-final-spec-v1'
N20_FINAL_NUMERIC_SCHEMA = 'ROOT-N20-fixedDEV-local-center-feature-error-CPU-numeric-shard-spec-v1'
N20_FINAL_PARTITION_SCHEMA = 'N20-fixed-thirty-two-CPU-numeric-subscopes-after-resource-deadline-v1'
N20_FINAL_CPU_RANGES = ((0, 6), (6, 12), (12, 18), (18, 24), (24, 30), (30, 36), (36, 42), (42, 47), (47, 53), (53, 59), (59, 65), (65, 71), (71, 77), (77, 83), (83, 89), (89, 94), (94, 100), (100, 106), (106, 112), (112, 118), (118, 124), (124, 130), (130, 136), (136, 141), (141, 147), (147, 153), (153, 159), (159, 165), (165, 171), (171, 177), (177, 183), (183, 187))
N20_FINAL_GPU_RANGES = ((0,47),(47,94),(94,141),(141,187))
N20_FINAL_PARENT_IDS = (0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3)
N20_FINAL_CPU_KEYS = {'cpu_shard_id','cpu_dataset_ordinal_range','source','spec','report','RAW',
    'admission','phase_binding','bootstrap','cpu_shard_owner_index'}
N20_FINAL_GPU_KEYS = {'shard_id','dataset_ordinal_range','source','spec','report','RAW',
    'admission','phase_binding','bootstrap','shard_owner_index'}
N20_FINAL_TRUE_FLAGS = ('parent_original_DEV_GPU0_CUDA0_bound',
    'all_actual_GPU_scout_logits_two_CPU_formulas_absolute1e4',
    'all_actual_true_F9_or_center_inputs_byte_exact','all_actual_saved_FP64_scores_canonical_exact',
    'all_selected_indices_masks_RLE_exact','all_PR_two_formulas_absolute1e4_strictsign',
    'all_native_R_and_new_score_class_query_order_exact','independent_CPU_proof_complete',
    'training_artifacts_recovery_qualified')
N20_FINAL_FALSE_FLAGS = ('qualified','ROOT_actual_CPU0_verified','current_owned_CPU_terminal_self_certified',
    'original_training_GPU_actual0_verified','DEV_winner_selected','CPU_scout_forward_used_for_ranking',
    'CPU_F_rerendered','full_native_teacher_model_loaded','AP_evaluated','official_COCO_called',
    'failed_partials_reused','failed26_partials_reused','CUDA_initialized')
N20_FINAL_SHARED_KEYS = ('completed_GPU_shards','N20_root_training_qualified_result',
    'expected_N20_root_training_qualified_result','N20_root_cache_qualified_result',
    'expected_N20_root_cache_qualified_result','N20_locked_protocol','expected_N20_locked_protocol',
    'N20_initialization_lock','expected_N20_initialization_digest','N20_failed_training_record',
    'N20_failed_training_stderr','N20_training_tail_source','N20_failed_DEV_capture_record',
    'N20_failed_DEV_capture_stderr','locked_protocol','cache_GPU_report','region_common_source',
    'scout_module_source','base_scout_source','paired_training_body_source',
    'original_paired_training_body_source','initialization_seed','legacy_qualified_result','legacy_shards',
    'n17_root_qualified_result_ref','expected_n17_root_qualified_result','new_eval_body_source',
    'legacy_eval_body_source','capture_body_source','coordinates_source','CPU_adapter_source',
    'selector_cpu_proof_source','legacy_selector_proof_source','endpoint_body_source','PR_formula_source',
    'labels_source','CPU_execution_partition_supplement','expected_CPU_execution_partition_supplement',
    'N20_failed_CPU_numeric_record','N20_failed26_helper_source','failed26_partials_reused')


def n20_final_exact(left, right):
    return json.dumps(left, sort_keys=True, separators=(',', ':'), allow_nan=False) == json.dumps(
        right, sort_keys=True, separators=(',', ':'), allow_nan=False)


def n20_final_policy_names():
    return ('native_official','strong_fixed_E24_Omega') + tuple('E%d_%s_%s' % (epoch,arm,mode)
        for epoch in (24,8) for arm in ('LOCAL_ERROR','CENTER_ERROR') for mode in ('atmost','exact'))


def n20_validate_final_header(spec):
    require(type(spec) is dict and spec['schema'] == N20_FINAL_CPU_SCHEMA
        and all(type(spec[k]) is int for k in ('normal_RAW_count','expected_normal_RAW_count',
            'total_failed_RAW_count','total_terminal_identity_pin_count','initialization_seed',
            'deadline_seconds','maximum_process_RSS_bytes'))
        and spec['normal_RAW_count'] == spec['expected_normal_RAW_count'] == 177
        and spec['total_failed_RAW_count'] == 26 and spec['total_terminal_identity_pin_count'] == 203
        and spec['initialization_seed'] == 202610150201 and spec['deadline_seconds'] == 1800
        and spec['automatic_retry'] is False and spec['maximum_process_RSS_bytes'] == 16*1024**3
        and spec['failed26_partials_reused'] is False
        and spec['allocator_limit_MiB'] == 4096 and spec['whole_GPU_limit_MiB'] == 8192
        and Path(__file__).resolve(strict=True) == Path(spec['entry_source']['path']).resolve(strict=True),
        'Independent final CPU177/failed26/pins203 own entry/schema/resources; no historical schema rewrite')
    records = spec['numeric_CPU_shards']; GPUs = spec['completed_GPU_shards']
    require(type(records) is list and len(records) == 32 and type(GPUs) is list and len(GPUs) == 4,
        'Exactly four completed original GPU and 32 independent CPU descriptors')
    for i, item in enumerate(GPUs):
        require(type(item) is dict and set(item) == N20_FINAL_GPU_KEYS
            and type(item['shard_id']) is int and item['shard_id'] == i
            and item['dataset_ordinal_range'] == list(N20_FINAL_GPU_RANGES[i])
            and type(item['shard_owner_index']) is int and item['shard_owner_index'] == 141+i,
            'Four original GPU141..144 whole-image scopes, never failed25 partials')
    for i, item in enumerate(records):
        require(type(item) is dict and set(item) == N20_FINAL_CPU_KEYS
            and type(item['cpu_shard_id']) is int and item['cpu_shard_id'] == i
            and item['cpu_dataset_ordinal_range'] == list(N20_FINAL_CPU_RANGES[i])
            and type(item['cpu_shard_owner_index']) is int and item['cpu_shard_owner_index'] == 145+i,
            '32 explicit CPU145..176 original owners and fixed child ranges')
    for group in (GPUs, records):
        for item in group:
            for key in ('source','spec','report','RAW','admission','phase_binding','bootstrap'):
                ref_shape(item[key])
        for key in ('spec','report','RAW','admission','phase_binding','bootstrap'):
            require(len({r[key]['path'] for r in group}) == len(group), 'Distinct actual completed artifact: '+key)
    return records


def n20_validate_final_PR_stats(stats, points):
    require(type(stats) is dict and set(stats) == {'original_head','independent_PR'}, 'Two original PR formulas')
    for value in stats.values():
        require(type(value) is dict and value['calls'] == (points+2047)//2048 and value['points'] == points
            and value['strict_sign_exact'] is True and all(type(value[k]) in (int,float)
                and math.isfinite(value[k]) and 0 <= value[k] <= 1e-4 for k in
                ('residual_max_absolute_error','combined_max_absolute_error')),
            'Complete original PR2048 counts/absolute1e-4/rtol0/strict sign summaries')


def n20_validate_final_image_proof(value, row, names):
    require(type(value) is dict and set(value) == {'dataset_index','image_id','queries','bundle','input_carrier','consumption','details'}
        and value['dataset_index'] == row['dataset_index'] and value['image_id'] == row['image_id']
        and value['queries'] == 100 and value['bundle'] == row['bundle'] and value['input_carrier'] == row['input_carrier'],
        'Original complete per-image CPU proof bound to its actual GPU bundle and input carrier')
    members = len(row['member_locators']); consumed = value['consumption']
    require(consumed['all_registered_members_consumed'] is True and type(consumed['members']) is int
        and consumed['members'] == members > 0 and consumed['queries'] == 100
        and consumed['ROOT_scientific_qualification_conferred'] is False,
        'All and only actual registered members consumed once per whole image')
    details = value['details']
    require(type(details) is list and len(details) == 100 and [d['query_id'] for d in details] == list(range(100)),
        'Independent numeric proofs are natural query order, separate from prediction export order')
    scout_names = {'E%d_%s' % (epoch,arm) for epoch in (24,8) for arm in ('LOCAL_ERROR','CENTER_ERROR')}
    for q, item in enumerate(details):
        query = row['query_profiles'][q]; K = query['K']; C = query['C_count']; active = C if K else 0
        require(item['K'] == K and item['C_points'] == C and all(item[k] is True for k in
                ('all_selected_indices_exact','all_full_masks_and_RLE_exact','actual_true_F9_or_center_inputs_byte_exact',
                 'actual_saved_FP64_scores_canonical_exact','original_query_ids_classes_S0_export_order_require_external_binding'))
            and all(item[k] is False for k in ('CPU_scout_outputs_used_for_ranking','AP_evaluated',
                'ROOT_scientific_qualification_conferred','CPU_feature_rerender_equivalence_proven',
                'actual_capture_provenance_proven_here','native_R_reference_reuse_proven_here'))
            and item['atol'] == 1e-4 and item['rtol'] == 0
            and set(item['profiles']) == set(names) and set(item['scout_numeric']) == scout_names,
            'All8 profiles/four actual-logit formula/true-F9/saved-FP64 exact proofs; no CPU ranking or self qualification')
        common = item['common_C_PR']
        require(common['strict_sign_exact'] is True, 'Original common C strict sign')
        n20_validate_final_PR_stats(common['formula_stats'], active)
        require(common['formula_chunk_calls'] == {k:v['calls'] for k,v in common['formula_stats'].items()},
            'Exact common original PR chunk summary')
        for stat in item['scout_numeric'].values():
            require(stat['chunk_calls'] == (active+2047)//2048 and stat['points'] == active
                and stat['actual_true_F9_or_center_inputs_byte_exact'] is True
                and all(type(stat[k]) in (int,float) and math.isfinite(stat[k]) and 0 <= stat[k] <= 1e-4
                    for k in ('factory_vs_actual_GPU_max_absolute_error','independent_vs_actual_GPU_max_absolute_error')),
                'Both same-input CPU scout formulas vs actual GPU remain absolute1e-4, without ranking tolerance')
        for name, profile in item['profiles'].items():
            require(type(profile['epoch']) is int and profile['epoch'] in (24,8)
                and profile['arm'] in ('LOCAL_ERROR','CENTER_ERROR') and profile['mode'] in ('atmost','exact')
                and name == 'E%d_%s_%s' % (profile['epoch'],profile['arm'],profile['mode'])
                and type(profile['selected_points']) is int and 0 <= profile['selected_points'] <= K
                and (profile['mode'] != 'exact' or profile['selected_points'] == K)
                and all(profile[k] is True for k in ('selected_indices_exact','full_mask_exact',
                    'outside_selected_native_exact','RLE_exact','PR_strict_sign_exact'))
                and profile['CPU_scout_outputs_used_for_ranking'] is False
                and profile['CPU_approximate_PR_used_for_ranking'] is False
                and profile['main_result'] is (profile['epoch'] == 24 and profile['mode'] == 'atmost'),
                'Exact actual support/mask/RLE/raw PR action; E24 atmost alone is main')
            n20_validate_final_PR_stats(profile['selected_PR_formula_stats'], profile['selected_points'])
    return members


def n20_final_completed_CPU(spec, gate, descriptor, numeric_template, GPU_evidence):
    c = gate['closure']; cid = descriptor['cpu_shard_id']; owner = descriptor['cpu_shard_owner_index']
    first, last = N20_FINAL_CPU_RANGES[cid]; pid = N20_FINAL_PARENT_IDS[cid]
    for key in ('source','spec','report','RAW','admission','phase_binding','bootstrap'):
        closed(c, descriptor[key])
    require(descriptor['source']['bytes'] == spec['numeric_CPU_shard_gate_source']['bytes']
        and descriptor['source']['sha256'] == spec['numeric_CPU_shard_gate_source']['sha256'],
        'Each completed numeric source keeps the exact canonical standalone CPU source bytes; paths stay original')
    numeric = load(descriptor['source'], '_N20_completed_numeric_CPU_scope_%d' % cid)
    require(numeric.N20_DEV_CPU_SCHEMA == N20_FINAL_NUMERIC_SCHEMA
        and tuple(numeric.N20_CPU_RANGES) == N20_FINAL_CPU_RANGES
        and tuple(numeric.N20_CPU_PARENT_GPU_IDS) == N20_FINAL_PARENT_IDS
        and numeric.N20_CPU_REPORT_SCHEMA == numeric_template.N20_CPU_REPORT_SCHEMA
        and numeric.N20_CPU_STATUS == numeric_template.N20_CPU_STATUS,
        'Closed completed standalone numeric CPU module/version/scope ABI; no own gate call')
    parent = read(closed(c, descriptor['spec'])); report = read(closed(c, descriptor['report']))
    numeric.n20_validate_cpu_header(parent)
    joined = gate['lineage'].owner_ref_join(gate['context'], gate['owners'], descriptor['RAW'], descriptor['admission'],
        descriptor['source'], descriptor['spec'], '')
    rawrow = spec['RAW_inputs'][owner]
    require(type(joined) is int and joined == owner == parent['normal_RAW_count'] == parent['expected_normal_RAW_count']
        and parent['entry_source'] == descriptor['source'] and parent['phase'] == rawrow['phase']
        and parent['RAW_inputs'] == spec['RAW_inputs'][:owner]
        and parent['total_failed_RAW_count'] == 26 and parent['total_terminal_identity_pin_count'] == owner+26
        and parent['cpu_shard_id'] == cid and parent['cpu_shard_owner_index'] == owner
        and parent['cpu_dataset_ordinal_range'] == [first,last] and parent['CPU_parent_GPU_id'] == pid
        and parent['GPU_shard'] == spec['completed_GPU_shards'][pid]
        and parent['GPU_outer_source'] == parent['GPU_shard']['source']
        and all(rawrow[k] == descriptor[v] for k,v in (('receipt','RAW'),('admission','admission'),
            ('phase_binding','phase_binding'),('bootstrap','bootstrap')))
        and all(c.get(ref['path']) == ref for ref in parent['sources'])
        and all(spec['terminal_identity_pins'].get(k) == v for k,v in parent['terminal_identity_pins'].items())
        and Path(parent['CPU_admission_file']).resolve(strict=True) == Path(descriptor['admission']['path']).resolve(strict=True)
        and Path(descriptor['report']['path']).resolve(strict=True) == (Path(parent['output_directory'])/'REPORT.json').resolve(strict=True),
        'Original CPUempty actual0 own source/ad/bootstrap/prefix/pins/report exact, never a future/live self claim')
    require(all(n20_final_exact(parent[k], spec[k]) for k in N20_FINAL_SHARED_KEYS)
        and all(n20_final_exact(v, spec[k]) for k,v in parent.items() if k.startswith('additional_preserved_')),
        'Every numeric scope retains all common frozen formulas/checkpoints/recovery/protocol and failed26 evidence')
    gp = GPU_evidence[pid]
    child = dict(gate)
    child.update(parent_GPU=gp['spec'], parent_GPU_report=gp['report'], parent_GPU_manifest=gp['manifest'],
        parent_GPU_START=gp['START'], parent_GPU_binding=gp['binding'], expected_predictions=gp['expected_predictions'],
        expected_prediction_groups=gp['expected_prediction_groups'])
    numeric.n20_scope_parent_GPU(parent, child)
    rows = child['scope_rows']; ids = child['scope_image_ids']; capture_refs = child['scope_capture_refs']
    binding = dict(source=descriptor['source'], spec=descriptor['spec'], CPU_admission=descriptor['admission'], phase=parent['phase'],
        GPU_shard=parent['GPU_shard'], CPU_parent_GPU_id=pid, capture_manifest=gp['report']['capture_manifest'], parent_GPU_binding=gp['binding'],
        N20_root_training_qualified_result=spec['N20_root_training_qualified_result'], N20_locked_protocol=spec['N20_locked_protocol'],
        legacy_qualified_result=spec['legacy_qualified_result'], CPU_execution_partition_supplement=spec['CPU_execution_partition_supplement'],
        checkpoints=gate['trained_checkpoints'], CPU_adapter_source=spec['CPU_adapter_source'], selector_cpu_proof_source=spec['selector_cpu_proof_source'],
        legacy_selector_proof_source=spec['legacy_selector_proof_source'], endpoint_body_source=spec['endpoint_body_source'],
        PR_formula_source=spec['PR_formula_source'], labels_source=spec['labels_source'], scout_module_source=spec['scout_module_source'],
        GPU_outer_source=parent['GPU_outer_source'],
        N20_failed_CPU_numeric_record=spec['N20_failed_CPU_numeric_record'],
        N20_failed26_helper_source=spec['N20_failed26_helper_source'])
    names = n20_final_policy_names()
    require(report['schema'] == numeric.N20_CPU_REPORT_SCHEMA and report['status'] == numeric.N20_CPU_STATUS
        and all(report[k] == v for k,v in binding.items()) and report['cpu_shard_id'] == cid
        and report['cpu_dataset_ordinal_range'] == [first,last] and report['cpu_shard_owner_index'] == owner
        and report['normal_RAW_count'] == owner and report['total_failed_RAW_count'] == 26
        and report['total_terminal_identity_pin_count'] == owner+26
        and report['scope_dataset_indices'] == list(range(first,last)) and report['scope_image_ids'] == ids
        and report['images'] == last-first and report['queries'] == (last-first)*100
        and report['global_images'] == 187 and report['global_queries'] == 18700 and report['global_GT'] == 380
        and report['fixed_main_epoch'] == 24 and report['fixed_curve_epoch'] == 8
        and report['numeric_atol'] == 1e-4 and report['numeric_rtol'] == 0
        and tuple(report['controls']) == names and set(report['control_results']) == set(names)
        and all(report[k] is True for k in N20_FINAL_TRUE_FLAGS) and all(report[k] is False for k in N20_FINAL_FALSE_FLAGS)
        and report['teacher_forward_calls'] == 0 and report['capture_refs'] == capture_refs
        and report['actual_capture_files_consumed'] == last-first and len(report['image_proofs']) == last-first,
        'Immutable completed numeric report binds the entire child scope/10 policies and retains original pending/limit flags')
    total_members = 0
    for proof_ref, row in zip(report['image_proofs'], rows):
        proof_path = closed(c, proof_ref)
        require(proof_path.resolve(strict=True) == (Path(parent['output_directory'])/('IMAGE_%03d_PROOF.json' % row['dataset_index'])).resolve(strict=True),
            'Original successful CPU owner imageproof location, never failed partial output')
        total_members += n20_validate_final_image_proof(read(proof_path), row, names[2:])
    require(total_members == report['registered_bundle_members_consumed'] == sum(len(row['member_locators']) for row in rows),
        'Full actual registered member coverage per independent whole-image scope')
    for name in names:
        output = report['control_results'][name]
        require(type(output) is dict and set(output) == {'predictions'}, 'Completed local predictions only; no local AP/COCO')
        require(read(closed(c, output['predictions'])) == child['scope_expected_predictions'][name],
            'Complete CPU predictions exact original GPU slice, including class/S0/export query order: '+name)
    return dict(descriptor=descriptor, parent=parent, report=report, child=child), total_members


def source_gate(path, spec):
    records = n20_validate_final_header(spec)
    failed22 = n20_failed22_raw_only(spec); failed23 = n20_failed23_raw_only(spec)
    failed24 = n20_failed24_raw_only(spec); failed25 = n20_failed25_raw_only(spec)
    failed26 = n20_failed26_raw_only(spec)
    gate = inherited_source_gate(path, spec, cpu=True)
    n20_failed22_closed(spec, gate, failed22); n20_failed23_closed(spec, gate, failed23)
    n20_failed24_closed(spec, gate, failed24); n20_failed25_closed(spec, gate, failed25)
    n20_failed26_closed(spec, gate, failed26)
    c = gate['closure']
    for key in ('numeric_CPU_shard_gate_source','GPU_outer_source','interpreter_source',
            'CPU_execution_partition_supplement','N20_failed_DEV_capture_stderr','N20_failed26_helper_source'):
        closed(c, spec[key])
    for item in spec['completed_GPU_shards']+records:
        for key in ('source','spec','report','RAW','admission','phase_binding','bootstrap'):
            closed(c, item[key])
    numeric = load(spec['numeric_CPU_shard_gate_source'], '_N20_completed_CPU_metadata_contract')
    require(numeric.N20_DEV_CPU_SCHEMA == N20_FINAL_NUMERIC_SCHEMA
        and tuple(numeric.N20_CPU_RANGES) == N20_FINAL_CPU_RANGES
        and tuple(numeric.N20_CPU_PARENT_GPU_IDS) == N20_FINAL_PARENT_IDS
        and callable(getattr(numeric,'n20_recovered_training_metadata_gate',None))
        and callable(getattr(numeric,'n20_completed_DEV_GPU_record',None))
        and callable(getattr(numeric,'n20_scope_parent_GPU',None)), 'Completed CPU pure metadata ABI, not its own-source gate')
    original = read(closed(c, spec['completed_GPU_shards'][0]['spec']))
    numeric.n20_recovered_training_metadata_gate(original, gate)
    partition = read(closed(c, spec['CPU_execution_partition_supplement']))
    require(type(partition) is dict and partition['schema'] == N20_FINAL_PARTITION_SCHEMA
        and type(spec['expected_CPU_execution_partition_supplement']) is dict
        and n20_final_exact(partition, spec['expected_CPU_execution_partition_supplement']), 'Exact complete prelocked 32-CPU execution supplement')
    gate['CPU_execution_partition_supplement'] = partition
    for key in numeric.N20_CPU_SOURCE_KEYS:
        closed(c, spec[key])
    GPU_evidence = [numeric.n20_completed_DEV_GPU_record(spec, gate, item) for item in spec['completed_GPU_shards']]
    require(spec['GPU_outer_source'] in [item['source'] for item in spec['completed_GPU_shards']], 'Actual completed GPU helper source')
    gpu = load(spec['GPU_outer_source'], '_N20_completed_GPU_final_metadata_helpers')
    require(tuple(gpu.n20_policy_names()) == n20_final_policy_names()
        and callable(getattr(gpu,'validate_legacy_COCO',None)), 'Original native/R inline COCO validator and ten fixed policy names')
    evidence = []; rows = []; proofs = []; members = 0
    for descriptor in records:
        item, count = n20_final_completed_CPU(spec, gate, descriptor, numeric, GPU_evidence)
        evidence.append(item); rows.extend(item['child']['scope_rows']); proofs.extend(item['report']['image_proofs']); members += count
    require([row['dataset_index'] for row in rows] == list(range(187)) and len({row['image_id'] for row in rows}) == 187
        and len(proofs) == len({ref['path'] for ref in proofs}) == 187
        and [item['parent']['GPU_shard']['shard_owner_index'] for item in evidence] == [141]*8+[142]*8+[143]*8+[144]*8,
        'All187 whole images/18700 natural queries exactly once across four GPU/32 CPU actual0 parents')
    gate.update(gpu_helpers=gpu, numeric_helpers=numeric, completed_GPU_shards=spec['completed_GPU_shards'],
        completed_GPU_evidence=GPU_evidence, numeric_CPU_evidence=evidence, full_rows=rows, image_proofs=proofs, registered_members=members)
    return gate
