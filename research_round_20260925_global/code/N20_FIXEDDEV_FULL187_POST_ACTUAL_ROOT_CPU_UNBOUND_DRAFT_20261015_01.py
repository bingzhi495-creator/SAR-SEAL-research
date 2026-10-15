"""UNBOUND post-only N20 front+tail. ROOT appends typed25 prefix; no CLI.

The inherited current CPU dispatch uses N20_FINAL_CPU_SCHEMA; historical schemas
remain intact. Original final153 must actually complete before binding this own
CPU154 spec. Parent modules are called only through pure metadata/interpretation
helpers. No parent own gate, model, numeric replay, training or COCO is called.
"""
import gc
import json
import os
from pathlib import Path
import sys
import time

N20_FINAL_CPU_SCHEMA = 'ROOT-N20-fixedDEV-full187-CPU-COCO-final-spec-v1'
N20_POST_RECORD_KEYS = {'source','spec','Q','RAW','admission','phase_binding','bootstrap','owner_index'}
N20_POST_SCHEMA = 'N20-fixedDEV-post-actual-final-ROOT-CPU-output-v1'
N20_POST_STATUS = 'N20_POST_ACTUAL_FINAL_ROOT_RESULT_WRITTEN_PENDING_POST_OWNER_ACTUAL0'
N20_POST_THREAD_ENV = ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')
N20_POST_FINAL_CHECKS = (
    'four_original_DEV_GPU0_eight_CPU0_Source_spec_ad_phase_prefix_exact',
    'all187_all100_registered_F9_FP64_score_numerical_imageproofs_complete',
    'all_actual_GPU_logits_two_CPU_formulas_absolute1e4_and_exact_selected_PR_masks_RLE_preserved',
    'all10_full18700_predictions_scores_classes_native_query_order_exact',
    'all10_original_official_full_COCO12_semantics','native_R_all12stats_inline_COCO_and_full_predictions_exact',
    'seed201_main_only_prelocked_interpreter_Source_exact','all25_failures_preserved_not_normal',
    'original_training_GPU_exit1_and_independent_CPU_recovery_ancestry_preserved')


def n20_validate_post_header(spec):
    require(type(spec) is dict and spec['schema'] == N20_FINAL_CPU_SCHEMA
        and all(type(spec[k]) is int for k in ('normal_RAW_count','expected_normal_RAW_count','total_failed_RAW_count',
            'total_terminal_identity_pin_count','deadline_seconds','allocator_limit_MiB','whole_GPU_limit_MiB',
            'maximum_process_RSS_bytes','initialization_seed'))
        and spec['normal_RAW_count'] == spec['expected_normal_RAW_count'] == 154
        and spec['total_failed_RAW_count'] == 25 and spec['total_terminal_identity_pin_count'] == 179
        and spec['initialization_seed'] == 202610150201 and spec['deadline_seconds'] == 1800
        and spec['allocator_limit_MiB'] == 4096 and spec['whole_GPU_limit_MiB'] == 8192
        and spec['maximum_process_RSS_bytes'] == 16*1024**3 and spec['automatic_retry'] is False,
        'Independent post CPU154/25/179 resources and unchanged first201 recipe')
    record = spec['final_record']
    require(type(record) is dict and set(record) == N20_POST_RECORD_KEYS
        and type(record['owner_index']) is int and record['owner_index'] == 153,
        'Exact original completed final record, explicit owner153, Q output key')
    for key in N20_POST_RECORD_KEYS-{'owner_index'}:
        ref_shape(record[key])
    ref_shape(spec['completed_final_CPU_source']); ref_shape(spec['entry_source'])
    require(spec['completed_final_CPU_source'] == record['source']
        and spec['entry_source']['path'] != record['source']['path']
        and Path(__file__).resolve(strict=True) == Path(spec['entry_source']['path']).resolve(strict=True)
        and Path(spec['CPU_admission_file']).is_absolute()
        and Path(spec['CPU_admission_file']).resolve() != Path(record['admission']['path']).resolve()
        and len(spec['RAW_inputs']) == 154 and type(spec['phase']) is str and bool(spec['phase'])
        and all(row['phase'] != spec['phase'] for row in spec['RAW_inputs']),
        'Fresh own post source/admission/phase; live post absent from the completed normal prefix')
    return True


def n20_post_parent_metadata(final, parent, gate):
    """Minimal pure metadata portion of frozen final front, no final own gate."""
    records = final.n20_validate_final_header(parent); c = gate['closure']
    for key in ('numeric_CPU_shard_gate_source','GPU_outer_source','interpreter_source','CPU_execution_partition_supplement'):
        closed(c, parent[key])
    for item in parent['completed_GPU_shards']+records:
        for key in ('source','spec','report','RAW','admission','phase_binding','bootstrap'):
            closed(c, item[key])
    numeric = load(parent['numeric_CPU_shard_gate_source'], '_N20_post_completed_CPU_metadata')
    require(numeric.N20_DEV_CPU_SCHEMA == final.N20_FINAL_NUMERIC_SCHEMA
        and tuple(numeric.N20_CPU_RANGES) == final.N20_FINAL_CPU_RANGES
        and tuple(numeric.N20_CPU_PARENT_GPU_IDS) == final.N20_FINAL_PARENT_IDS,
        'Original completed standalone numerical CPU metadata ABI')
    original = read(closed(c, parent['completed_GPU_shards'][0]['spec']))
    numeric.n20_recovered_training_metadata_gate(original, gate)
    partition = read(closed(c, parent['CPU_execution_partition_supplement']))
    require(type(partition) is dict and final.n20_final_exact(partition,parent['expected_CPU_execution_partition_supplement']),
        'Original complete prelocked execution supplement')
    gate['CPU_execution_partition_supplement'] = partition
    for key in numeric.N20_CPU_SOURCE_KEYS:
        closed(c, parent[key])
    GPUs = [numeric.n20_completed_DEV_GPU_record(parent,gate,item) for item in parent['completed_GPU_shards']]
    require(parent['GPU_outer_source'] in [r['source'] for r in parent['completed_GPU_shards']], 'Actual completed GPU helper source')
    gpu = load(parent['GPU_outer_source'], '_N20_post_original_inline_COCO_semantics')
    require(tuple(gpu.n20_policy_names()) == final.n20_final_policy_names()
        and callable(getattr(gpu,'validate_legacy_COCO',None)), 'Ten original policies/inline COCO metadata validator')
    evidence = []; rows = []; proofs = []; members = 0
    for record in records:
        item, count = final.n20_final_completed_CPU(parent,gate,record,numeric,GPUs)
        evidence.append(item); rows.extend(item['child']['scope_rows']); proofs.extend(item['report']['image_proofs']); members += count
    require([row['dataset_index'] for row in rows] == list(range(187)) and len({row['image_id'] for row in rows}) == 187
        and len(proofs) == len({ref['path'] for ref in proofs}) == 187
        and [item['parent']['GPU_shard']['shard_owner_index'] for item in evidence] == [141,141,142,142,143,143,144,144],
        'Four original GPU/eight CPU actual0 scopes, complete187 images, no failed partial reuse')
    gate.update(gpu_helpers=gpu,numeric_helpers=numeric,completed_GPU_evidence=GPUs,numeric_CPU_evidence=evidence,
        full_rows=rows,image_proofs=proofs,registered_members=members)
    return gate


def n20_post_validate_final(record, spec, gate, final):
    c = gate['closure']; context = gate['context']; owner = record['owner_index']; row = spec['RAW_inputs'][owner]
    require(context['normal_RAW_count'] == 154 and context['RAW_inputs'] == spec['RAW_inputs']
        and all(row[k] == record[v] for k,v in (('receipt','RAW'),('admission','admission'),
            ('phase_binding','phase_binding'),('bootstrap','bootstrap'))), 'Real completed own prefix includes original final153')
    raw = terminal_raw_only(record['RAW'],row['phase'],0)
    check_pin(spec['terminal_identity_pins'][record['RAW']['path']],raw,row['phase'],0)
    for key in N20_POST_RECORD_KEYS-{'owner_index'}:
        closed(c,record[key])
    joined = gate['lineage'].owner_ref_join(context,gate['owners'],record['RAW'],record['admission'],record['source'],record['spec'],'')
    parent = read(closed(c,record['spec'])); Q = read(closed(c,record['Q']))
    final.n20_validate_final_header(parent)
    require(type(joined) is int and joined == owner == 153 and parent['entry_source'] == record['source']
        and parent['phase'] == row['phase'] and parent['RAW_inputs'] == spec['RAW_inputs'][:153]
        and all(c.get(ref['path']) == ref for ref in parent['sources'])
        and all(spec['terminal_identity_pins'].get(k) == v for k,v in parent['terminal_identity_pins'].items())
        and Path(parent['CPU_admission_file']).resolve(strict=True) == Path(record['admission']['path']).resolve(strict=True)
        and Path(record['Q']['path']).resolve(strict=True) == (Path(parent['output_directory'])/'Q.json').resolve(strict=True)
        and all(final.n20_final_exact(parent[k],spec[k]) for k in final.N20_FINAL_SHARED_KEYS+(
            'numeric_CPU_shards','numeric_CPU_shard_gate_source','GPU_outer_source','interpreter_source'))
        and all(final.n20_final_exact(v,spec[k]) for k,v in parent.items() if k.startswith('additional_preserved_')),
        'Original final CPUempty0 actual/source/spec/ad/phase/prefix/full Sources/pins and immutable bindings')
    n20_post_parent_metadata(final,parent,gate)
    binding = dict(source=record['source'],spec=record['spec'],CPU_admission=record['admission'],phase=parent['phase'],
        numeric_CPU_shards=parent['numeric_CPU_shards'],completed_GPU_shards=parent['completed_GPU_shards'],
        numeric_CPU_shard_gate_source=parent['numeric_CPU_shard_gate_source'],CPU_execution_partition_supplement=parent['CPU_execution_partition_supplement'],
        N20_root_training_qualified_result=parent['N20_root_training_qualified_result'],N20_locked_protocol=parent['N20_locked_protocol'],
        legacy_qualified_result=parent['legacy_qualified_result'],interpreter_source=parent['interpreter_source'],checkpoints=gate['trained_checkpoints'],
        N20_failed_training_record=parent['N20_failed_training_record'],N20_failed_training_stderr=parent['N20_failed_training_stderr'],
        N20_failed_DEV_capture_record=parent['N20_failed_DEV_capture_record'],N20_failed_DEV_capture_stderr=parent['N20_failed_DEV_capture_stderr'])
    require(Q['schema'] == 'N20-fixedDEV-full187-CPU-COCO-qualification-v1'
        and Q['status'] == 'N20_ALL18700_FIXED_DEV_CPU_NUMERIC_AND_FULL_COCO_COMPLETE_PENDING_ROOT_ACTUAL0'
        and all(Q[k] == v for k,v in binding.items()) and Q['images'] == 187 and Q['GT'] == 380 and Q['natural_queries'] == 18700
        and Q['initialization_seed'] == 202610150201 and Q['normal_RAW_count'] == 153
        and Q['total_failed_RAW_count'] == 25 and Q['total_terminal_identity_pin_count'] == 178
        and tuple(Q['policies']) == final.n20_final_policy_names()
        and set(Q['checks']) == set(N20_POST_FINAL_CHECKS) and all(Q['checks'][k] is True for k in N20_POST_FINAL_CHECKS)
        and all(Q[k] is False for k in ('qualified','ROOT_actual_CPU0_verified','current_owned_CPU_terminal_self_certified',
            'numerical_forward_rerun','head_loaded','CUDA_initialized','local_AP_averaged','CPU_scout_forward_used_for_ranking',
            'AP_results_interpreted_as_qualified','original_training_GPU_actual0_verified','task_efficacy_established',
            'innovation_established','TGRS_readiness_established')) and Q['training_artifacts_recovery_qualified'] is True,
        'Immutable original pending final Q/full187/ten policies/checks/limits; actual0 comes from original RAW join')
    details = read(closed(c,Q['details']))
    require(Path(Q['details']['path']).resolve(strict=True) == (Path(parent['output_directory'])/'DETAILS.json').resolve(strict=True)
        and details['schema'] == 'N20-fixedDEV-full187-CPU-COCO-details-v1'
        and all(details[k] == v for k,v in binding.items()) and details['control_results'] == Q['control_results']
        and details['image_proofs'] == gate['image_proofs'] and details['registered_bundle_members_consumed'] == gate['registered_members']
        and all(details[k] is False for k in ('numerical_forward_rerun','head_loaded','local_AP_averaged','original_training_GPU_actual0_verified'))
        and details['training_artifacts_recovery_qualified'] is True, 'Original DETAILS and187 complete proof refs/flags')
    evidence = gate['numeric_CPU_evidence']
    manifests = list({item['child']['parent_GPU_report']['capture_manifest']['path']:
        item['child']['parent_GPU_report']['capture_manifest'] for item in evidence}.values())
    captures = [ref for item in evidence for ref in item['child']['scope_capture_refs']]
    require(details['capture_manifests'] == manifests and len(manifests) == 4
        and details['capture_refs'] == captures and len(captures) == len({r['path'] for r in captures}) == 187,
        'Original four full manifests/187 actual physical captures, correct origin bindings')
    for ref in details['image_proofs']+manifests+captures:
        closed(c,ref)
    require(set(Q['control_results']) == set(final.n20_final_policy_names()), 'Exactly ten complete final policies')
    for name in final.n20_final_policy_names():
        result = Q['control_results'][name]
        require(type(result) is dict and set(result) == {'predictions','COCO','stats'}, 'Inline COCO metadata, strictref predictions only')
        gate['gpu_helpers'].validate_legacy_COCO(result)
        values = read(closed(c,result['predictions']))
        expected = [value for item in evidence for value in item['child']['scope_expected_predictions'][name]]
        require(len(values) == 18700 and values == expected
            and Path(result['predictions']['path']).resolve(strict=True) ==
                (Path(parent['output_directory'])/(name+'_PREDICTIONS.json')).resolve(strict=True),
            'Original full18700 predictions/RLE/class/S0/export order exact all8 CPU/all4 GPU')
        if name in ('native_official','strong_fixed_E24_Omega'):
            baseline = gate['legacy_root']['control_results'][name]
            require(values == read(closed(c,baseline['predictions'])) and result['stats'] == baseline['stats']
                and result['COCO'] == baseline['COCO'], 'Native/R full predictions/all12 stats/inline COCO exact original reference')
        del values,expected
    interpreter = load(parent['interpreter_source'],'_N20_post_frozen_first201_interpreter')
    final.n20_validate_final_interpreter(interpreter,gate['N20_protocol'],parent['interpreter_source'])
    interpreted = final.n20_final_main_preview(interpreter,Q['control_results'])
    require(interpreted == Q['primary_interpretation_preview_pending_ROOT_actual0'], 'Only prelocked E24 atmost percent interpretation, unchanged from pending preview')
    return dict(schema='ROOT-N20-fixedDEV-full187-GPU-CPU-COCO-qualified-result-v1',
        status='ROOT_N20_ALL18700_FIXED_DEV_GPU_CPU_FULL_COCO_QUALIFIED',qualified=True,
        original_final_CPU_actual0_verified=True,final_owner=record,final_Q=record['Q'],final_details=Q['details'],
        **{k:binding[k] for k in binding if k not in ('source','spec','CPU_admission','phase')},
        control_results=Q['control_results'],images=187,GT=380,natural_queries=18700,initialization_seed=202610150201,
        primary_interpretation=interpreted,normal_after=154,preserved_failures=25,pins_after=179,
        original_training_GPU_actual0_verified=False,training_artifacts_recovery_qualified=True,
        formal_TEST_evaluated=False,cross_dataset_evaluated=False,numerical_forward_rerun=False,COCO_rerun=False,
        task_efficacy_established=False,innovation_established=False,TGRS_readiness_established=False,evidence_scope=Q['evidence_scope'])


def source_gate(path,spec):
    n20_validate_post_header(spec)
    failed22 = n20_failed22_raw_only(spec); failed23 = n20_failed23_raw_only(spec)
    failed24 = n20_failed24_raw_only(spec); failed25 = n20_failed25_raw_only(spec)
    gate = inherited_source_gate(path,spec,cpu=True)
    n20_failed22_closed(spec,gate,failed22); n20_failed23_closed(spec,gate,failed23)
    n20_failed24_closed(spec,gate,failed24); n20_failed25_closed(spec,gate,failed25)
    c = gate['closure']
    for key in N20_POST_RECORD_KEYS-{'owner_index'}:
        closed(c,spec['final_record'][key])
    closed(c,spec['completed_final_CPU_source'])
    final = load(spec['completed_final_CPU_source'],'_N20_completed_final_pure_metadata_API')
    require(final.N20_FINAL_CPU_SCHEMA == N20_FINAL_CPU_SCHEMA, 'Frozen final metadata API schema')
    qualified = n20_post_validate_final(spec['final_record'],spec,gate,final)
    gate.update(completed_final_module=final,final_qualified_result=qualified)
    return gate


def run(spec_path):
    started = time.monotonic(); require(Path(spec_path).is_absolute(),'Independent absolute post specification')
    path = Path(spec_path).resolve(strict=True); spec = read(path); n20_validate_post_header(spec)
    require(path != Path(spec['final_record']['spec']['path']).resolve(strict=True), 'New post spec cannot be the frozen final spec')
    output = Path(spec['output_directory'])
    require(output.is_absolute() and not output.exists() and not output.is_symlink() and output.parent.is_dir(), 'Fresh post output')
    gate = source_gate(path,spec)
    import psutil
    process = psutil.Process(); peak = 0; minimum = None
    def check():
        nonlocal peak,minimum
        rss = process.memory_info().rss; available = psutil.virtual_memory().available
        peak = max(peak,rss); minimum = available if minimum is None else min(minimum,available)
        require(time.monotonic()-started < 1800 and process.cpu_affinity() == [0,1]
            and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and all(os.environ.get(k) == '2' for k in N20_POST_THREAD_ENV)
            and rss < 16*1024**3 and available >= 16*1024**3, 'Post CPU01/emptyCUDA/two-thread/RAM/RSS/1800s bounds')
        if 'torch' in sys.modules:
            require(not sys.modules['torch'].cuda.is_initialized(),'Post metadata must never initialize CUDA')
    check(); require(psutil.disk_usage(str(output.parent)).free >= 1024**3,'Post output free1GiB')
    check()
    require(gate['source'] == spec['entry_source'] and gate['spec_ref'] == reference(path)
        and gate['admission_ref']['path'] == str(Path(spec['CPU_admission_file']).resolve(strict=True))
        and gate['context']['normal_RAW_count'] == 154 and gate['context']['RAW_inputs'] == spec['RAW_inputs'],
        'Real current own post context/source/spec/admission; original final context was not changed')
    for ref in spec['sources']:
        check(); bound(ref)
    require(reference(path) == gate['spec_ref'] and reference(__file__) == gate['source']
        and reference(spec['CPU_admission_file']) == gate['admission_ref'], 'Final post Source/spec/admission immutable')
    gate['lineage'].final_endpoint_check(gate['parent']); check(); gc.collect(); output.mkdir()
    value = dict(schema=N20_POST_SCHEMA,status=N20_POST_STATUS,source=gate['source'],spec=gate['spec_ref'],
        CPU_admission=gate['admission_ref'],phase=spec['phase'],completed_final_CPU_source=spec['completed_final_CPU_source'],
        final_record=spec['final_record'],final_qualified_result=gate['final_qualified_result'],normal_RAW_count=154,
        total_failed_RAW_count=25,total_terminal_identity_pin_count=179,post_only=True,original_final_actual0_checked=True,
        qualified=False,ROOT_post_actual_CPU0_verified=False,current_post_owned_terminal_self_certified=False,
        COCO_rerun=False,numerical_forward_rerun=False,head_loaded=False,old_context_modified=False,
        live_post_added_to_normal=False,peak_observed_process_RSS_bytes=peak,minimum_observed_available_RAM_bytes=minimum,
        elapsed_seconds=time.monotonic()-started,evidence_limits='Inner result is completed original final153; ROOT must separately verify original post154 actual0 and Source before publishing outer qualification.')
    write(output/'ROOT_RESULT.json',value); check()
    print(json.dumps(dict(status=N20_POST_STATUS,result=reference(output/'ROOT_RESULT.json'),qualified=False)),flush=True)
