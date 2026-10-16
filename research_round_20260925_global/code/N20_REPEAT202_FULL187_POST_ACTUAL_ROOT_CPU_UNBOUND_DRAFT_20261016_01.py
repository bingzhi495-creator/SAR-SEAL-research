"""UNBOUND_NOT_EXECUTED repeat202 full187 post suffix, no typed prefix/CLI.

Only completed-final metadata is consumed. Root must obtain the original final
terminal0 and a new pure-prefix actual0 before admitting this independent CPU.
This post owner's pending output cannot certify its own later terminal state.
"""
import gc
import json
import os
from pathlib import Path
import sys
import time

N20_REPEAT202_POST_SPEC_SCHEMA = 'ROOT-N20-repeat202-fixedDEV-full187-post-actual-CPU-spec-v1'
N20_FINAL_CPU_SCHEMA = N20_REPEAT202_POST_SPEC_SCHEMA
N20_PARENT_FINAL_SPEC_SCHEMA = 'ROOT-N20-repeat202-fixedDEV-full187-CPU-COCO-final-spec-v1'
N20_POST_RECORD_KEYS = {'source','spec','Q','RAW','admission','phase_binding','bootstrap','owner_index'}
N20_POST_SCHEMA = 'N20-repeat202-fixedDEV-post-actual-final-ROOT-CPU-output-v1'
N20_POST_STATUS = 'N20_REPEAT202_POST_ACTUAL_FINAL_ROOT_RESULT_WRITTEN_PENDING_POST_OWNER_ACTUAL0'
N20_POST_THREAD_ENV = ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')


def n20_validate_post_header(spec):
    require(type(spec) is dict and spec['schema'] == N20_FINAL_CPU_SCHEMA
        and all(type(spec[k]) is int for k in ('post_owner_index','final_owner_index','normal_RAW_count',
            'expected_normal_RAW_count','total_failed_RAW_count','total_terminal_identity_pin_count',
            'deadline_seconds','allocator_limit_MiB','whole_GPU_limit_MiB','maximum_process_RSS_bytes','initialization_seed'))
        and 219 <= spec['final_owner_index'] < spec['post_owner_index']
        and spec['normal_RAW_count'] == spec['expected_normal_RAW_count'] == spec['post_owner_index']
        and spec['total_failed_RAW_count'] == 26
        and spec['total_terminal_identity_pin_count'] == spec['post_owner_index']+26
        and spec['initialization_seed'] == 202610150202 and spec['deadline_seconds'] == 1800
        and spec['allocator_limit_MiB'] == 4096 and spec['whole_GPU_limit_MiB'] == 8192
        and spec['maximum_process_RSS_bytes'] == 16*1024**3 and spec['automatic_retry'] is False
        and spec['failed26_partials_reused'] is False,
        'Independent post explicit owner/26/pins and fixed repeat202 resources')
    record = spec['final_record']
    require(type(record) is dict and set(record) == N20_POST_RECORD_KEYS
        and type(record['owner_index']) is int and record['owner_index'] == spec['final_owner_index'],
        'Original completed final exact8 record with Q and explicit prior owner')
    for key in N20_POST_RECORD_KEYS-{'owner_index'}:
        ref_shape(record[key])
    ref_shape(spec['completed_final_CPU_source']); ref_shape(spec['entry_source'])
    require(spec['completed_final_CPU_source'] == record['source']
        and spec['entry_source']['path'] != record['source']['path']
        and Path(__file__).resolve(strict=True) == Path(spec['entry_source']['path']).resolve(strict=True)
        and Path(spec['CPU_admission_file']).is_absolute()
        and Path(spec['CPU_admission_file']).resolve() != Path(record['admission']['path']).resolve()
        and len(spec['RAW_inputs']) == spec['post_owner_index'] and type(spec['phase']) is str and bool(spec['phase'])
        and all(row['phase'] != spec['phase'] for row in spec['RAW_inputs']),
        'Fresh post own source/admission/phase; live post absent from completed prefix')
    return True


def n20_post_validate_pending_provenance(Q, details):
    """Completed training ancestry differs from this still-pending final/post."""
    for value in (Q, details):
        require(value['original_training_GPU_actual0_verified'] is True
            and value['training_saved_artifact_recovery'] is False
            and value['independent_endpoint_CPU_qualified'] is True
            and value.get('training_artifacts_recovery_qualified',False) is False
            and value['failed26_partials_reused'] is False,
            'Normal repeat202 training ancestry, not first201 saved-artifact recovery')
    require(all(Q[k] is False for k in ('qualified','ROOT_actual_CPU0_verified',
        'current_owned_CPU_terminal_self_certified','numerical_forward_rerun','head_loaded',
        'CUDA_initialized','local_AP_averaged','CPU_scout_forward_used_for_ranking',
        'AP_results_interpreted_as_qualified','task_efficacy_established',
        'innovation_established','TGRS_readiness_established'))
        and all(details[k] is False for k in ('numerical_forward_rerun','head_loaded','local_AP_averaged')),
        'Original final Q remains pending; lineage success cannot certify own terminal or innovation')
    return True


def n20_post_parent_metadata(final, parent, gate):
    """Delegate only complete pure metadata aggregation; never the final own gate."""
    require(callable(getattr(final,'n20_final_parent_metadata',None)), 'Completed final pure aggregation API')
    require(final.n20_final_parent_metadata(parent,gate) is gate, 'Shared full187 parent metadata gate')
    return gate


def n20_post_validate_final(record, spec, gate, final):
    c = gate['closure']; context = gate['context']; owner = record['owner_index']; row = spec['RAW_inputs'][owner]
    require(context['normal_RAW_count'] == spec['post_owner_index'] and context['RAW_inputs'] == spec['RAW_inputs']
        and all(row[k] == record[v] for k,v in (('receipt','RAW'),('admission','admission'),
            ('phase_binding','phase_binding'),('bootstrap','bootstrap'))), 'Original final belongs to completed post prefix')
    raw = terminal_raw_only(record['RAW'],row['phase'],0)
    check_pin(spec['terminal_identity_pins'][record['RAW']['path']],raw,row['phase'],0)
    for key in N20_POST_RECORD_KEYS-{'owner_index'}:
        closed(c,record[key])
    joined = gate['lineage'].owner_ref_join(context,gate['owners'],record['RAW'],record['admission'],record['source'],record['spec'],'')
    parent = read(closed(c,record['spec'])); Q = read(closed(c,record['Q']))
    final.n20_validate_final_header(parent)
    require(type(joined) is int and joined == owner == spec['final_owner_index'] == parent['final_owner_index']
        and parent['schema'] == N20_PARENT_FINAL_SPEC_SCHEMA and parent['entry_source'] == record['source']
        and parent['phase'] == row['phase'] and parent['RAW_inputs'] == spec['RAW_inputs'][:owner]
        and all(c.get(ref['path']) == ref for ref in parent['sources'])
        and all(spec['terminal_identity_pins'].get(k) == v for k,v in parent['terminal_identity_pins'].items())
        and Path(parent['CPU_admission_file']).resolve(strict=True) == Path(record['admission']['path']).resolve(strict=True)
        and Path(record['Q']['path']).resolve(strict=True) == (Path(parent['output_directory'])/'Q.json').resolve(strict=True)
        and all(final.n20_final_exact(parent[k],spec[k]) for k in final.N20_FINAL_SHARED_KEYS+(
            'numeric_CPU_shards','numeric_CPU_shard_gate_source','GPU_outer_source','interpreter_source'))
        and all(final.n20_final_exact(v,spec[k]) for k,v in parent.items() if k.startswith('additional_preserved_')),
        'Completed CPUempty final actual/source/spec/ad/phase/prefix/all Sources/pins unchanged')
    n20_post_parent_metadata(final,parent,gate)
    parent_gate = dict(gate,source=record['source'],spec_ref=record['spec'],admission_ref=record['admission'])
    binding = final.n20_final_binding(parent,parent_gate)
    require(Q['schema'] == 'N20-fixedDEV-full187-CPU-COCO-qualification-v1'
        and Q['status'] == 'N20_ALL18700_FIXED_DEV_CPU_NUMERIC_AND_FULL_COCO_COMPLETE_PENDING_ROOT_ACTUAL0'
        and all(Q[k] == v for k,v in binding.items()) and Q['images'] == 187 and Q['GT'] == 380 and Q['natural_queries'] == 18700
        and Q['initialization_seed'] == 202610150202 and Q['normal_RAW_count'] == owner
        and Q['total_failed_RAW_count'] == 26 and Q['total_terminal_identity_pin_count'] == owner+26
        and tuple(Q['policies']) == final.n20_final_policy_names()
        and set(Q['checks']) == set(final.N20_FINAL_CHECKS) and all(Q['checks'][k] is True for k in final.N20_FINAL_CHECKS),
        'Immutable pending repeat202 final Q/full187/ten policies/checks; actual0 is from original RAW')
    details = read(closed(c,Q['details']))
    require(Path(Q['details']['path']).resolve(strict=True) == (Path(parent['output_directory'])/'DETAILS.json').resolve(strict=True)
        and details['schema'] == 'N20-fixedDEV-full187-CPU-COCO-details-v1'
        and all(details[k] == v for k,v in binding.items()) and details['control_results'] == Q['control_results']
        and details['image_proofs'] == gate['image_proofs'] and details['registered_bundle_members_consumed'] == gate['registered_members'],
        'Original DETAILS full binding and all187 proof references')
    n20_post_validate_pending_provenance(Q,details)
    evidence = gate['numeric_CPU_evidence']
    require(len(evidence) == 32 and len(gate['full_rows']) == len(gate['image_proofs']) == 187
        and [r['dataset_index'] for r in gate['full_rows']] == list(range(187)), 'Complete fixed32 scopes and full187 order')
    manifests = list({item['child']['parent_GPU_report']['capture_manifest']['path']:
        item['child']['parent_GPU_report']['capture_manifest'] for item in evidence}.values())
    captures = [ref for item in evidence for ref in item['child']['scope_capture_refs']]
    require(details['capture_manifests'] == manifests and len(manifests) == 4
        and details['capture_refs'] == captures and len(captures) == len({r['path'] for r in captures}) == 187,
        'Original four full GPU manifests/187 physical captures; no partial reuse')
    for ref in details['image_proofs']+manifests+captures:
        closed(c,ref)
    require(set(Q['control_results']) == set(final.n20_final_policy_names()), 'Exactly ten complete final policies')
    for name in final.n20_final_policy_names():
        result = Q['control_results'][name]
        require(type(result) is dict and set(result) == {'predictions','COCO','stats'}, 'Only predictions are strictrefs; COCO is inline')
        gate['gpu_helpers'].validate_legacy_COCO(result)
        values = read(closed(c,result['predictions']))
        expected = [value for item in evidence for value in item['child']['scope_expected_predictions'][name]]
        require(len(values) == 18700 and values == expected
            and Path(result['predictions']['path']).resolve(strict=True) ==
                (Path(parent['output_directory'])/(name+'_PREDICTIONS.json')).resolve(strict=True),
            'All32 scoped predictions/RLE/class/S0/export order exactly equal original full18700')
        if name in ('native_official','strong_fixed_E24_Omega'):
            baseline = gate['legacy_root']['control_results'][name]
            require(values == read(closed(c,baseline['predictions'])) and result['stats'] == baseline['stats']
                and result['COCO'] == baseline['COCO'], 'Native/R original full predictions/inline COCO/all12stats unchanged')
        del values,expected
    interpreter = load(parent['interpreter_source'],'_N20_repeat202_post_prelocked_two_seed_interpreter')
    final.n20_validate_final_interpreter(interpreter,gate['N20_protocol'],parent['interpreter_source'])
    interpreted = final.n20_final_main_preview(interpreter,Q['control_results'],gate)
    require(interpreted == Q['primary_interpretation_preview_pending_ROOT_actual0'],
        'Qualified first201 plus current202 E24-atmost main only; unchanged pending preview')
    qualified = {k:binding[k] for k in binding if k not in ('source','spec','CPU_admission','phase')}
    qualified.update(schema='ROOT-N20-repeat202-fixedDEV-full187-GPU-CPU-COCO-qualified-result-v1',
        status='ROOT_N20_REPEAT202_ALL18700_FIXED_DEV_GPU_CPU_FULL_COCO_QUALIFIED',qualified=True,
        original_final_CPU_actual0_verified=True,final_owner=record,final_Q=record['Q'],final_details=Q['details'],
        control_results=Q['control_results'],images=187,GT=380,natural_queries=18700,
        primary_interpretation=interpreted,normal_after=owner+1,preserved_failures=26,pins_after=owner+27,
        original_training_GPU_actual0_verified=True,training_saved_artifact_recovery=False,independent_endpoint_CPU_qualified=True,
        failed26_partials_reused=False,formal_TEST_evaluated=False,cross_dataset_evaluated=False,
        numerical_forward_rerun=False,COCO_rerun=False,task_efficacy_established=False,
        innovation_established=False,TGRS_readiness_established=False,evidence_scope=Q['evidence_scope'])
    return qualified


def source_gate(path,spec):
    n20_validate_post_header(spec)
    failed = [globals()['n20_failed%d_raw_only'%i](spec) for i in range(22,27)]
    gate = inherited_source_gate(path,spec,cpu=True)
    c = gate['closure']
    closed(c,spec['N20_failed26_helper_source'])
    for key in N20_POST_RECORD_KEYS-{'owner_index'}:
        closed(c,spec['final_record'][key])
    closed(c,spec['completed_final_CPU_source'])
    final = load(spec['completed_final_CPU_source'],'_N20_repeat202_completed_final_pure_metadata_API')
    require(final.N20_FINAL_CPU_SCHEMA == N20_PARENT_FINAL_SPEC_SCHEMA, 'Separate completed final schema and current post schema')
    first, history_spec = final.n20_repeat202_first_permission(spec,gate)
    historical = final.n20_repeat202_history_view(spec,gate,history_spec)
    for i,raw in zip(range(22,27),failed):
        globals()['n20_failed%d_closed'%i](historical,gate,raw)
    qualified = n20_post_validate_final(spec['final_record'],spec,gate,final)
    gate.update(completed_final_module=final,final_qualified_result=qualified)
    return gate


def run(spec_path):
    started = time.monotonic(); require(Path(spec_path).is_absolute(),'Independent absolute post specification')
    path = Path(spec_path).resolve(strict=True); spec = read(path); n20_validate_post_header(spec)
    require(path != Path(spec['final_record']['spec']['path']).resolve(strict=True), 'Fresh post spec distinct from frozen final')
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
            and rss < 16*1024**3 and available >= 16*1024**3
            and 'torch' not in sys.modules and 'numpy' not in sys.modules,
            'Post metadata only/CPU01/emptyCUDA/two-thread/RAM/RSS/1800s bounds')
    check(); require(psutil.disk_usage(str(output.parent)).free >= 1024**3,'Post output free1GiB')
    require(gate['source'] == spec['entry_source'] and gate['spec_ref'] == reference(path)
        and gate['admission_ref']['path'] == str(Path(spec['CPU_admission_file']).resolve(strict=True))
        and gate['context']['normal_RAW_count'] == spec['post_owner_index'] and gate['context']['RAW_inputs'] == spec['RAW_inputs'],
        'Independent current post own context remains intact after parent metadata')
    for ref in spec['sources']:
        check(); bound(ref)
    require(reference(path) == gate['spec_ref'] and reference(__file__) == gate['source']
        and reference(spec['CPU_admission_file']) == gate['admission_ref'], 'Post Source/spec/admission immutable')
    gate['lineage'].final_endpoint_check(gate['parent']); check(); gc.collect(); output.mkdir()
    value = dict(schema=N20_POST_SCHEMA,status=N20_POST_STATUS,source=gate['source'],spec=gate['spec_ref'],
        CPU_admission=gate['admission_ref'],phase=spec['phase'],completed_final_CPU_source=spec['completed_final_CPU_source'],
        final_record=spec['final_record'],final_qualified_result=gate['final_qualified_result'],normal_RAW_count=spec['post_owner_index'],
        total_failed_RAW_count=26,total_terminal_identity_pin_count=spec['post_owner_index']+26,
        post_only=True,original_final_actual0_checked=True,qualified=False,ROOT_post_actual_CPU0_verified=False,
        current_post_owned_terminal_self_certified=False,COCO_rerun=False,numerical_forward_rerun=False,
        head_loaded=False,old_context_modified=False,failed26_partials_reused=False,live_post_added_to_normal=False,
        original_training_GPU_actual0_verified=True,training_saved_artifact_recovery=False,independent_endpoint_CPU_qualified=True,
        initialization_seed=202610150202,peak_observed_process_RSS_bytes=peak,minimum_observed_available_RAM_bytes=minimum,
        elapsed_seconds=time.monotonic()-started,
        evidence_limits='Inner result qualifies the completed original final only. Root must separately verify original post terminal0, then new pure-prefix actual0 and immutable Source/output before outer qualification.')
    write(output/'ROOT_RESULT.json',value); check()
    print(json.dumps(dict(status=N20_POST_STATUS,result=reference(output/'ROOT_RESULT.json'),qualified=False)),flush=True)
