"""UNBOUND v2 adapter: explicit historical lineage and canonical GPU relation.
Only stdlib is imported here. The original v1 common is bound after the
complete current RAW gate; its packet/region/head helpers remain unchanged.
No current process can certify its own terminal or claim cache efficacy.
"""
import os
from pathlib import Path
import types

CPU_SCHEMA='ROOT-N13-sameK-region-FIT-cache-CPU-spec-v2'
Q_SCHEMA='ROOT-N13-sameK-region-FIT-cache-CPU-qualification-v2'
Q_STATUS='ROOT_N13_SAMEK_REGION_FIT_CACHE_CPU_QUALIFIED_V2'
RELATION_SCHEMA='ROOT-N13-frozen-M-to-L-relationship-GPU-replay-spec-v1'
RELATION_REPORT_SCHEMA='N13-frozen-M-to-L-relationship-GPU-replay-report-v1'
RELATION_REPORT_STATUS='N13_FROZEN_M_TO_L_CANONICAL100_EXACT_BYTES_REPLAY_COMPLETE_NO_CACHE_Q'
ROOT_RELATION_SCHEMA='ROOT-N13-frozen-M-to-L-relationship-GPU-qualification-v1'
ROOT_RELATION_STATUS='ACTUAL_GPU_TERMINAL0_CANONICAL100_ALL_MATCHED_BYTES_EXACT_NOT_CACHE_Q'
RELATION_REF_KEYS=('relationship_entry_source','relationship_spec','relationship_report',
 'relationship_GPU_admission','relationship_GPU_RAW','relationship_ROOT_qualification')
VALIDATION_PROTOCOL=dict(
 canonical_GPU_M_to_L_exact_bytes_ROOT_qualification_required=True,
 canonical_GPU_relation_images=741,canonical_GPU_relation_matched_queries=1661,
 native_L_CPU_full_allclose='diagnostic only, original v1 rtol/atol unchanged',
 native_L_CPU_rtol=1e-4,native_L_CPU_atol=1e-4,
 sampled_F_rtol=1e-4,sampled_F_atol=1e-4,sampled_F_allclose_is_strong_gate=True,
 exact_saved_GPU_L_is_sole_selector_and_training_logit=True,
 saved_GPU_L_regions_draws_labels_CPU_exact_replayed=True,
 CPU_replay_never_substitutes_saved_GPU_L_P_B_K=True,
 unmatched_M200_recovered=False,native_teacher_forward_replayed=False,
 certified_GPU_error_bound=False,historical_failures_preserved=True)


def bind_common(common):
    """Forward original helpers; replace only qualified-cache metadata admission."""
    global _COMMON
    if common.GPU_SCHEMA!='ROOT-N13-sameK-region-FIT-cache-GPU-spec-v1' or common.ARMS!=('LOG_PR','GEO_PR'):
        raise RuntimeError('Actually pinned original N13 common ABI differs')
    _COMMON=common
    names=('require','read','reference','ref_shape','bound','load','write','owner_join','joined',
           'OWNER_SHA','GPU_SCHEMA','REPORT_SCHEMA','REPORT_STATUS','RAW_KEYS','REF_KEYS','MODULES',
           'RECIPE','REGIONS_RECIPE','NUMERIC')
    for name in names: globals()[name]=getattr(common,name)
    facade=types.ModuleType('_N13_v2_common_facade')
    facade.__dict__.update(common.__dict__)
    facade.validate_cache_input=validate_cache_input
    facade.CPU_SCHEMA=CPU_SCHEMA; facade.Q_SCHEMA=Q_SCHEMA; facade.Q_STATUS=Q_STATUS
    facade.VALIDATION_PROTOCOL=VALIDATION_PROTOCOL; facade.RELATION_REF_KEYS=RELATION_REF_KEYS
    return facade


def failure_prefix_and_pins(parent,current,extra_count):
    require(parent['prior_failed_CPU_receipt']==current['prior_failed_CPU_receipt'] and
        parent['additional_failed_RAW_inputs']==current['additional_failed_RAW_inputs'][:extra_count],
        'Historical failure list must remain an exact explicit prefix')
    if 'terminal_identity_pins' in parent:
        require(all(current['terminal_identity_pins'].get(k)==v for k,v in parent['terminal_identity_pins'].items()),
            'Historical original identity/admission pin changed')

def relationship_gate(spec,closure,owners):
    """Join ROOT record to already-held original RAW; report flags cannot replace it."""
    require(all(spec[k] in closure.values() for k in RELATION_REF_KEYS), 'Complete relationship primary closure required')
    rs,report,record=(read(bound(spec[k])) for k in
        ('relationship_spec','relationship_report','relationship_ROOT_qualification'))
    require(rs['schema']==RELATION_SCHEMA and rs['entry_source']==spec['relationship_entry_source'] and
        rs['RAW_inputs']==spec['RAW_inputs'][:38] and all(ref in closure.values() for ref in rs['sources']),
        'Full frozen relationship SPEC/prefix/closure differs')
    failure_prefix_and_pins(rs,spec,5)
    require(report['schema']==RELATION_REPORT_SCHEMA and report['status']==RELATION_REPORT_STATUS and
        report['source']==spec['relationship_entry_source'] and report['spec']==spec['relationship_spec'] and
        report['GPU_admission']==spec['relationship_GPU_admission'] and report['phase']==rs['phase'] and
        report['RAW_inputs']==rs['RAW_inputs'] and report['recipe']==rs['recipe'] and
        report['prior_failed_CPU_receipt']==rs['prior_failed_CPU_receipt'] and
        report['additional_failed_RAW_inputs']==rs['additional_failed_RAW_inputs'] and
        report['images']==741 and report['matched_queries']==1661 and report['canonical100_all_exact_bytes'] is True and
        report['canonical100_mismatched_images']==report['canonical100_mismatched_queries']==report['canonical100_mismatched_pixels']==0 and
        report['Q_written'] is False and report['cache_qualified'] is False and
        report['current_owned_GPU_terminal_self_certified'] is False and report['cleanup']['allocated_after']==0,
        'Actual all-image/all-matched-query canonical100 exact-byte relation required')
    mapping={'cache_GPU_spec':'cache_spec','cache_GPU_report':'cache_report',
        'cache_GPU_RAW':'cache_GPU_RAW','cache_GPU_admission':'cache_GPU_admission'}
    require(rs['cache_GPU_entry_source']==report['cache_GPU_source']==spec['cache_entry_source'] and
        all(rs[k]==report[k]==spec[v] for k,v in mapping.items()), 'Relationship is tied to a different GPU cache')
    ri=joined(report['source'],report['spec'],report['GPU_admission'],report['phase'],'0',spec['RAW_inputs'],owners)
    require(ri==38 and spec['RAW_inputs'][ri]['receipt']==spec['relationship_GPU_RAW'] and
        Path(rs['GPU_admission_file'])==Path(spec['relationship_GPU_admission']['path']),
        'Actual unique normal38 relationship owner/admission/RAW required')
    raw=read(bound(spec['relationship_GPU_RAW'])); original=raw['original_identity']
    require(set(original)=={'pid','birth_filetime_100ns','birth_unix'} and type(original['pid']) is int and
        type(original['birth_filetime_100ns']) is str and type(original['birth_unix']) is float,
        'Exact original relationship process identity schema required')
    expected_identity=dict(phase_label=spec['RAW_inputs'][ri]['phase'],process_pid=original['pid'],
        process_birth_time=original['birth_unix'],process_filetime=original['birth_filetime_100ns'],
        admission_sha256=raw['admission_sha256'],actual_exit_code=0,original_handle_wait_succeeded=True,
        process_reaped=True,original_process_birth_absent=True)
    require(record['schema']==ROOT_RELATION_SCHEMA and record['status']==ROOT_RELATION_STATUS and
        all(record[k]==spec[k] for k in RELATION_REF_KEYS if k!='relationship_ROOT_qualification') and
        record['actual_identity']==expected_identity and record['source_all_complete'] is True and
        record['checked']==dict(images=741,instances=1661,canonical100_all_exact_bytes=True,mismatch_images=0,
            Q_written=False,cache_qualified=False) and
        raw['returncode']==0 and raw['held_handle_exit']==dict(wait_state='WAIT_OBJECT_0',actual_exit_code=0) and
        raw['root_wait_reaped'] is True and raw['numeric_birth_absence']['original_birth_absent'] is True,
        'ROOT relation record must exactly join already-closed original held/reaped RAW')
    cached=read(bound(spec['cache_report'])); rows=report['rows']
    require(len(rows)==len(cached['rows'])==741, 'Complete relationship image rows required')
    instances=0
    for index,(rel,row) in enumerate(zip(rows,cached['rows'])):
        G=row['raw_GT']; identity=row['identity']; H,W=identity['original_shape']; canonical=rel['canonical100']
        require(rel['dataset_index']==row['dataset_index']==index and rel['image_id']==row['image_id'] and
            rel['identity']==identity and rel['query_ids']==row['query_ids'] and rel['matched_queries']==G and
            rel['packet']==row['packet']==cached['packet_refs'][index] and
            rel['validation_carrier']==row['validation_carrier']==cached['validation_carrier_refs'][index] and
            rel['canonical100_exact_bytes'] is True and rel['per_query_canonical100_exact_bytes']==[True]*G and
            canonical['exact_bytes'] is True and canonical['per_query_exact_bytes']==[True]*G and
            canonical['per_query_mismatched_pixels']==[0]*G and canonical['mismatched_pixels']==0 and
            canonical['compared_bytes']==4*G*H*W and canonical['max_abs']==0 and
            canonical['saved_shape']==canonical['replay_shape']==[G,H,W] and
            rel['canonical_input_shape']==[1,100,200,200] and
            rel['canonical_input_stride']==[4000000,40000,200,1] and
            rel['unmatched_slots_are_dummy'] is True and rel['original_full100_M_stride_archived'] is False and
            rel['subset_G_is_diagnostic_only'] is True and
            rel['input_tensor_digests']==dict(query_ids=row['tensor_digests']['native']['query_ids'],
                matched_M200=row['carrier_digests']['matched_M200'],matched_native_L=row['tensor_digests']['native']['matched_native_L']),
            'Per-image matched M/carrier/query/L exact canonical relationship differs')
        require(rel['packet'] in closure.values() and rel['validation_carrier'] in closure.values(),
            'Relationship packet/carrier closure missing')
        instances+=G
    require(instances==1661, 'Full relationship matched-query population differs')
    return dict(spec=rs,report=report,ROOT_record=record,normal_index=ri)


def source_gate(path, spec, entry_file, raws, closure, common):
    bind_common(common)
    cpu=True
    count,schema,cuda=39,CPU_SCHEMA,''
    require(spec['schema'] == schema and len(raws) == len(spec['RAW_inputs']) == count and
            all(k in spec for k in REF_KEYS+RELATION_REF_KEYS+('cache_contract_source',)) and
            spec['validation_protocol']==VALIDATION_PROTOCOL and spec['recipe'] == RECIPE and
            spec['regions_recipe'] == REGIONS_RECIPE and spec['numeric_protocol'] == NUMERIC and
            type(spec['deadline_seconds']) is int and spec['deadline_seconds'] > 0 and
            spec['allocator_limit_MiB'] == 4096 and spec['whole_GPU_limit_MiB'] == 8192,
            'Complete locked N13 cache SPEC required')
    extra = ('cache_entry_source','cache_spec','cache_report','cache_GPU_admission','cache_GPU_RAW') if cpu else ()
    required = [spec[k] for k in REF_KEYS + extra + RELATION_REF_KEYS + ('cache_contract_source',)]
    required += [row[k] for row in spec['RAW_inputs'] for k in RAW_KEYS if k != 'phase']
    require(all(ref in closure.values() for ref in required), 'Primary inputs absent from closure')
    source, spec_ref = reference(entry_file), reference(path)
    require(source == spec['entry_source'] and reference(common.__file__) == spec['region_common_source'] and
            reference(__file__)==spec['cache_contract_source'],
            'Actual cache entry/common SOURCE differs')
    owners = [owner_join(row, raw, closure) for row, raw in zip(spec['RAW_inputs'], raws)]
    admission_path = Path(spec['CPU_admission_file' if cpu else 'GPU_admission_file'])
    require(admission_path.is_absolute(), 'Absolute own admission required')
    admission_ref, admission = reference(admission_path), read(admission_path)
    require(admission['status'] == 'ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE' and
            admission['driver_sha256'] == OWNER_SHA and admission['phase'] == spec['phase'] and
            admission['scientific_spec'] == spec_ref and admission['entry'] == source['path'] and
            admission['entry_args'] == ['--spec', str(path), '--owner-execute'] and
            admission['allowed_cpus'] == [0,1] and admission['cuda_visible_devices'] == cuda and
            admission['automatic_retry'] is False and admission['whole_GPU_used_limit_MiB'] == 8192 and
            admission['qualified_leaf_deadline_seconds'] == spec['deadline_seconds'] and
            admission['bound_source_sha256'].get(str(path)) == spec_ref['sha256'] and
            all(admission['bound_source_sha256'].get(ref['path']) == ref['sha256'] for ref in spec['sources']) and
            os.environ.get('CUDA_VISIBLE_DEVICES') == cuda and all(os.environ.get(k) == '2' for k in
            ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')),
            'Actual own complete admission/resources differ')

    context, cq = read(bound(spec['context_GPU_spec'])), read(bound(spec['context_CPU_qualification']))
    require(context['schema'] == 'ROOT-N01-uniform-point-supervision-gradient-probe-spec-v1' and
            cq['schema'] == 'ROOT-N01-fixed-FIT-gradient-CPU-qualification-v1' and
            cq['status'] == 'ROOT_N01_FIXED_FIT_GRADIENT_CPU_QUALIFIED' and
            cq['GPU_spec'] == spec['context_GPU_spec'] and cq['FIT'] == context['FIT'] == spec['FIT'] and
            cq['images_checked'] == 32 and cq['selected_parameter_tensors'] == 8 and
            cq['selected_parameter_scalars'] == 263168 and cq['local_M200_VJP_images_checked'] == 5 and
            cq['optimizer_created'] is False and cq['weights_updated'] is False and
            cq['source'] == context['entry_source'] and cq['renderer_source'] == context['renderer_source'],
            'Actual original frozen E24/FIT context differs')
    for k in ('source','reader_source','spec','GPU_admission','parent_receipt','GPU_phase_binding','GPU_bootstrap','report','start','trace'):
        require(cq[k] in closure.values(), 'Original context primary lineage absent: '+k)
    require(all(ref in closure.values() for ref in context['sources']) and
            all(any(all(current[k] == row[k] for k in RAW_KEYS) for current in spec['RAW_inputs'])
                for row in context['RAW_inputs']) and all(context[k] in closure.values() for k in MODULES) and
            spec['image_root'] == context['image_root'], 'Entire native context/helpers/ancestor closure required')
    ci = joined(cq['source'], cq['GPU_spec'], cq['GPU_admission'],
                spec['RAW_inputs'][next(i for i,r in enumerate(spec['RAW_inputs']) if r['receipt']==cq['parent_receipt'])]['phase'],
                '0', spec['RAW_inputs'], owners)
    require(spec['RAW_inputs'][ci]['receipt'] == cq['parent_receipt'] and
            spec['RAW_inputs'][ci]['phase_binding'] == cq['GPU_phase_binding'] and
            spec['RAW_inputs'][ci]['bootstrap'] == cq['GPU_bootstrap'], 'Original N01 GPU join differs')
    require(any(o['entry']==cq['reader_source']['path'] and o['entry_args']==['--spec',cq['spec']['path'],'--owner-execute']
                for o in owners), 'Original N01 CPU execution absent')

    old = load(spec['original_cache_entry_source'], '_N13_original_N07_cache_ABI')
    old_gpu, old_report, old_q = (read(bound(spec[k])) for k in
        ('original_cache_spec','original_cache_report','original_cache_CPU_qualification'))
    require(old_gpu['schema'] == old.SCHEMA and old_gpu['entry_source'] == spec['original_cache_entry_source'] and
            old_gpu['recipe'] == old.RECIPE and old_gpu['FIT'] == spec['FIT'] and
            old_gpu['context_GPU_spec'] == spec['context_GPU_spec'] and
            old_gpu['context_CPU_qualification'] == spec['context_CPU_qualification'] and
            old_gpu['RAW_inputs'] == spec['RAW_inputs'][:17] and
            all(ref in closure.values() for ref in old_gpu['sources']), 'Complete original N07 cache SPEC differs')
    require(old_q['schema'] == 'ROOT-N07-frozen-E24-original-detail-point-cache-CPU-qualification-v1' and
            old_q['status'] == 'ROOT_N07_FROZEN_E24_ORIGINAL_DETAIL_POINT_CACHE_CPU_QUALIFIED' and
            old_q['source'] == spec['original_cache_entry_source'] and
            old_q['reader_source'] == spec['original_cache_CPU_source'] and old_q['spec'] == spec['original_cache_CPU_spec'] and
            old_q['CPU_admission'] == spec['original_cache_CPU_admission'] and
            old_q['cache_spec'] == spec['original_cache_spec'] and old_q['cache_report'] == spec['original_cache_report'] and
            old_q['cache_GPU_RAW'] == spec['original_cache_GPU_receipt'] and
            old_q['cache_GPU_admission'] == spec['original_cache_GPU_admission'] and
            old_q['RAW_inputs'] == spec['RAW_inputs'][:18] and old_q['FIT'] == spec['FIT'] and
            old_q['recipe'] == old.RECIPE and old_q['images'] == 741 and old_q['raw_GT'] == 1661 and
            old_q['hard_targets_CPU_checked'] is True and old_q['uniform_draws_CPU_replayed'] is True and
            old_q['F_e_L_shape_finiteness_digest_and_duplicate_consistency_CPU_checked'] is True,
            'Actual original FIT cache qualification differs')
    oi = joined(old_report['source'], old_report['spec'], old_report['GPU_admission'], old_report['phase'],
                '0', spec['RAW_inputs'], owners)
    oqi = joined(old_q['reader_source'], old_q['spec'], old_q['CPU_admission'], old_q['phase'], '', spec['RAW_inputs'], owners)
    require((oi,oqi) == (17,18) and spec['RAW_inputs'][oi]['receipt'] == spec['original_cache_GPU_receipt'] and
            spec['RAW_inputs'][oqi]['receipt'] == spec['original_cache_CPU_receipt'] and
            old_report['schema'] == 'N07-frozen-E24-point-cache-report-v1' and old_report['status'] == old.STATUS and
            old_report['source'] == spec['original_cache_entry_source'] and old_report['spec'] == spec['original_cache_spec'] and
            old_report['GPU_admission'] == spec['original_cache_GPU_admission'] and old_report['recipe'] == old.RECIPE and
            old_report['images'] == 741 and old_report['raw_GT'] == 1661 and
            old_report['all_raw_GT_retained'] is True and old_report['cleanup']['allocated_after'] == 0 and
            old_report['optimizer_created'] is False and old_report['weights_updated'] is False and
            old_q['packet_refs'] == old_report['packet_refs'] and len(old_report['packet_refs']) == len(old_report['rows']) == 741 and
            all(ref in closure.values() for ref in old_report['packet_refs']), 'Original cached GT/query packet lineage differs')
    require(spec['matching_loss_seed_base'] == old_gpu['matching_loss_seed_base'] and
            all(type(spec[k]) is int and 0 <= spec[k] < 2**63-741 for k in
                ('matching_loss_seed_base','region_rank_seed_base','uniform_seed_base')) and
            spec['region_rank_seed_base'] != spec['uniform_seed_base'], 'Locked independent seed streams required')

    n12, n12q = read(bound(spec['prior_N12_GPU_report'])), read(bound(spec['prior_N12_CPU_qualification']))
    require(n12q['schema'] == 'ROOT-N12-native-geometry-budget-regions-CPU-qualification-v1' and
            n12q['status'] == 'ROOT_N12_NATIVE_GEOMETRY_BUDGET_REGIONS_CPU_QUALIFIED' and
            n12q['reader_source'] == spec['prior_N12_CPU_source'] and n12q['spec'] == spec['prior_N12_CPU_spec'] and
            n12q['CPU_admission'] == spec['prior_N12_CPU_admission'] and
            n12q['GPU_report'] == spec['prior_N12_GPU_report'] and n12q['source'] == n12['source'] and
            n12q['GPU_spec'] == n12['spec'] and n12q['GPU_admission'] == n12['GPU_admission'] and
            n12q['RAW_inputs'] == spec['RAW_inputs'][:36] and
            n12q['prior_failed_CPU_receipt'] == spec['prior_failed_CPU_receipt'] and
            n12q['additional_failed_RAW_inputs'] == spec['additional_failed_RAW_inputs'][:3] and
            n12q['summaries'] == n12['summaries'] and n12q['H_region_gate'] == n12['H_region_gate'] and
            n12q['H_region_gate']['H_region_diagnostic_supported'] is True and
            n12q['exact_saved_GPU_L_used'] is True and n12q['approximate_CPU_L_or_P_B_K_substituted'] is False and
            n12q['actual_efficacy_established'] is False and n12q['innovation_established'] is False and
            n12['cleanup']['allocated_after'] == 0, 'Qualified pre-registered N12 diagnostic gate required')
    ni = joined(n12['source'],n12['spec'],n12['GPU_admission'],n12['phase'],'0',spec['RAW_inputs'],owners)
    nqi = joined(n12q['reader_source'],n12q['spec'],n12q['CPU_admission'],n12q['phase'],'',spec['RAW_inputs'],owners)
    require((ni,nqi)==(35,36) and spec['RAW_inputs'][ni]['receipt']==n12q['GPU_RAW'] and
            spec['RAW_inputs'][nqi]['receipt']==spec['prior_N12_CPU_receipt'], 'Actual N12 GPU02/CPU02 history differs')
    gate = dict(source=source,spec_ref=spec_ref,admission_ref=admission_ref,admission=admission,
                context=context,closure=closure,owners=owners,old=old,old_gpu=old_gpu,
                old_report=old_report,old_q=old_q,n12q=n12q)
    if cpu:
        gpu, report = read(bound(spec['cache_spec'])), read(bound(spec['cache_report']))
        require(gpu['schema']==GPU_SCHEMA and gpu['entry_source']==spec['cache_entry_source'] and
                gpu['RAW_inputs']==spec['RAW_inputs'][:37] and all(ref in closure.values() for ref in gpu['sources']) and
                all(gpu[k]==spec[k] for k in REF_KEYS if k!='entry_source') and
                all(gpu[k]==spec[k] for k in ('recipe','regions_recipe','numeric_protocol','FIT','image_root',
                    'matching_loss_seed_base','region_rank_seed_base','uniform_seed_base')),
                'Actual N13 GPU SPEC and complete prefix differ')
        gi=joined(spec['cache_entry_source'],spec['cache_spec'],spec['cache_GPU_admission'],gpu['phase'],
                  '0',spec['RAW_inputs'],owners)
        require(gi==37 and spec['RAW_inputs'][gi]['receipt']==spec['cache_GPU_RAW'] and
                Path(gpu['GPU_admission_file'])==Path(spec['cache_GPU_admission']['path']) and
                report['schema']==REPORT_SCHEMA and report['status']==REPORT_STATUS and
                report['source']==spec['cache_entry_source'] and report['spec']==spec['cache_spec'] and
                report['GPU_admission']==spec['cache_GPU_admission'] and report['phase']==gpu['phase'] and
                report['recipe']==RECIPE and report['regions_recipe']==REGIONS_RECIPE and report['numeric_protocol']==NUMERIC and
                report['images']==741 and report['raw_GT']==1661 and report['point_count_per_arm']==1700864 and
                report['teacher_forwards']==741 and report['one_teacher_forward_per_image'] is True and
                report['native610_unchanged_SOURCE_observed'] is True and report['native_gradient_tensors']==0 and
                report['optimizer_created'] is False and report['weights_updated'] is False and report['backward_calls']==0 and
                report['heads_loaded'] is False and report['AP_computed'] is False and report['no_DEV'] is True and
                report['no_TEST'] is True and report['cleanup']['allocated_after']==0 and
                report['teacher_checkpoint']==context['checkpointmap']['native0'], 'Actual full N13 GPU report differs')
        require(len(report['rows'])==len(report['packet_refs'])==741 and
                len(report['validation_carrier_refs'])==741 and
                all(ref in closure.values() for ref in [report['start'],report['trace']]+report['packet_refs']+report['validation_carrier_refs']) and
                Path(spec['cache_report']['path'])==Path(gpu['output_directory'])/'REPORT.json',
                'All cache packet/start/trace outputs must be current CPU inputs')
        failure_prefix_and_pins(gpu,spec,3)
        gate.update(gpu=gpu,report=report)
    gate.update(relationship=relationship_gate(spec,closure,owners))
    return gate


def validate_cache_input(spec, closure, owners):
    """Post-terminal training gate: reuse qualified cache, never claim own terminal."""
    keys=('cache_entry_source','cache_spec','cache_report','cache_GPU_admission','cache_GPU_RAW',
          'cache_CPU_qualification','cache_CPU_source','cache_CPU_spec','cache_CPU_admission','cache_CPU_RAW','FIT',
          'region_common_source','cache_contract_source')+RELATION_REF_KEYS
    require(all(spec[k] in closure.values() for k in keys) and spec['validation_protocol']==VALIDATION_PROTOCOL,
            'Training cache primary refs/evidence protocol absent')
    gpu,report,q=(read(bound(spec[k])) for k in ('cache_spec','cache_report','cache_CPU_qualification'))
    require(gpu['schema']==GPU_SCHEMA and gpu['entry_source']==spec['cache_entry_source'] and
            gpu['FIT']==spec['FIT'] and gpu['recipe']==RECIPE and gpu['regions_recipe']==REGIONS_RECIPE and
            gpu['numeric_protocol']==NUMERIC and gpu['RAW_inputs']==spec['RAW_inputs'][:37] and
            all(ref in closure.values() for ref in gpu['sources']), 'Qualified cache GPU SPEC differs')
    require(report['schema']==REPORT_SCHEMA and report['status']==REPORT_STATUS and
            report['source']==spec['cache_entry_source'] and report['spec']==spec['cache_spec'] and
            report['GPU_admission']==spec['cache_GPU_admission'] and report['recipe']==RECIPE and
            report['regions_recipe']==REGIONS_RECIPE and report['numeric_protocol']==NUMERIC and
            report['images']==741 and report['raw_GT']==1661 and report['point_count_per_arm']==1700864 and
            report['cleanup']['allocated_after']==0 and len(report['packet_refs'])==len(report['rows'])==741 and
            len(report['validation_carrier_refs'])==741 and
            all(ref in closure.values() for ref in report['packet_refs']+report['validation_carrier_refs']), 'Qualified cache report differs')
    require(q['schema']==Q_SCHEMA and q['status']==Q_STATUS and q['source']==spec['cache_entry_source'] and
            q['reader_source']==spec['cache_CPU_source'] and q['spec']==spec['cache_CPU_spec'] and
            q['CPU_admission']==spec['cache_CPU_admission'] and q['GPU_spec']==spec['cache_spec'] and
            q['GPU_report']==spec['cache_report'] and q['GPU_RAW']==spec['cache_GPU_RAW'] and
            q['GPU_admission']==spec['cache_GPU_admission'] and
            q['validation_protocol']==VALIDATION_PROTOCOL and q['cache_contract_source']==spec['cache_contract_source'] and
            q['region_common_source']==spec['region_common_source'] and
            all(q[k]==spec[k] for k in RELATION_REF_KEYS) and
            q['recipe']==RECIPE and q['regions_recipe']==REGIONS_RECIPE and q['numeric_protocol']==NUMERIC and
            q['FIT']==spec['FIT'] and q['packet_refs']==report['packet_refs'] and
            q['validation_carrier_refs']==report['validation_carrier_refs'] and
            q['teacher_checkpoint']==report['teacher_checkpoint'] and
            q['prior_N12_CPU_qualification']==gpu['prior_N12_CPU_qualification'] and
            q['original_cache_CPU_qualification']==gpu['original_cache_CPU_qualification'] and
            all(q[k] is True for k in ('canonical_GPU_M_to_L_exact_bytes_ROOT_qualified','saved_GPU_L_geometry_and_sampling_CPU_exact_replayed','hard_targets_CPU_checked',
                'F_e_L_shape_finiteness_digest_and_duplicate_consistency_CPU_checked','sampled_F_original_two_stage_CPU_replayed')) and
            q['native_L_CPU_full_allclose_is_qualification_gate'] is False and
            type(q['native_L_CPU_full_allclose_passed']) is bool and
            q['current_owned_CPU_terminal_self_certified'] is False and q['CPU_approximate_L_substituted_for_selection'] is False and
            q['CPU_affinity']==[0,1] and q['threads']==2 and q['CUDA_initialized'] is False,
            'Actual independent CPU region cache qualification required')
    gi=joined(report['source'],report['spec'],report['GPU_admission'],report['phase'],'0',spec['RAW_inputs'],owners)
    qi=joined(q['reader_source'],q['spec'],q['CPU_admission'],q['phase'],'',spec['RAW_inputs'],owners)
    cpu=read(bound(spec['cache_CPU_spec']))
    require(gi==37 and qi==len(cpu['RAW_inputs'])==39 and
            spec['RAW_inputs'][gi]['receipt']==spec['cache_GPU_RAW'] and
            spec['RAW_inputs'][qi]['receipt']==spec['cache_CPU_RAW'] and
            cpu['schema']==CPU_SCHEMA and cpu['entry_source']==spec['cache_CPU_source'] and
            cpu['cache_contract_source']==spec['cache_contract_source'] and cpu['region_common_source']==spec['region_common_source'] and
            cpu['RAW_inputs']==q['RAW_inputs']==spec['RAW_inputs'][:qi] and
            cpu['validation_protocol']==VALIDATION_PROTOCOL and
            all(cpu[k]==spec[k] for k in RELATION_REF_KEYS) and
            all(ref in closure.values() for ref in cpu['sources']), 'Actual successful cache GPU/CPU ownership and complete CPU SPEC differ')
    failure_prefix_and_pins(gpu,spec,3)
    failure_prefix_and_pins(cpu,spec,5)
    require(q['prior_failed_CPU_receipt']==cpu['prior_failed_CPU_receipt'] and
        q['additional_failed_RAW_inputs']==cpu['additional_failed_RAW_inputs'] and
        q['terminal_identity_pins']==cpu['terminal_identity_pins'] and
        len(q['per_image_checks'])==741 and
        q['native_L_CPU_full_allclose_passed']==all(row['native_L_CPU_full_allclose_passed'] for row in q['per_image_checks']) and
        q['native_L_CPU_full_allclose_failed_images']==sum(not row['native_L_CPU_full_allclose_passed'] for row in q['per_image_checks']),
        'Cache CPU history/pins and diagnostic aggregate differ')
    require(reference(_COMMON.__file__)==spec['region_common_source'] and
        reference(__file__)==spec['cache_contract_source'], 'Actually loaded original common/new adapter SOURCE differs')
    relationship_gate(spec,closure,owners)
    return gpu,report,q


