"""Independent UNBOUND N13 metadata, packet and region helpers.
Scientific libraries are supplied by the ROOT-owned caller after all terminal
and source gates. Importing this module performs no file access or numerics.
"""
import ast
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import types

OWNER_SHA = 'ef9a9f17d4fb1bc3464936095ef799021dd880f8b15488d558a871c08b6e1924'
GPU_SCHEMA = 'ROOT-N13-sameK-region-FIT-cache-GPU-spec-v1'
CPU_SCHEMA = 'ROOT-N13-sameK-region-FIT-cache-CPU-spec-v1'
PACKET_SCHEMA = 'N13-sameK-region-FIT-cache-packet-v1'
CARRIER_SCHEMA = 'N13-sameK-region-FIT-validation-carrier-v1'
REPORT_SCHEMA = 'N13-sameK-region-FIT-cache-report-v1'
REPORT_STATUS = 'N13_SAMEK_REGION_FIT_CACHE_GPU_COMPLETE_PENDING_CPU'
Q_SCHEMA = 'ROOT-N13-sameK-region-FIT-cache-CPU-qualification-v1'
Q_STATUS = 'ROOT_N13_SAMEK_REGION_FIT_CACHE_CPU_QUALIFIED'
ARMS = ('LOG_PR', 'GEO_PR')
RAW_KEYS = ('receipt', 'phase', 'admission', 'phase_binding', 'bootstrap')
MODULES = ('native_loader_source','evaluation_source','micro_preflight_source',
           'initializer_source','training_source','native_helpers','renderer_source',
           'joint_losses_source','joint_preflight_entry_source')
REF_KEYS = ('entry_source','region_common_source','context_GPU_spec',
 'context_CPU_qualification','FIT','original_cache_entry_source',
 'original_cache_spec','original_cache_report','original_cache_CPU_qualification',
 'original_cache_CPU_source','original_cache_CPU_spec','original_cache_GPU_admission',
 'original_cache_GPU_receipt','original_cache_CPU_admission','original_cache_CPU_receipt',
 'prior_N12_GPU_report','prior_N12_CPU_qualification','prior_N12_CPU_source',
 'prior_N12_CPU_spec','prior_N12_CPU_admission','prior_N12_CPU_receipt')
RECIPE = dict(images=741,raw_GT=1661,native_queries=100,points_per_GT=1024,
 region_points=768,uniform_points=256,arms=['LOG_PR','GEO_PR'],
 replacement=True,duplicates_preserved=True,common_region_rank_draws=True,
 independent_common_uniform_generator=True,K0_shared_uniform1024=True,
 fixed_cache_all_epochs_and_seeds=True,no_augmentation=True,
 band_abs_logit=2.,native_render_queries=100,
 native_render_order='natural all100 then select original matched queries',
 feature_channel_chunk=16,render_dtype='float32',native_weights_epoch=24,
 original_matching_and_GT_order_preserved=True,GT_directed_region_selection=False,
 native_F256_and_matched_M200_saved_for_practical_CPU_replay=True,
 JPEG_patch_used_by_PR=False,heads_loaded=False,optimizer_created=False,
 no_DEV=True,no_TEST=True,no_AP=True)
REGIONS_RECIPE = dict(
 old_B='abs(saved original GPU L)<=2',native_P='saved original GPU L>0',
 K='original exact old_B count per matched query',
 D='minimum in-domain opposing-class pixel-center Chebyshev distance',
 outside_domain_boundary=False,empty_or_full_D='infinity',
 no_opposite_fallback='uncertainty absL then flat',
 geometry_sort=['D','absL','flat'],output_indices='flat ascending',
 distance_not_truncated=True,
 GPU_leaf_distance='scipy.ndimage.distance_transform_cdt chessboard',
 CPU_independent_distance='cv2.distanceTransform DIST_C maskSize3')
NUMERIC = dict(native_L_rtol=1e-4,native_L_atol=1e-4,
 sampled_F_rtol=1e-4,sampled_F_atol=1e-4,
 native_e_and_C_equal_old_cache=True,
 CPU_replay_never_substitutes_saved_GPU_L_P_B_K=True,
 certified_GPU_error_bound=False)

def require(ok, message):
    if not ok: raise RuntimeError(message)


def read(path):
    def reject(value): raise ValueError('Nonfinite JSON constant: ' + value)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), parse_constant=reject)


def reference(path):
    path = Path(path).resolve(strict=True); digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''): digest.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def ref_shape(ref):
    require(type(ref) is dict and set(ref) == {'path', 'bytes', 'sha256'} and
            type(ref['path']) is str and Path(ref['path']).is_absolute() and
            type(ref['bytes']) is int and ref['bytes'] >= 0 and type(ref['sha256']) is str and
            len(ref['sha256']) == 64 and set(ref['sha256']) <= set('0123456789abcdef'), 'Malformed reference')


def bound(ref):
    ref_shape(ref); require(reference(ref['path']) == ref, 'Pinned input changed: ' + ref['path'])
    return Path(ref['path'])


def terminal(ref, phase, exit_code=0):
    ref_shape(ref); path = Path(ref['path'])
    require(type(phase) is str and phase and path.name == 'PARENT_RECEIPT.json' and
            path.parent.name == phase, 'Absolute original receipt/phase locator required')
    raw = read(path)
    require(raw['status'] == 'ACTUAL_OWNED_TERMINAL' and type(raw['returncode']) is int and raw['returncode'] == exit_code and
            type(raw['original_identity']) is dict and raw['original_identity'] and raw['root_wait_reaped'] is True and
            raw['held_handle_exit'] == dict(wait_state='WAIT_OBJECT_0', actual_exit_code=exit_code) and
            type(raw['held_handle_exit']['actual_exit_code']) is int and raw['numeric_birth_absence']['original_birth_absent'] is True and
            raw.get('error') is None and raw.get('abort') is None and raw['wait_errors'] is False and
            raw['GPU_observation_errors'] == [], 'Actual held/reaped terminal%d required' % exit_code)
    if 'phase' in raw: require(type(raw['phase']) is str and raw['phase'] == phase, 'Original RAW phase differs')
    sha, peak = raw['admission_sha256'], raw['whole_GPU_observed_peak_MiB']
    require(type(sha) is str and len(sha) == 64 and set(sha) <= set('0123456789abcdef') and
            type(peak) in (int, float) and math.isfinite(peak) and 0 <= peak <= 8192 and
            type(raw['whole_GPU_used_limit_MiB']) is int and raw['whole_GPU_used_limit_MiB'] == 8192 and
            raw['whole_leaf_VRAM_certified'] is False and raw['GPU_scope'] ==
            'whole GPU observed, not per-process WDDM allocation', 'Original RAW resource scope differs')
    return raw


def owner_join(row, raw, closure):
    owner, pb = read(bound(row['admission'])), read(bound(row['phase_binding']))
    bootstrap = bound(row['bootstrap']); runtime = Path(row['receipt']['path']).parent
    require(Path(row['phase_binding']['path']) == runtime/'PHASE_BINDING.json' and bootstrap == runtime/'bootstrap.py' and
            owner['status'] == 'ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE' and owner['phase'] == row['phase'] and
            owner['driver_sha256'] == OWNER_SHA and owner['automatic_retry'] is False and owner['allowed_cpus'] == [0, 1] and
            owner['cuda_visible_devices'] in ('', '0') and owner['whole_GPU_used_limit_MiB'] == 8192 and
            raw['admission_sha256'] == row['admission']['sha256'] == pb['admission_sha256'] and
            pb['bootstrap_sha256'] == row['bootstrap']['sha256'] and pb['argv'][-1] == str(bootstrap) and
            pb['CPU_affinity'] == owner['allowed_cpus'] and pb['CUDA_visible_devices'] == owner['cuda_visible_devices'] and
            pb['automatic_retry'] is False, 'Original owner/admission/PB/bootstrap differs')
    entry, args = owner['entry'], owner['entry_args']
    require(entry in closure and owner['bound_source_sha256'].get(entry) == closure[entry]['sha256'], 'Original entry absent/unbound')
    tree = ast.parse(bootstrap.read_text(encoding='utf-8'))
    argv = [ast.literal_eval(node.value) for node in ast.walk(tree) if isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == 'sys' and target.attr == 'argv'
        for target in node.targets)]
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and
             isinstance(node.func.value, ast.Name) and node.func.value.id == 'runpy' and node.func.attr == 'run_path']
    require(argv == [[entry]+args] and len(calls) == 1 and ast.literal_eval(calls[0].args[0]) == entry and
            {item.arg: ast.literal_eval(item.value) for item in calls[0].keywords} == {'run_name': '__main__'}, 'Original bootstrap differs')
    if '--spec' in args:
        require(args.count('--spec') == 1, 'Ambiguous original SPEC')
        path = args[args.index('--spec')+1]
        require(path in closure and owner['bound_source_sha256'].get(path) == closure[path]['sha256'], 'Original SPEC absent/unbound')
    if 'scientific_spec' in owner:
        ref = owner['scientific_spec']
        require(ref in closure.values() and owner['bound_source_sha256'].get(ref['path']) == ref['sha256'], 'Scientific SPEC absent/unbound')
    return owner


def load(ref, name):
    path = bound(ref); source = path.read_bytes()
    require(len(source) == ref['bytes'] and hashlib.sha256(source).hexdigest() == ref['sha256'], 'SOURCE bytes changed')
    module = types.ModuleType(name); module.__file__ = str(path)
    exec(compile(source, str(path), 'exec'), module.__dict__)
    return module


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def joined(source, spec_ref, admission, phase, cuda, rows, owners):
    matches = [i for i, owner in enumerate(owners)
               if owner['entry'] == source['path'] and
               owner['entry_args'] == ['--spec', spec_ref['path'], '--owner-execute']]
    require(len(matches) == 1, 'Unique original owned execution required')
    i = matches[0]
    require(rows[i]['phase'] == phase and rows[i]['admission'] == admission and
            owners[i]['scientific_spec'] == spec_ref and
            owners[i]['cuda_visible_devices'] == cuda,
            'Original phase/SPEC/admission/entry join differs')
    return i


def source_gate(path, spec, entry_file, raws, closure, cpu=False):
    count, schema, cuda = (38, CPU_SCHEMA, '') if cpu else (37, GPU_SCHEMA, '0')
    require(spec['schema'] == schema and len(raws) == len(spec['RAW_inputs']) == count and
            all(k in spec for k in REF_KEYS) and spec['recipe'] == RECIPE and
            spec['regions_recipe'] == REGIONS_RECIPE and spec['numeric_protocol'] == NUMERIC and
            type(spec['deadline_seconds']) is int and spec['deadline_seconds'] > 0 and
            spec['allocator_limit_MiB'] == 4096 and spec['whole_GPU_limit_MiB'] == 8192,
            'Complete locked N13 cache SPEC required')
    extra = ('cache_entry_source','cache_spec','cache_report','cache_GPU_admission','cache_GPU_RAW') if cpu else ()
    required = [spec[k] for k in REF_KEYS + extra]
    required += [row[k] for row in spec['RAW_inputs'] for k in RAW_KEYS if k != 'phase']
    require(all(ref in closure.values() for ref in required), 'Primary inputs absent from closure')
    source, spec_ref = reference(entry_file), reference(path)
    require(source == spec['entry_source'] and reference(__file__) == spec['region_common_source'],
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
            n12q['additional_failed_RAW_inputs'] == spec['additional_failed_RAW_inputs'] and
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
        gate.update(gpu=gpu,report=report)
    return gate


def validate_cache_input(spec, closure, owners):
    """Post-terminal training gate: reuse qualified cache, never claim own terminal."""
    keys=('cache_entry_source','cache_spec','cache_report','cache_GPU_admission','cache_GPU_RAW',
          'cache_CPU_qualification','cache_CPU_source','cache_CPU_spec','cache_CPU_admission','cache_CPU_RAW','FIT')
    require(all(spec[k] in closure.values() for k in keys), 'Training cache primary refs absent')
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
            q['GPU_admission']==spec['cache_GPU_admission'] and q['RAW_inputs']==spec['RAW_inputs'][:38] and
            q['recipe']==RECIPE and q['regions_recipe']==REGIONS_RECIPE and q['numeric_protocol']==NUMERIC and
            q['FIT']==spec['FIT'] and q['packet_refs']==report['packet_refs'] and
            q['validation_carrier_refs']==report['validation_carrier_refs'] and
            q['teacher_checkpoint']==report['teacher_checkpoint'] and
            q['prior_N12_CPU_qualification']==gpu['prior_N12_CPU_qualification'] and
            q['original_cache_CPU_qualification']==gpu['original_cache_CPU_qualification'] and
            all(q[k] is True for k in ('full_native_L_geometry_and_sampling_CPU_replayed','hard_targets_CPU_checked',
                'F_e_L_shape_finiteness_digest_and_duplicate_consistency_CPU_checked','sampled_F_original_two_stage_CPU_replayed')) and
            q['current_owned_CPU_terminal_self_certified'] is False and q['CPU_approximate_L_substituted_for_selection'] is False,
            'Actual independent CPU region cache qualification required')
    gi=joined(report['source'],report['spec'],report['GPU_admission'],report['phase'],'0',spec['RAW_inputs'],owners)
    qi=joined(q['reader_source'],q['spec'],q['CPU_admission'],q['phase'],'',spec['RAW_inputs'],owners)
    require((gi,qi)==(37,38) and spec['RAW_inputs'][gi]['receipt']==spec['cache_GPU_RAW'] and
            spec['RAW_inputs'][qi]['receipt']==spec['cache_CPU_RAW'], 'Actual successful cache GPU/CPU ownership differs')
    return gpu,report,q


def digest_tensor(value):
    array=value.detach().cpu().contiguous().numpy()
    return dict(shape=list(array.shape),dtype=str(array.dtype),sha256=hashlib.sha256(array.tobytes()).hexdigest())


def packet_digests(packet):
    top=('query_ids','query_embeddings','native_class_logits','matched_native_L')
    return dict(native={k:digest_tensor(packet[k]) for k in top},
                arms={arm:{k:digest_tensor(packet['arms'][arm][k]) for k in
                      ('F_samples','native_logits','targets','point_indices','point_in_region')} for arm in ARMS})


def packet_validate(torch, packet, row, report):
    G=len(packet['identity']['ann_ids']); H,W=packet['identity']['original_shape']
    require(packet['schema']==PACKET_SCHEMA and packet['identity']==row['identity'] and
            packet['dataset_index']==row['dataset_index'] and packet['source']==report['source'] and
            packet['spec']==report['spec'] and packet['phase']==report['phase'] and
            packet['teacher_checkpoint']==report['teacher_checkpoint'] and 1<=G<=100 and row['raw_GT']==G and
            tuple(packet['query_ids'].shape)==(G,) and packet['query_ids'].dtype==torch.int64 and
            len(set(packet['query_ids'].tolist()))==G and packet['query_ids'].tolist()==row['query_ids'] and
            all(0<=q<100 for q in row['query_ids']), 'Original packet/query/GT identity differs')
    require(packet['validation_carrier']==row['validation_carrier'] and
            packet['validation_carrier'] in report['validation_carrier_refs'], 'Validation carrier reference differs')
    shapes={'query_embeddings':(G,256),'native_class_logits':(1,100,2),'matched_native_L':(G,H,W)}
    for k,shape in shapes.items():
        t=packet[k]
        require(isinstance(t,torch.Tensor) and t.device.type=='cpu' and t.dtype==torch.float32 and
                tuple(t.shape)==shape and not t.requires_grad and bool(torch.isfinite(t).all()), 'Finite native packet field differs: '+k)
    require(set(packet['arms'])==set(ARMS) and len(packet['K'])==len(packet['native_P_rles'])==G and
            set(packet['region_rles'])==set(ARMS), 'Two fixed-region packet populations differ')
    for arm in ARMS:
        fields=packet['arms'][arm]
        for k,shape in (('F_samples',(G,1024,256)),('native_logits',(G,1024)),('targets',(G,1024)),
                        ('point_indices',(G,1024)),('point_in_region',(G,1024))):
            t=fields[k]; dtype=torch.int64 if k=='point_indices' else (torch.bool if k in ('targets','point_in_region') else torch.float32)
            require(isinstance(t,torch.Tensor) and t.device.type=='cpu' and t.dtype==dtype and tuple(t.shape)==shape and
                    not t.requires_grad and (dtype!=torch.float32 or bool(torch.isfinite(t).all())), 'Finite arm packet field differs: '+k)
        require(bool(((fields['point_indices']>=0)&(fields['point_indices']<H*W)).all()), 'Point index outside original image')
    require(packet_digests(packet)==row['tensor_digests'], 'Native/arm cache tensor digests differ')
    return G


def load_packet(torch, report, index):
    row,ref=report['rows'][index],report['packet_refs'][index]
    require(row['packet']==ref and row['dataset_index']==index, 'Canonical packet reference/order differs')
    packet=torch.load(str(bound(ref)),map_location='cpu'); packet_validate(torch,packet,row,report)
    return packet


def instance_inputs(torch, packet, g, arm, device):
    require(arm in ARMS and 0<=g<len(packet['query_ids']), 'Original arm/instance required')
    fields=packet['arms'][arm]
    return (fields['F_samples'][g].to(device), packet['query_embeddings'][g].to(device).expand(1024,256),
            fields['native_logits'][g].to(device),fields['targets'][g].to(device,dtype=torch.float32),
            fields['point_in_region'][g].to(device))


def region_logits(torch, head, F, e, L, regionflag):
    require(regionflag.dtype==torch.bool and regionflag.device==L.device and tuple(regionflag.shape)==tuple(L.shape),
            'Explicit actual-region bool flags required')
    result=head(F,e,L,L.new_zeros((L.numel(),49)),arm='PR')
    residual=result['residual']; require(tuple(residual.shape)==tuple(L.shape) and bool(torch.isfinite(residual).all()),
                                       'Original PR ungated residual required')
    logits=torch.where(regionflag,L.detach()+residual,L.detach())
    require(bool(torch.isfinite(logits).all()), 'Finite region-gated logits required')
    return dict(logits=logits,residual=residual,band=regionflag,editable=regionflag)


def rle(np, masks, binary):
    encoded=masks.encode(np.asfortranarray(binary.astype(np.uint8)))
    counts=encoded['counts']; return dict(size=list(encoded['size']),counts=counts.decode('ascii') if isinstance(counts,bytes) else counts)


def rle_decode(np, masks, value):
    require(type(value) is dict and set(value)=={'size','counts'} and type(value['counts']) is str, 'Canonical RLE required')
    return masks.decode(dict(size=value['size'],counts=value['counts'].encode('ascii'))).astype(np.bool_)


def gpu_regions(np, ndimage, L):
    require(L.ndim==2 and L.dtype==np.float32 and np.isfinite(L).all(), 'Exact finite original GPU L raster required')
    P=L>0; B=np.abs(L)<=np.float32(2); K=int(B.sum()); flat=np.arange(L.size,dtype=np.int64)
    absL=np.abs(L).reshape(-1); no_opposite=not P.any() or P.all()
    if no_opposite:
        D=np.full(P.shape,np.inf,dtype=np.float64)
    else:
        inside=ndimage.distance_transform_cdt(P,metric='chessboard')
        outside=ndimage.distance_transform_cdt(~P,metric='chessboard')
        D=np.where(P,inside,outside)
        require(bool((D>=1).all()), 'Finite in-domain opposite distances required')
    priority=np.lexsort((flat,absL,D.reshape(-1)))
    selected=np.sort(priority[:K]); G=np.zeros(L.size,dtype=np.bool_); G[selected]=True; G=G.reshape(L.shape)
    uncertain=np.zeros(L.size,dtype=np.bool_); uncertain[np.lexsort((flat,absL))[:K]]=True
    require(np.array_equal(uncertain.reshape(L.shape),B) and int(G.sum())==K, 'Exact K and uncertainty/oldB identity required')
    return P,{'LOG_PR':B,'GEO_PR':G},K,no_opposite


def gpu_sample_points(torch, np, region_rows, Ks, rank_seed, uniform_seed):
    rank_rng=torch.Generator(device='cpu'); rank_rng.manual_seed(rank_seed)
    uniform_rng=torch.Generator(device='cpu'); uniform_rng.manual_seed(uniform_seed)
    rows={arm:[] for arm in ARMS}
    for g,K in enumerate(Ks):
        area=region_rows['LOG_PR'][g].size
        if K:
            ranks=torch.randint(K,(768,),generator=rank_rng)
            uniform=torch.randint(area,(256,),generator=uniform_rng)
            for arm in ARMS:
                candidates=torch.from_numpy(np.flatnonzero(region_rows[arm][g]).astype(np.int64))
                require(len(candidates)==K, 'Same exact K sampling populations required')
                rows[arm].append(torch.cat((candidates[ranks],uniform)))
        else:
            uniform=torch.randint(area,(1024,),generator=uniform_rng)
            for arm in ARMS: rows[arm].append(uniform.clone())
    return {arm:torch.stack(rows[arm]).contiguous() for arm in ARMS}


def imported_files(torch, np, cv2, masks, COCO, closure, admission, ndimage=None):
    prefixes=('torch','numpy','cv2','pycocotools','psutil','mmcv','mmengine','mmdet')
    if ndimage is not None: prefixes=prefixes+('scipy',)
    files={}
    for name,module in list(sys.modules.items()):
        if module is None or not any(name==p or name.startswith(p+'.') for p in prefixes): continue
        path=getattr(module,'__file__',None)
        if path is None or not Path(path).is_file(): continue
        ref=reference(path)
        require(closure.get(ref['path'])==ref and admission['bound_source_sha256'].get(ref['path'])==ref['sha256'],
                'Actual numerical library/binary SOURCE unbound: '+ref['path'])
        files[name]=ref
    for name,module in (('torch_C',torch._C),('numpy_binary',np.core._multiarray_umath),('cv2_native',cv2._native),('mask_binary',masks._mask)):
        ref=reference(module.__file__)
        require(closure.get(ref['path'])==ref and admission['bound_source_sha256'].get(ref['path'])==ref['sha256'],
                'Actual numerical binary unbound: '+name)
        files[name]=ref
    return files
