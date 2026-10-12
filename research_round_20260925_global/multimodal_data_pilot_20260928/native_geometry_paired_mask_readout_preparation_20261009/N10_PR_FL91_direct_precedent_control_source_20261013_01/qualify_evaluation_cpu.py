"""Unbound SOURCE for independent ROOT-owned N10 fixed8 DEV evaluation.

All32 normal original RAWs and three preserved failed1 RAWs must be terminal
before the first SHA, SOURCE import or numerical operation. The GPU native
all100/full18700 reproduction is an exact prerequisite. Four trained C/PR
heads are independently reconstructed with functional CPU arithmetic at every
original-native-band pixel, using the same bound renderer and archived F/e/M.
Practical near-zero/band checks are not CUDA equality or a certified error
bound. Five official COCO12 arrays and matching pixels use the unchanged GPU
exports; CPU approximate masks never replace those fixed exported predictions.
No teacher, full optimizer trajectory, GT-directed inference or TEST is run.
ROOT separately qualifies this leaf's actual held process terminal after exit.
"""
import argparse
import gc
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time
import types


SCHEMA = 'ROOT-N10-C-PR-FL91-fixed8-evaluation-CPU-spec-v1'
Q_SCHEMA = 'ROOT-N10-C-PR-FL91-fixed8-evaluation-CPU-qualification-v1'
STATUS = 'ROOT_N10_C_PR_FL91_FIXED8_EVALUATION_CPU_QUALIFIED'
GPU_SCHEMA = 'ROOT-N10-C-PR-FL91-fixed8-fullDEV-GPU-spec-v1'
GPU_REPORT_SCHEMA = 'N10-C-PR-FL91-fixed8-fullDEV-GPU-report-v1'
GPU_STATUS = 'N10_C_PR_FL91_FIXED8_FULLDEV_GPU_COMPLETE_PENDING_CPU'
OWNER_SHA = 'ef9a9f17d4fb1bc3464936095ef799021dd880f8b15488d558a871c08b6e1924'
SEEDS = (202610120701,202610120702)
ARMS = ('C','PR')
KEYS = tuple('%s_seed%d' % (arm,seed) for seed in SEEDS for arm in ARMS)
CONTROLS = ('native_official',)+tuple(key+'_fixed_native_score' for key in KEYS)
N07_REPORT_SCHEMA = 'N07-original-detail-fixed-endpoint-evaluation-report-v1'
N07_STATUS = 'N07_ORIGINAL_DETAIL_FIXED_ENDPOINT_EVALUATION_COMPLETE_PENDING_CPU'
N07_Q_SCHEMA = 'ROOT-N07-original-detail-fixed-endpoint-evaluation-CPU-qualification-v1'
N07_Q_STATUS = 'ROOT_N07_ORIGINAL_DETAIL_FIXED_ENDPOINT_EVALUATION_CPU_QUALIFIED'
TRAIN_SCHEMA = 'ROOT-N10-C-PR-FL91-point-training-spec-v1'
TRAIN_REPORT_SCHEMA = 'N10-C-PR-FL91-point-training-report-v1'
TRAIN_STATUS = 'N10_C_PR_FL91_POINT_TRAINING_COMPLETE_PENDING_ROOT_QUALIFICATION'
TRAIN_Q_SCHEMA = 'ROOT-N10-C-PR-FL91-fixed8-training-CPU-qualification-v1'
TRAIN_Q_STATUS = 'ROOT_N10_C_PR_FL91_FIXED8_TRAINING_CPU_QUALIFIED'
PACKET_SCHEMA = 'N07-original-detail-all100-native-fields-v1'
CP_SCHEMA = 'N10-C-PR-FL91-point-head-checkpoint-v1'
RAW_KEYS = ('receipt','phase','admission','phase_binding','bootstrap')
THREAD_ENV = ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')
REF_KEYS = ('entry_source','evaluation_entry_source','evaluation_spec','evaluation_report',
    'evaluation_GPU_RAW','evaluation_GPU_admission','reader_source','renderer_source',
    'prior_N07_report','prior_N07_cpu_qualification','native_reference_predictions',
    'DEV','training_CPU_reader_source')
REQUIRED_KEYS = ('schema','phase','sources','RAW_inputs','prior_failed_CPU_receipt',
    'additional_failed_RAW_inputs','CPU_admission_file','deadline_seconds','Q_path',
    'numeric_protocol')+REF_KEYS
NUMERIC = dict(class_rtol=1e-6,class_atol=1e-6,quality_rtol=1e-4,quality_atol=1e-4,
    CPU_sign_and_band_rtol=1e-4,CPU_sign_and_band_atol=1e-4,COCO_stats_atol=1e-12,
    renderer_query_chunk=4,official_quality_eps=1e-6,certified_GPU_error_bound=False,
    head_pixel_chunk=2048,feature_channel_chunk=16,native_CPU_render_batch=100)


def require(ok,message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    def reject(value):
        raise ValueError('Nonfinite JSON constant: '+value)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'),parse_constant=reject)


def ref_shape(ref):
    require(type(ref) is dict and set(ref) == {'path','bytes','sha256'} and
        type(ref['path']) is str and Path(ref['path']).is_absolute() and
        type(ref['bytes']) is int and ref['bytes'] >= 0 and type(ref['sha256']) is str and
        len(ref['sha256']) == 64 and set(ref['sha256']) <= set('0123456789abcdef'),
        'Malformed absolute byte/SHA reference')


def reference(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):
            digest.update(block)
    return dict(path=str(path),bytes=path.stat().st_size,sha256=digest.hexdigest())


def bound(ref):
    ref_shape(ref)
    require(reference(ref['path']) == ref,'Actual pinned input changed: '+ref['path'])
    return Path(ref['path'])


def deadline(started,seconds):
    require(time.monotonic()-started < seconds,
        'Prelocked full187/all100 CPU deadline exceeded; no sampling, retry or fallback')


def terminal(ref,phase,code=0):
    """Original RAW JSON only, before every SHA or numerical import."""
    ref_shape(ref)
    path = Path(ref['path'])
    require(path.name == 'PARENT_RECEIPT.json' and type(phase) is str and phase and
        path.parent.name == phase,'Original absolute RAW/phase locator required')
    raw = read(path)
    require(raw['status'] == 'ACTUAL_OWNED_TERMINAL' and type(raw['returncode']) is int and
        raw['returncode'] == code and type(raw['original_identity']) is dict and raw['original_identity'] and
        raw['root_wait_reaped'] is True and
        raw['held_handle_exit'] == dict(wait_state='WAIT_OBJECT_0',actual_exit_code=code) and
        type(raw['held_handle_exit']['actual_exit_code']) is int and
        raw['numeric_birth_absence']['original_birth_absent'] is True and
        raw.get('error') is None and raw.get('abort') is None and raw['wait_errors'] is False and
        raw['GPU_observation_errors'] == [],'Original held/reaped terminal%d required'%code)
    if 'phase' in raw:
        require(type(raw['phase']) is str and raw['phase'] == phase,'Original RAW phase differs')
    sha,peak = raw['admission_sha256'],raw['whole_GPU_observed_peak_MiB']
    require(type(sha) is str and len(sha) == 64 and set(sha) <= set('0123456789abcdef') and
        type(peak) in (int,float) and math.isfinite(peak) and 0 <= peak <= 8192 and
        type(raw['whole_GPU_used_limit_MiB']) is int and raw['whole_GPU_used_limit_MiB'] == 8192 and
        raw['whole_leaf_VRAM_certified'] is False and
        raw['GPU_scope'] == 'whole GPU observed, not per-process WDDM allocation',
        'Original RAW owner/resource scope differs')
    return raw


def load(ref,name):
    path = bound(ref)
    source = path.read_bytes()
    require(path.suffix == '.py' and len(source) == ref['bytes'] and
        hashlib.sha256(source).hexdigest() == ref['sha256'],'Bound Python SOURCE bytes changed')
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(source,str(path),'exec'),module.__dict__)
    return module


def phase_index(source,spec_ref,admission,phase,cuda,rows,owners):
    matches = [index for index,owner in enumerate(owners) if owner['entry'] == source['path'] and
        owner['entry_args'] == ['--spec',spec_ref['path'],'--owner-execute']]
    require(len(matches) == 1,'Exactly one actual successful parent phase required')
    index = matches[0]
    require(rows[index]['phase'] == phase and rows[index]['admission'] == admission and
        owners[index]['scientific_spec'] == spec_ref and owners[index]['cuda_visible_devices'] == cuda,
        'Actual owned parent primary identity differs')
    return index


def stats_close(actual,expected,message):
    require(type(actual) is list and type(expected) is list and len(actual) == len(expected) == 12 and
        all(type(value) in (int,float) and math.isfinite(value) and (value == -1 or 0 <= value <= 1)
            for value in actual+expected) and
        all(abs(left-right) <= NUMERIC['COCO_stats_atol'] for left,right in zip(actual,expected)),message)


def resources(psutil,torch):
    require(psutil.Process().cpu_affinity() == [0,1] and
        torch.get_num_threads() == torch.get_num_interop_threads() == 2 and
        os.environ.get('CUDA_VISIBLE_DEVICES') == '' and
        all(os.environ.get(key) == '2' for key in THREAD_ENV) and not torch.cuda.is_initialized(),
        'Actual CPU01/threads2/cold CUDA differs')


def imported_files(psutil,torch,np,COCO,COCOeval,masks,gate):
    modules = dict(psutil=psutil,psutil_platform=psutil._psplatform,psutil_binary=psutil._psplatform.cext,
        torch=torch,torch_binary=torch._C,torch_tensor=sys.modules['torch._tensor'],
        torch_autograd=sys.modules['torch.autograd'],torch_random=torch.random,numpy=np,
        numpy_binary=np.core._multiarray_umath,COCO=sys.modules[COCO.__module__],
        COCOeval=sys.modules[COCOeval.__module__],masks=masks,mask_binary=masks._mask)
    result = {name:reference(module.__file__) for name,module in modules.items()}
    result['interpolation'] = reference(inspect.getsourcefile(torch.nn.functional.interpolate))
    result['serialization'] = reference(inspect.getsourcefile(torch.load))
    require(all(ref in gate['closure'].values() and
        gate['owner']['bound_source_sha256'].get(ref['path']) == ref['sha256']
        for ref in result.values()),'Actual imported CPU library/binary/math absent or unbound')
    return result


def prior_lineage(spec,closure,rows,owners):
    prior = read(bound(spec['prior_N07_report']))
    q = read(bound(spec['prior_N07_cpu_qualification']))
    require(prior['schema'] == N07_REPORT_SCHEMA and prior['status'] == N07_STATUS and
        q['schema'] == N07_Q_SCHEMA and q['status'] == N07_Q_STATUS and
        q['evaluation_report'] == spec['prior_N07_report'] and q['source'] == prior['source'] and
        q['evaluation_spec'] == prior['spec'] and q['reader_source'] == spec['reader_source'] and
        prior['DEV'] == q['DEV'] == spec['DEV'] and prior['images'] == q['images'] == 187 and
        prior['raw_GT'] == q['raw_GT'] == 380 and
        prior['predictions']['native_official'] == spec['native_reference_predictions'] ==
            q['control_results']['native_official']['predictions'] and
        q['complete_identity_fixed_scores_and_empty_query_population_CPU_checked'] is True and
        q['five_official_COCO12_CPU_recomputed'] is True and
        q['strong_E24_native_export_and_COCO_reference_CPU_checked'] is True and
        q['native_M200_original_render_CPU_replayed'] is True and
        q['packet_refs'] == prior['final_field_packets'] and len(q['packet_refs']) == 187,
        'Actual archived N07 native identity/fields/independent CPU qualification differs')
    require(prior['source'] in closure.values() and prior['spec'] in closure.values() and
        q['spec'] in closure.values() and all(ref in closure.values() for ref in q['packet_refs']) and
        all(prior[key] in closure.values() for key in ('image_rows','trace','native_query_score_factors')),
        'Complete referenced N07 packet/native identity inputs absent')
    prior_index = phase_index(prior['source'],prior['spec'],prior['GPU_admission'],prior['phase'],
        '0',rows,owners)
    q_index = phase_index(q['reader_source'],q['spec'],q['CPU_admission'],q['phase'],'',rows,owners)
    require(prior_index < q_index < 31 and rows[prior_index]['receipt'] == q['evaluation_GPU_RAW'] and
        q['evaluation_GPU_admission'] == prior['GPU_admission'],
        'Actual archived N07 GPU/independent CPU original RAW chronology differs')
    return prior,q


def training_lineage(evaluation,spec,closure,rows,owners,provenance):
    training = read(bound(evaluation['head_training_spec']))
    head_report = read(bound(evaluation['head_training_report']))
    tq = read(bound(evaluation['head_training_CPU_qualification']))
    require(training['schema'] == TRAIN_SCHEMA and training['entry_source'] == evaluation['head_training_entry_source'] and
        training['RAW_inputs'] == rows[:29] and
        training['prior_failed_CPU_receipt'] == spec['prior_failed_CPU_receipt'] and
        training['additional_failed_RAW_inputs'] == spec['additional_failed_RAW_inputs'] and
        training['point_heads_source'] == evaluation['point_heads_source'] and
        training['PR_point_heads_source'] == evaluation['PR_point_heads_source'] and
        all(ref in closure.values() for ref in training['sources']),
        'Exact original N10 fixed8 training/head closure differs')
    require(head_report['schema'] == TRAIN_REPORT_SCHEMA and head_report['status'] == TRAIN_STATUS and
        head_report['source'] == evaluation['head_training_entry_source'] and
        head_report['spec'] == evaluation['head_training_spec'] and
        head_report['GPU_admission'] == evaluation['head_training_GPU_admission'] and
        head_report['phase'] == evaluation['head_training_phase'] and
        head_report['completed_head_epochs'] == 8 and head_report['continued_all4_to24'] is False and
        head_report['all4_fixed_same_epochs'] is True and head_report['paired_order'] is True and
        head_report['paired_seed'] is True and head_report['paired_cold_native_identity'] is True and
        head_report['paired_order_and_initialization'] is False and
        head_report['point_heads_source'] == evaluation['point_heads_source'] and
        head_report['PR_point_heads_source'] == evaluation['PR_point_heads_source'] and
        head_report['head_CPU_qualification'] == training['head_CPU_qualification'] and
        head_report['recipe'] == training['recipe'] and
        head_report['no_DEV'] is True and head_report['no_TEST'] is True and head_report['AP_computed'] is False and
        set(head_report['checkpoints']) == set(KEYS) and
        all(head_report['checkpoints'][key]['checkpoint'] == evaluation['head_checkpoints'][key] and
            head_report['checkpoints'][key]['completed_head_epochs'] == 8 and
            head_report['checkpoints'][key]['updates'] == 13288 for key in KEYS),
        'Actual fixed8 four-head training report differs')
    require(provenance.Q_SCHEMA == TRAIN_Q_SCHEMA and provenance.STATUS == TRAIN_Q_STATUS and
        tq['schema'] == TRAIN_Q_SCHEMA and tq['status'] == TRAIN_Q_STATUS and
        tq['source'] == evaluation['head_training_entry_source'] and
        tq['reader_source'] == spec['training_CPU_reader_source'] and
        tq['reader_source'] == evaluation['head_training_CPU_entry_source'] and
        tq['spec'] == evaluation['head_training_CPU_spec'] and
        tq['CPU_admission'] == evaluation['head_training_CPU_admission'] and
        tq['training_spec'] == evaluation['head_training_spec'] and
        tq['training_report'] == evaluation['head_training_report'] and
        tq['training_GPU_RAW'] == evaluation['head_training_receipt'] and
        tq['training_GPU_admission'] == evaluation['head_training_GPU_admission'] and
        tq['point_heads_source'] == evaluation['point_heads_source'] and
        tq['PR_point_heads_source'] == evaluation['PR_point_heads_source'] and
        tq['head_CPU_qualification'] == training['head_CPU_qualification'] and
        tq['recipe'] == training['recipe'] and tq['parameter_tensors_by_arm'] == dict(C=6,PR=8) and
        tq['RAW_inputs'] == rows[:30] and tq['prior_failed_CPU_receipt'] == spec['prior_failed_CPU_receipt'] and
        tq['additional_failed_RAW_inputs'] == spec['additional_failed_RAW_inputs'] and
        tq['checkpoints'] == head_report['checkpoints'] and tq['completed_head_epochs'] == 8 and
        tq['updates_per_model'] == 13288 and tq['C_parameters'] == 40257 and tq['PR_parameters'] == 40497 and
        tq['per_arm_cold_initializer_proofs_CPU_checked'] is True and
        tq['private_image_order_and_endpoint_RNG_CPU_reconstructed'] is True and
        tq['AdamW_endpoint_states_and_stepcounts_CPU_checked'] is True and
        tq['all16_epoch_records_CPU_checked'] is True and
        tq['fixed8_no_extension_decision_CPU_checked'] is True and
        tq['final_fullFIT_endpoint_BCE_CPU_recomputed'] is True and
        tq['C_PR_identical_weight_initialization_claimed'] is False and
        tq['CPU_affinity'] == [0,1] and tq['threads'] == 2 and tq['CUDA_initialized'] is False,
        'Actual independent fixed8 training CPU qualification differs')
    for index,receipt,phase,admission,source,spec_ref,cuda in (
        (29,evaluation['head_training_receipt'],evaluation['head_training_phase'],
            evaluation['head_training_GPU_admission'],evaluation['head_training_entry_source'],
            evaluation['head_training_spec'],'0'),
        (30,evaluation['head_training_CPU_receipt'],tq['phase'],tq['CPU_admission'],tq['reader_source'],tq['spec'],'')):
        require(spec_ref in closure.values() and source in closure.values() and admission in closure.values() and
            phase_index(source,spec_ref,admission,phase,cuda,rows,owners) == index and
            rows[index]['receipt'] == receipt,'Actual N10 training GPU/CPU held0 ancestry differs')
    ts = read(bound(tq['spec']))
    require(ts['schema'] == provenance.SCHEMA and ts['entry_source'] == tq['reader_source'] and
        ts['RAW_inputs'] == rows[:30] and ts['training_entry_source'] == tq['source'] and
        ts['training_spec'] == tq['training_spec'] and ts['training_report'] == tq['training_report'] and
        ts['training_GPU_RAW'] == tq['training_GPU_RAW'] and ts['training_GPU_admission'] == tq['training_GPU_admission'] and
        ts['point_heads_source'] == evaluation['point_heads_source'] and
        ts['PR_point_heads_source'] == evaluation['PR_point_heads_source'] and
        ts['prior_failed_CPU_receipt'] == spec['prior_failed_CPU_receipt'] and
        ts['additional_failed_RAW_inputs'] == spec['additional_failed_RAW_inputs'] and
        tq['phase'] == ts['phase'] == evaluation['head_training_CPU_phase'] and
        all(ref in closure.values() for ref in ts['sources']),
        'Complete actual training CPU SPEC/source/input closure differs')
    return training,head_report,tq


def source_gate(path,spec,started):
    require(all(key in spec for key in REQUIRED_KEYS) and spec['schema'] == SCHEMA and
        type(spec['RAW_inputs']) is list and len(spec['RAW_inputs']) == 32 and
        all(type(row) is dict and set(RAW_KEYS) <= set(row) for row in spec['RAW_inputs']),
        'Complete32-normal-RAW ROOT-authored CPU SPEC required')
    rows = spec['RAW_inputs']
    for row in rows:
        ref_shape(row['receipt'])
    normal_paths = {row['receipt']['path'] for row in rows}
    require(len(normal_paths) == 32,'Thirty-two distinct actual normal RAWs required')
    raws = [terminal(row['receipt'],row['phase']) for row in rows]
    failed_refs = [spec['prior_failed_CPU_receipt']]
    ref_shape(failed_refs[0])
    terminal(failed_refs[0],Path(failed_refs[0]['path']).parent.name,1)
    extra = spec['additional_failed_RAW_inputs']
    require(type(extra) is list and len(extra) == 2,'Two additional preserved failed1 RAW rows required')
    for row in extra:
        require(type(row) is dict and set(row) == {'receipt','phase','returncode'} and
            type(row['returncode']) is int and row['returncode'] == 1,'Explicit historical failed1 row required')
        terminal(row['receipt'],row['phase'],1)
        failed_refs.append(row['receipt'])
    require(len({ref['path'] for ref in failed_refs}) == 3 and
        not normal_paths.intersection(ref['path'] for ref in failed_refs),
        'Failed1 cannot qualify a normal ancestor')
    # ALL32 normal plus failed3 terminal reads precede first SHA/import/payload.
    require(type(spec['sources']) is list and spec['sources'],'Complete CPU input closure required')
    for ref in spec['sources']:
        ref_shape(ref)
    closure = {ref['path']:ref for ref in spec['sources']}
    require(len(closure) == len(spec['sources']),'Unique complete CPU input closure required')
    required = [spec[key] for key in REF_KEYS]+failed_refs+[row[key]
        for row in rows for key in RAW_KEYS if key != 'phase']
    require(all(ref in closure.values() for ref in required),'All actual primary CPU refs absent from sources')
    require(type(spec['deadline_seconds']) is int and spec['deadline_seconds'] > 0 and
        os.environ.get('CUDA_VISIBLE_DEVICES') == '' and
        all(os.environ.get(key) == '2' for key in THREAD_ENV),'Prelocked CPU environment/deadline differs')
    for ref in spec['sources']:
        deadline(started,spec['deadline_seconds'])
        bound(ref)
    evaluator = load(spec['evaluation_entry_source'],'_N10_actual_GPU_evaluator_ABI')
    evaluation = read(bound(spec['evaluation_spec']))
    require(evaluator.SCHEMA == GPU_SCHEMA and evaluator.REPORT_SCHEMA == GPU_REPORT_SCHEMA and
        evaluator.STATUS == GPU_STATUS and evaluator.KEYS == KEYS and evaluator.CONTROLS == CONTROLS and
        evaluation['schema'] == GPU_SCHEMA and evaluation['entry_source'] == spec['evaluation_entry_source'] and
        evaluation['recipe'] == evaluator.RECIPE and evaluation['RAW_inputs'] == rows[:31] and
        evaluation['prior_failed_CPU_receipt'] == spec['prior_failed_CPU_receipt'] and
        evaluation['additional_failed_RAW_inputs'] == extra and
        evaluation['prior_N07_report'] == spec['prior_N07_report'] and
        evaluation['N07_cpu_qualification'] == spec['prior_N07_cpu_qualification'] and
        evaluation['native_predictions'] == spec['native_reference_predictions'] and
        evaluation['DEV'] == spec['DEV'] and evaluation['reader_source'] == spec['reader_source'] and
        evaluation['renderer_source'] == spec['renderer_source'] and
        evaluation['head_training_CPU_entry_source'] == spec['training_CPU_reader_source'] and
        evaluation['head_endpoint_epoch'] == 8 and set(evaluation['head_checkpoints']) == set(KEYS) and
        all(ref in closure.values() for ref in evaluation['sources']),
        'Actual fixed8 full187/all100 GPU evaluation recipe/parents/closure differs')
    cache = load(evaluation['cache_entry_source'],'_N10_CPU_original_owned_RAW_join')
    require(cache.OWNER_SHA == OWNER_SHA,'Original owner helper differs')
    owners = [cache.owner_join(row,raw,closure) for row,raw in zip(rows,raws)]
    prior,prior_q = prior_lineage(spec,closure,rows,owners)
    provenance = load(spec['training_CPU_reader_source'],'_N10_independent_CPU_C_PR_formula')
    training,head_report,training_q = training_lineage(evaluation,spec,closure,rows,owners,provenance)
    require(phase_index(spec['evaluation_entry_source'],spec['evaluation_spec'],spec['evaluation_GPU_admission'],
        evaluation['phase'],'0',rows,owners) == 31 and rows[31]['receipt'] == spec['evaluation_GPU_RAW'] and
        owners[31]['qualified_leaf_deadline_seconds'] == evaluation['deadline_seconds'] and
        all(owners[31]['bound_source_sha256'].get(ref['path']) == ref['sha256'] for ref in evaluation['sources']),
        'Actual held0 N10 GPU evaluation owner/full closure differs')
    source,spec_ref = reference(__file__),reference(path)
    require(source == spec['entry_source'],'Actual CPU entry SOURCE differs')
    owner_path = Path(spec['CPU_admission_file'])
    require(owner_path.is_absolute(),'Absolute own CPU admission required')
    owner_ref = reference(owner_path)
    owner = read(owner_path)
    require(owner['status'] == 'ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE' and
        owner['driver_sha256'] == OWNER_SHA and owner['phase'] == spec['phase'] and
        owner['entry'] == source['path'] and owner['entry_args'] == ['--spec',str(path),'--owner-execute'] and
        owner['scientific_spec'] == spec_ref and owner['allowed_cpus'] == [0,1] and
        owner['cuda_visible_devices'] == '' and owner['automatic_retry'] is False and
        owner['whole_GPU_used_limit_MiB'] == 8192 and
        owner['qualified_leaf_deadline_seconds'] == spec['deadline_seconds'] and
        owner['bound_source_sha256'].get(str(path)) == spec_ref['sha256'] and
        all(owner['bound_source_sha256'].get(ref['path']) == ref['sha256'] for ref in spec['sources']),
        'Actual current CPU owner/full own input closure differs')
    report = read(bound(spec['evaluation_report']))
    require(report['schema'] == GPU_REPORT_SCHEMA and report['status'] == GPU_STATUS and
        report['source'] == spec['evaluation_entry_source'] and report['spec'] == spec['evaluation_spec'] and
        report['GPU_admission'] == spec['evaluation_GPU_admission'] and report['phase'] == evaluation['phase'] and
        report['RAW_inputs'] == evaluation['RAW_inputs'] and
        report['prior_failed_CPU_receipt'] == spec['prior_failed_CPU_receipt'] and
        report['additional_failed_RAW_inputs'] == extra and report['recipe'] == evaluator.RECIPE and
        report['head_training_entry_source'] == evaluation['head_training_entry_source'] and
        report['head_training_spec'] == evaluation['head_training_spec'] and
        report['head_training_report'] == evaluation['head_training_report'] and
        report['head_training_CPU_qualification'] == evaluation['head_training_CPU_qualification'] and
        report['head_checkpoints'] == evaluation['head_checkpoints'] and report['head_endpoint_epoch'] == 8 and
        report['point_heads_source'] == evaluation['point_heads_source'] and
        report['PR_point_heads_source'] == evaluation['PR_point_heads_source'] and
        report['teacher_checkpoint'] == head_report['teacher_checkpoint'] == prior['teacher_checkpoint'] and
        report['prior_N07_report'] == spec['prior_N07_report'] and
        report['N07_cpu_qualification'] == spec['prior_N07_cpu_qualification'] and
        report['native_reference_predictions'] == spec['native_reference_predictions'] and
        report['reader_source'] == spec['reader_source'] and report['renderer_source'] == spec['renderer_source'] and
        report['DEV'] == spec['DEV'] and report['packet_refs'] == prior['final_field_packets'] and
        set(report['predictions']) == set(report['control_results']) == set(CONTROLS) and
        report['images'] == 187 and report['raw_GT'] == 380 and report['predictions_per_control'] == 18700 and
        report['native0_export_exact_reference'] is True and report['native0_COCO_exact_reference'] is True and
        report['native0_exact_queries'] == 18700 and report['native_M_equals_eF_exact'] is True and
        report['all100_identity_and_fixed_scores_exact'] is True and report['native_model_loaded'] is False and
        report['native_forward_calls'] == report['teacher_forwards'] == report['backward_calls'] == 0 and
        report['original_JPEG_decoded'] is False and report['optimizer_created'] is False and
        report['weights_updated'] is False and report['training_or_matching_loss_executed'] is False and
        report['no_GT_query_selection'] is True and report['no_NMS'] is True and report['no_score_cutoff'] is True and
        report['no_TEST'] is True and report['best_DEV_selection'] is False and
        report['cleanup']['allocated_after'] == 0,'Actual full187/all100 exact-native GPU report differs')
    require(report['native0_predictions'] == report['predictions']['native_official'] and
        report['native0_COCO'] == report['control_results']['native_official']['COCO'],
        'Actual native report aliases differ')
    outputs = [report[key] for key in ('start','trace','image_diagnostics')]
    outputs += list(report['predictions'].values())+list(report['packet_refs'])
    outputs += [head_report[key] for key in ('start','trace','decision')]
    outputs += [row[key] for row in head_report['checkpoints'].values() for key in ('checkpoint','proof')]
    require(all(ref in closure.values() for ref in outputs),'Complete GPU/head numerical outputs absent')
    reader_spec = read(bound(prior_q['spec']))
    require(reader_spec['renderer_source'] == spec['renderer_source'] and spec['numeric_protocol'] == NUMERIC,
        'Original renderer or prelocked practical CPU numeric protocol differs')
    target = Path(spec['Q_path'])
    details = target.with_name(target.stem+'_DETAILS.json')
    require(target.is_absolute() and target.parent.is_dir() and details != target and
        all(not item.exists() and not item.with_suffix(item.suffix+'.tmp').exists() for item in (target,details)),
        'Fresh absolute CPU Q/details paths required')
    return dict(closure=closure,owner=owner,owner_ref=owner_ref,source=source,spec_ref=spec_ref,
        evaluator=evaluator,evaluation=evaluation,report=report,prior=prior,prior_q=prior_q,
        provenance=provenance,training=training,head_report=head_report,training_q=training_q,
        packets=prior['final_field_packets'],details=details)


def packet_check(torch,reader,coco,value,row,index,gate):
    prior = gate['prior']
    identity = row['identity']
    iid = row['image_id']
    image = coco.imgs[iid]
    require(value['schema'] == PACKET_SCHEMA and value['source'] == prior['source'] and
        value['spec'] == prior['spec'] and value['phase'] == prior['phase'] and
        value['dataset_index'] == row['dataset_index'] == index and value['identity'] == identity and
        value['teacher_checkpoint'] == prior['teacher_checkpoint'] and
        value['head_checkpoints'] == prior['head_checkpoints'] and
        value['head_endpoint_epoch'] == prior['head_endpoint_epoch'] and
        value['GPU_original_logits_saved'] is False and
        value['prediction_query_ids'] == row['prediction_query_ids'] and
        value['field_digests'] == row['final_fields'],'Original N07 archived packet provenance differs')
    H,W = image['height'],image['width']
    h,w = identity['actual_img_shape']
    require(identity['image_id'] == iid and identity['original_shape'] == [H,W] and
        type(h) is int and type(w) is int and 0 < h <= 800 and 0 < w <= 800 and
        identity['batch_input_shape'] == [800,800] and
        identity['ann_ids'] == row['ann_ids'] == [ann['id'] for ann in coco.imgToAnns[iid]] and
        row['raw_GT'] == len(identity['ann_ids']),'Original full-image/all-GT metadata differs')
    qids = value['prediction_query_ids']
    require(type(qids) is list and len(qids) == 100 and all(type(query) is int for query in qids) and
        sorted(qids) == list(range(100)),'All100 original query IDs required')
    fields = value['native_fields']
    shapes = dict(F256=(1,256,200,200),e0=(1,100,256),M200=(1,100,200,200),final_class_logits=(1,100,2))
    require(set(fields) == set(shapes) == set(value['field_digests']),'Exactly four archived native fields required')
    for name,shape in shapes.items():
        tensor = fields[name]
        require(isinstance(tensor,torch.Tensor) and tensor.device.type == 'cpu' and
            tensor.dtype == torch.float32 and tuple(tensor.shape) == shape and not tensor.requires_grad and
            bool(torch.isfinite(tensor).all()) and reader.digest(tensor) == value['field_digests'][name],
            'Finite archived CPU native field/digest differs: '+name)
    require(dict(reader.digest(fields['final_class_logits'][0]),values=fields['final_class_logits'][0].tolist()) ==
        row['native_class_packet'],'Original archived class packet differs')
    return fields,identity,qids


def original_feature(torch,renderer,F,identity,started,seconds):
    H,W = identity['original_shape']
    result = torch.empty((256,H,W),dtype=torch.float32,device='cpu')
    for first in range(0,256,NUMERIC['feature_channel_chunk']):
        deadline(started,seconds)
        values = renderer.render_original(torch,F[:,first:first+NUMERIC['feature_channel_chunk']],identity)
        require(values.device.type == 'cpu' and values.dtype == torch.float32 and
            tuple(values.shape) == (NUMERIC['feature_channel_chunk'],H,W) and bool(torch.isfinite(values).all()),
            'Same bound renderer original F projection differs')
        result[first:first+NUMERIC['feature_channel_chunk']] = values
    return result


def head_state_check(torch,provenance,gate,start):
    evaluation,report = gate['evaluation'],gate['head_report']
    states,checks = {},{}
    tq = gate['training_q']
    for key in KEYS:
        arm,seed_text = key.split('_seed',1)
        ref = evaluation['head_checkpoints'][key]
        row = report['checkpoints'][key]
        cp = torch.load(str(bound(ref)),map_location='cpu')
        meta,state = cp['meta'],cp['state_dict']
        proof = read(bound(row['proof']))
        provenance.finite_state(torch,state,arm)
        count = 6 if arm == 'C' else 8
        scalars = 40257 if arm == 'C' else 40497
        factory_ref = evaluation['point_heads_source'] if arm == 'C' else evaluation['PR_point_heads_source']
        require(row['checkpoint'] == ref and meta['schema'] == CP_SCHEMA and
            meta['source'] == evaluation['head_training_entry_source'] and
            meta['spec'] == evaluation['head_training_spec'] and
            meta['GPU_admission'] == evaluation['head_training_GPU_admission'] and
            meta['phase'] == evaluation['head_training_phase'] and meta['arm'] == arm and
            meta['seed'] == int(seed_text) and meta['head_factory_source'] == factory_ref and
            meta['point_heads_source'] == evaluation['point_heads_source'] and
            meta['PR_point_heads_source'] == evaluation['PR_point_heads_source'] and
            meta['head_CPU_qualification'] == report['head_CPU_qualification'] and
            meta['completed_head_epochs'] == 8 and meta['updates'] == 13288 and
            meta['parameter_tensors'] == count and meta['parameter_scalars'] == scalars and
            meta['native_weights_updated'] is False and meta['no_pending_gradients'] is True,
            'Exact fixed8 C/PR head checkpoint metadata differs')
        values = {name:provenance.digest(value) for name,value in state.items()}
        require(proof['checkpoint'] == ref and proof['state'] == values and
            start['head_state_proofs'][key] == values and proof['updates'] == 13288 and
            proof['parameter_tensors'] == count and proof['parameter_scalars'] == scalars and
            proof['full_readback_exact'] is True and tq['checkpoints'][key] == row and
            tq['checkpoint_results'][key]['checkpoint'] == ref,
            'Actual head state/START/independent training CPU proof differs')
        states[key] = state
        checks[key] = dict(checkpoint=ref,proof=row['proof'],epoch=8,updates=13288,
            parameter_tensors=count,parameter_scalars=scalars,head_factory_source=factory_ref,
            CPU_finite_state_and_proofs_checked=True,prior_training_CPU_qualification_reused=True,
            optimizer_trajectory_CPU_replayed=False)
        del cp
    return states,checks


def dense_query(torch,provenance,state,F_original,e,z,arm,started,seconds):
    """All native-band pixels, with independent arithmetic, no head.forward."""
    refined = z.clone()
    indices = torch.nonzero(z.flatten().abs() <= 2,as_tuple=False).flatten()
    calls = 0
    for first in range(0,len(indices),NUMERIC['head_pixel_chunk']):
        deadline(started,seconds)
        idx = indices[first:first+NUMERIC['head_pixel_chunk']]
        sampled_F = F_original.flatten(1)[:,idx].transpose(0,1)
        sampled_e = e.expand(len(idx),256)
        sampled_L = z.flatten()[idx]
        patch = torch.zeros((len(idx),49),dtype=torch.float32,device='cpu')
        logits = provenance.independent_logits(torch,state,sampled_F,sampled_e,sampled_L,patch,arm)
        calls += 1
        require(logits.device.type == 'cpu' and logits.dtype == torch.float32 and
            tuple(logits.shape) == (len(idx),) and bool(torch.isfinite(logits).all()),
            'Independent full-band CPU C/PR formula ABI differs')
        refined.flatten()[idx] = logits
    require(torch.equal(refined[z.abs() > 2],z[z.abs() > 2]),'CPU dense refinement changed outside native band')
    return refined,int(len(indices)),calls


def practical_support(torch,np,masks,reader,exported,z,native_z=None):
    actual = masks.decode(reader.rle_bytes(exported['segmentation'])).astype(bool)
    require(actual.shape == tuple(z.shape),'Exported original-raster support shape differs')
    support = torch.from_numpy(np.ascontiguousarray(actual))
    tau = NUMERIC['CPU_sign_and_band_atol']+NUMERIC['CPU_sign_and_band_rtol']*z.abs()
    allowance = z.abs() <= tau
    if native_z is not None:
        native_tau = NUMERIC['CPU_sign_and_band_atol']+NUMERIC['CPU_sign_and_band_rtol']*native_z.abs()
        allowance = allowance | ((native_z.abs()-2).abs() <= native_tau)
    flips = support != (z > 0)
    require(not bool((flips & ~allowance).any()),
        'CPU/GPU support discrepancy outside prelocked practical zero/band allowance')
    return support,dict(CPU_GPU_support_flips=int(flips.sum()),
        practical_allowance_pixels=int(allowance.sum()),foreground_pixels=int(actual.sum()))


def replay_fields(torch,np,masks,reader,renderer,coco,gate,populations,factors,states,started,spec):
    prior,report = gate['prior'],gate['report']
    rows = read(bound(prior['image_rows']))
    diagnostics = read(bound(report['image_diagnostics']))
    def reject(value):
        raise ValueError('Nonfinite TRACE constant: '+value)
    trace = [json.loads(line,parse_constant=reject) for line in bound(report['trace']).read_text(
        encoding='utf-8').splitlines() if line.strip()]
    require(rows and len(rows) == len(diagnostics) == len(trace) == 187 and trace == diagnostics and
        [row['image_id'] for row in rows] == list(coco.imgs) and
        sum(row['raw_GT'] for row in rows) == 380,'Complete archived187/new TRACE diagnostic identity differs')
    total_names = ('added_pixels','removed_pixels','changed_pixels','expanded_queries','shrunk_queries',
        'changed_queries','native_empty_queries','refined_empty_queries',
        'native_empty_new_nonempty_keeps_zero_score','new_empty_keeps_positive_native_score')
    totals = {control:{name:0 for name in total_names} for control in CONTROLS[1:]}
    checks,cpu_calls = [],{control:0 for control in CONTROLS[1:]}
    with torch.no_grad():
        for index,(row,ref,diagnostic) in enumerate(zip(rows,gate['packets'],diagnostics)):
            deadline(started,spec['deadline_seconds'])
            require(row['packet'] == ref and diagnostic['dataset_index'] == index and
                diagnostic['image_id'] == row['image_id'] and diagnostic['packet'] == ref and
                diagnostic['queries'] == 100 and diagnostic['native_query_order'] == row['prediction_query_ids'] and
                diagnostic['native_M_equals_eF_exact'] is True and
                diagnostic['native_masks_scores_and_classes_exact'] is True and
                diagnostic['fixed_native_scores_and_class_and_order_exact'] is True and
                diagnostic['original_band_only'] is True and
                diagnostic['band_counts_natural_query_order'] == row['band_counts_all100'] and
                diagnostic['native_forward_calls'] == 0 and diagnostic['native_model_loaded'] is False and
                diagnostic['original_JPEG_decoded'] is False and diagnostic['GT_used_to_choose_queries_or_residual'] is False and
                set(diagnostic['controls']) == set(CONTROLS[1:]),'Actual exact-native full100 GPU per-image protocol differs')
            packet = torch.load(str(bound(ref)),map_location='cpu')
            fields,identity,qids = packet_check(torch,reader,coco,packet,row,index,gate)
            H,W = identity['original_shape']
            native_L = renderer.render_original(torch,fields['M200'],identity)
            require(native_L.device.type == 'cpu' and native_L.dtype == torch.float32 and
                tuple(native_L.shape) == (100,H,W) and bool(torch.isfinite(native_L).all()),
                'Same bound CPU natural all100 original renderer ABI differs')
            F_original = original_feature(torch,renderer,fields['F256'],identity,started,spec['deadline_seconds'])
            probabilities = fields['final_class_logits'][0].softmax(-1)[:,0]
            iid = row['image_id']
            image_queries = {control:[] for control in CONTROLS[1:]}
            native_checks = []
            for ordinal,qid in enumerate(qids):
                deadline(started,spec['deadline_seconds'])
                z = native_L[qid]
                before = populations['native_official'][iid][ordinal]
                require(before['query_id'] == qid,'Natural query/logit/original export ordinal alignment differs')
                native_support,native_stats = practical_support(torch,np,masks,reader,before,z)
                tau = NUMERIC['CPU_sign_and_band_atol']+NUMERIC['CPU_sign_and_band_rtol']*z.abs()
                near_band = (z.abs()-2).abs() <= tau
                native_band = z.abs() <= 2
                gpu_band_count = row['band_counts_all100'][qid]
                require(type(gpu_band_count) is int and 0 <= gpu_band_count <= H*W and
                    abs(int(native_band.sum())-gpu_band_count) <= int(near_band.sum()),
                    'Native CPU/GPU band count differs outside practical boundary allowance')
                factor = factors[iid,qid]
                probability = float(probabilities[qid])
                quality = float((z.sigmoid()*native_support).sum()/(native_support.sum()+1e-6))
                require(math.isclose(probability,factor['class_probability'],rel_tol=NUMERIC['class_rtol'],
                    abs_tol=NUMERIC['class_atol']) and math.isclose(quality,factor['mask_quality'],
                    rel_tol=NUMERIC['quality_rtol'],abs_tol=NUMERIC['quality_atol']),
                    'CPU native class/quality on exact exported GPU support differs')
                native_empty = not bool(native_support.any())
                if native_empty:
                    require(before['score'] == factor['mask_quality'] == 0.,'Native empty support must keep exact original zero S0')
                native_checks.append(dict(query_id=qid,native_export_ordinal=ordinal,
                    native_band_pixels_CPU=int(native_band.sum()),native_band_pixels_GPU=gpu_band_count,
                    practical_near_band_pixels=int(near_band.sum()),
                    absolute_class_probability_difference=abs(probability-factor['class_probability']),
                    absolute_mask_quality_difference=abs(quality-factor['mask_quality']),**native_stats))
                for key in KEYS:
                    arm = key.split('_seed',1)[0]
                    control = key+'_fixed_native_score'
                    refined,count,calls = dense_query(torch,gate['provenance'],states[key],F_original,
                        fields['e0'][0,qid],z,arm,started,spec['deadline_seconds'])
                    require(count == int(native_band.sum()),'All CPU native-band points must be evaluated for every head')
                    cpu_calls[control] += calls
                    after = populations[control][iid][ordinal]
                    require(after['query_id'] == qid,'Refined original query order differs')
                    support,support_stats = practical_support(torch,np,masks,reader,after,refined,z)
                    changed = support != native_support
                    require(not bool((changed & (z.abs() > 2+tau)).any()),
                        'Actual exported residual geometry changed outside practical original band')
                    added = int((~native_support & support).sum())
                    removed = int((native_support & ~support).sum())
                    empty = not bool(support.any())
                    observed = dict(query_id=qid,native_export_ordinal=ordinal,fixed_native_score=before['score'],
                        band_pixels=gpu_band_count,native_foreground_pixels=int(native_support.sum()),
                        refined_foreground_pixels=int(support.sum()),added_pixels=added,removed_pixels=removed,
                        changed_pixels=added+removed,native_empty=native_empty,refined_empty=empty,
                        native_empty_new_nonempty_keeps_zero_score=native_empty and not empty,
                        new_empty_keeps_positive_native_score=not native_empty and empty and before['score'] > 0)
                    require(diagnostic['controls'][control]['query_diagnostics'][ordinal] == observed,
                        'Actual per-query refined pixel/empty/fixed-score accounting differs')
                    image_queries[control].append(dict(observed,CPU_native_band_pixels=count,
                        CPU_independent_point_chunk_calls=calls,**support_stats))
                    del refined,support,changed
            image_summaries = {}
            for key in KEYS:
                control = key+'_fixed_native_score'
                values = image_queries[control]
                counts = {name:sum(value[name] for value in values) for name in (
                    'added_pixels','removed_pixels','changed_pixels','native_empty_new_nonempty_keeps_zero_score',
                    'new_empty_keeps_positive_native_score')}
                counts.update(changed_queries=sum(value['changed_pixels'] > 0 for value in values),
                    expanded_queries=sum(value['added_pixels'] > 0 for value in values),
                    shrunk_queries=sum(value['removed_pixels'] > 0 for value in values),
                    native_empty_queries=sum(value['native_empty'] for value in values),
                    refined_empty_queries=sum(value['refined_empty'] for value in values))
                gpu = diagnostic['controls'][control]
                require(gpu['arm'] == key.split('_seed',1)[0] and gpu['seed'] == int(key.split('_seed',1)[1]) and
                    all(gpu[name] == counts[name] for name in total_names) and
                    gpu['head_point_chunk_calls'] == sum((count+2047)//2048 for count in row['band_counts_all100']),
                    'Actual per-image four-head full-band/chunk/pixel accounting differs')
                for name in total_names:
                    totals[control][name] += counts[name]
                image_summaries[control] = dict(counts,CPU_GPU_support_flips=sum(
                    value['CPU_GPU_support_flips'] for value in values),CPU_independent_point_chunk_calls=sum(
                    value['CPU_independent_point_chunk_calls'] for value in values))
            checks.append(dict(dataset_index=index,image_id=iid,packet=ref,queries=100,
                native_CPU_render_batch=100,F256_original_renderer_CPU_replayed=True,
                all_four_independent_dense_head_formulas_CPU_replayed=True,native_queries=native_checks,
                refined_controls=image_summaries,refined_query_checks=image_queries))
            del native_L,F_original,probabilities,packet,fields
            gc.collect()
            print(json.dumps(dict(N10_CPU_dense_head_replayed_images=index+1,
                complete_all100_queries=100*(index+1))),flush=True)
    require(totals == report['change_totals'],'Full187/all100 independent expansion/shrink/empty totals differ')
    return checks,totals,cpu_calls


def fixed_native_pairs(np,masks,reader,coco,native_matching,populations):
    """Keep each native matched query fixed; preserve FN/ignored population."""
    summaries,details = {},{}
    for control in CONTROLS[1:]:
        control_summaries,control_details = [],[]
        for threshold in (.50,.75,.90):
            records = []
            for image in native_matching:
                cells = [cell for cell in image['thresholds'] if abs(cell['threshold']-threshold) <= 1e-12]
                require(len(cells) == 1,'Actual native matching threshold required')
                lookup = {row['query_id']:row for row in populations[control][image['image_id']]}
                for row in cells[0]['GT']:
                    pair = image['TP_pairs'][row['TP_pair_key']] if row['TP_pair_key'] else None
                    if pair is None:
                        records.append(dict(image_id=image['image_id'],gt_id=row['gt_id'],
                            status='ignored_GT' if row['ignored'] else 'native_FN',query_id=None,
                            native=None,refined=None,refined_minus_native=None))
                        continue
                    gt = coco.anns[row['gt_id']]
                    a = masks.decode(reader.rle_bytes(coco.annToRLE(gt))).astype(bool)
                    b = masks.decode(reader.rle_bytes(lookup[pair['query_id']]['segmentation'])).astype(bool)
                    require(a.shape == b.shape,'Fixed native query/GT raster shape differs')
                    tp,fp,fn = int(np.count_nonzero(a & b)),int(np.count_nonzero(~a & b)),int(np.count_nonzero(a & ~b))
                    union,total = tp+fp+fn,2*tp+fp+fn
                    refined = dict(IoU=tp/union if union else 0.,Dice=2*tp/total if total else 0.,
                        pixel_TP=tp,pixel_FP=fp,pixel_FN=fn)
                    native = {name:pair[name] for name in refined}
                    records.append(dict(image_id=image['image_id'],gt_id=row['gt_id'],status='native_matched_fixed_query',
                        query_id=pair['query_id'],native=native,refined=refined,
                        refined_minus_native={name:refined[name]-native[name] for name in refined}))
            require(len(records) == 380,'Fixed native pairing must retain all380 GT, including FN/ignore')
            selected = [row for row in records if row['refined_minus_native'] is not None]
            deltas = [row['refined_minus_native'] for row in selected]
            control_summaries.append(dict(threshold=threshold,GT=380,
                native_matched_pairs=len(selected),native_FN=sum(row['status'] == 'native_FN' for row in records),
                ignored_GT=sum(row['status'] == 'ignored_GT' for row in records),
                conditional_mean_IoU_delta=math.fsum(row['IoU'] for row in deltas)/len(deltas) if deltas else None,
                conditional_mean_Dice_delta=math.fsum(row['Dice'] for row in deltas)/len(deltas) if deltas else None,
                conditional_pixel_deltas={name:sum(row[name] for row in deltas) for name in ('pixel_TP','pixel_FP','pixel_FN')},
                fixed_native_query_and_assignment=True,not_global_pixel_risk_or_causal_decomposition=True))
            control_details.append(dict(threshold=threshold,GT_rows=records))
        summaries[control],details[control] = control_summaries,control_details
    return summaries,details


def atomic_write(path,value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    require(path.parent.is_dir() and not path.exists() and not temporary.exists(),'Fresh output file required')
    with temporary.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.write('\n')
    temporary.replace(path)


def qualify(spec_path):
    started = time.monotonic()
    path = Path(spec_path)
    require(path.is_absolute(),'Absolute ROOT-authored CPU SPEC required')
    path = path.resolve(strict=True)
    spec = read(path)
    gate = source_gate(path,spec,started)
    import psutil
    require(psutil.Process().cpu_affinity() == [0,1],'CPU01 required before numerical imports')
    import torch
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from pycocotools import mask as masks
    require(torch.__version__ == '2.0.1+cu118' and np.__version__ == '1.23.5' and
        not torch.cuda.is_initialized(),'Original Torch/NumPy and cold CUDA required')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    resources(psutil,torch)
    libraries = imported_files(psutil,torch,np,COCO,COCOeval,masks,gate)
    rng = torch.random.get_rng_state().clone()
    reader = load(spec['reader_source'],'_N10_original_independent_COCO_matching_helpers')
    renderer = load(spec['renderer_source'],'_N10_actual_bound_original_CPU_renderer')
    require(reader.Q_SCHEMA == N07_Q_SCHEMA and reader.STATUS == N07_Q_STATUS and
        all(reader.NUMERIC[key] == NUMERIC[key] for key in reader.NUMERIC),
        'Original independent matching/helper practical numeric ABI differs')
    coco = COCO(str(bound(spec['DEV'])))
    ids = list(coco.imgs)
    require(len(ids) == len(set(ids)) == 187 and ids == [row['id'] for row in coco.dataset['images']] and
        len(coco.anns) == 380 and set(coco.cats) == {1},'Complete canonical DEV187/380/category1 required')
    report,evaluation = gate['report'],gate['evaluation']
    predictions = {control:read(bound(report['predictions'][control])) for control in CONTROLS}
    populations = {control:reader.population(coco,rows,ids) for control,rows in predictions.items()}
    native = predictions['native_official']
    archived = read(bound(spec['native_reference_predictions']))
    require(native == archived,'All18700 native masks/class/order/S0 must equal archived reference exactly')
    for control in CONTROLS[1:]:
        require(all({key:value for key,value in left.items() if key != 'segmentation'} ==
            {key:value for key,value in right.items() if key != 'segmentation'}
            for left,right in zip(native,predictions[control])),
            'Every original query/class/order/S0, including empty refined geometry, must remain exact')
    start = read(bound(report['start']))
    require(start['schema'] == 'N10-C-PR-FL91-fixed8-fullDEV-GPU-start-v1' and
        start['source'] == report['source'] and start['spec'] == report['spec'] and
        start['GPU_admission'] == report['GPU_admission'] and start['phase'] == report['phase'] and
        start['recipe'] == report['recipe'] and start['head_training_report'] == report['head_training_report'] and
        start['head_training_CPU_qualification'] == report['head_training_CPU_qualification'] and
        start['head_checkpoints'] == report['head_checkpoints'] and start['head_state_proofs'] == report['head_state_proofs'] and
        start['point_heads_source'] == report['point_heads_source'] and
        start['PR_point_heads_source'] == report['PR_point_heads_source'] and
        start['prior_N07_report'] == spec['prior_N07_report'] and
        start['N07_cpu_qualification'] == spec['prior_N07_cpu_qualification'] and
        start['native_predictions'] == spec['native_reference_predictions'] and start['archived_packet_refs'] == gate['packets'] and
        start['native_model_loaded'] is False and start['native_forward_calls'] == 0 and
        start['original_JPEG_decoded'] is False and start['paired_seed_and_image_order'] is True and
        start['C_PR_identical_weight_initialization_claimed'] is False and start['no_optimizer'] is True and
        start['no_weight_update'] is True and start['no_TEST'] is True and start['best_DEV_selection'] is False,
        'Actual fixed8 fullDEV GPU START identity/scope differs')
    states,head_checks = head_state_check(torch,gate['provenance'],gate,start)
    factors,factor_proof = reader.factor_check(np,read(bound(gate['prior']['native_query_score_factors'])),native,ids)
    packet_checks,totals,cpu_calls = replay_fields(torch,np,masks,reader,renderer,coco,gate,
        populations,factors,states,started,spec)
    controls,matching = {},{}
    for control in CONTROLS:
        deadline(started,spec['deadline_seconds'])
        evaluator,metrics = reader.official_metrics(coco,predictions[control],ids,COCOeval)
        expected = report['control_results'][control]
        require(expected['predictions'] == report['predictions'][control] and expected['stats'] == expected['COCO']['stats'] and
            all(metrics[key] == expected['COCO'][key] for key in ('iouType','maxDets','original_GT_preserved',
                'score_filter','official_COCO_crowd_ignore_semantics')),'Actual original official COCO conventions differ')
        stats_close(metrics['stats'],expected['stats'],'Independent official COCO12 differs: '+control)
        matching[control] = reader.matching_images(np,masks,evaluator,ids)
        controls[control] = dict(predictions=report['predictions'][control],COCO=metrics,
            one_to_one=reader.matching_summary(matching[control]),all100_identity_and_fixed_scores_CPU_checked=True)
        del evaluator
        gc.collect()
        print(json.dumps(dict(N10_CPU_official_COCO_control=control)),flush=True)
    stats_close(controls['native_official']['COCO']['stats'],gate['prior_q']['expected_E24_COCO_stats'],
        'Independent strong E24 native COCO12 anchor differs')
    contrasts = [('native_official',control) for control in CONTROLS[1:]]+[
        ('C_seed%d_fixed_native_score'%seed,'PR_seed%d_fixed_native_score'%seed) for seed in SEEDS]
    paired,paired_details = {},{}
    for left,right in contrasts:
        label = right+'_minus_'+left
        paired[label],paired_details[label] = reader.paired_differences(matching[left],matching[right])
    fixed_summary,fixed_details = fixed_native_pairs(np,masks,reader,coco,matching['native_official'],populations)
    deadline(started,spec['deadline_seconds'])
    atomic_write(gate['details'],dict(schema='N10-C-PR-FL91-fixed8-independent-CPU-dense-and-matching-details-v1',
        source=gate['source'],spec=gate['spec_ref'],evaluation_report=spec['evaluation_report'],
        controls_actual_COCO_matching=matching,paired_reassignment_differences=paired_details,
        fixed_native_query_GT_pairs=fixed_details,full187_all100_CPU_formula_checks=packet_checks,
        fixed_native_pairs_separate_from_official_reassignment=True,
        no_GT_directed_inference=True,no_CPU_mask_substitution_for_exported_COCO=True))
    require(torch.equal(rng,torch.random.get_rng_state()) and not torch.cuda.is_initialized() and
        'mmdet' not in sys.modules,'CPU replay changed RNG, initialized CUDA or loaded native model')
    for ref in spec['sources']:
        deadline(started,spec['deadline_seconds'])
        bound(ref)
    require(reference(path) == gate['spec_ref'] and reference(__file__) == gate['source'] and
        reference(spec['CPU_admission_file']) == gate['owner_ref'] and
        imported_files(psutil,torch,np,COCO,COCOeval,masks,gate) == libraries,
        'Final SPEC/SOURCE/admission/library closure changed')
    resources(psutil,torch)
    deadline(started,spec['deadline_seconds'])
    q = dict(schema=Q_SCHEMA,status=STATUS,phase=spec['phase'],source=spec['evaluation_entry_source'],
        reader_source=gate['source'],spec=gate['spec_ref'],CPU_admission=gate['owner_ref'],
        evaluation_spec=spec['evaluation_spec'],evaluation_report=spec['evaluation_report'],
        evaluation_GPU_RAW=spec['evaluation_GPU_RAW'],evaluation_GPU_admission=spec['evaluation_GPU_admission'],
        evaluation_phase=evaluation['phase'],RAW_inputs=spec['RAW_inputs'],
        prior_failed_CPU_receipt=spec['prior_failed_CPU_receipt'],additional_failed_RAW_inputs=spec['additional_failed_RAW_inputs'],
        head_training_report=report['head_training_report'],head_training_CPU_qualification=report['head_training_CPU_qualification'],
        head_checkpoints=report['head_checkpoints'],head_endpoint_epoch=8,point_heads_source=report['point_heads_source'],
        PR_point_heads_source=report['PR_point_heads_source'],teacher_checkpoint=report['teacher_checkpoint'],
        prior_N07_report=spec['prior_N07_report'],N07_cpu_qualification=spec['prior_N07_cpu_qualification'],
        native_reference_predictions=spec['native_reference_predictions'],DEV=spec['DEV'],
        reader_helpers_source=spec['reader_source'],renderer_source=spec['renderer_source'],
        independent_head_formula_source=spec['training_CPU_reader_source'],
        images=187,raw_GT=380,predictions_per_control=18700,controls=list(CONTROLS),control_results=controls,
        packet_refs=gate['packets'],head_endpoint_state_checks=head_checks,native_factor_check=factor_proof,
        change_totals=totals,CPU_independent_point_chunk_calls=cpu_calls,
        paired_one_to_one_differences=paired,fixed_native_query_GT_pairs=fixed_summary,
        matching_details=reference(gate['details']),numeric_protocol=NUMERIC,
        native_full18700_exact_GPU_export_prerequisite=True,native_natural_all100_GPU_quality_exact_prerequisite=True,
        native_M_equals_eF_exact_GPU_prerequisite=True,
        complete_identity_fixed_scores_and_empty_query_population_CPU_checked=True,
        five_official_COCO12_GPU_exports_CPU_recomputed=True,
        strong_E24_native_export_and_COCO_reference_CPU_checked=True,
        actual_official_one_to_one_matching_and_matched_pixels_CPU_recomputed=True,
        fixed_native_query_GT_pair_pixels_CPU_recomputed=True,fixed_pairs_separate_from_reassignment=True,
        native_M200_same_renderer_CPU_replayed=True,native_CPU_render_batch=100,
        F256_same_original_renderer_CPU_replayed=True,all_four_head_dense_functional_formulas_CPU_replayed=True,
        all_four_original_native_band_pixels_CPU_replayed=True,
        practical_native_and_refined_support_consistency_CPU_checked=True,
        CPU_approximate_masks_substituted_for_exported_metrics=False,
        CPU_GPU_bitwise_equality_claimed=False,certified_GPU_error_bound=False,
        native_teacher_CPU_replayed=False,F_e_to_M_CPU_replayed=False,original_JPEG_decode_replayed=False,
        training_gradient_or_optimizer_trajectory_CPU_replayed=False,head_Module_forward_called=False,
        optimizer_created=False,backward_calls=0,weights_updated=False,
        no_GT_query_selection=True,no_NMS=True,no_score_cutoff=True,no_TEST=True,best_DEV_selection=False,
        task_efficacy_established=False,innovation_established=False,convergence_established=False,generalization_established=False,
        C_PR_identical_weight_initialization_claimed=False,actual_imported_files=libraries,
        CPU_affinity=[0,1],threads=2,CUDA_initialized=False,
        elapsed_seconds=time.monotonic()-started,deadline_seconds=spec['deadline_seconds'],automatic_retry=False,
        current_owned_CPU_terminal_self_certified=False,whole_leaf_VRAM_certified=False,
        evidence_limits=[
            'All18700 native exports reproduce exactly on the qualified original natural100 GPU render/reorder/quality path before any refinement.',
            'This CPU reader reconstructs every original-native-band pixel for four fixed8 C/PR states with independent functional formulas and the same bound renderer.',
            'CPU/GPU mask consistency uses prelocked practical near-zero/native-band allowances, not bitwise CUDA equality, numerical logit allclose or a certified error bound.',
            'Five official COCO12 and actual matching arrays use unchanged GPU exported masks; approximate CPU reconstructed masks never replace those exports.',
            'Fixed native query/GT pixels at .50/.75/.90 retain native FN/ignore counts but condition pixel deltas on native matches; they are separate from COCO reassignment and not global risk.',
            'Archived F/e/M/classes and prior identity are reused; no native teacher, JPEG decoder, F/e-to-M CPU projection or complete optimizer trajectory is replayed.',
            'C/PR share carrier, supervision, seed/image orders and cold native identity; feature paths/depth/width differ, and their weights are not asserted identical.',
            'Historical DEV evidence at fixed8 is not blind TEST, convergence, external SOTA superiority, feature causality, innovation or cross-dataset generalization.',
            'ROOT separately closes and checks this current actual held/reaped CPU process; the Q cannot certify its own terminal RAW.'
        ])
    atomic_write(Path(spec['Q_path']),q)
    del states
    gc.collect()
    print(json.dumps(dict(status=STATUS,Q=reference(spec['Q_path']),details=q['matching_details']),
        allow_nan=False),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',required=True)
    parser.add_argument('--owner-execute',action='store_true')
    args = parser.parse_args()
    require(args.owner_execute,'Explicit single ROOT-owned CPU execution required')
    qualify(args.spec)
