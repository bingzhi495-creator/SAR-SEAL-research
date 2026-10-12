"""Future ROOT-owned independent N09 full187/all100 CPU evaluation.

SOURCE preparation does not run science or qualify a numerical result. Every
normal ancestor and explicitly preserved failed1 is terminal before the first
SHA or non-stdlib import. ROOT owns the current admission and final RAW closure.
Only archived N07 M200/class fields are loaded; no native teacher, point head,
FIT cache/refit, optimizer, model forward or training trajectory is replayed.
The actual bound renderer is used on CPU with the already locked practical
atol=rtol=1e-4 near-threshold consistency allowance. This is not CUDA bitwise
equality or a certified rendering error bound. Native GPU reproduction remains
an exact prerequisite. Independent official COCO arrays provide reassignment;
fixed native query-GT pixel comparisons at .50/.75/.90 are reported separately.
"""
import argparse
import ast
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


SCHEMA = 'ROOT-N09-FIT-global-band-offset-evaluation-CPU-v1'
Q_SCHEMA = 'ROOT-N09-FIT-global-band-offset-evaluation-CPU-qualification-v1'
STATUS = 'ROOT_N09_FIT_GLOBAL_BAND_OFFSET_EVALUATION_CPU_QUALIFIED'
GPU_SCHEMA = 'ROOT-N09-FIT-global-band-offset-fullDEV-GPU-v1'
GPU_REPORT_SCHEMA = 'N09-FIT-global-band-offset-fullDEV-GPU-report-v1'
GPU_STATUS = 'N09_FIT_GLOBAL_BAND_OFFSET_FULLDEV_GPU_COMPLETE_PENDING_CPU'
FIT_REPORT_SCHEMA = 'N09-FIT-global-band-offset-report-v1'
FIT_STATUS = 'N09_FIT_GLOBAL_BAND_OFFSET_CPU_COMPLETE_PENDING_ROOT_QUALIFICATION'
N07_REPORT_SCHEMA = 'N07-original-detail-fixed-endpoint-evaluation-report-v1'
N07_STATUS = 'N07_ORIGINAL_DETAIL_FIXED_ENDPOINT_EVALUATION_COMPLETE_PENDING_CPU'
N07_Q_SCHEMA = 'ROOT-N07-original-detail-fixed-endpoint-evaluation-CPU-qualification-v1'
N07_Q_STATUS = 'ROOT_N07_ORIGINAL_DETAIL_FIXED_ENDPOINT_EVALUATION_CPU_QUALIFIED'
PACKET_SCHEMA = 'N07-original-detail-all100-native-fields-v1'
OWNER_SHA = 'ef9a9f17d4fb1bc3464936095ef799021dd880f8b15488d558a871c08b6e1924'
RAW_KEYS = ('receipt', 'phase', 'admission', 'phase_binding', 'bootstrap')
THREAD_ENV = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
REF_KEYS = ('entry_source', 'evaluation_entry_source', 'evaluation_spec', 'evaluation_report',
    'evaluation_GPU_admission', 'evaluation_GPU_RAW', 'FIT_offset_report', 'FIT_ROOT_record',
    'prior_N07_report', 'N07_cpu_qualification', 'native_reference_predictions', 'DEV',
    'reader_source', 'renderer_source', 'offset_source')
REQUIRED_SPEC_KEYS = ('schema', 'phase', 'sources', 'RAW_inputs', 'prior_failed_CPU_receipt',
    'CPU_admission_file', 'deadline_seconds', 'Q_path') + REF_KEYS
NUMERIC = dict(CPU_sign_and_band_rtol=1e-4, CPU_sign_and_band_atol=1e-4,
    COCO_stats_atol=1e-12, renderer_query_chunk=4, certified_GPU_error_bound=False)
CHANGE_KEYS = ('added_pixels', 'removed_pixels', 'changed_pixels', 'expanded_queries',
    'shrunk_queries', 'changed_queries', 'native_empty_queries', 'offset_empty_queries',
    'native_empty_new_nonempty_keeps_zero_score', 'new_empty_keeps_positive_native_score')
METRIC_PROTOCOL_KEYS = ('iouType', 'maxDets', 'original_GT_preserved', 'score_filter',
    'official_COCO_crowd_ignore_semantics')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    def reject(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), parse_constant=reject)


def ref_shape(ref):
    require(type(ref) is dict and set(ref) == {'path', 'bytes', 'sha256'} and
        type(ref['path']) is str and Path(ref['path']).is_absolute() and
        type(ref['bytes']) is int and ref['bytes'] >= 0 and type(ref['sha256']) is str and
        len(ref['sha256']) == 64 and set(ref['sha256']) <= set('0123456789abcdef'),
        'Malformed absolute byte/SHA reference')


def reference(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def bound(ref):
    ref_shape(ref)
    require(reference(ref['path']) == ref, 'Actual pinned input changed: ' + ref['path'])
    return Path(ref['path'])


def deadline(started, seconds):
    require(time.monotonic() - started < seconds, 'Prelocked CPU deadline exceeded; no fallback or retry')


def terminal(ref, phase, code=0):
    """Original RAW JSON only; every terminal gate precedes SHA/import."""
    ref_shape(ref)
    path = Path(ref['path'])
    require(path.name == 'PARENT_RECEIPT.json' and type(phase) is str and phase and
        path.parent.name == phase, 'Actual original RAW/phase locator required')
    raw = read(path)
    require(raw['status'] == 'ACTUAL_OWNED_TERMINAL' and type(raw['returncode']) is int and
        raw['returncode'] == code and type(raw['original_identity']) is dict and raw['original_identity'] and
        raw['root_wait_reaped'] is True and
        raw['held_handle_exit'] == dict(wait_state='WAIT_OBJECT_0', actual_exit_code=code) and
        type(raw['held_handle_exit']['actual_exit_code']) is int and
        raw['numeric_birth_absence']['original_birth_absent'] is True and raw.get('error') is None and
        raw.get('abort') is None and raw['wait_errors'] is False and raw['GPU_observation_errors'] == [],
        'Original held/reaped terminal%d required' % code)
    if 'phase' in raw:
        require(type(raw['phase']) is str and raw['phase'] == phase, 'Original RAW phase differs')
    sha, peak = raw['admission_sha256'], raw['whole_GPU_observed_peak_MiB']
    require(type(sha) is str and len(sha) == 64 and set(sha) <= set('0123456789abcdef') and
        type(peak) in (int, float) and math.isfinite(peak) and 0 <= peak <= 8192 and
        type(raw['whole_GPU_used_limit_MiB']) is int and raw['whole_GPU_used_limit_MiB'] == 8192 and
        raw['whole_leaf_VRAM_certified'] is False and
        raw['GPU_scope'] == 'whole GPU observed, not per-process WDDM allocation',
        'Original RAW owner/resource scope differs')
    return raw


def owner_join(row, raw, closure):
    owner, pb = read(bound(row['admission'])), read(bound(row['phase_binding']))
    bootstrap = bound(row['bootstrap'])
    runtime = Path(row['receipt']['path']).parent
    require(Path(row['phase_binding']['path']) == runtime/'PHASE_BINDING.json' and
        bootstrap == runtime/'bootstrap.py' and
        owner['status'] == 'ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE' and
        owner['phase'] == row['phase'] and owner['driver_sha256'] == OWNER_SHA and
        owner['automatic_retry'] is False and owner['allowed_cpus'] == [0, 1] and
        owner['cuda_visible_devices'] in ('', '0') and owner['whole_GPU_used_limit_MiB'] == 8192 and
        raw['admission_sha256'] == row['admission']['sha256'] == pb['admission_sha256'] and
        pb['bootstrap_sha256'] == row['bootstrap']['sha256'] and pb['argv'][-1] == str(bootstrap) and
        pb['CPU_affinity'] == owner['allowed_cpus'] and
        pb['CUDA_visible_devices'] == owner['cuda_visible_devices'] and pb['automatic_retry'] is False,
        'Original actual owner/PB/bootstrap differs')
    entry, args = owner['entry'], owner['entry_args']
    require(type(entry) is str and type(args) is list and entry in closure and
        owner['bound_source_sha256'].get(entry) == closure[entry]['sha256'], 'Original entry absent or unbound')
    tree = ast.parse(bootstrap.read_text(encoding='utf-8'))
    argv = [ast.literal_eval(node.value) for node in ast.walk(tree) if isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and
        target.value.id == 'sys' and target.attr == 'argv' for target in node.targets)]
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and
        isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and
        node.func.value.id == 'runpy' and node.func.attr == 'run_path']
    require(argv == [[entry]+args] and len(calls) == 1 and ast.literal_eval(calls[0].args[0]) == entry and
        {item.arg: ast.literal_eval(item.value) for item in calls[0].keywords} == {'run_name': '__main__'},
        'Actual original bootstrap entry/argv differs')
    if '--spec' in args:
        require(args.count('--spec') == 1, 'Ambiguous ancestor SPEC')
        locator = args[args.index('--spec')+1]
        require(locator in closure and owner['bound_source_sha256'].get(locator) == closure[locator]['sha256'],
            'Actual original SPEC absent or unbound')
    if 'scientific_spec' in owner:
        ref = owner['scientific_spec']
        require(ref in closure.values() and owner['bound_source_sha256'].get(ref['path']) == ref['sha256'],
            'Original scientific SPEC absent or unbound')
    return owner


def phase_index(source, spec_ref, admission, phase, cuda, rows, owners):
    matches = [i for i, owner in enumerate(owners) if owner['entry'] == source['path'] and
        owner['entry_args'] == ['--spec', spec_ref['path'], '--owner-execute']]
    require(len(matches) == 1, 'Exactly one actual successful parent phase required')
    index = matches[0]
    require(rows[index]['phase'] == phase and rows[index]['admission'] == admission and
        owners[index]['scientific_spec'] == spec_ref and owners[index]['cuda_visible_devices'] == cuda,
        'Parent primary refs do not join their actual owned RAW')
    return index


def source_gate(path, spec, started):
    require(all(key in spec for key in REQUIRED_SPEC_KEYS) and spec['schema'] == SCHEMA and
        type(spec['phase']) is str and spec['phase'] and type(spec['RAW_inputs']) is list and
        len(spec['RAW_inputs']) >= 1 and all(type(row) is dict and set(RAW_KEYS) <= set(row)
            for row in spec['RAW_inputs']), 'Complete ROOT-authored CPU SPEC required')
    rows = spec['RAW_inputs']
    receipt_paths = [row['receipt']['path'] for row in rows]
    require(len(set(receipt_paths)) == len(rows), 'Distinct actual normal RAWs required')
    raws = [terminal(row['receipt'], row['phase']) for row in rows]
    failed_refs = [spec['prior_failed_CPU_receipt']]
    terminal(failed_refs[0], Path(failed_refs[0]['path']).parent.name, 1)
    for row in spec.get('additional_failed_RAW_inputs', []):
        require(type(row) is dict and set(row) == {'receipt', 'phase', 'returncode'} and
            type(row['returncode']) is int and row['returncode'] == 1, 'Explicit preserved failed1 locator required')
        terminal(row['receipt'], row['phase'], 1)
        failed_refs.append(row['receipt'])
    require(len({ref['path'] for ref in failed_refs}) == len(failed_refs) and
        not set(receipt_paths) & {ref['path'] for ref in failed_refs}, 'Failed1 cannot qualify a normal ancestor')
    # All actual terminal gates above precede the first SHA or non-stdlib import.
    require(type(spec['sources']) is list and spec['sources'], 'Complete source/input closure required')
    for ref in spec['sources']:
        ref_shape(ref)
    closure = {ref['path']: ref for ref in spec['sources']}
    require(len(closure) == len(spec['sources']), 'Unique complete input closure required')
    required = [spec[key] for key in REF_KEYS] + failed_refs + [row[key] for row in rows
        for key in RAW_KEYS if key != 'phase']
    require(all(ref in closure.values() for ref in required), 'Actual primary refs absent from sources')
    require(type(spec['deadline_seconds']) is int and spec['deadline_seconds'] > 0 and
        os.environ.get('CUDA_VISIBLE_DEVICES') == '' and
        all(os.environ.get(key) == '2' for key in THREAD_ENV), 'Prelocked CPU environment/deadline differs')
    for ref in spec['sources']:
        deadline(started, spec['deadline_seconds'])
        bound(ref)
    source, spec_ref = reference(__file__), reference(path)
    require(source == spec['entry_source'], 'Actual CPU entry SOURCE differs')
    owners = [owner_join(row, raw, closure) for row, raw in zip(rows, raws)]
    admission_path = Path(spec['CPU_admission_file'])
    require(admission_path.is_absolute(), 'Absolute own CPU admission required')
    admission_ref = reference(admission_path)
    owner = read(admission_path)
    require(owner['status'] == 'ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE' and
        owner['driver_sha256'] == OWNER_SHA and owner['phase'] == spec['phase'] and
        owner['entry'] == source['path'] and owner['entry_args'] == ['--spec', str(path), '--owner-execute'] and
        owner['scientific_spec'] == spec_ref and owner['allowed_cpus'] == [0, 1] and
        owner['cuda_visible_devices'] == '' and owner['automatic_retry'] is False and
        owner['whole_GPU_used_limit_MiB'] == 8192 and
        type(owner['qualified_leaf_deadline_seconds']) is int and
        owner['qualified_leaf_deadline_seconds'] == spec['deadline_seconds'] and
        owner['bound_source_sha256'].get(str(path)) == spec_ref['sha256'] and
        all(owner['bound_source_sha256'].get(ref['path']) == ref['sha256'] for ref in spec['sources']),
        'Actual current CPU owner or complete closure differs')
    report = read(bound(spec['evaluation_report']))
    evaluation = read(bound(spec['evaluation_spec']))
    fit = read(bound(spec['FIT_offset_report']))
    prior = read(bound(spec['prior_N07_report']))
    q = read(bound(spec['N07_cpu_qualification']))
    require(report['schema'] == GPU_REPORT_SCHEMA and report['status'] == GPU_STATUS and
        report['source'] == spec['evaluation_entry_source'] and report['spec'] == spec['evaluation_spec'] and
        report['GPU_admission'] == spec['evaluation_GPU_admission'] and
        report['FIT_offset_report'] == spec['FIT_offset_report'] and
        report['prior_N07_report'] == spec['prior_N07_report'] and
        report['N07_cpu_qualification'] == spec['N07_cpu_qualification'] and
        report['native_reference_predictions'] == spec['native_reference_predictions'] and
        report['DEV'] == spec['DEV'] and report['reader_source'] == spec['reader_source'] and
        report['renderer_source'] == spec['renderer_source'] and report['offset_source'] == spec['offset_source'] and
        report['images'] == 187 and report['raw_GT'] == 380 and report['predictions_count'] == 18700 and
        report['native0_export_exact_reference'] is True and report['native0_COCO_exact_reference'] is True and
        report['native0_exact_queries'] == 18700 and report['all100_identity_and_fixed_scores_exact'] is True and
        report['native_model_loaded'] is False and report['point_head_loaded'] is False and
        report['teacher_forwards'] == report['head_forwards'] == report['backward_calls'] == 0 and
        report['optimizer_created'] is False and report['weights_updated'] is False and
        report['training_or_matching_loss_executed'] is False and report['no_GT_query_selection'] is True and
        report['no_NMS'] is True and report['no_score_cutoff'] is True and report['no_TEST'] is True and
        report['best_DEV_selection'] is False and report['cleanup']['allocated_after'] == 0,
        'Actual full187/all100 exact-native GPU report differs')
    require(evaluation['schema'] == GPU_SCHEMA and evaluation['entry_source'] == report['source'] and
        evaluation['phase'] == report['phase'], 'Actual N09 GPU SPEC identity differs')
    if 'recipe' in evaluation:
        require(evaluation['recipe'] == report['recipe'], 'Explicit actual N09 GPU recipe differs')
    require(evaluation['FIT_offset_report'] == spec['FIT_offset_report'] and
        evaluation['prior_N07_report'] == spec['prior_N07_report'] and
        evaluation['N07_cpu_qualification'] == spec['N07_cpu_qualification'] and
        evaluation['native_predictions'] == spec['native_reference_predictions'] and
        evaluation['DEV'] == spec['DEV'] and evaluation['reader_source'] == spec['reader_source'] and
        evaluation['renderer_source'] == spec['renderer_source'] and evaluation['offset_source'] == spec['offset_source'] and
        report['RAW_inputs'] == evaluation['RAW_inputs'] and
        report['prior_failed_CPU_receipt'] == evaluation['prior_failed_CPU_receipt'] == spec['prior_failed_CPU_receipt'] and
        report['additional_failed_RAW_inputs'] == evaluation.get('additional_failed_RAW_inputs', []) and
        all(ref in closure.values() for ref in evaluation['sources']), 'Actual GPU SPEC source/input closure differs')
    gpu_index = phase_index(report['source'], report['spec'], report['GPU_admission'], report['phase'], '0', rows, owners)
    require(rows[gpu_index]['receipt'] == spec['evaluation_GPU_RAW'] and
        all(row in rows[:gpu_index] for row in evaluation['RAW_inputs']) and
        all(ref in failed_refs for ref in [evaluation['prior_failed_CPU_receipt']] +
            [row['receipt'] for row in evaluation.get('additional_failed_RAW_inputs', [])]),
        'Actual GPU RAW and ancestor chronology differ')
    require(fit['schema'] == FIT_REPORT_SCHEMA and fit['status'] == FIT_STATUS and
        type(fit['delta']) is float and math.isfinite(fit['delta']) and
        fit['delta'] == report['delta'] == fit['FIT_fitted']['delta'] and
        fit['spec'] == report['FIT_spec'] and fit['offset_source'] == spec['offset_source'] and
        fit['FIT'] == dict(images=741, GT=1661, points_per_GT=1024, total_points=1700864) and
        fit['no_DEV'] is True and fit['no_TEST'] is True and fit['AP_computed'] is False and
        fit['FIT_fitted']['band_renormalized'] is False and fit['FIT_fitted']['regularizer_or_offset_cap'] is None,
        'Actual FIT-only single-offset report differs')
    fit_index = phase_index(fit['source'], fit['spec'], fit['CPU_admission'], fit['phase'], '', rows, owners)
    require(fit_index < gpu_index, 'FIT must precede the offset GPU evaluation')
    fit_root = read(bound(spec['FIT_ROOT_record']))
    require(fit_root['schema'] == 'ROOT-N09-FIT-only-CPU-terminal-qualified-v1' and
        fit_root['status'] == 'ROOT_N09_FIT_CPU_ACTUAL0_AND_FIT_REPORT_QUALIFIED' and
        fit_root['phase'] == fit['phase'] and fit_root['CPU_RAW'] == rows[fit_index]['receipt'] and
        fit_root['source'] == fit['source'] and fit_root['spec'] == fit['spec'] and
        fit_root['admission'] == fit['CPU_admission'] and fit_root['FIT_report'] == spec['FIT_offset_report'] and
        fit_root['start'] == fit['start'] and fit_root['fitted'] == fit['FIT_fitted'] and
        fit_root['root_wait_reaped'] is True and fit_root['original_birth_absent'] is True and
        fit_root['held_handle_exit'] == dict(wait_state='WAIT_OBJECT_0', actual_exit_code=0) and
        fit_root['no_DEV_AP_evaluation'] is True and fit_root['coefficient_independently_refitted'] is False,
        'Actual ROOT FIT terminal/report record differs; no cache refit is performed')
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
        q['native_M200_original_render_CPU_replayed'] is True,
        'Actual independently qualified strong N07 native parent differs')
    n07_index = phase_index(q['reader_source'], q['spec'], q['CPU_admission'], q['phase'], '', rows, owners)
    require(n07_index < fit_index, 'Qualified N07 native parent must precede FIT')
    packets = report['packet_refs']
    require(type(packets) is list and len(packets) == 187 and
        len({ref['path'] for ref in packets}) == 187 and packets == prior['final_field_packets'] == q['packet_refs'],
        'Exact full187 inherited final-field refs required')
    outputs = [report[key] for key in ('start', 'trace', 'image_diagnostics', 'predictions', 'native0_predictions',
        'source', 'spec', 'GPU_admission', 'export_source', 'COCO_source')] + packets
    outputs += [prior[key] for key in ('image_rows', 'native_query_score_factors', 'source', 'spec', 'GPU_admission')]
    outputs += [fit[key] for key in ('source', 'spec', 'CPU_admission', 'start')]
    outputs += [q[key] for key in ('spec', 'CPU_admission', 'E24_reference_predictions', 'E24_reference_Q')]
    require(all(ref in closure.values() for ref in outputs), 'Actual metadata/native outputs absent from current closure')
    qpath = Path(spec['Q_path'])
    require(qpath.is_absolute() and qpath.parent.is_dir() and not qpath.exists() and
        not qpath.with_suffix(qpath.suffix+'.tmp').exists(), 'Fresh absolute CPU Q path required')
    return dict(closure=closure, source=source, spec_ref=spec_ref, admission_ref=admission_ref,
        owner=owner, report=report, prior=prior, q=q, fit=fit, packets=packets, failed_refs=failed_refs,
        current_GPU_RAW=rows[gpu_index]['receipt'])


def load(ref, name):
    path = bound(ref)
    require(path.suffix == '.py', 'Bound Python SOURCE required')
    source = path.read_bytes()
    require(len(source) == ref['bytes'] and hashlib.sha256(source).hexdigest() == ref['sha256'],
        'SOURCE changed after binding')
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(source, str(path), 'exec'), module.__dict__)
    return module


def resources(psutil, torch):
    require(psutil.Process().cpu_affinity() == [0, 1] and
        torch.get_num_threads() == torch.get_num_interop_threads() == 2 and
        os.environ.get('CUDA_VISIBLE_DEVICES') == '' and all(os.environ.get(key) == '2' for key in THREAD_ENV) and
        not torch.cuda.is_initialized(), 'Actual CPU01/threads2/cold CUDA differs')


def imported_files(psutil, torch, np, cv2, COCO, COCOeval, masks, gate):
    modules = dict(psutil=psutil, psutil_platform=psutil._psplatform, psutil_binary=psutil._psplatform.cext,
        torch=torch, torch_binary=torch._C, torch_tensor=sys.modules['torch._tensor'],
        torch_autograd=sys.modules['torch.autograd'], numpy=np, numpy_binary=np.core._multiarray_umath,
        cv2=cv2, cv2_binary=cv2._native, COCO=sys.modules[COCO.__module__],
        COCOeval=sys.modules[COCOeval.__module__], masks=masks, mask_binary=masks._mask)
    refs = {name: reference(module.__file__) for name, module in modules.items()}
    refs['torch_serialization'] = reference(inspect.getsourcefile(torch.load))
    refs['interpolation'] = reference(inspect.getsourcefile(torch.nn.functional.interpolate))
    require(all(ref in gate['closure'].values() and
        gate['owner']['bound_source_sha256'].get(ref['path']) == ref['sha256'] for ref in refs.values()),
        'Actual imported CPU library/binary/interpolation absent or unbound')
    return refs


def stats_check(actual, expected, message):
    require(type(actual) is list and type(expected) is list and len(actual) == len(expected) == 12 and
        all(type(value) in (int, float) and math.isfinite(value) and (value == -1 or 0 <= value <= 1)
            for value in actual+expected) and all(abs(a-b) <= NUMERIC['COCO_stats_atol']
            for a, b in zip(actual, expected)), message)


def packet_check(torch, reader, coco, packet, row, index, ref, gate):
    prior = gate['prior']
    identity = row['identity']
    iid = row['image_id']
    image = coco.imgs[iid]
    require(row['dataset_index'] == packet['dataset_index'] == index and row['packet'] == ref and
        packet['schema'] == PACKET_SCHEMA and packet['identity'] == identity and
        packet['source'] == prior['source'] and packet['spec'] == prior['spec'] and packet['phase'] == prior['phase'] and
        packet['teacher_checkpoint'] == prior['teacher_checkpoint'] and
        packet['head_checkpoints'] == prior['head_checkpoints'] and packet['head_endpoint_epoch'] == 8 and
        packet['GPU_original_logits_saved'] is False and
        packet['prediction_query_ids'] == row['prediction_query_ids'] and
        packet['field_digests'] == row['final_fields'], 'Archived field packet lineage/order differs')
    H, W = image['height'], image['width']
    h, w = identity['actual_img_shape']
    require(identity['image_id'] == iid and identity['original_shape'] == [H, W] and
        type(h) is int and type(w) is int and 0 < h <= 800 and 0 < w <= 800 and
        identity['batch_input_shape'] == [800, 800] and
        identity['ann_ids'] == row['ann_ids'] == [ann['id'] for ann in coco.imgToAnns[iid]] and
        row['raw_GT'] == len(identity['ann_ids']), 'Archived original-size/GT/crop identity differs')
    qids = packet['prediction_query_ids']
    require(type(qids) is list and len(qids) == 100 and all(type(query) is int for query in qids) and
        sorted(qids) == list(range(100)), 'All100 archived original query IDs required')
    fields = packet['native_fields']
    require(set(fields) == set(packet['field_digests']) == {'F256', 'e0', 'M200', 'final_class_logits'},
        'Original four-field packet ABI differs')
    for name, shape in (('M200', (1, 100, 200, 200)), ('final_class_logits', (1, 100, 2))):
        value = fields[name]
        require(isinstance(value, torch.Tensor) and value.device.type == 'cpu' and value.dtype == torch.float32 and
            tuple(value.shape) == shape and not value.requires_grad and bool(torch.isfinite(value).all()) and
            reader.digest(value) == packet['field_digests'][name], 'Finite archived native field differs: '+name)
    require(dict(reader.digest(fields['final_class_logits'][0]), values=fields['final_class_logits'][0].tolist()) ==
        row['native_class_packet'], 'Original archived class packet differs')
    return fields['M200'], fields['final_class_logits'], identity, qids


def replay(torch, np, masks, coco, reader, renderer, offset, gate, spec, populations, factors, diagnostics, started):
    rows = read(bound(gate['prior']['image_rows']))
    ids = list(coco.imgs)
    trace = [reader.read_json_line(line) for line in bound(gate['report']['trace']).read_text(encoding='utf-8').splitlines()
        if line.strip()]
    require(type(rows) is list and len(rows) == len(diagnostics) == 187 and diagnostics == trace and
        [row['image_id'] for row in rows] == ids and sum(row['raw_GT'] for row in rows) == 380,
        'Canonical archived187/all380 rows and actual N09 TRACE differ')
    checks = []
    totals = {key: 0 for key in CHANGE_KEYS}
    with torch.no_grad():
        for index, (row, ref, diagnostic) in enumerate(zip(rows, gate['packets'], diagnostics)):
            deadline(started, spec['deadline_seconds'])
            packet = torch.load(str(bound(ref)), map_location='cpu')
            M, C, identity, qids = packet_check(torch, reader, coco, packet, row, index, ref, gate)
            iid = row['image_id']
            before_rows, after_rows = populations['native'][iid], populations['offset'][iid]
            require([r['query_id'] for r in before_rows] == qids == [r['query_id'] for r in after_rows] and
                diagnostic['dataset_index'] == index and diagnostic['image_id'] == iid and diagnostic['packet'] == ref and
                diagnostic['queries'] == 100 and diagnostic['native_query_order'] == qids and
                diagnostic['native_masks_scores_and_classes_exact'] is True and
                diagnostic['fixed_native_scores_and_class_and_order_exact'] is True and
                diagnostic['original_band_only'] is True and len(diagnostic['query_diagnostics']) == 100,
                'Actual N09 image/query diagnostic identity differs')
            classes = C[0].softmax(-1)[:, 0]
            stats = dict(native_CPU_GPU_support_flips=0, offset_CPU_GPU_support_flips=0,
                practical_near_native_zero_pixels=0, practical_near_offset_zero_pixels=0,
                practical_near_native_band_pixels=0, max_abs_class_difference=0., max_abs_quality_difference=0.)
            query_rows = []
            for first in range(0, 100, NUMERIC['renderer_query_chunk']):
                deadline(started, spec['deadline_seconds'])
                selected = torch.tensor(qids[first:first+4], dtype=torch.int64)
                L = renderer.render_original(torch, M[:, selected], identity)
                require(L.device.type == 'cpu' and L.dtype == torch.float32 and
                    tuple(L.shape) == (4,)+tuple(identity['original_shape']) and bool(torch.isfinite(L).all()),
                    'Same actual renderer CPU FP32 original-space ABI differs')
                shifted = offset.apply_offset(torch, L, gate['fit']['delta'])
                band = L.abs() <= 2.
                require(torch.equal(shifted[~band], L[~band]), 'CPU operator changed outside original native band')
                for j, (z, changed_z) in enumerate(zip(L, shifted)):
                    ordinal = first+j
                    before, after = before_rows[ordinal], after_rows[ordinal]
                    qid = qids[ordinal]
                    a = masks.decode(reader.rle_bytes(before['segmentation'])).astype(bool)
                    b = masks.decode(reader.rle_bytes(after['segmentation'])).astype(bool)
                    require(a.shape == b.shape == tuple(z.shape), 'Original full-raster binary size differs')
                    A, B = torch.from_numpy(np.ascontiguousarray(a)), torch.from_numpy(np.ascontiguousarray(b))
                    tau = NUMERIC['CPU_sign_and_band_atol'] + NUMERIC['CPU_sign_and_band_rtol']*z.abs()
                    shifted_tau = NUMERIC['CPU_sign_and_band_atol'] + NUMERIC['CPU_sign_and_band_rtol']*changed_z.abs()
                    near0, near_shifted0 = z.abs() <= tau, changed_z.abs() <= shifted_tau
                    near_band = (z.abs()-2.).abs() <= tau
                    native_flips, offset_flips = A != (z > 0), B != (changed_z > 0)
                    require(not bool((native_flips & ~near0).any()),
                        'Native CPU/GPU support discrepancy outside prelocked practical near-zero allowance')
                    require(not bool((offset_flips & ~(near_shifted0 | near_band)).any()),
                        'Offset CPU/GPU support discrepancy outside prelocked practical threshold/band allowance')
                    actual_change = A != B
                    require(not bool((actual_change & (z.abs() > 2.+tau)).any()),
                        'Actual GPU offset changed geometry outside practical original-band allowance')
                    added = int(np.count_nonzero(~a & b))
                    removed = int(np.count_nonzero(a & ~b))
                    require(not added if gate['fit']['delta'] <= 0 else not removed,
                        'Actual scalar offset violates its global expansion/shrinkage direction')
                    if gate['fit']['delta'] == 0:
                        require(before == after, 'Zero global offset must preserve every native export exactly')
                    native_empty, offset_empty = not bool(a.any()), not bool(b.any())
                    if native_empty:
                        require(before['score'] == after['score'] == 0., 'Native empty masks must keep original zero S0')
                    factor = factors[iid, qid]
                    probability = float(classes[qid])
                    quality = float((z.sigmoid()*A).sum()/(A.sum()+1e-6))
                    require(math.isclose(probability, factor['class_probability'],
                        rel_tol=reader.NUMERIC['class_rtol'], abs_tol=reader.NUMERIC['class_atol']) and
                        math.isclose(quality, factor['mask_quality'], rel_tol=reader.NUMERIC['quality_rtol'],
                            abs_tol=reader.NUMERIC['quality_atol']), 'CPU native class/quality on exact exported support differs')
                    stats['max_abs_class_difference'] = max(stats['max_abs_class_difference'], abs(probability-factor['class_probability']))
                    stats['max_abs_quality_difference'] = max(stats['max_abs_quality_difference'], abs(quality-factor['mask_quality']))
                    check = dict(query_id=qid, native_export_ordinal=ordinal, fixed_native_score=before['score'],
                        native_foreground_pixels=int(a.sum()), offset_foreground_pixels=int(b.sum()),
                        added_pixels=added, removed_pixels=removed, changed_pixels=added+removed,
                        native_empty=native_empty, offset_empty=offset_empty,
                        native_empty_new_nonempty_keeps_zero_score=native_empty and not offset_empty,
                        new_empty_keeps_positive_native_score=not native_empty and offset_empty and before['score'] > 0)
                    observed = diagnostic['query_diagnostics'][ordinal]
                    require(all(observed[key] == value for key, value in check.items()) and
                        type(observed['band_pixels']) is int and 0 <= observed['band_pixels'] <= a.size and
                        abs(int(band[j].sum())-observed['band_pixels']) <= int(near_band.sum()),
                        'Actual per-query pixel/empty/band accounting differs')
                    for key, value in (('native_CPU_GPU_support_flips', native_flips),
                        ('offset_CPU_GPU_support_flips', offset_flips), ('practical_near_native_zero_pixels', near0),
                        ('practical_near_offset_zero_pixels', near_shifted0), ('practical_near_native_band_pixels', near_band)):
                        stats[key] += int(value.sum())
                    query_rows.append(check)
                del selected, L, shifted, band
            image_counts = dict(added_pixels=sum(r['added_pixels'] for r in query_rows),
                removed_pixels=sum(r['removed_pixels'] for r in query_rows),
                changed_pixels=sum(r['changed_pixels'] for r in query_rows),
                expanded_queries=sum(r['added_pixels'] > 0 for r in query_rows),
                shrunk_queries=sum(r['removed_pixels'] > 0 for r in query_rows),
                changed_queries=sum(r['changed_pixels'] > 0 for r in query_rows),
                native_empty_queries=sum(r['native_empty'] for r in query_rows),
                offset_empty_queries=sum(r['offset_empty'] for r in query_rows),
                native_empty_new_nonempty_keeps_zero_score=sum(r['native_empty_new_nonempty_keeps_zero_score'] for r in query_rows),
                new_empty_keeps_positive_native_score=sum(r['new_empty_keeps_positive_native_score'] for r in query_rows))
            require(all(image_counts[key] == diagnostic[key] for key in CHANGE_KEYS), 'Complete per-image pixel population differs')
            for key in CHANGE_KEYS:
                totals[key] += image_counts[key]
            checks.append(dict(dataset_index=index, image_id=iid, packet=ref, queries=100,
                M200_and_class_shape_digest_CPU_checked=True, original_JPEG_decode_replayed=False,
                CPU_GPU_bitwise_equality_claimed=False, certified_GPU_error_bound=False,
                practical_tolerance=NUMERIC, query_pixel_counts=query_rows, **stats, **image_counts))
            del packet, M, C, classes
            print(json.dumps(dict(N09_CPU_archived_render_images=index+1)), flush=True)
    require(totals == gate['report']['change_totals'] and totals['changed_pixels'] ==
        totals['added_pixels']+totals['removed_pixels'], 'Complete N09 change/empty accounting differs')
    return checks, totals


def fixed_native_pairs(np, masks, reader, coco, native_images, offset_population, offset_delta):
    """Evaluate offset geometry at native-assigned query/GT; never rematch here."""
    details, summaries = [], []
    for threshold in (.5, .75, .9):
        records = []
        counts = dict(native_TP=0, native_FN=0, ignored_GT=0, offset_fixed_pair_IoU_ge_threshold=0)
        for image in native_images:
            candidates = [row for row in image['thresholds'] if abs(row['threshold']-threshold) <= 1e-12]
            require(len(candidates) == 1, 'Exact fixed native matching threshold required')
            lookup = {row['query_id']: row for row in offset_population[image['image_id']]}
            for gt in candidates[0]['GT']:
                key = gt['TP_pair_key']
                native = image['TP_pairs'][key] if key else None
                shifted = None
                pair_delta = None
                if gt['ignored']:
                    counts['ignored_GT'] += 1
                elif native is None:
                    counts['native_FN'] += 1
                else:
                    counts['native_TP'] += 1
                    query = native['query_id']
                    target = masks.decode(reader.rle_bytes(coco.annToRLE(coco.anns[gt['gt_id']]))).astype(bool)
                    prediction = masks.decode(reader.rle_bytes(lookup[query]['segmentation'])).astype(bool)
                    require(target.shape == prediction.shape, 'Fixed native-pair original raster shapes differ')
                    tp = int(np.count_nonzero(target & prediction))
                    fp = int(np.count_nonzero(~target & prediction))
                    fn = int(np.count_nonzero(target & ~prediction))
                    union, total = tp+fp+fn, 2*tp+fp+fn
                    shifted = dict(gt_id=gt['gt_id'], query_id=query, IoU=tp/union if union else 0.,
                        Dice=2*tp/total if total else 0., pixel_TP=tp, pixel_FP=fp, pixel_FN=fn,
                        empty_union_IoU_convention=0, offset_rematched=False)
                    if offset_delta <= 0:
                        require(shifted['pixel_FP'] <= native['pixel_FP'] and
                            shifted['pixel_FN'] >= native['pixel_FN'] and shifted['pixel_TP'] <= native['pixel_TP'],
                            'Nonpositive offset violates fixed-query/GT shrinkage pixel direction')
                    if offset_delta >= 0:
                        require(shifted['pixel_FP'] >= native['pixel_FP'] and
                            shifted['pixel_FN'] <= native['pixel_FN'] and shifted['pixel_TP'] >= native['pixel_TP'],
                            'Nonnegative offset violates fixed-query/GT expansion pixel direction')
                    pair_delta = {name: shifted[name]-native[name] for name in ('IoU', 'Dice', 'pixel_TP', 'pixel_FP', 'pixel_FN')}
                    counts['offset_fixed_pair_IoU_ge_threshold'] += int(shifted['IoU'] >= threshold)
                records.append(dict(image_id=image['image_id'], gt_id=gt['gt_id'], ignored=gt['ignored'],
                    native_FN=not gt['ignored'] and native is None, native_pair=native,
                    offset_same_native_query_pair=shifted, offset_minus_native=pair_delta))
        require(len(records) == 380 and counts['native_TP']+counts['native_FN']+counts['ignored_GT'] == 380,
            'Fixed native-pair GT/FN/ignore population does not conserve380')
        selected = [row for row in records if row['native_pair'] is not None]
        summaries.append(dict(threshold=threshold, counts=counts, conditioned_on_native_matching=True,
            offset_reassignment_used=False, conditional_pixels_do_not_cover_unmatched_instances=True,
            scalar_direction_fixed_query_pixel_FP_FN_checked=True,
            scalar_direction_imposes_no_reassigned_matching_count_direction=True,
            native_matched_pixel_totals={name: sum(row['native_pair'][name] for row in selected)
                for name in ('pixel_TP', 'pixel_FP', 'pixel_FN')},
            offset_same_query_pixel_totals={name: sum(row['offset_same_native_query_pair'][name] for row in selected)
                for name in ('pixel_TP', 'pixel_FP', 'pixel_FN')},
            conditional_mean_IoU_delta=math.fsum(row['offset_minus_native']['IoU'] for row in selected)/len(selected) if selected else None,
            conditional_mean_Dice_delta=math.fsum(row['offset_minus_native']['Dice'] for row in selected)/len(selected) if selected else None))
        details.append(dict(threshold=threshold, GT_rows=records, fixed_native_query_assignment=True))
    return summaries, details


def atomic_write(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    require(path.parent.is_dir() and not path.exists() and not temporary.exists(), 'Fresh output file required')
    with temporary.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
    temporary.replace(path)


def qualify(spec_path):
    started = time.monotonic()
    path = Path(spec_path)
    require(path.is_absolute(), 'Absolute ROOT-authored CPU SPEC required')
    path = path.resolve(strict=True)
    spec = read(path)
    gate = source_gate(path, spec, started)
    import psutil
    require(psutil.Process().cpu_affinity() == [0, 1], 'CPU01 required before numerical imports')
    import torch
    require(torch.__version__ == '2.0.1+cu118' and not torch.cuda.is_initialized(), 'Original Torch and cold CUDA required')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    import numpy as np
    import cv2
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from pycocotools import mask as masks
    require(np.__version__ == '1.23.5', 'Qualified original NumPy required')
    resources(psutil, torch)
    libraries = imported_files(psutil, torch, np, cv2, COCO, COCOeval, masks, gate)
    reader = load(spec['reader_source'], '_N09_independent_N07_matching_reader')
    renderer = load(spec['renderer_source'], '_N09_actual_CPU_renderer')
    offset = load(spec['offset_source'], '_N09_actual_FIT_only_offset_math')
    require(reader.Q_SCHEMA == N07_Q_SCHEMA and reader.STATUS == N07_Q_STATUS and offset.BAND_LIMIT == 2. and
        all(reader.NUMERIC[key] == value for key, value in NUMERIC.items()),
        'Already prelocked N07 practical numeric/helper ABI differs')
    coco = COCO(str(bound(spec['DEV'])))
    ids = list(coco.imgs)
    require(len(ids) == len(set(ids)) == 187 and len(coco.anns) == 380 and set(coco.cats) == {1},
        'Complete original DEV187/380/category1 required')
    report = gate['report']
    native_reference = read(bound(spec['native_reference_predictions']))
    native = read(bound(report['native0_predictions']))
    shifted = read(bound(report['predictions']))
    strong_reference = read(bound(gate['q']['E24_reference_predictions']))
    require(native == native_reference == strong_reference,
        'Native0 full18700 prediction identity must equal the strong E24 reference exactly')
    populations = dict(native=reader.population(coco, native, ids), offset=reader.population(coco, shifted, ids))
    require(all({key: value for key, value in before.items() if key != 'segmentation'} ==
        {key: value for key, value in after.items() if key != 'segmentation'} for before, after in zip(native, shifted)),
        'Every original query/class/score/order must remain exactly fixed')
    factors, factor_proof = reader.factor_check(np, read(bound(gate['prior']['native_query_score_factors'])), native, ids)
    diagnostics = read(bound(report['image_diagnostics']))
    render_checks, change_totals = replay(torch, np, masks, coco, reader, renderer, offset, gate, spec,
        populations, factors, diagnostics, started)
    resources(psutil, torch)
    controls, matching = {}, {}
    for name, predictions in (('native', native), ('offset', shifted)):
        deadline(started, spec['deadline_seconds'])
        evaluator, metrics = reader.official_metrics(coco, predictions, ids, COCOeval)
        expected = report['native0_COCO'] if name == 'native' else report['offset_COCO']
        require(all(metrics[key] == expected[key] for key in METRIC_PROTOCOL_KEYS), 'Actual official COCO conventions differ')
        stats_check(metrics['stats'], expected['stats'], 'Independent official COCO12 differs: '+name)
        if name == 'native':
            require(expected['stats'] == gate['q']['expected_E24_COCO_stats'] ==
                gate['q']['control_results']['native_official']['COCO']['stats'],
                'Native GPU COCO12 exact strong-reference gate differs')
            stats_check(metrics['stats'], gate['q']['expected_E24_COCO_stats'], 'Independent strong E24 COCO12 differs')
        images = reader.matching_images(np, masks, evaluator, ids)
        matching[name] = images
        controls[name] = dict(predictions=report['native0_predictions'] if name == 'native' else report['predictions'],
            COCO=metrics, actual_one_to_one=reader.matching_summary(images))
        del evaluator
        gc.collect()
        print(json.dumps(dict(N09_CPU_official_COCO_control=name)), flush=True)
    reassignment_summary, reassignment_details = reader.paired_differences(matching['native'], matching['offset'])
    fixed_summary, fixed_details = fixed_native_pairs(np, masks, reader, coco, matching['native'], populations['offset'], gate['fit']['delta'])
    qpath = Path(spec['Q_path'])
    detail_path = qpath.with_suffix('.matching_details.json')
    require(detail_path != qpath, 'Distinct matching detail and Q paths required')
    deadline(started, spec['deadline_seconds'])
    atomic_write(detail_path, dict(schema='N09-independent-official-and-fixed-native-pair-details-v1',
        source=gate['source'], spec=gate['spec_ref'], evaluation_report=spec['evaluation_report'],
        official_matching=matching, offset_reassignment_vs_native=reassignment_details,
        fixed_native_query_GT_pairs=fixed_details,
        fixed_pair_and_reassignment_are_separate=True, no_GT_directed_inference=True))
    for ref in spec['sources']:
        deadline(started, spec['deadline_seconds'])
        bound(ref)
    require(reference(path) == gate['spec_ref'] and reference(__file__) == gate['source'] and
        reference(spec['CPU_admission_file']) == gate['admission_ref'] and
        imported_files(psutil, torch, np, cv2, COCO, COCOeval, masks, gate) == libraries,
        'Final SPEC/SOURCE/admission/library closure changed')
    resources(psutil, torch)
    deadline(started, spec['deadline_seconds'])
    result = dict(schema=Q_SCHEMA, status=STATUS, phase=spec['phase'], source=spec['evaluation_entry_source'],
        reader_source=gate['source'], spec=gate['spec_ref'], CPU_admission=gate['admission_ref'],
        evaluation_spec=spec['evaluation_spec'], evaluation_report=spec['evaluation_report'],
        evaluation_GPU_admission=spec['evaluation_GPU_admission'], evaluation_GPU_RAW=spec['evaluation_GPU_RAW'],
        FIT_offset_report=spec['FIT_offset_report'], FIT_ROOT_record=spec['FIT_ROOT_record'],
        FIT_ROOT_record_terminal_report_metadata_join_checked=True, delta=gate['fit']['delta'],
        prior_N07_report=spec['prior_N07_report'], N07_cpu_qualification=spec['N07_cpu_qualification'],
        native_reference_predictions=spec['native_reference_predictions'], E24_reference_predictions=gate['q']['E24_reference_predictions'],
        DEV=spec['DEV'], renderer_source=spec['renderer_source'], matching_reader_source=spec['reader_source'],
        offset_source=spec['offset_source'], RAW_inputs=spec['RAW_inputs'], prior_failed_CPU_receipt=spec['prior_failed_CPU_receipt'],
        additional_failed_RAW_inputs=spec.get('additional_failed_RAW_inputs', []),
        images=187, raw_GT=380, queries_per_image=100, predictions_per_control=18700, packet_refs=gate['packets'],
        controls=controls, native_factor_check=factor_proof, native_score_factors=gate['prior']['native_query_score_factors'],
        raw_population=dict(images=187, GT=380, native_queries=18700, offset_queries=18700,
            all100_query_class_score_order_exact=True, native_full_export_equals_strong_E24=True,
            every_original_GT_including_FN_and_ignore_retained=True, empty_query_population=change_totals),
        render_checks=render_checks, change_totals=change_totals, numeric_protocol=NUMERIC,
        fixed_native_query_GT_pairs=fixed_summary, reassigned_offset_vs_native=reassignment_summary,
        matching_details=reference(detail_path), independent_two_official_COCO12_CPU_recomputed=True,
        actual_official_one_to_one_and_matched_pixels_CPU_recomputed=True,
        fixed_native_query_GT_pair_pixels_CPU_recomputed=True, fixed_pairs_separate_from_offset_reassignment=True,
        strong_E24_native_export_identity_exact_CPU_checked=True, all100_fixed_scores_classes_and_order_CPU_checked=True,
        native_M200_same_renderer_CPU_replayed=True, offset_operator_on_CPU_rendered_L_replayed=True,
        practical_near_threshold_support_consistency_checked=True, CPU_GPU_bitwise_equality_claimed=False,
        certified_GPU_error_bound=False, CUDA_numerics_independently_replayed=False,
        native_teacher_replayed=False, native_model_loaded=False, point_head_loaded=False,
        F_e_to_M_CPU_replayed=False, original_JPEG_decode_replayed=False, FIT_full_cache_refitted=False,
        training_trajectory_CPU_replayed=False, native_weights_updated=False, optimizer_created=False,
        teacher_forwards=0, head_forwards=0, backward_calls=0, no_NMS=True, no_score_cutoff=True,
        no_GT_query_selection=True, no_TEST=True, DEV_offset_selection_performed=False,
        innovation_established=False, causal_mechanism_established=False, generalization_established=False,
        convergence_established=False, current_owned_CPU_terminal_self_certified=False, whole_leaf_VRAM_certified=False,
        actual_imported_files=libraries, CPU_affinity=[0, 1], threads=2, CUDA_initialized=False,
        elapsed_seconds=time.monotonic()-started, deadline_seconds=spec['deadline_seconds'], automatic_retry=False,
        evidence_limits=[
            'Independent COCO12 and actual official one-to-one arrays use all historical DEV187/all100 exports and unchanged GT.',
            'Archived native M200/class fields and prior identity qualification are reused; teacher/head and F/e projection are not replayed.',
            'GPU native reproduction stays an exact prerequisite; CPU support comparisons use only the old practical atol=rtol=1e-4 allowance.',
            'Practical near-threshold checks are neither CUDA bitwise equality nor a certified numerical error bound.',
            'Fixed native query-GT pixels condition on native matches; they omit native FN and cannot replace offset reassignment or global COCO.',
            'The bound FIT ROOT terminal/report record is metadata-joined; the complete FIT cache is not reopened or refitted.',
            'N09 is a known single scalar control; this result supplies no innovation, causal mechanism, blind TEST, convergence or external generalization claim.',
            'ROOT separately closes this current CPU RAW and observes resources; this reader cannot certify its own final held process.'
        ])
    atomic_write(qpath, result)
    print(json.dumps(dict(status=STATUS, Q=reference(qpath), matching_details=reference(detail_path)), allow_nan=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--owner-execute', action='store_true')
    args = parser.parse_args()
    require(args.owner_execute, 'Explicit single ROOT-owned CPU execution required')
    qualify(args.spec)
