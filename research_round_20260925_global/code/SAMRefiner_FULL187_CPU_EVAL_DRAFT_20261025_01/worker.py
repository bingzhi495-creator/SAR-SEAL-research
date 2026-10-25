"""Independent native/R/SAM COCO comparison candidate; stdlib-only import."""
import math
import argparse
from copy import deepcopy
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import sys
import time
import traceback
import types

SCHEMA = 'ROOT-SAMRefiner-full187-CPU-eval-spec-v01'
OPERATION = 'CPU_SAMREFINER_FULL187_FIXED_NATIVE100_COCO_V01'
LIMITS = dict(allowed_cpus=[0,1], threads=2, maximum_RSS_bytes=16*1024**3,
              minimum_available_RAM_bytes=16*1024**3, deadline_seconds=1800)
SOURCE_PINS = {
    'eval_body': dict(bytes=24972, sha256='0e4ec8c999b0e954ce6d588b3ffae97510be1051c4465f9d4510c3c2ec3b34bf'),
    'official': dict(bytes=49923, sha256='2357929fcd38ffc3020c5b8d204ee371da8da3972ff2e2e03398c8225c55e916'),
}
LIBRARY_NAMES = dict(numpy='numpy', psutil='psutil', pycocotools='pycocotools',
    mask_api='pycocotools.mask', mask_extension='pycocotools._mask',
    coco_loader='pycocotools.coco', cocoeval='pycocotools.cocoeval')
INPUT_ROLES = {'context','legacy_context','legacy_result','native_predictions','R_predictions',
               'DEV','SAM_predictions','SAM_report'}
COCO_KEYS = {'stats','metrics','iouType','maxDets','original_GT_preserved','score_filter',
             'official_COCO_crowd_ignore_semantics'}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def validate_population(native, values, ids, shapes, *, queries_per_image=100):
    need(type(native) is list and type(values) is list
         and len(native) == len(values) == len(ids)*queries_per_image,
         'Complete candidate population; no filtering or truncation')
    need(len(ids) == len(set(ids)) and set(ids) == set(shapes), 'Unique original image roster')
    for index, image_id in enumerate(ids):
        block = native[index*queries_per_image:(index+1)*queries_per_image]
        need(all(type(row.get('image_id')) is int and row['image_id'] == image_id
                 and type(row.get('query_id')) is int for row in block)
             and {row['query_id'] for row in block} == set(range(queries_per_image)),
             'Complete unique native query IDs in original image order')
    for original, candidate in zip(native, values):
        need(type(original) is dict and type(candidate) is dict
             and set(original) == set(candidate) == {'image_id','query_id','category_id','score','segmentation'},
             'Exactly native segmentation-only export keys; no bbox injection')
        metadata = lambda row: {key:value for key,value in row.items() if key != 'segmentation'}
        signature = lambda value: json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
        need(signature(metadata(original)) == signature(metadata(candidate)),
             'Exact original image/query/class/S0/order and JSON types required')
        need(type(original['category_id']) is int and original['category_id'] == 1
             and type(original['score']) in (int,float) and math.isfinite(original['score'])
             and 0 <= original['score'] <= 1, 'Finite original bounded S0 and category1')
        for row in (original, candidate):
            rle = row['segmentation']
            need(type(rle) is dict and set(rle) == {'size','counts'}
                 and rle['size'] == shapes[original['image_id']]
                 and type(rle['counts']) is str and rle['counts'],
                 'Original-grid compressed COCO RLE required, including empty masks')
    return True


def delta_stats(left, right):
    need(len(left) == len(right) == 12 and all(type(v) in (int,float) and math.isfinite(v)
         and (v == -1 or 0 <= v <= 1) for v in left+right), 'Finite official COCO12 stats')
    return dict(left_stats=list(left), right_stats=list(right),
        percentage_points=[None if a == -1 or b == -1 else 100*(a-b) for a,b in zip(left,right)],
        undefined_policy='Raw -1 retained; difference undefined if either method has -1')


def strict_ref(ref):
    need(type(ref) is dict and set(ref) == {'path','bytes','sha256'}
         and type(ref['bytes']) is int and ref['bytes'] >= 0
         and type(ref['sha256']) is str and len(ref['sha256']) == 64
         and set(ref['sha256']) <= set('0123456789abcdef'), 'Exact file reference required')
    path = PurePosixPath(ref['path'])
    need(path.anchor == '/' and str(path) == ref['path'] and '..' not in path.parts
         and '\\' not in ref['path'], 'Canonical actual POSIX ref')


def reference(path):
    path = Path(path).resolve(strict=True)
    before = path.stat()
    digest, size = hashlib.sha256(), 0
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block); size += len(block)
    after = path.stat()
    stat = lambda s: (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    need(stat(before) == stat(after), 'File changed while hashing')
    return dict(path=str(path), bytes=size, sha256=digest.hexdigest())


def validate_spec(spec):
    need(type(spec) is dict and set(spec) == {'schema','operation','run_id','source_refs',
         'library_sources','input_refs','root_lease_ref','runtime_pythonpath_root','prelock',
         'output_directory','sources'}, 'Exact independent CPU spec keys')
    need(spec['schema'] == SCHEMA and spec['operation'] == OPERATION
         and type(spec['run_id']) is str and spec['run_id'].startswith('SAMRefiner-full187-CPU-v01-'),
         'Independent CPU evaluation identity')
    need(set(spec['source_refs']) == {'worker','eval_body','official'}
         and set(spec['library_sources']) == set(LIBRARY_NAMES)
         and set(spec['input_refs']) == INPUT_ROLES, 'Exact explicit roles')
    for role, pin in SOURCE_PINS.items():
        need(all(spec['source_refs'][role][key] == value for key,value in pin.items()),
             'Frozen original metrics helper identity: '+role)
    need(spec['prelock'] == dict(limits=LIMITS, methods=['native_official','R','SAMRefiner'],
         score_policy='ORIGINAL_NATIVE_S0_UNCHANGED', annotation_policy='ORIGINAL_DEV_UNCHANGED',
         expected_images=187, expected_annotations=380, expected_candidates=18700,
         GPU_hidden=True, new_training=False), 'Fixed same-protocol CPU prelock')
    refs = list(spec['source_refs'].values())+list(spec['library_sources'].values())
    refs += list(spec['input_refs'].values())+[spec['root_lease_ref']]
    unique = {}
    out = PurePosixPath(spec['output_directory'])
    need(out.anchor == '/' and str(out) == spec['output_directory'] and '..' not in out.parts
         and str(out) != '/', 'Canonical explicit output')
    for value in refs:
        strict_ref(value)
        need(value['path'] not in unique or unique[value['path']] == value, 'Conflicting refs')
        need(out not in PurePosixPath(value['path']).parents and str(out) != value['path'],
             'Output cannot contain an input')
        unique[value['path']] = value
    need(spec['sources'] == [unique[path] for path in sorted(unique)], 'Exact direct source union')
    package_root = PurePosixPath(spec['runtime_pythonpath_root'])
    need(str(package_root) == spec['runtime_pythonpath_root'] and package_root.anchor == '/'
         and package_root == PurePosixPath(spec['library_sources']['pycocotools']['path']).parent.parent,
         'Actual selected COCO package parent required')


class Control:
    """Real cooperative checks; Root must independently supervise/reap the group."""
    def __init__(self, psutil):
        self.psutil = psutil
        self.started = time.monotonic()
        self.peak_rss = self.minimum_available = 0
        self.samples = 0

    def check(self):
        process = self.psutil.Process()
        rss, available = process.memory_info().rss, int(self.psutil.virtual_memory().available)
        self.samples += 1
        self.peak_rss = max(self.peak_rss, rss)
        self.minimum_available = min(self.minimum_available or available, available)
        need(time.monotonic()-self.started < 1800 and process.cpu_affinity() == [0,1]
             and rss <= 16*1024**3 and available >= 16*1024**3,
             'Real finite CPU01/RSS16GiB/freeRAM16GiB cooperative resource checks')
        need(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and all(os.environ.get(key) == '2'
             for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')),
             'GPU-hidden two-thread environment')
        need(not any(name.split('.')[0] == 'torch' for name in sys.modules), 'CPU job must never import Torch')

    def closed(self, value):
        self.check()
        need(reference(value['path']) == value, 'Actual bytes/SHA source closure: '+value['path'])
        self.check()
        return Path(value['path'])

    def summary(self):
        self.check()
        return dict(samples=self.samples, peak_worker_RSS_bytes=self.peak_rss,
            minimum_available_RAM_bytes=self.minimum_available, elapsed_seconds=time.monotonic()-self.started,
            external_owned_group_live_supervision_self_certified=False)


def read(c, value):
    return json.loads(c.closed(value).read_bytes())


def load_definition(c, value, alias):
    path = c.closed(value)
    module = types.ModuleType(alias)
    module.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    return module


def write_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())
    return reference(path)


def same_content(left, right):
    return all(left[key] == right[key] for key in ('bytes','sha256'))


def run(c, spec):
    validate_spec(spec)
    for value in spec['sources']:
        c.closed(value)
    need(reference(__file__) == spec['source_refs']['worker'], 'This exact admitted worker')
    lease = read(c, spec['root_lease_ref'])
    need(lease.get('schema') == 'Root-SAMRefiner-full187-CPU-eval-lease-v01'
         and lease.get('status') == 'ROOT_GRANTED' and lease.get('run_id') == spec['run_id']
         and lease.get('worker_ref') == spec['source_refs']['worker']
         and lease.get('prelock') == spec['prelock']
         and lease.get('output_directory') == spec['output_directory']
         and lease.get('SAM_predictions_ref') == spec['input_refs']['SAM_predictions']
         and lease.get('SAM_report_ref') == spec['input_refs']['SAM_report']
         and lease.get('single_attempt') is True and lease.get('main_scientific_processes_reaped') is True
         and lease.get('exclusive_project_scientific_leaf') is True,
         'Exact Root CPU lease; ordinary JSON alone does not establish actual admission')
    need(platform.system() == 'Linux' and platform.machine() == 'aarch64'
         and platform.python_version() == '3.12.14', 'Selected actual DGX runtime')
    need(not any(name.split('.')[0] in ('numpy','pycocotools','torch') for name in sys.modules),
         'Fresh CPU interpreter before scientific imports')
    out = Path(spec['output_directory'])
    need(not out.exists() and out.parent.is_dir(), 'Fresh exclusive output with existing parent')
    out.mkdir(mode=0o700)
    report = dict(schema='SAMRefiner-full187-CPU-eval-report-v01', operation=OPERATION,
        run_id=spec['run_id'], status='STARTED', qualified=False, Root_external_admission_required=True,
        AP=False, AP_computation_status='NOT_STARTED', DEV=True, TEST=False, GT_JSON_loaded=False,
        new_training=False, formal_training_budget_credit=0, precision_improvement_established=False,
        innovation_established=False, TGRS_ready=False, automatic_retry=False, CPU_only=True,
        Torch_imported=False, full_binary_closure_claimed=False,
        original_input_refs=deepcopy(spec['input_refs']), source_refs=deepcopy(spec['sources']), results={})
    stat = Path('/proc/self/stat').read_text()
    report.update(pid=os.getpid(), pgid=os.getpgrp(), start_ticks=int(stat[stat.rfind(')')+2:].split()[19]),
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    exit_code, runtime, native, strong, refined, coco = 1, {}, None, None, None, None
    try:
        sys.path.insert(0, spec['runtime_pythonpath_root'])
        for role, name in LIBRARY_NAMES.items():
            value = importlib.import_module(name)
            need(reference(value.__file__) == spec['library_sources'][role], 'Actual selected module: '+role)
            runtime[role] = value
            c.check()
        need(runtime['psutil'] is c.psutil and str(runtime['numpy'].__version__) == '1.26.4'
             and str(c.psutil.__version__) == '6.1.1', 'Selected real NumPy/psutil versions')
        definitions = {role:load_definition(c,value,'_SAM_CPU_'+role)
            for role,value in spec['source_refs'].items() if role != 'worker'}
        context, old_context, legacy = [read(c,spec['input_refs'][role])
            for role in ('context','legacy_context','legacy_result')]
        need(same_content(context['legacy_context'],spec['input_refs']['legacy_context'])
             and context['image_ids'] == old_context['image_ids'], 'Original context bridge and full ordered roster')
        for current in (context,old_context):
            for old, actual in (('native_predictions','native_predictions'),('R_predictions','R_predictions'),
                                ('legacy_ROOT_result','legacy_result')):
                need(same_content(current[old],spec['input_refs'][actual]), 'Original direct input relocation: '+old)
        need(same_content(old_context['DEV'],spec['input_refs']['DEV'])
             and same_content(old_context['COCO_source'],spec['source_refs']['official'])
             and legacy['qualified'] is True, 'Original GT/helper and legacy result identity')
        native, strong, refined = [read(c,spec['input_refs'][role])
            for role in ('native_predictions','R_predictions','SAM_predictions')]
        sam_report = read(c,spec['input_refs']['SAM_report'])
        need(sam_report['schema'] == 'SAMRefiner-full187-shared-worker-report-v01'
             and sam_report['status'] == 'FULL187_AUTHOR_RETURNED_PENDING_ROOT_REVIEW'
             and sam_report['worker_exit_code'] == 0 and sam_report['images_completed'] == 187
             and sam_report['candidates_completed'] == 18700 and sam_report['GT_loaded'] is False
             and sam_report['output_predictions_ref'] == spec['input_refs']['SAM_predictions']
             and sam_report['global_order_and_nonsegmentation_metadata_preserved'] is True
             and sam_report['all100_candidates_submitted_each_image'] is True,
             'Actual full187 SAM output report; Root must separately admit its original execution')
        coco = runtime['coco_loader'].COCO(str(c.closed(spec['input_refs']['DEV'])))
        report['GT_JSON_loaded'] = True
        ids = list(coco.imgs)
        need(ids == context['image_ids'] and len(ids) == 187 and len(coco.anns) == 380
             and set(coco.cats) == {1}, 'Original full DEV187/380/category1 without annotation conversion')
        shapes = {image_id:[coco.imgs[image_id]['height'],coco.imgs[image_id]['width']] for image_id in ids}
        for predictions in (native,strong,refined):
            validate_population(native,predictions,ids,shapes)
        controls = dict(native_official=legacy['control_results']['native_official']['COCO'],
                        R=legacy['control_results']['strong_fixed_E24_Omega']['COCO'])
        report.update(images=187, annotations=380, candidates_per_method=18700,
            fixed_original_S0_class_order=True, inference_cost_source_ref=spec['input_refs']['SAM_report'],
            SAM_cost={key:sam_report[key] for key in ('elapsed_seconds','sampled_worker_peak_RSS_bytes',
                'CUDA_peak_allocated_bytes','model_load_count','speed_benchmark','background_load_locked') if key in sam_report})
        for label, predictions in (('native_official',native),('R',strong),('SAMRefiner',refined)):
            c.check()
            report.update(AP=None, AP_computation_status='STARTED_COMPLETION_UNKNOWN', current_method=label)
            metrics, per_iou, evaluation = definitions['eval_body'].official_metrics(
                runtime['numpy'],c,definitions['official'],runtime['cocoeval'].COCOeval,coco,predictions,ids)
            need(set(metrics) == COCO_KEYS, 'Complete original seven-key COCO result')
            report['results'][label] = dict(COCO=metrics, per_IoU=per_iou)
            if label in controls:
                need(metrics == controls[label], 'Exact original complete COCO12 reproduction: '+label)
            del evaluation
        report['differences'] = {name:delta_stats(report['results']['SAMRefiner']['COCO']['stats'],
            report['results'][label]['COCO']['stats']) for name,label in
            (('SAMRefiner_minus_native','native_official'),('SAMRefiner_minus_R','R'))}
        for value in spec['sources']:
            c.closed(value)
        report.update(AP=True, AP_computation_status='COMPLETE_PENDING_ROOT_ACCEPTANCE',
            original_native_R_complete_COCO12_exact=True, status='COMPLETE_PENDING_ROOT_ACCEPTANCE',
            no_shard_AP_average=True, standalone_FPS_claimed=False)
        exit_code = 0
    except BaseException as error:
        report.update(status='FAILED_PRESERVED_NO_RETRY', error_type=type(error).__name__,
            error=str(error), traceback=traceback.format_exc())
    finally:
        native = strong = refined = coco = None
        runtime.clear()
        gc.collect()
        try:
            report['resource_observations'] = c.summary()
        except BaseException as error:
            report.update(status='FAILED_RESOURCE_CLOSURE', resource_error=str(error))
            exit_code = 1
        report['worker_exit_code'] = exit_code
        write_json(out/'REPORT.json',report)
    return exit_code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',required=True)
    parser.add_argument('--spec-sha',required=True)
    args = parser.parse_args(argv)
    raw = Path(args.spec).read_bytes()
    need(hashlib.sha256(raw).hexdigest() == args.spec_sha, 'Actual fixed spec SHA')
    spec = json.loads(raw)
    validate_spec(spec)
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
        need(os.environ.get(key) == '2', 'Explicit two-thread environment')
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Explicit GPU-hidden CPU child')
    need(reference(spec['library_sources']['psutil']['path']) == spec['library_sources']['psutil'],
         'Admitted psutil before import')
    import psutil
    need(reference(psutil.__file__) == spec['library_sources']['psutil'], 'Actual psutil module')
    c = Control(psutil)
    c.check()
    return run(c,spec)


if __name__ == '__main__':
    raise SystemExit(main())
