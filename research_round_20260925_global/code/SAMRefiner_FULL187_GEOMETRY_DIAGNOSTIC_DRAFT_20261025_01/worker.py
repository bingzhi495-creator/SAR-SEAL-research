"""Posthoc native-fixed-pair geometry; stdlib import, Root execution pending."""
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
import traceback
import types

METHODS = ('native_official','R','SAMRefiner')
SCHEMA = 'ROOT-SAMRefiner-full187-fixed-native-geometry-spec-v01'
OPERATION = 'CPU_SAMREFINER_FULL187_NATIVE_FIXED_PAIR_GEOMETRY_V01'
LIMITS = dict(allowed_cpus=[0,1],threads=2,maximum_RSS_bytes=16*1024**3,
              minimum_available_RAM_bytes=16*1024**3,deadline_seconds=1800)
PRELOCK = dict(limits=LIMITS,analysis_status='POSTHOC_EXPLORATORY',methods=list(METHODS),
    selection_model='native_official',selection_IoU=.5,selection_area='all',selection_maxDets=100,
    thresholds=[.5,.75,.9,.95],score_policy='ORIGINAL_NATIVE_S0_UNCHANGED',
    annotation_policy='ORIGINAL_DEV_UNCHANGED',rematching=False,GT_for_inference=False,
    Omega_used=False,expected_images=187,expected_annotations=380,expected_candidates=18700)
PINS = {
    'cpu_helpers':dict(bytes=18971,sha256='0c4e8a3794c1146ada0920e0eafb775da87aee55c75d72e04e7d19c2aa965b27'),
    'eval_body':dict(bytes=24972,sha256='0e4ec8c999b0e954ce6d588b3ffae97510be1051c4465f9d4510c3c2ec3b34bf'),
    'official':dict(bytes=49923,sha256='2357929fcd38ffc3020c5b8d204ee371da8da3972ff2e2e03398c8225c55e916'),
    'geometry_diagnostic':dict(bytes=13722,sha256='6f6669eb91d0842f7c23eb09b992624642d16c9abe8d145c258ec9aae5ff36ac'),
}


def need(ok,message):
    if not ok:
        raise ValueError(message)


def fixed_native_pairs(geometry,evalImgs,detections,all_area,all_GT_ids):
    pairs = geometry.select_native_pairs(evalImgs,detections,all_area)
    considered,ignored = set(),set()
    for row in evalImgs:
        if row is None or list(row['aRng']) != list(all_area) or row['maxDet'] != 100 or row['category_id'] != 1:
            continue
        for gt,ignore in zip(row['gtIds'],row['gtIgnore']):
            gt = int(gt)
            need(gt not in considered, 'An original GT may appear only once in all-area scope')
            considered.add(gt)
            if bool(ignore):
                ignored.add(gt)
    need(considered == set(all_GT_ids), 'Preserve every original GT in the denominator')
    selected = len(pairs)
    return dict(pairs=pairs,coverage=dict(total_GT=len(considered),selected_native_nonignored_GT=selected,
        ignored_GT=len(ignored),unmatched_nonignored_GT=len(considered)-len(ignored)-selected,
        selected_fraction_of_all_GT=selected/len(considered) if considered else None))


def require_original_grid(rle,shape):
    need(type(rle) is dict and set(rle) == {'size','counts'}
         and type(rle['size']) is list and rle['size'] == list(shape)
         and all(type(v) is int and v > 0 for v in rle['size'])
         and type(rle['counts']) in (str,bytes) and len(rle['counts']) > 0,
         'Exact original-grid compressed RLE; never resize/transpose to fit GT')


def same_query_values(pair,mappings):
    need(set(mappings) == set(METHODS), 'Exactly the three actual method populations')
    identity = (pair['image_id'],pair['query_id'])
    need(all(identity in mappings[method] for method in METHODS),
         'Missing original query is an error; no rematching or alternate output')
    return {method:mappings[method][identity] for method in METHODS}


def bootstrap_module(value,alias):
    path = Path(value['path'])
    raw = path.read_bytes()
    need(len(raw) == value['bytes'] and hashlib.sha256(raw).hexdigest() == value['sha256'],
         'Exact closed stdlib helper source before execution')
    module = types.ModuleType(alias)
    module.__file__ = str(path)
    exec(compile(raw,str(path),'exec'),module.__dict__)
    return module


def validate_spec(spec,h):
    need(type(spec) is dict and set(spec) == {'schema','operation','run_id','source_refs','library_sources',
         'input_refs','root_lease_ref','runtime_pythonpath_root','prelock','output_directory','sources'},
         'Exact independent geometry SPEC keys')
    need(spec['schema'] == SCHEMA and spec['operation'] == OPERATION
         and type(spec['run_id']) is str and spec['run_id'].startswith('SAMRefiner-full187-geometry-v01-'),
         'Independent exploratory geometry identity')
    need(set(spec['source_refs']) == set(PINS)|{'worker'}
         and set(spec['library_sources']) == set(h.LIBRARY_NAMES)
         and set(spec['input_refs']) == h.INPUT_ROLES and spec['prelock'] == PRELOCK,
         'Exact source/input roles and prospective diagnostic settings')
    for role,pin in PINS.items():
        need(all(spec['source_refs'][role][key] == value for key,value in pin.items()),
             'Frozen helper content: '+role)
    out = PurePosixPath(spec['output_directory'])
    need(out.anchor == '/' and str(out) == spec['output_directory'] and '..' not in out.parts
         and str(out) != '/', 'Fresh canonical output location')
    union = {}
    for value in list(spec['source_refs'].values())+list(spec['library_sources'].values())+list(spec['input_refs'].values())+[spec['root_lease_ref']]:
        h.strict_ref(value)
        need(value['path'] not in union or union[value['path']] == value, 'Conflicting actual refs')
        need(out not in PurePosixPath(value['path']).parents and str(out) != value['path'],
             'Output cannot contain an admitted input')
        union[value['path']] = value
    need(spec['sources'] == [union[key] for key in sorted(union)], 'Exact sorted direct reference union')
    need(PurePosixPath(spec['runtime_pythonpath_root']) ==
         PurePosixPath(spec['library_sources']['pycocotools']['path']).parent.parent,
         'Explicit selected COCO package parent')


def validate_native_evaluator(np,evaluation,native,coco,ids):
    params,frozen = evaluation.params,evaluation._paramsEval
    for value in (params,frozen):
        need(set(value.imgIds) == set(ids) and len(value.imgIds) == 187
             and list(value.catIds) == [1] and list(value.maxDets) == [1,10,100]
             and np.array_equal(value.iouThrs,np.linspace(.5,.95,10)),
             'Actual full187 native COCO parameters with IoU.50 first')
    labels = list(params.areaRngLbl)
    need(labels.count('all') == 1, 'Single all-area range')
    area = list(params.areaRng[labels.index('all')])
    need(list(frozen.areaRng[list(frozen.areaRngLbl).index('all')]) == area,
         'Actual evaluated all-area range unchanged')
    mapping = evaluation.cocoDt.anns
    need(set(mapping) == set(range(1,18701)), 'Native COCO ordinal detection IDs')
    for ordinal,row in enumerate(native,1):
        actual = mapping[ordinal]
        need(actual['id'] == ordinal and actual['image_id'] == row['image_id']
             and actual['query_id'] == row['query_id'] and actual['category_id'] == 1
             and float(actual['score']).hex() == float(row['score']).hex(),
             'Resolve each native detection ID to actual original query/S0')
    need(len(coco.anns) == 380, 'Original annotation denominator')
    return area


def whole_grid_counts(np,pred,GT):
    tp = int(np.count_nonzero(pred & GT))
    fp = int(np.count_nonzero(pred & ~GT))
    fn = int(np.count_nonzero(~pred & GT))
    area,union = tp+fn,tp+fp+fn
    need(area > 0 and union > 0, 'Selected nonempty original GT')
    return dict(TP=tp,FP=fp,FN=fn,GT_raster_area=area,prediction_area=tp+fp,
        IoU=float(np.float64(tp)/np.float64(union)),
        area_ratio=float(np.float64(tp+fp)/np.float64(area)),
        FP_over_GT=float(np.float64(fp)/np.float64(area)),
        FN_over_GT=float(np.float64(fn)/np.float64(area)),
        threshold_pass={label:100*tp >= threshold*union
            for label,threshold in (('0.50',50),('0.75',75),('0.90',90),('0.95',95))})


def describe(np,values):
    if not values:
        return dict(n=0,mean=None,median=None,q25=None,q75=None)
    array = np.asarray(values,dtype=np.float64)
    return dict(n=len(values),mean=float(np.mean(array)),median=float(np.median(array)),
        q25=float(np.percentile(array,25)),q75=float(np.percentile(array,75)))


def summarize(np,coverage,records):
    methods = {}
    for method in METHODS:
        values = [row['methods'][method] for row in records]
        totals = {key:sum(row[key] for row in values) for key in ('TP','FP','FN')}
        union = sum(totals.values())
        methods[method] = dict(matched_instances=len(values),pixel_totals=totals,
            pooled_IoU=totals['TP']/union if union else None,
            IoU=describe(np,[row['IoU'] for row in values]),
            area_ratio=describe(np,[row['area_ratio'] for row in values]),
            FP_over_GT=describe(np,[row['FP_over_GT'] for row in values]),
            FN_over_GT=describe(np,[row['FN_over_GT'] for row in values]),
            threshold_pass_counts={key:sum(row['threshold_pass'][key] for row in values)
                for key in ('0.50','0.75','0.90','0.95')})
    differences = {method+'_minus_'+baseline:describe(np,[
        row['methods'][method]['IoU']-row['methods'][baseline]['IoU'] for row in records])
        for method,baseline in (('R','native_official'),('SAMRefiner','native_official'),('SAMRefiner','R'))}
    return dict(schema='SAMRefiner-full187-native-fixed-pair-geometry-public-summary-v01',
        status='COMPLETE_PENDING_ROOT_ACCEPTANCE',qualified=False,
        analysis_status='POSTHOC_EXPLORATORY',scope=dict(coverage,total_DEV_images=187,
            total_natural_queries=18700,selection_model='native_official',selection_IoU=.5,
            selection_area='all',selection_maxDets=100,conditioned_on_native_matched_nonignored_GT=True,
            unmatched_and_ignored_GT_excluded_from_geometry=True),methods=methods,
        paired_IoU_differences=differences,rematching=False,GT_for_inference=False,
        original_S0_class_order_preserved=True,Omega_used=False,
        native_control_COCO_recomputed=True,native_complete_COCO12_exact=True,
        new_method_AP_recomputed=False,TEST=False,new_training=False,
        innovation_established=False,TGRS_ready=False,
        private_pair_records_not_for_publication=True,standalone_FPS_claimed=False,
        interpretation_limit='Posthoc conditional geometry on native IoU.50 matches only; not all380-GT geometry, not new-method AP, not a unique explanation of AP loss, not physical contour truth or core innovation.')


def run(c,spec,h):
    validate_spec(spec,h)
    for value in spec['sources']:
        c.closed(value)
    need(h.reference(__file__) == spec['source_refs']['worker'], 'Execute the exact admitted geometry worker')
    lease = h.read(c,spec['root_lease_ref'])
    need(lease.get('schema') == 'Root-SAMRefiner-full187-fixed-native-geometry-lease-v01'
         and lease.get('status') == 'ROOT_GRANTED' and lease.get('run_id') == spec['run_id']
         and lease.get('worker_ref') == spec['source_refs']['worker']
         and lease.get('prelock') == spec['prelock'] and lease.get('output_directory') == spec['output_directory']
         and lease.get('SAM_predictions_ref') == spec['input_refs']['SAM_predictions']
         and lease.get('SAM_report_ref') == spec['input_refs']['SAM_report']
         and lease.get('single_attempt') is True and lease.get('main_scientific_processes_reaped') is True
         and lease.get('exclusive_project_scientific_leaf') is True,
         'Exact Root exploratory lease; actual process admission remains external')
    need(platform.system() == 'Linux' and platform.machine() == 'aarch64'
         and platform.python_version() == '3.12.14', 'Selected actual DGX CPU interpreter')
    need(not any(name.split('.')[0] in ('numpy','pycocotools','torch') for name in sys.modules),
         'Fresh CPU worker before scientific imports')
    out = Path(spec['output_directory'])
    need(not out.exists() and out.parent.is_dir(), 'Fresh exclusive output')
    out.mkdir(mode=0o700)
    report = dict(schema='SAMRefiner-full187-fixed-native-geometry-report-v01',operation=OPERATION,
        run_id=spec['run_id'],status='STARTED',qualified=False,Root_external_admission_required=True,
        analysis_status='POSTHOC_EXPLORATORY',GT_JSON_loaded=False,GT_for_inference=False,
        native_COCO_computation_status='NOT_STARTED',new_method_AP_recomputed=False,
        CPU_only=True,Torch_imported=False,new_training=False,formal_training_budget_credit=0,
        rematching=False,Omega_used=False,source_refs=deepcopy(spec['sources']),pairs_completed=0)
    stat = Path('/proc/self/stat').read_text()
    report.update(pid=os.getpid(),pgid=os.getpgrp(),start_ticks=int(stat[stat.rfind(')')+2:].split()[19]),
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    exit_code,runtime,records = 1,{},[]
    native = strong = sam = coco = evaluation = None
    try:
        sys.path.insert(0,spec['runtime_pythonpath_root'])
        for role,name in h.LIBRARY_NAMES.items():
            module = importlib.import_module(name)
            need(h.reference(module.__file__) == spec['library_sources'][role], 'Actual admitted module: '+role)
            runtime[role] = module
            c.check()
        np = runtime['numpy']
        need(runtime['psutil'] is c.psutil and str(np.__version__) == '1.26.4'
             and str(c.psutil.__version__) == '6.1.1', 'Admitted actual NumPy/psutil runtime')
        definitions = {role:h.load_definition(c,spec['source_refs'][role],'_SAM_GEO_'+role)
            for role in ('eval_body','official','geometry_diagnostic')}
        context,old_context,legacy = [h.read(c,spec['input_refs'][role])
            for role in ('context','legacy_context','legacy_result')]
        need(h.same_content(context['legacy_context'],spec['input_refs']['legacy_context'])
             and context['image_ids'] == old_context['image_ids'], 'Original roster/context bridge')
        for container in (context,old_context):
            for old,actual in (('native_predictions','native_predictions'),('R_predictions','R_predictions'),
                               ('legacy_ROOT_result','legacy_result')):
                need(h.same_content(container[old],spec['input_refs'][actual]), 'Exact original direct content relocation')
        need(h.same_content(old_context['DEV'],spec['input_refs']['DEV'])
             and h.same_content(old_context['COCO_source'],spec['source_refs']['official'])
             and legacy['qualified'] is True, 'Original GT/helper/reference identity')
        native,strong,sam = [h.read(c,spec['input_refs'][role])
            for role in ('native_predictions','R_predictions','SAM_predictions')]
        sam_report = h.read(c,spec['input_refs']['SAM_report'])
        need(sam_report['schema'] == 'SAMRefiner-full187-shared-worker-report-v01'
             and sam_report['status'] == 'FULL187_AUTHOR_RETURNED_PENDING_ROOT_REVIEW'
             and sam_report['worker_exit_code'] == 0 and sam_report['images_completed'] == 187
             and sam_report['candidates_completed'] == 18700 and sam_report['GT_loaded'] is False
             and sam_report['output_predictions_ref'] == spec['input_refs']['SAM_predictions'],
             'Actual completed full187 SAM report joined to frozen output; Root separately admits it')
        coco = runtime['coco_loader'].COCO(str(c.closed(spec['input_refs']['DEV'])))
        report['GT_JSON_loaded'] = True
        ids = list(coco.imgs)
        need(ids == context['image_ids'] and len(ids) == 187 and len(coco.anns) == 380
             and set(coco.cats) == {1}, 'Original full DEV187/380/category1')
        shapes = {image:[coco.imgs[image]['height'],coco.imgs[image]['width']] for image in ids}
        mappings = {}
        for method,values in zip(METHODS,(native,strong,sam)):
            h.validate_population(native,values,ids,shapes)
            mappings[method] = {(row['image_id'],row['query_id']):row for row in values}
        report['native_COCO_computation_status'] = 'STARTED_COMPLETION_UNKNOWN'
        metrics,unused,evaluation = definitions['eval_body'].official_metrics(
            np,c,definitions['official'],runtime['cocoeval'].COCOeval,coco,native,ids)
        report.update(native_COCO_computation_status='COMPLETE',native_control_COCO=metrics)
        need(metrics == legacy['control_results']['native_official']['COCO'],
             'Exact complete native COCO12 calibration before geometry')
        area = validate_native_evaluator(np,evaluation,native,coco,ids)
        geometry = definitions['geometry_diagnostic']
        fixed = fixed_native_pairs(geometry,evaluation.evalImgs,evaluation.cocoDt.anns,area,set(coco.anns))
        report['coverage'] = fixed['coverage']
        image_ordinals = {image:index for index,image in enumerate(ids)}
        for pair in fixed['pairs']:
            c.check()
            image = pair['image_id']
            ann = coco.anns[pair['GT_annotation_id']]
            need(ann['image_id'] == image and ann['category_id'] == 1, 'Actual fixed-pair GT association')
            shape = shapes[image]
            GT = np.ascontiguousarray(coco.annToMask(ann) != 0,dtype=np.bool_)
            need(GT.shape == tuple(shape) and bool(GT.any()), 'Actual nonempty original-raster GT')
            values = same_query_values(pair,mappings)
            record = dict(pair,dataset_index=image_ordinals[image],methods={})
            for method in METHODS:
                require_original_grid(values[method]['segmentation'],shape)
                pred = geometry._decode(np,runtime['mask_api'],values[method]['segmentation'],shape)
                counts = whole_grid_counts(np,pred,GT)
                if method == 'native_official':
                    need(2*counts['TP'] >= counts['TP']+counts['FP']+counts['FN'],
                         'Selected native mask must reproduce its IoU.50 match')
                record['methods'][method] = counts
                del pred
            records.append(record)
            report['pairs_completed'] += 1
            del GT
        for value in spec['sources']:
            c.closed(value)
        c.check()
        summary = summarize(np,fixed['coverage'],records)
        report['private_pair_records_ref'] = h.write_json(out/'PRIVATE_PAIR_RECORDS.json',records)
        report['public_summary_ref'] = h.write_json(out/'SUMMARY.json',summary)
        report.update(status='COMPLETE_PENDING_ROOT_ACCEPTANCE',native_complete_COCO12_exact=True)
        exit_code = 0
    except BaseException as error:
        report.update(status='FAILED_PRESERVED_NO_RETRY',error_type=type(error).__name__,
            error=str(error),traceback=traceback.format_exc())
    finally:
        native = strong = sam = coco = evaluation = None
        runtime.clear()
        gc.collect()
        try:
            report['resource_observations'] = c.summary()
        except BaseException as error:
            report.update(status='FAILED_RESOURCE_CLOSURE',resource_error=str(error))
            exit_code = 1
        report['worker_exit_code'] = exit_code
        h.write_json(out/'REPORT.json',report)
    return exit_code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',required=True)
    parser.add_argument('--spec-sha',required=True)
    args = parser.parse_args(argv)
    raw = Path(args.spec).read_bytes()
    need(hashlib.sha256(raw).hexdigest() == args.spec_sha, 'Exact actual SPEC SHA')
    spec = json.loads(raw)
    helper_ref = spec['source_refs']['cpu_helpers']
    need(all(helper_ref[key] == value for key,value in PINS['cpu_helpers'].items()),
         'Fixed previously reviewed CPU helper content')
    h = bootstrap_module(helper_ref,'_SAM_GEOMETRY_CPU_HELPERS')
    validate_spec(spec,h)
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and all(os.environ.get(key) == '2'
         for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')),
         'Explicit GPU-hidden two-thread CPU environment')
    need(h.reference(spec['library_sources']['psutil']['path']) == spec['library_sources']['psutil'],
         'Admitted psutil source before import')
    import psutil
    need(h.reference(psutil.__file__) == spec['library_sources']['psutil'], 'Actual admitted psutil')
    c = h.Control(psutil)
    c.check()
    return run(c,spec,h)


if __name__ == '__main__':
    raise SystemExit(main())

