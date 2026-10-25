"""ORACLE_DIAGNOSTIC_POSTHOC over each query's already exported single mask."""
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
import traceback
import types

METHODS = ('native_official','R','SAMRefiner')
SCHEMA = 'ROOT-SAMRefiner-full187-all100-oracle-coverage-spec-v01'
OPERATION = 'CPU_SAMREFINER_FULL187_ALL100_ORACLE_COVERAGE_V01'
LIMITS = dict(allowed_cpus=[0,1],threads=2,maximum_RSS_bytes=16*1024**3,
              minimum_available_RAM_bytes=16*1024**3,deadline_seconds=1800)
PRELOCK = dict(limits=LIMITS,analysis_status='ORACLE_DIAGNOSTIC_POSTHOC',methods=list(METHODS),
    expected_images=187,expected_annotations=380,queries_per_image=100,exported_masks_per_query=1,
    coverage_thresholds=[.5,.75,.9,.95],best_IoU_tie='FIRST_ORIGINAL_EXPORT_INDEX',
    score_rank_order='DESCENDING_ORIGINAL_S0_THEN_EXPORT_INDEX',
    GT_used_for_oracle_query_selection=True,GT_for_model_inference=False,
    query_identity_or_S0_modified=False,oracle_not_deployable_not_AP=True,
    author_internal_three_branches_considered=False,annotation_policy='ORIGINAL_DEV_UNCHANGED')
PINS = {
    'cpu_helpers':dict(bytes=18971,sha256='0c4e8a3794c1146ada0920e0eafb775da87aee55c75d72e04e7d19c2aa965b27'),
    'eval_body':dict(bytes=24972,sha256='0e4ec8c999b0e954ce6d588b3ffae97510be1051c4465f9d4510c3c2ec3b34bf'),
    'official':dict(bytes=49923,sha256='2357929fcd38ffc3020c5b8d204ee371da8da3972ff2e2e03398c8225c55e916'),
    'geometry_diagnostic':dict(bytes=13722,sha256='6f6669eb91d0842f7c23eb09b992624642d16c9abe8d145c258ec9aae5ff36ac'),
}


def need(ok,message):
    if not ok:
        raise ValueError(message)


def require_all100(values):
    need(type(values) is list and len(values) == 100, 'All100 original queries; no filtering')
    need(len({row['image_id'] for row in values}) == 1
         and all(type(row['query_id']) is int and row['category_id'] == 1
                 and type(row['score']) in (int,float) and math.isfinite(row['score'])
                 and 0 <= row['score'] <= 1 for row in values)
         and {row['query_id'] for row in values} == set(range(100)),
         'One image with every original query and finite unchanged S0')


def stable_score_ranks(values):
    require_all100(values)
    ranks = [None]*100
    for rank,index in enumerate(sorted(range(100),key=lambda index:(-values[index]['score'],index)),1):
        ranks[index] = rank
    return ranks


def best_export_position(ious):
    need(type(ious) is list and len(ious) == 100 and all(type(v) in (int,float)
         and math.isfinite(v) and 0 <= v <= 1 for v in ious), 'All100 finite official IoUs')
    return max(range(100),key=lambda index:ious[index])


def coverage_scope(GT_ids,native_matched):
    need(type(GT_ids) is list and len(GT_ids) == len(set(GT_ids)) == 380
         and set(native_matched) <= set(GT_ids), 'All380 original GT; never restrict to native matched subset')
    return dict(total_GT=380,native_matched_GT=len(native_matched),
                native_unmatched_or_ignored_GT=380-len(native_matched))


def bootstrap(value,alias):
    path = Path(value['path'])
    raw = path.read_bytes()
    need(len(raw) == value['bytes'] and hashlib.sha256(raw).hexdigest() == value['sha256'],
         'Exact original CPU helper before execution')
    module = types.ModuleType(alias)
    module.__file__ = str(path)
    exec(compile(raw,str(path),'exec'),module.__dict__)
    return module


def validate_spec(spec,h):
    need(type(spec) is dict and set(spec) == {'schema','operation','run_id','source_refs','library_sources',
         'input_refs','root_lease_ref','runtime_pythonpath_root','prelock','output_directory','sources'},
         'Exact independent oracle SPEC keys')
    need(spec['schema'] == SCHEMA and spec['operation'] == OPERATION
         and type(spec['run_id']) is str and spec['run_id'].startswith('SAMRefiner-full187-oracle-v01-')
         and spec['prelock'] == PRELOCK, 'Explicit posthoc oracle identity/settings')
    need(set(spec['source_refs']) == set(PINS)|{'worker'}
         and set(spec['library_sources']) == set(h.LIBRARY_NAMES)
         and set(spec['input_refs']) == h.INPUT_ROLES, 'Exact admitted roles')
    for role,pin in PINS.items():
        need(all(spec['source_refs'][role][key] == value for key,value in pin.items()),
             'Frozen helper source: '+role)
    out = PurePosixPath(spec['output_directory'])
    need(out.anchor == '/' and str(out) == spec['output_directory'] and '..' not in out.parts
         and str(out) != '/', 'Canonical output location')
    union = {}
    for value in list(spec['source_refs'].values())+list(spec['library_sources'].values())+list(spec['input_refs'].values())+[spec['root_lease_ref']]:
        h.strict_ref(value)
        need(value['path'] not in union or union[value['path']] == value, 'Conflicting input references')
        need(out not in PurePosixPath(value['path']).parents and str(out) != value['path'], 'Output cannot contain inputs')
        union[value['path']] = value
    need(spec['sources'] == [union[key] for key in sorted(union)], 'Exact sorted reference union')
    need(PurePosixPath(spec['runtime_pythonpath_root']) ==
         PurePosixPath(spec['library_sources']['pycocotools']['path']).parent.parent,
         'Selected COCO package parent')


def admit(c,spec,h):
    validate_spec(spec,h)
    for value in spec['sources']:
        c.closed(value)
    need(h.reference(__file__) == spec['source_refs']['worker'], 'This exact admitted oracle worker')
    lease = h.read(c,spec['root_lease_ref'])
    need(lease.get('schema') == 'Root-SAMRefiner-full187-all100-oracle-coverage-lease-v01'
         and lease.get('status') == 'ROOT_GRANTED' and lease.get('run_id') == spec['run_id']
         and lease.get('worker_ref') == spec['source_refs']['worker']
         and lease.get('prelock') == spec['prelock'] and lease.get('output_directory') == spec['output_directory']
         and lease.get('SAM_predictions_ref') == spec['input_refs']['SAM_predictions']
         and lease.get('SAM_report_ref') == spec['input_refs']['SAM_report']
         and lease.get('single_attempt') is True and lease.get('main_scientific_processes_reaped') is True
         and lease.get('exclusive_project_scientific_leaf') is True,
         'Root oracle lease; actual process qualification is external')
    need(platform.system() == 'Linux' and platform.machine() == 'aarch64'
         and platform.python_version() == '3.12.14', 'Admitted actual DGX CPU interpreter')
    need(not any(name.split('.')[0] in ('numpy','pycocotools','torch') for name in sys.modules),
         'Fresh interpreter before scientific imports')


def distribution(np,values):
    if not values:
        return dict(n=0,mean=None,median=None,q25=None,q75=None)
    a = np.asarray(values,dtype=np.float64)
    return dict(n=len(values),mean=float(np.mean(a)),median=float(np.median(a)),
                q25=float(np.percentile(a,25)),q75=float(np.percentile(a,75)))


def rank_distribution(np,rows):
    ranks = [row['S0_rank_1based'] for row in rows]
    return dict(statistics=distribution(np,ranks),bins={label:sum(lo <= rank <= hi for rank in ranks)
        for label,lo,hi in (('1',1,1),('2-5',2,5),('6-10',6,10),('11-20',11,20),('21-50',21,50),('51-100',51,100))})


def summarize(np,scope,records):
    groups = dict(all380=records,
        native_matched_nonignored=[row for row in records if row['native_COCO_matched_nonignored']],
        native_unmatched_nonignored=[row for row in records if not row['native_COCO_matched_nonignored'] and not row['native_COCO_effective_ignore']],
        native_effectively_ignored=[row for row in records if row['native_COCO_effective_ignore']])
    summaries = {}
    for name,group in groups.items():
        methods = {}
        for method in METHODS:
            values = [row['methods'][method] for row in group]
            thresholds = {label:sum(row['max_IoU'] >= threshold for row in values)
                for label,threshold in (('0.50',.5),('0.75',.75),('0.90',.9),('0.95',.95))}
            methods[method] = dict(GT_denominator=len(values),max_IoU=distribution(np,[row['max_IoU'] for row in values]),
                coverage_counts=thresholds,coverage_fractions={key:value/len(values) if values else None for key,value in thresholds.items()},
                zero_max_IoU_GT=sum(row['max_IoU'] == 0 for row in values),
                best_candidate_S0_rank=rank_distribution(np,values),
                best_candidate_S0_rank_given_max_IoU_ge0p50=rank_distribution(np,[row for row in values if row['max_IoU'] >= .5]))
        summaries[name] = dict(GT_denominator=len(group),methods=methods)
    return dict(schema='SAMRefiner-full187-all100-oracle-coverage-public-summary-v01',
        status='COMPLETE_PENDING_ROOT_ACCEPTANCE',qualified=False,analysis_status='ORACLE_DIAGNOSTIC_POSTHOC',
        scope=scope,groups=summaries,GT_used_for_oracle_query_selection=True,GT_for_model_inference=False,
        query_identity_or_S0_modified=False,original_AP_and_prediction_outputs_unchanged=True,
        oracle_not_deployable_not_AP=True,exported_masks_per_query=1,
        author_internal_three_branches_considered=False,
        IoU_semantics='Official mask.iou with original per-GT iscrowd; crowd entries use COCO overlap semantics',
        best_IoU_tie='FIRST_ORIGINAL_EXPORT_INDEX',score_rank_order='DESCENDING_ORIGINAL_S0_THEN_EXPORT_INDEX',
        native_control_COCO_recomputed=True,native_complete_COCO12_exact=True,new_method_AP_recomputed=False,
        TEST=False,new_training=False,innovation_established=False,TGRS_ready=False,
        private_GT_query_records_not_for_publication=True,
        interpretation_limit='Posthoc GT oracle over exported single masks; not deployable selection, not AP or one-to-one recall, not a bound on author internal branches, not a unique cause of AP loss or proof of core innovation.')


def run(c,spec,h,runtime):
    out = Path(spec['output_directory'])
    need(not out.exists() and out.parent.is_dir(), 'Fresh exclusive output')
    out.mkdir(mode=0o700)
    report = dict(schema='SAMRefiner-full187-all100-oracle-coverage-report-v01',run_id=spec['run_id'],
        operation=OPERATION,status='STARTED',qualified=False,Root_external_admission_required=True,
        analysis_status='ORACLE_DIAGNOSTIC_POSTHOC',GT_used_for_oracle_query_selection=True,
        GT_for_model_inference=False,query_identity_or_S0_modified=False,oracle_not_deployable_not_AP=True,
        GT_JSON_loaded=False,native_COCO_computation_status='NOT_STARTED',new_method_AP_recomputed=False,
        CPU_only=True,Torch_imported=False,new_training=False,formal_training_budget_credit=0,
        source_refs=deepcopy(spec['sources']),GT_completed=0,images_completed=0)
    stat = Path('/proc/self/stat').read_text()
    report.update(pid=os.getpid(),pgid=os.getpgrp(),start_ticks=int(stat[stat.rfind(')')+2:].split()[19]),
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    code,records = 1,[]
    native = strong = sam = coco = evaluation = None
    try:
        np = runtime['numpy']
        definitions = {role:h.load_definition(c,spec['source_refs'][role],'_SAM_ORACLE_'+role)
            for role in ('eval_body','official','geometry_diagnostic')}
        context,old,legacy = [h.read(c,spec['input_refs'][role]) for role in ('context','legacy_context','legacy_result')]
        need(h.same_content(context['legacy_context'],spec['input_refs']['legacy_context'])
             and context['image_ids'] == old['image_ids'], 'Original roster/context bridge')
        for container in (context,old):
            for source,actual in (('native_predictions','native_predictions'),('R_predictions','R_predictions'),('legacy_ROOT_result','legacy_result')):
                need(h.same_content(container[source],spec['input_refs'][actual]), 'Original content relocation')
        need(h.same_content(old['DEV'],spec['input_refs']['DEV'])
             and h.same_content(old['COCO_source'],spec['source_refs']['official']) and legacy['qualified'] is True,
             'Original GT/helper/reference identity')
        native,strong,sam = [h.read(c,spec['input_refs'][role]) for role in ('native_predictions','R_predictions','SAM_predictions')]
        sr = h.read(c,spec['input_refs']['SAM_report'])
        need(sr['schema'] == 'SAMRefiner-full187-shared-worker-report-v01'
             and sr['status'] == 'FULL187_AUTHOR_RETURNED_PENDING_ROOT_REVIEW' and sr['worker_exit_code'] == 0
             and sr['images_completed'] == 187 and sr['candidates_completed'] == 18700
             and sr['GT_loaded'] is False and sr['output_predictions_ref'] == spec['input_refs']['SAM_predictions'],
             'Completed full187 SAM output join; original admission remains Root responsibility')
        coco = runtime['coco_loader'].COCO(str(c.closed(spec['input_refs']['DEV'])))
        report['GT_JSON_loaded'] = True
        ids = list(coco.imgs)
        need(ids == context['image_ids'] and len(ids) == 187 and len(coco.anns) == 380 and set(coco.cats) == {1},
             'Original all187 images/all380 GT/category1')
        shapes = {image:[coco.imgs[image]['height'],coco.imgs[image]['width']] for image in ids}
        for values in (native,strong,sam):
            h.validate_population(native,values,ids,shapes)
        report['native_COCO_computation_status'] = 'STARTED_COMPLETION_UNKNOWN'
        metrics,unused,evaluation = definitions['eval_body'].official_metrics(
            np,c,definitions['official'],runtime['cocoeval'].COCOeval,coco,native,ids)
        report.update(native_COCO_computation_status='COMPLETE',native_control_COCO=metrics)
        need(metrics == legacy['control_results']['native_official']['COCO'], 'Exact native complete COCO12 calibration')
        for p in (evaluation.params,evaluation._paramsEval):
            need(set(p.imgIds) == set(ids) and list(p.catIds) == [1] and list(p.maxDets) == [1,10,100]
                 and np.array_equal(p.iouThrs,np.linspace(.5,.95,10)), 'Actual native full COCO parameters/.50 first')
        area = list(evaluation.params.areaRng[list(evaluation.params.areaRngLbl).index('all')])
        need(list(evaluation._paramsEval.areaRng[list(evaluation._paramsEval.areaRngLbl).index('all')]) == area,
             'Actually evaluated all-area range')
        dt = evaluation.cocoDt.anns
        need(set(dt) == set(range(1,18701)) and all(dt[i]['query_id'] == row['query_id']
             and dt[i]['image_id'] == row['image_id'] and float(dt[i]['score']).hex() == float(row['score']).hex()
             for i,row in enumerate(native,1)), 'Actual detection-ID to original query/S0 identity')
        pairs = definitions['geometry_diagnostic'].select_native_pairs(evaluation.evalImgs,dt,area)
        matched = {row['GT_annotation_id'] for row in pairs}
        ignored = {}
        for row in evaluation.evalImgs:
            if row is not None and list(row['aRng']) == area and row['maxDet'] == 100 and row['category_id'] == 1:
                for gt,ignore in zip(row['gtIds'],row['gtIgnore']):
                    need(int(gt) not in ignored, 'Original GT considered once in all-area scope')
                    ignored[int(gt)] = bool(ignore)
        need(set(ignored) == set(coco.anns), 'All380 GT considered, including native unmatched')
        scope = coverage_scope(list(coco.anns),matched)
        scope.update(total_DEV_images=187,candidates_per_method=18700,queries_per_image=100,
            native_effectively_ignored_GT=sum(ignored.values()),
            native_nonignored_unmatched_GT=sum(gt not in matched and not flag for gt,flag in ignored.items()),
            original_iscrowd_GT=sum(bool(ann.get('iscrowd',0)) for ann in coco.anns.values()),
            original_ignore_flag_GT=sum(bool(ann.get('ignore',0)) for ann in coco.anns.values()))
        seen = set()
        for ordinal,image in enumerate(ids):
            c.check()
            blocks = {method:values[ordinal*100:(ordinal+1)*100] for method,values in zip(METHODS,(native,strong,sam))}
            for block in blocks.values():
                require_all100(block)
            ranks = stable_score_ranks(blocks['native_official'])
            gt_ids = coco.getAnnIds(imgIds=[image],catIds=[1])
            annotations = [coco.anns[gt] for gt in gt_ids]
            need(not seen.intersection(gt_ids), 'No duplicate GT across original images')
            seen.update(gt_ids)
            image_records = [dict(image_id=image,GT_annotation_id=gt,dataset_index=ordinal,
                native_COCO_matched_nonignored=gt in matched,native_COCO_effective_ignore=ignored[gt],
                original_iscrowd=bool(ann.get('iscrowd',0)),original_ignore_flag=bool(ann.get('ignore',0)),methods={})
                for gt,ann in zip(gt_ids,annotations)]
            if annotations:
                gt_rles = [coco.annToRLE(ann) for ann in annotations]
                crowds = [int(bool(ann.get('iscrowd',0))) for ann in annotations]
                for method in METHODS:
                    matrix = runtime['mask_api'].iou([row['segmentation'] for row in blocks[method]],gt_rles,crowds)
                    need(isinstance(matrix,np.ndarray) and matrix.shape == (100,len(gt_ids))
                         and bool(np.isfinite(matrix).all()) and bool((matrix >= 0).all()) and bool((matrix <= 1).all()),
                         'Official all100 by all-original-GT overlap matrix')
                    for column,record in enumerate(image_records):
                        values = [float(value) for value in matrix[:,column]]
                        best = best_export_position(values)
                        row = blocks[method][best]
                        record['methods'][method] = dict(max_IoU=values[best],best_query_id=row['query_id'],
                            export_index_within_image=best,global_export_index=ordinal*100+best,
                            original_S0=row['score'],S0_rank_1based=ranks[best],
                            equal_best_IoU_candidates=sum(value == values[best] for value in values))
                    del matrix
            records.extend(image_records)
            report['images_completed'] += 1
            report['GT_completed'] += len(image_records)
        need(seen == set(coco.anns) and len(records) == 380, 'Every original GT exactly once in oracle denominator')
        for value in spec['sources']:
            c.closed(value)
        c.check()
        report['private_GT_records_ref'] = h.write_json(out/'PRIVATE_GT_ORACLE_RECORDS.json',records)
        report['public_summary_ref'] = h.write_json(out/'SUMMARY.json',summarize(np,scope,records))
        report.update(status='COMPLETE_PENDING_ROOT_ACCEPTANCE',native_complete_COCO12_exact=True)
        code = 0
    except BaseException as error:
        report.update(status='FAILED_PRESERVED_NO_RETRY',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
    finally:
        native = strong = sam = coco = evaluation = None
        gc.collect()
        try:
            report['resource_observations'] = c.summary()
        except BaseException as error:
            report.update(status='FAILED_RESOURCE_CLOSURE',resource_error=str(error)); code = 1
        report['worker_exit_code'] = code
        h.write_json(out/'REPORT.json',report)
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',required=True); parser.add_argument('--spec-sha',required=True)
    args = parser.parse_args(argv)
    raw = Path(args.spec).read_bytes()
    need(hashlib.sha256(raw).hexdigest() == args.spec_sha, 'Exact actual SPEC SHA')
    spec = json.loads(raw)
    value = spec['source_refs']['cpu_helpers']
    need(all(value[key] == pin for key,pin in PINS['cpu_helpers'].items()), 'Exact reviewed CPU helper')
    h = bootstrap(value,'_SAM_ORACLE_CPU_HELPERS')
    validate_spec(spec,h)
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and all(os.environ.get(key) == '2'
         for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')), 'GPU-hidden two-thread CPU environment')
    need(h.reference(spec['library_sources']['psutil']['path']) == spec['library_sources']['psutil'], 'Psutil bytes before import')
    import psutil
    need(h.reference(psutil.__file__) == spec['library_sources']['psutil'], 'Actual admitted psutil')
    c = h.Control(psutil)
    c.check(); admit(c,spec,h)
    runtime = {}
    sys.path.insert(0,spec['runtime_pythonpath_root'])
    for role,name in h.LIBRARY_NAMES.items():
        module = importlib.import_module(name)
        need(h.reference(module.__file__) == spec['library_sources'][role], 'Actual selected module: '+role)
        runtime[role] = module
        c.check()
    need(str(runtime['numpy'].__version__) == '1.26.4' and str(psutil.__version__) == '6.1.1'
         and runtime['psutil'] is c.psutil, 'Actual admitted NumPy/psutil versions')
    return run(c,spec,h,runtime)


if __name__ == '__main__':
    raise SystemExit(main())
