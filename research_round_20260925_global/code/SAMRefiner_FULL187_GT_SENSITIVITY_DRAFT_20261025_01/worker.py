"""Additional posthoc paired integer/local-float GT analysis; unbound candidate."""
import math
import json
import argparse
from copy import deepcopy
import gc
import hashlib
import importlib
import os
from pathlib import Path, PurePosixPath
import platform
import sys
import traceback
import types

RULE = 'int() truncation per polygon coordinate; bbox extrema max-minus-min; polygon area abs(shoelace)/2'
SOURCE_MEMBER = dict(bytes=1526458,sha256='4565548ce05136dba8072a4b1c1fb668f3d415e3c980ebfe05b5a8998dec1ac5')
METHODS = ('native_official','R','SAMRefiner')
GT_VERSIONS = ('INTEGER_PRIMARY','LOCAL_CANDIDATE_FLOAT')
SCHEMA = 'ROOT-SAMRefiner-full187-GT-sensitivity-spec-v01'
OPERATION = 'CPU_SAMREFINER_FULL187_PAIRED_INTEGER_FLOAT_GT_V01'
LIMITS = dict(allowed_cpus=[0,1],threads=2,maximum_RSS_bytes=16*1024**3,
              minimum_available_RAM_bytes=16*1024**3,deadline_seconds=1800)
PRELOCK = dict(limits=LIMITS,methods=list(METHODS),GT_versions=list(GT_VERSIONS),
    analysis_status='POSTHOC_ADDITIONAL_GT_SENSITIVITY',analysis_timing='ADDED_AFTER_BASE_SAM_RESULTS',
    expected_images=187,expected_annotations=380,expected_candidates=18700,
    primary_integer_GT_unchanged=True,query_identity_or_S0_modified=False,
    declared_mapping_rule=RULE,source_training_member_content=SOURCE_MEMBER,
    original_preregistration_claimed=False,first_E50_FULL_PLAIN_sensitivity_completed=False,
    official_archive_authenticated=False,GT_for_model_inference=False)
PINS = {
    'cpu_helpers':dict(bytes=18971,sha256='0c4e8a3794c1146ada0920e0eafb775da87aee55c75d72e04e7d19c2aa965b27'),
    'eval_body':dict(bytes=24972,sha256='0e4ec8c999b0e954ce6d588b3ffae97510be1051c4465f9d4510c3c2ec3b34bf'),
    'official':dict(bytes=49923,sha256='2357929fcd38ffc3020c5b8d204ee371da8da3972ff2e2e03398c8225c55e916'),
}


def need(ok,message):
    if not ok:
        raise ValueError(message)


def validate_GT_pair(integer,floating,*,expected_images=187,expected_annotations=380):
    canonical = lambda value: json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
    without = lambda value,fields: {key:item for key,item in value.items() if key not in fields}
    need(canonical(without(integer,{'annotations'})) == canonical(without(floating,{'annotations'})),
         'Image/category/top-level metadata must be identical')
    need(len(integer['images']) == len(floating['images']) == expected_images
         and len(integer['annotations']) == len(floating['annotations']) == expected_annotations,
         'Both complete original populations')
    ids = [row['id'] for row in integer['annotations']]
    need(len(set(ids)) == expected_annotations
         and ids == [row['id'] for row in floating['annotations']], 'Original unique annotation IDs and order')
    for original,restored in zip(integer['annotations'],floating['annotations']):
        need(canonical(without(original,{'segmentation','bbox','area'})) ==
             canonical(without(restored,{'segmentation','bbox','area'})), 'Only three geometry fields may differ')
        need(type(restored['bbox']) is list and len(restored['bbox']) == 4
             and all(type(v) in (int,float) and math.isfinite(v) for v in restored['bbox'])
             and type(restored['area']) in (int,float) and math.isfinite(restored['area'])
             and restored['area'] >= 0, 'Original finite float bbox/area retained without recomputation')
        seg = restored['segmentation']
        need(type(seg) is list and len(seg) == 1 and type(seg[0]) is list
             and len(seg[0]) >= 6 and len(seg[0])%2 == 0
             and all(type(v) in (int,float) and math.isfinite(v) for v in seg[0]),
             'One original finite polygon, no vertex/geometry search')
        points = [int(value) for value in seg[0]]
        x,y = points[0::2],points[1::2]
        area = abs(sum(x[i]*y[(i+1)%len(x)]-x[(i+1)%len(x)]*y[i] for i in range(len(x))))/2
        need(original['segmentation'] == [points] and original['bbox'] == [min(x),min(y),max(x)-min(x),max(y)-min(y)]
             and original['area'] == area, 'Exact existing int()/extrema/shoelace rule only')


def raster_counts(integer_area,float_area,intersection):
    need(all(type(value) is int and value >= 0 for value in (integer_area,float_area,intersection))
         and intersection <= min(integer_area,float_area), 'Valid exact official-raster pixel counts')
    union = integer_area+float_area-intersection
    return dict(integer_area_pixels=integer_area,float_area_pixels=float_area,intersection_pixels=intersection,
        union_pixels=union,XOR_pixels=integer_area+float_area-2*intersection,
        area_difference_float_minus_integer=float_area-integer_area,
        IoU=intersection/union if union else None,both_empty=union == 0,
        float_over_integer_area_ratio=float_area/integer_area if integer_area else None)


def same_content(left,right):
    return all(left[key] == right[key] for key in ('bytes','sha256'))


def validate_mapping_report(report,integer_ref,float_ref):
    need(report['schema'] == 'Root-SAMRefiner-DEV187-local-float-restoration-map-v1'
         and report['declared_rule'] == RULE and report['unique_mappings'] == 380
         and report['ambiguous'] == report['missing'] == 0
         and report['internal_IDs_population_preserved'] is True
         and report['official_archive_authenticated'] is False
         and report['posthoc_additional_diagnostic'] is True, 'Single-rule posthoc mapping scope')
    need(same_content(report['integer_DEV_ref'],integer_ref) and same_content(report['float_DEV_ref'],float_ref)
         and same_content(report['source_training_member_ref'],SOURCE_MEMBER),
         'Mapping refs may relocate by exact bytes/SHA; preserve original path strings')


def bootstrap(value,alias):
    path = Path(value['path'])
    raw = path.read_bytes()
    need(len(raw) == value['bytes'] and hashlib.sha256(raw).hexdigest() == value['sha256'], 'Closed CPU helper')
    module = types.ModuleType(alias)
    module.__file__ = str(path)
    exec(compile(raw,str(path),'exec'),module.__dict__)
    return module


def validate_spec(spec,h):
    need(type(spec) is dict and set(spec) == {'schema','operation','run_id','source_refs','library_sources',
         'input_refs','root_lease_ref','runtime_pythonpath_root','prelock','output_directory','sources'}, 'Exact SPEC keys')
    need(spec['schema'] == SCHEMA and spec['operation'] == OPERATION
         and type(spec['run_id']) is str and spec['run_id'].startswith('SAMRefiner-full187-GT-sensitivity-v01-')
         and spec['prelock'] == PRELOCK, 'Fixed additional sensitivity settings')
    need(set(spec['source_refs']) == set(PINS)|{'worker'}
         and set(spec['library_sources']) == set(h.LIBRARY_NAMES)
         and set(spec['input_refs']) == h.INPUT_ROLES|{'FLOAT_DEV','FLOAT_mapping_report'}, 'Exact source/input/library roles')
    for role,pin in PINS.items():
        need(all(spec['source_refs'][role][key] == value for key,value in pin.items()), 'Frozen source: '+role)
    out = PurePosixPath(spec['output_directory'])
    need(out.anchor == '/' and str(out) == spec['output_directory'] and '..' not in out.parts and str(out) != '/', 'Output path')
    union = {}
    for value in list(spec['source_refs'].values())+list(spec['library_sources'].values())+list(spec['input_refs'].values())+[spec['root_lease_ref']]:
        h.strict_ref(value)
        need(value['path'] not in union or union[value['path']] == value, 'Ref conflict')
        need(out not in PurePosixPath(value['path']).parents and str(out) != value['path'], 'Output/input overlap')
        union[value['path']] = value
    need(spec['sources'] == [union[key] for key in sorted(union)], 'Exact direct reference union')
    need(PurePosixPath(spec['runtime_pythonpath_root']) ==
         PurePosixPath(spec['library_sources']['pycocotools']['path']).parent.parent, 'Selected COCO parent')


def admit(c,spec,h):
    validate_spec(spec,h)
    for value in spec['sources']:
        c.closed(value)
    need(h.reference(__file__) == spec['source_refs']['worker'], 'Actual admitted worker')
    lease = h.read(c,spec['root_lease_ref'])
    need(lease.get('schema') == 'Root-SAMRefiner-full187-GT-sensitivity-lease-v01'
         and lease.get('status') == 'ROOT_GRANTED' and lease.get('run_id') == spec['run_id']
         and lease.get('worker_ref') == spec['source_refs']['worker']
         and lease.get('prelock') == spec['prelock'] and lease.get('output_directory') == spec['output_directory']
         and all(lease.get(role+'_ref') == spec['input_refs'][role]
                 for role in ('SAM_predictions','SAM_report','FLOAT_DEV','FLOAT_mapping_report'))
         and lease.get('single_attempt') is True and lease.get('main_scientific_processes_reaped') is True
         and lease.get('exclusive_project_scientific_leaf') is True, 'Root lease join; actual admission is external')
    need(platform.system() == 'Linux' and platform.machine() == 'aarch64'
         and platform.python_version() == '3.12.14', 'Admitted DGX runtime')
    need(not any(name.split('.')[0] in ('numpy','pycocotools','torch') for name in sys.modules), 'Fresh CPU interpreter')


def distribution(np,values):
    if not values:
        return dict(n=0,mean=None,median=None,q25=None,q75=None)
    a = np.asarray(values,dtype=np.float64)
    return dict(n=len(values),mean=float(np.mean(a)),median=float(np.median(a)),
                q25=float(np.percentile(a,25)),q75=float(np.percentile(a,75)))


def summarize_pixels(np,rows):
    need(len(rows) == 380, 'Every original annotation retained in pixel summary')
    areas_i = sum(row['integer_area_pixels'] for row in rows)
    areas_f = sum(row['float_area_pixels'] for row in rows)
    intersection = sum(row['intersection_pixels'] for row in rows)
    union = sum(row['union_pixels'] for row in rows)
    return dict(annotations=380,changed_raster_masks=sum(row['XOR_pixels'] > 0 for row in rows),
        unchanged_raster_masks=sum(row['XOR_pixels'] == 0 for row in rows),
        both_empty_count=sum(row['both_empty'] for row in rows),
        integer_empty_count=sum(row['integer_area_pixels'] == 0 for row in rows),
        float_empty_count=sum(row['float_area_pixels'] == 0 for row in rows),
        integer_area_pixels_sum=areas_i,float_area_pixels_sum=areas_f,
        XOR_pixels_sum=sum(row['XOR_pixels'] for row in rows),pooled_IoU=intersection/union if union else None,
        instance_IoU_defined=distribution(np,[row['IoU'] for row in rows if row['IoU'] is not None]),
        instance_IoU_undefined_both_empty_count=sum(row['IoU'] is None for row in rows),
        XOR_pixels=distribution(np,[row['XOR_pixels'] for row in rows]),
        raster_area_difference=distribution(np,[row['area_difference_float_minus_integer'] for row in rows]),
        float_over_integer_area_ratio_defined=distribution(np,[row['float_over_integer_area_ratio'] for row in rows
            if row['float_over_integer_area_ratio'] is not None]),
        area_ratio_undefined_integer_empty_count=sum(row['float_over_integer_area_ratio'] is None for row in rows),
        continuous_area_difference=distribution(np,[row['continuous_area_difference'] for row in rows]),
        bbox_changed_annotations=sum(row['bbox_changed'] for row in rows),
        polygon_changed_annotations=sum(row['polygon_changed'] for row in rows),
        denominator_policy='All380 retained; undefined both-empty IoUs and zero-denominator area ratios explicitly counted')


def run(c,spec,h,runtime):
    out = Path(spec['output_directory'])
    need(not out.exists() and out.parent.is_dir(), 'Fresh exclusive output')
    out.mkdir(mode=0o700)
    report = dict(schema='SAMRefiner-full187-paired-GT-sensitivity-report-v01',run_id=spec['run_id'],
        operation=OPERATION,status='STARTED',qualified=False,Root_external_admission_required=True,
        analysis_status='POSTHOC_ADDITIONAL_GT_SENSITIVITY',GT_JSON_loaded=False,
        AP_computation_status='NOT_STARTED',results={},GT_pairs_completed=0,
        first_E50_FULL_PLAIN_sensitivity_completed=False,original_preregistration_claimed=False,
        CPU_only=True,Torch_imported=False,GT_for_model_inference=False,new_training=False,
        formal_training_budget_credit=0,source_refs=deepcopy(spec['sources']))
    stat = Path('/proc/self/stat').read_text()
    report.update(pid=os.getpid(),pgid=os.getpgrp(),start_ticks=int(stat[stat.rfind(')')+2:].split()[19]),
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    code,rows = 1,[]
    native = strong = sam = integer = floating = coco_i = coco_f = evaluation = None
    try:
        np = runtime['numpy']
        definitions = {role:h.load_definition(c,spec['source_refs'][role],'_SAM_GT_'+role) for role in ('eval_body','official')}
        context,old,legacy = [h.read(c,spec['input_refs'][role]) for role in ('context','legacy_context','legacy_result')]
        need(h.same_content(context['legacy_context'],spec['input_refs']['legacy_context'])
             and context['image_ids'] == old['image_ids'], 'Original ordered roster/context bridge')
        for container in (context,old):
            for source,actual in (('native_predictions','native_predictions'),('R_predictions','R_predictions'),('legacy_ROOT_result','legacy_result')):
                need(h.same_content(container[source],spec['input_refs'][actual]), 'Original content relocation')
        need(h.same_content(old['DEV'],spec['input_refs']['DEV'])
             and h.same_content(old['COCO_source'],spec['source_refs']['official']) and legacy['qualified'] is True,
             'Original integer GT/helper/reference identity')
        mapping = h.read(c,spec['input_refs']['FLOAT_mapping_report'])
        validate_mapping_report(mapping,spec['input_refs']['DEV'],spec['input_refs']['FLOAT_DEV'])
        report['mapping_ref_relocations'] = {role:dict(original=mapping[role+'_ref'],actual=spec['input_refs'][actual])
            for role,actual in (('integer_DEV','DEV'),('float_DEV','FLOAT_DEV'))}
        report.update(mapping_source_ref=mapping['source_training_member_ref'],
            Root_mapper_execution_qualification_self_certified=False)
        integer,floating = [h.read(c,spec['input_refs'][role]) for role in ('DEV','FLOAT_DEV')]
        report['GT_JSON_loaded'] = True
        validate_GT_pair(integer,floating)
        native,strong,sam = [h.read(c,spec['input_refs'][role]) for role in ('native_predictions','R_predictions','SAM_predictions')]
        sr = h.read(c,spec['input_refs']['SAM_report'])
        need(sr['schema'] == 'SAMRefiner-full187-shared-worker-report-v01'
             and sr['status'] == 'FULL187_AUTHOR_RETURNED_PENDING_ROOT_REVIEW' and sr['worker_exit_code'] == 0
             and sr['images_completed'] == 187 and sr['candidates_completed'] == 18700
             and sr['GT_loaded'] is False and sr['output_predictions_ref'] == spec['input_refs']['SAM_predictions'],
             'Actual complete frozen SAM output join')
        coco_i,coco_f = [runtime['coco_loader'].COCO(str(c.closed(spec['input_refs'][role]))) for role in ('DEV','FLOAT_DEV')]
        ids = list(coco_i.imgs)
        need(ids == context['image_ids'] == list(coco_f.imgs) and len(ids) == 187
             and list(coco_i.anns) == list(coco_f.anns) and len(coco_i.anns) == 380
             and set(coco_i.cats) == set(coco_f.cats) == {1}, 'Same original IDs/order/population under both GTs')
        shapes = {image:[coco_i.imgs[image]['height'],coco_i.imgs[image]['width']] for image in ids}
        populations = dict(zip(METHODS,(native,strong,sam)))
        for values in populations.values():
            h.validate_population(native,values,ids,shapes)
        controls = dict(native_official=legacy['control_results']['native_official']['COCO'],R=legacy['control_results']['strong_fixed_E24_Omega']['COCO'])
        for version,coco in zip(GT_VERSIONS,(coco_i,coco_f)):
            report['results'][version] = {}
            for method in METHODS:
                c.check()
                report.update(AP_computation_status='STARTED_COMPLETION_UNKNOWN',current_GT=version,current_method=method)
                metrics,per_iou,evaluation = definitions['eval_body'].official_metrics(
                    np,c,definitions['official'],runtime['cocoeval'].COCOeval,coco,populations[method],ids)
                need(set(metrics) == h.COCO_KEYS, 'Complete official COCO12 dictionary')
                report['results'][version][method] = dict(COCO=metrics,per_IoU=per_iou)
                if version == 'INTEGER_PRIMARY' and method in controls:
                    need(metrics == controls[method], 'Exact original integer complete COCO12: '+method)
                del evaluation
        report['AP_computation_status'] = 'ALL_SIX_COMPLETE_PENDING_ROOT_ACCEPTANCE'
        for annotation_id in coco_i.anns:
            c.check()
            ai,af = coco_i.anns[annotation_id],coco_f.anns[annotation_id]
            shape = tuple(shapes[ai['image_id']])
            mi,mf = [np.ascontiguousarray(runtime['mask_api'].decode(coco.annToRLE(ann)) != 0,dtype=np.bool_)
                for coco,ann in ((coco_i,ai),(coco_f,af))]
            need(mi.shape == mf.shape == shape, 'Official original-grid GT rasters; no resizing')
            counts = raster_counts(int(np.count_nonzero(mi)),int(np.count_nonzero(mf)),int(np.count_nonzero(mi & mf)))
            need(counts['XOR_pixels'] == int(np.count_nonzero(mi != mf)), 'Exact official raster XOR identity')
            rows.append(dict(counts,annotation_id=annotation_id,image_id=ai['image_id'],
                continuous_area_difference=af['area']-ai['area'],bbox_changed=af['bbox'] != ai['bbox'],
                polygon_changed=af['segmentation'] != ai['segmentation']))
            report['GT_pairs_completed'] += 1
            del mi,mf
        need(len(rows) == 380, 'All380 GT pairs, no deletion')
        for value in spec['sources']:
            c.closed(value)
        c.check()
        paired = {method:h.delta_stats(report['results']['LOCAL_CANDIDATE_FLOAT'][method]['COCO']['stats'],
            report['results']['INTEGER_PRIMARY'][method]['COCO']['stats']) for method in METHODS}
        contrasts = {version:{left+'_minus_'+right:h.delta_stats(report['results'][version][left]['COCO']['stats'],
            report['results'][version][right]['COCO']['stats']) for left,right in
            (('R','native_official'),('SAMRefiner','native_official'),('SAMRefiner','R'))} for version in GT_VERSIONS}
        summary = dict(schema='SAMRefiner-full187-paired-GT-sensitivity-public-summary-v01',
            status='COMPLETE_PENDING_ROOT_ACCEPTANCE',qualified=False,
            analysis_status='POSTHOC_ADDITIONAL_GT_SENSITIVITY',analysis_timing='ADDED_AFTER_BASE_SAM_RESULTS',
            original_preregistration_claimed=False,first_E50_FULL_PLAIN_sensitivity_completed=False,
            internal_DEV_images=187,annotations=380,candidates_per_method=18700,TEST=False,
            GT_versions=list(GT_VERSIONS),methods=list(METHODS),results=report['results'],
            float_minus_integer_COCO12=paired,method_contrasts_by_GT=contrasts,GT_raster_sensitivity=summarize_pixels(np,rows),
            integer_native_R_complete_COCO12_exact=True,primary_integer_GT_unchanged=True,
            original_predictions_and_S0_unchanged=True,GT_for_model_inference=False,new_training=False,
            official_archive_authenticated=False,official_TEST_claimed=False,innovation_established=False,TGRS_ready=False,
            GT_choice_by_score=False,annotation_changes_include_polygon_bbox_and_continuous_area=True,
            interpretation_limit='Additional posthoc sensitivity to one local-candidate float restoration. It does not replace the first-E50 FULL/PLAIN policy, certify an official archive or physical contours, choose the better GT, or uniquely attribute AP changes to raster geometry rather than matching/area-bin changes.')
        report['private_GT_pixel_records_ref'] = h.write_json(out/'PRIVATE_GT_PIXEL_RECORDS.json',rows)
        report['public_summary_ref'] = h.write_json(out/'SUMMARY.json',summary)
        report.update(status='COMPLETE_PENDING_ROOT_ACCEPTANCE',integer_native_R_complete_COCO12_exact=True)
        code = 0
    except BaseException as error:
        report.update(status='FAILED_PRESERVED_NO_RETRY',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
    finally:
        native = strong = sam = integer = floating = coco_i = coco_f = evaluation = None
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
    need(hashlib.sha256(raw).hexdigest() == args.spec_sha, 'Exact SPEC SHA')
    spec = json.loads(raw)
    value = spec['source_refs']['cpu_helpers']
    need(all(value[key] == pin for key,pin in PINS['cpu_helpers'].items()), 'Reviewed CPU helper')
    h = bootstrap(value,'_SAM_GT_SENSITIVITY_CPU_HELPERS')
    validate_spec(spec,h)
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and all(os.environ.get(key) == '2'
         for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')), 'GPU-hidden two-thread CPU')
    need(h.reference(spec['library_sources']['psutil']['path']) == spec['library_sources']['psutil'], 'Psutil source before import')
    import psutil
    need(h.reference(psutil.__file__) == spec['library_sources']['psutil'], 'Actual psutil module')
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
         and runtime['psutil'] is c.psutil, 'Admitted actual NumPy/psutil versions')
    return run(c,spec,h,runtime)


if __name__ == '__main__':
    raise SystemExit(main())
