"""UNBOUND HRSID population pixel/raster checker; only stdlib imports.

Authoring tests use fabricated metadata/counters, never successful image/mask
substitutes. Future genuine modules, U42 checker and source/resource ownership
must be injected by the original caller. No runner, model, AP or file discovery.
"""
import hashlib
import json
import math
from pathlib import Path

EXPECTED={'train':dict(images=3642,annotations=11047),'test':dict(images=1962,annotations=5922)}
SPLITS=('train','test')
COUNTERS=('declared_images','declared_annotations','processed_images','recorded_images',
    'original_annotations_seen','image_checks_passed','image_checks_failed','pixel_decode_passed_images',
    'pixel_decode_failed_images','empty_annotation_images','raster_attempted_annotations',
    'raster_passed_annotations','raster_failed_annotations','annotation_metadata_failed_count','not_attempted_annotations')


def _need(ok,message):
    if not ok:raise ValueError(message)


def _strict_ref(value):
    _need(type(value) is dict and set(value)=={'path','bytes','sha256'}
          and type(value['path']) is str and Path(value['path']).is_absolute()
          and type(value['bytes']) is int and value['bytes']>=0
          and type(value['sha256']) is str and len(value['sha256'])==64
          and set(value['sha256'])<=set('0123456789abcdef'),'Exact original absolute strict reference')
    return value


def _path_key(path):
    return str(Path(path).resolve()).replace('\\','/').casefold()


def decode_metadata_json(raw):
    """Do not let duplicate JSON keys silently replace an original declaration."""
    def object_pairs(pairs):
        result={}
        for key,value in pairs:
            _need(key not in result,'Duplicate JSON key in original metadata: '+key)
            result[key]=value
        return result
    def invalid_constant(value):
        raise ValueError('Nonfinite JSON constant in original metadata: '+value)
    return json.loads(raw,object_pairs_hook=object_pairs,parse_constant=invalid_constant)


def validate_population_metadata(context,documents):
    """Pure original split metadata; IDs are local to each split, never global."""
    _need(type(context) is dict and set(context)=={'schema','image_root','annotations','image_refs_by_filename'}
          and context['schema']=='U43-HRSID-official-split-pixel-mask-check-context-v1',
          'Exact explicit original HRSID context, without discovery or merged annotations')
    root=Path(context['image_root'])
    _need(root.is_absolute() and type(context['annotations']) is dict
          and set(context['annotations'])==set(SPLITS) and type(documents) is dict
          and set(documents)==set(SPLITS),'Two separate explicit original annotation documents')
    for value in context['annotations'].values():_strict_ref(value)
    _need(len({_path_key(value['path']) for value in context['annotations'].values()})==2,
          'Separate original train/test JSON references')
    refs=context['image_refs_by_filename'];_need(type(refs) is dict,'Explicit original JPEG reference mapping')
    _need(len(refs)==5604,'Exactly the original5604 admitted JPEG references')
    plan=dict(splits={});all_names=[]
    for split in SPLITS:
        document=documents[split]
        _need(type(document) is dict and all(type(document[key]) is list for key in ('images','annotations','categories')),
              'Original COCO split lists')
        _need(len(document['categories'])==1 and type(document['categories'][0]['id']) is int
              and document['categories'][0]['id']==1,'Original single ship category1')
        _need(len(document['images'])==EXPECTED[split]['images']
              and len(document['annotations'])==EXPECTED[split]['annotations'],
              'Exact original split counts; never substitute the merged16951 version')
        images=document['images'];by_image={}
        for image in images:
            _need(type(image) is dict and type(image['id']) is int and type(image['width']) is int
                  and type(image['height']) is int and (image['width'],image['height'])==(800,800)
                  and type(image['file_name']) is str and Path(image['file_name']).name==image['file_name']
                  and Path(image['file_name']).suffix.lower()=='.jpg','Original integer ID/basename/800 JPEG declaration')
            name=image['file_name'];_need(name in refs,'Original filename has one explicitly admitted JPEG reference')
            reference=_strict_ref(refs[name])
            _need(_path_key(reference['path'])==_path_key(root/name),'Exact original filename/root/source association')
            _need(image['id'] not in by_image,'Duplicate local image ID within one original split')
            by_image[image['id']]=[];all_names.append(name)
        annotation_ids=set()
        for annotation in document['annotations']:
            _need(type(annotation) is dict and type(annotation['id']) is int
                  and type(annotation['image_id']) is int and annotation['image_id'] in by_image
                  and type(annotation['category_id']) is int and annotation['category_id']==1,
                  'Original local annotation/image ID and category association')
            _need(annotation['id'] not in annotation_ids,'Duplicate local annotation ID within one original split')
            annotation_ids.add(annotation['id'])
            by_image[annotation['image_id']].append(annotation)
        plan['splits'][split]=dict(images=images,annotations_by_image=by_image,
            declared_images=len(images),declared_annotations=len(document['annotations']))
    _need(len(set(all_names))==len(all_names) and set(all_names)==set(refs),
          'No cross-split filename overlap; exact admitted original JPEG union')
    _need(len({_path_key(_strict_ref(value)['path']) for value in refs.values()})==len(refs),
          'No path aliases masquerading as distinct original filenames')
    return plan


def completion_status(summaries):
    """Pure counter contract, not independent evidence of image/mask execution."""
    _need(type(summaries) is dict and set(summaries)==set(SPLITS),'Both original split summaries')
    complete=True
    for split in SPLITS:
        row=summaries[split];n=EXPECTED[split]['images'];g=EXPECTED[split]['annotations']
        _need(all(type(row[key]) is int and row[key]>=0 for key in COUNTERS),'Nonnegative integer actual counters')
        complete=complete and (row['declared_images']==row['processed_images']==row['recorded_images']
            ==row['image_checks_passed']==row['pixel_decode_passed_images']==n
            and row['declared_annotations']==row['original_annotations_seen']==row['raster_attempted_annotations']
            ==row['raster_passed_annotations']==g
            and row['image_checks_failed']==row['pixel_decode_failed_images']==row['raster_failed_annotations']
            ==row['annotation_metadata_failed_count']==row['not_attempted_annotations']==0)
    totals={key:sum(summaries[split][key] for split in SPLITS) for key in COUNTERS}
    visited=all(summaries[split]['processed_images']==EXPECTED[split]['images'] for split in SPLITS)
    audited=all(summaries[split]['recorded_images']==EXPECTED[split]['images'] for split in SPLITS)
    conserved=(totals['original_annotations_seen']==16969 and totals['raster_passed_annotations']
        +totals['raster_failed_annotations']+totals['annotation_metadata_failed_count']
        +totals['not_attempted_annotations']==16969)
    return dict(status='INPUT_CHECK_COMPLETE_PENDING_ROOT' if complete else 'INPUT_CHECK_FAILED_PENDING_ROOT',
        input_check_complete=bool(complete),full_image_visitation=visited,full_private_audit_coverage=audited,
        annotation_count_conserved=conserved,totals=totals)


def _encoded_duplicate_summary(plan,references):
    groups={}
    for split in SPLITS:
        for image in plan['splits'][split]['images']:
            reference=references[image['file_name']]
            group=groups.setdefault(reference['sha256'],dict(train=0,test=0,declared_lengths=set()))
            group[split]+=1;group['declared_lengths'].add(reference['bytes'])
    duplicate=[g for g in groups.values() if g['train']+g['test']>1]
    cross=[g for g in groups.values() if g['train'] and g['test']]
    return dict(grouping='Original strict-reference encoded SHA256; no pixel or scene identity inference',
        declared_images=5604,unique_encoded_SHA_groups=len(groups),duplicate_encoded_SHA_groups=len(duplicate),
        images_in_duplicate_SHA_groups=sum(g['train']+g['test'] for g in duplicate),
        duplicate_image_occurrences=sum(g['train']+g['test']-1 for g in duplicate),
        cross_split_SHA_groups=len(cross),train_images_in_cross_split_SHA_groups=sum(g['train'] for g in cross),
        test_images_in_cross_split_SHA_groups=sum(g['test'] for g in cross),
        within_split_duplicate_SHA_groups={s:sum(g[s]>1 for g in groups.values()) for s in SPLITS},
        within_split_duplicate_image_occurrences={s:sum(max(g[s]-1,0) for g in groups.values()) for s in SPLITS},
        same_SHA_conflicting_declared_length_groups=sum(len(g['declared_lengths'])>1 for g in groups.values()),
        images_removed=0,annotations_changed=0,duplicate_groups_independently_rehashed_here=False,
        annotation_semantic_disagreement_inferred=False,blind_TEST_or_scene_independence_inferred=False)


def _statistic():
    return dict(count=0,sum=0.0,absolute_sum=0.0,minimum=None,maximum=None,mean=None,absolute_mean=None,nonzero_count=0)


def _add_statistic(statistic,value):
    _need(type(value) in (int,float) and math.isfinite(value),'Finite scalar geometry-difference measurement')
    statistic['count']+=1;statistic['sum']+=value;statistic['absolute_sum']+=abs(value)
    _need(math.isfinite(statistic['sum']) and math.isfinite(statistic['absolute_sum']),'Finite aggregate geometry differences')
    statistic['minimum']=value if statistic['minimum'] is None else min(statistic['minimum'],value)
    statistic['maximum']=value if statistic['maximum'] is None else max(statistic['maximum'],value)
    statistic['nonzero_count']+=int(value!=0)
    statistic['mean']=statistic['sum']/statistic['count']
    statistic['absolute_mean']=statistic['absolute_sum']/statistic['count']


def _new_split_summary():
    result={key:0 for key in COUNTERS}
    result['EXIF_policy_counts']=dict(absent_no_transform_applied=0,declared_1_no_transform_applied=0,
        no_policy_recorded=0,EXIF_rejection_or_read_failure_images=0)
    result['geometry_difference_statistics']=dict(
        scope='Only successfully rasterized original instances; differences do not decide semantic alignment',
        declared_area=_statistic(),raster_area=_statistic(),area_difference_raster_minus_declared=_statistic(),
        bbox_difference_raster_minus_declared=[_statistic() for _ in range(4)])
    result['out_of_canvas_diagnostics']=dict(annotations_with_available_counts=0,annotations_without_available_counts=0,
        annotations_with_outside_coordinates=0,out_of_canvas_coordinate_count=0,out_of_canvas_component_count=0,
        outside_rule='coordinate<0 or coordinate>800; 800 is not outside',rejection_based_on_counts=False)
    return result


def _validate_image_result(result,image,annotations,reference):
    _need(type(result) is dict and result['schema']=='U42-single-image-pixel-mask-compatibility-pending-v1'
          and result['status'] in ('PASSED','FAILED') and result['image_id']==image['id']
          and result['image_ref']==reference and type(result['annotation_count']) is int
          and result['annotation_count']==len(annotations) and type(result['annotations']) is list
          and len(result['annotations'])==len(annotations),'Exact injected U42 original image/instance result ABI')
    _need(result['qualified'] is False and result['semantic_alignment_proven'] is False
          and result['population_qualified'] is False and result['new_training'] is False and result['AP'] is False
          and result['raw_pixels_or_masks_returned'] is False and result['pixel_array_released'] is True
          and result['per_annotation_arrays_released'] is True,'Only released, unqualified U42 data compatibility output')
    _need(type(result['pixel_decode_complete']) is bool and result['orientation_policy'] in
        (None,'absent_no_transform_applied','declared_1_no_transform_applied'),'Original explicit pixel/EXIF policy')
    passed=failed=metadata_failed=not_attempted=0
    for ordinal,(row,annotation) in enumerate(zip(result['annotations'],annotations)):
        _need(type(row) is dict and row['ordinal']==ordinal and row['annotation_id']==annotation['id']
              and row['status'] in ('PASSED','FAILED','NOT_ATTEMPTED'),'Original local ann ID/order retained; no component splitting')
        if row['status']=='PASSED':
            _need(row['error'] is None and row['iscrowd']==annotation['iscrowd']
                  and row['component_count']==len(annotation['segmentation']),
                  'Original polygon grouping/crowd retained by every passed annotation')
            passed+=1
        elif row['status']=='NOT_ATTEMPTED':
            _need(row['error'] is None,'Not-attempted metadata does not hide an error')
            not_attempted+=1
        else:
            _need(type(row['error']) is dict and type(row['error']['stage']) is str,'Retained explicit original annotation failure')
            if row['error']['stage']=='ANNOTATION_METADATA':metadata_failed+=1
            else:failed+=1
    actual=dict(raster_attempted_annotations=passed+failed,raster_passed_annotations=passed,
        raster_failed_annotations=failed,annotation_metadata_failed_count=metadata_failed,not_attempted_annotations=not_attempted)
    _need(all(type(result[key]) is int and result[key]==value for key,value in actual.items()),
          'Every original annotation is exactly passed/raster-failed/metadata-failed/not-attempted')
    _need(sum((passed,failed,metadata_failed,not_attempted))==len(annotations),'Original ann count conservation')
    success=(result['pixel_decode_complete'] and result['image_error'] is None and passed==len(annotations)
             and failed==metadata_failed==not_attempted==0)
    _need((result['status']=='PASSED')==success,'U42 status cannot substitute for actual pixel/raster coverage counters')
    return actual


def _accumulate(summary,result,annotations):
    summary['processed_images']+=1;summary['original_annotations_seen']+=len(annotations)
    summary['image_checks_passed']+=int(result['status']=='PASSED')
    summary['image_checks_failed']+=int(result['status']=='FAILED')
    summary['pixel_decode_passed_images']+=int(result['pixel_decode_complete'])
    summary['pixel_decode_failed_images']+=int(not result['pixel_decode_complete'])
    summary['empty_annotation_images']+=int(not annotations)
    for key in ('raster_attempted_annotations','raster_passed_annotations','raster_failed_annotations',
                'annotation_metadata_failed_count','not_attempted_annotations'):
        summary[key]+=result[key]
    policy=result['orientation_policy']
    summary['EXIF_policy_counts'][policy if policy is not None else 'no_policy_recorded']+=1
    if result['image_error'] is not None and result['image_error']['stage']=='EXIF_ORIENTATION':
        summary['EXIF_policy_counts']['EXIF_rejection_or_read_failure_images']+=1
    for row in result['annotations']:
        outside=summary['out_of_canvas_diagnostics']
        coordinates=row['out_of_canvas_coordinate_count'];components=row['out_of_canvas_component_count']
        if coordinates is not None and components is not None:
            _need(type(coordinates) is int and type(components) is int and coordinates>=0 and components>=0,
                  'Original per-ann outside diagnostics are nonnegative counts, never a filter')
            outside['annotations_with_available_counts']+=1
            outside['annotations_with_outside_coordinates']+=int(coordinates>0)
            outside['out_of_canvas_coordinate_count']+=coordinates;outside['out_of_canvas_component_count']+=components
        else:outside['annotations_without_available_counts']+=1
        if row['status']=='PASSED':
            stats=summary['geometry_difference_statistics']
            for key in ('declared_area','raster_area','area_difference_raster_minus_declared'):
                _add_statistic(stats[key],row[key])
            _need(type(row['bbox_difference_raster_minus_declared']) is list
                  and len(row['bbox_difference_raster_minus_declared'])==4,'Four original raster-minus-declared bbox components')
            for statistic,value in zip(stats['bbox_difference_raster_minus_declared'],row['bbox_difference_raster_minus_declared']):
                _add_statistic(statistic,value)


def _read_annotation_document(reference,closed_ref,check):
    check();_strict_ref(reference)
    path=Path(closed_ref(reference))
    _need(path.resolve()==Path(reference['path']).resolve(),'Exact original annotation path, no remapping')
    raw=path.read_bytes()
    _need(len(raw)==reference['bytes'] and hashlib.sha256(raw).hexdigest()==reference['sha256'],
          'Original split annotation bytes/hash exact')
    document=decode_metadata_json(raw);raw=None;check()
    return document


def run_population(Image,ImageFile,np,masks,context,checker,*,closed_ref,check,record_image,report):
    """Stream exactly the original5604 JPEG/U42 checks; no result/image corpus.

    Only U42's explicit image/raster failures continue. Source, resource, audit
    callback and unexpected checker exceptions propagate with a partial report.
    The caller owns bindings, modules, actual termination and Root acceptance.
    """
    _need(type(report) is dict and not report,'Caller owns one fresh mutable partial report')
    _need(all(callable(value) for value in (checker,closed_ref,check,record_image)),
          'Inject original U42 callable/source/resource/private audit writers')
    report.update(schema='U43-HRSID-population-pixel-mask-check-report-pending-v1',status='READING_SPLITS',
        qualified=False,input_check_complete=False,full_image_visitation=False,full_private_audit_coverage=False,
        annotation_count_conserved=False,metadata_validated=False,expected_split_counts={s:dict(EXPECTED[s]) for s in SPLITS},
        metadata_observed_split_counts={},split_summaries={s:_new_split_summary() for s in SPLITS},
        totals={key:0 for key in COUNTERS},new_training=False,AP=False,model_executed=False,
        semantic_alignment_proven=False,JPG_equals_PNG_pixels_proven=False,blind_TEST_proven=False,
        population_qualified=False,annotations_changed=0,images_removed=0,
        raw_pixels_or_masks_returned=False,complete_private_records_retained_by_core=False,
        checker_scope='Injected original U42 callable; exact module/source qualification belongs to the original caller',
        scope='Stored JPEG decode and official per-original-ann full-canvas raster compatibility; no semantic/model/AP claim')
    documents={};plan=None;result=None;audit_result=None;image_annotations=None
    try:
        check()
        _need(type(context) is dict and set(context)=={'schema','image_root','annotations','image_refs_by_filename'}
              and context['schema']=='U43-HRSID-official-split-pixel-mask-check-context-v1',
              'Exact explicit context schema/keys before reading any source')
        _need(type(context['annotations']) is dict and set(context['annotations'])==set(SPLITS),
              'Only two explicit original train/test annotation references')
        for split in SPLITS:
            documents[split]=_read_annotation_document(context['annotations'][split],closed_ref,check)
            document=documents[split]
            report['metadata_observed_split_counts'][split]={key:len(document[key]) if type(document.get(key)) is list else None
                                                           for key in ('images','annotations')}
        plan=validate_population_metadata(context,documents)
        for split in SPLITS:
            summary=report['split_summaries'][split]
            summary['declared_images']=plan['splits'][split]['declared_images']
            summary['declared_annotations']=plan['splits'][split]['declared_annotations']
        report.update(metadata_validated=True,status='CHECKING_IMAGES',
            context_metadata_sha256=hashlib.sha256(json.dumps(context,sort_keys=True,separators=(',',':'),
                ensure_ascii=False,allow_nan=False).encode()).hexdigest(),
            duplicate_encoded_SHA_diagnostics=_encoded_duplicate_summary(plan,context['image_refs_by_filename']))
        global_ordinal=0
        for split in SPLITS:
            current=plan['splits'][split];summary=report['split_summaries'][split]
            for ordinal,image in enumerate(current['images']):
                check()
                report['current_image']=dict(split=split,split_image_ordinal=ordinal,global_image_ordinal=global_ordinal)
                image_annotations=current['annotations_by_image'][image['id']]
                reference=context['image_refs_by_filename'][image['file_name']]
                result=checker(Image,ImageFile,np,masks,closed_ref,check,reference,image,image_annotations)
                _validate_image_result(result,image,image_annotations,reference)
                _accumulate(summary,result,image_annotations)
                # The callback owns the complete private per-image JSONL. No
                # filename/ID/per-image result list is retained by this core.
                audit_result=dict(result,split=split,split_image_ordinal=ordinal,global_image_ordinal=global_ordinal)
                record_image(audit_result)
                summary['recorded_images']+=1;global_ordinal+=1
                report['totals']={key:sum(report['split_summaries'][s][key] for s in SPLITS) for key in COUNTERS}
                result=audit_result=image_annotations=None
                check()
        report.update(completion_status(report['split_summaries']))
        report['current_image']=None
        check()
        return report
    except BaseException as error:
        # Keep actual visitation/audit/conservation counters even when a last
        # resource check or private audit write interrupts an otherwise full visit.
        report.update(completion_status(report['split_summaries']))
        report.update(status='INTERRUPTED',input_check_complete=False,
            error=dict(type=type(error).__name__,message=str(error)))
        raise
    finally:
        documents.clear();plan=result=audit_result=image_annotations=None
