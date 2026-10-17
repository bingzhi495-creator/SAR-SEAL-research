"""Fixed-original-native-pair geometry diagnostic; no scientific imports at import.

Pixel calculations in diagnose are pending real execution by Root. The only
executed tests use fabricated JSON-like COCO metadata, never masks or tensors.
"""
import json
import math


def _need(ok,message):
    if not ok:
        raise ValueError(message)


def _list(value):
    if type(value) not in (list,tuple) and callable(getattr(value,'tolist',None)):
        value=value.tolist()
    _need(type(value) in (list,tuple),'COCO metadata requires a sequence')
    return list(value)


def _integer(value,minimum=0):
    if type(value) not in (int,float) and callable(getattr(value,'tolist',None)):
        value=value.tolist()
    _need(type(value) in (int,float) and math.isfinite(value) and int(value)==value
          and value>=minimum,'Finite exact integral COCO identity required')
    return int(value)


def select_native_pairs(evalImgs,cocoDt_ann_mapping,areaRng):
    """First IoU row only; caller must assert it represents native IoU=.50.

    Uses gtMatches/gtIgnore at area=all,maxDet=100. Never uses new masks,
    alternative detections, or a highest-IoU substitution to choose a pair.
    """
    _need(type(cocoDt_ann_mapping) is dict,'Original native COCO DT annotation mapping required')
    area=_list(areaRng);_need(len(area)==2,'Exact all-area range required')
    result=[];seen_images=set();seen_GT=set();seen_DT=set();seen_queries=set()
    for row in evalImgs:
        if row is None:
            continue
        _need(type(row) is dict,'Original COCO evalImg metadata required')
        if _list(row['aRng'])!=area or row['maxDet']!=100 or row['category_id']!=1:
            continue
        image=_integer(row['image_id'])
        _need(image not in seen_images,'Repeated native all-area/max100 image entry')
        seen_images.add(image)
        GT=[_integer(value) for value in _list(row['gtIds'])]
        ignored=_list(row['gtIgnore']);matches=_list(row['gtMatches'])
        _need(len(matches)>0,'Native first IoU row required')
        matches=_list(matches[0])
        _need(len(GT)==len(ignored)==len(matches),'GT match/ignore columns must align')
        _need(len(set(GT))==len(GT) and not seen_GT.intersection(GT),'Repeated original GT identity')
        seen_GT.update(GT)
        DT=[_integer(value,1) for value in _list(row['dtIds'])]
        _need(len(set(DT))==len(DT) and len(DT)<=100,'Distinct original top100 DT identities')
        for gt,ignore_value,dt_value in zip(GT,ignored,matches):
            if type(ignore_value) not in (int,bool) and callable(getattr(ignore_value,'tolist',None)):
                ignore_value=ignore_value.tolist()
            _need(type(ignore_value) in (int,bool) and ignore_value in (0,1),'Exact original GT ignore flag')
            dt=_integer(dt_value)
            if ignore_value or dt==0:
                continue
            _need(dt in DT and dt in cocoDt_ann_mapping,'Native GT match must resolve to its original top100 DT')
            annotation=cocoDt_ann_mapping[dt]
            _need(type(annotation) is dict and _integer(annotation['id'],1)==dt
                  and _integer(annotation['image_id'])==image and annotation['category_id']==1,
                  'Original native detection id/image/category association')
            query=_integer(annotation['query_id']);_need(query<100,'Original natural query bound')
            _need(dt not in seen_DT and (image,query) not in seen_queries,
                  'A nonignored native DT/query may match at most one fixed GT')
            seen_DT.add(dt);seen_queries.add((image,query))
            result.append(dict(image_id=image,GT_annotation_id=gt,dt_id=dt,query_id=query))
    return result


def _decode(np,masks,rle,shape):
    _need(type(rle) is dict and list(rle['size'])==list(shape)
          and type(rle['counts']) in (str,bytes),'Original-grid compressed RLE metadata')
    counts=rle['counts'].encode('ascii') if type(rle['counts']) is str else rle['counts']
    result=np.ascontiguousarray(masks.decode(dict(size=list(shape),counts=counts))!=0,dtype=np.bool_)
    _need(result.shape==tuple(shape),'Decoded complete original grid required')
    return result


def _confusion(np,pred,GT,omega):
    result={}
    for name,value in (('TP',pred & GT),('FP',pred & ~GT),('FN',~pred & GT)):
        inside=int(np.count_nonzero(value & omega));outside=int(np.count_nonzero(value & ~omega))
        result[name+'_inside']=inside;result[name+'_outside']=outside;result[name]=inside+outside
    denominator=result['TP']+result['FP']+result['FN']
    _need(denominator>0,'Original selected nonempty GT yields a positive IoU union')
    result['IoU_FP64']=float(np.float64(result['TP'])/np.float64(denominator))
    return result


def _non_geometry(row):
    return json.dumps({key:value for key,value in row.items() if key!='segmentation'},
                      sort_keys=True,separators=(',',':'),allow_nan=False)


def diagnose(np,masks,coco,native_eval,predictions,support_records):
    """PENDING real pixel execution; caller owns Source and original CPU terminal.

    The original native evaluation alone fixes pairs. Each alternative is read
    at that original image/query, including empty/bad predictions. Unmatched or
    ignored GTs are excluded, never silently converted to zero-quality targets.
    Returned individual GT identifiers/counts are private; publish summary only.
    """
    methods=('native_official','R','SELECTIVE','WHOLE_NODE')
    _need(type(predictions) is dict and set(predictions)==set(methods),
          'All four fixed original-score method populations required')
    native=predictions['native_official']
    _need(type(native) is list and len(native)==18700
          and len(coco.imgs)==187 and len(coco.anns)==380 and set(coco.cats)=={1},
          'Complete DEV187/380 and original natural100 predictions')
    images=[];canonical=[];native_by={}
    for index in range(187):
        block=native[index*100:(index+1)*100]
        image=_integer(block[0]['image_id']);queries=[]
        _need(image in coco.imgs,'Original DEV image identity')
        for row in block:
            query=_integer(row['query_id'])
            _need(_integer(row['image_id'])==image and query<100 and row['category_id']==1
                  and type(row['score']) in (int,float) and math.isfinite(row['score']),
                  'Canonical original image/query/category/finite score')
            identity=(image,query);_need(identity not in native_by,'Duplicate native image/query')
            native_by[identity]=row;queries.append(query);canonical.append(identity)
        _need(len(set(queries))==100 and set(queries)==set(range(100)),'Each original natural query once')
        images.append(image)
    _need(len(set(images))==187 and set(images)==set(coco.imgs),'Complete unique DEV images')
    by_method={}
    for method in methods:
        values=predictions[method]
        _need(type(values) is list and len(values)==18700,'Every method retains18700 original candidates')
        mapping={}
        for identity,baseline,value in zip(canonical,native,values):
            _need(type(value) is dict and _non_geometry(value)==_non_geometry(baseline),
                  'Only segmentation may differ at the original canonical ordinal')
            mapping[identity]=value
        by_method[method]=mapping
    _need(type(support_records) is list and len(support_records)==18700,'Complete original Omega support records')
    supports={}
    for ordinal,(identity,value) in enumerate(zip(canonical,support_records)):
        _need(type(value) is dict and value['dataset_index']==ordinal//100
              and (value['image_id'],value['query_id'])==identity and identity not in supports,
              'Omega support order must retain exact original image/query identity')
        supports[identity]=value
    params=native_eval.params
    _need(set(params.imgIds)==set(images) and len(params.imgIds)==187
          and list(params.catIds)==[1] and list(params.maxDets)==[1,10,100]
          and np.array_equal(params.iouThrs,np.linspace(.5,.95,10)),
          'Original native official fullDEV/all100 evaluation and .50-first thresholds')
    labels=list(params.areaRngLbl);_need(labels.count('all')==1,'Original all-area COCO range')
    area=_list(params.areaRng[labels.index('all')])
    frozen=native_eval._paramsEval
    _need(set(frozen.imgIds)==set(images) and list(frozen.catIds)==[1]
          and list(frozen.maxDets)==[1,10,100]
          and np.array_equal(frozen.iouThrs,params.iouThrs)
          and _list(frozen.areaRng[list(frozen.areaRngLbl).index('all')])==area,
          'Evaluation must have actually used the same original parameters')
    dt_annotations=native_eval.cocoDt.anns
    _need(set(dt_annotations)==set(range(1,18701)),'Original COCO loadRes ordinal identities')
    for ordinal,row in enumerate(native,1):
        annotation=dt_annotations[ordinal]
        _need(annotation['id']==ordinal and annotation['image_id']==row['image_id']
              and annotation['query_id']==row['query_id'] and annotation['category_id']==1
              and float(annotation['score']).hex()==float(row['score']).hex(),
              'Native evaluation detection ids must resolve to the exact original query/score')
    considered_GT=set()
    for row in native_eval.evalImgs:
        if row is not None and _list(row['aRng'])==area and row['maxDet']==100 and row['category_id']==1:
            considered_GT.update(_integer(value) for value in _list(row['gtIds']))
    _need(considered_GT==set(coco.anns),'Native all-area evaluation retains every original GT id')
    pairs=select_native_pairs(native_eval.evalImgs,dt_annotations,area)
    totals={method:{name+suffix:0 for name in ('TP','FP','FN')
                    for suffix in ('','_inside','_outside')} for method in methods}
    ious={method:[] for method in methods}
    passes={method:{'0.90':0,'0.95':0} for method in methods}
    private=[]
    for pair in pairs:
        image,query=pair['image_id'],pair['query_id'];identity=(image,query)
        ann=coco.anns[pair['GT_annotation_id']]
        _need(ann['image_id']==image and ann['category_id']==1,'True original fixed-pair annotation')
        shape=(int(coco.imgs[image]['height']),int(coco.imgs[image]['width']))
        GT=np.ascontiguousarray(coco.annToMask(ann)!=0,dtype=np.bool_)
        _need(GT.shape==shape and bool(GT.any()),'Complete nonempty selected original instance GT')
        support=supports[identity];omega=_decode(np,masks,support['Omega'],shape)
        record=dict(dataset_index=support['dataset_index'],image_id=image,query_id=query,
                    GT_annotation_id=pair['GT_annotation_id'],native_dt_id=pair['dt_id'],
                    GT_area=int(np.count_nonzero(GT)),Omega_points=int(np.count_nonzero(omega)),methods={})
        for method in methods:
            pred=_decode(np,masks,by_method[method][identity]['segmentation'],shape)
            counts=_confusion(np,pred,GT,omega)
            denominator=counts['TP']+counts['FP']+counts['FN']
            if method=='native_official':
                _need(2*counts['TP']>=denominator,'Native .50 selected pair must reproduce its mask IoU')
            for name in totals[method]:totals[method][name]+=counts[name]
            ious[method].append(counts['IoU_FP64'])
            passes[method]['0.90']+=int(100*counts['TP']>=90*denominator)
            passes[method]['0.95']+=int(100*counts['TP']>=95*denominator)
            record['methods'][method]=counts
            del pred,counts
        private.append(record)
        del GT,omega
    selected=len(pairs);method_summary={}
    for method in methods:
        values=totals[method]
        error=values['FP']+values['FN'];outside=values['FP_outside']+values['FN_outside']
        denominator=values['TP']+error
        method_summary[method]=dict(matched_instances=selected,counts=values,
            pooled_error_pixels=error,pooled_inside_error_pixels=error-outside,
            pooled_outside_error_pixels=outside,
            pooled_outside_error_fraction=(float(np.float64(outside)/np.float64(error)) if error else None),
            mean_IoU=(float(np.mean(np.asarray(ious[method],dtype=np.float64))) if selected else None),
            pooled_IoU=(float(np.float64(values['TP'])/np.float64(denominator)) if denominator else None),
            fixed_pair_pass_counts=passes[method])
    summary=dict(schema='N28-original-native-fixed-pair-geometry-summary-v1',
        scope=dict(total_DEV_GT=380,total_DEV_images=187,total_natural_queries=18700,
            selected_native_nonignored_GT=selected,selected_GT_coverage_fraction=selected/380,
            selection_model='native_official',selection_IoU=.5,selection_area='all',selection_maxDets=100,
            conditioned_on_original_native_matched_GT=True,unmatched_or_ignored_GT_excluded=True,
            no_new_method_rematching=True,private_pair_records_not_for_publication=True),
        methods=method_summary,
        interpretation_limit='Conditional geometry on GTs originally matched by native_official at IoU .50; not all380-GT localization evidence, not AP and not an AP/learnability bound. Every method uses the same original query even if its new mask becomes empty or worse.',
        threshold_comparison='Exact integer TP/union ratios against .90/.95; no tolerance',
        Omega_basis='Original GPU-captured native_L absolute<=2; support mask is not inferred from a method mask',
        new_method_GT_pair_selection=False,new_training=False,AP_computed=False,
        qualified=False,ROOT_actual_terminal_self_certified=False)
    return dict(summary=summary,private_pair_records=private)
