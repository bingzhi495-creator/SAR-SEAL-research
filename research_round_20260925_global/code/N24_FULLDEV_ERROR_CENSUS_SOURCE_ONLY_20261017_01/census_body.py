"""Observe original complete COCO evaluation; offline association census only."""
import gc
import importlib.metadata
import importlib.util
import json
import sys
from pathlib import Path


def need(ok, message):
    if not ok:
        raise AssertionError(message)


def load(c, ref, name):
    sp=importlib.util.spec_from_file_location(name,str(c.closed(ref)))
    m=importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


def read(c, ref):
    return json.loads(c.closed(ref).read_bytes())


def run(np,spec,c,pre):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    ctx=read(c,spec['context'])
    need(ctx['schema']=='ROOT-N24-qualified-fullDEV-census-context-v1', 'Exact census context')
    official=load(c,ctx['COCO_source'],'N24_ORIGINAL_COMPLETE_OFFICIAL_COCO')
    core=load(c,spec['core_source'],'N24_OFFLINE_CARDINALITY_ONLY')
    parent=read(c,ctx['N23_ROOT_qualified'])
    need(parent['qualified'] and parent['operation']=='CPU_COCO' and parent['AP'] is True,
         'Prior complete qualified five-policy endpoint')
    coco=COCO(str(c.closed(ctx['DEV'])))
    ids=list(coco.imgs)
    need(ids==ctx['image_ids'] and len(ids)==187 and len(coco.anns)==380 and set(coco.cats)=={1},
         'Original complete187/380/category1 without normalization')
    output=Path(spec['output_directory'])
    output.mkdir()
    summaries={}
    detail_refs={}
    for model in pre['models']:
        c.check()
        values=read(c,ctx['predictions'][model])
        need(len(values)==18700 and all(v['category_id']==1 for v in values), 'All natural query predictions')
        keys=[(v['image_id'],v['query_id']) for v in values]
        need(len(set(keys))==18700 and keys==[tuple(v) for v in ctx['prediction_order']],
             'Original prediction order/identity, no population filtering')
        ordinal={key:i for i,key in enumerate(keys)}
        captured=[]
        def capture(*args,**kwargs):
            e=COCOeval(*args,**kwargs)
            captured.append(e)
            return e
        metrics=official.coco_metrics(coco,values,ids,capture)
        need(metrics==ctx['expected_COCO'][model] and len(captured)==1,
             'Original five-policy complete COCO12 must exactly reproduce')
        ev=captured.pop()
        need(list(ev.params.maxDets)==[1,10,100] and list(ev.params.catIds)==[1]
             and np.array_equal(ev.params.iouThrs,np.linspace(.5,.95,10)),
             'Original thresholds/category/budgets')
        image_evals={x['image_id']:x for x in ev.evalImgs if x is not None
                     and x['category_id']==1 and x['aRng']==ev.params.areaRng[0] and x['maxDet']==100}
        need(set(image_evals)==set(ids), 'Every canonical image contributes all-area evaluation')
        total_GT=0
        effective_GT=0
        ignored_GT=0
        crowd_GT=0
        totals={str(k):dict(official_greedy=[0]*10,maximum_one_to_one=[0]*10,
                            GT_best_ge_threshold=[0]*10) for k in pre['topDets']}
        details=[]
        all_best={str(k):[] for k in pre['topDets']}
        for image_id in ids:
            c.check()
            e=image_evals[image_id]
            raw_gt=ev._gts[(image_id,1)]
            raw_ids=[g['id'] for g in raw_gt]
            raw_pos={g:i for i,g in enumerate(raw_ids)}
            ordered_gt=e['gtIds']
            need(set(raw_ids)==set(ordered_gt), 'No GT lost or duplicated by ignore ordering')
            dts=ev.cocoDt.loadAnns(e['dtIds'])
            need(len(dts)==100 and len({v['query_id'] for v in dts})==100,
                 'Official score-sorted original natural100 candidates')
            dt_rows=[]
            for dt in dts:
                key=(image_id,dt['query_id'])
                original=values[ordinal[key]]
                need(dt['score']==original['score'] and dt['segmentation']==original['segmentation'],
                     'Explicit post-sort query/ordinal association, no guessed identity')
                dt_rows.append(dict(dt_id=dt['id'],query_id=dt['query_id'],
                                    original_prediction_ordinal=ordinal[key],score=dt['score']))
            raw_iou=np.asarray(ev.ious[(image_id,1)],dtype=np.float64)
            if raw_iou.size==0:
                raw_iou=np.zeros((100,len(raw_ids)),dtype=np.float64)
            need(raw_iou.shape==(100,len(raw_ids)), 'Official IoU matrix raw GT order')
            # evalImg internally moves ignored GT to the end with stable ordering.
            # The observer aligns columns by actual annotation ID without redoing IoU.
            ious=raw_iou[:,[raw_pos[g] for g in ordered_gt]]
            gt_ignore=np.asarray(e['gtIgnore'],dtype=np.bool_)
            crowd=np.array([bool(coco.anns[g].get('iscrowd',0)) for g in ordered_gt],dtype=np.bool_)
            active=~gt_ignore & ~crowd
            active_cols=np.flatnonzero(active)
            total_GT+=len(ordered_gt)
            effective_GT+=int(active.sum())
            ignored_GT+=int(gt_ignore.sum())
            crowd_GT+=int(crowd.sum())
            best_records=[]
            local={}
            for k in pre['topDets']:
                cut=min(k,len(dts))
                sub=ious[:cut,active_cols]
                counts=dict(official_greedy=[],maximum_one_to_one=[],GT_best_ge_threshold=[])
                for ti,threshold in enumerate(ev.params.iouThrs):
                    greedy=int(np.count_nonzero((e['gtMatches'][ti]>0) & active &
                               np.isin(e['gtMatches'][ti],e['dtIds'][:cut])))
                    # dtIgnore is assignment-dependent: never filter rows before the maximum matching.
                    pairs=core.maximum_matching(np,sub,float(threshold))
                    maximum=len(pairs)
                    possible=int(np.count_nonzero(sub.max(axis=0)>=threshold)) if cut else 0
                    need(greedy<=maximum<=possible<=int(active.sum()), 'Feasible maximum cardinality bounds')
                    for field,val in [('official_greedy',greedy),('maximum_one_to_one',maximum),('GT_best_ge_threshold',possible)]:
                        counts[field].append(val)
                        totals[str(k)][field][ti]+=val
                local[str(k)]=counts
                for gi,gt_id in enumerate(ordered_gt):
                    maximum=float(ious[:cut,gi].max()) if cut else None
                    candidates=np.flatnonzero(ious[:cut,gi]==maximum) if cut else []
                    best=min(candidates,key=lambda d:(dts[int(d)]['query_id'],ordinal[(image_id,dts[int(d)]['query_id'])])) if len(candidates) else None
                    if active[gi]:
                        all_best[str(k)].append(maximum)
                    best_records.append(dict(GT_id=gt_id,topDets=k,best_IoU=maximum,
                        best_query_id=None if best is None else dts[int(best)]['query_id'],
                        best_score_rank_1based=None if best is None else int(best)+1,
                        official_ignored=bool(gt_ignore[gi]),iscrowd=bool(crowd[gi]),
                        raw_ignore_flag=bool(coco.anns[gt_id].get('ignore',False)),
                        official_top100_matched_dt_ids=e['gtMatches'][:,gi].tolist(),
                        official_match_scope_topDets=100,
                        identity_kind='offline_best_overlap_not_model_matcher'))
            details.append(dict(image_id=image_id,dt_query_ordinal_sidecar=dt_rows,
                gt_ids=ordered_gt,official_gtIgnore=gt_ignore.tolist(),iscrowd=crowd.tolist(),
                original_raw_gt_order=raw_ids,official_IoU_matrix_in_gt_ids_order=ious.tolist(),
                dtScores=list(e['dtScores']),dtMatches=e['dtMatches'].tolist(),gtMatches=e['gtMatches'].tolist(),
                dtIgnore=e['dtIgnore'].tolist(),topDets=local,per_GT_best=best_records))
        need(total_GT==380 and effective_GT+ignored_GT==380, 'All GT recorded with effective/ignore denominator')
        rebuilt_AR={}
        for k in pre['topDets']:
            counts=totals[str(k)]
            for ti in range(10):
                counts.setdefault('greedy_association_opportunity',[]).append(
                    counts['maximum_one_to_one'][ti]-counts['official_greedy'][ti])
                counts.setdefault('one_to_one_candidate_shortage',[]).append(effective_GT-counts['maximum_one_to_one'][ti])
                counts.setdefault('individual_geometric_shortage',[]).append(effective_GT-counts['GT_best_ge_threshold'][ti])
                counts.setdefault('candidate_competition_shortage',[]).append(
                    counts['GT_best_ge_threshold'][ti]-counts['maximum_one_to_one'][ti])
            rebuilt=float(np.mean(np.asarray(counts['official_greedy'],dtype=np.float64)/effective_GT)) if effective_GT else None
            expected=metrics['stats'][7 if k==10 else 8]
            need(rebuilt is not None and abs(rebuilt-expected)<=1e-12,
                 'Independent aggregate matching counts must reconstruct official AR10/100')
            rebuilt_AR[str(k)]=rebuilt
        per_threshold_AP=[]
        for ti in range(10):
            precision=ev.eval['precision'][ti,:,0,0,2]
            valid=precision[precision>-1]
            per_threshold_AP.append(float(valid.mean()) if len(valid) else None)
        need(all(x is not None for x in per_threshold_AP)
             and abs(float(np.mean(per_threshold_AP))-metrics['stats'][0])<=1e-12,
             'Observed precision tensor must reconstruct original mean AP')
        quantiles={k:dict(min=float(np.min(v)),q10=float(np.quantile(v,.1)),median=float(np.median(v)),
                          q90=float(np.quantile(v,.9)),max=float(np.max(v)),zero=sum(x==0 for x in v))
                   for k,v in all_best.items() if v}
        dr=c.write_json(output/(model+'_PRIVATE_CENSUS.json'),dict(model=model,images=details,
            GT=total_GT,effective_GT=effective_GT,ignored_GT=ignored_GT,crowd_GT=crowd_GT))
        detail_refs[model]=dr
        summaries[model]=dict(COCO=metrics,stats_exact_to_qualified_N23=True,
            GT=total_GT,effective_GT=effective_GT,ignored_GT=ignored_GT,crowd_GT=crowd_GT,
            thresholds=ev.params.iouThrs.tolist(),topDets=totals,rebuilt_AR=rebuilt_AR,
            per_threshold_AP=per_threshold_AP,best_IoU_quantiles=quantiles)
        del ev,values,details,image_evals,raw_gt,dts,ious,raw_iou,e,sub
        gc.collect()
        c.check()
        print(json.dumps(dict(stage='N24_FULLDEV_ERROR_CENSUS',model=model,complete=True,
                              GT=total_GT,COCO12_exact=True)),flush=True)
    need('torch' not in sys.modules, 'No Torch/model/weights/native feature cache loaded')
    report=dict(schema='N24-fullDEV-error-census-report-pending-v1',images=187,GT=380,queries=18700,
        models=pre['models'],summaries=summaries,private_detail_refs=detail_refs,
        no_new_policy_AP=True,no_new_training=True,all_original_COCO12_exact=True,
        GT_used_for_offline_diagnosis=True,GT_used_for_deployable_prediction=False,
        maximum_matching_is_AP_upper_bound=False,score_is_unique_cause=False,structure_selected=False,
        Torch_imported=False,qualified=False,goal_complete=False,
        versions=dict(python=sys.version,numpy=np.__version__,pycocotools=importlib.metadata.version('pycocotools')))
    rr=c.write_json(output/'CENSUS_REPORT.json',report)
    return dict(report_ref=rr,images=187,GT=380,queries=18700,models=pre['models'],
                all_original_COCO12_exact=True,no_new_policy_AP=True,AP=False)
