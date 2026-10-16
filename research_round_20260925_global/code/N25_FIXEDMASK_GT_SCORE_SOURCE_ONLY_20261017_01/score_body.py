"""Fixed-mask GT-quality score diagnostic; full official COCO, never training.

All masks and original prediction ordinals remain unchanged. Both alternative
scores use DEV GT and are nondeployable diagnostics, not mathematical AP bounds.
Only score/identity/quality sidecars are saved; full prediction RLEs are not.
"""
import copy
import gc
import importlib.metadata
import importlib.util
import json
import sys
from pathlib import Path


MODELS = ['native_official', 'R', 'A_COMMON', 'A_ACTIVE', 'B_C']
SCORE_MODES = ['ORIGINAL_S0', 'GT_PRODUCT', 'GT_IOU_ONLY']


def need(ok, message):
    if not ok:
        raise AssertionError(message)


def no_torch():
    need(not any(name == 'torch' or name.startswith('torch.') for name in sys.modules),
         'No Torch, model, weights, or native feature cache in CPU score diagnostic')


def load(c, ref, name):
    sp = importlib.util.spec_from_file_location(name, str(c.closed(ref)))
    module = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(module)
    no_torch()
    return module


def read(c, ref):
    return json.loads(c.closed(ref).read_bytes())


def same_array_bytes(left, right):
    return (left.dtype == right.dtype and left.shape == right.shape
            and left.tobytes(order='C') == right.tobytes(order='C'))


def observe(np, c, official, COCOeval, masks, coco, values, ids,
            image_positions, canonical_gt_ids):
    """Evaluate once and align official IoUs by original ordinal and real GT ID."""
    captured = []

    def capture(*args, **kwargs):
        ev = COCOeval(*args, **kwargs)
        captured.append(ev)
        return ev

    # COCO.loadRes may add COCO bookkeeping to its arguments. Keep the immutable
    # original dictionaries separate, without reencoding any segmentation.
    metrics = official.coco_metrics(coco, copy.deepcopy(values), ids, capture)
    need(len(captured) == 1 and len(metrics['stats']) == 12
         and metrics['iouType'] == 'segm' and metrics['maxDets'] == [1,10,100]
         and metrics['original_GT_preserved'] is True and metrics['score_filter'] is None
         and metrics['official_COCO_crowd_ignore_semantics'] is True,
         'One complete original official COCO12 evaluation without score filtering')
    ev = captured.pop()
    thresholds = np.linspace(.5, .95, 10)
    need(list(ev.params.maxDets) == [1,10,100] and list(ev.params.catIds) == [1]
         and set(ev.params.imgIds) == set(ids) and len(ev.params.imgIds) == 187
         and np.array_equal(ev.params.iouThrs, thresholds), 'Original complete COCO scope')
    image_evals = {e['image_id']:e for e in ev.evalImgs if e is not None
                   and e['category_id'] == 1 and e['aRng'] == ev.params.areaRng[0]
                   and e['maxDet'] == 100}
    need(set(image_evals) == set(ids), 'All images retained in all-area top100 evaluation')
    geometry = {}
    for image_id in ids:
        c.check()
        e = image_evals[image_id]
        positions = image_positions[image_id]
        gt_ids = canonical_gt_ids[image_id]
        raw_gt = ev._gts[(image_id,1)]
        raw_ids = [g['id'] for g in raw_gt]
        need(len(raw_ids) == len(set(raw_ids)) == len(gt_ids)
             and set(raw_ids) == set(gt_ids) == set(e['gtIds'])
             and not any(bool(g.get('iscrowd',0)) or bool(g.get('ignore',False)) for g in raw_gt)
             and not bool(np.asarray(e['gtIgnore'], dtype=np.bool_).any()),
             'Original same-category GT identities with exactly zero crowd/ignore')
        dts = ev.cocoDt.loadAnns(e['dtIds'])
        need(len(dts) == len(positions) == 100
             and {d['query_id'] for d in dts} == set(range(100)), 'Full natural100 top100 coverage')
        local_by_query = {values[o]['query_id']:o for o in positions}
        need(len(local_by_query) == 100, 'Original unique natural query identity')
        score_rows = {}
        areas_by_ordinal = {}
        observed_order = []
        for rank, dt in enumerate(dts):
            q = dt['query_id']
            o = local_by_query[q]
            original = values[o]
            need(dt['image_id'] == original['image_id'] == image_id
                 and dt['category_id'] == original['category_id'] == 1
                 and dt['score'] == original['score']
                 and dt['segmentation'] == original['segmentation'],
                 'Official sorted row joined by actual query, unchanged mask and score')
            score_rows[q] = rank
            observed_order.append(o)
            area = int(masks.area(original['segmentation']))
            need(area >= 0, 'Original decoded mask area is nonnegative')
            areas_by_ordinal[o] = area
        expected_order = sorted(positions, key=lambda o:(-float(values[o]['score']),o))
        need(observed_order == expected_order,
             'Official stable score ties must preserve the original prediction ordinal')
        raw_iou = np.asarray(ev.ious[(image_id,1)], dtype=np.float64)
        if not len(gt_ids):
            need(raw_iou.size == 0, 'Zero same-category GT has no hidden IoU columns')
            raw_iou = np.zeros((100,0), dtype=np.float64)
        need(raw_iou.shape == (100,len(raw_ids)) and bool(np.isfinite(raw_iou).all())
             and bool(((raw_iou >= 0) & (raw_iou <= 1)).all()), 'Finite original raw official mask IoUs')
        raw_gt_position = {g:i for i,g in enumerate(raw_ids)}
        row_order = [score_rows[values[o]['query_id']] for o in positions]
        column_order = [raw_gt_position[g] for g in gt_ids]
        ious = raw_iou[row_order,:][:,column_order].copy()
        best_by_GT = ious.max(axis=0)
        reachable = [int(np.count_nonzero(best_by_GT >= t)) for t in thresholds]
        geometry[image_id] = dict(ordinals=list(positions), gt_ids=list(gt_ids),
            query_ids=[values[o]['query_id'] for o in positions], ious=ious,
            mask_areas=[areas_by_ordinal[o] for o in positions],
            best_by_GT=best_by_GT, reachable_GT_counts=reachable,
            top10_query_ids=[values[o]['query_id'] for o in observed_order[:10]])
    per_threshold_AP = []
    for ti in range(10):
        precision = ev.eval['precision'][ti,:,0,0,2]
        valid = precision[precision > -1]
        per_threshold_AP.append(float(valid.mean()) if len(valid) else None)
    need(all(v is not None for v in per_threshold_AP)
         and abs(float(np.mean(per_threshold_AP))-metrics['stats'][0]) <= 1e-12,
         'Original precision tensor reconstructs the reported mean AP')
    del ev,image_evals
    gc.collect()
    c.check()
    no_torch()
    return metrics,per_threshold_AP,geometry


def run(np, spec, c, pre):
    no_torch()
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from pycocotools import mask as masks
    need(pre['models'] == MODELS and pre['score_modes'] == SCORE_MODES,
         'Exactly five original endpoints and three preregistered scoring modes')
    ctx = read(c,spec['context'])
    need(ctx['schema'] == 'ROOT-N25-qualified-fixedmask-score-context-v1', 'Exact N25 context')
    n24_root,n24 = read(c,ctx['N24_ROOT_qualified']),read(c,ctx['N24_report'])
    need(n24_root['qualified'] is True and n24_root['operation'] == 'CPU_ERROR_CENSUS'
         and n24_root['result']['report_ref'] == ctx['N24_report']
         and n24['schema'] == 'N24-fullDEV-error-census-report-pending-v1'
         and n24['images'] == 187 and n24['GT'] == 380 and n24['queries'] == 18700
         and n24['models'] == MODELS and n24['all_original_COCO12_exact'] is True,
         'Accepted original full-DEV census and its exact report')
    official = load(c,ctx['COCO_source'],'N25_ORIGINAL_OFFICIAL_COCO')
    core = load(c,spec['core_source'],'N25_FIXEDMASK_GT_QUALITY_CORE')
    coco = COCO(str(c.closed(ctx['DEV'])))
    ids = list(coco.imgs)
    need(ids == ctx['image_ids'] and len(ids) == 187 and len(coco.anns) == 380
         and set(coco.cats) == {1}
         and all(not bool(g.get('iscrowd',0)) and not bool(g.get('ignore',False))
                 for g in coco.anns.values()), 'Original187/380 GT with no raw crowd or ignore flags')
    canonical_gt_ids = {i:[g['id'] for g in coco.imgToAnns[i] if g['category_id'] == 1] for i in ids}
    need(sum(len(v) for v in canonical_gt_ids.values()) == 380, 'Every original same-category GT retained')
    expected_order = [tuple(v) for v in ctx['prediction_order']]
    need(len(expected_order) == len(set(expected_order)) == 18700, 'Original fixed prediction ordinals')
    image_positions = {i:[] for i in ids}
    for ordinal,(image_id,q) in enumerate(expected_order):
        need(image_id in image_positions and type(q) is int and 0 <= q < 100, 'Original natural query identity')
        image_positions[image_id].append(ordinal)
    need(all(len(v) == 100 for v in image_positions.values()), 'No image loses a natural candidate')
    output = Path(spec['output_directory'])
    output.mkdir()
    results,audit_refs,deltas = {},{},{}
    for model in MODELS:
        c.check()
        values = read(c,ctx['predictions'][model])
        need(len(values) == 18700 and [(v['image_id'],v['query_id']) for v in values] == expected_order
             and all(v['category_id'] == 1 for v in values), 'Original unchanged model population and order')
        original_scores = np.asarray([v['score'] for v in values],dtype=np.float64)
        need(bool(np.isfinite(original_scores).all()), 'Finite original exported S0 without recalculation')
        baseline,baseline_T,baseline_geometry = observe(np,c,official,COCOeval,masks,coco,values,ids,
                                                       image_positions,canonical_gt_ids)
        need(baseline == ctx['expected_COCO'][model] == n24['summaries'][model]['COCO'],
             'Every original full COCO12 exactly reproduces accepted N23/N24')
        all_scores = {mode:np.full(18700,np.nan,dtype=np.float64) for mode in ['quality','GT_PRODUCT','GT_IOU_ONLY']}
        prediction_audit = [None]*18700
        image_audit = {}
        for image_id in ids:
            c.check()
            g = baseline_geometry[image_id]
            positions = g['ordinals']
            local_scores = original_scores[positions].copy()
            scores = core.quality_scores(np,g['ious'],local_scores)
            need(set(scores) == {'quality','GT_PRODUCT','GT_IOU_ONLY'}, 'Exact quality and two fixed score outputs')
            expected_q = g['ious'].max(axis=1) if len(g['gt_ids']) else np.zeros(100,dtype=np.float64)
            for mode in all_scores:
                array = scores[mode]
                need(isinstance(array,np.ndarray) and array.dtype == np.float64 and array.shape == (100,)
                     and bool(np.isfinite(array).all()), 'Finite FP64 quality/score arrays for all100')
                all_scores[mode][positions] = array
            need(same_array_bytes(scores['quality'],expected_q)
                 and same_array_bytes(scores['GT_IOU_ONLY'],scores['quality'])
                 and same_array_bytes(scores['GT_PRODUCT'],local_scores*scores['quality'])
                 and bool(((scores['quality'] >= 0) & (scores['quality'] <= 1)).all()),
                 'Unclipped max-all-GT quality and exact FP64 product, no epsilon or normalization')
            for li,o in enumerate(positions):
                q = float(scores['quality'][li])
                area = g['mask_areas'][li]
                need(area != 0 or q == 0.0, 'Original empty mask has exactly zero GT quality')
                ties = [gt for gi,gt in enumerate(g['gt_ids']) if g['ious'][li,gi] == q]
                prediction_audit[o] = dict(original_prediction_ordinal=o,image_id=image_id,
                    query_id=values[o]['query_id'],S0=float(original_scores[o]),quality=q,
                    GT_PRODUCT=float(scores['GT_PRODUCT'][li]),GT_IOU_ONLY=float(scores['GT_IOU_ONLY'][li]),
                    argmax_GT_ids=ties,mask_area=area,empty_mask=(area==0),
                    identity_kind='offline_max_any_samecategory_GT_not_model_matcher')
            image_audit[str(image_id)] = dict(GT_ids=g['gt_ids'],queries=100,
                original_top10_query_ids=g['top10_query_ids'],oracle_top10_query_ids={},
                original_top100_GT_best_IoUs=g['best_by_GT'].tolist(),
                original_top100_reachable_GT_counts=g['reachable_GT_counts'])
        need(all(v is not None for v in prediction_audit)
             and all(bool(np.isfinite(v).all()) for v in all_scores.values()), 'Every prediction receives a quality audit')
        model_results = {'ORIGINAL_S0':dict(COCO=baseline,per_threshold_AP=baseline_T,
            AP_points=baseline['stats'][0]*100,GT_used_for_score=False,original_COCO12_exact=True,
            top100_geometry_bitexact_to_ORIGINAL_S0=True)}
        for mode in ['GT_PRODUCT','GT_IOU_ONLY']:
            changed = [dict(v,score=float(all_scores[mode][o])) for o,v in enumerate(values)]
            need([(v['image_id'],v['query_id']) for v in changed] == expected_order
                 and all({k:v for k,v in a.items() if k != 'score'} ==
                         {k:v for k,v in b.items() if k != 'score'} for a,b in zip(changed,values)),
                 'Only score may change; same original RLE and ordinal, no filtering or reencoding')
            metrics,T,geometry = observe(np,c,official,COCOeval,masks,coco,changed,ids,
                                        image_positions,canonical_gt_ids)
            for image_id in ids:
                old,new = baseline_geometry[image_id],geometry[image_id]
                need(old['ordinals'] == new['ordinals'] and old['query_ids'] == new['query_ids']
                     and old['gt_ids'] == new['gt_ids'] and old['mask_areas'] == new['mask_areas']
                     and same_array_bytes(old['ious'],new['ious'])
                     and same_array_bytes(old['best_by_GT'],new['best_by_GT'])
                     and old['reachable_GT_counts'] == new['reachable_GT_counts'],
                     'Fresh official IoUs: fixed top100 geometry bitexact after query/GT identity alignment')
                image_audit[str(image_id)]['oracle_top10_query_ids'][mode] = new['top10_query_ids']
            model_results[mode] = dict(COCO=metrics,per_threshold_AP=T,AP_points=metrics['stats'][0]*100,
                Delta_AP_points=metrics['stats'][0]*100-baseline['stats'][0]*100,
                GT_used_for_score=True,oracle_not_deployable=True,no_math_AP_upper_bound=True,
                full_population_and_original_ordinal_preserved=True,
                top100_geometry_bitexact_to_ORIGINAL_S0=True,top10_unchanged_required=False)
            del changed,geometry
            c.check()
            print(json.dumps(dict(stage='N25_FIXEDMASK_GT_SCORE',model=model,score_mode=mode,
                                  images=187,queries=18700,geometry_bitexact=True)),flush=True)
        deltas[model] = {mode:model_results[mode]['Delta_AP_points'] for mode in ['GT_PRODUCT','GT_IOU_ONLY']}
        ar = c.write_json(output/(model+'_PRIVATE_SCORE_AUDIT.json'),dict(
            schema='N25-private-fixedmask-score-audit-v1',model=model,images=187,GT=380,queries=18700,
            prediction_records=prediction_audit,image_checks=image_audit,
            original_prediction_reference=ctx['predictions'][model],
            same_masks_and_natural100=True,all_fresh_oracle_top100_IoUs_bitexact=True,
            full_RLE_or_GT_annotations_copied=False,GT_used_for_diagnosis=True,
            oracle_not_deployable=True,qualified=False))
        results[model],audit_refs[model] = model_results,ar
        del values,original_scores,baseline_geometry,all_scores,prediction_audit,image_audit
        gc.collect()
        c.check()
    no_torch()
    primary = dict(model='R',score_mode='GT_PRODUCT',
        baseline_AP_points=results['R']['ORIGINAL_S0']['AP_points'],
        oracle_AP_points=results['R']['GT_PRODUCT']['AP_points'],
        Delta_AP_points=deltas['R']['GT_PRODUCT'])
    report = dict(schema='N25-fixedmask-GT-score-report-pending-v1',images=187,GT=380,queries=18700,
        models=MODELS,score_modes=SCORE_MODES,results=results,primary=primary,
        Delta_AP_points=deltas,private_score_audit_refs=audit_refs,
        context=spec['context'],N24_ROOT_qualified=ctx['N24_ROOT_qualified'],N24_report=ctx['N24_report'],
        all_original_COCO12_exact=True,all_oracle_top100_geometry_bitexact=True,
        no_new_deployable_policy_AP=True,oracle_not_deployable=True,no_math_AP_upper_bound=True,
        AP=True,AP_kind='GT_ORACLE_DIAGNOSTIC_NOT_LEARNED_METHOD',new_training=False,DEV=True,TEST=False,
        GT_used_for_score=True,GT_used_for_deployable_prediction=False,
        score_is_unique_cause=False,innovation_established=False,qualified=False,goal_complete=False,
        versions=dict(python=sys.version,numpy=np.__version__,pycocotools=importlib.metadata.version('pycocotools')))
    rr = c.write_json(output/'ORACLE_SCORE_REPORT.json',report)
    return dict(report_ref=rr,images=187,GT=380,queries=18700,models=MODELS,score_modes=SCORE_MODES,
        primary=primary,Delta_AP_points=deltas,all_original_COCO12_exact=True,
        all_oracle_top100_geometry_bitexact=True,AP=True,
        AP_kind='GT_ORACLE_DIAGNOSTIC_NOT_LEARNED_METHOD',no_new_deployable_policy_AP=True,
        oracle_not_deployable=True,no_math_AP_upper_bound=True)
