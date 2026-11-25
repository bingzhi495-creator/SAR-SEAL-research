"""Equal-image and count activation controls on exact frozen source ROI additions."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType

import numpy as np
from pycocotools import mask as codec
import torch
import torch.nn.functional as F


PROTOCOL='bbfde4ac70ba330ab44a0188844a103913aad48f74fbbf31a5ff4da3e1fe0c37'
NATIVE_ROI='a0a14935d2d9780aab938125b7ff457b648bf808bd4543f282ac5156656eaf82'
NATIVE_RESULT='d698e5f1af8c8a8b862e32f5ba62c58ff6c23371dc25b8a617795ddd8e513279'
NATIVE_TERMINAL='e74ad512313077766f274489044e6eb8f20858d23eadf5db46c4334381945e35'
CONTROLS=('rf_only','a_only','detector_only','hash_sham')
GATED=(*CONTROLS,'joint')
ARMS=('native','ungated',*GATED)


def check(ok,why):
    if not ok:
        raise ValueError(why)


def pin(path,digest):
    data=path.read_bytes()
    check(sha256(data).hexdigest()==digest,'Frozen exact artifact: '+path.name)
    return data


def module(path,digest,name):
    instance=ModuleType(name)
    instance.__file__=str(path)
    sys.modules[name]=instance
    exec(compile(pin(path,digest),str(path),'exec'),instance.__dict__)
    return instance


def midranks(values):
    n=len(values)
    if n==1:
        return np.asarray([.5])
    check(n>0 and np.isfinite(values).all(),'Nonempty finite within-image cue')
    order=np.argsort(values,kind='stable')
    groups=np.split(order,np.flatnonzero(np.diff(values[order]))+1)
    ranks=np.empty(n)
    a=0
    for ties in groups:
        length=len(ties)
        ranks[ties]=(a+.5*(length-1))/(n-1)
        a+=length
    return ranks


def keep_group(iid,rows):
    n=len(rows)
    check(n and len({r['k'] for r in rows})==1, 'Only exact k within one image group')
    rf=midranks(np.asarray([r['q_rf'] for r in rows]))
    a=midranks(np.asarray([r['q_a'] for r in rows]))
    sham=[]
    for r in rows:
        text='|'.join(map(str,('HRSID-ROI-ACTIVATION-HASH-V1',iid,r['rank'],r['query_id'],r['k'])))
        sham.append(int.from_bytes(sha256(text.encode()).digest(),'big'))
    cues=dict(rf_only=rf,a_only=a,detector_only=[r['detector'] for r in rows],hash_sham=sham,joint=(rf+a)/2)
    limit=(n+1)//2
    return {name:{rows[i]['pos'] for i in sorted(range(n), key=lambda j:(-v[j],rows[j]['rank']))[:limit]}
            for name,v in cues.items()}


def replay_with_cues(args,src,paired,corpus,box,rank,sparse,pilot,stock,infos,groups,schemas,folds,models,device,prior):
    raw_render,original_additions=src.replay_raw,src.additions
    store={}
    cues=defaultdict(dict)

    def observed_raw(*a,**k):
        result=raw_render(*a,**k)
        store['raw']=result
        return result

    def observed_additions(args,meta,pos,crop_bits_area_rle,a_logits,box,rank,sparse,pilot,cache):
        store.clear()
        result,details=original_additions(args,meta,pos,crop_bits_area_rle,a_logits,box,rank,sparse,pilot,cache)
        if result is not None:
            pixels=result['learned'][1]
            crop=crop_bits_area_rle[0]
            x0,y0,x1,y1=crop
            y,x=np.divmod(pixels,800)
            roi=F.interpolate(a_logits,size=(y1-y0,x1-x0),mode='bilinear',align_corners=False)[0,0]
            qa=roi[y-y0,x-x0].sigmoid().numpy()
            qrf=torch.from_numpy(store['raw'][y,x]).sigmoid().numpy()
            identity=meta['rows'][pos]
            iid=identity['image_id']
            check(pos not in cues[iid] and len(pixels)>0 and qa.shape==qrf.shape==pixels.shape
                  and np.isfinite(qa).all() and np.isfinite(qrf).all(),'Same archived probabilities on exact selected additions')
            cues[iid][pos]=dict(pos=pos,rank=identity['rank'],query_id=identity['query_id'],k=len(pixels),
                                q_a=float(qa.mean(dtype=np.float64)),q_rf=float(qrf.mean(dtype=np.float64)),detector=identity['score'])
        return result,details

    src.replay_raw=observed_raw
    src.additions=observed_additions
    try:
        worlds,_,actions,census,effects,budgets=src.infer(args,paired,corpus,box,rank,sparse,pilot,stock,infos,groups,
                                                        schemas,folds,models,device,prior)
    finally:
        src.replay_raw,src.additions=raw_render,original_additions
        store.clear()
    return worlds['learned'],actions,cues,census,effects,budgets


def gate_images(stock,ungated,actions,cues):
    worlds={arm:{} for arm in GATED}
    retained={arm:{} for arm in GATED}
    summary=Counter()
    budgets={arm:Counter() for arm in GATED}
    for iid,rows in stock.items():
        candidates=cues[iid]
        check(candidates.keys()==actions[iid].keys(),'Every preexisting action has exact q evidence')
        categories=defaultdict(list)
        selections={arm:set() for arm in GATED}
        for pos,row in candidates.items():
            check(row['k']==len(actions[iid][pos]['learned']),'Fixed original action coordinate count')
            categories[row['k']].append(row)
        for k,v in categories.items():
            summary['groups']+=1
            summary['singleton_groups']+=len(v)==1
            quota=(len(v)+1)//2
            summary['kept_each_arm']+=quota
            summary['pixels_each_arm']+=k*quota
            selected=keep_group(iid,v)
            for arm,keep in selected.items():
                check(len(keep)==quota and all(candidates[pos]['k']==k for pos in keep), 'Same image x k exact action and pixel budget')
                selections[arm].update(keep)
        for arm,chosen in selections.items():
            retained[arm][iid]=chosen
            worlds[arm][iid]=[ungated[iid][pos] if pos in chosen else native for pos,native in enumerate(rows)]
            for pos in chosen:
                data=worlds[arm][iid][pos]
                before,after=rows[pos]['segmentation'],data['segmentation']
                k=candidates[pos]['k']
                area=int(codec.area(before))
                intersection=int(codec.area(codec.merge([before,after],intersect=True)))
                check(area==intersection and int(codec.area(after))==area+k
                      and all(data[key]==rows[pos][key] for key in ('image_id','category_id','score')),
                      'Original exact full learned mask or immutable stock, no deletions or score edits')
                budgets[arm].update(actions=1,added_pixels=k,deleted_pixels=0)
    check(all(v['actions']==summary['kept_each_arm'] and v['added_pixels']==summary['pixels_each_arm'] and v['deleted_pixels']==0
              for v in budgets.values()),'Equal total and same image-bin budgets')
    return worlds,retained,dict(summary),{key:dict(v) for key,v in budgets.items()}


def geometry_and_oracle(gt,coco,matches,groups,worlds,actions,retained,pilot,sparse,ap):
    rows=defaultdict(list)
    geometry_axes=(*ARMS,'oracle')
    selected_hits={arm:Counter() for arm in (*GATED,'ungated','oracle')}
    paired_hits={arm:defaultdict(Counter) for arm in CONTROLS}
    edited=defaultdict(Counter)
    for ann in gt['annotations']:
        aid,iid=ann['id'],ann['image_id']
        group=groups[iid]
        if aid not in matches:
            rows[group].append([0.]*len(geometry_axes))
            continue
        found,pos=matches[aid]
        check(found==iid,'Exact original fixed candidate image')
        truth=coco.annToRLE(ann)
        rles=[worlds[arm][iid][pos]['segmentation'] for arm in ARMS]
        value=codec.iou(rles,[truth],[0]).reshape(-1)
        check(value[0]>=.50-1e-12,'Same original mask .50 fixed assignment')
        original_action=actions[iid].get(pos)
        oracle_selection=bool(original_action) and value[1]>value[0]
        rows[group].append([*value.tolist(),float(value[1] if oracle_selection else value[0])])
        for i,arm in enumerate(ARMS[:-1]):
            if rles[i]!=rles[-1]:
                edited[arm][group]+=1
        if original_action is None:
            continue
        pixels=original_action['learned']
        hits=int(pilot.sample_rle(truth,pixels,sparse).sum())
        k=len(pixels)
        selected_hits['ungated'].update(matched_actions=1,added=k,hits=hits)
        if oracle_selection:
            selected_hits['oracle'].update(matched_actions=1,added=k,hits=hits)
        for arm in GATED:
            if pos in retained[arm][iid]:
                selected_hits[arm].update(matched_actions=1,added=k,hits=hits)
        joint=pos in retained['joint'][iid]
        for arm in CONTROLS:
            c=pos in retained[arm][iid]
            category='shared' if joint and c else 'joint_only' if joint else 'control_only' if c else 'neither'
            paired_hits[arm][category].update(matched_actions=1,added=k,hits=hits)
    parent=np.asarray([np.mean(rows[g],axis=0) for g in sorted(rows)])
    overall=np.asarray([v for group in rows.values() for v in group])
    check(len(parent)==109 and len(overall)==496 and len(matches)==453,'Complete original native and unmatched denominator')
    draws=np.random.Generator(np.random.PCG64(5217)).integers(len(parent),size=(4096,len(parent)))
    prefix_values=100*parent.mean(axis=0)
    scores={name:dict(prefix_balanced_iou_points=float(prefix_values[i]),
                      threshold_reach={str(t):int((overall[:,i]>=t).sum()) for t in (.50,.75,.90)})
            for i,name in enumerate(geometry_axes)}
    def geometry_delta(left,right):
        A,B=(geometry_axes.index(name) for name in (left,right))
        vals=100*(parent[:,A]-parent[:,B])
        return dict(IoU_points=float(vals.mean()),percentile95_points=np.percentile(vals[draws].mean(axis=1),[2.5,97.5]).tolist())
    contrasts={}
    for arm in ARMS[:-1]:
        g=geometry_delta('joint',arm)
        d=ap['joint']['AP']-ap[arm]['AP']
        contrasts[arm]=dict(g,AP_points=d,changed_fixed_hulls=sum(edited[arm].values()),changed_prefixes=len(edited[arm]),
                            passes_metrics=bool(d>=.30 and g['IoU_points']>=.20 and g['percentile95_points'][0]>0))
    best=max(CONTROLS,key=lambda a:ap[a]['AP'])
    changed_floor=contrasts[best]['changed_fixed_hulls']>=20 and contrasts[best]['changed_prefixes']>=8
    take=changed_floor and all(contrasts[arm]['passes_metrics'] for arm in ('native',*CONTROLS))
    return dict(geometry_fixed_all_hulls=scores,joint_minus=contrasts,
                gate=dict(best_AP_same_budget_control=best,change_exposure_pass=changed_floor,advances=take),
                matched_hits_by_arm={a:dict(row) for a,row in selected_hits.items()},
                joint_vs_control_shared_actions={a:{k:dict(v) for k,v in row.items()} for a,row in paired_hits.items()},
                retrospective_oracle_geometry_only={a:geometry_delta('oracle',a) for a in ('native','ungated','joint')})


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for key in ('protocol','code','native-roi-run','paired-run','capture','acceptance','source-cohort','prior','corpus-run',
                'box-run','sibling-pilot','fit-gt','fit-images','model-source','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    start=perf_counter()
    check(not args.output.exists() and args.output.parent.is_dir(),'Fresh isolated source-only output')
    pin(args.protocol,PROTOCOL)
    root=args.native_roi_run
    prior=json.loads(pin(root/'result/RESULT_SAFE.json',NATIVE_RESULT))
    terminal=json.loads(pin(root/'observed_native_roi/TERMINAL.json',NATIVE_TERMINAL))
    check(terminal['actual_exit']==0 and terminal['output_footer_verified'] and terminal['result_sha256']==NATIVE_RESULT
          and terminal['worker_sha256']==prior['worker_sha256']==NATIVE_ROI and prior['source_hulls']==496,
          'Closed deterministic same-support ungated learned ROI')
    src=module(root/'native_roi.py',NATIVE_ROI,'_activation_previous_exact_execution')
    old=json.loads(pin(args.paired_run/'result/RESULT_SAFE.json',src.PAIRED_RESULT))
    prior_t=json.loads(pin(args.paired_run/'observed_paired/TERMINAL.json',src.PAIRED_TERMINAL))
    check(prior_t['actual_exit']==0 and prior_t['output_footer_verified'] and prior_t['result_sha256']==src.PAIRED_RESULT
          and prior_t['worker_sha256']==src.PAIRED,'Original paired training closure')
    paired=module(args.paired_run/'roi_paired.py',src.PAIRED,'_activation_original_render')
    device=paired.configure()
    check(str(device)==old['device'],'Same immutable ROI device and FP32 precision')
    corpus=module(args.corpus_run/'corpus.py',paired.CORPUS,'_activation_exact_input_sampling')
    box=module(args.box_run/'box_study.py',corpus.BOX_SOURCE,'_activation_original_stock_boxes')
    pilot=module(args.sibling_pilot/'pilot.py',paired.SIBLING_SOURCE,'_activation_pinned_coco')
    sparse=module(args.code/'RFDETR_SPARSE_RLE_EQUIVALENCE_20261120/sparse_rle.py',
                 'cb9b8e839de85588fdf116ce5554ed6cadbc20bdbe76e38180561528bfd870d5','_activation_exact_rle')
    rank=module(args.code/'DFINE_SEG_EXTERNAL_BASELINE_FIT32_E24_20261112/e24/fit109_probability_rank/rank_math.py',
                'cb438f8937f8e29dbdcbc441abdf5a788ac9c442fd63f1b3ad504b8ae0fcca0b','_activation_exact_shell')
    stock,infos,groups,oracle,capture=box.inputs(args)
    schemas={v['image_id']:v for v in json.loads(pin(args.source_cohort,corpus.SOURCE_COHORT))['images'] if v['core']}
    plan=module(args.code/oracle.FORMAL/'fit_plan.py',corpus.FOLD_PLAN,'_activation_original_fold_schema')
    folds=plan.source_folds(list(groups.values()))
    check(schemas.keys()==stock.keys() and capture['stock_predictions_sha256']==prior['original_stock_sha256']
          and len(folds)==109,'Frozen full decoded source roster')
    models,digests=src.restore_models(args,paired,old,device)
    ungated,actions,cues,admission,observed,b=replay_with_cues(args,src,paired,corpus,box,rank,sparse,pilot,
                                             stock,infos,groups,schemas,folds,models,device,old)
    check(admission==prior['candidate_gate_census'] and observed==prior['exact_budget_census']
          and prior['same_count_add_only_budgets']==b and b['learned']==dict(edited=4957,added=31613,removed=0),
          'Byte-pinned candidate procedure restores exact prior scientific action counts')
    gated,retained,sizes,budgets=gate_images(stock,ungated,actions,cues)
    label_time=perf_counter()
    print(json.dumps(dict(stage='verified_equal_budgets_before_label_load', **sizes)),flush=True)
    helper=module(args.code/'RFDETR_FIT109_QUOTA_COMPETITION_TARGETS_20261123/assemble_competition.py',
                  '3b0be851e38028248d5b646a108598585b6a261989496bf535d5ea8b54e6dd5b','_activation_original_author_labels')
    gt=helper.annotations(args,stock)
    coco,matches=oracle.fixed_coco(gt,stock)
    worlds={'native':stock,'ungated':ungated,**gated}
    metrics={arm:pilot.full_metrics(gt,worlds[arm],stock) for arm in ARMS}
    check(metrics['native']==prior['full_coco_points']['native'] and metrics['ungated']==prior['full_coco_points']['learned'],
          'Exact original COCO parity on native and preexisting ungated learned actions')
    result=geometry_and_oracle(gt,coco,matches,groups,worlds,actions,retained,pilot,sparse,metrics)
    for name,ref in (('native','native'),('ungated','learned')):
        check(abs(result['geometry_fixed_all_hulls'][name]['prefix_balanced_iou_points']-
                  prior['fixed_all_hull_discrete']['arms'][ref]['prefix_balanced_iou_points'])<1e-12,
              'Exact original fixed-all-hull geometry parity')
    result.update(status='HRSID_FIT109_EQUAL_BUDGET_ROI_ACTIVATION_COMPLETE_V1',phase='score',protocol_sha256=PROTOCOL,
                  worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),source_original_ungated_roi_sha256=NATIVE_RESULT,
                  original_rf_capture_sha256=box.CAPTURE,original_roi_head_tensor_hashes=digests,source_images=len(stock),
                  source_hulls=len(gt['annotations']),fixed_native_matches=len(matches),source_candidate_count=10488,
                  prefix_proxy_groups=109,archives_score_parity=True,original_detector_fit_exposed=True,
                  original_admission=admission,original_ungated_budget=b['learned'],same_exact_image_k_groups=sizes,
                  exact_equal_budgets=budgets,full_coco_points=metrics,oracle_has_no_AP=True,
                  optimizer_updates=0,detector_forwards=0,replayed_original_roi_head_forwards=9402,external_data_opened=False,
                  time_seconds=dict(before_label_load=label_time-start,total=perf_counter()-start))
    args.output.mkdir()
    stamp=corpus.write_artifact(args.output/'RESULT_SAFE.json',corpus.encode(result))
    print(json.dumps(dict(status=result['status'],result_sha256=stamp['sha256'])),flush=True)


if __name__=='__main__':
    main()
