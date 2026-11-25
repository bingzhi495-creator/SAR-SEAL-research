"""Test frozen ROI candidate ranking at exact native-preserving addition budgets."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType

import numpy as np
from pycocotools import mask as codec
import torch
import torch.nn.functional as F


PROTOCOL='e84e47358adfc6038376c9f864187f3ceea3b43f36672c1be5cfa47f789d519f'
PAIRED='1ad5d968ab07c564012c6f79999a5aa52c03b406891776346e7f6bd5384bb056'
PAIRED_RESULT='ff25ba9aad6106abdd2110deb8a8c108653b9b583d766bbffcf7b6e8ae3100ac'
PAIRED_TERMINAL='169104cd76362264674ba5c6fb47f13c9a5cc35bd31798e378e6893d79f36e95'
EDITS=('nearest','raw','sham','learned')
ARMS=('native','full_A',*EDITS)


def check(ok,why):
    if not ok:
        raise ValueError(why)


def pin(path,digest):
    data=path.read_bytes()
    check(sha256(data).hexdigest()==digest,'Original closed file: '+path.name)
    return data


def module(path,digest,name):
    m=ModuleType(name)
    m.__file__=str(path)
    sys.modules[name]=m
    exec(compile(pin(path,digest),str(path),'exec'),m.__dict__)
    return m


def budget(area,logits):
    return min(8,area//10,int(np.count_nonzero(logits>=0)))


def select(cells,priority,k):
    idx=np.lexsort((cells,priority))[:k]
    return cells[idx]


def shuffle(values,cells,shell,raw,identity):
    sham=values.copy()
    for ring in (1,2):
        group=np.flatnonzero(shell==ring)
        group=group[np.lexsort((cells[group],raw[group]))]
        for start in range(0,len(group),4):
            block=group[start:start+4]
            text='|'.join(map(str,('ROI_A_SHAM_V1',*identity,ring,start//4)))
            seed=int.from_bytes(sha256(text.encode()).digest()[:16],'little')
            sham[block]=values[np.random.Generator(np.random.PCG64(seed)).permutation(block)]
    return sham


def restore_models(args,paired,prior,device):
    source=module(args.model_source,paired.MODEL,'_native_roi_exact_user_model')
    models,digests={},{}
    for fold in range(5):
        name=f'fold{fold}_uniform_A.npz'
        evidence=prior['weights_closed_private'][name]
        raw=pin(args.paired_run/'result/weights_private'/name,evidence['sha256'])
        with np.load(BytesIO(raw),allow_pickle=False) as saved:
            state={key:torch.from_numpy(saved[key]) for key in saved.files}
        expected=prior['training'][str(fold)]['uniform_A']['final_tensor_sha256']
        check(len(raw)==evidence['bytes'] and paired.hash_weights(state)==evidence['tensor_sha256']==expected,
              'Exact original held-fold A tensor state')
        model=paired.create_model(source)
        check(sum(p.numel() for p in model.parameters())==paired.PARAMETERS, 'Original user model structure')
        model.load_state_dict(state,strict=True)
        models[fold]=model.to(device).eval()
        digests[fold]=expected
    return models,digests


def replay_raw(root,box,info,pos,bits,area,cache):
    if not cache:
        cache.update(box.native_logits(root,info))
    row=info['rows'][pos]
    expected,plane=cache[row['query_id']]
    check(expected['native_rank']==row['rank'] and expected['score']==row['score'] and expected['predicted_area']==area,
          'Closed original RF query/raster-specific input plane')
    original=torch.from_numpy(plane).reshape(1,1,144,144)
    native=F.interpolate(original,size=(800,800),mode='bilinear',align_corners=False)[0,0].numpy()
    check(np.array_equal(native>0,bits),'Original archived native raw logit sign parity')
    return native


def additions(args,meta,pos,coords_rgb,a_logits,box,rank,sparse,pilot,cache):
    crop,bits,area,rle=coords_rgb
    if not 32<=area<1024:
        return None,dict(outside_original_area_guard=1)
    box_coords=meta['boxes'][pos]
    pools,_=rank.editable_pools(bits,box_coords)
    one,two=pools['add1'],pools['add2']
    cells=np.concatenate((one[0],two[0]))
    if not len(cells):
        return None,dict(inside_area_guard=1,no_native_shell=1)
    distances=np.concatenate((one[1],two[1]))
    shell=np.repeat([1,2],[len(one[0]),len(two[0])])
    y,x=np.divmod(cells,800)
    x0,y0,x1,y1=crop
    check(len(np.unique(cells))==len(cells) and not bits.flat[cells].any() and np.all((x>=x0)&(x<x1)&(y>=y0)&(y<y1)),
          'Exact distinct original background ring within same frozen box')
    proposal=F.interpolate(a_logits,size=(y1-y0,x1-x0),mode='bilinear',align_corners=False)[0,0].numpy()
    score=proposal[y-y0,x-x0]
    k=budget(area,score)
    if not k:
        return None,dict(inside_area_guard=1,native_shell=1,zero_positive_budget=1)
    raw=replay_raw(args.capture,box,meta,pos,bits,area,cache).flat[cells]
    row=meta['rows'][pos]
    shuffled=shuffle(score,cells,shell,raw,(row['image_id'],row['rank'],row['query_id']))
    ranks=dict(nearest=distances,raw=-raw,learned=-score,sham=-shuffled)
    writes={}
    for arm,priority in ranks.items():
        pixels=select(cells,priority,k)
        result=sparse.xor_rle(rle,pixels)
        common=int(codec.area(codec.merge([rle,result],intersect=True)))
        check(len(np.unique(pixels))==k and not np.any(bits.flat[pixels]) and common==area
              and int(codec.area(result))==area+k and pilot.sample_rle(result,pixels,sparse).all(),
              'All native foreground unchanged, exact same-k original background additions')
        writes[arm]=(result,pixels)
    return writes,dict(inside_area_guard=1,native_shell=1,positive_k=1,total_k=k,
                       changed_sham_scores=int(np.count_nonzero(score!=shuffled)))


def infer(args,paired,corpus,box,rank,sparse,pilot,stock,infos,groups,schemas,folds,models,device,prior):
    worlds={arm:{} for arm in ARMS}
    all_active,issued={},{}
    census,effects=Counter(),Counter()
    by_arm={name:Counter() for name in EDITS}
    with torch.inference_mode():
        for ordinal,(iid,rows) in enumerate(stock.items(),1):
            rgb=corpus.rgb_from_original(args.fit_images,schemas[iid])
            info=infos[iid]
            for arm in ARMS:
                worlds[arm][iid]=[dict(row) for row in rows]
            take,features=[],[]
            all_active[iid],issued[iid]=set(),{}
            for pos,(row,native_box) in enumerate(zip(rows,info['boxes'],strict=True)):
                bits=codec.decode(row['segmentation']).astype(bool)
                crop=corpus.integer_crop(native_box,bits.shape)
                x0,y0,x1,y1=crop
                census['full_candidates']+=1
                if x0==x1 or y0==y1:
                    census['empty_integer_crops']+=1
                    continue
                if not paired.gate(bits,crop):
                    census['ineligible_due_to_native_fill']+=1
                    continue
                same,rgb48,stock48=corpus.sample_candidate(rgb,bits,native_box)
                check(same==crop,'Exact original label-free ROI input extraction')
                area=int(codec.area(row['segmentation']))
                take.append((pos,crop,bits,area,row['segmentation']))
                features.append((rgb48,stock48))
                census['admitted']+=1
                census['admitted_empty_stock']+=int(area==0)
                all_active[iid].add(pos)
            cache={}
            for first in range(0,len(take),48):
                x,m=zip(*features[first:first+48],strict=True)
                batch=torch.from_numpy(paired.tensor_channels(np.asarray(x),np.asarray(m))).to(device)
                scores=models[folds[groups[iid]]](batch).cpu()
                check(scores.shape==(len(x),1,48,48) and torch.isfinite(scores).all(),'Original ROI batch identity and finite logits')
                for j,(pos,crop,bits,area,rle) in enumerate(take[first:first+48]):
                    local=scores[j:j+1]
                    full,added,removed=paired.exact_native_write(bits,crop,local,corpus)
                    if added or removed:
                        worlds['full_A'][iid][pos]['segmentation']=full
                    effects.update(full_A_changed=int(bool(added or removed)),full_A_added=added,full_A_removed=removed)
                    writers,summary=additions(args,info,pos,(crop,bits,area,rle),local,box,rank,sparse,pilot,cache)
                    effects.update(summary)
                    if writers is None:
                        continue
                    issued[iid][pos]={}
                    for arm,(updated,pixels) in writers.items():
                        worlds[arm][iid][pos]['segmentation']=updated
                        issued[iid][pos][arm]=pixels
                        by_arm[arm].update(edited=1,added=len(pixels),removed=0)
            if ordinal%25==0:
                print(json.dumps(dict(stage='completed_native_preserving_mask_edits_before_labels',images=ordinal)),flush=True)
    old=prior['actual_masks_changed']['uniform_A']
    check(dict(census)==prior['equal_label_free_gate'] and census['full_candidates']==10488
          and effects['full_A_changed']==old['changed'] and effects['full_A_added']==old['added']
          and effects['full_A_removed']==old['removed'] and effects['positive_k']>0
          and all(v==by_arm['learned'] for v in by_arm.values()), 'Exact original full-A parity and matched added/no-deleted totals')
    return worlds,all_active,issued,dict(census),dict(effects),{name:dict(v) for name,v in by_arm.items()}


def fixed_geometry(gt,coco,matches,worlds,groups,issued,metrics,pilot,sparse):
    parent=defaultdict(list)
    recovered=Counter()
    changed=defaultdict(Counter)
    hit_comparison=defaultdict(Counter)
    for ann in gt['annotations']:
        aid,iid=ann['id'],ann['image_id']
        group=groups[iid]
        if aid not in matches:
            parent[group].append([0.]*len(ARMS))
            continue
        found,pos=matches[aid]
        check(iid==found,'Original fixed native paired image')
        reference=coco.annToRLE(ann)
        rles=[worlds[arm][iid][pos]['segmentation'] for arm in ARMS]
        values=codec.iou(rles,[reference],[0]).reshape(-1)
        check(values[0]>=.50-1e-12,'Original fixed .50 mask anchors')
        parent[group].append(values.tolist())
        action=issued[iid].get(pos,{})
        hits={}
        if action:
            recovered['active_original_fixed_hulls']+=1
            recovered['same_budget_each_arm_pixels']+=len(action['learned'])
        for i,arm in enumerate(ARMS[:-1]):
            if rles[i]!=rles[-1]:
                changed[arm][group]+=1
        if action:
            for arm,pixels in action.items():
                hits[arm]=int(pilot.sample_rle(reference,pixels,sparse).sum())
                recovered['real_author_foreground_'+arm]+=hits[arm]
            for arm in EDITS[:-1]:
                delta=hits['learned']-hits[arm]
                hit_comparison[arm].update(hulls_with_more=int(delta>0),hulls_with_less=int(delta<0),
                                           more_true_pixels=max(0,delta),fewer_true_pixels=max(0,-delta))
    groups_=np.asarray([np.mean(parent[g],axis=0) for g in sorted(parent)])
    all_hulls=np.asarray([v for g in parent.values() for v in g])
    check(len(groups_)==109 and len(all_hulls)==496 and len(matches)==453,'All original native fixed and unmatched hulls')
    rng=np.random.Generator(np.random.PCG64(5217))
    draws=rng.integers(len(groups_),size=(4096,len(groups_)))
    main_metrics={name:dict(prefix_balanced_iou_points=100*float(groups_[:,index].mean()),
                            reach={f'{t:.2f}':int((all_hulls[:,index]>=t).sum()) for t in (.50,.75,.90)},
                            lost_original50=int(((all_hulls[:,0]>=.50)&(all_hulls[:,index]<.50)).sum()))
                  for index,name in enumerate(ARMS)}
    comparisons={}
    for i,arm in enumerate(ARMS[:-1]):
        d=100*(groups_[:,-1]-groups_[:,i]); mean=float(d.mean())
        delta_ap=metrics['learned']['AP']-metrics[arm]['AP']
        ci=np.percentile(d[draws].mean(axis=1),[2.5,97.5]).tolist()
        comparisons[arm]=dict(AP_delta_points=delta_ap,IoU_delta_points=mean,IoU_95_CI_points=ci,
                              changed_fixed_hulls=sum(changed[arm].values()),changed_prefixes=len(changed[arm]),
                              arm_pass=bool(delta_ap>=.30 and mean>=.20 and ci[0]>0))
    exposure=comparisons['nearest']['changed_fixed_hulls']>=20 and comparisons['nearest']['changed_prefixes']>=8
    gate=exposure and all(comparisons[arm]['arm_pass'] for arm in ('native',*EDITS[:-1]))
    return dict(arms=main_metrics,planned_comparisons=comparisons,changed_mechanism_floor=exposure,passes=gate,
                additions_truth_hits=dict(recovered),truth_add_pair_differences={a:dict(v) for a,v in hit_comparison.items()})


def residual_geometry(gt,coco,matches,worlds,active,issued,infos,box):
    by_image=defaultdict(list)
    for row in gt['annotations']:
        by_image[row['image_id']].append(row)
    census=Counter()
    rescued={arm:Counter() for arm in ARMS}
    for iid,truth in by_image.items():
        anchors={h['id']:matches[h['id']][1] for h in truth if h['id'] in matches}
        candidates=box.assign(truth,infos[iid]['boxes'],set(anchors),set(anchors.values()))
        for item in truth:
            if item['id'] not in candidates:
                continue
            pos=candidates[item['id']]
            truth_rle=coco.annToRLE(item)
            rles=[worlds[arm][iid][pos]['segmentation'] for arm in ARMS]
            values=codec.iou(rles,[truth_rle],[0]).reshape(-1)
            check(values[0]<.50,'Exact original 30 stock-weak fixed residual assignment')
            census['residual_box_pairs']+=1
            census['original_roi_gate']+=pos in active[iid]
            area=int(codec.area(rles[0])); matched_add=issued[iid].get(pos,{})
            census['roi_and_original_area_guard']+=pos in active[iid] and 32<=area<1024
            census['all_matched_count_arms_positive_k']+=bool(matched_add)
            for i,arm in enumerate(ARMS):
                rescued[arm].update(actual_changed=int(rles[i]!=rles[0]),recovered50=bool(values[i]>=.50),
                                   recovered75=bool(values[i]>=.75),recovered90=bool(values[i]>=.90))
    check(len(gt['annotations'])-len(matches)==43 and census['residual_box_pairs']==30,'Complete original unmatched and fallback denominator')
    return dict(whole_original_unmatched=43,no_residual_box=13,gate_census=dict(census),fixed_rescued={a:dict(v) for a,v in rescued.items()})


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ('code','protocol','paired-run','corpus-run','box-run','prior','sibling-pilot','capture','acceptance',
                 'source-cohort','fit-gt','fit-images','model-source','output'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    wait=perf_counter()
    check(not args.output.exists() and args.output.parent.is_dir(),'Fresh evaluation output directory')
    pin(args.protocol,PROTOCOL)
    prior=json.loads(pin(args.paired_run/'result/RESULT_SAFE.json',PAIRED_RESULT))
    receipt=json.loads(pin(args.paired_run/'observed_paired/TERMINAL.json',PAIRED_TERMINAL))
    check(receipt['actual_exit']==0 and receipt['output_footer_verified'] and receipt['result_sha256']==PAIRED_RESULT
          and receipt['worker_sha256']==prior['worker_sha256']==PAIRED,'Original observed paired source fit concluded')
    paired=module(args.paired_run/'roi_paired.py',PAIRED,'_native_same_original_roi')
    device=paired.configure()
    check(str(device)==prior['device'],'Archived original precision and ROI inference device')
    corpus=module(args.corpus_run/'corpus.py',paired.CORPUS,'_native_exact_roi_corpus')
    box=module(args.box_run/'box_study.py',corpus.BOX_SOURCE,'_native_original_boxes')
    pilot=module(args.sibling_pilot/'pilot.py',paired.SIBLING_SOURCE,'_native_original_coco')
    sparse=module(args.code/'RFDETR_SPARSE_RLE_EQUIVALENCE_20261120/sparse_rle.py',
                  'cb9b8e839de85588fdf116ce5554ed6cadbc20bdbe76e38180561528bfd870d5','_native_exact_sparse_rle')
    spot=module(args.code/'RFDETR_CORE36_MATCHED_POINT_SPOT_20261120/run_spot.py',
                '75ff72a3177e99f06d212a878c8d8921de4db6da8f4249722fb762718083d0ee','_native_rank_pin')
    rank=module(args.code/'DFINE_SEG_EXTERNAL_BASELINE_FIT32_E24_20261112/e24/fit109_probability_rank/rank_math.py',
                spot.RANK_SHA,'_native_original_full_shell')
    stock,infos,groups,oracle,capture=box.inputs(args)
    schemas={v['image_id']:v for v in json.loads(pin(args.source_cohort,corpus.SOURCE_COHORT))['images'] if v['core']}
    plan=module(args.code/oracle.FORMAL/'fit_plan.py',corpus.FOLD_PLAN,'_native_original_prefix_split')
    folds=plan.source_folds(list(groups.values()))
    check(schemas.keys()==stock.keys() and capture['stock_predictions_sha256']==prior['original_stock_sha256'] and len(folds)==109,
          'All original stock candidate and source image roster identities')
    models,digests=restore_models(args,paired,prior,device)
    prepared=perf_counter()
    worlds,active,issued,census,effects,budgets=infer(args,paired,corpus,box,rank,sparse,pilot,
                                                  stock,infos,groups,schemas,folds,models,device,prior)
    before_labels=perf_counter()
    print(json.dumps(dict(stage='every_edit_verified_before_label_open',**effects)),flush=True)
    helper=module(args.code/'RFDETR_FIT109_QUOTA_COMPETITION_TARGETS_20261123/assemble_competition.py',
                  '3b0be851e38028248d5b646a108598585b6a261989496bf535d5ea8b54e6dd5b','_native_original_labels')
    gt=helper.annotations(args,stock)
    coco,matches=oracle.fixed_coco(gt,stock)
    metrics={arm:pilot.full_metrics(gt,worlds[arm],stock) for arm in ARMS}
    check(metrics['native']==prior['official_coco_points']['native'] and metrics['full_A']==prior['official_coco_points']['uniform_A'],
          'Both archived native and unmodified full A COCO scores reproduce exactly')
    geometry=fixed_geometry(gt,coco,matches,worlds,groups,issued,metrics,pilot,sparse)
    for arm,old in (('native','native'),('full_A','uniform_A')):
        check(abs(geometry['arms'][arm]['prefix_balanced_iou_points']-
                  prior['fixed_source_discrete']['arms'][old]['prefix_equal_iou_points'])<1e-12,
              'Exact original '+arm+' fixed geometry parity')
    residual=residual_geometry(gt,coco,matches,worlds,active,issued,infos,box)
    report=dict(status='HRSID_FIT109_NATIVE_PRESERVING_ROI_COMPLETE_V1',phase='score',protocol_sha256=PROTOCOL,
                worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),original_paired_sha256=PAIRED_RESULT,
                original_stock_sha256=capture['stock_predictions_sha256'],original_full_native_and_A_exact_parity=True,
                original_A_tensor_digests=digests,candidate_gate_census=census,exact_budget_census=effects,
                same_count_add_only_budgets=budgets,source_images=len(stock),source_hulls=len(gt['annotations']),
                source_native_matches=len(matches),heldout_source_prefix_proxies=len(folds),original_RF_source_fit_exposure=True,
                new_optimizer_updates=0,new_detector_forwards=0,archived_roi_head_candidate_forwards=census['admitted'],
                external_data_opened=False,full_coco_points=metrics,fixed_all_hull_discrete=geometry,fixed_residual_OP=residual,
                timing_seconds=dict(admission=prepared-wait,archived_roi_inference_and_no_label_edits=before_labels-prepared,
                score=perf_counter()-before_labels,total=perf_counter()-wait))
    args.output.mkdir()
    stamp=corpus.write_artifact(args.output/'RESULT_SAFE.json',corpus.encode(report))
    print(json.dumps(dict(status=report['status'],result_sha256=stamp['sha256'])),flush=True)


if __name__=='__main__':
    main()
