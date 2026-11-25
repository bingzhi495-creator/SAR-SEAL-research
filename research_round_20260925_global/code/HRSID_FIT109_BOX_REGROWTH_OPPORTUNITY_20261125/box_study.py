"""Fixed score-ordered RF candidate/box opportunity, no fitted or scored predictions."""

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


PROTOCOL = 'e1f90c707b9109ce8645175cdfeda932fd83fef86592d5ee8e61abf7d0314d36'
ORIGINAL = '10b1b0fde04d207bcbd220526b627fa7924158675707fdbfb7423e94b53ea8b2'
CAPTURE = '7f3f6dd112d287623360c8e3a174bae3a12a039c7f5505e4bf7c5d76b2e5ed6f'
OUTCOME = 'f105b46804c981913294caeb5c9bfe465513dc9aa8e8a379364968c63cbdcf70'
TERMINAL = 'ee66b96176ae2d0dfadfbeca46621d8b194c80da9a5e91536422cc1bb41c00cf'
METRICS=('native_raster_iou','crop_only','frozen_fixed_area','frozen_free_area','uniform_logit_threshold')


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def pin(path, digest):
    data=path.read_bytes()
    need(sha256(data).hexdigest()==digest,'Exact closed SHA: '+path.name)
    return data


def module(path,digest,name):
    source=ModuleType(name)
    source.__file__=str(path)
    sys.modules[name]=source
    exec(compile(pin(path,digest),str(path),'exec'),source.__dict__)
    return source


def inputs(args):
    p=args.prior
    previous=json.loads(pin(p/'diagnosis/RESULT_SAFE.json',OUTCOME))
    observed=json.loads(pin(p/'observed_diagnosis/TERMINAL.json',TERMINAL))
    need(observed['actual_exit']==0 and observed['output_footer_verified'] and observed['result_sha256']==OUTCOME
         and observed['worker_sha256']==ORIGINAL and previous['native_iou50_matches']==453, 'Original source score identities')
    oracle=module(p/'oracle.py',ORIGINAL,'_box_fixed_original')
    read=module(args.code/oracle.FORMAL/'run_fit109.py','9f989c3a976642d9e790d65199dda43da4ae6b946d649fa13d1106258e99f84e','_box_source_loader')
    spot=module(args.code/oracle.SPOT/'run_spot.py','75ff72a3177e99f06d212a878c8d8921de4db6da8f4249722fb762718083d0ee','_box_original_slots')
    args.acceptance_sha256=CAPTURE
    roster,capture=read.read_population(args,spot)
    raw=pin(args.capture/'STOCK_NATIVE_PREDICTIONS_PRIVATE.jsonl',capture['stock_predictions_sha256'])
    groups,stock,infos={},{},{}
    for (item,ordinal,block),line in zip(roster,raw.splitlines(),strict=True):
        frozen=json.loads(line)
        iid = item['image_id']
        rows=spot.native_candidate_rows(block)
        expect=[dict(image_id=iid,category_id=1,score=row['score'],segmentation=row['rle']) for row in rows]
        need(iid==frozen['image_id']==block['image_id'] and expect==frozen['coco']
             and frozen['native_ranks']==[row['rank'] for row in rows]
             and block['source_prefix']==item['group'], 'Exact native candidate identity, class, raster, confidence and order')
        all_boxes=spot.author_boxes(block['raw_boxes_cxcywh'],torch)
        stock[iid]=expect
        groups[iid]=item['group']
        infos[iid]=dict(boxes=np.asarray([all_boxes[row['query_id']] for row in rows],dtype=np.float64),
                        proof=block['logit_capture'],rows=rows,ordinal=ordinal)
    need(len(stock)==len(set(groups.values()))==109 and sum(map(len,stock.values()))==10488, 'All original prefix groups and mapped candidates')
    return stock,infos,groups,oracle,capture


def assign(annotations,boxes,reserved_hulls,reserved_candidates):
    assigned={}
    remaining=[ann for ann in annotations if ann['id'] not in reserved_hulls]
    for index,box in enumerate(boxes):
        if index in reserved_candidates or not remaining:
            continue
        gt=np.asarray([row['bbox'] for row in remaining],dtype=np.float64)
        lo,hi=gt[:,:2],gt[:,:2]+gt[:,2:]
        hit=np.maximum(0.,np.minimum(hi,box[2:])-np.maximum(lo,box[:2])).prod(axis=1)
        union=np.prod(box[2:]-box[:2])+np.prod(gt[:,2:],axis=1)-hit
        iou=np.divide(hit,union,out=np.zeros_like(hit),where=union>0.)
        target=int(np.argmax(iou))
        if iou[target]>=.50:
            assigned[remaining.pop(target)['id']]=index
    return assigned


def threshold_ceiling(logits,target,hit_outside,area_outside,truth_area):
    base=hit_outside/(area_outside+truth_area-hit_outside)
    if not len(logits):
        return float(base)
    order=np.argsort(-logits,kind='stable')
    ordered=logits[order]
    cuts=np.flatnonzero(np.r_[ordered[:-1]!=ordered[1:],True])
    numer=np.cumsum(target[order],dtype=np.int64)[cuts]+hit_outside
    denom=(cuts+1)+area_outside+truth_area-numer
    return float(max(base,float(np.max(numer/denom))))


def box_geometry(stock,truth,box):
    x0,y0,x1,y1=np.ceil(box).astype(np.int64)
    t=truth[y0:y1,x0:x1]
    m=stock[y0:y1,x0:x1]
    truth_area=int(truth.sum())
    area,area_in=int(stock.sum()),int(m.sum())
    hit,hit_in=int((stock & truth).sum()),int((m & t).sum())
    outside_hit,outside_area=hit-hit_in,area-area_in
    in_truth=int(t.sum())
    fixed=outside_hit+min(area_in,in_truth)
    free=outside_hit+in_truth
    return dict(native_raster_iou=hit/(area+truth_area-hit),crop_only=in_truth/truth_area,
                frozen_fixed_area=fixed/(area+truth_area-fixed),frozen_free_area=free/(outside_area+in_truth+truth_area-free),
                stock_area_over_gt=float(area/truth_area),stock_recall=hit/truth_area,
                stock_precision=hit/area if area else 0.,stock_inside_box_fraction=area_in/area if area else 0.,
                stock_area=area,truth_area=truth_area,outside_hit=outside_hit,outside_area=outside_area)


def native_logits(root,info):
    ref=info['proof']
    relative=ref['relative_path']
    need(relative==f'RAW_LOGITS/{info["ordinal"]:03d}.npy', 'Verified original raw logit sidecar')
    raw=pin(root/relative,ref['sha256'])
    with BytesIO(raw) as source:
        rows=np.load(source,allow_pickle=False)
        need(not source.read(1) and rows.dtype==np.dtype('<f4') and len(raw)==ref['bytes']
             and list(rows.shape)==ref['shape']==[len(ref['candidates']),144,144]
             and np.isfinite(rows).all(),'Closed exact original RF FP32 logit array')
    return {c['query_id']:(c,plane) for c,plane in zip(ref['candidates'],rows,strict=True)}


def threshold_at_position(root, info, index, stock, truth, bounds, available):
    if not 32<=bounds['stock_area']<1024:
        return None
    if not available:
        available.update(native_logits(root,info))
    row=info['rows'][index]
    proof,prior=available[row['query_id']]
    need(proof['native_rank']==row['rank'] and proof['score']==row['score'] and proof['predicted_area']==bounds['stock_area'], 'Exact archived query raw plane')
    with torch.inference_mode():
        shape=torch.from_numpy(prior).reshape(1,1,144,144)
        rendered=torch.nn.functional.interpolate(shape,size=(800,800),mode='bilinear',align_corners=False)[0,0].numpy()
    need(np.array_equal(rendered>0,stock),'Exact original RF native signs after original FP32 interpolation')
    x0,y0,x1,y1=np.ceil(info['boxes'][index]).astype(np.int64)
    return threshold_ceiling(rendered[y0:y1,x0:x1].ravel(),truth[y0:y1,x0:x1].ravel(),bounds['outside_hit'],bounds['outside_area'],bounds['truth_area'])


def summarize(rows):
    metrics={}
    names=(*METRICS,'stock_area_over_gt','stock_recall','stock_precision','stock_inside_box_fraction')
    for name in names:
        values=[row[name] for _,row in rows if row.get(name) is not None]
        result=dict(n=len(values),mean=float(np.mean(values)) if values else None)
        if name in METRICS:
            result['coverage']={f'{t:.2f}':sum(v>=t for v in values) for t in (.50,.75,.90)}
        if name=='stock_area_over_gt' and values:
            result['p10_median_p90']=np.percentile(values,[10,50,90]).tolist()
        metrics[name]=result
    return dict(assigned_hulls=len(rows),prefix_proxies=len({g for g,_ in rows}),metrics=metrics)


def study(gt,stock,infos,groups,anchors,coco,args):
    by_image=defaultdict(list)
    for ann in gt['annotations']:
        by_image[ann['image_id']].append(ann)
    scopes=defaultdict(list)
    counts=Counter()
    rescued=set()
    any_mask_before=set()
    for iid,hulls in by_image.items():
        info=infos[iid]
        original={ann['id']:anchors[ann['id']][1] for ann in hulls if ann['id'] in anchors}
        if original:
            need(len(set(original.values()))==len(original),'Original mask assignment is one-to-one')
        all_boxes=assign(hulls,info['boxes'],set(),set())
        residual=assign(hulls,info['boxes'],set(original),set(original.values()))
        old_masks=[p['segmentation'] for p in stock[iid]]
        raw_cache={}
        for ann in hulls:
            aid=ann['id']
            gt_rle=coco.annToRLE(ann)
            truth=codec.decode(gt_rle).astype(bool)
            g=groups[iid]
            if aid not in original:
                counts['original_unmatched']+=1
                max_native=float(np.max(codec.iou(old_masks,[gt_rle],[0]))) if old_masks else 0.
                if max_native>=.50:
                    counts['original_unmatched_with_any_original_mask_reaching50']+=1
                    any_mask_before.add(g)
                if aid not in residual:
                    counts['original_unmatched_without_residual_box']+=1
            for scope,current in (('residual_box',residual),('all_independent_box',all_boxes)):
                if aid not in current:
                    continue
                index=current[aid]
                stock_mask=codec.decode(old_masks[index]).astype(bool)
                bounds=box_geometry(stock_mask,truth,info['boxes'][index])
                med=scope+'_assigned'
                scopes[med].append((g,bounds))
                need(bounds['native_raster_iou']<=bounds['frozen_fixed_area']<=bounds['frozen_free_area'], 'Nested physically exact ROI ceilings')
                if bounds['native_raster_iou']>=.50:
                    continue
                scopes[scope+'_weak'].append((g,bounds))
                if scope!='residual_box':
                    continue
                limit=threshold_at_position(args.capture,info,index,stock_mask,truth,bounds,raw_cache)
                bounds['uniform_logit_threshold']=limit
                if limit is not None:
                    need(bounds['native_raster_iou']-1e-12<=limit<=bounds['frozen_free_area']+1e-12, 'Exact native-zero and nested threshold ceilings')
                elif bounds['stock_area']<32:
                    counts['weak_residual_threshold_missing_area_below32']+=1
                else:
                    counts['weak_residual_threshold_missing_area_ge1024']+=1
                if bounds['crop_only']>=.50 and bounds['frozen_free_area']>=.50:
                    counts['conservative_residual_opportunities']+=1
                    rescued.add(g)
                    counts['conservative_and_no_alt_native_mask_reaches50']+=int(max_native<.50)
    need(counts['original_unmatched']==43 and counts['original_unmatched_without_residual_box']+
         len(scopes['residual_box_assigned'])==43,'Complete original unmatched denominator')
    gate=counts['conservative_residual_opportunities']>=20 and len(rescued)>=8
    return dict(assignment_scope={name:summarize(scopes[name]) for name in ('residual_box_assigned','residual_box_weak',
              'all_independent_box_assigned','all_independent_box_weak')},counts=dict(counts),
              conservative_rescued_prefix_proxies=len(rescued),original_unmatched_any_native_ge50_prefixes=len(any_mask_before),
              passes_source_rescue_gate=gate)


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for attr in ('protocol','code','capture','acceptance','source-cohort','prior','fit-gt','output'):
        parser.add_argument('--'+attr,type=Path,required=True)
    args=parser.parse_args()
    need(not args.output.exists() and args.output.parent.is_dir(), 'Fresh independent result directory')
    t0=perf_counter()
    pin(args.protocol, PROTOCOL)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    need(sys.flags.isolated and sys.dont_write_bytecode and not torch.cuda.is_initialized(),'Original isolated CPU process')
    stock,infos,groups,oracle,capture=inputs(args)
    t_pre=perf_counter()
    print(json.dumps(dict(stage='verified_closed_original_stock_prior_to_labels', chips=len(stock))),flush=True)
    helper=module(args.code/'RFDETR_FIT109_QUOTA_COMPETITION_TARGETS_20261123/assemble_competition.py',
                  '3b0be851e38028248d5b646a108598585b6a261989496bf535d5ea8b54e6dd5b','_box_truth')
    gt=helper.annotations(args,stock)
    coco,matches=oracle.fixed_coco(gt,stock)
    need(len(matches)==453 and len(gt['annotations'])==496, 'Revalidate whole original mask match 453/496')
    result=study(gt,stock,infos,groups,matches,coco,args)
    result.update(status='HRSID_FIT109_FIXED_BOX_REGROWTH_OPPORTUNITY_V1',phase='score',protocol_sha256=PROTOCOL,
                  worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),source_capture_sha256=CAPTURE,
                  source_original_native_stock_sha256=capture['stock_predictions_sha256'],source_gt_sha256=helper.ORIGINAL_GT,
                  original_detector_fit_exposed=True,scope='HRSID_FIT109',images=len(stock),prefix_proxies=len(set(groups.values())),
                  source_candidates=sum(map(len,stock.values())),author_hulls=len(gt['annotations']),original_native_mask_matches=len(matches),
                  independent_all_box_is_not_recovery_denominator=True,label_conditioned_oracle_only=True,
                  AP_computed=False,new_model_predictions_written=False,extra_detector_forwards=0,optimizer_updates=0,point_head_restored=False,
                  elapsed_seconds=perf_counter()-t0,label_free_seconds=t_pre-t0)
    need(not torch.cuda.is_initialized(),'No GPU inference')
    raw=(json.dumps(result,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
    args.output.mkdir()
    with (args.output/'RESULT_SAFE.json').open('xb') as out:
        out.write(raw)
    print(json.dumps(dict(status=result['status'],result_sha256=sha256(raw).hexdigest())),flush=True)


if __name__=='__main__':
    main()
