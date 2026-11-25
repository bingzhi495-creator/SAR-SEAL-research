"""Export private native 48x48 source supervision and label-free ROI extraction."""

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
from PIL import Image
from pycocotools import mask as codec
import torch


PROTOCOL='9ac5073a4a20463438dd482706578b8cb88f999a302f2240ee5fcb47ada76f9d'
BOX_SOURCE='55b6481bce8ad1a068a651841ce8b23c2c3710c46d160791228675f44eac541f'
BOX_RESULT='d482e0e72343f9d4fdf7f564fad043e0f199929c6552b4f36dc523eb8e9671ea'
BOX_TERMINAL='c9436d662324f66ebaaa37c67f7711a6172afe449ba89cdeacd4acd93abb812f'
SOURCE_COHORT='70e9a0e8fb30f5ba70b48b9067c1b95d07bfec23d6b248d92c5983727acceb6e'
FOLD_PLAN='1fd0fe8f815b5ad4e01d0d8eeca0aa0e1480433292fc48f94fee17e5837e5c35'
SIZE=48
ROLES=('weak','regular','negative')


def need(ok,why):
    if not ok:
        raise ValueError(why)


def pinned(path,digest):
    raw=path.read_bytes()
    need(sha256(raw).hexdigest()==digest,'Exact artifact '+path.name)
    return raw


def module(path,digest,name):
    instance=ModuleType(name)
    instance.__file__=str(path)
    sys.modules[name]=instance
    exec(compile(pinned(path,digest),str(path),'exec'),instance.__dict__)
    return instance


def encode(data):
    return (json.dumps(data,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()


def integer_crop(box,hw):
    coordinates=np.asarray(box,dtype=np.float64)
    h,w=hw
    need(coordinates.shape==(4,) and np.isfinite(coordinates).all() and np.all(coordinates[:2]<=coordinates[2:])
         and ((0<=coordinates)&(coordinates<=np.asarray([w,h,w,h]))).all(),'Exact original noninverted native ROI')
    return np.ceil(coordinates).astype(int).tolist()


def nearest(value):
    h,w=value.shape
    need(h>0 and w>0,'Nonempty original integer crop')
    centers=2*np.arange(SIZE)+1
    y=(centers*h)//(2*SIZE)
    x=(centers*w)//(2*SIZE)
    return value[y[:,None],x[None,:]].astype(np.uint8)


def bilinear_rgb(rgb):
    h,w,channels=rgb.shape
    need(h>0 and w>0 and channels==3 and rgb.dtype==np.uint8,'Original exact native RGB crop')
    centers=(np.arange(SIZE,dtype=np.float64)+.5)/SIZE
    y=np.clip(centers*h-.5,0,h-1)
    x=np.clip(centers*w-.5,0,w-1)
    y0,x0=np.floor(y).astype(int),np.floor(x).astype(int)
    y1,x1=np.minimum(y0+1,h-1),np.minimum(x0+1,w-1)
    wy,wx=(y-y0)[:,None,None],(x-x0)[None,:,None]
    upper=rgb[y0[:,None],x0[None,:],:]*(1-wx)+rgb[y0[:,None],x1[None,:],:]*wx
    lower=rgb[y1[:,None],x0[None,:],:]*(1-wx)+rgb[y1[:,None],x1[None,:],:]*wx
    return np.ascontiguousarray(np.rint((1-wy)*upper+wy*lower).transpose(2,0,1),dtype=np.uint8)


def sample_candidate(rgb, stock, box):
    """Label-free entry for any original RF candidate, independent of corpus membership."""
    need(stock.shape==rgb.shape[:2] and stock.dtype==np.bool_,'Same original RF mask and image')
    crop=integer_crop(box,stock.shape)
    x0,y0,x1,y1=crop
    if x0==x1 or y0==y1:
        return None
    return crop,bilinear_rgb(rgb[y0:y1,x0:x1]),nearest(stock[y0:y1,x0:x1])


def encoded_crop(target):
    raw=codec.encode(np.asfortranarray(target,dtype=np.uint8))
    return dict(size=list(raw['size']),counts=raw['counts'].decode('ascii'))


def overlaps(box,gt):
    if not len(gt):
        return np.asarray([])
    lo,hi=gt[:,:2],gt[:,:2]+gt[:,2:]
    hit=np.maximum(0.,np.minimum(hi,box[2:])-np.maximum(lo,box[:2])).prod(axis=1)
    union=np.prod(box[2:]-box[:2])+np.prod(gt[:,2:],axis=1)-hit
    return np.divide(hit,union,out=np.zeros_like(hit),where=union>0.)


def negatives(boxes,gt,occupied,blocked,limit):
    selected=[]
    for index,box in enumerate(boxes):
        if len(selected)==limit:
            break
        if index in blocked:
            continue
        max_iou=float(np.max(overlaps(box,gt),initial=0.))
        if max_iou>=.20:
            continue
        x0,y0,x1,y1=integer_crop(box,occupied.shape)
        if x0==x1 or y0==y1 or occupied[y0:y1,x0:x1].any():
            continue
        selected.append((index,max_iou))
    return selected


def rgb_from_original(root,info):
    raw=pinned(root/info['native_basename'],info['encoded_sha256'])
    with Image.open(BytesIO(raw)) as p:
        need(p.mode=='RGB' and p.format=='PNG' and p.size==(800,800),'Original RGB 800px chip')
        values=np.asarray(p)
    need(len(raw)==info['encoded_bytes'] and sha256(values.tobytes()).hexdigest()==info['decoded_rgb_sha256'], 'Original decoded RGB identity')
    return values


def build_rows(args,box,stock,infos,groups,gt,coco,original_mask,schemas,folds):
    by_image=defaultdict(list)
    for ann in gt['annotations']:
        by_image[ann['image_id']].append(ann)
    rgb,stock48,targets,inputs,labels=[],[],[],[],[]
    counts=Counter()
    foldcount=defaultdict(Counter)
    role_groups=defaultdict(set)
    for iid, rows in stock.items():
        group=groups[iid]
        fold=folds[group]
        info=infos[iid]
        identity=schemas[iid]
        need(identity['group']==group,'Original source prefix role')
        geometry=info['boxes']
        hulls=by_image[iid]
        holdout=[original_mask[ann['id']][1] for ann in hulls if ann['id'] in original_mask]
        positive={index:aid for aid,index in box.assign(hulls,geometry,set(),set()).items()}
        rle={ann['id']:coco.annToRLE(ann) for ann in hulls}
        occupied=codec.decode(codec.merge(list(rle.values()))).astype(bool) if rle else np.zeros((800,800),dtype=bool)
        gt_boxes=np.asarray([ann['bbox'] for ann in hulls],dtype=np.float64)
        negative=dict(negatives(geometry,gt_boxes,occupied,set(positive)|set(holdout),len(positive)))
        counts['negative_shortfall']+=len(positive)-len(negative)
        counts['images_with_negative_shortfall']+=len(negative)<len(positive)
        if not positive and not negative:
            continue
        image=rgb_from_original(args.fit_images,identity)
        meta={ann['id']:ann for ann in hulls}
        for index in sorted(set(positive)|set(negative)):
            native=rows[index]['segmentation']
            source=info['rows'][index]
            crop,sampled,base=sample_candidate(image,codec.decode(native).astype(bool),geometry[index])
            x0,y0,x1,y1=crop
            if index in positive:
                aid=positive[index]
                own_rle=rle[aid]
                mask_iou=float(codec.iou([native],[own_rle],[0])[0,0])
                type_name='weak' if mask_iou<.50 else 'regular'
                target=codec.decode(own_rle)[y0:y1,x0:x1]
                box_match=float(overlaps(geometry[index],np.asarray([meta[aid]['bbox']],dtype=np.float64))[0])
                need(box_match>=.50,'Reproduce exact one-to-one box assignment')
            else:
                aid=None
                mask_iou=None
                type_name='negative'
                target=np.zeros((y1-y0,x1-x0),dtype=np.uint8)
                box_match=float(negative[index])
                need(not occupied[y0:y1,x0:x1].any() and box_match<.20,'Training-only unambiguous negative')
            sampled_truth=nearest(target)
            roi_index=len(inputs)
            counts[type_name]+=1
            foldcount[fold][type_name]+=1
            role_groups[type_name].add(group)
            counts['empty_resampled_positives']+=int(aid is not None and not sampled_truth.any())
            counts['empty_native_positive_target']+=int(aid is not None and not target.any())
            counts['empty_resampled_stock']+=int(not base.any())
            counts[type_name+'_exact_crop_true_pixels']+=int(target.sum())
            counts[type_name+'_resampled_true_pixels']+=int(sampled_truth.sum())
            counts['negative_samples_with_native_stock_foreground']+=int(type_name=='negative' and bool(codec.area(native)))
            rgb.append(sampled)
            stock48.append(base)
            targets.append(sampled_truth)
            inputs.append(dict(roi_index=roi_index,image_id=iid,source_prefix=group,fold=fold,native_rank=source['rank'],
                               query_id=source['query_id'],score=source['score'],box_xyxy=geometry[index].tolist(),crop_xyxy=crop,
                               native_hw=[800,800],sample_hw=[SIZE,SIZE]))
            labels.append(dict(roi_index=roi_index,role=type_name,annotation_id=aid,stock_iou=mask_iou,
                               box_iou_or_max_gt_box_iou=box_match,native_target_crop_rle=encoded_crop(target)))
    need(counts['weak']==37 and counts['regular']==444 and counts['weak']+counts['regular']==481
         and counts['negative']+counts['negative_shortfall']==481,'Original positive cohort and actual limited negative counts')
    n=len(inputs)
    need(n==len(labels)==sum(counts[key] for key in ROLES) and [r['roi_index'] for r in inputs]==list(range(n))
         and [r['roi_index'] for r in labels]==list(range(n)), 'Exactly aligned separated feature/label rows')
    rgb,stock48,targets=np.asarray(rgb),np.asarray(stock48),np.asarray(targets)
    need(rgb.shape==(n,3,SIZE,SIZE) and stock48.shape==targets.shape==(n,SIZE,SIZE)
         and rgb.dtype==stock48.dtype==targets.dtype==np.uint8 and set(np.unique(stock48))<={0,1}
         and set(np.unique(targets))<={0,1},'Unmodified binary target and exact RGB frame tensor schema')
    report=dict(counts=dict(counts),prefix_proxies_with_role={key:len(role_groups[key]) for key in ROLES},
                per_fold_counts={str(k):{role:foldcount[k][role] for role in ROLES} for k in range(5)},
                folds=5,supervised_roi_rows=n,source_prefix_proxies=len(folds))
    return (rgb,stock48,targets,inputs,labels),report


def write_artifact(path, data):
    with path.open('xb') as f:
        f.write(data)
    expected=sha256(data).hexdigest()
    need(sha256(path.read_bytes()).hexdigest()==expected,'Independent exact byte readback')
    return dict(sha256=expected,bytes=len(data))


def npz_bytes(**arrays):
    with BytesIO() as raw:
        np.savez(raw,**arrays)
        return raw.getvalue()


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for role in ('code','protocol','capture','acceptance','source-cohort','prior','box-run','fit-gt','fit-images','output'):
        parser.add_argument('--'+role,type=Path,required=True)
    args=parser.parse_args()
    t0=perf_counter()
    need(not args.output.exists() and args.output.parent.is_dir(),'Fresh private corpus directory')
    pinned(args.protocol,PROTOCOL)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    need(sys.flags.isolated and sys.dont_write_bytecode and not torch.cuda.is_initialized(),'CPU-only isolated ROI export')
    observed=json.loads(pinned(args.box_run/'observed_pilot/TERMINAL.json',BOX_TERMINAL))
    accepted=json.loads(pinned(args.box_run/'result/RESULT_SAFE.json',BOX_RESULT))
    need(observed['actual_exit']==0 and observed['output_footer_verified'] and observed['result_sha256']==BOX_RESULT
         and observed['worker_sha256']==BOX_SOURCE and accepted['passes_source_rescue_gate'], 'Qualified original box-support diagnostic')
    base=module(args.box_run/'box_study.py',BOX_SOURCE,'_roi_box_study')
    stock,infos,groups,oracle,capture=base.inputs(args)
    source=json.loads(pinned(args.source_cohort,SOURCE_COHORT))
    schemas={row['image_id']:row for row in source['images'] if row['core']}
    need(schemas.keys()==stock.keys(),'All and only exact 109 independent encoded and decoded images')
    plan=module(args.code/oracle.FORMAL/'fit_plan.py',FOLD_PLAN,'_roi_original_fold_plan')
    folds=plan.source_folds(list(groups.values()))
    prelabel=perf_counter()
    print(json.dumps(dict(stage='full_original_stock_roster_verified_before_label_read', images=len(stock),
                          full_candidates=sum(map(len,stock.values())))),flush=True)
    helper=module(args.code/'RFDETR_FIT109_QUOTA_COMPETITION_TARGETS_20261123/assemble_competition.py',
                  '3b0be851e38028248d5b646a108598585b6a261989496bf535d5ea8b54e6dd5b','_roi_original_author_gt')
    gt=helper.annotations(args,stock)
    coco,original=oracle.fixed_coco(gt,stock)
    need(len(original)==453 and len(gt['annotations'])==496,'Original source native match denominator unchanged')
    (rgb,masks,targets,features,labels),report=build_rows(args,base,stock,infos,groups,gt,coco,original,schemas,folds)
    written=perf_counter()
    args.output.mkdir()
    (args.output/'inputs').mkdir()
    (args.output/'labels').mkdir()
    values={'inputs/INPUTS_PRIVATE.npz':npz_bytes(rgb=rgb,stock=masks),
            'inputs/INPUT_ROSTER_PRIVATE.jsonl':b''.join(map(encode,features)),
            'labels/TARGETS_PRIVATE.npz':npz_bytes(target=targets),
            'labels/LABEL_ROSTER_PRIVATE.jsonl':b''.join(map(encode,labels))}
    artifacts={name:write_artifact(args.output/name,value) for name,value in values.items()}
    need(not torch.cuda.is_initialized(),'No detector or other GPU execution')
    report.update(status='HRSID_FIT109_NATIVE_WEAK_ROI_CORPUS_CLOSED_V1',phase='score',protocol_sha256=PROTOCOL,
                  worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),source_box_opportunity_sha256=BOX_RESULT,
                  source_capture_acceptance_sha256=base.CAPTURE,source_stock_full_roster_sha256=capture['stock_predictions_sha256'],
                  source_query_full_roster_sha256=capture['queries_sha256'],source_image_manifest_sha256=capture['image_manifest_sha256'],
                  source_ground_truth_sha256=helper.ORIGINAL_GT,detector_source_fit_exposure=True,source_images=len(stock),
                  source_full_candidates=sum(map(len,stock.values())),source_hulls=len(gt['annotations']),original_native_matches=len(original),
                  sampled_rgb_chw_shape=[3,SIZE,SIZE],sampled_binary_shape=[SIZE,SIZE],training_rows_label_conditioned=True,
                  full_roi_inference_extractor='sample_candidate',training_performed=False,model_forwards=0,AP_computed=False,
                  external_data_opened=False,artifacts=artifacts,timing_seconds=dict(label_free_seconds=prelabel-t0,
                  ground_truth_sampling_seconds=written-prelabel,write_verify_seconds=perf_counter()-written,total=perf_counter()-t0))
    stamp=write_artifact(args.output/'RESULT_SAFE.json',encode(report))
    print(json.dumps(dict(status=report['status'],result_sha256=stamp['sha256'])),flush=True)


if __name__=='__main__':
    main()
