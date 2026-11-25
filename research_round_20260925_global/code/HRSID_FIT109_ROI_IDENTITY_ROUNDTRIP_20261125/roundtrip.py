"""Measure only the frozen native-mask ROI sampling and inverse cost."""

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


PROTOCOL='256f4fc9d61b272006f1e77baee07e7a57c9a66099a590f93d86f27c4612df52'
ROI_PAIRED='1ad5d968ab07c564012c6f79999a5aa52c03b406891776346e7f6bd5384bb056'
ROI_SCORE='ff25ba9aad6106abdd2110deb8a8c108653b9b583d766bbffcf7b6e8ae3100ac'
ROI_TERMINAL='169104cd76362264674ba5c6fb47f13c9a5cc35bd31798e378e6893d79f36e95'


def check(ok,why):
    if not ok:
        raise ValueError(why)


def pin(path,digest):
    raw=path.read_bytes()
    check(sha256(raw).hexdigest()==digest,'Original exact file: '+path.name)
    return raw


def load(path,digest,name):
    m=ModuleType(name)
    m.__file__=str(path)
    sys.modules[name]=m
    exec(compile(pin(path,digest),str(path),'exec'),m.__dict__)
    return m


def logits_for(mask):
    return torch.from_numpy(2*mask.astype(np.float32)-1)[None,None]


def render_identity(stock,crop,mask,inverse):
    return inverse(stock,crop,logits_for(mask))


def reconstruct(args,stock,infos,schemas,corpus,paired,locked):
    worlds={}
    census,areas=Counter(),defaultdict(Counter)
    with torch.inference_mode():
        for ordinal,(iid,predictions) in enumerate(stock.items(),1):
            rgb=corpus.rgb_from_original(args.fit_images,schemas[iid])
            outputs=[]
            for pred,box in zip(predictions,infos[iid]['boxes'],strict=True):
                old=codec.decode(pred['segmentation']).astype(bool)
                crop=corpus.integer_crop(box,old.shape)
                x0,y0,x1,y1=crop
                census['full_candidates']+=1
                if x0==x1 or y0==y1:
                    census['empty_integer_crops']+=1
                    outputs.append(pred)
                    continue
                if not paired.gate(old,crop):
                    census['ineligible_due_to_native_fill']+=1
                    outputs.append(pred)
                    continue
                census['admitted']+=1
                empty=not old.any()
                census['admitted_empty_stock']+=int(empty)
                source,_,square=corpus.sample_candidate(rgb,old,box)
                check(source==crop,'Exactly the same native camera patch and forward stock sampler')
                final=render_identity(old,crop,square,paired.render)
                changed=final!=old
                count=int(changed.sum())
                label='at_most_48_each_axis' if max(x1-x0,y1-y0)<=48 else 'greater_than_48_some_axis'
                areas[label].update(admitted=1,changed=int(bool(count)))
                check(int(changed[y0:y1,x0:x1].sum())==count,'Every original exterior bit is identical')
                check(not empty or not count,'Roundtrip exactly preserves empty stock image')
                census['changed_candidates']+=int(bool(count))
                census['added_pixels']+=int((changed&final).sum())
                census['removed_pixels']+=int((changed&old).sum())
                outputs.append(dict(pred,segmentation=corpus.encoded_crop(final)) if count else pred)
            worlds[iid]=outputs
            if ordinal%40==0:
                print(json.dumps(dict(stage='native_mask_roundtrip_prelabel', images=ordinal)),flush=True)
    check(len(worlds)==109 and all(census[k]==v for k,v in locked.items())
          and census['admitted']==9402 and census['full_candidates']==10488,'Original label-free full gate census parity')
    census['unchanged_candidates']=census['full_candidates']-census['changed_candidates']
    return worlds,dict(census),{name:dict(counts) for name,counts in areas.items()}


def geometry(gt,coco,anchors,original,worlds,groups):
    by_prefix=defaultdict(list)
    transitions=Counter()
    for ann in gt['annotations']:
        aid,iid=ann['id'],ann['image_id']
        if aid not in anchors:
            by_prefix[groups[iid]].append([0.,0.])
            continue
        same,pos=anchors[aid]
        check(same==iid,'Same exact original mask and author fixed match')
        masks=[p[iid][pos]['segmentation'] for p in (original,worlds)]
        value=codec.iou(masks,[coco.annToRLE(ann)],[0]).reshape(-1)
        check(value[0]>=.50-1e-12,'Original stock exact .50 assignment')
        by_prefix[groups[iid]].append(value.tolist())
        before,after=value[0]>.90,value[1]>.90
        if before and not after:
            transitions['strict90_lost']+=1
        if after and not before:
            transitions['strict90_new']+=1
    parent=np.asarray([np.mean(by_prefix[group],axis=0) for group in sorted(by_prefix)])
    hulls=np.asarray([v for p in by_prefix.values() for v in p])
    check(len(parent)==109 and len(hulls)==496 and len(anchors)==453,'All original hulls include unmodified and unmatched zero')
    draws=np.random.Generator(np.random.PCG64(5217)).integers(len(parent),size=(4096,len(parent)))
    difference=100*(parent[:,1]-parent[:,0])
    return dict(native_parent_iou_points=float(100*parent[:,0].mean()),roundtrip_parent_iou_points=float(100*parent[:,1].mean()),
                roundtrip_minus_native_iou_points=float(difference.mean()),
                paired_iou_percentile95_points=np.percentile(difference[draws].mean(axis=1),[2.5,97.5]).tolist(),
                native_strict_gt90=int((hulls[:,0]>.90).sum()),roundtrip_strict_gt90=int((hulls[:,1]>.90).sum()),
                strict_gt90_lost=transitions['strict90_lost'],strict_gt90_new=transitions['strict90_new'])


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for attr in ('protocol','code','paired-run','corpus-run','box-run','prior','sibling-pilot','capture','acceptance',
                 'source-cohort','fit-gt','fit-images','output'):
        p.add_argument('--'+attr,type=Path,required=True)
    args=p.parse_args()
    start=perf_counter()
    check(not args.output.exists() and args.output.parent.is_dir(),'Fresh result directory')
    pin(args.protocol,PROTOCOL)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    check(sys.flags.isolated and sys.dont_write_bytecode and not torch.cuda.is_initialized(),'CPU-only exact interpolation')
    prev=json.loads(pin(args.paired_run/'result/RESULT_SAFE.json',ROI_SCORE))
    terminal=json.loads(pin(args.paired_run/'observed_paired/TERMINAL.json',ROI_TERMINAL))
    check(terminal['actual_exit']==0 and terminal['output_footer_verified'] and terminal['result_sha256']==ROI_SCORE
          and terminal['worker_sha256']==prev['worker_sha256']==ROI_PAIRED and prev['native_fixed_matches']==453,
          'Qualified original paired ROI context')
    paired=load(args.paired_run/'roi_paired.py',ROI_PAIRED,'_roundtrip_exact_crop_inverse')
    corpus=load(args.corpus_run/'corpus.py',paired.CORPUS,'_roundtrip_exact_corpus')
    box=load(args.box_run/'box_study.py',corpus.BOX_SOURCE,'_roundtrip_original_stock_boxes')
    pilot=load(args.sibling_pilot/'pilot.py',paired.SIBLING_SOURCE,'_roundtrip_exact_co_co')
    stock,infos,groups,oracle,capture=box.inputs(args)
    schemas={v['image_id']:v for v in json.loads(pin(args.source_cohort,corpus.SOURCE_COHORT))['images'] if v['core']}
    check(schemas.keys()==stock.keys() and capture['stock_predictions_sha256']==prev['original_stock_sha256'],'Exact original effective native candidates')
    transformed,census,by_scale=reconstruct(args,stock,infos,schemas,corpus,paired,prev['equal_label_free_gate'])
    before_labels=perf_counter()
    print(json.dumps(dict(stage='all_candidate_masks_written_before_labels',**census)),flush=True)
    helper=load(args.code/'RFDETR_FIT109_QUOTA_COMPETITION_TARGETS_20261123/assemble_competition.py',
                '3b0be851e38028248d5b646a108598585b6a261989496bf535d5ea8b54e6dd5b','_roundtrip_original_gt')
    gt=helper.annotations(args,stock)
    coco,anchors=oracle.fixed_coco(gt,stock)
    native=pilot.full_metrics(gt,stock,stock)
    variant=pilot.full_metrics(gt,transformed,stock)
    check(native==prev['official_coco_points']['native'],'Exact original native full COCO parity')
    result=geometry(gt,coco,anchors,stock,transformed,groups)
    check(abs(result['native_parent_iou_points']-prev['fixed_source_discrete']['arms']['native']['prefix_equal_iou_points'])<1e-12,
          'Exact original native fixed geometry parity')
    loss=native['AP']-variant['AP']
    report=dict(status='HRSID_FIT109_STOCK_ROI_IDENTITY_ROUNDTRIP_COMPLETE_V1',phase='score',protocol_sha256=PROTOCOL,
                worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),original_paired_roi_sha256=ROI_SCORE,
                original_native_stock_sha256=capture['stock_predictions_sha256'],source_images=len(stock),native_matches=len(anchors),
                source_hulls=len(gt['annotations']),loss_AP_points=loss,practically_material_loss=loss>=.30,
                coco_points=dict(native=native,roundtrip=variant),fixed_geometry=result,actual_candidate_edits=census,
                admitted_and_changed_by_crop_size=by_scale,training_updates=0,model_forwards=0,
                external_data_opened=False,elapsed_seconds=perf_counter()-start,label_free_seconds=before_labels-start)
    args.output.mkdir()
    data=corpus.encode(report)
    stamp=corpus.write_artifact(args.output/'RESULT_SAFE.json',data)
    print(json.dumps(dict(status=report['status'],result_sha256=stamp['sha256'])),flush=True)


if __name__=='__main__':
    main()
