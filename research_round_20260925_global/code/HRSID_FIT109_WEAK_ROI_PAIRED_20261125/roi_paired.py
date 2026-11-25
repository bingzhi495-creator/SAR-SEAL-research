"""Run the fixed paired CleanUNet++ source ROI weight experiment and full source score."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType

import numpy as np
from pycocotools import mask as codec
import torch
import torch.nn.functional as F


PROTOCOL='32223edd50a39073132b2a645d147fca260ead1db8d98c4cc1899f3d93f974f2'
MODEL='d070accd73c5fd08f5cc6b3733754e4569dfdd5adbe3d52360cc3eda646f61eb'
CORPUS='3e90ccaa511593675e0cfa24d8d17b10b3f3ce5c57d189a580567a2a716e3108'
CORPUS_RESULT='7d326fd6404e4611e754d94e730e768330f4440248377cb170fed0de25ecbe2a'
CORPUS_TERMINAL='f161cadcbb0103a2f2237445a4efb23ca3b1fd67f66ada6044b2d0de13d91fa2'
SIBLING_RESULT='f0bdbd6557a7cd40589a0a8007b081d5ce07bf23f867b34714095043a0e60323'
SIBLING_SOURCE='dd35a407d09786af0f6aa3cc61fe86f25edc9a3c3af7a212a9310639720b6960'
ARMS=('native','uniform_A','weak_weighted_B')
BATCH=12
UPDATES=160
PARAMETERS=573972


def need(ok,why):
    if not ok:
        raise ValueError(why)


def pinned(path,digest):
    payload=path.read_bytes()
    need(sha256(payload).hexdigest()==digest,'Exact frozen source or archive '+path.name)
    return payload


def module(path,digest,name):
    m=ModuleType(name)
    m.__file__=str(path)
    sys.modules[name]=m
    exec(compile(pinned(path,digest),str(path),'exec'),m.__dict__)
    return m


def create_model(user_source):
    return user_source.CleanUNetPlusPlus(num_classes=1,in_channels=2,base_channels=8,
                                        deep_supervision=False,norm_type='group')


def configure():
    need(sys.flags.isolated and sys.dont_write_bytecode,'Isolated normal model runtime')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if torch.cuda.is_available():
        need(os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8','Fixed deterministic CUDA workspace')
        need(torch.cuda.mem_get_info()[0]>=4*1024**3,'GPU supports the matched FP32 training schedule')
        return torch.device('cuda:0')
    return torch.device('cpu')


def per_instance_loss(logits,targets):
    bce=F.binary_cross_entropy_with_logits(logits,targets,reduction='none').mean(dim=(1,2,3))
    probabilities=logits.sigmoid()
    numerator=2*(probabilities*targets).sum(dim=(1,2,3))+1
    denominator=(probabilities+targets).sum(dim=(1,2,3))+1
    return bce+1-numerator/denominator


def objective(logits,targets,weights):
    return (per_instance_loss(logits,targets)*weights).sum()/weights.sum()


def tensor_channels(rgb,stock):
    weight=np.asarray([.299,.587,.114],dtype=np.float32)[None,:,None,None]
    gray=(rgb.astype(np.float32)*weight).sum(axis=1)/np.float32(255.)
    return np.stack((gray,stock.astype(np.float32)),axis=1)


def gate(stock,crop):
    x0,y0,x1,y1=crop
    area=(x1-x0)*(y1-y0)
    return area>0 and 20*np.count_nonzero(stock[y0:y1,x0:x1])<13*area


def render(stock,crop,logits):
    x0,y0,x1,y1=crop
    resized=F.interpolate(logits,size=(y1-y0,x1-x0),mode='bilinear',align_corners=False)
    target=resized[0,0].numpy()>=0
    result=stock.copy()
    result[y0:y1,x0:x1]=target
    return result


def hash_weights(state):
    h=sha256()
    for key in sorted(state):
        a=np.ascontiguousarray(state[key].numpy())
        h.update(json.dumps([key,list(a.shape),str(a.dtype)],separators=(',',':')).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def roi_data(args,corpus,info):
    root=args.corpus_run/'result'
    artifacts={name:pinned(root/name,ref['sha256']) for name,ref in info['artifacts'].items()}
    feature,labels=([json.loads(line) for line in artifacts[path].splitlines()] for path in (
                   'inputs/INPUT_ROSTER_PRIVATE.jsonl','labels/LABEL_ROSTER_PRIVATE.jsonl'))
    i=np.asarray([row['roi_index'] for row in feature])
    f=np.asarray([row['fold'] for row in feature])
    both={}
    for row,label in zip(feature,labels,strict=True):
        need(row['roi_index']==label['roi_index'] and 'role' not in row,'Separated original supervised role linkage')
        both.setdefault(row['source_prefix'],row['fold'])
        need(both[row['source_prefix']]==row['fold'],'No source-prefix leakage between original folds')
    need(np.array_equal(i,np.arange(840)) and len(labels)==840 and len(both)==109 and set(f.tolist())==set(range(5)),
         'Exact original source corpus rows and prefix folds')
    with np.load(BytesIO(artifacts['inputs/INPUTS_PRIVATE.npz']),allow_pickle=False) as a, (
            np.load(BytesIO(artifacts['labels/TARGETS_PRIVATE.npz']),allow_pickle=False)) as t:
        rgb,stock,y=a['rgb'],a['stock'],t['target']
    need(rgb.shape==(840,3,48,48) and stock.shape==y.shape==(840,48,48)
         and rgb.dtype==stock.dtype==y.dtype==np.uint8 and set(np.unique(stock))<={0,1}
         and set(np.unique(y))<={0,1},'Original exact two-input source arrays')
    weak=np.asarray([r['role']=='weak' for r in labels],dtype=bool)
    need(int(weak.sum())==37 and [int((weak & (f==j)).sum()) for j in range(5)]==[0,3,4,19,11],
         'Original held-source weak cases including zero fold')
    return tensor_channels(rgb,stock),y[:,None].astype(np.float32),weak,f,both,feature,labels


def train_pair(source,device,x,y,is_weak,folds):
    data=torch.from_numpy(x).to(device)
    labels=torch.from_numpy(y).to(device)
    weights=torch.from_numpy(np.where(is_weak,3.,1.).astype(np.float32)).to(device)
    models,receipts,states={},{},{}
    for fold in range(5):
        torch.manual_seed(5217+fold)
        common=create_model(source)
        initial={key:t.detach().clone() for key,t in common.state_dict().items()}
        total=sum(p.numel() for p in common.parameters())
        need(total==PARAMETERS and not common.deep_supervision,'Exact unchanged user CleanUNet++ structure')
        rest=np.flatnonzero(folds!=fold)
        draws=np.random.Generator(np.random.PCG64(20261125+fold)).choice(rest,size=(UPDATES,BATCH),replace=True)
        draw_ids=torch.from_numpy(draws.astype(np.int64)).to(device)
        models[fold],receipts[fold]={},{}
        for arm in ARMS[1:]:
            t0=perf_counter()
            m=create_model(source)
            m.load_state_dict(initial,strict=True)
            need(all(torch.equal(m.state_dict()[key],val) for key,val in initial.items()),'Identical initialization across arms')
            m.to(device).train()
            opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01,betas=(.9,.999),eps=1e-8,foreach=False)
            losses=[]
            for index in draw_ids:
                opt.zero_grad(set_to_none=True)
                selected=weights[index] if arm=='weak_weighted_B' else torch.ones(BATCH,device=device,dtype=torch.float32)
                loss=objective(m(data[index]),labels[index],selected)
                loss.backward()
                opt.step()
                losses.append(float(loss.detach().item()))
            final={key:t.detach().cpu().clone() for key,t in m.state_dict().items()}
            states[(fold,arm)]=final
            count={int(v['step'].item()) for v in opt.state.values()}
            need(count=={UPDATES} and np.isfinite(losses).all() and all(torch.isfinite(t).all() for t in final.values()),
                 'Every active parameter trained for all fixed updates with finite values')
            first_hash,last_hash=hash_weights(initial),hash_weights(final)
            need(first_hash!=last_hash,'Actual learned tensor update')
            if device.type=='cuda':
                torch.cuda.synchronize(device)
            models[fold][arm]=m.eval()
            doc=dict(updates=len(losses),optimizer_steps=UPDATES,training_rows=len(rest),heldout_rows=int((folds==fold).sum()),
                     heldout_weak=int((is_weak & (folds==fold)).sum()),weak_draws=int(is_weak[draws].sum()),
                     batch=BATCH,total_parameters=total,active_optimizer_parameter_tensors=len(opt.state),
                     initialization_sha256=first_hash,final_tensor_sha256=last_hash,batch_draw_sha256=sha256(draws.astype('<i8').tobytes()).hexdigest(),
                     seconds=perf_counter()-t0,first_loss=losses[0],last_loss=losses[-1])
            receipts[fold][arm]=doc
            print(json.dumps(dict(stage='trained_matched_heldout_prefix_fold', fold=fold,arm=arm,updates=UPDATES,
                                  seconds=doc['seconds'])),flush=True)
    return models,receipts,states


def exact_native_write(stock,crop,logits,corpus):
    mask=render(stock,crop,logits)
    x0,y0,x1,y1=crop
    changed=mask!=stock
    need(int(changed.sum())==int(changed[y0:y1,x0:x1].sum()),'Every modified cell confined to original integer box')
    return corpus.encoded_crop(mask),int((changed & mask).sum()),int((changed & stock).sum())


def infer_label_free(args,corpus,stock,infos,groups,schemas,index,models,device):
    worlds={arm:{} for arm in ARMS}
    census=Counter()
    changes={arm:Counter() for arm in ARMS[1:]}
    per_fold=defaultdict(Counter)
    active={}
    modified={arm:{} for arm in ARMS[1:]}
    with torch.inference_mode():
        for ordinal,(iid,rows) in enumerate(stock.items(),1):
            prefix=groups[iid]
            fold=index[prefix]
            rgb=corpus.rgb_from_original(args.fit_images,schemas[iid])
            roi,indices=[],[]
            active[iid]=set()
            for arm in ARMS:
                worlds[arm][iid]=[dict(row) for row in rows]
            for arm in ARMS[1:]:
                modified[arm][iid]=set()
            for pos,(row,box) in enumerate(zip(rows,infos[iid]['boxes'],strict=True)):
                bits=codec.decode(row['segmentation']).astype(bool)
                crop=corpus.integer_crop(box,bits.shape)
                x0,y0,x1,y1=crop
                if x0==x1 or y0==y1:
                    census['empty_integer_crops']+=1
                    continue
                if not gate(bits,crop):
                    census['ineligible_due_to_native_fill']+=1
                    continue
                actual,r,m=corpus.sample_candidate(rgb,bits,box)
                need(actual==crop,'Exact label-free crop function')
                roi.append((r,m))
                indices.append((pos,crop,bits))
                active[iid].add(pos)
                census['admitted']+=1
                census['admitted_empty_stock']+=int(not bits.any())
                per_fold[fold]['admitted']+=1
            for start in range(0,len(roi),48):
                rx,rm=zip(*roi[start:start+48],strict=True)
                batch=torch.from_numpy(tensor_channels(np.asarray(rx),np.asarray(rm))).to(device)
                for arm in ARMS[1:]:
                    predicted=models[fold][arm](batch).cpu()
                    local=indices[start:start+48]
                    need(predicted.shape==(len(local),1,48,48) and torch.isfinite(predicted).all(),'Finite 48 ROI logits')
                    for k,(pos,crop,bits) in enumerate(local):
                        rle,add,drop=exact_native_write(bits,crop,predicted[k:k+1],corpus)
                        if add or drop:
                            worlds[arm][iid][pos]['segmentation']=rle
                            modified[arm][iid].add(pos)
                        changes[arm].update(admitted=1,changed=int(bool(add or drop)),added=add,removed=drop)
                        per_fold[fold][arm+'_changed']+=int(bool(add or drop))
            census['full_candidates']+=len(rows)
            per_fold[fold]['full_candidates']+=len(rows)
            if ordinal%25==0:
                print(json.dumps(dict(stage='label_free_heldout_inference', images=ordinal,admitted=census['admitted'])),flush=True)
    need(census['full_candidates']==10488==census['admitted']+census['ineligible_due_to_native_fill']+census['empty_integer_crops']
         and all(v['admitted']==census['admitted'] for v in changes.values()),'Same complete original score candidate population and gate')
    return worlds,active,modified,dict(census),{a:dict(v) for a,v in changes.items()},{str(k):dict(v) for k,v in per_fold.items()}


def point_iou(worlds,iid,pos,truth):
    masks=[worlds[arm][iid][pos]['segmentation'] for arm in ARMS]
    return codec.iou(masks,[truth],[0]).reshape(-1)


def recovery_table(entries,coco,worlds,active,modified):
    count={arm:Counter() for arm in ARMS}
    transitions=Counter()
    for ann,iid,pos in entries:
        values=point_iou(worlds,iid,pos,coco.annToRLE(ann))
        need(values[0]<.50,'Exact original weak matched cohort')
        a,b=values[1]>=.50,values[2]>=.50
        key='both' if a and b else 'A_only' if a else 'B_only' if b else 'neither'
        transitions[key]+=1
        for i,arm in enumerate(ARMS):
            count[arm].update(rescued=bool(values[i]>=.50),reached75=bool(values[i]>=.75),reached90=bool(values[i]>=.90))
            if arm!='native':
                count[arm].update(admitted=pos in active[iid],changed=pos in modified[arm][iid])
    return dict(denominator=len(entries),arm={k:dict(v) for k,v in count.items()},
                recovery_transition={k:transitions[k] for k in ('both','A_only','B_only','neither')},
                net_B_vs_A=transitions['B_only']-transitions['A_only'])


def weak_and_residual(gt,worlds,coco,match,stock,infos,features,corpus_labels,box,active,modified):
    all_gt={g['id']:g for g in gt['annotations']}
    records=defaultdict(list)
    for ann in gt['annotations']:
        records[ann['image_id']].append(ann)
    rank_position={iid:{v['rank']:j for j,v in enumerate(infos[iid]['rows'])} for iid in stock}
    weak=[]
    for src,y in zip(features,corpus_labels,strict=True):
        if y['role']!='weak':
            continue
        iid,aid=src['image_id'],y['annotation_id']
        weak.append((all_gt[aid],iid,rank_position[iid][src['native_rank']]))
    residual=[]
    for iid,hulls in records.items():
        reserve={ann['id']:match[ann['id']][1] for ann in hulls if ann['id'] in match}
        for aid,pos in box.assign(hulls,infos[iid]['boxes'],set(reserve),set(reserve.values())).items():
            residual.append((all_gt[aid],iid,pos))
    need(len(weak)==37 and len(residual)==30,'Original independent weak samples and residual box-pair identities')
    return dict(held_original_weak=recovery_table(weak,coco,worlds,active,modified),
                original_30_unmatched_residual=recovery_table(residual,coco,worlds,active,modified))


def main_geometry(gt,coco,matches,worlds,groups,ap):
    values=defaultdict(list)
    mutations=Counter()
    for ann in gt['annotations']:
        aid,iid=ann['id'],ann['image_id']
        g=groups[iid]
        if aid not in matches:
            values[g].append([0.]*3)
            continue
        sample_iid,pos=matches[aid]
        need(sample_iid==iid,'Same frozen candidate and hull image')
        iou=point_iou(worlds,iid,pos,coco.annToRLE(ann))
        need(iou[0]>=.50-1e-12,'Native original .50 fixed anchor')
        for j,name in enumerate(ARMS[1:],1):
            for threshold in (.50,.75,.90):
                if iou[0]>=threshold and iou[j]<threshold:
                    mutations[(name,f'{threshold:.2f}')]+=1
        values[g].append(iou.tolist())
    parent=np.asarray([np.mean(values[g],axis=0) for g in sorted(values)])
    pooled=np.asarray([row for g in values.values() for row in g])
    need(len(parent)==109 and len(pooled)==496 and len(matches)==453,'Exact original source denominator across arms')
    draws=np.random.Generator(np.random.PCG64(5217)).integers(len(parent),size=(4096,len(parent)))
    deltas={}
    for j,name in enumerate(ARMS[:-1]):
        diff=100*(parent[:,-1]-parent[:,j])
        ci=np.percentile(diff[draws].mean(axis=1),[2.5,97.5]).tolist()
        ap_effect=ap['weak_weighted_B']['AP']-ap[name]['AP']
        deltas[name]=dict(AP=ap_effect,IoU=float(diff.mean()),percentile95=ci,
                          improvement_gate=bool(ap_effect>=.30 and diff.mean()>=.20 and ci[0]>0))
    metrics={name:dict(prefix_equal_iou_points=float(100*parent[:,j].mean()),
                       reach={f'{t:.2f}':int((pooled[:,j]>=t).sum()) for t in (.50,.75,.90)},
                       original_match_losses={f'{t:.2f}':mutations[(name,f'{t:.2f}')] for t in (.50,.75,.90)})
             for j,name in enumerate(ARMS)}
    return dict(arms=metrics,B_minus=deltas,advances=all(row['improvement_gate'] for row in deltas.values()))


def prepare(args,corpus,captured):
    accepted=json.loads(pinned(args.corpus_run/'result/RESULT_SAFE.json',CORPUS_RESULT))
    terminal=json.loads(pinned(args.corpus_run/'observed_export/TERMINAL.json',CORPUS_TERMINAL))
    need(terminal['actual_exit']==0 and terminal['result_sha256']==CORPUS_RESULT and terminal['output_footer_verified']
         and terminal['worker_sha256']==accepted['worker_sha256']==CORPUS
         and captured['stock_predictions_sha256']==accepted['source_stock_full_roster_sha256']
         and captured['queries_sha256']==accepted['source_query_full_roster_sha256'] and accepted['source_full_candidates']==10488,
         'Only closed exact source corpus and full original roster')
    return roi_data(args,corpus,accepted)


def save_weights(output,states,corpus,training):
    path=output/'weights_private'
    path.mkdir()
    docs={}
    for (fold,arm),state in states.items():
        arrays={name:a.numpy() for name,a in state.items()}
        data=corpus.npz_bytes(**arrays)
        name=f'fold{fold}_{arm}.npz'
        artifact=corpus.write_artifact(path/name,data)
        artifact['tensor_sha256']=training[fold][arm]['final_tensor_sha256']
        docs[name]=artifact
    return docs


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    keys=('protocol','model-source','code','corpus-run','capture','acceptance','source-cohort','prior','box-run',
          'sibling-pilot','fit-gt','fit-images','output')
    for attr in keys:
        parser.add_argument('--'+attr,type=Path,required=True)
    args=parser.parse_args()
    start=perf_counter()
    need(not args.output.exists() and args.output.parent.is_dir(),'Fresh private result directory')
    pinned(args.protocol,PROTOCOL)
    device=configure()
    source=module(args.model_source,MODEL,'_user_clean_unet_plus_plus')
    corpus=module(args.corpus_run/'corpus.py',CORPUS,'_paired_frozen_source_corpus')
    box=module(args.box_run/'box_study.py',corpus.BOX_SOURCE,'_paired_original_box_loader')
    pilot=module(args.sibling_pilot/'pilot.py',SIBLING_SOURCE,'_paired_original_full_coco')
    reference=json.loads(pinned(args.sibling_pilot/'result/RESULT_SAFE.json',SIBLING_RESULT))
    stock,infos,groups,oracle,capture=box.inputs(args)
    schemas={v['image_id']:v for v in json.loads(pinned(args.source_cohort,corpus.SOURCE_COHORT))['images'] if v['core']}
    need(schemas.keys()==stock.keys() and len(schemas)==109,'Exact 109-image full-population input identity')
    x,y,weak,folds,prefix_fold,features,corpuslabels=prepare(args,corpus,capture)
    plan=module(args.code/oracle.FORMAL/'fit_plan.py',corpus.FOLD_PLAN,'_paired_original_prefix_plan')
    need(prefix_fold==plan.source_folds(list(groups.values())), 'Every selected and full inference image uses original prefix folds')
    t_admitted=perf_counter()
    print(json.dumps(dict(stage='closed_paired_corpus_and_complete_original_roster_admitted',device=str(device),
                          rows=len(x),full_candidates=sum(map(len,stock.values())))),flush=True)
    models,training,states=train_pair(source,device,x,y,weak,folds)
    t_fit=perf_counter()
    worlds,active,modified,census,changes,per_fold=infer_label_free(args,corpus,stock,infos,groups,schemas,prefix_fold,models,device)
    t_infer=perf_counter()
    print(json.dumps(dict(stage='native_full_population_labels_not_used_for_inference', **census)),flush=True)
    helper=module(args.code/'RFDETR_FIT109_QUOTA_COMPETITION_TARGETS_20261123/assemble_competition.py',
                  '3b0be851e38028248d5b646a108598585b6a261989496bf535d5ea8b54e6dd5b','_paired_full_original_author_truth')
    gt=helper.annotations(args,stock)
    coco,match=oracle.fixed_coco(gt,stock)
    need(len(match)==453 and len(gt['annotations'])==496,'Original full score and fixed anchors')
    ap={arm:pilot.full_metrics(gt,worlds[arm],stock) for arm in ARMS}
    need(ap['native']==reference['full_coco_points']['native'],'Exact original full-source native AP and recall parity')
    geo=main_geometry(gt,coco,match,worlds,groups,ap)
    need(abs(geo['arms']['native']['prefix_equal_iou_points']-reference['fixed_all_hull_geometry']['native']['parent_discrete_iou_points'])<1e-12,
         'Exact native matching and discrete geometry parity')
    secondary=weak_and_residual(gt,worlds,coco,match,stock,infos,features,corpuslabels,box,active,modified)
    t_scored=perf_counter()
    args.output.mkdir()
    weights=save_weights(args.output,states,corpus,training)
    peak=int(torch.cuda.max_memory_allocated(device)) if device.type=='cuda' else None
    stamp=dict(status='HRSID_FIT109_PAIRED_WEAK_ROI_CLEAN_UNETPP_COMPLETE_V1',phase='score',protocol_sha256=PROTOCOL,
               worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),model_source_sha256=MODEL,
               corpus_result_sha256=CORPUS_RESULT,source_original_rf_capture_sha256=box.CAPTURE,
               original_stock_sha256=capture['stock_predictions_sha256'],source_detector_fit_exposed=True,
               no_new_detector_forwards=0,external_data_opened=False,supervised_rows=len(x),model_parameters=PARAMETERS,
               device=str(device),peak_cuda_allocated_bytes=peak,training=training,total_optimizer_updates=sum(
                   e['updates'] for fold in training.values() for e in fold.values()),weights_closed_private=weights,
               source_images=len(stock),source_total_candidates=sum(map(len,stock.values())),source_hulls=len(gt['annotations']),
               native_fixed_matches=len(match),original_match_absent_hulls=len(gt['annotations'])-len(match),
               exact_archived_full_native_parity=True,equal_label_free_gate=census,actual_masks_changed=changes,
               common_gate_and_actual_changes_by_fold=per_fold,fold_geometry_proxy_groups=len(prefix_fold),
               official_coco_points=ap,fixed_source_discrete=geo,secondary_rescue=secondary,
               timing_seconds=dict(input_admission=t_admitted-start,paired_training=t_fit-t_admitted,
                   full_inference=t_infer-t_fit,official_score_and_geometry=t_scored-t_infer,total=perf_counter()-start))
    payload=corpus.encode(stamp)
    corpus.write_artifact(args.output/'RESULT_SAFE.json',payload)
    print(json.dumps(dict(status=stamp['status'],result_sha256=sha256(payload).hexdigest())),flush=True)


if __name__=='__main__':
    main()
