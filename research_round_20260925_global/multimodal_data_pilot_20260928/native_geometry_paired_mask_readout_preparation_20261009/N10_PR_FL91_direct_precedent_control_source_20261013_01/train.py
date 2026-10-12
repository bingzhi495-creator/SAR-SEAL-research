"""Unbound N10 SOURCE: paired C/PR fixed-eight-epoch cached-point training.

C uses the original N07 factory unchanged. PR uses the new F/L shared head.
Both arms consume exactly the same qualified frozen FIT cache, two private
seeds and instance/image orders. Different architectures do not have identical
weights. No native model is loaded, no DEV/TEST is read and no AP is computed.
Actual numerical qualification and owned execution are ROOT responsibilities.
"""
import argparse
import gc
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time
import types

SCHEMA = 'ROOT-N10-C-PR-FL91-point-training-spec-v1'
STATUS = 'N10_C_PR_FL91_POINT_TRAINING_COMPLETE_PENDING_ROOT_QUALIFICATION'
CP_SCHEMA = 'N10-C-PR-FL91-point-head-checkpoint-v1'
CACHE_Q_SCHEMA = 'ROOT-N07-frozen-E24-original-detail-point-cache-CPU-qualification-v1'
CACHE_Q_STATUS = 'ROOT_N07_FROZEN_E24_ORIGINAL_DETAIL_POINT_CACHE_CPU_QUALIFIED'
HEAD_CPU_SPEC_SCHEMA = 'ROOT-N10-C-PR-FL91-synthetic-head-CPU-spec-v1'
HEAD_CPU_Q_SCHEMA = 'ROOT-N10-C-PR-FL91-synthetic-head-CPU-qualification-v1'
HEAD_CPU_Q_STATUS = 'ROOT_N10_C_PR_FL91_SYNTHETIC_HEAD_CPU_QUALIFIED'
SEEDS = (202610120701, 202610120702)
ARMS = ('C', 'PR')
NAMES = {
    'C': ('hidden1.weight','hidden1.bias','hidden2.weight','hidden2.bias','final.weight','final.bias'),
    'PR': ('hidden1.weight','hidden1.bias','hidden2.weight','hidden2.bias',
           'hidden3.weight','hidden3.bias','final.weight','final.bias')}
SHAPES = {
    'C': ((64,562),(64,),(64,64),(64,),(1,64),(1,)),
    'PR': ((91,257),(91,),(91,92),(91,),(91,92),(91,),(1,92),(1,))}
PARAMETERS = {'C': 40257, 'PR': 40497}
FACTORY_SPEC_KEYS = {'C': 'point_heads_source', 'PR': 'PR_point_heads_source'}
RECIPE = dict(arms=list(ARMS), seeds=list(SEEDS), fixed_epochs=8,
              points_per_GT=1024, updates_per_epoch_per_model=1661, images_per_epoch=741,
              instance_mean_BCE=True, one_update_per_GT_instance=True, no_accumulation=True,
              AdamW_lr=3e-4, AdamW_weight_decay=1e-4, constant_lr=True, gradient_clip=None,
              fixed_cached_points_all_epochs=True, no_augmentation=True, band_abs_logit=2.,
              paired_seed=True, paired_initialization=False, paired_cold_native_identity=True,
              paired_image_order=True, raw_GT_order_within_image=True,
              endpoint_BCE='full1661 cached GT instance means after each epoch',
              extension_rule='fixed8; no continuation or best-epoch selection in this stage',
              best_epoch_selection=False, native610_updated=False,
              head_parameters=dict(PARAMETERS), head_parameter_tensors={'C':6,'PR':8})


def require(ok, message):
    if not ok: raise RuntimeError(message)


def read(path):
    def reject(value): raise ValueError('Nonfinite JSON constant: '+value)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'),parse_constant=reject)


def reference(path):
    path=Path(path).resolve(strict=True); digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''): digest.update(block)
    return dict(path=str(path),bytes=path.stat().st_size,sha256=digest.hexdigest())


def bound(ref):
    require(type(ref) is dict and set(ref)=={'path','bytes','sha256'} and Path(ref['path']).is_absolute() and
            type(ref['bytes']) is int and type(ref['sha256']) is str and len(ref['sha256'])==64 and
            set(ref['sha256']) <= set('0123456789abcdef') and reference(ref['path'])==ref,'Actual pinned input differs')
    return Path(ref['path'])


def terminal(ref, phase, code=0):
    path=Path(ref['path'])
    require(path.is_absolute() and path.name=='PARENT_RECEIPT.json' and type(phase) is str and phase and path.parent.name==phase,
            'Explicit original RAW locator required')
    raw=read(path); require(raw['status']=='ACTUAL_OWNED_TERMINAL' and type(raw['returncode']) is int and raw['returncode']==code and
            type(raw['original_identity']) is dict and raw['original_identity'] and raw['root_wait_reaped'] is True and
            raw['held_handle_exit']==dict(wait_state='WAIT_OBJECT_0',actual_exit_code=code) and
            type(raw['held_handle_exit']['actual_exit_code']) is int and raw['numeric_birth_absence']['original_birth_absent'] is True and
            raw.get('error') is None and raw.get('abort') is None and raw['wait_errors'] is False and raw['GPU_observation_errors']==[],
            'All owned terminal states required before any SHA/import')
    if 'phase' in raw: require(type(raw['phase']) is str and raw['phase']==phase,'RAW phase differs')
    peak,sha=raw['whole_GPU_observed_peak_MiB'],raw['admission_sha256']
    require(type(sha) is str and len(sha)==64 and set(sha)<=set('0123456789abcdef') and type(peak) in (int,float) and
            math.isfinite(peak) and 0<=peak<=8192 and type(raw['whole_GPU_used_limit_MiB']) is int and raw['whole_GPU_used_limit_MiB']==8192 and
            raw['whole_leaf_VRAM_certified'] is False and raw['GPU_scope']=='whole GPU observed, not per-process WDDM allocation','RAW owner/resource differs')
    return raw


def load(ref,name):
    path=bound(ref); source=path.read_bytes()
    require(len(source)==ref['bytes'] and hashlib.sha256(source).hexdigest()==ref['sha256'],'SOURCE bytes changed')
    module=types.ModuleType(name); module.__file__=str(path)
    exec(compile(source,str(path),'exec'),module.__dict__); return module


def source_gate(path,spec):
    require(spec['schema']==SCHEMA and type(spec['RAW_inputs']) is list and len(spec['RAW_inputs'])==29 and
            len({row['receipt']['path'] for row in spec['RAW_inputs']})==29,
            'Twenty-nine actual normal ancestral RAWs, including own synthetic CPU qualification, required')
    raws=[terminal(row['receipt'],row['phase']) for row in spec['RAW_inputs']]
    failed=terminal(spec['prior_failed_CPU_receipt'],Path(spec['prior_failed_CPU_receipt']['path']).parent.name,1)
    extra=spec['additional_failed_RAW_inputs']
    require(type(extra) is list and len(extra)==2 and
            all(type(row) is dict and set(row)=={'receipt','phase','returncode'} and
                type(row['returncode']) is int and row['returncode']==1 for row in extra),
            'Two exact three-key historical failed RAW rows must be preserved')
    failed_extra=[terminal(row['receipt'],row['phase'],1) for row in extra]
    normal_paths={row['receipt']['path'] for row in spec['RAW_inputs']}
    failed_paths=[spec['prior_failed_CPU_receipt']['path']]+[row['receipt']['path'] for row in extra]
    require(len(set(failed_paths))==3 and not normal_paths.intersection(failed_paths),
            'Historical failures must remain distinct from all normal training ancestors')
    # All normal29 and historical failed3 terminal states precede every hash/import.
    for ref in spec['sources']: bound(ref)
    closure={ref['path']:ref for ref in spec['sources']}
    require(len(closure)==len(spec['sources']),'Unique complete sources required')
    keys=('entry_source','cache_entry_source','point_heads_source','PR_point_heads_source','cache_CPU_reader_source',
          'cache_spec','cache_report','cache_GPU_receipt','cache_GPU_admission','cache_CPU_qualification',
          'cache_CPU_receipt','cache_CPU_admission','prior_failed_CPU_receipt','head_CPU_qualification',
          'head_CPU_receipt','head_CPU_admission','head_CPU_entry_source','head_CPU_spec')
    refs=[spec[k] for k in keys]
    refs += [row[k] for row in spec['RAW_inputs'] for k in ('receipt','admission','phase_binding','bootstrap')]
    refs += [row['receipt'] for row in extra]
    require(all(ref in closure.values() for ref in refs),'All training, cache and head CPU inputs must be sources-bound')
    require(spec['point_heads_source']['path']!=spec['PR_point_heads_source']['path'],
            'Original cache/C factory and new PR factory must remain distinct')
    cache=load(spec['cache_entry_source'],'_N10_qualified_original_cache_ABI')
    owners=[cache.owner_join(row,raw,closure) for row,raw in zip(spec['RAW_inputs'],raws)]
    source,spec_ref=reference(__file__),reference(path)
    require(source==spec['entry_source'],'Actual N10 training SOURCE differs')
    admission_path=Path(spec['GPU_admission_file'])
    require(admission_path.is_absolute(),'Absolute current admission required')
    admission_ref=reference(admission_path); admission=read(admission_path)
    require(admission['status']=='ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE' and admission['driver_sha256']==cache.OWNER_SHA and
            type(spec['phase']) is str and spec['phase'] and admission['phase']==spec['phase'] and admission['scientific_spec']==spec_ref and
            admission['entry']==source['path'] and admission['entry_args']==['--spec',str(path),'--owner-execute'] and
            admission['allowed_cpus']==[0,1] and admission['cuda_visible_devices']=='0' and admission['automatic_retry'] is False and
            admission['whole_GPU_used_limit_MiB']==8192 and type(spec['deadline_seconds']) is int and spec['deadline_seconds']>0 and
            admission['qualified_leaf_deadline_seconds']==spec['deadline_seconds'] and
            admission['bound_source_sha256'].get(str(path))==spec_ref['sha256'] and
            all(admission['bound_source_sha256'].get(ref['path'])==ref['sha256'] for ref in spec['sources']),
            'Own complete N10 training owner differs')
    cached=read(bound(spec['cache_spec'])); report=read(bound(spec['cache_report'])); q=read(bound(spec['cache_CPU_qualification']))
    require(cached['schema']==cache.SCHEMA and cached['entry_source']==spec['cache_entry_source'] and
            cached['point_heads_source']==spec['point_heads_source'] and cached['recipe']==cache.RECIPE and
            all(ref in closure.values() for ref in cached['sources']) and cached['RAW_inputs']==spec['RAW_inputs'][:17],
            'Exact original fixed cache source, factory, recipe and ancestral closure required')
    require(q['schema']==CACHE_Q_SCHEMA and q['status']==CACHE_Q_STATUS and q['source']==spec['cache_entry_source'] and
            q['reader_source']==spec['cache_CPU_reader_source'] and q['cache_spec']==spec['cache_spec'] and q['cache_report']==spec['cache_report'] and
            q['cache_GPU_admission']==spec['cache_GPU_admission'] and q['cache_GPU_RAW']==spec['cache_GPU_receipt'] and
            q['CPU_admission']==spec['cache_CPU_admission'] and q['point_heads_source']==spec['point_heads_source'] and
            q['phase']==spec['cache_CPU_phase'] and q['cache_phase']==spec['cache_GPU_phase'] and q['recipe']==cache.RECIPE and
            q['FIT']==cached['FIT'] and q['image_root']==cached['image_root'] and
            q['images']==741 and q['raw_GT']==1661 and q['point_count']==1661*1024 and
            q['hard_targets_CPU_checked'] is True and q['indices_JPEG_patches_CPU_checked'] is True and
            q['point_head_synthetic_CPU_qualified'] is True,'Actual original CPU cache qualification required')
    require(q['spec'] in closure.values(),'Actual original cache CPU SPEC absent')
    cpu=read(bound(q['spec']))
    require(cpu['RAW_inputs']==spec['RAW_inputs'][:18] and all(ref in closure.values() for ref in cpu['sources']),
            'Complete original CPU cache input closure required')
    for index,key,phase_key,admission_key,entry,args in (
        (17,'cache_GPU_receipt','cache_GPU_phase','cache_GPU_admission',spec['cache_entry_source']['path'],
         ['--spec',spec['cache_spec']['path'],'--owner-execute']),
        (18,'cache_CPU_receipt','cache_CPU_phase','cache_CPU_admission',spec['cache_CPU_reader_source']['path'],
         ['--spec',q['spec']['path'],'--owner-execute'])):
        row,owner=spec['RAW_inputs'][index],owners[index]
        require(row['receipt']==spec[key] and row['phase']==spec[phase_key] and row['admission']==spec[admission_key] and
                owner['entry']==entry and owner['entry_args']==args,'Original cache GPU/CPU provenance differs')
    require(report['schema']=='N07-frozen-E24-point-cache-report-v1' and report['status']==cache.STATUS and report['source']==spec['cache_entry_source'] and
            report['spec']==spec['cache_spec'] and report['GPU_admission']==spec['cache_GPU_admission'] and report['phase']==spec['cache_GPU_phase'] and
            report['packet_refs']==q['packet_refs'] and len(report['packet_refs'])==len(report['rows'])==741 and
            report['images']==741 and report['raw_GT']==1661 and report['point_count']==1661*1024 and
            report['native610_unchanged_SOURCE_observed'] is True and report['all_raw_GT_retained'] is True and
            report['optimizer_created'] is False and report['weights_updated'] is False and report['cleanup']['allocated_after']==0,
            'Complete original frozen cache report differs')
    require(all(ref in closure.values() for ref in report['packet_refs']) and report['start'] in closure.values() and
            report['trace'] in closure.values(),'Complete741 packet/START/TRACE closure required')
    context=read(bound(cached['context_GPU_spec']))
    require(all(ref in closure.values() for ref in context['sources']),'Original E24 closure absent')
    hq=read(bound(spec['head_CPU_qualification'])); hs=read(bound(spec['head_CPU_spec']))
    require(hq['schema']==HEAD_CPU_Q_SCHEMA and hq['status']==HEAD_CPU_Q_STATUS and
            hq['source']==spec['head_CPU_entry_source'] and hq['spec']==spec['head_CPU_spec'] and
            hq['CPU_admission']==spec['head_CPU_admission'] and
            hq['original_point_heads_source']==spec['point_heads_source'] and hq['PR_point_heads_source']==spec['PR_point_heads_source'] and
            hq['C_parameters']==40257 and hq['PR_parameters']==40497 and
            hq['synthetic_cold_identity_and_band_checked'] is True and hq['synthetic_backward_update_checked'] is True and
            hq['synthetic_CPU_only'] is True and hq['no_task_efficacy_claim'] is True and
            hq['CPU_affinity']==[0,1] and hq['threads']==2 and hq['CUDA_initialized'] is False and
            hq['RAW_inputs']==spec['RAW_inputs'][:28] and hq['prior_failed_CPU_receipt']==spec['prior_failed_CPU_receipt'] and
            hq['additional_failed_RAW_inputs']==extra,'Actual independent synthetic C/PR CPU qualification required')
    require(hs['schema']==HEAD_CPU_SPEC_SCHEMA and hs['entry_source']==spec['head_CPU_entry_source'] and
            hs['original_point_heads_source']==spec['point_heads_source'] and hs['PR_point_heads_source']==spec['PR_point_heads_source'] and
            hs['RAW_inputs']==spec['RAW_inputs'][:28] and hs['prior_failed_CPU_receipt']==spec['prior_failed_CPU_receipt'] and
            hs['additional_failed_RAW_inputs']==extra and all(ref in closure.values() for ref in hs['sources']),
            'Complete head CPU qualification input closure required')
    row,owner=spec['RAW_inputs'][28],owners[28]
    require(row['receipt']==spec['head_CPU_receipt'] and row['phase']==spec['head_CPU_phase'] and
            row['admission']==spec['head_CPU_admission'] and owner['cuda_visible_devices']=='' and
            owner['entry']==spec['head_CPU_entry_source']['path'] and
            owner['entry_args']==['--spec',spec['head_CPU_spec']['path'],'--owner-execute'] and
            hs['phase']==spec['head_CPU_phase'] and hq['phase']==spec['head_CPU_phase'],
            'Original held-terminal0 head CPU owner/provenance differs')
    require(spec['recipe']==RECIPE and spec['allocator_limit_MiB']==4096 and spec['whole_GPU_limit_MiB']==8192 and
            os.environ.get('CUDA_VISIBLE_DEVICES')=='0' and all(os.environ.get(k)=='2' for k in
            ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')),
            'Locked C/PR fixed8 training/resource recipe differs')
    return cache,cached,report,q,context,closure,source,spec_ref,admission_ref


def write(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,indent=2,allow_nan=False); stream.write('\n')


def digest(torch,tensor):
    array=tensor.detach().cpu().contiguous().numpy()
    return dict(shape=list(array.shape),dtype=str(array.dtype),sha256=hashlib.sha256(array.tobytes()).hexdigest())


def cpu_tree(torch,value):
    if isinstance(value,torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value,dict): return {k:cpu_tree(torch,v) for k,v in value.items()}
    if isinstance(value,list): return [cpu_tree(torch,v) for v in value]
    if isinstance(value,tuple): return tuple(cpu_tree(torch,v) for v in value)
    return value


def same_tree(torch,left,right):
    if isinstance(left,torch.Tensor): return isinstance(right,torch.Tensor) and left.dtype==right.dtype and left.shape==right.shape and torch.equal(left,right)
    if type(left) is not type(right): return False
    if isinstance(left,dict): return left.keys()==right.keys() and all(same_tree(torch,left[k],right[k]) for k in left)
    if isinstance(left,(tuple,list)): return len(left)==len(right) and all(same_tree(torch,a,b) for a,b in zip(left,right))
    return left==right


def check_head(torch,head,arm):
    require(arm in ARMS,'Known C/PR arm required')
    require(tuple(n for n,p in head.named_parameters())==NAMES[arm] and
            tuple(tuple(p.shape) for p in head.parameters())==SHAPES[arm] and
            sum(p.numel() for p in head.parameters())==PARAMETERS[arm] and not tuple(head.buffers()) and
            all(p.dtype==torch.float32 and bool(torch.isfinite(p).all()) for p in head.parameters()),
            'Exact finite per-arm head required')


def packet(torch,cache,report,index):
    ref=report['packet_refs'][index]; value=torch.load(str(bound(ref)),map_location='cpu'); row=report['rows'][index]
    require(value['schema']==cache.PACKET_SCHEMA and value['dataset_index']==row['dataset_index']==index and value['identity']==row['identity'] and
            value['source']==report['source'] and value['spec']==report['spec'] and value['phase']==report['phase'] and row['packet']==ref,
            'Exact frozen cache row identity differs')
    G=len(value['identity']['ann_ids']); require(1<=G<=100 and G==row['raw_GT'],'Complete positive raw-GT cache row required')
    for name,shape in (('F_samples',(G,1024,256)),('query_embeddings',(G,256)),('native_logits',(G,1024)),('patches',(G,1024,49)),('native_class_logits',(1,100,2))):
        t=value[name]; require(isinstance(t,torch.Tensor) and t.device.type=='cpu' and t.dtype==torch.float32 and tuple(t.shape)==shape and
                not t.requires_grad and bool(torch.isfinite(t).all()),'Finite detached FP32 cache operand differs: '+name)
    require(value['targets'].dtype==torch.bool and tuple(value['targets'].shape)==(G,1024) and value['point_indices'].dtype==torch.int64 and
            tuple(value['point_indices'].shape)==(G,1024) and value['query_ids'].dtype==torch.int64 and tuple(value['query_ids'].shape)==(G,) and
            len(set(value['query_ids'].tolist()))==G and bool(((value['query_ids']>=0)&(value['query_ids']<100)).all()),'Cached exact labels/query/point shapes differ')
    require(bool(((value['patches']>=0)&(value['patches']<=1)).all()) and
            all(digest(torch,value[name])==row['cache_digests'][name] for name in row['cache_digests']), 'Cache values/digests differ')
    return value


def instance_inputs(torch,value,g,device):
    return (value['F_samples'][g].to(device),value['query_embeddings'][g].to(device).expand(1024,256),
            value['native_logits'][g].to(device),value['patches'][g].to(device),value['targets'][g].to(device,dtype=torch.float32))


def endpoint(torch,cache,report,heads,spec,started):
    losses={arm:[] for arm in ARMS}; modes={arm:head.training for arm,head in heads.items()}
    try:
        for head in heads.values(): head.eval()
        with torch.no_grad():
            for index in range(741):
                require(time.monotonic()-started<spec['deadline_seconds'],'Prelocked training deadline exceeded')
                value=packet(torch,cache,report,index)
                for g in range(len(value['query_ids'])):
                    F,e,L,patch,y=instance_inputs(torch,value,g,'cuda:0')
                    for arm,head in heads.items():
                        loss=torch.nn.functional.binary_cross_entropy_with_logits(head(F,e,L,patch,arm=arm)['logits'],y)
                        require(bool(torch.isfinite(loss)),'Endpoint point-BCE nonfinite'); losses[arm].append(float(loss.cpu()))
                del value
    finally:
        for arm,head in heads.items(): head.train(modes[arm])
    require(all(len(v)==1661 for v in losses.values()),'Complete1661 endpoint losses required')
    return {arm:math.fsum(values)/1661 for arm,values in losses.items()}


def optimizer_proof(torch,optimizer,head,updates,arm):
    state=cpu_tree(torch,optimizer.state_dict()); names=list(NAMES[arm])
    require(len(state['param_groups'])==1,'Single fixed AdamW group required')
    group=state['param_groups'][0]; count=len(names)
    require(group['lr']==3e-4 and group['weight_decay']==1e-4 and group['betas']==(.9,.999) and group['eps']==1e-8 and
            group['amsgrad'] is False and len(group['params'])==len(state['state'])==count,
            'Full per-arm AdamW endpoint required')
    proof={}
    for name,param,identifier in zip(names,head.parameters(),group['params']):
        entry=state['state'][identifier]
        require(set(entry)=={'step','exp_avg','exp_avg_sq'} and float(entry['step'])==updates and
                all(tuple(entry[k].shape)==tuple(param.shape) and entry[k].dtype==torch.float32 and
                    bool(torch.isfinite(entry[k]).all()) for k in ('exp_avg','exp_avg_sq')),
                'Actual per-parameter AdamW moments/counts differ')
        proof[name]={key:digest(torch,value) for key,value in entry.items()}; proof[name]['updates']=updates
    return state,proof


def save_checkpoint(torch,head,optimizer,order_rng,report,source,spec_ref,admission_ref,spec,output,seed,arm,epoch,history,orders,activity,teacher):
    check_head(torch,head,arm); updates=1661*epoch; state=cpu_tree(torch,head.state_dict())
    opt,proof=optimizer_proof(torch,optimizer,head,updates,arm)
    meta=dict(schema=CP_SCHEMA,source=source,spec=spec_ref,GPU_admission=admission_ref,phase=spec['phase'],arm=arm,seed=seed,
              point_heads_source=spec['point_heads_source'],PR_point_heads_source=spec['PR_point_heads_source'],
              head_factory_source=spec[FACTORY_SPEC_KEYS[arm]],head_CPU_qualification=spec['head_CPU_qualification'],
              cache_spec=spec['cache_spec'],cache_report=spec['cache_report'],cache_CPU_qualification=spec['cache_CPU_qualification'],
              teacher_checkpoint=teacher,native_weights_epoch=24,native_weights_updated=False,
              completed_head_epochs=epoch,updates=updates,GT_visits=updates,point_visits=updates*1024,image_visits=epoch*741,
              points_fixed_across_epochs=True,recipe=RECIPE,parameter_tensors=len(NAMES[arm]),parameter_scalars=PARAMETERS[arm],
              no_pending_gradients=all(p.grad is None for p in head.parameters()),formal_native_initializer=False,
              training_resume_eligible=False,endpoint_BCE_history=history,epoch_orders=orders,gradient_activity=activity,
              paired_initialization=False,paired_seed=True,paired_cold_native_identity=True,paired_image_order=True)
    require(meta['no_pending_gradients'],'Clear head gradients before checkpoint')
    cp=dict(state_dict=state,optimizer_state_dict=opt,private_order_state=order_rng.get_state().clone(),meta=meta)
    target=output/('HEAD_%s_SEED%d_EPOCH%02d.pth'%(arm,seed,epoch))
    with target.open('xb') as stream: torch.save(cp,stream)
    readback=torch.load(str(target),map_location='cpu')
    require(same_tree(torch,cp,readback),'Complete own head checkpoint readback differs')
    proof_path=target.with_suffix('.PROOF.json')
    write(proof_path,dict(checkpoint=reference(target),state={n:digest(torch,v) for n,v in state.items()},optimizer=proof,updates=updates,
                          parameter_tensors=len(NAMES[arm]),parameter_scalars=PARAMETERS[arm],
                          private_order_state=digest(torch,cp['private_order_state']),full_readback_exact=True))
    return dict(checkpoint=reference(target),proof=reference(proof_path),completed_head_epochs=epoch,updates=updates,
                mean_endpoint_BCE=history[-1]['BCE'],head_state_finite=True,no_pending_gradients=True)


def train_epochs(torch,cache,report,pair,spec,started,first,last,trace):
    heads,opts,seed=pair['heads'],pair['optimizers'],pair['seed']
    for epoch in range(first,last+1):
        order=torch.randperm(741,generator=pair['order_rng']).tolist()
        order_sha=hashlib.sha256(json.dumps(order,separators=(',',':')).encode('ascii')).hexdigest()
        pair['orders'].append(dict(epoch=epoch,indices=order,sha256=order_sha))
        online={arm:[] for arm in ARMS}; visits=0
        for index in order:
            require(time.monotonic()-started<spec['deadline_seconds'],'Prelocked paired training deadline exceeded')
            value=packet(torch,cache,report,index)
            for g in range(len(value['query_ids'])):
                F,e,L,patch,y=instance_inputs(torch,value,g,'cuda:0')
                for arm in ARMS:
                    head,opt=heads[arm],opts[arm]; opt.zero_grad(set_to_none=True)
                    loss=torch.nn.functional.binary_cross_entropy_with_logits(head(F,e,L,patch,arm=arm)['logits'],y)
                    require(bool(torch.isfinite(loss)) and loss.requires_grad,'Finite connected instance BCE required')
                    loss.backward()
                    for name,param in head.named_parameters():
                        require(param.grad is not None and bool(torch.isfinite(param.grad).all()),'Finite connected per-arm head gradients required')
                        pair['activity'][arm][name]['present']+=1
                        pair['activity'][arm][name]['nonzero']+=int(bool((param.grad!=0).any()))
                    opt.step(); opt.zero_grad(set_to_none=True); check_head(torch,head,arm); online[arm].append(float(loss.detach().cpu()))
                visits+=1
            del value
        require(visits==1661 and all(len(values)==1661 for values in online.values()),'Paired complete FIT update population differs')
        measured=endpoint(torch,cache,report,heads,spec,started)
        for arm in ARMS: pair['history'][arm].append(dict(epoch=epoch,BCE=measured[arm]))
        row=dict(seed=seed,epoch=epoch,image_order_sha256=order_sha,images=741,GT_instances=1661,updates_per_model=1661,
                 point_visits_per_model=1661*1024,online_BCE={arm:math.fsum(values)/1661 for arm,values in online.items()},
                 full_FIT_endpoint_BCE=measured,paired_initialization=False,paired_seed=True,paired_cold_native_identity=True,paired_image_order=True,
                 native_forward_calls=0,native_weights_updated=False)
        trace.write(json.dumps(row,allow_nan=False)+'\n'); trace.flush()
        print(json.dumps(dict(seed=seed,epoch=epoch,endpoint_BCE=measured)),flush=True)


def run(spec_path):
    started=time.monotonic(); path=Path(spec_path)
    require(path.is_absolute(),'Absolute ROOT SPEC required'); path=path.resolve(strict=True)
    spec=read(path); cache,cached,report,q,context,closure,source,spec_ref,admission_ref=source_gate(path,spec)
    output=Path(spec['output_directory']); require(output.is_absolute() and not output.exists(),'Fresh absolute training output required')
    import psutil
    require(psutil.Process().cpu_affinity()==[0,1],'Actual CPU01 required before science imports')
    import torch
    require(torch.__version__=='2.0.1+cu118' and not torch.cuda.is_initialized(),'Original Torch/cold CUDA required')
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    require(reference(torch.__file__) in closure.values() and
            Path(inspect.getsourcefile(torch.optim.AdamW)).resolve()==bound(context['adamw_SOURCE']).resolve() and
            Path(inspect.getsourcefile(torch.nn.functional.binary_cross_entropy_with_logits)).resolve()==bound(context['interpolation_SOURCE']).resolve(),
            'Actual Torch/AdamW/BCE library SOURCE differs')
    factories={arm:load(spec[FACTORY_SPEC_KEYS[arm]],'_N10_'+arm+'_point_heads') for arm in ARMS}
    external_CPU=torch.random.get_rng_state().clone(); pairs=[]
    for seed in SEEDS:
        heads={arm:factories[arm].make_point_head(torch,seed) for arm in ARMS}
        for arm,head in heads.items():
            check_head(torch,head,arm)
            require(not bool((head.final.weight!=0).any()) and not bool((head.final.bias!=0).any()),
                    'Both cold output layers must be zero')
        order_rng=torch.Generator(device='cpu'); order_rng.manual_seed(seed)
        pairs.append(dict(seed=seed,heads=heads,order_rng=order_rng,optimizers={},history={arm:[] for arm in ARMS},orders=[],
                          initial_state_proof={arm:{name:digest(torch,value) for name,value in heads[arm].state_dict().items()} for arm in ARMS},
                          activity={arm:{name:dict(present=0,nonzero=0) for name in NAMES[arm]} for arm in ARMS}))
    require(torch.equal(external_CPU,torch.random.get_rng_state()) and not torch.cuda.is_initialized(),
            'CPU initialization changed outside RNG/CUDA')
    torch.cuda.init(); torch.cuda.set_per_process_memory_fraction(min(1.,4096*1024**2/torch.cuda.get_device_properties(0).total_memory),0)
    torch.cuda.reset_peak_memory_stats(0); outside=[torch.random.get_rng_state().clone(),torch.cuda.get_rng_state(0).clone()]
    fresh_counts={}
    for pair in pairs:
        for arm,head in pair['heads'].items():
            head.to('cuda:0'); head.train(); opt=torch.optim.AdamW(head.parameters(),lr=3e-4,weight_decay=1e-4)
            key='%s_seed%d'%(arm,pair['seed']); fresh_counts[key]=len(opt.state)
            require(fresh_counts[key]==0,'Actually fresh per-arm AdamW required'); pair['optimizers'][arm]=opt
    require('mmdet' not in sys.modules,'Cache training must not load a native model')
    output.mkdir(); teacher=context['checkpointmap']['native0']
    write(output/'START.json',dict(schema='N10-C-PR-FL91-point-training-start-v1',source=source,spec=spec_ref,
          GPU_admission=admission_ref,phase=spec['phase'],cache_CPU_qualification=spec['cache_CPU_qualification'],
          cache_spec=spec['cache_spec'],cache_report=spec['cache_report'],teacher_checkpoint=teacher,
          point_heads_source=spec['point_heads_source'],PR_point_heads_source=spec['PR_point_heads_source'],
          head_CPU_qualification=spec['head_CPU_qualification'],recipe=RECIPE,fresh_optimizer_state_entries_per_model=fresh_counts,
          initial_head_state_proofs={('%s_seed%d'%(arm,pair['seed'])):pair['initial_state_proof'][arm] for pair in pairs for arm in ARMS},
          native_model_loaded=False,own_stage_cpu_qualification=False,budget_is_unmeasured_prelock=True,
          paired_seed=True,paired_initialization=False,paired_cold_native_identity=True,paired_image_order=True,
          no_DEV=True,no_TEST=True,no_AP=True))
    epochs=RECIPE['fixed_epochs']; final={}
    with (output/'TRACE.jsonl').open('x',encoding='utf-8',newline='\n') as trace:
        for pair in pairs:
            train_epochs(torch,cache,report,pair,spec,started,1,epochs,trace)
            for arm in ARMS:
                key='%s_seed%d'%(arm,pair['seed'])
                final[key]=save_checkpoint(torch,pair['heads'][arm],pair['optimizers'][arm],pair['order_rng'],report,source,spec_ref,
                    admission_ref,spec,output,pair['seed'],arm,epochs,pair['history'][arm],pair['orders'],pair['activity'][arm],teacher)
    decision_path=output/'FIXED8_DECISION.json'
    write(decision_path,dict(rule=RECIPE['extension_rule'],fixed_final_epoch=epochs,continue_all4=False,
          DEV_accessed=False,TEST_accessed=False,best_epoch_selection=False,automatic_extension=False))
    histories={('%s_seed%d'%(arm,pair['seed'])):pair['history'][arm] for pair in pairs for arm in ARMS}
    require(len(final)==4 and all(len(values)==8 for values in histories.values()) and
            all(row['updates']==13288 for row in final.values()),'All4 fixed-eight-epoch endpoints required')
    require(torch.equal(outside[0],torch.random.get_rng_state()) and torch.equal(outside[1],torch.cuda.get_rng_state(0)),
            'Outside RNG changed')
    peak=torch.cuda.max_memory_allocated(0)
    for pair in pairs:
        for head in pair['heads'].values(): head.cpu()
        pair['optimizers'].clear(); pair['heads'].clear()
    del head,opt,heads,pairs,pair; gc.collect(); allocated_before=torch.cuda.memory_allocated(0)
    torch.cuda.synchronize(); torch._C._cuda_clearCublasWorkspaces(); torch.cuda.empty_cache()
    require(torch.cuda.memory_allocated(0)==0,'Live GPU tensor/optimizer alias survived cleanup')
    require(psutil.Process().cpu_affinity()==[0,1] and torch.get_num_threads()==torch.get_num_interop_threads()==2,
            'Final resources differ')
    for ref in spec['sources']: bound(ref)
    require(reference(path)==spec_ref and reference(__file__)==source and reference(spec['GPU_admission_file'])==admission_ref,
            'Final provenance changed')
    result=dict(schema='N10-C-PR-FL91-point-training-report-v1',status=STATUS,source=source,spec=spec_ref,
                GPU_admission=admission_ref,phase=spec['phase'],start=reference(output/'START.json'),
                trace=reference(output/'TRACE.jsonl'),decision=reference(decision_path),
                cache_spec=spec['cache_spec'],cache_report=spec['cache_report'],cache_CPU_qualification=spec['cache_CPU_qualification'],
                teacher_checkpoint=teacher,point_heads_source=spec['point_heads_source'],PR_point_heads_source=spec['PR_point_heads_source'],
                head_CPU_qualification=spec['head_CPU_qualification'],recipe=RECIPE,continued_all4_to24=False,completed_head_epochs=epochs,
                checkpoints=final,epoch8_checkpoints=final,endpoint_BCE_histories=histories,all4_fixed_same_epochs=True,
                paired_order_and_initialization=False,paired_order=True,paired_seed=True,paired_cold_native_identity=True,
                native_model_loaded=False,native_forward_calls=0,native_weights_updated=False,
                independent_CPU_training_trajectory_replayed=False,own_stage_cpu_qualification=False,
                AP_computed=False,no_DEV=True,no_TEST=True,fresh_optimizer_state_entries_per_model=fresh_counts,
                budget_is_unmeasured_prelock=True,elapsed_seconds=time.monotonic()-started,torch_peak_allocated_bytes=peak,
                cleanup=dict(allocated_before=allocated_before,allocated_after=torch.cuda.memory_allocated(0)),
                evidence_limits=[
                'PR is a PointRend-style frozen-carrier structural adaptation, not the complete author pipeline.',
                'C and PR differ in information, width, depth and condition paths; approximate parameter/MAC budgets do not prove equal function capacity.',
                'The original C factory is used unchanged; C/PR share seeds and image orders, not identical weights.',
                'All head supervision uses the same1024 cached points per GT across all epochs; endpoint BCE is not AP or unseen-data convergence.',
                'All4 end at eight epochs; no FIT/DEV-conditioned continuation or best checkpoint selection is performed.',
                'Synthetic CPU qualification does not establish task efficacy or independently replay this training trajectory.',
                'Native weights are referenced but never loaded or updated; teacher cache compute is separate.',
                'SOURCE and checkpoint readback cannot qualify their own actual owned terminal.'])
    write(output/'REPORT.json',result)
    print(json.dumps(dict(status=STATUS,report=reference(output/'REPORT.json'))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--spec',required=True); parser.add_argument('--owner-execute',action='store_true')
    args=parser.parse_args(); require(args.owner_execute,'Explicit single ROOT-owned N10 training execution required'); run(args.spec)
