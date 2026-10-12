"""Future ROOT-owned N07 paired training endpoint CPU qualification.

All20 successful ancestors plus the preserved failed1 are observed before hashes
or imports. Recompute the four final FIT cached-point endpoints independently;
check saved epoch8/final states, moments, counters and private image orders.
Intermediate epoch losses and gradient activity are bound records, not replayed
trajectories. No native teacher, image/GT decoder, AP or optimizer step is run.
ROOT closes this CPU producer's actual terminal separately after tool completion.
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

SCHEMA='ROOT-N07-original-detail-paired-point-training-CPU-qualification-spec-v1'
Q_SCHEMA='ROOT-N07-original-detail-paired-point-training-CPU-qualification-v1'
STATUS='ROOT_N07_ORIGINAL_DETAIL_PAIRED_POINT_TRAINING_CPU_QUALIFIED'
SEEDS=(202610120701,202610120702)
ARMS=('C','R')
KEYS=tuple('%s_seed%d'%(arm,seed) for seed in SEEDS for arm in ARMS)
NAMES=('hidden1.weight','hidden1.bias','hidden2.weight','hidden2.bias','final.weight','final.bias')
SHAPES=((64,562),(64,),(64,64),(64,),(1,64),(1,))
NUMERIC=dict(endpoint_BCE_rtol=1e-4,endpoint_BCE_atol=1e-6,
             instance_BCE_arithmetic_rtol=1e-4,instance_BCE_arithmetic_atol=1e-6)


def require(ok,message):
    if not ok: raise RuntimeError(message)


def read(path):
    def reject(value): raise ValueError('Nonfinite JSON constant: '+value)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'),parse_constant=reject)


def reference(path):
    path=Path(path).resolve(strict=True); value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''): value.update(block)
    return dict(path=str(path),bytes=path.stat().st_size,sha256=value.hexdigest())


def bound(ref):
    require(type(ref) is dict and set(ref)=={'path','bytes','sha256'} and type(ref['path']) is str and
            Path(ref['path']).is_absolute() and type(ref['bytes']) is int and ref['bytes']>=0 and
            type(ref['sha256']) is str and len(ref['sha256'])==64 and set(ref['sha256'])<=set('0123456789abcdef') and
            reference(ref['path'])==ref,'Actual bound input differs')
    return Path(ref['path'])


def terminal(ref,phase,code=0):
    path=Path(ref['path'])
    require(path.is_absolute() and path.name=='PARENT_RECEIPT.json' and type(phase) is str and phase and path.parent.name==phase,
            'Original absolute RAW/phase locator required')
    raw=read(path)
    require(raw['status']=='ACTUAL_OWNED_TERMINAL' and type(raw['returncode']) is int and raw['returncode']==code and
            type(raw['original_identity']) is dict and raw['original_identity'] and raw['root_wait_reaped'] is True and
            raw['held_handle_exit']==dict(wait_state='WAIT_OBJECT_0',actual_exit_code=code) and
            type(raw['held_handle_exit']['actual_exit_code']) is int and raw['numeric_birth_absence']['original_birth_absent'] is True and
            raw.get('error') is None and raw.get('abort') is None and raw['wait_errors'] is False and raw['GPU_observation_errors']==[],
            'All20 normal plus preserved failed1 terminals must precede SHA/import')
    if 'phase' in raw: require(type(raw['phase']) is str and raw['phase']==phase,'Original RAW phase differs')
    sha,peak=raw['admission_sha256'],raw['whole_GPU_observed_peak_MiB']
    require(type(sha) is str and len(sha)==64 and set(sha)<=set('0123456789abcdef') and type(peak) in (int,float) and
            math.isfinite(peak) and 0<=peak<=8192 and type(raw['whole_GPU_used_limit_MiB']) is int and raw['whole_GPU_used_limit_MiB']==8192 and
            raw['whole_leaf_VRAM_certified'] is False and raw['GPU_scope']=='whole GPU observed, not per-process WDDM allocation',
            'Original owner/resource observation differs')
    return raw


def load(ref,name):
    path=bound(ref); source=path.read_bytes()
    require(len(source)==ref['bytes'] and hashlib.sha256(source).hexdigest()==ref['sha256'],'SOURCE bytes changed')
    module=types.ModuleType(name); module.__file__=str(path)
    exec(compile(source,str(path),'exec'),module.__dict__); return module


def cache_lineage(train,training,closure):
    """CPU metadata counterpart; never call the GPU-only train.source_gate."""
    refs=[training[k] for k in ('cache_entry_source','cache_spec','cache_report','cache_CPU_qualification','cache_CPU_reader_source',
          'cache_GPU_receipt','cache_GPU_admission','cache_CPU_receipt','cache_CPU_admission','point_heads_source')]
    require(all(ref in closure.values() for ref in refs),'Actual cache primary lineage absent')
    cache=load(training['cache_entry_source'],'_N07_CPU_cache_provenance')
    cached=read(bound(training['cache_spec'])); report=read(bound(training['cache_report'])); q=read(bound(training['cache_CPU_qualification']))
    require(cached['schema']==cache.SCHEMA and cached['entry_source']==training['cache_entry_source'] and
            cached['point_heads_source']==training['point_heads_source'] and cached['recipe']==cache.RECIPE and
            cached['RAW_inputs']==training['RAW_inputs'][:17] and all(ref in closure.values() for ref in cached['sources']),
            'Exact actual qualified cache SOURCE/recipe/closure differs')
    require(q['schema']==train.CACHE_Q_SCHEMA and q['status']==train.CACHE_Q_STATUS and q['source']==training['cache_entry_source'] and
            q['reader_source']==training['cache_CPU_reader_source'] and q['cache_spec']==training['cache_spec'] and q['cache_report']==training['cache_report'] and
            q['cache_GPU_admission']==training['cache_GPU_admission'] and q['cache_GPU_RAW']==training['cache_GPU_receipt'] and
            q['CPU_admission']==training['cache_CPU_admission'] and q['point_heads_source']==training['point_heads_source'] and
            q['phase']==training['cache_CPU_phase'] and q['cache_phase']==training['cache_GPU_phase'] and q['recipe']==cache.RECIPE and
            q['FIT']==cached['FIT'] and q['image_root']==cached['image_root'] and q['images']==741 and q['raw_GT']==1661 and q['point_count']==1700864 and
            q['hard_targets_CPU_checked'] is True and q['indices_JPEG_patches_CPU_checked'] is True and q['point_head_synthetic_CPU_qualified'] is True,
            'Actual own independent CPU cache/head qualification differs')
    require(q['spec'] in closure.values(),'Actual cache CPU SPEC absent')
    cpu=read(bound(q['spec'])); require(cpu['RAW_inputs']==training['RAW_inputs'][:18] and all(ref in closure.values() for ref in cpu['sources']),
            'Complete actual qualified cache CPU closure differs')
    require(report['schema']=='N07-frozen-E24-point-cache-report-v1' and report['status']==cache.STATUS and report['source']==training['cache_entry_source'] and
            report['spec']==training['cache_spec'] and report['GPU_admission']==training['cache_GPU_admission'] and report['phase']==training['cache_GPU_phase'] and
            report['packet_refs']==q['packet_refs'] and len(report['packet_refs'])==len(report['rows'])==741 and report['images']==741 and report['raw_GT']==1661 and
            report['point_count']==1700864 and report['native610_unchanged_SOURCE_observed'] is True and report['all_raw_GT_retained'] is True and
            report['optimizer_created'] is False and report['weights_updated'] is False and report['cleanup']['allocated_after']==0,
            'Actual immutable frozen cache report differs')
    require(all(ref in closure.values() for ref in report['packet_refs']) and report['start'] in closure.values() and report['trace'] in closure.values() and
            cached['context_GPU_spec'] in closure.values(),'All actual741 packet/START/TRACE/context inputs absent')
    context=read(bound(cached['context_GPU_spec'])); require(all(ref in closure.values() for ref in context['sources']),'Original E24 closure absent')
    return cache,cached,report,q,context


def source_gate(path,spec):
    require(spec['schema']==SCHEMA and type(spec['RAW_inputs']) is list and len(spec['RAW_inputs'])==20 and
            len({row['receipt']['path'] for row in spec['RAW_inputs']})==20,'Exactly20 distinct successful ancestors required')
    raws=[terminal(row['receipt'],row['phase']) for row in spec['RAW_inputs']]
    terminal(spec['prior_failed_CPU_receipt'],Path(spec['prior_failed_CPU_receipt']['path']).parent.name,1)
    require(spec['prior_failed_CPU_receipt']['path'] not in {r['receipt']['path'] for r in spec['RAW_inputs']},'Failed attempt cannot qualify an ancestor')
    # No first input SHA, SOURCE import, checkpoint or numerical input read above.
    for ref in spec['sources']: bound(ref)
    closure={ref['path']:ref for ref in spec['sources']}; require(len(closure)==len(spec['sources']),'Unique complete CPU closure required')
    roles=('entry_source','training_entry_source','training_spec','training_report','training_GPU_RAW','training_GPU_admission',
           'point_heads_source','prior_failed_CPU_receipt')
    required=[spec[k] for k in roles]+[row[k] for row in spec['RAW_inputs'] for k in ('receipt','admission','phase_binding','bootstrap')]
    require(all(ref in closure.values() for ref in required),'All actual CPU primary inputs must be bound')
    train=load(spec['training_entry_source'],'_N07_train_endpoint_provenance'); training=read(bound(spec['training_spec']))
    train_source,train_spec,train_admission=spec['training_entry_source'],spec['training_spec'],spec['training_GPU_admission']
    require(training['schema']==train.SCHEMA and training['entry_source']==train_source and training['recipe']==train.RECIPE and
            training['RAW_inputs']==spec['RAW_inputs'][:19] and training['prior_failed_CPU_receipt']==spec['prior_failed_CPU_receipt'] and
            all(ref in closure.values() for ref in training['sources']) and
            training['point_heads_source']==spec['point_heads_source'],'Complete actual own training/cache closure differs')
    cache,cached,cache_report,cache_q,context=cache_lineage(train,training,closure)
    owners=[cache.owner_join(row,raw,closure) for row,raw in zip(spec['RAW_inputs'],raws)]
    for index,key,phase_key,admission_key,entry,args in (
        (17,'cache_GPU_receipt','cache_GPU_phase','cache_GPU_admission',training['cache_entry_source']['path'],['--spec',training['cache_spec']['path'],'--owner-execute']),
        (18,'cache_CPU_receipt','cache_CPU_phase','cache_CPU_admission',training['cache_CPU_reader_source']['path'],['--spec',cache_q['spec']['path'],'--owner-execute'])):
        row,owner=spec['RAW_inputs'][index],owners[index]
        require(row['receipt']==training[key] and row['phase']==training[phase_key] and row['admission']==training[admission_key] and
                owner['entry']==entry and owner['entry_args']==args,'Own cache actualGPU/CPU lineage differs')
    row,owner=spec['RAW_inputs'][-1],owners[-1]
    require(row['receipt']==spec['training_GPU_RAW'] and row['phase']==training['phase'] and
            row['admission']==spec['training_GPU_admission']==train_admission and owner['entry']==train_source['path'] and
            owner['entry_args']==['--spec',train_spec['path'],'--owner-execute'],'Own training actualGPU0 lineage differs')
    require(owner['scientific_spec']==train_spec and owner['cuda_visible_devices']=='0' and
            owner['qualified_leaf_deadline_seconds']==training['deadline_seconds'] and
            all(owner['bound_source_sha256'].get(ref['path'])==ref['sha256'] for ref in training['sources']) and
            training['allocator_limit_MiB']==4096 and training['whole_GPU_limit_MiB']==8192,'Complete original training owner/recipe differs')
    source,spec_ref=reference(__file__),reference(path); require(source==spec['entry_source'],'Actual CPU SOURCE differs')
    admission_path=Path(spec['CPU_admission_file']); require(admission_path.is_absolute(),'Absolute current CPU admission required')
    admission_ref=reference(admission_path); admission=read(admission_path)
    require(admission['status']=='ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE' and admission['driver_sha256']==cache.OWNER_SHA and
            type(spec['phase']) is str and spec['phase'] and admission['phase']==spec['phase'] and admission['scientific_spec']==spec_ref and
            admission['entry']==source['path'] and admission['entry_args']==['--spec',str(path),'--owner-execute'] and
            admission['allowed_cpus']==[0,1] and admission['cuda_visible_devices']=='' and admission['automatic_retry'] is False and
            admission['whole_GPU_used_limit_MiB']==8192 and type(spec['deadline_seconds']) is int and spec['deadline_seconds']>0 and
            admission['qualified_leaf_deadline_seconds']==spec['deadline_seconds'] and admission['bound_source_sha256'].get(str(path))==spec_ref['sha256'] and
            all(admission['bound_source_sha256'].get(ref['path'])==ref['sha256'] for ref in spec['sources']),'Actual CPU owner/complete closure differs')
    report=read(bound(spec['training_report']))
    require(report['schema']=='N07-paired-point-training-report-v1' and report['status']==train.STATUS and report['source']==train_source and
            report['spec']==train_spec and report['GPU_admission']==train_admission and report['phase']==training['phase'] and report['recipe']==train.RECIPE and
            report['cache_spec']==training['cache_spec'] and report['cache_report']==training['cache_report'] and
            report['cache_CPU_qualification']==training['cache_CPU_qualification'] and report['teacher_checkpoint']==context['checkpointmap']['native0'] and
            report['point_heads_source']==spec['point_heads_source'] and report['native_model_loaded'] is False and report['native_forward_calls']==0 and
            report['native_weights_updated'] is False and report['all4_fixed_same_epochs'] is True and report['paired_order_and_initialization'] is True and
            report['no_DEV'] is True and report['no_TEST'] is True and report['AP_computed'] is False and report['cleanup']['allocated_after']==0,
            'Complete actual own paired training report differs')
    epochs=report['completed_head_epochs']; require(type(epochs) is int and epochs in (8,24) and report['continued_all4_to24'] is (epochs==24) and
            set(report['checkpoints'])==set(report['epoch8_checkpoints'])==set(report['endpoint_BCE_histories'])==set(KEYS),'Common final endpoint differs')
    require(set(report['fresh_optimizer_state_entries_per_model'])==set(KEYS) and
            all(type(v) is int and v==0 for v in report['fresh_optimizer_state_entries_per_model'].values()),'Actual fresh moments differ')
    outputs=[report[k] for k in ('start','trace','decision')]
    outputs += [row[k] for group in ('checkpoints','epoch8_checkpoints') for row in report[group].values() for k in ('checkpoint','proof')]
    require(all(ref in closure.values() for ref in outputs),'All numerical checkpoints/proofs/START/TRACE/decision must be CPU inputs')
    require(spec['recipe']==train.RECIPE and spec['numeric_protocol']==NUMERIC and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and
            all(os.environ.get(k)=='2' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')),
            'Prelocked CPU math/resources differ')
    target=Path(spec['Q_path']); require(target.is_absolute() and not target.exists() and not target.with_suffix(target.suffix+'.tmp').exists(),'Fresh absolute Q path required')
    return train,cache,cached,cache_report,cache_q,context,training,report,closure,source,spec_ref,admission_ref,admission


def digest(tensor):
    array=tensor.detach().cpu().contiguous().numpy()
    return dict(shape=list(array.shape),dtype=str(array.dtype),sha256=hashlib.sha256(array.tobytes()).hexdigest())


def imported_files(torch,np,context,closure):
    paths=dict(torch=Path(torch.__file__),torch_binary=Path(torch._C.__file__),numpy=Path(np.__file__),
               numpy_binary=Path(np.core._multiarray_umath.__file__),AdamW=Path(inspect.getsourcefile(torch.optim.AdamW)),
               point_math=Path(inspect.getsourcefile(torch.nn.functional.binary_cross_entropy_with_logits)))
    result={}
    for role,path in paths.items():
        ref=reference(path); require(ref in closure.values(),'Actual CPU library file absent: '+role); result[role]=ref
    require(result['AdamW']==context['adamw_SOURCE'] and result['point_math']==context['interpolation_SOURCE'],'Original math SOURCE differs')
    return result


def finite_state(torch,state):
    require(type(state) is dict and tuple(state)==NAMES and all(isinstance(state[n],torch.Tensor) and state[n].device.type=='cpu' and
            state[n].dtype==torch.float32 and tuple(state[n].shape)==shape and not state[n].requires_grad and bool(torch.isfinite(state[n]).all())
            for n,shape in zip(NAMES,SHAPES)) and sum(state[n].numel() for n in NAMES)==40257,'Exact six-tensor CPU state differs')


def checkpoint_check(torch,ref,proof_ref,key,epoch,training,report,spec,start,trace,factory):
    arm,seed_text=key.split('_seed'); seed=int(seed_text); cp=torch.load(str(bound(ref)),map_location='cpu'); proof=read(bound(proof_ref))
    require(set(cp)=={'state_dict','optimizer_state_dict','private_order_state','meta'},'Own complete checkpoint fields differ')
    state,meta=cp['state_dict'],cp['meta']; finite_state(torch,state); updates=epoch*1661
    require(meta['schema']=='N07-original-detail-paired-point-head-checkpoint-v1' and meta['source']==spec['training_entry_source'] and
            meta['spec']==spec['training_spec'] and meta['GPU_admission']==spec['training_GPU_admission'] and meta['phase']==training['phase'] and
            meta['arm']==arm and meta['seed']==seed and meta['point_heads_source']==spec['point_heads_source'] and
            meta['cache_spec']==report['cache_spec'] and meta['cache_report']==report['cache_report'] and meta['cache_CPU_qualification']==report['cache_CPU_qualification'] and
            meta['teacher_checkpoint']==report['teacher_checkpoint'] and meta['native_weights_epoch']==24 and meta['native_weights_updated'] is False and
            meta['completed_head_epochs']==epoch and meta['updates']==meta['GT_visits']==updates and meta['point_visits']==updates*1024 and
            meta['image_visits']==epoch*741 and meta['points_fixed_across_epochs'] is True and meta['recipe']==training['recipe'] and
            meta['parameter_tensors']==6 and meta['parameter_scalars']==40257 and meta['no_pending_gradients'] is True and
            meta['formal_native_initializer'] is False and meta['training_resume_eligible'] is False,'Own head metadata differs')
    require(proof['checkpoint']==ref and proof['updates']==updates and proof['parameter_tensors']==6 and proof['parameter_scalars']==40257 and
            proof['full_readback_exact'] is True and proof['state']=={n:digest(v) for n,v in state.items()},'Own checkpoint state proof differs')
    opt=cp['optimizer_state_dict']; require(len(opt['param_groups'])==1 and len(opt['state'])==6,'Six own AdamW moments required')
    group=opt['param_groups'][0]; identifiers=group['params']
    require(group['lr']==3e-4 and group['weight_decay']==1e-4 and group['betas']==(.9,.999) and group['eps']==1e-8 and group['amsgrad'] is False and
            len(identifiers)==len(set(identifiers))==6 and set(identifiers)==set(opt['state']),'Own constant AdamW group differs')
    for name,shape,identifier in zip(NAMES,SHAPES,identifiers):
        entry=opt['state'][identifier]; step=entry['step']
        require(set(entry)=={'step','exp_avg','exp_avg_sq'} and isinstance(step,torch.Tensor) and step.device.type=='cpu' and step.numel()==1 and
                step.dtype==torch.float32 and bool(torch.isfinite(step).all()) and float(step)==updates and
                all(isinstance(entry[k],torch.Tensor) and entry[k].device.type=='cpu' and entry[k].dtype==torch.float32 and tuple(entry[k].shape)==shape and
                    bool(torch.isfinite(entry[k]).all()) for k in ('exp_avg','exp_avg_sq')) and
                proof['optimizer'][name]==dict({k:digest(v) for k,v in entry.items()},updates=updates),'Own moment values/counts/proof differ')
    history=meta['endpoint_BCE_history']
    require(history==report['endpoint_BCE_histories'][key][:epoch] and len(history)==epoch and [row['epoch'] for row in history]==list(range(1,epoch+1)) and
            all(type(row['BCE']) in (int,float) and math.isfinite(row['BCE']) and row['BCE']>=0 for row in history),'Own endpoint history differs')
    generator=torch.Generator(device='cpu'); generator.manual_seed(seed); orders=meta['epoch_orders']
    require(len(orders)==epoch,'Complete private image order required')
    for number,row in enumerate(orders,1):
        order=torch.randperm(741,generator=generator).tolist(); sha=hashlib.sha256(json.dumps(order,separators=(',',':')).encode('ascii')).hexdigest()
        require(row==dict(epoch=number,indices=order,sha256=sha) and trace[(seed,number)]['image_order_sha256']==sha and
                trace[(seed,number)]['full_FIT_endpoint_BCE'][arm]==history[number-1]['BCE'],'Own private order/trace/loss history differs')
    private=cp['private_order_state']; require(isinstance(private,torch.Tensor) and private.device.type=='cpu' and private.dtype==torch.uint8 and
            torch.equal(generator.get_state(),private) and proof['private_order_state']==digest(private),'Own private RNG endpoint differs')
    activity=meta['gradient_activity']; require(set(activity)==set(NAMES) and all(set(v)=={'present','nonzero'} and type(v['present']) is int and
            v['present']==updates and type(v['nonzero']) is int and 0<=v['nonzero']<=updates for v in activity.values()),'Bound gradient activity counts differ')
    cold=factory.make_point_head(torch,seed); cold_state=cold.state_dict()
    require(tuple(cold_state)==NAMES and {n:digest(v) for n,v in cold_state.items()}==start['initial_head_state_proofs'][str(seed)] and
            not bool((cold.final.weight!=0).any()) and not bool((cold.final.bias!=0).any()),'Paired cold state proof differs')
    cold.load_state_dict(state,strict=True); require(not tuple(cold.buffers()),'Unexpected head buffers')
    del cold,cp,opt
    return state,dict(checkpoint=ref,proof=proof_ref,epoch=epoch,updates=updates,parameter_tensors=6,parameter_scalars=40257,
                      strict_CPU_loaded=True,AdamW_endpoint_step=updates,private_order_reconstructed=True,gradient_activity_record=activity)


def records_check(report,training,start,decision,trace_rows):
    epochs=report['completed_head_epochs']; fresh=report['fresh_optimizer_state_entries_per_model']
    require(start['schema']=='N07-paired-point-training-start-v1' and start['source']==report['source'] and start['spec']==report['spec'] and
            start['GPU_admission']==report['GPU_admission'] and start['phase']==training['phase'] and start['recipe']==training['recipe'] and
            start['cache_CPU_qualification']==report['cache_CPU_qualification'] and start['cache_spec']==report['cache_spec'] and
            start['cache_report']==report['cache_report'] and start['teacher_checkpoint']==report['teacher_checkpoint'] and
            start['point_heads_source']==report['point_heads_source'] and start['fresh_optimizer_state_entries_per_model']==fresh and
            start['native_model_loaded'] is False and start['no_DEV'] is True and start['no_TEST'] is True and start['no_AP'] is True,
            'Actual paired cold/fresh START differs')
    require(len(trace_rows)==2*epochs and {(r['seed'],r['epoch']) for r in trace_rows}=={(s,e) for s in SEEDS for e in range(1,epochs+1)},'Complete unique epoch traces required')
    trace={(r['seed'],r['epoch']):r for r in trace_rows}
    for row in trace_rows:
        require(row['images']==741 and row['GT_instances']==1661 and row['updates_per_model']==1661 and row['point_visits_per_model']==1700864 and
                row['paired_initialization'] is True and row['paired_image_order'] is True and row['native_forward_calls']==0 and row['native_weights_updated'] is False and
                set(row['online_BCE'])==set(row['full_FIT_endpoint_BCE'])==set(ARMS) and
                all(type(v) in (int,float) and math.isfinite(v) and v>=0 for field in ('online_BCE','full_FIT_endpoint_BCE') for v in row[field].values()),
                'Actual1661-instance epoch counters/finite losses differ')
    computed=[]
    for key in KEYS:
        arm,seed_text=key.split('_seed'); seed=int(seed_text); history=report['endpoint_BCE_histories'][key]
        require(len(history)==epochs and [r['epoch'] for r in history]==list(range(1,epochs+1)) and
                all(type(r['BCE']) in (int,float) and math.isfinite(r['BCE']) and r['BCE']>=0 for r in history),'Finite full history required')
        values={r['epoch']:r['BCE'] for r in history}; changes=[(values[a]-values[b])/values[a] if values[a]>0 else 0. for a,b in ((6,7),(7,8))]
        computed.append(dict(seed=seed,arm=arm,relative_decrease_6_to_7=changes[0],relative_decrease_7_to_8=changes[1],trigger=all(v>=.01 for v in changes)))
    extend=any(r['trigger'] for r in computed)
    require(decision['rule']==training['recipe']['extension_rule'] and decision['per_model']==computed and decision['continue_all4'] is extend and
            decision['fixed_final_epoch']==epochs==(24 if extend else 8) and decision['DEV_accessed'] is False and decision['best_epoch_selection'] is False and
            decision['own_optimizers_and_order_RNG_continue'] is extend,'Exact FIT-only all4 continuation decision differs')
    return trace,computed


def packet_check(torch,value,row,index,cache_report,cache):
    require(value['schema']==cache.PACKET_SCHEMA and value['dataset_index']==row['dataset_index']==index and value['identity']==row['identity'] and
            value['source']==cache_report['source'] and value['spec']==cache_report['spec'] and value['phase']==cache_report['phase'],'Own qualified cache row differs')
    G=len(value['identity']['ann_ids']); require(1<=G<=100 and G==row['raw_GT'] and len(set(value['identity']['ann_ids']))==G,'Complete positive raw-GT instance order required')
    shapes=dict(F_samples=(G,1024,256),query_embeddings=(G,256),native_logits=(G,1024),patches=(G,1024,49),native_class_logits=(1,100,2))
    for name,shape in shapes.items():
        t=value[name]; require(isinstance(t,torch.Tensor) and t.device.type=='cpu' and t.dtype==torch.float32 and tuple(t.shape)==shape and
                not t.requires_grad and bool(torch.isfinite(t).all()),'Actual detached FP32 cache operand differs: '+name)
    for name,shape,dtype in (('targets',(G,1024),torch.bool),('point_indices',(G,1024),torch.int64),('query_ids',(G,),torch.int64)):
        t=value[name]; require(isinstance(t,torch.Tensor) and t.device.type=='cpu' and t.dtype==dtype and tuple(t.shape)==shape and not t.requires_grad,'Cache discrete operand differs: '+name)
    H,W=value['identity']['original_shape']; require(bool(((value['point_indices']>=0)&(value['point_indices']<H*W)).all()) and
            len(set(value['query_ids'].tolist()))==G and bool(((value['query_ids']>=0)&(value['query_ids']<100)).all()) and
            bool(((value['patches']>=0)&(value['patches']<=1)).all()) and all(digest(value[n])==v for n,v in row['cache_digests'].items()),'Qualified point/query/value digests differ')
    return G


def independent_logits(torch,state,F,e,L,patch,arm):
    """Functional six-parameter FP32 reconstruction; no trained head.forward."""
    require(arm in ARMS,'Known paired arm required')
    a=torch.nn.functional.layer_norm(F,(256,),weight=None,bias=None,eps=1e-5)
    b=torch.nn.functional.layer_norm(e,(256,),weight=None,bias=None,eps=1e-5)
    x=torch.cat((a,b,L.clamp(-8,8).unsqueeze(1)/8,torch.zeros_like(patch) if arm=='C' else patch),dim=1)
    h=torch.relu(torch.nn.functional.linear(x,state['hidden1.weight'],state['hidden1.bias']))
    h=torch.relu(torch.nn.functional.linear(h,state['hidden2.weight'],state['hidden2.bias']))
    u=torch.nn.functional.linear(h,state['final.weight'],state['final.bias']).squeeze(1)
    result=torch.where(L.abs()<=2,L+u,L)
    require(bool(torch.isfinite(result).all()) and torch.equal(result[L.abs()>2],L[L.abs()>2]),'Independent band/finite logits differ')
    return result


def endpoint_check(torch,states,cache,cache_report,cache_q,spec,started):
    losses={key:[] for key in KEYS}; instances=[]; arithmetic_max=0.
    with torch.no_grad():
        for index,(ref,row) in enumerate(zip(cache_report['packet_refs'],cache_report['rows'])):
            require(time.monotonic()-started<spec['deadline_seconds'],'Prelocked CPU endpoint deadline exceeded')
            require(row['packet']==ref and ref==cache_q['packet_refs'][index],'Qualified741 packet order differs')
            value=torch.load(str(bound(ref)),map_location='cpu'); G=packet_check(torch,value,row,index,cache_report,cache)
            for g in range(G):
                F=value['F_samples'][g]; e=value['query_embeddings'][g].expand(1024,256); L=value['native_logits'][g]
                patch=value['patches'][g]; y=value['targets'][g].to(dtype=torch.float32)
                instances.append(dict(dataset_index=index,image_id=value['identity']['image_id'],annotation_id=value['identity']['ann_ids'][g],query_id=int(value['query_ids'][g])))
                for key in KEYS:
                    z=independent_logits(torch,states[key],F,e,L,patch,key[0])
                    per_point=z.clamp(min=0)-z*y+torch.log1p(torch.exp(-z.abs()))
                    functional=torch.nn.functional.binary_cross_entropy_with_logits(z,y,reduction='none')
                    require(torch.allclose(per_point,functional,rtol=NUMERIC['instance_BCE_arithmetic_rtol'],atol=NUMERIC['instance_BCE_arithmetic_atol']) and
                            bool(torch.isfinite(per_point).all()),'Independent stable point BCE arithmetic differs')
                    arithmetic_max=max(arithmetic_max,float((per_point-functional).abs().max()))
                    losses[key].append(float(per_point.mean()))
            del value
            print(json.dumps(dict(N07_CPU_training_endpoint_images=index+1,instances=len(instances))),flush=True)
    require(len(instances)==1661 and all(len(values)==1661 for values in losses.values()),'Full1661 fixed-cache endpoint population differs')
    return losses,instances,arithmetic_max


def qualify(path):
    started=time.monotonic(); path=Path(path); require(path.is_absolute(),'Absolute ROOT CPU SPEC required'); path=path.resolve(strict=True)
    spec=read(path); train,cache,cached,cache_report,cache_q,context,training,report,closure,source,spec_ref,admit_ref,admission=source_gate(path,spec)
    import psutil
    require(psutil.Process().cpu_affinity()==[0,1],'Actual CPU01 required before numerical imports')
    import torch
    import numpy as np
    require(torch.__version__=='2.0.1+cu118' and np.__version__=='1.23.5' and not torch.cuda.is_initialized(),'Original CPU Torch/NumPy, no CUDA required')
    torch.set_num_threads(2); torch.set_num_interop_threads(2); libraries=imported_files(torch,np,context,closure)
    external=torch.random.get_rng_state().clone(); factory=load(spec['point_heads_source'],'_N07_CPU_point_constructor')
    start=read(bound(report['start'])); decision=read(bound(report['decision']))
    trace_rows=[json.loads(line,parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v))) for line in bound(report['trace']).read_text(encoding='utf-8').splitlines() if line.strip()]
    trace,decision_rows=records_check(report,training,start,decision,trace_rows); states={}; results={}; first8={}
    for key in KEYS:
        early=report['epoch8_checkpoints'][key]
        require(early['completed_head_epochs']==8 and early['updates']==13288 and early['head_state_finite'] is True and early['no_pending_gradients'] is True,
                'Own complete epoch8 checkpoint result differs')
        early_state,first8[key]=checkpoint_check(torch,early['checkpoint'],early['proof'],key,8,training,report,spec,start,trace,factory)
        require(early['mean_endpoint_BCE']==report['endpoint_BCE_histories'][key][7]['BCE'],'Epoch8 bound mean differs')
        final=report['checkpoints'][key]; epoch=report['completed_head_epochs']
        require(final['completed_head_epochs']==epoch and final['updates']==1661*epoch and final['mean_endpoint_BCE']==report['endpoint_BCE_histories'][key][-1]['BCE'] and
                final['head_state_finite'] is True and final['no_pending_gradients'] is True,'Own final checkpoint result differs')
        if epoch==8:
            require(final==early,'No-extension final must be its own epoch8 checkpoint'); states[key],results[key]=early_state,first8[key]
        else:
            states[key],results[key]=checkpoint_check(torch,final['checkpoint'],final['proof'],key,24,training,report,spec,start,trace,factory)
    per_instance,instances,arithmetic_max=endpoint_check(torch,states,cache,cache_report,cache_q,spec,started); endpoints={}
    for key in KEYS:
        mean=math.fsum(per_instance[key])/1661; exported=report['endpoint_BCE_histories'][key][-1]['BCE']
        require(math.isclose(mean,exported,rel_tol=NUMERIC['endpoint_BCE_rtol'],abs_tol=NUMERIC['endpoint_BCE_atol']),
                'Independent final fullFIT mean BCE differs: '+key)
        endpoints[key]=dict(CPU_mean_BCE=mean,GPU_recorded_mean_BCE=exported,absolute_difference=abs(mean-exported),instances=1661,points=1700864)
    require(torch.equal(external,torch.random.get_rng_state()) and not torch.cuda.is_initialized() and 'mmdet' not in sys.modules,
            'CPU endpoint changed external RNG/CUDA or loaded native model')
    for ref in spec['sources']: bound(ref)
    require(reference(path)==spec_ref and reference(__file__)==source and reference(spec['CPU_admission_file'])==admit_ref and
            libraries==imported_files(torch,np,context,closure) and psutil.Process().cpu_affinity()==[0,1] and
            torch.get_num_threads()==torch.get_num_interop_threads()==2,'Final inputs/libraries/resources changed')
    q=dict(schema=Q_SCHEMA,status=STATUS,phase=spec['phase'],source=spec['training_entry_source'],reader_source=source,spec=spec_ref,
           training_spec=spec['training_spec'],training_report=spec['training_report'],training_GPU_RAW=spec['training_GPU_RAW'],training_GPU_admission=spec['training_GPU_admission'],
           CPU_admission=admit_ref,training_phase=training['phase'],cache_CPU_qualification=training['cache_CPU_qualification'],point_heads_source=spec['point_heads_source'],
           RAW_inputs=spec['RAW_inputs'],prior_failed_CPU_receipt=spec['prior_failed_CPU_receipt'],recipe=training['recipe'],numeric_protocol=NUMERIC,
           completed_head_epochs=report['completed_head_epochs'],images_per_epoch=741,instances_per_epoch=1661,points_per_endpoint=1700864,
           updates_per_model=1661*report['completed_head_epochs'],checkpoints=report['checkpoints'],checkpoint_results=results,epoch8_checkpoint_results=first8,
           paired_cold_initializer_CPU_checked=True,private_image_order_and_endpoint_RNG_CPU_reconstructed=True,AdamW_endpoint_states_and_stepcounts_CPU_checked=True,
           all_epoch_records_CPU_checked=True,all4_FIT_only_continuation_decision_CPU_checked=True,continuation_decision_rows=decision_rows,
           final_fullFIT_endpoint_BCE_CPU_recomputed=True,final_endpoints=endpoints,instance_order=instances,per_instance_CPU_BCE=per_instance,
           stable_instance_BCE_arithmetic_max_abs_difference=arithmetic_max,intermediate_epoch_BCE_numerically_replayed=False,
           every_gradient_or_AdamW_update_CPU_replayed=False,optimizer_continuation_trajectory_CPU_replayed=False,
           original_native_teacher_loaded_or_replayed=False,GT_or_JPEG_decode_repeated=False,qualified_cache_targets_and_patches_reused=True,
           native610_unchanged_SOURCE_observed=cache_q['native610_unchanged_SOURCE_observed'],native610_unchanged_independent_CPU_verified=False,
           optimizer_created=False,weights_updated=False,no_DEV=True,no_TEST=True,no_AP=True,
           actual_imported_files=libraries,CPU_execution=dict(affinity=[0,1],threads=2,CUDA_VISIBLE_DEVICES='',CUDA_initialized=False),elapsed_seconds=time.monotonic()-started,
           evidence_limits=['Independent final1024-point-per-instance FIT endpoint arithmetic, not full-frame risk, AP or unseen-data convergence.',
             'Intermediate epoch BCE and gradient activity are checked bound records; only saved epoch8/final states exist, not a full optimizer trajectory.',
             'Fresh moments and continuation are SOURCE-observed and record-consistent; endpoint moments cannot independently prove every intervening update.',
             'GT/JPEG identity relies on the independently qualified immutable cache; sampled F/e/L do not permit a native backbone or matching replay.',
             'CPU/GPU endpoint comparison uses a prelocked practical tolerance, not bitwise equality or a certified GPU numerical bound.',
             'ROOT separately closes this current CPU actual owned terminal; the Q does not self-claim its final RAW.'])
    target=Path(spec['Q_path']); temporary=target.with_suffix(target.suffix+'.tmp'); target.parent.mkdir(parents=True,exist_ok=True)
    require(not target.exists() and not temporary.exists(),'Fresh Q required at final write')
    with temporary.open('x',encoding='utf-8',newline='\n') as stream: json.dump(q,stream,indent=2,allow_nan=False); stream.write('\n')
    temporary.replace(target); del states; gc.collect(); print(json.dumps(dict(status=STATUS,Q=reference(target))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--spec',required=True); parser.add_argument('--owner-execute',action='store_true')
    args=parser.parse_args(); require(args.owner_execute,'Explicit single ROOT-owned CPU execution required'); qualify(args.spec)
