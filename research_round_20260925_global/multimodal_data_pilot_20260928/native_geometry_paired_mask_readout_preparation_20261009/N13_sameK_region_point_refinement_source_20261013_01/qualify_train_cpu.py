"""UNBOUND N13 SOURCE: independent CPU qualification of one training leaf.
All normal ancestors and four preserved failed owned RAWs are checked before
hash/import. The final paired FIT losses, cold initialization, loaded resume
state, cumulative order RNG, optimizer structures/steps and complete visit
records are checked. No optimizer step or intermediate gradient is replayed.
This process cannot certify its own actual terminal; ROOT closes that later.
No native model, image/GT decoder, DEV, TEST, AP or scientific efficacy claim.
"""
import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time
import types

SCHEMA='ROOT-N13-sameK-region-point-training-CPU-spec-v1'
Q_SCHEMA='ROOT-N13-sameK-region-point-training-CPU-qualification-v1'
STATUS='ROOT_N13_SAMEK_REGION_POINT_TRAINING_CPU_QUALIFIED'
TRAIN_SCHEMA='ROOT-N13-sameK-region-point-training-spec-v1'
TRAIN_REPORT_SCHEMA='N13-sameK-region-point-training-report-v1'
TRAIN_STATUS='N13_SAMEK_REGION_POINT_TRAINING_LEAF_COMPLETE_PENDING_ROOT_QUALIFICATION'
CP_SCHEMA='N13-sameK-region-PR-FL91-head-checkpoint-v1'
START_SCHEMA='N13-sameK-region-point-training-start-v1'
LOADED_SCHEMA='N13-sameK-region-point-training-loaded-state-v1'
OWNER_SHA='ef9a9f17d4fb1bc3464936095ef799021dd880f8b15488d558a871c08b6e1924'
ARMS=('LOG_PR','GEO_PR')
SEEDS=(202610120701,202610120702,202610120703)
NAMES=('hidden1.weight','hidden1.bias','hidden2.weight','hidden2.bias','hidden3.weight','hidden3.bias','final.weight','final.bias')
SHAPES=((91,257),(91,),(91,92),(91,),(91,92),(91,),(1,92),(1,))
THREAD_ENV=('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')
NUMERIC=dict(endpoint_BCE_rtol=1e-4,endpoint_BCE_atol=1e-6,point_BCE_arithmetic_rtol=1e-4,point_BCE_arithmetic_atol=1e-6)


def require(ok,message):
    if not ok: raise RuntimeError(message)


def read(path):
    def reject(value): raise ValueError('Nonfinite JSON constant: '+value)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'),parse_constant=reject)


def ref_shape(ref):
    require(type(ref) is dict and set(ref)=={'path','bytes','sha256'} and type(ref['path']) is str and Path(ref['path']).is_absolute() and
        type(ref['bytes']) is int and ref['bytes']>=0 and type(ref['sha256']) is str and len(ref['sha256'])==64 and
        set(ref['sha256'])<=set('0123456789abcdef'),'Malformed pinned reference')


def reference(path):
    path=Path(path).resolve(strict=True); h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''): h.update(block)
    return dict(path=str(path),bytes=path.stat().st_size,sha256=h.hexdigest())


def bound(ref):
    ref_shape(ref); require(reference(ref['path'])==ref,'Pinned input changed: '+ref['path']); return Path(ref['path'])


def terminal(ref,phase,code=0):
    ref_shape(ref); path=Path(ref['path'])
    require(path.name=='PARENT_RECEIPT.json' and type(phase) is str and phase and path.parent.name==phase,'Original RAW locator required')
    raw=read(path)
    require(raw['status']=='ACTUAL_OWNED_TERMINAL' and type(raw['returncode']) is int and raw['returncode']==code and
        type(raw['original_identity']) is dict and raw['original_identity'] and raw['root_wait_reaped'] is True and
        raw['held_handle_exit']==dict(wait_state='WAIT_OBJECT_0',actual_exit_code=code) and
        type(raw['held_handle_exit']['actual_exit_code']) is int and raw['numeric_birth_absence']['original_birth_absent'] is True and
        raw.get('error') is None and raw.get('abort') is None and raw['wait_errors'] is False and raw['GPU_observation_errors']==[],
        'Actual held exit/reap/original birth absence required before every SHA/import')
    if 'phase' in raw: require(raw['phase']==phase,'RAW phase differs')
    peak,sha=raw['whole_GPU_observed_peak_MiB'],raw['admission_sha256']
    require(type(sha) is str and len(sha)==64 and set(sha)<=set('0123456789abcdef') and
        type(peak) in (int,float) and math.isfinite(peak) and 0<=peak<=8192 and raw['whole_GPU_used_limit_MiB']==8192 and
        raw['whole_leaf_VRAM_certified'] is False and raw['GPU_scope']=='whole GPU observed, not per-process WDDM allocation','RAW resource differs')
    return raw


def pre_gate(spec):
    rows=spec['RAW_inputs']; count=spec['normal_RAW_count']
    require(type(count) is int and count>=40 and type(rows) is list and len(rows)==count and
        len({row['receipt']['path'] for row in rows})==count,'Exact prelocked normal40+ ancestor list required')
    raws=[terminal(row['receipt'],row['phase']) for row in rows]
    failed=terminal(spec['prior_failed_CPU_receipt'],Path(spec['prior_failed_CPU_receipt']['path']).parent.name,1)
    extra=spec['additional_failed_RAW_inputs']
    require(type(extra) is list and len(extra)==3 and all(type(row) is dict and set(row)=={'receipt','phase','returncode'} and
        type(row['returncode']) is int and row['returncode']==1 for row in extra),'Preserve exact three-key extra failed3 rows')
    failed_extra=[terminal(row['receipt'],row['phase'],1) for row in extra]
    failures=[spec['prior_failed_CPU_receipt']['path']]+[row['receipt']['path'] for row in extra]
    require(len(set(failures))==4 and not {row['receipt']['path'] for row in rows}.intersection(failures),'Distinct normal/failure locators required')
    # Only NOW may reference()/bound()/SOURCE or numerical imports execute.
    closure={ref['path']:ref for ref in spec['sources']}; require(len(closure)==len(spec['sources']),'Unique complete CPU closure required')
    for ref in spec['sources']: bound(ref)
    require(all(row[k] in closure.values() for row in rows for k in ('receipt','admission','phase_binding','bootstrap')) and
        spec['prior_failed_CPU_receipt'] in closure.values() and all(row['receipt'] in closure.values() for row in extra),'RAW metadata/failure refs absent')
    return raws,closure


def load(ref,name):
    path=bound(ref); source=path.read_bytes(); require(len(source)==ref['bytes'] and hashlib.sha256(source).hexdigest()==ref['sha256'],'SOURCE changed')
    module=types.ModuleType(name); module.__file__=str(path); exec(compile(source,str(path),'exec'),module.__dict__); return module


def source_gate(path,spec):
    require(spec['schema']==SCHEMA and spec['numeric_protocol']==NUMERIC,'Exact CPU endpoint schema/tolerances required')
    raws,closure=pre_gate(spec)
    keys=('entry_source','training_entry_source','training_spec','training_report','training_GPU_RAW','training_GPU_admission',
        'region_common_source','point_heads_source','cache_CPU_qualification')
    require(all(spec[k] in closure.values() for k in keys),'CPU/training/cache/factory closure absent')
    training=read(bound(spec['training_spec'])); report=read(bound(spec['training_report']))
    require(training['schema']==TRAIN_SCHEMA and report['schema']==TRAIN_REPORT_SCHEMA and report['status']==TRAIN_STATUS and
        report['source']==training['entry_source']==spec['training_entry_source'] and report['spec']==spec['training_spec'] and
        report['GPU_admission']==spec['training_GPU_admission'] and report['phase']==training['phase'] and
        spec['RAW_inputs'][:-1]==training['RAW_inputs'] and spec['RAW_inputs'][-1]['receipt']==spec['training_GPU_RAW'] and
        spec['RAW_inputs'][-1]['admission']==spec['training_GPU_admission'] and
        spec['RAW_inputs'][-1]['phase']==training['phase'] and spec['normal_RAW_count']==training['normal_RAW_count']+1 and
        spec['prior_failed_CPU_receipt']==training['prior_failed_CPU_receipt'] and
        spec['additional_failed_RAW_inputs']==training['additional_failed_RAW_inputs'] and
        all(ref in closure.values() for ref in training['sources']), 'Exact complete GPU leaf/CPU actual prefix required')
    train=load(spec['training_entry_source'],'_N13_qualified_train_ABI')
    require(train.SCHEMA==TRAIN_SCHEMA and train.REPORT_SCHEMA==TRAIN_REPORT_SCHEMA and train.ARMS==ARMS and train.SEEDS==SEEDS and
        train.NAMES==NAMES and train.SHAPES==SHAPES and train.PARAMETERS==40497,'Fixed N13 training ABI differs')
    gpu_gate=train.source_gate(Path(spec['training_spec']['path']),training,owner_environment=False)
    common=gpu_gate['common']; owners=[common.owner_join(row,raw,closure) for row,raw in zip(spec['RAW_inputs'],raws)]
    train.owner_ref_join(spec,owners,spec['training_GPU_RAW'],spec['training_GPU_admission'],spec['training_entry_source'],spec['training_spec'],'0')
    require(spec['recipe']==training['recipe']==report['recipe']==train.RECIPE and spec['seed']==training['seed']==report['seed'] and
        spec['first_epoch']==training['first_epoch']==report['first_epoch'] and spec['last_epoch']==training['last_epoch']==report['last_epoch'] and
        spec['region_common_source']==training['region_common_source'] and spec['point_heads_source']==training['point_heads_source'] and
        spec['cache_CPU_qualification']==training['cache_CPU_qualification'] and
        type(spec['deadline_seconds']) is int and 0<spec['deadline_seconds']<=1800,
        'Exact same recipe/seed/epochs/cache and CPU leaf deadline required')
    artifacts=[report[k] for k in ('start','trace','update_trace','decision','initial_loaded_state')]
    artifacts += [report['checkpoints'][arm][k] for arm in ARMS for k in ('checkpoint','proof')]
    require(set(report['checkpoints'])==set(ARMS) and all(ref in closure.values() for ref in artifacts),'Complete current GPU endpoint artifacts absent')
    source,spec_ref=reference(__file__),reference(path); require(source==spec['entry_source'],'Actual CPU reader differs')
    adpath=Path(spec['CPU_admission_file']); require(adpath.is_absolute(),'Absolute CPU owner admission required')
    admission_ref=reference(adpath); admission=read(adpath)
    require(admission['status']=='ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE' and admission['driver_sha256']==OWNER_SHA and
        admission['phase']==spec['phase'] and admission['scientific_spec']==spec_ref and admission['entry']==source['path'] and
        admission['entry_args']==['--spec',str(path),'--owner-execute'] and admission['allowed_cpus']==[0,1] and admission['cuda_visible_devices']=='' and
        admission['automatic_retry'] is False and admission['whole_GPU_used_limit_MiB']==8192 and
        admission['qualified_leaf_deadline_seconds']==spec['deadline_seconds'] and admission['bound_source_sha256'].get(str(path))==spec_ref['sha256'] and
        all(admission['bound_source_sha256'].get(ref['path'])==ref['sha256'] for ref in spec['sources']) and
        os.environ.get('CUDA_VISIBLE_DEVICES')=='' and all(os.environ.get(k)=='2' for k in THREAD_ENV),'Complete single owned CPU admission/environment differs')
    return dict(train=train,common=common,training=training,report=report,GPU_gate=gpu_gate,closure=closure,owners=owners,
        source=source,spec_ref=spec_ref,admission_ref=admission_ref,admission=admission)


def digest(torch,tensor):
    a=tensor.detach().cpu().contiguous().numpy()
    return dict(shape=list(a.shape),dtype=str(a.dtype),sha256=hashlib.sha256(a.tobytes()).hexdigest())


def same_tree(torch,a,b):
    if isinstance(a,torch.Tensor): return isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
    if type(a) is not type(b): return False
    if isinstance(a,dict): return a.keys()==b.keys() and all(same_tree(torch,a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)): return len(a)==len(b) and all(same_tree(torch,x,y) for x,y in zip(a,b))
    return a==b


def deadline(started,spec):
    require(time.monotonic()-started<spec['deadline_seconds'],'Prelocked CPU leaf endpoint deadline exceeded; no fallback/retry')


def resources(psutil,torch):
    require(psutil.Process().cpu_affinity()==[0,1] and torch.get_num_threads()==torch.get_num_interop_threads()==2 and
        not torch.cuda.is_initialized(),'Actual CPU01/threads2/coldCUDA required')


def imported_files(psutil,torch,np,gate):
    modules=dict(psutil=psutil,psutil_platform=psutil._psplatform,psutil_binary=psutil._psplatform.cext,
        torch=torch,torch_binary=torch._C,torch_tensor=sys.modules['torch._tensor'],torch_autograd=sys.modules['torch.autograd'],
        torch_random=torch.random,numpy=np,numpy_binary=np.core._multiarray_umath)
    result={n:reference(m.__file__) for n,m in modules.items()}
    objects=dict(torch_Module=torch.nn.Module,torch_Linear=torch.nn.Linear,AdamW=torch.optim.AdamW,
        Optimizer=torch.optim.Optimizer,point_math=torch.nn.functional.binary_cross_entropy_with_logits,torch_serialization=torch.load)
    result.update({n:reference(inspect.getsourcefile(obj)) for n,obj in objects.items()})
    require(all(ref in gate['closure'].values() and gate['admission']['bound_source_sha256'].get(ref['path'])==ref['sha256']
        for ref in result.values()),'Actual CPU libraries/binaries/serialization/math must be sources-bound')
    return result


def finite_state(torch,state):
    require(type(state) is dict and tuple(state)==NAMES,'Exact eight CPU state tensors required')
    for name,shape in zip(NAMES,SHAPES):
        t=state[name]; require(isinstance(t,torch.Tensor) and t.device.type=='cpu' and t.dtype==torch.float32 and tuple(t.shape)==shape and
            not t.requires_grad and bool(torch.isfinite(t).all()),'Exact finite CPU FP32 endpoint state required')
    require(sum(v.numel() for v in state.values())==40497,'Exact40497 parameter scalars required')


def optimizer_check(torch,state,updates):
    require(type(state) is dict and set(state)=={'state','param_groups'} and len(state['param_groups'])==1,'Exact single AdamW state group required')
    group=state['param_groups'][0]; ids=group['params']
    require(group['lr']==3e-4 and group['weight_decay']==1e-4 and group['betas']==(.9,.999) and group['eps']==1e-8 and
        group['amsgrad'] is False and group['maximize'] is False and group['capturable'] is False and
        group['differentiable'] is False and group['foreach'] is None and len(ids)==8 and len(set(ids))==8,'Exact fixed AdamW group recipe required')
    if updates==0:
        require(state['state']=={},'Cold optimizer must actually have no state'); return {}
    require(set(ids)==set(state['state']),'All8 endpoint moments required'); proof={}
    for name,shape,identifier in zip(NAMES,SHAPES,ids):
        entry=state['state'][identifier]
        require(set(entry)=={'step','exp_avg','exp_avg_sq'} and isinstance(entry['step'],torch.Tensor) and
            entry['step'].device.type=='cpu' and entry['step'].numel()==1 and not entry['step'].requires_grad and
            bool(torch.isfinite(entry['step']).all()) and float(entry['step'])==updates,
            'Exact cumulative endpoint AdamW step required')
        for k in ('exp_avg','exp_avg_sq'):
            t=entry[k]; require(t.device.type=='cpu' and t.dtype==torch.float32 and tuple(t.shape)==shape and not t.requires_grad and
                bool(torch.isfinite(t).all()),'Finite moment shape/dtype required')
        require(bool((entry['exp_avg_sq']>=0).all()),'AdamW second moment must be nonnegative')
        proof[name]={k:digest(torch,v) for k,v in entry.items()}; proof[name]['updates']=updates
    return proof


def load_and_check_states(torch,gate,spec):
    train,training,report=gate['train'],gate['training'],gate['report']; first,last=spec['first_epoch'],spec['last_epoch']; seed=spec['seed']
    loaded=torch.load(str(bound(report['initial_loaded_state'])),map_location='cpu')
    start=read(bound(report['start'])); decision=read(bound(report['decision']))
    require(start['schema']==START_SCHEMA and start['source']==report['source'] and start['spec']==report['spec'] and
        start['GPU_admission']==report['GPU_admission'] and start['phase']==report['phase'] and start['recipe']==train.RECIPE and
        start['seed']==seed and start['first_epoch']==first and start['last_epoch']==last and
        start['initial_loaded_state']==report['initial_loaded_state'] and start['point_heads_source']==training['point_heads_source'] and
        start['cache_CPU_qualification']==training['cache_CPU_qualification'] and start['cold_state_proofs']==report['cold_state_proofs'] and
        start['optimizer_reset'] is False and start['private_order_RNG_reset'] is False and start['paired_initialization'] is True and
        start['native_model_loaded'] is False and start['no_DEV'] is True and start['no_TEST'] is True and start['no_AP'] is True,
        'Exact actual pre-update START binding required')
    require(loaded['schema']==LOADED_SCHEMA and loaded['source']==report['source'] and loaded['spec']==report['spec'] and loaded['seed']==seed and
        loaded['first_epoch']==first and set(loaded['arms'])==set(ARMS),'Actual pre-update loaded-state artifact differs')
    external=torch.random.get_rng_state().clone(); factory=load(training['point_heads_source'],'_N13_CPU_original_PR_factory')
    cold_heads={arm:factory.make_point_head(torch,seed) for arm in ARMS}
    cold_states={arm:{n:p.detach().clone() for n,p in head.state_dict().items()} for arm,head in cold_heads.items()}
    cold_proofs={arm:{n:digest(torch,v) for n,v in state.items()} for arm,state in cold_states.items()}
    require(same_tree(torch,cold_states[ARMS[0]],cold_states[ARMS[1]]) and cold_proofs==loaded['cold_state_proofs']==start['cold_state_proofs']==report['cold_state_proofs'] and
        all(not bool((state['final.weight']!=0).any()) and not bool((state['final.bias']!=0).any()) for state in cold_states.values()) and
        torch.equal(external,torch.random.get_rng_state()) and not torch.cuda.is_initialized(),'Exact paired cold weights/zerooutput/outside RNG required')
    parent_cps={}
    if first>1:
        resume=training['resume']; parent_cps={arm:torch.load(str(bound(resume['checkpoints'][arm])),map_location='cpu') for arm in ARMS}
        require(start['resume_parent_CPU_qualification']==report['resume_parent_CPU_qualification']==resume['CPU_qualification'],
            'Actual START must point to the separately qualified CPU parent')
    else:
        require(training['resume'] is None and start['resume_parent_CPU_qualification'] is None and report['resume_parent_CPU_qualification'] is None,
            'Cold first leaf cannot have a parent')
    gen=torch.Generator(device='cpu'); gen.manual_seed(seed); initial_rng=gen.get_state().clone()
    for arm in ARMS:
        incoming=loaded['arms'][arm]; finite_state(torch,incoming['state_dict']); optimizer_check(torch,incoming['optimizer_state_dict'],1661*(first-1))
        if first==1:
            empty_activity={n:dict(present=0,nonzero=0) for n in NAMES}
            require(same_tree(torch,incoming['state_dict'],cold_states[arm]) and incoming['history']==[] and
                incoming['gradient_activity']==empty_activity and loaded['epoch_orders']==[] and torch.equal(loaded['private_order_state'],initial_rng),
                'First leaf must load exact cold weights/fresh optimizer/private initial RNG')
        else:
            cp=parent_cps[arm]; meta=cp['meta']; finite_state(torch,cp['state_dict'])
            require(meta['schema']==CP_SCHEMA and meta['source']==training['resume']['training_entry_source'] and
                meta['spec']==training['resume']['training_spec'] and meta['GPU_admission']==training['resume']['training_GPU_admission'] and
                meta['seed']==seed and meta['arm']==arm and meta['completed_head_epochs']==first-1 and meta['recipe']==train.RECIPE and
                meta['cache_CPU_qualification']==training['cache_CPU_qualification'] and meta['head_factory_source']==training['point_heads_source'],
                'Own qualified parent checkpoint identities required')
            require(same_tree(torch,incoming['state_dict'],cp['state_dict']) and same_tree(torch,incoming['optimizer_state_dict'],cp['optimizer_state_dict']) and
                incoming['history']==meta['endpoint_BCE_history'] and incoming['gradient_activity']==meta['gradient_activity'] and
                loaded['epoch_orders']==meta['epoch_orders'] and torch.equal(loaded['private_order_state'],cp['private_order_state']),
                'Actual pre-update weights/optimizer/history/private RNG must equal qualified parent exactly')
    cps={arm:torch.load(str(bound(report['checkpoints'][arm]['checkpoint'])),map_location='cpu') for arm in ARMS}; states={}
    for arm,cp in cps.items():
        finite_state(torch,cp['state_dict']); meta=cp['meta']; updates=1661*last
        require(meta['schema']==CP_SCHEMA and meta['source']==report['source'] and meta['spec']==report['spec'] and
            meta['GPU_admission']==report['GPU_admission'] and meta['phase']==report['phase'] and meta['arm']==arm and meta['seed']==seed and
            meta['head_factory_source']==training['point_heads_source'] and meta['cache_spec']==training['cache_spec'] and
            meta['cache_report']==training['cache_report'] and meta['cache_CPU_qualification']==training['cache_CPU_qualification'] and
            meta['recipe']==train.RECIPE and meta['first_epoch_this_leaf']==first and meta['completed_head_epochs']==last and
            meta['updates']==meta['GT_visits']==updates and meta['point_visits']==updates*1024 and meta['image_visits']==741*last and
            meta['parameter_tensors']==8 and meta['parameter_scalars']==40497 and meta['no_pending_gradients'] is True and
            meta['training_resume_eligible'] is True and meta['resume_only_qualified_N13'] is True and meta['native_weights_updated'] is False and
            meta['cold_state_proofs']==cold_proofs and meta['paired_initialization'] is True and meta['paired_image_order'] is True and
            meta['endpoint_BCE_history']==report['endpoint_histories'][arm] and meta['gradient_activity']==report['gradient_activity'][arm] and
            meta['parent_CPU_qualification']==report['resume_parent_CPU_qualification'], 'Complete cumulative own checkpoint metadata differs')
        opt_proof=optimizer_check(torch,cp['optimizer_state_dict'],updates); proof=read(bound(report['checkpoints'][arm]['proof']))
        require(proof['checkpoint']==report['checkpoints'][arm]['checkpoint'] and proof['state']=={n:digest(torch,v) for n,v in cp['state_dict'].items()} and
            proof['optimizer']==opt_proof and proof['private_order_state']==digest(torch,cp['private_order_state']) and proof['updates']==updates and
            proof['full_readback_exact'] is True and report['checkpoints'][arm]['updates']==updates and
            report['checkpoints'][arm]['completed_head_epochs']==last and report['checkpoints'][arm]['endpoint']==meta['endpoint_BCE_history'][-1],
            'Actual per-arm endpoint tensor/optimizer/RNG proofs differ')
        require(len(meta['endpoint_BCE_history'])==last and [r['epoch'] for r in meta['endpoint_BCE_history']]==list(range(1,last+1)) and
            meta['endpoint_BCE_history'][:first-1]==loaded['arms'][arm]['history'],'All48-bound cumulative loss history must extend exact parent prefix')
        expected_stability=train.stability(meta['endpoint_BCE_history'],last)
        require(meta['stability']==report['checkpoints'][arm]['stability']==expected_stability,'Fixed37–42/43–48 stability arithmetic differs')
        states[arm]=cp['state_dict']
    require(cps[ARMS[0]]['meta']['epoch_orders']==cps[ARMS[1]]['meta']['epoch_orders'] and
        torch.equal(cps[ARMS[0]]['private_order_state'],cps[ARMS[1]]['private_order_state']),'Paired cumulative image order/RNG differs')
    require(decision['fixed_final_epoch']==48 and decision['last_epoch_this_leaf']==last and decision['continue_to_prelocked48']==(last<48) and
        decision['automatic_extension'] is False and decision['best_epoch_selection'] is False and decision['DEV_accessed'] is False and decision['TEST_accessed'] is False and
        decision['stability']=={arm:train.stability(report['endpoint_histories'][arm],last) for arm in ARMS},'Fixed48/noDEV-selection decision differs')
    return states,cps,loaded,start,decision,cold_proofs


def records_check(torch,gate,spec,cps,loaded):
    report=gate['report']; first,last=spec['first_epoch'],spec['last_epoch']; seed=spec['seed']; orders=cps[ARMS[0]]['meta']['epoch_orders']
    require(len(orders)==last and orders[:first-1]==loaded['epoch_orders'],'Cumulative paired complete image orders required')
    generator=torch.Generator(device='cpu'); generator.manual_seed(seed)
    for epoch in range(1,last+1):
        indices=torch.randperm(741,generator=generator).tolist(); sha=hashlib.sha256(json.dumps(indices,separators=(',',':')).encode('ascii')).hexdigest()
        require(orders[epoch-1]==dict(epoch=epoch,indices=indices,sha256=sha),'Independent private image permutation differs')
        if epoch==first-1: require(torch.equal(generator.get_state(),loaded['private_order_state']),'Actual resume private RNG is not uninterrupted')
    require(torch.equal(generator.get_state(),cps[ARMS[0]]['private_order_state']),'Independent final private image RNG differs')
    trace=[json.loads(line) for line in bound(report['trace']).read_text(encoding='utf-8').splitlines() if line]
    require(len(trace)==last-first+1 and [row['epoch'] for row in trace]==list(range(first,last+1)),'One complete trace per new epoch required')
    records_path=bound(report['update_trace']); normal_rows=gate['GPU_gate']['cache_report']['rows']
    expected_steps=1661*(last-first+1); activity={arm:{n:dict(present=0,nonzero=0) for n in NAMES} for arm in ARMS}
    with records_path.open('r',encoding='utf-8') as stream:
        counted=0
        for epoch,row in zip(range(first,last+1),trace):
            require(row['seed']==seed and row['images']==741 and row['GT_instances']==1661 and row['updates_per_model']==1661 and
                row['cumulative_updates_per_model']==1661*epoch and row['point_visits_per_model']==1700864 and
                row['image_order_sha256']==orders[epoch-1]['sha256'] and row['paired_initialization'] is True and row['paired_image_order'] is True and
                row['native_forward_calls']==0 and row['native_weights_updated'] is False and set(row['online_BCE'])==set(ARMS),
                'Complete per-epoch paired visit/recipe trace required')
            online={arm:[] for arm in ARMS}; epoch_count=0
            for index in orders[epoch-1]['indices']:
                identity=normal_rows[index]['identity']; ann_ids=identity['ann_ids']
                # query IDs are verified against qualified packets by endpoint_check;
                # a tiny row query list is preferred, otherwise load the core packet.
                if 'query_ids' in normal_rows[index]: qids=normal_rows[index]['query_ids']
                else:
                    value=gate['train'].cache_packet(torch,gate['GPU_gate'],index); qids=value['query_ids'].tolist(); del value
                require(len(qids)==len(ann_ids)==normal_rows[index]['raw_GT'],'Qualified update identity population required')
                for g,(ann,qid) in enumerate(zip(ann_ids,qids)):
                    line=stream.readline(); require(bool(line),'Incomplete paired update records'); actual=json.loads(line)
                    require(actual['seed']==seed and actual['epoch']==epoch and actual['image_index']==index and actual['GT_index']==g and
                        actual['annotation_id']==ann and actual['query_id']==qid and actual['global_step']==1661*(epoch-1)+epoch_count+1 and
                        set(actual['BCE'])==set(actual['gradient_present'])==set(actual['gradient_nonzero'])==set(ARMS),'Exact ordered complete paired update identity required')
                    for arm in ARMS:
                        loss=actual['BCE'][arm]; present=actual['gradient_present'][arm]; nonzero=actual['gradient_nonzero'][arm]
                        require(type(loss) in (int,float) and math.isfinite(loss) and loss>=0 and present==[1]*8 and len(nonzero)==8 and
                            all(type(v) is int and v in (0,1) for v in nonzero),'Finite recorded BCE/8 connected gradient flags required')
                        online[arm].append(loss)
                        for n,nz in zip(NAMES,nonzero): activity[arm][n]['present']+=1; activity[arm][n]['nonzero']+=nz
                    epoch_count+=1; counted+=1
            require(epoch_count==1661,'Each epoch requires all1661 instances')
            for arm in ARMS:
                require(math.fsum(online[arm])/1661==row['online_BCE'][arm] and row['full_FIT_endpoint'][arm]==
                    {k:v for k,v in report['endpoint_histories'][arm][epoch-1].items() if k!='epoch'},'Record-derived online means and bound endpoint history differ')
        require(not stream.read().strip() and counted==expected_steps,'No missing/extra update visits allowed')
    for arm in ARMS:
        for n in NAMES:
            incoming=loaded['arms'][arm]['gradient_activity'][n]; expected={k:incoming[k]+activity[arm][n][k] for k in ('present','nonzero')}
            require(report['gradient_activity'][arm][n]==cps[arm]['meta']['gradient_activity'][n]==expected and
                expected['present']==1661*last and 0<=expected['nonzero']<=expected['present'],'Cumulative recorded gradient presence/counts differ')
    require(report['leaf_updates_per_model']==expected_steps and report['cumulative_updates_per_model']==1661*last and
        report['paired_initialization'] is True and report['paired_image_order'] is True and report['optimizer_reset'] is False and
        report['private_order_RNG_reset'] is False and report['native_model_loaded'] is False and report['native_forward_calls']==0 and
        report['native_weights_updated'] is False and report['AP_computed'] is False and report['no_DEV'] is True and report['no_TEST'] is True and
        report['own_stage_cpu_qualification'] is False and report['current_owned_terminal_self_certified'] is False and
        report['independent_training_trajectory_replayed'] is False and report['cleanup']['allocated_after']==0,'Report evidence boundaries differ')
    return dict(epochs_checked=last-first+1,paired_update_records_checked=counted,per_step_gradient_or_optimizer_replayed=False)


def independent_logits(torch,state,F,L,flag):
    # Explicit PR formula; no trained head.forward or GPU training wrapper.
    norm=torch.nn.functional.layer_norm(F,(256,),weight=None,bias=None,eps=1e-5); coarse=L.clamp(-8,8).unsqueeze(1)/8
    hidden=norm
    for name in ('hidden1','hidden2','hidden3'):
        hidden=torch.relu(torch.nn.functional.linear(torch.cat((hidden,coarse),dim=1),state[name+'.weight'],state[name+'.bias']))
    residual=torch.nn.functional.linear(torch.cat((hidden,coarse),dim=1),state['final.weight'],state['final.bias']).squeeze(1)
    result=L+flag.to(dtype=L.dtype)*residual
    require(bool(torch.isfinite(result).all()) and torch.equal(result[~flag],L[~flag]),'Independent external region gate differs')
    return result


def endpoint_check(torch,states,gate,spec,started):
    losses={arm:[] for arm in ARMS}; editable={arm:[] for arm in ARMS}; counts={arm:0 for arm in ARMS}; identities=[]; arithmetic_max=0.
    with torch.no_grad():
        for index in range(741):
            deadline(started,spec); value=gate['train'].cache_packet(torch,gate['GPU_gate'],index)
            for g in range(len(value['query_ids'])):
                identities.append(dict(dataset_index=index,annotation_id=value['identity']['ann_ids'][g],query_id=int(value['query_ids'][g])))
                for arm in ARMS:
                    F,e,L,y,flag=gate['common'].instance_inputs(torch,value,g,arm,'cpu')
                    logits=independent_logits(torch,states[arm],F,L,flag)
                    per=logits.clamp(min=0)-logits*y+torch.log1p(torch.exp(-logits.abs()))
                    functional=torch.nn.functional.binary_cross_entropy_with_logits(logits,y,reduction='none')
                    require(torch.allclose(per,functional,rtol=NUMERIC['point_BCE_arithmetic_rtol'],atol=NUMERIC['point_BCE_arithmetic_atol']) and
                        bool(torch.isfinite(per).all()),'Independent stable BCE arithmetic differs')
                    arithmetic_max=max(arithmetic_max,float((per-functional).abs().max())); losses[arm].append(float(per.mean()))
                    count=int(flag.sum()); counts[arm]+=count
                    if count: editable[arm].append(float(per[flag].mean()))
            del value
            if (index+1)%100==0: print(json.dumps(dict(N13_CPU_endpoint_images=index+1,instances=len(identities))),flush=True)
    require(len(identities)==1661 and all(len(v)==1661 for v in losses.values()),'Full1661 paired CPU endpoint population required')
    endpoints={}
    for arm in ARMS:
        cpu=dict(BCE=math.fsum(losses[arm])/1661,editable_BCE=(math.fsum(editable[arm])/len(editable[arm]) if editable[arm] else None),
            instances=1661,points=1700864,editable_instances=len(editable[arm]),
            empty_editable_instances=1661-len(editable[arm]),editable_points=counts[arm])
        gpu={k:v for k,v in gate['report']['endpoint_histories'][arm][-1].items() if k!='epoch'}
        require(all(cpu[k]==gpu[k] for k in ('instances','points','editable_instances','empty_editable_instances','editable_points')),'Independent editable population differs')
        differences={}
        for key in ('BCE','editable_BCE'):
            if cpu[key] is None or gpu[key] is None:
                require(cpu[key] is None and gpu[key] is None,'Editable empty population differs'); differences[key]=None
            else:
                require(math.isclose(cpu[key],gpu[key],rel_tol=NUMERIC['endpoint_BCE_rtol'],abs_tol=NUMERIC['endpoint_BCE_atol']),
                    'CPU/GPU final fullFIT endpoint loss differs beyond locked tolerance')
                differences[key]=abs(cpu[key]-gpu[key])
        endpoints[arm]=dict(CPU=cpu,GPU_recorded=gpu,absolute_differences=differences)
    return endpoints,identities,arithmetic_max


def write(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,indent=2,allow_nan=False); stream.write('\n')


def qualify(spec_path):
    started=time.monotonic(); path=Path(spec_path); require(path.is_absolute(),'Absolute CPU SPEC required'); path=path.resolve(strict=True)
    spec=read(path); gate=source_gate(path,spec)
    output=Path(spec['Q_path']); require(output.is_absolute() and not output.exists(),'Fresh absolute Q output required')
    import psutil
    require(psutil.Process().cpu_affinity()==[0,1],'CPU01 required before science imports')
    import torch
    import numpy as np
    require(torch.__version__=='2.0.1+cu118' and np.__version__=='1.23.5' and not torch.cuda.is_initialized(),'Pinned Torch/NumPy/coldCUDA required')
    torch.set_num_threads(2); torch.set_num_interop_threads(2); resources(psutil,torch)
    imported=imported_files(psutil,torch,np,gate); outside=torch.random.get_rng_state().clone()
    deadline(started,spec)
    states,cps,loaded,start,decision,cold=load_and_check_states(torch,gate,spec)
    deadline(started,spec)
    records=records_check(torch,gate,spec,cps,loaded)
    final,identities,arithmetic=endpoint_check(torch,states,gate,spec,started)
    resources(psutil,torch); require(torch.equal(outside,torch.random.get_rng_state()),'CPU qualification changed outside RNG')
    require(imported_files(psutil,torch,np,gate)==imported,'Actual CPU imported SOURCE changed')
    for ref in spec['sources']: bound(ref)
    require(reference(path)==gate['spec_ref'] and reference(__file__)==gate['source'] and reference(spec['CPU_admission_file'])==gate['admission_ref'],
        'Final CPU source/spec/admission changed')
    result=dict(schema=Q_SCHEMA,status=STATUS,source=spec['training_entry_source'],reader_source=gate['source'],spec=gate['spec_ref'],
        CPU_admission=gate['admission_ref'],phase=spec['phase'],training_spec=spec['training_spec'],training_report=spec['training_report'],
        training_GPU_RAW=spec['training_GPU_RAW'],training_GPU_admission=spec['training_GPU_admission'],recipe=spec['recipe'],
        seed=spec['seed'],first_epoch=spec['first_epoch'],last_epoch=spec['last_epoch'],fixed_final_epoch=48,
        normal_RAW_count=spec['normal_RAW_count'],RAW_inputs=spec['RAW_inputs'],prior_failed_CPU_receipt=spec['prior_failed_CPU_receipt'],
        additional_failed_RAW_inputs=spec['additional_failed_RAW_inputs'],point_heads_source=spec['point_heads_source'],
        region_common_source=spec['region_common_source'],cache_CPU_qualification=spec['cache_CPU_qualification'],
        checkpoints=gate['report']['checkpoints'],cold_state_proofs=cold,final_endpoints=final,endpoint_instances=identities,
        stable_BCE_vs_functional_max_absolute_difference=arithmetic,numeric_protocol=NUMERIC,records=records,
        cold_paired_weights_CPU_checked=True,resume_loaded_state_CPU_checked=True,private_image_order_and_endpoint_RNG_CPU_reconstructed=True,
        optimizer_endpoint_structure_and_stepcounts_CPU_checked=True,full_update_visit_sequence_CPU_checked=True,
        final_fullFIT_endpoint_BCE_CPU_recomputed=True,final_editable_BCE_CPU_recomputed=True,
        intermediate_epoch_losses_independently_recomputed=False,per_step_gradient_or_AdamW_replayed=False,
        native_model_loaded=False,native_forward_calls=0,optimizer_steps_executed=0,no_DEV=True,no_TEST=True,AP_computed=False,
        current_owned_CPU_terminal_self_certified=False,CPU_affinity=[0,1],threads=2,CUDA_initialized=False,
        imported_library_refs=imported,elapsed_seconds=time.monotonic()-started,
        evidence_limits=['CPU independently checks this leaf final cached-point losses, identities, paired cold weights and loaded resume continuity.',
            'CPU verifies complete recorded update visit order and cumulative AdamW structure/step counts; it does not replay gradients or updates.',
            'Intermediate epoch losses and their48-window stability arithmetic remain bound GPU records, not independently recomputed endpoints.',
            'Cached point BCE and its plateau cannot establish dense-mask efficacy, novelty, full convergence or TGRS readiness.',
            'The own actual held terminal remains ROOT responsibility after this process exits.'])
    deadline(started,spec)
    write(output,result); print(json.dumps(dict(status=STATUS,Q=reference(output))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--spec',required=True); parser.add_argument('--owner-execute',action='store_true')
    args=parser.parse_args(); require(args.owner_execute,'Explicit single ROOT-owned CPU execution required'); qualify(args.spec)
