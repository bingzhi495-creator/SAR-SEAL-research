"""UNEXECUTED independent U52 numerical equivalence body; stdlib at import only.

Future Root supplies a real Linux owner, two admitted U49 packages and private
scratch. No U40 checkpoint, population reader, entry.main, or evaluation is used.
"""
from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path, PurePosixPath
import platform
import random
import sys
import time
import traceback
import types


PINS = {
    'shared_initialization':'4ba968db7688a03d5975c8546607791a5e09e82ef6c5c7d096cd6a368d1b0292',
    'full_legacy':'c210b318be29aca7d26516e9f47c8849e2c6a76642d4049fe7b133726ee809b9',
    'full_dsconv':'b4b612edf9f87cf29dc48bb675d501658fdec5dbee1ea363a7f02d86d197e5f0',
    'full_head':'6caba966ac7601b611fd1b36fdbd1d758875c5d029758fefe586738c8009f16e',
    'full_feature':'f7953780580449dd246cbdf1803652183c1dd178f2fd60b37adb238a6d2b6a5a',
    'plain_legacy':'902e13e5c60b1579459854c6bf4393ad8f268092a73a1d851f0fa640c810f48d',
    'plain_feature':'d93ff973a57ec3abfde98bb7b9ab4fcf5a3443445a512c2acd16460fce5bfbad',
    'instance_adapter':'9ad221170f69c028474f76700c991cb76dfecf19b11f5198767843b8e378d8b3',
    'image_geometry':'65db107de80c5f1d4dade7cb2ffd3f241fd6bf7f9b233cb1aa9c90cbfd00abf7',
    'u36_helpers':'1e726092a19d80a06f58d6a56f85f4c15bff2b744f0468a7e0a35fa2c85999e5',
    'u39_training_ops':'378c955877674a65f6b2180d4c269e01355245881229ed0d2ec555b184ee7e5d',
    'u50_pilot':'003641b7c81e198bb70502073aba22f623f9dac82554ca9ce90ba059ac20625b',
    'u50_loader':'aceadae1454b6a99a1a191e9c71ad2b9b636914f4e9a85dc4907a92eeb09eee1',
    'candidate':'80e6b777ccd1b41cbdc5bd4ef34f4e4d3d429b827f039c830b78bc08a847eed9',
}


def need(ok, message):
    if not ok: raise ValueError(message)


def equivalence_contract():
    return dict(schema='U52-shared-encoder-equivalence-prelock-v1',
        environment=dict(python='3.12.14',torch='2.10.0+cu130',numpy='1.26.4',system='Linux',machine='aarch64'),
        arms=['PLAIN','FULL'],seed=202610173101,
        warmup=dict(dataset_index=54,G=1,step=1,implementation='original'),
        trajectories=['BASE_A','BASE_B','CANDIDATE'],comparison_indices=[710,54],comparison_steps=[2,3],
        baseline_control='bitexact_all_grad_model_Adam_BN_RNG',stop_on_baseline_control_mismatch=True,
        maximum_disposable_updates=14,query_chunk=1,image_batch_size=1,branch_size=400,canvas_size=800,
        optimizer=dict(name='AdamW',lr=3e-4,weight_decay=1e-4,betas=[.9,.999],eps=1e-8,foreach=False,fused=False),
        tolerances=dict(gradients=dict(rtol=1e-4,atol=1e-6),parameters=dict(rtol=1e-5,atol=1e-7),
            buffers=dict(rtol=1e-5,atol=1e-7),exp_avg=dict(rtol=1e-4,atol=1e-7),
            exp_avg_sq=dict(rtol=3e-4,atol=1e-10),loss=dict(rtol=1e-6,atol=1e-7)),
        integer_state_RNG_and_groups='exact',encoder_saves='save_on_cpu_pin_memory_true',
        AMP=False,compile=False,checkpoint_recompute=False,automatic_retry=False,
        main_training=False,formal_training_budget_credit=0,AP=False,DEV=False,TEST=False,
        resume_U40_checkpoint=False,baseline_and_candidate_start_same_warm_state=True,
        private_CPU_audit_snapshots=True,performance_benchmark=False,
        limits=dict(allowed_cpus=[0,1],threads=2,allocator_bytes=4*1024**3,
            maximum_RSS_bytes=16*1024**3,minimum_available_RAM_bytes=16*1024**3,deadline_seconds=1800))


def validate_prelock(value):
    need(json.dumps(value,sort_keys=True,allow_nan=False)==json.dumps(equivalence_contract(),sort_keys=True,allow_nan=False),
         'Exact prelocked warm/control/carried sequence, tolerances and independent resource scope')


def _strict_ref(value):
    need(type(value) is dict and set(value)=={'path','bytes','sha256'}
         and type(value['path']) is str and PurePosixPath(value['path']).is_absolute()
         and str(PurePosixPath(value['path']))==value['path'] and '..' not in PurePosixPath(value['path']).parts
         and type(value['bytes']) is int and value['bytes']>0
         and type(value['sha256']) is str and len(value['sha256'])==64
         and set(value['sha256'])<=set('0123456789abcdef'),'Strict future DGX-local file reference')
    return value


def validate_context_metadata(context):
    need(type(context) is dict and set(context)=={'portable_refs','scratch_directory'}
         and type(context['portable_refs']) is dict and set(context['portable_refs'])=={'54','710'},
         'Only two admitted private packages plus independent scratch')
    paths=[PurePosixPath(_strict_ref(v)['path']) for v in context['portable_refs'].values()]
    scratch=PurePosixPath(context['scratch_directory'])
    need(type(context['scratch_directory']) is str and scratch.is_absolute() and str(scratch)==context['scratch_directory']
         and scratch!=PurePosixPath('/') and '..' not in scratch.parts and len(set(paths))==2
         and all(scratch!=p and scratch not in p.parents for p in paths),'Separate canonical new scratch path')


def comparison_verdict(baseline_exact, candidate_within):
    need(type(baseline_exact) is type(candidate_within) is bool,'Actual comparison booleans')
    if not baseline_exact:return 'BASELINE_CONTROL_MISMATCH'
    return 'WITHIN_PRELOCKED_TOLERANCES' if candidate_within else 'CANDIDATE_OUTSIDE_PRELOCKED_TOLERANCES'


def _reference(path):
    path=Path(path);digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return dict(path=str(path),bytes=path.stat().st_size,sha256=digest.hexdigest())


def _closed(c,value):
    _strict_ref(value);path=Path(c.closed(value))
    need(path.resolve()==path==Path(value['path']) and _reference(path)==value,'Actual admitted canonical source/input bytes')
    return path


def _load(c,ref,alias,originals):
    path=_closed(c,ref);originals.setdefault(alias,(alias in sys.modules,sys.modules.get(alias)))
    module=types.ModuleType(alias);module.__file__=str(path);sys.modules[alias]=module
    exec(compile(path.read_bytes(),str(path),'exec'),module.__dict__)
    return module


def _load_sources(c,refs,own_ref,originals):
    need(type(refs) is dict and set(refs)==set(PINS),'Exact fourteen explicit original/candidate dependencies')
    all_paths=[]
    for role,pin in PINS.items():
        need(_strict_ref(refs[role])['sha256']==pin,'Fixed ordinary source pin: '+role)
        all_paths.append(str(_closed(c,refs[role])))
    need(len(set(all_paths))==len(all_paths),'Distinct explicit component source files')
    need(_closed(c,own_ref)==Path(__file__).resolve(),'Actual new equivalence body admitted')
    loader=_load(c,refs['u50_loader'],'_U52_U50_LOADER',originals)
    modules={role:loader.load_closed(lambda ref:_closed(c,ref),refs[role],alias,originals)
             for role,alias in loader.LOAD_ORDER}
    modules['u50_pilot']=_load(c,refs['u50_pilot'],'_U52_U50_PILOT',originals)
    modules['candidate']=_load(c,refs['candidate'],'_U52_CANDIDATE',originals)
    need(set(modules)==set(PINS)-{'u50_loader'},'Actual numeric roles loaded, no entry.main execution')
    return modules


def _cpu_tree(torch,value,c):
    if isinstance(value,torch.Tensor):
        c.check();return value.detach().cpu().clone()
    # Module.state_dict is an OrderedDict; retain all keys/values in a safe
    # ordinary dict, rather than requiring the concrete built-in dict type.
    if isinstance(value,dict):return {key:_cpu_tree(torch,item,c) for key,item in value.items()}
    if type(value) is list:return [_cpu_tree(torch,item,c) for item in value]
    if type(value) is tuple:return tuple(_cpu_tree(torch,item,c) for item in value)
    need(value is None or type(value) in (str,int,float,bool),'Only safe tensor/basic CPU audit payload')
    return value


def _rng(torch,np):
    state=np.random.get_state()
    return dict(python=deepcopy(random.getstate()),numpy=dict(generator=state[0],keys=state[1].tolist(),
        position=int(state[2]),has_gauss=int(state[3]),cached_gaussian=float(state[4])),
        torch_CPU=torch.get_rng_state().clone(),torch_CUDA=torch.cuda.get_rng_state(0).cpu().clone())


def _set_rng(torch,np,value):
    random.setstate(value['python']);n=value['numpy']
    np.random.set_state((n['generator'],np.asarray(n['keys'],dtype=np.uint32),n['position'],n['has_gauss'],n['cached_gaussian']))
    torch.set_rng_state(value['torch_CPU']);torch.cuda.set_rng_state(value['torch_CUDA'],0)


def _rng_equal(torch,a,b):
    return (a['python']==b['python'] and a['numpy']==b['numpy']
            and torch.equal(a['torch_CPU'],b['torch_CPU']) and torch.equal(a['torch_CUDA'],b['torch_CUDA']))


def _rng_digests(value):
    return {name:hashlib.sha256((json.dumps(item,sort_keys=True,allow_nan=False).encode()
        if name in ('python','numpy') else item.numpy().tobytes())).hexdigest() for name,item in value.items()}


def _groups(optimizer,params):
    names={id(value):name for name,value in params.items()}
    return [dict((key,[names[id(p)] for p in value] if key=='params' else deepcopy(value))
                 for key,value in group.items()) for group in optimizer.param_groups]


def _save(torch,c,path,payload):
    need(not path.exists(),'Exclusive private CPU audit snapshot, no overwrite/retry')
    c.check()
    with path.open('xb') as stream:torch.save(payload,stream)
    c.check();return _reference(path)


def _read(torch,c,ref):
    c.check();need(_reference(ref['path'])==ref,'Same owned CPU snapshot bytes before reuse')
    value=torch.load(ref['path'],map_location='cpu',weights_only=True)
    c.check();return value


def _equal_tensor(torch,current,reference):
    return (current.dtype==reference.dtype and current.shape==reference.shape
            and bool(torch.equal(current.detach().cpu(),reference)))


def _compare_tensor(torch,c,name,current,reference,policy,exact):
    c.check();row=dict(name=name,passed=False)
    if current is None or reference is None:
        row.update(both_none=current is reference,passed=current is reference)
        return row
    need(current.dtype==reference.dtype and current.shape==reference.shape,'Comparison tensor names/shapes/dtypes: '+name)
    a=current.detach().cpu();b=reference
    finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    row.update(shape=list(a.shape),dtype=str(a.dtype),elements=a.numel(),finite=finite,
               exact_required=bool(exact or not a.is_floating_point()))
    if not finite:return row
    x=a.to(torch.float64);y=b.to(torch.float64);difference=(x-y).abs()
    rtol=0. if row['exact_required'] else policy['rtol'];atol=0. if row['exact_required'] else policy['atol']
    violations=int((difference>atol+rtol*y.abs()).sum().item())
    error_norm=float(torch.linalg.vector_norm(x-y).item());reference_norm=float(torch.linalg.vector_norm(y).item())
    row.update(atol=atol,rtol=rtol,violations=violations,passed=violations==0,
        bitexact=bool(torch.equal(a,b)),max_abs=float(difference.max().item()) if a.numel() else 0.,
        max_relative=float((difference/y.abs().clamp_min(1e-30)).max().item()) if a.numel() else 0.,
        error_L2=error_norm,reference_L2=reference_norm,current_L2=float(torch.linalg.vector_norm(x).item()),
        relative_L2=error_norm/reference_norm if reference_norm else (0. if error_norm==0 else None))
    return row


class _Audit:
    """Observe gradients before original AdamW.step; no gradient/backward hooks."""
    def __init__(self,torch,np,c,params,effective,pre,reference,path):
        self.torch,self.np,self.c,self.params=torch,np,c,params
        self.effective,self.pre,self.reference,self.path=effective,pre,reference,path
        self.gradients=None;self.comparison=dict(gradients=[],parameters=[],buffers=[],Adam=[],passed=True)

    def before(self,optimizer):
        if self.reference is None:
            self.gradients={name:_cpu_tree(self.torch,p.grad,self.c) for name,p in self.params.items()}
        else:
            need(set(self.reference['gradients'])==set(self.params),'Every original effective/inactive gradient is compared')
            for name,p in self.params.items():
                self.comparison['gradients'].append(_compare_tensor(self.torch,self.c,name,p.grad,
                    self.reference['gradients'][name],self.pre['tolerances']['gradients'],self.exact))

    def after(self,adapter,optimizer,stage):
        torch,np,c=self.torch,self.np,self.c
        current_rng=_rng(torch,np);groups=_groups(optimizer,self.params)
        if self.reference is None:
            payload=dict(schema='U52-private-comparison-snapshot-v1',gradients=self.gradients,
                model=_cpu_tree(torch,adapter.state_dict(),c),
                Adam={name:_cpu_tree(torch,optimizer.state[self.params[name]],c) for name in sorted(self.effective)},
                groups=groups,rng=current_rng,loss={k:stage[k] for k in ('loss','BCE','softDice')})
            self.ref=_save(torch,c,self.path,payload);payload=None;self.gradients=None
            return dict(reference=self.ref,comparison=False)
        reference=self.reference
        need(reference['schema']=='U52-private-comparison-snapshot-v1'
             and set(reference['model'])==set(adapter.state_dict()) and set(reference['Adam'])==self.effective,
             'Complete model buffers/parameters and all actual Adam states')
        state=adapter.state_dict()
        for name,value in state.items():
            category='parameters' if name in self.params else 'buffers'
            self.comparison[category].append(_compare_tensor(torch,c,name,value,reference['model'][name],
                self.pre['tolerances'][category],self.exact))
        value=state=None
        for name in sorted(self.effective):
            actual=optimizer.state[self.params[name]];wanted=reference['Adam'][name]
            need(set(actual)==set(wanted)=={'step','exp_avg','exp_avg_sq'},'Every Adam moment and actual step')
            for key in ('step','exp_avg','exp_avg_sq'):
                self.comparison['Adam'].append(_compare_tensor(torch,c,name+'.'+key,actual[key],wanted[key],
                    self.pre['tolerances'].get(key,dict(rtol=0.,atol=0.)),self.exact or key=='step'))
        actual=wanted=None
        self.comparison['groups_exact']=groups==reference['groups']
        self.comparison['RNG_exact']=_rng_equal(torch,current_rng,reference['rng'])
        self.comparison['RNG_digests']=dict(current=_rng_digests(current_rng),reference=_rng_digests(reference['rng']))
        loss_policy=self.pre['tolerances']['loss'];loss_rows=[]
        for key in ('loss','BCE','softDice'):
            a,b=stage[key],reference['loss'][key];delta=abs(a-b)
            allowed=0. if self.exact else loss_policy['atol']+loss_policy['rtol']*abs(b)
            loss_rows.append(dict(name=key,current=a,reference=b,max_abs=delta,allowed=allowed,passed=delta<=allowed))
        self.comparison['loss']=loss_rows
        self.comparison['passed']=(self.comparison['groups_exact'] and self.comparison['RNG_exact']
            and all(row['passed'] for key in ('gradients','parameters','buffers','Adam','loss') for row in self.comparison[key]))
        return self.comparison


def _observed_optimizer(torch,params,effective,pre):
    class AuditedAdamW(torch.optim.AdamW):
        audit=None
        def step(self,closure=None):
            need(closure is None,'Original AdamW without closure or extra optimization step')
            if self.audit is not None:self.audit.before(self)
            return super().step()
    p=pre['optimizer']
    return AuditedAdamW([params[name] for name in sorted(effective)],lr=p['lr'],weight_decay=p['weight_decay'],
        betas=tuple(p['betas']),eps=p['eps'],foreach=False,fused=False)


def _warm_payload(torch,np,c,adapter,optimizer,params,effective,inactive):
    return dict(schema='U52-private-original-warm-state-v1',step=1,
        model=_cpu_tree(torch,adapter.state_dict(),c),optimizer=_cpu_tree(torch,optimizer.state_dict(),c),
        effective=sorted(effective),inactive=sorted(inactive),groups=_groups(optimizer,params),rng=_rng(torch,np))


def _restore_warm(torch,np,c,adapter,optimizer,params,effective,inactive,warm):
    need(warm['schema']=='U52-private-original-warm-state-v1' and warm['step']==1
         and warm['effective']==sorted(effective) and warm['inactive']==sorted(inactive), 'Original warm parameter/Adam mapping')
    adapter.load_state_dict(warm['model'],strict=True);optimizer.load_state_dict(warm['optimizer'])
    need(set(adapter.state_dict())==set(warm['model']) and _groups(optimizer,params)==warm['groups'], 'Exact restored names and Adam groups')
    checks=0
    for name,value in adapter.state_dict().items():
        c.check();need(_equal_tensor(torch,value,warm['model'][name]),'Exact warm model/BN restoration: '+name);checks+=1
    saved_ids=[key for group in warm['optimizer']['param_groups'] for key in group['params']]
    need(len(saved_ids)==len(effective) and set(saved_ids)==set(warm['optimizer']['state']), 'Complete original Adam index mapping')
    for name,key in zip(sorted(effective),saved_ids):
        actual=optimizer.state[params[name]];expected=warm['optimizer']['state'][key]
        need(set(actual)==set(expected)=={'step','exp_avg','exp_avg_sq'},'Original warm Adam state fields')
        for field in actual:
            c.check();need(_equal_tensor(torch,actual[field],expected[field]),'Exact warm Adam tensor: '+name+'.'+field);checks+=1
    _set_rng(torch,np,warm['rng']);need(_rng_equal(torch,_rng(torch,np),warm['rng']),'Exact warm Python/NumPy/CPU/CUDA RNG')
    return checks


def _gpu_zero(torch,c):
    gc.collect();torch.cuda.synchronize();torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache();torch.cuda.synchronize();c.check()
    result=dict(allocated_bytes=int(torch.cuda.memory_allocated()),reserved_bytes=int(torch.cuda.memory_reserved()))
    need(result==dict(allocated_bytes=0,reserved_bytes=0),'All owned model/optimizer/query GPU references released')
    return result


def _run_arm(torch,np,c,adapters,reader,modules,pre,scratch,arm,record,report):
    adapter=params=optimizer=audit=warm=reference=before_buffers=None;handles=[]
    ops=modules['u39_training_ops'];helpers=modules['u36_helpers']
    counters=dict(backbone=0,feature_exit=0,legacy_forward=0,**{name:0 for name in ops.HEAD_NAMES})
    record.update(status='STARTED',updates_completed=0,stages=[],baseline_control_exact=None,candidate_within=None)
    def count(name):
        def hook(module,inputs,output):counters[name]+=1
        return hook
    def update(index,step,implementation,trajectory):
        stage=dict(status='STARTED',dataset_index=index,step=step,implementation=implementation,trajectory=trajectory)
        record['stages'].append(stage);before=dict(counters)
        function=ops._update if implementation=='original' else modules['candidate'].update_shared_encoder
        function(torch,c,reader,index,adapter,optimizer,params,effective,inactive,helpers,step,stage,record,report)
        stage['calls']={name:counters[name]-before[name] for name in counters}
        need(stage['calls']['backbone']==stage['calls']['feature_exit']==1
             and stage['calls']['legacy_forward']==0 and all(stage['calls'][name]==0 for name in ops.HEAD_NAMES),
             'Exactly one original image encoding and no inactive legacy heads')
        stage['inactive_head_BN']=helpers._verify_buffers(torch,adapter.backbone.legacy,before_buffers)
        stage['resources']=ops._sample(torch,c,'actual_update_complete_before_CPU_audit',True)
        need(stage['resources']['CUDA_peak_allocated_bytes']<=pre['limits']['allocator_bytes'],'Actual stage within original 4GiB allocator')
        stage['status']='UPDATE_RETURNED';return stage
    try:
        adapter=adapters.pop(arm);adapter.train()
        effective,inactive,record['parameters']=ops._partition(adapter)
        legacy=adapter.backbone.legacy
        handles.extend((adapter.backbone.register_forward_hook(count('backbone')),
            legacy.CONV0_4.register_forward_hook(count('feature_exit')),legacy.register_forward_hook(count('legacy_forward'))))
        handles.extend(getattr(legacy,name).register_forward_hook(count(name)) for name in ops.HEAD_NAMES)
        before_buffers=helpers._head_buffers(torch,legacy);legacy=None
        need(len(before_buffers)==(12 if arm=='PLAIN' else 18),'Original inactive semantic BN inventory')
        random.seed(pre['seed']);np.random.seed(pre['seed']%2**32)
        torch.random.default_generator.manual_seed(pre['seed']);torch.cuda.manual_seed(pre['seed'])
        adapter.to(device='cuda:0',dtype=torch.float32);params=dict(adapter.named_parameters())
        optimizer=_observed_optimizer(torch,params,effective,pre)
        update(54,1,'original','WARMUP')
        record['warm_head_nonzero']=bool((adapter.head[-1].weight!=0).any() or (adapter.head[-1].bias!=0).any())
        need(record['warm_head_nonzero'],'Nonvacuous comparison: real original warmup made final head nonzero')
        warm=_warm_payload(torch,np,c,adapter,optimizer,params,effective,inactive)
        report['private_audit_snapshot_write_attempted']=True
        record['warm_ref']=_save(torch,c,scratch/(arm+'_WARM.pt'),warm);warm=None
        report['private_audit_weights_written']=True
        optimizer=None;adapter.cpu();params=None
        record['after_warm_release']=_gpu_zero(torch,c)
        references={};control_pass=True;candidate_pass=True
        for trajectory in pre['trajectories']:
            if trajectory=='CANDIDATE' and not control_pass:break
            warm=_read(torch,c,record['warm_ref']);adapter.to(device='cuda:0',dtype=torch.float32)
            params=dict(adapter.named_parameters());optimizer=_observed_optimizer(torch,params,effective,pre)
            restored=_restore_warm(torch,np,c,adapter,optimizer,params,effective,inactive,warm);warm=None
            run=record.setdefault('trajectories',{})[trajectory]=dict(exact_restored_tensor_checks=restored,steps=[],passed=True)
            for index,step in zip(pre['comparison_indices'],pre['comparison_steps']):
                reference=None if trajectory=='BASE_A' else _read(torch,c,references[step])
                audit=_Audit(torch,np,c,params,effective,pre,reference,scratch/(arm+'_BASE_A_STEP'+str(step)+'.pt'))
                audit.exact=trajectory=='BASE_B';optimizer.audit=audit
                stage=update(index,step,'candidate' if trajectory=='CANDIDATE' else 'original',trajectory)
                result=audit.after(adapter,optimizer,stage)
                if trajectory=='BASE_A':references[step]=result['reference']
                else:run['passed']=run['passed'] and result['passed']
                stage['comparison']=result;stage['status']='AUDITED';run['steps'].append(step)
                optimizer.audit=None;audit=reference=None;gc.collect();c.check()
            if trajectory=='BASE_B':control_pass=run['passed']
            if trajectory=='CANDIDATE':candidate_pass=run['passed']
            optimizer=None;adapter.cpu();params=None
            run['after_release']=_gpu_zero(torch,c)
        record.update(baseline_control_exact=control_pass,candidate_within=candidate_pass if control_pass else None,
            verdict=comparison_verdict(control_pass,candidate_pass),calls=counters,status='COMPARISON_COMPLETED' if control_pass else 'CONTROL_STOPPED')
        need(record['updates_completed']==(7 if control_pass else 5),'Actual warm plus returned trajectory updates, no hidden steps')
        return control_pass,candidate_pass
    except BaseException as error:
        # Clear failed callee traceback frames before testing GPU zero; they can
        # otherwise retain CUDA tensors despite the update's own finally block.
        record.update(status='FAILED',error=_error(error))
        return False,False
    finally:
        for handle in handles:handle.remove()
        handles.clear()
        if optimizer is not None:optimizer.audit=None
        audit=reference=warm=optimizer=params=before_buffers=None
        adapter=None;gc.collect()
        try:record['final_CUDA_cleanup']=_gpu_zero(torch,c)
        except BaseException as error:record.update(status='FAILED',cleanup_error=_error(error))


def _error(error):
    result=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
    current=error;seen=set()
    while current is not None and id(current) not in seen:
        seen.add(id(current));following=current.__cause__ or current.__context__
        traceback.clear_frames(current.__traceback__);current.__traceback__=None;current=following
    return result


def run_equivalence(torch,np,c,*,context,source_refs,prelock,body_source_ref,report):
    """Future real runtime only; returns a partial/completed JSON report, never models.

    c.closed/check/psutil come from a separately admitted real resource owner.
    That owner must reject FAILED/CONTROL_STOPPED and observe actual reaping.
    Numerical mismatch is a recorded diagnostic outcome, never an auto-retry.
    """
    need(type(report) is dict and not report,'Fresh caller-owned partial report')
    started=time.monotonic();aliases={};modules=adapters=reader=items=records=None
    outside_cpu=outside_cuda=outside_python=outside_numpy=None
    report.update(schema='U52-shared-encoder-equivalence-report-pending-v1',status='STARTED',
        prelock=deepcopy(prelock),source_refs=deepcopy(source_refs),context=deepcopy(context),arms={},
        successful_optimizer_updates=0,optimizer_steps_entered=0,new_training=False,pilot_only=True,
        main_training=False,formal_training_budget_credit=0,U40_frontier_modified=False,
        AP=False,DEV=False,TEST=False,qualified=False,innovation_established=False,speedup_established=False,
        cross_environment_equivalence_claimed=False,whole_population_qualified=False,
        whole_GPU_observed_peak_MiB=None,whole_GPU_WDDM8GiB_contract_certified=False,
        private_audit_snapshot_write_attempted=False,private_audit_weights_written=False,
        model_weights_adopted=False,automatic_retry=False,cleanup_errors=[])
    try:
        validate_prelock(prelock);validate_context_metadata(context);c.check()
        actual=dict(python=platform.python_version(),torch=str(torch.__version__),numpy=str(np.__version__),system=platform.system(),machine=platform.machine())
        need(actual==prelock['environment'] and not torch.cuda.is_initialized() and torch.get_default_dtype()==torch.float32
             and torch.get_num_threads()==torch.get_num_interop_threads()==2 and c.psutil.Process().cpu_affinity()==[0,1],
             'Exact cold DGX runtime and two-thread CPU01 owner')
        need(torch.is_grad_enabled() and not torch.is_autocast_enabled()
             and torch.backends.cudnn.deterministic is True and torch.backends.cudnn.benchmark is False
             and torch.backends.cudnn.allow_tf32 is False and torch.backends.cuda.matmul.allow_tf32 is False,
             'Original deterministic FP32/noAMP/noTF32 execution policy')
        outside_cpu=torch.get_rng_state().clone();outside_python=random.getstate();outside_numpy=np.random.get_state()
        modules=_load_sources(c,source_refs,body_source_ref,aliases)
        scratch=Path(context['scratch_directory'])
        need(scratch.resolve()==scratch and not scratch.exists() and not scratch.is_symlink(),'Exclusive new canonical private scratch')
        for value in list(source_refs.values())+[body_source_ref]:
            need(scratch!=Path(value['path']) and scratch not in Path(value['path']).parents,'Scratch cannot contain admitted sources')
        scratch.mkdir(parents=True,exist_ok=False);report['scratch_directory']=str(scratch)
        pilot=modules['u50_pilot'];items={};records={};report['portable_inputs']={}
        for index in (54,710):
            item,row,proof=pilot._load_item(torch,c,index,context['portable_refs'][str(index)],modules['image_geometry'])
            items[index]=item;records[index]=row;report['portable_inputs'][str(index)]=proof;item=row=proof=None
        reader=pilot._PortableReader(items,records);items=records=None
        initializer=modules['shared_initialization']
        adapters,report['initialization']=initializer.build_pair(torch,
            {key:modules[key] for key in initializer.SOURCE_SHA256},{key:source_refs[key] for key in initializer.SOURCE_SHA256})
        initial=report['initialization']
        need(initial['seed']==initial['adapter_seed']==prelock['seed'] and len(initial['common_state'])==104
             and initial['common_parameter_tensors']==68 and initial['common_BN_buffers']==36
             and not torch.cuda.is_initialized(),'Actual original common104 and shared adapter initialization, still CPU only')
        initial=None;gc.collect();c.check()
        torch.cuda.init();torch.cuda.set_device(0)
        total=int(torch.cuda.get_device_properties(0).total_memory);cap=prelock['limits']['allocator_bytes']
        need(total>=cap,'Device supports prelocked 4GiB allocator')
        torch.cuda.set_per_process_memory_fraction(cap/float(total),device=0)
        need(torch.cuda.memory_allocated()==torch.cuda.memory_reserved()==0,'Allocator cap before any model/input transfer')
        outside_cuda=torch.cuda.get_rng_state(0).cpu().clone()
        report['CUDA_allocator_limit']=dict(bytes=cap,total_bytes=total,set_before_model_transfer=True)
        control=True;candidate=True
        for arm in prelock['arms']:
            record=report['arms'][arm]={}
            a,b=_run_arm(torch,np,c,adapters,reader,modules,prelock,scratch,arm,record,report)
            need(record['status']!='FAILED','Actual arm/update or cleanup failure; preserved detailed partial record')
            control=control and a;candidate=candidate and b
            if not control:break
        report['verdict']=comparison_verdict(control,candidate)
        report['status']='COMPARISON_COMPLETED_PENDING_ROOT' if control else 'BASELINE_CONTROL_STOPPED'
        report['private_audit_weights_written']=True
        report['reader_visits']=list(reader.visits)
        if control:
            need(not adapters and report['successful_optimizer_updates']==report['optimizer_steps_entered']==14
                 and reader.visits==([54,710,54,710,54,710,54]*2),'Exactly14 real disposable updates on prelocked two-image trajectories')
        for value in list(source_refs.values())+[body_source_ref]:_closed(c,value)
    except BaseException as error:
        report.update(status='FAILED',error=_error(error))
    finally:
        adapters=reader=items=records=modules=None;gc.collect()
        if torch.cuda.is_initialized():
            try:
                report['final_CUDA_cleanup']=_gpu_zero(torch,c)
                if outside_cuda is not None:
                    torch.cuda.set_rng_state(outside_cuda,0)
                    report['caller_postinitialization_CUDA_RNG_restored']=bool(torch.equal(outside_cuda,torch.cuda.get_rng_state(0).cpu()))
            except BaseException as error:report['cleanup_errors'].append(_error(error));report['status']='FAILED'
        for alias,(present,original) in reversed(list(aliases.items())):
            if present:sys.modules[alias]=original
            else:sys.modules.pop(alias,None)
        report['module_aliases_restored']=all((alias in sys.modules)==present and (not present or sys.modules[alias] is original)
            for alias,(present,original) in aliases.items())
        if outside_cpu is not None:
            torch.set_rng_state(outside_cpu);np.random.set_state(outside_numpy);random.setstate(outside_python)
            report['caller_CPU_RNG_restored']=bool(torch.equal(outside_cpu,torch.get_rng_state()))
            report['caller_Python_RNG_restored']=random.getstate()==outside_python
            after=np.random.get_state()
            report['caller_NumPy_RNG_restored']=(after[0]==outside_numpy[0] and bool(np.array_equal(after[1],outside_numpy[1])) and after[2:]==outside_numpy[2:])
        report['elapsed_seconds']=time.monotonic()-started
        if report['elapsed_seconds']>1800:report.update(status='FAILED',deadline_exceeded=True)
        if not report['module_aliases_restored']:report['status']='FAILED'
    json.dumps(report,allow_nan=False)
    return report
