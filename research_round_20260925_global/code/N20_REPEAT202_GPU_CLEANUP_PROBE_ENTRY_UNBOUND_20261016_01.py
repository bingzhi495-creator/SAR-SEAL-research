# -*- coding: utf-8 -*-
"""Single synthetic allocator diagnostic. No SAR data, weights or AP."""
import argparse,gc,hashlib,importlib.util,json,os,sys,textwrap,time
from pathlib import Path
def ref(p):
    p=Path(p);v=p.read_bytes();return dict(path=str(p),bytes=len(v),sha256=hashlib.sha256(v).hexdigest())
def require(ok,message):
    if not ok:raise RuntimeError(message)
def run(spec_path):
    started=time.monotonic();path=Path(spec_path).resolve(strict=True);s=json.loads(path.read_text(encoding='utf-8'))
    assert s['schema']=='ROOT-N20-repeat202-synthetic-GPU-cleanup-probe-spec-v1' and s['normal_RAW_count']==180
    assert s['total_failed_RAW_count']==26 and s['total_terminal_identity_pin_count']==206 and len(s['RAW_inputs'])==180
    assert s['entry_source']==ref(__file__) and s['initialization_seed']==202610150202 and s['automatic_retry'] is False
    assert os.environ['CUDA_VISIBLE_DEVICES']=='0'
    for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):assert os.environ[k]=='2'
    sources={v['path']:v for v in s['sources']}
    for v in sources.values():assert ref(v['path'])==v
    ad_path=Path(s['GPU_admission_file']);ad_ref=ref(ad_path);ad=json.loads(ad_path.read_text(encoding='utf-8'));sp_ref=ref(path)
    assert ad['scientific_spec']==sp_ref and ad['phase']==s['phase'] and ad['entry']==s['entry_source']['path']
    assert ad['bound_source_sha256']==dict({v['path']:v['sha256'] for v in s['sources']},**{str(path):sp_ref['sha256']})
    out=Path(s['output_directory']);assert out.parent.is_dir() and not out.exists()
    import psutil
    process=psutil.Process()
    def check():
        assert process.cpu_affinity()==[0,1] and process.memory_info().rss<=16*1024**3
        assert psutil.virtual_memory().available>=16*1024**3 and time.monotonic()-started<=1800
    check()
    def load(key):
        r=s['math_sources'][key];assert sources[r['path']]==r;d=importlib.util.spec_from_file_location('_N20_probe_'+key,r['path']);m=importlib.util.module_from_spec(d);d.loader.exec_module(m);return m
    body,head,base,old=(load(k) for k in ('body','head','base','old'))
    import torch
    assert torch.__version__=='2.0.1+cu118' and not torch.cuda.is_initialized()
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    assert torch.cuda.is_available() and callable(getattr(torch._C,'_cuda_clearCublasWorkspaces',None))
    torch.cuda.set_device(0);total=torch.cuda.get_device_properties(0).total_memory;assert total>=4096*1024**2
    torch.cuda.set_per_process_memory_fraction(4096*1024**2/total,0);torch.cuda.reset_peak_memory_stats(0)
    state=body.make_paired_state(torch,head,base,old,device='cuda:0',seed=202610150202)
    def exercise(state):
        rng=torch.Generator(device='cpu');rng.manual_seed(202610150202)
        center=torch.randn((4096,256),generator=rng).cuda();emb=torch.randn((4096,256),generator=rng).cuda()
        logits_native=torch.randn((4096,),generator=rng).cuda();neighbors=torch.randn((4096,9,256),generator=rng).cuda()
        labels=(torch.arange(4096,dtype=torch.int64)%3).cuda();values={}
        for step in range(8):
            for arm in body.ARMS:
                optimizer=state['optimizers'][arm];optimizer.zero_grad(set_to_none=True)
                local=neighbors if arm=='LOCAL_ERROR' else center[:,None,:].repeat(1,9,1)
                logits=state['heads'][arm](center,emb,logits_native,local)
                loss=torch.nn.functional.cross_entropy(logits,labels);assert bool(torch.isfinite(loss))
                loss.backward();optimizer.step();values[arm]=float(loss.detach().cpu())
        return values
    losses=exercise(state);check()
    state['history']=[dict(synthetic=True,steps_per_arm=8,losses=losses)];state['epoch_orders']=[[0]]
    before_history=json.loads(json.dumps(state['history']));before_orders=json.loads(json.dumps(state['epoch_orders']))
    tail=s['new_tail_source'];assert sources[tail['path']]==tail and ref(tail['path'])==tail
    text=Path(tail['path']).read_text(encoding='utf-8');start=text.index('    report_history = json.loads(');end=text.index("    write(output / 'REPORT.json'",start)
    fragment=textwrap.dedent(text[start:end]);function='def cleanup_probe(state):\n'+textwrap.indent(fragment,'    ')+'    return cleanup, report_history, report_epoch_orders\n'
    namespace=dict(torch=torch,gc=gc,json=json,require=require,check=check,N20_TRAIN_SEED=202610150202)
    exec(compile(function,tail['path']+'::exact_cleanup_block','exec'),namespace)
    cleanup,history,orders=namespace['cleanup_probe'](state)
    assert not state and cleanup['allocated_after']==cleanup['reserved_after']==0
    assert history==before_history and orders==before_orders;check()
    for v in sources.values():assert ref(v['path'])==v
    assert ref(path)==sp_ref and ref(ad_path)==ad_ref
    result=dict(schema='N20-repeat202-synthetic-GPU-cleanup-probe-report-v1',source=s['entry_source'],spec=sp_ref,GPU_admission=ad_ref,
        phase=s['phase'],new_tail_source=tail,cleanup=cleanup,synthetic_steps_per_arm=8,points_per_step=4096,
        actual_head_tensors=5,actual_head_parameters=18851,history_and_orders_preserved=True,
        SAR_data_read=False,SAR_training_executed=False,AP_evaluated=False,original_failure24_cause_established=False,
        qualified=False,ROOT_actual_GPU0_verified=False,current_owned_terminal_self_certified=False,
        elapsed_seconds=time.monotonic()-started)
    out.mkdir();target=out/'REPORT.json'
    with target.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status='N20_SYNTHETIC_CLEANUP_PROBE_PENDING_ROOT_ACTUAL0',report=ref(target),qualified=False)),flush=True)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--spec',required=True);a.add_argument('--owner-execute',action='store_true');v=a.parse_args();assert v.owner_execute;run(v.spec)
