# -*- coding: utf-8 -*-
"""Own CPU-only random initialization; no SAR data, checkpoints or training."""
import argparse,gc,hashlib,importlib.util,json,os,sys,time
from pathlib import Path

def ref(p):
    p=Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())

def run(spec_path):
    started=time.monotonic();sp=Path(spec_path).resolve(strict=True)
    s=json.loads(sp.read_text(encoding='utf-8'))
    assert s['schema']=='ROOT-N20-source-only-repeat202-initialization-spec-v1'
    assert s['initialization_seed']==202610150202 and s['normal_RAW_count']==179 and s['total_failed_RAW_count']==26
    assert s['total_terminal_identity_pin_count']==205 and len(s['RAW_inputs'])==179
    assert s['entry_source']==ref(__file__) and s['deadline_seconds']==1800 and s['automatic_retry'] is False
    assert s['maximum_process_RSS_bytes']==16*1024**3 and os.environ['CUDA_VISIBLE_DEVICES']==''
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):assert os.environ[key]=='2'
    sources={v['path']:v for v in s['sources']}
    for v in sources.values():assert ref(v['path'])==v
    ad_path=Path(s['CPU_admission_file']);ad_ref=ref(ad_path);ad=json.loads(ad_path.read_text(encoding='utf-8'));spec_ref=ref(sp)
    assert ad['scientific_spec']==spec_ref and ad['entry']==s['entry_source']['path'] and ad['phase']==s['phase']
    assert ad['entry_args']==['--spec',str(sp),'--owner-execute'] and ad['allowed_cpus']==[0,1] and ad['cuda_visible_devices']==''
    assert ad['bound_source_sha256']==dict({v['path']:v['sha256'] for v in s['sources']},**{str(sp):spec_ref['sha256']})
    out=Path(s['output_directory']);assert out.is_absolute() and out.parent.is_dir() and not out.exists()
    import psutil
    proc=psutil.Process()
    def check():
        assert proc.cpu_affinity()==[0,1] and proc.memory_info().rss<=16*1024**3
        assert psutil.virtual_memory().available>=16*1024**3 and time.monotonic()-started<=1800
    check()
    def load(name,key):
        v=s['math_sources'][key];assert sources[v['path']]==v and ref(v['path'])==v
        d=importlib.util.spec_from_file_location(name,v['path']);m=importlib.util.module_from_spec(d);d.loader.exec_module(m);return m
    body=load('_N20_seed202_body','body');head=load('_N20_seed202_head','head')
    base=load('_N20_seed202_base','base');old=load('_N20_seed202_old','old');tail=load('_N20_seed202_digest','digest_helper')
    assert body.REPEAT_SEEDS==(202610150201,202610150202,202610150203) and tail.N20_TRAIN_SEED==202610150202
    tail.require=body.require
    import torch
    assert torch.__version__=='2.0.1+cu118' and not torch.cuda.is_initialized()
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    outside=torch.get_rng_state().clone()
    state=body.make_paired_state(torch,head,base,old,device='cpu',seed=202610150202)
    digest,tensors=tail.n20_training_state_digest(torch,state['heads']['LOCAL_ERROR'].state_dict())
    proofs=tail.n20_verify_training_initialization(torch,state,digest)
    assert digest!=s['first201_initialization_digest'] and torch.equal(outside,torch.get_rng_state()) and not torch.cuda.is_initialized()
    check()
    value=dict(schema='Root-N20-source-only-private-initialization-lock-v1',seed=202610150202,combined_digest=digest,proofs=proofs,
        paired_state_exact=True,paired_storage_independent=True,spatial_weights_zero=True,torch_version=torch.__version__,
        CPU_affinity=proc.cpu_affinity(),threads=2,CUDA_initialized=False,external_CPU_RNG_unchanged=True,training_executed=False,
        source_sha256={Path(v['path']).name:v['sha256'] for v in s['math_sources'].values()},
        source=s['entry_source'],spec=spec_ref,CPU_admission=ad_ref,phase=s['phase'],source_only=True,
        qualified=False,ROOT_actual_CPU0_verified=False,current_owned_terminal_self_certified=False)
    for v in sources.values():assert ref(v['path'])==v
    assert ref(sp)==spec_ref and ref(ad_path)==ad_ref and not torch.cuda.is_initialized()
    state['optimizers'].clear();state['heads'].clear();state.clear();del state;gc.collect();check()
    out.mkdir();target=out/'INITIALIZATION_LOCK.json'
    with target.open('x',encoding='utf-8') as stream:json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status='N20_REPEAT202_SOURCE_ONLY_INITIALIZATION_PENDING_ROOT_ACTUAL0',metadata=ref(target),digest=digest,qualified=False)),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--spec',required=True);a.add_argument('--owner-execute',action='store_true');v=a.parse_args();assert v.owner_execute;run(v.spec)
