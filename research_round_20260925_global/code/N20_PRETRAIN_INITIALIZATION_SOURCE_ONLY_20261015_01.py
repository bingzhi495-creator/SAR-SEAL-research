"""Source-only private initialization lock; no SAR inputs or trained weights."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='2'
import psutil
psutil.Process().cpu_affinity([0,1])
import hashlib
import importlib.util
import json
from pathlib import Path

root=Path(__file__).resolve().parent
def load(name,filename):
    path=root/filename
    definition=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(definition)
    definition.loader.exec_module(module)
    return module,path

body,bp=load('n20_init_pair_body','N20_PAIRED_LOCAL_CENTER_ERROR_TRAIN_BODY_UNBOUND_DRAFT_20261015_01.py')
head,hp=load('n20_init_head','N20_FEATURE_ADAPTER_UNBOUND_20261015_01.py')
base,fp=load('n20_init_original_factory','N16_SCOUT_MODULE_UNBOUND_DRAFT_20261014_01.py')
old,op=load('n20_init_original_helpers','N16_PAIRED_SCOUT_TRAIN_BODY_UNBOUND_DRAFT_20261014_02.py')
tail,tp=load('n20_init_state_digest','N20_PAIRED_TRAIN_GPU_RUN_TAIL_UNBOUND_DRAFT_20261015_01.py')
tail.require=body.require
import torch
assert not torch.cuda.is_initialized()
torch.set_num_threads(2);torch.set_num_interop_threads(2)
outside=torch.get_rng_state().clone()
state=body.make_paired_state(torch,head,base,old,device='cpu',seed=202610150201)
digest,tensors=tail.n20_training_state_digest(torch,state['heads']['LOCAL_ERROR'].state_dict())
proofs=tail.n20_verify_training_initialization(torch,state,digest)
assert torch.equal(outside,torch.get_rng_state()) and not torch.cuda.is_initialized()
payload=dict(schema='Root-N20-source-only-private-initialization-lock-v1',
    seed=202610150201,combined_digest=digest,proofs=proofs,
    paired_state_exact=True,paired_storage_independent=True,spatial_weights_zero=True,
    torch_version=torch.__version__,CPU_affinity=psutil.Process().cpu_affinity(),threads=2,
    CUDA_initialized=False,external_CPU_RNG_unchanged=True,training_executed=False,
    source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (bp,hp,fp,op,tp)})
target=root/'N20_PRETRAIN_INITIALIZATION_SOURCE_ONLY_RESULT_20261015_01.json'
with target.open('x',encoding='utf8') as stream:
    json.dump(payload,stream,indent=2,allow_nan=False)
print(json.dumps(dict(status='SOURCE_ONLY_INITIALIZATION_LOCK_PASS',seed=payload['seed'],
    digest=digest,target=str(target),CUDA_initialized=False)))
