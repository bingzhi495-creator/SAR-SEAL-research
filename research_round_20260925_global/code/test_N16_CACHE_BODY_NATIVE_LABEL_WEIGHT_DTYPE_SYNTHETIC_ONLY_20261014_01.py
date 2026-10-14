import os,sys,types
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='2'
import torch
from pathlib import Path
torch.set_num_threads(2);torch.set_num_interop_threads(2)
require=lambda v: None if v else (_ for _ in ()).throw(AssertionError('native fixture contract'))
p=Path(sys.argv[1]);m=types.ModuleType('_N16_dtype_fixture');m.__file__=str(p)
exec(compile(p.read_bytes(),str(p),'exec'),m.__dict__)
labels=torch.ones(100,dtype=torch.int64);labels[:2]=0
lw=labels.new_ones((100,));mw=torch.zeros(100,dtype=torch.float32);mw[:2]=1
obs=dict(labels=labels,label_weights=lw,mask_weights=mw,positive_query_indices=[0,1],negative_query_indices=list(range(2,100)),assigned_GT_indices=[0,1])
a=m.final_assignment(torch,obs,[0,1],[101,102],1)
require(a['raw_GT_index_by_query'].tolist()==[0,1]+[-1]*98)
obs['label_weights']=lw.float()
try:m.final_assignment(torch,obs,[0,1],[101,102],1)
except RuntimeError:pass
else:raise AssertionError('native label weights must not be cast FP32')
require(not torch.cuda.is_initialized())
print('N16_NATIVE_INT64_CLASS_WEIGHT_AND_FP32_MASK_WEIGHT_FIXTURE_GREEN')
