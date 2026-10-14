"""SYNTHETIC_ONLY: one CPU GPU-entry interface test; no own Source/RAW/science run."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='2'
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
torch.set_num_threads(2);torch.set_num_interop_threads(2)
if torch.cuda.is_initialized():raise RuntimeError('Cold CUDA required')
ROOT=Path(__file__).resolve().parent
PUBLIC=ROOT
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
entry=load('_n17_GPU_entry_synthetic',ROOT/'N17_LOCAL_CENTER_ERROR_PAIRED_TRAIN_GPU_ROOT_UNBOUND_DRAFT_20261015_01.py')
kernel=load('_n17_module_synthetic',ROOT/'N17_NATIVE_L_NEIGHBORS_AND_SCOUT_INIT_UNBOUND_DRAFT_20261015_01.py')
body=load('_n17_train_interface_synthetic',ROOT/'N17_PAIRED_LOCAL_CENTER_ERROR_TRAIN_BODY_UNBOUND_DRAFT_20261015_01.py')
base=load('_n16_factory_interface_synthetic',PUBLIC/'N16_SCOUT_MODULE_UNBOUND_DRAFT_20261014_01.py')
old=load('_n16_training_interface_synthetic',PUBLIC/'N16_PAIRED_SCOUT_TRAIN_BODY_UNBOUND_DRAFT_20261014_02.py')
class TrainingEntryInterface(unittest.TestCase):
    def test_ROOT_locked_literal_field_ABI(self):
        with tempfile.TemporaryDirectory(prefix='SYNTHETIC_N17_LOCK_ABI_',dir=ROOT) as directory:
            root=Path(directory).resolve();self.assertEqual(root.parent,ROOT.resolve())
            refs={}
            for name in ('parent','cache','previous','init','base','module'):
                path=root/(name+'.json');path.write_text('{}',encoding='utf8');refs[name]=entry.reference(path)
            digest='5b9f58a3dece9720f84b257d03ae6928a7129e6f364e218f6bd49475d5fb7aa7'
            # Synthetic strict refs; literal lock key ABI supplied by ROOT, never actual lock/data.
            lock=dict(schema='N17-local-native-logit-error-paired-experiment-locked-v1',parent_protocol=refs['parent'],
                qualified_parent_cache=refs['cache'],qualified_previous_result=refs['previous'],previous_N16_decision='STOP',
                architecture=dict(first_seed=202610150171,repeat_seeds=[202610150172,202610150173],parameters=16803,layers=[521,32,3],tensors=4,
                    old_factory_SHA256=refs['base']['sha256'],neighbor_module_SHA256=refs['module']['sha256'],
                    initialization_metadata=refs['init'],initialization_digest_by_seed={str(202610150171):digest}),
                training=dict(epochs=24,updates_each_arm=17784,point_exposure_each_arm=72843264,order_seed=202610140160001,
                    AdamW_lr=3e-4,weight_decay=1e-4,betas=[.9,.999],eps=1e-8,AMP=False),
                inputs=dict(offsets=[list(v) for v in entry.OFFSETS],border='replicate',normalization='clip[-8,8]/8'))
            lockpath=root/'LOCK.json';lockpath.write_text(json.dumps(lock),encoding='utf8');lockref=entry.reference(lockpath)
            spec=dict(experiment_lock=lockref,expected_experiment_lock=lock,locked_protocol=refs['parent'],
                base_scout_source=refs['base'],scout_module_source=refs['module'])
            closure={ref['path']:ref for ref in list(refs.values())+[lockref]}
            self.assertEqual(entry.validate_experiment_lock(spec,closure),lock)
    def test_real_refs_point_binding_strict_loader_and_four_inputs(self):
        with tempfile.TemporaryDirectory(prefix='SYNTHETIC_N17_GPU_ENTRY_',dir=ROOT) as directory:
            root=Path(directory).resolve();self.assertEqual(root.parent,ROOT.resolve())
            native_path=root/'native_must_not_load.pth';native_path.write_bytes(b'NOT A TORCH PAYLOAD: THIN TRAINING MUST NOT READ THIS')
            native_ref=entry.reference(native_path)
            pointer=lambda key:dict(path=str(root/(key+'.json')),bytes=0,sha256='0'*64)
            report={key:pointer(key) for key in ('source','spec','GPU_admission','fixed_PR_checkpoint','teacher_checkpoint')};report['phase']='SYNTHETIC'
            identity=dict(dataset_index=0,image_id=7,original_shape=[2,3],ann_ids=[41,42])
            q=torch.arange(4096,dtype=torch.int64)%2;flat=torch.arange(4096,dtype=torch.int64)%6
            L=torch.linspace(-9,9,4096);labels=torch.arange(4096,dtype=torch.int64)%3
            packet=dict(report,dataset_index=0,identity=identity,protocol=pointer('old_lock'),natural_native_L_carrier=native_ref,
                sampled_F_unique=torch.arange(256,dtype=torch.float32).unsqueeze(0),native_e0=torch.zeros((1,100,256)),sample_to_unique=torch.zeros(4096,dtype=torch.int64),
                samples=dict(query_ids=q,flat_indices=flat,native_L=L,importance_weights=torch.ones(4096),error_labels=labels,utility_labels=labels,
                    delta=labels!=2,stratum_ids=torch.zeros(4096,dtype=torch.uint8)))
            packet_path=root/'thin.pth';torch.save(packet,packet_path);packet_ref=entry.reference(packet_path)
            npy_path=root/'real_features.npy';array=np.repeat(np.linspace(-1,1,4096,dtype=np.float32)[:,None],8,axis=1);np.save(npy_path,array,allow_pickle=False);npy_ref=entry.reference(npy_path)
            row=dict(dataset_index=0,image_id=7,original_shape=[2,3],packet_ref=packet_ref,native_L_ref=native_ref,neighbors8_ref=npy_ref,
                point_order_sha256=hashlib.sha256(q.numpy().tobytes()+flat.numpy().tobytes()).hexdigest())
            oldrow=dict(dataset_index=0,image_id=7,identity=identity,packet=packet_ref,natural_native_L_carrier=native_ref)
            closure={ref['path']:ref for ref in (packet_ref,native_ref,npy_ref)}
            reads=[]
            class TorchReader:
                def __getattr__(self,name):return getattr(torch,name)
                def load(self,path,**kwargs):reads.append(Path(path));return torch.load(path,**kwargs)
            loaded=entry.load_training_image(TorchReader(),np,0,row,oldrow,report,closure,packet['protocol'],lambda:None)
            self.assertEqual(reads,[packet_path]);self.assertEqual(set(loaded),{'packet','neighbors8'})
            self.assertTrue(np.array_equal(loaded['neighbors8'].view(np.uint32),array.view(np.uint32)))
            state=body.make_paired_state(torch,kernel,base.make_scout,old)
            digest,tensors=entry.state_digest(torch,state['heads']['LOCAL_ERROR'].state_dict())
            self.assertEqual(digest,'5b9f58a3dece9720f84b257d03ae6928a7129e6f364e218f6bd49475d5fb7aa7')
            gate=dict(initialization_metadata=dict(rows=[dict(seed=202610150171,combined_digest=digest,tensors=tensors,paired_state_exact=True,
                paired_storage_independent=True,new_columns_zero=True)]),experiment_lock=dict(architecture=dict(initialization_digest_by_seed={str(202610150171):digest})))
            self.assertEqual(set(entry.verify_initialization(torch,state,gate)),{'LOCAL_ERROR','CENTER_ERROR'})
            inputs=body._inputs(torch,loaded,'cpu',old,True)
            for arm in ('LOCAL_ERROR','CENTER_ERROR'):
                logits=state['heads'][arm](inputs['F'],inputs['e'],inputs['L'],inputs['extra8'][arm]);self.assertEqual(tuple(logits.shape),(4096,3))
                self.assertTrue(torch.equal(inputs['labels'][arm],labels))
            row['point_order_sha256']='0'*64
            with self.assertRaises(RuntimeError):entry.load_training_image(TorchReader(),np,0,row,oldrow,report,closure,packet['protocol'],lambda:None)
            with self.assertRaises(RuntimeError):entry.load(packet_path,'_malformed_Path_instead_of_ref')
            self.assertNotIn(native_path,reads)
if __name__=='__main__':unittest.main(verbosity=2)
