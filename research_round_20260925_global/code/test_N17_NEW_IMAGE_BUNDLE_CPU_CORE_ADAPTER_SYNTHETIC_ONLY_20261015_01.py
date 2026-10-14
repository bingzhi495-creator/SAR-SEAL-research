"""SYNTHETIC_ONLY: one tiny complete schema fixture, fake CUDA metadata is not science."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='2'
import copy
import importlib.util
from pathlib import Path
import unittest
import numpy as np
import torch
from scipy import ndimage
from pycocotools import mask as masks
torch.set_num_threads(2);torch.set_num_interop_threads(2)
if torch.cuda.is_initialized():raise RuntimeError('Cold CUDA required')
ROOT=Path('E:/SAR/seal/tmp');PUBLIC=Path('E:/SAR/seal/github_release_20261011/research_round_20260925_global/code')
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
adapter=load('_n17_adapter_synthetic',ROOT/'N17_NEW_IMAGE_BUNDLE_CPU_CORE_ADAPTER_UNBOUND_DRAFT_20261015_01.py')
writer=load('_n17_writer_API_synthetic',ROOT/'N17_FIXEDDEV_LOCAL_CENTER_ERROR_GPU_SHARD_ROOT_UNBOUND_DRAFT_20261015_01.py')
body=load('_n17_eval_adapter_synthetic',ROOT/'N17_ALL100_LOCAL_CENTER_ERROR_SELECTOR_EVAL_BODY_UNBOUND_DRAFT_20261015_01.py')
kernel=load('_n17_kernel_adapter_synthetic',ROOT/'N17_NATIVE_L_NEIGHBORS_AND_SCOUT_INIT_UNBOUND_DRAFT_20261015_01.py')
base=load('_n16_base_adapter_synthetic',PUBLIC/'N16_SCOUT_MODULE_UNBOUND_DRAFT_20261014_01.py')
legacy=load('_n16_eval_adapter_synthetic',PUBLIC/'N16_ALL100_SCOUT_SELECTOR_EVAL_BODY_UNBOUND_DRAFT_20261014_01.py')
helper=load('_n16_score_adapter_synthetic',PUBLIC/'N16_SCOUT_ACTION_LABELS_AND_SELECTION_UNBOUND_20261014_01.py')
def encode(mask):
    value=masks.encode(np.asfortranarray(mask.astype(np.uint8)));return dict(size=value['size'],counts=value['counts'].decode('ascii'))
class PR(torch.nn.Module):
    def __init__(self):super().__init__();self.value=torch.nn.Parameter(torch.tensor(.25))
    def forward(self,F,e,L,D,arm='PR'):return dict(residual=torch.full_like(L,float(self.value)))
class AdapterTiny(unittest.TestCase):
    def test_complete100_schema_decoder_hooks_order_and_rejections(self):
        ref=lambda name:dict(path=str(ROOT/(name+'.pth')),bytes=1,sha256='1'*64)
        oldref,newref=ref('SYNTHETIC_OLD'),ref('SYNTHETIC_NEW');oldbinding=dict(phase='SYNTHETIC_OLD',source=ref('OLD_SOURCE'));binding=dict(phase='SYNTHETIC_NEW',source=ref('NEW_SOURCE'))
        carrier=dict(bundle=oldref,shared_member='i000/SHARED_ACTUAL_INPUTS.pth',original_GPU_binding=oldbinding)
        L=torch.full((100,3,4),-9.,dtype=torch.float32);L[0]=torch.tensor([[-1.,0.,3.,9.],[-2.,2.,-9.,-3.],[.5,-.5,4.,-4.]])
        F=torch.arange(12*256,dtype=torch.float32).reshape(12,256)/1000;e=torch.zeros((100,256));union=torch.arange(12,dtype=torch.int64)
        domains=[legacy.domains_from_native(np,ndimage,L[q].numpy()) for q in range(100)];identity=dict(image_id=9,original_shape=[3,4]);qids=list(reversed(range(100)))
        shared=dict(schema='N16-actual-GPU-image-shared-inputs-v1',dataset_index=0,image_id=9,identity=identity,natural_query_ids=list(range(100)),native_export_query_ids=qids,
            native_L=L,e0=e,union_flat_indices=union,F_unique=F,U=12,C_indices=[torch.from_numpy(d[3]) for d in domains],C_counts=[len(d[3]) for d in domains],K_counts=[d[4] for d in domains],actual_GPU_F_captured=True,CPU_F_rerendered=False)
        native=[dict(image_id=9,query_id=q,category_id=1,score=float(q)/100,segmentation=encode(L[q].numpy()>0)) for q in qids];byq={p['query_id']:p for p in native}
        heads={}
        for epoch in (24,8):
            for arm in ('LOCAL_ERROR','CENTER_ERROR'):
                head=kernel.make_scout(torch,202610150171,base.make_scout)
                with torch.no_grad():
                    for p in head.parameters():p.zero_()
                heads[(epoch,arm)]=head.requires_grad_(False).eval()
        pr=PR().requires_grad_(False).eval();members={};registry={};roles={};query_profiles=[]
        def put(name,value):
            members[name]=value;registry[name]=dict(schema=value['schema'],query_id=value['query_id'],stage=value.get('stage','profile'),scout_epoch=value.get('scout_epoch',value.get('epoch')),
                control=value.get('control',value.get('arm')),evalmode=value.get('evalmode',value.get('mode')),chunk_first=value.get('chunk_first'),raw_tensor_payload_bytes=writer.tensor_bytes(torch,value));return name
        def capture(event):
            value={k:(v.detach().cpu().clone() if isinstance(v,torch.Tensor) else v) for k,v in event.items() if k not in ('sampled_F','e0','native_L')}
            value.update(schema='N17-actual-GPU-selector-point-chunk-v1',actual_device_type='cuda',synthetic_only=False,actual_observed_F_e_L_bitexact_to_input_carrier=True)
            name=put(writer.new_point_member_id(0,value),value);roles.setdefault((value['query_id'],value['scout_epoch'],value['control'],value['evalmode'],value['stage']),[]).append(name)
        for result in body.evaluate_all100(torch,np,ndimage,helper,legacy,kernel,L,e,union,F,heads,pr,point_callback=capture,synthetic_only=True):
            q=result['query_id'];profiles={}
            for name,item in result['profiles'].items():
                epoch,arm,mode=item['epoch'],item['arm'],item['mode']
                value=dict(schema='N17-actual-GPU-selector-query-profile-v1',query_id=q,epoch=epoch,arm=arm,mode=mode,K=result['K'],
                    C_scout_chunks=roles.get((q,epoch,arm,'shared-atmost-exact','scout_C_logits'),[]),common_C_PR_chunks=roles.get((q,None,'COMMON_C_REFERENCE','shared','full_C_PR'),[]),
                    selected_PR_chunks=roles.get((q,epoch,arm,mode,'selected_PR'),[]),actual_selected_flat_indices=torch.from_numpy(item['selected_flat_indices'].copy()),
                    actual_mask_packbits=torch.from_numpy(np.packbits(item['mask'].ravel(),bitorder='little')),actual_mask_shape=[3,4],actual_mask_bitorder='little',actual_RLE=encode(item['mask']),
                    original_prediction_nonsegmentation_fields={k:v for k,v in byq[q].items() if k!='segmentation'},PR_compatibility=item['PR_compatibility'])
                profiles[name]=put(writer.new_profile_member_id(0,q,epoch,arm,mode),value)
            query_profiles.append(dict(query_id=q,profiles=profiles,K=result['K'],C_count=len(result['C_flat_indices'])))
        bundle=dict(schema='N17-actual-GPU-selector-image-bundle-v1',binding=binding,input_carrier=carrier,dataset_index=0,image_id=9,members=members,member_registry=registry,
            member_count=len(members),raw_tensor_payload_bytes=sum(v['raw_tensor_payload_bytes'] for v in registry.values()))
        row=dict(dataset_index=0,image_id=9,identity=identity,qids=qids,input_carrier=carrier,bundle=newref,member_locators=list(members),query_profiles=query_profiles,natural_queries=100)
        def decode(bundle=bundle,row=row):return adapter.decode_image(torch,np,shared,bundle,row,binding,carrier,native)
        decoder=decode()
        with self.assertRaises(ValueError):decoder.require_all_consumed()
        for q in range(100):
            args=decoder.query(q)
            self.assertEqual(args['query_id'],q)
            for epoch,arm in heads:
                for item in args['iter_scout_chunks'](epoch,arm):
                    flat,f,emb,nativeL,extra,logits=item;self.assertTrue(torch.equal(f,F[flat]));self.assertEqual(tuple(extra.shape),(len(flat),8))
                for mode in ('atmost','exact'):
                    selected,blocks,mask,actualRLE=args['read_profile'](epoch,arm,mode);self.assertEqual(tuple(mask.shape),(3,4));self.assertEqual(actualRLE,encode(mask.numpy()))
        self.assertTrue(decoder.require_all_consumed()['all_registered_members_consumed'])
        predictions=decoder.predictions('E24_LOCAL_ERROR_atmost');self.assertEqual([p['query_id'] for p in predictions],qids)
        for expected,actual in zip(native,predictions):self.assertEqual({k:v for k,v in expected.items() if k!='segmentation'},{k:v for k,v in actual.items() if k!='segmentation'})
        first_profile=next(v for v in members.values() if v['schema']=='N17-actual-GPU-selector-query-profile-v1')
        profile_id=writer.new_profile_member_id(0,first_profile['query_id'],first_profile['epoch'],first_profile['arm'],first_profile['mode'])
        bad=copy.deepcopy(bundle);bad['members'][profile_id]['actual_mask_bitorder']='big'
        with self.assertRaises(ValueError):decode(bundle=bad)
        bad=copy.deepcopy(row);bad['member_locators']=list(reversed(bad['member_locators']))
        with self.assertRaises(ValueError):decode(row=bad)
        bad=copy.deepcopy(bundle);del bad['members'][profile_id]
        with self.assertRaises(ValueError):decode(bundle=bad)
        bad=copy.deepcopy(bundle);point=next(v for v in bad['members'].values() if v['schema']=='N17-actual-GPU-selector-point-chunk-v1');point['control']='UTILITY'
        with self.assertRaises(ValueError):decode(bundle=bad)
if __name__=='__main__':unittest.main(verbosity=2)
