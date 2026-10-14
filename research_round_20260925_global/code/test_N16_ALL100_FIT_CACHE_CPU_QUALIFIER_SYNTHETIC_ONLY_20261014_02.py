"""SYNTHETIC_ONLY: hand-made 2x3 FIT packets; no SAR IO/cache/weights."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='2'
import copy
import importlib.util
from pathlib import Path
import unittest
import numpy as np
import cv2
import torch

torch.set_num_threads(2)
torch.set_num_interop_threads(2)
if torch.__version__!='2.0.1+cu118' or torch.cuda.is_initialized():
    raise RuntimeError('Qualified seal_gpu CPU2/coldCUDA runtime required')
path=Path(__file__).with_name('N16_ALL100_FIT_CACHE_CPU_QUALIFIER_BODY_UNBOUND_DRAFT_20261014_02.py')
spec=importlib.util.spec_from_file_location('_n16_synthetic_cpu_body',path)
body=importlib.util.module_from_spec(spec)
spec.loader.exec_module(body)


class SyntheticPR(torch.nn.Module):
    """Only a test stub: eight synthetic parameters, deterministic raw u."""
    def __init__(self):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(7)
            self.hidden1=torch.nn.Linear(257,91)
            self.hidden2=torch.nn.Linear(92,91)
            self.hidden3=torch.nn.Linear(92,91)
            self.final=torch.nn.Linear(92,1)
        self.requires_grad_(False)
        self.eval()

    def forward(self,F,e,L,neighbor,arm):
        if arm!='PR' or neighbor.shape!=(len(L),49):
            raise RuntimeError('Synthetic PR stub call differs')
        return {'residual':F[:,0]/4}


def explicit_stub(torch_module,state,F,L):
    # This is NOT the real PR arithmetic; it isolates validator input routing.
    u=F[:,0]*.25
    return u,L+u


def fixture():
    raw=torch.tensor([[[1,1,1],[0,0,0]],[[0,0,0],[1,1,0]]],dtype=torch.bool)
    L=torch.zeros(100,2,3)
    L[0]=torch.tensor([[-1,.5,3],[-3,0,1]])
    L[1]=torch.tensor([[-3,-1,0],[1,3,-3]])
    L[3]=3  # Physically nonempty C but K=0: must never be sampled.
    e=torch.zeros(1,100,256)
    labels=torch.ones(100,dtype=torch.int64)
    labels[:2]=0
    lw=torch.zeros(100,dtype=torch.int64)
    lw[:4]=1
    mw=torch.zeros(100)
    mw[:2]=1
    observations=[]
    matches=[]
    for _ in range(10):
        obs=dict(labels=labels.clone(),label_weights=lw.clone(),mask_weights=mw.clone(),
            positive_query_indices=[0,1],negative_query_indices=[2,3],assigned_GT_indices=[0,1],
            sampler_positive_query_indices=[0,1],sampler_negative_query_indices=[2,3],
            mask_targets_digest={'shape':[2,2,3],'dtype':'bool','sha256':'synthetic-processed-only'})
        observations.append(obs)
        matches.append(dict(positive_query_indices=[0,1],assigned_GT_indices=[0,1],positives=2,
            target_labels=body.digest(labels)))
    assignment=dict(natural_query_ids=list(range(100)),
        raw_GT_index_by_query=torch.tensor([0,1,-1,-1]+[-2]*96),
        raw_GT_annotation_id_by_query=[101,102]+[None]*98,
        supervision_kind_by_query=['matched','matched','background','background']+['ignored']*96,
        valid_targets=torch.tensor([True]*4+[False]*96),
        native_positive_query_indices=[0,1],native_negative_query_indices=[2,3],
        native_ignored_query_indices=list(range(4,100)),
        final_native_labels=labels.clone(),final_native_label_weights=lw.clone(),final_native_mask_weights=mw.clone(),
        background_empty_set_is_new_supervision_extension=True,
        original_native_negative_mask_supervised=False,ignore_never_zero_filled=True)
    seed=19001
    rng=torch.Generator(device='cpu').manual_seed(seed)
    before=rng.get_state().clone()
    qs=[]
    fs=[]
    ws=[]
    strata=[]
    # Hand-built 2x3 domains are all six pixels for the active queries.
    # Independent fixture generation; never imports or calls the cache writer.
    for group,queries,weight in ((0,[0,1],4/3),(1,[2],2/3)):
        draws=torch.randint(len(queries),(2048,),generator=rng).tolist()
        for ordinal in draws:
            qs.append(queries[ordinal])
            fs.append(int(torch.randint(6,(1,),generator=rng)))
            ws.append(weight)
            strata.append(group)
    q=torch.tensor(qs)
    flat=torch.tensor(fs)
    unique,inverse=torch.unique(flat,sorted=True,return_inverse=True)
    F=torch.zeros(6,256)
    F[:,0]=torch.tensor([8,6,-16,20,6,-8])
    native=L.flatten(1)[q,flat]
    u=F[flat,0]*.25
    combined=native+u
    target=torch.zeros(4096,dtype=torch.bool)
    for i,(query,pixel) in enumerate(zip(qs,fs)):
        if query<2:
            target[i]=raw[query].flatten()[pixel]
    # Labels are enumerated by hand, independently of the validator predicates.
    utility_table=torch.tensor([[0,2,1,1,1,0],[2,1,2,2,2,2],[1,1,2,1,1,2]])
    error_table=torch.tensor([[1,2,2,2,2,0],[2,2,2,2,2,2],[2,2,2,2,2,2]])
    plan={
        'matched':dict(query_ids=[0,1],n_stratum=2,n_active=3,p_s=.5,draws=2048,weight=4/3),
        'background':dict(query_ids=[2],n_stratum=1,n_active=3,p_s=.5,draws=2048,weight=2/3)}
    sample=dict(query_ids=q,flat_indices=flat,importance_weights=torch.tensor(ws),
        stratum_ids=torch.tensor(strata,dtype=torch.uint8),sampler_state_before=before,
        sampler_state_after=rng.get_state().clone(),plan=plan,seed=seed,replacement=True,
        duplicates_preserved=True,draw_order='matched then background; vector query draw then sequential point draws',
        native_L=native,actual_PR_residual=u,actual_combined=combined,native_P=native>0,
        actual_after_A=combined>0,targets=target,utility_labels=utility_table[q,flat],error_labels=error_table[q,flat],
        delta=(native>0)!=(combined>0),error=(native>0)!=target)
    maximum=torch.zeros(100)
    maximum[0]=.5
    maximum[1]=1
    maximum[3]=.5
    packet=dict(schema=body.PACKET_SCHEMA,dataset_index=0,
        identity=dict(dataset_index=0,image_id=999,original_shape=[2,3],ann_ids=[101,102]),
        natural_query_ids=list(range(100)),matching_loss_seed=18001,
        natural_native_L=L,native_e0=e,native_class_logits=torch.zeros(1,100,2),raw_GT_ids=[101,102],
        raw_GT_digest=body.digest(raw),raw_target_mapping=[0,1],all_ten_target_observations=observations,
        native_stage_matches=matches,assignment=assignment,max_native_mask_IoU_to_any_GT=maximum,
        max_IoU_is_FIT_diagnostic_only=True,C_counts_all100=[6]*100,K_counts_all100=[4,3,6,0]+[6]*96,
        active_queries=[0,1,2],active_matched_queries=[0,1],active_background_queries=[2],samples=sample,
        unique_flat_indices=unique,sample_to_unique=inverse,sampled_F_unique=F[unique],
        actual_raw_processed_GT_correspondence={'synthetic_only':True},
        processed_target_proof={'synthetic_only':True},original_cache_packet={'synthetic_only':True},
        new_background_empty_set_supervision=True,candidate_GT_independent=True,
        original_matcher_negative_mask_supervision=False,paired_scout_same_samples=True,
        fixed_refiner_has_seen_FIT=True,PR_label_generation_chunk_calls=2,actual_label_generation_points=4096,
        helper_classes=dict(utility=['help','harm','noop'],error=['FP','FN','correct']))
    recipe=dict(point_chunk=2048,points_two_groups_each=2048,points_single_group=4096,num_classes=1,
        PR_atol=1e-4,PR_rtol=0,background_empty_set_extension=True,carrier_epoch=24,
        carrier_seed=202610120701,point_seed_base=19001,matching_loss_seed_base=18001)
    head=SyntheticPR()
    return packet,recipe,head.state_dict(),{'original_head':head,'independent_PR':explicit_stub},raw


def validate(args):
    return body.validate_packet(torch,np,cv2,*args)


class QualifierTests(unittest.TestCase):
    def test_hand_packet_passes_and_preserves_rng_with_real_bg_extension(self):
        args=fixture()
        before=torch.get_rng_state().clone()
        row=validate(args)
        self.assertEqual(row['points'],4096)
        self.assertEqual(row['active_background_queries'],1)
        self.assertEqual(row['native_background_queries'],2)
        self.assertEqual(row['native_background_max_IoU_ge_point5'],1)
        self.assertEqual(row['PR_errors'],dict.fromkeys(row['PR_errors'],0.))
        self.assertTrue(all(n>0 for n in row['utility_class_counts']))
        self.assertTrue(all(n>0 for n in row['error_class_counts']))
        self.assertFalse(row['processed_correspondence_independently_proved_by_this_body'])
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        self.assertFalse(torch.cuda.is_initialized())

    def test_wrong_importance_query_and_pixel_are_rejected_at_sampler(self):
        for field,new in (('importance_weights',9.),('query_ids',99),('flat_indices',5)):
            with self.subTest(field=field):
                args=fixture()
                tensor=args[0]['samples'][field]
                tensor[0]=new if tensor[0]!=new else 4
                with self.assertRaisesRegex(RuntimeError,'Private sampler/importance exact replay differs'):
                    validate(args)

    def test_native_negative_mask_supervision_and_invalid_target_are_rejected(self):
        args=fixture()
        args[0]['all_ten_target_observations'][-1]['mask_weights'][2]=1
        with self.assertRaisesRegex(RuntimeError,'foreground/background target policy differs'):
            validate(args)
        args=fixture()
        background=int((args[0]['samples']['query_ids']==2).nonzero()[0])
        args[0]['samples']['targets'][background]=True
        with self.assertRaisesRegex(RuntimeError,'target differs from actual rawGT'):
            validate(args)
        args=fixture()
        args[0]['assignment']['raw_GT_index_by_query'][4]=-1
        with self.assertRaisesRegex(RuntimeError,'Background empty-set must be explicit'):
            validate(args)

    def test_incorrect_labels_and_even_near_zero_sign_are_rejected(self):
        args=fixture()
        args[0]['samples']['utility_labels'][0]=(int(args[0]['samples']['utility_labels'][0])+1)%3
        with self.assertRaisesRegex(RuntimeError,'class/bit differs: utility_labels'):
            validate(args)
        args=fixture()
        args[0]['samples']['error_labels'][0]=(int(args[0]['samples']['error_labels'][0])+1)%3
        with self.assertRaisesRegex(RuntimeError,'class/bit differs: error_labels'):
            validate(args)
        # Absolute closeness alone cannot stand in for the locked strict sign.
        with self.assertRaisesRegex(RuntimeError,'Strict PR final sign failed'):
            body.absolute(torch,torch.tensor([1e-7]),torch.tensor([-1e-7]),'synthetic near-zero',sign=True)
        args=fixture()
        args[0]['samples']['actual_combined'][0]*=-1
        with self.assertRaisesRegex(RuntimeError,'PR compatibility failed'):
            validate(args)


if __name__=='__main__':
    unittest.main(verbosity=2)
