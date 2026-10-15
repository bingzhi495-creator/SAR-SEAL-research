# UNBOUND_DRAFT / SYNTHETIC_ONLY / NOT_EXECUTED: one public proof_query integration.
# Every field called GPU below is a synthetic CPU fixture, never an observed GPU artifact.
import importlib.util
from pathlib import Path
import unittest

def ordinary(name,filename):
    path=Path(__file__).with_name(filename)
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

core=ordinary('_N20_synthetic_core','N20_SELECTED_NUMERIC_CPU_PROOF_BODY_UNBOUND_DRAFT_20261015_01.py')
endpoint=ordinary('_N20_synthetic_endpoint','N20_PAIRED_FEATURE_ERROR_ENDPOINT_CPU_QUALIFIER_BODY_UNBOUND_DRAFT_20261015_01.py')
adapter=ordinary('_N20_synthetic_head','N20_FEATURE_ADAPTER_UNBOUND_20261015_01.py')
legacy=ordinary('_N16_synthetic_PR_kernel','N16_ALL100_SCOUT_SELECTOR_CPU_PROOF_BODY_UNBOUND_DRAFT_20261014_01.py')
helper=ordinary('_N16_synthetic_FP64_helper','N16_SCOUT_ACTION_LABELS_AND_SELECTION_UNBOUND_20261014_01.py')

class PublicNumericProof(unittest.TestCase):
    def test_all_eight_profiles_and_saved_score_tamper(self):
        import torch
        import numpy as np
        import cv2
        from pycocotools import mask as masks
        self.assertFalse(torch.cuda.is_initialized())
        torch.set_num_threads(2);torch.set_num_interop_threads(2);cv2.setNumThreads(2)
        q=7
        # Positive native pixels make radius8 dilation cover the whole2x3 grid.
        # Omega is exactly flat1,2, hence C=[0..5] and K=2, calculated by hand.
        L=torch.tensor([[-4.,-1.,1.],[3.,-3.,4.]],dtype=torch.float32)
        grid=torch.full((100,2,3),-4.,dtype=torch.float32);grid[q]=L
        flat=torch.arange(6,dtype=torch.int64);e=torch.arange(256,dtype=torch.float32)
        F=torch.zeros((6,256),dtype=torch.float32);F[:,0]=1.;F[:,1]=-1.
        mapping=torch.tensor([[0,0,1,0,0,1,3,3,4],[0,1,2,0,1,2,3,4,5],
            [1,2,2,1,2,2,4,5,5],[0,0,1,3,3,4,3,3,4],
            [0,1,2,3,4,5,3,4,5],[1,2,2,4,5,5,4,5,5]],dtype=torch.int64)
        context=dict(union_flat_indices=flat,F_unique=F,extra_flat_indices=torch.empty(0,dtype=torch.int64),
            extra_F=torch.empty((0,256),dtype=torch.float32),neighbor_to_combined=mapping)
        heads={};states={};scouts={};scores={}
        for epoch in (24,8):
            for arm in ('LOCAL_ERROR','CENTER_ERROR'):
                head=adapter.create_head(torch)
                with torch.no_grad():
                    for value in head.parameters():value.zero_()
                    head.spatial_weight[0,0]=.5
                    head.hidden.weight[0,0]=.001
                    head.hidden.weight[0,512]=1.
                    head.hidden.bias[0]=1.
                    head.output.weight[0,0]=head.output.weight[1,0]=1.
                head.eval().requires_grad_(False);key=(epoch,arm);heads[key]=head
                states[key]={name:value.detach().clone() for name,value in head.state_dict().items()}
                neighbors=F[mapping] if arm=='LOCAL_ERROR' else F[:,None,:].repeat(1,9,1)
                with torch.no_grad():synthetic_GPU_logits=head(F,e[None].expand(6,256),L.reshape(-1),neighbors)
                scouts[key]=(flat.clone(),F.clone(),e.clone(),L.reshape(-1).clone(),neighbors.clone(),synthetic_GPU_logits.clone())
                scores[key]=torch.from_numpy(helper.error_score(synthetic_GPU_logits.numpy()).copy())
                self.assertGreater(float(head.spatial_weight.abs().sum()),0.)
        # All F are identical, so the normalized F term is a positive constant.
        # Logits=(t,t,0), t=1+L/8+positive_constant: every score>0 and increases
        # strictly with L. Thus BOTH modes choose the two largest L: flat3,5.
        chosen=torch.tensor([3,5],dtype=torch.int64)
        hand_mask=np.array([[False,False,True],[False,False,False]],dtype=np.bool_)
        # Synthetic PR u=-4 removes the selected native positives3,5; flat5 has
        # L+u==0, exercising strict>0. Unselected native positive flat2 stays.
        def block(indices):
            raw=L.reshape(-1)[indices].clone();u=torch.full_like(raw,-4.)
            return dict(flat_indices=indices.clone(),sampled_F=F[indices].clone(),e0=e.clone(),native_L=raw,
                GPU_raw_residual=u,GPU_combined=raw+u)
        def original_head(f,embedding,raw):
            self.assertEqual(tuple(embedding.shape),(len(raw),256))
            u=torch.full_like(raw,-4.);return u,raw+u
        def independent_PR(f,embedding,raw):
            u=torch.zeros_like(raw)-4.;return u,raw-4.
        def encode(mask):
            value=masks.encode(np.asfortranarray(mask.astype(np.uint8)))
            return dict(size=[int(v) for v in value['size']],counts=value['counts'].decode('ascii'))
        expected_rle=encode(hand_mask)
        def iter_scout_chunks(epoch,arm):
            yield tuple(value.clone() for value in scouts[(epoch,arm)])
        def read_scores(epoch,arm):return scores[(epoch,arm)].clone()
        def read_profile(epoch,arm,mode):
            self.assertIn(mode,('atmost','exact'))
            return chosen.clone(),[block(chosen)],torch.from_numpy(hand_mask.copy()),dict(expected_rle)
        kwargs=dict(query_id=q,native_L_grid=grid,e0=e,C_flat_indices=flat,K=2,feature_context=context,
            heads=heads,states=states,common_PR_chunks=[block(flat)],iter_scout_chunks=iter_scout_chunks,
            read_scores=read_scores,read_profile=read_profile,
            formulas=dict(original_head=original_head,independent_PR=independent_PR),rle_encoder=encode)
        proof=core.proof_query(torch,np,cv2,helper,legacy,endpoint,**kwargs)
        expected_names={'E%d_%s_%s'%(ep,arm,mode) for ep in (24,8)
            for arm in ('LOCAL_ERROR','CENTER_ERROR') for mode in ('atmost','exact')}
        self.assertEqual((proof['query_id'],proof['C_points'],proof['K']),(q,6,2))
        self.assertEqual(set(proof['profiles']),expected_names)
        self.assertTrue(proof['actual_saved_FP64_scores_canonical_exact'])
        self.assertFalse(proof['CPU_scout_outputs_used_for_ranking'])
        self.assertFalse(proof['ROOT_scientific_qualification_conferred'])
        for value in proof['profiles'].values():
            self.assertEqual(value['selected_points'],2)
            self.assertTrue(np.array_equal(value['reconstructed_mask'].numpy(),hand_mask))
            self.assertTrue(value['RLE_exact'] and value['PR_strict_sign_exact'])
            for stats in value['selected_PR_formula_stats'].values():
                self.assertEqual((stats['calls'],stats['points']),(1,2))
        scores[(24,'LOCAL_ERROR')][0]+=.001
        with self.assertRaisesRegex(ValueError,'Saved FP64 scores must be EXACT'):
            core.proof_query(torch,np,cv2,helper,legacy,endpoint,**kwargs)
        self.assertFalse(torch.cuda.is_initialized())

if __name__=='__main__':
    unittest.main()
