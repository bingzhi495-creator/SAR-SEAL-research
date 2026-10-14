"""Pure CPU synthetic captures; never reads scientific data/states/records."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import copy
import importlib.util
from pathlib import Path
import unittest
import numpy as np
import torch
import cv2
from scipy import ndimage
torch.set_num_threads(2)
cv2.setNumThreads(1)
BASE=Path('E:/SAR/seal')


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


helper=load('n16_frozen_helper',BASE/'github_release_20261011/research_round_20260925_global/code/N16_SCOUT_ACTION_LABELS_AND_SELECTION_UNBOUND_20261014_01.py')
ev=load('n16_ordinary_eval',BASE/'tmp/N16_ALL100_SCOUT_SELECTOR_EVAL_BODY_UNBOUND_DRAFT_20261014_01.py')
proof=load('n16_cpu_proof',BASE/'tmp/N16_ALL100_SCOUT_SELECTOR_CPU_PROOF_BODY_UNBOUND_DRAFT_20261014_01.py')


class Scout(torch.nn.Module):
    def __init__(self,values,table=False):
        super().__init__();self.register_buffer('values',torch.tensor(values,dtype=torch.float32))
        self.table=table;self.eval()
    def forward(self,F,e,L):
        return self.values[F[:,0].to(torch.int64)].clone() if self.table else self.values.expand(len(L),3).clone()


class PR(torch.nn.Module):
    def __init__(self,u):super().__init__();self.u=u;self.eval()
    def forward(self,F,e,L,D,arm):return {'residual':torch.full_like(L,self.u)}


def formulas(u=4.,shift=0.):
    def original(F,e,L):
        residual=torch.full_like(L,u);return residual,L+residual
    def independent(F,e,L):
        residual=torch.full_like(L,u+shift);return residual,L+residual
    return {'original_head':original,'independent_PR':independent}


def fixture(values=((-3.,-1.,4.),),mode='atmost',u=4.,utility=None,error=None):
    L0=torch.tensor(values,dtype=torch.float32);h,w=L0.shape
    F=torch.zeros((256,h,w),dtype=torch.float32);F[0]=torch.arange(h*w).reshape(h,w)
    e=torch.zeros((100,256),dtype=torch.float32);L=L0[None].expand(100,h,w).clone()
    utility=utility if utility is not None else Scout((5.,0.,0.))
    error=error if error is not None else Scout((5.,5.,0.))
    events=[]
    row=next(ev.evaluate_all100_scout_selectors(torch,np,ndimage,helper,F,e,L,utility,error,PR(u),
        evalmode=mode,point_callback=events.append,synthetic_only=True))
    def chunks(control,stage):
        out=[]
        for event in events:
            if event['control']!=control or event['stage']!=stage:continue
            common={k:event[k].clone() for k in ('flat_indices','sampled_F','e0','native_L')}
            keys=('scout_logits',) if stage=='scout_C_logits' else ('GPU_raw_residual','GPU_combined')
            common.update({k:event[k].clone() for k in keys});out.append(common)
        return out
    records={}
    for name,result in row['controls'].items():
        records[name]=dict(C_scout_chunks=chunks(name,'scout_C_logits'),
            selected_PR_chunks=chunks(name,'selected_PR'),
            actual_selected_indices=torch.from_numpy(result['selected_flat_indices'].copy()),
            actual_mask=torch.from_numpy(result['mask'].copy()))
    return dict(schema=proof.CAPTURE_SCHEMA,query_id=0,evalmode=mode,native_L=L0.clone(),e0=e[0].clone(),
        K=row['K'],C_flat_indices=torch.from_numpy(row['C_flat_indices'].copy()),
        common_C_PR_chunks=chunks('COMMON_C_REFERENCE','full_C_PR'),controls=records)


def check(packet,fs=None):return proof.proof_query(torch,np,cv2,helper,packet,fs or formulas())


class Tests(unittest.TestCase):
    def test_four_controls_and_exact_two_scouts(self):
        packet=fixture();report=check(packet)
        self.assertEqual(set(report['controls']),set(ev.MAIN_CONTROLS))
        self.assertFalse(report['CPU_scout_forward_used_for_ranking'])
        for name,result in report['controls'].items():
            self.assertTrue(result['selected_indices_exact'] and result['full_mask_exact'])
            self.assertTrue(torch.equal(result['reconstructed_mask'],packet['controls'][name]['actual_mask']))
        exact=check(fixture(mode='exact'))
        self.assertEqual(set(exact['controls']),{'UTILITY','ERROR'})

    def test_near_tie_actual_GPU_logits_rank_not_CPU_approximation(self):
        above=np.nextafter(np.float32(1.),np.float32(np.inf))
        logits=np.array([[1.,0.,0.],[above,0.,0.],[0.,5.,0.],[0.,5.,0.]],dtype=np.float32)
        packet=fixture(values=((-3.,-3.,-1.,4.),),utility=Scout(logits,table=True))
        np.testing.assert_array_equal(packet['controls']['UTILITY']['actual_selected_indices'].numpy(),[1])
        alternative=logits.copy();alternative[0,0]=above;alternative[1,0]=1.
        self.assertLess(float(np.max(np.abs(alternative-logits))),1.e-4)
        L=packet['native_L'].numpy();C=np.ones_like(L,dtype=np.bool_)
        hypothetical=helper.choose_candidates(L,C,helper.utility_score(alternative).reshape(L.shape),1)
        np.testing.assert_array_equal(np.flatnonzero(hypothetical),[0])
        self.assertTrue(check(packet)['controls']['UTILITY']['selected_indices_exact'])
        changed=copy.deepcopy(packet);changed['controls']['UTILITY']['actual_selected_indices']=torch.tensor([0],dtype=torch.int64)
        with self.assertRaises(proof.ProofFailure):check(changed)

    def test_empty_K0_positive_K0_and_true_noop(self):
        for values in (((-20.,-20.),),((20.,20.),)):
            report=check(fixture(values=values))
            self.assertEqual(report['K'],0)
            self.assertEqual(report['common_C_PR']['formula_chunk_calls'],{'original_head':0,'independent_PR':0})
            self.assertTrue(all(c['selected_points']==0 for c in report['controls'].values()))
        report=check(fixture(u=0.),formulas(u=0.))
        self.assertEqual(report['controls']['UTILITY']['selected_points'],1)
        self.assertTrue(report['controls']['UTILITY']['full_mask_exact'])

    def test_nonpositive_atmost_and_real_exact(self):
        u=Scout((0.,5.,0.));e=Scout((-100.,-100.,100.))
        most=check(fixture(utility=u,error=e))
        self.assertEqual(most['controls']['UTILITY']['selected_points'],0)
        exact=check(fixture(mode='exact',utility=u,error=e))
        self.assertEqual(exact['controls']['UTILITY']['selected_points'],exact['K'])

    def test_formula_abs_and_strict_sign_failures(self):
        with self.assertRaises(proof.ProofFailure):check(fixture(),formulas(shift=.001))
        self.assertTrue(check(fixture(),formulas(shift=1.e-5))['common_C_PR']['strict_sign_exact'])
        with self.assertRaises(proof.ProofFailure):check(fixture(u=3.),formulas(u=3.,shift=1.e-6))

    def test_support_input_provenance_and_full_mask_tampering(self):
        packet=fixture()
        changes=[lambda p:p['controls']['UTILITY']['actual_mask'].__setitem__((0,0),True),
            lambda p:p['controls']['UTILITY']['selected_PR_chunks'][0]['sampled_F'].__setitem__((0,0),99.),
            lambda p:p['controls']['UTILITY']['C_scout_chunks'][0]['e0'].__setitem__(0,1.),
            lambda p:p.__setitem__('K',True),
            lambda p:p.__setitem__('C_flat_indices',torch.tensor([0,2],dtype=torch.int64)),
            lambda p:p['controls']['UTILITY']['C_scout_chunks'][0]['scout_logits'].__setitem__((0,0),float('nan')),
            lambda p:p['controls']['UTILITY']['C_scout_chunks'].clear(),
            lambda p:p['common_C_PR_chunks'][0]['GPU_raw_residual'].__setitem__(0,9.)]
        for change in changes:
            bad=copy.deepcopy(packet);change(bad)
            with self.assertRaises(proof.ProofFailure):check(bad)

    def test_canonical_chunks_and_formula_output_contract(self):
        packet=fixture(values=([0.]*2050,),u=0.)
        self.assertEqual([len(b['flat_indices']) for b in packet['common_C_PR_chunks']],[2048,2])
        self.assertTrue(check(packet,formulas(u=0.))['controls']['UTILITY']['full_mask_exact'])
        wrong=formulas();wrong['independent_PR']=lambda F,e,L:(torch.full_like(L,4.),L)
        with self.assertRaises(proof.ProofFailure):check(fixture(),wrong)

    def test_all100_order_complete_no_GT(self):
        packet=fixture(values=((-20.,),))
        stream=(dict(packet,query_id=q) for q in range(100))
        report=proof.proof_all100_queries(torch,np,cv2,helper,stream,formulas(),evalmode='atmost')
        self.assertEqual(report['queries'],100)
        with self.assertRaises(proof.ProofFailure):
            proof.proof_all100_queries(torch,np,cv2,helper,[packet]*100,formulas())
        with self.assertRaises(proof.ProofFailure):
            proof.proof_all100_queries(torch,np,cv2,helper,[dict(packet,query_id=q) for q in range(99)],formulas())
        with self.assertRaises(TypeError):proof.proof_query(torch,np,cv2,helper,packet,formulas(),GT=np.ones((1,1)))


if __name__=='__main__':unittest.main()
