"""Synthetic CPU-only contracts; no real data, cached features, or weights."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '2'
import psutil
psutil.Process().cpu_affinity([0,1])
import unittest
import numpy as np
import torch
torch.set_num_threads(2)
torch.set_num_interop_threads(2)
assert not torch.cuda.is_initialized()

import quality_head


class QualityHeadContractTests(unittest.TestCase):
    def test_target_uses_full_original_grid_and_empty_union_is_zero(self):
        pred = np.array([[1,1,0],[0,1,0]],dtype=np.bool_)
        target = np.array([[1,0,0],[0,1,1]],dtype=np.bool_)
        score = quality_head.matched_mask_iou(np,pred,target)
        self.assertIsInstance(score,np.float64)
        self.assertEqual(score,np.float64(.5))
        zero = np.zeros((2,3),dtype=np.bool_)
        self.assertEqual(quality_head.matched_mask_iou(np,zero,zero),np.float64(0))
        # Same cropped shape at disjoint locations must not become a perfect IoU.
        first = np.zeros((4,6),dtype=np.bool_);first[0,0]=True
        second = np.zeros((4,6),dtype=np.bool_);second[3,5]=True
        self.assertEqual(quality_head.matched_mask_iou(np,first,second),np.float64(0))
        self.assertFalse(torch.cuda.is_initialized())

    def test_predicted_non_square_bbox_is_unpadded_bilinear_and_translation_invariant(self):
        pattern = torch.tensor([[True,False,False],[False,False,True]],dtype=torch.bool)
        pred = torch.zeros((6,9),dtype=torch.bool);pred[1:3,2:5]=pattern
        translated = torch.zeros((9,14),dtype=torch.bool);translated[4:6,7:10]=pattern
        before = pred.clone()
        feature = quality_head.make_mask_features(torch,pred)
        shifted = quality_head.make_mask_features(torch,translated)
        self.assertEqual(tuple(feature.shape),(784,))
        self.assertEqual(feature.dtype,torch.float32)
        self.assertEqual(feature.device.type,'cpu')
        self.assertFalse(feature.requires_grad)
        self.assertTrue(torch.equal(feature,shifted))
        self.assertTrue(torch.equal(pred,before))
        resized = feature.reshape(28,28)
        self.assertEqual(resized[0,0].item(),1.0)
        self.assertEqual(resized[-1,-1].item(),1.0)
        self.assertAlmostEqual(resized[0,5].item(),51/56,places=6)
        empty = quality_head.make_mask_features(torch,torch.zeros((3,7),dtype=torch.bool))
        self.assertTrue(torch.equal(empty,torch.zeros(784,dtype=torch.float32)))
        self.assertFalse(torch.cuda.is_initialized())

    def test_score_product_preserves_original_order_and_q1_preserves_exact_S0(self):
        S0 = np.array([-0.0,.2,.2,1.25],dtype=np.float64)
        original_bytes = S0.tobytes()
        identity = quality_head.score_product(np,S0,np.ones(4,dtype=np.float64))
        self.assertEqual(identity.dtype,np.dtype('float64'))
        self.assertEqual(identity.tobytes(),original_bytes)
        qhat = np.array([1,.5,0,.4],dtype=np.float64)
        product = quality_head.score_product(np,S0,qhat)
        np.testing.assert_array_equal(product,np.array([-0.0,.1,0,.5],dtype=np.float64))
        self.assertEqual(S0.tobytes(),original_bytes)
        np.testing.assert_array_equal(qhat,[1,.5,0,.4])

    def test_quality_learning_has_no_gradient_or_mutation_path_into_teacher_embedding(self):
        teacher = torch.linspace(-1,1,512,dtype=torch.float32).reshape(2,256).requires_grad_()
        before = teacher.detach().clone()
        mask_features = quality_head.make_mask_features(torch,torch.zeros((4,7),dtype=torch.bool))
        features = torch.cat((teacher,mask_features.unsqueeze(0).expand(2,784)),dim=1)
        head = quality_head.build_head(torch)
        quality = head(features)
        self.assertEqual(tuple(quality.shape),(2,))
        self.assertEqual(quality.dtype,torch.float32)
        self.assertTrue(bool(torch.isfinite(quality).all()))
        self.assertTrue(bool(((quality >= 0) & (quality <= 1)).all()))
        quality.sum().backward()
        self.assertIsNone(teacher.grad)
        self.assertTrue(torch.equal(teacher.detach(),before))
        self.assertTrue(all(p.grad is not None for p in head.parameters()))
        self.assertTrue(any(bool((p.grad != 0).any()) for p in head.parameters()))
        self.assertFalse(torch.cuda.is_initialized())

    def test_head_rejects_wrong_width_dtype_shape_and_nonfinite_input(self):
        head = quality_head.build_head(torch)
        invalid = [torch.zeros((2,1041)),torch.zeros(1040),
                   torch.zeros((2,1040),dtype=torch.float64),
                   torch.full((2,1040),float('nan')),torch.full((2,1040),float('inf')),
                   [[0.0]*1040]]
        for value in invalid:
            with self.subTest(value_type=type(value)):
                with self.assertRaises(ValueError):
                    head(value)

    def test_feature_accepts_only_predicted_CPU_bool_mask_and_never_GT(self):
        for pred in [torch.ones((2,3),dtype=torch.float32),torch.zeros((0,3),dtype=torch.bool),
                     torch.zeros((2,3,1),dtype=torch.bool),np.zeros((2,3),dtype=np.bool_)]:
            with self.subTest(value_type=type(pred)):
                with self.assertRaises(ValueError):
                    quality_head.make_mask_features(torch,pred)
        pred = torch.ones((2,3),dtype=torch.bool)
        with self.assertRaises(TypeError):
            quality_head.make_mask_features(torch,pred,GT=pred)
        with self.assertRaises(TypeError):
            quality_head.make_mask_features(torch,pred,e0=torch.zeros(256))

    def test_target_and_product_reject_nonbinary_nonfinite_and_misaligned_inputs(self):
        mask=np.zeros((2,3),dtype=np.bool_)
        bad_targets=[np.zeros((3,2),dtype=np.bool_),np.full((2,3),2,dtype=np.int64),
                     np.full((2,3),float('nan')),np.zeros((0,3),dtype=np.bool_),np.zeros(6)]
        for target in bad_targets:
            with self.subTest(target_shape=target.shape):
                with self.assertRaises(ValueError):
                    quality_head.matched_mask_iou(np,mask,target)
        for scores,quality in [
            (np.ones(2),np.ones(3)),(np.ones((1,2)),np.ones((1,2))),
            (np.ones(2),np.array([0.,1.1])),(np.ones(2),np.array([0.,-.1])),
            (np.ones(2),np.array([0.,float('nan')])),
            (np.array([1.,float('inf')]),np.ones(2))]:
            with self.subTest(scores_shape=scores.shape,quality_shape=quality.shape):
                with self.assertRaises(ValueError):
                    quality_head.score_product(np,scores,quality)


if __name__ == '__main__':
    unittest.main()
