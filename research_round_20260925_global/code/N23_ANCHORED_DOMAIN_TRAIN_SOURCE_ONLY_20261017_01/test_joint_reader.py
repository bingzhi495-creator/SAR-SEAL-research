# encoding: utf-8
"""UNIT_FIXTURE only: original-point / C-cache joins on synthetic NumPy arrays."""
import copy
import unittest
import numpy as np

import joint_reader


def fixture():
    ref = lambda name: dict(path='E:/UNIT/' + name, bytes=1, sha256='a' * 64)
    new_binding = {k: ref(k) for k in
                   ('source', 'spec', 'GPU_admission', 'teacher_checkpoint', 'fixed_PR_checkpoint')}
    new_binding['phase'] = 'UNIT_N16'
    new_binding['cache_body_source'] = dict(path='E:/UNIT/cache_body.py', bytes=23893,
        sha256='77de4c5445ac455672095b8c795e16046a4ed6822bae50f739f40cb4625ea5bd')
    new_binding['C_engine_source'] = dict(path='E:/UNIT/C_engine.py', bytes=7684,
        sha256='8f0bd504ce7899225d868381d9b656402f8882bb19f6bd880060085fe269f9cc')
    old_binding = {k: ref('OLD_' + k) for k in ('source', 'spec', 'GPU_admission')}
    old_binding.update(phase='UNIT_N13', teacher_checkpoint=new_binding['teacher_checkpoint'])
    qids = np.array([0, 2, 0, 0, 1, 1], dtype=np.int64)
    flats = np.array([0, 3, 0, 4, 2, 2], dtype=np.int64)
    unique, inverse = np.unique(flats, return_inverse=True)
    raw = np.full(100, -2, dtype=np.int64); raw[0] = 0; raw[2] = 1; raw[1] = -1
    kinds = ['ignored'] * 100; kinds[0] = kinds[2] = 'matched'; kinds[1] = 'background'
    identity = dict(image_id=42, ann_ids=[10, 20], original_shape=[2, 3])
    new_packet = dict(schema='N16-all100-FIT-action-sampled-cache-body-v1', dataset_index=3,
        identity=copy.deepcopy(identity), natural_query_ids=list(range(100)), raw_GT_ids=[10, 20],
        sampled_F_unique=np.arange(len(unique)*256, dtype=np.float32).reshape(-1, 256),
        native_e0=np.arange(100*256, dtype=np.float32).reshape(1, 100, 256),
        unique_flat_indices=unique, sample_to_unique=inverse.astype(np.int64),
        assignment=dict(raw_GT_index_by_query=raw, raw_GT_annotation_id_by_query=[10,None,20]+[None]*97,
            supervision_kind_by_query=kinds, native_positive_query_indices=[0,2],
            native_negative_query_indices=[1], native_ignored_query_indices=list(range(3,100))),
        C_counts_all100=[6]*100, K_counts_all100=[1]*100,
        active_matched_queries=[0,2], active_background_queries=[1], active_queries=[0,1,2],
        samples=dict(query_ids=qids, flat_indices=flats,
            native_L=np.array([-.5,3,-.5,-5,-3,-3], np.float32),
            targets=np.array([1,0,1,1,0,0], np.bool_),
            importance_weights=np.array([.8,.8,.8,.8,1.2,1.2],np.float32),
            stratum_ids=np.array([0,0,0,0,1,1], np.uint8),
            replacement=True, duplicates_preserved=True))
    new_packet.update({k:v for k,v in new_binding.items() if k not in ('cache_body_source','C_engine_source')})
    new_row = dict(dataset_index=3,image_id=42,identity=copy.deepcopy(identity),points=6,
        C_counts_all100=[6]*100,K_counts_all100=[1]*100)
    old_L = np.array([[.5,-3,.5,4],[3,-3,2,-2]],np.float32)
    old_packet = dict(schema='N13-sameK-region-FIT-cache-packet-v1',dataset_index=3,
        identity=copy.deepcopy(identity),query_ids=np.array([0,2],np.int64),
        query_embeddings=new_packet['native_e0'][0,[0,2]].copy(),K=[1,1],
        replacement=True,duplicates_preserved=True,
        arms={'LOG_PR':dict(F_samples=np.arange(2*4*256,dtype=np.float32).reshape(2,4,256),
            native_logits=old_L,targets=np.array([[1,1,0,1],[0,0,1,0]],np.bool_),
            point_indices=np.array([[0,0,4,2],[3,3,1,5]],np.int64),
            point_in_region=np.abs(old_L)<=np.float32(2))})
    old_packet.update({k:v for k,v in old_binding.items() if k != 'GPU_admission'})
    old_row = dict(dataset_index=3,image_id=42,identity=copy.deepcopy(identity),raw_GT=2,
        query_ids=[0,2],points_per_arm=8,K=[1,1],raw_GT_ids=[10,20])
    old_packet['raw_GT_digest']=dict(shape=[2,2,3],dtype='bool',sha256='c'*64)
    new_packet['raw_GT_digest']=copy.deepcopy(old_packet['raw_GT_digest'])
    return old_packet,old_row,new_packet,new_row,old_binding,new_binding


class JointReaderTests(unittest.TestCase):
    def test_original_GT_order_draw_order_and_duplicates_are_preserved(self):
        args=fixture()
        result=joint_reader.build_joint_image(np,*args,synthetic_only=True)
        self.assertEqual(result['raw_GT_order'],[10,20])
        self.assertEqual([b['query_id'] for b in result['batches']],[0,2])
        first=result['batches'][0]
        np.testing.assert_array_equal(first['old']['F'],args[0]['arms']['LOG_PR']['F_samples'][0])
        np.testing.assert_array_equal(first['old']['flat_indices'],[0,0,4,2])
        np.testing.assert_array_equal(first['old']['point_in_region'],[True,False,True,False])
        np.testing.assert_array_equal(first['new']['flat_indices'],[0,0,4])
        np.testing.assert_array_equal(first['new']['targets'],[True,True,True])
        self.assertEqual(result['audit']['old_draws'],8)
        self.assertEqual(result['audit']['new_matched_draws'],4)
        self.assertEqual(result['audit']['discarded_new_background_draws'],2)

    def test_identity_binding_teacher_and_embedding_mismatches_are_rejected(self):
        mutations = (
            lambda a:a[0].update(phase='different'),
            lambda a:a[0].update(schema='different'),
            lambda a:a[0]['identity'].update(image_id=999),
            lambda a:a[1].update(raw_GT_ids=[20,10]),
            lambda a:a[4].update(teacher_checkpoint=dict(path='E:/UNIT/wrong',bytes=1,sha256='b'*64)),
            lambda a:a[0]['query_embeddings'].__setitem__((0,0),np.float32(-0.0)),
            lambda a:a[4].pop('GPU_admission'),
            lambda a:a[2]['raw_GT_digest'].update(sha256='d'*64),
        )
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                args=fixture();mutate(args)
                with self.assertRaises(RuntimeError):
                    joint_reader.build_joint_image(np,*args,synthetic_only=True)

    def test_K0_real_GT_keeps_old_draws_without_inventing_new_points(self):
        args=fixture()
        args[2]['samples']['query_ids'][1]=0
        args[2]['K_counts_all100'][2]=args[3]['K_counts_all100'][2]=0
        args[0]['K'][1]=args[1]['K'][1]=0
        args[0]['arms']['LOG_PR']['native_logits'][1]=np.array([3,-3,4,-4],np.float32)
        args[0]['arms']['LOG_PR']['point_in_region'][1]=False
        result=joint_reader.build_joint_image(np,*args,synthetic_only=True)
        self.assertEqual(result['raw_GT_order'],[10,20])
        self.assertIsNone(result['batches'][1]['new'])
        self.assertEqual(result['batches'][1]['old']['n'],4)
        self.assertEqual(result['audit']['K0_real_GT_instances'],1)
        self.assertEqual(result['audit']['K_positive_real_GT_instances'],1)
        self.assertEqual(result['audit']['old_draws'],8)
        args[0]['arms']['LOG_PR']['native_logits'][1,0]=np.float32(.5)
        args[0]['arms']['LOG_PR']['point_in_region'][1,0]=True
        with self.assertRaises(RuntimeError):
            joint_reader.build_joint_image(np,*args,synthetic_only=True)

    def test_wrong_GT_query_K_region_and_incomplete_positive_K_join_are_rejected(self):
        def swapped_GT_queries(a):
            a[0]['query_ids'] = np.array([2,0],np.int64)
            a[1]['query_ids'] = [2,0]
            a[0]['query_embeddings'] = a[0]['query_embeddings'][[1,0]].copy()
        mutations = (
            swapped_GT_queries,
            lambda a:a[0]['K'].__setitem__(0,2),
            lambda a:a[0]['arms']['LOG_PR']['point_in_region'].__setitem__((0,1),True),
            lambda a:a[0]['arms']['LOG_PR']['point_indices'].__setitem__((0,1),6),
            lambda a:a[2]['samples']['query_ids'].__setitem__(1,0),
            lambda a:a[0].update(duplicates_preserved=False),
            lambda a:a[1].update(points_per_arm=9),
        )
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                args=fixture();mutate(args)
                with self.assertRaises(RuntimeError):
                    joint_reader.build_joint_image(np,*args,synthetic_only=True)

    def test_caller_arrays_are_unchanged_and_fixture_requires_explicit_flag(self):
        args=fixture();before=copy.deepcopy(args)
        result=joint_reader.build_joint_image(np,*args,synthetic_only=True)
        result['raw_GT_order'][0]=999
        result['batches'][0]['old']['F'][0,0]=-99
        result['batches'][0]['old']['targets'][0]=False
        result['batches'][0]['old']['flat_indices'][0]=5
        result['batches'][0]['old']['e0'][0]=-99
        result['batches'][0]['new']['F'][0,0]=-99
        result['batches'][0]['new']['targets'][0]=False
        for key in ('F_samples','native_logits','targets','point_indices','point_in_region'):
            np.testing.assert_array_equal(args[0]['arms']['LOG_PR'][key],before[0]['arms']['LOG_PR'][key])
        np.testing.assert_array_equal(args[0]['query_embeddings'],before[0]['query_embeddings'])
        np.testing.assert_array_equal(args[2]['sampled_F_unique'],before[2]['sampled_F_unique'])
        np.testing.assert_array_equal(args[2]['samples']['targets'],before[2]['samples']['targets'])
        self.assertEqual(args[0]['identity'],before[0]['identity'])
        with self.assertRaises(RuntimeError):
            joint_reader.build_joint_image(np,*args)


if __name__ == '__main__':
    unittest.main()
