# SOURCE_ONLY_SYNTHETIC_TEST: Root actual exit0/fb9fe9; no SAR inputs.
"""ROOT runs RED after the actual GPU terminal; qualifier is not implemented yet.
Expected future module: N20_FEATURE_NEIGHBOR_CACHE_CPU_QUALIFIER_UNBOUND_20261015_01.py.
No GPU coordinate builder, renderer, teacher, PR or GT is used by this fixture.
"""
import copy
import hashlib
import importlib.util
from pathlib import Path
import unittest


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _index_digest(value):
    # ROOT confirmed ordinary N07 qualify_cache_cpu.py:180 / train.py:93 ABI.
    array = value.detach().cpu().contiguous().numpy()
    return dict(shape=list(array.shape), dtype=str(array.dtype),
                sha256=hashlib.sha256(array.tobytes()).hexdigest())


class HandNativeFeatureNeighbors(unittest.TestCase):
    def test_hand_grid_and_sample_order_reject_wrong_physical_mapping(self):
        qualifier = _load('n20_feature_neighbor_qualifier_under_test',
            Path(__file__).resolve().with_name(
                'N20_FEATURE_NEIGHBOR_CACHE_CPU_QUALIFIER_UNBOUND_20261015_01.py'))
        import numpy as np
        import torch
        self.assertFalse(torch.cuda.is_initialized())
        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)
        identity = dict(image_id=700001, original_shape=[2,3])
        teacher = dict(path='E:/SYNTHETIC_ONLY/never_open_teacher.pth',
                       bytes=0, sha256='0'*64)  # Compared as metadata, never loaded.
        unique = torch.tensor([0,4], dtype=torch.int64)
        inverse = torch.tensor([1,0,1,0], dtype=torch.int64)
        flats = torch.tensor([4,0,4,0], dtype=torch.int64)
        qids = torch.tensor([5,2,5,7], dtype=torch.int64)
        extra = torch.tensor([1,2,3,5], dtype=torch.int64)
        # Hand-computed NW/N/NE/W/C/E/SW/S/SE on the 2x3 original grid.
        expected_coordinates = np.array([[0,0,1,0,0,1,3,3,4],
                                         [0,1,2,3,4,5,3,4,5]], dtype=np.int64)
        mapping = torch.tensor([[0,0,2,0,0,2,4,4,1],
                                [0,2,3,4,1,5,4,1,5]], dtype=torch.int64)
        combined_indices = np.array([0,4,1,2,3,5], dtype=np.int64)
        np.testing.assert_array_equal(combined_indices[mapping.numpy()], expected_coordinates)
        np.testing.assert_array_equal(expected_coordinates[inverse.numpy()],
                                     expected_coordinates[[1,0,1,0]])
        channels = torch.arange(256, dtype=torch.float32)
        old_F = torch.stack((channels,4000+channels))
        extra_F = torch.stack(tuple(1000*int(flat)+channels for flat in extra))
        combined_F = torch.cat((old_F,extra_F),dim=0)
        self.assertTrue(torch.equal(combined_F[mapping][:,4,:], old_F))
        self.assertTrue(torch.equal(combined_F[mapping][inverse][:,4,:],
                                    torch.stack((old_F[1],old_F[0],old_F[1],old_F[0]))))
        samples = dict(flat_indices=flats,query_ids=qids,seed=202610140160000,
                       replacement=True,duplicates_preserved=True)
        old_packet = dict(dataset_index=0,identity=identity,teacher_checkpoint=teacher,
                          matching_loss_seed=202610122001,natural_query_ids=list(range(100)),
                          unique_flat_indices=unique,sample_to_unique=inverse,
                          sampled_F_unique=old_F,samples=samples)
        old_row = dict(dataset_index=0,image_id=identity['image_id'],identity=identity,
                       points=4,unique_sampled_F_points=2,natural_queries=100,
                       matching_loss_seed=202610122001,point_seed=202610140160000)
        digests = {name:_index_digest(value) for name,value in (
            ('unique_flat_indices',unique),('sample_to_unique',inverse),
            ('sample_flat_indices',flats),('sample_query_ids',qids))}
        new_packet = dict(schema='N20-true-F-neighbor-capture-body-v1',dataset_index=0,
                          identity=identity,teacher_checkpoint=teacher,matching_loss_seed=202610122001,
                          point_seed=202610140160000,original_unique_count=2,extra_count=4,sampled_points=4,
                          input_index_digests=digests,center_column=4,
                          neighbor_order=['NW','N','NE','W','C','E','SW','S','SE'],
                          combined_order='old unique order then sorted extra union-minus-centers',
                          extra_F=extra_F,extra_flat_indices=extra,neighbor_to_combined=mapping)
        new_row = dict(dataset_index=0,image_id=identity['image_id'],identity=identity,
                       natural_queries=100,points=4,unique_centers=2,original_unique_count=2,
                       extra_count=4,matching_loss_seed=202610122001,point_seed=202610140160000,
                       old_F_centers_byte_exact=True,original_samples_and_order_reused=True,
                       extra_feature_bytes=4096,raw_tensor_payload_bytes=4272)
        proof = qualifier.qualify_image(torch,np,old_packet,new_packet,old_row,new_row)
        self.assertEqual(proof['original_unique_count'],2)
        self.assertEqual(proof['extra_count'],4)
        self.assertEqual(proof['sampled_points'],4)
        for key in ('neighbor_coordinates_exact','extra_union_exact',
                    'center_features_byte_exact','original_sample_order_exact'):
            self.assertIs(proof[key],True)
        self.assertIs(proof['qualified'],False)  # Numeric body cannot grant ROOT qualification.
        for label,at,replacement in [('row_crossing',(0,5),4),
                                     ('wrong_center',(0,4),1)]:
            with self.subTest(malformed=label):
                bad = copy.deepcopy(new_packet)
                bad['neighbor_to_combined'][at] = replacement
                with self.assertRaises((ValueError,RuntimeError)):
                    qualifier.qualify_image(torch,np,old_packet,bad,old_row,new_row)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
