import unittest
import numpy as np
from dev_readout_contract import encode_json, unordered_ring, unchanged_identities


class DevReadoutContract(unittest.TestCase):
    def test_ring_preserves_all_samples_in_each_ring_and_common_values(self):
        a=np.arange(76,dtype=np.float32).reshape(2,38)
        x=a.copy()
        x[:,6:]=x[:,6:].reshape(-1,4,8)[:,:,::-1].reshape(-1,32)
        np.testing.assert_array_equal(unordered_ring(x),a)
        self.assertFalse(np.array_equal(x, a))

    def test_prediction_identity_and_canonical_newline(self):
        row=dict(image_id=5,category_id=1,score=.25,segmentation=dict(size=[2,1],counts="2"))
        self.assertEqual(encode_json([row]),b'[{"image_id":5,"category_id":1,"score":0.25,"segmentation":{"size":[2,1],"counts":"2"}}]\n')
        self.assertTrue(unchanged_identities([row],[dict(row,segmentation=dict(size=[2,1],counts="11"))]))
        self.assertFalse(unchanged_identities([row],[dict(row,score=.5)]))


if __name__=="__main__":
    unittest.main()

