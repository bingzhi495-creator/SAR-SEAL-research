import unittest
import os
import importlib.util
from hashlib import sha256
from pathlib import Path
from feature_adapter import select_images


def original_module(variable, expected):
    path = Path(os.environ[variable])
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != expected:
        raise ValueError('Exact ordinary published numerical source required')
    spec = importlib.util.spec_from_file_location(variable, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FeatureTests(unittest.TestCase):
    def test_real_rf_shapes_render_and_native_strict_zero_parity(self):
        import numpy as np
        import torch
        from feature_adapter import render_posterior
        adapter=original_module('RF_POSTERIOR_SOURCE','5b9a25a4e9c8c125342dd9e8fd7e0d1aeb3ecb546f5cf95b8512dfe09116b0f2')
        self.assertFalse(torch.cuda.is_initialized())
        raw=np.zeros((144,144),dtype=np.float32)
        stock=np.zeros((800,800),dtype=bool)
        p=render_posterior(torch,np,adapter,raw,stock)
        self.assertEqual(p.shape,(800,800))
        self.assertEqual(p.dtype,np.float32)
        self.assertTrue(np.all(p==.5))
        with self.assertRaises(ValueError):
            render_posterior(torch,np,adapter,raw,~stock)
        self.assertFalse(torch.cuda.is_initialized())

    def test_sparse_actual_masks_match_ordinal_ties_and_exact_counts(self):
        import numpy as np
        from feature_adapter import readouts_sparse, apply_sparse
        stock = np.zeros((10,10),dtype=bool)
        stock[:5] = True
        pairs = np.array([[50,2],[51,0],[52,1]],dtype=np.int32)
        small = np.array([[.9,.1],[.4,.6],[.3,.7]],dtype=np.float32)
        large = np.array([[.9,.55],[.85,.6],[.8,.65]],dtype=np.float32)
        half = np.full((3,2),.5,dtype=np.float32)
        original = stock.copy()
        result = readouts_sparse(np,stock,pairs,small,large,half,2)
        self.assertEqual(set(result),{'N','SF','SU=F','SU','LF=S','LU=S','LW=S','NP=S'})
        for name,edit in result.items():
            self.assertNotIn('mask',edit)
            mask = apply_sparse(np,stock,edit)
            self.assertEqual(np.count_nonzero(mask!=stock),0 if name=='N' else 2)
            self.assertEqual(edit['area_delta'],2 if name=='LW=S' else 0)
            self.assertIn('nonpositive_actions',edit)
            self.assertIn('nonpositive_flips',edit)
        self.assertEqual(result['NP=S']['flipped'].tolist(),[50,2])
        self.assertEqual(result['LW=S']['flipped'].tolist(),[50,51])
        tied=readouts_sparse(np,stock,pairs,small,half,half,2)
        self.assertEqual(tied['LW=S']['flipped'].tolist(),[0,1])
        self.assertEqual(tied['LW=S']['nonpositive_actions'],2)
        self.assertEqual(tied['LW=S']['area_delta'],-2)
        self.assertTrue(np.array_equal(stock,original))
        tampered=dict(result['LW=S'],area_delta=0)
        with self.assertRaises(ValueError):
            apply_sparse(np,stock,tampered)
        zero=readouts_sparse(np,stock,pairs,half,half,half,2)
        self.assertTrue(all(v['edits']==0 for v in zero.values()))

    def test_real_published_fourteen_features_use_fp32_rf_posterior(self):
        import numpy as np
        from feature_adapter import extract_candidate
        pair = original_module('RF_PAIR_SOURCE', '8fe7a2c146c08c89f377b1f20b321276a60356de4bdbb27be2a00544dfc77962')
        rank = original_module('RF_RANK_SOURCE', 'cb438f8937f8e29dbdcbc441abdf5a788ac9c442fd63f1b3ad504b8ae0fcca0b')
        stock = np.zeros((800,800),dtype=bool)
        stock[20:28,30:38] = True
        prob = np.full((800,800), np.float32(.500123), dtype=np.float32)
        rgb = np.zeros((800,800,3), dtype=np.uint8)
        rgb[:,:,0] = np.arange(800,dtype=np.uint16).astype(np.uint8)[None,:]
        image = pair.image_features(rgb)
        result = extract_candidate(np,pair,rank,stock,[28.,18.,40.,30.],.73,prob,image)
        coords, features = result['coordinates'],result['features']
        self.assertEqual(features.shape,(32,2,14))
        self.assertEqual(features.dtype,np.float32)
        self.assertTrue(np.array_equal(features[:,:,0], np.full((32,2),.500123,dtype=np.float32)))
        for side in (0,1):
            y,x=np.divmod(coords[:,side],800)
            self.assertTrue(np.array_equal(features[:,side,9:],image[:,y,x].T))
            self.assertTrue(np.array_equal(features[:,side,1],np.full(32,side)))
            self.assertTrue(np.array_equal(stock.ravel()[coords[:,side]],np.full(32,bool(side))))
        self.assertTrue(np.all(features[:,:,6]==6.))
        self.assertTrue(np.all(features[:,:,8]==np.float32(.73)))
        with self.assertRaises(ValueError):
            extract_candidate(np,pair,rank,stock,[28.,18.,40.,30.],.73,prob.astype(np.float16),image)

    def test_label_free_selection_digest_then_source_id(self):
        class Forbidden:
            def __iter__(self):
                raise AssertionError('Labels must not be visited')
        rows = [dict(image_id=i, decoded_rgb_sha256=d * 64, annotations=Forbidden())
                for i,d in [(8,'b'),(7,'a'),(2,'a'),(1,'c'),(6,'d')]]
        chosen = select_images(rows,4)
        self.assertEqual([r['image_id'] for r in chosen],[2,7,8,1])
        self.assertIs(chosen[0], rows[2])
        with self.assertRaises(ValueError):
            select_images(rows+rows[:1],4)


if __name__ == '__main__':
    unittest.main()
