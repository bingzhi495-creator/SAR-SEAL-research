"""Original terminal and independent support checks, synthetic only."""
import unittest

from verify_dev_readout import verify_terminal


class ReadbackTest(unittest.TestCase):

    def test_native_points_reject_unsealed_spatial_support(self):
        from pathlib import Path
        import os
        import numpy as np
        from verify_dev_readout import verify_points
        from types import ModuleType
        sources=[Path(os.environ["POINT_PAIR_SOURCE"]),Path(os.environ["POINT_RANK_SOURCE"])]
        modules=[]
        for name,path in zip(("pair_math","rank_math"),sources):

            module=ModuleType(name)
            exec(compile(path.read_bytes(),str(path),"exec"),module.__dict__)
            modules.append(module)
        pair,rank=modules
        stock=np.zeros((800,800),dtype=bool)
        stock[20:28,20:28]=True
        box=np.array([10,10,40,40],dtype=np.float32)
        pools,_=rank.editable_pools(stock,box)
        add=np.concatenate([pools[k][0] for k in ("add1","add2")])
        drop=np.concatenate([pools[k][0] for k in ("drop1","drop2")])
        n=min(32,len(add),len(drop))
        coords=pair.pair_assignment(add[:n],drop[:n],800)
        self.assertFalse(verify_points(np,stock,box,coords,pair,rank))
        coords=coords.copy()
        coords[0,0]=0
        with self.assertRaisesRegex(ValueError,"support"):
            verify_points(np,stock,box,coords,pair,rank)


    def test_cached_mask_witness_still_rejects_bad_metadata_and_duplicates(self):
        import os
        from pathlib import Path
        from types import ModuleType, SimpleNamespace
        import numpy as np
        from pycocotools import mask as codec
        from verify_dev_readout import verify_masks
        path=Path(os.environ["SPOT_SOURCE"])/"feature_adapter.py"
        real=ModuleType("_real_sparse_apply")
        exec(compile(path.read_bytes(),str(path),"exec"),real.__dict__)
        stock=np.zeros((12,12),dtype=bool)
        stock[4:8,2:10]=True
        coords=np.array([[3*12+2,4*12+2]],dtype=np.int32)
        packed=codec.encode(np.asfortranarray(stock,dtype=np.uint8))
        source=dict(size=packed["size"],counts=packed["counts"].decode())
        action=dict(flipped=coords.reshape(-1),edits=2,adds=1,drops=1,area_delta=0,
                    actions=1,action_kind="swap",nonpositive_actions=0,nonpositive_flips=0,predicted_gain_sum=.5)
        calls=[]
        feature=SimpleNamespace(apply_sparse=lambda *args:(calls.append(1),real.apply_sparse(*args))[1])
        cache={}
        read=lambda *args:action
        first=verify_masks(np,codec,stock,source,coords,("SF",),read,feature,5217,1,cache=cache)
        repeated=verify_masks(np,codec,stock,source,coords,("SF",),read,feature,5218,1,cache=cache)
        self.assertEqual(first,repeated)
        self.assertEqual(len(calls),1)
        uncached=verify_masks(np,codec,stock,source,coords,("SF",),read,real,5218,1)
        self.assertEqual(first,uncached)
        action["adds"]=0
        with self.assertRaises(ValueError):
            verify_masks(np,codec,stock,source,coords,("SF",),read,feature,5219,1,cache=cache)
        action["adds"]=1
        action["flipped"]=np.array([coords[0,0],coords[0,1],coords[0,0]],dtype=np.int32)
        with self.assertRaises(ValueError):
            verify_masks(np,codec,stock,source,coords,("SF",),read,feature,5219,1,cache=cache)
        self.assertEqual(len(calls),1)


        zero=dict(action,flipped=np.empty(0,dtype=np.int32),edits=0,adds=0,drops=0,
                  area_delta=0,actions=0,predicted_gain_sum=0.)
        read_zero=lambda *args:zero
        n1=verify_masks(np,codec,stock,source,coords,("N",),read_zero,feature,5217,1,cache=cache)
        n2=verify_masks(np,codec,stock,source,coords,("N",),read_zero,feature,5218,1,cache=cache)
        self.assertEqual(n1,n2)
        self.assertEqual(len(calls),2)

    def test_original_terminal_digest_must_match(self):
        digest="a"*64
        wait=dict(launch={"session_id":17},terminal={"exit_code":0,"output":
                  '{"status":"RFDETR_MATCHED_POINTS_DEV705_READOUT_COMPLETE","result_sha256":"'+digest+'"}\n'})
        verify_terminal(wait,digest)
        with self.assertRaises(ValueError):
            verify_terminal(wait,"b"*64)


if __name__=="__main__":
    unittest.main()
