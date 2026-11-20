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

    def test_original_terminal_digest_must_match(self):
        digest="a"*64
        wait=dict(launch={"session_id":17},terminal={"exit_code":0,"output":
                  '{"status":"RFDETR_MATCHED_POINTS_DEV705_READOUT_COMPLETE","result_sha256":"'+digest+'"}\n'})
        verify_terminal(wait,digest)
        with self.assertRaises(ValueError):
            verify_terminal(wait,"b"*64)


if __name__=="__main__":
    unittest.main()
