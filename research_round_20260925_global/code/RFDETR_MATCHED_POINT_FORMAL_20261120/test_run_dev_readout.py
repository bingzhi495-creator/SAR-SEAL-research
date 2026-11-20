"""Pure header and sparse-schema checks without any DEV inputs."""
import unittest

from run_dev_readout import validate_capture_receipt


class HeaderTest(unittest.TestCase):

    def test_legacy_stock_encoding_restores_original_rle_order(self):
        from run_dev_readout import legacy_stock_bytes
        pool=[dict(rank=3,query_id=17,score=.75,area=1,rle={"counts":"012","size":[2,2]})]
        raw=legacy_stock_bytes(7,pool)
        self.assertIn(b'"segmentation":{"size":[2,2],"counts":"012"}',raw)
        self.assertTrue(raw.startswith(b'{"image_id":7,"raw_query_count":164,'))

    def test_nonzero_eight_arm_sparse_readout_has_exact_counts(self):
        from collections import Counter
        from pathlib import Path
        from types import SimpleNamespace
        import os
        import numpy as np
        from pycocotools import mask as codec
        from run_dev_readout import ARMS, SEEDS, emit_candidate, encoded, load, PINS
        feat=load(Path(os.environ["SPOT_SOURCE"])/"feature_adapter.py",PINS["feature_adapter.py"],"_synthetic_sparse")
        stock=np.zeros((12,12),dtype=bool)
        stock[4:8,2:10]=True
        coords=np.column_stack((np.array([3*12+2,3*12+3,3*12+4,3*12+5]),
                                np.array([4*12+2,4*12+3,4*12+4,4*12+5]))).astype(np.int32)
        small=np.asarray([[.9,.1],[.9,.1],[.1,.9],[.1,.9]],dtype=np.float32)
        large=np.tile(np.asarray([.9,.4],dtype=np.float32),(4,1))
        features=np.zeros((4,2,14),dtype=np.float32)
        features[:,:,0]=[.4,.6]
        rle=codec.encode(np.asfortranarray(stock,dtype=np.uint8))
        rle["counts"]=rle["counts"].decode()
        record=dict(coordinates=coords,features=features,rle=rle)
        trainer=SimpleNamespace(predict=lambda head,*args:small if head=="S" else large)
        refits={seed:({"S":"S","L":"L"},None,None) for seed in SEEDS}
        budgets={seed:{arm:Counter() for arm in ARMS} for seed in SEEDS}
        out=emit_candidate(np,stock,record,refits,trainer,feat,codec,budgets)
        for seed in SEEDS:
            self.assertNotIn("N",out[str(seed)])
            self.assertEqual(out[str(seed)]["SU"]["edits"],4)
            self.assertEqual(out[str(seed)]["LW=S"]["area_delta"],4)
            self.assertTrue(all(out[str(seed)][arm]["edits"]==4 for arm in ARMS if arm!="N"))
        self.assertIn(b'"flipped":',encoded(out))

    def test_identity_hash_preserves_joint_rank_order(self):
        from run_dev_readout import mapped_identity_sha256
        pool=[dict(rank=3,query_id=17,score=.75,area=31),dict(rank=6,query_id=22,score=.5,area=32)]
        self.assertNotEqual(mapped_identity_sha256(pool),mapped_identity_sha256(pool[::-1]))

    def test_cpu_stock_mismatch_rejected(self):
        row=dict(status="RF_CENTER5202_FULL_CAPTURE_INDEPENDENTLY_ACCEPTED_V1",mode="dev",
                 FIT=False,DEV=True,TEST=False,GT_opened=False,AP=False,images=705,
                 original_os_exit=0,job_drained=True,model_forwards=0,native_mapped_rows=67386,
                 source_checkpoint_sha256="011bd30ed056c871be1d61e14d71b3b2b258330a9d20cf2f164ca4a7c4f23c11",
                 stock_predictions_sha256="da55f2d3f9bbf08b26d5fae76606b2496a5b482c5fb61f4fab53c23172e5c06e",
                 same_device_raster_exact=5,raw_logit_rows=5,raw_logit_files=705,
                 independent_CPU_raster_disagree_rows=1,independent_CPU_raster_disagree_pixels=1)
        with self.assertRaisesRegex(ValueError,"CPU"):
            validate_capture_receipt(row)
        row["independent_CPU_raster_disagree_rows"]=0
        row["independent_CPU_raster_disagree_pixels"]=0
        validate_capture_receipt(row)


if __name__=="__main__":
    unittest.main()
