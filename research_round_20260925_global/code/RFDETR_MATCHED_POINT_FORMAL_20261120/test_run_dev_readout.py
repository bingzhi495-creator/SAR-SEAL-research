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
        calls=[]
        def count_encode(*args):
            calls.append(1)
            return codec.encode(*args)
        counted=SimpleNamespace(encode=count_encode,area=codec.area)
        out=emit_candidate(np,stock,record,refits,trainer,feat,counted,budgets)
        self.assertEqual(len(calls),2,"Two distinct nonzero pixel edits, one encoding each")
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


    def test_repair_header_only_admits_one_bound_row(self):
        from run_dev_readout import validate_capture_receipt
        capture=dict(status="RF_CENTER5202_FULL_CAPTURE_INDEPENDENTLY_ACCEPTED_V1",mode="dev",
                     images=705,raw_logit_files=705,native_mapped_rows=67386,
                     source_checkpoint_sha256="011bd30ed056c871be1d61e14d71b3b2b258330a9d20cf2f164ca4a7c4f23c11",
                     stock_predictions_sha256="da55f2d3f9bbf08b26d5fae76606b2496a5b482c5fb61f4fab53c23172e5c06e",
                     same_device_raster_exact=5,raw_logit_rows=5,model_forwards=0,original_os_exit=0,job_drained=True,
                     DEV=True,FIT=False,TEST=False,GT_opened=False,AP=False,
                     independent_CPU_raster_disagree_rows=1,independent_CPU_raster_disagree_pixels=1,
                     capture_result_sha256="a"*64,queries_sha256="b"*64)
        repair=dict(capture_result_sha256="a"*64,queries_sha256="b"*64)
        validate_capture_receipt(capture,repair)
        capture["independent_CPU_raster_disagree_pixels"]=2
        with self.assertRaises(ValueError):
            validate_capture_receipt(capture,repair)


    def test_single_probability_tensor_consumed_exactly_once(self):
        from hashlib import sha256
        from pathlib import Path
        import tempfile
        import numpy as np
        import torch
        from pycocotools import mask as codec
        from run_dev_readout import encoded, repaired_probability, repair_summary
        raw=np.zeros((144,144),dtype=np.float32)
        stock=np.zeros((800,800),dtype=bool)
        stock[0,0]=True
        packed=codec.encode(np.asfortranarray(stock,dtype=np.uint8))
        rle=dict(size=packed["size"],counts=packed["counts"].decode())
        probability=np.full((800,800),.5,dtype=np.float32)
        probability[0,0]=.6
        identity=dict(ordinal=2,image_id=17,query_id=31,native_rank=8,logit_row=4)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"prob.npy"
            np.save(path,probability,allow_pickle=False)
            source=Path(folder)/"source.npy"
            np.save(source,raw,allow_pickle=False)
            ref=dict(relative_path="RAW_LOGITS/002.npy",path=str(source),bytes=source.stat().st_size,
                     sha256=sha256(source.read_bytes()).hexdigest())
            repair=dict(identity,_consumed=0,_receipt_sha256="c"*64,source_logit_ref=ref,
                        source_raw_plane_sha256=sha256(raw.tobytes()).hexdigest(),
                        stock_rle_sha256=sha256(encoded(rle)).hexdigest(),
                        strict_cuda_logit_sign_tensor_sha256=sha256(stock.astype(np.uint8).tobytes()).hexdigest(),
                        native_probability_tensor_sha256=sha256(probability.tobytes()).hexdigest(),
                        native_probability_ref=dict(path=str(path),bytes=path.stat().st_size,
                                                    sha256=sha256(path.read_bytes()).hexdigest()))
            out=repaired_probability(np,torch,raw,stock,rle,identity,ref,repair)
            np.testing.assert_array_equal(out,probability)
            self.assertEqual(repair_summary(dict(independent_CPU_raster_disagree_rows=1,
                             independent_CPU_raster_disagree_pixels=1),repair)["probability_repair_consumed"],1)
            with self.assertRaises(ValueError):
                repaired_probability(np,torch,raw,stock,rle,identity,ref,repair)

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
