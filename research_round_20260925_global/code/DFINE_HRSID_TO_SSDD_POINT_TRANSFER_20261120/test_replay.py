"""Synthetic sidecar and native mask score contracts."""
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

import ssdd_replay as replay
import ssdd_score as score


class NativeReplayTest(unittest.TestCase):
    def test_frozen_head_and_native_point_masks(self):
        root=Path(os.environ["REFIT_ROOT"])
        pair=Path(os.environ["PAIR_ROOT"])
        args=SimpleNamespace(
            refit=root/"frozen_refit.py",refit_result=root/"ACTUAL/RESULT_SAFE.json",
            refit_acceptance=root/"INDEPENDENT/RESULT_SAFE.json",weights=root/"ACTUAL/POINT_HEAD_PRIVATE.npz",
            common=Path(os.environ["COMMON_ROOT"])/"equal_count.py",
            gray=Path(os.environ["GRAY_ROOT"])/"gray_ablation.py",pair_math=pair/"pair_math.py",
            rank_math=Path(os.environ["RANK_ROOT"])/"rank_math.py",
            free=Path(os.environ["FREE_ROOT"])/"free_pairing.py",training=pair/"train_point.py")
        old=replay.load(Path(os.environ["OLD_ROOT"])/"dev_endpoint.py",replay.OLD,"_closed_source_only")
        native=replay.load(Path(os.environ["DRAFT"])/"native_points.py",replay.NATIVE,"_native_source_only")
        original,rank,free,learner,head,center,scale=old.restore(args)
        original.raw=replay.pin(args.pair_math,native.PAIR)
        n,r=native.make_native(original,rank,193,214)
        from pycocotools import mask as codec
        from collections import Counter
        stock=np.zeros((193,214),dtype=np.uint8)
        stock[40:48,60:68]=1
        packed=codec.encode(np.asfortranarray(stock))
        packed["counts"]=packed["counts"].decode()
        row=dict(image_id=1,category_id=1,score=.9,segmentation=packed)
        audit=dict(native_rank=1,score=.9,mask_area=64,box_xyxy=[50.,30.,80.,60.])
        result=old.image_edits([row],[audit],np.full((1,160,160),.5,dtype=np.float32),
                              original.image_features(np.zeros((193,214,3),dtype=np.uint8)),
                              (n,r,free,learner,head,center,scale),Counter())
        for modes in result.values():
            for item in modes.values():
                old.validate_edit(packed,item["rle"],64,item["swaps"])
        self.assertEqual(score.load_worker(Path(os.environ["DRAFT"])/"ssdd_replay.py").PROTOCOL,replay.PROTOCOL)


if __name__=="__main__":
    unittest.main()
