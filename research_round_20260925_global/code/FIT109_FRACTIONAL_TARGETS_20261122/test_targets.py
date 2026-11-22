import argparse
from pathlib import Path
from runpy import run_path
from types import SimpleNamespace
import unittest

import numpy as np
import torch


cli = argparse.ArgumentParser()
cli.add_argument("--pilot", type=Path, required=True)
cli.add_argument("--classical", type=Path, required=True)
cli.add_argument("--helper", type=Path, required=True)
cli.add_argument("--helper-sha256", required=True)
args, rest = cli.parse_known_args()
worker = run_path(str(Path(__file__).with_name("compare_targets.py")))


class FractionalTargetFixture(unittest.TestCase):
    def test_exact_matched_identity_and_inclusive_half_threshold(self):
        gt = dict(
            images=[dict(id=7, width=8, height=6)],
            categories=[dict(id=1, name="ship")],
            annotations=[
                dict(id=1,image_id=7,category_id=1,iscrowd=0,segmentation=[[1,1,3,1,3,3,1,3]]),
                dict(id=2,image_id=7,category_id=1,iscrowd=0,segmentation=[[3,1,5,1,5,3,3,3]]),
            ],
        )
        original = np.array([[.5, .49999999], [1., 0.]])
        coords = np.array([[9,10],[17,20]])
        # The first annotation covers the first three cells. The second covers the fourth.
        slot = dict(row=dict(image_id=7,segmentation=dict(size=[6,8])),gt_id=1,role="valid",
                    truth=original,data=dict(coordinates=coords))
        arms, counters = worker["target_populations"]([slot], gt)
        np.testing.assert_array_equal(arms["fractional"][0]["truth"], original)
        np.testing.assert_array_equal(arms["threshold50"][0]["truth"], [[1,0],[1,0]])
        np.testing.assert_array_equal(arms["coco_raster"][0]["truth"], [[1,1],[1,0]])
        self.assertEqual(counters["exact_half"], 1)
        self.assertEqual(counters["threshold_raster_disagreements"], 1)

    def test_seed_condition_pairs_reset_identically_and_updates_differ_by_target(self):
        torch.set_num_threads(1)
        prior = worker["module"](args.classical, worker["CLASSICAL"])
        base = prior.import_checked(args.pilot, prior.PILOT_SHA)
        a = SimpleNamespace(helper=args.helper, helper_sha256=args.helper_sha256, private_root=base.PRIVATE, code=base.CODE)
        m = base.sources(a)
        m.learner.HEADS = ("full",)
        rng = np.random.default_rng(55)
        f = rng.uniform(.1,.9,(3,2,14)).astype(np.float32)
        slots = [dict(fold=i,group=str(i),truth=np.array([[.1,.65],[.6,.8],[.4,.05]]),
                      data={"features": f+i*.015}) for i in range(3)]
        hard = [dict(s,truth=(s["truth"]>=.5).astype(np.float32)) for s in slots]
        init = set()
        draws = set()
        for seed, sampler in worker["SEEDS"]:
            train = worker["seed_trainer"](a,m.learner,base.REFERENCES["training"][1],seed,sampler)
            _, center, scale, frac = train(slots,0)
            _, center2, scale2, control = train(hard,0)
            np.testing.assert_array_equal(center,center2)
            np.testing.assert_array_equal(scale,scale2)
            fa, ca = frac["heads"]["full"], control["heads"]["full"]
            self.assertEqual(fa["steps"],200)
            self.assertEqual(fa["parameters"],1321)
            for key in ("initial_tensor_sha256","batch_sha256"):
                self.assertEqual(fa[key],ca[key])
            self.assertNotEqual(fa["last_batch_loss"],ca["last_batch_loss"])
            init.add(fa["initial_tensor_sha256"])
            draws.add(fa["batch_sha256"])
        self.assertEqual(len(init),3)
        self.assertEqual(len(draws),3)


if __name__ == "__main__":
    unittest.main(argv=[__file__, *rest])
