"""Regression checks for the frozen action anchor and source weighting."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

import fit
import policies


def candidate():
    moves = np.asarray([[0,2],[1,1],[2,0]],dtype=np.int16)
    features = np.array([[-.02,.02,-.006,.003,-.001,100,.9,.021,1],
                         [0,0,0,0,0,100,.9,.021,1],
                         [.02,.02,.007,-.002,.002,100,.9,.021,1]])
    return moves,features


class LearnTests(unittest.TestCase):
    def test_balanced_anchor_and_simple_controls(self):
        moves, features = candidate()
        coeffs = np.zeros((3,25))
        model = policies.Model(np.ones(25),coeffs,-8)
        self.assertEqual(policies.select_actions(model,1,moves,features),
                         dict(A=1,M=1,G=1,B=1,R=2,J=0,C=0))
        self.assertTrue(np.all(policies.basis(features)[1] == 0))
        self.assertEqual(policies.best_index(np.array([1.,0.,1.]),moves[:,0],1),0)
        self.assertEqual(policies.select_actions(model,0,np.array([[0,0]]),
                         np.array([[0,0,0,0,0,100,.9,.021,0]])),dict.fromkeys(policies.ARM_NAMES,0))

    def test_weighted_ridge_keeps_labels_and_source_structure(self):
        ids = np.array([[5217,7,1,1],[5217,7,2,2],[5218,7,1,1],[5219,7,1,1]])
        actual = fit.candidate_weights(ids,np.array([True,True,True,False]))
        np.testing.assert_array_equal(actual,np.array([.25,.25,.5,0]))
        X = np.ones((4,1))
        y = np.array([0.,2.,1.,np.nan])
        coef, error = fit.fit_ridge(X,y,actual,.05)
        np.testing.assert_allclose(coef,np.array([1/1.05]),atol=1e-12,rtol=0)
        self.assertGreater(error,0)

    def test_source_channels_require_complete_action_alignment(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            for tag in ("FEATURE_SHARDS_PRIVATE","TARGET_SHARDS_PRIVATE","COMPETITION_SHARDS_PRIVATE"):
                (base/tag).mkdir()
            moves, features = candidate()
            # The second candidate edits two pixels with its only feasible allocation.
            moves = np.vstack((moves,moves[1]))
            features = np.vstack((features,features[1]))
            identity = np.array([[5217,37,1,5],[5217,37,2,9]],dtype=np.int64)
            offsets = np.array([0,3,4],dtype=np.int16)
            ref = dict(file="001.npz",candidate_rows=2,actions=4,image_id=37)
            geo = dict(supervised_candidate_seed_rows=2,supervised_actions=4)
            ap = {}

            def save(dir_, **arrays):
                path = base/dir_/ref["file"]
                np.savez(path,**arrays)
                return fit.sha256(path.read_bytes()).hexdigest()

            ref["sha256"] = save("FEATURE_SHARDS_PRIVATE",identity=identity,actions=moves,features=features,
                                 feature_names=np.asarray(policies.FEATURES),action_offsets=offsets,
                                 balanced_count=np.array([1,1],dtype=np.int16))
            geo["target_sha256"] = save("TARGET_SHARDS_PRIVATE",supervised_candidate=np.array([True,True]),
                                        delta_iou=np.array([-.04,0.,.05,0.]))
            values = np.array([.001,0.,-.002,0.])
            ap["sha256"] = save("COMPETITION_SHARDS_PRIVATE",identity=identity,actions=moves,action_offsets=offsets,
                                delta_ap=values,delta_ap_by_iou=np.repeat(values[:,None],10,axis=1))
            block, counts, _ = fit.load_shard(base,base,ref,geo,ap,policies)
            np.testing.assert_array_equal(block[2],values[[0,2]])
            np.testing.assert_array_equal(block[1],np.full((2,2),.5))
            self.assertEqual(counts["matched_trainable_actions"],2)
            self.assertEqual(counts["nonzero_budget_single_action_rows"],1)
            ap["sha256"] = save("COMPETITION_SHARDS_PRIVATE",identity=identity,actions=moves[::-1],action_offsets=offsets,
                                delta_ap=values,delta_ap_by_iou=np.repeat(values[:,None],10,axis=1))
            with self.assertRaisesRegex(ValueError,"original feature action identities"):
                fit.load_shard(base,base,ref,geo,ap,policies)


if __name__ == "__main__":
    unittest.main()
