"""Synthetic arrays only. Set FIT109_PUBLISHED_CODE to the ordinary published code directory."""
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
import os
from pathlib import Path
import unittest

import numpy as np
import torch


def load(name, path):
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HERE = Path(__file__).resolve().parent
CODE = Path(os.environ.get("FIT109_PUBLISHED_CODE", str(HERE.parent)))
heads = load("new_heads", HERE / "fit109_heads.py")
plan = load("old_plan", CODE / "RFDETR_MATCHED_POINT_FORMAL_20261120/fit_plan.py")
step = load("old_step", CODE / "RFDETR_CORE36_MATCHED_POINT_SPOT_20261120/fit_step.py")
reference = load("old_edits", CODE / "RFDETR_CORE36_MATCHED_POINT_SPOT_20261120/feature_adapter.py")
trainer = step.load_trainer(CODE / "DFINE_SEG_EXTERNAL_BASELINE_FIT32_E24_20261112/e24/fit109_paired_point/train_point.py")


class Synthetic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(17)
        cls.group_ids = [f"synthetic-{i}" for i in range(109)]
        cls.features, cls.labeled, cls.old = [], [], {}
        assignment = plan.source_folds(cls.group_ids)
        for i, group in enumerate(cls.group_ids):
            n = i % 3
            old = rng.normal(size=(n, 2, 14)).astype(np.float32)
            old[:, :, 0] = rng.random(size=(n, 2))
            coords = np.arange(n*2, dtype=np.int32).reshape(n, 2)
            arms = {}
            for name in heads.ARMS:
                extra = rng.normal(size=(n, 2, 16)).astype(np.float32)
                if name == "zero":
                    extra.fill(0)
                arms[name] = np.concatenate((old, extra), axis=2)
            row = dict(group=group, image_id=i, rank=1, query_id=i % 10, old14=old, cap=n,
                       features=arms, coordinates=coords)
            cls.features.append(row)
            cls.labeled.append(dict(row, truth=None if i % 7 == 0 else rng.random(size=(n, 2))))
        for f in range(5):
            selected = [dict(r, features=r["old14"]) for r in cls.labeled if assignment[r["group"]] != f]
            _, _, center, scale, groups = step.prepared(selected, trainer)
            # Independent transcription of the historic index stream.
            gen = np.random.Generator(np.random.PCG64(20261112))
            draws = gen.integers(len(groups), size=(200, 64))
            batches = [[groups[g][gen.integers(len(groups[g]))] for g in row] for row in draws]
            old_sha = sha256(np.asarray(batches, dtype="<i8").tobytes()).hexdigest()
            torch.manual_seed(5217)
            model = trainer.PointHead(14).eval()
            cls.old[f] = dict(model=model, center=center, scale=scale, seed=5217, fold=f,
                              initial_tensor_sha256=step.tensor_hash(model), common_batch_sha256=old_sha)
        cls.budgets = heads.seal_oof_budgets(cls.features, cls.group_ids, cls.old, step, plan)

    def test_initial_lift(self):
        old = self.old[0]["model"]
        new = heads._initial_model(trainer, step, step.tensor_hash(old))
        w, source = new.network[0].weight, old.network[0].weight
        self.assertTrue(torch.equal(w[:, :14], source[:, :14]))
        self.assertTrue(torch.equal(w[:, 30:44], source[:, 14:]))
        self.assertEqual(torch.count_nonzero(w[:, 14:30]) + torch.count_nonzero(w[:, 44:]), 0)
        for i in (0, 2, 4):
            self.assertTrue(torch.equal(new.network[i].bias, old.network[i].bias))
            if i:
                self.assertTrue(torch.equal(new.network[i].weight, old.network[i].weight))

    def test_selector_matches_native_with_ties_and_zero_gain(self):
        rng = np.random.default_rng(119)
        for n in (0, 1, 8, 32):
            coords = np.arange(2*n, dtype=np.int32).reshape(n, 2)
            stock = np.ones((12, 12), dtype=bool)
            stock.reshape(-1)[coords[:, 0]] = False
            for _ in range(9):
                q = rng.choice([.125, .5, .875], size=(n, 2)).astype(np.float32)
                cap = min(8, n, int(stock.sum()) // 10)
                expected = reference.readouts_sparse(np, stock, coords, q, q, q, cap)["SU"]
                actions = heads.free_selection(q, cap)
                got = np.column_stack((coords[actions[:, 0], 0], coords[actions[:, 1], 1])).ravel()
                np.testing.assert_array_equal(got, expected["flipped"])

    def test_prelabel_and_alignment_guards(self):
        with self.assertRaisesRegex(ValueError, "Label-derived"):
            heads.seal_oof_budgets(self.labeled, self.group_ids, self.old, step, plan)
        with self.assertRaisesRegex(ValueError, "order changed"):
            next(heads.fit_oof(self.labeled[::-1], self.group_ids, self.old, trainer, step, plan, self.budgets))
        invalid = self.old.copy()
        invalid[0] = dict(invalid[0], fold=1)
        with self.assertRaisesRegex(ValueError, "fold/seed"):
            heads.seal_oof_budgets(self.features, self.group_ids, invalid, step, plan)
        invalid[0] = dict(self.old[0], common_batch_sha256="bad")
        with self.assertRaisesRegex(ValueError, "sampled pair indices"):
            next(heads.fit_oof(self.labeled, self.group_ids, invalid, trainer, step, plan, self.budgets))
        bad = self.labeled.copy()
        changed = dict(bad[1]["features"])
        changed["ordinary_centered"] = changed["ordinary_centered"].copy()
        changed["ordinary_centered"][0, 0, 17] += 1
        bad[1] = dict(bad[1], features=changed)
        with self.assertRaisesRegex(ValueError, "budget mismatch"):
            next(heads.fit_oof(bad, self.group_ids, self.old, trainer, step, plan, self.budgets))

    def test_all_folds_shared_schedule_and_aligned_predictions(self):
        observed = set()
        for fold, held, audit in heads.fit_oof(self.labeled, self.group_ids, self.old, trainer, step,
                                              plan, self.budgets):
            self.assertEqual(set(audit["heads"]), set(heads.ARMS))
            self.assertFalse(set(audit["training_groups"]) & set(audit["held_groups"]))
            for w in audit["heads"].values():
                self.assertEqual(w["optimizer_steps"], 200)
                self.assertEqual(w["common_batch_sha256"], self.old[fold]["common_batch_sha256"])
            for r in held:
                key = heads.identity(r)
                self.assertNotIn(key, observed)
                observed.add(key)
                old = self.budgets[key]
                self.assertEqual(r["k"], old["k"])
                for name in heads.ARMS:
                    q, flipped = r["probabilities"][name], r["flipped"][name]
                    self.assertEqual(q.shape, r["coordinates"].shape)
                    self.assertEqual(q.dtype, np.float32)
                    self.assertEqual(len(np.unique(flipped)), 2*r["k"])
                    self.assertTrue(np.isin(flipped[::2], r["coordinates"][:, 0]).all())
                    self.assertTrue(np.isin(flipped[1::2], r["coordinates"][:, 1]).all())
        self.assertEqual(observed, set(self.budgets))


if __name__ == "__main__":
    unittest.main()
