"""Pure metadata tests only; no Torch/NumPy imports, models or actual inputs."""
import copy
import unittest
import train_core as core


class OrderAndCursorTests(unittest.TestCase):
    def test_complete_reproducible_private_order(self):
        import random
        before = random.getstate()
        first = core.generate_order()
        self.assertEqual(first, core.generate_order())
        self.assertEqual(before, random.getstate())
        core.validate_order(first, require_python38=False)
        self.assertEqual(len(first['order']), 37050)
        for epoch in range(50):
            self.assertEqual(sorted(first['order'][epoch*741:(epoch+1)*741]), list(range(741)))
        self.assertNotEqual(first['order'][:741], first['order'][741:1482])
        self.assertNotEqual(first['generator_initial_state'], first['generator_final_state'])

    def test_segment_cursor_and_boundaries(self):
        plan = core.generate_order()
        first = core.validate_segment(dict(seed=core.SEED,start=0,stop=743,order_sha256=plan['order_sha256']), plan)
        self.assertEqual(first, (0,743))
        self.assertEqual(core.cursor(743), dict(global_updates=743,completed_epochs=1,within_epoch=2))
        self.assertEqual(core.cursor(37050), dict(global_updates=37050,completed_epochs=50,within_epoch=0))
        self.assertEqual(core.expected_trace(plan, 743), core.append_trace(core.expected_trace(plan, 742), 742, plan['order'][742]))

    def test_reject_wrong_seed_order_budget_and_digest(self):
        with self.assertRaises(ValueError): core.generate_order(3101)
        good = core.generate_order()
        for field,value in [('seed',3101),('images',740),('epochs',49)]:
            bad = copy.deepcopy(good); bad[field]=value
            with self.assertRaises(ValueError): core.validate_order(bad,require_python38=False)
        bad=copy.deepcopy(good); bad['order'][0]=bad['order'][1]
        with self.assertRaises(ValueError): core.validate_order(bad,require_python38=False)
        bad=copy.deepcopy(good); bad['generator_final_state'][1][0] ^= 1
        with self.assertRaises(ValueError): core.validate_order(bad,require_python38=False)
        for start,stop in [(0,0),(-1,1),(0,37051),(True,2),(3,2)]:
            with self.assertRaises(ValueError):
                core.validate_segment(dict(seed=core.SEED,start=start,stop=stop,order_sha256=good['order_sha256']),good)
        with self.assertRaises(ValueError):
            core.validate_segment(dict(seed=core.SEED,start=0,stop=1,order_sha256='0'*64),good)


if __name__ == '__main__':
    unittest.main()
