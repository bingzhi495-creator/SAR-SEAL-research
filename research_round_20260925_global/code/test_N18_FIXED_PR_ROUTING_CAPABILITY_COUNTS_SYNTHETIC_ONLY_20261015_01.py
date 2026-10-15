# UNBOUND_DRAFT — hand-built boolean fixtures only; no scientific files.
import importlib.util
from pathlib import Path
import unittest

p = Path(__file__).with_name('N18_FIXED_PR_ROUTING_CAPABILITY_COUNTS_UNBOUND_20261015_01.py')
s = importlib.util.spec_from_file_location('n18_counts_synthetic', p)
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)


class HandCounts(unittest.TestCase):
    def test_hand_example(self):
        # E={0,1}, B={0,1}, H={2,3}, actual flips={0,1,2,3}.
        p = [True, False, True, False, True, False]
        g = [False, True, True, False, True, False]
        a = [False, True, False, True, True, False]
        supports = {
            'E24_LOCAL_atmost': [True, False, True, False, False, False],
            'E24_CENTER_atmost': [False, True, False, False, True, False],
            'Omega_strong_R': [False, False, True, True, False, False],
        }
        r = m.count_query(p, g, a, 2, supports)
        self.assertEqual((r['E'], r['B'], r['H'], r['actual_action_flips']), (2, 2, 2, 4))
        self.assertEqual(r['budget_repairable_denominator'], 2)
        for name, expected in [('E24_LOCAL_atmost', (2, 1, 1, 2, 0)),
                               ('E24_CENTER_atmost', (2, 1, 0, 1, 1)),
                               ('Omega_strong_R', (2, 0, 2, 2, 0))]:
            v = r['selections'][name]
            self.assertEqual(tuple(v[k] for k in ('selected_support', 'helpful_selected',
                             'harmful_selected', 'actual_flips', 'selected_noop')), expected)
        self.assertEqual(r['selections']['E24_LOCAL_atmost']['helpful_over_budget_repairable'], .5)

        # K=0 can have nonempty C and native errors, but no PR capability.
        empty_supports = {name: [False] * 6 for name in m.POLICIES}
        k0 = m.count_query(p, g, None, 0, empty_supports)
        self.assertEqual(k0['E'], 2)
        self.assertIsNone(k0['B'])
        self.assertIsNone(k0['PR_capability_E_denominator'])
        self.assertIsNone(k0['budget_repairable_denominator'])
        z = m.count_query([True], [True], [True], 1,
                          {name: [True] for name in m.POLICIES})
        self.assertEqual((z['E'], z['B'], z['H']), (0, 0, 0))
        self.assertIsNone(z['PR_repairable_fraction'])
        self.assertIsNone(z['selections']['Omega_strong_R']['helpful_over_budget_repairable'])
        micro = m.aggregate_counts([r, k0, z])
        self.assertEqual((micro['matched_queries'], micro['PR_observed_queries'],
                          micro['K0_queries'], micro['E_all_queries'],
                          micro['PR_capability_E_denominator'], micro['B'], micro['H']),
                         (3, 2, 1, 4, 2, 2, 2))
        self.assertEqual(micro['budget_repairable_denominator'], 2)
        self.assertEqual(micro['selections']['E24_LOCAL_atmost']['helpful_over_budget_repairable'], .5)
        self.assertIsNone(m.aggregate_counts([k0])['PR_repairable_fraction'])
        self.assertIsNone(m.aggregate_counts([])['budget_repairable_denominator'])
        bad_supports = dict(supports, E24_LOCAL_atmost=[1, 0, 1, 0, 0, 0])
        bad_length = dict(supports, E24_CENTER_atmost=[True])
        bad_omega = dict(supports, Omega_strong_R=[False] * 6)
        for args in [(p, g, a, True, supports), (p, g, a, 2, bad_supports),
                     (p, g, a, 2, bad_length), (p, g, a, 2, bad_omega),
                     (p, g, a, 0, empty_supports), (p, g, None, 2, supports)]:
            with self.assertRaises(ValueError):
                m.count_query(*args)


if __name__ == '__main__':
    unittest.main()
