"""SYNTHETIC_ONLY stdlib fixtures; no actual result/Source/science IO."""
import copy
import importlib.util
import math
from pathlib import Path
import unittest
p=Path(__file__).with_name('N16_FIRST_SEED_RESULT_INTERPRETER_UNBOUND_DRAFT_20261014_01.py')
s=importlib.util.spec_from_file_location('_n16_synthetic_interpreter',p)
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def fixture(U=.8,E=.79):
    r={name:dict(stats=[.5]*12) for name in m.POLICIES}
    r['strong_fixed_E24_Omega']['stats'][0]=m.REFERENCE_AP_PERCENT/100
    r['E24_UTILITY_atmost']['stats'][0]=U;r['E24_ERROR_atmost']['stats'][0]=E
    return r
class ResultTests(unittest.TestCase):
    def test_exact_one_and_adjacent_float_boundaries(self):
        value=m.interpret_first_seed(fixture())
        self.assertEqual(value['decision'],'GO');self.assertEqual(value['Delta_AP_points'],1.)
        self.assertEqual(value['allowed_extra_seeds'],[202610140162,202610140163])
        low=m.interpret_first_seed(fixture(math.nextafter(80.,0.)/100))
        high=m.interpret_first_seed(fixture(math.nextafter(80.,100.)/100))
        self.assertEqual(low['decision'],'ONE_EXTRA_SEED');self.assertEqual(low['allowed_extra_seeds'],[202610140162])
        self.assertLess(low['Delta_AP_points'],1.)
        self.assertEqual(high['decision'],'GO');self.assertGreater(high['Delta_AP_points'],1.)
    def test_zero_negative_and_fixed_reference_stop(self):
        for U,E in ((.79,.79),(.78,.79),(.68,.60)):
            value=m.interpret_first_seed(fixture(U,E))
            self.assertEqual(value['decision'],'STOP');self.assertEqual(value['allowed_extra_seeds'],[])
        value=m.interpret_first_seed(fixture(.70,.695))
        self.assertEqual(value['decision'],'ONE_EXTRA_SEED')
    def test_secondary_winners_never_replace_primary(self):
        r=fixture(.78,.79)
        for name in m.POLICIES:
            if name not in ('E24_UTILITY_atmost','E24_ERROR_atmost'):r[name]['stats'][0]=.99
        value=m.interpret_first_seed(r)
        self.assertEqual(value['decision'],'STOP');self.assertEqual(value['UTILITY_AP_percent'],78.)
        self.assertEqual(value['ERROR_AP_percent'],79.)
    def test_complete_original_precision_table_and_undefined_metric(self):
        r=fixture();r['E24_UTILITY_atmost']['stats'][1]=.8123456789012345
        r['E24_UTILITY_atmost']['stats'][2]=.7234567890123456
        r['E24_UTILITY_atmost']['stats'][8]=.9345678901234567
        r['E8_UTILITY_atmost']['stats'][2]=-1
        original=copy.deepcopy(r);value=m.interpret_first_seed(r)
        self.assertEqual(r,original);self.assertEqual([v['policy'] for v in value['table']],list(m.POLICIES))
        rows={v['policy']:v for v in value['table']}
        self.assertEqual(rows['E24_UTILITY_atmost']['AP50'],r['E24_UTILITY_atmost']['stats'][1]*100)
        self.assertEqual(rows['E24_UTILITY_atmost']['AP75'],r['E24_UTILITY_atmost']['stats'][2]*100)
        self.assertEqual(rows['E24_UTILITY_atmost']['AR100'],r['E24_UTILITY_atmost']['stats'][8]*100)
        self.assertIsNone(rows['E8_UTILITY_atmost']['AP75'])
        self.assertIn('AP75',rows['E8_UTILITY_atmost']['undefined_metrics'])
    def test_undefined_main_does_not_invent_stop_or_go(self):
        r=fixture();r['E24_UTILITY_atmost']['stats'][0]=-1
        value=m.interpret_first_seed(r)
        self.assertEqual(value['decision'],'RULE_NOT_EVALUABLE');self.assertIsNone(value['Delta_AP_points'])
        self.assertEqual(value['allowed_extra_seeds'],[])
    def test_nan_invalid_units_missing_count_and_boolean_reject(self):
        for scalar in (float('nan'),float('inf'),-.1,1.0001,79.,True,'0.8'):
            r=fixture();r['E8_ERROR_exact']['stats'][11]=scalar
            with self.assertRaises((TypeError,ValueError)):m.interpret_first_seed(r)
        r=fixture();r.pop('E24_MAGNET_atmost')
        with self.assertRaises((TypeError,ValueError)):m.interpret_first_seed(r)
        r=fixture();r['E24_UTILITY_atmost']['stats'].pop()
        with self.assertRaises((TypeError,ValueError)):m.interpret_first_seed(r)
        with self.assertRaises((TypeError,ValueError)):m.interpret_first_seed(fixture(),m.REFERENCE_AP_PERCENT/100)
if __name__=='__main__':unittest.main(verbosity=2)
