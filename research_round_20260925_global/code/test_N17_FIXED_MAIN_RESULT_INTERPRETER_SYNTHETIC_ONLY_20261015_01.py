"""Pure stdlib hand AP fixtures only; no scientific inputs, qualification or runs."""
import importlib.util
import math
from pathlib import Path
import unittest
ENTRY=Path(__file__).with_name('N17_FIXED_MAIN_RESULT_INTERPRETER_UNBOUND_DRAFT_20261015_01.py')
class Contract(unittest.TestCase):
    def test_locked_thresholds_repeat_permissions_and_rejections(self):
        spec=importlib.util.spec_from_file_location('_n17_main_result',ENTRY);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        R=m.STRONG_AP
        def row(seed=202610150171,L=R,C=R-1,**updates):
            result=dict(seed=seed,epoch=24,mode='atmost',AP_unit='percent',LOCAL_AP=L,CENTER_AP=C,strong_AP=R);result.update(updates);return result
        for L,decision,seeds in ((R,'STOP',[]),(R-1,'STOP',[]),(R+.5,'ONE_EXTRA_SEED',[202610150172]),
            (R+1-1e-10,'ONE_EXTRA_SEED',[202610150172]),(R+1,'TWO_EXTRA_SEEDS',[202610150172,202610150173])):
            result=m.interpret_fixed_main([row(L=L)]);self.assertEqual(result['decision'],decision);self.assertEqual(result['allowed_next_seeds'],seeds)
        result=m.interpret_fixed_main([row(L=R+.5),row(202610150172,R+1.5)])
        self.assertEqual(result['decision'],'ALLOW_THIRD_SEED');self.assertEqual(result['allowed_next_seeds'],[202610150173])
        result=m.interpret_fixed_main([row(L=R+.5),row(202610150172,R+.6)])
        self.assertEqual(result['decision'],'STOP');self.assertFalse(result['accepted_under_locked_rule'])
        # Gate is delta of means, not mean of individually max-referenced deltas.
        first=row(L=R+.5,C=R);second=row(202610150172,R+3,C=R+1.5)
        result=m.interpret_fixed_main([first,second]);self.assertAlmostEqual(result['Delta_AP_points'],1);self.assertEqual(result['decision'],'ALLOW_THIRD_SEED')
        wins=[row(L=R+1),row(202610150172,R+1),row(202610150173,R+1)]
        self.assertTrue(m.interpret_fixed_main(wins)['accepted_under_locked_rule'])
        fails=[row(L=R+2),row(202610150172,R+2),row(202610150173,R+2,C=R+2)]
        self.assertEqual(m.interpret_fixed_main(fails)['decision'],'STOP')
        first_go=[row(L=R+1),row(202610150172,R-1)]
        self.assertEqual(m.interpret_fixed_main(first_go)['allowed_next_seeds'],[202610150173])
        for bad in (row(202610150170),row(epoch=8),row(mode='exact'),row(AP_unit='fraction'),row(L=math.nan),row(C=math.inf),
            row(L=True),row(L=101),row(strong_AP=.6809938494745545),dict(row(),E8_LOCAL_AP=100),{k:v for k,v in row().items() if k!='strong_AP'}):
            with self.subTest(bad=bad),self.assertRaises((TypeError,ValueError)):m.interpret_fixed_main([bad])
        for bad in ([row(),row(202610150172,R+1)], [row(L=R+.5),row(202610150172,R+.5),row(202610150173,R+3)],
            [row(L=R+1),row(202610150173,R+1)], [row(L=-1),row(202610150172,R+1)]):
            with self.assertRaises(ValueError):m.interpret_fixed_main(bad)
        undefined=m.interpret_fixed_main([row(L=-1)]);self.assertEqual(undefined['decision'],'RULE_NOT_EVALUABLE');self.assertIsNone(undefined['Delta_AP_points'])
        self.assertEqual(m.interpret_fixed_main([row(L=.5,C=.4)])['LOCAL_AP_percent'],.5)
        self.assertFalse(m.interpret_fixed_main(wins)['qualification_performed'])
if __name__=='__main__':unittest.main(verbosity=2)
