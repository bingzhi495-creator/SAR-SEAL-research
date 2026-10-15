# SYNTHETIC_ONLY: policy behavior; no real results/data/Torch/qualification.
import runpy,unittest,sys
m=runpy.run_path(r'E:/SAR/seal/tmp/N20_FIXED_FIRST_SEED_MAIN_RESULT_INTERPRETER_UNBOUND_DRAFT_20261015_01.py',run_name='_pure_N20_decision')
f=m['interpret_fixed_main'];R=68.09938494745545
def row(local,center):return dict(seed=202610150201,epoch=24,mode='atmost',AP_unit='percent',LOCAL_AP=local,CENTER_AP=center,strong_AP=R)
class MainDecision(unittest.TestCase):
 def test_main_strongest_comparator_and_budget_boundaries(self):
  for l,c,expected,seeds in [(68.5,69.,'STOP',[]),(R,68.,'STOP',[]),(R+.5,68.,'ONE_EXTRA_SEED',[202610150202]),(R+1.,68.,'TWO_EXTRA_SEEDS',[202610150202,202610150203]),(-1.,68.,'RULE_NOT_EVALUABLE',[])]:
   with self.subTest(local=l,center=c):
    v=f([row(l,c)]);self.assertEqual(v['decision'],expected);self.assertEqual(v['allowed_next_seeds'],seeds)
    for k in ('qualification_performed','task_efficacy_established','innovation_established','TGRS_readiness_established','secondary_results_may_replace_main'):self.assertIs(v[k],False)
 def test_wrong_seed_endpoint_units_or_reference_rejected(self):
  for key,value in [('seed',202610150202),('epoch',8),('mode','exact'),('AP_unit','fraction'),('strong_AP',67.),('LOCAL_AP',True)]:
   with self.subTest(key=key):
    v=row(69.,68.);v[key]=value
    with self.assertRaises(ValueError):f([v])
 def test_no_future_repeats_or_extra_fields_accepted(self):
  with self.assertRaises(ValueError):f([row(69.,68.),row(69.,68.)])
  v=row(69.,68.);v['E8_AP']=99.
  with self.assertRaises(ValueError):f([v])
if __name__=='__main__':
 assert 'torch' not in sys.modules
 unittest.main()
