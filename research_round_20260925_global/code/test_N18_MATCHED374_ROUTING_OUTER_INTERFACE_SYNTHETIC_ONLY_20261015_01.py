"""One stdlib-only synthetic outer/count-body interface check; no science IO."""
import importlib.util
import os
from pathlib import Path
import unittest
HERE=Path(__file__).parent
ENTRY=Path(os.environ.get('N18_ROUTING_OUTER_SYNTHETIC',str(HERE/'N18_FIXED_PR_MATCHED374_ROUTING_CPU_ROOT_UNBOUND_DRAFT_20261015_01.py')))
def load(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
class Interface(unittest.TestCase):
    def test_fixed_matched_bridge_and_K0_NA(self):
        outer=load(ENTRY,'_N18_outer_interface_SYNTHETIC')
        body=load(HERE/'N18_FIXED_PR_ROUTING_CAPABILITY_COUNTS_UNBOUND_20261015_01.py','_N18_counts_interface_SYNTHETIC')
        pair=dict(image_id=7,gt_id=90,query_id=4,status='native_matched_fixed_query')
        supports={'E24_LOCAL_atmost':[True,False,False],'E24_CENTER_atmost':[False,True,False],'Omega_strong_R':[True,False,False]}
        value=outer.count_pair(body,pair,[False,False,True],[True,False,True],[True,True,True],1,supports)
        self.assertEqual(value['identity'],pair);self.assertEqual((value['counts']['B'],value['counts']['H']),(1,1))
        self.assertEqual(value['rates']['routing_miss_rate']['E24_LOCAL_atmost'],0)
        self.assertEqual(value['rates']['selected_flip_harm_fraction']['E24_CENTER_atmost'],1)
        zero={key:[False,False,False] for key in supports}
        k0=outer.count_pair(body,pair,[False,False,True],[True,False,True],None,0,zero)
        self.assertIsNone(k0['rates']['PR_unrepairable_fraction']);self.assertIsNone(k0['rates']['routing_miss_rate']['E24_LOCAL_atmost'])
        aggregate=body.aggregate_counts([value['counts'],k0['counts']]);self.assertEqual(aggregate['PR_capability_E_denominator'],1)
        with self.assertRaises(RuntimeError):outer.count_pair(body,dict(pair,status='native_FN',query_id=None),[False],[True],None,0,{key:[False] for key in supports})
if __name__=='__main__':unittest.main()
