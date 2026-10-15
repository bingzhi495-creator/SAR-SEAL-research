"""One stdlib metadata/main-rule check; no data, logits, model or COCO execution."""
import importlib.util
import os
from pathlib import Path
import unittest
HERE=Path(__file__).parent
ENTRY=Path(os.environ.get('N19_OUTER_SYNTHETIC',str(HERE/'N19_FIXED_HALF_READOUT_CPU_ROOT_UNBOUND_DRAFT_20261015_01.py')))
class Metadata(unittest.TestCase):
    def test_explicit137_owner_and_only_DIR_main(self):
        d=importlib.util.spec_from_file_location('_N19_outer_SYNTHETIC',ENTRY);m=importlib.util.module_from_spec(d);d.loader.exec_module(m)
        stats=lambda v:dict(stats=[v]+[0.]*11)
        baseline={'E24_UTILITY_atmost':stats(.67),'E24_ERROR_atmost':stats(.68),'strong_fixed_E24_Omega':stats(.6809938494745545)}
        results={'DIR_HALF':stats(.70),'GLOBAL_HALF':stats(.71),'REVERSED_HALF':stats(.65)}
        value=m.main_result(results,baseline);self.assertEqual(value['decision'],'STOP');self.assertAlmostEqual(value['Delta'],-1.)
        results['DIR_HALF']=stats(.73);self.assertEqual(m.main_result(results,baseline)['decision'],'EXPLORATORY_POSITIVE_REQUIRES_FRESH_PRELOCK')
        results['REVERSED_HALF']=stats(-1);self.assertIsNone(m.main_result(results,baseline)['Delta'])
        spec=dict(schema='ROOT-N19-fixedDEV-full187-fixed-half-readout-CPU-spec-v1',entry_source=dict(path=str(ENTRY.resolve())),normal_RAW_count=137,
            expected_normal_RAW_count=137,total_failed_RAW_count=21,total_terminal_identity_pin_count=158,maximum_process_RSS_bytes=16*1024**3,
            deadline_seconds=1800,automatic_retry=False,allocator_limit_MiB=4096,whole_GPU_limit_MiB=8192,cpu_shard_owner_index=127)
        m.probe_header(spec)
        with self.assertRaises(RuntimeError):m.probe_header(dict(spec,normal_RAW_count=136,expected_normal_RAW_count=136,total_terminal_identity_pin_count=157))
if __name__=='__main__':unittest.main()
