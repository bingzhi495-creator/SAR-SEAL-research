"""SYNTHETIC_ONLY: fake JSON refs/metadata; no scientific artifact IO."""
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

root=Path(__file__).parent
path=root/'N16_ALL100_FIT_CACHE_CPU_ROOT_UNBOUND_DRAFT_20261014_04.py'
if not path.exists():
    path=root/'N16_ALL100_FIT_CACHE_CPU_ROOT_UNBOUND_DRAFT_20261014_03.py'
spec=importlib.util.spec_from_file_location('_n16_fake_lineage_bridge',path)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def ref(name):
    return dict(path='E:/SYNTHETIC_ONLY/'+name,bytes=1,sha256='0'*64)

def fixture():
    refs={name:ref(name+'.json') for name in ('n07report','n07spec','n07source','n13spec','context','teacher','FIT')}
    old=dict(source=refs['n07source'],spec=refs['n07spec'],images=741,raw_GT=1661,
             packet_refs=[None]*741,rows=[None]*741)
    region=dict(spec=refs['n13spec'],teacher_checkpoint=refs['teacher'])
    data={refs['n07report']['path']:old,
          refs['n13spec']['path']:dict(original_cache_report=refs['n07report'],
             original_cache_spec=refs['n07spec'],original_cache_entry_source=refs['n07source']),
          refs['n07spec']['path']:dict(context_GPU_spec=refs['context']),
          refs['context']['path']:dict(FIT=refs['FIT'],checkpointmap=dict(native0=refs['teacher']))}
    cpu=dict(original_fit_cache_report=refs['n07report'])
    gate=dict(cache_report=region)
    evidence=dict(report=dict(teacher_checkpoint=refs['teacher']))
    closure={r['path']:r for r in refs.values()}
    return closure,cpu,gate,evidence,data

def call(args):
    closure,cpu,gate,evidence,data=args
    def fake_closed(c,r):
        module.require(c.get(r['path'])==r,'Synthetic closure differs')
        return r['path']
    with patch.object(module,'closed',side_effect=fake_closed),patch.object(module,'read',side_effect=lambda p:data[str(p)]):
        return module.cache_lineage_bridge(closure,cpu,gate,evidence)

class BridgeTests(unittest.TestCase):
    def test_distinct_N07_and_N13_reports_bridge_and_N07_has_no_teacher_field(self):
        args=fixture()
        old=args[-1][args[1]['original_fit_cache_report']['path']]
        self.assertNotEqual(old,args[2]['cache_report'])
        self.assertNotIn('teacher_checkpoint',old)
        self.assertTrue(call(args)['teacher_checkpoint']==args[3]['report']['teacher_checkpoint'])

    def test_wrong_N07_report_source_or_spec_is_rejected(self):
        for key in ('original_cache_report','original_cache_entry_source','original_cache_spec'):
            with self.subTest(key=key):
                args=fixture()
                args[-1][args[2]['cache_report']['spec']['path']][key]=ref('wrong')
                with self.assertRaises(RuntimeError):
                    call(args)

    def test_each_wrong_teacher_path_is_rejected(self):
        for location in ('context','region','completed'):
            with self.subTest(location=location):
                args=fixture()
                if location=='context':
                    context=next(v for v in args[-1].values() if 'checkpointmap' in v)
                    context['checkpointmap']['native0']=ref('wrong')
                elif location=='region':
                    args[2]['cache_report']['teacher_checkpoint']=ref('wrong')
                else:
                    args[3]['report']['teacher_checkpoint']=ref('wrong')
                with self.assertRaises(RuntimeError):
                    call(args)

if __name__=='__main__':
    unittest.main(verbosity=2)
