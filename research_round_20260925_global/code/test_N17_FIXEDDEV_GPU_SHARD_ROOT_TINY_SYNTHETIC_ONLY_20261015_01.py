"""One ordinary GPU-entry ABI check; no Torch/CUDA/SAR/actual refs."""
import importlib.util
import ast
from pathlib import Path
import unittest
ENTRY=Path(__file__).with_name('N17_FIXEDDEV_LOCAL_CENTER_ERROR_GPU_SHARD_ROOT_UNBOUND_DRAFT_20261015_01.py')
class Contract(unittest.TestCase):
    def test_new_header_and_new_arm_member_identifiers(self):
        s=importlib.util.spec_from_file_location('_n17_gpu_tiny',ENTRY);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
        value=dict(schema=m.SCHEMA,shard_id=0,dataset_ordinal_range=[0,47],normal_RAW_count=122,expected_normal_RAW_count=122,
            total_failed_RAW_count=17,total_terminal_identity_pin_count=139,initialization_seed=202610150171,
            deadline_seconds=1800,automatic_retry=False,allocator_limit_MiB=4096,whole_GPU_limit_MiB=8192,
            entry_source=dict(path=str(ENTRY.resolve()),bytes=0,sha256='0'*64))
        self.assertTrue(m.validate_gpu_header(value))
        for key,v in (('schema',m.TRAIN_SCHEMA),('normal_RAW_count',121),('shard_id',True),('dataset_ordinal_range',[0,46])):
            with self.subTest(key=key),self.assertRaises(RuntimeError):m.validate_gpu_header(dict(value,**{key:v}))
        event=dict(query_id=0,control='LOCAL_ERROR',scout_epoch=24,evalmode='shared-atmost-exact',stage='scout_C_logits',chunk_first=0)
        first=m.new_point_member_id(0,event)
        other=m.new_point_member_id(0,dict(event,control='CENTER_ERROR'))
        self.assertNotEqual(first,other)
        self.assertNotEqual(m.new_profile_member_id(0,0,24,'LOCAL_ERROR','atmost'),m.new_profile_member_id(0,0,24,'LOCAL_ERROR','exact'))
        with self.assertRaises(RuntimeError):m.new_point_member_id(0,dict(event,control='UTILITY'))
        tree=ast.parse(ENTRY.read_text(encoding='utf-8'))
        run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
        self.assertFalse(any(isinstance(n,ast.Name) and n.id=='reference' and isinstance(n.ctx,ast.Store) for n in ast.walk(run)))
        self.assertTrue(any(isinstance(n,ast.Delete) and any(isinstance(t,ast.Name) and t.id=='profile' for t in n.targets) for n in ast.walk(run)))
        budget=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='capture_budget_gate')
        self.assertTrue(any(isinstance(n,ast.Constant) and n.value==800 for n in ast.walk(budget)))
if __name__=='__main__':unittest.main()
