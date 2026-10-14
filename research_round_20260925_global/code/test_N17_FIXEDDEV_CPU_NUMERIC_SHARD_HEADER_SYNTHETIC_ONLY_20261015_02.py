"""One metadata-only fake-ref fixture. No Torch/CUDA/SAR/actual qualification IO."""
import ast
import importlib.util
from pathlib import Path
import unittest
ENTRY=Path(__file__).with_name('N17_FIXEDDEV_LOCAL_CENTER_ERROR_CPU_NUMERIC_SHARD_ROOT_UNBOUND_DRAFT_20261015_02.py')

class Contract(unittest.TestCase):
    def test_CPU_header_and_explicit_original_parent(self):
        info=importlib.util.spec_from_file_location('_N17_CPU_header_fake',ENTRY);module=importlib.util.module_from_spec(info);info.loader.exec_module(module)
        fake=lambda name:dict(path=str(ENTRY.parent/name),bytes=0,sha256='0'*64)
        parent=dict(shard_id=0,dataset_ordinal_range=[0,47],source=fake('FAKE_GPU03.py'),spec=fake('FAKE_GPU_SPEC.json'),
            report=fake('FAKE_GPU_REPORT.json'),RAW=fake('FAKE_GPU_RAW.json'),admission=fake('FAKE_AD.json'),
            phase_binding=fake('FAKE_PHASE.json'),bootstrap=fake('FAKE_BOOT.py'),shard_owner_index=122)
        spec=dict(schema=module.CPU_SCHEMA,entry_source=fake(ENTRY.name),cpu_shard_id=0,cpu_dataset_ordinal_range=[0,47],cpu_shard_owner_index=127,
            expected_normal_RAW_count=127,normal_RAW_count=127,total_failed_RAW_count=20,total_terminal_identity_pin_count=147,
            deadline_seconds=1800,automatic_retry=False,allocator_limit_MiB=4096,whole_GPU_limit_MiB=8192,
            initialization_seed=202610150171,maximum_process_RSS_bytes=16*1024**3,GPU_shard=parent,GPU_outer_source=parent['source'],
            execution_partition_supplement=fake('FAKE_PARTITION_LOCK.json'))
        self.assertTrue(module.validate_cpu_header(spec));self.assertEqual(module.validate_parent_record(spec),parent)
        with self.assertRaises(RuntimeError):module.validate_cpu_header(dict(spec,cpu_shard_owner_index=123,
            normal_RAW_count=123,expected_normal_RAW_count=123,total_terminal_identity_pin_count=143))
        for key,value in (('schema',module.SCHEMA),('total_failed_RAW_count',19),('cpu_shard_id',True),('cpu_dataset_ordinal_range',[0,46]),('cpu_shard_owner_index',122)):
            with self.subTest(key=key),self.assertRaises(RuntimeError):module.validate_cpu_header(dict(spec,**{key:value}))
        for key,value in (('shard_owner_index',121),('dataset_ordinal_range',[0,46]),('source',dict(path='relative.py',bytes=0,sha256='0'*64))):
            with self.subTest(parent=key),self.assertRaises(RuntimeError):module.validate_parent_record(dict(spec,GPU_shard=dict(parent,**{key:value})))
        for sid,scope in enumerate(((0,47),(47,70),(70,94),(94,141),(141,187))):
            mixed=dict(parent,shard_id=sid,dataset_ordinal_range=list(scope),shard_owner_index=122+sid,source=fake('FAKE_GPU03.py' if sid==0 else 'FAKE_GPU04.py'))
            current=dict(spec,cpu_shard_id=sid,cpu_dataset_ordinal_range=list(scope),cpu_shard_owner_index=127,
                expected_normal_RAW_count=127,normal_RAW_count=127,total_terminal_identity_pin_count=147,GPU_shard=mixed,GPU_outer_source=mixed['source'])
            self.assertTrue(module.validate_cpu_header(current));self.assertEqual(module.validate_parent_record(current),mixed)
        tree=ast.parse(ENTRY.read_text(encoding='utf-8'))
        self.assertFalse(any(isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and
            node.func.value.id=='gpu' and node.func.attr=='source_gate' for node in ast.walk(tree)))
        self.assertFalse(any(isinstance(node,ast.Assign) and any(isinstance(t,ast.Attribute) and isinstance(t.value,ast.Name) and t.value.id=='gpu'
            for t in node.targets) for node in ast.walk(tree)))

if __name__=='__main__':unittest.main()
