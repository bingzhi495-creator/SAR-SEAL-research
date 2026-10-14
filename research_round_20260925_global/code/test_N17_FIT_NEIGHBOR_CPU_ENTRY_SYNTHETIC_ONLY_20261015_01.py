"""Tiny stdlib SYNTHETIC_ONLY own-contract checks; no scientific sources or data."""
import copy
import importlib.util
from pathlib import Path
import unittest
import tempfile
p=Path(__file__).with_name('N17_FIT_NATIVE_NEIGHBOR_CACHE_CPU_ROOT_UNBOUND_DRAFT_20261015_01.py')
s=importlib.util.spec_from_file_location('_n17_entry_synthetic',p)
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

class Contract(unittest.TestCase):
    def fixture(self):
        return dict(schema=m.LEGACY_SCHEMA,N17_schema=m.N17_SCHEMA,
            normal_RAW_count=119,expected_normal_RAW_count=119,total_failed_RAW_count=17,
            total_terminal_identity_pin_count=136,RAW_inputs=[{} for _ in range(119)],
            terminal_identity_pins={str(i):{} for i in range(127)},deadline_seconds=1800,
            allocator_limit_MiB=4096,whole_GPU_limit_MiB=8192,automatic_retry=False,
            N17_cache_scope=dict(split='FIT',images=741,natural_queries=74100,
                sampled_points=3035136,payload_bytes=97124352,no_DEV_inputs=True,
                no_teacher_forward=True,no_new_training=True))
    def test_own119_and_FIT_only_before_source_import(self):
        m.validate_spec(self.fixture())
        for field,value in [('normal_RAW_count',True),('total_failed_RAW_count',16),
                ('expected_normal_RAW_count',118),('total_terminal_identity_pin_count',135)]:
            bad=self.fixture();bad[field]=value
            with self.assertRaises(RuntimeError):m.validate_spec(bad)
    def test_BOM_metadata_read_uses_real_supported_codec(self):
        with tempfile.TemporaryDirectory(prefix='SYN_N17_METADATA_') as directory:
            f=Path(directory)/'fake.json'
            f.write_text('{"synthetic_only":true}',encoding='utf-8-sig')
            self.assertEqual(m.read(f),dict(synthetic_only=True))
        for field,value in [('split','DEV'),('images',187),('no_DEV_inputs',False),('sampled_points',1)]:
            bad=self.fixture();bad['N17_cache_scope'][field]=value
            with self.assertRaises(RuntimeError):m.validate_spec(bad)

if __name__=='__main__':unittest.main(verbosity=2)
