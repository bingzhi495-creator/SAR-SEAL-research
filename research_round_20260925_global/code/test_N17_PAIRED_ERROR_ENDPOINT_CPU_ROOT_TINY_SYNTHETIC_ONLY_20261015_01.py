"""One synthetic CPU-entry contract check; no actual owners/SOURCE/SAR/Torch/CUDA."""
import importlib.util
from pathlib import Path
import types
import unittest

ENTRY=Path(__file__).with_name('N17_LOCAL_CENTER_ERROR_PAIRED_ENDPOINT_CPU_ROOT_UNBOUND_DRAFT_20261015_01.py')

class EntryContract(unittest.TestCase):
    def test_cpu121_header_hooks_and_paired_trace_substream(self):
        definition=importlib.util.spec_from_file_location('_n17_tiny_cpu_entry',ENTRY)
        entry=importlib.util.module_from_spec(definition);definition.loader.exec_module(entry)
        spec=dict(schema=entry.CPU_SCHEMA,normal_RAW_count=121,expected_normal_RAW_count=121,
            total_failed_RAW_count=17,total_terminal_identity_pin_count=138,initialization_seed=202610150171,
            deadline_seconds=1800,automatic_retry=False,allocator_limit_MiB=4096,whole_GPU_limit_MiB=8192,
            entry_source=dict(path=str(ENTRY.resolve()),bytes=0,sha256='0'*64))
        self.assertTrue(entry.validate_cpu_header(spec))
        for key,value in (('schema',entry.SCHEMA),('normal_RAW_count',120),('total_failed_RAW_count',16),
                          ('total_terminal_identity_pin_count',137),('normal_RAW_count',True)):
            with self.subTest(key=key,value=value),self.assertRaises(RuntimeError):
                entry.validate_cpu_header(dict(spec,**{key:value}))
        updates=[dict(event='paired_image_update',epoch=1,dataset_index=i) for i in (2,0)]
        records=[dict(event='before_image',epoch=1,dataset_index=2),updates[0],
            dict(event='before_fixed_endpoint_image',epoch=8,dataset_index=0),updates[1]]
        self.assertEqual(entry.paired_update_substream(records),updates)
        fake=types.SimpleNamespace(ARMS=entry.ARMS,INIT_SEED=entry.SEED,ORDER_SEED=entry.ORDER_SEED,
            STATE_SHAPES=entry.STATE_SHAPES,ADAMW=entry.ADAMW,ABS_LOGITS=1e-4,ABS_MEAN_CE=1e-4,
            independent_logits=lambda *a:None,validate_state_evidence=lambda *a:None,
            validate_endpoint=lambda *a:None,validate_trace=lambda *a:None)
        self.assertIs(entry.bind_endpoint_body_api(fake),fake)
        fake.ARMS=('UTILITY','ERROR')
        with self.assertRaises(RuntimeError):entry.bind_endpoint_body_api(fake)

if __name__=='__main__':unittest.main()
