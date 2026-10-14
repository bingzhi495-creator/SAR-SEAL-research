"""SYNTHETIC_ONLY stdlib; tiny real fixture refs, never actual science/RAW/Source."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
ROOT_TMP=Path(__file__).resolve().parent
path=ROOT_TMP/'N16_FIXED_E24_ALL100_SCOUT_CPU_NUMERIC_SHARDS_AGGREGATE_ROOT_UNBOUND_DRAFT_20261014_02.py'
spec=importlib.util.spec_from_file_location('_synthetic_final_cpu02',path)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class ReferenceTests(unittest.TestCase):
    def test_source_gate_passes_strict_ref_after_closure_not_Path(self):
        with tempfile.TemporaryDirectory(prefix='SYNTHETIC_N16_REF17_',dir=ROOT_TMP) as directory:
            root=Path(directory).resolve();self.assertTrue(root.is_relative_to(ROOT_TMP.resolve()))
            source=root/'fixture_source.py'
            source.write_text("SYNTHETIC_ONLY=True\ndef check_common_completed_aggregate(spec,gate):\n    raise RuntimeError('SYNTHETIC_ONLY_STRICT_REFERENCE_REACHED')\n",encoding='utf8')
            ref=m.reference(source);closure={ref['path']:ref};events=[]
            original_closed=m.closed;original_load=m.load
            def closed(c,r):events.append(('closed',r));return original_closed(c,r)
            def load(r,n):events.append(('load',r));return original_load(r,n)
            own=dict(schema=m.SCHEMA,normal_RAW_count=118,expected_normal_RAW_count=118,total_failed_RAW_count=17,
                total_terminal_identity_pin_count=135,aggregate_CPU_owner_index=113,entry_source=m.reference(path),
                numeric_CPU_shard_gate_source=ref)
            with patch.object(m,'inherited_source_gate',return_value={'closure':closure}),patch.object(m,'closed',side_effect=closed),patch.object(m,'load',side_effect=load):
                with self.assertRaisesRegex(RuntimeError,'SYNTHETIC_ONLY_STRICT_REFERENCE_REACHED'):
                    m.source_gate(path,own)
            self.assertEqual(events,[('closed',ref),('load',ref)])
    def test_load_keeps_rejecting_Path_and_changed_ref(self):
        with tempfile.TemporaryDirectory(prefix='SYNTHETIC_N16_STRICT17_',dir=ROOT_TMP) as directory:
            root=Path(directory).resolve();self.assertTrue(root.is_relative_to(ROOT_TMP.resolve()))
            source=root/'fixture_source.py';source.write_text('SYNTHETIC_ONLY=17\n',encoding='utf8')
            ref=m.reference(source);closure={ref['path']:ref}
            self.assertEqual(m.load(ref,'_synthetic_17').SYNTHETIC_ONLY,17)
            with self.assertRaisesRegex(RuntimeError,'Malformed absolute pinned reference'):m.load(m.closed(closure,ref),'_bad_Path')
            bad=dict(ref,bytes=ref['bytes']+1)
            with self.assertRaises(RuntimeError):m.closed(closure,bad)
    def test_failure17_typed_exit1_abortNone_pin_and_distinctness(self):
        with tempfile.TemporaryDirectory(prefix='SYNTHETIC_N16_RAW17_',dir=ROOT_TMP) as directory:
            root=Path(directory).resolve();self.assertTrue(root.is_relative_to(ROOT_TMP.resolve()))
            phase='SYNTHETIC_FAILURE17';folder=root/phase;folder.mkdir();target=folder/'PARENT_RECEIPT.json'
            raw=dict(status='ACTUAL_OWNED_TERMINAL',returncode=1,original_identity={'SYNTHETIC_ONLY':17},root_wait_reaped=True,
                held_handle_exit=dict(wait_state='WAIT_OBJECT_0',actual_exit_code=1),numeric_birth_absence={'original_birth_absent':True},
                error=None,abort=None,wait_errors=False,GPU_observation_errors=[],admission_sha256='0'*64,
                whole_GPU_observed_peak_MiB=0,whole_GPU_used_limit_MiB=8192,whole_leaf_VRAM_certified=False,
                GPU_scope='whole GPU observed, not per-process WDDM allocation',phase=phase)
            target.write_text(json.dumps(raw),encoding='utf8');ref=m.reference(target)
            pin=dict(phase=phase,returncode=1,original_identity=raw['original_identity'],admission_sha256='0'*64)
            extra=dict(RAW_ref=ref,phase=phase,pin=pin,returncode=1,original_Q_absent=True)
            for key in ('admission','phase_binding','bootstrap','entry_source','spec'):
                extra[key]=dict(path=str(root/(key+'.json')),bytes=0,sha256='0'*64)
            owner=dict(additional_preserved_N16_final_CPU_reference_ABI_failure=extra,terminal_identity_pins={},RAW_inputs=[])
            for i,key in enumerate(m.FAILURE17_PRIOR_KEYS):
                owner[key]={'RAW_ref':dict(path=str(root/('prior%d.json'%i)),bytes=0,sha256='0'*64)}
            self.assertEqual(m.seventeenth_final_CPU_reference_ABI_failure_raw_only(owner),raw)
            for field,value in (('returncode',True),('abort','resource'),('error','bad'),('root_wait_reaped',False)):
                bad=dict(raw);bad[field]=value;target.write_text(json.dumps(bad),encoding='utf8')
                with self.assertRaises(RuntimeError):m.seventeenth_final_CPU_reference_ABI_failure_raw_only(owner)
            target.write_text(json.dumps(raw),encoding='utf8')
            bad=copy.deepcopy(owner);bad['additional_preserved_N16_final_CPU_reference_ABI_failure']['original_Q_absent']=False
            with self.assertRaises(RuntimeError):m.seventeenth_final_CPU_reference_ABI_failure_raw_only(bad)
            bad=copy.deepcopy(owner);bad['terminal_identity_pins'][ref['path']]=pin
            with self.assertRaises(RuntimeError):m.seventeenth_final_CPU_reference_ABI_failure_raw_only(bad)
if __name__=='__main__':unittest.main(verbosity=2)
