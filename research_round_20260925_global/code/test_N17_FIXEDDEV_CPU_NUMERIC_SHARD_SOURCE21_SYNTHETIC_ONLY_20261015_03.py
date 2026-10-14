"""Pure fake metadata/RAW only; no Torch, scientific data, actual refs or execution."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
ENTRY=Path(__file__).with_name('N17_FIXEDDEV_LOCAL_CENTER_ERROR_CPU_NUMERIC_SHARD_ROOT_UNBOUND_DRAFT_20261015_03.py')
class Contract(unittest.TestCase):
    def test_failed21_front_current_counts_and_original_failed20_preserved(self):
        s=importlib.util.spec_from_file_location('_n17_cpu03_meta',ENTRY);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
        header=dict(schema='ROOT-N17-fixedDEV-local-center-error-CPU-numeric-shard-spec-v3',cpu_shard_id=0,cpu_dataset_ordinal_range=[0,47],
            normal_RAW_count=127,expected_normal_RAW_count=127,cpu_shard_owner_index=127,total_failed_RAW_count=21,total_terminal_identity_pin_count=148,
            deadline_seconds=1800,automatic_retry=False,allocator_limit_MiB=4096,whole_GPU_limit_MiB=8192,maximum_process_RSS_bytes=16*1024**3,
            initialization_seed=202610150171,entry_source=dict(path=str(ENTRY.resolve()),bytes=0,sha256='0'*64),
            execution_partition_supplement=dict(path=str(ENTRY.resolve()),bytes=0,sha256='0'*64))
        self.assertTrue(m.validate_cpu_header(header))
        for key,wrong in (('total_failed_RAW_count',20),('total_terminal_identity_pin_count',147),('schema',m.PRIOR_CPU_SCHEMA),('normal_RAW_count',126)):
            with self.subTest(header=key),self.assertRaises(RuntimeError):m.validate_cpu_header(dict(header,**{key:wrong}))
        with tempfile.TemporaryDirectory(prefix='n17_fake_CPU_SOURCE21_') as temporary:
            phase=Path(temporary)/'FAKE21';phase.mkdir();p=phase/'PARENT_RECEIPT.json'
            raw=dict(status='ACTUAL_OWNED_TERMINAL',returncode=1,original_identity={'pid':321,'birth':'FAKE'},root_wait_reaped=True,
                held_handle_exit=dict(wait_state='WAIT_OBJECT_0',actual_exit_code=1),numeric_birth_absence=dict(original_birth_absent=True),
                error=None,abort=None,wait_errors=False,GPU_observation_errors=[],admission_sha256='1'*64,
                whole_GPU_observed_peak_MiB=0,whole_GPU_used_limit_MiB=8192,whole_leaf_VRAM_certified=False,
                GPU_scope='whole GPU observed, not per-process WDDM allocation',phase='FAKE21')
            p.write_text(json.dumps(raw),encoding='utf-8');fake=lambda name:dict(path=str(phase/name),bytes=0,sha256='0'*64)
            extra=dict(RAW_ref=fake('PARENT_RECEIPT.json'),phase='FAKE21',pin=dict(phase='FAKE21',returncode=1,
                original_identity=raw['original_identity'],admission_sha256='1'*64),returncode=1,admission=fake('AD.json'),phase_binding=fake('PHASE.json'),
                bootstrap=fake('BOOT.py'),entry_source=fake('ENTRY.py'),spec=fake('SPEC.json'),original_report_absent=True)
            spec=dict(header,terminal_identity_pins={},RAW_inputs=[dict(receipt=fake('NORMAL_%03d.json'%i)) for i in range(127)])
            for key in m.FAILURE21_PRIOR_KEYS:spec[key]=dict(RAW_ref=fake(key+'.json'))
            key21='additional_preserved_N17_independent_CPU_SOURCE_closure_failure';spec[key21]=extra
            # Even a stubbed hash/import is forbidden until every RAW-only check passed.
            original_reference=m.reference;m.reference=lambda *a:(_ for _ in ()).throw(AssertionError('SHA before RAW gate'))
            m.load=lambda *a:(_ for _ in ()).throw(AssertionError('Source import before RAW gate'))
            self.assertEqual(m.twentyfirst_independent_CPU_SOURCE_closure_failure_raw_only(spec),raw)
            for mutate in (lambda x:x.pop(key21),lambda x:x[key21].update(returncode=True),lambda x:x[key21].update(original_report_absent=False),
                lambda x:x[key21]['pin'].update(admission_sha256='2'*64),lambda x:x[m.FAILURE21_PRIOR_KEYS[-1]].update(RAW_ref=extra['RAW_ref'])):
                bad=copy.deepcopy(spec);mutate(bad)
                with self.assertRaises((KeyError,RuntimeError)):m.twentyfirst_independent_CPU_SOURCE_closure_failure_raw_only(bad)
            bad=copy.deepcopy(raw);bad['abort']=m.ABORT;p.write_text(json.dumps(bad),encoding='utf-8')
            with self.assertRaises(RuntimeError):m.twentyfirst_independent_CPU_SOURCE_closure_failure_raw_only(spec)
            p.write_text(json.dumps(raw),encoding='utf-8');m.reference=original_reference
            adapter=fake('OMITTED_FROM_ORIGINAL_SOURCES_ADAPTER.py');spec['CPU_adapter_source']=adapter
            old=dict(schema=m.PRIOR_CPU_SCHEMA,entry_source=extra['entry_source'],phase='FAKE21',cpu_shard_id=0,cpu_dataset_ordinal_range=[0,47],
                cpu_shard_owner_index=127,normal_RAW_count=127,expected_normal_RAW_count=127,total_failed_RAW_count=20,total_terminal_identity_pin_count=147,
                RAW_inputs=spec['RAW_inputs'],sources=[],terminal_identity_pins={},output_directory=str(phase/'OLD_OUTPUT'),CPU_adapter_source=adapter)
            for key in m.FAILURE21_PRIOR_KEYS:old[key]=spec[key]
            (phase/'SPEC.json').write_text(json.dumps(old),encoding='utf-8');recorded=[]
            m.closed=lambda closure,ref:(recorded.append(ref) or Path(ref['path']))
            owner=dict(scientific_spec=extra['spec'],entry=extra['entry_source']['path'],cuda_visible_devices='')
            lineage=types.SimpleNamespace(owner_join=lambda row,raw,c:owner)
            self.assertEqual(m.closed_independent_CPU_SOURCE_failure21(spec,{},lineage,raw),owner)
            self.assertTrue(all(extra[k] in recorded for k in ('RAW_ref','admission','phase_binding','bootstrap','entry_source','spec')))
            # The actual failed source omitted its adapter ref; preserving that fact is legal.
            self.assertNotIn(adapter,old['sources'])
            for key,wrong in (('cuda_visible_devices','0'),('scientific_spec',fake('WRONG_SPEC.json'))):
                with self.assertRaises(RuntimeError):m.closed_independent_CPU_SOURCE_failure21(spec,{},types.SimpleNamespace(owner_join=lambda row,raw,c:dict(owner,**{key:wrong})),raw)
            bad=copy.deepcopy(old);bad['total_failed_RAW_count']=21;(phase/'SPEC.json').write_text(json.dumps(bad),encoding='utf-8')
            with self.assertRaises(RuntimeError):m.closed_independent_CPU_SOURCE_failure21(spec,{},lineage,raw)
        oldtree=ast.parse(ENTRY.with_name('N17_FIXEDDEV_LOCAL_CENTER_ERROR_CPU_NUMERIC_SHARD_ROOT_UNBOUND_DRAFT_20261015_02.py').read_text(encoding='utf-8'))
        tree=ast.parse(ENTRY.read_text(encoding='utf-8'));oldfuncs={n.name:ast.dump(n,include_attributes=False) for n in oldtree.body if isinstance(n,ast.FunctionDef)}
        newfuncs={n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,ast.FunctionDef)}
        self.assertEqual({name for name in oldfuncs if oldfuncs[name]!=newfuncs[name]}, {'ninth_failed_raw_only','inherited_source_gate','validate_cpu_header','run'})
        gate=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='inherited_source_gate')
        calls=[n.func.id for n in ast.walk(gate) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
        self.assertLess(calls.index('twentyfirst_independent_CPU_SOURCE_closure_failure_raw_only'),calls.index('pre_gate'))
        # Numeric loop/functions remain identical; only emitted current ancestry counts differ.
        oldrun=next(n for n in oldtree.body if isinstance(n,ast.FunctionDef) and n.name=='run');newrun=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
        oldloops=[ast.dump(n,include_attributes=False) for n in oldrun.body if isinstance(n,(ast.For,ast.FunctionDef))]
        newloops=[ast.dump(n,include_attributes=False) for n in newrun.body if isinstance(n,(ast.For,ast.FunctionDef))]
        self.assertEqual(oldloops,newloops)
if __name__=='__main__':unittest.main(verbosity=2)
