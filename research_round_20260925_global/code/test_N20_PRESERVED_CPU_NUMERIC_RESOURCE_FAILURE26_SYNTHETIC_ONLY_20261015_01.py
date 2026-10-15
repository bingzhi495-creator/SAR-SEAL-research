"""SYNTHETIC_ONLY: invented stdlib RAW/ref fixtures; no Torch or science IO."""
import ast
import copy
import math
from pathlib import Path
import types
import unittest

HERE = Path(__file__).resolve().parent
HELPER = HERE / 'N20_PRESERVED_CPU_NUMERIC_RESOURCE_FAILURE26_UNBOUND_20261015_01.py'
PRIOR = HERE / 'N17_FIXEDDEV_LOCAL_CENTER_ERROR_GPU_SHARD_ROOT_UNBOUND_DRAFT_20261015_04.py'

class Failure26Synthetic(unittest.TestCase):
    def test_preserved_resource_exit1_and_closed_CPU_owner(self):
        namespace = dict(math=math, Path=Path,
            ABORT='QUALIFIED_ORIGINAL_LEAF_RESOURCE_OR_DEADLINE', MIN_RAM_BYTES=16*1024**3,
            GPU_LIMIT_MIB=8192, GPU_SCOPE='whole GPU observed, not per-process WDDM allocation')
        tree = ast.parse(PRIOR.read_text(encoding='utf8'))
        names = {'require', 'ref_shape', 'check_pin', 'RawGateError', 'helpercheck'}
        selected = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(PRIOR), 'exec'), namespace)
        exec(compile(HELPER.read_text(encoding='utf8'), str(HELPER), 'exec'), namespace)
        def ref(name, size=1):
            return dict(path=str(HERE / ('SYNTHETIC_ONLY_' + name)), bytes=size, sha256='a'*64)
        phase='SYNTHETIC_ONLY_failed26'
        receipt=ref('receipt'); receipt['path']=str(HERE / phase / 'PARENT_RECEIPT.json')
        identity=dict(pid=77760, birth_filetime_100ns='134365278288498169', birth_unix=1792054228.8498173)
        pin=dict(phase=phase, returncode=1, original_identity=identity, admission_sha256='a'*64)
        empty='e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
        raw=dict(status='ACTUAL_OWNED_TERMINAL', returncode=1, original_identity=identity,
            root_wait_reaped=True, held_handle_exit=dict(wait_state='WAIT_OBJECT_0',actual_exit_code=1),
            numeric_birth_absence=dict(original_birth_absent=True), error=None,
            abort='QUALIFIED_ORIGINAL_LEAF_RESOURCE_OR_DEADLINE', wait_errors=False,
            GPU_observation_errors=[], admission_sha256='a'*64, minimum_ram_bytes=30790778880,
            whole_GPU_observed_peak_MiB=2146, whole_GPU_used_limit_MiB=8192,
            whole_leaf_VRAM_certified=False, GPU_scope=namespace['GPU_SCOPE'],
            elapsed_seconds=1800.342999999877, logs={'stdout.log':empty,'stderr.log':empty})
        extra=dict(RAW_ref=receipt, phase=phase, pin=pin, returncode=1, admission=ref('admission'),
            phase_binding=ref('phase'), bootstrap=ref('bootstrap'), entry_source=ref('entry',159357),
            spec=ref('spec'), original_report_absent=True)
        spec=dict(N20_failed_CPU_numeric_record=extra, N20_failed26_helper_source=ref('helper'),
            failed26_partials_reused=False, normal_RAW_count=145, expected_normal_RAW_count=145,
            total_failed_RAW_count=26, total_terminal_identity_pin_count=171,
            RAW_inputs=[dict(receipt=ref('normal_'+str(i))) for i in range(145)],
            terminal_identity_pins={str(i):{} for i in range(153)})
        namespace['FAILURE21_PRIOR_KEYS']=('synthetic_prior',)
        prior_keys=('synthetic_prior','additional_preserved_N17_independent_CPU_SOURCE_closure_failure',
            'additional_preserved_N20_training_output_precondition_failure',
            'additional_preserved_N20_current_failure_counter_failure','N20_failed_training_record',
            'N20_failed_DEV_capture_record')
        for i,key in enumerate(prior_keys): spec[key]=dict(RAW_ref=ref('failure_'+str(i)))
        def read(path):
            if str(path)==receipt['path']: return copy.deepcopy(raw)
            if str(path)==extra['spec']['path']: return copy.deepcopy(original)
            raise AssertionError('Unexpected metadata read: '+str(path))
        namespace['read']=read
        verified=namespace['n20_failed26_raw_only'](spec)
        self.assertEqual(verified,raw)
        self.assertNotEqual(len(spec['terminal_identity_pins']),spec['total_terminal_identity_pin_count'])
        for mutate in (lambda s:s.pop('N20_failed_CPU_numeric_record'),
                       lambda s:s.update(total_failed_RAW_count=25),
                       lambda s:s.update(failed26_partials_reused=True),
                       lambda s:s['N20_failed_CPU_numeric_record'].update(returncode=0),
                       lambda s:s['N20_failed_CPU_numeric_record']['pin'].update(returncode=True)):
            bad=copy.deepcopy(spec); mutate(bad)
            with self.assertRaises((KeyError,RuntimeError,ValueError)): namespace['n20_failed26_raw_only'](bad)
        for key,value in (('abort',None),('elapsed_seconds',1700.0),('root_wait_reaped',False),
                          ('minimum_ram_bytes',1),('original_identity',dict(identity,pid=1))):
            saved=raw[key]; raw[key]=value
            with self.assertRaises((RuntimeError,ValueError)): namespace['n20_failed26_raw_only'](spec)
            raw[key]=saved
        # Fake metadata filesystem; no scientific/real output directory is opened.
        class FakePath:
            report_exists=False
            def __init__(self,value): self.value=str(value)
            def __truediv__(self,value): return FakePath(self.value+'/'+value)
            def is_dir(self): return True
            def is_symlink(self): return False
            def exists(self): return self.report_exists and self.value.endswith('/REPORT.json')
            def resolve(self,strict=False): return self.value
        namespace['Path']=FakePath
        namespace['closed']=lambda closure,r: r['path'] if closure.get(r['path'])==r else (_ for _ in ()).throw(RuntimeError('not closed'))
        preserved=namespace['N20_FAILURE26_PRESERVED_KEYS']
        for key in preserved: spec.setdefault(key,ref(key))
        spec['completed_GPU_shards']=[dict(shard_id=i) for i in range(4)]
        original={key:copy.deepcopy(spec[key]) for key in preserved}
        original.update(schema='ROOT-N20-fixedDEV-local-center-feature-error-CPU-numeric-shard-spec-v1',
            phase=phase,entry_source=extra['entry_source'],normal_RAW_count=145,expected_normal_RAW_count=145,
            total_failed_RAW_count=25,total_terminal_identity_pin_count=170,cpu_shard_id=0,
            cpu_shard_owner_index=145,CPU_parent_GPU_id=0,cpu_dataset_ordinal_range=[0,24],
            RAW_inputs=copy.deepcopy(spec['RAW_inputs']),sources=[extra['entry_source']],
            terminal_identity_pins=copy.deepcopy(spec['terminal_identity_pins']),
            completed_GPU_shards=copy.deepcopy(spec['completed_GPU_shards']),
            deadline_seconds=1800,automatic_retry=False,maximum_process_RSS_bytes=16*1024**3,
            allocator_limit_MiB=4096,whole_GPU_limit_MiB=8192,output_directory='SYNTHETIC_ONLY_OUTPUT',
            CPU_admission_file=extra['admission']['path'])
        closure={r['path']:r for r in [extra[k] for k in namespace['N20_FAILURE26_REF_KEYS']]+[spec['N20_failed26_helper_source']]}
        owner=dict(scientific_spec=extra['spec'],entry=extra['entry_source']['path'],cuda_visible_devices='')
        gate=dict(closure=closure,lineage=types.SimpleNamespace(owner_join=lambda row,r,c:copy.deepcopy(owner)))
        self.assertEqual(namespace['n20_failed26_closed'](spec,gate,verified),owner)
        saved=owner['cuda_visible_devices']; owner['cuda_visible_devices']='0'
        with self.assertRaises(RuntimeError): namespace['n20_failed26_closed'](spec,gate,verified)
        owner['cuda_visible_devices']=saved
        original['total_failed_RAW_count']=26
        with self.assertRaises(RuntimeError): namespace['n20_failed26_closed'](spec,gate,verified)
        original['total_failed_RAW_count']=25
        FakePath.report_exists=True
        with self.assertRaises(RuntimeError): namespace['n20_failed26_closed'](spec,gate,verified)

if __name__=='__main__': unittest.main()
