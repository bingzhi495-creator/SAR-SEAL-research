"""Mini ordinary fake RAW test; no Source/hash/SCI I/O or storage replay."""
import ast
import importlib.util
from pathlib import Path
import unittest

SOURCE=Path(__file__).with_name('N16_FIXED_E24_ALL100_SCOUT_FULLDEV_GPU_ROOT_UNBOUND_DRAFT_20261014_05.py')
EXTRA='additional_preserved_N16_fullDEV_controlled_storage_failure'


def fixture():
    d=importlib.util.spec_from_file_location('ordinary_failed14',str(SOURCE));m=importlib.util.module_from_spec(d);d.loader.exec_module(m)
    phase='SYNTHETIC_ONLY_CONTROLLED14';ref=lambda s:dict(path='E:/SAR/seal/tmp/SYNTHETIC_ONLY/'+s,bytes=0,sha256='0'*64)
    raw=dict(status='ACTUAL_OWNED_TERMINAL',returncode=1,root_wait_reaped=True,original_identity={'SYNTHETIC_PID':14},
        held_handle_exit=dict(wait_state='WAIT_OBJECT_0',actual_exit_code=1),numeric_birth_absence=dict(original_birth_absent=True),
        error=None,abort=m.ABORT,wait_errors=False,GPU_observation_errors=[],minimum_ram_bytes=32*1024**3,elapsed_seconds=1800.25,
        admission_sha256='0'*64,whole_GPU_observed_peak_MiB=4000,whole_GPU_used_limit_MiB=8192,whole_leaf_VRAM_certified=False,GPU_scope=m.GPU_SCOPE)
    pin=dict(phase=phase,returncode=1,original_identity={'SYNTHETIC_PID':14},admission_sha256='0'*64)
    extra=dict(RAW_ref=ref(phase+'/PARENT_RECEIPT.json'),phase=phase,pin=pin,returncode=1,admission=ref('AD'),
        phase_binding=ref('PHASE'),bootstrap=ref('BOOT'),entry_source=ref('ENTRY'),spec=ref('SPEC'),original_report_absent=True)
    spec={EXTRA:extra,'terminal_identity_pins':{'SYNTHETIC_LEGACY':pin}}
    for i,k in enumerate(('additional_preserved_independent_CPU_failure','additional_preserved_N16_cache_failure',
        'additional_preserved_N16_cache_target_dtype_failure','additional_preserved_N16_controlled_resource_failure',
        'additional_preserved_N16_fullDEV_packet_ABI_failure')):spec[k]={'RAW_ref':ref('OTHER%d/PARENT_RECEIPT.json'%i)}
    def forbidden(*args,**kwargs):raise AssertionError('Pure RAW gate reached Source/hash/import')
    m.read=lambda p:raw;m.bound=m.closed=m.reference=m.load=forbidden
    return m,spec,raw


class ControlledStorageFailure14(unittest.TestCase):
    def test_typed_controlled_failure_accepts_without_source(self):
        m,s,r=fixture();self.assertIs(m.fourteenth_controlled_storage_failure_raw_only(s),r)

    def test_abort_none_wrongphase_wrongpin_and_terminal_resource_reject(self):
        cases=(lambda s,r:r.update(abort=None),lambda s,r:s[EXTRA].update(phase='WRONG_PHASE'),
            lambda s,r:s[EXTRA]['pin'].update(original_identity={'SYNTHETIC_PID':99}),
            lambda s,r:s[EXTRA]['pin'].update(admission_sha256='1'*64),lambda s,r:r.update(returncode=0),
            lambda s,r:r['held_handle_exit'].update(actual_exit_code=0),lambda s,r:r.update(elapsed_seconds=1799.99),
            lambda s,r:r.update(elapsed_seconds=1805.01),lambda s,r:r.update(minimum_ram_bytes=16*1024**3-1),
            lambda s,r:s[EXTRA].update(RAW_ref=s['additional_preserved_N16_fullDEV_packet_ABI_failure']['RAW_ref']))
        for i,change in enumerate(cases):
            with self.subTest(i=i):
                m,s,r=fixture();change(s,r)
                with self.assertRaises((RuntimeError,m.RawGateError)):m.fourteenth_controlled_storage_failure_raw_only(s)

    def test_raw14_precedes_first_sha_and_storage_code_is_unchanged(self):
        new=ast.parse(SOURCE.read_text(encoding='utf-8'));old=ast.parse(SOURCE.with_name(SOURCE.name.replace('_05.py','_04.py')).read_text(encoding='utf-8'))
        gate=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name=='inherited_source_gate')
        calls=[n.value.func.id for n in gate.body if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name)]
        self.assertLess(calls.index('fourteenth_controlled_storage_failure_raw_only'),calls.index('pre_gate'))
        for name in ('ImageBundleCaptureSink','evaluate_image','shared_actual_image','strip_event_to_shared','lookup_CPU_bundle_member',
                'validate_CPU_image_bundle','decode_CPU_query_core_bundle','model_load_cpu','original_image'):
            left=next(n for n in old.body if getattr(n,'name',None)==name);right=next(n for n in new.body if getattr(n,'name',None)==name)
            self.assertEqual(ast.dump(left),ast.dump(right),name)


if __name__=='__main__':unittest.main()
