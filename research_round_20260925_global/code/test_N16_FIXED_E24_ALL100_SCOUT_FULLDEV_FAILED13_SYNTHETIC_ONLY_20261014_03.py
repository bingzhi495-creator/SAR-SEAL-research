"""Ordinary-only fake RAW/AST regression; no science, hash, CLI or CUDA."""
import ast
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

PATH=Path(__file__).with_name('N16_FIXED_E24_ALL100_SCOUT_FULLDEV_GPU_ROOT_UNBOUND_DRAFT_20261014_03.py')
EXTRA='additional_preserved_N16_fullDEV_packet_ABI_failure'


def fixture():
    definition=importlib.util.spec_from_file_location('ordinary_failed13',str(PATH))
    module=importlib.util.module_from_spec(definition);definition.loader.exec_module(module)
    phase='SYNTHETIC_ONLY_FAILED13'
    ref=lambda name:dict(path='E:/SAR/seal/tmp/SYNTHETIC_ONLY/'+name,bytes=0,sha256='0'*64)
    raw=dict(status='ACTUAL_OWNED_TERMINAL',returncode=1,original_identity={'SYNTHETIC_PID':13},root_wait_reaped=True,
        held_handle_exit=dict(wait_state='WAIT_OBJECT_0',actual_exit_code=1),numeric_birth_absence=dict(original_birth_absent=True),
        error=None,abort=None,wait_errors=False,GPU_observation_errors=[],admission_sha256='0'*64,
        whole_GPU_observed_peak_MiB=1024,whole_GPU_used_limit_MiB=8192,whole_leaf_VRAM_certified=False,
        GPU_scope='whole GPU observed, not per-process WDDM allocation')
    pin=dict(phase=phase,returncode=1,original_identity=copy.deepcopy(raw['original_identity']),admission_sha256='0'*64)
    extra=dict(RAW_ref=ref(phase+'/PARENT_RECEIPT.json'),phase=phase,pin=pin,returncode=1,
        admission=ref('AD.json'),phase_binding=ref('PHASE.json'),bootstrap=ref('BOOT.py'),entry_source=ref('ENTRY.py'),
        spec=ref('SPEC.json'),original_report_absent=True)
    spec={EXTRA:extra,'terminal_identity_pins':{'SYNTHETIC_LEGACY_PIN':pin},'normal_RAW_count':109,
        'total_failed_RAW_count':13,'total_terminal_identity_pin_count':122}
    for i,key in enumerate(('additional_preserved_independent_CPU_failure','additional_preserved_N16_cache_failure',
            'additional_preserved_N16_cache_target_dtype_failure','additional_preserved_N16_controlled_resource_failure')):
        spec[key]={'RAW_ref':ref('OTHER_FAILED_%d/PARENT_RECEIPT.json'%i)}
    reads=[]
    def fake_read(path):reads.append(str(path));return raw
    def forbidden(*args,**kwargs):raise AssertionError('RAW-only gate reached Source/hash/import')
    module.read=fake_read;module.bound=module.closed=module.reference=module.load=forbidden
    return module,spec,raw,reads


class Failed13Regression(unittest.TestCase):
    def test_actual1_abort_none_fake_accepts_without_source_or_hash(self):
        m,s,r,reads=fixture();self.assertIs(m.thirteenth_failed_raw_only(s),r)
        self.assertEqual(len(reads),1);self.assertEqual(s['normal_RAW_count'],109)
        self.assertEqual((s['total_failed_RAW_count'],s['total_terminal_identity_pin_count']),(13,122))

    def test_terminal_identity_scope_rejections(self):
        changes=(lambda r:r.update(returncode=0),lambda r:r.update(abort='QUALIFIED_ORIGINAL_LEAF_RESOURCE_OR_DEADLINE'),
            lambda r:r.update(root_wait_reaped=False),lambda r:r['held_handle_exit'].update(actual_exit_code=0),
            lambda r:r['numeric_birth_absence'].update(original_birth_absent=False),lambda r:r.update(admission_sha256='1'*64),
            lambda r:r.update(original_identity={'SYNTHETIC_PID':14}),lambda r:r.update(GPU_observation_errors=['fake']),
            lambda r:r.update(whole_GPU_observed_peak_MiB=8193))
        for i,change in enumerate(changes):
            with self.subTest(i=i):
                m,s,r,_=fixture();change(r)
                with self.assertRaises(RuntimeError):m.thirteenth_failed_raw_only(s)

    def test_no_report_claim_or_legacy_owner_promotion(self):
        for change in (lambda s:s[EXTRA].update(original_report_absent=False),
                lambda s:s[EXTRA].update(returncode=True),
                lambda s:s[EXTRA].update(RAW_ref=s['additional_preserved_independent_CPU_failure']['RAW_ref'])):
            m,s,_,_=fixture();change(s)
            with self.assertRaises(RuntimeError):m.thirteenth_failed_raw_only(s)

    def test_all_failed_raw_checks_precede_first_source_gate(self):
        tree=ast.parse(PATH.read_text(encoding='utf-8'))
        gate=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='inherited_source_gate')
        calls=[n.value.func.id for n in gate.body if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name)]
        self.assertLess(calls.index('thirteenth_failed_raw_only'),calls.index('pre_gate'))
        prior=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='source_gate')
        update=next(n.value for n in prior.body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
            and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='update'
            and any(k.arg=='prior' for k in n.value.keywords))
        self.assertTrue(any(k.arg=='report' and isinstance(k.value,ast.Name) and k.value.id=='prior' for k in update.keywords))
        original=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='original_image')
        selector=next(n for n in ast.walk(original) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
            and n.func.attr=='selector_packet')
        self.assertEqual(ast.dump(selector.args[1]),ast.dump(ast.parse("gate['context']",mode='eval').body))


if __name__=='__main__':unittest.main()
