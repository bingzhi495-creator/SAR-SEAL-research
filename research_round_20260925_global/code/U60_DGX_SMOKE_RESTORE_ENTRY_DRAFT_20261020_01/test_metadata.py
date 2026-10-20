"""Stdlib-only U60 outer wiring tests. No real model/runtime acceptance."""
import importlib.util
import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


entry = load('u60_metadata_entry', Path(__file__).with_name('entry.py'))


class Wiring(unittest.TestCase):
    def test_cpu_environment_refuses_visible_cuda_before_any_builder(self):
        good = dict(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                    OPENBLAS_NUM_THREADS='2')
        entry.validate_environment(good)
        for key, value in [('CUDA_VISIBLE_DEVICES', '0'), ('OMP_NUM_THREADS', '8'),
                           ('MKL_NUM_THREADS', None), ('OPENBLAS_NUM_THREADS', '1')]:
            bad = dict(good)
            if value is None:
                bad.pop(key)
            else:
                bad[key] = value
            with self.assertRaises(ValueError):
                entry.validate_environment(bad)

    def test_original_pair_is_built_once_then_both_distinct_smoke_arms(self):
        calls = []
        adapters = {'PLAIN': object(), 'FULL': object()}
        receipt = {'from': 'actual stub builder, never caller input'}
        def build_pair(torch, components, references):
            calls.append(('construct', tuple(components), tuple(references)))
            return adapters, receipt
        def restore(torch, np, control, adapter, arm, metadata, fresh, **kwargs):
            self.assertIs(adapter, adapters[arm])
            self.assertIs(fresh, receipt)
            calls.append(('restore', arm))
            kwargs['report'].update(status='RESTORED_CPU_ONLY', formal_endpoint=False,
                                    mode='SMOKE', model_forward_executed=False)
            return kwargs['report']
        modules = {'shared_initialization': SimpleNamespace(
            SOURCE_SHA256={'x': 'pin'}, build_pair=build_pair), 'x': object()}
        helper = SimpleNamespace(validate_smoke_endpoint=lambda *a: {
            'mode': 'SMOKE', 'cursor': {'global_updates': 2, 'completed_epochs': 0, 'within_epoch': 2}},
            restore_smoke_arm=restore)
        modules.update(training_state=object(), training_entry=object())
        torch = SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: False))
        report = {}
        result = entry.restore_pair(torch, object(), SimpleNamespace(check=lambda: None), helper,
                                    modules, {'x': 'reference'}, {'spec': {'prelock': {'mode': 'SMOKE'}}}, report)
        self.assertIs(result, report)
        self.assertEqual(calls, [('construct', ('x',), ('x',)),
                                 ('restore', 'PLAIN'), ('restore', 'FULL')])
        self.assertFalse(result['qualified'])
        self.assertFalse(result['numerical_runtime_qualified'])
        self.assertFalse(result['AP'])
        self.assertEqual(result['status'], 'COMPLETE_PENDING_ROOT')

    def test_formal_refused_before_helper_or_constructor(self):
        with self.assertRaises(ValueError):
            entry.restore_pair(None, None, None, None, {}, {},
                               {'spec': {'prelock': {'mode': 'FORMAL'}}}, {})

    def test_only_direct_restore_inputs_are_admitted(self):
        spec = fixture()
        admitted = entry.source_table(spec)
        self.assertEqual(len(admitted), len(spec['sources']))
        bad = deepcopy(spec)
        bad['sources'].append(ref('forbidden/FIT-package'))
        with self.assertRaises(ValueError):
            entry.source_table(bad)
        for change in ('missing_arm', 'wrong_helper', 'wrong_runtime', 'wrong_limits'):
            bad = deepcopy(spec)
            if change == 'missing_arm':
                del bad['checkpoint_refs']['FULL']
            elif change == 'wrong_helper':
                bad['restore_helper_source']['sha256'] = '0' * 64
            elif change == 'wrong_runtime':
                bad['prelock']['environment']['torch'] = '2.0.1+cu118'
            else:
                bad['prelock']['limits']['threads'] = 8
            with self.assertRaises(ValueError):
                entry.source_table(bad)

    def test_dedicated_owner_rejects_training_and_cannot_claim_audit_741(self):
        owner = load('u60_metadata_owner', Path(__file__).with_name('linux_owner.py'))
        spec = fixture()
        owner.validate_spec(spec)
        bad = deepcopy(spec)
        bad['operation'] = 'GPU_U57_DGX_ORIGINAL_UPDATE_TRAINING'
        with self.assertRaises(ValueError):
            owner.validate_spec(bad)
        self.assertEqual(owner.resource_violations(1., 1, 16*1024**3, [0, 1]), [])
        for values in ((1801., 1, 16*1024**3, [0, 1]),
                       (1., 16*1024**3+1, 16*1024**3, [0, 1]),
                       (1., 1, 16*1024**3-1, [0, 1]), (1., 1, 16*1024**3, [0])):
            self.assertTrue(owner.resource_violations(*values))
        self.assertTrue(owner.terminal_success(0, True, True, [], [], 'COMPLETE_PENDING_ROOT'))
        self.assertFalse(owner.terminal_success(1, True, True, [], [], 'COMPLETE_PENDING_ROOT'))
        self.assertFalse(owner.terminal_success(0, True, True, [{'pid': 1}], [], 'COMPLETE_PENDING_ROOT'))

    def test_second_arm_failure_retains_partial_and_does_not_retry(self):
        calls = []
        def builder(*args):
            calls.append('construct')
            return {'PLAIN': object(), 'FULL': object()}, {}
        def restore(*args, **kwargs):
            arm = args[4]
            calls.append(arm)
            if arm == 'FULL':
                raise ValueError('synthetic full arm failure')
            kwargs['report'].update(status='RESTORED_CPU_ONLY', formal_endpoint=False,
                                    mode='SMOKE', model_forward_executed=False)
            return kwargs['report']
        helper = SimpleNamespace(validate_smoke_endpoint=lambda *a: dict(mode='SMOKE',
            cursor=dict(global_updates=2, completed_epochs=0, within_epoch=2)), restore_smoke_arm=restore)
        modules = dict(training_state=object(), training_entry=object(), shared_initialization=SimpleNamespace(
            SOURCE_SHA256={}, build_pair=builder))
        report = {}
        with self.assertRaisesRegex(ValueError, 'synthetic full'):
            entry.restore_pair(SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: False)),
                object(), SimpleNamespace(check=lambda: None), helper, modules, {},
                {'spec': {'prelock': {'mode': 'SMOKE'}}}, report)
        self.assertEqual(calls, ['construct', 'PLAIN', 'FULL'])
        self.assertEqual(report['status'], 'FAILED')
        self.assertEqual(report['arms']['PLAIN']['status'], 'RESTORED_CPU_ONLY')
        self.assertFalse(report['qualified'])

    def test_entry_owner_pure_contracts_agree_and_original_supervision_is_retained(self):
        owner_path = Path(__file__).with_name('linux_owner.py')
        owner = load('u60_contract_owner', owner_path)
        for key in ('SCHEMA', 'OPERATION', 'LIMITS', 'ENVIRONMENT', 'RESTORE_PINS', 'AUXILIARY_PINS'):
            self.assertEqual(getattr(entry, key), getattr(owner, key))
        source = owner_path.read_text(encoding='utf-8')
        self.assertNotIn('private_audit_ref', source)
        self.assertNotIn('CPU_validation_complete', source)
        tree = ast.parse(source)
        run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run_owner')
        calls = [node for node in ast.walk(run) if isinstance(node, ast.Call)]
        self.assertTrue(any(isinstance(node.func, ast.Attribute) and node.func.attr == 'Popen'
            and any(k.arg == 'start_new_session' and isinstance(k.value, ast.Constant)
                    and k.value.value is True for k in node.keywords) for node in calls))
        # Importing either draft is standard-library only; these are source checks.
        for path in (owner_path, Path(__file__).with_name('entry.py')):
            for node in ast.parse(path.read_text(encoding='utf-8')).body:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    roots = ([n.name.split('.')[0] for n in node.names] if isinstance(node, ast.Import)
                             else [node.module.split('.')[0]])
                    self.assertFalse(set(roots) & {'torch', 'numpy', 'psutil'})


def ref(name, pin='a' * 64):
    return dict(path='/metadata-test-only/' + name, bytes=1, sha256=pin)


def fixture():
    spec = dict(schema=entry.SCHEMA, operation=entry.OPERATION, phase='metadata-test-only',
        entry_source=ref('entry.py'), owner_source=ref('linux_owner.py'),
        restore_helper_source=ref('restore_state.py', entry.HELPER_SHA256),
        restore_sources={k: ref(k + '.py', v) for k, v in entry.RESTORE_PINS.items()},
        auxiliary_sources={k: ref(k + '.py', v) for k, v in entry.AUXILIARY_PINS.items()},
        library_sources={k: ref(k + '/__init__.py') for k in ('psutil', 'torch', 'numpy')},
        endpoint_refs={k: ref(k + '.json') for k in ('qualified', 'spec', 'pair_commit', 'order')},
        checkpoint_refs={k: ref(k + '.pth') for k in ('PLAIN', 'FULL')},
        prelock=entry.prelock(), output_directory='/metadata-test-only/out/WORKER',
        owner_output_directory='/metadata-test-only/out/OWNER')
    values = [spec[k] for k in ('entry_source', 'owner_source', 'restore_helper_source')]
    for key in ('restore_sources', 'auxiliary_sources', 'library_sources', 'endpoint_refs', 'checkpoint_refs'):
        values += list(spec[key].values())
    spec['sources'] = sorted(values, key=lambda v: v['path'])
    return spec


if __name__ == '__main__':
    unittest.main()
