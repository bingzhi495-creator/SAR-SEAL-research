"""U71-only synthetic JSON and process-boundary stubs; no real child or science."""
from copy import deepcopy
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import tempfile
from types import SimpleNamespace
from types import ModuleType
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location('u71_'+name, HERE/(name+'.py'))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def ref(path, raw=b'x'):
    return dict(path=path, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def envelope():
    root = HERE.parent
    raw = (root/'U70_DGX_FIXED50_POPULATION_METRICS_WORKER_DRAFT_20261021_01/worker.py').read_bytes()
    worker = load_source(raw, '/synthetic/u70.py')
    request = dict(worker_source=ref('/synthetic/u70.py', raw), core_source=dict(path='/synthetic/core.py', **worker.CORE_PIN),
        module_sources={k: ref('/synthetic/'+k+'.py') for k in worker.MODULE_ROLES},
        library_sources={k: ref('/synthetic/lib/'+k+'.py') for k in worker.LIBRARY_ROLES},
        input_refs={k: ref('/synthetic/'+k+'.json') for k in worker.INPUT_ROLES},
        shard_refs=[ref('/synthetic/shard'+str(i)+'.json') for i in range(4)], output_directory='/synthetic/WORKER')
    for k, pin in worker.SOURCE_PINS.items():
        request['module_sources'][k].update(pin)
    admissions = [ref('/synthetic/NOT_REAL_ADMISSION_'+str(i)+'.json') for i in range(4)]
    entry = load('entry')
    spec = dict(schema=entry.SCHEMA, operation=entry.OPERATION, phase='synthetic-only',
        entry_source=ref('/synthetic/entry.py'), owner_source=ref('/synthetic/owner.py'), request=request,
        limits=deepcopy(entry.LIMITS), external_shard_admission_refs=admissions,
        output_directory='/synthetic/ENTRY', owner_output_directory='/synthetic/OWNER')
    refs = worker.direct_references(request)+[spec['entry_source'],spec['owner_source']]+admissions
    spec['sources'] = sorted(refs, key=lambda r: r['path'])
    return spec, raw


def load_source(raw, path):
    value = SimpleNamespace()
    scope = {'__name__':'_SYNTHETIC_DEFINITIONS', '__file__':path}
    exec(compile(raw, path, 'exec'), scope)
    value.__dict__.update(scope)
    return value


class MemoryPath:
    def __init__(self, path, raw):
        self.path, self.raw = path, raw

    def __str__(self):
        return self.path

    def read_bytes(self):
        return self.raw


class Tests(unittest.TestCase):
    def test_CPU_launch_has_real_session_wait_contract_without_GPU_environment(self):
        owner = load('linux_owner')
        with patch('subprocess.Popen', return_value='synthetic-child') as spawn, patch('os.getpid', return_value=900):
            result = owner.launch_child({'entry_source': {'path': '/synthetic/entry.py'}}, Path('/synthetic/SPEC.json'), 'a'*64, object(), object())
        self.assertEqual(result, 'synthetic-child')
        spawn.assert_called_once()
        args, kwargs = spawn.call_args
        self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], '')
        self.assertTrue(kwargs['start_new_session'])
        self.assertEqual(kwargs['preexec_fn'].args, (900,))
        self.assertEqual(owner.LIMITS['allowed_cpus'], [0,1])
        self.assertNotIn('allocator_bytes', owner.LIMITS)
        self.assertEqual(owner.OPERATION, 'CPU_U71_DGX_FIXED50_POPULATION_METRICS')

    def test_exact_U70_union_and_external_gate_refs_precede_payload_reads(self):
        entry = load('entry')
        spec, raw = envelope()
        visited = []

        def closed(value):
            visited.append(value['path'])
            self.assertEqual(value, spec['request']['worker_source'])
            return MemoryPath(value['path'], raw)

        admitted = entry.close_source_union(spec, closed)
        self.assertEqual(admitted, {r['path']: r for r in spec['sources']})
        self.assertEqual(visited, ['/synthetic/u70.py'])
        spec['sources'].append(ref('/synthetic/forbidden-extra.json'))
        spec['sources'].sort(key=lambda r: r['path'])
        with self.assertRaisesRegex(ValueError, 'union'):
            entry.close_source_union(spec, closed)
        spec['external_shard_admission_refs'] = []
        with self.assertRaisesRegex(ValueError, 'four'):
            entry.validate_envelope(spec)

    def test_entry_verifies_actual_three_outputs_and_preserves_success_AP(self):
        entry = load('entry')
        spec, raw = envelope()
        with tempfile.TemporaryDirectory(dir=HERE) as root:
            base = Path(root).resolve()
            spec['output_directory'], spec['owner_output_directory'] = str(base/'ENTRY'), str(base/'OWNER')
            spec['request']['output_directory'] = str(base/'WORKER')
            control = SimpleNamespace(check=lambda: None, closed=lambda r: MemoryPath(r['path'], b'x'),
                summary=lambda: dict(schema='U71-Linux-child-control-observations-v1', samples=1, peak_RSS_bytes=1234,
                    minimum_available_RAM_bytes=16*1024**3, violations=[], elapsed_seconds=0.1))
            worker = ModuleType('_synthetic_worker');worker.__file__=spec['request']['worker_source']['path']

            def run(c, request, *, report):
                out=Path(request['output_directory']);out.mkdir()
                summary=dict(status='METRICS_COMPLETE_PENDING_ROOT_ACCEPTANCE',qualified=False,ROOT_actual_terminal_self_certified=False,
                    AP=True,DEV=True,TEST=False,new_training=False,formal_training_budget_credit=0,images=187,GT=380,queries=18700,
                    geometry={'scope':{'selected_native_nonignored_GT':0}})
                report.update(schema='U70-population-metrics-worker-report-v1',status='COMPLETE_PENDING_ROOT_ACCEPTANCE',
                    qualified=False,ROOT_actual_terminal_self_certified=False,AP=True,AP_computation_status='COMPLETE_PENDING_ROOT_ACCEPTANCE',
                    DEV=True,DEV_technical_input=True,TEST=False,new_training=False,formal_training_budget_credit=0,CPU_only=True,
                    Torch_imported=False,CUDA_visible_devices='',automatic_retry=False,full_binary_closure_claimed=False,
                    cleanup_errors=[],source_aliases_restored=True,outer_runtime_states_restored=True,
                    all_direct_input_bytes_rechecked_after=True,annotation_JSON_parsed=True,evaluation_started=True,
                    request=deepcopy(request))
                report['summary_ref']=entry.write_json(out/'SUMMARY.json',summary)
                report['private_pair_records_ref']=entry.write_json(out/'FIXED_NATIVE_PAIRS_PRIVATE.json',[])
                entry.write_json(out/'REPORT.json',report)
                return report

            worker.run_worker=run
            report={}
            with patch.object(entry,'validate_envelope',return_value={}),patch.object(entry,'close_source_union',return_value={}), \
                 patch.object(entry,'no_scientific_imports',return_value=True):
                returned=entry.execute_owned(spec,ref('/synthetic/SPEC.json'),control,worker,report=report)
            self.assertIs(returned,report)
            self.assertTrue(report['AP']);self.assertFalse(report['qualified'])
            self.assertEqual(report['status'],'COMPLETE_PENDING_ROOT_ACCEPTANCE')
            self.assertTrue(report['flags']['private_pairs_bytes_verified'])
            self.assertEqual(json.loads((base/'ENTRY/REPORT.json').read_text()),report)

    def test_wrong_spec_SHA_never_creates_output_or_imports_runtime(self):
        entry=load('entry')
        with tempfile.TemporaryDirectory(dir=HERE) as root:
            path=Path(root)/'SPEC.json'
            path.write_text(json.dumps({'output_directory':str(Path(root)/'ENTRY')}))
            with patch('importlib.import_module',side_effect=AssertionError('no runtime allowed')):
                with self.assertRaisesRegex(ValueError,'SHA'):
                    entry.main(['--spec',str(path.resolve()),'--spec-sha','a'*64])
            self.assertFalse((Path(root)/'ENTRY').exists())

    def test_CLI_guard_runs_after_all_helper_definitions(self):
        for name in ('entry','linux_owner'):
            tree=ast.parse((HERE/(name+'.py')).read_text())
            self.assertIsInstance(tree.body[-1],ast.If)
            self.assertIn('__main__',ast.dump(tree.body[-1]))

    def test_failed_worker_AP_unknown_and_primary_error_survive_resource_cleanup(self):
        entry=load('entry');spec,raw=envelope();started=[False]
        with tempfile.TemporaryDirectory(dir=HERE) as root:
            base=Path(root).resolve();spec['output_directory']=str(base/'ENTRY');spec['request']['output_directory']=str(base/'WORKER')
            def check():
                if started[0]:raise ValueError('synthetic resource failure')
            control=SimpleNamespace(check=check,closed=lambda r:MemoryPath(r['path'],b'x'),summary=lambda:dict(
                schema='U71-Linux-child-control-observations-v1',samples=2,peak_RSS_bytes=1234,
                minimum_available_RAM_bytes=16*1024**3-1,violations=['free_RAM'],elapsed_seconds=0.1))
            worker=ModuleType('_synthetic_worker');worker.__file__=spec['request']['worker_source']['path']
            def fail(c,request,*,report):
                started[0]=True;out=Path(request['output_directory']);out.mkdir()
                report.update(status='FAILED',AP=None,AP_computation_status='STARTED_COMPLETION_UNKNOWN',evaluation_started=True,
                    annotation_JSON_parsed=True)
                entry.write_json(out/'REPORT.json',report)
                raise RuntimeError('synthetic original metric failure')
            worker.run_worker=fail;report={}
            with patch.object(entry,'validate_envelope',return_value={}),patch.object(entry,'close_source_union',return_value={}), \
                 patch.object(entry,'no_scientific_imports',return_value=True):
                with self.assertRaisesRegex(RuntimeError,'original metric failure'):
                    entry.execute_owned(spec,ref('/synthetic/SPEC.json'),control,worker,report=report)
            saved=json.loads((base/'ENTRY/REPORT.json').read_text())
            self.assertIsNone(saved['AP']);self.assertTrue(saved['evaluation_started'])
            self.assertIn('original metric failure',saved['error']['traceback'])
            self.assertEqual(saved['resource_observations']['violations'],['free_RAM'])
            self.assertIn('worker_report_ref',saved)

    def test_owner_actual_wait_and_birth_protection_are_retained(self):
        owner=load('linux_owner')
        original=dict(pid=200,start_ticks=100,state='R',parent_pid=900,process_group=200,session=200)
        class Child:
            pid=200
            def poll(self):return 0
            def wait(self,timeout=None):return 5
        receipt=dict(status='RUNNING',violations=[],observations=[],peak_group_RSS_bytes=0,
            minimum_available_RAM_bytes=None,returncode=None,reaped=False)
        psutil=SimpleNamespace(process_iter=lambda fields:[],virtual_memory=lambda:SimpleNamespace(available=16*1024**3))
        with patch.object(owner,'proc_identity',return_value=None),patch.object(owner,'_json_write',return_value=None):
            owner.supervise_child(psutil,Child(),original,receipt,owner.time.monotonic(),Path('/synthetic/OWNER'))
        self.assertEqual(receipt['returncode'],5);self.assertTrue(receipt['reaped'])
        self.assertFalse(owner.terminal_success(5,True,True,[],[],'COMPLETE_PENDING_ROOT_ACCEPTANCE'))
        with patch.object(owner,'proc_identity',return_value=dict(original,start_ticks=101)),patch('os.killpg',create=True) as kill:
            with self.assertRaisesRegex(ValueError,'PID was reused'):
                owner.supervise_child(psutil,Child(),original,deepcopy(receipt),owner.time.monotonic(),Path('/synthetic/OWNER'))
            kill.assert_not_called()


if __name__ == '__main__':
    unittest.main()
