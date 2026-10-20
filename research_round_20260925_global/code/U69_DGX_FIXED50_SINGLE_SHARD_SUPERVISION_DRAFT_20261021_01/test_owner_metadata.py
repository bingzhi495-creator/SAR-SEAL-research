"""New U69 stdlib/OS-boundary stubs only; never launch a process or import ML."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
SCHEMA = 'ROOT-U69-DGX-fixed50-single-shard-supervision-spec-v1'
OPERATION = 'GPU_U69_DGX_FIXED50_SINGLE_SHARD_PREDICTION'


def module():
    spec = importlib.util.spec_from_file_location('u69_owner', HERE/'linux_owner.py')
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def reference(path, raw):
    return dict(path=path, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


class MemoryPath:
    def __init__(self, path, raw):
        self.path, self.raw = path, raw

    def __str__(self):
        return self.path

    def read_bytes(self):
        return self.raw


def fixture():
    text = ("SCHEMA="+repr(SCHEMA)+"\nOPERATION="+repr(OPERATION)+"\n"
        "def validate_envelope(spec):\n"
        "    if spec['schema']!=SCHEMA or spec['operation']!=OPERATION:raise ValueError('wrong synthetic schema')\n"
        "    return {r['path']:r for r in spec['sources']}\n"
        "def close_source_union(spec,closed):\n"
        "    closed(spec['request']['worker_source']).read_bytes()\n"
        "    closed(spec['request']['manifest_ref']).read_bytes()\n"
        "    if spec.get('reject_union'):raise ValueError('synthetic extra payload refused')\n"
        "    return validate_envelope(spec)\n"
        "def completed_entry(report,spec,spec_ref):\n"
        "    if report['status']!='COMPLETE_PENDING_ROOT_ACCEPTANCE':raise ValueError('incomplete synthetic entry')\n"
        "    return report['worker_report']\n")
    objects = {'/synthetic/entry.py': text.encode(), '/synthetic/owner.py': b'# owner',
               '/synthetic/u68.py': b'# harmless worker', '/synthetic/manifest.json': b'{}'}
    refs = {p: reference(p, raw) for p, raw in objects.items()}
    spec = dict(schema=SCHEMA, operation=OPERATION, phase='synthetic-only',
        entry_source=refs['/synthetic/entry.py'], owner_source=refs['/synthetic/owner.py'],
        request=dict(worker_source=refs['/synthetic/u68.py'], manifest_ref=refs['/synthetic/manifest.json']),
        sources=[refs[p] for p in sorted(refs)])
    observed = []

    class Files:
        def __init__(self, references):
            self.admitted = {r['path']: r for r in references}

        def closed(self, ref, check=None):
            observed.append(ref['path'])
            if self.admitted.get(ref['path']) != ref:
                raise ValueError('synthetic path not admitted')
            raw = objects[ref['path']]
            if reference(ref['path'], raw) != ref:
                raise ValueError('synthetic byte mismatch')
            return MemoryPath(ref['path'], raw)

    return spec, objects, observed, Files


class OwnerTests(unittest.TestCase):
    def test_only_byte_closed_stdlib_entry_contract_defines_envelope(self):
        owner = module()
        spec, objects, observed, files = fixture()
        with patch.object(owner, 'FileAllowlist', files):
            admitted = owner.validate_spec(spec)
            self.assertEqual(admitted, {r['path']: r for r in spec['sources']})
            self.assertEqual(observed, ['/synthetic/entry.py'])
            objects['/synthetic/entry.py'] += b'\nraise RuntimeError("unadmitted bytes must not execute")'
            with self.assertRaisesRegex(ValueError, 'byte mismatch'):
                owner.validate_spec(spec)

    def test_manifest_union_precedes_any_payload_hash(self):
        owner = module()
        spec, objects, observed, files = fixture()
        extra = reference('/synthetic/extra.pth', b'never-read')
        spec['sources'].append(extra)
        spec['reject_union'] = True
        with patch.object(owner, 'FileAllowlist', files):
            with self.assertRaisesRegex(ValueError, 'extra payload refused'):
                owner.admit_sources(spec)
        self.assertEqual(observed, ['/synthetic/entry.py', '/synthetic/u68.py', '/synthetic/manifest.json'])
        self.assertNotIn(extra['path'], observed)

    def test_same_cpu_RSS_free_RAM_and_deadline_guards_apply_to_child_control(self):
        owner = module()
        spec, objects, observed, files = fixture()
        spec['limits'] = deepcopy(owner.LIMITS)
        available = [16*1024**3]
        process = SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=1234), cpu_affinity=lambda: [0, 1])
        psutil = SimpleNamespace(Process=lambda: process,
            virtual_memory=lambda: SimpleNamespace(available=available[0]))
        with patch.object(owner, 'FileAllowlist', files), patch('time.monotonic', return_value=0):
            control = owner.make_control(spec, psutil)
            control.check()
            self.assertEqual(control.summary()['peak_RSS_bytes'], 1234)
            available[0] -= 1
            with self.assertRaisesRegex(ValueError, 'free_RAM'):
                control.check()
            self.assertIn('free_RAM', control.summary()['violations'])
        self.assertEqual(owner.resource_violations(1801, 16*1024**3+1, 16*1024**3-1, [2]),
                         ['deadline', 'RSS', 'free_RAM', 'CPU_affinity'])

    def test_monitor_uses_actual_wait_return_and_preserves_terminal_evidence(self):
        owner = module()
        for actual in (0, 5):
            child, original, psutil, identity = process_fixture(actual)
            receipt = process_receipt()
            writes = []
            with patch.object(owner, 'proc_identity', side_effect=identity, create=True), \
                 patch.object(owner, '_json_write', side_effect=lambda p, r, **k: writes.append(deepcopy(r)), create=True), \
                 patch('time.monotonic', return_value=0), patch('time.sleep', return_value=None):
                owner.supervise_child(psutil, child, original, receipt, 0, Path('/synthetic/owner'))
            self.assertEqual(child.waits, 1)
            self.assertEqual(receipt['returncode'], actual)
            self.assertTrue(receipt['reaped'])
            self.assertTrue(writes)
            self.assertEqual(owner.terminal_success(actual, True, True, [], [],
                             'COMPLETE_PENDING_ROOT_ACCEPTANCE'), actual == 0)

    def test_launch_boundary_fixes_environment_and_session_without_launching(self):
        owner = module()
        spec = {'entry_source': {'path': '/synthetic/entry.py'}}
        fake, stdout, stderr = object(), object(), object()
        with patch('subprocess.Popen', return_value=fake) as popen, patch('os.getpid', return_value=900), \
             patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': 'old', 'CUBLAS_WORKSPACE_CONFIG': 'old'}):
            child = owner.launch_child(spec, Path('/synthetic/spec.json'), 'a'*64, stdout, stderr)
            self.assertIs(child, fake)
            popen.assert_called_once()
            args, kwargs = popen.call_args
            self.assertEqual(args[0][1:], ['-u', '/synthetic/entry.py', '--spec', str(Path('/synthetic/spec.json')), '--spec-sha', 'a'*64])
            self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], '0')
            self.assertEqual(kwargs['env']['CUBLAS_WORKSPACE_CONFIG'], ':4096:8')
            self.assertTrue(all(kwargs['env'][k] == '2' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')))
            self.assertTrue(kwargs['start_new_session'])
            self.assertTrue(kwargs['close_fds'])
            self.assertEqual(kwargs['preexec_fn'].args, (900,))
            self.assertIs(kwargs['stdout'], stdout)
            self.assertIs(kwargs['stderr'], stderr)
            self.assertEqual(owner.os.environ['CUDA_VISIBLE_DEVICES'], 'old')

    def test_owner_receipt_follows_stub_child_terminal_not_completed_report(self):
        owner = module()
        for actual in (0, 5):
            with tempfile.TemporaryDirectory(dir=HERE) as root:
                base = Path(root).resolve()
                entry, outputs, worker_out = base/'entry.py', base/'ENTRY', base/'WORKER'
                entry.write_text('# synthetic unused entry')
                spec = dict(phase='synthetic', operation=OPERATION, schema=SCHEMA,
                    entry_source={'path': str(entry)}, owner_source={'path': owner.__file__},
                    output_directory=str(outputs), owner_output_directory=str(base/'OWNER'),
                    request={'output_directory': str(worker_out)}, sources=[], limits=deepcopy(owner.LIMITS))
                spec_path = base/'SPEC.json'
                spec_path.write_text(json.dumps(spec))
                digest = hashlib.sha256(spec_path.read_bytes()).hexdigest()
                child, original, psutil, identity = process_fixture(actual)
                affinity = [[0, 1, 2]]

                def cpu(value=None):
                    if value is not None:
                        affinity[0] = value
                    return affinity[0]

                parent = SimpleNamespace(cpu_affinity=cpu, memory_info=lambda: SimpleNamespace(rss=1234))
                psutil.Process = lambda: parent
                psutil.__version__ = '6.1.1'

                def launch(*args):
                    outputs.mkdir()
                    worker_out.mkdir()
                    (outputs/'REPORT.json').write_text(json.dumps(dict(status='COMPLETE_PENDING_ROOT_ACCEPTANCE',
                        worker_report=dict(status='COMPLETE_PENDING_ROOT_ACCEPTANCE', prediction_ref={'synthetic': True}),
                        worker_report_ref={'synthetic': True}, resource_observations={'synthetic': True})))
                    return child

                contract = {'completed_entry': lambda r, s, sr: r['worker_report']}
                files = SimpleNamespace(closed=lambda ref, check=None: Path(ref['path']))
                with patch.object(owner, 'admit_sources', return_value=(files, contract)), \
                     patch.object(owner, 'validate_spec', return_value={}), \
                     patch.object(owner, 'launch_child', side_effect=launch) as launch_stub, \
                     patch.object(owner, 'proc_identity', side_effect=lambda pid: identity(pid) if pid == 200 else {'pid': 900}), \
                     patch.object(owner, 'boot_id', return_value='synthetic-boot', create=True), \
                     patch('platform.system', return_value='Linux'), patch('platform.machine', return_value='aarch64'), \
                     patch('platform.python_version', return_value='3.12.14'), \
                     patch('os.getpid', return_value=900), patch('os.sched_getaffinity', return_value={0, 1, 2}, create=True), \
                     patch('signal.getsignal', return_value=None), patch('signal.signal', return_value=None), \
                     patch.object(signal, 'SIGHUP', 1, create=True), patch('time.sleep', return_value=None):
                    result = owner.run_owner(spec, spec_path, digest, psutil)
                launch_stub.assert_called_once()
                self.assertEqual(result['returncode'], actual)
                self.assertEqual(result['status'], 'OWNED_CHILD_COMPLETE_PENDING_ROOT_REVIEW' if actual == 0 else 'FAILED')
                self.assertFalse(result['qualified'])
                self.assertIsNone(result['whole_GPU_observed_bytes'])
                self.assertFalse(result['whole_GPU_WDDM8GiB_contract_certified'])
                self.assertNotIn('owner_process_exit_code', result)
                self.assertIn('independently observe', result['controller_exit_observation'])
                self.assertEqual(json.loads((base/'OWNER/OWNER_RECEIPT.json').read_text()), result)

    def test_inherited_resource_kill_is_owned_and_reused_PID_is_never_signalled(self):
        owner = module()
        child, original, psutil, identity = process_fixture(-15, 16*1024**3-1)
        receipt = process_receipt()

        def signal_owned(group, signum):
            self.assertEqual(group, 200)
            self.assertEqual(signum, signal.SIGTERM)
            child.alive = False

        with patch.object(owner, 'proc_identity', side_effect=identity), \
             patch.object(owner, '_json_write', return_value=None), \
             patch.object(signal, 'SIGKILL', SimpleNamespace(name='SIGKILL'), create=True), \
             patch('os.killpg', side_effect=signal_owned, create=True) as kill, \
             patch('time.monotonic', return_value=0), patch('time.sleep', return_value=None):
            owner.supervise_child(psutil, child, original, receipt, 0, Path('/synthetic/owner'))
        kill.assert_called_once()
        self.assertEqual(receipt['status'], 'RESOURCE_OR_DEADLINE_FAILURE')
        self.assertEqual(receipt['returncode'], -15)
        self.assertTrue(receipt['reaped'])
        self.assertIn('free_RAM', receipt['violations'])
        child, original, psutil, identity = process_fixture()
        reused = dict(original, start_ticks=101)
        with patch.object(owner, 'proc_identity', return_value=reused), \
             patch.object(owner, '_json_write', return_value=None), \
             patch('os.killpg', create=True) as kill:
            with self.assertRaisesRegex(ValueError, 'PID was reused'):
                owner.supervise_child(psutil, child, original, process_receipt(), 0, Path('/synthetic/owner'))
        kill.assert_not_called()


def process_receipt():
    return dict(status='RUNNING', returncode=None, reaped=False, violations=[], observations=[],
                peak_group_RSS_bytes=0, minimum_available_RAM_bytes=None, signals=[], termination_requested=False)


def process_fixture(actual=0, available=16*1024**3):
    original = dict(pid=200, start_ticks=100, state='R', parent_pid=900, process_group=200, session=200)

    class Child:
        pid = 200

        def __init__(self):
            self.alive, self.polls, self.waits = True, 0, 0

        def poll(self):
            self.polls += 1
            if self.polls >= 2:
                self.alive = False
                return 0  # Never use this planned/poll value as actual wait evidence.
            return None

        def wait(self, timeout=None):
            self.waits += 1
            self.alive = False
            return actual

    child = Child()
    proc = SimpleNamespace(info={'pid': 200}, memory_info=lambda: SimpleNamespace(rss=1234),
                           cpu_affinity=lambda: [0, 1])
    psutil = SimpleNamespace(process_iter=lambda fields: [proc] if child.alive else [],
        virtual_memory=lambda: SimpleNamespace(available=available), NoSuchProcess=ProcessLookupError)
    return child, original, psutil, lambda pid: deepcopy(original) if child.alive else None


if __name__ == '__main__':
    unittest.main()
