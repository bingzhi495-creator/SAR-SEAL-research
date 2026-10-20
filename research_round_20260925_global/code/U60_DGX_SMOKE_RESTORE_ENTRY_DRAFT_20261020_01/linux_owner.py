"""UNEXECUTED dedicated U60 SMOKE2 CPU restore owner; adapted from U54.

The original owned-session lifecycle and resource supervision are retained.
Only this specific restoration contract replaces the U54 FIT validation ABI.

No scientific imports, model construction, remote connection or launch occurs
on import. Actual psutil is imported only by the explicitly invoked parent CLI.
"""
import argparse
import ctypes
from copy import deepcopy
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import signal
import subprocess
import sys
import time
import traceback

LIMITS = dict(allowed_cpus=[0, 1], threads=2, maximum_RSS_bytes=16*1024**3,
              minimum_available_RAM_bytes=16*1024**3, deadline_seconds=1800)
SCHEMA = 'ROOT-U60-DGX-SMOKE2-CPU-restore-spec-v1'
OPERATION = 'CPU_U60_DGX_INDEPENDENT_SMOKE2_RESTORE'


HELPER_SHA256 = '0943096b6d9787bdb370212c8c311427c37f5ef1eb4c32ab397c1ea35faf3d14'

ENVIRONMENT = dict(python='3.12.14', torch='2.10.0+cu130', numpy='1.26.4', system='Linux', machine='aarch64')

RESTORE_PINS = dict(
    training_state='d80ab26ad7092431f4001535705e7afcc4fda7cbf66bacb4dc2cb66edd3d6724',
    training_entry='cd62a640e63210bcb63e58e09bb1e429f1a68ea66dcfec240831a5f1db31aca9',
    shared_initialization='4ba968db7688a03d5975c8546607791a5e09e82ef6c5c7d096cd6a368d1b0292',
    full_legacy='c210b318be29aca7d26516e9f47c8849e2c6a76642d4049fe7b133726ee809b9',
    full_dsconv='b4b612edf9f87cf29dc48bb675d501658fdec5dbee1ea363a7f02d86d197e5f0',
    full_head='6caba966ac7601b611fd1b36fdbd1d758875c5d029758fefe586738c8009f16e',
    full_feature='f7953780580449dd246cbdf1803652183c1dd178f2fd60b37adb238a6d2b6a5a',
    plain_legacy='902e13e5c60b1579459854c6bf4393ad8f268092a73a1d851f0fa640c810f48d',
    plain_feature='d93ff973a57ec3abfde98bb7b9ab4fcf5a3443445a512c2acd16460fce5bfbad',
    instance_adapter='9ad221170f69c028474f76700c991cb76dfecf19b11f5198767843b8e378d8b3',
    u36_helpers='1e726092a19d80a06f58d6a56f85f4c15bff2b744f0468a7e0a35fa2c85999e5',
    u39_training_ops='378c955877674a65f6b2180d4c269e01355245881229ed0d2ec555b184ee7e5d')

AUXILIARY_PINS = dict(
    image_geometry='65db107de80c5f1d4dade7cb2ffd3f241fd6bf7f9b233cb1aa9c90cbfd00abf7',
    portable_reader='b65ed9ecb7a9a704899f3c75983b2cb9676e26b42d87f65336cfbcb16bd94557')

def prelock():
    return dict(schema='U60-DGX-SMOKE2-CPU-restore-prelock-v1', environment=deepcopy(ENVIRONMENT),
        limits=deepcopy(LIMITS), CUDA_visible_devices='', mode='SMOKE', endpoint_updates=2,
        no_new_training=True, no_forward=True, no_FIT=True, no_DEV=True, no_TEST=True, no_AP=True)

def source_table(spec):
    keys = {'schema', 'operation', 'phase', 'entry_source', 'owner_source', 'restore_helper_source',
            'restore_sources', 'auxiliary_sources', 'library_sources', 'endpoint_refs', 'checkpoint_refs',
            'prelock', 'output_directory', 'owner_output_directory', 'sources'}
    need(type(spec) is dict and set(spec) == keys and spec['schema'] == SCHEMA and spec['operation'] == OPERATION
         and type(spec['phase']) is str and 1 <= len(spec['phase']) <= 160
         and '/' not in spec['phase'] and '\\' not in spec['phase'], 'Exact dedicated U60 spec')
    need(type(spec['prelock']) is dict and json.dumps(spec['prelock'], sort_keys=True, allow_nan=False)
         == json.dumps(prelock(), sort_keys=True, allow_nan=False), 'Exact typed CPU SMOKE2 prelock and unchanged limits')
    values = [spec[k] for k in ('entry_source', 'owner_source', 'restore_helper_source')]
    need(strict_ref(spec['restore_helper_source'])['sha256'] == HELPER_SHA256, 'Exact reviewed U58 helper')
    for key, pins in (('restore_sources', RESTORE_PINS), ('auxiliary_sources', AUXILIARY_PINS)):
        need(type(spec[key]) is dict and set(spec[key]) == set(pins), 'Exact original code roles: ' + key)
        for role, pin in pins.items():
            need(strict_ref(spec[key][role])['sha256'] == pin, 'Exact original code pin: ' + role)
        values += list(spec[key].values())
    for key, roles in (('library_sources', {'psutil', 'torch', 'numpy'}),
                       ('endpoint_refs', {'qualified', 'spec', 'pair_commit', 'order'}),
                       ('checkpoint_refs', {'PLAIN', 'FULL'})):
        need(type(spec[key]) is dict and set(spec[key]) == roles, 'Complete explicit role set: ' + key)
        values += list(spec[key].values())
    admitted = {}
    for value in values:
        strict_ref(value)
        need(value['path'] not in admitted, 'Distinct direct input/source roles, no aliasing')
        admitted[value['path']] = value
    need(spec['sources'] == [admitted[k] for k in sorted(admitted)], 'Only exact direct restoration Source closure')
    outputs = []
    for key in ('output_directory', 'owner_output_directory'):
        value = spec[key]
        need(type(value) is str and '\\' not in value and PurePosixPath(value).is_absolute()
             and str(PurePosixPath(value)) == value and value != '/' and '..' not in PurePosixPath(value).parts,
             'Canonical explicit output directory')
        outputs.append(PurePosixPath(value))
    need(outputs[0] != outputs[1] and outputs[0] not in outputs[1].parents
         and outputs[1] not in outputs[0].parents, 'Separate owner and worker outputs')
    for value in admitted:
        path = PurePosixPath(value)
        need(all(out != path and out not in path.parents for out in outputs), 'Outputs cannot contain admitted sources')
    return admitted

def need(ok, message):
    if not ok:
        raise ValueError(message)


def strict_ref(value):
    need(type(value) is dict and set(value) == {'path', 'bytes', 'sha256'}
         and type(value['path']) is str and '\\' not in value['path']
         and PurePosixPath(value['path']).is_absolute() and str(PurePosixPath(value['path'])) == value['path']
         and '..' not in PurePosixPath(value['path']).parts and type(value['bytes']) is int and value['bytes'] > 0
         and type(value['sha256']) is str and len(value['sha256']) == 64
         and set(value['sha256']) <= set('0123456789abcdef'), 'Exact positive canonical POSIX file reference')
    return value


class FileAllowlist:
    def __init__(self, references):
        need(type(references) is list and len(references) > 0, 'Finite explicit file allowlist')
        self.references = {}
        for reference in references:
            strict_ref(reference)
            key = str(Path(reference['path']).resolve(strict=True))
            need(key == reference['path'] and key not in self.references,
                 'Unique exact canonical path; Linux paths are case-sensitive')
            self.references[key] = deepcopy(reference)

    def closed(self, reference, check=None):
        strict_ref(reference)
        path = Path(reference['path'])
        need(self.references.get(str(path)) == reference, 'File is not explicitly admitted')
        need(str(path.resolve(strict=True)) == str(path), 'Admitted canonical path changed')
        before = path.stat()
        digest = hashlib.sha256()
        size = 0
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024*1024), b''):
                if check is not None:
                    check()
                size += len(block)
                need(size <= reference['bytes'], 'File exceeded declared byte count')
                digest.update(block)
        after = path.stat()
        identity = lambda st: (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
        need(identity(before) == identity(after) and size == reference['bytes']
             and digest.hexdigest() == reference['sha256'], 'Stable exact admitted bytes required')
        return path


def resource_violations(elapsed, rss, available, affinity):
    need(type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed >= 0
         and type(rss) is int and rss >= 0 and type(available) is int and available >= 0,
         'Actual finite nonnegative resource observations required')
    result = []
    if elapsed > LIMITS['deadline_seconds']:
        result.append('deadline')
    if rss > LIMITS['maximum_RSS_bytes']:
        result.append('RSS')
    if available < LIMITS['minimum_available_RAM_bytes']:
        result.append('free_RAM')
    if affinity != LIMITS['allowed_cpus']:
        result.append('CPU_affinity')
    return result


class ChildControl:
    def __init__(self, spec, psutil_module):
        need(spec['prelock']['limits'] == LIMITS, 'Unchanged U60 CPU limits')
        self.psutil = psutil_module
        self.files = FileAllowlist(spec['sources'])
        self.started = time.monotonic()
        self.process = psutil_module.Process()
        self.observed = dict(schema='U60-Linux-child-control-observations-v1',
                             samples=0, peak_RSS_bytes=0,
                             minimum_available_RAM_bytes=None, violations=[])

    def check(self):
        rss = int(self.process.memory_info().rss)
        available = int(self.psutil.virtual_memory().available)
        self.observed['samples'] += 1
        self.observed['peak_RSS_bytes'] = max(self.observed['peak_RSS_bytes'], rss)
        previous = self.observed['minimum_available_RAM_bytes']
        self.observed['minimum_available_RAM_bytes'] = available if previous is None else min(previous, available)
        violations = resource_violations(time.monotonic()-self.started, rss, available,
                                         self.process.cpu_affinity())
        self.observed['violations'].extend(v for v in violations if v not in self.observed['violations'])
        need(not self.observed['violations'], 'Child resource violation: '+','.join(self.observed['violations']))

    def closed(self, reference):
        self.check()
        result = self.files.closed(reference, self.check)
        self.check()
        return result

    def summary(self):
        result = deepcopy(self.observed)
        result['elapsed_seconds'] = time.monotonic()-self.started
        return result


def make_control(spec, psutil_module):
    return ChildControl(spec, psutil_module)


def parse_proc_stat(text):
    closing = text.rfind(')')
    need(closing >= 0 and '(' in text[:closing], 'Linux /proc stat format')
    pid = int(text[:text.index('(')].strip())
    fields = text[closing+1:].split()
    need(len(fields) >= 20, 'Complete Linux /proc stat identity')
    return dict(pid=pid, state=fields[0], parent_pid=int(fields[1]),
                process_group=int(fields[2]), session=int(fields[3]), start_ticks=int(fields[19]))


def proc_identity(pid):
    try:
        return parse_proc_stat(Path('/proc/%d/stat' % pid).read_text())
    except FileNotFoundError:
        return None


def birth_absent(original, current):
    return current is None or (current['pid'], current['start_ticks']) != (original['pid'], original['start_ticks'])


def terminal_success(returncode, reaped, original_birth_absent, remaining_group, violations, worker_status):
    return (type(returncode) is int and returncode == 0 and reaped is True
            and original_birth_absent is True and not remaining_group and not violations
            and worker_status == 'COMPLETE_PENDING_ROOT')


def _json_write(path, value, exclusive=False):
    raw = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode('utf-8')
    if exclusive:
        with Path(path).open('xb') as stream:
            stream.write(raw);stream.flush();os.fsync(stream.fileno())
    else:
        # A signal may interrupt a previous write. Keep that partial evidence;
        # never let its temporary filename prevent the terminal state write.
        temporary = Path(str(path)+'.%d.%d.next' % (os.getpid(), time.monotonic_ns()))
        with temporary.open('xb') as stream:
            stream.write(raw);stream.flush();os.fsync(stream.fileno())
        os.replace(temporary, path)


def validate_spec(spec):
    return source_table(spec)


def completed_validation(report, spec, spec_ref):
    """Only JSON checks in the stdlib parent; no CP deserialize or model replay."""
    source = FileAllowlist([spec['entry_source']]).closed(spec['entry_source'])
    namespace = dict(__name__='_U60_OWNER_WORKER_CONTRACT', __file__=str(source))
    exec(compile(source.read_bytes(), str(source), 'exec'), namespace)
    need(namespace['SCHEMA'] == SCHEMA and namespace['OPERATION'] == OPERATION,
         'Exact dedicated U60 worker ABI')
    return namespace['completed_report'](report, spec, spec_ref)


def _group_members(psutil_module, original):
    current = proc_identity(original['pid'])
    need(current is None or not birth_absent(original, current), 'Owned leader PID was reused; refuse to signal another birth')
    members = []
    for process in psutil_module.process_iter(['pid']):
        pid = process.info['pid']
        identity = proc_identity(pid)
        if identity is None or identity['process_group'] != original['pid']:
            continue
        need(identity['session'] == original['pid'] and identity['start_ticks'] >= original['start_ticks'],
             'Only the session/process group created for this child may be observed or signalled')
        try:
            rss = int(process.memory_info().rss)
            affinity = process.cpu_affinity()
        except psutil_module.NoSuchProcess:
            continue
        members.append(dict(identity, RSS_bytes=rss, CPU_affinity=affinity))
    return members


def _kill_owned_group(psutil_module, child, original, receipt):
    """Only this Popen's unique session/group; never signal an external user job."""
    receipt['termination_requested'] = True
    for signum, seconds in ((signal.SIGTERM, 2.), (signal.SIGKILL, 5.)):
        members = _group_members(psutil_module, original)
        if not members:
            break
        try:
            os.killpg(original['pid'], signum)
        except ProcessLookupError:
            continue  # Group finished between observation and signal; never retry another PID.
        receipt['signals'].append(dict(signal=signum.name, owned_group=original['pid'],
                                       member_identities=members, monotonic=time.monotonic()))
        end = time.monotonic()+seconds
        while time.monotonic() < end:
            child.poll()  # actual waitpid(WNOHANG) reaps the direct child when ready
            if not _group_members(psutil_module, original):
                break
            time.sleep(.05)
    try:
        receipt['returncode'] = child.wait(timeout=1.)
        receipt['reaped'] = True
    except subprocess.TimeoutExpired:
        receipt['reaped'] = False
    receipt['remaining_owned_group'] = _group_members(psutil_module, original)


def _prepare_child(expected_parent_pid):
    # Parent is single-threaded. This runs after setsid and before exec, with no ML.
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
    libc.prctl.restype = ctypes.c_int
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
        raise OSError(ctypes.get_errno(), 'PR_SET_PDEATHSIG failed')
    # Close the race if the owner died before the kernel death signal was armed.
    if os.getppid() != expected_parent_pid:
        os.kill(os.getpid(), signal.SIGKILL)
    os.sched_setaffinity(0, LIMITS['allowed_cpus'])


def run_owner(spec, spec_path, spec_sha, psutil_module):
    """One attempt, no retries. Output receipt is evidence pending Root acceptance."""
    validate_spec(spec)
    need(platform.system() == 'Linux' and platform.machine() == 'aarch64', 'Dedicated Linux/aarch64 owner only')
    need(not any(name.split('.')[0] in ('torch', 'numpy', 'PIL', 'pycocotools') for name in sys.modules),
         'Parent must remain free of scientific imports')
    need(Path(spec['owner_source']['path']).resolve(strict=True) == Path(__file__).resolve(strict=True), 'Actual admitted owner source')
    out = Path(spec['owner_output_directory'])
    worker_out = Path(spec['output_directory'])
    need(not out.exists() and not out.is_symlink() and not worker_out.exists() and not worker_out.is_symlink(),
         'Both outputs must be fresh; existing attempt may never be restarted')
    need(out.parent.is_dir() and worker_out.parent.is_dir(), 'Root must precreate the dedicated parent directories')
    out.mkdir()
    started = time.monotonic()
    receipt = dict(schema='U60-Linux-owned-SMOKE2-CPU-restore-receipt-v1', status='PREFLIGHT', operation=OPERATION,
        phase=spec['phase'], spec_ref=dict(path=str(spec_path), bytes=spec_path.stat().st_size, sha256=spec_sha),
        owner_pid=os.getpid(), owner_identity=proc_identity(os.getpid()),
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        child_identity=None, child_launched=False, returncode=None, reaped=False,
        original_birth_absent=None, numeric_pid_present=None, remaining_owned_group=[],
        final_group_scan_resolved=False,
        termination_requested=False, signals=[], violations=[], observations=[],
        CPU_only=True, CUDA_visible_devices='', GPU_allocator_contract_applicable=False,
        qualified=False, automatic_retry=False, U40_frontier_modified=False, formal_training_budget_credit=0,
        AP=False, DEV=False, TEST=False, new_training=False, pilot_only=False, main_training=False,
        innovation_established=False, speedup_established=False, CPU_restore_complete=False,
        restored_arms=[], restored_cursor=None,
        peak_group_RSS_bytes=0, minimum_available_RAM_bytes=None, worker_report_ref=None,
        sys_executable=sys.executable, python_version=platform.python_version(),
        psutil_version=str(psutil_module.__version__),
        source_footprint='Explicit selected source/module files, not complete Python/OS/shared-library closure')
    child = None
    original = None
    stdout = stderr = None
    last_written = 0.
    handled_signals = (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)
    outside_signals = {value: signal.getsignal(value) for value in handled_signals}
    termination_seen = []

    def terminate_owner(signum, frame):
        if not termination_seen:
            termination_seen.append(signum)
            raise RuntimeError('Owner received '+signal.Signals(signum).name+'; terminate and reap its owned group')

    for value in handled_signals:
        signal.signal(value, terminate_owner)
    try:
        import threading
        need(threading.active_count() == 1, 'Owner must be single-threaded before child preexec affinity')
        need(str(psutil_module.__version__) == '6.1.1', 'Actual reviewed psutil6.1.1 runtime')
        parent_process = psutil_module.Process()
        # Keep the small owner off the child's dedicated CPU0/1 where possible.
        available_cpus = sorted(os.sched_getaffinity(0))
        need(0 in available_cpus and 1 in available_cpus, 'CPU0/1 must be available in this Linux cpuset')
        other_cpus = [cpu for cpu in available_cpus if cpu not in (0, 1)]
        if other_cpus:
            parent_process.cpu_affinity([other_cpus[0]])
        receipt['owner_CPU_affinity'] = parent_process.cpu_affinity()
        receipt['child_CPU_affinity'] = [0, 1]
        files = FileAllowlist(spec['sources'])

        def preflight_check():
            need(time.monotonic()-started <= LIMITS['deadline_seconds'], 'Owner preflight deadline exceeded')
            need(int(psutil_module.virtual_memory().available) >= LIMITS['minimum_available_RAM_bytes'], 'Owner preflight free RAM below16GiB')
            need(int(parent_process.memory_info().rss) <= LIMITS['maximum_RSS_bytes'], 'Owner preflight RSS exceeded16GiB')

        for reference in spec['sources']:
            files.closed(reference, preflight_check)
        need(hashlib.sha256(spec_path.read_bytes()).hexdigest() == spec_sha, 'Spec unchanged before launch')
        environment = os.environ.copy()
        environment.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                           OPENBLAS_NUM_THREADS='2', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
        command = [sys.executable, '-u', spec['entry_source']['path'], '--spec', str(spec_path), '--spec-sha', spec_sha]
        stdout = (out/'stdout.log').open('xb')
        stderr = (out/'stderr.log').open('xb')
        preflight_check()
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                 env=environment, start_new_session=True, preexec_fn=partial(_prepare_child, os.getpid()),
                                 close_fds=True, cwd=str(Path(spec['entry_source']['path']).parent))
        receipt['child_launched'] = True
        original = proc_identity(child.pid)
        need(original is not None and original['process_group'] == child.pid and original['session'] == child.pid,
             'Capture real Linux PID/startticks and independent session before monitoring')
        receipt.update(child_identity=original, status='RUNNING', launch_elapsed_seconds=time.monotonic()-started)
        receipt['parent_death_signal'] = dict(signal='SIGKILL', prctl='PR_SET_PDEATHSIG',
            expected_parent_pid=os.getpid(), preexec_succeeded=True,
            scope='Direct child only; not inherited automatically by descendant processes')
        _json_write(out/'RUN_STATE.json', receipt, exclusive=True)
        while True:
            returncode = child.poll()
            members = _group_members(psutil_module, original)
            elapsed = time.monotonic()-started
            rss = sum(row['RSS_bytes'] for row in members)
            available = int(psutil_module.virtual_memory().available)
            affinity = [0, 1] if all(row['CPU_affinity'] == [0, 1] for row in members) else []
            violations = resource_violations(elapsed, rss, available, affinity)
            receipt['peak_group_RSS_bytes'] = max(receipt['peak_group_RSS_bytes'], rss)
            prior = receipt['minimum_available_RAM_bytes']
            receipt['minimum_available_RAM_bytes'] = available if prior is None else min(prior, available)
            receipt['violations'].extend(v for v in violations if v not in receipt['violations'])
            if elapsed-last_written >= 5. or returncode is not None or violations:
                receipt['observations'].append(dict(elapsed_seconds=elapsed, group_RSS_bytes=rss,
                    available_RAM_bytes=available, member_identities=members, child_returncode=returncode))
                receipt['elapsed_seconds'] = elapsed
                _json_write(out/'RUN_STATE.json', receipt)
                last_written = elapsed
            if violations:
                receipt['status'] = 'RESOURCE_OR_DEADLINE_FAILURE'
                _kill_owned_group(psutil_module, child, original, receipt)
                break
            if returncode is not None:
                receipt['returncode'] = child.wait()
                receipt['reaped'] = True
                if members:
                    receipt['violations'].append('owned_group_survived_direct_child')
                    _kill_owned_group(psutil_module, child, original, receipt)
                break
            time.sleep(.1)
        report_path = worker_out/'REPORT.json'
        if report_path.is_file():
            need(report_path.stat().st_size <= 64*1024**2, 'Bounded worker JSON report')
            raw = report_path.read_bytes()
            worker_report = json.loads(raw)
            receipt['worker_report_ref'] = dict(path=str(report_path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
            receipt['worker_report_status'] = worker_report.get('status')
            body = worker_report.get('body_report', {})
            receipt['body_status'] = body.get('status')
            if receipt['worker_report_status']=='COMPLETE':
                body=completed_validation(worker_report,spec,receipt['spec_ref'])
                receipt.update(CPU_restore_complete=True,restored_arms=['PLAIN','FULL'],
                               restored_cursor=deepcopy(body['cursor']))
                receipt['worker_cost']=deepcopy(worker_report['cost'])
                receipt['child_resource_observations']=deepcopy(worker_report['resource_observations'])

    except BaseException as error:
        receipt.update(status='FAILED', error=dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc()))
        if child is not None and original is not None:
            try:
                _kill_owned_group(psutil_module, child, original, receipt)
            except BaseException as cleanup_error:
                receipt['cleanup_error'] = dict(type=type(cleanup_error).__name__, message=str(cleanup_error))
        elif child is not None:
            # No unverified group signal. The still-owned Popen leader can be killed/reaped.
            child.kill()
            try:
                receipt['returncode'] = child.wait(timeout=5.)
                receipt['reaped'] = True
            except subprocess.TimeoutExpired:
                receipt['reaped'] = False
    finally:
        if stdout is not None:
            stdout.close()
        if stderr is not None:
            stderr.close()
        receipt['elapsed_seconds'] = time.monotonic()-started
        if original is not None:
            current = proc_identity(original['pid'])
            receipt['original_birth_absent'] = birth_absent(original, current)
            receipt['numeric_pid_present'] = current is not None
            try:
                receipt['remaining_owned_group'] = _group_members(psutil_module, original)
                receipt['final_group_scan_resolved'] = True
            except BaseException as error:
                receipt['violations'].append('final_group_identity_unresolved')
                receipt['group_error'] = str(error)
        success = terminal_success(receipt['returncode'], receipt['reaped'], receipt['original_birth_absent'],
            receipt['remaining_owned_group'], receipt['violations'], receipt.get('body_status'))
        success = success and receipt.get('worker_report_status') == 'COMPLETE' and receipt['final_group_scan_resolved']
        success = success and receipt['CPU_restore_complete'] is True
        if success and 'error' not in receipt and 'cleanup_error' not in receipt:
            receipt['status'] = 'OWNED_CHILD_COMPLETE_PENDING_ROOT_REVIEW'
        elif receipt['status'] not in ('RESOURCE_OR_DEADLINE_FAILURE',):
            receipt['status'] = 'FAILED'
        receipt['controller_returning'] = True
        receipt['controller_exit_observation'] = 'Caller must independently observe this parent actual terminal; not self-certified'
        receipt['child_alive'] = child is not None and child.poll() is None
        receipt['all_owned_processes_absent'] = (receipt['original_birth_absent'] is True
            and receipt['final_group_scan_resolved'] and not receipt['remaining_owned_group'])
        receipt['owner_termination_signal'] = termination_seen[0] if termination_seen else None
        receipt['logs'] = {}
        for name in ('stdout.log', 'stderr.log'):
            path = out/name
            if path.exists():
                digest = hashlib.sha256()
                with path.open('rb') as stream:
                    for block in iter(lambda: stream.read(1024*1024), b''):
                        digest.update(block)
                receipt['logs'][name] = dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())
        _json_write(out/'RUN_STATE.json', receipt)
        _json_write(out/'OWNER_RECEIPT.json', receipt, exclusive=True)
        for value, handler in outside_signals.items():
            signal.signal(value, handler)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--spec-sha', required=True)
    args = parser.parse_args(argv)
    path = Path(args.spec)
    need(path.is_absolute() and path.is_file() and path.stat().st_size <= 2*1024**2, 'Small absolute spec JSON')
    raw = path.read_bytes()
    need(hashlib.sha256(raw).hexdigest() == args.spec_sha, 'Explicit Root-pinned spec SHA')
    spec = json.loads(raw)
    validate_spec(spec)
    need(platform.system() == 'Linux' and platform.machine() == 'aarch64', 'Linux/aarch64 actual owner only')
    # Verify selected psutil source before the sole non-stdlib parent import.
    files = FileAllowlist(spec['sources'])
    expected = spec['library_sources']['psutil']
    files.closed(expected)
    import psutil
    need(str(Path(psutil.__file__).resolve(strict=True)) == expected['path'], 'Actual psutil imported from admitted source')
    receipt = run_owner(spec, path, args.spec_sha, psutil)
    return 0 if receipt['status'] == 'OWNED_CHILD_COMPLETE_PENDING_ROOT_REVIEW' else 1


if __name__ == '__main__':
    sys.exit(main())
