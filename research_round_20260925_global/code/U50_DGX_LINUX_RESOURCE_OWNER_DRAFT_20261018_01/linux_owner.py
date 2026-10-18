"""Unbound U50 Linux owner and explicit child control; stdlib-only on import.

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
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import traceback

LIMITS = dict(allowed_cpus=[0, 1], threads=2, allocator_bytes=4*1024**3,
              maximum_RSS_bytes=16*1024**3, minimum_available_RAM_bytes=16*1024**3,
              deadline_seconds=1800)
SCHEMA = 'ROOT-U50-DGX-paired-pilot-spec-v1'
OPERATION = 'GPU_U50_DGX_PAIRED_DISPOSABLE_PILOT'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def strict_ref(value):
    need(type(value) is dict and set(value) == {'path', 'bytes', 'sha256'}
         and type(value['path']) is str and Path(value['path']).is_absolute()
         and type(value['bytes']) is int and value['bytes'] >= 0
         and type(value['sha256']) is str and len(value['sha256']) == 64
         and set(value['sha256']) <= set('0123456789abcdef'), 'Exact absolute path/bytes/SHA256 ref')
    return value


class FileAllowlist:
    def __init__(self, references):
        need(type(references) is list and 1 <= len(references) <= 64, 'Finite explicit file allowlist')
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
        need(spec['prelock']['limits'] == LIMITS, 'Unchanged U50 limits')
        self.psutil = psutil_module
        self.files = FileAllowlist(spec['sources'])
        self.started = time.monotonic()
        self.process = psutil_module.Process()
        self.observed = dict(schema='U50-Linux-child-control-observations-v1',
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
            and worker_status == 'COMPLETED_PENDING_ROOT_REVIEW')


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
    keys = {'schema', 'operation', 'phase', 'entry_source', 'pilot_source', 'owner_source',
            'source_refs', 'library_sources', 'portable_refs', 'prelock', 'output_directory',
            'owner_output_directory', 'sources'}
    need(type(spec) is dict and set(spec) == keys and spec['schema'] == SCHEMA
         and spec['operation'] == OPERATION and type(spec['phase']) is str and spec['phase'], 'Exact U50 owner spec')
    need(set(spec['source_refs']) == {'shared_initialization', 'full_legacy', 'full_dsconv', 'full_head',
         'full_feature', 'plain_legacy', 'plain_feature', 'instance_adapter', 'image_geometry', 'u36_helpers',
         'u39_training_ops'} and set(spec['library_sources']) == {'psutil', 'torch', 'numpy'}
         and set(spec['portable_refs']) == {'54', '710'}, 'Eleven numeric roles, three libraries, two data packages')
    need(spec['prelock']['limits'] == LIMITS, 'Exact independent U50 owner limits')
    values = [spec[key] for key in ('entry_source', 'pilot_source', 'owner_source')]
    values += list(spec['source_refs'].values())+list(spec['library_sources'].values())+list(spec['portable_refs'].values())
    expected = {}
    for value in values:
        strict_ref(value)
        need(value['path'] not in expected or expected[value['path']] == value, 'Conflicting source refs')
        expected[value['path']] = value
    need(spec['sources'] == [expected[path] for path in sorted(expected)], 'Exact sorted finite Linux source allowlist')
    out, worker = (Path(spec[key]) for key in ('owner_output_directory', 'output_directory'))
    need(out.is_absolute() and worker.is_absolute() and out != worker
         and out not in worker.parents and worker not in out.parents,
         'Distinct nonnested absolute owner and worker output directories')


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
    receipt = dict(schema='U50-Linux-owned-pilot-receipt-v1', status='PREFLIGHT', operation=OPERATION,
        phase=spec['phase'], spec_ref=dict(path=str(spec_path), bytes=spec_path.stat().st_size, sha256=spec_sha),
        owner_pid=os.getpid(), owner_identity=proc_identity(os.getpid()),
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        child_identity=None, child_launched=False, returncode=None, reaped=False,
        original_birth_absent=None, numeric_pid_present=None, remaining_owned_group=[],
        final_group_scan_resolved=False,
        termination_requested=False, signals=[], violations=[], observations=[],
        whole_GPU_observed_peak_MiB=None, whole_GPU_WDDM8GiB_contract_certified=False,
        GPU_observation_scope='GB10 UMA whole-card used/total unavailable here; None is not zero or a passed WDDM gate',
        allocator_limit_bytes=LIMITS['allocator_bytes'], allocator_enforced_by='original pilot body before model/input CUDA allocation',
        qualified=False, automatic_retry=False, U40_frontier_modified=False, formal_training_budget_credit=0,
        AP=False, DEV=False, TEST=False, new_training=None, pilot_only=True,
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
        environment.update(CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
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
            receipt['new_training'] = worker_report.get('new_training')
            receipt['successful_optimizer_updates'] = worker_report.get('successful_optimizer_updates')
            receipt['optimizer_steps_entered'] = worker_report.get('optimizer_steps_entered')
            need(worker_report['schema'] == 'U50-DGX-paired-pilot-worker-report-v1'
                 and worker_report['operation'] == OPERATION and worker_report['phase'] == spec['phase']
                 and worker_report['spec_ref'] == receipt['spec_ref'], 'Exact worker report identity')
            if receipt['worker_report_status'] == 'COMPLETE':
                need(receipt['body_status'] == 'COMPLETED_PENDING_ROOT_REVIEW'
                     and all(worker_report[key] == body[key] == 6 for key in
                             ('successful_optimizer_updates', 'optimizer_steps_entered'))
                     and worker_report['new_training'] is body['new_training'] is True,
                     'Worker success requires actual body six returned/entered updates')
                need(body['main_training'] is False and body['pilot_only'] is True
                     and body['formal_training_budget_credit'] == 0,
                     'Disposable independent pilot, no main-training budget credit')
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
