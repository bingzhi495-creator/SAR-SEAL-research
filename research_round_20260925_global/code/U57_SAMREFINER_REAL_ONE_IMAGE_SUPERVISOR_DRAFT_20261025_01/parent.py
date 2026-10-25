"""Unbound one-attempt SAMRefiner supervisor; stdlib-only at import."""
import time


def wait_child(child, sample, *, deadline_seconds=1800):
    """Observe resources, then actually wait/reap one already owned child."""
    started = time.monotonic()
    while True:
        sample()
        if child.poll() is not None:
            return dict(returncode=child.wait(), reaped=True)
        if time.monotonic() - started >= deadline_seconds:
            raise TimeoutError('Owned single-attempt deadline exceeded')
        time.sleep(0.05)


import argparse
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import threading
import types

OPERATION = 'GPU_SAMREFINER_ORIGINAL_ORDINAL0_ONE_IMAGE_TRIAL'
WORKER_PIN = dict(bytes=27920, sha256='a9b7910fcd18d53752122b8e983044b9ab4855a0484950cd4e90b88079b2c8b8')
OWNER_PIN = dict(bytes=37956, sha256='6c6e2867bc7a36624f09e6483eadaaf6cd2f8340001e839000b7d85c5ac6659e')
CONTROL_PIN = dict(bytes=33056, sha256='763c097e77c1ec5faa4c69adde5512feced9e52ad8b3e942c5337027434ef373')
DEADLINE = 1800


def need(ok, message):
    if not ok:
        raise ValueError(message)


def reference(path):
    path = Path(path)
    digest = hashlib.sha256()
    size = 0
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
            size += len(block)
    return dict(path=str(path), bytes=size, sha256=digest.hexdigest())


def load_source(ref, alias):
    path = Path(ref['path'])
    need(str(path.resolve(strict=True)) == ref['path'] and reference(path) == ref,
         'Exact canonical source bytes required')
    module = types.ModuleType(alias)
    module.__file__ = str(path)
    sys.modules[alias] = module
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    return module


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--spec-sha', required=True)
    args = parser.parse_args(argv)
    spec_path = Path(args.spec)
    need(spec_path.is_absolute() and spec_path.stat().st_size <= 2*1024**2, 'Small absolute supervisor SPEC required')
    raw = spec_path.read_bytes()
    need(hashlib.sha256(raw).hexdigest() == args.spec_sha, 'Exact Root-pinned parent SPEC digest required')
    spec = json.loads(raw)
    need(spec.get('schema') == 'ROOT-SAMRefiner-one-image-supervisor-spec-v1'
         and spec.get('operation') == OPERATION, 'Independent one-image supervisor scope required')
    need(platform.system() == 'Linux' and platform.machine() == 'aarch64'
         and not any(name.split('.')[0] in ('torch', 'numpy') for name in sys.modules)
         and threading.active_count() == 1, 'Fresh single-threaded Linux/aarch64 stdlib parent required')
    sys.dont_write_bytecode = True
    for role, pin in (('owner_source', OWNER_PIN), ('worker_source', WORKER_PIN)):
        need(all(spec[role][key] == value for key, value in pin.items()), 'Unchanged reviewed source required: '+role)
    out = Path(spec['output_directory'])
    need(out.is_absolute() and str(out.resolve()) == str(out), 'Canonical fresh private parent output required')
    out.mkdir(mode=0o700, exist_ok=False)
    started = time.monotonic()
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    receipt = dict(schema='SAMRefiner-one-image-parent-control-v1', operation=OPERATION,
        run_id=spec['run_id'], status='PREFLIGHT', qualified=False, automatic_retry=False,
        maximum_attempts=1, attempts_entered=0, deadline_seconds=DEADLINE,
        boot_id=boot, pid=os.getpid(), worker_source=spec['worker_source'],
        child_launched=False, returncode=None, reaped=False, original_birth_absent=None,
        final_group_scan_resolved=False, remaining_owned_group=None, signals=[], violations=[],
        samples=0, peak_group_RSS_bytes=0, minimum_available_RAM_bytes=None, observations=[],
        external_live_GPU_exclusivity_self_certified=False, AP=False, training_updates=0)
    child = original = owner = psutil = None
    stdout = stderr = None
    worker_spec = None
    cleanup_in_progress = [False]
    termination_seen = []
    handled_signals = (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)
    old_handlers = {value: signal.getsignal(value) for value in handled_signals}

    def interrupt(signum, frame):
        if not termination_seen:
            termination_seen.append(signum)
            if not cleanup_in_progress[0]:
                raise RuntimeError('Supervisor received '+signal.Signals(signum).name)

    for value in handled_signals:
        signal.signal(value, interrupt)

    def save():
        receipt['elapsed_seconds'] = time.monotonic()-started
        temporary = out/'CONTROL.next.json'
        temporary.write_text(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False)+'\n')
        os.replace(temporary, out/'CONTROL.json')

    def preflight_check():
        need(Path('/proc/sys/kernel/random/boot_id').read_text().strip() == boot, 'Same boot required')
        need(time.monotonic()-started < DEADLINE, 'Parent preflight deadline exceeded')
        need(int(psutil.virtual_memory().available) >= 16*1024**3, 'Parent available-RAM floor violation')
        need(int(psutil.Process().memory_info().rss) <= 16*1024**3, 'Parent RSS violation')

    save()
    try:
        owner = load_source(spec['owner_source'], '_SAM_PROBE_FIXED_OWNER_HELPERS')
        values = [spec[key] for key in ('parent_source', 'owner_source', 'worker_source',
                  'worker_spec_ref', 'root_lease_ref', 'checkpoint_stage_ref', 'timeout_ref')]
        values.append(spec['library_sources']['psutil'])
        unique = {}
        for value in values:
            owner.strict_ref(value)
            need(value['path'] not in unique or unique[value['path']] == value, 'Repeated refs must retain bytes/SHA')
            unique[value['path']] = value
        need(spec['sources'] == [unique[path] for path in sorted(unique)], 'Exact finite parent source/metadata closure required')
        files = owner.FileAllowlist(spec['sources'])
        files.closed(spec['library_sources']['psutil'])
        import psutil as selected_psutil
        psutil = selected_psutil
        need(reference(Path(psutil.__file__).resolve()) == spec['library_sources']['psutil']
             and str(psutil.__version__) == '6.1.1', 'Actual reviewed psutil source/version required')
        for value in spec['sources']:
            files.closed(value, preflight_check)
        need(Path(__file__).resolve() == files.closed(spec['parent_source']), 'Actual admitted parent source required')
        need(hashlib.sha256(spec_path.read_bytes()).hexdigest() == args.spec_sha, 'Parent SPEC changed')
        worker = load_source(spec['worker_source'], '_SAM_PROBE_WORKER_METADATA_CONTRACT')
        need(not any(name.split('.')[0] in ('torch', 'numpy') for name in sys.modules), 'Parent imported scientific runtime')
        worker_spec = json.loads(files.closed(spec['worker_spec_ref']).read_bytes())
        worker.validate_spec(worker_spec)
        need(worker_spec['source_refs']['worker'] == spec['worker_source']
             and worker_spec['root_lease_ref'] == spec['root_lease_ref']
             and worker_spec['checkpoint_stage_ref'] == spec['checkpoint_stage_ref']
             and worker_spec['library_sources']['psutil'] == spec['library_sources']['psutil']
             and worker_spec['run_id'] == spec['run_id'], 'Parent/child exact reference and run identity join required')
        control_ref = worker_spec['source_refs']['resource_control']
        need(all(control_ref[key] == value for key, value in CONTROL_PIN.items()),
             'Child must retain the original U65 CPU/RSS control component')
        pre = worker_spec['prelock']
        need(pre['limits']['allocator_bytes'] == 16*1024**3
             and pre['startup_min_available_RAM_bytes'] == 48*1024**3,
             'This supervisor is prospectively bound only to the independent 16GiB/48GiB plan')
        worker_out = Path(worker_spec['output_directory'])
        need(not worker_out.exists() and worker_out != out
             and out not in worker_out.parents and worker_out not in out.parents,
             'Fresh separate parent and child output directories required')
        need(all(out != Path(value['path']) and out not in Path(value['path']).parents
                 for value in spec['sources'] + worker_spec['sources']), 'Parent output cannot contain admitted inputs')
        stage = json.loads(files.closed(spec['checkpoint_stage_ref']).read_bytes())
        worker.validate_checkpoint_stage(worker_spec, stage)
        receipt['Root_checkpoint_certificate_consumed'] = True
        lease = json.loads(files.closed(spec['root_lease_ref']).read_bytes())
        need(lease.get('schema') == 'Root-SAMRefiner-one-image-lease-v1' and lease.get('status') == 'ROOT_GRANTED'
             and lease.get('operation') == OPERATION and lease.get('run_id') == spec['run_id']
             and lease.get('worker_ref') == spec['worker_source'] and lease.get('prelock') == pre
             and lease.get('single_attempt') is True and lease.get('exclusive_GPU') is True
             and lease.get('main_scientific_processes_reaped') is True
             and lease.get('scope') == 'one_image_resource_and_function_probe_only'
             and lease.get('parent_source_ref') == spec['parent_source']
             and lease.get('parent_output_directory') == str(out)
             and lease.get('worker_output_directory') == str(worker_out)
             and lease.get('boot_id') == boot, 'Exact independent Root lease required; this parent never grants one')
        priors = lease.get('prior_scientific_groups')
        need(type(priors) is list and len(priors) > 0, 'Explicit Root-reaped prior scientific group births required')
        for prior in priors:
            need(type(prior) is dict and all(type(prior.get(key)) is int and prior[key] > 0
                 for key in ('pid', 'start_ticks', 'process_group', 'session'))
                 and prior['pid'] == prior['process_group'] == prior['session'], 'Actual prior group leader birth required')
            need(owner.birth_absent(prior, owner.proc_identity(prior['pid']))
                 and not owner._group_members(psutil, prior), 'Prior scientific birth/group is not strictly absent')
        receipt.update(prior_scientific_groups_absent=True, Root_lease_record_verified=True,
                       allocator_bytes=16*1024**3, startup_min_available_RAM_bytes=48*1024**3)
        preflight_check()
        available = int(psutil.virtual_memory().available)
        need(available >= 48*1024**3, 'Independent startup RAM gate below48GiB')
        receipt['startup_available_RAM_bytes'] = available
        cpus = sorted(os.sched_getaffinity(0))
        need(0 in cpus and 1 in cpus, 'CPU0/1 must be available')
        other = [cpu for cpu in cpus if cpu not in (0, 1)]
        if other:
            psutil.Process().cpu_affinity([other[0]])
        python = Path(spec['worker_python'])
        need(python.is_absolute() and python.is_file()
             and python.resolve() == Path(sys.executable).resolve(), 'Explicit same-base qualified worker interpreter required')
        timeout = files.closed(spec['timeout_ref'])
        need(timeout.name == 'timeout' and os.access(timeout, os.X_OK), 'Admitted GNU timeout executable required')
        remaining = DEADLINE-(time.monotonic()-started)
        need(remaining > 1, 'No remaining one-attempt launch budget')
        command = [str(timeout), '--signal=KILL', '%.3fs' % remaining,
                   str(python), '-B', '-u', spec['worker_source']['path'],
                   '--spec', spec['worker_spec_ref']['path'], '--spec-sha', spec['worker_spec_ref']['sha256']]
        environment = os.environ.copy()
        environment.update(CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                           OPENBLAS_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
        stdout = (out/'stdout.log').open('xb')
        stderr = (out/'stderr.log').open('xb')
        receipt.update(status='LAUNCHING', attempts_entered=1, command=command)
        save()
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
            close_fds=True, cwd=str(Path(spec['worker_source']['path']).parent), env=environment,
            start_new_session=True, preexec_fn=partial(owner._prepare_child, os.getpid()))
        receipt['child_launched'] = True
        original = owner.proc_identity(child.pid)
        need(original is not None and original['pid'] == original['process_group'] == original['session'] == child.pid,
             'Capture the real timeout leader PID=PGID=SID before monitoring')
        receipt.update(status='RUNNING', child_identity=original,
            parent_death_signal_scope='Direct timeout child only; worker descendants require owned-group cleanup')
        save()
        last_saved = [-5.0]
        def sample():
            preflight_check()
            members = owner._group_members(psutil, original)
            elapsed = time.monotonic()-started
            rss = sum(member['RSS_bytes'] for member in members)
            available = int(psutil.virtual_memory().available)
            violations = []
            if elapsed >= DEADLINE: violations.append('deadline')
            if rss > 16*1024**3: violations.append('group_RSS')
            if available < 16*1024**3: violations.append('free_RAM')
            if any(member['CPU_affinity'] != [0, 1] for member in members): violations.append('CPU_affinity')
            receipt['samples'] += 1
            receipt['peak_group_RSS_bytes'] = max(receipt['peak_group_RSS_bytes'], rss)
            old = receipt['minimum_available_RAM_bytes']
            receipt['minimum_available_RAM_bytes'] = available if old is None else min(old, available)
            receipt['violations'].extend(value for value in violations if value not in receipt['violations'])
            if elapsed-last_saved[0] >= 5 or violations or child.poll() is not None:
                receipt['observations'].append(dict(elapsed_seconds=elapsed, group_RSS_bytes=rss,
                    available_RAM_bytes=available, group_members=members))
                save(); last_saved[0] = elapsed
            need(not violations, 'Sampled resource violation: '+','.join(violations))
        receipt.update(wait_child(child, sample, deadline_seconds=max(0.001, DEADLINE-(time.monotonic()-started))))
    except BaseException as error:
        receipt.update(status='FAILED_PRESERVED_NO_RETRY', error_type=type(error).__name__, error=str(error))
    finally:
        cleanup_in_progress[0] = True
        if child is not None:
            if original is not None:
                try:
                    current_returncode = child.poll()
                    remaining = owner._group_members(psutil, original)
                    if current_returncode is not None and remaining:
                        receipt['violations'].append('owned_group_survived_direct_child')
                    if current_returncode is None or remaining:
                        owner._kill_owned_group(psutil, child, original, receipt)
                    if not receipt['reaped']:
                        receipt.update(returncode=child.wait(timeout=1), reaped=True)
                    need(Path('/proc/sys/kernel/random/boot_id').read_text().strip() == boot, 'Boot changed before closure')
                    receipt['original_birth_absent'] = owner.birth_absent(original, owner.proc_identity(original['pid']))
                    receipt['remaining_owned_group'] = owner._group_members(psutil, original)
                    receipt['final_group_scan_resolved'] = True
                except BaseException as error:
                    receipt.update(cleanup_error_type=type(error).__name__, cleanup_error=str(error))
            else:
                child.kill()
                try:
                    receipt.update(returncode=child.wait(timeout=1), reaped=True)
                except subprocess.TimeoutExpired:
                    pass
                receipt['unresolved_child_identity'] = True
        if stdout is not None: stdout.close()
        if stderr is not None: stderr.close()
        terminal = (receipt['returncode'] == 0 and receipt['reaped'] is True
                    and receipt['original_birth_absent'] is True and receipt['final_group_scan_resolved'] is True
                    and receipt['remaining_owned_group'] == [] and not receipt['violations']
                    and not termination_seen and 'error' not in receipt and 'cleanup_error' not in receipt)
        if terminal:
            try:
                report_path = Path(worker_spec['output_directory'])/'REPORT.json'
                need(report_path.stat().st_size <= 2*1024**2, 'Bounded child report required')
                body = json.loads(report_path.read_bytes())
                need(body.get('schema') == 'SAMRefiner-real-one-image-trial-report-v1'
                     and body.get('run_id') == spec['run_id'] and body.get('operation') == OPERATION
                     and body.get('boot_id') == boot and body.get('pgid') == original['pid']
                     and type(body.get('pid')) is int and type(body.get('start_ticks')) is int
                     and body['start_ticks'] >= original['start_ticks']
                     and body.get('status') == 'ONE_IMAGE_AUTHOR_REFINER_RETURNED_PENDING_ROOT_REVIEW'
                     and body.get('worker_exit_code') == 0 and body.get('inference_completed') is True
                     and body.get('author_refiner_returned') is True
                     and body.get('CUDA_synchronized_after_refiner') is True
                     and body.get('CUDA_allocated_after_cleanup') == body.get('CUDA_reserved_after_cleanup') == 0
                     and body.get('CUDA_peak_allocated_bytes', 16*1024**3+1) <= 16*1024**3,
                     'Real completed author return, CUDA synchronization and cleanup required')
                need(owner.birth_absent(body, owner.proc_identity(body['pid'])), 'Reported worker birth must be absent')
                receipt['reported_worker_birth_absent'] = True
                receipt['worker_report_ref'] = reference(report_path)
                for key in ('output_masks_ref', 'output_identity_ref'):
                    ref = body[key]
                    need(Path(ref['path']).parent == report_path.parent and reference(ref['path']) == ref,
                         'Actual child output bytes must close inside its own output directory')
                receipt['status'] = 'OWNED_ONE_IMAGE_CHILD_COMPLETE_PENDING_ROOT_REVIEW'
            except BaseException as error:
                receipt.update(status='FAILED_CHILD_REPORT_OR_OUTPUT_CLOSURE', closure_error=str(error))
                terminal = False
        else:
            receipt['status'] = 'FAILED_PRESERVED_NO_RETRY'
        receipt.update(controller_returning=True, controller_exit_observation_required=True,
                       termination_signals=termination_seen,
                       all_owned_processes_absent=(receipt['original_birth_absent'] is True
                           and receipt['final_group_scan_resolved'] is True and receipt['remaining_owned_group'] == []))
        save()
        for value, handler in old_handlers.items():
            signal.signal(value, handler)
    return 0 if terminal else 1


if __name__ == '__main__':
    raise SystemExit(main())

