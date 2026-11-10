"""Own two serial DGX FIT qualification arms under the existing project GPU leases."""
import argparse
from contextlib import ExitStack
from hashlib import sha256
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil


MODES = ("stepwise_center", "one_pass")
GIB = 1024**3
HOST_FLOOR, RSS_CAP, SECONDS = 40 * GIB, 8 * GIB, 300
LEASE_NAMES = ("P250_HRSID_TO_SSDD_FROZEN_TRANSFER_GPU_OWNER.lock",
               "SSDD_PHASE_DGX_GPU_OWNER_20261110.lock")
Q164 = "63ff93afdd977ba79b5b5a46ba8b4839ef6c40eb0722b29e4667d4932404fbc2"
INDEX = "e95cf26cc60a2c6fb651e907541380a256f2df7b1e153244a6c0f803b0b19f8f"
BOOT = Path("/proc/sys/kernel/random/boot_id")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def validate_document(raw):
    def unique(pairs):
        result = dict(pairs)
        require(len(result) == len(pairs), "Duplicate request field")
        return result
    result = json.loads(raw, object_pairs_hook=unique)
    require(type(result) is dict, "Exact request object required")
    return result


def birth(proc):
    return dict(pid=proc.pid, created=proc.create_time(), boot_id=BOOT.read_text().strip())


def group_live(pid):
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return False
    return True


def kill_group(pid):
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def write_json(path, value):
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return dict(path=path.as_posix(), bytes=len(raw), sha256=sha256(raw).hexdigest())


def closed_ref(path, limit, *, empty=False):
    require(path.is_absolute() and path.resolve(strict=True) == path and path.is_file(),
            "Canonical closed regular file required")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    require((empty or len(raw) > 0) and len(raw) <= limit, "Closed file exceeds its scope")
    return dict(path=path.as_posix(), bytes=len(raw), sha256=sha256(raw).hexdigest()), raw


def make_arm_lock(root, mode, platform, scientific_sources):
    stage = root / mode
    require(mode in MODES, "Exact B/C mode required")
    return dict(status="HRSID_ONE_PASS_COMMON_FIT_SHORT_LOCK_V1",
                training_protocol=platform["training_protocol"], implementation_commit=platform["implementation_commit"],
                stage="qualification", mode=mode, training_seed=5202, initialization_seed=5201,
                initial_expanded_sha256=Q164, fit_index_sha256=INDEX,
                native_output=(stage / "ACTUAL").as_posix(), audit_output=(stage / "AUDIT").as_posix(),
                sources=scientific_sources, qualification_pair_ref=None)


def supervise_child(stage, command, env, host_floor=HOST_FLOOR, rss_cap=RSS_CAP, seconds=SECONDS):
    require(psutil.virtual_memory().available >= host_floor, "DGX host available RAM floor before worker")
    out, err = stage / "WORKER.stdout", stage / "WORKER.stderr"
    start = time.monotonic()
    peak, stop, child_birth = 0, None, None
    with out.open("xb") as stdout, err.open("xb") as stderr:
        child = subprocess.Popen(command, env=env, cwd=stage, stdout=stdout, stderr=stderr, start_new_session=True)
        try:
            proc = psutil.Process(child.pid)
            child_birth = birth(proc)
            write_json(stage / "WORKER_BIRTH_PRIVATE.json", child_birth)
            while child.poll() is None:
                try:
                    processes = [proc, *proc.children(recursive=True)]
                    rss = sum(item.memory_info().rss for item in processes if item.is_running())
                    peak = max(peak, rss)
                except psutil.NoSuchProcess:
                    continue
                if psutil.virtual_memory().available < host_floor:
                    stop = "host available RAM floor"
                elif peak > rss_cap:
                    stop = "process-tree RSS cap"
                elif time.monotonic() - start > seconds:
                    stop = "original worker deadline"
                if stop:
                    break
                time.sleep(0.25)
        except (Exception, KeyboardInterrupt) as error:
            stop = type(error).__name__
        finally:
            if child.poll() is None:
                kill_group(child.pid)
            exit_code = child.wait()
            if group_live(child.pid):
                stop = stop or "descendant remained after original wait"
                kill_group(child.pid)
            for log in (stdout, stderr):
                log.flush()
                os.fsync(log.fileno())
    terminal = dict(original_os_wait=True, original_os_exit=exit_code,
                    worker_closed=not group_live(child.pid), worker=child_birth,
                    resource_stop=stop is not None, stop_reason=stop, peak_rss_bytes=peak,
                    seconds=time.monotonic() - start, stdout_sha256=closed_ref(out, 2 * 1024**2, empty=True)[0]["sha256"],
                    stderr_sha256=closed_ref(err, 2 * 1024**2, empty=True)[0]["sha256"])
    write_json(stage / "TERMINAL_PRIVATE.json", terminal)
    return terminal


def run(request, request_sha):
    import fcntl

    identity, raw = closed_ref(request, 64 * 1024)
    require(identity["sha256"] == request_sha, "Exact prospective owner request SHA-256 required")
    cfg = validate_document(raw)
    require(set(cfg) == {"status", "platform", "worker", "scientific_sources", "lease_directory"}
            and cfg["status"] == "P298_HRSID_DGX_BC_SHORT_OWNER_REQUEST_V1",
            "Exact prospective owner request required")
    platform, worker, sources = cfg["platform"], cfg["worker"], cfg["scientific_sources"]
    root, lease_root = Path(platform["work_root"]), Path(cfg["lease_directory"])
    require(root.is_absolute() and root.parent.resolve(strict=True) == root.parent == lease_root
            and not root.exists() and not root.is_symlink()
            and platform["status"] == "P297_HRSID_DGX_BC_SHORT_PLATFORM_LOCK_V1"
            and platform["python"] == sys.executable and "owner_birth_ref" not in platform
            and len(sources) == 16, "Fresh same-lease private study and exact P297 platform required")
    worker_path = Path(worker["path"])
    require(closed_ref(worker_path, 64 * 1024)[0] == worker and platform["worker_sha256"] == worker["sha256"],
            "Exact P297 worker bytes required")
    require(sys.platform == "linux" and sys.flags.isolated == sys.flags.utf8_mode == 1
            and sys.dont_write_bytecode and psutil.virtual_memory().available >= HOST_FLOOR,
            "Exact isolated Linux owner and host memory floor required")
    os.sched_setaffinity(0, {0, 1})
    with ExitStack() as stack:
        for name in LEASE_NAMES:
            path = lease_root / name
            require(path.resolve(strict=True) == path and path.is_file(), "Existing canonical project GPU lease required")
            lease = stack.enter_context(path.open("r+b"))
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        root.mkdir(mode=0o700)
        owner = birth(psutil.Process())
        owner_ref = write_json(root / "OWNER_BIRTH_PRIVATE.json", owner)
        platform_ref = write_json(root / "PLATFORM_LOCK_PRIVATE.json", dict(platform, owner_birth_ref=owner_ref))
        terminals, failure = {}, None
        try:
            for mode in MODES:
                stage = root / mode
                stage.mkdir(mode=0o700)
                scratch = stage / "SCRATCH"
                scratch.mkdir(mode=0o700)
                lock = write_json(stage / "TRAINING_LOCK_PRIVATE.json", make_arm_lock(root, mode, platform, sources))
                env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                           HF_HOME=str(scratch), TORCH_HOME=str(scratch), TMPDIR=str(scratch),
                           OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
                command = [sys.executable, "-I", "-B", "-X", "utf8", worker["path"],
                           "--platform-lock", platform_ref["path"], "--platform-lock-sha256", platform_ref["sha256"],
                           "--training-lock", lock["path"], "--training-lock-sha256", lock["sha256"], "--mode", mode]
                print(json.dumps(dict(status="P298_DGX_BC_ORIGINAL_ARM_START", mode=mode)), flush=True)
                terminal = supervise_child(stage, command, env)
                terminals[mode] = closed_ref(stage / "TERMINAL_PRIVATE.json", 16384)[0]
                require(terminal["original_os_exit"] == 0 and terminal["worker_closed"]
                        and not terminal["resource_stop"] and terminal["worker"] is not None,
                        f"Original {mode} failed or stopped")
        except (Exception, KeyboardInterrupt) as error:
            failure = type(error).__name__ + ": " + str(error)
        status = ("P298_DGX_BC_ORIGINALS_CLOSED_PENDING_INDEPENDENT_ACCEPTANCE"
                  if failure is None else "P298_DGX_BC_FAILED_OR_STOPPED")
        write_json(root / "ORIGINAL_COORDINATOR_OUTCOME_PRIVATE.json", dict(
            status=status, request_sha256=request_sha, owner=owner, platform_ref=platform_ref, terminals=terminals,
            failure=failure, FIT=True, DEV=False, TEST=False, AP=False))
        print(json.dumps(dict(status=status, failure=failure)), flush=True)
        return 0 if failure is None else 4


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--request-sha256", required=True)
    args = parser.parse_args()
    return run(args.request, args.request_sha256)


if __name__ == "__main__":
    raise SystemExit(main())

