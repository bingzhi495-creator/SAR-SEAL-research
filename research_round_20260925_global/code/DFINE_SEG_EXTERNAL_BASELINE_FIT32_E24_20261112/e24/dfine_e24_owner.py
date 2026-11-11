"""Own the locked D-FINE-seg-S E24 comparator with the original Windows JobObject."""
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from time import monotonic
from types import ModuleType, SimpleNamespace

import psutil


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "HRSID_DFINE_E24_TRAIN_ACTUAL_20261112_01"
WORKER = HERE / "dfine_e24_worker.py"
WORKER_SHA = "a3d3f9ae300f209ca4251b9e564ee262c3637429c208ccbd4e390d6b2b628d1d"
PRELOCK = HERE / "PROTOCOL.md"
PRELOCK_SHA = "684ab3a18dddc3a9887819050dc545c28736bbb3ccc664e83be9c67ef2b1d0a3"
JOB_SOURCE = ROOT / "P206_ROOT_HRSID_MASK_PHASE_DEV705_COORDINATOR_PRIVATE_20261109.py"
JOB_SHA = "55cb9a4a18a6b58fa5eac28a259422c058831523ae10ca0448818ee856ef0b39"
PYTHON = ROOT / "DFINE_SEG_EXTERNAL_BASELINE_INTAKE_20261112_01/CUDA_RUNTIME_QUAL_PRIVATE_20261112_01/VENV_PRIVATE/Scripts/python.exe"
CONTROL = OUT / "CONTROL"
KEY = "DFINE_SEG_S_HRSID_E24_WINDOWS_CUDA_20261112_01"
WALL_SECONDS = 30*60*60
STATE_LIMIT = 16 * 1024**2
SCOPE = dict(FIT=True, DEV=False, TEST=False, AP=False)


def need(ok, why):
    if not ok:
        raise ValueError(why)


def emit(record):
    print(json.dumps(record, sort_keys=True, allow_nan=False), flush=True)


def pin(path, digest):
    need(path.is_file() and path.resolve(strict=True) == path, "Exact canonical owned input")
    raw = path.read_bytes()
    need(sha256(raw).hexdigest() == digest, "Exact owned source and prospective request")
    return raw


def main():
    pin(PRELOCK, PRELOCK_SHA)
    lock = dict(minimum_free_cuda_bytes=12*1024**3,minimum_host_available_bytes=16*1024**3,
                max_worker_tree_rss_bytes=10*1024**3,wall_seconds=WALL_SECONDS)
    pin(WORKER, WORKER_SHA)
    job = ModuleType("_exact_reused_original_windows_job")
    job.__file__ = str(JOB_SOURCE)
    source = pin(JOB_SOURCE, JOB_SHA).decode("utf8")
    deadline = "time.monotonic() - start > 7200:"
    need(source.count(deadline)==1,"Exact inherited Windows hard deadline seam")
    source = source.replace(deadline,f"time.monotonic() - start > {WALL_SECONDS+120}:",1)
    exec(compile(source, str(JOB_SOURCE), "exec"), job.__dict__)
    raw = job.P189["stable"](job.STATE, STATE_LIMIT)
    state = json.loads(raw)
    need(KEY not in state and not OUT.exists() and PYTHON.is_file()
         and state["GPU_owner_or_worker_active"] is state["gpu_training_active"] is False
         and all(row.get("active_attempt") is None for row in state.values() if type(row) is dict),
         "Fresh exclusive original GPU admission")
    def observed_gpu_free():
        return int(subprocess.check_output(
            ["nvidia-smi","-i","0","--query-gpu=memory.free","--format=csv,noheader,nounits"],
            timeout=12, creationflags=subprocess.CREATE_NO_WINDOW, text=True).strip())
    gpu_free_mib = observed_gpu_free()
    host_initial = psutil.virtual_memory().available
    need(gpu_free_mib * 1024**2 >= lock["minimum_free_cuda_bytes"]
         and host_initial >= lock["minimum_host_available_bytes"], "Current original GPU and host admission")
    CONTROL.mkdir(parents=True)
    scratch = CONTROL/"scratch"
    scratch.mkdir()
    coordinator = job.birth(os.getpid())
    state[KEY] = dict(status="DFINE_E24_GPU_ORIGINAL_RESERVED", active_attempt=KEY, resource_closed=False,
                      owner_birth=coordinator, prelock_sha256=PRELOCK_SHA, worker_source_sha256=WORKER_SHA, **SCOPE)
    state.update(GPU_owner_or_worker_active=True, gpu_training_active=True)
    job.cas(raw,state,"DFINE_HRSID_E24_WINDOWS_CUDA_RESERVE")
    reserved = job.P189["stable"](job.STATE,STATE_LIMIT)
    need(json.loads(reserved) == state,"Exact exclusive Windows GPU reservation")
    emit(dict(status="DFINE_E24_GPU_ORIGINAL_RESERVED", owner_pid=coordinator["pid"],
              prelock_sha256=PRELOCK_SHA, free_gpu_mib=gpu_free_mib, **SCOPE))

    venv_config = dict(line.split(" = ",1) for line in (PYTHON.parent.parent/"pyvenv.cfg").read_text("utf8").splitlines() if " = " in line)
    win = Path(os.environ["SystemRoot"])
    env = dict(os.environ, PATH=os.pathsep.join(map(str,(win/"System32",win,Path(venv_config["home"]),PYTHON.parent))),
               CUDA_VISIBLE_DEVICES="0", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               NO_ALBUMENTATIONS_UPDATE="1", WANDB_MODE="disabled", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", TMP=str(scratch), TEMP=str(scratch), TORCH_HOME=str(scratch/"torch"))
    child, reason, peak, floor = None, None, 0, host_initial
    began = monotonic()
    last_gpu_poll, gpu_free_floor, gpu_watches = began, gpu_free_mib, 1

    def popen(*args, **kwargs):
        nonlocal child
        child = subprocess.Popen(*args, env=env, **kwargs)
        return child

    def permitted():
        nonlocal reason, peak, floor, last_gpu_poll, gpu_free_floor, gpu_watches
        if job.STATE.read_bytes() != reserved:
            reason = "SHARED_RESERVATION_CHANGED"
        elif monotonic() - began >= lock["wall_seconds"]:
            reason = "ORIGINAL_30_HOUR_TECHNICAL_LIMIT"
        else:
            now = monotonic()
            if now - last_gpu_poll >= 15:
                gpu_free_floor = min(gpu_free_floor,observed_gpu_free())
                last_gpu_poll = now
                gpu_watches += 1
                if gpu_free_floor*1024**2 < lock["minimum_free_cuda_bytes"]:
                    reason = "GPU_FREE_MEMORY_BELOW_12_GIB"
            floor = min(floor, psutil.virtual_memory().available)
            if floor < lock["minimum_host_available_bytes"]:
                reason = "HOST_MEMORY_BELOW_16_GIB"
            if child is not None and child.poll() is None:
                try:
                    parent = psutil.Process(child.pid)
                    peak = max(peak, sum(process.memory_info().rss for process in [parent, *parent.children(recursive=True)]
                                         if process.is_running()))
                except psutil.NoSuchProcess:
                    pass
                if peak > lock["max_worker_tree_rss_bytes"]:
                    reason = "WORKER_TREE_RSS_ABOVE_10_GIB"
        return reason is None

    job.CONTROL, job.OWNER_PYTHON = CONTROL, PYTHON
    job.subprocess = SimpleNamespace(Popen=popen, CREATE_NO_WINDOW=subprocess.CREATE_NO_WINDOW,
                                     BELOW_NORMAL_PRIORITY_CLASS=subprocess.BELOW_NORMAL_PRIORITY_CLASS)
    terminal = job.wait_owner(WORKER, [], permitted)
    closed = terminal["original_os_wait"] is True and terminal["job_active_after_wait"] == 0 and not job.live(terminal["owner"])
    success = (closed and terminal["original_os_exit"] == 0 and terminal["job_assigned"] is True
               and terminal["job_active_before_drain"] == 0 and terminal["stop_reason"] is reason is None)
    guard = dict(reason=reason, peak_worker_tree_rss_bytes=peak, minimum_host_bytes=floor,
                 original_gpu_free_mib=gpu_free_mib, minimum_gpu_free_mib=gpu_free_floor, gpu_watch_count=gpu_watches, wall_limit_seconds=lock["wall_seconds"])
    job.write_json(CONTROL/"RESOURCE_GUARD_PRIVATE.json",guard)
    summary = dict(status="DFINE_E24_GPU_ORIGINAL_CLOSED_PENDING_ACCEPTANCE" if success else "DFINE_E24_GPU_ORIGINAL_FAILED",
                   terminal_ref=job.ref(CONTROL/"OWNER_TERMINAL_PRIVATE.json"), resource_guard=guard,
                   original_exit=terminal["original_os_exit"], original_wall_seconds=terminal["wall_seconds"],
                   worker_pid=terminal["owner"]["pid"] if terminal["owner"] else None,
                   resource_closed=closed, failure=terminal["stop_reason"] or reason,
                   worker_source_sha256=WORKER_SHA, prelock_sha256=PRELOCK_SHA, **SCOPE)
    if success:
        summary["result_ref"] = job.ref(OUT/"ACTUAL/RESULT_SAFE.json")
    job.write_json(CONTROL/"ORIGINAL_OWNER_SUMMARY_PRIVATE.json",summary)
    if closed:
        state[KEY].update(status=summary["status"],active_attempt=None,resource_closed=True,
                          failure=summary["failure"], terminal_ref=summary["terminal_ref"],
                          summary_ref=job.ref(CONTROL/"ORIGINAL_OWNER_SUMMARY_PRIVATE.json"))
        state.update(GPU_owner_or_worker_active=False,gpu_training_active=False)
        job.cas(reserved,state,"DFINE_HRSID_E24_WINDOWS_CUDA_CLOSE")
    emit(dict(status=summary["status"], resource_closed=closed, original_exit=summary["original_exit"],
              original_wall_seconds=summary["original_wall_seconds"], terminal_sha256=summary["terminal_ref"]["sha256"],
              result_sha256=summary.get("result_ref",{}).get("sha256"), failure=summary["failure"], **SCOPE))
    return 0 if success else 5


if __name__ == "__main__":
    raise SystemExit(main())
