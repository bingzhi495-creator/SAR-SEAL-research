"""Retain the original OS waits for sequential, job-contained float FIT CPU arms."""
import argparse
import ctypes
from ctypes import wintypes
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import time

import psutil

WORKER = Path(__file__).with_name("cpu_short.py")
GIB = 1024**3


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    return sha256(path.read_bytes()).hexdigest()


def save(path, data):
    with path.open("x", encoding="ascii", newline="\n") as stream:
        json.dump(data, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def event(status, **kwargs):
    print(json.dumps(dict(status=status, **kwargs), sort_keys=True), flush=True)


# JobObject functions extracted unchanged from the executed containment helper.
class BasicLimits(ctypes.Structure):
    _fields_ = [
        ("process_time", ctypes.c_longlong), ("job_time", ctypes.c_longlong),
        ("flags", wintypes.DWORD), ("min_working_set", ctypes.c_size_t),
        ("max_working_set", ctypes.c_size_t), ("active_limit", wintypes.DWORD),
        ("affinity", ctypes.c_size_t), ("priority", wintypes.DWORD), ("scheduling", wintypes.DWORD),
    ]

class ExtendedLimits(ctypes.Structure):
    _fields_ = [
        ("basic", BasicLimits), ("io_counters", ctypes.c_ulonglong * 6),
        ("process_memory", ctypes.c_size_t), ("job_memory", ctypes.c_size_t),
        ("peak_process_memory", ctypes.c_size_t), ("peak_job_memory", ctypes.c_size_t),
    ]

class Accounting(ctypes.Structure):
    _fields_ = [
        ("user_time", ctypes.c_longlong), ("kernel_time", ctypes.c_longlong),
        ("period_user_time", ctypes.c_longlong), ("period_kernel_time", ctypes.c_longlong),
        ("faults", wintypes.DWORD), ("total", wintypes.DWORD),
        ("active", wintypes.DWORD), ("terminated", wintypes.DWORD),
    ]

class WindowsJob:
    def __init__(self):
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        handle, dword, ptr, ok = wintypes.HANDLE, wintypes.DWORD, ctypes.c_void_p, wintypes.BOOL
        for name, result, args in (
            ("CreateJobObjectW", handle, (ptr, wintypes.LPCWSTR)),
            ("SetInformationJobObject", ok, (handle, ctypes.c_int, ptr, dword)),
            ("QueryInformationJobObject", ok, (handle, ctypes.c_int, ptr, dword, ptr)),
            ("OpenProcess", handle, (dword, ok, dword)),
            ("AssignProcessToJobObject", ok, (handle, handle)),
            ("TerminateJobObject", ok, (handle, wintypes.UINT)),
            ("CloseHandle", ok, (handle,)),
        ):
            call = getattr(self.kernel, name)
            call.restype, call.argtypes = result, args
        self.handle = self.checked(self.kernel.CreateJobObjectW(None, None))
        limits = ExtendedLimits()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        try:
            self.checked(self.kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)))
        except OSError:
            self.kernel.CloseHandle(self.handle)
            raise

    @staticmethod
    def checked(value):
        if not value:
            raise ctypes.WinError(ctypes.get_last_error())
        return value

    def assign(self, pid):
        process = self.checked(self.kernel.OpenProcess(0x0001 | 0x0100, False, pid))
        try:
            self.checked(self.kernel.AssignProcessToJobObject(self.handle, process))
        finally:
            self.kernel.CloseHandle(process)

    def active(self):
        info = Accounting()
        self.checked(self.kernel.QueryInformationJobObject(self.handle, 1, ctypes.byref(info), ctypes.sizeof(info), None))
        return info.active

    def stop(self):
        self.checked(self.kernel.TerminateJobObject(self.handle, 70))

    def close(self):
        self.checked(self.kernel.CloseHandle(self.handle))


def run(mode, args, worker_sha):
    arm = args.output_dir / mode
    arm.mkdir()
    stdout, stderr = arm / "ORIGINAL.stdout", arm / "ORIGINAL.stderr"
    now = time.monotonic()
    host = psutil.virtual_memory().available
    require(host >= 18*GIB, "Initial CPU job requires 18 GiB host headroom")
    job = WindowsJob()
    pid = birth = stop = code = None
    peak = 0
    attached = False
    active = None
    try:
        with stdout.open("xb") as out, stderr.open("xb") as err:
            command = [str(args.worker_python), "-I", "-B", "-X", "utf8", str(WORKER),
                       "--mode", mode, "--arm-dir", arm.as_posix()]
            for name in ("code_root", "fit_layout", "layout_receipt", "dependency_overlay", "coco"):
                command.extend(("--" + name.replace("_", "-"), str(getattr(args, name))))
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
            child = subprocess.Popen(command, cwd=arm, stdout=out, stderr=err, env=env,
                creationflags=subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS | 4)
            pid = child.pid
            try:
                proc = psutil.Process(pid)
                birth = proc.create_time()
                job.assign(pid)
                attached = True
                require(job.active() == 1, "Single suspended process in JobObject required")
                proc.resume()
                event("FIT_FLOAT_CPU_ORIGINAL_LAUNCHED", mode=mode, pid=pid, birth=birth, worker_sha256=worker_sha)
                while child.poll() is None:
                    try:
                        peak = max(peak, proc.memory_info().rss)
                    except psutil.NoSuchProcess:
                        break
                    host = min(host, psutil.virtual_memory().available)
                    if host < 16*GIB or peak > 10*GIB or time.monotonic()-now > 900:
                        stop = "cpu_resource_guard"
                        job.stop()
                        break
                    time.sleep(0.25)
            except BaseException as error:
                stop = type(error).__name__
                if child.poll() is None:
                    job.stop() if attached else child.kill()
            finally:
                code = child.wait(timeout=20)
                if job.active():
                    job.stop()
                deadline = time.monotonic() + 5
                while job.active() and time.monotonic() < deadline:
                    time.sleep(0.05)
                active = job.active()
    finally:
        job.close()
    result = dict(status="FIT_FLOAT_CPU_ORIGINAL_OS_WAIT", mode=mode, pid=pid, birth=birth, original_os_exit=code,
                  original_os_wait=True, job_attached=attached, process_group_closed=active == 0, stop_reason=stop,
                  peak_rss_bytes=peak, minimum_host_available_bytes=host, seconds=time.monotonic()-now,
                  stdout_sha256=sha(stdout), stderr_sha256=sha(stderr), worker_sha256=worker_sha)
    path = arm / "ORIGINAL_WAIT_PRIVATE.json"
    save(path, result)
    event(result["status"], mode=mode, original_os_exit=code, process_group_closed=active == 0,
          stop_reason=stop, wait_sha256=sha(path))
    if stop or code != 0 or active != 0:
        raise SystemExit(code if code not in (None, 0) else 70)
    report = arm / "RESULT_SAFE.json"
    value = json.loads(report.read_bytes())
    footer = json.loads(stdout.read_bytes().splitlines()[-1])
    require(value["mode"] == mode and value["status"] == footer["status"] == "SSDD_FLOAT_RFDETR_CPU_SHORT_ARM_COMPLETE"
            and value["visits"] == footer["visits"] == 32 and value["optimizer_updates"] == value["ema_updates"] == 8
            and footer["report_sha256"] == sha(report) and value["common_ledger_sha256"] == sha(arm/"COMMON_VISITS_PRIVATE.jsonl"),
            "Exact closed worker report, original footer and common ledger required")
    return value, sha(report), sha(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("worker-python", "code-root", "fit-layout", "layout-receipt", "dependency-overlay", "coco", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--worker-sha256", required=True)
    args = parser.parse_args()
    for name in ("worker_python", "code_root", "fit_layout", "layout_receipt", "dependency_overlay", "coco"):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    args.output_dir = args.output_dir.resolve()
    worker_sha = args.worker_sha256
    require(sha(WORKER) == worker_sha, "Exact portable worker SHA-256 required")
    require(args.output_dir != args.fit_layout and args.fit_layout not in args.output_dir.parents
            and args.output_dir not in args.fit_layout.parents, "Fresh isolated output directory required")
    args.output_dir.mkdir()
    a, a_sha, aw_sha = run("native", args, worker_sha)
    b, b_sha, bw_sha = run("center", args, worker_sha)
    shared = ("visits", "optimizer_updates", "ema_updates", "q100_initial_sha256", "classifier_initial_sha256",
              "common_ledger_sha256", "common_targets_total", "common_empty_visits", "changed_mask_visits",
              "prefetched_unconsumed", "unconsumed_prefetch_sha256")
    require(all(a[key] == b[key] for key in shared) and a["changed_mask_visits"] > 0
            and a["selected_mask_stream_sha256"] != b["selected_mask_stream_sha256"]
            and a["classifier_ending_sha256"] != b["classifier_ending_sha256"],
            "Exact shared CPU input ledger with a real mask and fitted parameter difference required")
    safe = dict(status="SSDD_FLOAT_RFDETR_CPU_SHORT_PAIR_COMPLETE", device="CPU", precision="32-true",
                native_arm_sha256=a_sha, center_arm_sha256=b_sha, native_wait_sha256=aw_sha,
                center_wait_sha256=bw_sha, FIT=True, DEV=False, TEST=False, AP=False, cuda_initialized=False,
                **{k: a[k] for k in shared})
    result = args.output_dir / "PAIR_SAFE.json"
    save(result, safe)
    event(safe["status"], report_sha256=sha(result))


if __name__ == "__main__":
    main()
