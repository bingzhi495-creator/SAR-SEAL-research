"""Capture original bounded CPU process exit separately from the scientific receipt."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter


def write(path, raw):
    with path.open("xb") as output:
        output.write(raw)
        output.flush()
        os.fsync(output.fileno())
    return sha256(raw).hexdigest()


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--worker", type=Path, required=True)
parser.add_argument("--worker-sha", required=True)
parser.add_argument("--terminal", type=Path, required=True)
parser.add_argument("arguments", nargs=argparse.REMAINDER)
args = parser.parse_args()
worker_bytes = args.worker.read_bytes()
if sha256(worker_bytes).hexdigest() != args.worker_sha or args.terminal.exists():
    raise SystemExit("Exact sealed worker and fresh terminal path required")
command = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
stage = command[0]
env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
           MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
started = perf_counter()
process = subprocess.Popen([sys.executable, "-I", "-B", "-X", "utf8", str(args.worker), *command],
                           env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
timed_out = False
try:
    stdout, stderr = process.communicate(timeout=120)
except subprocess.TimeoutExpired:
    timed_out = True
    process.kill()
    stdout, stderr = process.communicate()
write(args.terminal.with_suffix(".stdout"), stdout)
write(args.terminal.with_suffix(".stderr"), stderr)
result_digest = None
if not process.returncode and not timed_out:
    result = json.loads(stdout.decode().strip().splitlines()[-1])
    if result["stage"] != stage or len(result["output_sha256"]) != 64:
        raise SystemExit("Natural worker summary mismatch")
    result_digest = result["output_sha256"]
terminal = dict(stage=stage, original_os_exit=process.returncode, timed_out=timed_out, child_closed=True,
                worker_source_sha256=args.worker_sha, output_sha256=result_digest,
                stdout_sha256=sha256(stdout).hexdigest(), stderr_sha256=sha256(stderr).hexdigest(),
                elapsed_seconds=perf_counter()-started)
digest = write(args.terminal, (json.dumps(terminal, sort_keys=True) + "\n").encode())
print(json.dumps(dict(terminal_sha256=digest, **terminal), sort_keys=True), flush=True)
if process.returncode or timed_out:
    sys.stderr.buffer.write(stderr)
    raise SystemExit(1)