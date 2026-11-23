"""Observe a process and verify its final complete result."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def save(path, document):
    with path.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(document, f, sort_keys=True, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--phase", choices=("request", "inference", "edits", "score"), required=True)
    for name in ("worker", "result", "observer"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    a = parser.parse_args()
    command = a.command[1:] if a.command[:1] == ["--"] else a.command
    worker = a.worker.resolve(strict=True)
    if a.observer.exists() or a.result.exists() or not any(Path(arg).resolve() == worker for arg in command[1:]):
        raise ValueError("Fresh outputs and exact worker identity required")
    worker_digest = sha256(worker.read_bytes()).hexdigest()
    a.observer.mkdir()
    started = timestamp()
    outpath = a.observer / "STDOUT.log"
    with outpath.open("xb") as out, (a.observer / "STDERR.log").open("xb") as err:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                 creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        meta = dict(phase=a.phase, started_utc=started, observer_pid=os.getpid(), worker_pid=child.pid,
                    worker_sha256=worker_digest)
        save(a.observer / "START.json", meta)
        exit_code = child.wait()
    result_digest = None
    footer_verified = False
    if exit_code == 0 and a.result.is_file():
        raw = a.result.read_bytes()
        result_digest = sha256(raw).hexdigest()
        try:
            last = outpath.read_text(encoding="utf-8").splitlines()[-1]
            answer, doc = json.loads(last), json.loads(raw)
            footer_verified = (answer["status"] == doc["status"] and answer["result_sha256"] == result_digest
                               and doc["worker_sha256"] == worker_digest and doc["phase"] == a.phase)
        except (KeyError, ValueError, IndexError):
            pass
    meta.update(actual_exit=exit_code, ended_utc=timestamp(), output_footer_verified=footer_verified,
                result_sha256=result_digest)
    save(a.observer / "TERMINAL.json", meta)
    print(json.dumps(meta), flush=True)
    return exit_code if exit_code else (0 if footer_verified else 2)


if __name__ == "__main__":
    raise SystemExit(main())
