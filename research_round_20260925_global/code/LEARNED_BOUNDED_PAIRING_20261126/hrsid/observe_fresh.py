import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys

def digest(path):
    with path.open("rb") as stream:
        return sha256(stream.read()).hexdigest()

def stamp():
    return datetime.now(timezone.utc).isoformat()

def record(path, data):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(data, sort_keys=True) + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("args", nargs=argparse.REMAINDER)
    options = parser.parse_args()
    options.log.mkdir()
    worker = options.worker.resolve(strict=True)
    started = stamp()
    details = dict(phase=options.phase, started_utc=started, observer_pid=os.getpid(), worker_sha256=digest(worker))
    stdout = options.log / "STDOUT.log"
    with stdout.open("xb") as output, (options.log / "STDERR.log").open("xb") as error:
        with subprocess.Popen([sys.executable, "-I", "-B", "-X", "utf8", str(worker), *options.args[1:]],
                              stdout=output, stderr=error, creationflags=subprocess.CREATE_NO_WINDOW) as process:
            details["worker_pid"] = process.pid
            record(options.log / "START.json", details)
            result = process.wait()
    verification = False
    result_hash = None
    if result == 0 and options.result.is_file():
        result_hash = digest(options.result)
        try:
            verification = json.loads(stdout.read_bytes().splitlines()[-1]).get("result_sha256") == result_hash
        except (UnicodeError, ValueError, IndexError):
            pass
    record(options.log / "TERMINAL.json", dict(details, ended_utc=stamp(), actual_exit=result,
                                              result_sha256=result_hash, output_footer_verified=verification))
    return result or int(not verification)

if __name__ == "__main__":
    raise SystemExit(main())
