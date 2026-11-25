"""Observe one real worker exit and bind its output footer and result bytes."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
from subprocess import CREATE_NO_WINDOW, Popen


def write(path,value):
    with path.open("x",encoding="utf-8",newline="\n") as stream:
        json.dump(value,stream,sort_keys=True,allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument("--phase",required=True,choices=("prepare-edits","verify-edits","score","render"))
    p.add_argument("--worker",type=Path,required=True)
    p.add_argument("--result",type=Path,required=True)
    p.add_argument("--observer",type=Path,required=True)
    p.add_argument("command",nargs=argparse.REMAINDER)
    a=p.parse_args()
    cmd=a.command[1:] if a.command[:1] == ["--"] else a.command
    worker=a.worker.resolve(strict=True)
    worker_sha=sha256(worker.read_bytes()).hexdigest()
    if a.observer.exists() or a.result.exists() or not any(Path(v).resolve() == worker for v in cmd[1:]):
        raise ValueError("Fresh actual process destination with one identifiable worker")
    a.observer.mkdir()
    started=now()
    with (a.observer/"STDOUT.log").open("xb") as stdout,(a.observer/"STDERR.log").open("xb") as stderr:
        child=Popen(cmd,stdin=None,stdout=stdout,stderr=stderr,creationflags=CREATE_NO_WINDOW)
        meta=dict(phase=a.phase,observer_pid=os.getpid(),worker_pid=child.pid,started_utc=started,worker_sha256=worker_sha)
        write(a.observer/"START.json",meta)
        actual_exit=child.wait()
    result_sha=None
    footer_verified=False
    if actual_exit == 0 and a.result.is_file():
        result=a.result.read_bytes()
        result_sha=sha256(result).hexdigest()
        lines=(a.observer/"STDOUT.log").read_text(encoding="utf-8").splitlines()
        if lines:
            try:
                footer=json.loads(lines[-1])
                result_doc=json.loads(result)
            except (ValueError,TypeError):
                pass
            else:
                footer_verified=(footer.get("result_sha256") == result_sha
                                 and footer.get("status") == result_doc.get("status"))
    meta.update(ended_utc=now(),actual_exit=actual_exit,result_sha256=result_sha,
                output_footer_verified=footer_verified)
    write(a.observer/"TERMINAL.json",meta)
    print(json.dumps(meta),flush=True)
    return actual_exit if actual_exit != 0 else (0 if footer_verified else 2)


if __name__ == "__main__":
    raise SystemExit(main())
