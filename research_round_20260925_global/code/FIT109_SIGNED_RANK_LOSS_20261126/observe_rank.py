"""Observe the actual child and validate a result footer only after natural exit."""
from argparse import ArgumentParser, REMAINDER
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def main():
    p=ArgumentParser()
    for t in ("observed","result","worker"):
        p.add_argument("--"+t,type=Path,required=True)
    p.add_argument("--phase",required=True)
    p.add_argument("cmd",nargs=REMAINDER)
    a=p.parse_args()
    cmd=a.cmd[1:] if a.cmd[0]=="--" else a.cmd
    worker_sha=digest(a.worker)
    a.observed.mkdir(exist_ok=False)
    beginning=datetime.now(timezone.utc).isoformat()
    with (a.observed/"WORKER.stdout").open("xb") as out, (a.observed/"WORKER.stderr").open("xb") as err:
        proc=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=out,stderr=err)
        print("observed child PID",proc.pid,a.phase,flush=True)
        code=proc.wait()
    end=datetime.now(timezone.utc).isoformat()
    checked=False
    content_sha=None
    try:
        result=read(a.result)
        footer=json.loads((a.observed/"WORKER.stdout").read_bytes().splitlines()[-1])
        content_sha=digest(a.result)
        checked = (result["status"] == footer["status"] and footer["result_sha256"] == content_sha
                   and result["worker_sha256"] == worker_sha == digest(a.worker))
    except (OSError,ValueError,IndexError,KeyError):
        pass
    terminal=dict(phase=a.phase,actual_exit=code,natural_exit=True,worker_pid=proc.pid,
                  started_utc=beginning,ended_utc=end,output_footer_verified=checked,
                  worker_sha256=worker_sha,result_sha256=content_sha)
    with (a.observed/"TERMINAL.json").open("x",encoding="utf-8",newline="\n") as out:
        out.write(json.dumps(terminal,sort_keys=True,indent=2)+"\n")
    print(json.dumps(terminal),flush=True)
    if code or not checked:
        print((a.observed/"WORKER.stderr").read_bytes().decode("utf-8",errors="replace"),file=sys.stderr)
    sys.exit(code or (0 if checked else 2))


if __name__=="__main__":
    main()

