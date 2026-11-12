"""Own one fixed D-FINE DEV705 export under a drained Windows JobObject."""

from argparse import ArgumentParser
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from time import monotonic, sleep
from types import ModuleType

import psutil


MIB,GIB = 1024**2,1024**3
EVAL_PROTOCOL = "14fbf70e29c71d220280fc5c2a82f47b53030ff555b7515685f963fee9967b61"
WORKER_SHA = "5be622cdec5c889ca8520aab5547f6f15f01a54568a0263423a816dca7f96626"
JOB_SHA = "81bac5b6124bbc1b5bb2e401596c5e671485a30f4a5eea5106a8de6662efe9cc"
CLOSURE_COMPLETE = "HRSID_DFINE_E24_DEV705_EXPORT_OUTER_NATURAL_CLOSE_V1"
CLOSURE_FAILED = "HRSID_DFINE_E24_DEV705_EXPORT_OUTER_FAILED_V1"
CHILD_COMPLETE = "HRSID_DFINE_E24_DEV705_EXPORT_COMPLETE_V1"
ATTESTED = "HRSID_DFINE_E24_DEV705_EXPORT_OS_CLOSED_V1"
BOOT = """
from hashlib import sha256
from pathlib import Path
import sys
expected=sys.argv[1]
sys.argv=sys.argv[2:]
path=Path(sys.argv[0])
raw=path.read_bytes()
if sha256(raw).hexdigest()!=expected:
    raise SystemExit(73)
exec(compile(raw,str(path),"exec"),{"__name__":"__main__","__file__":str(path)})
"""


def need(ok,reason):
    if not ok:
        raise ValueError(reason)


def is_sha(value):
    return isinstance(value,str) and re.fullmatch("[0-9a-f]{64}",value) is not None


def exact(path,digest=None,limit=MIB,allow_empty=False):
    need(path.is_absolute() and path.resolve(strict=True)==path and path.is_file(),
         "Expected canonical regular file: "+path.name)
    size=path.stat().st_size
    need((0 if allow_empty else 1)<=size<=limit,"Expected bounded exact file size: "+path.name)
    raw=path.read_bytes()
    need(len(raw)==size and (digest is None or is_sha(digest) and sha256(raw).hexdigest()==digest),
         "Exact pinned closed bytes required: "+path.name)
    return raw


def write(path,value):
    raw=(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return sha256(raw).hexdigest()


def emit(value):
    print(json.dumps(value,sort_keys=True,allow_nan=False),flush=True)


def load_job(path):
    obj=ModuleType("_dfine_inherited_windows_job")
    obj.__file__=str(path)
    exec(compile(exact(path,JOB_SHA),str(path),"exec"),obj.__dict__)
    return obj.WindowsJob


def clean_job(term):
    return (term["original_os_wait"] is True and type(term["original_os_exit"]) is int
            and term["original_os_exit"]==0 and term["job_assigned"] is True
            and term["job_active_before_drain"]==term["job_active_after_wait"]==0
            and term["stop_reason"] is None)


def run_job(job_factory,command,control,authorize,*,timeout,env=None,rss_limit=None):
    need(control.is_absolute() and control.parent.resolve(strict=True)==control.parent
         and not control.exists(),"Fresh isolated job control directory")
    control.mkdir(exist_ok=False)
    stdout_path,stderr_path=control/"CHILD.stdout",control/"CHILD.stderr"
    child=None
    assigned=False
    active=remaining=None
    exit_code=stop=born=None
    peak=0
    started=monotonic()
    job=job_factory()
    try:
        with stdout_path.open("xb") as stdout,stderr_path.open("xb") as stderr:
            try:
                authorize()
                child=subprocess.Popen(command,cwd=control,stdout=stdout,stderr=stderr,env=env,
                    creationflags=4|subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS)
                process=psutil.Process(child.pid)
                born=dict(pid=child.pid,creation_time=process.create_time())
                job.assign(child.pid)
                assigned=True
                need(job.active()==1,"Single suspended JobObject child required")
                authorize()
                process.resume()
                while child.poll() is None:
                    need(monotonic()-started<timeout,"Original export wall limit")
                    authorize()
                    if rss_limit is not None:
                        try:
                            current=sum(p.memory_info().rss for p in [process,*process.children(recursive=True)])
                            peak=max(peak,current)
                        except psutil.NoSuchProcess:
                            pass
                        need(peak<=rss_limit,"Original export process-tree RSS limit")
                    sleep(.25)
            except (Exception,KeyboardInterrupt) as error:
                stop=type(error).__name__+": "+str(error)
            finally:
                if child is not None:
                    if child.poll() is None:
                        if assigned:
                            job.stop()
                        else:
                            child.kill()
                    exit_code=child.wait()
                active=job.active()
                if active:
                    stop=stop or "Original child left a process in its JobObject"
                    job.stop()
                limit=monotonic()+30
                while job.active() and monotonic()<limit:
                    sleep(.05)
                remaining=job.active()
                for stream in (stdout,stderr):
                    stream.flush()
                    os.fsync(stream.fileno())
    finally:
        job.close()
    term=dict(original_os_wait=child is not None,original_os_exit=exit_code,job_assigned=assigned,
              job_active_before_drain=active,job_active_after_wait=remaining,stop_reason=stop,
              peak_worker_tree_rss_bytes=peak,elapsed_seconds=monotonic()-started,child_birth=born,
              stdout_sha256=sha256(exact(stdout_path,limit=8*MIB,allow_empty=True)).hexdigest(),
              stderr_sha256=sha256(exact(stderr_path,limit=8*MIB,allow_empty=True)).hexdigest())
    write(control/"CHILD_TERMINAL_PRIVATE.json",term)
    return term


def verify_result(result):
    need(result["status"]==CHILD_COMPLETE and result["protocol_sha256"]==EVAL_PROTOCOL
         and result["export_source_sha256"]==WORKER_SHA and result["images"]==705
         and result["DEV"] is True and result["FIT"] is result["TEST"] is result["GT_opened"] is result["AP"] is False,
         "Exact closed child export result and no-label scope")


def parse_tail(output):
    return json.loads(next((line for line in reversed(output.splitlines()) if line.strip()),""))


def validate_attestation(closure,terminal,wait,result,digests):
    verify_result(result)
    need(all(is_sha(digests[k]) for k in ("closure","terminal","result","tool"))
         and clean_job(terminal) and closure["status"]==CLOSURE_COMPLETE and closure["resource_closed"] is True
         and closure["protocol_sha256"]==EVAL_PROTOCOL and closure["worker_source_sha256"]==WORKER_SHA
         and closure["child_terminal_sha256"]==digests["terminal"] and closure["result_sha256"]==digests["result"],
         "Exact naturally closed child JobObject and result required")
    launch,tool=wait["launch"],wait["terminal"]
    need(type(launch.get("session_id")) is int and launch["session_id"]>0
         and type(tool.get("exit_code")) is int and tool["exit_code"]==0 and type(tool.get("output")) is str,
         "Exact original outer tool wait exit0 required")
    try:
        tail=parse_tail(tool["output"])
    except (TypeError,ValueError):
        raise ValueError("Exact original outer tool wait status required") from None
    need(tail==dict(status=CLOSURE_COMPLETE,closure_sha256=digests["closure"],result_sha256=digests["result"]),
         "Exact original outer tool wait must bind child closure and result")
    return dict(status=ATTESTED,original_os_wait=True,original_os_exit=0,job_drained=True,resource_stop=False,
                export_result_sha256=digests["result"],protocol_sha256=EVAL_PROTOCOL,
                original_tool_wait_sha256=digests["tool"],outer_closure_sha256=digests["closure"],
                child_terminal_sha256=digests["terminal"],DEV=True,FIT=False,TEST=False,AP=False,GT_opened=False)



def cli_attest(args):
    closed=args.control
    closure_raw=exact(closed/"OWNER_CLOSURE_PRIVATE.json",args.closure_sha)
    closure=json.loads(closure_raw)
    terminal_raw=exact(closed/"CHILD_TERMINAL_PRIVATE.json",closure["child_terminal_sha256"])
    terminal=json.loads(terminal_raw)
    result_raw=exact(closed/"EXPORT"/"RESULT_SAFE.json",closure["result_sha256"])
    wait_raw=exact(args.tool_wait,args.tool_wait_sha)
    stdout=exact(closed/"CHILD.stdout",terminal["stdout_sha256"],8*MIB).decode("utf-8")
    exact(closed/"CHILD.stderr",terminal["stderr_sha256"],8*MIB,allow_empty=True)
    need(parse_tail(stdout)==dict(status=CHILD_COMPLETE,result_sha256=closure["result_sha256"]),
         "Exact original child stdout result witness")
    digests=dict(closure=sha256(closure_raw).hexdigest(),terminal=sha256(terminal_raw).hexdigest(),
                 result=sha256(result_raw).hexdigest(),tool=sha256(wait_raw).hexdigest())
    receipt=validate_attestation(closure,terminal,json.loads(wait_raw),json.loads(result_raw),digests)
    need(args.output.is_absolute() and args.output.parent.resolve(strict=True)==args.output.parent
         and not args.output.exists(),"Fresh exact terminal attestation")
    result=write(args.output,receipt)
    emit(dict(status=ATTESTED,export_terminal_sha256=result,export_result_sha256=digests["result"]))
    return 0


def reservation_contract(state,key):
    row=state[key]
    need(state["GPU_owner_or_worker_active"] is True and state["gpu_training_active"] is False
         and row["active_attempt"]==key and row["resource_closed"] is False and row["DEV"] is True
         and row["FIT"] is row["TEST"] is row["AP"] is False
         and all(k==key or not isinstance(v,dict) or v.get("active_attempt") is None for k,v in state.items()),
         "Exact exclusive evaluator reservation before any GPU process")


def export_guard(path,expected,key):
    raw=exact(path,expected,16*MIB)
    reservation_contract(json.loads(raw),key)
    checked_at=None
    def authorize():
        nonlocal checked_at
        need(path.read_bytes()==raw,"Original exclusive GPU reservation changed")
        need(psutil.virtual_memory().available>=16*GIB,"Original DEV export host memory floor")
        now=monotonic()
        if checked_at is None or now-checked_at>=15:
            free=int(subprocess.check_output(["nvidia-smi","-i","0","--query-gpu=memory.free","--format=csv,noheader,nounits"],
                     creationflags=subprocess.CREATE_NO_WINDOW,timeout=12,text=True).strip())
            need(free>=12*1024,"Original DEV export GPU free-memory floor")
            checked_at=now
    return authorize


def cli_export(args):
    exact(args.worker,WORKER_SHA)
    exact(args.worker.with_name("PROTOCOL.md"),EVAL_PROTOCOL)
    accepted=json.loads(exact(args.acceptance,args.acceptance_sha))
    need(accepted["status"]=="DFINE_SEG_S_HRSID_E24_CUDA_INDEPENDENT_READBACK_PASSED"
         and accepted["completed_epochs"]==24 and accepted["optimizer_updates"]==accepted["ema_updates"]==17640
         and accepted["original_exit"]==0 and accepted["job_drained"] is True,
         "Independent exact E24 acceptance before original export")
    authorize=export_guard(args.gpu_reservation,args.gpu_reservation_sha,args.reservation_key)
    python=args.worker_python
    need(python.is_absolute() and python.resolve(strict=True)==python and python.is_file(),"Exact evaluator Python path")
    lines=(python.parent.parent/"pyvenv.cfg").read_text("utf-8").splitlines()
    venv=dict(line.split(" = ",1) for line in lines if " = " in line)
    windows=Path(os.environ["SystemRoot"])
    env=dict(os.environ,PATH=os.pathsep.join(map(str,(windows/"System32",windows,Path(venv["home"]),python.parent))),
             CUDA_VISIBLE_DEVICES="0",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",
             HF_HUB_OFFLINE="1",HF_DATASETS_OFFLINE="1",TRANSFORMERS_OFFLINE="1",WANDB_MODE="disabled",
             NO_ALBUMENTATIONS_UPDATE="1")
    command=[str(python),"-I","-B","-X","utf8","-c",BOOT,WORKER_SHA,str(args.worker),"export",
             "--output",str(args.control/"EXPORT"),"--protocol-sha",EVAL_PROTOCOL,
             "--acceptance-sha",args.acceptance_sha]
    for name in ("dev_root","acceptance","author_source","author_tree","archive","ema","codec_wheel"):
        command.extend(("--"+name.replace("_","-"),str(getattr(args,name))))
    terminal=run_job(load_job(args.job_helper),command,args.control,authorize,timeout=1800.,env=env,rss_limit=10*GIB)
    return finish_owner(args.control,terminal)


def finish_owner(control,terminal):
    child_sha=sha256(exact(control/"CHILD_TERMINAL_PRIVATE.json")).hexdigest()
    closure=dict(status=CLOSURE_FAILED,resource_closed=terminal["job_active_after_wait"]==0,
                 protocol_sha256=EVAL_PROTOCOL,worker_source_sha256=WORKER_SHA,child_terminal_sha256=child_sha,
                 result_sha256=None)
    if clean_job(terminal):
        try:
            raw=exact(control/"EXPORT"/"RESULT_SAFE.json")
            verify_result(json.loads(raw))
            result_sha=sha256(raw).hexdigest()
            stdout=exact(control/"CHILD.stdout",terminal["stdout_sha256"],8*MIB).decode("utf-8")
            need(parse_tail(stdout)==dict(status=CHILD_COMPLETE,result_sha256=result_sha),"Exact original child result stdout")
            closure.update(status=CLOSURE_COMPLETE,result_sha256=result_sha)
        except (OSError,ValueError,KeyError,TypeError) as error:
            closure["failure"]=type(error).__name__+": "+str(error)
    digest=write(control/"OWNER_CLOSURE_PRIVATE.json",closure)
    emit(dict(status=closure["status"],closure_sha256=digest,result_sha256=closure["result_sha256"]))
    return 0 if closure["status"]==CLOSURE_COMPLETE else 5


def main():
    parser=ArgumentParser(description=__doc__,allow_abbrev=False)
    sub=parser.add_subparsers(dest="mode",required=True)
    export=sub.add_parser("export",allow_abbrev=False)
    attest=sub.add_parser("attest",allow_abbrev=False)
    export.set_defaults(run=cli_export)
    attest.set_defaults(run=cli_attest)
    for name in ("worker","worker-python","job-helper","control","dev-root","acceptance","author-source","author-tree",
                 "archive","ema","codec-wheel","gpu-reservation"):
        export.add_argument("--"+name,type=Path,required=True)
    for name in ("acceptance-sha","gpu-reservation-sha","reservation-key"):
        export.add_argument("--"+name,required=True)
    for name in ("control","tool-wait","output"):
        attest.add_argument("--"+name,type=Path,required=True)
    for name in ("closure-sha","tool-wait-sha"):
        attest.add_argument("--"+name,required=True)
    args=parser.parse_args()
    need(os.name=="nt" and sys.flags.isolated and sys.dont_write_bytecode,"Isolated Windows owner process")
    raise SystemExit(args.run(args))


if __name__=="__main__":
    main()

