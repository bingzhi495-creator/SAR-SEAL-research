"""Exercise the DEV705 owner and attestation joins with a data-free dummy child."""

from pathlib import Path
import sys

from dfine_dev705_owner import EVAL_PROTOCOL, WORKER_SHA, finish_owner, load_job, run_job


DUMMY = """
from hashlib import sha256
import json
from pathlib import Path
import sys
stage=Path(sys.argv[1])
stage.mkdir()
status="HRSID_DFINE_E24_DEV705_EXPORT_COMPLETE_V1"
raw=(json.dumps(dict(status=status,protocol_sha256=sys.argv[2],export_source_sha256=sys.argv[3],
    FIT=False,DEV=True,TEST=False,GT_opened=False,AP=False,images=705,synthetic_fixture=True),sort_keys=True)+"\\n").encode()
with (stage/"RESULT_SAFE.json").open("xb") as stream:
    stream.write(raw)
print(json.dumps(dict(status=status,result_sha256=sha256(raw).hexdigest())),flush=True)
"""


if __name__=="__main__":
    control,helper=map(Path,sys.argv[1:])
    cmd=[sys.executable,"-I","-B","-c",DUMMY,str(control/"EXPORT"),EVAL_PROTOCOL,WORKER_SHA]
    result=run_job(load_job(helper),cmd,control,lambda:None,timeout=10.)
    raise SystemExit(finish_owner(control,result))

