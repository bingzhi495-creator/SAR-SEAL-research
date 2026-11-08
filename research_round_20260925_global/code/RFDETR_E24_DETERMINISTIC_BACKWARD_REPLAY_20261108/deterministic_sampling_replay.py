"""Keep CUDA forward and request strict native CUDA determinism only for backward."""
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

import torch

D=Path(__file__).resolve().parent.parent
COMBINED=D/"RFDETR_E24_COMBINED_SAMPLING_BACKWARD_PROBE_CANDIDATE_20261108_01/combined_sampling_backward_probe.py"
COMBINED_SHA="b90dcaf0cc408b078c2fcad2adb2c646f0e2632fe034f6a4cf50792cff8822e2"


class StrictBackward:
    def __init__(self, stock):
        self.stock=stock
        self.calls=0
        self.seconds=0.0

    def __call__(self,*args,**kwargs):
        initial=torch.are_deterministic_algorithms_enabled()
        warned=torch.is_deterministic_algorithms_warn_only_enabled()
        start=time.monotonic()
        torch.use_deterministic_algorithms(True,warn_only=False)
        try:
            value=self.stock(*args,**kwargs)
            self.calls+=1
            return value
        finally:
            torch.use_deterministic_algorithms(initial,warn_only=warned)
            self.seconds+=time.monotonic()-start


def require(condition,message):
    if not condition:
        raise ValueError(message)


def main():
    if sys.argv[1:]==["--help"]:
        print("Use --attempt F0|F1 --output-dir <fresh directory> --protocol-commit <verified SHA>")
        return
    require(torch.__version__=="2.10.0+cu130"
         and torch.version.git_version=="449b1768410104d3ed79d3bcfe4ba1d65c7f22c0"
         and os.environ.get("CUBLAS_WORKSPACE_CONFIG")==":4096:8",
         "Exact PyTorch version and pre-CUDA cuBLAS workspace")
    require(not torch.are_deterministic_algorithms_enabled(),"Preserve native forward policy")
    b=COMBINED.read_bytes()
    require(sha256(b).hexdigest()==COMBINED_SHA,"Exact closed combined sampling wrapper")
    spec=importlib.util.spec_from_file_location("_deterministic_replay_combined",COMBINED)
    combined=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=combined
    spec.loader.exec_module(combined)
    trace=StrictBackward(torch.Tensor.backward)
    def backward(tensor,*args,**kwargs):
        return trace(tensor,*args,**kwargs)
    with patch.object(torch.Tensor,"backward",backward):
        combined.main()
    require(trace.calls==4 and not torch.are_deterministic_algorithms_enabled(),"Exact four strict backward windows")
    out=Path(sys.argv[sys.argv.index("--output-dir")+1])
    native=(out/"RESULT.json").read_bytes()
    routed=(out/"COMBINED_SAMPLING_BACKWARD_PROBE_PRIVATE.json").read_bytes()
    result=dict(status="STRICT_NATIVE_BACKWARD_AND_CPU_SAMPLING_REPLAY_COMPLETE_NO_UPDATE",
        combined_result_sha256=sha256(routed).hexdigest(),native_forward_result_sha256=sha256(native).hexdigest(),
        strict_backward_calls=trace.calls,strict_backward_seconds=trace.seconds,strict_warn_only=False,
        cuda_forward_deterministic_flag=False,cublas_workspace=":4096:8",
        optimizer_updates=0,ema_updates=0,AP=False,DEV=False,TEST=False)
    with (out/"DETERMINISTIC_BACKWARD_PRIVATE.json").open("x",encoding="utf-8",newline="\n") as f:
        json.dump(result,f,sort_keys=True,allow_nan=False)
        f.write("\n")
    print(json.dumps(result,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
