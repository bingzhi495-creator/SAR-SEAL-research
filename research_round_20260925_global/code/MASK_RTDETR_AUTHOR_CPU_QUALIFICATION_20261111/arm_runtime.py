"""Pinned Linux ARM CPU runtime for the separate native training qualification."""
import os
from pathlib import Path
import platform
import random
import sys

from qualify import SEED,require


def load_runtime(source):
    require(platform.machine()=="aarch64" and sys.version_info[:3]==(3,12,3)
            and os.environ.get("CUDA_VISIBLE_DEVICES")=="" and os.environ.get("OMP_NUM_THREADS")=="2"
            and sorted(os.sched_getaffinity(0))==[2,3],"Exact admitted ARM CPU runtime and affinity")
    sys.path.insert(0,str(source))
    import numpy as np
    import paddle
    import ppdet
    require(paddle.__version__=="2.6.2" and not paddle.is_compiled_with_cuda()
            and Path(ppdet.__file__).resolve().parent==(source/"ppdet").resolve(),
            "Exact CPU paddle and author source")
    paddle.set_device("cpu")
    random.seed(SEED)
    np.random.seed(SEED)
    paddle.seed(SEED)
    return np,paddle

