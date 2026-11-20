"""Independently verify every closed edit is inside the frozen native band."""
import argparse
from collections import Counter
from pathlib import Path
import sys

import numpy as np
from pycocotools import mask as codec


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("replay-source","old","dev-source","rank-math","manifest","capture","accepted","replay","output"):
        parser.add_argument("--"+key,type=Path,required=True)
    for key in ("accepted-sha","replay-sha"):
        parser.add_argument("--"+key,required=True)
    a=parser.parse_args()
    raw=a.replay_source.read_bytes()
    from hashlib import sha256
    if sha256(raw).hexdigest()!="ee1db45898c42262ba263e15a14ee24c72ef1ee00d5015b5bd243daaeeee01d3":
        raise ValueError("Exact replay source")
    from types import ModuleType
    w=ModuleType("_independent_closed_replay")
    w.__file__=str(a.replay_source)
    exec(compile(raw,str(a.replay_source),"exec"),w.__dict__)
    old=w.load(a.old,w.OLD,"_original_mask_edits")
    dev=w.load(a.dev_source,w.DEV,"_original_author_validation")
    dev.cpu_runtime()
    rank=w.load(a.rank_math,old.RANK_MATH,"_native_support_math")
    prepared=w.require_prepare(a)
    receipt=w.accepted(a,dev)
    totals=Counter()
    for _,rows,audit,block in w.edit_blocks(a,dev,old,w.manifest(a),receipt,prepared):
        for key,arms in block["changes"].items():
            i=int(key)-1
            original=codec.decode(rows[i]["segmentation"]).astype(bool,order="C")
            pools,_=rank.editable_pools(original,audit[i]["box_xyxy"])
            adds=np.concatenate([pools[k][0] for k in ("add1","add2")])
            drops=np.concatenate([pools[k][0] for k in ("drop1","drop2")])
            for arm,v in arms.items():
                new=codec.decode(v["rle"]).astype(bool,order="C")
                add=np.flatnonzero(new.ravel() & ~original.ravel())
                drop=np.flatnonzero(original.ravel() & ~new.ravel())
                w.need(len(add)==len(drop)==v["swaps"] and np.isin(add,adds).all() and np.isin(drop,drops).all(),
                       "All exact add/drop cells inside same native Chebyshev band and author integer box")
                totals.update({arm+"_masks":1,arm+"_swaps":len(add)})
    a.output.mkdir()
    digest=dev.write_new(a.output/"RESULT_SAFE.json",dict(status="DFINE_SSDD_POINT_EDIT_NATIVE_SUPPORT_VERIFIED",
            protocol_sha256=w.PROTOCOL,replay_sha256=a.replay_sha,accepted_sha256=a.accepted_sha,
            exact_native_support=True,images=187,counts=dict(totals),FIT=False,DEV=True,TEST=False,GT_opened=False,AP=False))
    print(digest,flush=True)


if __name__=="__main__":
    main()
