"""Small deterministic synthetic fixture benchmark; no actual research payload."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import statistics
from time import perf_counter, process_time

import numpy as np
from pycocotools import mask as codec

from sparse_rle import decode_runs, xor_rle


def encode(mask):
    rle=codec.encode(np.asfortranarray(mask,dtype=np.uint8))
    return dict(size=rle["size"],counts=rle["counts"].decode("ascii"))


def dense_xor(source,pixels):
    mask=codec.decode(source).copy(order="C")
    mask.reshape(-1)[pixels]^=1
    return encode(mask)


def measurements(function,repeats):
    elapsed=[]
    cpu=[]
    for _ in range(3):
        function()
        start,task=perf_counter(),process_time()
        for _ in range(repeats):
            function()
        elapsed.append(1e6*(perf_counter()-start)/repeats)
        cpu.append(1e6*(process_time()-task)/repeats)
    return dict(wall_us=statistics.median(elapsed),cpu_us=statistics.median(cpu))


def fixtures():
    pixels=np.asarray([200*800+379+i for i in range(8)]+[201*800+380+i for i in range(8)],dtype=np.int64)
    empty=np.zeros((800,800),dtype=np.uint8)
    full=np.ones((800,800),dtype=np.uint8)
    compact=empty.copy()
    compact[200:225,380:400]=1
    fleet=empty.copy()
    for i in range(5):
        fleet[100+80*i:110+80*i,200+50*i:207+50*i]=1
    checker=np.indices((64,64)).sum(axis=0).astype(np.uint8)%2
    return [("empty800",empty,pixels),("full800",full,pixels),("rectangle800",compact,pixels),
            ("five_rectangles800",fleet,pixels),("fragmented64",checker,np.arange(16,dtype=np.int64))]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repeats",type=int,default=50)
    p.add_argument("--output",type=Path)
    args=p.parse_args()
    if args.repeats<=0:
        raise ValueError("Positive repeat count")
    records=[]
    for name,mask,pixels in fixtures():
        source=encode(mask)
        expected=dense_xor(source,pixels)
        if xor_rle(source,pixels)!=expected:
            raise ValueError("Fixture exact compressed-byte equivalence")
        fast=measurements(lambda:xor_rle(source,pixels),args.repeats)
        dense=measurements(lambda:dense_xor(source,pixels),args.repeats)
        records.append(dict(fixture=name,shape=list(mask.shape),flips=len(pixels),source_runs=len(decode_runs(source["counts"])),
                            source_compressed_bytes=len(source["counts"]),byte_exact=True,sparse=fast,dense=dense,
                            wall_ratio_dense_over_sparse=dense["wall_us"]/fast["wall_us"]))
    result=dict(status="SYNTHETIC_SPARSE_RLE_EQUIVALENCE_AND_TIMING",repeats_per_round=args.repeats,rounds=3,
                units="microseconds per call, median of round means",fixtures=records,actual_images=False,
                extrapolation=False,fast_path_raster_materialization=False)
    raw=(json.dumps(result,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
    if args.output:
        args.output.open("xb").write(raw)
    print(raw.decode(),end="")
    print("sha256 "+sha256(raw).hexdigest())


if __name__=="__main__":
    main()
