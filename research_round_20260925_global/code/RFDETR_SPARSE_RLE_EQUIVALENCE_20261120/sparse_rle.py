"""Sparse XOR of canonical compressed COCO RLE, without building a raster."""
from heapq import merge
from itertools import accumulate, groupby
from operator import index

from pycocotools import mask as codec


def need(ok,why):
    if not ok:
        raise ValueError(why)


def decode_runs(compressed):
    """Same signed 5-bit/delta convention as published verify_coverage.py."""
    text=compressed.decode("ascii") if isinstance(compressed,bytes) else compressed
    need(isinstance(text,str) and text,"Nonempty compressed COCO counts")
    runs=[]
    bit=value=0
    for character in text:
        symbol=ord(character)-48
        need(0<=symbol<=63,"Valid compressed RLE symbol")
        value|=(symbol&31)<<bit
        bit+=5
        need(bit<=40,"Compressed RLE integer width")
        if symbol&32:
            continue
        if symbol&16:
            value-=1<<bit
        if len(runs)>=3:
            value+=runs[-2]
        need(value>=0,"Nonnegative decoded COCO run")
        runs.append(value)
        bit=value=0
    need(bit==0 and runs and all(v>0 for v in runs[1:]),"Complete canonical COCO counts")
    return runs


def xor_rle(source,row_major_pixels):
    """Return exactly the raster XOR encoding; preserve the input object on no-op.

    Input pixels use row*width+col. COCO run offsets use col*height+row.
    Only runs and sparse boundary events are materialized in this fast path.
    """
    need(set(source)=={"size","counts"} and type(source["size"]) is list and len(source["size"])==2 and
         all(type(v) is int and v>0 for v in source["size"]) and isinstance(source["counts"],(str,bytes)),
         "Canonical original COCO RLE")
    height,width=source["size"]
    pixels=[]
    for pixel in row_major_pixels:
        need(not isinstance(pixel,bool),"Integer native pixel indices")
        try:
            value=index(pixel)
        except TypeError as exc:
            raise ValueError("Integer native pixel indices") from exc
        need(0<=value<height*width,"Native pixel index bounds")
        pixels.append(value)
    need(len(set(pixels))==len(pixels),"Duplicate pixel edits are rejected")
    if not pixels:
        return source
    runs=decode_runs(source["counts"])
    total=height*width
    need(sum(runs)==total,"Original RLE covers exactly the native canvas")
    events=set()
    for pixel in pixels:
        offset=(pixel%width)*height+pixel//width
        for boundary in (offset,offset+1):
            if boundary==total:
                continue
            if boundary in events:
                events.remove(boundary)
            else:
                events.add(boundary)
    output=[]
    previous=0
    # XOR flips toggle the same transition boundaries as the source runs.
    # Equal boundaries cancel, joining adjacent runs instead of emitting zeros.
    boundaries=merge(accumulate(runs[:-1]),sorted(events))
    for boundary,occurrences in groupby(boundaries):
        if sum(1 for _ in occurrences)%2:
            output.append(boundary-previous)
            previous=boundary
    output.append(total-previous)
    result=codec.frPyObjects(dict(size=[height,width],counts=output),height,width)
    if isinstance(source["counts"],str):
        result["counts"]=result["counts"].decode("ascii")
    return result
