"""Original FIT point feature logic generalized only to native canvas size."""
from hashlib import sha256
from types import ModuleType

import numpy as np

PAIR="8fe7a2c146c08c89f377b1f20b321276a60356de4bdbb27be2a00544dfc77962"
RANK="cb438f8937f8e29dbdcbc441abdf5a788ac9c442fd63f1b3ad504b8ae0fcca0b"


def need(ok,why):
    if not ok:
        raise ValueError(why)


def load(path,digest,name):
    raw=path.read_bytes()
    need(sha256(raw).hexdigest()==digest,"Exact original point feature source")
    obj=ModuleType(name)
    obj.__file__=str(path)
    obj.raw=raw
    exec(compile(raw,str(path),"exec"),obj.__dict__)
    return obj


def load_old(pair,rank):
    return load(pair,PAIR,"_original_pair_math"),load(rank,RANK,"_original_rank_math")


def make_native(old,rank,height,width):
    need(type(height) is type(width) is int and height>0 and width>0,"Positive native canvas")
    source=old.raw.decode()
    edits=(
        ('v=np.clip(np.ceil(box).astype(np.int64),0,800)',
         'v=np.clip(np.ceil(box).astype(np.int64),0,np.array([WIDTH,HEIGHT,WIDTH,HEIGHT]))'),
        ('sample_probability(prob,coords,800)','sample_probability(prob,coords,WIDTH)'),
        ('pair_assignment(available[0][0],available[1][0],800)',
         'pair_assignment(available[0][0],available[1][0],WIDTH)'),
        ('np.divmod(paired[:,side],800)','np.divmod(paired[:,side],WIDTH)'))
    for before,after in edits:
        need(source.count(before)==1,"Exact original point canvas seam")
        source=source.replace(before,after,1)
    native=ModuleType("_native_pair_math")
    native.WIDTH,native.HEIGHT=width,height
    exec(compile(source,old.__file__,"exec"),native.__dict__)
    adapted=ModuleType("_native_probability_feature")
    adapted.editable_pools=rank.editable_pools
    def sample_probability(plane,coords,canvas_width):
        need(canvas_width==width,"Exact target probability width")
        half=plane.astype(np.float16).astype(np.float32)
        yy,xx=np.divmod(coords,width)
        need(np.all((coords>=0)&(coords<width*height)),"Exact native point range")
        def indices(target,length):
            scale=np.float32(160)/np.float32(length)
            position=np.maximum(np.float32(0),(target.astype(np.float32)+np.float32(.5))*scale-np.float32(.5))
            base=np.floor(position).astype(np.intp)
            fraction=position-base.astype(np.float32)
            return np.minimum(base,159),np.minimum(base+1,159),fraction
        y0,y1,dy=indices(yy,height)
        x0,x1,dx=indices(xx,width)
        top=(np.float32(1)-dx)*half[y0,x0]+dx*half[y0,x1]
        bottom=(np.float32(1)-dx)*half[y1,x0]+dx*half[y1,x1]
        return ((np.float32(1)-dy)*top+dy*bottom).astype(np.float16)
    adapted.sample_probability=sample_probability
    return native,adapted
