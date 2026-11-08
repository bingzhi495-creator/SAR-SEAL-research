"""Fit matched SSDD point heads and seal DEV187 masks without DEV annotations."""
import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import time
from types import ModuleType

import numpy as np

sys.path.insert(0,str(Path(__file__).parent))
from dev_readout_contract import ARMS,DEV_MANIFEST_SHA,NATIVE_SHA,SEEDS,encode_json,unchanged_identities,unordered_ring

SELECTIVE_SHA="4840f5511d4225ef1701e88f0b96da2669149ed529b7c071b4dd908d882b074b"
POLICY_SHA="44c79d7459a59d9dd601d5192c9743ecce815725dd24e1456ba14c70573e0031"
FIT_REFERENCE_SHA="17c79b4645e70d585d48ee1fab33048fc4f65bac0e4ed48a44fadb0bf0ec36a4"
RNG_SUPPORT=5201
K=4
CAP=128


def need(ok,message):
    if not ok:
        raise ValueError(message)


def load_source(path,digest,name):
    raw=path.read_bytes()
    need(sha256(raw).hexdigest()==digest,"Pinned source bytes")
    module=ModuleType(name)
    module.__file__=str(path)
    sys.modules[name]=module
    exec(compile(raw,str(path),"exec"),module.__dict__)
    return module


def checked_bytes(path,size,digest):
    need(path.is_absolute() and path.resolve(strict=True)==path and path.stat().st_size==size,
         "Exact declared immutable input")
    raw=path.read_bytes()
    need(len(raw)==size and sha256(raw).hexdigest()==digest,"Pinned input hash")
    return raw


def write(path,payload):
    with path.open("xb") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    return dict(bytes=len(payload),sha256=sha256(payload).hexdigest())


def read_manifest(path):
    doc=json.loads(checked_bytes(path,49477,DEV_MANIFEST_SHA))
    items=doc["items"]
    paths=[Path(x["image_ref"]["path"]) for x in items]
    ids=[x["image_id"] for x in items]
    groups={x["image_ref"]["sha256"] for x in items}
    need(doc["population"]=="internal_DEV187" and doc["expected_ids"]==ids and len(set(ids))==len(set(paths))==187
         and len(groups)==183 and len({p.parent for p in paths})==1
         and all(p.parent.name=="valid" and p.suffix==".jpg" and p.resolve(strict=True)==p for p in paths),
         "Exact image-only DEV187 population, including four duplicate encodings")
    return items,paths,groups


def image_only_guard(paths,archive):
    images=set(paths)
    parent=paths[0].parent.parent.parent
    opened=set()
    def guard(event,args):
        if event!="open" or not isinstance(args[0],(str,bytes)):
            return
        path=Path(os.path.abspath(os.fsdecode(args[0])))
        mode,flags=args[1:3]
        writing=(isinstance(mode,str) and any(c in mode for c in "wax+")) or (
                 isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)))
        private_gt=path.name.casefold() in {"dev.json","test.json","instances_val.json","instances_test.json"}
        protected=path.is_relative_to(parent) or path.suffix.lower() in {".pt",".pth",".ckpt",".safetensors"}
        if private_gt or (protected and (writing or path not in images|{archive})):
            raise PermissionError("Outside admitted read-only DEV image / closed E24 archive roles")
        if path in images:
            opened.add(path)
    sys.addaudithook(guard)
    return opened


def fit_heads(env,teacher,base,prior,local,items,seed,projection,reference):
    data,counts=prior.collect(env,teacher,local,items,"dev_readout_fit",projection)
    targets=np.concatenate([r["truth"]!=r["inside"] for r in data])
    w=prior.weights(data)
    logit=np.concatenate([r["inputs"]["logit"] for r in data])
    center=np.concatenate([r["inputs"]["point"] for r in data])
    inputs=dict(local_logit=logit,center=center,ring=unordered_ring(logit))
    heads={}
    loss={}
    train_seconds={}
    old=base.SEED
    try:
        base.SEED=seed
        for name,value in inputs.items():
            ts=time.perf_counter()
            heads[name],loss[name]=base.train_model(env.torch,value,targets,w,True)
            train_seconds[name]=time.perf_counter()-ts
    finally:
        base.SEED=old
    need(counts["used_matches"]==reference["point_fit_pairs"] and counts["points"]==reference["point_fit_points"],
         "Exact common 290-image fitting points")
    if seed==5201:
        need(loss["local_logit"]==reference["point_fit_loss"],"Exact original FIT head loss at seed 5201")
    return heads,loss,counts,train_seconds


def original(env,image,hw):
    with env.torch.inference_mode():
        raw,f,q=env.infer(image)
        z=raw["pred_masks"][0]
        mask,_,keys,scores=env.core._read_stock(raw,hw,env.top,env.torch)
        keep=[j for j,(_,slot) in enumerate(keys) if slot==0]
        query=[keys[j][0] for j in keep]
        ctx=env.head.candidate_context(f,q,z)
    return dict(hw=hw,basis=f,z=z,active=ctx["active"],query=query,
                rank=[j+1 for j in keep],scores=[scores[j] for j in keep],native=mask[keep,0].cpu().numpy())


def point_fields(c,index,teacher,base,local,torch,projection):
    supported=[]
    for j,q in enumerate(c["query"]):
        if not bool(c["active"][q]):
            continue
        xy,_,_,distance,inside=local.boundary_samples(c["native"][j],CAP,RNG_SUPPORT+100*index+q)
        if int(inside.sum())>=K and int((~inside).sum())>=K:
            supported.append((j,q,xy,distance,inside))
    if not supported:
        return supported,None,None
    queries=[x[1] for x in supported]
    planes=[]
    for i in range(0,len(queries),16):
        planes.extend(teacher.render(c["z"][queries[i:i+16]],c["hw"]).detach().cpu().numpy())
    h,w=c["hw"]
    all_xy=np.concatenate([v[2] for v in supported])
    norm=(2*(all_xy+.5)/np.array((w,h))-1).astype(np.float32)
    with torch.no_grad():
        f=c["basis"].float()
        basis=(projection.T@f.flatten(1)).reshape(32,*f.shape[-2:])
        grid=torch.as_tensor(norm,device="cuda")[None,None]
        center=torch.nn.functional.grid_sample(basis[None],grid,mode="bilinear",
                    align_corners=False,padding_mode="border")[0,:,0].T.cpu().numpy()
    rows=[]
    for (j,_,xy,distance,inside),plane in zip(supported,planes,strict=True):
        area=float(np.log1p(np.count_nonzero(c["native"][j])))
        rows.append(base.logit_inputs(plane,xy,distance,inside,area,c["scores"][j],c["rank"][j]))
    return supported,np.concatenate(rows),center


def canonical_rows(iid,scores,rles):
    return [dict(image_id=iid,category_id=1,score=score,
                 segmentation=dict(size=rle["size"],counts=rle["counts"].decode("ascii")))
            for score,rle in zip(scores,rles,strict=True)]


def readouts(env,c,iid,support,x,center,heads,policy,timing):
    native=env.s.coerce_rles(c["native"],env.codec)
    result=dict(native=canonical_rows(iid,c["scores"],native))
    sizes=[len(v[2]) for v in support]
    cut=np.cumsum(sizes)[:-1]
    for arm in ARMS[1:]:
        start=time.perf_counter()
        rles=list(native)
        if len(support):
            if arm=="uncertainty":
                score=-np.abs(x[:,0])
            elif arm=="center":
                score=heads[arm](np.column_stack((x[:,:6],center)))
            elif arm=="ring":
                score=heads[arm](unordered_ring(x))
            else:
                score=heads[arm](x)
            for (j,_,xy,_,inside),z in zip(support,np.split(score,cut),strict=True):
                pixel=xy[:,1]*c["hw"][1]+xy[:,0]
                add,delete=policy.choose_points(pixel,inside,z,K)
                updated=policy.edited_mask(c["native"][j],pixel[add],pixel[delete])
                rles[j]=env.codec.encode(np.asfortranarray(updated.astype(np.uint8)))
        result[arm]=canonical_rows(iid,c["scores"],rles)
        timing[arm]+=time.perf_counter()-start
    return result


def run(args,teacher,base,prior,local,policy):
    start=time.perf_counter()
    need(not args.output.exists() and args.output.parent.resolve(strict=True)==args.output.parent,
         "Fresh output scope")
    items,paths,dev_groups=read_manifest(args.manifest)
    opened=image_only_guard(paths,args.archive)
    reference=json.loads(checked_bytes(args.fit_reference,args.fit_reference.stat().st_size,FIT_REFERENCE_SHA))
    env=teacher.Environment(args)
    torch=env.torch
    torch.backends.cudnn.allow_tf32=True
    fit_index=json.loads(env.cap.exact_bytes(args.fit_index,348469,env.cap.FIT_INDEX_SHA))["items"]
    need(not dev_groups&{x["image_sha256"] for x in fit_index},"No DEV/FIT encoded image overlap")
    fit=base.parts(env.fit)["point_fit"]
    need(len(fit)==290 and len({x["image_sha256"] for x in fit})==288,"Exact point-fitting role")
    matrix=np.linalg.qr(np.random.default_rng(RNG_SUPPORT).normal(size=(256,32)))[0].astype(np.float32)
    projection=torch.as_tensor(matrix,device="cuda")
    ts=time.perf_counter()
    heads,loss,counts,head_times=fit_heads(env,teacher,base,prior,local,fit,args.seed,projection,reference)
    fit_total=time.perf_counter()-ts
    pred={k:[] for k in ARMS}
    keys=[]
    edited=empty=0
    timing=Counter()
    for i,(item,path) in enumerate(zip(items,paths,strict=True)):
        t=time.perf_counter()
        ref=item["image_ref"]
        raw=checked_bytes(path,ref["bytes"],ref["sha256"])
        hw=tuple(item["original_hw"])
        image,scale=env.decode(raw,draft_size=None)
        need(image.mode=="RGB" and image.size==(hw[1],hw[0]) and scale==(1.,1.),"Original DEV JPEG dimensions")
        c=original(env,image,hw)
        timing["read_decode_and_native"]+=time.perf_counter()-t
        t=time.perf_counter()
        support,x,center=point_fields(c,i,teacher,base,local,torch,projection)
        timing["shared_boundary_inputs"]+=time.perf_counter()-t
        iid=item["image_id"]
        group=readouts(env,c,iid,support,x,center,heads,policy,timing)
        keys.extend((iid,rank,q,0,score.hex()) for rank,q,score in zip(c["rank"],c["query"],c["scores"],strict=True))
        for k in ARMS:
            pred[k].extend(group[k])
        edited+=len(support)
        empty+=int((~c["native"].any(axis=(1,2))).sum())
        if (i+1)%32==0 or i+1==len(items):
            print(json.dumps(dict(event="DEV187_IMAGE_EXPORT_PROGRESS",images=i+1,candidates=len(keys))),flush=True)
    env.close()
    need(opened==set(paths) and len(keys)==15434 and empty==5180
         and all(unchanged_identities(pred["native"],v) for v in pred.values()),"Exact native DEV images and row identity")
    native=encode_json(pred["native"])
    need(sha256(native).hexdigest()==NATIVE_SHA,"Byte-exact same-DGX native DEV187 prediction replay")
    args.output.mkdir(mode=0o700)
    files={k:write(args.output/(k+".json"),native if k=="native" else encode_json(pred[k])) for k in ARMS}
    files["ROW_KEYS"]=write(args.output/"ROW_KEYS.json",encode_json(keys))
    summary=dict(status="DEV187_FIXED_LOGIT_IMAGE_ONLY_EXPORT_COMPLETE",protocol=args.protocol,seed=args.seed,
                 predictions=files,images=187,encoded_images=183,native_top100=18700,mapped_rows=len(keys),
                 empty_native_masks=empty,eligible_rows=edited,edits_each_non_native=edited*8,
                 original_candidate_identity_sha256=files["ROW_KEYS"]["sha256"],training=counts,
                 updates_each_learned=96,detector_updates=0,loss=loss,fit_total_seconds=fit_total,
                 per_head_training_seconds=head_times,shared_and_incremental_seconds=timing,
                 head_fit_encoded_groups=288,native_exact_sha=NATIVE_SHA,dev_gt_read=False,TEST=False,
                 cuda_peak_bytes=torch.cuda.max_memory_allocated(),elapsed_s=time.perf_counter()-start)
    write(args.output/"EXPORT.json",encode_json(summary))
    print(json.dumps(summary,sort_keys=True),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher",type=Path,required=True)
    parser.add_argument("--selective-source",type=Path,required=True)
    first,_=parser.parse_known_args()
    base=load_source(first.selective_source,SELECTIVE_SHA,"_prospective_selective")
    teacher=base.teacher_module(first.teacher)
    teacher.common_args(parser)
    for n in ("local-source","local-math","policy","fit-reference","manifest"):
        parser.add_argument("--"+n,type=Path,required=True)
    parser.add_argument("--seed",type=int,choices=SEEDS,required=True)
    args=parser.parse_args()
    prior=load_source(args.local_source,base.LOCAL_SHA,"_prospective_local")
    local=load_source(args.local_math,base.MATH_SHA,"_prospective_math")
    policy=load_source(args.policy,POLICY_SHA,"_prospective_action")
    run(args,teacher,base,prior,local,policy)

