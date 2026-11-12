"""D-FINE E24 FIT-only original author binary masks and retained query probability planes."""
import argparse
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import re
import sys
from time import perf_counter
from types import ModuleType
from zipfile import ZipFile

PROTOCOL = "3f63fd555086853ec9a668e99cf66af9459f69c5a471e9bfbe461cf999f0b64b"
BASE = "5be622cdec5c889ca8520aab5547f6f15f01a54568a0263423a816dca7f96626"
ACCEPT = "7401f031c996298caff1761c6ef691fae0c301e65c87c15f51d186dcf81f25d8"
EMA = "82ca80e2cb9a0b9673a136fad3d7de3097b7327f4f76e3a2d3e7184ab07e7fd8"
COHORT = "70e9a0e8fb30f5ba70b48b9067c1b95d07bfec23d6b248d92c5983727acceb6e"
ORDER = "c6bd988e2728f0f50e44dcd52134a970ca1a135c300ff15d828bc9950771dc0f"
MAP = "5731bf5b6422171f9c90108b32a0607f131164f6e785e1b2a476c0ce82391939"
LAYOUT = "a7e4d2a7ff9e8ff238adbf94a1d0397553d8746e2558c316db4e417f86f8fec9"
SPLITS = {"core": (109,"8d94462e7bae8cdb1178f6371dbb6e30889aa179b646ac43bdb4c1906e217d57"),
          "stress": (5,"bfa31cdfb3d5ff468eb51259588f1395992f041822a27b2f39d0996cff777dc2")}
STATUS = "DFINE_E24_FIT114_IMAGE_CAPTURE_COMPLETE"
MIB = 2**20


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def encoded(value):
    return (json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()


def pin(path, digest, limit=4*MIB):
    need(path.is_absolute() and path.resolve(strict=True)==path and path.is_file()
         and 0 < path.stat().st_size <= limit, "Exact bounded canonical file")
    raw=path.read_bytes()
    need(sha256(raw).hexdigest()==digest, "Exact input digest "+path.name)
    return raw


def sync_write(path, raw):
    with path.open("xb") as out:
        out.write(raw)
        out.flush()
        os.fsync(out.fileno())
    return sha256(raw).hexdigest()


def token(status):
    return status.st_dev,status.st_ino,status.st_size,status.st_mtime_ns


def roster(args, archive):
    raw=archive.read("COHORT_PRIVATE.json")
    need(sha256(raw).hexdigest()==COHORT and len(raw)<MIB,"Exact historical FIT cohort")
    doc=json.loads(raw)
    rows=doc["images"]
    names=json.loads(pin(args.fit_map,MAP))
    layout_rows=json.loads(pin(args.fit_layout,LAYOUT))["images"]
    layout={row["image_id"]:row for row in layout_rows}
    first=json.loads(pin(args.fit_order,ORDER).splitlines()[0])
    need(doc["fit_only"] is True and len(rows)==114 and first["epoch"]==1
         and len(names)==len(layout)==2937 and all(r["core"] != r["stress"] for r in rows), "Exact native FIT cohort")
    positions={}
    for i,identity in enumerate(i for b in first["batches"] for i in b):
        positions.setdefault(identity,i)
    for row in rows:
        name=row["native_basename"]
        identity=row["image_id"]
        original=layout[identity]
        need(re.fullmatch(r"P[0-9]+_[A-Za-z0-9_]+\.png",name) is not None and name.split("_",1)[0]==row["group"]
             and names[str(identity)]==name and all(original[k]==row[k] for k in
             ("native_basename","encoded_sha256","decoded_rgb_sha256","encoded_bytes")),"Exact native image metadata")
    out={}
    for split,(count,digest) in SPLITS.items():
        subset=sorted((r for r in rows if r[split]),key=lambda r:positions[r["image_id"]])
        ids=[r["image_id"] for r in subset]
        raw_ids=json.dumps(ids,separators=(",",":")).encode()
        need(len(subset)==len(set(ids))==count and sha256(raw_ids).hexdigest()==digest,"Exact original 5202 visit order")
        out[split]=subset
    core=out["core"]
    need(all(len({r[k] for r in core})==109 for k in ("group","encoded_sha256","decoded_rgb_sha256")),
         "Exact source-diverse and content-unique core")
    return out


def read_png(archive,row,np,Image):
    info=archive.getinfo("images/"+row["native_basename"])
    need(0 < info.file_size==row["encoded_bytes"] <= 4*MIB,"Exact native PNG member")
    raw=archive.read(info)
    need(len(raw)==row["encoded_bytes"] and sha256(raw).hexdigest()==row["encoded_sha256"],"Exact encoded PNG")
    with Image.open(BytesIO(raw)) as original:
        need(original.format=="PNG" and original.size==(800,800),"Exact native image")
        rgb=np.asarray(original.convert("RGB"),dtype=np.uint8)
    need(rgb.shape==(800,800,3) and sha256(rgb.tobytes(order="C")).hexdigest()==row["decoded_rgb_sha256"],
         "Exact admitted decoded FIT pixels")
    return rgb


def initialize(args):
    base=ModuleType("_original_dfine_dev")
    base.__file__=str(args.base)
    exec(compile(pin(args.base,BASE),str(args.base),"exec"),base.__dict__)
    accepted=base.admitted_e24(json.loads(pin(args.acceptance,ACCEPT)))
    need(accepted["ema_checkpoint_sha256"]==EMA,"Exact original E24 EMA")
    base.admit_source(args.author_source,args.author_tree)
    import torch
    import torchvision
    from PIL import __version__ as pillow
    from dfine_seg.infer.torch_model import TorchModel
    need(sys.version_info[:3]==(3,11,15) and torch.__version__=="2.13.0+cu130"
         and torchvision.__version__=="0.28.0+cu130" and pillow=="13.0.0"
         and Path(sys.modules[TorchModel.__module__].__file__).resolve()==args.author_source/"dfine_seg/infer/torch_model.py",
         "Exact original author runtime")
    need(torch.cuda.is_available() and torch.cuda.device_count()==1,"Sole isolated CUDA device")
    device=torch.device("cuda:0")
    free,total=torch.cuda.mem_get_info(device)
    need(free>=6*1024**3,"Original CUDA export worker resource floor")
    torch.cuda.set_per_process_memory_fraction(min(.35,6*1024**3/total),device)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32=False
    torch.set_num_threads(2)
    before=token(args.ema.stat())
    checkpoint=torch.load(BytesIO(pin(args.ema,EMA,256*MIB)),map_location="cpu",weights_only=True)
    need(set(checkpoint)=={"model","meta"} and checkpoint["meta"]==base.AUTHOR_META and
         before[2]==accepted["ema_checkpoint_bytes"],"Exact original author checkpoint")
    model=TorchModel(str(args.ema),conf_thresh=0.,rect=False,keep_ratio=False,apply_nms=True,
                     nms_iou_thresh=.70,binarize_masks=True,mask_threshold=.50,device="cuda",task="segment")
    base.check_wrapper(model,checkpoint["model"],torch)
    del checkpoint
    need(token(args.ema.stat())==before,"Exact accepted EMA unchanged")
    return base,torch,model,sys.modules[TorchModel.__module__]


def bind_observers(model, author, torch):
    forward,nms=[],[]
    original_nms=author.nms

    def observe(_,inputs,out):
        need(len(inputs)==1 and tuple(inputs[0].shape)==(1,3,640,640) and inputs[0].dtype==torch.float32
             and tuple(out["pred_logits"].shape)==(1,300,1) and tuple(out["pred_boxes"].shape)==(1,300,4)
             and tuple(out["pred_masks"].shape)==(1,300,160,160) and out["pred_masks"].dtype==torch.float32,
             "Exact original 300-query one-view probability carrier")
        forward.append(out)

    def nms_read(boxes,scores,threshold):
        indices=original_nms(boxes,scores,threshold)
        need(threshold==.70,"Exact stock NMS threshold")
        nms.append((boxes,scores,indices))
        return indices

    hook=model.model.register_forward_hook(observe)
    author.nms=nms_read
    return forward,nms,hook,original_nms


def capture_row(image,row,model,author,torch,np,base,forward,nms):
    forward.clear()
    nms.clear()
    predicted,=model(image,bgr=False,conf_thresh=0.)
    need(len(forward)==len(nms)==1,"Exact one original forward and NMS observation")
    raw=forward[0]
    nms_boxes,nms_scores,keep=nms[0]
    scores,query=torch.topk(torch.sigmoid(raw["pred_logits"]).flatten(1),300,dim=-1)
    native_boxes=model.process_boxes(raw["pred_boxes"],[(640,640)],[(800,800)],False)[0]
    need(torch.equal(scores[0],nms_scores) and torch.equal(native_boxes[query[0]],nms_boxes)
         and torch.equal(predicted["scores"],nms_scores[keep]) and torch.equal(predicted["boxes"],nms_boxes[keep]),
         "Exact original pre/post-NMS score and box lineage")
    query=query[0,keep]
    qids=query.tolist()
    need(len(qids)==len(set(qids))==len(predicted["scores"]),"Exact post-NMS raw query lineage")
    limit=min(100,len(qids))
    planes=raw["pred_masks"][0,query[:limit]].contiguous()
    need(bool(torch.isfinite(planes).all()) and bool(((planes>=0)&(planes<=1)).all()),"Original FP32 probabilities")
    if limit:
        exact_resize=model.process_masks(planes[None],(640,640),[(800,800)],False)[0]
        exact_binary=author.cleanup_masks((exact_resize>=.50).view(torch.uint8),predicted["boxes"][:limit])
        need(torch.equal(exact_binary,predicted["masks"][:limit]),"Original author top100 exact binary mask parity")
    array=planes.cpu().numpy().astype("<f4",copy=False)
    masks=predicted.get("masks")
    coco,audit=base.pack_candidates(row["image_id"],(800,800),predicted["labels"].cpu().numpy(),
                                    predicted["scores"].cpu().numpy(),predicted["boxes"].cpu().numpy(),
                                    None if masks is None else masks.cpu().numpy())
    for i,candidate in enumerate(audit["candidates"]):
        candidate.update(query_id=qids[i],probability_row=i if i<limit else None)
    return coco,audit,array


def run_export(args):
    need(os.environ.get("CUDA_VISIBLE_DEVICES")=="0","Sole authorized Windows FIT GPU")
    base,torch,model,author=initialize(args)
    import numpy as np
    from PIL import Image

    args.output.mkdir()
    base.prepare_export_codec(args.codec_wheel,args.output/"CODEC_PRIVATE")
    reports={}
    started=perf_counter()
    forward,nms,hook,original_nms=bind_observers(model,author,torch)
    try:
        with args.bundle.open("rb") as stream, ZipFile(stream) as archive, torch.inference_mode():
            before=token(os.fstat(stream.fileno()))
            need(before[2]==151320868,"Exact known FIT bundle size")
            selected=roster(args,archive)
            for split,rows in selected.items():
                directory=args.output/split
                directory.mkdir()
                counts=Counter()
                predictions_sha=sha256()
                with (directory/"CANDIDATES_PRIVATE.jsonl").open("xb") as predictions:
                    for ordinal,row in enumerate(rows,1):
                        rgb=read_png(archive,row,np,Image)
                        coco,audit,planes=capture_row(rgb,row,model,author,torch,np,base,forward,nms)
                        file=f"{ordinal:03d}.npy"
                        buf=BytesIO()
                        np.save(buf,planes,allow_pickle=False)
                        raw=buf.getvalue()
                        ps=sync_write(directory/file,raw)
                        block=dict(ordinal=ordinal,image_id=row["image_id"],group=row["group"],
                                   basename=row["native_basename"],rgb_sha256=row["decoded_rgb_sha256"],
                                   coco=coco["coco"],candidates=audit["candidates"],
                                   probabilities=dict(filename=file,sha256=ps,bytes=len(raw),rows=len(planes)),
                                   exact_author_binary_replay=len(planes))
                        line=encoded(block)
                        predictions.write(line)
                        predictions_sha.update(line)
                        counts.update(images=1,post_nms=len(audit["candidates"]),probability_rows=len(planes),
                                      empty=sum(c["mask_area"]==0 for c in audit["candidates"]),
                                      exact_author_binary_replay=len(planes),probability_file_bytes=len(raw))
                        if ordinal%20==0 or ordinal==len(rows):
                            print(json.dumps(dict(event="DFINE_FIT114_CAPTURE_PROGRESS",split=split,images=ordinal)),
                                  flush=True)
                    predictions.flush()
                    os.fsync(predictions.fileno())
                reports[split]=dict(predictions_sha256=predictions_sha.hexdigest(),**counts)
            need(before==token(os.fstat(stream.fileno()))==token(args.bundle.stat()),"Original FIT bundle unchanged")
    finally:
        hook.remove()
        author.nms=original_nms
    torch.cuda.synchronize()
    result=dict(status=STATUS,protocol_sha256=PROTOCOL,source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                FIT=True,DEV=False,TEST=False,GT_opened=False,AP=False,optimizer_updates=0,images=114,
                model_forwards=114,core_images=109,stress_images=5,probability_semantics="decoder_sigmoid_FP32",
                original_binary_parity="author_FP16_resize_threshold_and_box_cleanup",ema_sha256=EMA,
                e24_acceptance_sha256=ACCEPT,fit_order_sha256=ORDER,split_reports=reports,
                elapsed_seconds=perf_counter()-started,peak_allocated=torch.cuda.max_memory_allocated())
    digest=sync_write(args.output/"RESULT_SAFE.json",encoded(result))
    print(json.dumps(dict(status=STATUS,result_sha256=digest),sort_keys=True),flush=True)


def run_verify(args):
    import numpy as np
    from pycocotools import mask as codec

    result=json.loads(pin(args.capture/"RESULT_SAFE.json",args.result_sha))
    terminal=json.loads(pin(args.terminal,args.terminal_sha))
    wait=json.loads(pin(args.tool_wait,args.tool_wait_sha))
    child=json.loads(pin(args.terminal.with_name("CHILD_TERMINAL_PRIVATE.json"),terminal["child_terminal_sha256"]))
    end=wait["terminal"]
    tail=json.loads(end["output"].strip().splitlines()[-1])
    need(type(wait["launch"].get("session_id")) is int and end["exit_code"]==0
         and tail["closure_sha256"]==args.terminal_sha and tail["result_sha256"]==args.result_sha
         and child["original_os_wait"] and child["original_os_exit"]==0 and child["job_assigned"]
         and child["job_active_before_drain"]==child["job_active_after_wait"]==0,
         "Exact natural original owner and drained child")
    need(terminal["status"]=="DFINE_E24_FIT114_OUTER_NATURAL_CLOSE" and terminal["resource_closed"]
         and terminal["result_sha256"]==args.result_sha and result["status"]==STATUS
         and result["images"]==result["model_forwards"]==114 and result["FIT"] is True
         and not any(result[k] for k in ("DEV","TEST","GT_opened","AP"))
         and result["source_sha256"]==sha256(Path(__file__).read_bytes()).hexdigest(),"Exact closed label-free capture")
    with ZipFile(args.bundle) as archive:
        selected=roster(args,archive)
    for split,rows in selected.items():
        directory=args.capture/split
        receipt=result["split_reports"][split]
        blocks=[json.loads(x) for x in pin(directory/"CANDIDATES_PRIVATE.jsonl",receipt["predictions_sha256"],32*MIB).splitlines()]
        counts=Counter()
        for i,(block,row) in enumerate(zip(blocks,rows,strict=True),1):
            need(block["ordinal"]==i and block["image_id"]==row["image_id"] and block["basename"]==row["native_basename"]
                 and block["rgb_sha256"]==row["decoded_rgb_sha256"] and block["group"]==row["group"],
                 "Exact frozen ordered original image identity")
            ref=block["probabilities"]
            meta=block["candidates"]
            limit=min(100,len(meta))
            need(ref["filename"]==f"{i:03d}.npy" and ref["rows"]==limit==block["exact_author_binary_replay"]
                 and len(meta)==len(block["coco"])<=300 and len({m["query_id"] for m in meta})==len(meta),
                 "Exact bounded candidate and array linkage")
            raw=pin(directory/ref["filename"],ref["sha256"],11*MIB)
            planes=np.load(BytesIO(raw),allow_pickle=False)
            need(planes.dtype==np.dtype("<f4") and planes.shape==(limit,160,160) and np.isfinite(planes).all()
                 and ((planes>=0)&(planes<=1)).all() and len(raw)==ref["bytes"],"Exact FP32 plane bytes and values")
            prior=1.
            for j,(m,p) in enumerate(zip(meta,block["coco"],strict=True)):
                need(m["native_rank"]==j+1 and m["probability_row"]==(j if j<limit else None)
                     and 0<=m["query_id"]<300 and p["image_id"]==row["image_id"] and p["category_id"]==1
                     and 0<=m["score"]==p["score"]<=prior and p["segmentation"]["size"]==[800,800]
                     and int(codec.area(p["segmentation"]))==m["mask_area"],"Exact candidate rank, scores and mask area")
                prior=m["score"]
            counts.update(images=1,post_nms=len(meta),probability_rows=limit,
                          empty=sum(m["mask_area"]==0 for m in meta),exact_author_binary_replay=limit,
                          probability_file_bytes=len(raw))
        need(dict(counts)=={k:v for k,v in receipt.items() if k!="predictions_sha256"},"Exact full split accounting")
    digest=sync_write(args.output,encoded(dict(status="DFINE_E24_FIT114_IMAGE_CAPTURE_INDEPENDENT_ACCEPTED",
                     protocol_sha256=PROTOCOL,result_sha256=args.result_sha,terminal_sha256=args.terminal_sha,
                     original_tool_wait_sha256=args.tool_wait_sha,split_reports=result["split_reports"],images=114,FIT=True,DEV=False,TEST=False,
                     GT_opened=False,AP=False,model_forwards=0,raw_probability_arrays_verified=True)))
    print(json.dumps(dict(status="DFINE_E24_FIT114_IMAGE_CAPTURE_INDEPENDENT_ACCEPTED",result_sha256=digest),
                     sort_keys=True),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    modes=p.add_subparsers(dest="mode",required=True)
    common=("protocol","bundle","fit-map","fit-layout","fit-order","output")
    for key, fields in (("export",("base","acceptance","ema","author-source","author-tree","codec-wheel")),
                        ("verify",("capture","terminal","tool-wait"))):
        stage=modes.add_parser(key)
        for f in common+fields:
            stage.add_argument("--"+f,required=True,type=Path)
        if key=="verify":
            stage.add_argument("--result-sha",required=True)
            stage.add_argument("--terminal-sha",required=True)
            stage.add_argument("--tool-wait-sha",required=True)
    a=p.parse_args()
    pin(a.protocol,PROTOCOL)
    def offline(event,_):
        if event.startswith(("socket.","subprocess.")):
            raise RuntimeError("Offline original FIT-only worker")
    if a.mode=="export":
        import torch
    sys.addaudithook(offline)
    (run_export if a.mode=="export" else run_verify)(a)


if __name__=="__main__":
    main()