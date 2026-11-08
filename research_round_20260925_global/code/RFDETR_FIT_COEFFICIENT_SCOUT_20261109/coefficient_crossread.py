"""Fixed-rank matched/unmatched readout of the closed SSDD coefficient scout."""
import argparse
from collections import Counter
from contextlib import redirect_stdout
from hashlib import sha256
from importlib import import_module
from importlib.metadata import version
from io import BytesIO, StringIO
import json
import os
from pathlib import Path
import sys
import time
from types import ModuleType
from zipfile import ZipFile

import numpy as np

SCOUT_SHA="69dd7c06e346eab10117ac0ae77df8ba89ebd1b19773b3b9a62211df44f75ed3"
SCOUT_RESULT_SHA="fb985286825a3c878f66744edee99ddaeebf50edade5255ec8324b3317da3e28"


def split_edits(native, edited, match_indices):
    gate=np.zeros(len(native),dtype=bool)
    gate[sorted(match_indices)]=True
    keep=gate[:,None,None]
    return np.where(keep,edited,native),np.where(keep,native,edited)


def run(args):
    start=time.perf_counter()
    p=Path(__file__).with_name("coefficient_scout.py")
    raw=p.read_bytes()
    if sha256(raw).hexdigest()!=SCOUT_SHA:
        raise ValueError("Exact closed scout source")
    s=ModuleType("_closed_coefficient_scout")
    s.__file__=str(p)
    exec(compile(raw,str(p),"exec"),s.__dict__)
    need=s.need
    cap=s.module(args.capture_source,s.CAPTURE_SHA,"_closed_stock_capture")
    core=s.module(args.core,s.CORE_SHA,"_closed_native_candidate_core")
    ema=s.module(args.ema,s.EMA_SHA,"_closed_ema_loader")
    sys.addaudithook(cap.offline)
    import torch
    from pycocotools import mask as codec
    from pycocotools.coco import COCO
    from rfdetr.datasets.coco import make_coco_transforms_square_div_64
    from rfdetr.datasets.io_utils import decode_pil_image_bytes
    from rfdetr.models.postprocess import PostProcess
    from torch.nn.functional import interpolate

    head=s.module(Path(__file__).with_name("coefficient_head.py"),s.HEAD_SHA,"_scout_coefficient_head")
    need(sys.version_info[:3]==(3,12,14) and os.environ.get("CUDA_VISIBLE_DEVICES")=="0"
         and os.environ.get("HF_HUB_OFFLINE")==os.environ.get("TRANSFORMERS_OFFLINE")=="1"
         and all(version(k)==v for k,v in cap.VERSIONS.items())
         and all(sha256(Path(import_module(k).__file__).read_bytes()).hexdigest()==v
                 for k,v in cap.NATIVE_SOURCE.items()),"Exact isolated stock runtime")
    need(torch.cuda.is_available() and torch.cuda.device_count()==1,"One CUDA device")
    torch.cuda.set_per_process_memory_fraction(min(.35,cap.CAP/torch.cuda.get_device_properties(0).total_memory))
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("high")
    need(args.output.parent.resolve(strict=True)==args.output.parent and not args.output.exists(),
         "Fresh private output")
    report_raw=args.scout_result.read_bytes()
    need(sha256(report_raw).hexdigest()==SCOUT_RESULT_SHA,"Exact closed scout report")
    report=json.loads(report_raw)
    need(not report["gate"] and report["updates"]==dict(contrast=484,foreground=484,query=484),
         "Fixed completed negative scout")
    with args.adapter.open("rb") as f:
        before=os.fstat(f.fileno())
        need(cap.file_digest(f)==report["adapter_archive_sha256"],"Exact archived adapter bytes")
        f.seek(0)
        a=torch.load(f,map_location="cpu",weights_only=True)
        need(cap.identity(before)==cap.identity(os.fstat(f.fileno()))==cap.identity(args.adapter.stat()),
             "Closed adapter unchanged")
    need(a["protocol"]==report["protocol"] and a["steps"]==report["updates"],"Closed adapter lineage")
    adapter=head.MaskCoefficientHead()
    adapter.load_state_dict(a["arms"]["contrast"],strict=True)
    need(all(torch.isfinite(x).all() for x in adapter.parameters()),"Finite saved adapter")
    adapter.requires_grad_(False).eval().cuda()

    rows=json.loads(cap.exact_bytes(args.fit_index,348469,cap.FIT_INDEX_SHA))["items"]
    blocked={x["image_sha256"] for x,_ in cap.view_plan(rows,"cohort32")}
    hold=[x for x in rows if x["partition"]=="head_holdout" and x["image_sha256"] not in blocked]
    need(len(hold)==111 and len({x["image_sha256"] for x in hold})==110,"Exact frozen image scope")
    bundle=ZipFile(BytesIO(cap.exact_bytes(args.fit_bundle,33079338,cap.FIT_BUNDLE_SHA)))
    gt_raw=bundle.read("annotations/fit.json")
    need(len(gt_raw)==498760 and sha256(gt_raw).hexdigest()==cap.FIT_GT_SHA,"Exact original FIT GT")
    gt=json.loads(gt_raw)
    gt.setdefault("info",{})
    coco=COCO()
    coco.dataset=gt
    with redirect_stdout(StringIO()):
        coco.createIndex()
    detector,digest=s.native_detector(args,cap,ema,torch)
    tf=make_coco_transforms_square_div_64("val",576,multi_scale=False,expanded_scales=False,
       skip_random_resize=True,patch_size=12,num_windows=1,aug_config=None,scale_jitter=True,gpu_postprocess=False)
    infer=s.NativeForward(detector,torch,tf)
    top=core.make_native_top100(PostProcess)
    names=("native","contrast","matched_only","unmatched_only")
    detections={k:[] for k in names}
    geometry={k:Counter() for k in names}
    counts=Counter()
    for i,row in enumerate(hold,1):
        iid=row["image_id"]
        encoded=bundle.read(f'images/{row["image_sha256"]}.jpg')
        need(len(encoded)==row["image_bytes"] and sha256(encoded).hexdigest()==row["image_sha256"],
             "Exact FIT JPEG")
        hw=tuple(row["original_hw"])
        image,scale=decode_pil_image_bytes(encoded,draft_size=None)
        need(image.size==(hw[1],hw[0]) and scale==(1.,1.) and image.mode=="RGB","Original image geometry")
        anns=[x for x in coco.imgToAnns[iid] if not x.get("iscrowd",0)]
        truth=[coco.annToRLE(x) for x in anns]
        truth_bits=codec.decode(truth).transpose(2,0,1).astype(bool) if truth else None
        raw,f,q=infer(image)
        z=raw["pred_masks"][0]
        with torch.no_grad():
            masks,_,ids,scores=core._read_stock(raw,hw,top,torch)
            keep=[j for j,(_,slot) in enumerate(ids) if slot==0]
            query=[ids[j][0] for j in keep]
            native=masks[keep,0].cpu().numpy()
            ctx=head.candidate_context(f,q,z)
            edited=head.bounded_readout(f,q,z,ctx,adapter(ctx["contrast"]))[query]
            contrast=np.concatenate([interpolate(v[:,None],size=hw,mode="bilinear",align_corners=False)[:,0]
                                      .gt(0).cpu().numpy() for v in edited.split(32)]) if keep else native
        original_rles=s.coerce_rles(native,codec)
        iou=codec.iou(original_rles,truth,[0]*len(truth)) if keep and truth else np.zeros((len(keep),len(truth)))
        pairs=s.greedy(iou)
        matched,unmatched=split_edits(native,contrast,{j for j,_ in pairs})
        counts.update(matched=len(pairs),retained=len(keep),images=1)
        for k,planes in zip(names,(native,contrast,matched,unmatched),strict=True):
            rles=original_rles if k=="native" else s.coerce_rles(planes,codec)
            need(len(rles)==len(keep),"Frozen candidate ranks")
            iou=codec.iou(rles,truth,[0]*len(truth)) if keep and truth else iou
            for j,t in pairs:
                pred=planes[j]
                bits=truth_bits[t]
                fp=int((pred & ~bits).sum())
                fn=int((~pred & bits).sum())
                strict=int(iou[j,t]>=.90)
                geometry[k].update(fp=fp,fn=fn,strict=strict,
                                   small_strict=strict*int(anns[t]["area"]<1024),mask_area=int(pred.sum()))
            detections[k].extend(dict(image_id=iid,category_id=1,score=scores[j],segmentation=x)
                                 for j,x in zip(keep,rles,strict=True))
        if i%25==0 or i==len(hold):
            print(json.dumps(dict(event="COEFFICIENT_CROSSREAD_PROGRESS",images=i)),flush=True)
    infer.close()
    bundle.close()
    need(cap.state_digest(detector.state_dict())==digest,"Frozen EMA and no optimizer")
    ids=[x["image_id"] for x in hold]
    idset=set(ids)
    gt_eval=dict(info=gt["info"],images=[x for x in gt["images"] if x["id"] in idset],
                 annotations=[x for x in gt["annotations"] if x["image_id"] in idset],categories=gt["categories"])
    stats={k:s.fit_stats(gt_eval,pred,ids) for k,pred in detections.items()}
    for k in ("native","contrast"):
        need(stats[k]==report["metrics"][k]
             and geometry[k]["strict"]==report["matched_geometry"][k]["strict"]
             and geometry[k]["small_strict"]==report["matched_geometry"][k]["small_strict"],
             "Exact native and all-edit endpoint replay")
    identity=lambda v:[(x["image_id"],x["category_id"],x["score"].hex()) for x in v]
    need(all(identity(v)==identity(detections["native"]) for v in detections.values()),"Frozen candidates and score bits")
    decomposition={}
    for k in ("AP","AP75","AP90","APsmall","AR100"):
        n=stats["native"][k]
        m,u=stats["matched_only"][k]-n,stats["unmatched_only"][k]-n
        f=stats["contrast"][k]-n
        decomposition[k]=dict(full=f,matched_only=m,unmatched_only=u,interaction=f-m-u)
    result=dict(status="SSDD_FIT_COEFFICIENT_SCOUT_CROSSREAD_COMPLETE",protocol=args.protocol,
                scout_protocol=report["protocol"],metrics=stats,delta=decomposition,geometry=geometry,
                scope=dict(counts),FIT_only=True,DEV=False,TEST=False,AP=True,optimizer_updates=0,
                exact_endpoint_replay=True,seconds=time.perf_counter()-start,
                cuda_peak_bytes=torch.cuda.max_memory_allocated())
    args.output.mkdir()
    with (args.output/"RESULT.json").open("x",encoding="utf-8") as f:
        json.dump(result,f,indent=2,sort_keys=True,allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps(result,sort_keys=True),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("capture-source","core","ema","fit-index","fit-bundle","archive","recipe","scout-result","adapter","output"):
        p.add_argument("--"+n,type=Path,required=True)
    p.add_argument("--protocol",required=True)
    run(p.parse_args())
