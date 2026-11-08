"""One equal-budget frozen-EMA coefficient scout with all-candidate COCO evaluation."""
import argparse
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
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

CAPTURE_SHA="7a39df88f849d36ff139195d6001ccf7a5ea258144d9bcd4a3440e7a1a572703"
CORE_SHA="459115f5f279a4e7d1c598dca410bb8db117e97694be329fa74480be8d85ed13"
EMA_SHA="7bf88b0880a360f52de9263b215bfda30d34ab14d91183dc33a13a13b8aa26f5"
HEAD_SHA="12ebb61582d1356751d90f8c6e4a4a0e50b4e50dfb1b08b0de507b1607b26221"
SEED=5201


def need(ok,msg):
    if not ok:
        raise ValueError(msg)


def module(path,digest,name):
    raw=path.read_bytes()
    need(sha256(raw).hexdigest()==digest,"Exact source digest")
    mod=ModuleType(name)
    mod.__file__=str(path)
    sys.modules[name]=mod
    exec(compile(raw,str(path),"exec"),mod.__dict__)
    return mod


def greedy(iou):
    """Match native ranks in order, preserving source target tie order."""
    pairs=[]
    used=set()
    for i,row in enumerate(iou):
        eligible=[j for j in range(len(row)) if j not in used and row[j]>=.5]
        if eligible:
            j=max(eligible,key=lambda j:row[j])
            used.add(j)
            pairs.append((i,j))
    return pairs


def native_detector(args, cap, ema, torch):
    import yaml
    from rfdetr.config import RFDETRSegNanoConfig, SegmentationTrainConfig
    from rfdetr.models._defaults import MODEL_DEFAULTS
    from rfdetr.models.lwdetr import build_model_from_config

    recipe=yaml.safe_load(cap.exact_bytes(args.recipe,1815,cap.RECIPE_SHA))
    m=RFDETRSegNanoConfig(**recipe["model"]["model_config"]["init_args"])
    train=SegmentationTrainConfig(**recipe["model"]["train_config"]["init_args"])
    need(m.num_queries==m.num_select==100 and m.num_classes==1 and m.group_detr==13
         and m.resolution==576 and m.patch_size==12 and m.num_windows==1,"Exact stock model configuration")
    p=args.archive
    before=p.stat()
    with p.open("rb") as f:
        need(before.st_size==543050422 and cap.file_digest(f)==cap.ARCHIVE_SHA,"Exact accepted native E24 archive")
        f.seek(0)
        a=torch.load(f,map_location="cpu",weights_only=True)
        need(cap.identity(before)==cap.identity(os.fstat(f.fileno()))==cap.identity(p.stat()),"Unchanged closed archive")
    need(a["epoch"]==23 and a["global_step"]==4464,"Native E24 optimizer clock")
    detector=build_model_from_config(m,train,defaults=replace(MODEL_DEFAULTS,force_no_pretrain=True))
    witness=ema.strict_ema_load(detector,a,torch)
    need(witness["tensors"]==545 and witness["ema_updates"]==4464,"Exact strict E24 EMA readback")
    model_digest=cap.state_digest(detector.state_dict())
    detector.requires_grad_(False).cuda().eval()
    return detector,model_digest


class NativeForward:
    def __init__(self,model,torch,transform):
        self.model=model
        self.torch=torch
        self.transform=transform
        self.hits={}
        h=model.segmentation_head
        self.handles=[layer.register_forward_hook(self.hook(name)) for name,layer in
                      (("f",h.spatial_features_proj),("q",h.query_features_proj))]

    def hook(self,name):
        def record(_mod,_inputs,out):
            self.hits[name]=out
        return record

    def __call__(self,image):
        from rfdetr.utilities.tensors import nested_tensor_from_tensor_list

        self.hits.clear()
        x,_=self.transform(image,None)
        with self.torch.no_grad():
            sample=nested_tensor_from_tensor_list([x],block_size=12).to("cuda")
            raw=self.model(sample)
            f,q=self.hits["f"],self.hits["q"]
            z=self.torch.einsum("bchw,bnc->bnhw",f,q)+self.model.segmentation_head.bias
        need(tuple(f.shape)==(1,256,144,144) and tuple(q.shape)==(1,100,256)
             and self.torch.equal(z,raw["pred_masks"]),"Exact dense original inference readout")
        return raw,f[0],q[0]

    def close(self):
        for h in self.handles:
            h.remove()


def fit_pairs(torch,raw_mask,query_ids,gt,codec,active):
    from torch.nn.functional import interpolate

    if not gt or not len(query_ids):
        return [],None
    y=torch.from_numpy(np.ascontiguousarray(codec.decode(gt).transpose(2,0,1))).float().cuda()
    y=interpolate(y[:,None],size=(144,144),mode="nearest-exact")[:,0].bool()
    x=(raw_mask[query_ids]>0).flatten(1).float()
    flat=y.flatten(1).float()
    inter=x@flat.T
    union=x.sum(1)[:,None]+flat.sum(1)[None]-inter
    iou=(inter/union.clamp_min(1)).cpu().numpy()
    pairs=[(int(query_ids[i]),j) for i,j in greedy(iou) if bool(active[query_ids[i]])]
    return pairs,y


def coerce_rles(planes,codec):
    return [codec.encode(np.asfortranarray(p.astype(np.uint8))) for p in planes]


def fit_stats(gt, preds, ids):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with redirect_stdout(StringIO()):
        stock=COCO()
        stock.dataset=deepcopy(gt)
        stock.createIndex()
        output=stock.loadRes(deepcopy(preds)) if preds else COCO()
        if not preds:
            output.dataset=dict(images=deepcopy(gt["images"]),categories=gt["categories"],annotations=[])
            output.createIndex()
        ev=COCOeval(stock,output,"segm")
        ev.params.imgIds=sorted(ids)
        ev.params.catIds=[1]
        need(tuple(ev.params.maxDets)==(1,10,100),"Exact stock COCO maxDets")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    i=int(np.flatnonzero(np.isclose(ev.params.iouThrs,.90))[0])
    ap90=ev.eval["precision"][i,:,0,0,-1]
    return dict(AP=float(ev.stats[0]),AP75=float(ev.stats[2]),AP90=float(ap90[ap90>=0].mean()),
                APsmall=float(ev.stats[3]),AR100=float(ev.stats[8]),predictions=len(preds))


def run(args):
    started=time.perf_counter()
    cap=module(args.capture_source,CAPTURE_SHA,"_closed_stock_capture")
    core=module(args.core,CORE_SHA,"_closed_native_candidate_core")
    ema=module(args.ema,EMA_SHA,"_closed_ema_loader")
    sys.addaudithook(cap.offline)
    import torch
    from pycocotools import mask as codec
    from pycocotools.coco import COCO
    from rfdetr.datasets.coco import make_coco_transforms_square_div_64
    from rfdetr.datasets.io_utils import decode_pil_image_bytes
    from rfdetr.models.postprocess import PostProcess
    from torch.nn.functional import interpolate

    head=module(Path(__file__).with_name("coefficient_head.py"),HEAD_SHA,"_scout_coefficient_head")
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
    torch.manual_seed(SEED)
    need(args.output.parent.resolve(strict=True)==args.output.parent and not args.output.exists(),
         "Fresh private output")
    rows=json.loads(cap.exact_bytes(args.fit_index,348469,cap.FIT_INDEX_SHA))["items"]
    blocked={x["image_sha256"] for x,_ in cap.view_plan(rows,"cohort32")}
    fit=[x for x in rows if x["partition"]=="head_fit" and x["image_sha256"] not in blocked]
    hold=[x for x in rows if x["partition"]=="head_holdout" and x["image_sha256"] not in blocked]
    groups=lambda v:{x["image_sha256"] for x in v}
    need((len(rows),len(blocked),len(fit),len(hold),len(groups(fit)),len(groups(hold)))==(741,32,484,111,479,110)
         and not groups(fit)&groups(hold),"Exact paired scout group scope")
    bundle=ZipFile(BytesIO(cap.exact_bytes(args.fit_bundle,33079338,cap.FIT_BUNDLE_SHA)))
    gt_raw=bundle.read("annotations/fit.json")
    need(len(gt_raw)==498760 and sha256(gt_raw).hexdigest()==cap.FIT_GT_SHA,"Exact original FIT GT")
    gt=json.loads(gt_raw)
    gt.setdefault("info",{})
    coco=COCO()
    coco.dataset=gt
    with redirect_stdout(StringIO()):
        coco.createIndex()
    detector,digest=native_detector(args,cap,ema,torch)
    tf=make_coco_transforms_square_div_64("val",576,multi_scale=False,expanded_scales=False,
       skip_random_resize=True,patch_size=12,num_windows=1,aug_config=None,scale_jitter=True,gpu_postprocess=False)
    infer=NativeForward(detector,torch,tf)
    top=core.make_native_top100(PostProcess)
    arms=head.ARMS
    first=head.MaskCoefficientHead().cuda()
    adapters={k:deepcopy(first) for k in arms}
    optim={k:torch.optim.AdamW(h.parameters(),lr=3e-4,weight_decay=1e-4) for k,h in adapters.items()}
    updates=Counter()
    counts=Counter()
    losses=defaultdict(list)

    def read_image(row):
        raw=bundle.read(f'images/{row["image_sha256"]}.jpg')
        hw=tuple(row["original_hw"])
        need(len(raw)==row["image_bytes"] and sha256(raw).hexdigest()==row["image_sha256"],"Exact original FIT JPEG")
        image,scale=decode_pil_image_bytes(raw,draft_size=None)
        need(image.size==(hw[1],hw[0]) and scale==(1.,1.) and image.mode=="RGB","Original image geometry")
        return image,hw

    def image_truth(iid):
        anns=[x for x in coco.imgToAnns[iid] if not x.get("iscrowd",0)]
        return [coco.annToRLE(x) for x in anns],[float(x["area"])<1024 for x in anns]

    print(json.dumps(dict(event="COEFFICIENT_SCOUT_ADMITTED",fit_images=len(fit),holdout_images=len(hold),
                          parameters=sum(p.numel() for p in first.parameters()))),flush=True)
    train_start=time.perf_counter()
    for i,row in enumerate(fit,1):
        image,_=read_image(row)
        raw,f,q=infer(image)
        z=raw["pred_masks"][0]
        with torch.no_grad():
            score,slot,qi=PostProcess._select_topk(top,raw["pred_logits"])
            kept=qi[0,slot[0]==0]
            context=head.candidate_context(f,q,z)
            truth,_=image_truth(row["image_id"])
            pairs,y=fit_pairs(torch,z,kept,truth,codec,context["active"])
        counts.update(train_visits=1,supervised_rows=len(pairs),zero_supervised_images=int(not pairs))
        if pairs:
            qs,gs=zip(*pairs)
            qt=torch.tensor(qs,dtype=torch.int64,device=z.device)
            yt=y[list(gs)]
            for k in arms:
                optim[k].zero_grad(set_to_none=True)
                edited=head.bounded_readout(f,q,z,context,adapters[k](context[k]))
                loss=head.mask_loss(edited,z,qt,yt)
                need(bool(torch.isfinite(loss)),"Finite same-support training loss")
                loss.backward()
                optim[k].step()
                losses[k].append(float(loss.detach()))
                updates[k]+=1
        if i%100==0 or i==len(fit):
            print(json.dumps(dict(event="COEFFICIENT_SCOUT_TRAIN_PROGRESS",images=i,updates=dict(updates))),flush=True)
    torch.cuda.synchronize()
    train_seconds=time.perf_counter()-train_start
    need(len(set(updates.values()))==1 and set(updates)==set(arms)
         and min(updates.values())>=300,"Equal actual training counts with adequate FIT support")
    for k in arms:
        state=optim[k].state.values()
        need(len(state)==4 and all(int(x["step"])==updates[k] for x in state),"Exact actual AdamW clocks")
        adapters[k].eval()
    names=("native",)+arms
    detections={k:[] for k in names}
    by_image=[]
    edits=defaultdict(Counter)
    eval_start=time.perf_counter()
    for i,row in enumerate(hold,1):
        iid=row["image_id"]
        image,hw=read_image(row)
        raw,f,q=infer(image)
        z=raw["pred_masks"][0]
        truth,small=image_truth(iid)
        with torch.no_grad():
            masks,planes,identities,scores=core._read_stock(raw,hw,top,torch)
            keep=[j for j,(_,slot) in enumerate(identities) if slot==0]
            query=[identities[j][0] for j in keep]
            native=masks[keep,0].cpu().numpy()
            context=head.candidate_context(f,q,z)
            stock_rles=coerce_rles(native,codec)
            stock_iou=codec.iou(stock_rles,truth,[0]*len(truth)) if keep and truth else np.zeros((len(keep),len(truth)))
            pairs=greedy(stock_iou)
            matchset={j for j,_ in pairs}
            metrics={}
            for k in names:
                if k=="native":
                    planes_np=native
                    rles=stock_rles
                else:
                    edited=head.bounded_readout(f,q,z,context,adapters[k](context[k]))[query]
                    planes_np=np.concatenate([interpolate(v[:,None],size=hw,mode="bilinear",align_corners=False)[:,0]
                                                .gt(0).cpu().numpy() for v in edited.split(32)]) if keep else native
                    rles=coerce_rles(planes_np,codec)
                need(len(rles)==len(keep),"Every retained original candidate stays in its rank")
                iou=codec.iou(rles,truth,[0]*len(truth)) if keep and truth else stock_iou
                good=[bool(iou[j,t]>=.90) for j,t in pairs]
                metrics[k]=dict(strict=sum(good),small_strict=sum(g for g,(_,t) in zip(good,pairs) if small[t]))
                for pos,rle in enumerate(rles):
                    det=dict(image_id=iid,category_id=1,score=scores[keep[pos]],segmentation=rle)
                    detections[k].append(det)
                    count=int(np.count_nonzero(planes_np[pos]!=native[pos]))
                    edits[k].update(changed_rows=int(count>0),changed_pixels=count,
                       unmatched_changed_rows=int(pos not in matchset and count>0),
                       unmatched_changed_pixels=count if pos not in matchset else 0)
            by_image.append(metrics)
        if i%25==0 or i==len(hold):
            print(json.dumps(dict(event="COEFFICIENT_SCOUT_HOLDOUT_PROGRESS",images=i)),flush=True)
    infer.close()
    bundle.close()
    need(cap.state_digest(detector.state_dict())==digest,"Frozen EMA tensors and buffers unchanged")
    ids=[x["image_id"] for x in hold]
    idset=set(ids)
    gt_eval=dict(info=gt["info"],images=[x for x in gt["images"] if x["id"] in idset],
                 annotations=[x for x in gt["annotations"] if x["image_id"] in idset],categories=gt["categories"])
    need(len(gt_eval["images"])==111,"All original holdout GT images")
    stats={k:fit_stats(gt_eval,v,ids) for k,v in detections.items()}
    row_identity=lambda x:(x["image_id"],x["category_id"],x["score"].hex())
    native_ids=[row_identity(x) for x in detections["native"]]
    need(all([row_identity(x) for x in v]==native_ids for v in detections.values()),"Exact frozen ranks and score bits")
    rng=np.random.default_rng(SEED)
    groups_of_image=[x["image_sha256"] for x in hold]
    group_keys=sorted(set(groups_of_image))
    bootstrap=rng.integers(0,len(group_keys),size=(2048,len(group_keys)))
    paired={}
    for k in ("native","query","foreground"):
        paired[k]={}
        for key in ("strict","small_strict"):
            group=defaultdict(list)
            for image_key,x in zip(groups_of_image,by_image,strict=True):
                group[image_key].append(x["contrast"][key]-x[k][key])
            vals=np.array([np.mean(group[g]) for g in group_keys])
            paired[k][key]=dict(net=sum(x["contrast"][key]-x[k][key] for x in by_image),
                per_group=float(vals.mean()),percentile95=np.quantile(vals[bootstrap].mean(1),[.025,.975]).tolist())
    gate=all(stats["contrast"]["AP"]-stats[k]["AP"]>=.001
             and stats["contrast"]["AP75"]>=stats[k]["AP75"]
             and stats["contrast"]["AP90"]>stats[k]["AP90"]
             and paired[k]["small_strict"]["percentile95"][0]>0
             for k in ("native","query","foreground"))
    result=dict(status="SSDD_FIT_FROZEN_COEFFICIENT_SCOUT_COMPLETE",protocol=args.protocol,FIT_only=True,DEV=False,
        TEST=False,AP=True,formal=False,fit_images=len(fit),holdout_images=len(hold),fit_encoded_groups=len(groups(fit)),
        holdout_encoded_groups=len(groups(hold)),excluded_oracle_groups=len(blocked),frozen_ema_tensors=545,
        head_parameters=sum(p.numel() for p in first.parameters()),seed=SEED,updates=dict(updates),
        loss_mean={k:float(np.mean(v)) for k,v in losses.items()},fit_counts=dict(counts),metrics=stats,gate=gate,
        matched_geometry={k:{m:sum(v[k][m] for v in by_image) for m in ("strict","small_strict")} for k in names},
        paired_contrast=paired,edit_footprint={k:dict(v) for k,v in edits.items()},training_seconds=train_seconds,
        holdout_seconds=time.perf_counter()-eval_start,cuda_peak_bytes=torch.cuda.max_memory_allocated(),
        total_seconds=time.perf_counter()-started)
    args.output.mkdir()
    with (args.output/"ADAPTERS_PRIVATE.pt").open("xb") as f:
        torch.save(dict(protocol=args.protocol,steps=dict(updates),
                        arms={k:{name:v.detach().cpu() for name,v in h.state_dict().items()}
                              for k,h in adapters.items()}),f)
        f.flush()
        os.fsync(f.fileno())
    with (args.output/"ADAPTERS_PRIVATE.pt").open("rb") as f:
        result["adapter_archive_sha256"]=cap.file_digest(f)
    with (args.output/"RESULT.json").open("x",encoding="utf-8") as f:
        json.dump(result,f,sort_keys=True,indent=2,allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps(result,sort_keys=True),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("capture-source","core","ema","fit-index","fit-bundle","archive","recipe","output"):
        p.add_argument("--"+n,type=Path,required=True)
    p.add_argument("--protocol",required=True)
    run(p.parse_args())
