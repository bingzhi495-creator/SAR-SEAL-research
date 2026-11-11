"""Trace SRNet mask target identities in the pinned author training path on FIT."""
import argparse
import ast
from collections import Counter
from hashlib import sha256
import inspect
import json
import logging
import os
from pathlib import Path
import random
import sys
from textwrap import dedent
from time import perf_counter
from types import ModuleType
from zipfile import ZipFile

P336_SHA="186d848c277467bae1809612b6ef8f8f4f0c61802df06e6d11e7c773c3486a48"
P336_REPORT_SHA="cef3b52db9621d5f3e1f50fcb581560106b7a6ae3523892e5de1c0c65cd824f8"
MORE_PINS={
    "mmrotate/models/dense_heads/SRNet_head.py":"144552cf27b5970c46ff4e3af1b4e0f59b1b273daa0aadf89cdcc513c8e2dba1",
    "mmrotate/models/dense_heads/SRNet_utils.py":"7d9210f7619fa31ad4bfacfc30e380cb1b4fce5d77e7217de15c4c14b5632c7f",
    "mmrotate/models/detectors/SRNet.py":"7fad564a8c2fb6f34f9e7dc197c7cafc643fe72b16817846fe173e52fe74a40a",
}
STATUS="SRNET_AUTHOR_REFINE_POSITIVE_MASK_TARGET_TRACE_COMPLETE"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load_source(path,pin,name):
    raw=path.read_bytes()
    require(0<len(raw)<=1024**2 and sha256(raw).hexdigest()==pin,"Exact prior ordinary source required")
    m=ModuleType(name)
    m.__file__=str(path)
    exec(compile(raw,str(path),"exec"),m.__dict__)
    return m


def cpu_loss(original, sample):
    """Change the target tensor device literal only and observe the original helper."""
    source=inspect.getsource(original)
    parsed=ast.parse(dedent(source))
    sites=[n for n in ast.walk(parsed) if isinstance(n,ast.Call)
           and isinstance(n.func,ast.Attribute) and n.func.attr=="to" and len(n.args)==1
           and isinstance(n.args[0],ast.Constant) and n.args[0].value=="cuda"]
    require(len(sites)==1,"One exact author target-device site required")
    sites[0].args[0].value="cpu"
    scope=dict(original.__globals__,sample_feature_and_gt_masks=sample)
    exec(compile(parsed,inspect.getsourcefile(original),"exec"),scope)
    return scope[original.__name__]


def ordered_target_ids(ids, levels, level):
    positions=levels.eq(level).nonzero(as_tuple=False).flatten()
    return positions,ids[:positions.numel()],ids.index_select(0,positions)


def mask_trace(model, features, rois, sampling, gt, meta, torch):
    original=type(model.mask_head).loss
    stock_sample=original.__globals__["sample_feature_and_gt_masks"]
    sample,=sampling
    masks,=gt
    ids=sample.pos_assigned_gt_inds.long()
    levels=model.mask_head.map_roi_levels(sample.pos_bboxes)
    require(ids.numel()>0 and bool((sample.pos_gt_labels==0).all()),
            "One-class actual positive refine assignments required")
    groups=sorted(levels.unique().tolist())
    queue=iter(groups)
    totals=Counter(positives=int(ids.numel()),pyramid_groups=len(groups),mixed_levels=int(len(groups)>1))
    hist=Counter()
    full=torch.from_numpy(masks.masks).float().index_select(0,ids)

    def observe(boxes, basis, stock_masks, img_meta, angle, n):
        level=next(queue)
        positions,source_ids,intended=ordered_target_ids(ids,levels,level)
        require(n==28 and angle=="le90" and torch.equal(boxes,sample.pos_bboxes.index_select(0,positions))
                and torch.equal(stock_masks,full),"Exact level-specific boxes and stock unpartitioned targets")
        pixels,old=stock_sample(boxes,basis,stock_masks,img_meta,angle,n)
        bad=(source_ids!=intended)
        hist[level]+=positions.numel()
        totals["wrong_identity"]+=int(bad.sum())
        if bool(bad.any()):
            new_pixels,new=stock_sample(boxes,basis,stock_masks.index_select(0,positions),img_meta,angle,n)
            require(torch.equal(pixels,new_pixels),"Pairing-only diagnostic must preserve feature samples")
            changed=(old!=new).flatten(1)
            logits=model.mask_head.logit(pixels)[:,0]
            old_cost=torch.nn.functional.binary_cross_entropy_with_logits(logits,old,reduction="sum")
            new_cost=torch.nn.functional.binary_cross_entropy_with_logits(logits,new,reduction="sum")
            totals["corrected_minus_stock_BCE_sum"]+=float(new_cost-old_cost)
            actual_changed=changed.any(dim=1)
            require(not bool((actual_changed&~bad).any()),"Correct identities must yield identical mask samples")
            totals["changed_sampled_masks"]+=int(actual_changed.sum())
            totals["changed_sampled_pixels"]+=int(changed.sum())
            totals["legacy_empty_corrected_nonempty"]+=int(((~old.bool().flatten(1).any(dim=1))
                                                           &new.bool().flatten(1).any(dim=1)).sum())
        return pixels,old

    losses=cpu_loss(original,observe)(model.mask_head,features,rois,sampling,gt,meta)
    require(next(queue,None) is None and sum(hist.values())==ids.numel()
            and all(bool(torch.isfinite(x).all()) for v in losses.values() for x in v),
            "Actual full stock mask loss and positive count must close")
    totals["images"]=1
    totals["images_with_wrong_identity"]=int(totals["wrong_identity"]>0)
    return totals,hist


def prepare_rows(dataset, fixed):
    by_id={row["id"]:i for i,row in enumerate(dataset.data_infos)}
    picks=[by_id[row["id"]] for row in fixed]
    stress=[]
    mixed=0
    for idx,row in enumerate(dataset.data_infos):
        b=dataset.get_ann_info(idx)["bboxes"]
        h,w=row["height"],row["width"]
        scale=(b[:,2]*b[:,3])**.5*608/max(h,w)
        if len(scale)>1 and min(scale)<112<=max(scale):
            mixed+=1
            if idx not in picks:
                stress.append((-float(max(scale)),row["id"],idx))
    normal=[("hash_core",i) for i in picks]
    normal.extend(("mixed_gt_scale_stress",x[2]) for x in sorted(stress)[:8])
    return normal,mixed


def run(a):
    started=perf_counter()
    p336=load_source(a.p336_source,P336_SHA,"_p339_p336")
    prior=json.loads(p336.read_exact(a.p336_report,P336_REPORT_SHA,65536))
    require(prior["status"]==p336.STATUS and prior["test"] is False,"Exact original author inference smoke required")
    p335=p336.load_prior(a.p335_source)
    for name,pin in dict(p335.SOURCE_PINS,**MORE_PINS).items():
        p336.read_exact(a.source/name,pin,1024**2)
    index=a.fit/"_annotations.coco.json"
    gt_index=json.loads(p336.read_exact(index,p335.FIT_INDEX_SHA,2*1024**2))
    p336.read_exact(a.archive,p335.FIT_SHA,34*1024**2)
    with ZipFile(a.archive) as z:
        rows={r["image_id"]:r for r in json.loads(z.read("manifest.json"))["rows"]}
    require(len(rows)==741,"Exact native FIT scope")
    require(len(p336.read_exact(a.r50,p336.R50_SHA,p336.R50_BYTES))==p336.R50_BYTES,
            "Exact author backbone initial bytes")
    sys.path.insert(0,str(a.source))
    import cv2
    import mmcv
    import mmrotate
    import numpy as np
    import torch
    from mmrotate.datasets import build_dataset
    from mmrotate.models import build_detector

    require(Path(mmrotate.__file__).resolve()==a.source/"mmrotate"/"__init__.py",
            "Exact author registry required")
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    logging.getLogger("mmcv").setLevel(logging.ERROR)
    cfg=mmcv.Config.fromfile(str(a.source/p335.AUTHOR_CONFIG))
    st=cfg.data.train.dataset.copy()
    st.ann_file=str(index)
    st.img_prefix=str(a.fit)+os.sep
    ds=build_dataset(st)
    visits,mixed=prepare_rows(ds,p335.choose(gt_index["images"]))
    torch.manual_seed(5201)
    cfg.model.backbone.init_cfg.checkpoint=str(a.r50)
    model=build_detector(cfg.model)
    model.init_weights()
    model.cpu().train()
    totals={}
    levels={}
    with torch.no_grad():
        for k,(stratum,idx) in enumerate(visits):
            info=ds.data_infos[idx]
            item=rows[info["id"]]
            require(info["filename"]==item["original_basename"],"Exact immutable prospective FIT view")
            p336.read_exact(a.fit/info["filename"],item["original_image_ref"]["sha256"],4*1024**2)
            seed=5201+k
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            data=ds.prepare_train_img(idx)
            if data is None:
                totals.setdefault(stratum,Counter())["augmentation_empty_or_filtered"]+=1
                continue
            tensor=data["img"].data[None]
            meta=[data["img_metas"].data]
            boxes=[data["gt_bboxes"].data]
            labels=[data["gt_labels"].data]
            masks=[data["gt_masks"].data]
            base,features=model.extract_feat(tensor)
            coarse=model.coarse_head(features)
            anchors=model.coarse_head.refine_bboxes(*coarse)
            refined=model.refine_head(model.align_conv(features,anchors))
            _,sampling=model.refine_head.loss(*refined,boxes,labels,meta,rois=anchors[0])
            rois=model.refine_head.refine_bboxes(*refined,rois=anchors[0])
            mask_feature,=model.mask_head((base,)+features)
            pair,level=mask_trace(model,mask_feature,rois,sampling,masks,meta,torch)
            totals.setdefault(stratum,Counter()).update(pair)
            levels.setdefault(stratum,Counter()).update(level)
            print(json.dumps(dict(event="SRNET_NATIVE_TARGET_TRACE_PROGRESS",visits=k+1,
                                  wrong_gt_identity_rows=pair["wrong_identity"])),flush=True)
    require(not torch.cuda.is_initialized(),"CPU-only stock mask loss trace")
    return dict(status=STATUS,author_commit=p335.AUTHOR_COMMIT,source_pins=MORE_PINS,
                fit=True,dev=False,test=False,ap=False,optimizer_updates=0,training_mode_forward_only=True,
                coarse_loss_skipped=True,coarse_refine_forward_and_refine_loss_stock=True,
                device_literal_change_in_mask_loss="cuda to cpu; other author operations unchanged",
                source_checkpoint_scope="official ImageNet backbone plus author initialized task heads",
                model_initialized_seed=5201,fit_index_sha256=p335.FIT_INDEX_SHA,
                mixed_gt_scale_candidate_images=mixed,prospectively_selected_visits=len(visits),
                strata={k:dict(v,level_positive_counts=dict(levels.get(k,{}))) for k,v in totals.items()},
                elapsed_seconds=perf_counter()-started)


def deny_network(event,_):
    if event.startswith("socket."):
        raise PermissionError("closed-input CPU target trace")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("p335-source","p336-source","p336-report","source","fit","archive","r50","output"):
        p.add_argument("--"+key,type=Path,required=True)
    a=p.parse_args()
    require(all(x.is_absolute() and x.resolve(strict=True)==x for x in (a.source,a.fit))
            and a.output.is_absolute() and a.output.parent.resolve(strict=True)==a.output.parent
            and not a.output.exists() and os.environ.get("CUDA_VISIBLE_DEVICES")=="-1",
            "Fresh isolated CPU target trace required")
    sys.addaudithook(deny_network)
    raw=(json.dumps(run(a),sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    with a.output.open("xb") as out:
        out.write(raw)
        out.flush()
        os.fsync(out.fileno())
    print(json.dumps(dict(status=STATUS,result_sha256=sha256(raw).hexdigest())),flush=True)


if __name__=="__main__":
    main()
