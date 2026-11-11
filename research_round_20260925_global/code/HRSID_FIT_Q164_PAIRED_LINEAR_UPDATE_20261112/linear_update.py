"""Actual four-step held-prefix paired AdamW assay on one native Linear."""
from argparse import ArgumentParser
from collections import defaultdict
from hashlib import sha256
from importlib import import_module
import io
import json
import os
from pathlib import Path
from statistics import mean
import sys
from time import perf_counter
from types import ModuleType

import numpy as np
from pycocotools import mask
import torch
import torch.nn.functional as F

COUNT_SHA = "7aadef7ea98eb595520b714086d94aa50fd926c4d504d0c2cb7ee92d62e5bb74"
PRODUCER_FOLDER = "HRSID_FIT_Q164_STOCK_POINT_BANK_DRAFT_20261112_01"
READOUT = "research_round_20260925_global/code/HRSID_FIT_LARGE_PAIRED_FOREGROUND_COUNT_20261112/count_readout.py"


def check(ok, why):
    if not ok:
        raise ValueError(why)


def exact(path, digest, maximum=2**20):
    check(path == path.resolve(strict=True) and 0 < path.stat().st_size <= maximum, "Canonical bounded source")
    raw = path.read_bytes()
    check(sha256(raw).hexdigest() == digest, "Closed source digest " + path.name)
    return raw


def module(path, digest, name):
    obj = ModuleType(name)
    obj.__file__ = str(path)
    exec(compile(exact(path,digest),str(path),"exec"),obj.__dict__)
    return obj


def emit(path, value):
    data = (json.dumps(value, allow_nan=False, sort_keys=True)+"\n").encode()
    with path.open("xb") as stream:
        stream.write(data)
    return sha256(data).hexdigest()


def load_views(args):
    root = args.root
    source = module(root/PRODUCER_FOLDER/"point_bank.py",args.bank_source_sha,"_closed_point_bank")
    safe = json.loads(exact(args.bank_dir/"RESULT_SAFE.json",args.bank_report_sha))
    check(safe["FIT"] is True and safe["DEV"] is safe["TEST"] is safe["AP"] is False
          and safe["bitwise_loss_and_gradient_witnesses"] == 56 and safe["rows"] == 113
          and safe["original_criterion_sha256"] == source.CRITERION_SHA, "Admitted full original stock loss witness")
    bank = torch.load(io.BytesIO(exact(args.bank_dir/"POINT_BANK_PRIVATE.pt",safe["bank_sha256"],8*2**20)),
                      map_location="cpu",weights_only=True)
    capture, rep = source.admit_capture(root)
    prev = source.load(root/"P392_HRSID_FIT_Q164_SHARED_BASIS_GRADIENT_DRAFT_PRIVATE_20261112_01/shared_gradient_validation_followup.py",
                       source.PRIOR_SOURCE,"_original_projection_matcher")
    old, _ = prev.prior(root)
    roster, streams, old_entries, spatial_dir, restore = old.input_authority(root)
    new_entries = json.loads(source.pin(capture/"INPUT_INDEX_PRIVATE.json",rep["private_index_sha256"]))
    params = np.frombuffer(source.pin(capture/"PROJECTION_PRIVATE.bin",rep["projection_sha256"]),dtype="<f4").copy()
    gt = old.read(root/"HRSID_FIT_NATIVE_COCO_LAYOUT_ACTUAL_PRIVATE_20261109_01/train/_annotations.coco.json",
                  old.FIT,32*2**20)
    target_rows = defaultdict(list)
    for row in gt["annotations"]:
        target_rows[row["image_id"]].append(row)
    initial = (torch.from_numpy(params[:65_536].reshape(256,256)), torch.from_numpy(params[65_536:]))
    check(tuple(p.numel() for p in initial)==(65_536,256) and len(bank)==28,"Same native projection dimensions")
    views = []
    for bank_row, image, raw, old_entry, q_entry in zip(bank,roster,streams,old_entries,new_entries,strict=True):
        iid = image["image_id"]
        archived = json.loads(raw)
        ordinal = bank_row["ordinal"]
        check(ordinal == old_entry["ordinal"] == q_entry["ordinal"] == len(views)+1 and iid == archived["image_id"]
              == old_entry["image_id"] and bank_row["q_input_sha256"] == q_entry["sha256"]
              and bank_row["old_spatial_sha256"] == old_entry["feature_sha256"]
              and bank_row["prefix"] == image["native_basename"].split("_",1)[0], "Aligned ordered FIT parents")
        spatial, projected, bias = restore.restore(old.pin(spatial_dir/"FEATURES_PRIVATE"/old_entry["file"],
            old_entry["feature_sha256"],21_401_604),old_entry["native_tensor_strides"])
        inputs = np.frombuffer(source.pin(capture/q_entry["file"],q_entry["sha256"]),dtype="<f4").copy().reshape(q_entry["shape"])
        query_input = torch.empty_strided(q_entry["shape"],q_entry["stride"]).copy_(torch.from_numpy(inputs))
        check(torch.equal(F.linear(query_input,*initial),projected), "Closed original Linear exact parity")
        large = bank_row["large"]
        source_targets = target_rows[iid]
        ids = bank_row["target_ids"]
        queries = bank_row["query_ids"]
        check(len(ids) == len(source_targets) == len(queries) and sorted(ids.tolist())==list(range(len(ids)))
              and len(queries.unique())==len(ids) and torch.equal(bank_row["native"][large],bank_row["nonlarge_center"][large]),
              "Same all-row IDs and unchanged large labels")
        truth, anchors = [], []
        for row_idx in torch.nonzero(large,as_tuple=True)[0]:
            target = source_targets[int(ids[row_idx])]
            check(target["area"]>=9216 and not target.get("iscrowd",0),"Original matched large annotation")
            rle = mask.merge(mask.frPyObjects(target["segmentation"],800,800))
            truth.append(mask.decode(rle).astype(bool))
            anchors.append(mask.decode(archived["all_rles"][int(queries[row_idx])]).astype(bool))
        views.append(dict(bank_row, spatial=spatial, input=query_input,bias=bias,truth=truth,anchors=anchors))
    check(sum(len(v["truth"]) for v in views)==50 and sum(not v["large"].all() for v in views)==17,"Original cohort")
    return views, initial, source


def point_loss(view, params, labels, stock, large_only=False):
    ids = view["query_ids"]
    batch = torch.zeros_like(ids)
    q = F.linear(view["input"],*params)[batch,ids]
    low = torch.einsum("chw,nc->nhw",view["spatial"][batch[-1]],q)+view["bias"]
    z = stock.point_sample(torch.cat([low]).unsqueeze(1),view["coords"],align_corners=False).squeeze(1)
    if large_only:
        z, labels = z[view["large"]], labels[view["large"]]
    d = view["denominator"]
    return 5*stock.sigmoid_ce_loss_jit(z,labels,d)+5*stock.dice_loss_jit(z,labels,d)


def train(views, initial, labels, stock):
    params = tuple(x.clone().requires_grad_(True) for x in initial)
    opt = torch.optim.AdamW(params,lr=1e-4,weight_decay=1e-4,betas=(.9,.999),eps=1e-8,
                            foreach=False,fused=False)
    losses = []
    for _ in range(4):
        opt.zero_grad(set_to_none=True)
        rows = []
        for view in views:
            loss = point_loss(view,params,view[labels],stock)
            check(torch.isfinite(loss),"Finite full-row stock point loss")
            (loss/len(views)).backward()
            rows.append(float(loss.detach()))
        check(all(p.grad is not None and torch.isfinite(p.grad).all() for p in params),"Finite accumulated Linear gradient")
        opt.step()
        losses.append(mean(rows))
    check(all(int(opt.state[p]["step"])==4 and torch.isfinite(p).all() for p in params),"Four actual finite AdamW updates")
    d = float(torch.linalg.vector_norm(torch.cat([(x-y).detach().reshape(-1) for x,y in zip(params,initial)])))
    check(d>0,"Actual finite parameter movement")
    return tuple(p.detach() for p in params), dict(losses=losses,parameter_motion=d,updates=int(opt.state[params[0]]["step"]))


def score(view,params,readout,stock,check_origin=False):
    q = F.linear(view["input"],*params)
    dense = (torch.einsum("bchw,bnc->bnhw",view["spatial"],q)+view["bias"])[0]
    queries = view["query_ids"][view["large"]]
    resize = F.interpolate(dense[queries,None],size=(800,800),mode="bilinear",align_corners=False)[:,0].numpy()
    by_mode = defaultdict(list)
    for logits,truth,anchor in zip(resize,view["truth"],view["anchors"],strict=True):
        natural = logits>0.
        fixed = readout.top_area(logits,int(anchor.sum()))
        if check_origin:
            check(np.array_equal(natural,anchor) and np.array_equal(fixed,anchor),"Original matched query stock mask")
        for mode,pred in (("fixed",fixed),("stock",natural)):
            by_mode[mode].append(readout.metrics(pred,truth))
    out = {mode:{key:mean(item[key] for item in scores) for key in ("iou","fp_gt","fn_gt")}
           for mode,scores in by_mode.items()}
    out["large_point_loss"] = float(point_loss(view,params,view["native"],stock,large_only=True))
    return out


def aggregate(rows):
    return {mode:{key:mean(item[mode][key] for item in rows) for key in ("iou","fp_gt","fn_gt")}
            for mode in ("fixed","stock")} | {"large_point_loss":mean(item["large_point_loss"] for item in rows)}


def run(args):
    start = perf_counter()
    exact(Path(__file__).resolve().parent/"PROTOCOL.md",args.protocol_sha)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    views, initial, bank_source = load_views(args)
    stock = import_module("rfdetr.models.criterion")
    exact(Path(stock.__file__).resolve(),"fe3a1b05acc7edc4476c03954d75e2b9c91d5612da333782aecd23d0d1cea5c3")
    reader = module(args.root/"github_release_20261011"/READOUT,COUNT_SHA,"_stable_area_reader")
    for count in sorted({int(v["large"].numel()) for v in views}):
        bank_source.warm_stock_loss(torch,stock,count,1296)
    with torch.inference_mode():
        for view in views:
            score(view,initial,reader,stock,check_origin=True)
    folds = []
    for prefix in sorted({v["prefix"] for v in views}):
        hold = [v for v in views if v["prefix"]==prefix]
        training = [v for v in views if v["prefix"]!=prefix and not v["large"].all()]
        a, ah = train(training,initial,"native",stock)
        b, bh = train(training,initial,"nonlarge_center",stock)
        separation = float(torch.linalg.vector_norm(torch.cat([(x-y).reshape(-1) for x,y in zip(a,b)])))
        with torch.inference_mode():
            qa = aggregate([score(v,a,reader,stock) for v in hold])
            qb = aggregate([score(v,b,reader,stock) for v in hold])
        folds.append(dict(train_images=len(training),hold_images=len(hold),large=sum(len(v["truth"]) for v in hold),
                          mixed_hold=any(not v["large"].all() for v in hold),a=qa,b=qb,
                          a_train=ah,b_train=bh,parameter_separation=separation,
                          fixed_iou_delta=100*(qb["fixed"]["iou"]-qa["fixed"]["iou"]),
                          stock_iou_delta=100*(qb["stock"]["iou"]-qa["stock"]["iou"]),
                          large_point_loss_delta=qb["large_point_loss"]-qa["large_point_loss"]))
    active = [f for f in folds if f["mixed_hold"]]
    check(len(folds)==4 and len(active)==3 and not torch.cuda.is_initialized(),"Fixed three mixed plus one pure prefix")
    summary = {k:mean(f[k] for f in active) for k in ("fixed_iou_delta","stock_iou_delta","large_point_loss_delta")}
    lead = summary["fixed_iou_delta"]<=-.25 and all(f["fixed_iou_delta"]<0 for f in active) and summary["large_point_loss_delta"]>0
    out = dict(status="HRSID_Q164_LARGE_PAIRED_LINEAR_UPDATE_ACTUAL", FIT=True, DEV=False, TEST=False, AP=False,
               model_forward_calls=0, optimizer_updates=sum(f["a_train"]["updates"]+f["b_train"]["updates"] for f in folds), images=28,large_targets=50,gate="DIRECTIONAL_LEAD" if lead else "NO_DIRECTIONAL_LEAD",
               folds=folds,summary=summary,protocol_sha256=args.protocol_sha,bank_report_sha256=args.bank_report_sha,
               worker_seconds=perf_counter()-start)
    digest=emit(args.output,out)
    print(json.dumps(dict(status=out["status"],result_sha256=digest,gate=out["gate"])),flush=True)


def main():
    parser=ArgumentParser()
    for name in ("root","output","bank-dir"):
        parser.add_argument("--"+name,type=Path,required=True)
    for name in ("protocol-sha","bank-report-sha","bank-source-sha"):
        parser.add_argument("--"+name,required=True)
    args=parser.parse_args()
    check(os.environ.get("CUDA_VISIBLE_DEVICES")=="-1" and sys.flags.isolated,"Isolated CPU worker")
    def offline(event, _):
        if event.startswith("socket."):
            raise RuntimeError("Closed FIT worker is offline")
    sys.addaudithook(offline)
    run(args)


if __name__=="__main__":
    main()

