"""Compare the four pairings of SRNet's crop grid and inverse paste geometry."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType

P338_SHA="2809a86664716dc8193184534eccb986ee8763e7fe4166f39dea0fe324969cad"
P338_REPORT_SHA="83f0fab1106bc73782d13e49600cfb692d4c6432f4083a9abe6d245bd5b66ae8"
STATUS="SRNET_FIT28_CROP_PASTE_FACTORIAL_COMPLETE"
ARMS=("interior_full","center_full","interior_contracted","center_contracted")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def exact(path,pin):
    raw=path.read_bytes()
    require(0<len(raw)<=1024**2 and sha256(raw).hexdigest()==pin,"Exact closed P338 evidence required")
    return raw


def four_paths(box, truth, stock, torch, oracle_metric):
    import torch.nn.functional as f

    n=28
    height,width=truth.shape
    side=max(height,width)
    canvas=torch.zeros((1,1,side,side),dtype=torch.float32)
    canvas[0,0,:height,:width]=torch.from_numpy(truth)
    roi=torch.as_tensor(box[None],dtype=torch.float32)
    contracted=roi.clone()
    contracted[:,2:4]*=n/(n+1)
    meta=dict(pad_shape=(side,side,3),img_shape=(height,width,3),ori_shape=(height,width,3))

    def crop(points):
        samples=f.grid_sample(canvas,(points/(side/2)-1)[:,None],align_corners=False)
        return (samples.reshape(1,1,n,n)>=.5).to(torch.float32)

    train_points=stock.sample_points_train(roi,"le90",n)
    infer_points=stock.sample_points_test(roi,"le90",n)
    contracted_centers=stock.sample_points_test(contracted,"le90",n)
    error=float((train_points-contracted_centers).abs().max())
    require(error<=1e-4,"Contracted paste bin centers must coincide with the interior crop grid")
    interior=crop(train_points)
    center=crop(infer_points)

    def paste(rois,mask):
        return oracle_metric(stock.get_mask(rois,[0],mask,version="le90",output_size=n,img_meta=meta,
                                            thr=.5,num_classes=1,rescale=False)[0][0],truth)

    return dict(interior_full=paste(roi,interior),center_full=paste(roi,center),
                interior_contracted=paste(contracted,interior),center_contracted=paste(contracted,center),
                matched_center_coordinate_abs_error=error,sampled_binary_nodes_changed=int(torch.count_nonzero(interior!=center)))


def run(a):
    started=perf_counter()
    raw=exact(a.p338_source,P338_SHA)
    prior=json.loads(exact(a.p338_report,P338_REPORT_SHA))
    p=ModuleType("_p341_p338")
    p.__file__=str(a.p338_source)
    exec(compile(raw,str(a.p338_source),"exec"),p.__dict__)
    require(prior["status"]==p.STATUS and prior["oracle_targets"]==20,"Closed fixed original oracle cohort required")
    original=p.oracle_pair
    records=[]

    def observe(box,truth,n,stock,torch):
        if n!=28:
            return original(box,truth,n,stock,torch)
        res=four_paths(box,truth,stock,torch,p.oracle_metric)
        records.append(res)
        return dict(train=res["interior_full"],test=res["center_full"],
                    sampled_binary_nodes_changed=res["sampled_binary_nodes_changed"])

    p.oracle_pair=observe
    replay=p.run(a)
    require({k:v for k,v in replay.items() if k!="elapsed_seconds"}==
            {k:v for k,v in prior.items() if k!="elapsed_seconds"} and len(records)==20,
            "Original P338 stock-paste results must reproduce exactly")
    metrics={}
    for arm in ARMS:
        m=[row[arm] for row in records]
        metrics[arm]=dict(mean_iou=sum(x["iou"] for x in m)/len(m),
                          at_75=sum(x["strict_075"] for x in m),at_90=sum(x["strict_090"] for x in m))
    comparisons={}
    for first,second in (("center_full","interior_full"),("interior_contracted","interior_full"),
                         ("center_full","interior_contracted"),("center_full","center_contracted")):
        diff=[r[first]["iou"]-r[second]["iou"] for r in records]
        comparisons[first+"_minus_"+second]=dict(mean=sum(diff)/len(diff),
            wins=sum(x>0 for x in diff),ties=sum(x==0 for x in diff),losses=sum(x<0 for x in diff))
    return dict(status=STATUS,fit=True,dev=False,test=False,ap=False,model_forward_calls=0,optimizer_updates=0,
                author_commit=prior["author_commit"],original_p338_report_sha256=P338_REPORT_SHA,
                prior_stock_paste_full_replay_exact=True,images=8,targets=20,grid_size=28,
                matched_inverse_scale=28/29,max_abs_coordinate_error=max(x["matched_center_coordinate_abs_error"] for x in records),
                arms=metrics,paired_differences=comparisons,elapsed_seconds=perf_counter()-started)


def deny_network(event,_):
    if event.startswith("socket."):
        raise PermissionError("offline fixed FIT geometry")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("p338-source","p338-report","p335-source","p335-result","p335-terminal","source","fit","archive","output"):
        p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args()
    require(a.output.is_absolute() and a.output.parent.resolve(strict=True)==a.output.parent
            and not a.output.exists() and os.environ.get("CUDA_VISIBLE_DEVICES")=="-1",
            "Fresh canonical CPU geometry output required")
    sys.addaudithook(deny_network)
    raw=(json.dumps(run(a),sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    with a.output.open("xb") as out:
        out.write(raw)
        out.flush()
        os.fsync(out.fileno())
    print(json.dumps(dict(status=STATUS,result_sha256=sha256(raw).hexdigest())),flush=True)


if __name__=="__main__":
    main()
