"""Build and validate bounded teacher targets on the exact SSDD original raster."""
import argparse
from collections import Counter, defaultdict
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
GEOMETRY_SHA="a1e9501251456e368e601fe78ed154070388166b4f62de490f785659e4a62ef7"
SOLVER_SHA="04016d572bb2675278554ea54ffa4b6171815cc40831999e2615e08b1bf1d115"
SEED=5201


def module(path,digest,name):
    raw=path.read_bytes()
    if sha256(raw).hexdigest()!=digest:
        raise ValueError("Exact published dependency digest")
    mod=ModuleType(name)
    mod.__file__=str(path)
    sys.modules[name]=mod
    exec(compile(raw,str(path),"exec"),mod.__dict__)
    return mod


class Environment:
    def __init__(self,args):
        s=module(args.scout,SCOUT_SHA,"_closed_coefficient_scout")
        cap=s.module(args.capture_source,s.CAPTURE_SHA,"_closed_stock_capture")
        core=s.module(args.core,s.CORE_SHA,"_closed_native_candidate_core")
        ema=s.module(args.ema,s.EMA_SHA,"_closed_ema_loader")
        self.head=s.module(args.head,s.HEAD_SHA,"_scout_coefficient_head")
        self.geometry=module(Path(__file__).with_name("teacher_geometry.py"),GEOMETRY_SHA,"_teacher_geometry")
        sys.addaudithook(cap.offline)
        import torch
        from pycocotools import mask as codec
        from pycocotools.coco import COCO
        from rfdetr.datasets.coco import make_coco_transforms_square_div_64
        from rfdetr.datasets.io_utils import decode_pil_image_bytes
        from rfdetr.models.postprocess import PostProcess

        s.need(sys.version_info[:3]==(3,12,14) and os.environ.get("CUDA_VISIBLE_DEVICES")=="0"
             and os.environ.get("HF_HUB_OFFLINE")==os.environ.get("TRANSFORMERS_OFFLINE")=="1"
             and all(version(k)==v for k,v in cap.VERSIONS.items())
             and all(sha256(Path(import_module(k).__file__).read_bytes()).hexdigest()==v
                     for k,v in cap.NATIVE_SOURCE.items()),"Exact pinned DGX runtime")
        s.need(torch.cuda.is_available() and torch.cuda.device_count()==1,"One DGX CUDA device")
        torch.cuda.set_per_process_memory_fraction(min(.35,cap.CAP/torch.cuda.get_device_properties(0).total_memory))
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.set_float32_matmul_precision("high")
        torch.manual_seed(SEED)
        s.need(args.output.parent.resolve(strict=True)==args.output.parent and not args.output.exists(),
               "Fresh private output")
        rows=json.loads(cap.exact_bytes(args.fit_index,348469,cap.FIT_INDEX_SHA))["items"]
        blocked={x["image_sha256"] for x,_ in cap.view_plan(rows,"cohort32")}
        self.fit=[x for x in rows if x["partition"]=="head_fit" and x["image_sha256"] not in blocked]
        self.hold=[x for x in rows if x["partition"]=="head_holdout" and x["image_sha256"] not in blocked]
        groups=lambda v:{x["image_sha256"] for x in v}
        s.need((len(blocked),len(self.fit),len(self.hold),len(groups(self.fit)),len(groups(self.hold)))
                ==(32,484,111,479,110) and not groups(self.fit)&groups(self.hold),"Exact original scout cohort")
        self.bundle=ZipFile(BytesIO(cap.exact_bytes(args.fit_bundle,33079338,cap.FIT_BUNDLE_SHA)))
        gt=self.bundle.read("annotations/fit.json")
        s.need(len(gt)==498760 and sha256(gt).hexdigest()==cap.FIT_GT_SHA,"Exact original-grid FIT GT")
        self.gt=json.loads(gt)
        self.gt.setdefault("info",{})
        self.coco=COCO()
        self.coco.dataset=self.gt
        with redirect_stdout(StringIO()):
            self.coco.createIndex()
        self.detector,self.initial=s.native_detector(args,cap,ema,torch)
        tf=make_coco_transforms_square_div_64("val",576,multi_scale=False,expanded_scales=False,
            skip_random_resize=True,patch_size=12,num_windows=1,aug_config=None,scale_jitter=True,gpu_postprocess=False)
        self.infer=s.NativeForward(self.detector,torch,tf)
        self.top=core.make_native_top100(PostProcess)
        self.torch,self.s,self.cap,self.core,self.codec,self.decode=torch,s,cap,core,codec,decode_pil_image_bytes

    def image(self,row):
        torch,s=self.torch,self.s
        encoded=self.bundle.read(f'images/{row["image_sha256"]}.jpg')
        s.need(len(encoded)==row["image_bytes"] and sha256(encoded).hexdigest()==row["image_sha256"],
               "Exact original FIT JPEG")
        hw=tuple(row["original_hw"])
        image,scale=self.decode(encoded,draft_size=None)
        s.need(image.mode=="RGB" and image.size==(hw[1],hw[0]) and scale==(1.,1.),"Exact original size")
        raw,f,q=self.infer(image)
        z=raw["pred_masks"][0]
        with torch.no_grad():
            masks,_,identities,scores=self.core._read_stock(raw,hw,self.top,torch)
            keep=[j for j,(_,slot) in enumerate(identities) if slot==0]
            query=[identities[j][0] for j in keep]
            native=masks[keep,0].cpu().numpy()
            ctx=self.head.candidate_context(f,q,z)
        return dict(hw=hw,basis=f,q=q,z=z,ctx=ctx,query=query,native=native,
                    rank=[j+1 for j in keep],scores=[scores[j] for j in keep])

    def matches(self,row,c):
        iid=row["image_id"]
        anns=[a for a in self.coco.imgToAnns[iid] if not a.get("iscrowd",0)]
        truth=[self.coco.annToRLE(a) for a in anns]
        rles=self.s.coerce_rles(c["native"],self.codec)
        iou=self.codec.iou(rles,truth,[0]*len(truth)) if rles and truth else np.zeros((len(rles),len(truth)))
        pairs=self.s.greedy(iou)
        bits=self.codec.decode(truth).transpose(2,0,1).astype(bool) if truth else np.empty((0,*c["hw"]),bool)
        return anns,bits,pairs,rles

    def close(self):
        self.infer.close()
        self.bundle.close()
        self.s.need(self.cap.state_digest(self.detector.state_dict())==self.initial,"Frozen EMA unchanged")


def pixel_grid(index,hw,torch,device):
    h,w=hw
    y,x=np.divmod(index,w)
    coordinates=np.column_stack(((x+.5)/w,(y+.5)/h)).astype(np.float32)
    return torch.from_numpy(coordinates).to(device)*2-1


def sample_points(planes,index,hw,torch):
    """Sample one feature or logit bank on the admitted original-grid centers."""
    from torch.nn.functional import grid_sample
    points=pixel_grid(index,hw,torch,planes.device)[None,None]
    return grid_sample(planes[None],points,mode="bilinear",padding_mode="border",
                       align_corners=False)[0,:,0].T


def render(planes,hw):
    from torch.nn.functional import interpolate
    return interpolate(planes[:,None],size=hw,mode="bilinear",align_corners=False)[:,0]


def paired_rates(values):
    grouped=defaultdict(list)
    for key,a,b in values:
        grouped[key].append((a,b))
    rates=np.array([np.mean(grouped[k],axis=0) for k in sorted(grouped)])
    d=rates[:,0]-rates[:,1]
    boot=np.random.default_rng(SEED).integers(0,len(d),size=(2048,len(d)))
    return dict(images=len(d),coeff=float(rates[:,0].mean()),offset=float(rates[:,1].mean()),
                difference=float(d.mean()),percentile95=np.quantile(d[boot].mean(1),[.025,.975]).tolist())


def build_image(env,item,solver):
    t=env.torch
    c=env.image(item)
    anns,truth,matches,_=env.matches(item,c)
    z,q,f,ctx,hw=(c[k] for k in ("z","q","basis","ctx","hw"))
    pairs=[(j,k) for j,k in matches if bool(ctx["active"][c["query"][j]])]
    totals=Counter(matched=len(matches),active_matched=len(pairs),suppressed_support=len(matches)-len(pairs))
    if not pairs:
        return [],totals
    staged=[]
    f_flat=f.flatten(1).T.detach().cpu().numpy().astype(np.float64)
    global_gram=f_flat.T@f_flat/len(f_flat)
    proposed=t.zeros_like(q)
    offsets=t.zeros((len(q),1,1),dtype=z.dtype,device=z.device)
    for pos,k in pairs:
        qi=c["query"][pos]
        native=c["native"][pos]
        gt=truth[k]
        stock=env.geometry.signed_counts(native,gt)
        support=env.geometry.sampling_support(gt,native,SEED+item["dataset_index"]*100+qi)
        idx=support["index"]
        label=gt.ravel()[idx]
        weights=support["weights"]
        totals.update(already_strict=stock["strict"],fit_points=len(idx),small=int(stock["area"]<1024))
        kept_held=bool((gt & support["held"]).any())
        delta=np.zeros(256,np.float32)
        if not stock["strict"]:
            local=ctx["local"][qi].flatten().cpu().numpy()
            q0=q[qi].detach().cpu().numpy().astype(np.float64)
            local_f=f_flat[local]
            radius=float(ctx["radius"][qi])
            metric=(global_gram+local_f.T@local_f/len(local_f))*(.5/radius**2)+np.eye(256)/max(q0@q0,1e-12)
            design=sample_points(f,idx,hw,t).cpu().numpy().astype(np.float64)
            base=sample_points(z[qi:qi+1],idx,hw,t)[:,0].cpu().numpy().astype(np.float64)
            fitted=solver.solve_bounded(design,metric,base,label.astype(np.float64),weights)
            scalar=solver.solve_bounded(np.ones((len(idx),1)),np.array([[1/radius**2]]),base,
                                        label.astype(np.float64),weights)
            delta=fitted.iterative.astype(np.float32)
            proposed[qi]=t.from_numpy(delta).to(device=q.device)
            offsets[qi]=float(scalar.iterative[0])
            totals.update(non_strict=1,small_non_strict=int(stock["area"]<1024),
                          projection_stall=int(fitted.projected_gradient>.01))
        staged.append((pos,k,qi,stock,kept_held,support,delta))
    with t.no_grad():
        coeff=env.head.bounded_readout(f,q,z,ctx,proposed)
        queries=[x[2] for x in staged]
        generated=[render(planes[queries],hw).gt(0).cpu().numpy() for planes in (z,coeff,z+offsets)]
    rows=[]
    for i,(pos,k,qi,stock,held_gt,support,delta) in enumerate(staged):
        original,cf,off=[v[i] for v in generated]
        gt=truth[k]
        env.s.need(np.array_equal(original,c["native"][pos]),"Exact original stock mask sign")
        held=support["held"]
        raw=[]
        for name,v in (("coeff",cf),("offset",off)):
            m=env.geometry.signed_counts(v,gt)
            a=env.geometry.signed_counts(v[held],gt[held])["error"]-env.geometry.signed_counts(
                    original[held],gt[held])["error"]
            keep=held_gt and env.geometry.accepted(stock,m,a,stock["area"])
            raw.append((keep,m))
            if not stock["strict"]:
                totals.update({f"{name}_raw_crossings":m["strict"],f"{name}_raw_fp_delta":m["fp"]-stock["fp"],
                    f"{name}_raw_fn_delta":m["fn"]-stock["fn"],f"{name}_accepted":int(keep),
                    f"{name}_verified_crossings":int(keep)*m["strict"],
                    f"{name}_accepted_small":int(keep and stock["area"]<1024),
                    f"{name}_verified_crossings_small":int(keep and stock["area"]<1024)*m["strict"]})
        keep,_=raw[0]
        idx=support["index"]
        rows.append(dict(query=qi,native_rank=c["rank"][pos],annotation_id=anns[k]["id"],
                         stock_mask_sha256=sha256(original.tobytes()).hexdigest(),keep=bool(keep),
                         strict_before=stock["strict"],index=t.from_numpy(idx),labels=t.from_numpy(gt.ravel()[idx]),
                         weights=t.from_numpy(support["weights"].astype(np.float32)),
                         delta=t.from_numpy(delta if keep else np.zeros_like(delta))))
    return rows,totals


def run(args):
    started=time.perf_counter()
    solver=module(args.solver,SOLVER_SHA,"_pinned_bounded_solver")
    env=Environment(args)
    if version("scipy")!="1.15.3":
        raise ValueError("Exact SciPy numerical runtime")
    t=env.torch
    archive=[]
    totals=Counter()
    rates=[]
    accepted_groups=set()
    for i,item in enumerate(env.fit,1):
        rows,counts=build_image(env,item,solver)
        archive.append(dict(image_id=item["image_id"],dataset_index=item["dataset_index"],rows=rows))
        totals.update(counts)
        if counts["non_strict"]:
            rates.append((item["image_sha256"],counts["coeff_verified_crossings"]/counts["non_strict"],
                          counts["offset_verified_crossings"]/counts["non_strict"]))
        if counts["coeff_accepted"]:
            accepted_groups.add(item["image_sha256"])
        if i%50==0 or i==len(env.fit):
            print(json.dumps(dict(event="ORIGINAL_GRID_TEACHER_PROGRESS",images=i,accepted=totals["coeff_accepted"],
                                  verified_crossings=totals["coeff_verified_crossings"])),flush=True)
    env.close()
    comparison=paired_rates(rates)
    gate=(len(accepted_groups)>=60 and totals["coeff_accepted"]>=100 and totals["coeff_accepted_small"]>=30
          and comparison["difference"]>=.10 and comparison["percentile95"][0]>0)
    args.output.mkdir()
    with (args.output/"TEACHERS_PRIVATE.pt").open("xb") as f:
        t.save(dict(protocol=args.protocol,records=archive,gate=gate),f)
        f.flush()
        os.fsync(f.fileno())
    with (args.output/"TEACHERS_PRIVATE.pt").open("rb") as f:
        digest=env.cap.file_digest(f)
    result=dict(status="SSDD_FIT_ORIGINAL_GRID_TEACHER_COMPLETE",protocol=args.protocol,
                FIT_only=True,DEV=False,TEST=False,AP=False,optimizer_updates=0,teacher_gate=gate,
                images=len(archive),holdout_images_used=0,accepted_image_groups=len(accepted_groups),
                totals=dict(totals),paired=comparison,teacher_archive_sha256=digest,
                seconds=time.perf_counter()-started,cuda_peak_bytes=t.cuda.max_memory_allocated())
    with (args.output/"RESULT.json").open("x",encoding="utf-8") as f:
        json.dump(result,f,indent=2,sort_keys=True,allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps(result,sort_keys=True),flush=True)


def common_args(p):
    for n in ("capture-source","scout","head","core","ema","fit-index","fit-bundle","archive","recipe","output"):
        p.add_argument("--"+n,type=Path,required=True)
    p.add_argument("--protocol",required=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    common_args(p)
    p.add_argument("--solver",type=Path,required=True)
    run(p.parse_args())
