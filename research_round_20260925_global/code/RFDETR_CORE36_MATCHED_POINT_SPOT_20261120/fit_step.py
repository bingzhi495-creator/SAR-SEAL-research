"""Shared source-balanced point fitting for compact and larger residual heads."""
from collections import defaultdict
from contextlib import redirect_stdout
from hashlib import sha256
from io import StringIO
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType, SimpleNamespace

import numpy as np
import torch
from torch import nn

TRAINER_SHA="3111c34afd791cb15987ed94c5e898de4c3ff2cad61677b616420651335d812c"
GEOMETRY_SHA="7cad8b51a9230101e2233db8455f84c28163abfb2ee8ddd8cbe8718a2e719bd0"
MATCHER_SHA="12fd1684b262a3c9b6732b90a0c1e624034c523a7d130919a079e17728f80075"


def need(ok,reason):
    if not ok:
        raise ValueError(reason)


def load(path,digest,name):
    source=path.read_bytes()
    need(sha256(source).hexdigest()==digest,"Exact published source "+path.name)
    module=ModuleType(name)
    module.__file__=str(path)
    exec(compile(source,str(path),"exec"),module.__dict__)
    return module


def load_trainer(path):
    return load(path,TRAINER_SHA,"_original_source_balanced_trainer")


def load_geometry(code,shapely_site):
    sys.path.append(str(shapely_site.resolve(strict=True)))
    import shapely
    need(shapely.__version__=="2.0.2" and shapely.geos_version_string=="3.11.2","Exact unit-cell polygon runtime")
    source=load(code/"MASK_RTDETR_DEV705_CONTINUOUS_AP_20261112/continuous_endpoint.py",
                GEOMETRY_SHA,"_stock_continuous_geometry")
    matcher=load(code/"HRSID_FIT_HULL_CDF_20261112/stock_coco_match.py",MATCHER_SHA,"_stock_native_matcher")
    return SimpleNamespace(modules=source.geometry_modules(code),matcher=matcher,shapely=shapely)


def attach_truth(image,annotations,candidates,records,helpers):
    """Freeze original stock .50 IDs, then attach continuous occupancy to the same sealed points."""
    from pycocotools import mask as codec

    need(all(r["image_id"]==image["id"] for r in records) and
         len({r["rank"] for r in records})==len(records) and
         len({r["rank"] for r in candidates})==len(candidates),
         "Exact original candidate and point identities")
    source=dict(images=[image],categories=[dict(id=1,name="ship")],annotations=annotations)
    modules,shapely=helpers.modules,helpers.shapely
    truth,neutral,_=modules.geometry.polygon_support(source,modules.vector)
    matched={}
    if candidates and annotations:
        with redirect_stdout(StringIO()):
            matched,_,_=helpers.matcher.freeze_coco(image,annotations,candidates)
    by_rank={rank:key for key,rank in matched.items()}
    by_candidate={c["rank"]:c for c in candidates}
    result=[]
    counts=defaultdict(int)
    for row in records:
        need(row["rank"] in by_candidate,"Original mapped native rank")
        out=dict(row,truth=None,matched_gt_id=by_rank.get(row["rank"]),raw_continuous=None)
        identity=out["matched_gt_id"]
        role="valid" if identity in truth and identity not in neutral else "neutral" if identity is not None else "unmatched"
        out["role"]=role
        counts[role]+=1
        if role=="valid":
            cand=by_candidate[row["rank"]]
            coords=np.asarray(row["coordinates"])
            need(coords.ndim==2 and coords.shape[1]==2 and np.issubdtype(coords.dtype,np.integer) and
                 np.all((coords>=0)&(coords<image["width"]*image["height"])),"Exact native point coordinates")
            polygon=truth[identity][0]
            y,x=np.divmod(coords,image["width"])
            q=np.asarray(shapely.area(shapely.intersection(shapely.box(x,y,x+1,y+1),polygon)))
            need(q.shape==coords.shape and np.isfinite(q).all() and (q>=0).all() and (q<=1+1e-12).all(),
                 "Exact continuous fractional occupancy on fixed points")
            out["truth"]=q
            rectangles=modules.geometry.boxes_from_counts(modules.shift.runs(cand["rle"]["counts"]),*cand["rle"]["size"])
            area=cand["area"]
            need(int(codec.area(cand["rle"]))==area and
                 np.prod(rectangles[:,2:]-rectangles[:,:2],axis=1).sum()==area,"Original native mask unit cells")
            overlap=modules.geometry.overlap_area(rectangles,polygon)
            need(-1e-8<=overlap<=min(area,polygon.area)+1e-8,"Exact original continuous intersection")
            out["raw_continuous"]=dict(predicted_area=area,truth_area=polygon.area,intersection=overlap)
            counts["supervised_pairs"]+=len(coords)
        result.append(out)
    return result,dict(stock_matches=len(matched),**dict(counts))


class LargeHead(nn.Module):
    def __init__(self,dimensions=14):
        super().__init__()
        self.network=nn.Sequential(nn.Linear(2*dimensions,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU(),
                                   nn.Linear(256,256),nn.ReLU(),nn.Linear(256,1))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self,own,partner,probabilities):
        residual=self.network(torch.cat((own,partner),dim=-1)).squeeze(-1)
        return torch.logit(probabilities.clamp(1e-4,1-1e-4))+residual


def tensor_hash(model):
    return sha256(b"".join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest()


def prepared(records,trainer):
    need(len({(r["image_id"],r["rank"],r["query_id"]) for r in records})==len(records),"Unique original point records")
    valid=[r for r in records if r["truth"] is not None and len(r["features"])]
    need(valid,"Fixed selection has no supervised point pairs")
    groups=defaultdict(list)
    end=0
    for row in valid:
        f,q=row["features"],row["truth"]
        need(f.ndim==3 and f.shape[1:]==(2,14) and f.dtype==np.float32 and q.shape==f.shape[:2] and
             np.isfinite(f).all() and np.isfinite(q).all() and
             (f[:,:,0]>=0).all() and (f[:,:,0]<=1).all() and (q>=0).all() and (q<=1+1e-12).all(),
             "Exact finite common point features and unit-cell labels")
        n=len(f)
        groups[row["group"]].extend(range(end,end+n))
        end+=n
    group_indices=[np.asarray(groups[k]) for k in sorted(groups)]
    own=np.concatenate([r["features"] for r in valid])
    targets=np.concatenate([r["truth"] for r in valid]).astype(np.float32)
    center,scale=trainer.group_moments(own,group_indices)
    return own,targets,center,scale,group_indices


def predict(model,features,center,scale):
    standard=torch.from_numpy(np.asarray((features-center)/scale,dtype=np.float32))
    with torch.inference_mode():
        result=model(standard,standard,torch.from_numpy(features[:,:,0].copy())).sigmoid().numpy()
    need(result.shape==features.shape[:2] and np.isfinite(result).all(),"Exact native endpoint prediction shape")
    return result


def fit_heads(records,trainer,*,steps=2,batch=64,seed=5217,sample_seed=20261112):
    need(type(steps) is type(batch) is int and steps>0 and batch>0,"Positive fixed point update budget")
    torch.set_num_threads(1)
    started=perf_counter()
    own,targets,center,scale,groups=prepared(records,trainer)
    x=torch.from_numpy(np.asarray((own-center)/scale,dtype=np.float32))
    p=torch.from_numpy(own[:,:,0].copy())
    q=torch.from_numpy(targets)
    rng=np.random.Generator(np.random.PCG64(sample_seed))
    draws=rng.integers(len(groups),size=(steps,batch))
    batches=np.asarray([[groups[g][rng.integers(len(groups[g]))] for g in row] for row in draws],dtype=np.int64)
    audit=dict(shared_pairs=len(own),source_groups=len(groups),steps=steps,batch=batch,
               sampled_pair_rows=steps*batch,sampled_point_rows=2*steps*batch,
               common_batch_sha256=sha256(batches.astype("<i8",copy=False).tobytes()).hexdigest(),heads={})
    models={}
    outputs={}
    for name,constructor,expected in (("S",trainer.PointHead,1321),("L",LargeHead,139265)):
        torch.manual_seed(seed)
        model=constructor(14)
        initial=tensor_hash(model)
        parameters=sum(v.numel() for v in model.parameters())
        need(parameters==expected and torch.count_nonzero(model.network[-1].weight)==0 and
             torch.count_nonzero(model.network[-1].bias)==0,"Fixed architecture and zero residual initialization")
        optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01)
        losses,gradient_norms=[],[]
        for rows in batches:
            optimizer.zero_grad(set_to_none=True)
            logits=model(x[rows],x[rows],p[rows])
            loss=nn.functional.binary_cross_entropy_with_logits(logits,q[rows])
            need(bool(torch.isfinite(loss)),"Finite actual BCE loss")
            loss.backward()
            need(all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters()),
                 "Actual finite gradients of every point-head tensor")
            gradient_norms.append(float(torch.sqrt(sum(v.grad.double().square().sum() for v in model.parameters()))))
            optimizer.step()
            losses.append(float(loss.detach()))
        observed=sorted({int(v["step"].item()) for v in optimizer.state.values()})
        need(observed==[steps] and all(torch.isfinite(v).all() for v in model.parameters()),"Exact actual optimizer steps and finite parameters")
        audit["heads"][name]=dict(parameters=parameters,optimizer_steps=observed,losses=losses,
                                gradient_norms=gradient_norms,initial_tensor_sha256=initial,
                                final_tensor_sha256=tensor_hash(model),finite_parameters=True)
        model.eval()
        models[name]=model
        outputs[name]=[predict(model,r["features"],center,scale) for r in records]
    audit["cpu_fit_and_predict_seconds"]=perf_counter()-started
    return dict(heads=models,center=center,scale=scale,prediction=outputs,audit=audit)
