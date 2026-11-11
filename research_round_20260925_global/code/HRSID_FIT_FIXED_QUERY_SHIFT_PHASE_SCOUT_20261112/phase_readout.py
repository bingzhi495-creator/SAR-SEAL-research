"""Build and score grouped FIT-only local mask correction features."""
from collections import Counter, defaultdict
from dataclasses import dataclass
from statistics import mean

import numpy as np
from pycocotools import mask as coco
from scipy.ndimage import distance_transform_edt, label

MODES=("zero","native","mean","generic","normal","tangent")
FIT_MODES=("generic","normal","tangent")
CONNECTIVITY=np.ones((3,3),dtype=bool)


@dataclass
class Example:
    ordinal: int
    shape: np.ndarray
    points_xy: np.ndarray
    inside: np.ndarray
    target_occupancy: np.ndarray
    features: dict
    gt_area: float
    intersection: float
    budget: int
    neighbor_masks: list


def input_geometry(prediction, points_xy):
    yy,xx=np.nonzero(prediction)
    x,y=points_xy.T
    h,w=prediction.shape
    margin=8
    y0,y1=max(0,min(yy.min(),y.min())-margin),min(h,max(yy.max(),y.max())+margin+1)
    x0,x1=max(0,min(xx.min(),x.min())-margin),min(w,max(xx.max(),x.max())+margin+1)
    window=prediction[y0:y1,x0:x1]
    signed=distance_transform_edt(window)-distance_transform_edt(~window)
    g_y,g_x=np.gradient(signed)
    cy,cx=y-y0,x-x0
    outward=-np.column_stack((g_x[cy,cx],g_y[cy,cx]))
    length=np.linalg.norm(outward,axis=1)
    active=length>0.
    outward[active]/=length[active,None]
    return signed[cy,cx],outward


def example(ordinal,prediction,points,inside,logits,truth,intersection,neighbors,math):
    import shapely
    points=np.asarray(points,dtype=np.int64)
    inside=np.asarray(inside,dtype=bool)
    if points.shape!=(len(inside),2) or len(np.unique(points,axis=0))!=len(points):
        raise ValueError("Exact unique original points required")
    if not np.array_equal(prediction[points[:,1],points[:,0]],inside):
        raise ValueError("Fixed original point side must match RLE")
    x,y=points.T
    boxes=shapely.box(x,y,x+1,y+1)
    occupancy=np.clip(shapely.area(shapely.intersection(boxes,truth)),0,1)
    signed,normal=input_geometry(prediction,points)
    features=math.shift_features(logits,signed,normal)
    encoded=coco.encode(np.asfortranarray(prediction,dtype=np.uint8))
    nearby=coco.iou([encoded],neighbors,[0]*len(neighbors))[0] if neighbors else np.empty(0)
    budget=0 if np.any(nearby>=.20) else min(8,int(prediction.sum())//10,int(inside.sum()),int((~inside).sum()))
    return Example(ordinal,prediction,points,inside,occupancy,features,float(truth.area),intersection,budget,neighbors)


def fit(examples,math):
    per_image=Counter(a.ordinal for a in examples)
    w=np.concatenate([np.full(len(a.inside),1./(len(a.inside)*per_image[a.ordinal])) for a in examples])
    targets=np.concatenate([a.target_occupancy for a in examples])
    return {mode:math.fit_weighted(np.concatenate([a.features[mode] for a in examples]),targets,w) for mode in FIT_MODES}


def contour_metrics(example,intersection):
    predicted=float(example.shape.sum())
    target=example.gt_area
    if not -1e-8<=intersection<=min(predicted,target)+1e-8:
        raise ValueError("Translated footprint has invalid overlap")
    inter=max(0.,min(intersection,predicted,target))
    return dict(iou=100*inter/(predicted+target-inter),
                fp=100*(predicted-inter)/target,
                fn=100*(target-inter)/target)


def score(example,models,math):
    original=contour_metrics(example,example.intersection)
    original.update(component_change=False,new_neighbor_overlap=False,edit_pairs=0)
    out={"zero":original}
    control=dict(native=example.features["native_order"],mean=example.features["mean_order"])
    for mode in FIT_MODES:
        control[mode]=math.predict(example.features[mode],models[mode])
    for mode,pred in control.items():
        if not example.budget:
            out[mode]=original.copy()
            continue
        add,remove=math.edits(pred,example.inside,example.budget)
        diff=example.target_occupancy[add].sum()-example.target_occupancy[remove].sum()
        new=example.shape.copy()
        p,q=example.points_xy[add],example.points_xy[remove]
        new[p[:,1],p[:,0]]=True
        new[q[:,1],q[:,0]]=False
        if new.sum()!=example.shape.sum() or np.count_nonzero(new^example.shape)!=2*example.budget:
            raise ValueError("Exact matched area and edit count required")
        changed=label(new,CONNECTIVITY)[1]!=label(example.shape,CONNECTIVITY)[1]
        neighbor=False
        if example.neighbor_masks:
            rle=coco.encode(np.asfortranarray(new,dtype=np.uint8))
            neighbor=bool(np.any(coco.iou([rle],example.neighbor_masks,[0]*len(example.neighbor_masks))>=.20))
        out[mode]=contour_metrics(example,example.intersection+diff)
        out[mode].update(component_change=changed,new_neighbor_overlap=neighbor,edit_pairs=example.budget)
    return out


def summarize(rows):
    if not rows:
        raise ValueError("Fixed cohort requires at least one row")
    result={}
    for mode in MODES:
        v=[row[mode] for row in rows]
        result[mode]={key:float(mean(r[key] for r in v)) for key in ("iou","fp","fn")}
        result[mode].update(
            strict=sum(r["iou"]>=90. for r in v),edit_pairs=sum(r["edit_pairs"] for r in v),
            component_changes=sum(r["component_change"] for r in v),
            neighbor_overlaps=sum(r["new_neighbor_overlap"] for r in v))
    return result


def crossfit(examples,math):
    folds=[]
    pooled=[]
    for fold in range(3):
        train=[e for e in examples if e.ordinal%3!=fold]
        held=[e for e in examples if e.ordinal%3==fold]
        models=fit(train,math)
        outputs=[score(e,models,math) for e in held]
        folds.append(dict(train=len(train),held=len(held),metrics=summarize(outputs)))
        pooled.extend(outputs)
    metrics=summarize(pooled)
    gain={m:metrics["normal"]["iou"]-metrics[m]["iou"] for m in ("zero","generic","tangent")}
    by_fold={m:sum(f["metrics"]["normal"]["iou"]>f["metrics"][m]["iou"] for f in folds) for m in ("generic","tangent")}
    lead=(gain["zero"]>=.25 and min(gain["generic"],gain["tangent"])>=.10
          and min(by_fold.values())>=2 and metrics["normal"]["strict"]>=metrics["zero"]["strict"])
    return dict(rows=len(examples),metrics=metrics,folds=folds,normal_gain=gain,
                positive_folds=by_fold,gate="DIRECTIONAL_LEAD" if lead else "NO_DIRECTIONAL_LEAD")

