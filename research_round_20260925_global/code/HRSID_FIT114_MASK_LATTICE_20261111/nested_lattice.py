"""Test half-pixel-preserving nested bilinear mask lattices on fixed HRSID FIT truth."""
import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
from types import ModuleType

import numpy as np


P322_SHA = "a7d3e80f9d475a37e6122cdbe2d49a4560ec5224ce74975b236b51cf5b5bf60d"
N = 144
METHODS = ("base_144","ordinary_288","nested_x","nested_y","nested_xy","nested_with_coarse_fallback")
WEIGHTS = {"75":(4,3),"90":(10,9)}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def interpolate_axis(values, numerator, denominator, axis):
    base = np.moveaxis(np.asarray(values,dtype=np.float64),axis,0)
    num = np.clip(numerator,0,denominator*(len(base)-1))
    low,rem = np.divmod(num,denominator)
    high = np.minimum(low+1,len(base)-1)
    out = (base[low]*(denominator-rem[:,None])+base[high]*rem[:,None])/denominator
    return np.moveaxis(out,0,axis)


def sample_nested(image, parent, axis):
    """Sample original pixels at nested edge coordinates (i+1)S/(2N)."""
    need(image.ndim==2 and min(image.shape)>0 and parent>=2,"Invalid nesting shape")
    length=image.shape[axis]
    numerator=length*np.arange(1,2*parent,dtype=np.int64)-parent
    return interpolate_axis(image,numerator,2*parent,axis)


def evaluate_axis(values,pixels,axis,parent):
    """Interpolate over the same outer clamp and old physical knot positions."""
    need(values.ndim==2 and values.shape[axis]==2*parent-1 and pixels>0,"Unexpected nested axis")
    numerator=(2*np.arange(pixels,dtype=np.int64)+1)*parent-pixels
    return interpolate_axis(values,numerator,pixels,axis)


def prolongate(grid,axis):
    """Keep all old nodes and insert adjacent means."""
    source=np.moveaxis(grid,axis,0)
    out=np.empty((2*len(source)-1,*source.shape[1:]),dtype=np.float64)
    out[::2]=source
    out[1::2]=(source[:-1]+source[1:])/2
    return np.moveaxis(out,0,axis)


def load(path,pin,name):
    raw=path.read_bytes()
    need(sha256(raw).hexdigest()==pin,"Exact released geometry source required")
    mod=ModuleType(name)
    mod.__file__=str(path)
    exec(compile(raw,str(path),"exec"),mod.__dict__)
    return mod


def evaluate_nested(nodes,hw,x_fine,y_fine,p322):
    x = evaluate_axis(nodes,hw[1],1,N) if x_fine else p322.resize_axis(nodes,hw[1],1)
    return evaluate_axis(x,hw[0],0,N) if y_fine else p322.resize_axis(x,hw[0],0)


def sampled_witnesses(truth,p322):
    hw=truth.shape
    distance=p322.signed_distance(truth)
    parent=p322.resize_bilinear(distance,(N,N))
    base=p322.resize_bilinear(parent,hw)
    embedded=evaluate_nested(prolongate(prolongate(parent,1),0),hw,True,True,p322)
    error=float(np.max(np.abs(base-embedded)))
    need(error <= 1e-10,"Prolonged coarse field differs beyond FP64 tolerance")

    dx=sample_nested(distance,N,1)
    dxy=sample_nested(dx,N,0)
    nx=p322.resize_axis(dx,N,0)
    ny=sample_nested(p322.resize_axis(distance,N,1),N,0)
    need(np.allclose(dxy[::2,::2],parent,rtol=0,atol=1e-10),
         "Original coarse positions are not the even nested nodes")
    fields={
        "base_144":base,
        "ordinary_288":p322.resize_bilinear(p322.resize_bilinear(distance,(288,288)),hw),
        "nested_x":evaluate_nested(nx,hw,True,False,p322),
        "nested_y":evaluate_nested(ny,hw,False,True,p322),
        "nested_xy":evaluate_nested(dxy,hw,True,True,p322),
    }
    values={k:p322.threshold_witnesses(truth,f) for k,f in fields.items()}
    preserved=p322.threshold_witnesses(truth,embedded)
    need(preserved==values["base_144"],"Coarse extension changed the guarded witness counts")
    values["nested_with_coarse_fallback"]={
        level:max((preserved[level],values["nested_xy"][level]),key=lambda x:x[0]/x[1])
        for level in p322.THRESHOLDS
    }
    return values,error


def summary(rows):
    out={}
    for k,(q,p) in WEIGHTS.items():
        s={}
        base=[any(q*i>=p*u for i,u,_ in result["base_144"].values()) for _,result,_ in rows]
        for method in METHODS:
            fields=[result[method] for _,result,_ in rows]
            fits=[any(q*i>=p*u for i,u,_ in v.values()) for v in fields]
            s[method]=dict(witnesses=sum(fits),new_vs_coarse=sum(b and not a for a,b in zip(base,fits,strict=True)),
                lost_vs_coarse=sum(a and not b for a,b in zip(base,fits,strict=True)),
                zero_threshold_witnesses=sum(q*v[0.] [0]>=p*v[0.][1] for v in fields),
                groups_with_new_witness=len({g for (g,_,_),a,b in zip(rows,base,fits,strict=True) if b and not a}),
                coarse_certified_impossible_with_witness=sum(fit and cert[k] for fit,(_,_,cert) in zip(fits,rows,strict=True)))
        out[k]=s
    return dict(instances=len(rows),source_groups=len({g for g,_,_ in rows}),thresholds=out)


def run(a):
    need(os.environ.get("CUDA_VISIBLE_DEVICES")=="-1" and os.environ.get("OMP_NUM_THREADS")=="1",
         "CPU-only fixed FIT geometry required")
    p322=load(a.p322_source,P322_SHA,"_p361_exact_witness")
    bound=p322.load_pinned(a.p320_source,p322.P320_SHA256,"_p361_exact_bound")
    p317=p322.load_pinned(a.p317_source,bound.P317_SHA256,"_p361_exact_fit_scope")
    cohort,index=p317.load_scope(a.bundle)
    roles,groups=p317.split_roles(cohort)
    need(p317.split_digest(roles,groups)==p317.ROLE_SHA,"Exact original FIT role digest required")
    truth=p317.ground_truth(index,cohort)
    core=[row["image_id"] for row in cohort if row["core"]]
    errors=[]
    strata={name:[] for name in ("small","medium","large")}
    all_rows=[]
    for ordinal,iid in enumerate(core):
        for rle in truth[iid]:
            gt=p317.codec.decode(rle).astype(bool,copy=False)
            size=int(gt.sum())
            need(gt.shape==(800,800) and size==int(p317.codec.area(rle)) and size>0,
                 "Exact original FIT mask required")
            values,error=sampled_witnesses(gt,p322)
            cert=bound.cell_cost_bounds(gt,N)
            row=(groups[iid],values,{k:cert[k]>size for k in WEIGHTS})
            errors.append(error)
            strata["small" if size<1024 else "medium" if size<9216 else "large"].append(row)
            all_rows.append(row)
        if (ordinal+1)%25==0 or ordinal+1==len(core):
            print(json.dumps(dict(event="NESTED_LATTICE_FIT_IMAGES_DONE",images=ordinal+1)),flush=True)
    need(len(core)==109 and tuple(map(len,strata.values()))==(447,42,7),"Fixed prior FIT population differs")
    counts={k:summary(v) for k,v in dict(all=all_rows,**strata).items()}
    read=counts["all"]["thresholds"]
    need((read["75"]["base_144"]["witnesses"],read["90"]["base_144"]["witnesses"],
          read["75"]["ordinary_288"]["witnesses"],read["90"]["ordinary_288"]["witnesses"])==(424,228,491,437),
          "Original witness counts differ")
    need(sum(cert["90"] for _,_,cert in all_rows)==3 and all(
        read[key]["nested_with_coarse_fallback"]["lost_vs_coarse"]==0 for key in WEIGHTS),
        "Exact coarse certificates and nested inheritance required")
    return dict(status="HRSID_FIT114_NESTED_LATTICE_ACTUAL",counts=counts,
                preserved_coarse_knots=True,old_outer_clamps_preserved=True,
                max_coarse_extension_field_abs_difference=max(errors),parent=144,refined_axis=287,
                threshold_values=p322.THRESHOLDS,guard=p322.FLOAT_GUARD,core_images=len(core),
                FIT=True,DEV=False,TEST=False,AP=False,model_forward_calls=0,optimizer_updates=0,
                mask_constructions_use_ground_truth=True,threshold_sweep_is_oracle=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("p322-source","p320-source","p317-source","bundle","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    a=parser.parse_args()
    need(a.output.parent.resolve(strict=True)==a.output.parent,"Exact fresh stage required")
    data=(json.dumps(run(a),sort_keys=True,allow_nan=False)+"\n").encode()
    with a.output.open("xb") as output:
        output.write(data)
    print(json.dumps(dict(event="HRSID_FIT114_NESTED_LATTICE_ACTUAL",sha256=sha256(data).hexdigest())),flush=True)


if __name__=="__main__":
    main()
