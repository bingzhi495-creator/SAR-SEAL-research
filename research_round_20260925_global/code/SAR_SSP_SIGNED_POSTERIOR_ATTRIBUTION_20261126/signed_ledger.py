"""Account for endpoints in the signed score of two frozen write choices."""
import numpy as np


def audit(p,y,fixed_action,variable_action,area,truth_area,native_hit):
    p,y = np.asarray(p,dtype=np.float64),np.asarray(y,dtype=np.int8)
    n=len(p)
    k=len(fixed_action[0])
    if not (p.shape==y.shape==(n,2) and 0<k<=n<=area and len(fixed_action[1])==k
            and sum(map(len,variable_action))==2*k and np.isfinite(p).all()
            and np.all((0<=p)&(p<=1)) and np.all((y==0)|(y==1))):
        raise ValueError("Well-defined original endpoints, exact equal total edit counts and binary author labels")

    def mask_for(action):
        inside=np.zeros((n,2),dtype=bool)
        inside[:,1]=True
        for side,positions in enumerate(action):
            if len(positions)!=len(set(positions)) or any(j<0 or j>=n for j in positions):
                raise ValueError("One valid selection per original endpoint")
            inside[np.asarray(positions,dtype=np.intp),side] = side==0
        return inside

    fixed=mask_for(fixed_action)
    variable=mask_for(variable_action)
    delta=variable.astype(np.int8)-fixed.astype(np.int8)
    area_delta=int(delta.sum())
    if not np.all(delta.sum(axis=0)==len(variable_action[0])-k):
        raise ValueError("At fixed total budget, both sidewise area contributions are equal")
    result=dict(area_delta=area_delta,native_area=area)
    for side,region in enumerate(("outside","inside")):
        for label,sign in (("plus",1),("minus",-1)):
            changed=delta[:,side]==sign
            key=f"{region}_{label}_"
            result[key+"count"]=int(changed.sum())
            result[key+"p_sum"]=float(p[changed,side].sum())
            result[key+"y_sum"]=int(y[changed,side].sum())

    pred_gain=float(np.sum(delta*p))
    true_gain=int(np.sum(delta*y))
    outside=area-n
    fixed_intersection=native_hit-int(y[:,1].sum())+int(y[fixed].sum())
    fixed_pred=outside+float(p[fixed].sum())
    author_total=truth_area
    predicted_total=outside+float(p.sum())
    true_margin=true_gain-fixed_intersection*area_delta/(area+author_total)
    pred_margin=pred_gain-fixed_pred*area_delta/(area+predicted_total)
    slope_error=(fixed_intersection/(area+author_total)-fixed_pred/(area+predicted_total))*area_delta
    if not (0<=fixed_intersection<=min(area,author_total) and 0<=fixed_intersection+true_gain<=min(area+area_delta,author_total)):
        raise ValueError("Valid original author area and exact intersection")
    result.update(pred_intersection_gain=pred_gain,true_intersection_gain=true_gain,
                  overlay_error=pred_gain-true_gain,slope_error=float(slope_error),
                  score_error=float(pred_margin-true_margin),pred_margin=float(pred_margin),true_margin=float(true_margin),
                  true_iou_delta=float((fixed_intersection+true_gain)/(area+area_delta+author_total-fixed_intersection-true_gain)
                                       -fixed_intersection/(area+author_total-fixed_intersection)),
                  pred_iou_delta=float((fixed_pred+pred_gain)/(area+area_delta+predicted_total-fixed_pred-pred_gain)
                                       -fixed_pred/(area+predicted_total-fixed_pred)))
    if abs(result["score_error"]-result["overlay_error"]-result["slope_error"])>1e-10:
        raise ValueError("Exact additivity of overlap and area-slope error")
    return result
