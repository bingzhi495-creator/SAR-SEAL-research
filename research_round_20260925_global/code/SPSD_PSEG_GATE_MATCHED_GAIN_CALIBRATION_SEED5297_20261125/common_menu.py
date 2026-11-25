"""Label-free own/shared action menus and gain-component diagnostics."""
from base64 import b64decode, b64encode
from collections import defaultdict
from itertools import product

import numpy as np


HEADS = ("regression", "midpoint", "gap")
TOL = 1e-12


def pack(value, dtype):
    return b64encode(np.asarray(value,dtype=dtype).tobytes()).decode("ascii")


def unpack(encoded, dtype, shape):
    return np.frombuffer(b64decode(encoded,validate=True),dtype=dtype).reshape(shape)


def gains(values, actions):
    add=values[actions[:,0],0,:].sum(axis=1)
    drop=values[actions[:,1],1,:].sum(axis=1)
    signed=add-drop
    return signed-signed[0]


def components(deltas):
    return deltas.mean(axis=-1), np.abs(deltas[...,0]-deltas[...,1])*.5


def freeze_menu(coordinates, posteriors, incumbent, choices, k, selector):
    own={}
    for head in HEADS:
        p=np.asarray(posteriors[head],dtype=np.float64)
        add=selector._candidate_sets(p[:,0,:],incumbent[0],k,largest=True)
        drop=selector._candidate_sets(p[:,1,:],incumbent[1],k,largest=False)
        own[head]=list(product(add,drop))
    union=set().union(*own.values())
    menu=[incumbent,*sorted(union-{incumbent})]
    index={action:i for i,action in enumerate(menu)}
    chosen=[index[choices[key+"_robust"]] for key in HEADS]
    actions=np.asarray(menu,dtype=np.uint8)
    estimates=np.stack([gains(np.asarray(posteriors[key],dtype=np.float64),actions) for key in HEADS],axis=1)
    return dict(k=k,n=len(coordinates),actions_count=len(actions),chosen=chosen,
                own_menu_indices=[[index[action] for action in own[key]] for key in HEADS],
                coordinates=pack(coordinates,"<i4"),actions=pack(actions,"u1"),
                predicted_gains=pack(estimates,"<f8"))


def read_menu(doc):
    k,n,m=(doc[key] for key in ("k","n","actions_count"))
    if not 0<k<=n<=32 or not 0<m<=430:
        raise ValueError("Original feasible positive quota and finite union size")
    coords=unpack(doc["coordinates"],"<i4",(n,2))
    actions=unpack(doc["actions"],"u1",(m,2,k))
    pred=unpack(doc["predicted_gains"],"<f8",(m,3,2))
    chosen=doc["chosen"]
    own=doc["own_menu_indices"]
    keys=[(tuple(a),tuple(d)) for a,d in actions]
    if (len(set(coords.ravel().tolist()))!=coords.size or np.any(coords<0) or np.any(actions>=n) or
        np.any(np.diff(actions.astype(np.int16),axis=2)<=0) or
        len(set(keys))!=m or sorted(keys[1:])!=keys[1:] or
        not np.isfinite(pred).all() or not np.all(pred[0]==0.) or np.max(abs(pred))>2*k+TOL or
        len(chosen)!=3 or len(own)!=3 or not all(0<=i<m for i in chosen) or
        not all(0<len(idx)<=144 and idx[0]==0 and len(idx)==len(set(idx)) and
                all(0<=i<m for i in idx) for idx in own) or
        set().union(*map(set,own))!=set(range(m))):
        raise ValueError("Unique feasible ordered menu and exact stable own support")
    for h,idx in enumerate(own):
        mid,gap=components(pred[idx,h])
        if idx[int(np.argmax(mid-gap))]!=chosen[h]:
            raise ValueError("Predicted midpoint minus gap must select each exact original joint action")
    return coords,actions,pred,chosen,own


def measure(values, actions, pred, selected, own, k):
    truth=gains(values,actions)
    truth_mid,truth_gap=components(truth)
    predicted_mid,predicted_gap=components(pred)
    direct=sorted({0,*selected})
    err_m=abs(predicted_mid-truth_mid[:,None])
    err_h=abs(predicted_gap-truth_gap[:,None])
    minimum=truth.min(axis=1)
    outcome=minimum[selected]
    swaps={}
    for label in ("predicted","truth_midpoint","truth_gap","truth_both"):
        changed=[]
        for h,idx in enumerate(own):
            midpoint=truth_mid[idx] if label in ("truth_midpoint","truth_both") else predicted_mid[idx,h]
            gap=truth_gap[idx] if label in ("truth_gap","truth_both") else predicted_gap[idx,h]
            changed.append(idx[int(np.argmax(midpoint-gap))])
        swaps[label]=dict(indices=np.asarray(changed),actual=minimum[changed])
    if not np.array_equal(swaps["predicted"]["indices"],selected):
        raise ValueError("Predicted components reproduce the selected joint actions")
    return dict(direct_actions=len(direct),full_actions=len(actions),direct_midpoint_mae=err_m[direct].mean(axis=0),
                direct_gap_mae=err_h[direct].mean(axis=0), full_midpoint_mae=err_m.mean(axis=0),full_gap_mae=err_h.mean(axis=0),
                selected_minimum_raw=outcome, selected_minimum_normalized=outcome/(2*k),
                nonstatic=np.asarray(selected)!=0,harmful=(np.asarray(selected)!=0)&(outcome<0),
                swaps=swaps,k=k)


def _group_table(rows, get):
    g=defaultdict(lambda:defaultdict(list))
    for iid,group,r in rows:
        g[group][iid].append(get(r))
    return np.stack([np.mean([np.mean(v,axis=0) for v in g[group].values()],axis=0) for group in sorted(g)])


def _paired(table,draws):
    return dict(points=table.mean(axis=0).tolist(),
                percentile95=np.percentile(table[draws].mean(axis=1),[2.5,97.5],axis=0).tolist())


def summarize(rows):
    """Aggregate group-equal paired effects, preserving the common selected-hull scope."""
    groups=len({g for _,g,_ in rows})
    draws=np.random.Generator(np.random.PCG64(5297)).integers(groups,size=(4096,groups))
    report=dict(candidates=len(rows),images=len({i for i,_,_ in rows}),groups=groups,
                total_shared_direct_actions=sum(v["direct_actions"] for _,_,v in rows),
                total_shared_full_actions=sum(v["full_actions"] for _,_,v in rows),head_order=HEADS)
    names=["direct_midpoint_mae","direct_gap_mae","full_midpoint_mae","full_gap_mae",
           "selected_minimum_raw","selected_minimum_normalized"]
    names.extend(key+"_normalized" for key in names[:4])
    for key in names:
        def value(row):
            if key.endswith("_mae_normalized"):
                return row[key.removesuffix("_normalized")]/(2*row["k"])
            return row[key]
        tab=_group_table(rows,value)
        report[key]=dict(head_group_means=tab.mean(axis=0).tolist(),
                         midpoint_minus_regression=_paired(tab[:,1]-tab[:,0],draws),
                         midpoint_minus_gap=_paired(tab[:,1]-tab[:,2],draws))
    for field in ("nonstatic","harmful"):
        report[field+"_hulls"]=np.sum([v[field] for _,_,v in rows],axis=0).tolist()
    swaps={}
    for rule in ("predicted","truth_midpoint","truth_gap","truth_both"):
        actual=_group_table(rows,lambda row:row["swaps"][rule]["actual"])
        delta=_group_table(rows,lambda row:row["swaps"][rule]["actual"]-row["selected_minimum_raw"])
        delta_norm=_group_table(rows,lambda row:(row["swaps"][rule]["actual"]-row["selected_minimum_raw"])/(2*row["k"]))
        v=np.stack([row["swaps"][rule]["actual"] for _,_,row in rows])
        idx=np.stack([row["swaps"][rule]["indices"] for _,_,row in rows])
        swaps[rule]=dict(selected_minimum_raw_group_means=actual.mean(axis=0).tolist(),
                         improvement_over_predicted_raw=_paired(delta,draws),
                         improvement_over_predicted_normalized=_paired(delta_norm,draws),
                         harmful_hulls=(v<0).sum(axis=0).tolist(),positive_hulls=(v>0).sum(axis=0).tolist(),
                         static_choices=(idx==0).sum(axis=0).tolist())
    report["truth_swap"]=swaps
    return report
