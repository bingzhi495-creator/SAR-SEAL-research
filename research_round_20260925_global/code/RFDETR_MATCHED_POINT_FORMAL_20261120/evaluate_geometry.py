"""Fixed native-match geometry and the preregistered RF point-refinement gate.

Pure CPU array/scalar arithmetic. No matching, labels, RLE, models, files or COCO.
Caller supplies the complete original area-eligible candidate ledger, including
valid matches without point support. Such identities remain unchanged in means.
"""
from collections import Counter
from math import isfinite
import numpy as np

ARMS = ('N','SF','SU=F','SU','LF=S','LU=S','LW=S','NP=S')
SEEDS = (5217,5218,5219)
TRANSITIONS = ('clean_retained','edited_retained','lost','new','stable_miss')
DRAW_COUNT = 4096
BOOTSTRAP_SEED = 5217


def need(ok, message):
    if not ok:
        raise ValueError(message)


def scalar(value):
    need(isinstance(value,(int,float,np.integer,np.floating)) and not isinstance(value,(bool,np.bool_)),
         'Numeric scalar, not a boolean')
    result=float(value)
    need(isfinite(result),'Finite scalar')
    return result


def identity(row):
    result=tuple(row[k] for k in ('image_id','rank','query_id'))
    need(all(type(v) is int for v in result) and result[1]>0 and result[2]>=0,
         'Original integer image/rank/query identity')
    need(type(row['group']) is str and bool(row['group']),'Fixed source prefix')
    return result


def point_geometry(row):
    coords=np.asarray(row['coordinates'])
    need(coords.ndim==2 and coords.shape[1]==2 and coords.dtype.kind in 'iu'
         and len(coords)<=32 and np.all(coords>=0) and np.unique(coords).size==coords.size,
         'Disjoint native add/drop indices, at most 32 pairs')
    return coords


def sparse_edit(row,edit):
    coords=point_geometry(row)
    indices=np.asarray(edit['flipped'])
    if indices.size==0:
        indices=np.empty(0,dtype=np.int64)
    need(indices.ndim==1 and indices.dtype.kind in 'iu' and np.unique(indices).size==indices.size,
         'Unique sparse native edits')
    need(np.isin(indices,coords.reshape(-1)).all(),'Edits restricted to sealed point support')
    add=np.isin(coords[:,0],indices)
    drop=np.isin(coords[:,1],indices)
    nadd,ndrop=int(add.sum()),int(drop.sum())
    need(edit['edits']==indices.size==nadd+ndrop and edit['adds']==nadd and edit['drops']==ndrop
         and edit['area_delta']==nadd-ndrop,'Sparse count/area metadata consistent')
    return add,drop,nadd,ndrop


def strict_transition(before,after,edits,threshold):
    """Strict > threshold, not COCO's >= threshold matching convention."""
    old,new=scalar(before)>threshold,scalar(after)>threshold
    if old and new:
        return 'edited_retained' if edits else 'clean_retained'
    return 'lost' if old else 'new' if new else 'stable_miss'


def candidate_geometry(row,edit):
    """I'=I+sum(q_add)-sum(q_drop); P'=P+nadd-ndrop, fixed T and match."""
    identity(row)
    need(row['role']=='valid' and row['matched_gt_id'] is not None,
         'Original .50 nonignored valid fixed match required')
    coords=point_geometry(row)
    q=(np.empty((0,2),dtype=np.float64) if row['truth'] is None and len(coords)==0
       else np.asarray(row['truth'],dtype=np.float64))
    need(q.shape==coords.shape and np.isfinite(q).all() and np.all((q>=0)&(q<=1+1e-12)),
         'Exact finite native unit-cell occupancy')
    original=row['raw_continuous']
    I,P,T=(scalar(original[k]) for k in ('intersection','predicted_area','truth_area'))
    need(32<=P<1024 and P.is_integer() and T>0 and -1e-8<=I<=min(P,T)+1e-8,
         'Original eligible P and valid continuous I/T')
    add,drop,nadd,ndrop=sparse_edit(row,edit)
    need(nadd+ndrop<=2*min(8,int(P)//10,len(coords)), 'Locked total native edit cap')
    updated_I=I+float(q[add,0].sum())-float(q[drop,1].sum())
    updated_P=P+nadd-ndrop
    need(updated_P>=0 and -1e-8<=updated_I<=min(updated_P,T)+1e-8
         and updated_P+T-updated_I>0,'Physical updated intersection and union')
    # Do not clamp/round geometry or replace a changing-area union with original P.
    return dict(I=updated_I,P=updated_P,T=T,iou=updated_I/(updated_P+T-updated_I),
                raw_I=I,raw_P=P,raw_iou=I/(P+T-I),adds=nadd,drops=ndrop,edits=nadd+ndrop,
                area_delta=nadd-ndrop,unsupported_noop=len(coords)==0)


def evaluate_geometry(records, edits_by_seed):
    """Aggregate a fixed eligible-candidate ledger without rematching.

    edits_by_seed[seed][(image_id,rank,query_id)][arm] is a sparse edit dict.
    Every candidate and arm must be explicit, including unsupported no-ops.
    Summary is aggregate-only; private_witness retains fixed match identities.
    Source bootstrap treats seed-averaged paired effects as observations.
    """
    keys=[identity(row) for row in records]
    need(len(keys)==len(set(keys)),'Unique original candidate identities')
    need(set(edits_by_seed)==set(SEEDS),'Exactly the three locked seeds')
    counts=Counter(row['role'] for row in records)
    need(set(counts)<=set(('valid','neutral','unmatched')),'Explicit original match roles')
    valid=[i for i,row in enumerate(records) if row['role']=='valid']
    matches=[(records[i]['image_id'],records[i]['matched_gt_id']) for i in valid]
    need(all(key is not None for _,key in matches) and len(set(matches))==len(matches),
         'One fixed nonignored GT match per eligible candidate identity')
    counts['eligible']=len(records)
    counts['valid_unsupported_noop']=sum(len(point_geometry(records[i]))==0 for i in valid)
    groups=sorted({records[i]['group'] for i in valid})
    group_indices=[np.array([j for j,i in enumerate(valid) if records[i]['group']==group],dtype=np.intp)
                   for group in groups]
    values=np.empty((len(SEEDS),len(valid),len(ARMS)),dtype=np.float64)
    private=[]
    per_seed={}
    for si,seed in enumerate(SEEDS):
        supplied=edits_by_seed[seed]
        need(set(supplied)==set(keys),'Same complete fixed candidate identities for every seed')
        budgets={arm:Counter() for arm in ARMS}
        strict={threshold:{arm:Counter({name:0 for name in TRANSITIONS}) for arm in ARMS}
                for threshold in (.75,.90)}
        vi=0
        for row,key in zip(records,keys):
            edits=supplied[key]
            need(set(edits)==set(ARMS),'All eight arm identities required')
            need(edits['N']['edits']==0,'Native baseline must remain unchanged')
            need(edits['SF']['edits']==edits['SU=F']['edits']
                 and all(edits[a]['edits']==edits['SU']['edits'] for a in ('LF=S','LU=S','LW=S','NP=S')),
                 'Per-candidate realized edit counts exactly matched')
            geometries={}
            for ai,arm in enumerate(ARMS):
                _,_,nadd,ndrop=sparse_edit(row,edits[arm])
                need(arm=='LW=S' or nadd==ndrop,'All non-LW rows conserve original area')
                budget=budgets[arm]
                budget.update(edits=nadd+ndrop,adds=nadd,drops=ndrop,area_delta=nadd-ndrop,
                              edited=int(nadd+ndrop>0),no_op=int(nadd+ndrop==0),
                              edited_valid=int(row['role']=='valid' and nadd+ndrop>0),
                              edited_neutral=int(row['role']=='neutral' and nadd+ndrop>0),
                              edited_unmatched=int(row['role']=='unmatched' and nadd+ndrop>0))
                if row['role']!='valid':
                    continue
                geometry=candidate_geometry(row,edits[arm])
                values[si,vi,ai]=geometry['iou']
                geometry['strict75']=strict_transition(geometry['raw_iou'],geometry['iou'],geometry['edits'],.75)
                geometry['strict90']=strict_transition(geometry['raw_iou'],geometry['iou'],geometry['edits'],.90)
                strict[.75][arm][geometry['strict75']]+=1
                strict[.90][arm][geometry['strict90']]+=1
                geometries[arm]=geometry
            if row['role']=='valid':
                private.append(dict(seed=seed,image_id=row['image_id'],rank=row['rank'],query_id=row['query_id'],
                                    group=row['group'],matched_gt_id=row['matched_gt_id'],arms=geometries))
                vi+=1
        metrics={}
        for ai,arm in enumerate(ARMS):
            metrics[arm]=dict(group_equal_iou_points=(float(np.mean([values[si,index,ai].mean()
                                              for index in group_indices]))*100 if groups else None),
                              pair_equal_iou_points=float(values[si,:,ai].mean())*100 if valid else None)
        per_seed[seed]=dict(arms=metrics,budgets={k:dict(v) for k,v in budgets.items()},
                            strict75={k:dict(v) for k,v in strict[.75].items()},
                            strict90={k:dict(v) for k,v in strict[.90].items()})
    seed_average=values.mean(axis=0)
    comparisons={}
    if groups:
        # One common source-index matrix for all paired comparisons. No seed or
        # candidate resampling; source-equal means include unsupported no-ops.
        rng=np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
        draws=rng.integers(len(groups),size=(DRAW_COUNT,len(groups)))
    for arm in ARMS:
        if arm=='SU':
            continue
        if groups:
            effects=seed_average[:,ARMS.index('SU')]-seed_average[:,ARMS.index(arm)]
            grouped=np.array([effects[index].mean() for index in group_indices],dtype=np.float64)*100
            bootstrap=grouped[draws].mean(axis=1)
            comparisons['SU_minus_'+arm]=dict(delta_points=float(grouped.mean()),
                                              interval95_points=np.quantile(bootstrap,[.025,.975]).tolist())
        else:
            comparisons['SU_minus_'+arm]=dict(delta_points=None,interval95_points=None)
    mean_metrics={arm:dict(group_equal_iou_points=(float(np.mean([
        per_seed[s]['arms'][arm]['group_equal_iou_points'] for s in SEEDS])) if groups else None)) for arm in ARMS}
    summary=dict(schema='RFDETR-fixed-match-eight-policy-geometry-v1',seeds=list(SEEDS),arms=list(ARMS),
                 matching_performed=False,required_input_match='original nonignored COCO .50 frozen identities',
                 counts={k:int(counts[k]) for k in ('eligible','valid','neutral','unmatched','valid_unsupported_noop')},
                 source_groups=len(groups),bootstrap_draws=DRAW_COUNT,bootstrap_seed=BOOTSTRAP_SEED,
                 bootstrap_unit='fixed source prefix; seed-averaged paired effects',
                 means=mean_metrics,per_seed=per_seed,comparisons=comparisons)
    return dict(summary=summary,private_witness=private)


def locked_gate(coco_by_seed, geometry_summary):
    """Apply the fixed gate to official COCO fractions and the geometry summary.

    No COCO computation or claim of official-source verification is performed.
    -1/None is undefined, never zero or a reduced-seed mean. LU remains visible
    even when it beats the preselected S primary; it never replaces that primary.
    """
    need(set(coco_by_seed)==set(SEEDS),'Exactly three locked COCO seeds')
    need(geometry_summary['seeds']==list(SEEDS) and geometry_summary['arms']==list(ARMS)
         and geometry_summary['bootstrap_draws']==DRAW_COUNT
         and geometry_summary['bootstrap_seed']==BOOTSTRAP_SEED,
         'Same locked geometry seeds, arms and source bootstrap')
    missing=[]
    normalized={}
    for seed in SEEDS:
        need(set(coco_by_seed[seed])==set(ARMS),'All eight official-score arms required')
        normalized[seed]={}
        for arm in ARMS:
            row=coco_by_seed[seed][arm]
            values={}
            for metric in ('AP','AP75'):
                value=row[metric]
                if value is None or value==-1:
                    values[metric]=None
                    missing.append(dict(seed=seed,arm=arm,metric=metric))
                else:
                    number=scalar(value)
                    need(0<=number<=1,'Official COCO fraction, not percentage')
                    values[metric]=number
            normalized[seed][arm]=values
    means={}
    for arm in ARMS:
        means[arm]={}
        for metric in ('AP','AP75'):
            values=[normalized[seed][arm][metric] for seed in SEEDS]
            means[arm][metric]=(float(np.mean(values)) if all(v is not None for v in values) else None)
    contrasts={}
    for arm in ARMS:
        if arm=='SU':
            continue
        one,two=means['SU']['AP'],means[arm]['AP']
        contrasts['SU_minus_'+arm]=None if one is None or two is None else 100*(one-two)
    gates={'all_eight_COCO_defined':not missing}
    for arm,threshold in (('N',.50),('SF',.30),('LF=S',.30),('LW=S',.30)):
        value=contrasts['SU_minus_'+arm]
        gates['AP_vs_'+arm]=value is not None and value>=threshold
    for arm in ('N','LF=S','LW=S'):
        comparison=geometry_summary['comparisons']['SU_minus_'+arm]
        effect,bounds=comparison['delta_points'],comparison['interval95_points']
        if effect is None or bounds is None:
            gates['geometry_vs_'+arm]=False
        else:
            need(len(bounds)==2 and scalar(bounds[0])<=scalar(bounds[1]),'Finite ordered paired interval')
            gates['geometry_vs_'+arm]=(geometry_summary['source_groups']>0
                                       and scalar(effect)>=.20 and scalar(bounds[0])>0)
    primary,baseline=means['SU']['AP75'],means['N']['AP75']
    gates['AP75_retained']=primary is not None and baseline is not None and primary>=baseline
    gates['positive_each_seed_vs_N']=all(
        normalized[seed]['SU']['AP'] is not None and normalized[seed]['N']['AP'] is not None
        and normalized[seed]['SU']['AP']>normalized[seed]['N']['AP'] for seed in SEEDS)
    return dict(schema='RFDETR-fixed-point-refinement-gate-v1',primary='SU',seeds=list(SEEDS),
                mean_COCO=means,AP_contrasts_points=contrasts,undefined_COCO=missing,
                gates=gates,advance=all(gates.values()),
                scope='fixed recipe development gate; not journal readiness or full PointRend superiority')
