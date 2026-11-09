"""Same-budget nested image-group OOF census on 484 historical FIT images."""
import argparse
from collections import Counter,defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from types import ModuleType

import numpy as np

ANCHOR_SHA="2751702e3ac30fbd26b02ec893a2df54d70ad18678eb7f4f7caa8c1f44fa16ff"
SLACK_SHA="1286df02c04781f6b82d95c474f4f6ca0dd0db56d7f960a70bbe6754f99bdb06"
RESULT_SHA="317d813bb8d462ff75fa5b868658cfc0c35f3ed66a5113cf9d00054ef4baf488"
SELECTIVE_SHA="4840f5511d4225ef1701e88f0b96da2669149ed529b7c071b4dd908d882b074b"
POLICY_SHA="44c79d7459a59d9dd601d5192c9743ecce815725dd24e1456ba14c70573e0031"
ARMS=("small","large")
FOLDS=5


def pinned(path,digest,name=None):
    raw=path.read_bytes()
    if sha256(raw).hexdigest()!=digest:
        raise ValueError("Exact frozen source or result required")
    if name is None:
        return json.loads(raw)
    m=ModuleType(name)
    m.__file__=str(path)
    sys.modules[name]=m
    exec(compile(raw,str(path),"exec"),m.__dict__)
    return m


def nested_scores(torch,rows,old,folds,prior,base):
    z={k:{} for k in ARMS}
    report=[]
    for f in range(FOLDS):
        held=[(i,r) for i,r in enumerate(rows) if folds[r["group"]]==f]
        train=[r for r in rows if folds[r["group"]]!=f]
        sizes={}
        for arm in ARMS:
            pool=[r for r in train if r["group"] in old] if arm=="small" else train
            x,y=(np.concatenate([r[k] for r in pool]) for k in ("x","y"))
            scorer,loss=base.train_model(torch,x,y,prior.weights(pool),True)
            sizes[arm]=dict(train_rows=len(pool),loss=loss,updates=96)
            for keep_old in (True,False):
                sub=[(i,r) for i,r in held if (r["group"] in old)==keep_old]
                xx=np.concatenate([r["x"] for _,r in sub])
                cuts=np.cumsum([len(r["x"]) for _,r in sub])[:-1]
                for (i,_),v in zip(sub,np.split(scorer(xx),cuts),strict=True):
                    z[arm][i]=v
        report.append(dict(fold=f,old_held_rows=sum(r["group"] in old for _,r in held),
                           added_held_rows=sum(r["group"] not in old for _,r in held),
                           old_held_groups=sum(folds[g]==f for g in old),
                           added_held_groups=sum(n==f and g not in old for g,n in folds.items()),fits=sizes))
    if not all(set(v)==set(range(len(rows))) for v in z.values()):
        raise ValueError("Every row scored only in its group-excluded fold")
    return z,report


def support_rows(env,items,rows,scores,old,folds,anchor,local,policy,slack_math):
    out=[]
    pos=0
    for i,item in enumerate(items,1):
        c=env.image(item)
        _,truth,matches,_=env.matches(item,c)
        group=item["image_sha256"]
        for j,k in matches:
            q=c["query"][j]
            if not bool(c["ctx"]["active"][q]):
                continue
            mask=c["native"][j]
            ymask=truth[k]
            xy,_,_,_,inside=local.boundary_samples(mask,128,5201+100*item["dataset_index"]+q)
            if not len(xy):
                continue
            y=ymask[xy[:,1],xy[:,0]]!=inside
            row=rows[pos]
            if row["group"]!=group or not np.array_equal(row["y"],y) or not np.array_equal(row["inside"],inside):
                raise ValueError("Exact original point-label and group row")
            limit=slack_math.bounds(y,inside)
            hits={}
            if limit is not None:
                px=xy[:,1]*c["hw"][1]+xy[:,0]
                for arm in ARMS:
                    aa,dd=policy.choose_points(px,inside,scores[arm][pos],4)
                    hits[arm]=int(y[aa].sum()+y[dd].sum())
            out.append(dict(group=group,fold=folds[group],old=group in old,limit=limit,
                            slack=anchor.slack(mask,ymask),hits=hits))
            pos+=1
        if i%100==0 or i==len(items):
            print(json.dumps(dict(event="BROAD_AP75_CENSUS_PROGRESS",images=i)),flush=True)
    if pos!=len(rows):
        raise ValueError("Complete original training-label population")
    return out


def census(records,arm,m):
    sums=Counter()
    role=[Counter() for _ in range(FOLDS)]
    names=("native","oof_bce","oof_bce_only")
    vulnerable={k:set() for k in names}
    min_hits={k:Counter() for k in names}
    statuses={k:Counter() for k in names}
    bce_margin=Counter()
    for row in records:
        if row["limit"] is None:
            sums["ineligible_short"]+=1
            continue
        lo,hi=row["limit"]
        h=row["hits"][arm]
        if not lo<=h<=hi:
            raise ValueError("Action within exact feasible support")
        s0=row["slack"]
        r=m.needs_hits(s0)
        native=s0>=0
        bce=m.slack_after(s0,h)>=0
        bce_only=bce and not native
        sums.update(eligible=1,native_pass=int(native),oof_bce_pass=int(bce),oof_bce_only_pass=int(bce_only))
        fold=row["fold"]
        role[fold].update(eligible=1)
        for name,passes in zip(names,(native,bce,bce_only),strict=True):
            if not passes:
                continue
            if r>hi:
                if name!="native":
                    raise ValueError("Realized BCE passes must be feasible")
                statuses[name]["all_actions_fail"]+=1
            elif r<=lo:
                statuses[name]["all_actions_pass"]+=1
            else:
                statuses[name]["choice_matters"]+=1
                vulnerable[name].add(row["group"])
                min_hits[name][r]+=1
                role[fold][name+"_vulnerable"]+=1
                if name=="oof_bce":
                    bce_margin[h-r]+=1
    return dict(totals=sums,role_counts=role,anchor_statuses=statuses,oof_bce_vulnerable_hit_margin=bce_margin,
                vulnerable={k:dict(groups=len(v),cases=sum(min_hits[k].values()),needed_hits=min_hits[k])
                            for k,v in vulnerable.items()})


def pairing(records,factors):
    counts=Counter()
    availability=defaultdict(Counter)
    realized=defaultdict(Counter)
    group_hits={k:Counter() for k in ARMS}
    hard_groups={k:set() for k in ARMS}
    full_groups={k:set() for k in ARMS}
    fold_hits={k:set() for k in ARMS}
    feasible_groups=set()
    for row in records:
        if row["limit"] is None:
            continue
        hits=row["hits"]
        answer=factors.rescue_case(row["slack"],*row["limit"],hits["small"],hits["large"])
        if answer is None:
            continue
        r,possible,actual=answer
        counts.update(native_fail=1,**{possible:1})
        availability[str(r) if r<=8 else "above_eight"][possible]+=1
        if possible!="choice_matters":
            continue
        feasible_groups.add(row["group"])
        realized[str(r)][actual]+=1
        for arm in ARMS:
            if hits[arm]>=r:
                group_hits[arm][row["group"]]+=1
                fold_hits[arm].add(row["fold"])
                if r>=7:
                    hard_groups[arm].add(row["group"])
                if r==8:
                    full_groups[arm].add(row["group"])
    def summary(k):
        g=group_hits[k]
        return dict(cases=sum(g.values()),groups=len(g),folds=len(fold_hits[k]),
                    r7_or_r8_groups=len(hard_groups[k]),r8_groups=len(full_groups[k]),largest_group_cases=max(g.values(),default=0))
    return dict(native_fail_cases=counts,feasible_choice_groups=len(feasible_groups),
                geometric_by_required_hits=dict(availability),paired_realized_by_required_hits=dict(realized),
                vulnerable_bce_only={k:summary(k) for k in ARMS})


def run(args,teacher,base):
    anchor=pinned(args.anchor,ANCHOR_SHA,"_broad_anchor")
    m=pinned(args.slack,SLACK_SHA,"_broad_slack")
    fac=pinned(args.factors,args.factors_sha,"_broad_factors")
    previous=pinned(args.reference,RESULT_SHA)
    prior=pinned(args.local_source,base.LOCAL_SHA,"_broad_prior")
    local=pinned(args.local_math,base.MATH_SHA,"_broad_local")
    policy=pinned(args.policy,POLICY_SHA,"_broad_policy")
    env=teacher.Environment(args)
    try:
        roles=base.parts(env.fit)
        role_groups={k:{r["image_sha256"] for r in v} for k,v in roles.items()}
        old=role_groups["point_fit"]
        added=role_groups["gate_fit"]|role_groups["gate_cal"]
        if ([len(roles[k]) for k in ("point_fit","gate_fit","gate_cal")]!=[290,98,96] or
            [len(role_groups[k]) for k in ("point_fit","gate_fit","gate_cal")]!=[288,96,95] or
            len(env.fit)!=484 or len(added)!=191):
            raise ValueError("Exact closed 484 FIT role after cohort32 exclusion")
        folds=fac.fold_map(old,added)
        rows=base.point_training(env,teacher,prior,local,env.fit)
        old_rows=[r for r in rows if r["group"] in old]
        if len(old_rows)!=623 or sum(len(r["y"]) for r in old_rows)!=142170:
            raise ValueError("Exact original 290-image point training support")
        z,reports=nested_scores(env.torch,rows,old,folds,prior,base)
        for new,orig in zip(reports,previous["folds"],strict=True):
            if (new["fold"],new["old_held_rows"],new["old_held_groups"],new["fits"]["small"]["train_rows"])!=(
                orig["fold"],orig["held_rows"],orig["held_groups"],orig["train_rows"]):
                raise ValueError("Exact P175 group exclusion and fit counts")
        cases=support_rows(env,env.fit,rows,z,old,folds,anchor,local,policy,m)
    finally:
        env.close()
    populations={"old":[r for r in cases if r["old"]],"added":[r for r in cases if not r["old"]],"all":cases}
    summary={k:{a:census(v,a,m) for a in ARMS} for k,v in populations.items()}
    if json.loads(json.dumps(summary["old"]["small"],sort_keys=True))!=previous["train_census"]:
        raise ValueError("Exact closed 290-image small-fit OOF census replay")
    paired={k:pairing(v,fac) for k,v in populations.items()}
    d=paired["all"]["vulnerable_bce_only"]["large"]
    gate=(d["groups"]>=12 and d["folds"]>=4 and d["r7_or_r8_groups"]>=3 and 3*d["largest_group_cases"]<=d["cases"])
    result=dict(status="HISTORICAL_FIT_AP75_BROAD_SUPPORT_COMPLETE",protocol=args.protocol,fold_reports=reports,
                census=summary,paired_rescue=paired,role_images=dict(old=290,added=194,all=484),
                role_groups=dict(old=288,added=191,all=479),training_points=sum(len(r["y"]) for r in rows),
                training_matches=len(rows),exact_p175_small_old_replay=True,support_gate=gate,AP=False,DEV=False,TEST=False)
    args.output.mkdir()
    with (args.output/"RESULT.json").open("x",encoding="utf8") as f:
        json.dump(result,f,sort_keys=True,indent=2,allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps(result,sort_keys=True,allow_nan=False),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("teacher","selective-source"):
        p.add_argument("--"+k,type=Path,required=True)
    first,_=p.parse_known_args()
    base=pinned(first.selective_source,SELECTIVE_SHA,"_broad_selective")
    teacher=base.teacher_module(first.teacher)
    teacher.common_args(p)
    for k in ("local-source","local-math","policy","anchor","slack","factors","reference"):
        p.add_argument("--"+k,type=Path,required=True)
    p.add_argument("--factors-sha",required=True)
    run(p.parse_args(),teacher,base)
