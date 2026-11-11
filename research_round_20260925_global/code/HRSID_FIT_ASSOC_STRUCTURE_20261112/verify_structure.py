from collections import Counter, defaultdict, deque
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

base=Path(sys.argv[1])
sys.path.insert(0,str(base/'HRSID_FIT_SEED_CONSENSUS_VERIFY_DRAFT_20261112_01'))
from verify_coverage import mask_matrix

stage=base/'HRSID_FIT_ASSOC_STRUCTURE_ACTUAL_20261112_01'
capture=base/'HRSID_FIT_SEED_CONSENSUS_ACTUAL_20261112_01'
seeds=(5201,5202,5203)
donors=(5201,5203)
def verified(path, expected):
    raw=path.read_bytes()
    assert sha256(raw).hexdigest()==expected, path.name
    return raw
result_sha='ade28b44d1ef5701e8986be93624663bfd054e60f75b6eef1cef8c1acd6199ad'
result=json.loads(verified(stage/'RESULT_SAFE.json',result_sha))
original=json.loads(verified(capture/'RESULT_SAFE.json',result['capture_sha256']))
arms={int(a['seed']):[json.loads(line) for line in verified(capture/str(a['seed'])/'CANDIDATES_PRIVATE.jsonl',a['prediction_sha256']).splitlines()] for a in original['arms']}
witness=[json.loads(line) for line in verified(stage/'WITNESS_PRIVATE.jsonl',result['witness_sha256']).splitlines()]
assert len(witness)==36
assert result['FIT'] is True and all(result[k] is False for k in ['AP','DEV','TEST','GT_opened'])
assert all(result[k]==0 for k in ['model_forwards','optimizer_updates'])

def overlaps(left, right):
    a,aa=left
    b,bb=right
    i=(a @ b.T).toarray()
    return i/(aa[:,None]+bb[None,:]-i)

def representative_ids(all_ranks,same,threshold):
    out={}
    available=list(all_ranks)
    while available:
        anchor=available[0]
        hit=[other for other in available if same[anchor,other] >= threshold]
        out.update(dict.fromkeys(hit,anchor))
        hit=set(hit)
        available=[other for other in available if other not in hit]
    return sorted(a for a,b in out.items() if a==b),out

def reciprocal(ranks, donor_ranks, score, margin):
    out={}
    if not ranks or not donor_ranks:
        return out
    for a in ranks:
        d=max(donor_ranks,key=lambda j:(score[a,j],-j))
        if score[a,d] < .50:
            continue
        if max(ranks,key=lambda i:(score[i,d],-i)) != a:
            continue
        row_runner=max((score[a,j] for j in donor_ranks if j!=d),default=0.)
        col_runner=max((score[i,d] for i in ranks if i!=a),default=0.)
        if score[a,d]-max(row_runner,col_runner)+1e-12 >= margin:
            out[a]=d
    return out

def greedy(eligible, roster, iou):
    out={}
    unused={s:set(roster[s]) for s in donors}
    for a in eligible:
        picks=[max(unused[s],key=lambda j:(iou[s][a,j],-j),default=None) for s in donors]
        if all(d is not None and iou[s][a,d]>=.50 for s,d in zip(donors,picks,strict=True)):
            out[a]=tuple(picks)
            for s,d in zip(donors,picks,strict=True):
                unused[s].remove(d)
    return out

def certify(eligible, donor_roster, iou, pairs):
    allowed=set()
    source,sink=(-1,0),(-2,0)
    allowed.update((source,(0,d)) for d in donor_roster[5201])
    allowed.update(((3,d),sink) for d in donor_roster[5203])
    for a in eligible:
        allowed.add(((1,a),(2,a)))
        allowed.update(((0,d),(1,a)) for d in donor_roster[5201] if iou[5201][a,d]>=.50)
        allowed.update(((2,a),(3,d)) for d in donor_roster[5203] if iou[5203][a,d]>=.50)
    used=set()
    for a,(l,r) in pairs.items():
        path=(source,(0,l),(1,a),(2,a),(3,r),sink)
        links=set(zip(path,path[1:]))
        assert links<=allowed and not links&used
        used.update(links)
    adjacency=defaultdict(set)
    for a,b in allowed:
        u,v=(b,a) if (a,b) in used else (a,b)
        adjacency[u].add(v)
    seen={source}
    queue=deque([source])
    while queue:
        for x in adjacency[queue.popleft()] - seen:
            seen.add(x)
            queue.append(x)
    assert sink not in seen
    assert sum(a in seen and b not in seen for a,b in allowed)==len(pairs)

summary=defaultdict(Counter)
group_counts=Counter()
nonzero_groups=Counter()
total=Counter()
denom_groups=Counter()
for idx,w in enumerate(witness):
    blocks={s:arms[s][idx] for s in seeds}
    anchor=blocks[5202]
    assert all((b['ordinal'],b['basename'],b['group'],b['rgb_sha256'])==(idx+1,anchor['basename'],anchor['group'],anchor['rgb_sha256']) for b in blocks.values())
    assert w['ordinal']==idx+1 and w['basename']==anchor['basename']
    rows={s:blocks[s]['candidates'] for s in seeds}
    names={s:[row['rank'] for row in rows[s]] for s in seeds}
    idxs={s:{r:i for i,r in enumerate(names[s])} for s in seeds}
    mat={s:mask_matrix(rows[s]) for s in seeds}
    same={s:overlaps(mat[s],mat[s]) for s in seeds}
    iou={s:overlaps(mat[5202],mat[s]) for s in donors}
    raw_rec=None
    counts={}
    for tag,pct in [('raw',None),('nms90',90),('nms85',85)]:
        kept={}
        for s in seeds:
            all_ids=list(range(len(rows[s])))
            if pct is None:
                kept[s]=all_ids
                continue
            kept[s],mapping=representative_ids(all_ids,same[s],pct/100)
            assert {str(names[s][a]):names[s][b] for a,b in mapping.items()}==w['reps'][str(pct)][str(s)]
            counts[f'{tag}_suppressed_{s}']=len(all_ids)-len(kept[s])
        eligible=[i for i in kept[5202] if 32<=rows[5202][i]['area']<1024]
        counts[tag+'_eligible']=len(eligible)
        counts[tag+'_both_edges']=sum(all(any(iou[s][a,d]>=.50 for d in kept[s]) for s in donors) for a in eligible)
        methods={}
        for name,margin in [('reciprocal',.10),('no_margin',0.)]:
            l,r=[reciprocal(kept[5202],kept[s],iou[s],margin) for s in donors]
            methods[name]={a:(l[a],r[a]) for a in eligible if a in l and a in r}
        if pct is None:
            raw_rec=methods['reciprocal']
        else:
            methods['original_on_reps']={a:p for a,p in raw_rec.items() if a in eligible}
        methods['first']=greedy(eligible,kept,iou)
        def saved(name):
            pairs=w['methods'][tag+'_'+name]['pairs']
            return {idxs[5202][int(a)]:(idxs[5201][pair[0]],idxs[5203][pair[1]]) for a,pair in pairs.items()}
        for k,v in methods.items():
            assert saved(k)==v,(idx,tag,k)
        methods['flow']=saved('flow')
        certify(eligible,kept,iou,methods['flow'])
        for key,pairs in methods.items():
            name=tag+'_'+key
            offsets=[sorted((rows[5201][l]['area'],rows[5202][a]['area'],rows[5203][r]['area']))[1]-rows[5202][a]['area'] for a,(l,r) in pairs.items()]
            summary[name].update('add' if d>0 else 'drop' if d<0 else 'zero' for d in offsets)
            group_counts[name]+=bool(offsets)
            nonzero_groups[name]+=any(offsets)
    total.update(counts)
    denom_groups.update(k for k,v in counts.items() if v>0)
assert dict(total)==result['totals'] and dict(denom_groups)==result['denominator_groups']
for name,c in summary.items():
    cc=sum(c.values())
    nc=c['add']+c['drop']
    g=group_counts[name]
    ng=nonzero_groups[name]
    actual=dict(covered=cc,covered_groups=g,nonzero=nc,nonzero_groups=ng,
                offsets={k:c[k] for k in ('add','drop','zero')},old_gate_counts_met=cc>=60 and g>=20 and nc>=30 and ng>=15)
    assert actual==result['methods'][name], (name,actual)
assert set(summary)==set(result['methods'])
print(json.dumps(dict(status='INDEPENDENT_STRUCTURE_VERIFIED',result_sha=result_sha,images=len(witness),minimum_cut_certificates=3*len(witness),methods_checked=len(summary),both_edge_raw=total['raw_both_edges'],both_edge_nms90=total['nms90_both_edges']),sort_keys=True))
