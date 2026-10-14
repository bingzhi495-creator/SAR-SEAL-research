"""UNBOUND_DRAFT: pure N17 locked-main interpretation; no IO or qualification.
Caller supplies already qualified E24-atmost AP in explicit percent units.
"""
import math
STRONG_AP=68.09938494745545
SEEDS=(202610150171,202610150172,202610150173)
KEYS={'seed','epoch','mode','AP_unit','LOCAL_AP','CENTER_AP','strong_AP'}

def interpret_fixed_main(results):
    """Interpret an ordered 171/172/173 prefix; secondary results are not inputs."""
    if type(results) not in (list,tuple) or not 1<=len(results)<=3:
        raise ValueError('One to three fixed main summaries required')
    rows=[]
    for index,row in enumerate(results):
        if type(row) is not dict or set(row)!=KEYS:
            raise ValueError('Exact main-only summary keys, including strong reference, required')
        if type(row['seed']) is not int or row['seed']!=SEEDS[index] or type(row['epoch']) is not int or row['epoch']!=24 or row['mode']!='atmost' or row['AP_unit']!='percent':
            raise ValueError('Fixed initialization seed order, E24, atmost and explicit AP percent required')
        for name in ('LOCAL_AP','CENTER_AP','strong_AP'):
            value=row[name]
            if type(value) not in (int,float) or not math.isfinite(value) or not (value==-1 or 0<=value<=100):
                raise ValueError('Finite AP percent0..100 or -1 undefined, never bool, required')
        if row['strong_AP']!=STRONG_AP:
            raise ValueError('Exact unchanged strong fixed E24 reference required')
        rows.append(dict(row))
    def aggregate(items):
        if any(r[k]==-1 for r in items for k in ('LOCAL_AP','CENTER_AP')):
            return None,None,None
        local=math.fsum(r['LOCAL_AP'] for r in items)/len(items)
        center=math.fsum(r['CENTER_AP'] for r in items)/len(items)
        return local,center,local-max(center,STRONG_AP)
    first=aggregate(rows[:1])[2]
    if len(rows)>1 and (first is None or first<=0):
        raise ValueError('STOP or undefined first main result permits no extra initialization seed')
    if len(rows)==3 and first<1:
        two=aggregate(rows[:2])[2]
        if two is None or two<1 or not all(r['LOCAL_AP']>r['CENTER_AP'] for r in rows[:2]):
            raise ValueError('Seed173 requires prelocked two-group mean gate and both paired wins')
    local,center,delta=aggregate(rows)
    wins=[None if r['LOCAL_AP']==-1 or r['CENTER_AP']==-1 else r['LOCAL_AP']>r['CENTER_AP'] for r in rows]
    decision='RULE_NOT_EVALUABLE';allowed=[];accepted=False
    if delta is not None:
        if len(rows)==1:
            if delta<=0:decision='STOP'
            elif delta<1:decision='ONE_EXTRA_SEED';allowed=[SEEDS[1]]
            else:decision='TWO_EXTRA_SEEDS';allowed=list(SEEDS[1:])
        elif len(rows)==2:
            if first>=1:decision='THIRD_SEED_PRELOCKED';allowed=[SEEDS[2]]
            elif delta>=1 and all(wins):decision='ALLOW_THIRD_SEED';allowed=[SEEDS[2]]
            else:decision='STOP'
        else:
            accepted=delta>=1 and all(wins)
            decision='ACCEPTED_INITIALIZATION_SCOPE' if accepted else 'STOP'
    return dict(schema='N17-fixed-main-result-interpretation-v1',main_epoch=24,main_mode='atmost',AP_unit='percent',
        seeds=[r['seed'] for r in rows],main_results=rows,LOCAL_AP_percent=local,CENTER_AP_percent=center,strong_AP_percent=STRONG_AP,
        Delta_AP_points=delta,aggregate_formula='mean(LOCAL)-max(mean(CENTER),strong)',paired_LOCAL_above_CENTER=wins,
        decision=decision,allowed_next_seeds=allowed,accepted_under_locked_rule=accepted,
        qualification_performed=False,secondary_results_may_replace_main=False,
        evidence_scope='Private scout initialization repeats under the same frozen teacher/PR, data, points and budget only',
        innovation_established=False,TGRS_readiness_established=False)
