"""UNBOUND_DRAFT: pure-stdlib N16 first-seed interpretation; no qualification, model or file IO.
Caller supplies already qualified canonical COCO stats in fractions, never AP percentages.
"""
import math

REFERENCE_AP_PERCENT=68.09938494745545
POLICIES=('native_official','strong_fixed_E24_Omega','E24_UTILITY_atmost','E24_ERROR_atmost',
    'E24_UTILITY_exact','E24_ERROR_exact','E8_UTILITY_atmost','E8_ERROR_atmost','E8_UTILITY_exact','E8_ERROR_exact',
    'E24_DENSE_ERROR_FLIP_atmost','E24_MAGNET_atmost')
TABLE_METRICS=(('AP',0),('AP50',1),('AP75',2),('AR100',8))

def _canonical_stats(value,policy):
    if type(value) is not dict or 'stats' not in value:
        raise TypeError('Each policy requires a stats field: '+policy)
    stats=value['stats']
    if type(stats) not in (list,tuple) or len(stats)!=12:
        raise ValueError('Exactly canonical COCO stats0..11 required: '+policy)
    checked=[]
    for index,number in enumerate(stats):
        if type(number) not in (int,float):
            raise TypeError('Real numeric COCO stat required, no bool: %s[%d]'%(policy,index))
        if not math.isfinite(number) or not (number==-1 or 0<=number<=1):
            raise ValueError('Finite COCO fraction0..1 or -1 undefined required: %s[%d]'%(policy,index))
        checked.append(float(number))
    return checked

def interpret_first_seed(control_results,reference_AP_percent=REFERENCE_AP_PERCENT):
    """Return unrounded percent table and only the prelocked main E24 decision.

    -1 maps to None (undefined), never zero. No ranking of secondary policies.
    This function does not inspect or certify Source, RAW, Q or actual execution.
    """
    if type(control_results) is not dict or set(control_results)!=set(POLICIES):
        raise ValueError('Exactly the twelve prelocked N16 policy results required')
    if type(reference_AP_percent) not in (int,float) or not math.isfinite(reference_AP_percent) or reference_AP_percent!=REFERENCE_AP_PERCENT:
        raise ValueError('Use the fixed prelocked E24 reference in AP percent, unchanged')
    stats={name:_canonical_stats(control_results[name],name) for name in POLICIES}
    table=[]
    for policy in POLICIES:
        row=dict(policy=policy)
        undefined=[]
        for name,index in TABLE_METRICS:
            raw=stats[policy][index]
            row[name]=None if raw==-1 else raw*100.0
            if raw==-1:undefined.append(name)
        row['undefined_metrics']=undefined
        table.append(row)
    U_raw=stats['E24_UTILITY_atmost'][0];E_raw=stats['E24_ERROR_atmost'][0]
    U=None if U_raw==-1 else U_raw*100.0
    E=None if E_raw==-1 else E_raw*100.0
    reference=float(reference_AP_percent)
    comparison=None if E is None else max(E,reference)
    delta=None if U is None or E is None else U-comparison
    paired_win=None if U is None or E is None else U>E
    if delta is None:
        decision='RULE_NOT_EVALUABLE';extra=[]
        rule='A main COCO AP is undefined; do not invent STOP, GO or a repeat trigger.'
    elif delta>=1.0 and paired_win:
        decision='GO';extra=[202610140162,202610140163]
        rule='Main Delta>=1 AP point and Utility>Error: allow the two prelocked initialization repeats.'
    elif delta<=0.0:
        decision='STOP';extra=[]
        rule='Main Delta<=0: stop expansion of this fixed N16 recipe.'
    else:
        decision='ONE_EXTRA_SEED';extra=[202610140162]
        rule='0<main Delta<1: at most one prelocked extra initialization seed162; no immediate seed163 authorization.'
    return dict(schema='N16-first-seed-fixed-main-result-interpretation-v1',
        first_initialization_seed=202610140161,table=table,table_units='percent',
        canonical_stats_fraction=stats,UTILITY_AP_fraction=None if U_raw==-1 else U_raw,
        ERROR_AP_fraction=None if E_raw==-1 else E_raw,UTILITY_AP_percent=U,ERROR_AP_percent=E,
        fixed_reference_AP_percent=reference,comparison_reference_AP_percent=comparison,
        Delta_AP_points=delta,Utility_strictly_above_paired_Error=paired_win,decision=decision,
        allowed_extra_seeds=extra,extra_seed_count_limit=len(extra),decision_rule=rule,
        decision_precision='original Python float values; no rounding or threshold tolerance',
        main_policies=['E24_UTILITY_atmost','E24_ERROR_atmost'],
        secondary_policies_never_replace_main=True,qualification_performed=False,
        interpretation_scope='conditional first-seed development rule for the fixed teacher/PR/cache/N16 recipe',
        innovation_established=False,TGRS_readiness_established=False)
