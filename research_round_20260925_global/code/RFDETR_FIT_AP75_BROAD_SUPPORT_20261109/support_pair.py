"""Fixed grouped folds and exact native-failure feasibility."""
from hashlib import sha256


def fold_map(old,added):
    if old & added:
        raise ValueError("Content groups must be disjoint")
    folds={}
    for role,salt in ((old,"P175_AP75_GROUP_OOF_V1:"),(added,"P177_AP75_ADDED_GROUP_OOF_V1:")):
        keys=sorted(role,key=lambda s:sha256((salt+s).encode()).digest())
        folds.update({s:i%5 for i,s in enumerate(keys)})
    return folds


def rescue_case(native_slack,lo,hi,small_hits,large_hits):
    if not (0<=lo<=hi<=8 and lo<=small_hits<=hi and lo<=large_hits<=hi):
        raise ValueError("Paired actions must obey support bounds")
    if native_slack>=0:
        return None
    r=4-native_slack//7
    feasible="unreachable" if r>hi else "all_actions_pass" if r<=lo else "choice_matters"
    small=small_hits>=r
    large=large_hits>=r
    actual="both" if small and large else "small_only" if small else "large_only" if large else "neither"
    return r,feasible,actual
