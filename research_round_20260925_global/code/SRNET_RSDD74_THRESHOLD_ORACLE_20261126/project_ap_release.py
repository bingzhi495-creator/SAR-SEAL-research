"""Publish an explicit aggregate projection of the witnessed threshold receipt.

The original archived receipt stays private. Its auxiliary per-arm status field
uses native image keys. Correct per-arm statuses are derived from the already
valid transition matrix marginals, with agreement across all three contrasts.
"""
from argparse import ArgumentParser
from hashlib import sha256
import json
import math
from pathlib import Path


SOURCE_RESULT="42db4140f6aa441359250716af6434a8306fa40fca67306138f7eb300e9e56fb"
SOURCE_TERMINAL="a801a2de3ead0a33f20e721f4b7a1290f52d49a9f760a405b0b7afd3c07ab41d"
SOURCE_WORKER="91ac7f3b786215f703447e8def8e561df500a9a520ade06ca72ef082e384bfed"
PROTOCOL="23e98ad86fae80a9fbfed17f22f9654f66cec33342cb4226fb60d931f411a4c9"
SCOPES=("full74","nested42_sensor_date")
ARMS=("raw","fixed","free_matched_count")
PAIRS=((ARMS[2],ARMS[1]),(ARMS[2],ARMS[0]),(ARMS[1],ARMS[0]))
STATES=("TP","FP","ignore")
MATCH_KEYS=("TP_to_same_reference","TP_to_different_reference")
CROSSES=("above_to_above","below_to_above","above_to_below","below_to_below")
CROSS_GROUPS=("original_native_match_raster_all","original_native_match_raster_eligible",
              "original_native_match_continuous_eligible")
VIEW_FIELDS=("images","original_candidates","author_annotations","original_native_matches","original_eligible_native_matches")
THRESHOLDS=tuple(f"{t/100:.2f}" for t in range(50,100,5))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pinned_json(path, expected):
    data=path.read_bytes()
    require(sha256(data).hexdigest()==expected, "Unexpected original receipt bytes")
    return json.loads(data)


def counts(source, expected_keys):
    require(set(source)==set(expected_keys),"Aggregate field whitelist mismatch")
    require(all(type(source[k]) is int and source[k]>=0 for k in expected_keys),"Nonnegative integer aggregate")
    return {k:source[k] for k in expected_keys}


def points(x):
    require(type(x) in (float,int) and math.isfinite(x) and -100<=x<=100,"Finite AP percentage points")
    return x


def public_scope(source):
    view=counts({k:source[k] for k in VIEW_FIELDS},VIEW_FIELDS)
    rows=source["by_threshold"]
    require(set(rows)==set(THRESHOLDS),"Fixed ten thresholds")
    public={}
    pairs=tuple(f"{a} minus {b}" for a,b in PAIRS)
    transition_keys=tuple(f"{a} to {b}" for a in STATES for b in STATES)
    for threshold in THRESHOLDS:
        row=rows[threshold]
        require(set(row["ap_points"])==set(ARMS) and set(row["contrasts"])==set(pairs),"Three fixed comparators")
        ap={arm:points(row["ap_points"][arm]) for arm in ARMS}
        by_arm={}
        all_contrasts={}
        for (new,old),contrast_name in zip(PAIRS,pairs,strict=True):
            contrast=row["contrasts"][contrast_name]
            match=contrast["authoritative_maxdets100"]
            table=counts(match["transition_counts"],transition_keys)
            assignments=counts({k:match[k] for k in MATCH_KEYS},MATCH_KEYS)
            require(sum(table.values())==view["original_candidates"] and sum(assignments.values())==table["TP to TP"],
                    "Whole candidate and retained TP census")
            arms=((old,{a:sum(table[f"{a} to {b}"] for b in STATES) for a in STATES}),
                  (new,{b:sum(table[f"{a} to {b}"] for a in STATES) for b in STATES}))
            for arm,marginal in arms:
                require(by_arm.setdefault(arm,marginal)==marginal,"All contrast marginals agree")
            crossed={group:counts(contrast[group],CROSSES) for group in CROSS_GROUPS}
            require(sum(crossed[CROSS_GROUPS[0]].values())==view["original_native_matches"] and
                    all(sum(crossed[group].values())==view["original_eligible_native_matches"] for group in CROSS_GROUPS[1:]),
                    "Original matched anchor census")
            delta=points(contrast["ap_delta_points"])
            require(abs(ap[new]-ap[old]-delta)<1e-10,"Exact threshold AP arithmetic")
            all_contrasts[contrast_name]=dict(ap_delta_points=delta,
                authoritative_maxdets100=dict(transition_counts=table,**assignments),**crossed)
        public[threshold]=dict(ap_points=ap,authoritative_detection_status=by_arm,contrasts=all_contrasts)
    return dict(view,by_threshold=public)


def main():
    p=ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ("result","terminal","output"):
        p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args()
    raw=pinned_json(a.result,SOURCE_RESULT)
    terminal=pinned_json(a.terminal,SOURCE_TERMINAL)
    require(raw["status"]=="SRNET_RSDD74_THRESHOLD_AP_TRANSITIONS_COMPLETE" and
            raw["worker_sha256"]==terminal["worker_sha256"]==SOURCE_WORKER and raw["protocol_sha256"]==PROTOCOL and
            terminal["actual_exit"]==0 and terminal["phase"]=="score" and terminal["output_footer_verified"] is True and
            terminal["result_sha256"]==SOURCE_RESULT and raw["old_arm_metrics_and_geometry_exact"] is True and
            raw["independent_precision_envelope_exact"] is True and set(raw["views"])==set(SCOPES),
            "Previously witnessed successful threshold audit")
    release=dict(status="SRNET_RSDD74_THRESHOLD_AP_PUBLIC_AGGREGATE",source_result_sha256=SOURCE_RESULT,
                 source_terminal_sha256=SOURCE_TERMINAL,source_worker_sha256=SOURCE_WORKER,protocol_sha256=PROTOCOL,
                 old_arm_metrics_and_geometry_exact=True,independent_precision_envelope_exact=True,
                 public_per_arm_status="Aggregate transition row and column marginals verified across all contrasts",
                 views={view:public_scope(raw["views"][view]) for view in SCOPES})
    payload=(json.dumps(release,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    with a.output.open("xb") as f:
        f.write(payload)
    print(json.dumps(dict(status=release["status"],result_sha256=sha256(payload).hexdigest())))


if __name__=="__main__":
    main()
