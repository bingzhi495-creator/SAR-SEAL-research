"""Recompute the paired wall/CPU timing summaries and bind the observed process exit."""
import argparse
from hashlib import sha256
import json
from math import floor, isclose, isfinite
from pathlib import Path
from statistics import fmean, median


def require(value, message):
    if not value:
        raise ValueError(message)


def near(x,y):
    return isclose(x,y,rel_tol=1e-12,abs_tol=1e-8)


def stats(rows):
    values = sorted(fmean(row) for row in rows)
    i = (len(values)-1) * .95
    j = floor(i)
    return dict(mean_ms=fmean(values),median_ms=median(values),p95_ms=values[j]+(i-j)*(values[j+1]-values[j]),
                summed_seconds=sum(values)/1000)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--result-sha",required=True)
    p.add_argument("--session",type=int,required=True)
    p.add_argument("--exit",type=int,required=True)
    p.add_argument("--chunk",required=True)
    args=p.parse_args()
    root=Path(__file__).resolve().parent
    def sealed(path,digest):
        raw=(root/path).read_bytes()
        require(sha256(raw).hexdigest()==digest,"Closed "+path)
        return raw
    require(args.exit==0 and args.session>0,"Observed original session exit")
    result=json.loads(sealed("RESULT_SAFE.json",args.result_sha))
    for path,digest in result["input_refs_sha256"].items():
        sealed(path,digest)
    actual=json.loads(sealed("PAIRED_TIMES_SAFE.json",result["paired_times_sha256"]))
    require(result["images"]==232 and result["measured_pairs_per_image"]==2 and
            result["warm_and_measured_parity_passes_per_arm"]==696 and result["all_warm_sample_uint16_bits_equal"]
            and result["all_complete_candidate_records_exact"],"Coverage of timing and parity checks")
    for k in ("original","optimized"):
        rows=actual[k]
        require(len(rows)==232 and all(len(row)==2 and all(len(pair)==2 and all(isfinite(x) and x>=0 for x in pair)
            for pair in row) for row in rows),"Complete nonnegative paired timings")
    verified=0
    for j,label in enumerate(("wall","process_cpu")):
        arms={k:[[pair[j] for pair in row] for row in actual[k]] for k in ("original","optimized")}
        arms["saving"]=[[a-b for a,b in zip(old,new)] for old,new in zip(arms["original"],arms["optimized"])]
        for arm,rows in arms.items():
            for k,val in stats(rows).items():
                require(near(result["stage_summary"][label+"_"+arm][k],val),label+" "+arm+" "+k)
                verified+=1
            if label=="wall":
                for i in range(2):
                    t=sum(row[i] for row in rows)/1000
                    require(near(result["repetition_totals_wall_seconds"][arm][i],t),"Full population paired total")
                    verified+=1
        if label=="wall":
            total=lambda k:sum(fmean(x) for x in arms[k])
            ratio=total("original")/total("optimized")
            require(near(result["ratio_original_over_optimized"],ratio) and
                    near(result["mean_wall_saving_percent"],100*(1-1/ratio)),"Paired ratio of total wall times")
            signs=[fmean(row) for row in arms["saving"]]
            require((result["faster_images"],result["slower_images"],result["tied_images"])==
                    (sum(x>0 for x in signs),sum(x<0 for x in signs),sum(x==0 for x in signs)),"Per image directions")
            verified+=5
    receipt=dict(status="INDEPENDENT_PAIRED_TIMING_ARITHMETIC_VERIFIED",result_sha256=args.result_sha,
                 verified_fields=verified,original_tool_session=args.session,original_exit=args.exit,
                 original_final_chunk=args.chunk,new_detector_forwards=0,GT_opened=False)
    raw=json.dumps(receipt,indent=2).encode()+b"\n"
    with (root/"INDEPENDENT_ARITHMETIC_SAFE.json").open("xb") as f:
        f.write(raw)
    print(json.dumps(dict(status=receipt["status"],sha256=sha256(raw).hexdigest())),flush=True)


if __name__=="__main__":
    main()
