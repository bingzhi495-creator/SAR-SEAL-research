"""Check the observed process result and independently recompute paired timing summaries."""
import argparse
from hashlib import sha256
import json
from math import isclose, floor
from pathlib import Path
from statistics import fmean, median


def require(value, reason):
    if not value:
        raise ValueError(reason)


def percentile95(values):
    values = sorted(values)
    index = (len(values) - 1) * .95
    first = floor(index)
    return values[first] + (values[first + 1] - values[first]) * (index - first)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-sha", required=True)
    parser.add_argument("--session", type=int, required=True)
    parser.add_argument("--exit-code", type=int, required=True)
    parser.add_argument("--last-chunk", required=True)
    a = parser.parse_args()
    root = Path(__file__).resolve().parent
    actual = (root/"RESULT_SAFE.json").read_bytes()
    require(sha256(actual).hexdigest() == a.result_sha and a.session > 0 and a.exit_code == 0,
            "Successful observed original session and stdout footer result")
    result = json.loads(actual)
    raw_data = (root/"PAIRED_TIMES_SAFE.json").read_bytes()
    require(result["paired_times_sha256"] == sha256(raw_data).hexdigest(), "Original timing ledger")
    time = json.loads(raw_data)
    require(result["worker_sha256"] == sha256((root/"benchmark.py").read_bytes()).hexdigest()
            and result["prelock_sha256"] == sha256((root/"BENCHMARK_PROTOCOL.md").read_bytes()).hexdigest(),
            "Original protocol and worker")
    require(result["images"] == len(time["raw"]) == len(time["natural"]) == 232
            and result["raw_natural_output_checks_per_arm"] == 696 and result["complete_rle_score_rank_parity"],
            "Entire paired population and original parity checks")
    require(all(len(row) == 2 and all(len(pair) == 2 and all(x >= 0 for x in pair) for pair in row)
                for a_ in ("raw","natural") for row in time[a_]), "All nonnegative two-clock measurements")
    verified = 0
    for j, clock in enumerate(("wall","process_cpu")):
        rt = [[pair[j] for pair in row] for row in time["raw"]]
        nt = [[pair[j] for pair in row] for row in time["natural"]]
        dt = [[b-a for a,b in zip(r,n)] for r,n in zip(rt,nt)]
        for mode, values in (("raw",rt),("natural",nt),("incremental",dt)):
            x = [fmean(row) for row in values]
            want = dict(mean_ms=fmean(x),median_ms=median(x),p95_ms=percentile95(x),summed_seconds=sum(x)/1000)
            got = result["timings"][clock+"_"+mode]
            for key, v in want.items():
                require(isclose(v,got[key],rel_tol=1e-12,abs_tol=1e-9),"Recomputed "+clock+" "+mode+" "+key)
                verified += 1
        if clock == "wall":
            for i, expected in enumerate(result["aggregate_incremental_wall_seconds_per_repetition"]):
                require(isclose(sum(row[i] for row in dt)/1000,expected,rel_tol=1e-12), "Paired repetition total")
                verified += 1
    observed = dict(status="PSEG_TEST232_PAIRED_CPU_TIMINGS_ARITHMETIC_VERIFIED",
                    original_source_tool_session=a.session,original_observed_exit=a.exit_code,original_final_chunk=a.last_chunk,
                    original_result_sha256=a.result_sha,verified_timing_fields=verified,images=232,
                    new_detector_forwards=0,new_timing_runs=0,GT_opened=False)
    value = json.dumps(observed,indent=2).encode()+b"\n"
    with (root/"INDEPENDENT_ARITHMETIC_SAFE.json").open("xb") as out:
        out.write(value)
    print(json.dumps(dict(status=observed["status"],result_sha256=sha256(value).hexdigest())),flush=True)


if __name__ == "__main__":
    main()
