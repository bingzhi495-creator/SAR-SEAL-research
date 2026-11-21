"""Independent readback of the terminal receipt, historical rows, gates, topology and count closure."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def read(path, digest=None):
    data = path.read_bytes()
    if digest is not None and hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"Incorrect digest for {path.name}")
    return data


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def verify(root, historical):
    term = json.loads(read(root/"PROCESS_TERMINAL_SAFE.json"))
    require(term["actual_process_terminal"] is True and term["actual_exit"] == 0, "Actual successful child exit")
    request = json.loads(read(root/"REQUEST_PRIVATE.json", term["request_sha256"]))
    start = json.loads(read(root/"START_PRIVATE.json"))
    require(start["owner_pid"] == term["owner_pid"] and start["worker_pid"] == term["worker_pid"], "Process continuity")
    read(root/"STDERR_PRIVATE.log", term["stderr_sha256"])
    stdout = read(root/"STDOUT_PRIVATE.log", term["stdout_sha256"]).splitlines()
    foot = json.loads(stdout[-1])
    require(foot == term["footer"] and foot["result_sha256"] == term["result_sha256"], "Terminal result binding")
    result = json.loads(read(root/"ACTUAL/RESULT_SAFE.json", term["result_sha256"]))
    pre = json.loads(read(root/"ACTUAL/PRELABEL_SAFE.json", result["prelabel_sha256"]))
    require(request["worker_sha256"] == term["worker_sha256"] if "worker_sha256" in request else
            request["engine_sha256"] == term["worker_sha256"], "Worker digest")
    require(request["git_prelock"] == "d9d82be0e3de3831819d383481f87cd8342ffb67" and
            pre["worker_sha256"] == term["worker_sha256"] and
            pre["prelock_sha256"] == request["prelock_sha256"] and pre["helper_sha256"] == request["helper_sha256"],
            "Published prelock continuity")
    old = json.loads(read(historical, "66867e13d7697ac18aff843433d5f2d171773e002a2cf8a1d9bf63da4ccddd7f"))
    require(result["status"] == foot["status"] and result["FIT"] and not result["TEST"] and not result["DEV"] and
            result["detector_forwards"] == result["detector_optimizer_updates"] == 0, "Actual FIT-only scope")
    require(result["fold_training"] == old["training"] and result["original_usage"] == old["usage"] and
            result["match_scope"] == old["match_scope"], "Original training and stock matches")
    folds = [json.loads(x) for x in stdout[:-1]]
    require([r["fold"] for r in folds] == list(range(5)) and sum(r["steps"] for r in folds) ==
            result["point_optimizer_updates"] == 1000, "One five-fold, 1000-update fit")
    for arm in ("raw", "fixed", "free_natural", "free_matched_count"):
        require(all(result[k][arm] == old[k][arm] for k in ("cohorts", "strict90_transitions", "budgets"))
                and result["fixed_geometry"]["metrics"][arm] == old["geometry"]["metrics"][arm], "Historical arm parity")
    hist = result["realized_count_histogram"]
    require(sum(row["candidates"] for row in hist) == 10900 and
            all(0 <= row["simple_count"] <= row["natural_count"] <= 8 for row in hist), "Complete per-candidate budgets")
    for arm, key in (("simple", "simple_count"), ("free_natural", "natural_count")):
        edits = sum(row["candidates"] for row in hist if row[key])
        swaps = sum(row["candidates"]*row[key] for row in hist)
        require((edits,swaps) == (result["budgets"][arm]["edited"],result["budgets"][arm]["swaps"]),
                "Independent edited and swap totals")
    require(not any(result["topology"]["simple"].values()), "Exact simple topology count closure")
    witness = [json.loads(row) for row in read(root/"ACTUAL/STRICT_TRANSITIONS_PRIVATE.jsonl",
                                               result["transition_witness_sha256"]).splitlines()]
    require(len(witness) == len({(r["image_id"],r["native_rank"],r["original_gt_id"]) for r in witness}) == 412,
            "Unique original .50 matches")
    for arm, claimed in result["strict90_transitions"].items():
        actual = Counter(row["arms"][arm]["state"] for row in witness)
        require(all(actual[k] == n for k,n in claimed.items()), "Independent strict identity counts")
    new = "simple"
    checks = {}
    for control in ("free_at_simple_count", "layer1_at_simple_count"):
        require(result["budgets"][new] == result["budgets"][control] and all(
                r["arms"][new]["swaps"] == r["arms"][control]["swaps"] for r in witness), "Realized equal counts")
        delta = result["fixed_geometry"]["simple_minus"][control]
        scores = result["cohorts"]
        losses = sum(r["arms"][new]["state"] == "lost" and r["arms"][control]["state"] != "lost" for r in witness)
        checks[control] = dict(AP=scores[new]["AP"]-scores[control]["AP"] >= .30,
                               continuous_IoU=delta["delta_points"] >= .20,
                               positive_lower95=delta["interval95_points"][0] > 0,
                               AP75_retained=scores[new]["AP75"] >= scores[control]["AP75"],
                               original_strict90_losses=losses == 0)
    require(checks == result["gates"] and all(all(v.values()) for v in checks.values()) ==
            result["followup_gate_met"] == foot["followup_gate_met"], "Independent protocol gate closure")
    return dict(status="DFINE_FIT109_SIMPLE_POINT_TERMINAL_AGGREGATE_READBACK_V1", actual_exit=0,
                result_sha256=term["result_sha256"], protocol_sha256=pre["prelock_sha256"],
                historical_parity=True, original_matches=412, all_controls_checked=True,
                followup_gate_met=result["followup_gate_met"], full_scientific_recomputation=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("historical", type=Path)
    args = parser.parse_args()
    receipt = verify(args.root, args.historical)
    encoded = (json.dumps(receipt, sort_keys=True, separators=(",",":"))+"\n").encode()
    with (args.root/"TERMINAL_AGGREGATE_READBACK_SAFE.json").open("xb") as output:
        output.write(encoded)
    print(encoded.decode().strip())
