"""Check the selection capacity of the closed, fixed-quota source roster."""
from argparse import ArgumentParser
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

RENDER = "7aab5753eafe1e18464d0740a6b440fe09997dca2020c64a0d48180c228728fd"
RENDER_WORKER = "c4841925513aeee64078064d483c73557b48c352594dd0d1edbe67a15be4b5b8"
SCORE = "b93e0f4d355f538ae751b8622bf359541dd5705f25d1dfe6eca836984521b052"
SELECTIVE_LOCK = "4eba0ba9d289ba7544df104aadb899a6056af9e04d04e13bc02a5f284dd2a85e"


def digest(raw):
    return sha256(raw).hexdigest()


def exact(path, expected):
    raw = path.read_bytes()
    if digest(raw) != expected:
        raise ValueError("Input digest mismatch: " + path.name)
    return raw


def bounds(values):
    order = sorted(values)
    eligible = Counter(order)
    amount = len(order) // 3
    select = {k: n // 3 for k, n in eligible.items()}
    remaining = amount - sum(select.values())
    remainders = sorted((k for k in eligible if eligible[k] % 3), key=lambda k: (-(eligible[k] % 3), k))
    for k in remainders[:remaining]:
        select[k] += 1
    assert sum(select.values()) == amount
    assert all(0 <= select[k] <= n for k, n in eligible.items())
    variable = {k for k, n in eligible.items() if 0 < select[k] < n}
    switches = 2 * sum(min(select[k], n - select[k]) for k, n in eligible.items())
    return dict(
        eligible=len(order), adopted=amount, strata=len(eligible), flexible_strata=len(variable),
        flexible_eligible=sum(eligible[k] for k in variable),
        strata_with_zero_adoption=int(sum(select[k]==0 for k in eligible)),
        strata_with_complete_adoption=int(sum(select[k]==n for k,n in eligible.items())),
        quota_matched_per_sign=sum(k * n for k, n in select.items()),
        min_cardinality_only_per_sign=sum(order[:amount]),
        max_cardinality_only_per_sign=sum(order[len(order)-amount:]),
        maximum_membership_disagreement=switches,
        images_with_any_adoption=int(amount>0),
        images_with_multiple_legal_policies=int(switches>0),
    )


def main(a):
    started = perf_counter()
    if digest(a.protocol.read_bytes()) != a.protocol_sha:
        raise ValueError("Unexpected audit protocol")
    exact(a.selective_lock, SELECTIVE_LOCK)
    rt = json.loads((a.prior / "RENDER_TERMINAL.json").read_bytes())
    st = json.loads((a.prior / "SCORE_TERMINAL.json").read_bytes())
    if not (rt["actual_exit"] == st["actual_exit"] == 0 and rt["output_footer_verified"]
            and st["output_footer_verified"] and rt["result_sha256"] == RENDER
            and rt["worker_sha256"] == RENDER_WORKER and st["result_sha256"] == SCORE):
        raise ValueError("Original render/score closure required")
    render = json.loads(exact(a.prior / "render/RESULT_SAFE.json", RENDER))
    score = json.loads(exact(a.prior / "score/RESULT_SAFE.json", SCORE))
    if score["render_result_sha256"] != RENDER or render["source_gate_labels_opened"]:
        raise ValueError("Unqualified archived preparation")
    stored = exact(a.prior / "render/GATE_RENDER_PRIVATE.jsonl", render["render_stream_sha256"])
    data = [json.loads(line) for line in stored.splitlines()]
    stats = Counter(images=len(data))
    if len(data) != len({r["image_id"] for r in data}) or len(data) != 183:
        raise ValueError("Expected all 183 frozen records")
    for row in data:
        candidates = row["candidates"]
        if len(candidates) != 100:
            raise ValueError("Expected every one of 100 original candidate slots")
        stats["positive_quotas"] += sum(p["quota"] > 0 for p in candidates)
        stats["all_arms_exchanges_per_sign"] += sum(p["quota"] for p in candidates)
        eligible = [p["quota"] for p in candidates
                    if p["quota"] and p["changes"]["bce_robust"] != p["changes"]["static"]]
        stats.update(bounds(eligible))
    if stats["positive_quotas"] != 8785 or stats["all_arms_exchanges_per_sign"] != 57692:
        raise ValueError("Frozen full-roster signed budget mismatch")
    result = dict(status="SPSD_PSEG_BCE_ADOPTION_CAPACITY_COMPLETE",
                  original_render_sha256=RENDER, original_score_sha256=SCORE,
                  selective_protocol_sha256=SELECTIVE_LOCK, protocol_sha256=a.protocol_sha,
                  worker_sha256=digest(Path(__file__).read_bytes()),
                  label_access=False, inference=False, splits=["source_gate"], capacity=stats,
                  elapsed_seconds=perf_counter()-started)
    payload = (json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n").encode("utf-8")
    a.output.mkdir()
    with (a.output / "RESULT_SAFE.json").open("xb") as output:
        output.write(payload)
    print(json.dumps(dict(status=result["status"],result_sha256=digest(payload))),flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for key in ("prior", "protocol", "selective-lock", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--protocol-sha", required=True)
    main(parser.parse_args())
