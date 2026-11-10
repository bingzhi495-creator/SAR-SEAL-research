"""Verify the fixed replay and leave out each repeated four-image block."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import replay
from scope import BATCH_SIZE, canonical


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("bundle", "plan", "upstream", "original", "replayed", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("plan-sha256", "original-sha256", "protocol-commit", "replay-source-sha256"):
        p.add_argument("--" + name, required=True)
    args = p.parse_args()
    assert sha256(Path(replay.__file__).read_bytes()).hexdigest() == args.replay_source_sha256
    original_raw = args.original.read_bytes()
    assert sha256(original_raw).hexdigest() == args.original_sha256
    prior = json.loads(original_raw)
    collected = []
    original_summary = replay.group_summary
    def capture(grouped, keep):
        if not collected:
            collected.append(grouped)
        else:
            assert collected[0] is grouped
        return original_summary(grouped, keep)
    replay.group_summary = capture
    try:
        replay.run(SimpleNamespace(bundle=args.bundle, plan=args.plan, upstream=args.upstream,
                                  plan_sha256=args.plan_sha256, protocol_commit=args.protocol_commit,
                                  output=args.replayed))
    finally:
        replay.group_summary = original_summary
    readback = json.loads(args.replayed.read_bytes())
    assert {k:v for k,v in prior.items() if k!="elapsed_seconds"} == {
        k:v for k,v in readback.items() if k!="elapsed_seconds"}
    groups = [row["group"] for row in json.loads(args.plan.read_bytes())["sources"]]
    assert set(groups) == set(collected[0]) and len(groups) == 24
    omitted = []
    for offset in range(0,len(groups),BATCH_SIZE):
        excluded = set(groups[offset:offset+BATCH_SIZE])
        remaining = {g:v for g,v in collected[0].items() if g not in excluded}
        reading = original_summary(remaining,lambda _:True)
        assert reading["groups"] == 20 and reading["y_absolute"]["groups"] == 20
        omitted.append(dict(ordinal=offset//BATCH_SIZE+1, groups=reading["groups"],
                            vertical_absolute_improvement_px=reading["y_absolute"]["favorable_delta"],
                            continuous_iou_gain=reading["iou"]["favorable_delta"],
                            retention_rate_gain=reading["retain"]["favorable_delta"]))
    result=dict(status="MASK_RTDETR_HRSID_FIT_FIXED_BATCH_BLOCK_SENSITIVITY_ACTUAL_V1",
                original_result_sha256=args.original_sha256,
                replay_equal_except_elapsed=True, additional_unique_views=0,
                fixed_batch_blocks=len(omitted), groups_per_block=BATCH_SIZE,
                leave_one_block=omitted, official_DEV_opened=False, official_TEST_opened=False,
                model_executed=False)
    raw=canonical(result)
    with args.output.open("xb") as f:
        f.write(raw)
    print(json.dumps(dict(status=result["status"], sha256=sha256(raw).hexdigest()),sort_keys=True),flush=True)


if __name__=="__main__":
    main()

