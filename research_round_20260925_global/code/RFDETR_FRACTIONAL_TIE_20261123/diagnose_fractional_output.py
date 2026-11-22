"""Count frozen prediction disagreements between supervision arms without reading target labels."""

import argparse
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path

from pycocotools import mask as codec


SEEDS = (5217, 5218, 5219)
OTHERS = ("T", "H", "C")
EDITS = "80158e7aa4a7fc793130b7bd74444406ee19fb6da4dbda94b89285dde344e860"
REPLAY = "a7fdfa0dc1caa7b94c81a3240904501b6394625eec4919b42ecd8f7d761c6f4a"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(data):
    return sha256(data).hexdigest()


def diagnose(replay, terminal):
    event = terminal.read_bytes()
    obs = json.loads(event)
    require(obs["actual_exit"] == 0 and obs["output_footer_verified"] is True
            and obs["worker_sha256"] == REPLAY, "Verified exit and footer of the reserve replay required")
    summary = (replay / "RESULT_SAFE.json").read_bytes()
    result = json.loads(summary)
    require(digest(summary) == obs["result_sha256"] and result["worker_sha256"] == REPLAY
            and result["status"] == "RFDETR_FRACTIONAL_TIE_RESERVE74_REPLAY_COMPLETE"
            and result["edits_sha256"] == EDITS and result["GT_opened"] is False
            and result["target_labels_opened"] is False and result["signed_xor_exact"] is True
            and result["historical_5217_F_exact"] is True, "Closed fractional supervision output required")
    stats = {str(seed): {arm: dict(different_masks=0, F_only_foreground_cells=0,
                 other_only_foreground_cells=0, symmetric_difference_cells=0) for arm in OTHERS} for seed in SEEDS}
    labels = {f"{s}/{a}" for s in SEEDS for a in ("F", *OTHERS)}
    images, candidates, swaps = set(), 0, 0
    stream = sha256()
    with (replay/"EDITS_PRIVATE.jsonl").open("rb") as input_rows:
        for raw in input_rows:
            stream.update(raw)
            row = json.loads(raw)
            require(row["image_id"] not in images, "Duplicate image row")
            images.add(row["image_id"])
            for actions in row["changes"].values():
                require(set(actions) == labels, "All twelve frozen arms required")
                k = actions["5217/F"]["swaps"]
                require(type(k) is int and 0 < k <= 8 and all(x["swaps"] == k for x in actions.values()),
                        "Every edited candidate must keep the same positive swap count")
                candidates += 1
                swaps += k
                for seed in SEEDS:
                    f = actions[f"{seed}/F"]["rle"]
                    area = int(codec.area(f))
                    for arm, totals in stats[str(seed)].items():
                        other = actions[f"{seed}/{arm}"]["rle"]
                        require(f["size"] == other["size"] == [512, 512] and int(codec.area(other)) == area,
                                "Equal-area masks on the same chip required")
                        common = int(codec.area(codec.merge([f, other], intersect=True)))
                        side = area - common
                        require(0 <= side <= 2*k and (side > 0) == (f != other),
                                "Canonical RLE disagreement must match the actual pixel-set disagreement")
                        totals["different_masks"] += int(side > 0)
                        totals["F_only_foreground_cells"] += side
                        totals["other_only_foreground_cells"] += side
                        totals["symmetric_difference_cells"] += 2*side
    require(stream.hexdigest() == EDITS and len(images) == result["images"] == 74
            and candidates == result["budget_candidates"] == 4092 and swaps == result["budget_swaps"] == 30841,
            "The closed entire reserve stream, population and budget must match")
    return dict(status="RFDETR_FRACTIONAL_TIE_OUTPUT_DIAGNOSIS_COMPLETE", images=len(images),
        positive_budget_candidates=candidates, swaps_per_arm=swaps, mapped_candidates=result["mapped_candidates"],
        pair_comparison="F versus the specified arm, pooled over all positive-budget candidates",
        counts_by_seed_and_comparator=stats, GT_opened=False, detector_forwards=0, optimizer_updates=0,
        protocol_sha256=result["protocol_sha256"], edits_sha256=EDITS, replay_result_sha256=digest(summary),
        replay_terminal_sha256=digest(event), worker_sha256=digest(Path(__file__).read_bytes()),
        pycocotools_version=version("pycocotools"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("replay", "replay-terminal", "output"):
        parser.add_argument("--"+name, required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "A fresh output directory is required")
    result = diagnose(args.replay, args.replay_terminal)
    args.output.mkdir()
    raw = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n").encode()
    with (args.output/"RESULT_SAFE.json").open("xb") as stream:
        stream.write(raw)
    print(json.dumps(dict(status=result["status"], result_sha256=digest(raw))), flush=True)


if __name__ == "__main__":
    main()
