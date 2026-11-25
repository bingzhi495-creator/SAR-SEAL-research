"""Check the portable writer against a completed Instance-RSDD reserve timing cache."""
from argparse import ArgumentParser
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import runpy
from time import perf_counter

def require(ok, message):
    if not ok:
        raise ValueError(message)


parser = ArgumentParser(description=__doc__)
parser.add_argument("--collect", required=True, type=Path, help="Closed benchmark collect directory")
parser.add_argument("--output", required=True, type=Path, help="Fresh audit directory")
args = parser.parse_args()

collection = args.collect
audit_root = args.output
writer = Path(__file__).with_name("replay_fixed_count.py")
terminal_path = collection.parent / "observed_collect" / "TERMINAL.json"
result_bytes = (collection / "RESULT_SAFE.json").read_bytes()
terminal = json.loads(terminal_path.read_bytes())
closure = json.loads(result_bytes)
require(terminal["actual_exit"] == 0 and terminal["output_footer_verified"] is True, "Collection closure required")
require(terminal["result_sha256"] == sha256(result_bytes).hexdigest(), "Collection receipt mismatch")
cache = (collection / "CANDIDATES_PRIVATE.jsonl").read_bytes()
require(closure["candidate_cache_sha256"] == sha256(cache).hexdigest(), "Candidate cache mismatch")
require(closure["label_files_opened"] == closure["detector_forwards"] == closure["optimizer_updates"] == 0,
        "Only the closed label-free timing cache is accepted")
audit_root.mkdir()
stage_input = audit_root / "candidate_input_PRIVATE.jsonl"
expected = []
count = Counter()
with stage_input.open("x", encoding="utf-8") as stream:
    for frame_id, raw in enumerate(cache.splitlines(), start=1):
        image_rows = json.loads(raw)
        count.update(images=1, candidates=len(image_rows))
        for rank, row in enumerate(image_rows, start=1):
            native = row["native"]
            metadata = {"image_id": frame_id, "category_id": 1, "score": 0.5, "rank": rank}
            record = dict(candidate={**metadata, "segmentation": native},
                          pairs=row.get("coords", []), occupancy=row.get("q", []))
            stream.write(json.dumps(record) + "\n")
            expected.append((metadata, row["expected"]["F"], row["expected"]["I"]) if "expected" in row
                            else (metadata, native, native))
            if "k" in row:
                count.update(positive_candidates=1, swaps=row["k"])

start = perf_counter()
output = audit_root / "masks_PRIVATE"
runpy.run_path(str(writer))["run"](stage_input, output, writer.parent.parent)
elapsed = perf_counter() - start
fixed = json.loads((output / "fixed.json").read_text(encoding="utf-8"))
independent = json.loads((output / "independent.json").read_text(encoding="utf-8"))
report = json.loads((output / "summary.json").read_text(encoding="utf-8"))
require(dict(count) == closure["counts"], "Archive roster mismatch")
require(report == {"status": "complete", "candidates": count["candidates"],
                  "positive_candidates": count["positive_candidates"], "swaps_per_arm": count["swaps"]},
        "Writer count mismatch")
for left, right, (metadata, ref_fixed, ref_independent) in zip(fixed, independent, expected, strict=True):
    require(left == {**metadata, "segmentation": ref_fixed}, "Fixed mask or record mismatch")
    require(right == {**metadata, "segmentation": ref_independent}, "Independent mask or record mismatch")
safe = {
    "status": "completed_exact_mask_replay",
    "counts": dict(count),
    "closed_cache_sha256": closure["candidate_cache_sha256"],
    "closed_cache_safe_receipt_sha256": terminal["result_sha256"],
    "writer_sha256": sha256(writer.read_bytes()).hexdigest(),
    "verifier_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
    "fixed_and_independent_rle_equal_at_all_slots": True,
    "positive_fp32_selection_slots": count["positive_candidates"],
    "zero_budget_passthrough_slots": count["candidates"] - count["positive_candidates"],
    "placeholder_scores_used_only_for_record_alignment": True,
    "actual_metrics_computed": False,
    "detector_forwards": 0, "optimizer_updates": 0, "label_files_opened": 0,
    "writer_replay_seconds": elapsed,
}
(audit_root / "RESULT_SAFE.json").write_text(json.dumps(safe, indent=2) + "\n", encoding="utf-8")
print(json.dumps(safe))

