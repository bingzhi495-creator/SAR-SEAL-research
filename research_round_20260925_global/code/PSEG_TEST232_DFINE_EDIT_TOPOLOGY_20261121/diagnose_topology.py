"""Count topology changes in closed PSeg boundary edits, without target labels."""

import argparse
from collections import Counter
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType

import numpy as np
from pycocotools import mask as codec
from scipy.ndimage import generate_binary_structure, label


ADAPTER_SHA = "dd8e435976d2331146c8f83d70f368ee4e0954c05520c54fa9a3b1205838d40f"
RECEIPT_SHA = "e99d0f2bbd9e1839e260a88075f6af187e05ffc2b86abaf6cc33b788372b7b71"
PAIRS = (
    ("fixed", "free_matched_count"),
    ("native_matched_count", "free_natural"),
)
FG = generate_binary_structure(2, 2)
BG = generate_binary_structure(2, 1)


def topology(rle):
    mask = codec.decode(rle)
    y, x = np.nonzero(mask)
    if not len(y):
        return 0, 0, 0, 0
    crop = np.pad(mask[y.min():y.max()+1, x.min():x.max()+1], 1).astype(bool)
    components, foreground = label(crop, structure=FG)
    _, background = label(~crop, structure=BG)
    area = np.bincount(components[crop])
    return foreground, background - 1, int(np.count_nonzero(area[1:] == 1)), int(area[1:].max())


def count_change(counts, before, after, area):
    old_c, old_h, old_single, old_largest = before
    new_c, new_h, new_single, new_largest = after
    counts.update(edited=1, components_increased=int(new_c > old_c),
                  components_decreased=int(new_c < old_c),
                  newly_disconnected=int(old_c == 1 and new_c > 1),
                  singleton_count_increased=int(new_single > old_single),
                  holes_increased=int(new_h > old_h),
                  holes_decreased=int(new_h < old_h),
                  largest_component_lost_pixels=max(0, old_largest-new_largest))
    counts["largest_component_share_delta_sum"] += (new_largest-old_largest) / area


def summarize(blocks, receipt, excluded):
    sets = {"official": set(), "exclude_exact_fit_overlap": set(excluded)}
    results = {}
    for name, omit in sets.items():
        included = [(key, val) for key, val in blocks.items() if key not in omit]
        results[name] = dict(images=len(included), candidates=sum(len(y[0]) for _, y in included),
                             arms={}, comparisons={})
    for first, second in PAIRS:
        stats = {name: {first: Counter(), second: Counter(), "comparison": Counter()} for name in sets}
        for name, (rows, edits, _) in blocks.items():
            for rank, change in edits.items():
                if first not in change:
                    continue
                if second not in change or change[first]["swaps"] != change[second]["swaps"]:
                    raise ValueError("Same pairs and equal edit count required")
                original = rows[int(rank)-1]["segmentation"]
                area = int(codec.area(original))
                before = topology(original)
                a = topology(change[first]["rle"])
                b = topology(change[second]["rle"])
                for sub, omit in sets.items():
                    if name in omit:
                        continue
                    counters = stats[sub]
                    count_change(counters[first], before, a, area)
                    count_change(counters[second], before, b, area)
                    counters["comparison"].update(
                        candidate_edits=1, swaps=change[first]["swaps"],
                        more_foreground_components=int(b[0] > a[0]),
                        fewer_foreground_components=int(b[0] < a[0]),
                        more_holes=int(b[1] > a[1]), fewer_holes=int(b[1] < a[1]),
                    )
        for name, group in stats.items():
            for arm in (first, second):
                metrics = dict(group[arm])
                if name == "official" and metrics["edited"] != receipt["counts"][arm+"_edited"]:
                    raise ValueError("Independently accepted real edited candidate count")
                total = metrics.pop("largest_component_share_delta_sum")
                metrics["largest_component_share_mean_delta_points"] = 100 * total / metrics["edited"]
                results[name]["arms"][arm] = metrics
            results[name]["comparisons"][second + " minus " + first] = dict(group["comparison"])
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("adapter","accepted","manifest","predictions","candidate-audit","edits","output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if not sys.flags.isolated or not sys.dont_write_bytecode or os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise ValueError("Use the original isolated CPU runtime")
    source = args.adapter.read_bytes()
    if sha256(source).hexdigest() != ADAPTER_SHA:
        raise ValueError("Exact published frozen score input adapter")
    reader = ModuleType("_pinned_pseg_input")
    reader.__file__ = str(args.adapter)
    exec(compile(source, str(args.adapter), "exec"), reader.__dict__)
    start = perf_counter()
    receipt = json.loads(reader.pinned(args.accepted, RECEIPT_SHA, 1024**2))
    reader.check_acceptance(receipt)
    roster = reader.validate_manifest(json.loads(reader.pinned(args.manifest, receipt["manifest_sha256"], 4*1024**2)))
    streams = []
    for path, key in ((args.predictions,"input_predictions_sha256"),(args.candidate_audit,"candidate_audit_sha256"),
                      (args.edits,"edits_sha256")):
        raw = reader.pinned(path, receipt[key], 256*1024**2)
        if not raw.endswith(b"\n"):
            raise ValueError("Complete native stream")
        streams.append([json.loads(line) for line in raw.splitlines()])
    blocks = reader.join_blocks(roster, *streams)
    result = dict(status="PSEG_TEST232_DFINE_EDIT_TOPOLOGY_DIAGNOSTIC_COMPLETE",
                  purpose="Historical target-split, descriptive label-free mask topology audit",
                  target_labels_opened=False, model_forwards=0, optimizer_updates=0,
                  connectivity=dict(foreground=8, background=4),
                  source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(), source_receipt_sha256=RECEIPT_SHA,
                  runtime=dict(numpy=np.__version__,scipy=version("scipy"),pycocotools=version("pycocotools")),
                  views=summarize(blocks, receipt, reader.EXCLUDED), seconds=perf_counter()-start)
    raw = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with args.output.open("xb") as out:
        out.write(raw)
    print(json.dumps({"status": result["status"], "result_sha256": sha256(raw).hexdigest()}), flush=True)


if __name__ == "__main__":
    main()
