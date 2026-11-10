import hashlib
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

if torch.__version__ != "2.10.0+cu130" or torch.cuda.is_initialized():
    raise ValueError("Expected exact closed-audit CPU torch runtime")
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
root = Path(sys.argv[1])
pins = dict(SSDD="fc7bdbe03d99fa4a77b5ec67d90a8b6b086ea5f3727627611cce8c258fda3d38",
            HRSID="2e4603a8bf45f6ffeb541ecd3b92fc99b5ccba0aff8982217903124703d1babe")
cache = {}
for kind, pin in pins.items():
    raw = (root / (kind+"_PRIVATE") / "ACTUAL" / "VIEWS_PRIVATE.jsonl").read_bytes()
    if hashlib.sha256(raw).hexdigest() != pin:
        raise ValueError("Closed ledger does not match its terminal result")
    views = [json.loads(line) for line in raw.splitlines()]
    for path in ("direct", "crop"):
        pairs = {(axis, step["before"][axis], step["after"][axis])
                 for view in views if view["path"] == path
                 for step in view["steps"] if step["kind"] == "resize" for axis in (0, 1)}
        differences = 0
        mismatched_shapes = 0
        max_delta = 0
        for axis, old, new in sorted(pairs):
            key = (axis, old, new)
            if key not in cache:
                source = torch.arange(old, dtype=torch.float32).view((1, 1, old, 1) if axis == 0 else (1, 1, 1, old))
                output = F.interpolate(source, size=(new, 1) if axis == 0 else (1, new), mode="nearest-exact").reshape(-1)
                exact = ((2 * torch.arange(new, dtype=torch.int64) + 1) * old) // (2 * new)
                distance = (output.to(torch.int64) - exact).abs()
                cache[key] = (int(distance.ne(0).sum()), int(distance.max()))
            count, delta = cache[key]
            differences += count
            mismatched_shapes += int(count > 0)
            max_delta = max(max_delta, delta)
        print(json.dumps(dict(dataset=kind, path=path, distinct_axis_resize_shapes=len(pairs),
                              axis_locations=sum(new for _, _, new in pairs),
                              mismatched_axis_shapes=mismatched_shapes, mismatched_indices=differences,
                              max_abs_index_difference=max_delta), sort_keys=True))
if torch.cuda.is_initialized():
    raise ValueError("CPU audit unexpectedly initialized CUDA")

for kind, pin in pins.items():
    raw = (root / (kind+"_PRIVATE") / "ACTUAL" / "VIEWS_PRIVATE.jsonl").read_bytes()
    if hashlib.sha256(raw).hexdigest() != pin:
        raise ValueError("Closed ledger changed")
    crop = [view for view in map(json.loads, raw.splitlines()) if view["path"] == "crop"]
    for discrepancies in (False, True):
        views = [view for view in crop if any(
            cache[(axis, step["before"][axis], step["after"][axis])][0] > 0
            for step in view["steps"] if step["kind"] == "resize" for axis in (0, 1)
        ) is discrepancies]
        print(json.dumps(dict(dataset=kind, arithmetic_disagreement_possible=discrepancies,
                              crop_views=len(views),
                              common_masks=sum(view["common_nonempty"] for view in views),
                              changed_masks=sum(view["common_changed_masks"] for view in views),
                              changed_pixels=sum(view["common_changed_pixels"] for view in views)), sort_keys=True))
