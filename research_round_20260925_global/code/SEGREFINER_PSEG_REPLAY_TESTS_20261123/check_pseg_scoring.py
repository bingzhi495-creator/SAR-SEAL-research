"""Exercise PSeg write admission, signed overlap, and location support on analytic masks."""

import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
from shapely.geometry import box

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--code", type=Path, default=Path(__file__).resolve().parents[1],
                    help="Release code directory containing the original pinned scorers")
code = parser.parse_args().code

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, code / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

source = load("source", "PSEG232_PHOENIX_SUPPORT_BUDGET_20261123/score_factorial.py")
stock = load("stock", "SEGREFINER_STOCK_20261123/score_pseg.py")
sign = load("sign", "SEGREFINER_SIGN_20261123/score_pseg.py")
depth = load("depth", "SEGREFINER_DEPTH_20261123/score_pseg.py")
deletion = load("deletion", "SEGREFINER_DEPTH_20261123/depth_priority.py")
geom = load("geom", "PSEG_TEST232_PAIRED_GEOMETRY_20261121/paired_geometry.py")
base = geom.load(code / "DFINE_SEG_EXTERNAL_BASELINE_FIT32_E24_20261112/e24/dev705_free_endpoint/score_endpoint.py", geom.OLD)
continuous = geom.load(code / "MASK_RTDETR_DEV705_CONTINUOUS_AP_20261112/continuous_endpoint.py", geom.GEOM)
modules = continuous.geometry_modules(code)

def segmentation(mask):
    return dict(segmentation=source.encode(mask))

def edit(mask):
    return dict(rle=source.encode(mask))

high = np.zeros((40, 40), bool)
high[4:20, 4:36] = True
low, modified = high.copy(), high.copy()
low[4, 4] = False
modified[10, 10] = False
native = [segmentation(high), segmentation(low)]
payload = dict(ordinal=1, image_id=7, changes={"1": edit(modified)})
roster = [dict(image_ref=dict(path="synthetic"), image_id=7)]

def qualify(change):
    records = {"synthetic": dict(areas=[512, 511], native=native)}
    blocks = {"synthetic": (native, {}, (40, 40))}
    qualified, _ = stock.qualify_edits(source, records, roster, blocks, [change])
    return qualified, records["synthetic"]

qualified, record = qualify(payload)
assert qualified["area_admitted"] == qualified["exact_native_bypass"] == 1
assert stock.mask_at(record, 2, stock.STOCK) == native[1]["segmentation"]
illegal = copy.deepcopy(payload)
illegal["changes"]["2"] = edit(low)
try:
    qualify(illegal)
except ValueError as failure:
    assert "gate-positive" in str(failure), str(failure)
else:
    raise AssertionError("Below-gate write was accepted")

raw = np.zeros((40, 40), bool)
raw[8:32, 8:32] = True
observed = raw.copy()
observed[18, 18:21] = False
observed[18, 32] = True
records = {
    n: dict(native=[segmentation(raw)], edits={"1": {stock.STOCK: edit(observed)}},
            areas=[576], rgb=f"group-{n//3}", size=(40, 40))
    for n in range(24)
}
truth = {n: (box(8, 8, 32, 32),) for n in records}
anchors = {n: (n, 1) for n in records}
counts = sign.extend_masks(source, records)
assert all(row["added_pixels"] == 1 and row["removed_pixels"] == 3 for row in counts.values())
# Replicate the two production evaluate() entry points' arm selection.
stock.ARMS = sign.ARMS
_, exposed, energies, signs, _ = sign.geometry_for_view(source, stock, geom, base, modules, truth, records, anchors, {})
assert sign.primary_deletion(exposed, signs)["both_contexts_supported"]
assert exposed["eligible_pairs"] == 24 and exposed["decoded_image_groups_with_pairs"] == 8
assert np.isclose(energies["gate_exposed"]["delete_truth_pct"]["value"], 300 / 576)
assert np.isclose(energies["gate_exposed"]["add_background_pct"]["value"], 100 / 576)
reduced = {n: pair for n, pair in anchors.items() if n < 21}
_, sub, _, sub_signs, _ = sign.geometry_for_view(source, stock, geom, base, modules, truth, records, reduced, {})
assert sub["eligible_pairs"] == 21 and sub["decoded_image_groups_with_pairs"] == 7
assert not sign.primary_deletion(sub, sub_signs)["both_contexts_supported"]

stock.ARMS = depth.ARMS
changes, _, _ = depth.extend_variants(source, deletion, records)
moved = sum(row["changed_deletes"] for row in changes.values())
assert moved == 72
_, g, support = depth.geometry_views(stock, source, geom, base, modules, truth, records, anchors, {})
key = depth.PERIPHERY + "_minus_" + stock.STOCK
metric = g["pairwise"][depth.PERIPHERY + " minus " + stock.STOCK]["continuous_iou_points"]
assert support[key]["different_hulls"] == 24 and np.isclose(metric["delta"], 0)
assert not depth.primary_gate(.30, metric, support[key])["supported_advantage"]
print(json.dumps(dict(synthetic_only=True, actual_data_read=False, stock_512_gate=True, below_gate_bypass=True,
                      signed_conservation=True, signs_24_in_8_pass=True, signs_21_in_7_fail=True,
                      moved_deletions=moved, equal_overlap_does_not_advance=True)))
