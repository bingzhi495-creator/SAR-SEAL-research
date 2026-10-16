"""UNBOUND ideal-geometry APIs. No real FIT, feature, LayerNorm or AP evidence.

Actual owned execution must separately lock the complete row list, count=32,
seed=202610170301, cap=64, geometry contract and source identities. Tiny fixture
overrides do not alter that future execution protocol. These functions do not
open metadata/data files, run a model or certify any scientific execution.
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

_geometry_spec = importlib.util.spec_from_file_location(
    "_n21_unbound_native_support", Path(__file__).with_name("native_support.py")
)
_native = importlib.util.module_from_spec(_geometry_spec)
_geometry_spec.loader.exec_module(_native)

DEFAULT_SEED = 202610170301
SHAPE_KEYS = ("native_shape", "canvas_shape", "crop_shape", "image_shape")


def _seed(seed):
    if type(seed) is not int:
        raise ValueError("seed must be an explicit integer")


def _checked_rows(rows):
    if not isinstance(rows, (list, tuple)) or not rows:
        raise ValueError("rows must be a nonempty list or tuple")
    seen = set()
    for row in rows:
        if type(row) is not dict or any(k not in row for k in ("image_key",) + SHAPE_KEYS):
            raise ValueError("Every row requires image_key and all four explicit shapes")
        key = row["image_key"]
        if type(key) is not str or not key or key in seen:
            raise ValueError("image_key must be a unique nonempty string")
        seen.add(key)
        for field in SHAPE_KEYS:
            shape = row[field]
            if (not isinstance(shape, (tuple, list)) or len(shape) != 2
                    or any(type(x) is not int or x < 1 for x in shape)):
                raise ValueError(field + " must contain two positive integers")
        if any(a > b for a, b in zip(row["crop_shape"], row["canvas_shape"])):
            raise ValueError("Top-left crop cannot exceed the canvas")
    return rows


def select_rows(rows, seed=DEFAULT_SEED, count=32):
    """Validate all rows, then hash-order without any shape-dependent filtering."""
    _seed(seed)
    _checked_rows(rows)
    if type(count) is not int or count < 1 or len(rows) < count:
        raise ValueError("Not enough unique valid rows for the positive requested count")
    def order(row):
        key = row["image_key"]
        digest = hashlib.sha256(("N21-image-v1|" + str(seed) + "|" + key).encode("utf-8")).digest()
        return digest, key
    return copy.deepcopy(sorted(rows, key=order)[:count])


def unique_centers(row, seed, cap=64):
    """Hash candidates from k=0, keeping first occurrence; never retry a seed."""
    _seed(seed)
    _checked_rows([row])
    if type(cap) is not int or not 1 <= cap <= 64:
        raise ValueError("cap must be an integer in [1,64]; owned execution fixes64")
    height, width = row["image_shape"]
    target = min(cap, height * width)
    limit = max(10000, 100 * target)
    centers, seen = [], set()
    for k in range(limit):
        prefix = "N21-center-v1|" + str(seed) + "|" + row["image_key"] + "|" + str(k) + "|"
        y = int.from_bytes(hashlib.sha256((prefix + "y").encode("utf-8")).digest(), "big") % height
        x = int.from_bytes(hashlib.sha256((prefix + "x").encode("utf-8")).digest(), "big") % width
        center = (y, x)
        if center not in seen:
            seen.add(center)
            centers.append(center)
            if len(centers) == target:
                return centers
    raise RuntimeError("Center candidates exhausted: " + str(limit) + "; target=" + str(target))


def _empty_stratum():
    return dict(denominator=0, same_cell_count=0,
                same_cell_histogram={"true": 0, "false": 0},
                rank_histogram={}, union_support_histogram={})


def _tally(stratum, group):
    stratum["denominator"] += 1
    same = group["same_native_2x2_cell"]
    stratum["same_cell_count"] += int(same)
    stratum["same_cell_histogram"]["true" if same else "false"] += 1
    for field, value in (("rank_histogram", group["rank"]),
                         ("union_support_histogram", group["union_support_count"])):
        key = str(value)
        stratum[field][key] = stratum[field].get(key, 0) + 1


def analyze_rows(selected, seed):
    """Return JSON-ready {cases, summary, decision} for the fixed ideal geometry.

    cases contains {row, groups} per selected image, in the supplied order.
    Each group records its center, nine exact rational sparse supports and
    coefficient rank. Summary denominators count centers, not images. Main is
    no_any_clamp: none of the neighborhood/first-resize/second-resize clamps may
    occur at any of the nine points. All thresholds use integer cross products.
    This verdict says nothing about actual F, LayerNorm, AP or innovation.
    """
    _seed(seed)
    _checked_rows(selected)
    summary = {name: _empty_stratum() for name in ("all", "no_any_clamp", "clamped")}
    cases = []
    for row in selected:
        case = dict(row=json.loads(json.dumps(row, allow_nan=False)), groups=[])
        for center in unique_centers(row, seed):
            trace = _native.trace_neighborhood(*(row[key] for key in SHAPE_KEYS), center)
            supports = [
                [dict(y=y, x=x, numerator=value.numerator, denominator=value.denominator)
                 for (y, x), value in sorted(sparse.items())]
                for sparse in trace["supports"]
            ]
            group = dict(center=list(center), points=[list(p) for p in trace["points"]],
                         supports=supports, union_support_count=trace["union_support_count"],
                         rank=trace["rank"], same_native_2x2_cell=trace["same_native_2x2_cell"],
                         clamp_flags=copy.deepcopy(trace["clamp_flags"]),
                         any_clamped=trace["any_clamped"])
            case["groups"].append(group)
            _tally(summary["all"], group)
            _tally(summary["clamped" if group["any_clamped"] else "no_any_clamp"], group)
        cases.append(case)
    total = summary["all"]["denominator"]
    main = summary["no_any_clamp"]["denominator"]
    same = summary["no_any_clamp"]["same_cell_count"]
    if 2 * main < total:
        decision = "INCONCLUSIVE_CLAMP_COVERAGE"
    elif 5 * same >= 4 * main:
        decision = "GEOMETRY_PREMISE_SUPPORTED"
    else:
        decision = "SHARED_FOUR_PREMISE_REJECTED"
    return dict(cases=cases, summary=summary, decision=decision)
