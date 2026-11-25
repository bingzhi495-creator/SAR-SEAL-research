"""Replay the paper's fixed and independent writers from local candidate rows."""
from argparse import ArgumentParser
from importlib.util import module_from_spec, spec_from_file_location
import json
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from pycocotools import mask as codec

E24 = "DFINE_SEG_EXTERNAL_BASELINE_FIT32_E24_20261112/e24"


def load_source(path):
    spec = spec_from_file_location(path.stem, path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def exact_selectors(code_root):
    base = code_root / E24
    fixed = load_source(base / "fit109_paired_point/pair_math.py")
    independent = load_source(base / "fit109_free_pairing/free_pairing.py")
    return fixed.choose_pairs, independent.choose_free


def valid_pairs(value, stock):
    if not isinstance(value, list) or len(value) > 32:
        raise ValueError("Expected at most 32 ordered endpoint pairs")
    for pair in value:
        if not isinstance(pair, list) or len(pair) != 2 or any(
            type(index) is not int or index < 0 or index >= stock.size for index in pair
        ):
            raise ValueError("Endpoint indices must be in the native row-major raster")
    pairs = np.asarray(value, dtype=np.intp).reshape(-1, 2)
    if len(np.unique(pairs)) != pairs.size or stock.flat[pairs[:, 0]].any() or not stock.flat[pairs[:, 1]].all():
        raise ValueError("Addition and removal endpoints must be unique and on the original mask sides")
    return pairs


def native_mask(candidate):
    if type(candidate["image_id"]) is not int or type(candidate["category_id"]) is not int:
        raise ValueError("COCO integer image and category IDs are required")
    score = candidate["score"]
    if type(score) not in (int, float) or not isfinite(score):
        raise ValueError("COCO finite detection score is required")
    rle = candidate["segmentation"]
    size = rle["size"]
    counts = rle["counts"]
    if (not isinstance(size, list) or len(size) != 2
            or any(type(v) is not int or v <= 0 for v in size)
            or not isinstance(counts, str) or not counts.isascii()):
        raise ValueError("Native mask must be compressed COCO RLE with integer height and width")
    return codec.decode(rle)


def write_mask(candidate, stock, additions, removals):
    if not len(additions):
        return candidate
    edited = stock.copy(order="F")
    edited.flat[additions] = 1
    edited.flat[removals] = 0
    rle = codec.encode(edited)
    return {**candidate, "segmentation": {"size": rle["size"], "counts": rle["counts"].decode("ascii")}}


def apply(row, select_fixed, select_independent):
    candidate = row["candidate"]
    stock = native_mask(candidate)
    pairs = valid_pairs(row["pairs"], stock)
    occupancy = np.asarray(row["occupancy"], dtype=np.float64)
    if not len(pairs) and occupancy.shape == (0,):
        occupancy = occupancy.reshape(0, 2)
    if (occupancy.shape != pairs.shape or not np.isfinite(occupancy).all()
            or np.any((occupancy < 0) | (occupancy > 1))):
        raise ValueError("One aligned probability pair in [0, 1] is required per endpoint pair")
    occupancy = occupancy.astype(np.float32)
    area_cap = min(8, int(np.count_nonzero(stock)) // 10, len(pairs))
    cap = row.get("cap", area_cap)
    if type(cap) is not int or not 0 <= cap <= area_cap:
        raise ValueError("Explicit cap exceeds the native-area and endpoint limits")
    fixed = select_fixed(occupancy[:, 0] - occupancy[:, 1], cap)
    k = len(fixed)
    free = np.asarray(select_independent(occupancy, cap, fixed_count=k), dtype=np.intp).reshape(-1, 2)
    return (
        write_mask(candidate, stock, pairs[fixed, 0], pairs[fixed, 1]),
        write_mask(candidate, stock, pairs[free[:, 0], 0], pairs[free[:, 1], 1]),
        k,
    )


def run(input_path, output, code_root):
    if output.exists():
        raise FileExistsError(f"Use a fresh output directory: {output}")
    select_fixed, select_independent = exact_selectors(code_root)
    rows = positives = swaps = 0
    with TemporaryDirectory(dir=output.parent) as staging:
        folder = Path(staging)
        with (
            input_path.open(encoding="utf-8") as source,
            (folder / "fixed.json").open("w", encoding="utf-8") as fixed,
            (folder / "independent.json").open("w", encoding="utf-8") as independent,
        ):
            fixed.write("[\n")
            independent.write("[\n")
            for number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    f, i, k = apply(json.loads(line), select_fixed, select_independent)
                    saved = (json.dumps(f, allow_nan=False), json.dumps(i, allow_nan=False))
                except (KeyError, ValueError, TypeError, OverflowError) as error:
                    raise ValueError(f"Invalid candidate at input line {number}: {error}") from error
                separator = ",\n" if rows else ""
                fixed.write(separator + saved[0])
                independent.write(separator + saved[1])
                rows += 1
                positives += k > 0
                swaps += k
            fixed.write("\n]\n")
            independent.write("\n]\n")
        if not rows:
            raise ValueError("At least one candidate is required")
        summary = {"status": "complete", "candidates": rows, "positive_candidates": positives,
                   "swaps_per_arm": swaps}
        (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        folder.rename(output)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Label-free candidate JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Fresh result directory")
    parser.add_argument("--code-root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="Repository research code directory")
    args = parser.parse_args()
    run(args.input, args.output, args.code_root)


if __name__ == "__main__":
    main()

