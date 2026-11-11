"""Measure OpenCV mask index differences on the Mask RT-DETR resize envelope."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from time import perf_counter


SIZES = tuple(range(480, 801, 32))
MAX_SOURCE = 3200


def indices(cv2, np, source, target, axis, mode):
    values = np.arange(source, dtype=np.uint16)
    shape = (1, source) if axis == "x" else (source, 1)
    size = (target, 1) if axis == "x" else (1, target)
    low = cv2.resize(values.astype(np.uint8).reshape(shape), size, interpolation=mode)
    high = cv2.resize((values >> 8).astype(np.uint8).reshape(shape), size, interpolation=mode)
    return low.ravel().astype(np.int32) + (high.ravel().astype(np.int32) << 8)


def run(cv2, np):
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    start = perf_counter()
    results = []
    for mode in (cv2.INTER_NEAREST, cv2.INTER_NEAREST_EXACT):
        for target in SIZES:
            counts = Counter()
            first_non_tie = None
            original_800 = None
            maximum_error = 0
            for source in range(1, MAX_SOURCE + 1):
                num = (2 * np.arange(target, dtype=np.int32) + 1) * source
                expected, remainder = np.divmod(num, 2 * target)
                actual = indices(cv2, np, source, target, "x", mode)
                vertical = indices(cv2, np, source, target, "y", mode)
                if np.any(actual < 0) or np.any(actual >= source):
                    raise RuntimeError("Index outside the source")
                if not np.array_equal(actual, vertical):
                    raise RuntimeError(f"Axis mismatch for {source} to {target}")
                error = actual - expected
                differs = error != 0
                tied = remainder == 0
                tie_diff = int(np.count_nonzero(differs & tied))
                non_tie_diff = int(np.count_nonzero(differs & ~tied))
                counts.update(positions=target, changed=tie_diff + non_tie_diff, ties=tie_diff,
                              non_ties=non_tie_diff, shapes_changed=int(tie_diff + non_tie_diff > 0))
                maximum_error = max(maximum_error, int(np.abs(error).max()))
                if first_non_tie is None and non_tie_diff:
                    j = int(np.flatnonzero(differs & ~tied)[0])
                    first_non_tie = [source, j, int(expected[j]), int(actual[j])]
                if source == 800:
                    original_800 = {"ties": tie_diff, "non_ties": non_tie_diff}
            results.append(dict(mode="nearest" if mode == cv2.INTER_NEAREST else "nearest_exact",
                                target=target, max_absolute_index_error=maximum_error,
                                first_non_tie_example=first_non_tie, original_source_800=original_800,
                                **counts))
    return dict(opencv=cv2.__version__, numpy=np.__version__, single_thread=True, opencl=False,
                max_source=MAX_SOURCE, target_sizes=SIZES, axes_verified_equal=True,
                per_target=results, elapsed_seconds=perf_counter() - start)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runtime = args.runtime.resolve(strict=True)
    sys.path.insert(0, str(runtime))
    import cv2
    import numpy as np

    if cv2.__version__ != "5.0.0" or np.__version__ != "2.2.6":
        raise RuntimeError("OpenCV 5.0.0 and NumPy 2.2.6 are required")
    if not all(Path(package.__file__).is_relative_to(runtime) for package in (cv2, np)):
        raise RuntimeError("Both packages must come from the isolated runtime")
    report = run(cv2, np)
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(report, out, sort_keys=True, indent=2, allow_nan=False)
        out.write("\n")
    print(json.dumps({k: v for k, v in report.items() if k != "per_target"}))


if __name__ == "__main__":
    main()

