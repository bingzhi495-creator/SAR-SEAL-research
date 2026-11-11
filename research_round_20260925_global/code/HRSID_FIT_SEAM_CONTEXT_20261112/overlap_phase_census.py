"""Lock and verify native HRSID FIT overlap pairs without annotations."""

from argparse import ArgumentParser
from collections import Counter, defaultdict
from hashlib import sha256
from io import BytesIO
from itertools import combinations
import json
from pathlib import Path
import re
from time import perf_counter

import numpy as np
from PIL import Image


LAYOUT_SHA = "a7e4d2a7ff9e8ff238adbf94a1d0397553d8746e2558c316db4e417f86f8fec9"
SALT = "HRSID-fit-overlap-mask-phase-v1|"
WINDOW = re.compile(r"(P[0-9]+)_([0-9]+)_([0-9]+)_([0-9]+)_([0-9]+)\.png")
SIDE = 800
MASK_SIDE = 144
MAX_PAIRS = 36
FIELDS = ("native_basename", "encoded_sha256", "decoded_rgb_sha256", "encoded_bytes")


def need(valid, message):
    if not valid:
        raise ValueError(message)


def exact(path, digest):
    data = path.read_bytes()
    need(sha256(data).hexdigest() == digest, "Unexpected digest for " + path.name)
    return data


def write_new(path, obj):
    data = (json.dumps(obj, sort_keys=True, allow_nan=False, indent=2) + "\n").encode()
    with path.open("xb") as handle:
        handle.write(data)
    return sha256(data).hexdigest()


def rank(key):
    return sha256((SALT + key).encode()).digest(), key


def parse_window(name):
    match = WINDOW.fullmatch(name)
    if match is None:
        return None
    group, *numbers = match.groups()
    top, bottom, left, right = tuple(map(int, numbers))
    need(bottom - top == right - left == SIDE, "Unexpected explicit crop dimensions")
    return group, (top, bottom, left, right)


def pair_geometry(a, b):
    """Return shared crop rectangle and signed native / mask-grid shifts."""
    dy, dx = b[0] - a[0], b[2] - a[2]
    overlap = (max(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), min(a[3], b[3]))
    if ((dy == dx == 0) or overlap[1] <= overlap[0] or overlap[3] <= overlap[2]
            or dy * MASK_SIDE % SIDE or dx * MASK_SIDE % SIDE):
        return None
    return overlap, (dy, dx), (dy * MASK_SIDE // SIDE, dx * MASK_SIDE // SIDE)


def stratum(shift):
    return f"dy={abs(shift[0])},dx={abs(shift[1])}"


def lock(args):
    started = perf_counter()
    records = json.loads(exact(args.native_layout, LAYOUT_SHA).decode("utf-8-sig"))["images"]
    names = [r["native_basename"] for r in records]
    need(len(records) == len(set(names)) == 2937, "Unexpected FIT image census")
    groups = defaultdict(list)
    windows = 0
    for rec in records:
        parsed = parse_window(rec["native_basename"])
        if parsed is None:
            continue
        group, window = parsed
        groups[group].append((rec, window))
        windows += 1
    candidates = Counter()
    candidates_by_group = defaultdict(list)
    for group, crops in groups.items():
        ordered = sorted(crops, key=lambda x: x[0]["native_basename"])
        for (a, aw), (b, bw) in combinations(ordered, 2):
            result = pair_geometry(aw, bw)
            if result is None:
                continue
            overlap, shift, mask_shift = result
            candidates[stratum(shift)] += 1
            pair = [dict({k: r[k] for k in FIELDS}, window=w) for r, w in ((a, aw), (b, bw))]
            token = pair[0]["native_basename"] + "|" + pair[1]["native_basename"]
            candidates_by_group[group].append((rank(token), dict(
                group=group, pair=pair, overlap=overlap, shift_yx=shift, mask_shift_yx=mask_shift,
            )))
    pairs = []
    for ordinal, group in enumerate(sorted(candidates_by_group, key=rank)[:MAX_PAIRS], start=1):
        _, record = min(candidates_by_group[group], key=lambda x: x[0])
        pairs.append(dict(ordinal=ordinal, **record))
    metrics = dict(
        native_filenames=len(records),
        explicit_rectangles=windows,
        groups_with_explicit_rectangles=len(groups),
        geometric_candidate_pairs=sum(candidates.values()),
        geometric_candidate_groups=len(candidates_by_group),
        geometric_offset_strata=dict(candidates),
        selected_pairs=len(pairs),
        selected_offset_strata=dict(Counter(stratum(x["shift_yx"]) for x in pairs)),
    )
    sha = write_new(args.output, dict(
        status="HRSID_FIT_OVERLAP_MASK_PHASE_ROSTER_PRELOCKED",
        FIT=True, DEV=False, TEST=False, GT_opened=False, PNG_contents_opened=0,
        native_layout_sha256=LAYOUT_SHA, selection_salt=SALT, metrics=metrics, pairs=pairs,
    ))
    print(json.dumps(dict(metrics, lock_sha256=sha, worker_seconds=perf_counter() - started), sort_keys=True))


def load_image(directory, expected):
    name = expected["native_basename"]
    need(Path(name).name == name, "Unsafe FIT filename")
    raw = exact(directory / name, expected["encoded_sha256"])
    need(len(raw) == expected["encoded_bytes"], "Encoded byte count differs")
    with Image.open(BytesIO(raw)) as im:
        need(im.format == "PNG" and im.size == (SIDE, SIDE), "Unexpected encoded FIT image")
        rgb = np.array(im.convert("RGB"), dtype=np.uint8)
    need(sha256(rgb.tobytes()).hexdigest() == expected["decoded_rgb_sha256"], "Decoded RGB differs")
    return rgb


def overlap_pixels(rgb, window, overlap):
    top, _, left, _ = window
    ot, ob, ol, or_ = overlap
    return rgb[ot-top:ob-top, ol-left:or_-left]


def check(args):
    started = perf_counter()
    roster = json.loads(exact(args.lock, args.lock_sha256))
    pairs = roster["pairs"]
    need(roster["status"] == "HRSID_FIT_OVERLAP_MASK_PHASE_ROSTER_PRELOCKED" and
         roster["native_layout_sha256"] == LAYOUT_SHA and roster["PNG_contents_opened"] == 0 and
         len(pairs) == len({x["group"] for x in pairs}) <= MAX_PAIRS,
         "Expected exact closed, label-free prelock")
    passed = Counter()
    failed = Counter()
    total_pixels = nonidentical_pixels = largest_pair_difference = 0
    for ordinal, row in enumerate(pairs, start=1):
        a, b = row["pair"]
        aw, bw = tuple(a["window"]), tuple(b["window"])
        need(row["ordinal"] == ordinal and
             parse_window(a["native_basename"]) == (row["group"], aw) and
             parse_window(b["native_basename"]) == (row["group"], bw), "Prelocked source identity differs")
        expected = tuple(row["overlap"]), tuple(row["shift_yx"]), tuple(row["mask_shift_yx"])
        need(pair_geometry(aw, bw) == expected, "Prelocked crop geometry differs")
        left, right = [
            overlap_pixels(load_image(args.fit_train, image), image["window"], row["overlap"])
            for image in (a, b)
        ]
        need(left.shape == right.shape and left.size > 0, "Overlap crop shape differs")
        difference = int(np.count_nonzero(np.any(left != right, axis=2)))
        (failed if difference else passed)[stratum(row["shift_yx"])] += 1
        total_pixels += left.shape[0] * left.shape[1]
        nonidentical_pixels += difference
        largest_pair_difference = max(largest_pair_difference, difference)
    result = dict(
        status="HRSID_FIT_OVERLAP_MASK_PHASE_CENSUS_ACTUAL",
        FIT=True, DEV=False, TEST=False, GT_opened=False, model_forwards=0, optimizer_updates=0,
        lock_sha256=args.lock_sha256, native_layout_sha256=LAYOUT_SHA, **roster["metrics"],
        verified_source_distinct_pairs=len(pairs), exact_RGB_pairs=sum(passed.values()),
        exact_RGB_groups=sum(passed.values()), nonidentical_RGB_pairs=sum(failed.values()),
        exact_offset_strata=dict(passed), nonidentical_offset_strata=dict(failed),
        decoded_PNG_count=2 * len(pairs), total_overlap_pixels=total_pixels,
        nonidentical_overlap_pixels=nonidentical_pixels,
        largest_single_pair_nonidentical_pixels=largest_pair_difference,
        worker_seconds=perf_counter() - started,
    )
    digest = write_new(args.output, result)
    print(json.dumps(dict(result, result_sha256=digest), sort_keys=True))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(required=True)
    make = subs.add_parser("lock")
    make.add_argument("--native-layout", type=Path, required=True)
    make.add_argument("--output", type=Path, required=True)
    make.set_defaults(handler=lock)
    verify = subs.add_parser("check")
    verify.add_argument("--lock", type=Path, required=True)
    verify.add_argument("--lock-sha256", required=True)
    verify.add_argument("--fit-train", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)
    verify.set_defaults(handler=check)
    args = parser.parse_args()
    args.handler(args)
