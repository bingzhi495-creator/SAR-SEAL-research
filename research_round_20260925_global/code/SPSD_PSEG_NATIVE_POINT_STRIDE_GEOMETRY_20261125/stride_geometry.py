"""Measure fixed native-window endpoint geometry on the original 160-grid."""
from argparse import ArgumentParser
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from time import perf_counter

import numpy as np

GRID = 160
POINTS = "87693897f873a2129764781611b0f7c35cadde51c0c4e830d7f8187c3b684049"
WORKER = "926f87b220c4ba9f1a7074088a52641cd66c17d4016bac91fba20c84f450f766"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def checked(path, expected):
    payload = path.read_bytes()
    require(sha256(payload).hexdigest() == expected, "Changed fixed input " + path.name)
    return payload


def coordinates(ids, shape):
    h, w = shape
    y, x = np.divmod(ids.astype(np.int64), w)
    xx = np.clip((2 * x + 1) * GRID - w, 0, (GRID - 1) * 2 * w)
    yy = np.clip((2 * y + 1) * GRID - h, 0, (GRID - 1) * 2 * h)
    anchors = (yy // (2 * h)) * GRID + (xx // (2 * w))
    exact = yy * (2 * w * GRID + 1) + xx
    return anchors, exact


def summarize(root, split, frozen):
    directory = root / split
    data = checked(directory / "POINT_INDEX_PRIVATE.jsonl", frozen["point_index_sha256"])
    rows = [json.loads(line) for line in data.splitlines()]
    reference = frozen["counts"]
    require(len(rows) == len({i["image_id"] for i in rows}) == reference["images"], "Exact roster")
    totals = Counter(images=len(rows))
    cells_unique, coords_unique = [], []
    for row in rows:
        h, w = row["original_hw"]
        raw = checked(directory / row["filename"], row["sha256"])
        require(len(raw) == row["bytes"], "Exact point packet size")
        with np.load(BytesIO(raw), allow_pickle=False) as packet:
            rank = packet["native_ranks"]
            loc = packet["signed_coordinates"]
            offsets = packet["point_offsets"]
            static = packet["static_offsets"]
        n = len(rank)
        require(n == row["supported_candidates"] and len(set(rank)) == n
                and offsets.shape == static.shape == (n + 1,)
                and np.all(rank > 0) and np.all(rank <= 100)
                and loc.shape == (row["paired_sites"], 2)
                and offsets[0] == static[0] == 0 and offsets[-1] == len(loc)
                and np.all(loc >= 0) and np.all(loc < h * w), "Exact existing point geometry")
        sizes, budgets = np.diff(offsets), np.diff(static)
        require(np.all(sizes > 0) and np.all(sizes <= 32) and np.all(budgets >= 0)
                and np.all(budgets <= sizes), "Exact supported and quota structure")
        totals.update(supported=n, paired_sites=len(loc), static_exchanges=int(static[-1]),
                      static_positive_candidates=int((budgets > 0).sum()))
        all_cells, all_exact = coordinates(loc, (h, w))
        for k in np.flatnonzero(budgets):
            sl = slice(offsets[k], offsets[k+1])
            a, b = all_cells[sl].T, all_exact[sl].T
            shared_cells = np.intersect1d(*a)
            shared_coordinates = np.intersect1d(*b)
            ca, cb = np.isin(a, shared_cells).sum(axis=1)
            ea, eb = np.isin(b, shared_coordinates).sum(axis=1)
            totals.update(positive_points_per_side=len(a[0]), positive_with_shared_anchor=int(bool(len(shared_cells))),
                          positive_with_coincident_point=int(bool(len(shared_coordinates))),
                          addition_shared_anchor=int(ca), removal_shared_anchor=int(cb),
                          addition_coincident=int(ea), removal_coincident=int(eb))
            cells_unique.append([len(np.unique(v)) for v in a])
            coords_unique.append([len(np.unique(v)) for v in b])
    for field in ("images", "supported", "paired_sites", "static_exchanges", "static_positive_candidates"):
        require(totals[field] == reference[field], "Full closed count parity on " + field)
    return dict(totals, median_unique_anchor_cells=np.median(cells_unique, axis=0).tolist(),
                median_unique_exact_coordinates=np.median(coords_unique, axis=0).tolist())


def main(a):
    start = perf_counter()
    lock = a.protocol.read_bytes()
    require(sha256(lock).hexdigest() == a.protocol_sha, "Locked geometry protocol")
    term = json.loads((a.points / "observed_points_retry/TERMINAL.json").read_bytes())
    require(term["actual_exit"] == 0 and term["output_footer_verified"] is True
            and term["worker_sha256"] == WORKER and term["result_sha256"] == POINTS, "Closed point artifact")
    root = a.points / "actual_retry"
    info = json.loads(checked(root / "RESULT_SAFE.json", POINTS))
    require(info["worker_sha256"] == WORKER and not any(info[x] for x in
            ("test_images_opened", "test_labels_opened", "train_labels_opened")), "Scope of original preparation")
    status = "SPSD_PSEG_NATIVE_POINT_STRIDE_GEOMETRY_COMPLETE"
    result = dict(status=status, point_result_sha256=POINTS, grid=[GRID, GRID],
                  protocol_sha256=a.protocol_sha, worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                  fit=summarize(root, "fit", info["partitions"]["fit"]),
                  source_gate=summarize(root, "source_gate", info["partitions"]["source_gate"]),
                  detector_forwards=0, label_access=False, new_feature_access=False)
    result["elapsed_seconds"] = perf_counter() - start
    a.output.mkdir()
    payload = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with (a.output / "RESULT_SAFE.json").open("xb") as out:
        out.write(payload)
    print(json.dumps(dict(status=status, result_sha256=sha256(payload).hexdigest())), flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for flag in ("points", "protocol", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--protocol-sha", required=True)
    main(parser.parse_args())
