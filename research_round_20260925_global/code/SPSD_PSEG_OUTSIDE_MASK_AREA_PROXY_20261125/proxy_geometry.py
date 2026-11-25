"""Exact geometry of the frozen native-outside IoU area proxy."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from time import perf_counter
from types import ModuleType

import numpy as np
from pycocotools import mask as codec

DIAGNOSE = "c44b34518377d3a932078f10fc37e9f1da683d0ef55418be3fbe7828000c0c1c"
POINTS = "87693897f873a2129764781611b0f7c35cadde51c0c4e830d7f8187c3b684049"
REFS = ("pseg", "spsd")
STATS = ("theta", "proxy_theta", "theta_gap", "outside_fp", "outside_fn",
         "relative_reference_area_error")


def sha(raw):
    return sha256(raw).hexdigest()


def pin(path, expected):
    raw = path.read_bytes()
    if sha(raw) != expected:
        raise ValueError("Fixed input changed: " + path.name)
    return raw


def load(path, expected):
    mod = ModuleType(path.stem)
    mod.__file__ = str(path)
    exec(compile(pin(path, expected), str(path), "exec"), vars(mod))
    return mod


def penalty(p, t, i, f, n):
    slope = i / (p + t)
    prox = (i + f) / (p + t + f - n)
    algebra = ((p + t - i) * f + i * n) / ((p + t) * (p + t + f - n))
    if abs(prox - slope - algebra) > 1e-12 or prox + 1e-12 < slope:
        raise ValueError("Failed outside-proxy break-even identity")
    return slope, prox, algebra


def hull_metrics(native, static, truth, positions, k):
    initial = codec.decode(native).astype(bool).ravel()
    base = codec.decode(static).astype(bool).ravel()
    y = codec.decode(truth).astype(bool).ravel()
    support = positions.ravel()
    if len(set(support)) != len(support) or initial.shape != base.shape or y.shape != base.shape:
        raise ValueError("Unexpected signed support/author extent")
    p, t, i = (int(x.sum()) for x in (base, y, base & y))
    delta = initial ^ base
    if int(initial.sum()) != p or delta.sum() != 2*k or delta[support].sum() != 2*k:
        raise ValueError("Exact original static signed write required")
    fp, fn = p - i, t - i
    valid = base[support]
    reference = y[support]
    f = fp - int((valid & ~reference).sum())
    n = fn - int((~valid & reference).sum())
    if f < 0 or n < 0:
        raise ValueError("Impossible outside support error counts")
    theta, proxy, gap = penalty(p, t, i, f, n)
    return (theta, proxy, gap, f, n, (f-n)/t), fp+fn


def main(a):
    start = perf_counter()
    pin(a.protocol, a.protocol_sha)
    d = load(a.diagnostic / "diagnose.py", DIAGNOSE)
    fam = d.load(a.diagnostic / "family.py", d.FAMILY)
    reader, _, _, _, native, records, groups = fam.closed(a)

    done = json.loads((a.points / "observed_points_retry/TERMINAL.json").read_bytes())
    d.need(done["actual_exit"] == 0 and done["output_footer_verified"]
           and done["result_sha256"] == POINTS, "Closed point preparation")
    root = a.points / "actual_retry"
    info = json.loads(pin(root / "RESULT_SAFE.json", POINTS))
    root = root / "source_gate"
    index = [json.loads(line) for line in pin(root/"POINT_INDEX_PRIVATE.jsonl",
                info["partitions"]["source_gate"]["point_index_sha256"]).splitlines()]
    d.need(len(index) == 183 and {r["image_id"] for r in index} == set(native)
           and not info["train_labels_opened"] and not info["test_labels_opened"],
           "Exact no-label coordinate source")

    truths = reader.labels(a, records)
    match = {name: d.stock_matches(truths[name], native) for name in REFS}
    d.need([len(match[name]) for name in REFS] == [311, 308], "Reproduced original native matches")
    sets = {name: {match[name][t["id"]]: t["segmentation"] for t in truths[name]["annotations"]
                   if t["id"] in match[name]} for name in REFS}
    observed = {ref: defaultdict(lambda: defaultdict(list)) for ref in REFS}
    counters = {ref: Counter() for ref in REFS}
    for row in index:
        iid = row["image_id"]
        group = groups[iid]
        raw = pin(root/row["filename"], row["sha256"])
        with np.load(BytesIO(raw), allow_pickle=False) as points:
            ranks, loc, offsets, sizes = (points[x] for x in (
                "native_ranks", "signed_coordinates", "point_offsets", "static_offsets"))
        d.need(row["source_group"] == group and len(raw) == row["bytes"]
               and loc.shape == (row["paired_sites"],2), "Exact coordinates and group")
        for j, rank in enumerate(ranks):
            k = int(sizes[j+1] - sizes[j])
            pos = native[iid][rank-1]
            d.need(k == pos["quota"], "Original static edit count")
            if k == 0:
                continue
            support = loc[offsets[j]:offsets[j+1]]
            for ref in REFS:
                target = sets[ref].get((iid, rank))
                if target is None:
                    continue
                values, errors = hull_metrics(pos["native"], pos["changes"]["static"], target, support, k)
                observed[ref][group][iid].append(values)
                counters[ref].update(matched_positive_hulls=1,
                                     has_outside_error=int(values[3]+values[4]>0),
                                     has_excess_penalty=int(values[2]>1e-12),
                                     outside_errors=int(values[3]+values[4]), all_static_errors=errors)

    result = {}
    for ref in REFS:
        group = observed[ref]
        rows = [v for images in group.values() for vals in images.values() for v in vals]
        group_rows = np.asarray([np.mean([np.mean(v, axis=0) for v in group[g].values()], axis=0)
                                 for g in sorted(group)])
        n = len(group_rows)
        gen = np.random.Generator(np.random.PCG64(5297))
        draws = gen.integers(n, size=(4096,n))
        boot = np.percentile(group_rows[draws].mean(axis=1), [2.5,97.5], axis=0).T
        result[ref] = dict(counters[ref], groups=n,
                           outside_error_fraction=(counters[ref]["outside_errors"]/counters[ref]["all_static_errors"]),
                           metrics={key: dict(hull_median=float(np.median(rows,axis=0)[index]),
                                              group_mean=float(group_rows.mean(axis=0)[index]),
                                              percentile95=boot[index].tolist())
                                    for index,key in enumerate(STATS)})
    status = "SPSD_PSEG_NATIVE_OUTSIDE_PROXY_GEOMETRY_COMPLETE"
    data = dict(status=status, protocol_sha256=a.protocol_sha, worker_sha256=sha(Path(__file__).read_bytes()),
                prior_score_sha256=fam.OLD_SCORE, point_result_sha256=POINTS, images=183,
                group_bootstrap_draws=4096, source_gate_labels_opened=True, historical_test_opened=False,
                fit_labels_opened=False, new_writer_opened=False, optimizer_updates=0, detector_forwards=0,
                ap_scored=False, references=result, elapsed_seconds=perf_counter()-start)
    a.output.mkdir()
    payload = (json.dumps(data,sort_keys=True,indent=2,allow_nan=False)+"\n").encode("utf-8")
    with (a.output / "RESULT_SAFE.json").open("xb") as out:
        out.write(payload)
    print(json.dumps(dict(status=status,result_sha256=sha(payload))), flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for flag in ("protocol", "diagnostic", "gate", "points", "labels", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--protocol-sha", required=True)
    main(parser.parse_args())
