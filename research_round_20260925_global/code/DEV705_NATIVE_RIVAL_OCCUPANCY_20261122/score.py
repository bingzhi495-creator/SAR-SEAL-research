"""Measure whole-DEV COCO parity and exact paired neighboring hull geometry."""

from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def near_rows(gt, blocks, witness, m, modules, shapely):
    valid, neutral, _ = modules.geometry.polygon_support(gt, modules.vector)
    need(not set(valid).intersection(neutral), "Disjoint valid and neutral reference shapes")
    by_image = defaultdict(dict)
    for annotation in gt["annotations"]:
        aid = annotation["id"]
        if aid in valid:
            by_image[annotation["image_id"]][aid] = valid[aid][0]
    keyed = {block[0]["image_id"]: block for block in blocks}
    result = []
    codec = m.old.codec
    for raw in witness:
        image, rank, aid = [raw[key] for key in ("image_id", "native_rank", "original_gt_id")]
        own = by_image[image][aid]
        others = [shape for gid, shape in by_image[image].items() if gid != aid]
        if min((own.distance(other) for other in others), default=float("inf")) > 4.0:
            continue
        exclusive = shapely.union_all(others).difference(own)
        item, coco, audit, changes, coverage = keyed[image]
        key = str(rank)
        stock = coco[rank-1]["segmentation"]
        raw_mask = codec.decode(stock).ravel(order="C").astype(bool)
        area = audit[rank-1]["mask_area"]
        scores = {}
        for arm in (m.control, *m.arms):
            edit = changes.get(key, {}).get(arm)
            final = edit["rle"] if edit else stock
            area_other = 0.0
            if edit:
                new_mask = codec.decode(final).ravel(order="C").astype(bool)
                added = np.flatnonzero(new_mask & ~raw_mask)
                need(len(added) == edit["swaps"], "Actual opposite sign edits")
                y, x = np.divmod(added, stock["size"][1])
                cells = shapely.box(x, y, x+1, y+1)
                area_other = float(shapely.area(shapely.intersection(cells, exclusive)).sum())
            scores[arm] = (area_other, m.score.continuous(final, area, own, modules)[0])
        flags = ("true_changed", "placebo_changed", "placebo_displaced", "true_placebo_changed")
        state = coverage.get(key, {})
        result.append(dict(group=m.score.as_group(item), arms=scores,
                           exposed=state.get("native_rival", False) and state.get("count", 0) > 0,
                           **{key: state.get(key, False) for key in flags}))
    need(len(result) == 47 and len({row["group"] for row in result}) == 14, "Identical original near population")
    return result


def evaluate(a, m, b, blocks, capture, module, pinned, score_sha):
    sys.path.append(str(b.shapely_site.resolve(strict=True)))
    import shapely

    need(shapely.__version__ == "2.0.2" and shapely.geos_version_string == "3.11.2", "Original vector runtime")
    adapter = module(a.code / "DIFFSAR_NATIVE_CPU_COCO_ADAPTER_DRAFT_20261107_01/worker.py", m.dev.SCORE_ADAPTER_SHA)
    geom = module(a.code / "MASK_RTDETR_DEV705_CONTINUOUS_AP_20261112/continuous_endpoint.py", m.score.GEOM)
    stock = module(a.code / "HRSID_FIT_HULL_CDF_20261112/stock_coco_match.py", m.score.MATCH)
    modules = geom.geometry_modules(a.code)
    reference = json.loads(pinned(a.reference, score_sha))
    expected = reference["geometry_points"]
    packed = [block[:4] for block in blocks]
    m.score.MODES = (*m.old.MODES, *m.arms)

    gt = m.dev.dev_json(b.dev_root, "DEV_ORIGINAL_GT_PRIVATE.json")
    m.dev.verify_labels(gt, m.dev.dev_json(b.dev_root, "DEV_EMPTY_RASTERS_PRIVATE.json"), [x[0] for x in blocks])
    coco = {}
    for arm in ("raw", m.control, *m.arms):
        rows = [row for _, pool, _, edits in packed for row in m.score.candidate_rows(pool, edits, arm)]
        coco[arm] = m.score.official(gt, rows, adapter)
    need(all(coco[arm] == reference["coco_fraction"][arm] for arm in ("raw", m.control)),
         "Full original raw and matched-count COCO parity")
    geometry, witness = m.score.geometry(gt, packed, stock, modules)
    need(geometry["counts"] == expected["counts"] and geometry["source_groups"] == 27
         and all(all(geometry[key][arm] == expected[key][arm] for arm in m.old.MODES)
                 for key in ("metrics", "strict90_transitions", "budgets")), "Original geometry and all old budgets")
    points = {arm: {"AP": 100 * metrics["AP"]} for arm, metrics in coco.items()}
    near = m.fit.near_contrasts(near_rows(gt, blocks, witness, m, modules, shapely), points)
    return dict(status="DFINE_DEV705_RIVAL_FROZEN_ACTUAL", source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                protocol_sha256=a.protocol_sha, DEV=True, FIT=False, TEST=False, GT_opened=True,
                hypothesis_motivated_by_this_DEV_population=True, original_parity=True, detector_forwards=0,
                detector_updates=0, point_optimizer_updates=0, capture_sha256=a.capture_sha,
                readback_sha256=a.readback_sha, edit_counts=capture["counts"], coco_fraction=coco,
                geometry_points=geometry, near=near, images=705, candidates=70500)
