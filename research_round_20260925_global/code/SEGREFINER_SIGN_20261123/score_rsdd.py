"""Decompose the observed SegRefiner-LR masks into signed native-grid actions."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
from time import perf_counter
from types import ModuleType

import numpy as np
from pycocotools import mask as codec
import shapely


LOCK_NAME = "SEGREFINER_SIGN_PRELOCK_20261123.md"
LOCK = "3e5918d16dad9f6cf0642e4c3222d16673f53a56835045de3b2687ab9776a7b1"
STOCK_SOURCE = "ccbcc1d253b10f5cbd1320c7eff29662b1d2f525b04de17443732d687df22734"
STOCK_SCORE = "ca904c131af51f2fb49daa6b5f30808917a37635c11de12815782f65c6c06eff"
STATUS = "RSDD74_SEGREFINER_SIGN_SCORE_COMPLETE"
ARMS = ("raw", "add_only", "delete_only", "stock_lr")
PAIRS = (("delete_only", "raw"), ("stock_lr", "add_only"), ("add_only", "raw"), ("stock_lr", "delete_only"))
PRIMARY = ("delete_only_minus_raw", "stock_lr_minus_add_only")
VIEWS = ("full", "date_separated")
EPS = 1e-8
AREA_FIELDS = ("true_added", "false_added", "true_deleted", "false_deleted")


def require(test, message):
    if not test:
        raise ValueError(message)


def digest(raw):
    return sha256(raw).hexdigest()


def pinned_module(path, expected):
    raw = path.read_bytes()
    require(digest(raw) == expected, "Archived scorer implementation differs")
    module = ModuleType(path.stem)
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def encode_mask(mask):
    packed = codec.encode(np.asfortranarray(mask, dtype=np.uint8))
    return dict(size=packed["size"], counts=packed["counts"].decode("ascii"))


def split_masks(native_rle, stock_rle):
    native = codec.decode(native_rle).astype(bool)
    stock = codec.decode(stock_rle).astype(bool)
    require(native.shape == stock.shape, "Frozen stock write must keep the native grid")
    added, deleted = stock & ~native, native & ~stock
    add_only, delete_only = native | stock, native & stock
    require(np.array_equal(add_only & ~native, added) and np.array_equal(native & ~delete_only, deleted)
            and np.array_equal(add_only & ~deleted, stock) and np.array_equal(delete_only | added, stock)
            and not np.any(added & deleted), "Signed partitions must reconstruct the exact observed stock pixels")
    add_rle, delete_rle = encode_mask(add_only), encode_mask(delete_only)
    require(np.array_equal(codec.decode(add_rle), add_only) and np.array_equal(codec.decode(delete_rle), delete_only),
            "Native-grid signed RLE round trip changed")
    return add_rle, delete_rle, int(added.sum()), int(deleted.sum())


def attach_signed_masks(blocks, areas, counts, gate):
    totals = Counter()
    for image, block in blocks.items():
        for slot, det in zip(block["ranks"], block["stock"], strict=True):
            area = areas[image][slot]
            opts = block["changes"].get(slot, {})
            totals["native_candidates"] += 1
            if "stock_lr" not in opts:
                require(area < gate and not {"add_only", "delete_only"} & opts.keys(),
                        "All and only original gate-positive candidates must have observed edits")
                totals["native_bypasses"] += 1
                continue
            require(area >= gate, "Observed author write crossed the frozen gate")
            native, stock = det["segmentation"], opts["stock_lr"]["rle"]
            add_rle, delete_rle, add, drop = split_masks(native, stock)
            opts.update(add_only={"rle": add_rle}, delete_only={"rle": delete_rle})
            totals.update(gate_candidates=1, added_pixels=add, deleted_pixels=drop,
                          candidates_with_additions=int(add > 0), candidates_with_deletions=int(drop > 0),
                          candidates_with_both=int(add > 0 and drop > 0), stock_masks_changed=int(add > 0 or drop > 0))
    require(totals["native_candidates"] == counts["native_candidates"] == 7346
            and totals["gate_candidates"] == counts["eligible_candidates"] == 3074
            and totals["native_bypasses"] == counts["bypassed_candidates"] == 4272
            and totals["stock_masks_changed"] == counts["eligible_masks_changed"]
            and totals["added_pixels"] - totals["deleted_pixels"]
            == counts["eligible_refined_pixels"] - counts["eligible_native_pixels"],
            "Signed action census differs from the closed author replay")
    return dict(totals)


def rles_for(anchor, blocks):
    image, slot, det = anchor
    raw = det["segmentation"]
    options = blocks[image]["changes"].get(slot, {})
    return [raw if arm == "raw" else options.get(arm, {"rle": raw})["rle"] for arm in ARMS]


def action_vector(rles, polygon, core, whole):
    vectors, areas, cache = [], [], {}
    for rle in rles:
        area = int(codec.area(rle))
        key = rle["counts"]
        if key not in cache:
            iou = core.geometric_iou(rle, polygon, shapely)
            cache[key] = whole.geometry_vector(iou, area, polygon.area)
        areas.append(area)
        vectors.append(cache[key])
    native, add_only, delete_only, stock = np.asarray(vectors)
    pixel_add = areas[1] - areas[0]
    pixel_del = areas[0] - areas[2]
    true_add = add_only[-1] - native[-1]
    true_del = native[-1] - delete_only[-1]
    values = np.array([true_add, pixel_add-true_add, true_del, pixel_del-true_del])
    require(np.min(values) >= -EPS and areas[3] == areas[0] + pixel_add - pixel_del
            and abs(stock[-1] - (native[-1] + true_add - true_del)) <= EPS,
            "Reference-area signed components must conserve observed intersection")
    values = np.maximum(values, 0)
    interaction = stock - add_only - delete_only + native
    normalized = 100*values/polygon.area
    joint_errors = np.array([normalized[1]-normalized[3], normalized[2]-normalized[0]])
    require(np.allclose(joint_errors, (stock-native)[1:3], rtol=0, atol=EPS),
            "Signed truth and background terms must exactly explain the joint FP/FN difference")
    return np.r_[values, normalized, interaction, joint_errors]


def changed_mask_exposure(anchors, blocks, dates):
    exposure = {pair: Counter() for pair in PAIRS}
    for anchor in anchors.values():
        parent = blocks[anchor[0]]["group"]
        masks = rles_for(anchor, blocks)
        for arm, base in PAIRS:
            if masks[ARMS.index(arm)] != masks[ARMS.index(base)]:
                exposure[(arm, base)][parent] += 1
    return {a + "_minus_" + b: dict(matches=sum(c.values()), parents=len(c),
                                    sensor_date_blocks=len({dates[parent] for parent in c}))
            for (a, b), c in exposure.items()}


def breakdown(anchors, blocks, polygons, dates, whole, deps):
    _, _, cfa, _, reserve, _, _, _, core = deps
    by_parent = defaultdict(list)
    for gid, anchor in anchors.items():
        parent = blocks[anchor[0]]["group"]
        masks = rles_for(anchor, blocks)
        by_parent[parent].append(action_vector(masks, polygons[gid], core, whole))
    fields = [s + "_px2" for s in AREA_FIELDS] + [s + "_per_truth_area_points" for s in AREA_FIELDS]
    fields += ["interaction_" + k for k in cfa.FIELDS]
    fields += ["joint_fp_per_truth_area_points", "joint_fn_per_truth_area_points"]
    parents = sorted(by_parent)
    require(parents, "At least one frozen polygon anchor required")
    means = np.stack([np.mean(by_parent[parent], axis=0) for parent in parents])
    bounds = reserve.bootstrap(means, [dates[parent] for parent in parents])
    values = {key: dict(value=float(center), interval_95=ci)
              for key, center, ci in zip(fields, means.mean(axis=0), bounds, strict=True)}
    return dict(hulls=len(anchors), represented_parents=len(parents), sensor_date_blocks=len({dates[p] for p in parents}),
                measured=values, changed_mask_exposure=changed_mask_exposure(anchors, blocks, dates))


def negative_criterion(geometry, changed):
    interval = geometry["interval_95"]
    exposure_supported = changed["matches"] >= 20 and changed["sensor_date_blocks"] >= 8
    negative = interval[1] < 0
    return dict(**changed, exposure_supported=exposure_supported, upper_interval_strictly_negative=negative,
                supported_negative=bool(exposure_supported and negative))


def score_view(ids, gt, blocks, areas, polygons, dates, reference, whole, deps, gate):
    _, _, cfa, _, reserve, _, evaluator, _, core = deps
    view = whole.score_view(ids, gt, blocks, polygons, dates, reference, deps)
    require(all(view["coverage"][arm] == reference["coverage"][arm] for arm in ("raw", "stock_lr")),
            "Observed stock and original matching coverage changed")
    subset = {i: block for i, block in blocks.items() if i in ids}
    partial = dict(gt, images=[im for im in gt["images"] if im["id"] in ids],
                   annotations=[ann for ann in gt["annotations"] if ann["image_id"] in ids])
    _, anchors = evaluator.score_one(partial, subset, "raw", anchors=True)
    valid = {gid: row for gid, row in anchors.items() if 32 <= areas[row[0]][row[1]] < 1024
             and polygons[gid].is_valid and polygons[gid].area > 0}
    exposed = {gid: row for gid, row in valid.items() if areas[row[0]][row[1]] >= gate}
    require(len(valid) == view["geometry"]["counts"]["eligible_matches"], "Fixed original geometry anchors differ")
    whole_sign = breakdown(valid, subset, polygons, dates, whole, deps)
    gate_sign = breakdown(exposed, subset, polygons, dates, whole, deps)
    geometry = whole.fixed_geometry(exposed, subset, polygons, {ann["id"]: ann for ann in partial["annotations"]},
                                   dates, cfa, core, reserve)
    prior = reference["gate_exposed_geometry"]
    require(all(geometry[k] == prior[k] for k in ("counts", "represented_groups", "sensor_date_blocks"))
            and all(abs(v-prior["group_equal_arms"][arm][field]) < EPS
                    for arm in ("raw", "stock_lr") for field, v in geometry["group_equal_arms"][arm].items()),
            "Previously published gate-exposed stock and original geometry differ")
    for name, contrast in view["contrasts"].items():
        contrast["criteria"] = {
            "all_fixed": negative_criterion(view["geometry"]["paired"][name]["continuous_iou_points"],
                                           whole_sign["changed_mask_exposure"][name]),
            "gate_exposed": negative_criterion(geometry["paired"][name]["continuous_iou_points"],
                                              gate_sign["changed_mask_exposure"][name]),
        }
    metrics = view["metrics_fraction"]
    interaction = {}
    for metric in metrics["raw"]:
        native, add, drop, joint = (metrics[arm][metric] for arm in ARMS)
        interaction[metric] = 100*(joint-add-drop+native) if None not in (native, add, drop, joint) else None
    view.update(gate_exposed_geometry=geometry, signed_geometry=whole_sign, gate_exposed_signed_geometry=gate_sign,
                descriptive_COCO_interaction_points=interaction,
                original_match_changed_mask_exposure=changed_mask_exposure(anchors, subset, dates),
                both_primary_deletion_contrasts_supported_adverse=all(
                    view["contrasts"][name]["criteria"]["gate_exposed"]["supported_negative"] for name in PRIMARY))
    return view


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("code", "run", "stock-root", "point-root", "author-labels", "protocol", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("score-terminal", "inference-terminal"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--protocol-sha256", required=True)
    a = parser.parse_args()
    for key, path in vars(a).items():
        if isinstance(path, Path):
            setattr(a, key, path.resolve(strict=key not in {"author_labels", "output"}))
    require(a.protocol.name == LOCK_NAME and a.protocol_sha256 == digest(a.protocol.read_bytes()) == LOCK
            and not a.output.exists(), "Published signed decomposition protocol and fresh output required")
    require((np.__version__, version("pycocotools"), shapely.__version__, shapely.geos_version_string)
            == ("1.26.4", "2.0.11", "2.0.2", "3.11.2"), "Historical mask and polygon runtime required")
    a.score_terminal = a.score_terminal or a.run / "observed_score" / "TERMINAL.json"
    a.inference_terminal = a.inference_terminal or a.run / "observed_inference" / "TERMINAL.json"
    a.cohort, a.support_root = 74, None
    start = perf_counter()
    previous = pinned_module(a.code / "SEGREFINER_STOCK_20261123/score_rsdd.py", STOCK_SOURCE)
    exposure_source = previous.exposure_source()
    whole, deps = exposure_source.sources(a)
    _, api, cfa, history, reserve, old, _, adapter, core = deps
    reference, score_sha, score_terminal = api.observed(a.run / "score", a.score_terminal, "score", previous.STATUS,
                                                        STOCK_SCORE, worker=STOCK_SOURCE)
    require(reference["protocol_sha256"] == previous.LOCK and reference["ledgers_verified_before_labels"] is True
            and reference["original_coco_and_local_geometry_exact"] is True, "Closed stock comparison differs")
    infer, inference_sha, infer_terminal = api.observed(
        a.run / "inference", a.inference_terminal, "inference", previous.INFER_STATUS,
        reference["inference_result_sha256"], reference["inference_terminal_sha256"], reference["inference_worker_sha256"])
    require(infer["target_labels_opened"] is False and infer["scientific_replay"] is True
            and infer["exact_original_bypass"] is True and infer["candidate_ordinals_preserved"] is True
            and infer["refined_masks_sha256"] == reference["refined_masks_sha256"],
            "Observed label-free masks with original bypass and ordinals required")
    loader = cfa.source(a.code / "POINTREND_SIGNED_QUOTA_SWAP_20261123/inputs.py", cfa.INPUTS_SHA)
    rows, lineage = loader.load_population(a)
    require(lineage == reference["input_lineage"] == infer["input_lineage"], "Original full candidate lineage differs")
    blocks, areas, positions = exposure_source.native_blocks(rows, adapter, core)
    stream = api.pin(a.run / "inference" / previous.LEDGER, reference["refined_masks_sha256"])
    counts = previous.attach_verified_masks(rows, blocks, areas, positions, stream, infer["counts"])
    require(counts == reference["counts"], "Observed stock run census differs")
    signed = attach_signed_masks(blocks, areas, counts, previous.GATE)
    print(json.dumps(dict(phase="signed_arms_verified", images=len(blocks), gate_candidates=signed["gate_candidates"],
                          target_labels_opened=False)), flush=True)

    a.author_labels = a.author_labels.resolve(strict=True)
    gt, rekeyed, polygons, sets, dates, label_sha = history.views_and_labels(a, old, reserve, blocks)
    require(label_sha == reference["author_labels_sha256"], "Original matched reference differs")
    whole.ARMS, whole.MATCHED, whole.PAIRS = ARMS, ("raw", "stock_lr"), PAIRS
    views = {}
    for name in VIEWS:
        views[name] = score_view(sets[name], gt, blocks, areas, polygons, dates, reference["views"][name],
                                 whole, deps, previous.GATE)
        print(json.dumps(dict(phase="scored_view", view=name, images=views[name]["images"])), flush=True)

    result = dict(
        status=STATUS, phase="score", worker_sha256=digest(Path(__file__).read_bytes()), protocol_sha256=a.protocol_sha256,
        stock_score_sha256=score_sha, stock_score_terminal_sha256=score_terminal, inference_result_sha256=inference_sha,
        inference_terminal_sha256=infer_terminal, refined_masks_sha256=reference["refined_masks_sha256"], input_lineage=lineage,
        author_labels_sha256=label_sha, author_ids_rekeyed=rekeyed, arms=list(ARMS), native_mask_gate=previous.GATE,
        masks_decomposed_before_labels=True, native_and_stock_metrics_exact=True, counts=signed,
        optimizer_updates=0, detector_forwards=0, segrefiner_forwards=0, target_labels_opened=True,
        primary_contrasts=list(PRIMARY), primary_geometry_scope="gate_exposed", bootstrap_seed=5217, bootstrap_draws=4096,
        cluster_unit="sensor/date; equal parent weight", views=views, score_seconds=perf_counter()-start,
    )
    a.output.mkdir()
    stamp = api.write(a.output / "RESULT_SAFE.json", api.encode(result))
    print(json.dumps(dict(status=STATUS, result_sha256=stamp)), flush=True)


if __name__ == "__main__":
    main()
