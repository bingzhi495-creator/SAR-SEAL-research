"""Apply frozen Phoenix endpoint probabilities and evaluate the observed equal-count masks."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from types import ModuleType

import numpy as np
from pycocotools import mask as codec


BUILDER = "da5ff070df83b3f4ab190baf494a71daee1f1fb9eff335aa255c2a20984b5021"
CFA_SCORER = "549dab0faf1660a3fc13b18520248d61c2b80d590d1a42331c098d32de6511bc"
CFA_SCORE_RESULT = "6a6d78baaea049130d6442d463dc1c69a2870aee12d3700d6826797cec95a10d"
CFA_SCORE_TERM = "f7c256f9903c2d359d563b406ee3c3e4992a20ff1b0d6454677072538cac1955"
INFER = "PHOENIX_RSDD74_EQUAL_COUNT_INFERENCE_COMPLETE"
INFER_SHA = "ea98a849ae181f06f3ffc0e41b29d2a91a7f867b4a4bcba6909c957f27db2b99"
AUTHOR = "13ee3996ef3d0292fab144316cc8e03b69c0ecfa"
WEIGHT_SHA = "9b6ba211e0d6d434085d3d5e379eb527d8b56ef413628f42cb587153cc51af7d"
EDIT = "PHOENIX_RSDD74_EQUAL_COUNT_EDITS_COMPLETE"
SCORE = "PHOENIX_RSDD74_EQUAL_COUNT_SCORE_COMPLETE"
PHOENIX = "phoenix"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def builder_api():
    path = Path(__file__).with_name("build_request.py")
    raw = path.read_bytes()
    require(sha256(raw).hexdigest() == BUILDER, "Frozen request builder required")
    module = ModuleType("verified_request")
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def ranked_mask(cand, prediction, tally):
    require(set(prediction) == {"position", "probabilities"} and prediction["position"] == cand["position"],
            "Phoenix probability row position differs")
    coords = np.asarray(cand["coords"])
    prob = np.asarray(prediction["probabilities"])
    k = cand["k"]
    rle = cand["segmentation"]
    stock = codec.decode(rle).astype(bool)
    require(stock.shape == (512, 512) and isinstance(k, int) and 0 < k <= min(8, int(stock.sum()) // 10, len(coords))
            and 32 <= stock.sum() < 1024 and coords.ndim == 2 and coords.shape[1] == 2 and prob.shape == coords.shape
            and coords.dtype.kind in "iu" and prob.dtype.kind in "fiu" and np.isfinite(prob).all()
            and np.all((prob >= 0) & (prob <= 1)) and np.all((coords >= 0) & (coords < stock.size))
            and np.unique(coords).size == coords.size and not stock.flat[coords[:, 0]].any()
            and stock.flat[coords[:, 1]].all(), "Invalid posterior, support, sign or frozen count")
    # The JSON numbers preserve the supplied floating values. Convert exact integers to avoid unsigned negation.
    prob = prob.astype(np.float64, copy=False)
    chosen = []
    for col, name in enumerate(("add", "drop")):
        scores = -prob[:, col] if col == 0 else prob[:, col]
        order = np.argsort(scores, kind="stable")
        independent = sorted(range(len(scores)), key=lambda i: (float(scores[i]), i))
        require(order.tolist() == independent, "Independent endpoint order differs")
        chosen.append(coords[order[:k], col])
        tally[name + "_tied_scores"] += len(scores) - len(np.unique(scores))
        tally[name + "_constant_candidate"] += int(np.all(scores == scores[0]))
        tally[name + "_cutoff_tie_candidates"] += int(k < len(order) and scores[order[k-1]] == scores[order[k]])
    out = stock.copy(order="C")
    out.flat[chosen[0]], out.flat[chosen[1]] = True, False
    adds = np.flatnonzero(out & ~stock)
    drops = np.flatnonzero(stock & ~out)
    require(np.array_equal(adds, np.sort(chosen[0])) and np.array_equal(drops, np.sort(chosen[1]))
            and len(adds) == len(drops) == k, "Actual changed pixels differ from signed selections")
    encoded = codec.encode(np.asfortranarray(out, dtype=np.uint8))
    return dict(swaps=k, rle=dict(size=encoded["size"], counts=encoded["counts"].decode("ascii")))


def input_chain(a, api):
    req, rsha, rtsha = api.observed(a.request, a.request_terminal, "request", api.STATUS, worker=BUILDER)
    infer, isha, itsha = api.observed(a.inference, a.inference_terminal, "inference", INFER, worker=INFER_SHA)
    require(req["protocol_sha256"] == infer["protocol_sha256"] == api.LOCK and req["target_labels_opened"] is False
            and req["counts"] == api.TOTALS and req["request_sha256"] == infer["request_sha256"]
            and infer["phase"] == "inference" and infer["target_labels_opened"] is False
            and infer["native_detector_forwards"] == infer["optimizer_updates"] == 0
            and infer["author_commit"] == AUTHOR and infer["weight_sha256"] == WEIGHT_SHA
            and (infer["seed"], infer["batch_size"], infer["refine_iters"]) == (5202, 2, 5)
            and {k:infer["runtime"][k] for k in ("FastGeodis", "numpy", "torch")} ==
                {"FastGeodis":"1.0.5", "numpy":"1.26.4", "torch":"2.10.0+cu130"},
            "Closed inference must bind the exact image-only request")
    payload = json.loads(api.pin(a.request / "REQUEST_PRIVATE.json", req["request_sha256"]))
    endpoint_data = api.pin(a.endpoints, infer["endpoint_sha256"])
    rows, ledger, lineage, previous = api.native_inputs(a)
    require(payload["status"] == api.REQUEST and payload["protocol_sha256"] == api.LOCK and
            req["input_lineage"] == lineage == payload["input_lineage"] == infer["input_lineage"] and
            req["source_replay"] == previous == payload["source_replay"] == infer["source_replay"], "CFA or request lineage differs")
    chain = dict(request_sha256=req["request_sha256"], request_result_sha256=rsha, request_terminal_sha256=rtsha,
                 inference_result_sha256=isha, inference_terminal_sha256=itsha, inference_worker_sha256=infer["worker_sha256"],
                 endpoint_sha256=infer["endpoint_sha256"], source_replay=previous, input_lineage=lineage,
                 inference_counts=infer["counts"], author_commit=AUTHOR, weight_sha256=WEIGHT_SHA)
    return rows, ledger, payload["images"], map(json.loads, endpoint_data.splitlines()), chain


def construct(a, api):
    from PIL import Image
    from io import BytesIO
    rows, ledger, images, endpoints, chain = input_chain(a, api)
    n, edits = Counter(), []
    for ordinal, (row, archive, image, scores) in enumerate(zip(rows, ledger, images, endpoints, strict=True), 1):
        item, audit, pred = row[:3]
        image_id = item["image_id"]
        require(image["image_id"] == scores["image_id"] == archive["image_id"] == image_id and
                image["image_file"] == f"image_{ordinal:03d}.png" and image["decoded_rgb_sha256"] == item["rgb_sha256"]
                and set(scores) == {"image_id", "candidates"}, "Original request row or endpoint image differs")
        raw_png = api.pin(a.request / image["image_file"], image["encoded_sha256"])
        with Image.open(BytesIO(raw_png)) as im:
            require(im.format == "PNG" and im.mode == "RGB" and im.size == (512, 512)
                    and api.digest(im.tobytes()) == item["rgb_sha256"], "Original native RGB differs")
        expected = sorted(map(int, archive["changes"]))
        require([c["position"] for c in image["candidates"]] == expected, "Positive-budget candidate roster differs")
        changes = {}
        for pos, cand, scores_row in zip(expected, image["candidates"], scores["candidates"], strict=True):
            info, det = audit["candidates"][pos-1], pred["coco"][pos-1]
            old = archive["changes"][str(pos)]
            require(set(old) == set(api.ARMS) and all(cand["k"] == edit["swaps"] for edit in old.values())
                    and cand["position"] == info["position"] == pos and cand["native_rank"] == info["native_rank"]
                    and cand["query_id"] == info["query_id"] and cand["score"] == info["score"] == det["score"]
                    and cand["segmentation"] == det["segmentation"], "Frozen stock, identity or budget differs")
            changes[str(pos)] = {**old, PHOENIX: ranked_mask(cand, scores_row, n)}
            n.update(positive_budget_candidates=1, swaps_per_arm=cand["k"], endpoint_pairs=len(cand["coords"]))
            n["default_area_rule_hits"] += int(int(codec.area(det["segmentation"])) >= .01 * 512 * 512)
        n.update(images=1, mapped_candidates=len(pred["coco"]), eligible_planes=len(audit["logit_ranks"]))
        n["image_encodes"] += int(bool(expected))
        n["decoder_batches"] += (len(expected) + 1) // 2
        edits.append(dict(image_id=image_id, changes=changes))
    require(all(n[k] == v for k, v in api.TOTALS.items()) and n["default_area_rule_hits"] == 0,
            "Frozen census, quota or default area-rule activation differs")
    actual = chain["inference_counts"]
    expected = dict(images=n["images"], image_encodes=n["image_encodes"], decoder_batches=n["decoder_batches"],
                    decoded_candidates=n["positive_budget_candidates"], candidate_refinement_iterations=5*n["positive_budget_candidates"],
                    native_count_sum=n["swaps_per_arm"], endpoint_pairs=n["endpoint_pairs"], default_area_rule_active=0,
                    equal_score_extra_add=n["add_tied_scores"], equal_score_extra_drop=n["drop_tied_scores"],
                    constant_add_scores=n["add_constant_candidate"], constant_drop_scores=n["drop_constant_candidate"])
    require(all(actual[k] == v for k, v in expected.items()), "Phoenix execution counts differ from validated request and endpoint values")
    return rows, edits, chain, dict(n)


def extended_evaluator(a, api):
    cfa = api.module(a.code / "INSTANCE_RSDD_CLASSICAL_FAIR_MATCH_20261123/score.py", CFA_SCORER)
    cfa.COMPARATORS = (*cfa.COMPARATORS, PHOENIX)
    cfa.BASELINES = (*cfa.BASELINES, PHOENIX)
    cfa.EDITED = (*cfa.EDITED, PHOENIX)
    cfa.ARMS = (*cfa.ARMS, PHOENIX)
    return cfa


def parity(view, archive, arms):
    require(all(view[k] == archive[k] for k in ("images", "annotations", "source_groups")),
            "CFA historical view census differs")
    for arm in arms:
        require(view["metrics_fraction"][arm] == archive["metrics_fraction"][arm],
                "Original full COCO vector differs for " + arm)
        new = view["geometry"]["group_equal_arms"][arm]
        old = archive["geometry"]["group_equal_arms"][arm]
        require(set(new) == set(old) and all(abs(new[f]-old[f]) < 1e-9 for f in new),
                "Original fixed-geometry vector differs for " + arm)


def score_view(ids, gt, blocks, polygons, dates, core, evaluator, reserve, h, cfa, contextual):
    view = cfa.score_view(ids, gt, blocks, polygons, dates, core, evaluator, reserve, h)
    geo = view["geometry"]
    geo["joint_exposure"]["all_seeds_all_three_comparators"] = geo["joint_exposure"].pop("all_seeds_both_comparators")
    scope = {i: block for i, block in blocks.items() if i in ids}
    part = dict(gt, images=[r for r in gt["images"] if r["id"] in scope],
                annotations=[r for r in gt["annotations"] if r["image_id"] in scope])
    _, anchors = evaluator.score_one(part, scope, "raw", anchors=True)
    # Compute paired (not differences of marginal intervals) Phoenix-minus-classical effects.
    # Three identical names reuse the existing three-seed code; only one deterministic contrast is retained.
    ctx = contextual.fixed_geometry(anchors, scope, polygons, dates, core, reserve, h)
    view["phoenix_minus_baselines"] = {
        b: dict(COCO_points=h.difference(view["metrics_fraction"][PHOENIX], view["metrics_fraction"][b]),
                fixed_geometry=ctx["per_seed"][b]["5217"]) for b in ("raw", "dynamic", "static")}
    signed = [v["AP"] for v in view["per_seed_differences_points"][PHOENIX].values()]
    bounds = geo["pooled"].get(PHOENIX, {}).get("continuous_iou_points", {}).get("interval_95")
    e = geo["exposure"][PHOENIX]["all_seeds"]
    exp = e["matches"] >= 20 and e["sensor_date_blocks"] >= 8
    pos = all(v is not None and v > 0 for v in signed)
    neg = all(v is not None and v < 0 for v in signed)
    view["phoenix_criteria"] = dict(common_changed_exposure=exp, all_AP_positive=pos, all_AP_negative=neg,
        occupancy_advantage=bool(exp and pos and bounds and bounds[0] > 0),
        phoenix_advantage=bool(exp and neg and bounds and bounds[1] < 0))
    del view["joint_criterion"]
    return view


def run(a, api):
    worker = api.digest(Path(__file__).read_bytes())
    edit_receipt = None
    if a.phase == "score":
        edit_receipt, ed_sha, term_sha = api.observed(a.edits, a.edits_terminal, "edits", EDIT, worker=worker)
        require(edit_receipt["protocol_sha256"] == api.LOCK and edit_receipt["target_labels_opened"] is False,
                "Closed label-free edit stage required")
    rows, edits, chain, counts = construct(a, api)
    encoded = b"".join(api.encode(row) for row in edits)
    edit_sha = api.digest(encoded)
    base = dict(phase=a.phase, protocol_sha256=api.LOCK, edits_sha256=edit_sha, **chain,
                counts=counts, worker_sha256=worker, detector_forwards=0, optimizer_updates=0)
    if a.phase == "edits":
        a.output.mkdir()
        api.write(a.output / "EDITS_PRIVATE.jsonl", encoded)
        return dict(**base, status=EDIT, independent_ranking_exact=True, archived_arms_exact=True, target_labels_opened=False)
    require(all(edit_receipt[k] == v for k, v in chain.items()) and edit_receipt["counts"] == counts
            and edit_receipt["independent_ranking_exact"] is edit_receipt["archived_arms_exact"] is True
            and edit_receipt["edits_sha256"] == edit_sha and api.pin(a.edits / "EDITS_PRIVATE.jsonl", edit_sha) == encoded,
            "Recomputed complete edits differ from closed label-free rankings")
    reference, _, _ = api.observed(a.cfa_score, a.cfa_score_terminal, "score",
        "INSTANCE_RSDD_CLASSICAL_FAIR_MATCH_SCORE_COMPLETE", CFA_SCORE_RESULT, CFA_SCORE_TERM, CFA_SCORER)
    require(reference["edits_sha256"] == api.CFA_EDITS and reference["input_lineage"] == chain["input_lineage"],
            "Expected historical fully scored policies")
    cfa = extended_evaluator(a, api)
    contextual = api.module(a.code / "INSTANCE_RSDD_CLASSICAL_FAIR_MATCH_20261123/score.py", CFA_SCORER)
    contextual.LEARNED = (PHOENIX,) * 3
    contextual.ARMS = ("raw", "dynamic", "static", PHOENIX)
    h = cfa.source(a.code / "POINTREND_SIGNED_QUOTA_SWAP_20261123/score_quota_swap.py", cfa.HISTORY_SHA)
    reserve = h.published_module(a.code / "INSTANCE_RSDD_ACQUISITION_RESERVE_20261123/score_reserve.py", h.RESERVE_SCORE)
    old = h.published_module(a.code / "INSTANCE_RSDD_FROZEN_HRSID_20261122/score_rsdd.py", reserve.HISTORIC)
    evaluator = old.module(a.code / "PSEG_RFDETR_FROZEN_SOURCE_HEAD_20261122/score.py", old.SCORE_SOURCE)
    adapter = old.module(a.code / "UEIKAP_RFDETR_CARRIER_20261122/score_rf.py", old.GEOMETRY_SOURCE)
    core = adapter.load_geometry(a.code / "UEIKAP_RESERVED_HULL_FROZEN_20261122/score.py")
    require(cfa.shapely.__version__ == "2.0.2" and cfa.shapely.geos_version_string == "3.11.2", "Original geometry runtime")
    blocks, budget, location = cfa.checked_blocks(rows, edits, adapter, core)
    require(budget["positive_budget_candidates"] == counts["positive_budget_candidates"] and budget["swaps_per_arm"] == 29900,
            "Rendered and evaluated signed counts differ")
    # No label data enter construction or verification of any arm.
    gt, rekeyed, polygons, sets, dates, labels_sha = h.views_and_labels(a, old, reserve, blocks)
    cache, views = {}, {}
    for name, ids in sets.items():
        key = frozenset(ids)
        if key not in cache:
            cache[key] = score_view(ids, gt, blocks, polygons, dates, core, evaluator, reserve, h, cfa, contextual)
        views[name] = cache[key]
        parity(views[name], reference["views"][name], ("raw", *api.ARMS))
        print(json.dumps(dict(scored_view=name, images=len(ids))), flush=True)
    a.output.mkdir()
    return dict(**base, status=SCORE, target_labels_opened=True, historical_arms_exact=True,
                author_labels_sha256=labels_sha, author_ids_rekeyed=rekeyed, historical_cfa_score_sha256=CFA_SCORE_RESULT,
                edits_result_sha256=ed_sha, edits_terminal_sha256=term_sha, actual_candidate_disagreement=location,
                archived_default_policy="native on the whole positive-count population", primary_view="full",
                primary_criteria=views["full"]["phoenix_criteria"], views=views)


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = p.add_subparsers(dest="phase", required=True)
    for phase in ("edits", "score"):
        sub = commands.add_parser(phase, allow_abbrev=False)
        names = ["code", "stock-root", "point-root", "cfa-verify", "cfa-terminal", "request", "request-terminal",
                 "inference", "inference-terminal", "endpoints", "protocol", "output"]
        if phase == "score":
            names += ["edits", "edits-terminal", "cfa-score", "cfa-score-terminal", "author-labels"]
        for name in names:
            sub.add_argument("--"+name, required=True, type=Path)
        sub.add_argument("--protocol-sha256", required=True)
    a = p.parse_args()
    for key, v in vars(a).items():
        if isinstance(v, Path):
            setattr(a, key, v.resolve(strict=key != "output"))
    api = builder_api()
    require(not a.output.exists() and api.digest(a.protocol.read_bytes()) == a.protocol_sha256 == api.LOCK,
            "Fresh output and fixed experiment lock required")
    a.cohort, a.support_root = 74, None
    start = perf_counter()
    result = run(a, api)
    result["seconds"] = perf_counter() - start
    result_sha = api.write(a.output / "RESULT_SAFE.json", api.encode(result))
    print(json.dumps(dict(status=result["status"], result_sha256=result_sha)), flush=True)


if __name__ == "__main__":
    main()
