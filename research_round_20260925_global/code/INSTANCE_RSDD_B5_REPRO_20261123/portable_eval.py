"""Run frozen detector capture, label-free point replay and polygon scoring on the 74-chip reserve."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
from functools import partial
from hashlib import sha256
from importlib.metadata import version
from io import StringIO
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace


RESERVE = "INSTANCE_RSDD_ACQUISITION_RESERVE_20261123"
NATIVE = "INSTANCE_RSDD_FROZEN_HRSID_20261122"
UEIKAP = "UEIKAP_RFDETR_CARRIER_20261122"
ROSTER = "2614e567e9ef1b2c791c1b357b03adfe7302ece53e911c1963c40e623ad405fc"
CHECKPOINT = "011bd30ed056c871be1d61e14d71b3b2b258330a9d20cf2f164ca4a7c4f23c11"
TRAINING_COMMIT = "04308c52bd2b4730f1474335c6316f6b8a11762e"
POINT_HEAD = "3a60f658149f4262451cd9118902bbe52dc83345d9fbcd23973a11dc473fcdf1"
CAPTURE_SOURCE = "6549dd151178a17146de518fbe16a3a045d90c5b4ed8345d4d8e272db0e51c2d"
REPLAY_SOURCE = "7238e6c5ca52133d51173388b01a28c48ec2a81dccd66139f6b7b2c4cbd8d072"
SCORE_SOURCE = "5ff671b2b83bb12e322b0c766897bf3cdacaedd233bf0d6edcdfcccf2c938df4"
MANIFEST = "IMAGE_MANIFEST_PRIVATE.json"
PREDICTIONS = "PREDICTIONS_PRIVATE.jsonl"
AUDIT = "CANDIDATE_AUDIT_PRIVATE.jsonl"
EDITS = "EDITS_PRIVATE.jsonl"
RESULT = "RESULT_SAFE.json"


def check(condition, message):
    if not condition:
        raise ValueError(message)


def pin(path, digest):
    raw = path.read_bytes()
    check(sha256(raw).hexdigest() == digest, f"SHA-256 mismatch for {path.name}")
    return raw


def module(path, digest):
    loaded = ModuleType(path.stem)
    loaded.__file__ = str(path)
    exec(compile(pin(path, digest), str(path), "exec"), loaded.__dict__)
    return loaded


def load_intake(args, native):
    roster = csv.DictReader(StringIO(pin(args.roster, ROSTER).decode("utf-8")))
    rows = list(roster)
    images = [dict(id=int(r["author_train_image_id"]), file_name=r["file_name"], height=512, width=512)
              for r in rows]
    raw = (args.intake / MANIFEST).read_bytes()
    items = native.roster({"images": images}, json.loads(raw), size=74)
    for item, row in zip(items, rows, strict=True):
        check((item["acquisition_group"], item["sensor"]) == (int(row["paper_scene"]), row["sensor"]),
              "Image-to-parent and sensor assignment differs from the frozen roster")
    check(Counter(r["sensor"] for r in items) == {"GF-3": 42, "TerraSAR-X": 32}
          and len({r["acquisition_group"] for r in items}) == 13
          and len({r["rgb_sha256"] for r in items}) == 74, "Reserve sensor, parent or RGB census differs")
    return items, sha256(raw).hexdigest()


def detector_components(code, checkpoint):
    """Resolve the published transform, EMA loader, checkpoint and top-100 postprocessor."""
    carrier = module(code / "RFDETR_CENTER5202_FULL_CAPTURE_20261120/capture.py",
                     "97449f58d1ad1a62c52c6f100fc872690e51d465e85e241660befba7a9690a71")
    post = module(code / "HRSID_FIT_NATIVE_QUERY_RETENTION_20261109/fit_query_export.py",
                  "6a2d7e38d69b82ef3df790a5c359c408b1e74778c39672b47a62c4dc86cabbf5")
    filtered = module(code / "HRSID_COCO_E24_FILTERED_DEV705_NATIVE_EVAL_20261109/hrsid_native_dev705_filtered.py",
                      "9a69500d10d5b0a371ea504f1ba89b7335aca130c1a28d96c4f93a87b2a4b9c0")
    base = filtered.load_original_evaluator(code / "HRSID_COCO_E24_DEV705_NATIVE_EVAL_20261109/hrsid_native_dev705.py")
    base.FIT_AUDIT = code / "HRSID_OFFICIAL_PNG_FIT_AUDIT_20261109/hrsid_png_fit_audit.py"
    base.RECIPE = code.parent / "configs/RFDETR_SEG_NANO_BASELINE_576_E100_20261107.yaml"
    filtered.TRAINING_PROTOCOL = TRAINING_COMMIT
    # The source model loader verifies checkpoint bytes before deserialization, then checks
    # the repaired E24 marker, EMA clock and every model tensor against the built architecture.
    base.check_endpoint = partial(filtered.check_filtered_endpoint, original_check=base.check_endpoint,
                                  qualification=filtered.AcceptedEndpoint(CHECKPOINT, 0))
    core = SimpleNamespace(CHECKPOINTS={"center": checkpoint}, TRAIN_PROTOCOL=TRAINING_COMMIT)
    return carrier, core, base, post


def make_detector(code, checkpoint, output, torch, device="cuda"):
    """Load the frozen center E24 EMA and native val576 transform on CPU or CUDA."""
    check(device in {"cpu", "cuda"}, "Use CPU or CUDA")
    carrier, core, base, _ = detector_components(code, checkpoint)
    paths = SimpleNamespace(root=code, output=output, mode="dev" if device == "cuda" else "fit")
    return carrier.model_and_transform(paths, core, base, torch)


def capture(args, items, manifest_sha, native):
    import torch

    native.install_guard(args.intake, items)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    from rfdetr.models.postprocess import PostProcess  # installs historical RF-DETR CUDA precision policy

    check(torch.get_float32_matmul_precision() == "high" and torch.backends.cuda.matmul.allow_tf32,
          "Expected RF-DETR high precision CUDA matmul policy")
    carrier, core, base, post = detector_components(args.code, args.checkpoint)
    # Replace only historical receipt/path discovery in this fresh module instance.
    carrier.load_core = lambda _: core
    carrier.setup = lambda args, core: (base, post)
    common = native.reference(args.code / UEIKAP / "capture.py")
    common.exact_source = lambda _: carrier
    # The public code directory occupies the former root argument. No path is looked up through it.
    paths = SimpleNamespace(private_root=args.code, intake=args.intake, output=args.output)
    result = native.capture(paths, items, torch, common)
    check(not torch.backends.cudnn.allow_tf32 and result["model_forward_calls"] == len(items),
          "Expected one native forward per image and the original cuDNN precision")
    return dict(result, phase="capture", status="B5_RESERVE74_CAPTURE_COMPLETE", images=len(items),
                checkpoint_sha256=CHECKPOINT, manifest_sha256=manifest_sha, target_labels_opened=False)


def read_capture(args, items, manifest_sha):
    import numpy as np
    from pycocotools import mask as codec

    native = module(args.code / NATIVE / "capture_instance_rsdd.py", CAPTURE_SOURCE)
    summary = json.loads((args.capture / RESULT).read_bytes())
    check(summary["status"] == "B5_RESERVE74_CAPTURE_COMPLETE" and summary["manifest_sha256"] == manifest_sha
          and summary["checkpoint_sha256"] == CHECKPOINT and summary["roster_sha256"] == ROSTER
          and summary["target_labels_opened"] is False and summary["model_forward_calls"] == len(items),
          "Capture cohort or detector differs")
    preds, audits = ([json.loads(line) for line in pin(args.capture / name, summary[key]).splitlines()]
                     for name, key in ((PREDICTIONS, "predictions_sha256"), (AUDIT, "audit_sha256")))
    totals = dict(mapped=0, eligible=0, overrides=0, rank_gaps=0)
    for item, pred, row in zip(items, preds, audits, strict=True):
        i = item["image_id"]
        check(i == pred["image_id"] == row["image_id"] and row["image_sha256"] == item["image_ref"]["sha256"]
              and all(row[k] == item[k] for k in ("acquisition_group", "sensor", "source_image_name", "rgb_sha256", "original_hw")),
              "Capture row differs from the materialized author image")
        probs = np.asarray(row["raw_query_probabilities"], np.float32)
        boxes = np.asarray(row["raw_boxes_cxcywh"], np.float32)
        check(probs.shape == (164, 2) and ((0 <= probs) & (probs <= 1)).all()
              and boxes.shape == (164, 4) and np.isfinite(boxes).all(), "Invalid full detector queries")
        best = np.argsort(-probs.ravel(), kind="stable")[:100]
        retained = [(r, int(q)//2) for r, q in enumerate(best, 1) if q % 2 == 0]
        ranks, ids = [r for r, _ in retained], [q for _, q in retained]
        xyxy = native.native_xyxy(boxes, item["original_hw"])[ids].tolist()
        check(pred["native_ranks"] == row["native_ranks"] == ranks and pred["raw_query_count"] == 164
              and row["query_ids"] == ids and row["native_boxes_xyxy"] == xyxy
              and pred["native_count"] == 100 and pred["unmapped_slot1_count"] == 100 - len(ids),
              "Original joint top-100 ranks, IDs or boxes differ")
        eligible = []
        for pos, ((rank, q), box, co, candidate) in enumerate(zip(retained, xyxy, pred["coco"], row["candidates"], strict=True), 1):
            check(co["image_id"] == i and co["category_id"] == 1 and co["segmentation"]["size"] == item["original_hw"]
                  and co["score"] == float(probs[q, 0]), "Unexpected native score, category or mask")
            area = int(codec.area(co["segmentation"]))
            check(candidate == dict(position=pos, native_rank=rank, query_id=q, score=co["score"], mask_area=area,
                                    box_xyxy=box), "Unexpected candidate-to-query mapping")
            if 32 <= area < 1024:
                eligible.append((rank, q))
        check(eligible == list(zip(row["logit_ranks"], row["logit_query_ids"], strict=True)),
              "Eligible plane identities differ")
        overrides = [(e["native_rank"], e["query_id"]) for e in row["posterior_overrides"]]
        check(len(overrides) == len(set(overrides)) and set(overrides) <= set(eligible), "Invalid CUDA overrides")
        native.count_rows(totals, retained, eligible, overrides)
    check(totals == summary["counts"], "Candidate totals differ")
    return summary, list(zip(audits, preds, strict=True))


def initialize_replay(code, intake, items, capture_path, point_head):
    global replay_module
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    native = module(code / NATIVE / "capture_instance_rsdd.py", CAPTURE_SOURCE)
    native.install_guard(intake, items)
    replay_module = module(code / RESERVE / "replay_reserve.py", REPLAY_SOURCE)
    edit = module(code / UEIKAP / "edit_rf.py", replay_module.EDIT)
    base = edit.load_module(code / edit.BASE / "replay.py", edit.REPLAY_SHA)
    methods = base.methods(code)
    step = base.read_source(code / base.SPOT / "fit_step.py", methods.readout.PINS["fit_step.py"])
    plan = base.read_source(code / base.RF / "fit_plan.py", methods.readout.PINS["fit_plan.py"])
    trainer = step.load_trainer(code / base.BASE / "fit109_paired_point/train_point.py")
    heads, center, scale = plan.restore_bundle(point_head, POINT_HEAD, trainer, step)
    predict = lambda features: step.predict(heads["S"], features, center, scale)
    image_paths = SimpleNamespace(images=intake / "images", capture=capture_path)
    replay_module.worker = (edit, base, methods, predict, image_paths, False)


def replay_both(pair):
    worker = replay_module.worker
    replay_module.worker = (*worker[:-1], False)
    first = replay_module.replay(pair)
    replay_module.worker = (*worker[:-1], True)
    second = replay_module.replay(pair)
    check(first == second, "Independent candidate-support and edit decisions differ")
    return first


def replay(args, items, manifest_sha, native):
    native.install_guard(args.intake, items)
    summary, pairs = read_capture(args, items, manifest_sha)
    # Also validate rows with no eligible point support, which replay_image would skip.
    for item in items:
        native.decode_native(args.intake, item)
    args.output.mkdir()
    total, digest = Counter(), sha256()
    initializer_args = (args.code, args.intake, items, args.capture, args.point_head)
    with (args.output / EDITS).open("xb") as stream, ProcessPoolExecutor(
            max_workers=args.workers, initializer=initialize_replay, initargs=initializer_args) as pool:
        for item, (changes, counts) in zip(items, pool.map(replay_both, pairs, chunksize=2), strict=True):
            check(changes["image_id"] == item["image_id"], "Replay order differs")
            data = (json.dumps(changes, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
            stream.write(data)
            digest.update(data)
            total.update(counts)
    check(total["images"] == len(items) and total["mapped_candidates"] == summary["counts"]["mapped"],
          "Incomplete point replay")
    return dict(phase="replay", status="B5_RESERVE74_REPLAY_COMPLETE", images=len(items), counts=dict(total),
                manifest_sha256=manifest_sha, checkpoint_sha256=CHECKPOINT, point_head_sha256=POINT_HEAD,
                predictions_sha256=summary["predictions_sha256"], audit_sha256=summary["audit_sha256"],
                edits_sha256=digest.hexdigest(), independent_decisions_equal=True, target_labels_opened=False)


def scoring_blocks(items, pairs, edits, scorer, core, adapter):
    blocks = {}
    for item, (audit, pred), row in zip(items, pairs, edits, strict=True):
        i = item["image_id"]
        check(row["image_id"] == i, "Edit row order differs")
        masks = adapter.native_masks(core, pred, audit, i)
        changes = {}
        for key, options in row["changes"].items():
            j = int(key)
            check(str(j) == key and 1 <= j <= len(masks) and options and set(options) <= set(scorer.ARMS[1:]),
                  "Unexpected active compact candidate or method")
            stock = masks[j - 1]
            area = int(core.codec.area(stock))
            check(32 <= area < 1024 and options.get(scorer.FIXED, {}).get("swaps", 0)
                  == options.get(scorer.MATCHED, {}).get("swaps", 0), "Fixed and matched candidate quotas differ")
            for update in options.values():
                check(set(update) == {"swaps", "rle"}, "Unexpected candidate edit fields")
                after = core.checked_rle(update["rle"], *item["original_hw"])
                k = update["swaps"]
                overlap = int(core.codec.area(core.codec.merge([stock, after], intersect=True)))
                check(type(k) is int and 1 <= k <= min(8, area // 10) and int(core.codec.area(after)) == area
                      and area - overlap == k, "Native additions, deletions, foreground area or cap differs")
            changes[j] = options
        blocks[i] = dict(stock=pred["coco"], ranks=[r["position"] for r in audit["candidates"]],
                         changes=changes, rgb=item["rgb_sha256"], group=item["acquisition_group"],
                         sensor=item["sensor"], file=item["source_image_name"])
    return blocks


def score(args, items, manifest_sha):
    import numpy as np
    import shapely

    check(shapely.__version__ == "2.0.2" and shapely.geos_version_string == "3.11.2", "Geometry runtime differs")
    scorer = module(args.code / RESERVE / "score_reserve.py", SCORE_SOURCE)
    prior = module(args.code / NATIVE / "score_rsdd.py", scorer.HISTORIC)
    evaluator = prior.module(args.code / "PSEG_RFDETR_FROZEN_SOURCE_HEAD_20261122/score.py", prior.SCORE_SOURCE)
    adapter = prior.module(args.code / UEIKAP / "score_rf.py", prior.GEOMETRY_SOURCE)
    core = adapter.load_geometry(args.code / "UEIKAP_RESERVED_HULL_FROZEN_20261122/score.py")
    capture_summary, pairs = read_capture(args, items, manifest_sha)
    replay_summary = json.loads((args.replay / RESULT).read_bytes())
    check(replay_summary["status"] == "B5_RESERVE74_REPLAY_COMPLETE"
          and all(replay_summary[k] == capture_summary[k] for k in
                  ("manifest_sha256", "roster_sha256", "checkpoint_sha256", "images", "predictions_sha256", "audit_sha256"))
          and replay_summary["point_head_sha256"] == POINT_HEAD and replay_summary["independent_decisions_equal"] is True
          and replay_summary["target_labels_opened"] is False, "Edit or capture lineage differs")
    edits = [json.loads(line) for line in pin(args.replay / EDITS, replay_summary["edits_sha256"]).splitlines()]
    blocks = scoring_blocks(items, pairs, edits, scorer, core, adapter)
    census_path = args.code.parent / "results/INSTANCE_RSDD_PARENT_RESERVE_CENSUS_SAFE_20261123.json"
    census = json.loads(pin(census_path, scorer.CENSUS))
    check(Counter((v["group"], v["sensor"]) for v in blocks.values())
          == {(p["paper_scene"], p["sensor"]): p["images"] for p in census["per_parent"]}, "Parent census differs")
    gt, rekeyed = prior.coco_annotation_ids(scorer.select_labels(pin(args.author_train, scorer.TRAIN), blocks))
    polygons = {r["id"]: shapely.Polygon(np.asarray(r["segmentation"][0]).reshape(-1, 2)) for r in gt["annotations"]}
    dates = {p["paper_scene"]: (p["sensor"], p["calendar_date"]) for p in census["per_parent"]}
    dated = set(census["date_separated_sensitivity"]["parent_scene_ids"])
    full = set(blocks)
    gf = {i for i, b in blocks.items() if b["sensor"] == "GF-3"}
    subsets = dict(full=full, gf3=gf, terrasar_x=full - gf,
                   date_separated={i for i, b in blocks.items() if b["group"] in dated},
                   full_deduplicated=prior.representatives(blocks, full))
    cache = {}

    def view(ids):
        key = frozenset(ids)
        if key not in cache:
            cache[key] = scorer.score_view(gt, ids, blocks, polygons, dates, evaluator, core, shapely)
        return cache[key]

    views = {name: view(ids) for name, ids in subsets.items()}
    by_parent = {}
    for parent in sorted(dates):
        v = view({i for i, b in blocks.items() if b["group"] == parent})
        by_parent[parent] = dict(images=v["images"], annotations=v["annotations"], sensor_date=dates[parent],
                                 metrics_fraction=v["metrics_fraction"], counts=v["geometry"]["counts_by_parent"][parent])
    args.output.mkdir()
    return dict(phase="score", status="B5_RESERVE74_SCORE_COMPLETE", images=len(items), views=views, by_parent=by_parent,
                author_train_sha256=scorer.TRAIN, census_sha256=scorer.CENSUS, manifest_sha256=manifest_sha,
                checkpoint_sha256=CHECKPOINT, point_head_sha256=POINT_HEAD, edits_sha256=replay_summary["edits_sha256"],
                predictions_sha256=capture_summary["predictions_sha256"], audit_sha256=capture_summary["audit_sha256"],
                evaluator_annotation_ids_rekeyed=rekeyed, non_simple_polygons_retained_in_ap=sum(not p.is_valid for p in polygons.values()),
                source_exposure_check_rerun=False, per_candidate_fixed_matched_budget_equal=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest="phase", required=True)
    for phase in ("capture", "replay", "score"):
        command = commands.add_parser(phase, allow_abbrev=False)
        command.add_argument("--code", type=Path, default=Path(__file__).resolve().parent.parent,
                             help="Published research_round_20260925_global/code directory")
        command.add_argument("--intake", type=Path, required=True,
                             help="Converted image directory containing images/ and IMAGE_MANIFEST_PRIVATE.json")
        command.add_argument("--roster", type=Path, required=True, help="Frozen generated 74-image CSV")
        command.add_argument("--output", type=Path, required=True, help="New phase directory")
        if phase == "capture":
            command.add_argument("--checkpoint", type=Path, required=True, help="HRSID center E24 checkpoint")
        else:
            command.add_argument("--capture", type=Path, required=True, help="Completed portable capture")
        if phase == "replay":
            command.add_argument("--point-head", type=Path, required=True, help="Frozen 5217_all_fit.npz")
            command.add_argument("--workers", type=int, default=4)
        if phase == "score":
            command.add_argument("--replay", type=Path, required=True, help="Completed portable replay")
            command.add_argument("--author-train", type=Path, required=True, help="Unmodified author train.json")
    args = parser.parse_args()
    for k, val in vars(args).items():
        if isinstance(val, Path):
            setattr(args, k, val.resolve(strict=k != "output"))
    check(not args.output.exists(), "Use a new phase output directory")
    if args.phase != "capture":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    expected = {"numpy": "1.26.4", "pycocotools": "2.0.11"}
    if args.phase != "score":
        expected.update({"torch": "2.10.0+cu130", "rfdetr": "1.12.0", "Pillow": "12.3.0"})
    runtime = {k: version(k) for k in expected}
    check(runtime == expected, f"Frozen operator versions differ: expected {expected}, found {runtime}")
    native = module(args.code / NATIVE / "capture_instance_rsdd.py", CAPTURE_SOURCE)
    items, manifest_sha = load_intake(args, native)
    if args.phase == "capture":
        result = capture(args, items, manifest_sha, native)
    elif args.phase == "replay":
        check(args.workers > 0, "Replay worker count must be positive")
        result = replay(args, items, manifest_sha, native)
    else:
        result = score(args, items, manifest_sha)
    result.update(roster_sha256=ROSTER, source_partition="author train", optimizer_updates=0,
                  operator_versions=runtime, worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    raw = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with (args.output / RESULT).open("xb") as stream:
        stream.write(raw)
    print(json.dumps(dict(status=result["status"], result_sha256=sha256(raw).hexdigest())), flush=True)


if __name__ == "__main__":
    main()
