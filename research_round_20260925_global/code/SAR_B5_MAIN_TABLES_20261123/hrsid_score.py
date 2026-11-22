"""Evaluate freshly generated HRSID official streams on the three reconstructed frozen cohorts."""

import json
import os
from pathlib import Path
from zipfile import ZipFile


def run(args, inputs):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    check = inputs.check
    check(not args.output.exists(), "Use a fresh score output directory")
    refine = inputs.load(args.code / "HRSID_OFFICIAL_FROZEN_REFINEMENT_SCORE_20261121/score_refinement.py",
                         "_contrasts", "f047034705e6048230764d7f5712419546a387cd1d3c3a92ec00c57561dea385")
    score = refine.score_module(args.code / "HRSID_OFFICIAL_BASELINE_SCORE_20261121/score_native.py")
    adapter = score.adapter_from(args.code / "DIFFSAR_NATIVE_CPU_COCO_ADAPTER_DRAFT_20261107_01/worker.py")
    runtime = score.runtime()
    clean = dict(python="3.10.13", numpy="1.26.4", pycocotools="2.0.11")
    check(runtime in (refine.RUNTIME, clean), "Use the historical scoring or clean portable inference environment")
    receipt = args.capture / "RESULT_SAFE.json"
    readout_sha = inputs.file_digest(receipt)
    readout = json.loads(receipt.read_bytes())
    check(readout["status"] == "B5_HRSID1962_CAPTURE_COMPLETE" and
          readout["images"] == readout["model_forward_calls"] == 1962 and
          readout["mapped_rows"] == 187796 and readout["GT_opened"] is readout["AP"] is False and
          readout["optimizer_updates"] == 0 and readout["endpoint_sha256"] == inputs.ENDPOINT and
          readout["manifest_sha256"] == inputs.MANIFEST_SHA and readout["native_predictions_sha256"] == refine.NATIVE and
          all(readout[k] is True for k in ("exact_candidate_identity", "exact_native_per_image",
              "exact_per_candidate_budgets", "actual_XOR_area_RLE_verified", "point_score_decisions_replayed")),
          "Fresh capture lacks complete native identity and independent pixel replay")
    check(inputs.file_digest(args.intake / inputs.MANIFEST) == inputs.MANIFEST_SHA and
          inputs.file_digest(args.intake / inputs.PLAN) == score.COHORT, "Original regenerated evaluation populations")
    views = score.validate_plan(json.loads((args.intake / inputs.PLAN).read_bytes()))
    with ZipFile(args.archive) as archive:
        raw = archive.read("annotations/test2017.json")
    check(score.digest(raw) == score.GT, "Original author labels differ")
    gt = json.loads(raw)
    ids = views["official"]["images"]
    mapped = readout["mapped_rows"]
    native_path = args.capture / "N.jsonl"
    rows, digests, signatures = refine.load_arm(native_path,
        dict(bytes=readout["native_prediction_bytes"], sha256=refine.NATIVE), ids, (800, 800), mapped, score)
    native = score.evaluate_views(gt, views, rows, digests, adapter)
    del rows, digests
    ref = readout["streams"]
    check(set(ref) == set(refine.SEEDS) and all(set(ref[s]) == set(refine.ARMS) for s in ref),
          "Three point-head seeds and four treatment arms")
    args.output.mkdir()
    treatments = {}
    for s in refine.SEEDS:
        treatments[s] = {}
        for arm in refine.ARMS:
            path = args.capture / (s + "_" + arm + ".jsonl")
            rows, digests, _ = refine.load_arm(path, ref[s][arm], ids, (800, 800), mapped, score, signatures)
            metrics = score.evaluate_views(gt, views, rows, digests, adapter)
            del rows, digests
            for name in views:
                check(all(metrics[name][k] == native[name][k] for k in ("images", "annotations", "images_without_annotations",
                          "mapped_predictions", "images_without_predictions")), "Treatment population differs")
            treatments[s][arm] = metrics
            print(json.dumps(dict(event="SCORED", seed=s, arm=arm)), flush=True)
    return dict(status="B5_HRSID1962_SCORE_COMPLETE", TEST=True, AP=True, historical_test_selection_exposure=True,
                model_forward_calls=0, optimizer_updates=0, images=1962,
                metric_units="COCO fraction; -1 is undefined", contrast_units="percentage points; null is undefined",
                seeds=[int(s) for s in refine.SEEDS], arms=["N"] + list(refine.ARMS), runtime=runtime,
                original_scoring_runtime=(runtime == refine.RUNTIME), readout_sha256=readout_sha,
                native=native, treatments=treatments, paired_contrasts=refine.contrasts(native, treatments),
                endpoint_sha256=inputs.ENDPOINT, gt_sha256=score.GT, cohort_sha256=score.COHORT,
                native_predictions_sha256=refine.NATIVE, streams=ref, score_source_sha256=refine.SCORE_SOURCE,
                adapter_sha256=score.ADAPTER, worker_sha256=inputs.file_digest(Path(__file__)))

