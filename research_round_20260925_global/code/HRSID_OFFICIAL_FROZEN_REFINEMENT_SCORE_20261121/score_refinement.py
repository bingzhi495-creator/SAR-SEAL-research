"""Score twelve closed frozen treatment streams and one shared native stream on the three official cohorts."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
from statistics import mean
from struct import pack
import sys
from types import ModuleType


SEEDS = ("5217", "5218", "5219")
ARMS = ("SF", "SU=F", "SU", "LW=S")
CONTRASTS = (("SU", "N"), ("SU=F", "SF"), ("SU", "SU=F"), ("SU", "LW=S"))
SCORE_SOURCE = "c3322233db5234bf89510ea370c88a255dc4c995d80fc483d9903ef43ef01628"
NATIVE_SCORE = "0b5759262f96214457064735b0680a7c2f020b1503223613dcf56cc7072eed05"
NATIVE = "9c16ecb5275ae1ba23926a46aa74a5e10c979a3761838d7ff279859986a742b4"
FIT = "476b0f3fa748d1d3fd2393bdeb1af2d83e1532995d58b12625d57222e3f2638a"
RUNTIME = dict(python="3.8.17", numpy="1.23.5", pycocotools="2.0.7")
ACCEPTED = "HRSID_OFFICIAL_TEST_FROZEN_REFINEMENT_READOUT_INDEPENDENT_ACCEPTED"


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def score_module(path):
    data = path.read_bytes()
    require(sha256(data).hexdigest() == SCORE_SOURCE, "Original official scorer implementation")
    module = ModuleType("_original_official_scorer")
    module.__file__ = str(path)
    exec(compile(data, str(path), "exec"), module.__dict__)
    return module


def validate_readout(receipt, native_scorer):
    require(receipt.get("status") == ACCEPTED and type(receipt.get("original_exit")) is int and receipt["original_exit"] == 0
            and receipt.get("TEST") is True and receipt.get("AP") is False and receipt.get("GT_opened") is False
            and type(receipt.get("optimizer_updates")) is int and receipt["optimizer_updates"] == 0,
            "Accepted frozen label-free official TEST readout")
    require(receipt.get("endpoint_sha256") == native_scorer.ENDPOINT and receipt.get("fit_accepted_sha256") == FIT
            and receipt.get("native_predictions_sha256") == NATIVE and receipt.get("images") == 1962
            and receipt.get("mapped_rows") == 187796 and all(receipt.get(key) is True for key in
            ("exact_candidate_identity", "exact_native_per_image", "exact_per_candidate_budgets", "actual_XOR_area_RLE_verified")),
            "Independent per-image native, query, frozen head, budget and raster verification")
    streams = receipt["streams"]
    require(set(streams) == set(SEEDS) and all(set(streams[s]) == set(ARMS) for s in SEEDS), "All three frozen heads and four treatments")
    for seed in SEEDS:
        for ref in streams[seed].values():
            require(native_scorer.valid_sha(ref.get("sha256")) and type(ref.get("bytes")) is int and ref["bytes"] > 0,
                    "Accepted treatment stream identity")
    return streams


def candidate_signature(block):
    ranks, rows = block["native_ranks"], block["coco"]
    require(set(block) == {"image_id", "raw_query_count", "native_count", "unmapped_slot1_count", "native_ranks", "coco"}
            and len(ranks) == len(rows) and all(type(r) is int and 1 <= r <= 100 for r in ranks), "Canonical native block fields")
    digest = sha256(pack(">qiii", block["image_id"], block["raw_query_count"], block["native_count"], block["unmapped_slot1_count"]))
    for rank, row in zip(ranks, rows):
        require(type(row["score"]) is float, "Original confidence representation")
        digest.update(pack(">Bqid", rank, row["image_id"], row["category_id"], row["score"]))
    return digest.digest()


def load_arm(path, ref, ids, hw, mapped, scorer, native_signatures=None):
    signatures = []
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        require(before.st_size == ref["bytes"], "Accepted stream length")

        class CheckedStream:
            def readline(self, bound):
                line = stream.readline(bound)
                require(line.endswith(b"\n"), "Complete native image block")
                signature = candidate_signature(json.loads(line))
                if native_signatures is not None:
                    require(signature == native_signatures[len(signatures)], "Per-image native IDs, rank order and exact scores")
                signatures.append(signature)
                return line

            def read(self, bound):
                return stream.read(bound)

        rows, digests = scorer.parse_native(CheckedStream(), ids, hw[0], hw[1],
                                            dict(predictions_sha256=ref["sha256"], mapped_rows=mapped, prediction_bytes=ref["bytes"]))
        after = os.fstat(stream.fileno())
        require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), "Stable closed stream")
    return rows, digests, signatures


def contrasts(native, treatments):
    results = {}
    for left, right in CONTRASTS:
        per_view = {}
        for name in native:
            seeds = {}
            for seed in SEEDS:
                a = treatments[seed][left][name]["metrics"]
                b = native[name]["metrics"] if right == "N" else treatments[seed][right][name]["metrics"]
                seeds[seed] = {m:100 * (a[m] - b[m]) if min(a[m], b[m]) >= 0 else None for m in a}
            avg = {}
            for m in seeds[SEEDS[0]]:
                values = [seeds[s][m] for s in SEEDS]
                avg[m] = mean(values) if all(v is not None for v in values) else None
            per_view[name] = dict(per_head_seed_points=seeds, arithmetic_mean_points=avg)
        results[left + " minus " + right] = per_view
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("score-source", "adapter", "baseline-score", "readout", "paths", "native", "cohorts", "gt", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--readout-sha256", required=True)
    args = parser.parse_args()
    scorer = score_module(args.score_source)
    require(sys.flags.isolated and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
            and scorer.runtime() == RUNTIME, "Original isolated CPU scoring environment")
    require(not args.output.exists() and args.output.parent.is_dir(), "Fresh score output")
    receipt = json.loads(scorer.pinned(args.readout, args.readout_sha256, 4 * 1024**2))
    streams = validate_readout(receipt, scorer)
    paths = json.loads(args.paths.read_text(encoding="utf-8-sig"))
    require(set(paths) == set(SEEDS) and all(set(paths[s]) == set(ARMS) for s in SEEDS)
            and all(Path(p).is_absolute() for arms in paths.values() for p in arms.values()), "All closed absolute treatment paths")
    views = scorer.validate_plan(json.loads(scorer.pinned(args.cohorts, scorer.COHORT, 4 * 1024**2)))
    primary = json.loads(scorer.pinned(args.baseline_score, NATIVE_SCORE, 1024**2))
    ids = views["official"]["images"]
    adapter = scorer.adapter_from(args.adapter)
    mapped = primary["mapped_rows"]
    require(primary["predictions_sha256"] == NATIVE and mapped == receipt["mapped_rows"], "Original qualified native score and stream")
    rows, digests, signatures = load_arm(args.native, dict(bytes=primary["prediction_bytes"], sha256=NATIVE), ids,
                                         (800,800), mapped, scorer)
    gt = json.loads(scorer.pinned(args.gt, scorer.GT, 16 * 1024**2))
    native = scorer.evaluate_views(gt, views, rows, digests, adapter)
    require(native == primary["scores"], "Exact N reproduction of all 39 values, population counts and block digests")
    del rows, digests
    treatments = {}
    for seed in SEEDS:
        treatments[seed] = {}
        for arm in ARMS:
            rows, digests, _ = load_arm(Path(paths[seed][arm]), streams[seed][arm], ids, (800,800), mapped, scorer, signatures)
            stats = scorer.evaluate_views(gt, views, rows, digests, adapter)
            for name, reference in native.items():
                require(all(stats[name][k] == reference[k] for k in ("images", "annotations", "images_without_annotations",
                            "mapped_predictions", "images_without_predictions")), "Same native population for every treatment")
            treatments[seed][arm] = stats
            del rows, digests
            print(json.dumps(dict(event="HRSID_FROZEN_REFINEMENT_SCORED", seed=seed, arm=arm)), flush=True)
    result = dict(status="HRSID_OFFICIAL_TEST_FROZEN_REFINEMENT_SCORE_COMPLETE", historical_test_selection_exposure=True,
                  TEST=True, AP=True, model_forward_calls=0, optimizer_updates=0,
                  seeds=[int(s) for s in SEEDS], arms=["N"] + list(ARMS), native=native, treatments=treatments,
                  metric_units="COCO fraction; -1 is undefined", contrast_units="percentage points; null is undefined",
                  paired_contrasts=contrasts(native, treatments), cohort_sha256=scorer.COHORT, gt_sha256=scorer.GT,
                  native_score_sha256=NATIVE_SCORE, native_predictions_sha256=NATIVE, endpoint_sha256=scorer.ENDPOINT,
                  fit_accepted_sha256=FIT, readout_sha256=args.readout_sha256, runtime=RUNTIME,
                  streams={s:{a:{k:streams[s][a][k] for k in ("bytes", "sha256")} for a in ARMS} for s in SEEDS},
                  score_source_sha256=SCORE_SOURCE, adapter_sha256=scorer.ADAPTER,
                  refinement_score_source_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    raw = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with args.output.open("xb") as out:
        require(out.write(raw) == len(raw), "Complete score output")
        out.flush()
        os.fsync(out.fileno())
    print(json.dumps(dict(status=result["status"], output_sha256=sha256(raw).hexdigest())), flush=True)


if __name__ == "__main__":
    main()
