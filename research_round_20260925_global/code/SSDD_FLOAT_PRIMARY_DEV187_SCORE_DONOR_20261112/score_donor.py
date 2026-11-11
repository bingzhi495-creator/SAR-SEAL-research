"""Fixed SSDD float DEV187 equal mapped-prefix and score-donor diagnostic."""

from argparse import ArgumentParser
from collections import Counter, defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from types import ModuleType

ARMS = ("native", "center")
PROTOCOL_SHA = "a1f7336baf259a479af27ef6904e112e416cd74254ef92a7acb2be732be2b4b2"
ORIGINAL_WORKER_SHA = "5463f294191929bceaac3b04fb6c17179c05066eaa29b6d859ef4943cca686c7"
ACCEPTED_SHA = "07bbf74198bec0e55f131a2aeeca2f7efa4c6008654dd3816e7b3ea57d8178f4"
REQUEST_SHA = "46b41a2261a746523049e74910cf7d417ecc595c583e80c57fd97de42c4759ce"
RESULT_STATUS = "SSDD_FLOAT_DEV187_EQUAL_MAPPED_SCORE_DONOR_ACTUAL_V1"
VERIFY_STATUS = "SSDD_FLOAT_DEV187_EQUAL_MAPPED_SCORE_DONOR_INDEPENDENT_READBACK_V1"
MIB = 1024**2


def need(ok, detail):
    if not ok:
        raise ValueError(detail)


def source(path, expected):
    need(path.is_absolute() and path.resolve(strict=True) == path, "Exact canonical source path required")
    with path.open("rb") as stream:
        raw = stream.read(MIB)
    need(sha256(raw).hexdigest() == expected, "Exact pinned source bytes required")
    return raw


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return sha256(canonical(value)).hexdigest()


def admitted(code, request_path, accepted_path):
    worker_path = code / "SSDD_FLOAT_PRIMARY_DEV187_20261112/float_dev187.py"
    raw = source(worker_path, ORIGINAL_WORKER_SHA)
    app = ModuleType("_closed_ssdd_float_score")
    app.__file__ = str(worker_path)
    exec(compile(raw, str(worker_path), "exec"), app.__dict__)
    stage, modules = app.load_stock(code)
    stage.admit_runtime("SCORE")
    sys.addaudithook(stage.reject_network)
    sys.addaudithook(stage.scope_guard(allowed=())[0])
    scorer = modules["ssdd_phase_dev_score"]
    original = modules["ssdd_phase_dev_admission"]
    request = app.read_json(stage, app.file_ref(request_path, REQUEST_SHA))
    accepted = app.read_json(stage, app.file_ref(accepted_path, ACCEPTED_SHA))
    need(accepted["status"] == "SSDD_FLOAT_COMMON_DEV187_SCORE_INDEPENDENTLY_ACCEPTED_V1"
         and accepted["images"] == 187 and accepted["annotations"] == 380
         and accepted["DEV"] is True and accepted["TEST"] is False and accepted["FIT"] is False
         and accepted["seed"] == 5201 and accepted["max_dets"] == [1, 10, 100]
         and accepted["floating_gt_sha256"] == app.FLOAT_GT_SHA == request["float_gt_ref"]["sha256"],
         "Exact independently accepted original float-primary scope required")
    train = stage.train_args(request)
    evaluation = request["evaluation_protocol"]
    pair = request["training_pair_ref"]
    need(accepted["training_pair_sha256"] == pair["sha256"]
         and accepted["evaluation_protocol"] == evaluation, "Original scoring and training receipts must join")
    _, trained = app.read_training_pair(original, pair, **train)
    sessions = {app.verify_export_acceptance(stage, original, request[mode], mode, pair_sha=pair["sha256"],
                                             evaluation_protocol=evaluation) for mode in ARMS}
    need(len(sessions) == 2, "Distinct accepted export sessions required")
    ref = request["image_manifest_ref"]
    need(ref["sha256"] == accepted["image_manifest_sha256"], "Exact accepted original manifest required")
    artifact = scorer.Artifact(ref, stage.read_artifact(ref, MIB))
    _, ids, items = scorer.fixed_images(artifact)
    apis = app.score_apis(scorer)
    gt = app.read_json(stage, request["float_gt_ref"])
    scorer.validate_gt(gt, ids, items)
    predictions, indexes = {}, {}
    for mode in ARMS:
        index, rows, safe = app.load_export(stage, original, scorer, request[mode], mode, trained[mode],
                                          pair_sha=pair["sha256"], arm_sha=train[mode + "_arm_ref"]["sha256"],
                                          evaluation_protocol=evaluation)
        need(safe == accepted["arms"][mode], "Original export metadata differs from accepted score")
        document = apis.bridge.group_canonical_rows(index, rows, items)
        flat, _ = apis.adapter.normalize_export(document, gt["images"])
        need(flat == rows, "Original normalization must preserve every mapped row and score")
        predictions[mode], indexes[mode] = flat, index["images"]
    return app, scorer, apis.official, gt, ids, accepted, predictions, indexes


def prefixes_from_groups(predictions, ids):
    pools = {}
    for mode, rows in predictions.items():
        groups = defaultdict(list)
        for row in rows:
            groups[row["image_id"]].append(row)
        pools[mode] = [groups[image_id] for image_id in ids]
    return pools


def prefixes_from_index(predictions, indexes, ids):
    pools = {}
    for mode, image_rows in indexes.items():
        need([image["image_id"] for image in image_rows] == ids, "Exact index ordering required")
        rows = predictions[mode]
        pools[mode] = [rows[image["prediction_start"]:image["prediction_start"] + image["prediction_count"]]
                       for image in image_rows]
    return pools


def build_cells(pools, ids):
    from pycocotools import mask

    counts = [min(len(a), len(b)) for a, b in zip(*(pools[mode] for mode in ARMS), strict=True)]
    prefixes = {mode: [rows[:k] for rows, k in zip(pools[mode], counts, strict=True)] for mode in ARMS}
    cells = {arm: {donor: [] for donor in ARMS} for arm in ARMS}
    for arm, donor_set in cells.items():
        for donor, stream in donor_set.items():
            for geometry, scoring in zip(prefixes[arm], prefixes[donor], strict=True):
                for row, donor_row in zip(geometry, scoring, strict=True):
                    need(row["image_id"] == donor_row["image_id"], "Donor rank is paired within the same image")
                    stream.append(dict(row, score=donor_row["score"]))
            if arm == donor:
                need(stream == [row for rows in prefixes[arm] for row in rows], "Own donor must leave prefix intact")
    need(sum(counts) > 0 and all(len(stream) == sum(counts) for donors in cells.values() for stream in donors.values()),
         "Exact nonempty shared candidate ledger required")
    original_counts = {mode: sum(map(len, pools[mode])) for mode in ARMS}
    kept_nonempty = {}
    for mode in ARMS:
        rles = [row["segmentation"] for rows in prefixes[mode] for row in rows]
        kept_nonempty[mode] = int((mask.area(rles) > 0).sum())
    census = dict(original_mapped=original_counts, kept_per_arm=sum(counts),
                  dropped={mode: original_counts[mode] - sum(counts) for mode in ARMS},
                  kept_nonempty=kept_nonempty, kept_zero_masks={mode: sum(counts)-kept_nonempty[mode] for mode in ARMS},
                  images=len(ids), zero_k_images=counts.count(0), k_min=min(counts), k_max=max(counts),
                  k_histogram={str(k): n for k, n in sorted(Counter(counts).items())},
                  per_image_count_digest=digest(list(zip(ids, counts, strict=True))))
    return cells, census


def compute(args, independent):
    app, scorer, official, gt, ids, accepted, predictions, indexes = admitted(args.code, args.request, args.accepted)
    score = lambda rows: app.score_record(scorer.score_one(official, gt, rows, ids))
    unmodified = {mode: score(predictions[mode]) for mode in ARMS}
    witness = dict(unmodified, center_minus_native_points=app.delta(unmodified["native"], unmodified["center"]))
    need(witness == accepted["primary"], "Untouched float-primary COCO12/AP90 witness differs from accepted score")
    pools = prefixes_from_index(predictions, indexes, ids) if independent else prefixes_from_groups(predictions, ids)
    streams, counts = build_cells(pools, ids)
    cells = {arm: {donor: score(stream) for donor, stream in donors.items()} for arm, donors in streams.items()}
    fingerprints = {arm: {donor: digest(stream) for donor, stream in donors.items()} for arm, donors in streams.items()}
    candidate = {donor: app.delta(cells["native"][donor], cells["center"][donor]) for donor in ARMS}
    score_donor = {arm: app.delta(cells[arm]["native"], cells[arm]["center"]) for arm in ARMS}
    interaction = {key: None if a is None or b is None else b-a
                   for key, a, b in ((key, score_donor["native"][key], score_donor["center"][key])
                                     for key in app.METRICS)}
    return dict(status=RESULT_STATUS, device="CPU", FIT=False, DEV=True, TEST=False, AP=True, model_forward_calls=0,
                training=False, images=187, annotations=380, training_seed=5201, max_dets=[1, 10, 100],
                posthoc_development_diagnostic=True, official_source_float_GT_sha256=app.FLOAT_GT_SHA,
                protocol_sha256=PROTOCOL_SHA, accepted_original_sha256=ACCEPTED_SHA,
                prediction_sha256={mode: accepted["arms"][mode]["prediction_sha256"] for mode in ARMS},
                full_primary_scores_exact=True, unmodified_original=witness,
                equal_mapped_candidate_census=counts, candidate_donor_stream_sha256=fingerprints,
                candidate_donor_score_cells=cells, candidate_effect_at_fixed_donor_points=candidate,
                score_donor_effect_at_fixed_candidate_points=score_donor,
                interaction_points=interaction, center_own_minus_native_own_points=app.delta(cells["native"]["native"],
                                                                                           cells["center"]["center"]))


def write(path, value):
    need(path.is_absolute() and path.parent.resolve(strict=True) == path.parent and not path.exists(),
         "Fresh canonical output required")
    raw = canonical(value) + b"\n"
    with path.open("xb") as stream:
        stream.write(raw)
    return sha256(raw).hexdigest()


def main():
    parser = ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("code", "protocol", "accepted", "request", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--verify-result", type=Path)
    parser.add_argument("--verify-sha256")
    args = parser.parse_args()
    need(all(os.environ.get(key) == "1" for key in
             ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")),
         "Exact one-thread local CPU environment required")
    source(args.protocol, PROTOCOL_SHA)
    independent = args.verify_result is not None
    need(independent == (args.verify_sha256 is not None), "Both independent result references required")
    result = compute(args, independent)
    if independent:
        previous = json.loads(source(args.verify_result, args.verify_sha256))
        need(previous == result, "Independent index-sliced donor streams and exact COCO scores differ")
        result = dict(status=VERIFY_STATUS, original_result_sha256=args.verify_sha256,
                      protocol_sha256=PROTOCOL_SHA, reconstructed_cells=4, recomputed_untouched_arms=2,
                      all_scores_exact=True, candidate_donor_stream_sha256=result["candidate_donor_stream_sha256"],
                      FIT=False, DEV=True, TEST=False)
    print(json.dumps(dict(status=result["status"], result_sha256=write(args.output, result)), sort_keys=True),
          flush=True)


if __name__ == "__main__":
    main()

