"""Fit the archived compact occupancy head on accepted PSeg-SSDD FIT109 evidence."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
from time import perf_counter
from types import ModuleType

import numpy as np
from pycocotools import mask as codec
import torch


RF_SPOT = "RFDETR_CORE36_MATCHED_POINT_SPOT_20261120"
OLD_POINTS = "DFINE_SEG_EXTERNAL_BASELINE_FIT32_E24_20261112/e24/fit109_paired_point/train_point.py"
STEP_SHA = "94e60f5b5557787687d1f228c33c249a1d7889172a225373b97bd789ec8b72b1"
REPLAY_SHA = "be1516489123153a018ebc721806190d83f05285617e7c108e9b3c1064f66655"
FIT_SHA = "97162d23a33b9387fe0b47a117f7e5b1f9dde7471033f244ab336cee1ca9039b"
CHECKPOINT_SHA = "bbe80576d9d8f3ff1b4dc308bd87f199a707ad9dc44c4d48d68533c309aeb9bb"
PRELOCK_SHA = "11a74f36a4b662cffd1a78fcea7b77b9e5d5e18d4543cc191ac296642752fb02"
CAPTURE_DONE = "PSEG_TARGET_RFDETR5201_FIT109_CAPTURE_COMPLETE_V1"
CAPTURE_VERIFIED = "PSEG_TARGET_RFDETR5201_FIT109_CAPTURE_INDEPENDENTLY_VERIFIED_V1"
COMPLETE = "PSEG_TARGET_RF5201_FIT109_POINT_HEAD_COMPLETE_V1"
SEED, SAMPLE_SEED, STEPS, BATCH = 5217, 20261112, 200, 64
LR, DECAY, PARAMS = .002, .01, 1321


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def pin(path, digest):
    data = path.read_bytes()
    need(sha256(data).hexdigest() == digest, "Exact input bytes: " + path.name)
    return data


def load(path, digest):
    helper = ModuleType(path.stem)
    helper.__file__ = str(path)
    exec(compile(pin(path, digest), str(path), "exec"), helper.__dict__)
    return helper


def encode(obj):
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def write(path, obj):
    raw = encode(obj)
    with path.open("xb") as out:
        out.write(raw)
        out.flush()
        os.fsync(out.fileno())
    return sha256(raw).hexdigest()


def training_sources(code):
    step = load(code / RF_SPOT / "fit_step.py", STEP_SHA)
    return step, step.load_trainer(code / OLD_POINTS)


def fit_compact(records, step, trainer):
    started = perf_counter()
    own, truth, center, scale, groups = step.prepared(records, trainer)
    x = torch.from_numpy(np.asarray((own - center) / scale, dtype=np.float32))
    p = torch.from_numpy(own[:, :, 0].copy())
    q = torch.from_numpy(truth)
    rng = np.random.Generator(np.random.PCG64(SAMPLE_SEED))
    draws = rng.integers(len(groups), size=(STEPS, BATCH))
    batches = np.asarray([[groups[g][rng.integers(len(groups[g]))] for g in row] for row in draws],
                         dtype=np.int64)
    torch.manual_seed(SEED)
    model = trainer.PointHead(14)
    initial = step.tensor_hash(model)
    need(sum(v.numel() for v in model.parameters()) == PARAMS
         and torch.count_nonzero(model.network[-1].weight) == 0
         and torch.count_nonzero(model.network[-1].bias) == 0, "Archived compact identity initialization")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=DECAY)
    losses, gradient_norms = [], []
    optimizer_started = perf_counter()
    for rows in batches:
        optimizer.zero_grad(set_to_none=True)
        logits = model(x[rows], x[rows], p[rows])
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, q[rows])
        need(bool(torch.isfinite(loss)), "Finite fractional occupancy BCE")
        loss.backward()
        need(all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters()),
             "Finite gradients for every parameter")
        gradient_norms.append(float(torch.sqrt(sum(v.grad.double().square().sum() for v in model.parameters()))))
        optimizer.step()
        losses.append(float(loss.detach()))
    optimizer_seconds = perf_counter() - optimizer_started
    counts = {int(v["step"].item()) for v in optimizer.state.values()}
    final = step.tensor_hash(model)
    need(counts == {STEPS} and all(torch.isfinite(v).all() for v in model.parameters()) and final != initial,
         "Exact finite, nontrivial compact optimizer endpoint")
    model.eval()
    audit = dict(optimizer_updates=next(iter(counts)), head_parameter_count=PARAMS, seed=SEED,
                 sample_seed=SAMPLE_SEED, batch_pairs=BATCH, sampled_pair_rows=STEPS * BATCH,
                 sampled_point_rows=2 * STEPS * BATCH, learning_rate=LR, weight_decay=DECAY,
                 supervised_pairs=len(own), supervised_rgb_groups=len(groups),
                 common_batch_sha256=sha256(batches.astype("<i8", copy=False).tobytes()).hexdigest(),
                 initial_tensor_sha256=initial, final_tensor_sha256=final, loss_first=losses[0],
                 loss_last=losses[-1], finite_losses=len(losses), gradient_norm_range=[min(gradient_norms),
                 max(gradient_norms)], optimizer_seconds=optimizer_seconds, fit_seconds=perf_counter() - started)
    return dict(model=model, center=center, scale=scale, audit=audit)


def load_head(code, weights, weights_sha):
    step, trainer = training_sources(code)
    model = trainer.PointHead(14)
    state = model.state_dict()
    with np.load(BytesIO(pin(weights, weights_sha)), allow_pickle=False) as bundle:
        need(set(bundle.files) == {"center", "scale", *(f"S.{key}" for key in state)}, "Exact small-head bundle")
        center, scale = bundle["center"], bundle["scale"]
        for key, original in state.items():
            array = bundle[f"S.{key}"]
            need(array.shape == original.shape and array.dtype == np.float32 and np.isfinite(array).all(),
                 "Exact finite and shaped compact weights")
            state[key] = torch.from_numpy(array)
        need(all(x.shape == (14,) and x.dtype == np.float32 and np.isfinite(x).all() for x in (center, scale))
             and np.all(scale > 0), "Exact finite scaler with positive scales")
    model.load_state_dict(state, strict=True)
    model.eval()
    return step, model, center, scale


def make_predictor(code, weights, weights_sha):
    step, model, center, scale = load_head(code, weights, weights_sha)
    return lambda features: step.predict(model, features, center, scale)


def read_capture(a):
    verify = json.loads(pin(a.capture_accepted, a.capture_accepted_sha))
    capture = json.loads(pin(a.capture / "RESULT_SAFE.json", verify["capture_result_sha256"]))
    terminal = json.loads(a.capture_terminal.read_bytes())
    need(terminal["phase"] == "verify" and terminal["actual_exit"] == 0
         and terminal["output_footer_verified"] is True and terminal["result_sha256"] == a.capture_accepted_sha,
         "Actual independent capture verification and observed output")
    need(capture["status"] == CAPTURE_DONE and verify["status"] == CAPTURE_VERIFIED
         and verify["native_raster_parity"] is True, "Verified FIT capture native mask parity")
    for x in (capture, verify):
        need(x["FIT"] is True and all(x[key] is False for key in ("DEV", "TEST", "GT_opened", "AP"))
             and x["images"] == 109 and x["optimizer_updates"] == 0 and x["checkpoint_sha256"] == CHECKPOINT_SHA
             and x["protocol_sha256"] == PRELOCK_SHA, "Closed frozen capture and FIT-only scope")
    for key in ("capture_stream_sha256", "predictions_sha256", "image_manifest_sha256", "selection_sha256",
                "mapped_candidates", "eligible_planes", "native_overrides"):
        need(capture[key] == verify[key], "Independent capture identity: " + key)
    selected = json.loads(pin(a.capture / "FIT_IMAGE_SELECTION_PRIVATE.json", verify["selection_sha256"]))
    blocks = [json.loads(line) for line in
              pin(a.capture / "CAPTURE_PRIVATE.jsonl", verify["capture_stream_sha256"]).splitlines()]
    predictions = json.loads(pin(a.capture / "PARITY_EXPORT/PREDICTIONS.json", verify["predictions_sha256"]))
    ids = [r["image_id"] for r in selected["items"]]
    rgb = [r["rgb_sha256"] for r in selected["items"]]
    need(selected["schema"] == "PSeg-SSDD-FIT109-RGB-selection-v1" and ids == selected["expected_ids"]
         and ids == [r["image_id"] for r in blocks] and rgb == [r["rgb_sha256"] for r in blocks]
         and len(set(rgb)) == len(set(ids)) == len(ids) == 109 and len(predictions) == verify["mapped_candidates"],
         "Exact selected images and distinct decoded-RGB groups in capture order")
    return blocks, predictions, verify


def prepare_features(a, blocks, predictions, replay):
    methods = replay.methods(a.code)
    totals, feature_seal = Counter(), sha256()
    offset, data = 0, []
    for row in blocks:
        hw = tuple(row["original_hw"])
        end = offset + row["stock_prediction_count"]
        need(row["stock_prediction_start"] == offset, "Native mapped stream order")
        preds = predictions[offset:end]
        offset = end
        ranks, queries, boxes = row["native_ranks"], row["query_ids"], row["native_boxes_xyxy"]
        need(ranks == sorted(set(ranks)) and len(queries) == len(set(queries)), "Original unique candidate identities")
        candidates = []
        for rank, query, target in zip(ranks, queries, preds, strict=True):
            rle = target["segmentation"]
            need(rle["size"] == list(hw) and target["image_id"] == row["image_id"] and target["category_id"] == 1,
                 "Native FIT candidate canvas and image identity")
            candidates.append(dict(image_id=row["image_id"], rank=rank, query_id=query,
                                   category_id=1, label=0, score=target["score"], rle=rle,
                                   area=int(codec.area(rle))))
        eligible = [(c, box) for c, box in zip(candidates, boxes, strict=True) if 32 <= c["area"] < 1024]
        need([c["rank"] for c, _ in eligible] == row["logit_ranks"]
             and [c["query_id"] for c, _ in eligible] == row["logit_query_ids"], "Exact retained logit order")
        planes = replay.load_array(a.capture, row["logits_ref"])
        need(planes.shape == (len(eligible), 144, 144), "Raw native coarse planes")
        overrides = {r["native_rank"]: r for r in row["posterior_overrides"]}
        rgb = replay.load_rgb(a.fit_images, row)
        features = methods.pair.image_features(rgb)
        pair, sampler = replay.canvas(methods, hw)
        records = []
        for (c, raw_box), plane in zip(eligible, planes, strict=True):
            stock = codec.decode(c["rle"]).astype(bool)
            ref = overrides.get(c["rank"])
            need(ref is None or ref["query_id"] == c["query_id"], "Posterior override tied to exact native query")
            probability = replay.native_probability(plane, stock,
                                                   None if ref is None else replay.load_array(a.capture, ref))
            box = np.asarray(raw_box, dtype=np.float32)
            need(box.shape == (4,) and np.isfinite(box).all() and np.all(box[2:] >= box[:2])
                 and np.all(box[:2] >= 0) and box[2] <= hw[1] and box[3] <= hw[0], "Native clipped box")
            if not np.all(box[2:] > box[:2]):
                continue
            result = pair.paired_points(stock, box, c["area"], c["score"], probability, features, sampler)
            if result is None:
                continue
            coords, point_features, _, _, _ = result
            identity = dict(image_id=c["image_id"], rank=c["rank"], query_id=c["query_id"], group=row["rgb_sha256"])
            feature_seal.update(encode(identity))
            feature_seal.update(coords.astype("<i4", copy=False).tobytes())
            feature_seal.update(point_features.astype("<f4", copy=False).tobytes())
            records.append(dict(**identity, coordinates=coords, features=point_features))
            totals["pairs"] += len(coords)
        totals.update(dict(mapped=len(candidates), eligible=len(eligible), candidate_point_sets=len(records),
                           overrides=len(overrides)))
        data.append((row, candidates, records))
    need(offset == len(predictions) and totals["pairs"] > 0, "Entire original native FIT stream used")
    return data, dict(totals), feature_seal.hexdigest()


def label(a, data, step):
    gt = json.loads(pin(a.fit_gt, FIT_SHA))
    rows = {image["id"]: image for image in gt["images"]}
    need(len(rows) == len(gt["images"]) == 741 and len(gt["annotations"]) == 1661
         and [r["id"] for r in gt["categories"]] == [1], "Archived original target FIT741 float annotations")
    chosen = {row["image_id"] for row, _, _ in data}
    annotations = defaultdict(list)
    for ann in gt["annotations"]:
        if ann["image_id"] in chosen:
            annotations[ann["image_id"]].append(ann)
    records, counts = [], Counter()
    helpers = step.load_geometry(a.code, a.shapely_site)
    for row, candidates, points in data:
        image_id = row["image_id"]
        image = rows[image_id]
        need(row["original_hw"] == [image["height"], image["width"]]
             and Path(row["source_image_name"]).name == Path(image["file_name"]).name,
             "Exact selected FIT labels and original canvas")
        joined, audit = step.attach_truth(image, annotations[image_id], candidates, points, helpers)
        counts.update(audit)
        records.extend(joined)
    counts["selected_annotations"] = sum(map(len, annotations.values()))
    return records, dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("protocol", "capture", "capture-accepted", "capture-terminal", "fit-images", "fit-gt",
                 "code", "shapely-site", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--capture-accepted-sha", required=True)
    a = parser.parse_args()
    pin(a.protocol, PRELOCK_SHA)
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" and not torch.cuda.is_initialized(),
         "Explicit CPU-only optimization")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    capture_rows, predictions, verify = read_capture(a)
    a.output.mkdir()
    started = perf_counter()
    replay = load(a.code / "PSEG_RFDETR_FROZEN_SOURCE_HEAD_20261122/replay.py", REPLAY_SHA)
    data, prepared, feature_sha = prepare_features(a, capture_rows, predictions, replay)
    feature_seconds = perf_counter() - started
    need(prepared["mapped"] == verify["mapped_candidates"] and prepared["eligible"] == verify["eligible_planes"]
         and prepared["overrides"] == verify["native_overrides"], "Exact independent capture totals")
    prepare_sha = write(a.output / "PRELABEL_SAFE.json", dict(status="PSEG_TARGET_RF5201_FIT109_PRELABEL_V1",
                        FIT=True, DEV=False, TEST=False, GT_opened=False, feature_sha256=feature_sha,
                        images=len(data), preparation=prepared, capture_accepted_sha256=a.capture_accepted_sha))
    step, trainer = training_sources(a.code)
    label_start = perf_counter()
    records, labels = label(a, data, step)
    label_seconds = perf_counter() - label_start
    del data
    fitted = fit_compact(records, step, trainer)
    need(fitted["audit"]["supervised_pairs"] == labels["supervised_pairs"] and not torch.cuda.is_initialized(),
         "All and only original matched FIT supervision on CPU")
    arrays = dict(center=fitted["center"], scale=fitted["scale"])
    arrays.update({f"S.{name}": value.detach().numpy() for name, value in fitted["model"].state_dict().items()})
    weights = a.output / "target_5217_all_fit.npz"
    with weights.open("xb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    weights_sha = sha256(weights.read_bytes()).hexdigest()
    restored_step, restored, center, scale = load_head(a.code, weights, weights_sha)
    need(restored_step.tensor_hash(restored) == fitted["audit"]["final_tensor_sha256"]
         and np.array_equal(center, arrays["center"]) and np.array_equal(scale, arrays["scale"]),
         "Exact compact weight and scaler readback")
    audit = dict(status=COMPLETE, FIT=True, DEV=False, TEST=False, AP=False, FIT_GT_opened=True,
                 detector_forwards=0, detector_optimizer_updates=0, images=len(capture_rows), selected_rgb_groups=109,
                 preparation=prepared, supervision=labels, weights_sha256=weights_sha, weights_name=weights.name,
                 protocol_sha256=PRELOCK_SHA, checkpoint_sha256=CHECKPOINT_SHA, fit_gt_sha256=FIT_SHA,
                 capture_accepted_sha256=a.capture_accepted_sha, selection_sha256=verify["selection_sha256"],
                 prelabel_sha256=prepare_sha, feature_sha256=feature_sha, feature_source_sha256=REPLAY_SHA,
                 training_source_sha256=STEP_SHA, worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                 dtype="float32", device="cpu", torch_version=torch.__version__,
                 feature_seconds=feature_seconds, label_seconds=label_seconds,
                 elapsed_seconds=perf_counter() - started, **fitted["audit"])
    result_sha = write(a.output / "RESULT_SAFE.json", audit)
    print(json.dumps(dict(status=COMPLETE, result_sha256=result_sha, optimizer_updates=STEPS)), flush=True)


if __name__ == "__main__":
    main()
