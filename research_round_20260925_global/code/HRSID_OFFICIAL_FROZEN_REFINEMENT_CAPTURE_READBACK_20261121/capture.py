"""Label-free per-image treatment for the closed 800px CPU native export.

The owner supplies a fresh CPU forward, image-only RGB and one exact reference
line. It controls the whole-population stream, endpoint and process admission.
"""
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from types import ModuleType, SimpleNamespace

import numpy as np
from pycocotools import mask as codec
import torch


SEEDS = (5217, 5218, 5219)
ARMS = ("N", "SF", "SU=F", "SU", "LW=S")
READOUT_SHA = "543bc7ffae51cb0de5b247fef59eeae6309a155440021d3ed50a7b3716ec8e9f"
FIT_ACCEPTED_SHA = "476b0f3fa748d1d3fd2393bdeb1af2d83e1532995d58b12625d57222e3f2638a"


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def load_policy(code_dir):
    """Load the exact published point policy, feature modules and head classes."""
    code = Path(code_dir)
    formal = code / "RFDETR_MATCHED_POINT_FORMAL_20261120"
    spot_dir = code / "RFDETR_CORE36_MATCHED_POINT_SPOT_20261120"
    dfine = code / "DFINE_SEG_EXTERNAL_BASELINE_FIT32_E24_20261112/e24"
    path = formal / "run_dev_readout.py"
    source = path.read_bytes()
    require(sha256(source).hexdigest() == READOUT_SHA, "Published matched policy source")
    readout = ModuleType("_matched_readout")
    readout.__file__ = str(path)
    exec(compile(source, str(path), "exec"), readout.__dict__)

    def original(directory, name):
        return readout.load(directory / name, readout.PINS[name], "_" + Path(name).stem)

    spot = original(spot_dir, "run_spot.py")
    fit_step = original(spot_dir, "fit_step.py")
    return SimpleNamespace(
        readout=readout, spot=spot, fit_step=fit_step, plan=original(formal, "fit_plan.py"),
        features=original(spot_dir, "feature_adapter.py"),
        posterior=fit_step.load(code / "RFDETR_MATCHED_POINT_REFINEMENT_20261120/adapter.py",
                                spot.ADAPTER_SHA, "_native_posterior"),
        pair=fit_step.load(dfine / "fit109_paired_point/pair_math.py", spot.PAIR_SHA, "_pairs"),
        rank=fit_step.load(dfine / "fit109_probability_rank/rank_math.py", spot.RANK_SHA, "_rank"),
        trainer=fit_step.load_trainer(dfine / "fit109_paired_point/train_point.py"),
    )


def restore_refits(policy, fit_result, fit_acceptance, heads_dir, *, fit_result_sha, fit_acceptance_sha):
    """Restore each all-FIT S/L bundle and the exact original 14-feature scaler."""
    require(fit_acceptance_sha == FIT_ACCEPTED_SHA, "Independently accepted original FIT109 receipt")
    readout = policy.readout
    accepted = json.loads(readout.pin(Path(fit_acceptance), fit_acceptance_sha))
    result = json.loads(readout.pin(Path(fit_result), fit_result_sha, 8 * 1024**2))
    rows = readout.validated_refits(result, accepted, fit_result_sha)
    return {seed: policy.plan.restore_bundle(Path(heads_dir) / row["relative_weights"],
                                             row["weights_sha256"], policy.trainer, policy.fit_step)
            for seed, row in rows.items()}


def encode_native(block):
    return (json.dumps(block, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode()


def mapped_queries(raw, image_id, expected_line, native_rows, postprocess_class):
    class Observed(postprocess_class):
        def _select_topk(self, logits):
            self.selection = super()._select_topk(logits)
            return self.selection

    post = Observed(num_select=100, upsample_masks_to_image_size=True)
    native = native_rows(raw, image_id=image_id, hw=(800, 800), torch=torch,
                         postprocess=post, np=np, codec=codec)
    require(encode_native(native) == expected_line, "Original native prediction line byte parity")
    scores, labels, ids = (value[0] for value in post.selection)
    flat_scores = raw["pred_logits"][0].sigmoid().flatten()
    order = torch.argsort(flat_scores, descending=True, stable=True)[:100]
    require(torch.equal(ids, order // 2) and torch.equal(labels, order % 2) and
            torch.equal(scores, flat_scores.index_select(0, order)), "Stable author joint query selection")
    positions = (labels == 0).nonzero(as_tuple=True)[0]
    require(native["native_ranks"] == (positions + 1).tolist() and
            [r["score"] for r in native["coco"]] == scores.index_select(0, positions).tolist(),
            "Mapped native ranks and scores")
    return native, ids.index_select(0, positions).tolist()


def edited_rle(stock, edit, features):
    changed = features.apply_sparse(np, stock, edit)
    rle = codec.encode(np.asfortranarray(changed, dtype=np.uint8))
    require(int(codec.area(rle)) == int(stock.sum()) + edit["area_delta"], "Edited raster area")
    return dict(size=rle["size"], counts=rle["counts"].decode("ascii"))


def f32_bytes(values):
    return np.asarray(values, dtype="<f4", order="C").tobytes(order="C")


def _treatment(raw, rgb, native, query_ids, policy, refits):
    # No-op indices share their immutable native records; each arm has its own row list.
    predictions = {s: {a: dict(native, coco=list(native["coco"])) for a in ARMS} for s in SEEDS}
    boxes = policy.spot.author_boxes(raw["pred_boxes"][0], torch)
    image_features = None
    candidates = []
    eligible = supported = 0
    for i, (rank, qid, row) in enumerate(zip(native["native_ranks"], query_ids, native["coco"], strict=True)):
        original = row["segmentation"]
        area = int(codec.area(original))
        witness = dict(rank=rank, query_id=qid, area=area, score=row["score"])
        candidates.append(witness)
        if not 32 <= area < 1024:
            continue
        eligible += 1
        stock = codec.decode(original).astype(bool)
        probability = policy.features.render_posterior(
            torch, np, policy.posterior, raw["pred_masks"][0, qid].numpy(), stock)
        witness.update(coordinates=[], edits={})
        if not policy.spot.has_author_support(boxes[qid]):
            continue
        if image_features is None:
            image_features = policy.pair.image_features(rgb)
        record = policy.features.extract_candidate(np, policy.pair, policy.rank, stock, boxes[qid],
                                                   row["score"], probability, image_features)
        if record is None:
            continue
        supported += 1
        coords, points = record["coordinates"], record["features"]
        witness["coordinates"] = coords.tolist()
        witness["feature_sha256"] = sha256(f32_bytes(points)).hexdigest()
        witness["point_scores_f32le_hex"] = {}
        cap = min(8, area // 10, len(coords))
        # Reuse identical edit sets across heads/arms only within the same original mask.
        encoded_edits = {}
        for seed in SEEDS:
            heads, center, scale = refits[seed]
            small = policy.fit_step.predict(heads["S"], points, center, scale)
            large = policy.fit_step.predict(heads["L"], points, center, scale)
            witness["point_scores_f32le_hex"][str(seed)] = {"S": f32_bytes(small).hex(),
                                                           "L": f32_bytes(large).hex()}
            edits = policy.features.readouts_sparse(np, stock, coords, small, large, points[:, :, 0], cap)
            selected = {}
            for arm in ARMS[1:]:
                edit = edits[arm]
                if edit["edits"]:
                    selected[arm] = policy.readout.sparse_edit(edit)
                    key = tuple(sorted(selected[arm]["flipped"]))
                    if key not in encoded_edits:
                        encoded_edits[key] = edited_rle(stock, edit, policy.features)
                    predictions[seed][arm]["coco"][i] = dict(row, segmentation=encoded_edits[key])
            if selected:
                witness["edits"][str(seed)] = selected
    return predictions, dict(image_id=native["image_id"], mapped=len(candidates), eligible=eligible,
                             supported=supported, candidates=candidates)


@torch.inference_mode()
def compute_image(*, image_id, rgb, expected_native_line, forward, native_rows,
                  postprocess_class, policy, refits):
    """Return all seed/arm native blocks, one rank/query witness and CPU wall timings.

    forward() returns raw output for the owner's checked CPU sample. Time its input
    decoding/preparation separately if performed before this call.
    """
    require(set(refits) == set(SEEDS) and type(image_id) is int, "Three frozen seeds and original image ID")
    require(isinstance(rgb, np.ndarray) and rgb.shape == (800, 800, 3) and rgb.dtype == np.uint8,
            "Original RGB800 pixels")
    require(type(expected_native_line) is bytes and expected_native_line.endswith(b"\n"), "Closed native line")
    t0 = perf_counter()
    raw = forward()
    t1 = perf_counter()
    for key, shape in (("pred_logits", (1, 164, 2)), ("pred_boxes", (1, 164, 4)), ("pred_masks", (1, 164, 144, 144))):
        val = raw[key]
        require(val.device.type == "cpu" and val.dtype == torch.float32 and tuple(val.shape) == shape,
                "Exact CPU FP32 detector output " + key)
    native, query_ids = mapped_queries(raw, image_id, expected_native_line, native_rows, postprocess_class)
    t2 = perf_counter()
    predictions, witness = _treatment(raw, rgb, native, query_ids, policy, refits)
    t3 = perf_counter()
    witness["native_line_sha256"] = sha256(expected_native_line).hexdigest()
    witness["joint_scores_f32le_hex"] = f32_bytes(raw["pred_logits"][0].sigmoid().numpy()).hex()
    return predictions, witness, dict(detector_forward_seconds=t1-t0, native_postprocess_parity_seconds=t2-t1,
                                      refinement_cpu_seconds=t3-t2)
