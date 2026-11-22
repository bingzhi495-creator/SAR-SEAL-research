"""Prepare four real HRSID FIT sources and fit the compact residual occupancy head."""
import argparse
from contextlib import redirect_stdout
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
from io import BytesIO, StringIO
import json
from pathlib import Path

import numpy as np
import torch

CODE = Path(__file__).resolve().parents[1]
CARRIER = "011bd30ed056c871be1d61e14d71b3b2b258330a9d20cf2f164ca4a7c4f23c11"
ANNOTATIONS = "4ac69031accd771b784f90e128f206373f7332293a0130324333154d723541fc"
# First four images in the accepted FIT109 manifest, ordered by decoded RGB digest.
# (author PNG name, historical FIT id, PNG SHA-256, optional closed feature archive SHA-256)
SOURCES = (
    ("P0120_3000_3800_7200_8000.png", 3310, "3c9a6b7fb778bf9b7f0decce1c7c2b78fdc7d6c148b84f821204a02b98e8055e", "55dc5de6f457cee99d18d88f7e3b0e80686522a2b70a030253b94f91391b0832"),
    ("P0084_1800_2600_600_1400.png", 2676, "ff22b8542ba71d1d19eb799743a788fc742843756e705ab751a95d875ab60180", "3b2fcf0e190136ad99084afe97e250be26a5100c35d06e8aceb72fba8c7ac66b"),
    ("P0131_13200_14000_9000_9800.png", 3480, "5abff190cfb17c0ea36bfd66701c7c9796a6729f5328810f759a841138bb30c8", "0e72fa76de1e78ce0dda818c43b6bfacc59882a9bae93c136a670f4cc075b2a0"),
    ("P0031_4800_5600_3000_3800.png", 1071, "66d7a07085b28d61b3c9b16c580f86a1b08d740d2961357ca21f048cc5a37b64", "a6c9cd9404c44a98da6681f921c4198c9ae3c9d4c2381a28cb3cef7680c838d8"),
)
QUERIES = "62737cefda8d04fe44069e0e9e515fb1081038ebea78ccbfcbad491c80e3da15"
# Independent archived HRSID features, paired with the original author polygons.
# Other CPU implementations can report their exact numerical agreement separately.
REFERENCE_TENSORS = {
    "features": "73680fb05e505abf14f227ea965145298d26d4cdd0cd499fa468ddf65fad70e1",
    "targets": "f53a2da5f3a184f56ac75445b73df913c49161e40693cb1a83a826641afff6d8",
    "groups": "352dc2ab5afcf557af3ad6e61496be49ea6c2c9fae63c87edd55a8df924f6114",
    "matches": "ee57ac2e07d7b24d3105aa485674597f1c963ccf86c8bf43de948af2ec5a18f6",
}
E24 = "DFINE_SEG_EXTERNAL_BASELINE_FIT32_E24_20261112/e24/"
SPOT = "RFDETR_CORE36_MATCHED_POINT_SPOT_20261120/"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def pinned(path, digest):
    data = path.read_bytes()
    require(sha256(data).hexdigest() == digest, "Unexpected SHA-256 for " + path.name)
    return data


def module(code, relative):
    path = code / relative
    spec = spec_from_file_location("_b5_" + path.stem, path)
    result = module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def candidates(query):
    from pycocotools import mask
    pool = []
    for rank, qid in zip(query["native_ranks"], query["native_ids"], strict=True):
        rle = query["all_rles"][qid]
        pool.append(dict(rank=rank, query_id=qid, rle=rle, area=int(mask.area(rle)),
                         score=query["ship_scores"][qid]))
    return pool


def archived_sources(args):
    query_lines = pinned(args.queries, QUERIES).splitlines()
    for index, (name, image_id, _, asset_hash) in enumerate(SOURCES):
        query = json.loads(query_lines[index])
        require(query["image_id"] == image_id and query["native_basename"] == name, "Source query identity")
        name_npz = f"NATIVE_ENDPOINT_BASIS_{index + 1:02d}_PRIVATE.npz"
        with np.load(BytesIO(pinned(args.features_dir / name_npz, asset_hash)), allow_pickle=False) as data:
            require(int(data["image_id"]) == image_id, "Source feature image identity")
            ranks, qids, offsets = data["ranks"], data["query_ids"], data["offsets"]
            coords, features = data["coords"], data["old14"]
        supported = {int(rank): (int(qids[i]), coords[offsets[i]:offsets[i+1]], features[offsets[i]:offsets[i+1]])
                     for i, rank in enumerate(ranks)}
        yield name, candidates(query), supported


def fresh_sources(args):
    from PIL import Image
    from pycocotools import mask
    from rfdetr.models.postprocess import PostProcess
    from rfdetr.utilities.tensors import nested_tensor_from_tensor_list

    entry = module(Path(__file__).resolve().parent, "portable_eval.py")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    model, transform, config = entry.make_detector(args.code, args.checkpoint, args.output, torch, device="cpu")
    post = module(args.code, "HRSID_FIT_NATIVE_QUERY_RETENTION_20261109/fit_query_export.py")
    pair = module(args.code, E24 + "fit109_paired_point/pair_math.py")
    rank_math = module(args.code, E24 + "fit109_probability_rank/rank_math.py")
    adapter = module(args.code, SPOT + "feature_adapter.py")
    spot = module(args.code, SPOT + "run_spot.py")
    for name, _, png_sha, _ in SOURCES:
        with Image.open(BytesIO(pinned(args.images / name, png_sha))) as img:
            require(img.mode == "RGB" and img.size == (800, 800), "Native HRSID PNG expected")
            texture = pair.image_features(np.asarray(img))
            tensor, _ = transform(img, None)
        require(tensor.shape == (3, 576, 576), "Frozen validation transform")
        with torch.inference_mode():
            batch = nested_tensor_from_tensor_list([tensor], block_size=config.patch_size * config.num_windows)
            raw = model(batch)
            query = post.aligned_query_rows(raw, (800, 800), torch=torch, np=np, codec=mask, processor=PostProcess)
            boxes = spot.author_boxes(raw["pred_boxes"][0], torch)
            pool = candidates(query)
            supported = {}
            for row in pool:
                qid = row["query_id"]
                if not (32 <= row["area"] < 1024 and spot.has_author_support(boxes[qid])):
                    continue
                logits = raw["pred_masks"][0, qid][None, None]
                plane = torch.nn.functional.interpolate(logits, size=(800, 800), mode="bilinear", align_corners=False)[0, 0]
                stock = mask.decode(row["rle"]).astype(bool)
                require(np.array_equal(plane.numpy() > 0, stock), "Carrier mask and logit raster disagree")
                points = adapter.extract_candidate(np, pair, rank_math, stock, boxes[qid], row["score"],
                                                   plane.sigmoid().numpy(), texture)
                if points is not None:
                    supported[row["rank"]] = (qid, points["coordinates"], points["features"])
        yield name, pool, supported


def prepare(args, archived):
    import shapely
    require(not args.output.exists(), "A fresh destination for HRSID samples is required")
    source = json.loads(pinned(args.hrsid_annotations, ANNOTATIONS))
    images_by_name = {im["file_name"]: im for im in source["images"]}
    geometry = module(args.code, "HRSID_DEV705_CONTINUOUS_AP_20261111/full_continuous_ap.py")
    vector = module(args.code, "HRSID_FIT114_VECTOR_LABEL_GEOMETRY_20261110/vector_label_geometry.py")
    matcher = module(args.code, "HRSID_FIT_HULL_CDF_20261112/stock_coco_match.py")
    features, targets, groups, matched = [], [], [], []
    rows = archived_sources(args) if archived else fresh_sources(args)
    for index, (name, pool, supported) in enumerate(rows):
        image = images_by_name[name]
        anns = [ann for ann in source["annotations"] if ann["image_id"] == image["id"]]
        subset = dict(images=[image], annotations=anns, categories=[dict(id=1, name="ship")])
        truth, neutral, _ = geometry.polygon_support(subset, vector)
        with redirect_stdout(StringIO()):
            matches, _, _ = matcher.freeze_coco(image, anns, pool)
        by_rank = {rank: aid for aid, rank in matches.items()}
        candidate_count = 0
        for row in pool:
            rank = row["rank"]
            aid = by_rank.get(rank)
            if rank not in supported or aid not in truth or aid in neutral:
                continue
            qid, coords, values = supported[rank]
            require(qid == row["query_id"] and np.all((0 <= coords) & (coords < 800**2)), "FIT endpoint identity")
            y, x = np.divmod(coords, 800)
            cells = shapely.box(x, y, x + 1, y + 1)
            q = shapely.area(shapely.intersection(cells, truth[aid][0]))
            require(q.shape == values.shape[:2] and np.isfinite(q).all() and np.all((q >= 0) & (q <= 1 + 1e-12)),
                    "Invalid unit-pixel polygon occupancy")
            features.append(values)
            targets.append(q.astype(np.float32))
            groups.append(np.full(len(q), index, dtype=np.int32))
            candidate_count += 1
        matched.append(candidate_count)
    require(len(matched) == len(SOURCES) and features, "Empty HRSID calibration population")
    with args.output.open("xb") as out:
        np.savez_compressed(out, features=np.concatenate(features), targets=np.concatenate(targets), groups=np.concatenate(groups),
                            matches=np.asarray(matched, dtype=np.int32), annotation_sha256=ANNOTATIONS, carrier_sha256=CARRIER,
                            source_route="closed_features" if archived else "fresh_png_cpu",
                            source_names=np.array([row[0] for row in SOURCES]))


def load_head(checkpoint, code=None):
    trainer = module(CODE if code is None else Path(code), E24 + "fit109_paired_point/train_point.py")
    model = trainer.PointHead(14)
    with np.load(checkpoint, allow_pickle=False) as bank:
        states = {"S." + name: orig for name, orig in model.state_dict().items()}
        require(set(bank.files) == set(states) | {"center", "scale"}, "Unexpected checkpoint tensors")
        for key, orig in states.items():
            weights = bank[key]
            require(weights.shape == tuple(orig.shape) and weights.dtype == np.float32 and np.isfinite(weights).all(),
                    "Invalid point head tensor")
        model.load_state_dict({k[2:]: torch.from_numpy(bank[k]) for k in states})
        center, scale = bank["center"], bank["scale"]
    require(all(v.shape == (14,) and v.dtype == np.float32 and np.isfinite(v).all() for v in (center, scale))
            and np.all(scale > 0), "Invalid source normalization")
    return model.eval(), center, scale


def train(args):
    require(not args.output.exists(), "A fresh training output directory is required")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    with np.load(args.samples, allow_pickle=False) as bank:
        require(bank["source_names"].tolist() == [row[0] for row in SOURCES] and str(bank["carrier_sha256"]) == CARRIER
                and str(bank["annotation_sha256"]) == ANNOTATIONS, "Expected source PNG cohort and reference labels")
        feat, q, spans = bank["features"], bank["targets"], bank["groups"]
        route, matches = str(bank["source_route"]), bank["matches"]
    n = len(feat)
    require(n > 0 and feat.shape == (n, 2, 14) and feat.dtype == np.float32 and np.isfinite(feat).all()
            and np.all((feat[:, :, 0] >= 0) & (feat[:, :, 0] <= 1)), "FP32 paired native point features")
    require(q.dtype == np.float32 and q.shape == (n, 2) and np.isfinite(q).all() and np.all((q >= 0) & (q <= 1))
            and spans.dtype == np.int32 and spans.shape == (n,) and np.all((spans >= 0) & (spans < len(SOURCES)))
            and matches.shape == (len(SOURCES),) and route in {"fresh_png_cpu", "closed_features"}, "HRSID sample layout")
    fractional = int(np.count_nonzero((q > 0) & (q < 1)))
    require(fractional > 0, "Expected genuinely fractional polygon intersections")
    tensor_hashes = {name: sha256(value.tobytes(order="C")).hexdigest() for name, value in
                     (("features", feat), ("targets", q), ("groups", spans), ("matches", matches))}
    require(route != "closed_features" or tensor_hashes == REFERENCE_TENSORS, "Closed FIT samples changed")
    trainer = module(args.code, E24 + "fit109_paired_point/train_point.py")
    order = sorted(np.unique(spans), key=lambda i: SOURCES[i][0])
    group_rows = [np.flatnonzero(spans == g) for g in order]
    center, scale = trainer.group_moments(feat, group_rows)
    x = torch.from_numpy((feat - center) / scale)
    p, y = torch.from_numpy(feat[:, :, 0].copy()), torch.from_numpy(q)
    torch.manual_seed(5217)
    model = trainer.PointHead(14)
    initial = [v.detach().clone() for v in model.parameters()]
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.01)
    rng = np.random.default_rng(20261112)
    losses, norms = [], []
    draw = rng.integers(len(group_rows), size=(args.steps, 64))
    for group_ids in draw:
        k = [group_rows[g][rng.integers(len(group_rows[g]))] for g in group_ids]
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x[k], x[k], p[k]), y[k])
        require(bool(torch.isfinite(loss)), "Nonfinite occupancy loss")
        loss.backward()
        norms.append(float(torch.stack([v.grad.detach().double().norm() for v in model.parameters()]).norm()))
        require(np.isfinite(norms[-1]), "Nonfinite occupancy gradient")
        optimizer.step()
        losses.append(float(loss.detach()))
    update = float(torch.stack([(v.detach() - old).double().norm() for v, old in zip(model.parameters(), initial)]).norm())
    require(update > 0 and np.isfinite(update) and max(norms) > 0
            and all(int(v["step"]) == args.steps for v in optimizer.state.values()), "Actual optimizer updates required")
    args.output.mkdir()
    checkpoint = args.output / "HEAD.npz"
    with checkpoint.open("xb") as out:
        np.savez(out, center=center, scale=scale, **{"S." + key: val.detach().numpy() for key, val in model.state_dict().items()})
    restored, c, s = load_head(checkpoint, args.code)
    with torch.inference_mode():
        pred = restored(x, x, p).sigmoid()
        require(np.array_equal(c, center) and np.array_equal(s, scale) and torch.equal(model.eval()(x, x, p).sigmoid(), pred)
                and bool(torch.isfinite(pred).all()), "Checkpoint prediction readback failed")
    report = dict(status="HRSID_MINIMAL_FIT_COMPLETE", source_dataset="HRSID", source_route=route, source_images=len(SOURCES),
                  source_groups=len(order), source_pairs=n, source_matches=int(matches.sum()), fractional_endpoints=fractional,
                  matched_per_source=matches.tolist(), source_tensor_sha256=tensor_hashes,
                  exact_closed_source_samples=tensor_hashes == REFERENCE_TENSORS,
                  source_annotations_sha256=ANNOTATIONS, source_samples_sha256=sha256(args.samples.read_bytes()).hexdigest(),
                  carrier_sha256=CARRIER, seed=5217, sample_seed=20261112, optimizer="AdamW", optimizer_steps=args.steps,
                  batch_pairs=64, learning_rate=.002, weight_decay=.01, parameters=sum(v.numel() for v in model.parameters()),
                  loss_first=losses[0], loss_last=losses[-1], gradient_norm_max=max(norms), parameter_update_l2=update,
                  checkpoint_sha256=sha256(checkpoint.read_bytes()).hexdigest(), reload_predictions_exact=True,
                  detector_optimizer_updates=0, target_labels_used=False)
    (args.output / "FIT_REPORT.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    for name in ("prepare", "prepare-archive", "train"):
        command = actions.add_parser(name)
        command.add_argument("--code", type=Path, default=CODE, help="published research_round_20260925_global/code")
        command.add_argument("--output", type=Path, required=True)
        if name != "train":
            command.add_argument("--hrsid-annotations", type=Path, required=True, help="official HRSID PNG train2017.json")
        if name == "prepare":
            command.add_argument("--images", type=Path, required=True, help="directory containing the four original PNGs")
            command.add_argument("--checkpoint", type=Path, required=True, help="source RF-DETR E24 detector")
        elif name == "prepare-archive":
            command.add_argument("--features-dir", type=Path, required=True)
            command.add_argument("--queries", type=Path, required=True)
        else:
            command.add_argument("--samples", type=Path, required=True)
            command.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()
    args.code = args.code.resolve(strict=True)
    args.output = args.output.resolve()
    if args.action == "train":
        require(args.steps > 0, "Positive update count")
        train(args)
    else:
        if args.action == "prepare":
            args.checkpoint = args.checkpoint.resolve(strict=True)
        prepare(args, args.action == "prepare-archive")


if __name__ == "__main__":
    main()
