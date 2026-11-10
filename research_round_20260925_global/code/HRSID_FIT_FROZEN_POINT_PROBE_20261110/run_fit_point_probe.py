"""Fixed FIT-only paired-view probe of point target construction and allocation."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from statistics import fmean
from time import perf_counter
from types import ModuleType
from zipfile import ZipFile
import json
import os
import sys


PAIRS = 8
PILOT_SHA = "0b9726556a27dcca40ca9571390a8d04bb6c0dea7f4d982c29235283791858d9"
CHECKPOINT_SHA = "198bc09cb93019e719342914965dffb31160bb284003a17b334f998c1f540c56"
TRAIN_PROTOCOL = "d66baf3c57fafdba4242ad1b55e08cf76bfd77f7"
EXTRA_RUNTIME = {
    "rfdetr.models.matcher": "2521d58fd5c3270f8f352c14ae299364347bd27792e18696de8bb20c637765f6",
    "rfdetr._namespace": "7b5ab31bd5a8f99b2588c766a3a8c5193d79fd7326c84f3610282f33143f7280",
    "rfdetr.models.heads.segmentation": "5de847861e2bd2fab8853f442e263f4e9c0eb75c5b3fbc6a03bb25e9c7115e1e",
}
BANKS = ("phase", "uncertainty", "random", "random_expected")
EFFECT = "mean_abs_polygon_minus_raster"


def require(ok, why):
    if not ok:
        raise ValueError(why)


def module(path, expected, name):
    raw = path.read_bytes()
    require(path.resolve(strict=True) == path and sha256(raw).hexdigest() == expected,
            "Pinned FIT probe source differs: " + path.name)
    target = ModuleType(name)
    target.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), target.__dict__)
    return target


def selected_pairs(history, count):
    groups = {}
    for cursor, row in enumerate(history):
        if row["layer"] == "core":
            groups.setdefault(row["ordinal"], {}).setdefault(row["path"], cursor)
    return [(ordinal, paths["crop"], paths["direct"]) for ordinal, paths in sorted(groups.items())
            if paths.keys() >= {"crop", "direct"}][:count]


def boundary_bank(hard):
    import numpy as np
    from scipy.ndimage import binary_dilation, binary_erosion

    require(hard.ndim == 2 and hard.dtype == np.bool_ and hard.any(), "Nonempty binary mask required")
    kernel = np.ones((3, 3), dtype=bool)
    foreground_one = binary_erosion(hard, structure=kernel, border_value=0)
    foreground_two = binary_erosion(foreground_one, structure=kernel, border_value=0)
    grown_one = binary_dilation(hard, structure=kernel, border_value=0)
    grown_two = binary_dilation(grown_one, structure=kernel, border_value=0)
    labels = np.full(hard.shape, -1, dtype=np.int8)
    labels[hard & ~foreground_one] = 0
    labels[foreground_one & ~foreground_two] = 1
    labels[grown_one & ~hard] = 2
    labels[grown_two & ~grown_one] = 3
    y, x = np.nonzero(labels >= 0)
    return np.column_stack((x, y)), labels[y, x]


def fixed_view(native, audit, image, targets, old, torch):
    from common_mask_supervision import ROW_ID, common_targets
    from one_pass_fit_transform import substitute_one_pass_masks
    from paired_fit_transform import _check_output, _prepare, _rng, _rng_bytes
    from replay_fit import normalized_trace
    from trace_resize import capture_transform

    original = _prepare(image, targets)
    transformed = []
    for mode in ("native", "center"):
        native.set_seed(old["seed"])
        with capture_transform(native._torchvision, mode=mode) as trace:
            pixels, target = audit.transform(image.copy(), {key: value.clone() for key, value in original.items()})
        _check_output(pixels, target, original)
        transformed.append((pixels, target, trace, _rng_bytes(_rng())))
    pixels, a, trace, rng = transformed[0]
    pixels_b, b, trace_b, rng_b = transformed[1]
    a_live, b_live = (set(t[ROW_ID][~native.mask_empty(t)].tolist()) for t in (a, b))
    prev = old["rows"]
    steps = normalized_trace(trace)
    require(torch.equal(pixels, pixels_b) and trace == trace_b and rng == rng_b
            and old["steps"] == steps
            and old["flip"] == any(row["op"] == "hflip" and row["applied"] for row in trace)
            and (prev["raw_rows"], prev["legacy_present"], prev["center_present"],
                 prev["legacy_nonempty"], prev["center_nonempty"], prev["common_nonempty"])
            == (len(original[ROW_ID]), len(a[ROW_ID]), len(b[ROW_ID]), len(a_live), len(b_live), len(a_live & b_live)),
            "Actual paired augmentation must match closed historical view")
    c = substitute_one_pass_masks(original["masks"], b, trace)
    a, b, c = common_targets(a, b, c)
    require(old["path"] != "direct" or torch.equal(b["masks"], c["masks"]), "Direct B/C exact equality required")
    return pixels, original, b, c, steps


def analysis_summary(rows):
    from math import isclose

    summary = {}
    for path in ("crop", "direct"):
        selected = [item for item in rows if item["path"] == path]
        groups = defaultdict(list)
        for item in selected:
            groups[item["ordinal"]].append(item["summary"])
        require(len(groups) == PAIRS, "Every fixed source must provide a small eligible row on both paths")
        for group in groups.values():
            require(all(x["k"] <= 16 for x in group), "Exact per-row point budget")
        fields = (EFFECT, "mean_abs_raster_minus_hard_c", "mean_abs_polygon_minus_hard_c")
        means = {
            bank: {metric: fmean(fmean(x["banks"][bank][metric] for x in group) for group in groups.values())
                   for metric in fields} for bank in BANKS
        }
        target_keys = ("mean_bce", "mean_abs_bce_gradient", "dice_loss")
        loss = {bank: {target: {key: (
            None if bank == "random_expected" and key == "dice_loss" else
            fmean(fmean(x["banks"][bank]["targets"][target][key] for x in group) for group in groups.values())
        ) for key in target_keys} for target in ("hard_c", "raster", "polygon")} for bank in BANKS}
        effects = []
        for ordinal, group in sorted(groups.items()):
            values = {bank: fmean(x["banks"][bank][EFFECT] for x in group) for bank in BANKS}
            effects.append(dict(ordinal=ordinal, rows=len(group),
                                phase_minus_expected=values["phase"]-values["random_expected"],
                                phase_minus_uncertainty=values["phase"]-values["uncertainty"]))
        gap = means["phase"][EFFECT] - means["random_expected"][EFFECT]
        require(isclose(gap, fmean(x["phase_minus_expected"] for x in effects), abs_tol=1e-12),
                "Independent paired point summary identity")
        ratio = (means["phase"][EFFECT] / means["random_expected"][EFFECT]
                 if means["random_expected"][EFFECT] else None)
        summary[path] = dict(
            images=len(groups), rows=len(selected), cells=sum(x["summary"]["n"] for x in selected),
            selected_points=sum(x["summary"]["k"] for x in selected),
            effective_proxy_rows=sum(bool(x["summary"]["effective_proxy"]) for x in selected),
            image_equal_target_difference=means, image_equal_loss_readback=loss, per_image_effect=effects,
            phase_minus_expected=gap, phase_over_expected=ratio,
            phase_minus_uncertainty=means["phase"][EFFECT]-means["uncertainty"][EFFECT],
            positive_images=sum(x["phase_minus_expected"] > 0 for x in effects),
        )
    crop, direct = summary["crop"], summary["direct"]
    return dict(by_path=summary, crop_minus_direct_phase_effect=crop["phase_minus_expected"]-direct["phase_minus_expected"],
                descriptive_gate=(crop["phase_minus_expected"] >= 0.01 and crop["phase_over_expected"] is not None
                                  and crop["phase_over_expected"] >= 1.25 and crop["positive_images"] >= 5))


def accepted_phase_evaluator(code, receipts):
    phase = module(code / "HRSID_MASK_PHASE_E24_DEV705_20261109/hrsid_mask_phase_dev705.py",
                   "488f9ed73943cc842162c0d45822b6bac42aecabf37c8ecf0d4dde8e96c0959b", "_p252_frozen_phase")
    arm = phase.read_pair(
        receipts / "PAIR_READY_PRIVATE.json",
        "3e8720e8fad83d6a4b313ee8b38e261cabb4f1a2a3eefe3a5e33e83d574ac1a1",
        native_path=receipts / "native/ARM_E24_ACCEPTED_PRIVATE.json",
        center_path=receipts / "center/ARM_E24_ACCEPTED_PRIVATE.json",
        mode="native",
    )
    require(phase.TRAINING_PROTOCOL == TRAIN_PROTOCOL and arm["checkpoint_sha256"] == CHECKPOINT_SHA,
            "Exact accepted phase checkpoint and training protocol required")
    filtered, base = phase.load_evaluator(
        code / "HRSID_COCO_E24_FILTERED_DEV705_NATIVE_EVAL_20261109/hrsid_native_dev705_filtered.py",
        code / "HRSID_COCO_E24_DEV705_NATIVE_EVAL_20261109/hrsid_native_dev705.py",
    )
    phase.bind_evaluator(base, filtered, arm)
    return base


def run(args):
    begin = perf_counter()
    require(sys.version_info[:3] == (3, 10, 13) and sys.flags.isolated == sys.flags.utf8_mode == 1
            and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1",
            "Exact isolated Windows CPU profile required")
    code = args.code
    for path in (code, args.bundle, args.checkpoint, args.shapely_site, args.receipts):
        require(path.is_absolute() and path.resolve(strict=True) == path, "Canonical sealed inputs required")
    require(args.output.is_absolute() and args.output.parent.resolve(strict=True) == args.output.parent
            and not args.output.exists() and not args.output.is_symlink(), "Fresh output required")
    pilot_dir = code / "HRSID_FIT114_VECTOR_LABEL_GEOMETRY_20261110"
    pilot = module(pilot_dir / "run_fit114_vector_trial.py", PILOT_SHA, "_p252_closed_pilot")
    native, np, torch, _ = pilot.install_support(code, args.shapely_site)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common_mask_supervision import ROW_ID
    from point_bank_analysis import summarize_row
    from point_occupancy import cell_occupancy
    from rfdetr.models.heads.segmentation import point_sample
    from rfdetr.models.lwdetr import build_criterion_from_config
    from rfdetr.utilities.tensors import nested_tensor_from_tensor_list
    from vector_label_geometry import UnsupportedPolygon, polygon_raster, polygon_rings

    base = accepted_phase_evaluator(code, args.receipts)
    base.RECIPE = code.parent / "configs/RFDETR_SEG_NANO_BASELINE_576_E100_20261107.yaml"
    base.FIT_AUDIT = code / "HRSID_OFFICIAL_PNG_FIT_AUDIT_20261109/hrsid_png_fit_audit.py"
    for name, expected in dict(base.MODEL_SOURCES, **EXTRA_RUNTIME).items():
        require(sha256(Path(import_module(name).__file__).read_bytes()).hexdigest() == expected,
                "Exact frozen model/matcher runtime differs: " + name)
    require(version("scipy") == "1.15.3" and not torch.cuda.is_initialized(), "Exact CPU geometry and matching")
    model, config, train, _ = base.load_model(args.checkpoint, CHECKPOINT_SHA, torch, args.output.parent, TRAIN_PROTOCOL)
    model.eval()
    criterion, _ = build_criterion_from_config(config, train)
    matcher = criterion.matcher
    audit = native.NativeInputAudit()
    rows, views = [], []
    skipped = Counter()
    ledger_sha = sha256()
    args.output.mkdir()
    with args.bundle.open("rb") as stream:
        before = pilot.fingerprint(os.fstat(stream.fileno()))
        source_sha = sha256()
        for chunk in iter(lambda: stream.read(1024**2), b""):
            source_sha.update(chunk)
        require(before[2] == pilot.BUNDLE_BYTES and source_sha.hexdigest() == pilot.BUNDLE_SHA, "Exact closed FIT packet")
        stream.seek(0)
        with (ZipFile(stream) as bundle, torch.inference_mode(),
              (args.output / "POINT_ROWS_PRIVATE.jsonl").open("xb") as ledger):
            cohort = json.loads(pilot.pinned_member(bundle, "COHORT_PRIVATE.json", pilot.COHORT_SHA, 1024**2))
            index = json.loads(pilot.pinned_member(bundle, "FIT114_COCO_PRIVATE.json", pilot.SUBSET_SHA, 4*1024**2))
            history = [json.loads(line) for line in pilot.pinned_member(
                bundle, "HISTORICAL_VIEWS_PRIVATE.jsonl", pilot.HISTORY_SHA, 1024**2).splitlines()]
            image_names = {row["id"]: row["file_name"] for row in index["images"]}
            labels = defaultdict(list)
            for annotation in index["annotations"]:
                require(annotation["image_id"] in image_names, "Closed FIT annotation image identity")
                labels[annotation["image_id"]].append(annotation)
            pair_list = selected_pairs(history, PAIRS)
            require(cohort["fit_only"] is True and len(cohort["images"]) == len(image_names) == 114
                    and len(history) == 456 and len(pair_list) == PAIRS, "Exact FIT-only paired probe population")
            for ordinal, crop_cursor, direct_cursor in pair_list:
                record = cohort["images"][ordinal]
                image_id, name = record["image_id"], record["native_basename"]
                require(record["core"] and image_names[image_id] == name, "Preselected core source")
                png = pilot.pinned_member(bundle, "images/" + name, record["encoded_sha256"], 4*1024**2)
                image, scales = native.io_utils.decode_pil_image_bytes(png, draft_size=None)
                require(image.size == (800, 800) and image.mode == "RGB" and scales == (1., 1.)
                        and len(png) == record["encoded_bytes"]
                        and pilot.digest(np.asarray(image).tobytes()) == record["decoded_rgb_sha256"], "Exact frozen decoded image")
                source_rows = labels[image_id]
                image, target = audit.convert(image, dict(image_id=image_id, annotations=source_rows))
                source = pilot.source_annotations(source_rows, target, (800, 800), torch)
                for cursor, path in ((crop_cursor, "crop"), (direct_cursor, "direct")):
                    old = history[cursor]
                    require(old["ordinal"] == ordinal and old["seed"] == pilot.BASE_SEED+cursor
                            and old["path"] == path and old["group"] == record["group"], "Exact historical view selection")
                    pixels, original, b, c, steps = fixed_view(native, audit, image, target, old, torch)
                    common_ids = c[ROW_ID].tolist()
                    require(tuple(pixels.shape) == (3, 576, 576) and 0 < len(common_ids) <= config.num_queries,
                            "Source view/query canvas")
                    sample = nested_tensor_from_tensor_list([pixels], block_size=config.patch_size*config.num_windows)
                    raw = model(sample)
                    require(tuple(raw["pred_masks"].shape) == (1, 164, 144, 144) and not torch.cuda.is_initialized(),
                            "Exact frozen source mask canvas")
                    torch.manual_seed(252001000+cursor)
                    [(queries, positions)] = matcher(raw, [c], group_detr=1)
                    assignment = dict(zip(positions.tolist(), queries.tolist(), strict=True))
                    require(set(assignment) == set(range(len(common_ids))) and len(set(assignment.values())) == len(common_ids),
                            "One common stock Hungarian match for every kept row")
                    valid = 0
                    for pos, row_id in enumerate(common_ids):
                        annotation = source[row_id]
                        if float(annotation["area"]) >= 1024:
                            continue
                        hard = c["masks"][pos].numpy()
                        previous = b["masks"][pos].numpy()
                        xy, strata = boundary_bank(hard)
                        seg = annotation["segmentation"]
                        raw_bitmap = original["masks"][row_id].numpy()
                        try:
                            occupancy = cell_occupancy(raw_bitmap, seg, steps, xy)
                            require(np.array_equal(raw_bitmap, polygon_raster(polygon_rings(seg), (800, 800))),
                                    "Same source polygon and stock bitmap")
                        except UnsupportedPolygon as exc:
                            skipped[str(exc)] += 1
                            continue
                        x, y = xy.T
                        query = assignment[pos]
                        coord = torch.as_tensor((xy+0.5)/576, dtype=torch.float32)[None]
                        z = point_sample(raw["pred_masks"][:, query:query+1], coord, align_corners=False).flatten().numpy()
                        bank = summarize_row(hard[y, x], previous[y, x], occupancy["bitmap"], occupancy["polygon"],
                                             z, strata, seed=252010000+1000*cursor+row_id)
                        item = dict(ordinal=ordinal, cursor=cursor, path=path, summary=bank)
                        rows.append(item)
                        line = (json.dumps(item, sort_keys=True, allow_nan=False)+"\n").encode()
                        require(ledger.write(line) == len(line), "Complete private audit row")
                        ledger_sha.update(line)
                        valid += 1
                    require(valid > 0, "Every prelocked view needs an eligible small row")
                    views.append(dict(ordinal=ordinal, cursor=cursor, path=path, common_rows=len(common_ids), eligible=valid))
                    print(json.dumps(dict(event="P252_FIT_POINT_PROBE_PROGRESS", views=len(views), last_rows=valid)), flush=True)
            ledger.flush()
            os.fsync(ledger.fileno())
        require(pilot.fingerprint(os.fstat(stream.fileno())) == pilot.fingerprint(args.bundle.stat()) == before,
                "Exact closed packet remained stable")
    require(len(views) == 2*PAIRS and not model.training and not torch.cuda.is_initialized(), "Complete fixed frozen probe")
    require(Counter(row["path"] for row in rows) == {"crop": 31, "direct": 35}
            and sum(skipped.values()) == 1, "Exact independently established FIT point eligibility")
    result = dict(status="P252_FIT_FROZEN_POINT_SUPERVISION_PROBE_COMPLETE", FIT=True, DEV=False, TEST=False, AP=False,
                  images=PAIRS, views=views, model_forward_calls=len(views), optimizer_updates=0, device="cpu", skipped=dict(skipped),
                  source_checkpoint_sha256=CHECKPOINT_SHA, fit_bundle_sha256=pilot.BUNDLE_SHA,
                  private_row_audit_sha256=ledger_sha.hexdigest(), summary=analysis_summary(rows), wall_seconds=perf_counter()-begin)
    path = args.output/"RESULT_SAFE.json"
    encoded = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()
    with path.open("xb") as out:
        require(out.write(encoded) == len(encoded), "Complete safe report")
        out.flush()
        os.fsync(out.fileno())
    print(json.dumps(dict(status=result["status"], result_sha256=sha256(encoded).hexdigest(),
                          views=len(views), rows=len(rows), wall_seconds=result["wall_seconds"])), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("code", "bundle", "checkpoint", "shapely-site", "receipts", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    run(parser.parse_args())
