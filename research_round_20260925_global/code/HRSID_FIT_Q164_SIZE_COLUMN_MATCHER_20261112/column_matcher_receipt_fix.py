"""Switch exact RF-DETR matcher target columns on a frozen HRSID FIT carrier."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from hashlib import sha256
from importlib import import_module
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType

PLAN = "b88de1299f215c84b80742b53c5a73c030534cece5f6a37bf2ea7a983e25ab7d"
FIT = "e95cf26cc60a2c6fb651e907541380a256f2df7b1e153244a6c0f803b0b19f8f"
MANIFEST = "696ce59e06a847166609bc8840a7e11b898e793da19a75f6ad21b27541106d5b"
PAIR_378 = "f7539b13ea4bb5ab8b89c6d036ee89109b9759b78dd43fb01b0743319e51dca5"
PAIR_380 = "3f48c8ca037f432a4233fcbb564e7c95a9451118dbdcebed6745e9b16cbe855b"
CELLS = ("none", "nonlarge", "large", "all")
CONTRASTS = (("none", "nonlarge"), ("none", "large"), ("none", "all"), ("nonlarge", "all"))
GROUPS = ("all_images", "mixed", "large_only")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def pin(path, digest, max_bytes=1024**2):
    require(path.is_absolute() and path.resolve(strict=True) == path
            and path.is_file() and 0 < path.stat().st_size <= max_bytes, "Canonical bounded input required")
    raw = path.read_bytes()
    require(sha256(raw).hexdigest() == digest, "Pinned input differs: " + path.name)
    return raw


def read(path, digest, maximum=1024**2):
    return json.loads(pin(path, digest, maximum))


def load(path, digest, name):
    mod = ModuleType(name)
    mod.__file__ = str(path)
    exec(compile(pin(path, digest), str(path), "exec"), mod.__dict__)
    return mod


def closed(path, digest):
    w = read(path, digest)
    require(w["original_os_wait"] and w["original_os_exit"] == 0 and w["process_group_closed"]
            and w["stop_reason"] is None, "Closed original worker required")


def tool_wait(path, digest):
    w = read(path, digest)
    require(w["terminal_observed"] and w["original_exec"]["exit_code"] == 0, "Closed original owner required")


def input_authority(root):
    a = root / "P378_HRSID_FIT_LARGE_PAIR_Q164_CPU_ACTUAL_PRIVATE_20261111_01"
    ad = root / "P378_HRSID_FIT_LARGE_PAIR_Q164_CPU_DRAFT_PRIVATE_20261111_01"
    b = root / "P380_HRSID_FIT_LARGE_FINAL_BASIS_ACTUAL_PRIVATE_20261111_01"
    bd = root / "P380_HRSID_FIT_LARGE_FINAL_BASIS_DRAFT_PRIVATE_20261111_01"
    tool_wait(ad / "TOOL_TERMINAL_PRIVATE.json", "dabf027581810c2eeb7ad33159a92879c1d2dfd07a4bbbb198addad9531309a4")
    tool_wait(bd / "TOOL_TERMINAL_PRIVATE.json", "7b65b55a20d55664623846791c602fc0c57a55c13fa6fd5dec4ced5b695811e3")
    pair_a = read(a / "PAIR_SAFE.json", PAIR_378)
    pair_b = read(b / "PAIR_SAFE.json", PAIR_380)
    readback = read(bd / "INDEPENDENT_READBACK_SAFE.json",
                    "05cb59201e4baa8a6119ea6754fc15d0aedca514f14b65f11f6e0c337a1f7266")
    require(pair_a["image_only_manifest_sha256"] == MANIFEST
            and pair_b["p378_entire_prediction_rows_bitwise_equal"] == 56
            and pair_b["serialized_q164_roundtrip_bitwise_reproduced"] == 56
            and readback["source_pair_sha256"] == PAIR_380, "Original archive carrier parity required")
    for p in (pair_a, pair_b):
        require(p["FIT"] is True and p["DEV"] is p["TEST"] is p["AP"] is False, "Only original FIT captures allowed")
    ref_a, ref_b = pair_a["arm_results"][0], pair_b["arms"][0]
    require(ref_a["arm"] == ref_b["arm"] == "native", "Only native E24 snapshot")
    closed(a / "native/ORIGINAL_WAIT_PRIVATE.json", ref_a["wait_sha256"])
    closed(b / "native/ORIGINAL_WAIT_PRIVATE.json", ref_b["wait_sha256"])
    manifest = read(ad / "IMAGE_ONLY_MANIFEST_PRIVATE.json", MANIFEST)
    require(manifest["FIT"] and manifest["DEV"] is manifest["TEST"] is False, "Closed FIT manifest scope")
    stream = pin(a / "native/capture/QUERIES_PRIVATE.jsonl", ref_a["prediction_sha256"], 8*1024**2).splitlines(keepends=True)
    directory = b / "native/capture"
    index = read(directory / "FEATURE_INDEX_PRIVATE.json", ref_b["feature_index_sha256"])
    restore = load(bd / "readback.py", "8c1ff2746f8c9e218c2882190c306e6540f51bdc45751c7ebeacebb183619aa2", "_basis_restore")
    require(len(stream) == len(index) == len(manifest["images"]) == 28, "Exact 28 archived images")
    return manifest["images"], stream, index, directory, restore


def make_matcher(release, cap):
    import yaml
    from rfdetr.config import RFDETRSegNanoConfig, SegmentationTrainConfig
    from rfdetr.models.lwdetr import build_criterion_from_config
    for name in ("rfdetr.models.lwdetr", "rfdetr.models.matcher", "rfdetr.models.heads.segmentation",
                 "rfdetr.utilities.box_ops", "rfdetr.utilities.tensors",
                 "rfdetr.datasets.coco", "rfdetr.datasets.transforms", "rfdetr.datasets._torchvision"):
        pin(Path(import_module(name).__file__).resolve(), cap.NATIVE_SOURCE[name])
    recipe = yaml.safe_load(pin(release / "configs/RFDETR_SEG_NANO_BASELINE_576_E100_20261107.yaml", cap.RECIPE_SHA))
    conf = dict(recipe["model"]["model_config"]["init_args"], num_queries=164)
    m = RFDETRSegNanoConfig(**conf)
    t = SegmentationTrainConfig(**recipe["model"]["train_config"]["init_args"])
    require(m.resolution == 576 and m.patch_size == 12 and m.num_windows == 1 and t.mask_point_sample_ratio == 16,
            "Exact original image grid and mask sampling")
    criterion, _ = build_criterion_from_config(m, t)
    matcher = criterion.matcher.cpu().eval()
    require((matcher.cost_bbox, matcher.cost_class, matcher.cost_giou, matcher.cost_mask_ce, matcher.cost_mask_dice)
            == (5, 2, 2, 5, 5), "Exact original stock matching weights")
    return matcher, m, t


def prepare(iid, anns, converter, transform):
    import torch
    from PIL import Image
    from torch.nn.functional import interpolate
    dummy = Image.new("RGB", (800, 800))
    _, raw = converter(dummy, dict(image_id=iid, annotations=anns))
    retained = []
    for ann in anns:
        if ann.get("iscrowd", 0):
            continue
        x, y, w, h = torch.tensor(ann["bbox"], dtype=torch.float32)
        if (x+w).clamp(0, 800) > x.clamp(0, 800) and (y+h).clamp(0, 800) > y.clamp(0, 800):
            retained.append(ann)
    require(len(retained) == len(raw["labels"]) and len({x["id"] for x in retained}) == len(retained)
            and torch.equal(raw["area"], torch.tensor([a["area"] for a in retained], dtype=torch.float32)),
            "Exact original row IDs, prepared area and stock box filtering")
    source = raw["masks"].as_subclass(torch.Tensor)
    _, native = transform(dummy, raw)
    near = interpolate(source[:, None].float(), size=(576, 576), mode="nearest")[:, 0].bool()
    exact = interpolate(source[:, None].float(), size=(576, 576), mode="nearest-exact")[:, 0].bool()
    require(tuple(native["masks"].shape[1:]) == (576, 576) and torch.equal(native["masks"], near)
            and len(native["labels"]) == len(retained) and (native["labels"] == 0).all()
            and (native["boxes"] >= 0).all() and (native["boxes"] <= 1).all(),
            "Stock direct nearest labels, original class mapping, normalized boxes")
    keep = near.flatten(1).any(1) & exact.flatten(1).any(1)
    kept = [a for a, use in zip(retained, keep.tolist(), strict=True) if use]
    require(len(kept) > 0, "Every fixed image has a common nonempty large target")
    return {key: native[key][keep] for key in ("boxes", "labels")}, near[keep], exact[keep], kept


def score_cases(matcher, api, cap, carrier, base, near, exact, large, ordinal):
    import torch
    changes = dict(none=torch.zeros_like(large), nonlarge=~large, large=large, all=torch.ones_like(large))
    output = {}
    for cell in (*CELLS, *(("repeat",) if ordinal == 1 else ())):
        selector = changes["none" if cell == "repeat" else cell]
        selected = torch.where(selector[:, None, None], exact, near)
        require(torch.equal(selected[~selector], near[~selector])
                and torch.equal(selected[selector], exact[selector]), "Exact selected and untouched mask bits")
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            torch.random.default_generator.manual_seed(52021000+ordinal)
            packet = cap.capture_final(matcher, api, carrier, [dict(base, masks=selected)], width=164, groups=1)
        require(packet["uniform_draw"].shape == (1296, 2), "Exactly 1296 actual stock matcher points")
        if cell != "none":
            ref = output["none"]
            for key in ("uniform_draw", "native_pred_at_draw", "nonmask"):
                require(torch.equal(ref[key], packet[key]), "Shared points, prediction logits and non-mask costs")
            for key in ("native_total", "native_mask_ce", "native_mask_dice"):
                require(torch.equal(ref[key][:, ~selector], packet[key][:, ~selector]),
                        "Bitwise unchanged unselected target cost columns")
            require(torch.equal(ref["native_target_at_draw"][~selector], packet["native_target_at_draw"][~selector]),
                    "Bitwise unchanged unselected target samples")
        order = packet["native_dst"].argsort()
        require(torch.equal(packet["native_dst"][order], torch.arange(len(large))), "Exact target ID alignment")
        packet["assigned"] = packet["native_src"][order]
        output[cell] = packet
    if ordinal == 1:
        require(all(torch.equal(output["repeat"][key], output["none"][key]) for key in (
            "native_total", "native_mask_ce", "native_mask_dice", "native_target_at_draw", "assigned")),
            "Repeated stock baseline must be bitwise identical")
    if not (~large).any():
        for x, y in (("none", "nonlarge"), ("large", "all")):
            require(all(torch.equal(output[x][key], output[y][key]) for key in (
                "native_total", "native_mask_ce", "native_mask_dice", "native_target_at_draw", "assigned")),
                "Large-only negative controls must reproduce exactly")
    return output


def summarize(output, sizes, store):
    import torch
    indexes = torch.arange(len(sizes))
    for a, b in CONTRASTS:
        left, right = output[a], output[b]
        ids_left, ids_right = left["assigned"], right["assigned"]
        changed = ids_left != ids_right
        c0, c1 = left["native_total"].double(), right["native_total"].double()
        record = store[a+"__"+b]
        record["images_with_reassignment"] += bool(changed.any())
        record["changed_positive_query_set_images"] += set(ids_left.tolist()) != set(ids_right.tolist())
        record["positive_query_additions"] += len(set(ids_right.tolist()) - set(ids_left.tolist()))
        record["changed_sampled_label_values"] += int((left["native_target_at_draw"] != right["native_target_at_draw"]).sum())
        record["changed_total_cost_columns"] += int((c0 != c1).any(0).sum())
        record["objective_gain_against_original_assignment_under_new_cost"] += float(
            (c1[ids_left, indexes]-c1[ids_right,indexes]).sum())
        for size in ("small", "medium", "large"):
            record["changed_"+size+"_assignments"] += sum(bool(x) and y == size for x, y in zip(changed, sizes))


def main():
    parser = ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("root", "protocol", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    pin(args.protocol, PLAN)
    def offline(event, _):
        if event.startswith("socket."):
            raise PermissionError("Only offline FIT matching permitted")
    sys.addaudithook(offline)
    import torch
    from rfdetr.datasets.coco import ConvertCoco, filter_parent_categories, make_coco_transforms_square_div_64
    start = perf_counter()
    release = args.root / "github_release_20261011/research_round_20260925_global"
    code = release / "code"
    cap = load(code / "RFDETR_E24_FIT_MATCHER_COST_ATTRIBUTION_20261109/capture_matcher_cost.py",
               "7a39df88f849d36ff139195d6001ccf7a5ea258144d9bcd4a3440e7a1a572703", "_stock_matcher_trace")
    guard = load(code / "MASK_RTDETR_DEV705_EQUAL_PREFIX_20261112/equal_prefix.py",
                 "503ef0b1c7e808ba1e5d9b6e0ac39ae9ef5b967493e6965fb1bbe1f2b0767699", "_cpu_guard")
    guard.admit_worker()
    require(torch.__version__ == "2.10.0+cu130" and version("rfdetr") == "1.12.0"
            and version("torchvision") == "0.25.0+cu130" and not torch.cuda.is_initialized(),
            "Exact original Windows CPU scientific runtime")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    matcher, m, t = make_matcher(release, cap)
    api = import_module("rfdetr.models.matcher")
    roster, stream, index, directory, restore = input_authority(args.root)
    gt = read(args.root / "HRSID_FIT_NATIVE_COCO_LAYOUT_ACTUAL_PRIVATE_20261109_01/train/_annotations.coco.json", FIT, 32*1024**2)
    require(len(gt["images"]) == 2937 and len(gt["annotations"]) == 9260, "Exact independent FIT scope")
    used = {int(x["category_id"]) for x in gt["annotations"]}
    remap = {int(c["id"]): i for i,c in enumerate(filter_parent_categories(gt["categories"], used))}
    require(len(remap) == 1 and set(remap.values()) == {0}, "Native ship-only class mapping")
    converter = ConvertCoco(include_masks=True, cat2label=remap)
    transform = make_coco_transforms_square_div_64("val", 576, multi_scale=False, expanded_scales=False,
        skip_random_resize=True, patch_size=m.patch_size, num_windows=m.num_windows, aug_config=None,
        scale_jitter=t.scale_jitter, gpu_postprocess=False)
    by_image = defaultdict(list)
    for ann in gt["annotations"]:
        by_image[ann["image_id"]].append(ann)
    image_map = {x["id"]: x for x in gt["images"]}
    stats = {g: defaultdict(Counter) for g in GROUPS}
    membership = {g: Counter() for g in GROUPS}
    for ordinal, (row, packed_row, entry) in enumerate(zip(roster, stream, index, strict=True), 1):
        iid = row["image_id"]
        archived = json.loads(packed_row)
        require(sha256(packed_row).hexdigest() == entry["p378_row_sha256"]
                and iid == archived["image_id"] == entry["image_id"] and entry["ordinal"] == ordinal
                and image_map[iid]["height"] == image_map[iid]["width"] == 800
                and entry["file"] == f"VIEW_{ordinal:03}.bin" and entry["bytes"] == 21401604,
                "Exact image, P378 row, packet name and size")
        packet = pin(directory / "FEATURES_PRIVATE" / entry["file"], entry["feature_sha256"], 21401604)
        spatial, query, bias = restore.restore(packet, entry["native_tensor_strides"])
        carrier = dict(pred_logits=torch.tensor([archived["raw_classifier_logits"]], dtype=torch.float32),
                       pred_boxes=torch.tensor([archived["raw_boxes_cxcywh"]], dtype=torch.float32),
                       pred_masks=dict(spatial_features=spatial, query_features=query, bias=bias))
        require(carrier["pred_logits"].shape == (1,164,2) and carrier["pred_boxes"].shape == (1,164,4)
                and all(torch.isfinite(x).all() for x in (carrier["pred_logits"], carrier["pred_boxes"], spatial, query, bias)),
                "Exact finite native raw carrier")
        anns = by_image[iid]
        base, near, exact, kept = prepare(iid, anns, converter, transform)
        sizes = ["large" if x["area"] >= 9216 else "medium" if x["area"] >= 1024 else "small" for x in kept]
        large = torch.tensor([x == "large" for x in sizes], dtype=torch.bool)
        original_l = sum(x["area"] >= 9216 for x in anns)
        require(int(large.sum()) == original_l > 0, "No original large rows lost")
        mixed = "mixed" if any(x["area"] < 9216 for x in anns) else "large_only"
        results = score_cases(matcher, api, cap, carrier, base, near, exact, large, ordinal)
        for name in ("all_images", mixed):
            state = membership[name]
            state["images"] += 1
            state["original_rows"] += len(anns)
            state["retained_rows"] += len(kept)
            state["original_large"] += original_l
            state["retained_large"] += int(large.sum())
            state["retained_nonlarge"] += int((~large).sum())
            state["changed_nonlarge_mask_rows"] += int(((near!=exact).flatten(1).any(1) & ~large).sum())
            state["changed_large_mask_rows"] += int(((near!=exact).flatten(1).any(1) & large).sum())
            summarize(results, sizes, stats[name])
    require([(membership[g]["images"], membership[g]["original_large"]) for g in GROUPS]
            == [(28,50),(17,33),(11,17)] and not torch.cuda.is_initialized(),
            "Exact fixed FIT population and no CUDA")
    document = dict(status="HRSID_FIT_Q164_SIZE_COLUMN_MATCHER_ACTUAL", protocol_sha256=PLAN,
        FIT=True, DEV=False, TEST=False, AP=False, model_forward_calls=0, optimizer_updates=0,
        one_group_eval_FP32=True, original_nonmask_and_RNG_equal=True, exact_stock_case_calls=113,
        source_pairs_sha256=[PAIR_378,PAIR_380], membership=membership, contrasts=stats, elapsed_seconds=perf_counter()-start)
    raw = (json.dumps(document, sort_keys=True, allow_nan=False, indent=2)+"\n").encode()
    with args.output.open("xb") as f:
        require(f.write(raw) == len(raw), "Complete fresh aggregate")
    print(json.dumps(dict(status=document["status"], result_sha256=sha256(raw).hexdigest())), flush=True)


if __name__ == "__main__":
    main()