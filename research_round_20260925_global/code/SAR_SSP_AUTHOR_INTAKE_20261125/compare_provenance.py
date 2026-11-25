"""Audit SAR-SSP source-scene geometry and optional SARDet-100K train lineage.

Read author JSON members and existing AIR-SARShip detection labels. Emit only
aggregate JSON. No mask or image reads, inference, or per-instance output.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from zipfile import ZipFile


AIR_LABEL = re.compile(r"SARShip-1\.0-(\d+)\.txt")
SARDET_AIR = re.compile(r"1AIR_SARShip_1_(\d+)__512__(\d+)___(\d+)\.[^.]+")
PANOPTIC = "datasets/SAR-SSP/annotations/panoptic_{}2017.json"
THRESHOLDS = (0.30, 0.50, 0.75)


def read_sarssp(archive_path):
    crops = {"train": {}, "val": {}}
    boxes = defaultdict(list)
    positive_crops = Counter()
    positive_sources = defaultdict(set)
    with ZipFile(archive_path) as archive:
        for split, split_crops in crops.items():
            data = json.loads(archive.read(PANOPTIC.format(split)))
            (category,) = (
                cat["id"] for cat in data["categories"]
                if cat["name"] == "ship" and cat["isthing"] == 1
            )
            images = {image["id"]: image for image in data["images"]}
            if len(images) != len(data["images"]):
                raise ValueError(f"Duplicate image ID in {split}")
            for annotation in data["annotations"]:
                image_id = annotation["image_id"]
                image = images[image_id]
                fields = ("source_image_no", "tile_x", "tile_y", "sea_state")
                coordinates = tuple(int(image[key]) for key in fields)
                if coordinates != tuple(int(annotation[key]) for key in fields):
                    raise ValueError("Image and annotation metadata disagree")
                source, x, y, sea_state = coordinates
                if (source, x, y) in split_crops:
                    raise ValueError(f"Duplicate crop in {split}")
                if (int(image["width"]), int(image["height"])) != (500, 500):
                    raise ValueError("Unexpected crop size")
                split_crops[source, x, y] = sea_state
                ships = [
                    segment for segment in annotation["segments_info"]
                    if segment["category_id"] == category
                ]
                if ships:
                    positive_crops[split] += 1
                    positive_sources[split].add(source)
                for segment in ships:
                    left, top, width, height = segment["bbox"]
                    if width <= 0 or height <= 0:
                        raise ValueError("Non-positive ship bounding box")
                    boxes[source].append((x + left, y + top, x + left + width, y + top + height))
            if len(split_crops) != len(images):
                raise ValueError(f"Expected one distinct crop per image in {split}")
    return crops, boxes, {
        split: {
            "tiles": len(split_crops),
            "ship_positive_tiles": positive_crops[split],
            "source_scenes": len({source for source, _, _ in split_crops}),
            "ship_positive_source_scenes": len(positive_sources[split]),
        }
        for split, split_crops in crops.items()
    }


def read_air_boxes(label_dir):
    scenes = {}
    for path in sorted(label_dir.glob("*/labels/SARShip-1.0-*.txt")):
        match = AIR_LABEL.fullmatch(path.name)
        if not match:
            raise ValueError(f"Unexpected AIR-SARShip label filename {path.name}")
        source = int(match[1])
        if source in scenes:
            raise ValueError(f"Duplicate AIR-SARShip scene {source}")
        boxes = []
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            coords = [float(value) for value in line.split()[:8]]
            xs, ys = coords[::2], coords[1::2]
            boxes.append((min(xs), min(ys), max(xs), max(ys)))
        scenes[source] = boxes
    return scenes


def area(box):
    x0, y0, x1, y1 = box
    return max(0, x1 - x0) * max(0, y1 - y0)


def intersection(a, b):
    x0, y0, x1, y1 = a
    u0, v0, u1, v1 = b
    return area((max(x0, u0), max(y0, v0), min(x1, u1), min(y1, v1)))


def iou(a, b):
    common = intersection(a, b)
    return common / (area(a) + area(b) - common)


def compare_scenes(sarssp, air):
    if set(sarssp) != set(air) or len(sarssp) < 2:
        raise ValueError("Expected the same nontrivial scene-ID roster from both sources")
    maxima = {
        source: {
            candidate: [max((iou(box, other) for other in research_boxes), default=0) for box in boxes]
            for candidate, research_boxes in air.items()
        }
        for source, boxes in sarssp.items()
    }
    outcomes = {}
    for threshold in THRESHOLDS:
        same, alternative = [], []
        for source, candidates in maxima.items():
            votes = {
                candidate: sum(value >= threshold for value in values)
                for candidate, values in candidates.items()
            }
            same.append(votes[source])
            alternative.append(max(votes[candidate] for candidate in votes if candidate != source))
        outcomes[f"{threshold:.2f}"] = {
            "same_id_matching_ship_box_records": sum(same),
            "sum_of_per_scene_best_other_id_matching_box_counts": sum(alternative),
            "source_scenes_same_id_uniquely_best": sum(s > a for s, a in zip(same, alternative)),
        }
    return {
        "sarssp_ship_box_records": sum(map(len, sarssp.values())),
        "air_detection_box_records": sum(map(len, air.values())),
        "source_scenes": len(sarssp),
        "thresholds": outcomes,
    }


def crop_box(x, y, size):
    return x, y, x + size, y + size


def group_crops(crops, size):
    groups = defaultdict(list)
    for source, x, y in crops:
        groups[source].append(crop_box(x, y, size))
    return groups


def crop_relationships(crops):
    train = group_crops(crops["train"], 500)
    val = group_crops(crops["val"], 500)
    union = group_crops(set(crops["train"]) | set(crops["val"]), 500)
    extents = Counter((max(box[2] for box in boxes), max(box[3] for box in boxes)) for boxes in union.values())
    cover = Counter(
        common
        for scene in train.keys() & val.keys()
        for a in val[scene] for b in train[scene]
        if (common := intersection(a, b)) > 0
    )
    return {
        "crop_extent_to_number_of_source_scenes": {f"{x}x{y}": n for (x, y), n in sorted(extents.items())},
        "shared_train_val_source_scenes": len(train.keys() & val.keys()),
        "identical_train_val_crop_origins": len(crops["train"].keys() & crops["val"].keys()),
        "train_val_same_scene_crop_pair_intersection_area_counts_px": dict(sorted(cover.items())),
    }


def sardet_exposure(train_path, mapping_path, test_crops):
    train = json.loads(train_path.read_text(encoding="utf-8-sig"))
    mapping = json.loads(mapping_path.read_text(encoding="utf-8-sig"))
    originals = {standardized: original for original, standardized in mapping.items()}
    air = []
    for item in train["images"]:
        original = originals[item["file_name"]]
        if original.startswith("1AIR_SARShip_1_"):
            match = SARDET_AIR.fullmatch(original)
            if not match:
                raise ValueError("Unexpected SARDet-100K AIR-SARShip-1.0 naming")
            air.append(tuple(map(int, match.groups())))
    test = group_crops(test_crops, 500)
    train_scenes = {source for source, _, _ in air}
    axis_diagnostics = {}
    for name, chips in (
        ("as_listed", air),
        ("coordinates_swapped", [(source, y, x) for source, x, y in air]),
    ):
        train_groups = group_crops(chips, 512)
        maxima = [
            max((intersection(box, other) for other in train_groups.get(source, [])), default=0)
            for source, boxes in test.items() for box in boxes
        ]
        axis_diagnostics[name] = {
            f"area_above_{value}_px": sum(common > value for common in maxima)
            for value in (0, 125000, 200000)
        }
    return {
        "sardet_train_air_sarship_1_0_tiles": len(air),
        "sardet_train_air_sarship_1_0_parent_scenes": len(train_scenes),
        "shared_with_sarssp_val_parent_scenes": len(test.keys() & train_scenes),
        "sarssp_val_tile_count_by_max_footprint_overlap": axis_diagnostics,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slgsea_zip", type=Path)
    parser.add_argument("airsarship_dota_root", type=Path, help="AIR-SARShip-1.0/dota, with train and val label directories")
    parser.add_argument("--sardet-train", type=Path)
    parser.add_argument("--sardet-mapping", type=Path)
    args = parser.parse_args()
    if bool(args.sardet_train) != bool(args.sardet_mapping):
        parser.error("Provide both optional SARDet-100K metadata files together")

    crops, boxes, split_counts = read_sarssp(args.slgsea_zip)
    result = {
        "sarssp_splits": split_counts,
        "crop_relationships": crop_relationships(crops),
        "air_sarship_1_0_geometry": compare_scenes(boxes, read_air_boxes(args.airsarship_dota_root)),
    }
    if args.sardet_train:
        result["historical_sardet_100k_train_exposure"] = sardet_exposure(
            args.sardet_train, args.sardet_mapping, crops["val"]
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
