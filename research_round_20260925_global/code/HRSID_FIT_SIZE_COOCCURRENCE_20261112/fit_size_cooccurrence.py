"""Audit co-occurrence by original annotated area in the frozen HRSID FIT split."""
from argparse import ArgumentParser
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import median
from time import perf_counter


FIT_SHA = "e95cf26cc60a2c6fb651e907541380a256f2df7b1e153244a6c0f803b0b19f8f"


def need(ok, message):
    if not ok:
        raise ValueError(message)


def population(index):
    images = {image["id"]: Counter() for image in index["images"]}
    names = {image["id"]: image["file_name"].split("_", 1)[0] for image in index["images"]}
    annotations = index["annotations"]
    need(len(images) == len(index["images"]) == 2937
         and len(annotations) == len({row["id"] for row in annotations}) == 9260,
         "Exact original FIT image and annotation populations required")
    for row in annotations:
        area = row["area"]
        need(row["image_id"] in images and row["category_id"] == 1 and math.isfinite(area) and area >= 0,
             "Original finite one-class FIT area and known image required")
        size = "small" if area < 1024 else "medium" if area < 9216 else "large"
        images[row["image_id"]][size] += 1
    need(sum(images.values(), Counter()) == dict(small=8121, medium=1089, large=50),
         "Exact published FIT census must reproduce")
    return images, names


def summarize(images, names):
    large = [(key, counts) for key, counts in images.items() if counts["large"]]
    need(len(large) == 28 and len({names[key] for key, _ in large}) == 4, "Exact large source population")
    totals = sum((counts for _, counts in large), Counter())
    grouped = []
    for name in sorted({names[key] for key, _ in large}):
        group = [counts for key, counts in large if names[key] == name]
        grouped.append(dict(images=len(group), large_rows=sum(c["large"] for c in group),
                            medium_rows=sum(c["medium"] for c in group), small_rows=sum(c["small"] for c in group),
                            mixed_images=sum(c["small"] + c["medium"] > 0 for c in group)))
    mixed = [counts for _, counts in large if counts["small"] + counts["medium"] > 0]
    same_image_counts = [c["small"] + c["medium"] for _, c in large]
    return dict(fit_images=len(images), fit_annotations=sum(c.total() for c in images.values()),
                large_images=len(large), large_rows=totals["large"],
                small_rows_on_large_images=totals["small"], medium_rows_on_large_images=totals["medium"],
                mixed_large_nonlarge_images=len(mixed), mixed_large_nonlarge_large_rows=sum(c["large"] for c in mixed),
                large_and_small_images=sum(c["small"] > 0 for _, c in large),
                large_and_medium_images=sum(c["medium"] > 0 for _, c in large),
                large_small_medium_images=sum(c["small"] > 0 and c["medium"] > 0 for _, c in large),
                max_nonlarge_rows_on_one_large_image=max(same_image_counts),
                median_nonlarge_rows_per_large_image=median(same_image_counts),
                equal_image_equal_row_large_share=sum(c["large"] / c.total() for _, c in large) / len(images),
                all_row_large_share=totals["large"] / sum(c.total() for c in images.values()),
                large_image_share=len(large) / len(images), group_counts=grouped)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--fit-index", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    started = perf_counter()
    with args.fit_index.open("rb") as inp:
        raw = inp.read(32 * 1024**2 + 1)
    need(0 < len(raw) <= 32 * 1024**2 and sha256(raw).hexdigest() == FIT_SHA,
         "Exact FIT-only COCO metadata required")
    images, names = population(json.loads(raw))
    out = dict(status="HRSID_FIT_ORIGINAL_AREA_COOCCURRENCE_ACTUAL", original_fit_index_sha256=FIT_SHA,
               FIT=True, DEV=False, TEST=False, images_decoded=0, masks_decoded=0, model_forward_calls=0,
               elapsed_seconds=perf_counter() - started, **summarize(images, names))
    with args.output.open("x", encoding="utf-8", newline="\n") as dst:
        json.dump(out, dst, sort_keys=True, indent=2, allow_nan=False)
        dst.write("\n")
    print(json.dumps(out, sort_keys=True))


if __name__ == "__main__":
    main()

