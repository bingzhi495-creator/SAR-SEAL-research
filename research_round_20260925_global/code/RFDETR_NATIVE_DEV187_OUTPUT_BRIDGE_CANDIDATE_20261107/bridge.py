"""Join native RF-DETR image accounting and flat COCO rows without changing ranks."""
import hashlib
import json


def need(condition, message):
    if not condition:
        raise ValueError(message)


def group_canonical_rows(manifest, predictions, admitted_items):
    """Construct the CPU evaluator's ordered image blocks from already admitted bytes."""
    images = manifest["images"]
    need(manifest["schema"] == "RFDETR-E24-canonical-DEV-image-index-v1"
         and (manifest["prediction_offset_base"], manifest["native_rank_base"]) == (0, 1)
         and type(images) is list and type(predictions) is list
         and len(images) == len(admitted_items) == manifest["image_count"] == 187
         and manifest["native_count_total"] == 18700, "Exact ordered RF-DETR DEV187 export")
    grouped = []
    ids = set()
    offset = unmapped = zero_images = nonempty = zero_masks = 0
    for row, source in zip(images, admitted_items):
        image = row["image_id"]
        count = row["prediction_count"]
        ranks = row["retained_native_ranks"]
        skipped = row["unmapped_slot1_count"]
        need(type(image) is int and image == source["image_id"] and image not in ids
             and row["original_hw"] == source["original_hw"] and row["image_sha256"] == source["image_ref"]["sha256"],
             "Manifest must preserve admitted original image order, shape and bytes")
        need(type(count) is int and 0 <= count <= 100 and type(skipped) is int and count + skipped == 100
             and row["raw_query_count"] == row["native_count"] == 100
             and row["prediction_start"] == offset and offset + count <= len(predictions)
             and row["coco_rows"] == row["codec_roundtrip_masks"] == count
             and row["nonempty_masks"] + row["all_zero_masks"] == count
             and type(ranks) is list and len(ranks) == count
             and all(type(rank) is int and 1 <= rank <= 100 for rank in ranks)
             and all(a < b for a, b in zip(ranks, ranks[1:])), "Exact original flat offset and retained ranks")
        rows = predictions[offset:offset + count]
        need(all(p["image_id"] == image and p["category_id"] == 1 for p in rows),
             "Flat prediction image or class changed")
        payload = json.dumps(rows, separators=(",", ":"), allow_nan=False).encode()
        need(hashlib.sha256(payload).hexdigest() == row["canonical_json_sha256"],
             "Per-image prediction bytes differ from worker accounting")
        grouped.append(dict(image_id=image, raw_query_count=100, native_count=100,
                            unmapped_slot1_count=skipped, native_ranks=ranks, coco=rows))
        ids.add(image)
        offset += count
        unmapped += skipped
        zero_images += count == 0
        nonempty += row["nonempty_masks"]
        zero_masks += row["all_zero_masks"]
    need(offset == len(predictions) == manifest["prediction_count"] == manifest["codec_roundtrip_masks"]
         and offset > 0 and offset + unmapped == 18700
         and unmapped == manifest["unmapped_slot1_total"] and zero_images == manifest["zero_prediction_images"]
         and nonempty == manifest["nonempty_masks"] and zero_masks == manifest["all_zero_masks"],
         "Complete flat and image-accounting reconciliation")
    return dict(schema="Root-RFDETR-DEV187-ordered-coco-export-v1", images=grouped)
