"""Synthetic paired SSDD scoring against the real legacy COCO helpers."""
from hashlib import sha256
import json
import unittest
from unittest.mock import patch

import numpy as np
from pycocotools import mask

import ssdd_phase_dev_score as subject


TRAIN = "1" * 40
EVAL = "2" * 40
CHECKPOINTS = {"native": "a" * 64, "center": "b" * 64}
ARM_PINS = {"native": "c" * 64, "center": "d" * 64}


def encode(value):
    return json.dumps(value, separators=(",", ":"), allow_nan=False).encode()


def artifact(path, value):
    raw = encode(value)
    return subject.Artifact(dict(path=path, bytes=len(raw), sha256=sha256(raw).hexdigest()), raw)


def rle(y, x):
    image = np.zeros((16, 16), np.uint8, order="F")
    image[y:y+2, x:x+2] = 1
    row = mask.encode(image)
    return dict(size=row["size"], counts=row["counts"].decode("ascii"))


def fixture():
    ids = list(range(1, 188))
    items = [
        dict(image_id=i, original_hw=[16, 16],
             image_ref=dict(path=f"/source/valid/{i}.jpg", bytes=10, sha256=f"{i:064x}"))
        for i in ids
    ]
    images = [dict(id=i, height=16, width=16, file_name=f"{i}.jpg") for i in ids]
    gts = [
        dict(id=j, image_id=(j-1) % 187+1, category_id=1, iscrowd=0, bbox=[1, 1, 2, 2], area=4,
             segmentation=[[1, 1, 1, 3, 3, 3, 3, 1]])
        for j in range(1, 381)
    ]
    gt = artifact("/sealed/gt_integer.json", dict(info={}, images=images, annotations=gts,
                                                  categories=[dict(id=1, name="ship")]))
    source = artifact("/sealed/source_images.json", dict(
        schema="Root-RFDETR-DEV187-image-only-manifest-v1", population="internal_DEV187",
        expected_ids=ids, items=items, model_execution=False, AP=False, TEST=False))
    pair = artifact("/sealed/pair.json", dict(
        status="SSDD_MASK_PHASE_E24_PAIR_INDEPENDENTLY_ACCEPTED_V1", training_protocol=TRAIN,
        seed=5201, completed_epochs=24, visits_per_arm=17856, optimizer_updates_per_arm=4464,
        ema_updates_per_arm=4464, ordered_fit_visits_match=True, initial_online_sha256=subject.Q100_SHA,
        coco_asset_sha256=subject.COCO_SHA, recipe_sha256=subject.RECIPE_SHA,
        FIT=True, DEV=False, TEST=False, AP=False, source_manifest_sha256="e" * 64,
        qualification_pair_sha256="f" * 64, fit_ids_sha256="0" * 64, ordered_fit_visits_sha256="1" * 64,
        **{f"{mode}_checkpoint_sha256": CHECKPOINTS[mode] for mode in CHECKPOINTS},
        **{f"{mode}_arm_receipt_sha256": ARM_PINS[mode] for mode in ARM_PINS},
    ))

    def arm(mode, prediction):
        name = f"/sealed/{mode}"
        rows, index = [], []
        for item in items:
            i = item["image_id"]
            mapped = [dict(image_id=i, category_id=1, score=.5, segmentation=prediction)] if i > 1 else []
            count = len(mapped)
            index.append(dict(image_id=i, original_hw=[16, 16], image_sha256=item["image_ref"]["sha256"],
                              prediction_start=len(rows), prediction_count=count, raw_query_count=100,
                              native_count=100, unmapped_slot1_count=100-count,
                              retained_native_ranks=[8] if count else [], coco_rows=count,
                              codec_roundtrip_masks=count, nonempty_masks=count, all_zero_masks=0,
                              canonical_json_sha256=sha256(encode(mapped)).hexdigest()))
            rows.extend(mapped)
        predictions = artifact(name + "/PREDICTIONS.json", rows)
        totals = dict(image_count=187, native_count_total=18700, unmapped_slot1_total=18700-len(rows),
                      prediction_count=len(rows), zero_prediction_images=1, codec_roundtrip_masks=len(rows),
                      nonempty_masks=len(rows), all_zero_masks=0)
        manifest = artifact(name + "/IMAGE_MANIFEST.json", dict(
            schema="RFDETR-E24-canonical-DEV-image-index-v1", checkpoint_sha256=CHECKPOINTS[mode],
            recipe_sha256=subject.RECIPE_SHA, prediction_offset_base=0, native_rank_base=1,
            predictions_ref=predictions.ref, images=index, **totals))
        report = artifact(name + "/EXPORT_REPORT.json", dict(
            status="SSDD_MASK_PHASE_DEV187_IMAGE_ONLY_EXPORT_PENDING_INDEPENDENT_ACCEPTANCE_V1",
            training_protocol=TRAIN, evaluation_protocol=EVAL, mode=mode, training_seed=5201,
            training_pair_sha256=pair.ref["sha256"], training_arm_sha256=ARM_PINS[mode],
            image_manifest_sha256=source.ref["sha256"],
            result=dict(status="NATIVE_E24_EMA_DEV187_EXPORT_COMPLETE_PENDING_ROOT", fixed_epoch=24,
                        optimizer_updates=4464, checkpoint_sha256=CHECKPOINTS[mode],
                        recipe_sha256=subject.RECIPE_SHA, full_DEV=True, qualified=False, terminal_self_certified=False,
                        GT_read=False, TEST=False, AP=False, training=False, tensor_precision="float32",
                        autocast=False, checkpoint_weights_only_cpu=True, predictions_ref=predictions.ref,
                        image_manifest_ref=manifest.ref, **totals)))
        closure = dict(original_os_wait=True, original_os_exit=0, resource_stop=False, worker_closed=True,
                       terminal_sha256=sha256(mode.encode()).hexdigest(), report_sha256=report.ref["sha256"])
        checkpoint = dict(path=name + "/checkpoint_23.ckpt", bytes=123, sha256=CHECKPOINTS[mode])
        return subject.ClosedExport(closure, report, predictions, manifest, checkpoint)

    return dict(native=arm("native", rle(10, 10)), center=arm("center", rle(1, 1)),
                training_pair=pair, image_manifest=source, integer_gt=gt, training_protocol=TRAIN,
                evaluation_protocol=EVAL), arm


def evaluate(args):
    gt = args["integer_gt"].ref
    source = args["image_manifest"].ref
    with patch.multiple(subject, INTEGER_GT_SHA=gt["sha256"], INTEGER_GT_BYTES=gt["bytes"],
                        IMAGE_MANIFEST_SHA=source["sha256"]):
        return subject.score_closed_pair(**args)



def rebind(arm, *, report=None, index=None, predictions=None):
    source = json.loads(arm.report.raw) if report is None else report
    manifest = arm.export_manifest
    if predictions is not None:
        index = json.loads(manifest.raw) if index is None else index
        index["predictions_ref"] = predictions.ref
        source["result"]["predictions_ref"] = predictions.ref
    if index is not None:
        manifest = artifact(manifest.ref["path"], index)
        source["result"]["image_manifest_ref"] = manifest.ref
    receipt = artifact(arm.report.ref["path"], source)
    return arm._replace(report=receipt, export_manifest=manifest, predictions=predictions or arm.predictions,
                        closure=dict(arm.closure, report_sha256=receipt.ref["sha256"]))


class SSDDPhaseScoreTests(unittest.TestCase):
    def test_complete_pair_scores_fixed_integer_gt_and_preserves_zero_candidate_image(self):
        args, _ = fixture()
        result = evaluate(args)
        self.assertEqual(result["images"], 187)
        self.assertEqual(result["annotations"], 380)
        self.assertEqual(result["primary"]["native"]["AP"], 0.)
        self.assertGreater(result["primary"]["center"]["AP"], .45)
        self.assertEqual(result["arms"]["center"]["zero_prediction_images"], 1)
        self.assertEqual(result["arms"]["center"]["prediction_count"], 186)
        self.assertIsNone(result["sensitivity"])


    def test_arm_mode_checkpoint_and_process_closure_must_bind(self):
        args, _ = fixture()
        center = args["center"]
        with self.assertRaisesRegex(ValueError, "process closure"):
            evaluate(dict(args, center=center._replace(closure=dict(center.closure, worker_closed=False))))
        with self.assertRaisesRegex(ValueError, "arm checkpoint"):
            evaluate(dict(args, center=center._replace(checkpoint_ref=args["native"].checkpoint_ref)))
        report = json.loads(center.report.raw)
        report["mode"] = "native"
        with self.assertRaisesRegex(ValueError, "mode and image manifest"):
            evaluate(dict(args, center=rebind(center, report=report)))

    def test_exact_prediction_path_and_checkpoint_in_image_index(self):
        args, _ = fixture()
        center = args["center"]
        prediction = center.predictions._replace(ref=dict(center.predictions.ref, path="/unbound/PREDICTIONS.json"))
        with self.assertRaisesRegex(ValueError, "prediction path"):
            evaluate(dict(args, center=center._replace(predictions=prediction)))
        index = json.loads(center.export_manifest.raw)
        index["checkpoint_sha256"] = CHECKPOINTS["native"]
        with self.assertRaisesRegex(ValueError, "bind the evaluated checkpoint"):
            evaluate(dict(args, center=rebind(center, index=index)))

    def test_original_group_bridge_rejects_changed_row_and_invalid_retained_rank(self):
        args, _ = fixture()
        center = args["center"]
        changed = json.loads(center.predictions.raw)
        changed[0]["score"] = .4
        prediction = artifact(center.predictions.ref["path"], changed)
        with self.assertRaisesRegex(ValueError, "Per-image prediction bytes differ"):
            evaluate(dict(args, center=rebind(center, predictions=prediction)))
        index = json.loads(center.export_manifest.raw)
        index["images"][1]["retained_native_ranks"] = [0]
        with self.assertRaisesRegex(ValueError, "retained ranks"):
            evaluate(dict(args, center=rebind(center, index=index)))

    def test_independently_exported_identical_predictions_are_a_valid_tie(self):
        args, arm = fixture()
        center = arm("center", rle(10, 10))
        self.assertEqual(center.predictions.ref["sha256"], args["native"].predictions.ref["sha256"])
        result = evaluate(dict(args, center=center))
        self.assertEqual(result["primary"]["center_minus_native_points"]["AP"], 0.)
        center = center._replace(closure=dict(center.closure,
                                              terminal_sha256=args["native"].closure["terminal_sha256"]))
        with self.assertRaisesRegex(ValueError, "independent terminal"):
            evaluate(dict(args, center=center))

    def test_float_sensitivity_requires_a_prelocked_exact_variant_and_keeps_integer_primary(self):
        args, _ = fixture()
        gt = json.loads(args["integer_gt"].raw)
        for row in gt["annotations"]:
            row["segmentation"] = [[value + .2 for value in row["segmentation"][0]]]
            row["bbox"] = [1.2, 1.2, 2., 2.]
        floating = artifact("/sealed/gt_float.json", gt)
        with self.assertRaisesRegex(ValueError, "must be prelocked"):
            evaluate(dict(args, floating_gt=floating))
        with self.assertRaisesRegex(ValueError, "sensitivity artifact"):
            evaluate(dict(args, floating_gt=floating, expected_floating_gt_sha256="9" * 64))
        result = evaluate(dict(args, floating_gt=floating, expected_floating_gt_sha256=floating.ref["sha256"]))
        self.assertEqual(result["primary"], evaluate(args)["primary"])
        self.assertEqual(result["sensitivity_GT"], "LOCAL_CANDIDATE_FLOAT")
        self.assertEqual(result["floating_gt_sha256"], floating.ref["sha256"])
        self.assertEqual(len(result["sensitivity"]["center"]["all12"]), 12)

        gt["annotations"][0]["iscrowd"] = 1
        wrong = artifact("/sealed/gt_float.json", gt)
        with self.assertRaisesRegex(ValueError, "Only three geometry fields"):
            evaluate(dict(args, floating_gt=wrong, expected_floating_gt_sha256=wrong.ref["sha256"]))


if __name__ == "__main__":
    unittest.main()

