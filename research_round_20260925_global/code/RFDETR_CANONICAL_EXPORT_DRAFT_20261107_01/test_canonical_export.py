"""Standard-library synthetic fixtures for the public export interface."""
import json
import unittest

from canonical_export import export_coco_mask_predictions


MASK = [[True, False], [False, True]]


def synthetic_encoder(mask):
    """Injected fixture for this one 2x2 mask, not a COCO codec verification."""
    if mask != MASK:
        raise AssertionError("unexpected synthetic mask")
    return {"size": [2, 2], "counts": b"0120"}


class CanonicalExportTests(unittest.TestCase):
    def test_golden_low_score_and_canonical_identity_survive_json(self):
        predictions = {"labels": [0], "scores": [0.125], "masks": [[MASK]]}
        result = export_coco_mask_predictions(
            predictions, image_id=42, original_hw=(2, 2),
            label2cat={0: 1}, encode_mask=synthetic_encoder)
        expected = [{"image_id": 42, "category_id": 1, "score": 0.125,
                     "segmentation": {"size": [2, 2], "counts": "0120"}}]
        self.assertEqual(json.loads(json.dumps(result, allow_nan=False)), expected)


    def test_unknown_rows_are_removed_after_native_top100_without_refill(self):
        labels = [1, 0] + [1] * 97 + [0]
        scores = [1.0 - index / 100.0 for index in range(100)]
        predictions = {"labels": labels, "scores": scores, "masks": [[MASK] for _ in labels]}
        before = json.dumps(predictions)
        encoded_masks = []

        def recording_encoder(mask):
            encoded_masks.append(mask)
            return synthetic_encoder(mask)

        result = export_coco_mask_predictions(
            predictions, image_id=7, original_hw=(2, 2),
            label2cat={0: 1}, encode_mask=recording_encoder)
        self.assertEqual([row["score"] for row in result], [scores[1], scores[99]])
        self.assertEqual([row["category_id"] for row in result], [1, 1])
        self.assertEqual(len(encoded_masks), 2)
        self.assertEqual(json.dumps(predictions), before)



    def test_missing_or_noncanonical_inverse_mapping_is_rejected(self):
        predictions = {"labels": [0], "scores": [0.125], "masks": [[MASK]]}
        for mapping in (None, {}, {1: 1}, {0: 0}, {False: 1}, {0: True}, {0: 1, 1: 1}):
            with self.subTest(mapping=mapping):
                with self.assertRaises(ValueError):
                    export_coco_mask_predictions(
                        predictions, image_id=42, original_hw=(2, 2),
                        label2cat=mapping, encode_mask=synthetic_encoder)
        with self.assertRaises(TypeError):
            export_coco_mask_predictions(
                predictions, image_id=42, original_hw=(2, 2), encode_mask=synthetic_encoder)



    def test_requires_original_size_binary_masks_with_one_channel(self):
        cases = [
            ((3, 2), [[MASK]]),
            ((2, 2), [MASK]),
            ((2, 2), [[MASK, MASK]]),
            ((2, 2), [[[[1, 0], [0, 1]]]]),
            ((2, 2), [[[[True], [False]]]]),
            ((0, 2), [[MASK]]),
        ]
        for original_hw, masks in cases:
            with self.subTest(original_hw=original_hw, masks=masks):
                with self.assertRaises(ValueError):
                    export_coco_mask_predictions(
                        {"labels": [0], "scores": [0.125], "masks": masks},
                        image_id=42, original_hw=original_hw,
                        label2cat={0: 1}, encode_mask=synthetic_encoder)



    def test_rejects_bad_identity_misaligned_rows_and_non_native_scores(self):
        good = {"labels": [0], "scores": [0.125], "masks": [[MASK]]}
        too_many = {"labels": [0] * 101, "scores": [0.125] * 101, "masks": [[MASK]] * 101}
        cases = [
            (True, good),
            (42, {"labels": [0], "scores": [], "masks": [[MASK]]}),
            (42, {"labels": [False], "scores": [0.125], "masks": [[MASK]]}),
            (42, {"labels": [0.0], "scores": [0.125], "masks": [[MASK]]}),
            (42, {"labels": [0], "scores": [float("nan")], "masks": [[MASK]]}),
            (42, {"labels": [0], "scores": [-0.1], "masks": [[MASK]]}),
            (42, {"labels": [0], "scores": [True], "masks": [[MASK]]}),
            (42, {"labels": [0], "masks": [[MASK]]}),
            (42, too_many),
        ]
        for image_id, predictions in cases:
            with self.subTest(image_id=image_id, predictions=predictions):
                with self.assertRaises(ValueError):
                    export_coco_mask_predictions(
                        predictions, image_id=image_id, original_hw=(2, 2),
                        label2cat={0: 1}, encode_mask=synthetic_encoder)
        self.assertEqual(export_coco_mask_predictions(
            {"labels": [], "scores": [], "masks": []},
            image_id=42, original_hw=(2, 2), label2cat={0: 1},
            encode_mask=synthetic_encoder), [])



    def test_coder_output_is_original_size_compressed_ascii_and_keeps_zero_mask(self):
        predictions = {"labels": [0], "scores": [0.125], "masks": [[MASK]]}
        for counts in (b"0120", "0120"):
            with self.subTest(valid_counts=counts):
                result = export_coco_mask_predictions(
                    predictions, image_id=42, original_hw=(2, 2), label2cat={0: 1},
                    encode_mask=lambda mask: {"size": [2, 2], "counts": counts})
                self.assertEqual(result[0]["segmentation"], {"size": [2, 2], "counts": "0120"})
        invalid_rles = [
            {}, {"size": [1, 2], "counts": b"2"},
            {"size": [2, 2], "counts": [0, 1, 2, 1]},
            {"size": [2, 2], "counts": b""}, {"size": [2, 2], "counts": ""},
            {"size": [2, 2], "counts": bytes([255])},
            {"size": [2, 2], "counts": chr(233)},
        ]
        for rle in invalid_rles:
            with self.subTest(invalid_rle=rle):
                with self.assertRaises(ValueError):
                    export_coco_mask_predictions(
                        predictions, image_id=42, original_hw=(2, 2), label2cat={0: 1},
                        encode_mask=lambda mask: rle)
        with self.assertRaises(ValueError):
            export_coco_mask_predictions(
                predictions, image_id=42, original_hw=(2, 2), label2cat={0: 1}, encode_mask=None)
        zero_mask = [[False, False], [False, False]]
        result = export_coco_mask_predictions(
            {"labels": [0], "scores": [0.0], "masks": [[zero_mask]]},
            image_id=42, original_hw=(2, 2), label2cat={0: 1},
            encode_mask=lambda mask: {"size": [2, 2], "counts": b"4"})
        self.assertEqual(result[0]["score"], 0.0)
        self.assertEqual(result[0]["segmentation"], {"size": [2, 2], "counts": "4"})



    def test_rejects_labels_outside_the_single_class_two_logit_layout(self):
        for label in (-1, 2, 9, 90):
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    export_coco_mask_predictions(
                        {"labels": [label], "scores": [0.9], "masks": [[MASK]]},
                        image_id=42, original_hw=(2, 2), label2cat={0: 1},
                        encode_mask=synthetic_encoder)


if __name__ == "__main__":
    unittest.main()
