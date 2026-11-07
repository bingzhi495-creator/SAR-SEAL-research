"""Synthetic content audit using the actual frozen materialization fixture."""
import unittest

from content_post import verify_materialized
from test_materialize import annotation, build, inputs, np, synthetic_coco


def sample():
    rows, source, gray = inputs()
    truth = [annotation()]
    materialized = build(rows, source, gray, truth)
    return rows, source, materialized, synthetic_coco(truth)


def check(rows, source, materialized, coco):
    return verify_materialized(np, rows_document=rows, source_arrays=source,
                               match_rows=materialized["match_rows"], feature_arrays=materialized["arrays"],
                               integer_coco=coco)


class VerifyMaterializedTest(unittest.TestCase):
    def test_all_source_candidates_and_exact_matched_pixel_labels(self):
        rows, source, output, coco = sample()
        aggregate = check(rows, source, output, coco)
        self.assertEqual(aggregate, {
            "images": 1, "candidates": 3, "matched": 1, "unmatched": 2, "ignored": 0,
            "empty_masks": 1, "empty_bands": 1, "support_pixels": 64,
            "valid_pixels": int(output["arrays"]["valid"].sum()),
            "labeled_pixels": 32, "flip_correct_pixels": 2,
        })
        self.assertGreater(aggregate["valid_pixels"], 0)


    def test_candidate_order_and_exact_sparse_data_cannot_change(self):
        rows, source, output, coco = sample()
        output["match_rows"][0]["query_id"] += 1
        with self.assertRaisesRegex(ValueError, "Candidate rank"):
            check(rows, source, output, coco)
        output["match_rows"][0]["query_id"] -= 1
        output["arrays"]["positions"] = source["positions"].copy()
        output["arrays"]["positions"][0] += np.uint32(1)
        with self.assertRaisesRegex(ValueError, "sparse arrays"):
            check(rows, source, output, coco)

    def test_signed_zero_in_original_sparse_logits_is_bit_exact(self):
        rows, source, gray = inputs()
        source["logits"][0] = np.float32(-0.0)
        truth = [annotation()]
        output = build(rows, source, gray, truth)
        coco = synthetic_coco(truth)
        self.assertEqual(check(rows, source, output, coco)["matched"], 1)
        output["arrays"]["logits"] = source["logits"].copy()
        output["arrays"]["logits"][0] = np.float32(0.0)
        with self.assertRaisesRegex(ValueError, "sparse arrays"):
            check(rows, source, output, coco)

    def test_feature_neutrality_and_geometric_validity_are_both_checked(self):
        rows, source, output, coco = sample()
        arrays = output["arrays"]
        invalid = int(np.flatnonzero(~arrays["valid"])[0])
        arrays["features"][invalid, 1] = np.float32(0.25)
        with self.assertRaisesRegex(ValueError, "neutral"):
            check(rows, source, output, coco)
        arrays["features"][invalid, 1] = np.float32(0)
        arrays["valid"][invalid] = True
        with self.assertRaisesRegex(ValueError, "neighbor validity"):
            check(rows, source, output, coco)

    def test_label_corruption_with_same_sum_and_unmatched_labels_are_rejected(self):
        rows, source, output, coco = sample()
        arrays = output["arrays"]
        y = arrays["y"].copy()
        true_index = int(np.flatnonzero(y)[0])
        false_index = int(np.flatnonzero(~y)[0])
        y[true_index], y[false_index] = False, True
        original_y = arrays["y"]
        arrays["y"] = y
        with self.assertRaisesRegex(ValueError, "GT XOR"):
            check(rows, source, output, coco)
        arrays["y"] = np.append(original_y, np.bool_(False))
        arrays["label_offsets"][2:] += 1
        with self.assertRaisesRegex(ValueError, "remain unlabeled"):
            check(rows, source, output, coco)


if __name__ == "__main__":
    unittest.main()
