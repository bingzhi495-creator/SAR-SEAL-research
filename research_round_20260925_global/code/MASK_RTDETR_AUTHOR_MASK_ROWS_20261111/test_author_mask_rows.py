import unittest

from author_mask_rows import prepare_mask_rows


def frame_row(image_id, score, counts):
    keys = {"image_id": image_id, "category_id": 1, "score": score}
    return (
        {**keys, "bbox": [0.0, 0.0, 80.0, 80.0]},
        {**keys, "segmentation": {"size": [8, 9], "counts": counts}},
    )


class AuthorMaskRowsTest(unittest.TestCase):
    def test_preserves_author_order_and_keeps_boxes_out_of_mask_scoring(self):
        pairs = [frame_row(7, 0.2, "P1"), frame_row(7, 0.9, "013"),
                 frame_row(3, 0.5, "013"), frame_row(3, 0.5, "P1")]
        export = {"bbox": [p[0] for p in pairs], "mask": [p[1] for p in pairs]}
        rows = prepare_mask_rows(export, {7: (8, 9), 3: (8, 9)}, expected_per_image=2)
        self.assertEqual(rows, export["mask"])
        self.assertTrue(all("bbox" not in row for row in rows))
        rows[0]["segmentation"]["size"][0] = 1
        self.assertEqual(export["mask"][0]["segmentation"]["size"], [8, 9])

    def test_rejects_bad_counts_and_box_mask_score_disagreement(self):
        box, mask = frame_row(3, 0.5, "P1")
        with self.assertRaises(ValueError):
            prepare_mask_rows({"bbox": [box], "mask": [mask]}, {3: (8, 9)}, expected_per_image=2)
        mask["score"] = 0.6
        with self.assertRaises(ValueError):
            prepare_mask_rows({"bbox": [box], "mask": [mask]}, {3: (8, 9)}, expected_per_image=1)

    def test_rejects_manifest_reorder_and_wrong_native_mask_canvas(self):
        b7, m7 = frame_row(7, 0.5, "P1")
        b3, m3 = frame_row(3, 0.5, "P1")
        export = {"bbox": [b7, b3], "mask": [m7, m3]}
        with self.assertRaises(ValueError):
            prepare_mask_rows(export, {3: (8, 9), 7: (8, 9)}, expected_per_image=1)
        with self.assertRaises(ValueError):
            prepare_mask_rows(export, {7: (640, 640), 3: (8, 9)}, expected_per_image=1)


if __name__ == "__main__":
    unittest.main()
