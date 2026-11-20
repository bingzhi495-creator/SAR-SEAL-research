import unittest

from run_spot import has_author_support, native_candidate_rows


class NativeRowsTest(unittest.TestCase):
    def test_zero_area_author_box_has_no_editable_support(self):
        self.assertFalse(has_author_support([0.0, 20.0, 0.0, 60.0]))
        self.assertTrue(has_author_support([0.0, 20.0, 2.0, 60.0]))
        with self.assertRaises(ValueError):
            has_author_support([2.0, 20.0, 0.0, 60.0])
    def test_only_class_zero_rows_in_original_joint_order(self):
        block = {
            "image_id": 12,
            "native_ids": [8, 2],
            "native_ranks": [1, 3],
            "joint_query_ids": [8, 8, 2],
            "joint_labels": [0, 1, 0],
            "joint_scores": [0.92, 0.63, 0.32],
            "ship_scores": [0.1] * 164,
            "all_rles": [None] * 164,
        }
        block["ship_scores"][8] = 0.92
        block["ship_scores"][2] = 0.32
        block["all_rles"][8] = {"size": [800, 800], "counts": "z8"}
        block["all_rles"][2] = {"size": [800, 800], "counts": "z2"}
        rows = native_candidate_rows(block, joint_count=3)
        self.assertEqual([(r["rank"], r["query_id"], r["score"]) for r in rows],
                         [(1, 8, 0.92), (3, 2, 0.32)])
        self.assertEqual([r["rle"]["counts"] for r in rows], ["z8", "z2"])
        block["native_ranks"] = [1, 2]
        with self.assertRaises(ValueError):
            native_candidate_rows(block, joint_count=3)


if __name__ == "__main__":
    unittest.main()
