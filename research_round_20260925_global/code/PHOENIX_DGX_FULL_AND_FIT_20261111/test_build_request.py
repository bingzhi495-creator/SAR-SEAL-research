import unittest

from build_request import candidate_rows, choose


class RequestTest(unittest.TestCase):
    def test_fixed_selection_and_preserves_empty_native_rows(self):
        rows = [
            dict(image_id=1, group="P1", encoded_sha256="01"*32, core=True, stress=False),
            dict(image_id=2, group="P1", encoded_sha256="02"*32, core=True, stress=False),
            dict(image_id=3, group="P2", encoded_sha256="03"*32, core=True, stress=False),
            dict(image_id=4, group="P3", encoded_sha256="03"*32, core=True, stress=False),
            dict(image_id=5, group="P4", encoded_sha256="04"*32, core=True, stress=True),
        ]
        chosen = choose(rows)
        self.assertEqual(chosen, choose(list(reversed(rows))))
        self.assertEqual(len({row["group"] for row in chosen}), 2)
        self.assertEqual(len({row["encoded_sha256"] for row in chosen}), 2)
        self.assertTrue(all(not row["stress"] for row in chosen))
        empty = dict(size=[800, 800], counts="XXa20")
        nonempty = dict(size=[800, 800], counts="someRLE")
        masks = [empty] * 164
        masks[1] = nonempty
        out = candidate_rows(dict(image_id=1, native_ids=[9, 1], native_ranks=[2, 8],
                                  ship_scores=[.5] * 164, all_rles=masks))
        self.assertEqual([(row["native_rank"], row["query_id"], row["segmentation"]) for row in out],
                         [(2, 9, empty), (8, 1, nonempty)])
        self.assertEqual([row["category_id"] for row in out], [1, 1])


    def test_rejects_non_native_rank_order(self):
        mask = dict(size=[800, 800], counts="XXa20")
        with self.assertRaisesRegex(ValueError, "native ship rank"):
            candidate_rows(dict(image_id=1, native_ids=[9, 1], native_ranks=[8, 2],
                                ship_scores=[.5] * 164, all_rles=[mask] * 164))

if __name__ == "__main__":
    unittest.main()
