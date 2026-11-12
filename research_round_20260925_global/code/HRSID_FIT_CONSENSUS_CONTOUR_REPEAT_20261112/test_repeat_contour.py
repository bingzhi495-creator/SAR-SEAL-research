import unittest

from repeat_contour import assert_disjoint, read_variants, validate_pairs


class RepeatCarrier(unittest.TestCase):
    def test_duplicate_rgb_is_not_disjoint(self):
        a = [dict(group="P1", basename="P1_a", encoded_sha256="a", decoded_rgb_sha256="b")]
        b = [dict(group="P2", basename="P2_a", encoded_sha256="c", decoded_rgb_sha256="b")]
        with self.assertRaisesRegex(ValueError, "decoded_rgb_sha256"):
            assert_disjoint(a, b)

    def test_one_donor_cannot_be_reused(self):
        record = dict(area=60, logit_row=0)
        candidates = {5202: {1: record, 2: record}, 5201: {1: record}, 5203: {1: record, 2: record}}
        with self.assertRaisesRegex(ValueError, "exclusive"):
            list(validate_pairs({"1": [1, 1], "2": [1, 2]}, candidates))

    def test_roundtrip_rank_recover(self):
        case = dict(basename="P1_a", candidates=[dict(rank=9)])
        variants = {mode: dict(counts="a", size=[800, 800]) for mode in ("vote", "soft", "anchor_count")}
        self.assertEqual(read_variants([case], [dict(basename="P1_a", variants={"9": variants})]),
                         [dict(basename="P1_a", variants={9: variants})])

if __name__ == "__main__":
    unittest.main()