import unittest

import numpy as np

from add_nested_oracle import donor_domain


class Domain(unittest.TestCase):
    def test_at_least_one_keeps_both_and_excludes_agreement_and_off_band(self):
        anchor = np.array([[0, 1, 1, 0, 1]], dtype=bool)
        first = np.array([[1, 0, 1, 0, 0]], dtype=bool)
        second = np.array([[1, 1, 0, 0, 0]], dtype=bool)
        band = np.array([[1, 1, 1, 1, 0]], dtype=bool)
        np.testing.assert_array_equal(donor_domain(anchor, first, second, band),
                                      np.array([[1, 1, 1, 0, 0]], dtype=bool))


if __name__ == "__main__":
    unittest.main()