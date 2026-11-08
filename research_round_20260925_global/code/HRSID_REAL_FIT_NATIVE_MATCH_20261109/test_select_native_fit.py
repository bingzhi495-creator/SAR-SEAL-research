import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from select_native_fit import select_cases


class NativeRoleSelectionTest(unittest.TestCase):
    def test_first_crop_empty_and_first_maximum_direct(self):
        images = [{"id": n} for n in range(8)]
        groups = {n: list(range(k)) for n, k in enumerate((3, 2, 163, 1, 0, 163, 1, 163))}
        visits = []

        def materialize(image, anns, seed):
            visits.append((image["id"], len(anns), seed))
            return {"labels": {1: 2, 2: 87, 3: 0, 5: 163}[image["id"]]}, image["id"]

        selected = select_cases(images, groups, 20, lambda seed: seed in (20, 24, 25, 27), materialize)
        self.assertEqual(selected["empty"]["sample"], 3)
        self.assertEqual(selected["dense"]["sample"], 5)
        self.assertEqual(visits, [(1, 2, 21), (2, 163, 22), (3, 1, 23), (5, 163, 25)])

    def test_rejects_incorrect_dense_transform(self):
        images = [{"id": 0}, {"id": 1}]
        groups = {0: [0], 1: list(range(163))}
        with self.assertRaisesRegex(ValueError, "must retain exactly 163"):
            select_cases(images, groups, 10, lambda seed: seed == 11,
                         lambda image, anns, seed: ({"labels": 0 if seed == 10 else 162}, None))


if __name__ == "__main__":
    unittest.main()
