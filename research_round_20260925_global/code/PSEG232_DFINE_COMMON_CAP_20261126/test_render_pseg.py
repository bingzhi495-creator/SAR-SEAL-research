"""Synthetic native-size signed budget, sparse write, and distance-cap controls."""
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
from pycocotools import mask as codec

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_pseg import native_distances, signed_record, verify_geometry


def source(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def encode(mask):
    record = codec.encode(np.asfortranarray(mask, dtype=np.uint8))
    record["counts"] = record["counts"].decode("ascii")
    return record


def mask_edit(stock, coordinates, pairs):
    updated = stock.copy()
    for add, drop in pairs:
        updated.flat[coordinates[add, 0]] = True
        updated.flat[coordinates[drop, 1]] = False
    return encode(updated)


class NativeSizeTest(unittest.TestCase):
    def test_row_major_geometry_uses_original_width(self):
        support = np.array([[2, 9], [15, 22], [21, 8]], dtype=np.int64)
        np.testing.assert_array_equal(native_distances(support, 7),
                                      np.array([[1, 10, 2], [2, 1, 1], [8, 1, 5]]))

    def test_rectangular_sparse_readback_uses_actual_native_area(self):
        code = Path(os.environ["SAR_SEAL_PUBLISHED_CODE"])
        solver = source(code / "PHOENIX_BOUNDED_MATCH_20261123/matching.py")
        sparse = source(code / "RFDETR_SPARSE_RLE_EQUIVALENCE_20261120/sparse_rle.py")
        coords = np.array([[2, 9], [15, 22], [21, 8]])
        probs = np.array([[.9, .99], [.51, .05], [.8, .3]], dtype=np.float32)
        native = np.zeros((5, 7), dtype=bool)
        native.flat[coords[:, 1]] = True
        fixed = SimpleNamespace(choose_pairs=lambda gains, quota: [2])
        free = SimpleNamespace(choose_free=lambda q, quota, k: [(0, 1)])
        old = SimpleNamespace(edit=mask_edit)
        predecessor = dict(fixed=dict(swaps=1, rle=mask_edit(native, coords, [(2, 2)])),
                           free_matched_count=dict(swaps=1, rle=mask_edit(native, coords, [(0, 1)])))
        record, _ = signed_record(predecessor, coords, probs, native, 20, fixed, free, old, solver, np)
        self.assertEqual(record["cap_squared"], 5)
        self.assertEqual(record["independent_pairs"], [[0, 1]])
        self.assertEqual(record["bounded_pairs"], [[2, 1]])
        verify_geometry(sparse, encode(native), record, np)
        record["cap_squared"] = 1
        self.assertRaises(ValueError, verify_geometry, sparse, encode(native), record, np)


if __name__ == "__main__":
    unittest.main()
