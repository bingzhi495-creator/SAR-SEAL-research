import io
import os
from pathlib import Path
import runpy
import unittest

import numpy as np
import torch


SOURCE = Path(__file__).with_name("run_core4.py")
CORE = runpy.run_path(str(SOURCE))


class EndpointArchiveTests(unittest.TestCase):
    def test_empty_image_retains_typed_native_archive(self):
        arrays, metrics = CORE["capture_image_endpoints"](
            [], 42, None, None, None, None, np, torch
        )
        buffer = io.BytesIO()
        np.savez(buffer, **arrays)
        buffer.seek(0)
        with np.load(buffer, allow_pickle=False) as saved:
            self.assertEqual(int(saved["image_id"]), 42)
            for key, expected in {
                "coords": ((0, 2), np.int32),
                "old14": ((0, 2, 14), np.float32),
                "layers": ((0, 2), np.int8),
                "spatial": ((0, 2, 256), np.float32),
                "projected_queries": ((0, 256), np.float32),
                "ranks": ((0,), int),
                "query_ids": ((0,), int),
            }.items():
                self.assertEqual(saved[key].shape, expected[0])
                self.assertEqual(saved[key].dtype, np.dtype(expected[1]))
            np.testing.assert_array_equal(saved["offsets"], [0])
        self.assertEqual(metrics, dict(moved=0, informative=0, grid_max=0., grid_exceedances=0))

    def test_supported_candidates_keep_order_and_exact_native_dense_values(self):
        basis = CORE["load"](Path(os.environ["SAR_CORE4_BASIS_SOURCE"]), CORE["BASIS_SHA"], "_core4_test_basis")
        residual = CORE["load"](Path(os.environ["SAR_CORE4_RESIDUAL_SOURCE"]), CORE["RESIDUAL_SHA"], "_core4_test_residual")
        torch.set_num_threads(1)
        generator = torch.Generator().manual_seed(5217)
        spatial = torch.randn((1, 256, 144, 144), generator=generator, dtype=torch.float32) * 8
        query_bank = torch.randn((1, 164, 256), generator=generator, dtype=torch.float32)
        captured = basis.FrozenBasis(spatial, query_bank, torch.zeros(1))
        coords = np.array([[0, 799], [639200, 639999], [28801, 555555], [201474, 587269], [30747, 40006]],
                          dtype=np.int32)
        original = np.arange(5*2*14, dtype=np.float32).reshape(5,2,14)
        layers = np.ones((5,2), dtype=np.int8)
        points = [
            dict(coordinates=coords[part], features=original[part], layers=layers[part], rank=rank, query_id=query_id)
            for part, rank, query_id in ((slice(0,2), 2, 17), (slice(2,5), 9, 3))
        ]

        arrays, metrics = CORE["capture_image_endpoints"](
            points, 42, captured, basis, residual, torch.tensor([0,1,7,31,63,95,127,255]), np, torch
        )

        np.testing.assert_array_equal(arrays["ranks"], [2,9])
        np.testing.assert_array_equal(arrays["query_ids"], [17,3])
        np.testing.assert_array_equal(arrays["coords"], coords)
        np.testing.assert_array_equal(arrays["old14"], original)
        np.testing.assert_array_equal(arrays["layers"], layers)
        np.testing.assert_array_equal(arrays["offsets"], [0,2,5])
        np.testing.assert_array_equal(arrays["projected_queries"], query_bank[0, [17,3]].numpy())
        channels = [15,31,62,102,123,144,202,240]
        dense = torch.nn.functional.interpolate(spatial[:, channels].contiguous(),
                                                size=(800,800), mode="bilinear", align_corners=False)
        reference = dense[0].reshape(len(channels),-1)[:, coords.reshape(-1)].T.reshape(5,2,len(channels))
        np.testing.assert_array_equal(arrays["spatial"][:,:,channels], reference.numpy())
        self.assertEqual(metrics["moved"], 10)
        self.assertEqual(metrics["informative"], 10)


if __name__ == "__main__":
    unittest.main()
