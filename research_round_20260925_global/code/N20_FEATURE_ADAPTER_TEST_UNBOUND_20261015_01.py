# UNBOUND_DRAFT: source-only synthetic behavior tests; no SAR inputs.
"""The first test was run RED before the adapter implementation was written.

Expected future ordinary module: N20_FEATURE_ADAPTER_UNBOUND_20261015_01.py.
Old ordinary reference: N16_SCOUT_MODULE_UNBOUND_DRAFT_20261014_01.py.
Confirmed old path: non-affine LN(F/e, eps=1e-5), raw L/8,
Linear(513,32), nn.ReLU(), Linear(32,3). No alternate activation.
"""
import importlib.util
from pathlib import Path
import unittest


def _load_ordinary(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ZeroSpatialAdapterBehavior(unittest.TestCase):
    def test_zero_spatial_weights_preserve_old_logits_for_real_and_center_neighbors(self):
        # Loading the not-yet-implemented future module is expected to RED.
        root = Path(__file__).resolve().parent
        adapter = _load_ordinary('n20_unbound_adapter_under_test',
            root / 'N20_FEATURE_ADAPTER_UNBOUND_20261015_01.py')
        old = _load_ordinary('n20_unbound_old_513_reference',
            root / 'N16_SCOUT_MODULE_UNBOUND_DRAFT_20261014_01.py')
        import torch
        self.assertFalse(torch.cuda.is_initialized())
        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)
        generator = torch.Generator(device='cpu').manual_seed(2026101502001)
        shape = (3, 4, 256)  # A real non-square synthetic native feature grid.
        grid_one = torch.randn(shape, generator=generator, dtype=torch.float32)
        grid_two = torch.randn(shape, generator=generator, dtype=torch.float32)
        coordinates = ((0, 0), (1, 2), (2, 3))
        center = torch.stack([grid_one[y, x].clone() for y, x in coordinates])
        for index, (y, x) in enumerate(coordinates):
            grid_two[y, x] = center[index]  # Same center, different real surroundings.
        offsets = tuple((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1))

        def gather_native_3x3(grid):
            return torch.stack([torch.stack([
                grid[min(max(y + dy, 0), 2), min(max(x + dx, 0), 3)]
                for dy, dx in offsets]) for y, x in coordinates])

        neighbors_one = gather_native_3x3(grid_one)
        neighbors_two = gather_native_3x3(grid_two)
        neighbors_center = center[:, None, :].repeat(1, 9, 1)
        self.assertFalse(torch.equal(neighbors_one, neighbors_two))
        self.assertFalse(torch.equal(neighbors_one, neighbors_center))
        for neighbors in (neighbors_one, neighbors_two, neighbors_center):
            self.assertTrue(torch.equal(neighbors[:, 4, :], center))
        query_e = torch.randn((3, 256), generator=generator, dtype=torch.float32)
        native_L = torch.tensor([-3.0, 0.0, 5.0], dtype=torch.float32)
        baseline = old.make_scout(torch, 2026101502002).eval()
        # Test construction uses CPU only; no claim about a future ctor's RNG policy.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(2026101502003)
            head = adapter.create_head(torch, feature_channels=256,
                                       hidden=32, num_classes=3).eval()
        self.assertTrue(all(p.device.type == 'cpu' and p.dtype == torch.float32
                            for p in head.parameters()))
        parameters = list(head.parameters())
        spatial = [p for p in parameters if tuple(p.shape) == (9, 256)]
        self.assertEqual(len(spatial), 1)
        self.assertTrue(torch.equal(spatial[0], torch.zeros_like(spatial[0])))
        # Set the exact same four MLP tensors, without inventing new parameter names.
        with torch.no_grad():
            for name, source in baseline.named_parameters():
                matches = [p for p in parameters if tuple(p.shape) == tuple(source.shape)]
                self.assertEqual(len(matches), 1, 'Unique old MLP tensor shape: ' + name)
                matches[0].copy_(source)
            expected = baseline(center, query_e, native_L)
            expected_bytes = expected.contiguous().view(torch.uint8)
            for label, neighbors in (('native_grid_one', neighbors_one),
                                     ('native_grid_two', neighbors_two),
                                     ('center_copy', neighbors_center)):
                with self.subTest(neighborhood=label):
                    actual = head(center, query_e, native_L, neighbors)
                    self.assertEqual(tuple(actual.shape), (3, 3))
                    self.assertEqual(actual.dtype, torch.float32)
                    self.assertEqual(actual.device.type, 'cpu')
                    self.assertTrue(torch.isfinite(actual).all().item())
                    self.assertTrue(torch.equal(actual.contiguous().view(torch.uint8),
                                                expected_bytes),
                                    'Zero adapter must preserve old logits byte for byte')
        self.assertFalse(torch.cuda.is_initialized())


    def test_zero_weights_allow_noncenter_output_and_gradient(self):
        root = Path(__file__).resolve().parent
        adapter = _load_ordinary('n20_unbound_adapter_active_behavior',
            root / 'N20_FEATURE_ADAPTER_UNBOUND_20261015_01.py')
        old = _load_ordinary('n20_unbound_old_513_active_reference',
            root / 'N16_SCOUT_MODULE_UNBOUND_DRAFT_20261014_01.py')
        import torch
        self.assertFalse(torch.cuda.is_initialized())
        torch.set_num_threads(2)
        # The other method sets interop threads once in this standalone suite.
        # Keep the same ordinary factory/MLP initialization as the first test.
        baseline = old.make_scout(torch, 2026101502002).eval()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(2026101502003)
            head = adapter.create_head(torch, feature_channels=256,
                                       hidden=32, num_classes=3).eval()
        parameters = list(head.parameters())
        spatial_matches = [p for p in parameters if tuple(p.shape) == (9, 256)]
        self.assertEqual(len(spatial_matches), 1)
        spatial = spatial_matches[0]
        self.assertTrue(torch.equal(spatial, torch.zeros_like(spatial)))
        with torch.no_grad():
            for name, source in baseline.named_parameters():
                matches = [p for p in parameters if tuple(p.shape) == tuple(source.shape)]
                self.assertEqual(len(matches), 1, 'Unique old MLP tensor shape: ' + name)
                matches[0].copy_(source)
        # Deterministic nonconstant features. Only offset0 (not center offset4)
        # differs; center/e/L and every other neighbor stay identical.
        center = torch.linspace(-1.0, 1.0, 256, dtype=torch.float32).reshape(1, 256)
        query_e = torch.linspace(1.0, -1.0, 256, dtype=torch.float32).reshape(1, 256)
        native_L = torch.tensor([1.0], dtype=torch.float32)
        unchanged_neighbors = torch.zeros((1, 9, 256), dtype=torch.float32)
        unchanged_neighbors[:, 4, :] = center
        changed_neighbors = unchanged_neighbors.clone()
        changed_neighbors[:, 0, :] = torch.linspace(1.0, 2.0, 256, dtype=torch.float32)
        # Exactly one backward at the original zero spatial weights. A forward
        # implementation that always discards neighbors cannot satisfy this.
        head.zero_grad(set_to_none=True)
        zero_output = head(center, query_e, native_L, changed_neighbors)
        loss = (zero_output * torch.tensor([[1.0, .5, -.25]], dtype=torch.float32)).sum()
        loss.backward()
        self.assertIsNotNone(spatial.grad)
        self.assertTrue(torch.isfinite(spatial.grad).all().item())
        self.assertGreater(torch.count_nonzero(spatial.grad[0]).item(), 0,
                           'A noncenter spatial weight must receive a real finite gradient')
        with torch.no_grad():
            # A single nonzero coefficient: F_effective[0,0] = -1 + .25*1.
            # This reference uses the independent OLD module's final logits,
            # not the adapter's intermediate feature calculation.
            spatial[0, 0] = .25
            effective_F = center.clone()
            effective_F[0, 0] = -.75
            expected_unchanged = baseline(center, query_e, native_L)
            expected_changed = baseline(effective_F, query_e, native_L)
            actual_unchanged = head(center, query_e, native_L, unchanged_neighbors)
            actual_changed = head(center, query_e, native_L, changed_neighbors)
            for actual, expected in ((actual_unchanged, expected_unchanged),
                                     (actual_changed, expected_changed)):
                self.assertEqual(tuple(actual.shape), (1, 3))
                self.assertEqual(actual.dtype, torch.float32)
                self.assertEqual(actual.device.type, 'cpu')
                self.assertTrue(torch.equal(actual.contiguous().view(torch.uint8),
                                            expected.contiguous().view(torch.uint8)))
            self.assertGreater(torch.max(torch.abs(actual_changed-actual_unchanged)).item(),
                               1e-6, 'One changed noncenter neighbor must change final logits')
        self.assertFalse(torch.cuda.is_initialized())



if __name__ == '__main__':
    unittest.main()
