"""CPU tensor fixtures for the original-grid FIT candidate capture."""
import unittest

import torch
from torch.nn.functional import interpolate

from fit_native_candidates import iter_native_candidates, make_native_top100


class FixturePostProcess:
    def __init__(self, *, num_select, upsample_masks_to_image_size):
        assert num_select == 100 and upsample_masks_to_image_size is True
        self.flip_native_mask = False

    def _select_topk(self, logits):
        probabilities = logits.sigmoid().view(1, -1)
        indices = torch.argsort(probabilities, descending=True, stable=True)[:, :100]
        return probabilities.gather(1, indices), indices % 2, indices // 2

    def __call__(self, raw, sizes, *, score_threshold):
        assert score_threshold is None
        scores, labels, queries = self._select_topk(raw['pred_logits'])
        planes = raw['pred_masks'][0].index_select(0, queries[0]).unsqueeze(1)
        masks = interpolate(planes, size=tuple(sizes[0].tolist()), mode='bilinear', align_corners=False) > 0
        if self.flip_native_mask:
            masks[0, 0, 0, 0] = ~masks[0, 0, 0, 0]
        return [dict(scores=scores[0], labels=labels[0], masks=masks)]


class NativeFITCaptureTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        masks = torch.arange(100, dtype=torch.float32).view(1, 100, 1, 1).expand(1, 100, 144, 144).clone() - 1
        self.raw = dict(pred_logits=torch.zeros(1, 100, 2), pred_boxes=torch.zeros(1, 100, 4), pred_masks=masks)
        self.processor = make_native_top100(FixturePostProcess)

    def test_preserves_stock_ties_mapped_rank_and_full_original_grid(self):
        batches = []

        def traced(planes, **kwargs):
            batches.append((planes.shape[0], kwargs))
            return interpolate(planes, **kwargs)

        rows = list(iter_native_candidates(self.raw, (3, 5), processor=self.processor,
                                           torch_api=torch, interpolate=traced, mask_chunk_size=32))
        self.assertEqual([row['query_id'] for row in rows], list(range(50)))
        self.assertEqual([row['native_rank'] for row in rows], list(range(1, 100, 2)))
        self.assertEqual([row['score'] for row in rows], [0.5] * 50)
        self.assertEqual([size for size, _ in batches], [32, 32, 32, 4])
        self.assertTrue(all(kw == {'size': (3, 5), 'mode': 'bilinear', 'align_corners': False}
                            for _, kw in batches))
        for query, row in enumerate(rows):
            self.assertEqual(tuple(row['mask_logits'].shape), (3, 5))
            self.assertEqual(row['mask_logits'].dtype, torch.float32)
            self.assertEqual(row['mask'].dtype, torch.bool)
            self.assertTrue(torch.equal(row['mask_logits'] > 0, row['mask']))
            self.assertEqual(row['mask_logits'][0, 0].item(), query - 1)
        self.assertFalse(bool(rows[1]['mask'].any().item()))


    def test_rejects_stock_mask_pixel_difference_and_releases_selection(self):
        self.processor.flip_native_mask = True
        with self.assertRaisesRegex(ValueError, 'reproduce every stock mask pixel'):
            list(iter_native_candidates(self.raw, (3, 5), processor=self.processor,
                                        torch_api=torch, interpolate=interpolate, mask_chunk_size=32))
        self.assertIsNone(self.processor.selection)
        self.processor.flip_native_mask = False
        self.assertEqual(len(list(iter_native_candidates(
            self.raw, (3, 5), processor=self.processor,
            torch_api=torch, interpolate=interpolate, mask_chunk_size=32))), 50)

    def test_all_unmapped_top100_is_not_refilled(self):
        self.raw['pred_logits'][:, :, 1] = 1
        chunks = []

        def traced(planes, **kwargs):
            chunks.append(planes.shape[0])
            return interpolate(planes, **kwargs)

        rows = list(iter_native_candidates(self.raw, (3, 5), processor=self.processor,
                                           torch_api=torch, interpolate=traced, mask_chunk_size=32))
        self.assertEqual(rows, [])
        self.assertEqual(chunks, [32, 32, 32, 4])


if __name__ == '__main__':
    unittest.main()
