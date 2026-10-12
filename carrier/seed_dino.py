# -*- coding: utf-8 -*-
"""SeedDINO: minimal evidence-seed injection into mmdet DINO (detection-only).

Per the NQS injection decision (seal_v4_workdir/nqs_injection_decision.md):
inject seed boxes into `reference_points` (recomputed per-layer by the decoder
via look-forward-twice), NOT into static PE -> structurally avoids the verdict-3
"PE washout" fatal flaw. Shapes are kept fixed (REPLACE the lowest-priority
learnable queries' reference points with seed boxes) so the Hungarian matching,
dn_mask and attention are untouched -- "种子并入 matching" minimal form.

Two modes (set by `seed_source`):
  'oracle' : GT boxes as seeds (ceiling diagnostic, verdict-3 required change 2).
             If even GT-box seeds don't lift inshore AP, the bottleneck is
             feature-level -> stop seeding, go to Plan B.
  'pred'   : precomputed NQS evidence seeds from data_sample.seeds (real run).

This is detection-only (no mask head yet) -- the gate is "are seeds worth it"
before paying for the self-ported mask head.
"""
from __future__ import annotations

import torch

from mmdet.models.detectors.dino import DINO
from mmdet.registry import MODELS
from mmdet.structures.bbox import bbox_xyxy_to_cxcywh


@MODELS.register_module()
class SeedDINO(DINO):
    def __init__(self, *args, seed_source: str = 'oracle', max_seeds: int = 50,
                 set_content: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        assert seed_source in ('oracle', 'pred', 'random', 'off')
        self.seed_source = seed_source
        self.max_seeds = max_seeds
        self.set_content = set_content  # also set content query from memory@seed (fuller form)

    def _seed_boxes_norm(self, data_sample, device):
        """Return (n,4) normalized cxcywh seed boxes in [0,1], or None."""
        h, w = data_sample.img_shape[:2]
        if self.seed_source == 'oracle':
            gt = data_sample.gt_instances.bboxes
            if gt is None or len(gt) == 0:
                return None
            boxes = gt.tensor if hasattr(gt, 'tensor') else gt
            boxes = boxes.to(device).float()
            cxcywh = bbox_xyxy_to_cxcywh(boxes)
        elif self.seed_source == 'pred':
            seeds = getattr(data_sample, 'seeds', None)
            if seeds is None or len(seeds) == 0:
                return None
            boxes = torch.as_tensor(seeds, device=device, dtype=torch.float32)  # xyxy abs
            cxcywh = bbox_xyxy_to_cxcywh(boxes)
        elif self.seed_source == 'random':
            # P1-5 trivial-prior control: random boxes matching the COUNT of real
            # NQS seeds (so the only difference vs 'pred' is location/statistic vs
            # random). If 'random' gets the same AR boost -> the gain is "any
            # high-recall proposal injection", not the MNE statistic specifically.
            seeds = getattr(data_sample, 'seeds', None)
            n = min(len(seeds), self.max_seeds) if seeds is not None and len(seeds) else 0
            if n == 0:
                return None
            cx = torch.rand(n, device=device); cy = torch.rand(n, device=device)
            wh = 0.02 + 0.10 * torch.rand(n, 2, device=device)  # small-ship-scale boxes
            return torch.stack([cx, cy, wh[:, 0], wh[:, 1]], dim=1).clamp(1e-4, 1 - 1e-4)
        else:
            return None
        norm = cxcywh / cxcywh.new_tensor([w, h, w, h])
        norm = norm.clamp(1e-4, 1 - 1e-4)
        if len(norm) > self.max_seeds:
            norm = norm[:self.max_seeds]
        return norm

    def pre_decoder(self, memory, memory_mask, spatial_shapes,
                    batch_data_samples=None):
        decoder_inputs_dict, head_inputs_dict = super().pre_decoder(
            memory, memory_mask, spatial_shapes, batch_data_samples)
        if self.seed_source == 'off' or batch_data_samples is None:
            return decoder_inputs_dict, head_inputs_dict

        ref = decoder_inputs_dict['reference_points']  # (bs, nq_total, 4) in [0,1]
        bs = ref.shape[0]
        # non-dn queries are the LAST num_queries entries; inject into the tail
        nq = self.num_queries
        for b in range(bs):
            seeds = self._seed_boxes_norm(batch_data_samples[b], ref.device)
            if seeds is None:
                continue
            k = min(len(seeds), nq)
            # replace the last k learnable-query reference points with seed boxes
            ref[b, -k:, :] = seeds[:k]
        decoder_inputs_dict['reference_points'] = ref
        return decoder_inputs_dict, head_inputs_dict
