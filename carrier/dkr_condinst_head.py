# -*- coding: utf-8 -*-
"""DKRCondInstMaskHead -- CondInst mask head with our DS-3x3 per-instance dynamic-kernel
readout as the NATIVE PRIMARY readout (no rank-1, no gamma gate).

CondInst's mask head IS already a pure per-instance dynamic-kernel stack (controller -> grouped
F.conv2d), byte-identical in form to carrier/dkr_op.py:_apply_dynamic. We only swap the 1x1-only
layer spec for ds3x3 (depthwise-separable dynamic 3x3 first layer -> per-instance spatial DOF in
the boundary band, where all the high-IoU error lives). kernel_mode in {'1x1','dense3x3','ds3x3'}
= the F1 kernel-shape ablation (the core claim). Unlike the Mask2Former graft (geofield_m2f_head.py:188
rank1 + gated dkr, which TIED in D1/D2), here the dynamic kernel IS the readout -> native, primary, ungated.

PARAM-ORDER NOTE: dkr_op._apply_dynamic slices params [w0,b0,w1,b1,...] (interleaved per layer), which
DIFFERS from stock CondInst.parse_dynamic_params ([all w then all b]). So this head's controller is
trained from scratch (COCO controller weights do NOT transfer); backbone/FPN/FCOS-trunk/mask_feature_head
DO transfer via load_from. The stock-CondInst B0 baseline (native parse, full COCO transfer) is a
SEPARATE config -- the honest same-carrier bar. The F1 arms (1x1/ds3x3/dense3x3) all run through this
head (all fresh controller, identical transfer) so the comparison isolates kernel SHAPE only.
"""
from __future__ import annotations
import torch

from mmdet.registry import MODELS
from mmdet.models.dense_heads.condinst_head import CondInstMaskHead
from mmdet.models.utils import aligned_bilinear, relative_coordinate_maps

from .dkr_op import _layer_specs, _nparams, _apply_dynamic


@MODELS.register_module()
class DKRCondInstMaskHead(CondInstMaskHead):
    """CondInst mask head whose dynamic-conv stack is set by ``kernel_mode``.

    Args:
        kernel_mode (str): '1x1' (== CondInst readout, 169 params @cs10/ch8),
            'dense3x3' (dense dynamic 3x3 first layer, 809 params, overfit control),
            'ds3x3' (depthwise-separable dynamic 3x3 first layer, 269 params, THE proposal).
            The bbox_head ``num_params`` MUST equal _nparams(specs) for this mode (set in config).
    """

    def __init__(self, *args, kernel_mode: str = 'ds3x3', **kwargs) -> None:
        # stash BEFORE super().__init__ (base __init__ calls self._init_layers)
        self.kernel_mode = kernel_mode
        super().__init__(*args, **kwargs)

    def _init_layers(self) -> None:
        # CondInst always concatenates 2 relative-coordinate channels onto the mask feature,
        # so the dynamic stack's substrate is in_channels + 2.
        cs = self.in_channels + 2
        self.specs = _layer_specs(self.kernel_mode, cs, self.feat_channels)
        self.num_params = _nparams(self.specs)
        # kept for API compatibility with the base class; unused by our forward_single
        self.weight_nums = []
        self.bias_nums = []

    def forward_single(self, mask_feat, positive_info):
        """Per-image mask prediction via the DS-3x3 dynamic kernel (replaces base 1x1-only path)."""
        pos_param_preds = positive_info.get('param_preds')
        pos_points = positive_info.get('points')
        pos_strides = positive_info.get('strides')

        num_inst = pos_param_preds.shape[0]
        mask_feat = mask_feat[None].repeat(num_inst, 1, 1, 1)
        _, _, H, W = mask_feat.size()
        if num_inst == 0:
            return (pos_param_preds.new_zeros((0, 1, H, W)), )

        locations = self.prior_generator.single_level_grid_priors(
            mask_feat.size()[2:], 0, device=mask_feat.device)
        rel_coords = relative_coordinate_maps(locations, pos_points, pos_strides,
                                              self.size_of_interest, mask_feat.size()[2:])
        mask_head_inputs = torch.cat([rel_coords, mask_feat], dim=1)   # (N, cs, H, W)
        mask_head_inputs = mask_head_inputs.reshape(1, -1, H, W)       # (1, N*cs, H, W)

        # native primary dynamic readout (interleaved [w,b] per layer; see dkr_op._apply_dynamic)
        mask_preds = _apply_dynamic(mask_head_inputs, pos_param_preds, self.specs, num_inst)
        mask_preds = mask_preds.reshape(-1, H, W)
        mask_preds = aligned_bilinear(
            mask_preds.unsqueeze(0),
            int(self.mask_feat_stride / self.mask_out_stride)).squeeze(0)
        return (mask_preds, )
