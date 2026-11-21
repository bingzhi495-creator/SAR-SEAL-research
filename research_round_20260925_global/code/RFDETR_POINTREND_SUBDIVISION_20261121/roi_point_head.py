"""Adaptation of PointRend's binary point head and sampling for frozen RF-DETR.

The single-image RF feature is its projected mask basis. The frozen query
logits supply a sampled 7x7 coarse RoI in place of a trained coarse RoI head.
Boxes are continuous XYXY edges on the original, unpadded image. The RF view
must be a direct resize of that image. Reject degenerate boxes before passing
them here and preserve their original masks in the outer candidate roster.

Author reference (point_head.py, point_features.py, mask_head.py):
https://github.com/facebookresearch/detectron2/tree/247691a1c292654867deb35cabb27f18d75a1363/projects/PointRend/point_rend
"""

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _check_boxes(boxes: Tensor, image_hw):
    if len(image_hw) != 2 or any(type(v) is not int or v < 1 for v in image_hw):
        raise ValueError("image_hw must be a positive integer (height, width)")
    if boxes.ndim != 2 or boxes.shape[1] != 4 or boxes.dtype not in (torch.float32, torch.float64):
        raise ValueError("boxes must be floating [R,4] XYXY edges")
    height, width = image_hw
    if not torch.isfinite(boxes).all() or (boxes[:, :2] < 0).any() or (
        boxes[:, 2:] > boxes.new_tensor([width, height])
    ).any() or (boxes[:, 2:] <= boxes[:, :2]).any():
        raise ValueError("each box must be clipped, finite and positive in both dimensions")


def _image_coordinates(boxes, roi_xy, image_hw):
    _check_boxes(boxes, image_hw)
    if (roi_xy.ndim != 3 or roi_xy.shape[0] != len(boxes) or roi_xy.shape[2] != 2
            or roi_xy.device != boxes.device or roi_xy.dtype != boxes.dtype):
        raise ValueError("roi_xy must be aligned floating [R,P,2] in box dtype and device")
    if not torch.isfinite(roi_xy).all() or ((roi_xy < 0) | (roi_xy > 1)).any():
        raise ValueError("RoI coordinates must be finite and normalized to [0,1]")
    origin, end = boxes[:, None, :2], boxes[:, None, 2:]
    height, width = image_hw
    return (origin + roi_xy * (end - origin)) / boxes.new_tensor([width, height])


def _sample(planes, unit_xy):
    if (planes.ndim != 4 or planes.shape[0] != unit_xy.shape[0] or min(planes.shape[1:]) < 1
            or planes.device != unit_xy.device or planes.dtype != unit_xy.dtype):
        raise ValueError("sampled planes must be [N,C,H,W] in the coordinates' dtype and device")
    return F.grid_sample(planes, (2 * unit_xy - 1).unsqueeze(2), mode="bilinear",
                         padding_mode="zeros", align_corners=False).squeeze(-1)


def fine_features_at_roi_points(feature_map: Tensor, boxes: Tensor, roi_xy: Tensor, image_hw) -> Tensor:
    """Sample one [1,C,Hf,Wf] spatial basis at [R,P,2] RoI locations, yielding [R,C,P]."""
    xy = _image_coordinates(boxes, roi_xy, image_hw)
    rows, points = xy.shape[:2]
    sampled = _sample(feature_map, xy.reshape(1, rows * points, 2))
    return sampled[0].reshape(feature_map.shape[1], rows, points).permute(1, 0, 2)


def coarse_roi_logits(query_logits: Tensor, boxes: Tensor, image_hw) -> Tensor:
    """Sample the selected [R,1,Hc,Wc] RF query planes at 7x7 regular RoI centers."""
    if query_logits.ndim != 4 or query_logits.shape[1] != 1:
        raise ValueError("query_logits must carry one aligned logit plane per box")
    centers = (torch.arange(7, dtype=boxes.dtype, device=boxes.device) + 0.5) / 7
    y, x = torch.meshgrid(centers, centers, indexing="ij")
    roi_xy = torch.stack((x, y), dim=-1).reshape(1, 49, 2).expand(len(boxes), -1, -1)
    xy = _image_coordinates(boxes, roi_xy, image_hw)
    return _sample(query_logits, xy).reshape(len(boxes), 1, 7, 7)


def paste_probabilities_at_pixels(roi_logits: Tensor, boxes: Tensor, pixels_xy: Tensor, image_hw) -> Tensor:
    """Sample sigmoid(RoI logits) at native integer (x,y) pixel centers, returning [R,P].

    The default Detectron2 binary paste thresholds these probabilities at >=0.5.
    The unthresholded values also support ranking native pixel edits.
    """
    _check_boxes(boxes, image_hw)
    if (roi_logits.ndim != 4 or roi_logits.shape[1] != 1
            or roi_logits.shape[2] != roi_logits.shape[3]):
        raise ValueError("roi_logits must contain one square logit plane per box")
    if (pixels_xy.ndim != 3 or pixels_xy.shape[0] != len(boxes) or pixels_xy.shape[2] != 2
            or pixels_xy.dtype not in (torch.int32, torch.int64) or pixels_xy.device != boxes.device):
        raise ValueError("pixels_xy must be aligned integer [R,P,2] on the boxes' device")
    height, width = image_hw
    if ((pixels_xy < 0) | (pixels_xy >= pixels_xy.new_tensor([width, height]))).any():
        raise ValueError("native pixel index outside image")
    origin, end = boxes[:, None, :2], boxes[:, None, 2:]
    unit = (pixels_xy.to(boxes.dtype) + 0.5 - origin) / (end - origin)
    return _sample(roi_logits.sigmoid(), unit)[:, 0]


class BinaryPointHead(nn.Module):
    """Single-class 256-channel StandardPointHead. Seed torch before construction."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Conv1d(257, 256, 1)
        self.fc2 = nn.Conv1d(257, 256, 1)
        self.fc3 = nn.Conv1d(257, 256, 1)
        self.predictor = nn.Conv1d(257, 1, 1)
        for layer in (self.fc1, self.fc2, self.fc3):
            nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
            nn.init.zeros_(layer.bias)
        nn.init.normal_(self.predictor.weight, std=0.001)
        nn.init.zeros_(self.predictor.bias)

    def forward(self, fine: Tensor, coarse: Tensor) -> Tensor:
        """Predict [R,1,P] logits from aligned [R,256,P] and [R,1,P] samples."""
        x = torch.cat((fine, coarse), dim=1)
        for layer in (self.fc1, self.fc2, self.fc3):
            x = torch.cat((F.relu(layer(x)), coarse), dim=1)
        return self.predictor(x)


@torch.no_grad()
def sample_training_points(coarse: Tensor, generator: torch.Generator) -> Tensor:
    """Return [R,196,2] RoI points using the caller's advancing, device-local RNG."""
    if (coarse.ndim != 4 or coarse.shape[1:] != (1, 7, 7)
            or coarse.dtype not in (torch.float32, torch.float64)):
        raise ValueError("training samples require [R,1,7,7] float coarse logits")
    if not isinstance(generator, torch.Generator) or generator.device != coarse.device:
        raise ValueError("supply an explicit generator on the coarse tensor's device")
    options = dict(generator=generator, device=coarse.device, dtype=coarse.dtype)
    rows = len(coarse)
    pool = torch.rand(rows, 588, 2, **options)
    sampled = _sample(coarse, pool)[:, 0]
    chosen = (-sampled.abs()).topk(147, dim=1).indices
    uncertain = pool.gather(1, chosen.unsqueeze(-1).expand(-1, -1, 2))
    fresh = torch.rand(rows, 49, 2, **options)
    return torch.cat((uncertain, fresh), dim=1)


def binary_point_training_loss(head: BinaryPointHead, feature_map: Tensor, coarse: Tensor,
                               boxes: Tensor, matched_bitmasks: Tensor, image_hw,
                               generator: torch.Generator) -> Tensor:
    """Compute BCE on one image's matched RoIs. Only the point head receives gradients.

    matched_bitmasks are the original boolean instance masks, in the same
    positive matched-box order as coarse. The caller selects the training
    sources, matches and optimizer budget. A batch needs at least one match.
    """
    if (not len(boxes) or matched_bitmasks.shape != (len(boxes), *image_hw)
            or matched_bitmasks.dtype != torch.bool or matched_bitmasks.device != boxes.device):
        raise ValueError("supply one native boolean instance mask per positive matched box")
    with torch.no_grad():
        coords = sample_training_points(coarse, generator)
        xy = _image_coordinates(boxes, coords, image_hw)
        target = _sample(matched_bitmasks[:, None].to(boxes.dtype), xy)
        fine = fine_features_at_roi_points(feature_map, boxes, coords, image_hw)
        coarse_samples = _sample(coarse, coords)
    return F.binary_cross_entropy_with_logits(head(fine, coarse_samples), target)
