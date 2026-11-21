"""Held-source inference from a frozen RF basis, an eval-mode head and stock candidate geometry.

The same rendered posterior supplies the fixed-pair (F), independent balanced
(B), and independent joint (J) edits at each candidate's supplied budget.
C pastes the original sampled coarse RoI, P pastes the learned subdivided RoI.
No GT is an input. Native sparse indices use y*width+x; COCO RLE is column-major.
"""

from numbers import Integral

import numpy as np
import torch
from pycocotools import mask as codec


def _stats(adds=0, drops=0, nonpositive=0):
    return dict(adds=int(adds), drops=int(drops), edits=int(adds+drops),
                area_delta=int(adds-drops), nonpositive_flips=int(nonpositive))


def _row_inputs(row, image_hw):
    h, w = image_hw
    identity = tuple(row[key] for key in ("image_id", "rank", "query_id"))
    if any(not isinstance(v, Integral) or isinstance(v, bool) for v in identity):
        raise ValueError("image_id, rank and query_id must be integers")
    coords = np.asarray(row["coordinates"])
    k = row["k"]
    if (coords.ndim != 2 or coords.shape[1] != 2 or coords.dtype.kind not in "iu"
            or not isinstance(k, Integral) or isinstance(k, bool) or not 0 <= k <= min(8, len(coords))):
        raise ValueError("require integer native [K,2] endpoint pairs and an attainable budget")
    if coords.size and (coords.min() < 0 or coords.max() >= h*w or np.unique(coords).size != coords.size):
        raise ValueError("endpoints must be unique, disjoint and inside the native image")
    if tuple(row["rle"]["size"]) != tuple(image_hw):
        raise ValueError("stock RLE must be on the original raster")
    box = torch.as_tensor(row["box_xyxy01"], dtype=torch.float32, device="cpu")
    if (box.shape != (4,) or not box.isfinite().all() or ((box < 0) | (box > 1)).any()
            or (box[2:] < box[:2]).any()):
        raise ValueError("invalid normalized clipped XYXY box")
    native_box = box * box.new_tensor([w, h, w, h])
    missing = row.get("raw_logits") is None or (native_box[2:] == native_box[:2]).any().item()
    if missing and (len(coords) or k):
        raise ValueError("original-mask fallback requires empty support and zero frozen budget")
    return tuple(map(int, identity)), coords.astype(np.int64, copy=False), int(k), native_box, missing


def _crop_grid(box, image_hw):
    # This is Detectron2 _do_paste_mask's CPU skip_empty rectangle. On the
    # outer side of a RoI edge the zero-padded bilinear weights sum to < .5;
    # equality exactly on an edge can pass >=.5. The one-pixel halo also
    # covers the floating coordinate rounding used by the author's crop.
    h, w = image_hw
    x0, y0 = torch.floor(box[:2]).to(torch.int64).tolist()
    x1, y1 = torch.ceil(box[2:]).to(torch.int64).tolist()
    x0, y0 = max(0, x0-1), max(0, y0-1)
    x1, y1 = min(w, x1+1), min(h, y1+1)
    y, x = torch.meshgrid(torch.arange(y0, y1), torch.arange(x0, x1), indexing="ij")
    return (slice(y0, y1), slice(x0, x1)), torch.stack((x, y), dim=-1).reshape(1, -1, 2)


def _paste_rle(adapter, logits, box, image_hw, crop, pixels, stock, stock_area, canvas):
    prob = adapter.paste_probabilities_at_pixels(logits, box, pixels, image_hw)
    probability = prob.reshape(stock[crop].shape).numpy()
    native = probability >= 0.5
    common = np.count_nonzero(native & stock[crop])
    equal_adds = np.count_nonzero((probability == .5) & ~stock[crop])
    stats = _stats(np.count_nonzero(native)-common, stock_area-common, equal_adds)
    canvas[crop] = native
    rle = codec.encode(canvas)
    rle["counts"] = rle["counts"].decode("ascii")
    canvas[crop] = 0
    return rle, stats


def _edits(coords, probs, k, budget):
    adds = [(int(i), float(q)) for i, q in zip(coords[:, 0], probs[:, 0])]
    drops = [(int(i), float(q)) for i, q in zip(coords[:, 1], probs[:, 1])]
    balanced, joint = budget.project_matched_budget(adds, drops, k)
    selected = np.argsort(-(probs[:, 0]-probs[:, 1]), kind="stable")[:k]
    fixed = coords[selected].reshape(-1).copy()
    fixed_nonpositive = np.count_nonzero(probs[selected, 0] <= .5) + np.count_nonzero(probs[selected, 1] >= .5)
    indices = {"F": fixed}
    stats = {"F": _stats(k, k, fixed_nonpositive)}
    for name, choice in (("B", balanced), ("J", joint)):
        indices[name] = np.asarray(choice.additions + choice.removals, dtype=np.int64)
        stats[name] = _stats(len(choice.additions), len(choice.removals), choice.nonpositive_flips)
    if any(len(idx) != 2*k or np.unique(idx).size != 2*k for idx in indices.values()):
        raise ValueError("realized point choices violate per-candidate exact 2k")
    return indices, stats


def _readouts(adapter, budget, row, coords, k, box, coarse, learned, image_hw, canvas):
    stock = codec.decode(row["rle"])
    if stock.shape != tuple(image_hw):
        raise ValueError("original mask must decode to one native plane")
    stock = stock.astype(bool, copy=False)
    if len(coords) and (stock.flat[coords[:, 0]].any() or not stock.flat[coords[:, 1]].all()):
        raise ValueError("fixed endpoints must be stock-background additions and stock-foreground removals")
    crop, pixels = _crop_grid(box[0], image_hw)
    area = int(codec.area(row["rle"]))
    coarse_rle, cstats = _paste_rle(adapter, coarse, box, image_hw, crop, pixels, stock, area, canvas)
    pred_rle, pstats = _paste_rle(adapter, learned, box, image_hw, crop, pixels, stock, area, canvas)
    probs = np.empty((0, 2), np.float32)
    if len(coords):
        cells = torch.as_tensor(coords.reshape(-1))
        width = image_hw[1]
        native_xy = torch.stack((cells % width, torch.div(cells, width, rounding_mode="floor")), dim=-1)[None]
        probs = adapter.paste_probabilities_at_pixels(learned, box, native_xy, image_hw).reshape(-1, 2).numpy()
    choices, stats = _edits(coords, probs, k, budget)
    stats.update(C=cstats, P=pstats)
    return dict(C=coarse_rle, P=pred_rle, **choices, diagnostics=stats)


@torch.inference_mode()
def infer_image(head, spatial, rows, *, adapter, render, budget, image_hw=(800, 800), chunk_size=12):
    """Render one image's rows. Return (dict keyed by original identity, aggregate counts).

    Row keys: image_id, rank, query_id, rle, box_xyxy01, raw_logits, coordinates, k.
    F/B/J arrays are the native stock bits to toggle. B/J put additions before
    removals; F keeps pair order. Caller retains the original full candidate roster.
    """
    if head.training or type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("require an eval-mode point head and positive chunk size")
    if len(image_hw) != 2 or any(type(n) is not int or n <= 0 for n in image_hw):
        raise ValueError("positive integer native height and width required")
    spatial = torch.as_tensor(spatial)
    if spatial.ndim == 3:
        spatial = spatial.unsqueeze(0)
    if (spatial.ndim != 4 or spatial.shape[:2] != (1, 256) or spatial.device.type != "cpu"
            or spatial.dtype != torch.float32 or min(spatial.shape[-2:]) < 1):
        raise ValueError("one CPU FP32 256-channel spatial basis required")
    outputs, pending = {}, []
    ranks, queries = set(), set()
    image_id = None
    for row in rows:
        identity, coords, k, box, missing = _row_inputs(row, image_hw)
        if (identity[1] in ranks or identity[2] in queries
                or (image_id is not None and identity[0] != image_id)):
            raise ValueError("one image with unique original candidate ranks and queries required")
        image_id = identity[0]
        ranks.add(identity[1])
        queries.add(identity[2])
        if missing:
            outputs[identity] = dict(C=row["rle"], P=row["rle"],
                                     **{name: np.empty(0, np.int64) for name in "FBJ"},
                                     diagnostics={name: _stats() for name in "CPFBJ"})
        else:
            outputs[identity] = None  # Reserve original candidate order before batched rendering.
            pending.append((identity, row, coords, k, box))
    counts = dict(rows=len(outputs), rendered=len(pending), fallback=len(outputs)-len(pending), point_evaluations=0)
    canvas = np.zeros(image_hw, dtype=np.uint8, order="F")
    for start in range(0, len(pending), chunk_size):
        chunk = pending[start:start+chunk_size]
        boxes = torch.stack([record[4] for record in chunk])
        raw = torch.stack([torch.as_tensor(record[1]["raw_logits"]) for record in chunk])
        if raw.device.type != "cpu" or raw.dtype != torch.float32 or raw.shape[1:] != spatial.shape[-2:]:
            raise ValueError("all CPU FP32 query planes must align with the frozen basis")
        coarse = adapter.coarse_roi_logits(raw[:, None], boxes, image_hw)
        if not coarse.isfinite().all():
            raise ValueError("nonfinite sampled coarse RoI logits")
        def predict(xy, samples):
            return head(adapter.fine_features_at_roi_points(spatial, boxes, xy, image_hw), samples)
        result = render.subdivide_binary_mask(coarse, predict, initial_size=(7, 7),
                                              native_size=(224, 224), points_per_step=784)
        if result.point_evaluations != 3136 * len(chunk) or not result.logits.isfinite().all():
            raise ValueError("expected four finite PointRend stages with 3136 point evaluations/RoI")
        counts["point_evaluations"] += result.point_evaluations
        for i, (identity, row, coords, k, _) in enumerate(chunk):
            outputs[identity] = _readouts(adapter, budget, row, coords, k, boxes[i:i+1],
                                          coarse[i:i+1], result.logits[i:i+1], image_hw, canvas)
    for name in "CPFBJ":
        for key in ("adds", "drops", "edits", "area_delta", "nonpositive_flips"):
            counts[f"{name}_{key}"] = sum(row["diagnostics"][name][key] for row in outputs.values())
    return outputs, counts
