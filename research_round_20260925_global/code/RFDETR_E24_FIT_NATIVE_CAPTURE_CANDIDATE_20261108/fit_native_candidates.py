"""Capture stock RF-DETR FIT candidates and exact original-grid mask logits."""
import math


NATIVE_ROWS = 100
MASK_CHUNK = 32
MAX_ORIGINAL_PIXELS = 2 * 1024**2


def need(condition, message):
    if not condition:
        raise ValueError(message)


def make_native_top100(postprocess_type):
    """Observe the pinned stock selector from the same native top100 call."""
    class ObserveStockSelection(postprocess_type):
        def _select_topk(self, logits):
            need(self.selection is None, 'Expected one native selector call')
            self.selection = super()._select_topk(logits)
            return self.selection

    processor = ObserveStockSelection(num_select=NATIVE_ROWS, upsample_masks_to_image_size=True)
    processor.selection = None
    return processor


def _read_stock(raw, hw, processor, torch):
    logits, boxes, masks = (raw[key] for key in ('pred_logits', 'pred_boxes', 'pred_masks'))
    need(tuple(logits.shape) == (1, NATIVE_ROWS, 2) and tuple(boxes.shape) == (1, NATIVE_ROWS, 4)
         and tuple(masks.shape) == (1, NATIVE_ROWS, 144, 144) and raw.get('pred_keypoints') is None
         and logits.device == boxes.device == masks.device
         and all(tensor.dtype == torch.float32 and bool(torch.isfinite(tensor).all().item())
                 for tensor in (logits, boxes, masks)), 'Expected finite dense FP32 Seg Nano 576 outputs')

    need(processor.selection is None, 'Unexpected stale native selection')
    try:
        output = processor(raw, torch.tensor([list(hw)], dtype=torch.int64, device=logits.device),
                           score_threshold=None)
        selection = processor.selection
    finally:
        processor.selection = None
    need(type(output) is list and len(output) == 1 and type(selection) is tuple and len(selection) == 3,
         'Expected one stock top100 result and its exact selection')
    scores, labels, queries = selection
    native = output[0]
    native_masks = native['masks']
    need(tuple(scores.shape) == tuple(labels.shape) == tuple(queries.shape) == (1, NATIVE_ROWS)
         and scores.dtype == torch.float32 and labels.dtype == queries.dtype == torch.int64
         and scores.device == labels.device == queries.device == native_masks.device == masks.device
         and tuple(native_masks.shape) == (NATIVE_ROWS, 1, *hw) and native_masks.dtype == torch.bool
         and torch.equal(native['scores'], scores[0]) and torch.equal(native['labels'], labels[0]),
         'Stock top100 scores, labels or mask rank alignment differs')

    identities = list(zip(queries[0].detach().cpu().tolist(), labels[0].detach().cpu().tolist()))
    values = scores[0].detach().cpu().tolist()
    need(len(set(identities)) == NATIVE_ROWS
         and all(type(q) is type(slot) is int and 0 <= q < NATIVE_ROWS and slot in (0, 1)
                 for q, slot in identities)
         and all(type(score) is float and math.isfinite(score) and 0 <= score <= 1 for score in values)
         and all(left > right or (left == right and a < b)
                 for left, right, a, b in zip(values, values[1:], identities, identities[1:])),
         'Stock top100 did not preserve stable sigmoid query/class order')
    planes = masks[0].detach().index_select(0, queries[0])
    return native_masks, planes, identities, values


def _owned_cpu_plane(tensor):
    # On CPU an ordinary slice would retain the full chunk or native mask storage.
    value = tensor.detach()
    return value.clone() if value.device.type == 'cpu' else value.cpu()


def iter_native_candidates(raw, original_hw, *, processor, torch_api, interpolate, mask_chunk_size):
    """Yield only retained slot0 rows as rank, query, score and two owned CPU planes.

    The caller owns the admitted FIT identity, frozen model/transform, writer, and
    terminal evidence. Consume this iterator fully to check every stock chunk.
    """
    need(type(original_hw) in (list, tuple) and len(original_hw) == 2
         and all(type(size) is int and size > 0 for size in original_hw)
         and math.prod(original_hw) <= MAX_ORIGINAL_PIXELS, 'Expected bounded original H/W')
    need(type(mask_chunk_size) is int and mask_chunk_size == MASK_CHUNK,
         'Expected stock RF-DETR 1.12.0 32-row mask chunks')
    hw = tuple(original_hw)
    native_masks, planes, identities, scores = _read_stock(raw, hw, processor, torch_api)
    for start in range(0, NATIVE_ROWS, mask_chunk_size):
        stop = min(start + mask_chunk_size, NATIVE_ROWS)
        logits = interpolate(planes[start:stop].unsqueeze(1), size=hw, mode='bilinear', align_corners=False)
        need(logits.dtype == torch_api.float32 and logits.device == planes.device
             and tuple(logits.shape) == (stop - start, 1, *hw)
             and bool(torch_api.isfinite(logits).all().item())
             and torch_api.equal(logits > 0, native_masks[start:stop]),
             'Original-grid FP32 logits must reproduce every stock mask pixel at >0')
        for position in range(start, stop):
            query, slot = identities[position]
            if slot == 0:
                yield dict(native_rank=position + 1, query_id=query, score=scores[position],
                           mask=_owned_cpu_plane(native_masks[position, 0]),
                           mask_logits=_owned_cpu_plane(logits[position - start, 0]))
