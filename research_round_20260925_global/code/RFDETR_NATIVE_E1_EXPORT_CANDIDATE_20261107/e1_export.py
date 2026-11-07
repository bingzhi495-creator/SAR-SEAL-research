"""Raw eager E1 output to stock RF-DETR selection and canonical COCO rows."""
import math


QUERIES = 100
CLASS_SLOTS = 2
NATIVE_ROWS = 100
RAW_FIELDS = ('pred_logits', 'pred_boxes', 'pred_masks')


def need(condition, message):
    if not condition:
        raise ValueError(message)


def finite(tensor, torch_api):
    return bool(torch_api.isfinite(tensor).all().item())


def validate_raw(outputs, torch_api):
    need(isinstance(outputs, dict) and all(name in outputs for name in RAW_FIELDS)
         and outputs.get('pred_keypoints') is None, 'Expected native segmentation eval outputs')
    logits, boxes, masks = (outputs[name] for name in RAW_FIELDS)
    need(all(torch_api.is_tensor(x) and x.is_floating_point() for x in (logits, boxes, masks)),
         'Expected dense native eval tensors, not sparse training outputs')
    mask_shape = tuple(masks.shape)
    need(tuple(logits.shape) == (1, QUERIES, CLASS_SLOTS) and tuple(boxes.shape) == (1, QUERIES, 4)
         and len(mask_shape) == 4 and mask_shape[:2] == (1, QUERIES) and min(mask_shape[2:]) > 0,
         'Frozen Nano eval requires 100 actual queries and two class slots')
    need(logits.device == boxes.device == masks.device and
         all(finite(tensor, torch_api) for tensor in (logits, boxes, masks)),
         'Nonfinite or mixed-device native outputs')
    return logits.device


def export_e1_image(raw_outputs, *, image_id, original_hw, torch_api, postprocess_type,
                    canonical_export, encode_mask):
    """Retain the stock all-slot top100 tensors and export mapped COCO masks.

    Call on one externally admitted EMA eval forward. Inject the actual pinned
    torch, PostProcess type, canonical export function and COCO codec wrapper.
    The return value's native tensor dictionary is private and is not JSON.
    """
    need(type(image_id) is int and isinstance(original_hw, (tuple, list)) and len(original_hw) == 2
         and all(type(value) is int and value > 0 for value in original_hw), 'Invalid original image identity or H/W')
    need(all(callable(value) for value in (postprocess_type, canonical_export, encode_mask)),
         'Missing stock postprocess or canonical codec dependency')
    original_hw = tuple(original_hw)
    device = validate_raw(raw_outputs, torch_api)
    target_sizes = torch_api.tensor([list(original_hw)], dtype=torch_api.int64, device=device)
    postprocess = postprocess_type(num_select=NATIVE_ROWS, upsample_masks_to_image_size=True)
    raw = {name: raw_outputs[name] for name in RAW_FIELDS}
    result = postprocess(raw, target_sizes, score_threshold=None)
    need(type(result) is list and len(result) == 1 and isinstance(result[0], dict),
         'Expected exactly one stock PostProcess result')
    native = result[0]
    need(all(torch_api.is_tensor(native.get(name)) for name in ('scores', 'labels', 'boxes', 'masks')),
         'Missing stock prediction tensors')
    labels, scores, boxes, masks = (native[name] for name in ('labels', 'scores', 'boxes', 'masks'))
    need(tuple(labels.shape) == tuple(scores.shape) == (NATIVE_ROWS,)
         and tuple(boxes.shape) == (NATIVE_ROWS, 4)
         and tuple(masks.shape) == (NATIVE_ROWS, 1, *original_hw)
         and labels.dtype == torch_api.int64 and masks.dtype == torch_api.bool
         and scores.is_floating_point() and boxes.is_floating_point()
         and finite(boxes, torch_api), 'Stock top100 tensor row count, dtype or original geometry differs')
    label_values = labels.detach().cpu().tolist()
    score_values = scores.detach().cpu().tolist()
    need(all(type(label) is int and label in (0, 1) for label in label_values)
         and all(type(score) in (int, float) and math.isfinite(score) and 0 <= score <= 1 for score in score_values)
         and all(a >= b for a, b in zip(score_values, score_values[1:])),
         'Invalid labels, scores or ordering from stock top100')
    cpu_masks = masks.detach().cpu()
    canonical = []
    for rank, (label, score) in enumerate(zip(label_values, score_values)):
        # Chunk the existing export function by row. All selection and all
        # original-size masks have already been produced by stock PostProcess.
        rows = canonical_export(dict(labels=[label], scores=[score], masks=[cpu_masks[rank].tolist()]),
                                image_id=image_id, original_hw=original_hw, label2cat={0: 1}, encode_mask=encode_mask)
        need(type(rows) is list and len(rows) == int(label == 0), 'Canonical slot filtering differs')
        canonical.extend(rows)
    return dict(native=native, native_count=NATIVE_ROWS, raw_query_count=QUERIES,
                unmapped_slot1_count=NATIVE_ROWS - len(canonical), coco=canonical)
