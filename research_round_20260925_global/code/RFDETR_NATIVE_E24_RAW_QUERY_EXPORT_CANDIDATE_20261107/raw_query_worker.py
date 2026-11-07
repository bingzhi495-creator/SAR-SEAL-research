"""Private E24 raw-query diagnostic, anchored to the accepted native top100."""
from dataclasses import replace
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import re
import time


IMAGE_COUNT = 187
QUERY_COUNT = 100
ALL_SLOT_ROWS = 200
PRIMARY_ROWS = 100
MAX_MANIFEST_BYTES = 4 * 1024**2
MAX_OUTPUT_BYTES = 256 * 1024**2
OUTFILE = 'RAW_QUERY_SLOT0.jsonl'


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha_string(value):
    return type(value) is str and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def json_bytes(value):
    return json.dumps(value, separators=(',', ':'), allow_nan=False).encode('utf-8')


def integer(value, lower, upper):
    return type(value) is int and lower <= value <= upper


def same_identity_pairs(items):
    result = dict(items)
    need(len(result) == len(items), 'Duplicate accepted image manifest key')
    return result


def read_primary_manifest(ref, pilot_api):
    path = pilot_api.original_ref(ref, max_bytes=MAX_MANIFEST_BYTES, suffix='.json')
    need(path.name == 'IMAGE_MANIFEST.json', 'Expected the accepted canonical image index')
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno())
        need(before.st_size == ref['bytes'], 'Accepted image manifest length differs')
        raw = stream.read(ref['bytes'] + 1)
        need(len(raw) == ref['bytes'] and hashlib.sha256(raw).hexdigest() == ref['sha256'],
             'Accepted image manifest digest differs')
        pilot_api.unchanged_stat(stream, path, before)
    return json.loads(raw, object_pairs_hook=same_identity_pairs)


def primary_rows(manifest, selected, *, manifest_ref, archive_ref, recipe_sha256):
    """Verify accepted index identity without opening its prediction file."""
    need(type(manifest) is dict and manifest['schema'] == 'RFDETR-E24-canonical-DEV-image-index-v1'
         and manifest['checkpoint_sha256'] == archive_ref['sha256']
         and manifest['recipe_sha256'] == recipe_sha256
         and integer(manifest['image_count'], IMAGE_COUNT, IMAGE_COUNT)
         and integer(manifest['prediction_offset_base'], 0, 0)
         and integer(manifest['native_rank_base'], 1, 1)
         and integer(manifest['native_count_total'], IMAGE_COUNT * PRIMARY_ROWS, IMAGE_COUNT * PRIMARY_ROWS),
         'Accepted image manifest checkpoint, recipe or population differs')
    prediction_ref = manifest['predictions_ref']
    path = Path(manifest_ref['path']).parent / 'PREDICTIONS.json'
    need(type(prediction_ref) is dict and set(prediction_ref) == {'path', 'bytes', 'sha256'}
         and prediction_ref['path'] == str(path) and integer(prediction_ref['bytes'], 1, MAX_OUTPUT_BYTES)
         and sha_string(prediction_ref['sha256']), 'Accepted original prediction ref metadata differs')
    images = manifest['images']
    need(type(images) is list and len(images) == len(selected) == IMAGE_COUNT,
         'Accepted complete canonical DEV population required')
    cursor = 0
    for primary, (_, ref, image_id, hw) in zip(images, selected):
        count = primary['prediction_count']
        ranks = primary['retained_native_ranks']
        need(type(primary['image_id']) is int and primary['image_id'] == image_id
             and primary['image_sha256'] == ref['sha256'] and primary['original_hw'] == list(hw)
             and integer(primary['raw_query_count'], QUERY_COUNT, QUERY_COUNT)
             and integer(primary['native_count'], PRIMARY_ROWS, PRIMARY_ROWS)
             and integer(primary['prediction_start'], cursor, cursor)
             and integer(count, 0, PRIMARY_ROWS) and integer(primary['unmapped_slot1_count'], 0, PRIMARY_ROWS)
             and primary['unmapped_slot1_count'] + count == PRIMARY_ROWS
             and type(ranks) is list and len(ranks) == count
             and all(integer(rank, 1, PRIMARY_ROWS) for rank in ranks)
             and all(a < b for a, b in zip(ranks, ranks[1:]))
             and sha_string(primary['canonical_json_sha256']), 'Accepted image identity, rank or row slice differs')
        cursor += count
    need(integer(manifest['prediction_count'], cursor, cursor)
         and integer(manifest['unmapped_slot1_total'], IMAGE_COUNT * PRIMARY_ROWS - cursor,
                     IMAGE_COUNT * PRIMARY_ROWS - cursor), 'Accepted mapped population accounting differs')
    return images


def image_record(primary, slot0_rows):
    """Preserve raw index order and prove the original mapped prefix's exact bytes."""
    need(type(slot0_rows) is list and len(slot0_rows) == QUERY_COUNT, 'Expected all100 slot0 query masks')
    image_id, hw = primary['image_id'], primary['original_hw']
    masks, scores, ranks = [None] * QUERY_COUNT, [None] * QUERY_COUNT, [None] * QUERY_COUNT
    retained, canonical, previous_rank, previous_score = [], [], 0, 1.0
    for rank, query, score, rle in slot0_rows:
        need(integer(rank, 1, ALL_SLOT_ROWS) and rank > previous_rank
             and integer(query, 0, QUERY_COUNT-1) and masks[query] is None
             and type(score) in (int, float) and math.isfinite(score) and 0 <= score <= previous_score,
             'Unexpected stock rank, duplicate raw query or sigmoid score')
        need(type(rle) is dict and set(rle) == {'size', 'counts'} and rle['size'] == hw
             and type(rle['counts']) is str and bool(rle['counts']) and rle['counts'].isascii(),
             'Unexpected raw query compressed RLE')
        masks[query], scores[query], ranks[query] = rle, score, rank
        previous_rank, previous_score = rank, score
        if rank <= PRIMARY_ROWS:
            retained.append(dict(native_rank=rank, query_id=query))
            canonical.append(dict(image_id=image_id, category_id=1, score=score, segmentation=rle))
    need(len(canonical) == primary['prediction_count']
         and [row['native_rank'] for row in retained] == primary['retained_native_ranks']
         and hashlib.sha256(json_bytes(canonical)).hexdigest() == primary['canonical_json_sha256'],
         'Stock top200 mapped prefix differs from the accepted original top100')
    return dict(image_id=image_id, image_sha256=primary['image_sha256'], original_hw=hw,
                primary_canonical_json_sha256=primary['canonical_json_sha256'], raw_masks=masks,
                slot0_scores=scores, slot0_native_ranks=ranks, retained=retained)


def observed_postprocess(base):
    """Observe the pinned stock selector without changing any of its outputs."""
    class ObserveStockSelection(base):
        def _select_topk(self, logits):
            need(self.selection is None, 'Unexpected multiple native selection calls for one image')
            self.selection = super()._select_topk(logits)
            return self.selection

    result = ObserveStockSelection(num_select=ALL_SLOT_ROWS, upsample_masks_to_image_size=True)
    result.selection = None
    return result


def stock_slot0(raw, hw, *, processor, torch, np, codec):
    logits, boxes, masks = (raw[key] for key in ('pred_logits', 'pred_boxes', 'pred_masks'))
    need(tuple(logits.shape) == (1, QUERY_COUNT, 2) and tuple(boxes.shape) == (1, QUERY_COUNT, 4)
         and tuple(masks.shape) == (1, QUERY_COUNT, 144, 144)
         and logits.device == boxes.device == masks.device and raw.get('pred_keypoints') is None
         and all(tensor.dtype == torch.float32 and bool(torch.isfinite(tensor).all().item())
                 for tensor in (logits, boxes, masks)), 'Unexpected raw E24 dense two-slot FP32 output')
    processor.selection = None
    result = processor(raw, torch.tensor([list(hw)], dtype=torch.int64, device=logits.device), score_threshold=None)
    need(type(result) is list and len(result) == 1 and processor.selection is not None,
         'Expected one stock top200 result')
    native = result[0]
    selected_scores, selected_labels, selected_queries = processor.selection
    need(tuple(selected_scores.shape) == tuple(selected_labels.shape) == tuple(selected_queries.shape) == (1, ALL_SLOT_ROWS)
         and selected_labels.dtype == selected_queries.dtype == torch.int64
         and tuple(native['masks'].shape) == (ALL_SLOT_ROWS, 1, *hw) and native['masks'].dtype == torch.bool
         and torch.equal(native['scores'], selected_scores[0]) and torch.equal(native['labels'], selected_labels[0]),
         'Stock postprocess changed native rank alignment, dtype or shape')
    labels = selected_labels[0].detach().cpu().tolist()
    queries = selected_queries[0].detach().cpu().tolist()
    scores = selected_scores[0].detach().cpu().tolist()
    pairs = [(query, label) for query, label in zip(queries, labels)]
    need(set(pairs) == {(query, label) for query in range(QUERY_COUNT) for label in (0, 1)}
         and all(type(score) is float and math.isfinite(score) and 0 <= score <= 1 for score in scores)
         and all(left > right or (left == right and a < b)
                 for left, right, a, b in zip(scores, scores[1:], pairs, pairs[1:])),
         'Stock all-slot selection is not the full stable query/class order')
    slot0 = [position for position, label in enumerate(labels) if label == 0]
    planes = native['masks'].index_select(
        0, torch.tensor(slot0, dtype=torch.int64, device=native['masks'].device)
    ).detach().cpu().numpy()
    rows = []
    for plane, pos in zip(planes[:, 0], slot0):
        mask = np.asfortranarray(plane, dtype=np.uint8)
        rle = codec.encode(mask)
        need(type(rle) is dict and rle.get('size') == list(hw), 'Native raw query RLE has wrong shape')
        counts = rle.get('counts')
        if type(counts) is bytes:
            counts = counts.decode('ascii')
        need(type(counts) is str and bool(counts) and counts.isascii(), 'Invalid native raw query RLE counts')
        decoded = codec.decode(dict(size=list(hw), counts=counts.encode('ascii')))
        need(decoded.shape == hw and decoded.dtype == np.uint8 and np.array_equal(decoded, plane),
             'Raw query mask changed during canonical RLE round trip')
        rows.append((pos + 1, queries[pos], scores[pos], dict(size=list(hw), counts=counts)))
    return rows


def write_records(out, selected, primary, infer, *, event_sink=None):
    """Write only verified private raw-query JSONL and preserve partial failure evidence."""
    need(len(selected) == len(primary) == IMAGE_COUNT, 'Raw query export requires exact DEV187')
    out.mkdir(mode=0o700)
    partial = out / (OUTFILE + '.partial')
    final = out / OUTFILE
    digest = hashlib.sha256()
    size, retained = 0, 0
    with partial.open('xb') as stream:
        for index, (item, expected) in enumerate(zip(selected, primary)):
            record = image_record(expected, infer(item))
            line = json_bytes(record) + b'\n'
            need(size + len(line) <= MAX_OUTPUT_BYTES, 'Raw-query artifact exceeds its bounded byte budget')
            stream.write(line)
            stream.flush()
            size += len(line)
            digest.update(line)
            retained += len(record['retained'])
            if event_sink is not None and (index + 1 in (2, IMAGE_COUNT) or (index + 1) % 32 == 0):
                event_sink(dict(event='e24_raw_query_hashes_verified', images=index+1, retained_rows=retained))
        need(retained == sum(row['prediction_count'] for row in primary),
             'Verified retained row count differs before publication')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, final)
    return dict(path=final.as_posix(), bytes=size, sha256=digest.hexdigest()), retained


def run_e24_raw_query_export(archive_ref, items, expected_ids, model_config, train_config, *,
                             primary_manifest_ref, recipe_sha256, validate_archive_progress,
                             boundary_api, pilot_api, e24_api, output_directory, event_sink=None):
    """Read the same accepted E24 EMA and DEV bytes in one independently owned original."""
    started = time.monotonic()
    import numpy as np
    from pycocotools import mask as codec
    import torch
    from rfdetr import config
    from rfdetr.datasets.coco import make_coco_transforms_square_div_64
    from rfdetr.datasets.io_utils import decode_pil_image_bytes
    from rfdetr.models._defaults import MODEL_DEFAULTS
    from rfdetr.models.lwdetr import build_model_from_config
    from rfdetr.models.postprocess import PostProcess
    from rfdetr.utilities.tensors import nested_tensor_from_tensor_list

    need(sha_string(recipe_sha256) and callable(validate_archive_progress)
         and all(version(name) == expected for name, expected in pilot_api.VERSIONS.items()),
         'Different pinned runtime or recipe identity')
    need(torch.cuda.is_available() and torch.cuda.device_count() == 1
         and not torch.is_autocast_enabled('cuda') and not torch.is_autocast_enabled('cpu'),
         'Expected the original single-device FP32 eager inference policy')
    selected = e24_api.select_items(items, expected_ids, pilot_api)
    out = e24_api.output_path(output_directory)
    need(out is not None, 'Caller must provide a fresh private diagnostic output directory')
    protected_dirs = (selected[0][0].parent.parent, Path(primary_manifest_ref['path']).parent,
                      Path(archive_ref['path']).parent)
    need(all(out != path and out not in path.parents and path not in out.parents for path in protected_dirs),
         'Diagnostic output must be independent of originals')
    manifest = read_primary_manifest(primary_manifest_ref, pilot_api)
    primary = primary_rows(manifest, selected, manifest_ref=primary_manifest_ref,
                           archive_ref=archive_ref, recipe_sha256=recipe_sha256)
    checkpoint = e24_api.read_archive(archive_ref, torch, pilot_api)
    validate_archive_progress(checkpoint, 24, boundary_api=boundary_api, recipe_sha256=recipe_sha256)
    pilot_api.match_model_config(checkpoint, model_config, train_config, config)
    model = build_model_from_config(model_config, train_config, defaults=replace(MODEL_DEFAULTS, force_no_pretrain=True))
    ema = e24_api.strict_ema_load(model, checkpoint, torch)
    del checkpoint
    device = torch.device('cuda:0')
    torch.cuda.reset_peak_memory_stats(device)
    model = model.to(device).eval()
    need(not model.training and model.num_queries == QUERY_COUNT and model.group_detr == 13,
         'Expected one native eval query group with 100 queries')
    processor = observed_postprocess(PostProcess)
    transform = make_coco_transforms_square_div_64(
        'val', 576, multi_scale=False, expanded_scales=False, skip_random_resize=True,
        patch_size=model_config.patch_size, num_windows=model_config.num_windows, aug_config=None,
        scale_jitter=train_config.scale_jitter, gpu_postprocess=False,
    )

    def infer(item):
        path, ref, image_id, hw = item
        image, scales = decode_pil_image_bytes(pilot_api.read_image(path, ref), draft_size=None)
        need(image.mode == 'RGB' and image.size == (hw[1], hw[0]) and scales == (1.0, 1.0),
             'Original DEV RGB JPEG decode or dimensions differ')
        tensor, target = transform(image, None)
        need(target is None and tensor.dtype == torch.float32 and tuple(tensor.shape) == (3, 576, 576)
             and bool(torch.isfinite(tensor).all().item()), 'Original image-only 576 transform differs')
        with torch.inference_mode():
            samples = nested_tensor_from_tensor_list(
                [tensor], block_size=model_config.patch_size * model_config.num_windows,
            ).to(device)
            need(tuple(samples.tensors.shape) == (1, 3, 576, 576) and samples.mask is not None
                 and not bool(samples.mask.any().item()), 'Unexpected native spatial padding')
            raw = model(samples)
            rows = stock_slot0(raw, hw, processor=processor, torch=torch, np=np, codec=codec)
            if image_id == selected[-1][2]:
                torch.cuda.synchronize(device)
        return rows

    raw_query_ref, retained = write_records(out, selected, primary, infer, event_sink=event_sink)
    need(retained == manifest['prediction_count'], 'Verified retained row count differs from accepted E24')
    return dict(status='NATIVE_E24_RAW_QUERY_DEV187_EXPORT_COMPLETE_PENDING_ROOT', qualified=False,
                terminal_self_certified=False, AP=False, TEST=False, GT_read=False, training=False,
                image_count=IMAGE_COUNT, raw_slot0_masks=IMAGE_COUNT*QUERY_COUNT,
                stock_top200_rows=IMAGE_COUNT*ALL_SLOT_ROWS, primary_per_image_hashes_verified=IMAGE_COUNT,
                primary_retained_rows_verified=retained, raw_query_ref=raw_query_ref,
                raw_query_format='RFDETR-E24-raw-query-slot0-jsonl-v1', primary_manifest_ref=dict(primary_manifest_ref),
                checkpoint_sha256=archive_ref['sha256'], recipe_sha256=recipe_sha256, strict_ema_readback=ema,
                tensor_precision='float32', autocast=False, cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device), worker_seconds=time.monotonic()-started)
