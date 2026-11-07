"""Stream the locked native E24 EMA predictions for the canonical DEV population."""
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import replace
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import time


EPOCH = 24
E24_STEPS = 4464
DEV_IMAGES = 187
NATIVE_ROWS = 100
ARCHIVE_NAME = 'checkpoint_23.ckpt'
EMA_KEY = 'RFDETREMACallback'
MODEL_PREFIX = 'module.model.'
CLASSIFIER = 'class_embed.weight'
PREDICTIONS = 'PREDICTIONS.json'
MANIFEST = 'IMAGE_MANIFEST.json'
MAX_PREDICTION_BYTES = 256 * 1024**2


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha_string(value):
    return type(value) is str and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def json_bytes(value):
    return json.dumps(value, separators=(',', ':'), allow_nan=False).encode('utf-8')


def exact(value, keys, label):
    need(type(value) is dict and set(value) == set(keys), 'Unexpected ' + label + ' fields')


def select_items(items, expected_ids, pilot_api):
    """Validate caller-authenticated metadata in the exact canonical order."""
    need(type(expected_ids) in (list, tuple) and len(expected_ids) == DEV_IMAGES
         and all(type(value) is int and value >= 0 for value in expected_ids)
         and len(set(expected_ids)) == DEV_IMAGES, 'Expected unique canonical DEV187 image order')
    need(type(items) in (list, tuple) and len(items) == DEV_IMAGES, 'Expected exactly 187 declared image items')
    selected, paths = [], set()
    for item, expected_id in zip(items, expected_ids):
        exact(item, ('image_ref', 'image_id', 'original_hw'), 'DEV image item')
        hw = item['original_hw']
        need(type(item['image_id']) is int and item['image_id'] == expected_id
             and type(hw) in (list, tuple) and len(hw) == 2 and all(type(n) is int and n > 0 for n in hw)
             and hw[0] * hw[1] <= pilot_api.MAX_PIXELS, 'Unexpected canonical DEV image ID/order or original H/W')
        ref = item['image_ref']
        path = pilot_api.original_ref(ref, max_bytes=pilot_api.MAX_JPEG_BYTES, suffix='.jpg')
        need(path.parent.name == 'valid' and path not in paths
             and (not selected or path.parent == selected[0][0].parent),
             'Each selected file must be a distinct JPG in the same native valid directory')
        paths.add(path)
        selected.append((path, dict(ref), expected_id, tuple(hw)))
    return selected


def read_archive(archive_ref, torch, pilot_api):
    path = pilot_api.original_ref(archive_ref, max_bytes=pilot_api.MAX_ARCHIVE_BYTES, suffix='.ckpt')
    need(path.name == ARCHIVE_NAME, 'Expected the fixed native E24 full checkpoint_23.ckpt')
    with path.open('rb') as stream:
        first = os.fstat(stream.fileno())
        need(first.st_size == archive_ref['bytes']
             and hashlib.file_digest(stream, 'sha256').hexdigest() == archive_ref['sha256'],
             'Original E24 full checkpoint byte identity differs')
        stream.seek(0)
        checkpoint = torch.load(stream, map_location='cpu', weights_only=True)
        pilot_api.unchanged_stat(stream, path, first)
    return checkpoint


def ema_state_for_model(checkpoint, expected_model_keys):
    """E24 version of the admitted native EMA wrapper-to-model mapping."""
    need(type(checkpoint['global_step']) is int and checkpoint['global_step'] == E24_STEPS,
         'Expected E24 cursor 4464')
    raw = checkpoint['state_dict']
    callback = checkpoint['callbacks'][EMA_KEY]
    avg = callback['average_model_state_dict']
    count = avg['n_averaged']
    count = count.item() if callable(getattr(count, 'item', None)) else count
    need(type(count) is type(callback['latest_update_step']) is int and count == callback['latest_update_step'] == E24_STEPS,
         'Expected 4464 native EMA updates')
    need(isinstance(raw, dict) and isinstance(avg, dict)
         and all(type(key) is str for key in (*raw, *avg)), 'Unexpected native state dictionary')
    wrapped = {'module.' + key for key in raw if key.rsplit('.', 1)[-1] != '_extra_state'}
    need(set(avg) == wrapped | {'n_averaged'}, 'Native EMA wrapper keys do not match the raw module namespace')
    selected = OrderedDict((key[len(MODEL_PREFIX):], value)
                           for key, value in avg.items() if key.startswith(MODEL_PREFIX))
    need(set(selected) == set(expected_model_keys) and CLASSIFIER in selected,
         'Native E24 eager model key set differs')
    metadata = getattr(avg, '_metadata', None)
    if metadata is not None:
        prefix = MODEL_PREFIX[:-1]
        selected._metadata = OrderedDict(
            (key[len(MODEL_PREFIX):] if key != prefix else '', value)
            for key, value in metadata.items() if key == prefix or key.startswith(MODEL_PREFIX)
        )
    return selected, raw['model.' + CLASSIFIER]


def strict_ema_load(model, checkpoint, torch):
    eager = model.state_dict()
    selected, raw_classifier = ema_state_for_model(checkpoint, eager.keys())
    for name, target in eager.items():
        weight = selected[name]
        need(torch.is_tensor(weight) and torch.is_tensor(target)
             and target.device.type == weight.device.type == 'cpu'
             and weight.shape == target.shape and weight.dtype == target.dtype
             and (not weight.is_floating_point() or bool(torch.isfinite(weight).all().item())),
             'Invalid E24 EMA tensor: ' + name)
    classifier = selected[CLASSIFIER]
    need(torch.is_tensor(raw_classifier) and raw_classifier.shape == classifier.shape
         and classifier.ndim == 2 and classifier.shape[0] == 2
         and raw_classifier.dtype == classifier.dtype == torch.float32
         and bool(torch.isfinite(raw_classifier).all().item()), 'Invalid E24 two-slot classifier witness')
    loaded = model.load_state_dict(selected, strict=True)
    need(not loaded.missing_keys and not loaded.unexpected_keys, 'Native strict E24 EMA load was incomplete')
    readback = model.state_dict()
    need(set(readback) == set(selected) and all(torch.equal(value, selected[name]) for name, value in readback.items()),
         'Native E24 eager EMA did not read back exactly')
    return dict(tensors=len(selected), class_slots=2, ema_updates=E24_STEPS, strict_readback_exact=True,
                raw_ema_classifier_max_abs_difference=(raw_classifier - classifier).abs().max().item())


def output_path(value):
    if value is None:
        return None
    path = Path(value)
    need(path.is_absolute() and path.resolve() == path and path.parent.is_dir()
         and not path.exists() and not path.is_symlink(), 'Expected a fresh absolute private export directory')
    return path


def write_population(items, infer_one, *, output_directory, checkpoint_sha256, recipe_sha256, event_sink=None):
    """Stream flat COCO rows in declared order, keeping a bounded image index."""
    need(len(items) == DEV_IMAGES and sha_string(checkpoint_sha256) and sha_string(recipe_sha256),
         'Expected fixed population and source hashes')
    serialize_started = time.monotonic()
    out = output_path(output_directory)
    if out is not None:
        out.mkdir(mode=0o700)
    partial = out / (PREDICTIONS + '.partial') if out is not None else None
    digest = hashlib.sha256()
    size = 0
    images, total = [], 0

    def emit(handle, value):
        nonlocal size
        if handle is not None:
            need(size + len(value) <= MAX_PREDICTION_BYTES, 'Prediction artifact exceeds its bounded byte budget')
            handle.write(value)
            digest.update(value)
            size += len(value)

    with (partial.open('xb') if partial is not None else nullcontext(None)) as stream:
        emit(stream, b'[')
        serialization_seconds = time.monotonic() - serialize_started
        for index, item in enumerate(items):
            _, ref, image_id, hw = item
            rows, diagnostic = infer_one(item)
            exact(diagnostic, ('raw_query_count', 'native_count', 'unmapped_slot1_count', 'retained_native_ranks',
                               'coco_rows', 'codec_roundtrip_masks', 'nonempty_masks', 'all_zero_masks',
                               'canonical_json_sha256'), 'per-image interface diagnostic')
            need(type(rows) is list, 'Expected canonical COCO prediction list')
            count = len(rows)
            ranks = diagnostic['retained_native_ranks']
            counters = ('raw_query_count', 'native_count', 'unmapped_slot1_count', 'coco_rows',
                        'codec_roundtrip_masks', 'nonempty_masks', 'all_zero_masks')
            need(all(type(diagnostic[key]) is int and diagnostic[key] >= 0 for key in counters)
                 and type(ranks) is list and len(ranks) == count
                 and all(type(rank) is int and 1 <= rank <= NATIVE_ROWS for rank in ranks)
                 and all(a < b for a, b in zip(ranks, ranks[1:]))
                 and diagnostic['raw_query_count'] == diagnostic['native_count'] == NATIVE_ROWS
                 and diagnostic['unmapped_slot1_count'] + count == NATIVE_ROWS
                 and diagnostic['coco_rows'] == diagnostic['codec_roundtrip_masks'] == count
                 and diagnostic['nonempty_masks'] + diagnostic['all_zero_masks'] == count,
                 'Native row, mapped rank or round-trip accounting differs')
            need(all(row['image_id'] == image_id and row['category_id'] == 1 for row in rows),
                 'Unexpected canonical row image/category')
            serialize_started = time.monotonic()
            payload = json_bytes(rows)
            need(hashlib.sha256(payload).hexdigest() == diagnostic['canonical_json_sha256'],
                 'Canonical rows changed after RLE verification')
            if count:
                if total:
                    emit(stream, b',')
                emit(stream, payload[1:-1])
            images.append(dict(image_id=image_id, image_sha256=ref['sha256'], original_hw=list(hw),
                               prediction_start=total, prediction_count=count, **diagnostic))
            total += count
            if stream is not None:
                stream.flush()
            serialization_seconds += time.monotonic() - serialize_started
            if index == 1:
                need(total > 0, 'First two fixed images did not exercise the real mapped RLE codec')
                if event_sink is not None:
                    event_sink(dict(event='e24_first_two_fixed_images_passed', canonical_rows=total))
            if event_sink is not None and (index + 1) % 32 == 0:
                event_sink(dict(event='e24_native_dev_export_progress', images=index+1, canonical_rows=total))
        serialize_started = time.monotonic()
        emit(stream, b']\n')
        if stream is not None:
            stream.flush()
            os.fsync(stream.fileno())

    native = sum(row['native_count'] for row in images)
    unmapped = sum(row['unmapped_slot1_count'] for row in images)
    zero_images = sum(row['prediction_count'] == 0 for row in images)
    need(len(images) == DEV_IMAGES and native == DEV_IMAGES * NATIVE_ROWS and total + unmapped == native,
         'Whole native DEV export accounting differs')
    summary = dict(image_count=len(images), native_count_total=native, unmapped_slot1_total=unmapped,
                   prediction_count=total, zero_prediction_images=zero_images,
                   codec_roundtrip_masks=sum(row['codec_roundtrip_masks'] for row in images),
                   nonempty_masks=sum(row['nonempty_masks'] for row in images),
                   all_zero_masks=sum(row['all_zero_masks'] for row in images))
    predictions_ref = manifest_ref = None
    if out is not None:
        prediction_path = out / PREDICTIONS
        manifest_path = out / MANIFEST
        predictions_ref = dict(path=prediction_path.as_posix(), bytes=size, sha256=digest.hexdigest())
        manifest = dict(schema='RFDETR-E24-canonical-DEV-image-index-v1',
                        checkpoint_sha256=checkpoint_sha256, recipe_sha256=recipe_sha256,
                        prediction_offset_base=0, native_rank_base=1, predictions_ref=predictions_ref,
                        images=images, **summary)
        manifest_payload = json_bytes(manifest) + b'\n'
        manifest_partial = out / (MANIFEST + '.partial')
        with manifest_partial.open('xb') as handle:
            handle.write(manifest_payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, prediction_path)
        os.replace(manifest_partial, manifest_path)
        manifest_ref = dict(path=manifest_path.as_posix(), bytes=len(manifest_payload),
                            sha256=hashlib.sha256(manifest_payload).hexdigest())
    serialization_seconds += time.monotonic() - serialize_started
    return dict(predictions_ref=predictions_ref, image_manifest_ref=manifest_ref,
                serialization_and_optional_write_seconds=serialization_seconds, **summary)


def run_e24_dev_export(archive_ref, items, expected_ids, model_config, train_config, *,
                       recipe_sha256, validate_archive_progress, boundary_api, pilot_api,
                       export_e1_image, canonical_export, output_directory=None, event_sink=None):
    """Use one admitted original for the first-two gate and complete DEV187."""
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

    need(sha_string(recipe_sha256)
         and all(callable(fn) for fn in (validate_archive_progress, export_e1_image, canonical_export))
         and all(version(name) == expected for name, expected in pilot_api.VERSIONS.items()),
         'Different pinned runtime, checkpoint or export interface')
    need(torch.cuda.is_available() and torch.cuda.device_count() == 1
         and not torch.is_autocast_enabled('cuda') and not torch.is_autocast_enabled('cpu'),
         'Expected one CUDA device and the frozen FP32 eager inference policy')
    selected = select_items(items, expected_ids, pilot_api)
    out = output_path(output_directory)
    if out is not None:
        native_view = selected[0][0].parent.parent
        archive_path = Path(archive_ref['path'])
        need(out != native_view and out not in native_view.parents and native_view not in out.parents
             and out not in archive_path.parents, 'Export directory overlaps original input paths')
    checkpoint = read_archive(archive_ref, torch, pilot_api)
    validate_archive_progress(checkpoint, EPOCH, boundary_api=boundary_api, recipe_sha256=recipe_sha256)
    pilot_api.match_model_config(checkpoint, model_config, train_config, config)
    model = build_model_from_config(model_config, train_config, defaults=replace(MODEL_DEFAULTS, force_no_pretrain=True))
    ema = strict_ema_load(model, checkpoint, torch)
    del checkpoint
    if event_sink is not None:
        event_sink(dict(event='e24_native_ema_strict_readback_complete', tensors=ema['tensors'], ema_updates=E24_STEPS))
    device = torch.device('cuda:0')
    torch.cuda.reset_peak_memory_stats(device)
    model = model.to(device).eval()
    need(not model.training and model.num_queries == NATIVE_ROWS and model.group_detr == 13,
         'Expected one native eval query group')
    transform = make_coco_transforms_square_div_64(
        'val', 576, multi_scale=False, expanded_scales=False, skip_random_resize=True,
        patch_size=model_config.patch_size, num_windows=model_config.num_windows, aug_config=None,
        scale_jitter=train_config.scale_jitter, gpu_postprocess=False,
    )

    torch.cuda.synchronize(device)
    timing = dict(setup_and_strict_load=time.monotonic()-started, jpeg_read_decode_transform=0.0,
                  input_pack_transfer=0.0, synchronized_cuda_forward=0.0, stock_postprocess_rle_roundtrip=0.0)

    def infer_one(item):
        path, ref, image_id, hw = item
        phase_started = time.monotonic()
        image, scales = decode_pil_image_bytes(pilot_api.read_image(path, ref), draft_size=None)
        need(image.mode == 'RGB' and image.size == (hw[1], hw[0]) and scales == (1.0, 1.0),
             'Native DEV JPEG decode differs from fixed original dimensions')
        tensor, target = transform(image, None)
        need(target is None and tensor.dtype == torch.float32 and tuple(tensor.shape) == (3, 576, 576)
             and bool(torch.isfinite(tensor).all().item()), 'Native image-only 576 transform differs')
        timing['jpeg_read_decode_transform'] += time.monotonic() - phase_started
        with torch.inference_mode():
            phase_started = time.monotonic()
            samples = nested_tensor_from_tensor_list(
                [tensor], block_size=model_config.patch_size * model_config.num_windows,
            ).to(device)
            need(tuple(samples.tensors.shape) == (1, 3, 576, 576) and samples.mask is not None
                 and not bool(samples.mask.any().item()), 'Unexpected native spatial padding')
            torch.cuda.synchronize(device)
            timing['input_pack_transfer'] += time.monotonic() - phase_started
            phase_started = time.monotonic()
            raw = model(samples)
            torch.cuda.synchronize(device)
            timing['synchronized_cuda_forward'] += time.monotonic() - phase_started
            phase_started = time.monotonic()
            need(tuple(raw['pred_masks'].shape) == (1, NATIVE_ROWS, 144, 144)
                 and all(raw[key].dtype == torch.float32 for key in ('pred_logits', 'pred_boxes', 'pred_masks')),
                 'Native FP32 dense mask-head geometry differs')
            output = export_e1_image(
                raw, image_id=image_id, original_hw=hw, torch_api=torch, postprocess_type=PostProcess,
                canonical_export=canonical_export, encode_mask=lambda mask: codec.encode(np.asfortranarray(mask, dtype=np.uint8)),
            )
            checked = pilot_api.codec_roundtrip(output, image_id, hw, np, codec)
            labels = output['native']['labels'].detach().cpu().tolist()
            ranks = [index + 1 for index, label in enumerate(labels) if label == 0]
            torch.cuda.synchronize(device)
            timing['stock_postprocess_rle_roundtrip'] += time.monotonic() - phase_started
        return output['coco'], dict(raw_query_count=output['raw_query_count'], native_count=output['native_count'],
                                   unmapped_slot1_count=output['unmapped_slot1_count'],
                                   retained_native_ranks=ranks, **checked)

    evidence = write_population(selected, infer_one, output_directory=out,
                                checkpoint_sha256=archive_ref['sha256'], recipe_sha256=recipe_sha256,
                                event_sink=event_sink)
    timing['serialization_and_optional_write'] = evidence.pop('serialization_and_optional_write_seconds')
    timing['worker_elapsed'] = time.monotonic() - started
    return dict(status='NATIVE_E24_EMA_DEV187_EXPORT_COMPLETE_PENDING_ROOT',
                qualified=False, terminal_self_certified=False, AP=False, TEST=False, GT_read=False,
                training=False, full_DEV=True, checkpoint_weights_only_cpu=True,
                fixed_epoch=EPOCH, optimizer_updates=E24_STEPS, checkpoint_sha256=archive_ref['sha256'],
                recipe_sha256=recipe_sha256, eager_model_state=ema, timing_seconds=timing,
                tensor_precision='float32', autocast=False,
                float32_matmul_precision=torch.get_float32_matmul_precision(),
                cudnn_tf32=torch.backends.cudnn.allow_tf32,
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device), **evidence)
