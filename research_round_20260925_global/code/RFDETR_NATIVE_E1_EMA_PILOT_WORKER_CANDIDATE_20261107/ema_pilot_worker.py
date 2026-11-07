"""Two-image eager E1 EMA export qualification worker."""
from collections import OrderedDict
from dataclasses import replace
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import re


E1_STEPS = 186
EMA_KEY = 'RFDETREMACallback'
MODEL_PREFIX = 'module.model.'
CLASSIFIER = 'class_embed.weight'
EXPECTED_MODEL = dict(num_classes=1, resolution=576, positional_encoding_size=48,
                      num_channels=3, patch_size=12, num_windows=1, num_queries=100,
                      num_select=100, group_detr=13, mask_downsample_ratio=4,
                      segmentation_head=True, compile=False, cuda_graphs=False,
                      fused_optimizer=False, use_grouppose_keypoints=False, backbone_lora=False)
VERSIONS = dict(rfdetr='1.12.0', torch='2.10.0+cu130', torchvision='0.25.0+cu130',
                numpy='1.26.4', pycocotools='2.0.11', simplejpeg='1.9.0')
MAX_ARCHIVE_BYTES = 1024**3
MAX_JPEG_BYTES = 16 * 1024**2
MAX_PIXELS = 2 * 1024**2


def need(condition, message):
    if not condition:
        raise ValueError(message)


def int_value(value, expected):
    return type(value) is int and value == expected


def exact(obj, keys, label):
    need(type(obj) is dict and set(obj) == set(keys), 'Unexpected ' + label + ' fields')


def ema_state_for_model(checkpoint, expected_model_keys):
    """Map exactly the native AveragedModel namespace into eager model keys."""
    need(isinstance(checkpoint, dict) and int_value(checkpoint.get('global_step'), E1_STEPS),
         'Expected native E1 optimizer cursor')
    raw = checkpoint['state_dict']
    callback = checkpoint['callbacks'][EMA_KEY]
    avg = callback['average_model_state_dict']
    count = avg['n_averaged']
    count = count.item() if callable(getattr(count, 'item', None)) else count
    need(int_value(callback['latest_update_step'], E1_STEPS) and int_value(count, E1_STEPS),
         'Expected exactly 186 native EMA updates')
    need(isinstance(raw, dict) and isinstance(avg, dict)
         and all(type(key) is str for key in (*raw, *avg)), 'Unexpected native state dictionary')
    wrapped = {'module.' + key for key in raw if key.rsplit('.', 1)[-1] != '_extra_state'}
    need(set(avg) == wrapped | {'n_averaged'}, 'Native EMA wrapper keys do not match the raw module namespace')
    selected = OrderedDict((key[len(MODEL_PREFIX):], tensor)
                           for key, tensor in avg.items() if key.startswith(MODEL_PREFIX))
    need(set(selected) == set(expected_model_keys) and CLASSIFIER in selected,
         'Native EMA eager model key set differs')
    metadata = getattr(avg, '_metadata', None)
    if metadata is not None:
        prefix = MODEL_PREFIX[:-1]
        selected._metadata = OrderedDict(
            (key[len(MODEL_PREFIX):] if key != prefix else '', value)
            for key, value in metadata.items() if key == prefix or key.startswith(MODEL_PREFIX)
        )
    return selected, raw['model.' + CLASSIFIER]


def original_ref(ref, *, max_bytes, suffix):
    exact(ref, ('path', 'bytes', 'sha256'), 'original file reference')
    raw_path = ref['path']
    need(type(raw_path) is str and raw_path.startswith('/') and '\\' not in raw_path
         and not any(ord(char) < 32 for char in raw_path)
         and type(ref['bytes']) is int and 0 < ref['bytes'] <= max_bytes
         and type(ref['sha256']) is str and re.fullmatch(r'[0-9a-f]{64}', ref['sha256']) is not None,
         'Invalid bounded original file reference')
    path = Path(raw_path)
    need(str(path) == raw_path and '..' not in path.parts and path.suffix == suffix
         and path.resolve(strict=True) == path and path.is_file(), 'Original reference is not a canonical file')
    return path


def unchanged_stat(stream, path, before):
    def identity(state):
        return state.st_dev, state.st_ino, state.st_size, state.st_mtime_ns

    need(identity(before) == identity(os.fstat(stream.fileno())) == identity(path.stat())
         and path.resolve(strict=True) == path, 'Original file changed during readback')


def load_archive(ref, torch):
    path = original_ref(ref, max_bytes=MAX_ARCHIVE_BYTES, suffix='.ckpt')
    need(path.name == 'checkpoint_0.ckpt', 'Expected native E1 full archive name')
    with path.open('rb') as stream:
        first = os.fstat(stream.fileno())
        need(first.st_size == ref['bytes']
             and hashlib.file_digest(stream, 'sha256').hexdigest() == ref['sha256'],
             'E1 archive byte identity differs')
        stream.seek(0)
        checkpoint = torch.load(stream, map_location='cpu', weights_only=True)
        unchanged_stat(stream, path, first)
    return checkpoint


def select_items(items):
    need(type(items) in (list, tuple) and len(items) == 2, 'Exactly two externally fixed DEV images required')
    paths, ids = set(), set()
    selected = []
    for item in items:
        exact(item, ('image_ref', 'image_id', 'original_hw'), 'DEV item')
        path = original_ref(item['image_ref'], max_bytes=MAX_JPEG_BYTES, suffix='.jpg')
        hw = item['original_hw']
        image_id = item['image_id']
        need(type(image_id) is int and image_id >= 0 and image_id not in ids
             and type(hw) in (list, tuple) and len(hw) == 2
             and all(type(n) is int and n > 0 for n in hw) and hw[0] * hw[1] <= MAX_PIXELS
             and path.parent.name == 'valid' and path not in paths, 'Invalid fixed DEV image identity or geometry')
        ids.add(image_id)
        paths.add(path)
        selected.append((path, item['image_ref'], image_id, tuple(hw)))
    need(selected[0][0].parent == selected[1][0].parent, 'Both images must share the fixed DEV directory')
    return selected


def read_image(path, ref):
    with path.open('rb') as stream:
        first = os.fstat(stream.fileno())
        need(first.st_size == ref['bytes'], 'Original JPEG size differs')
        raw = stream.read(ref['bytes'] + 1)
        need(len(raw) == ref['bytes'] and raw.startswith(b'\xff\xd8\xff')
             and hashlib.sha256(raw).hexdigest() == ref['sha256'], 'Original JPEG byte identity differs')
        unchanged_stat(stream, path, first)
    return raw


def match_model_config(checkpoint, config, train, config_module):
    need(type(config) is config_module.RFDETRSegNanoConfig and type(train) is config_module.SegmentationTrainConfig
         and train.square_resize_div_64 is True and train.multi_scale is config_module.MultiScale.OFF
         and train.augmentation_backend == config_module.AugmentationBackend.TV and train.aug_config is None,
         'Expected admitted RF-DETR Seg Nano 576 and stock validation transform recipe')
    stored = checkpoint['model_config']
    need(isinstance(stored, dict), 'Native E1 checkpoint model description missing')
    for key, expected in EXPECTED_MODEL.items():
        actual = getattr(config, key)
        saved = stored.get(key)
        need(type(actual) is type(saved) is type(expected) and actual == saved == expected,
             'Native E1 model recipe differs: ' + key)


def strict_ema_load(model, checkpoint, torch):
    model_state = model.state_dict()
    selected, raw_classifier = ema_state_for_model(checkpoint, model_state.keys())
    for name, target in model_state.items():
        weight = selected[name]
        need(torch.is_tensor(weight) and torch.is_tensor(target)
             and target.device.type == weight.device.type == 'cpu'
             and weight.shape == target.shape and weight.dtype == target.dtype
             and (not weight.is_floating_point() or bool(torch.isfinite(weight).all().item())),
             'Invalid native EMA tensor: ' + name)
    classifier = selected[CLASSIFIER]
    need(torch.is_tensor(raw_classifier) and raw_classifier.shape == classifier.shape
         and classifier.ndim == 2 and classifier.shape[0] == 2 and classifier.dtype == torch.float32
         and raw_classifier.dtype == classifier.dtype and bool(torch.isfinite(raw_classifier).all().item())
         and not torch.equal(raw_classifier, classifier), 'Native raw/EMA two-slot witness differs from admission')
    loaded = model.load_state_dict(selected, strict=True)
    need(not loaded.missing_keys and not loaded.unexpected_keys, 'Native strict EMA load was incomplete')
    readback = model.state_dict()
    need(set(readback) == set(selected) and all(torch.equal(tensor, selected[name]) for name, tensor in readback.items()),
         'Native eager EMA state did not read back exactly')
    return dict(tensors=len(selected), class_slots=2, ema_updates=E1_STEPS, strict_readback_exact=True,
                raw_ema_classifier_max_abs_difference=(raw_classifier - classifier).abs().max().item())


def codec_roundtrip(output, image_id, original_hw, np, codec):
    native = output['native']
    labels = native['labels'].detach().cpu().tolist()
    scores = native['scores'].detach().cpu().tolist()
    bool_masks = native['masks'].detach().cpu().numpy()
    selected = [rank for rank, label in enumerate(labels) if label == 0]
    raw = json.dumps(output['coco'], separators=(',', ':'), allow_nan=False).encode('utf-8')
    rows = json.loads(raw)
    need(type(rows) is list and len(rows) == len(selected) == 100 - output['unmapped_slot1_count'],
         'Canonical selected-row count differs')
    nonempty = 0
    for rank, row in zip(selected, rows):
        exact(row, ('image_id', 'category_id', 'score', 'segmentation'), 'canonical mask record')
        need(row['image_id'] == image_id and row['category_id'] == 1 and row['score'] == scores[rank],
             'Canonical image, category, score or rank differs')
        rle = row['segmentation']
        exact(rle, ('size', 'counts'), 'canonical compressed RLE')
        need(rle['size'] == list(original_hw) and type(rle['counts']) is str
             and bool(rle['counts']) and rle['counts'].isascii(), 'Invalid canonical RLE after JSON')
        decoded = codec.decode(dict(size=rle['size'], counts=rle['counts'].encode('ascii')))
        original = bool_masks[rank, 0]
        need(decoded.shape == original.shape == original_hw and decoded.dtype == np.uint8
             and np.array_equal(decoded, original), 'Canonical RLE round-trip pixel mismatch')
        nonempty += int(original.any())
    return dict(coco_rows=len(rows), codec_roundtrip_masks=len(rows), nonempty_masks=nonempty,
                all_zero_masks=len(rows)-nonempty, canonical_json_sha256=hashlib.sha256(raw).hexdigest())


def run_ema_pilot(archive_ref, items, model_config, train_config, *, recipe_sha256, validate_checkpoint,
                  export_e1_image, canonical_export, event_sink=None):
    """Read one admitted archive and exactly two frozen DEV JPEGs in an owned process."""
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

    def event(name, **fields):
        if event_sink is not None:
            event_sink(dict(event=name, **fields))

    need(all(callable(fn) for fn in (validate_checkpoint, export_e1_image, canonical_export)),
         'Missing independently pinned checkpoint or export interface')
    need(type(recipe_sha256) is str and re.fullmatch(r'[0-9a-f]{64}', recipe_sha256) is not None,
         'Expected exact E1 recipe SHA256')
    need(all(version(name) == expected for name, expected in VERSIONS.items()),
         'Unexpected scientific runtime versions')
    need(torch.cuda.is_available() and torch.cuda.device_count() == 1
         and not torch.is_autocast_enabled('cuda') and not torch.is_autocast_enabled('cpu'),
         'Expected single CUDA device and FP32 eager pilot')
    fixed_items = select_items(items)
    checkpoint = load_archive(archive_ref, torch)
    validate_checkpoint(checkpoint, recipe_sha256=recipe_sha256)
    match_model_config(checkpoint, model_config, train_config, config)
    model = build_model_from_config(model_config, train_config, defaults=replace(MODEL_DEFAULTS, force_no_pretrain=True))
    state = strict_ema_load(model, checkpoint, torch)
    del checkpoint
    event('native_e1_ema_strict_cpu_readback_complete', model_tensors=state['tensors'], ema_updates=E1_STEPS)
    device = torch.device('cuda:0')
    torch.cuda.reset_peak_memory_stats(device)
    model = model.to(device).eval()
    need(not model.training and model.num_queries == 100 and model.group_detr == 13,
         'Expected admitted eager eval model with 100 inference queries')
    transform = make_coco_transforms_square_div_64(
        'val', 576, multi_scale=False, expanded_scales=False, skip_random_resize=True,
        patch_size=model_config.patch_size, num_windows=model_config.num_windows, aug_config=None,
        scale_jitter=train_config.scale_jitter, gpu_postprocess=False,
    )
    observations = []
    for index, (path, ref, image_id, hw) in enumerate(fixed_items):
        image, scales = decode_pil_image_bytes(read_image(path, ref), draft_size=None)
        need(image.mode == 'RGB' and image.size == (hw[1], hw[0]) and scales == (1.0, 1.0),
             'Native DEV JPEG decode differs from fixed original dimensions')
        tensor, target = transform(image, None)
        need(target is None and tensor.dtype == torch.float32 and tuple(tensor.shape) == (3, 576, 576)
             and bool(torch.isfinite(tensor).all().item()), 'Native image-only 576 transform differs')
        with torch.inference_mode():
            samples = nested_tensor_from_tensor_list(
                [tensor], block_size=model_config.patch_size * model_config.num_windows,
            ).to(device)
            need(tuple(samples.tensors.shape) == (1, 3, 576, 576) and samples.mask is not None
                 and not bool(samples.mask.any().item()), 'Unexpected native spatial padding')
            raw = model(samples)
            need(tuple(raw['pred_masks'].shape) == (1, 100, 144, 144)
                 and all(raw[key].dtype == torch.float32 for key in ('pred_logits', 'pred_boxes', 'pred_masks')),
                 'Native E1 FP32 dense mask-head geometry differs')
            output = export_e1_image(
                raw, image_id=image_id, original_hw=hw, torch_api=torch, postprocess_type=PostProcess,
                canonical_export=canonical_export, encode_mask=lambda mask: codec.encode(np.asfortranarray(mask, dtype=np.uint8)),
            )
            checked = codec_roundtrip(output, image_id, hw, np, codec)
        observations.append(dict(image_id=image_id, image_sha256=ref['sha256'], original_hw=list(hw),
                                 native_queries=output['raw_query_count'], native_rows=output['native_count'],
                                 unmapped_slot1=output['unmapped_slot1_count'], mask_head_hw=[144, 144], **checked))
        event('native_e1_ema_dev_image_export_checked', sample_index=index, canonical_rows=checked['coco_rows'])
        del output, raw, samples
    torch.cuda.synchronize(device)
    need(sum(row['codec_roundtrip_masks'] for row in observations) > 0,
         'Neither fixed image yielded a mapped row, so the real codec was not exercised')
    return dict(status='NATIVE_E1_EMA_TWO_DEV_EXPORT_COMPLETE_PENDING_ROOT',
                qualified=False, terminal_self_certified=False, AP=False, TEST=False,
                training=False, validation_metrics=False, GT_read=False, full_DEV=False,
                checkpoint_sha256=archive_ref['sha256'], recipe_sha256=recipe_sha256,
                checkpoint_weights_only_cpu=True, eager_model_state=state, image_observations=observations,
                tensor_precision='float32', autocast=False,
                float32_matmul_precision=torch.get_float32_matmul_precision(),
                cudnn_tf32=torch.backends.cudnn.allow_tf32,
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
