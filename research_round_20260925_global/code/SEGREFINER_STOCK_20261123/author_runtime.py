"""Inference adapter for the published SegRefiner-LR COCO configuration.

The three externally supplied author files execute without source edits. Only
the OpenMMLab registration, constructor and bitmap interfaces are emulated.
Inference uses the caller's PyTorch backend and RNG; seed it before replay.
"""

import builtins
import hashlib
from pathlib import Path
from types import ModuleType, SimpleNamespace

import cv2
import numpy as np
import torch
from torch import nn


AUTHOR_COMMIT = '53419a2d38ea3da0b6e2be77e5b45e139195a0b3'
SOURCE_SHA256 = {
    'diffusion_unet_head.py': 'd782f93cbd402943928cbb504b62c709d2820514ec80674999b43e72d4c2e15e',
    'base.py': 'c94561e5d5ee9f5873c13f03e9bf6ff93b09d0e0e1ff9a24dd60de150cb0ea1a',
    'instance.py': 'b77677c507220ce076ae5595cbd476abca8236e977b6dd5c110b0f9b95f55049',
}
CHECKPOINT_SHA256 = 'd5806a41a4bff6f1e313d1a9dcf18d11dab28e3a8e1497871a156214cd8fe07a'
CHECKPOINT_SIZE = 474603251
AREA_THRESHOLD = 512


class _BaseModule(nn.Module):
    def __init__(self, init_cfg=None):
        super().__init__()
        self.init_cfg = init_cfg


def _author_classes(directory):
    registry = SimpleNamespace(register_module=lambda: (lambda cls: cls))
    builder = SimpleNamespace(HEADS=registry, DETECTORS=registry)
    bridges = {(0, 'mmcv.runner'): SimpleNamespace(BaseModule=_BaseModule),
               (2, 'builder'): builder}

    def local_import(name, globals=None, locals=None, fromlist=(), level=0):
        if (level, name) in bridges:
            return bridges[level, name]
        return builtins.__import__(name, globals, locals, fromlist, level)

    def load(name):
        path = Path(directory) / name
        source = path.read_bytes()
        if hashlib.sha256(source).hexdigest() != SOURCE_SHA256[name]:
            raise ValueError('Unexpected upstream source: ' + name)
        module = ModuleType('segrefiner_author_' + path.stem)
        module.__file__ = str(path)
        module.__dict__['__builtins__'] = dict(vars(builtins), __import__=local_import)
        exec(compile(source, str(path), 'exec'), module.__dict__)
        return module

    head = load('diffusion_unet_head.py').DenoiseUNet

    def build_head(config):
        cfg = config.copy()
        if cfg.pop('type') != 'DenoiseUNet':
            raise ValueError('Unsupported denoiser')
        return head(**cfg)

    builder.build_head = build_head
    # Author inference never calls training losses. Calling a stub raises.
    builder.build_loss = lambda cfg: nn.Module()
    bridges[1, 'segrefiner_base'] = load('base.py')
    return load('instance.py').SegRefinerInstance


def _load_checkpoint(head, checkpoint):
    with Path(checkpoint).open('rb') as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
        if stream.tell() != CHECKPOINT_SIZE or digest.hexdigest() != CHECKPOINT_SHA256:
            raise ValueError('Unexpected SegRefiner-LR checkpoint size or SHA-256')
        stream.seek(0)
        weights = torch.load(stream, weights_only=True, map_location='cpu')
    prefix = 'denoise_model.'
    denoiser = {k[len(prefix):]: value for k, value in weights.items() if k.startswith(prefix)}
    training_keys = {k for k in weights if not k.startswith(prefix)}
    if len(denoiser) != 453 or training_keys != {'loss_texture.sobel.conv_x.weight',
                                                   'loss_texture.sobel.conv_y.weight'}:
        raise ValueError('Unexpected SegRefiner-LR parameter names')
    head.load_state_dict(denoiser, strict=True)


def _normalized_rgb(rgb):
    # MMCV 1.7.1 imnormalize_, starting from already-decoded RGB uint8.
    mean = np.array([[123.675, 116.28, 103.53]], dtype=np.float32).astype(np.float64)
    std = np.array([[58.395, 57.12, 57.375]], dtype=np.float32).astype(np.float64)
    image = np.array(rgb, dtype=np.float32, order='C', copy=True)
    cv2.subtract(image, mean, image)
    cv2.multiply(image, 1.0 / std, image)
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).unsqueeze(0)


class AuthorRefiner:
    """Replay single-image stock inference; labels are zero-based COCO indices."""

    def __init__(self, author_dir: Path, checkpoint: Path, device: str):
        self.device = torch.device(device)
        instance = _author_classes(author_dir)
        self.model = instance(
            task='instance', step=6,
            denoise_model=dict(type='DenoiseUNet', in_channels=4, out_channels=1,
                               model_channels=128, num_res_blocks=2, num_heads=4,
                               num_heads_upsample=-1, attention_strides=(16, 32),
                               learn_time_embd=True, channel_mult=(1, 1, 2, 2, 4, 4),
                               dropout=0.0),
            diffusion_cfg=dict(betas=dict(type='linear', start=0.8, stop=0,
                                         num_timesteps=6), diff_iter=False),
            test_cfg=dict(pad_width=20, model_size=256, batch_max=32, area_thr=AREA_THRESHOLD),
        )
        _load_checkpoint(self.model.denoise_model, checkpoint)
        self.model.to(device=self.device, dtype=torch.float32).eval()

    @torch.inference_mode()
    def refine(self, rgb_uint8_HWC, masks_bool_NHW, boxes_xyxy_float32_N5, labels_int64_N):
        rgb, masks, boxes, labels = map(np.asarray, (rgb_uint8_HWC, masks_bool_NHW,
                                                       boxes_xyxy_float32_N5, labels_int64_N))
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1:
            raise ValueError('Expected native RGB uint8 image with shape H x W x 3')
        if masks.dtype != np.bool_ or masks.ndim != 3 or masks.shape[1:] != rgb.shape[:2]:
            raise ValueError('Expected native boolean N x H x W masks')
        n = len(masks)
        if boxes.dtype != np.float32 or boxes.shape != (n, 5) or not np.isfinite(boxes).all():
            raise ValueError('Expected finite float32 xyxy+score rows')
        if labels.dtype != np.int64 or labels.shape != (n,) or ((labels < 0) | (labels >= 80)).any():
            raise ValueError('Expected int64 class indices in [0, 79]')
        if not n:
            return masks.copy()

        areas = masks.sum(axis=(1, 2), dtype=np.int64)
        bitmaps = SimpleNamespace(masks=np.ascontiguousarray(masks, dtype=np.uint8), areas=areas)
        detections = np.column_stack((boxes, labels))
        meta = dict(img_shape=rgb.shape, ori_shape=rgb.shape)
        [(class_boxes, class_masks)] = self.model(
            img_metas=[meta], return_loss=False, img=_normalized_rgb(rgb).to(self.device),
            coarse_masks=[bitmaps], dt_bboxes=[detections],
        )
        # The author emits qualifying masks first, then tiny masks, grouped by
        # class. Stable integer positions also distinguish identical boxes/scores.
        order = np.concatenate((np.flatnonzero(areas >= AREA_THRESHOLD),
                                np.flatnonzero(areas < AREA_THRESHOLD)))
        restored = np.empty_like(masks)
        for label in np.unique(labels):
            positions = order[labels[order] == label]
            if (len(class_masks[label]) != len(positions) or
                    not np.array_equal(class_boxes[label], boxes[positions])):
                raise ValueError('Returned instance metadata differs from frozen input')
            restored[positions] = np.stack(class_masks[label])
        return restored
