"""Test the original SegRefiner inference entry point on synthetic SAR-like arrays."""

import argparse
from hashlib import sha256
from importlib import util
import json
from pathlib import Path
import random
from time import perf_counter

import numpy as np
import torch

RUNTIME_SHA256 = "85a6a95205c6842e1e4f5f24805d911a0ef5bb0eded0d5ff67d7e020fe04d816"


def load_author_runtime(path):
    source = path.read_bytes()
    if sha256(source).hexdigest() != RUNTIME_SHA256:
        raise ValueError("Expected the unmodified stock wrapper")
    spec = util.spec_from_file_location("segrefiner_stock_runtime", path)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.AuthorRefiner


def run(author_source, checkpoint, admitted, runtime):
    random.seed(5217)
    np.random.seed(5217)
    torch.manual_seed(5217)
    device = torch.device("cuda:0")
    before_free, before_total = torch.cuda.mem_get_info(device)
    model = load_author_runtime(runtime)(author_source, checkpoint, str(device))

    height, width = 256, 256
    yy, xx = np.indices((height, width))
    channel = ((7 * xx + 11 * yy + (xx * yy) % 19) % 251).astype(np.uint8)
    rgb = np.repeat(channel[:, :, None], 3, axis=2)
    masks = np.zeros((admitted + 1, height, width), dtype=bool)
    for i, mask in enumerate(masks):
        y, x = 22 + 5 * (i // 8), 30 + 6 * (i % 8)
        mask[y:y + 16, x:x + 32] = True
        if i == admitted:
            mask[y, x] = False
        else:
            rgb[y:y + 16, x:x + 32] = 220
    boxes = np.empty((admitted + 1, 5), dtype=np.float32)
    for i, mask in enumerate(masks):
        y, x = np.nonzero(mask)
        boxes[i] = [x.min(), y.min(), x.max() + 1, y.max() + 1, 1 - i / 100]
    labels = np.zeros(len(masks), dtype=np.int64)
    assert (masks.sum(axis=(1, 2))[:-1] == 512).all() and masks[-1].sum() == 511

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    start = perf_counter()
    result = model.refine(rgb, masks.copy(), boxes, labels)
    torch.cuda.synchronize(device)
    elapsed = perf_counter() - start
    assert result.shape == masks.shape and result.dtype == np.bool_
    assert np.array_equal(result[-1], masks[-1])
    return dict(
        synthetic_only=True, actual_data_opened=False, stock_semantics=True,
        device=torch.cuda.get_device_name(device), capability=torch.cuda.get_device_capability(device),
        torch=torch.__version__, compiled_cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
        checkpoint_strict_verified=True, input_masks=len(masks), admitted=admitted,
        exact_bypass=1, changed_admitted=int(np.any(result[:-1] != masks[:-1], axis=(1, 2)).sum()),
        setup_available_gib=before_free / 2**30, device_total_gib=before_total / 2**30,
        torch_peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30,
        torch_peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 2**30,
        inference_seconds=elapsed,
    )


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--author", required=True, type=Path)
    cli.add_argument("--checkpoint", required=True, type=Path)
    cli.add_argument("--runtime", type=Path, default=Path(__file__).with_name("author_runtime.py"))
    cli.add_argument("--admitted", type=int, default=8)
    args = cli.parse_args()
    if not 1 <= args.admitted <= 32:
        cli.error("--admitted must be between 1 and 32")
    print(json.dumps(run(args.author, args.checkpoint, args.admitted, args.runtime), sort_keys=True))

