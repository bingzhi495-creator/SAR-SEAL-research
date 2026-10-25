"""One original image and full native candidate input check, never SAM/AP."""

import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


def join_original_image(stage, identity):
    """Map a relocated byte copy while retaining the original provenance."""
    original = identity['image']
    if type(stage['ordinal']) is not int or stage['ordinal'] != 0:
        raise ValueError('Only the prospectively fixed first DEV ordinal is admitted')
    if (type(stage['image_id']) is not int or type(identity['image_id']) is not int
            or stage['image_id'] != identity['image_id']):
        raise ValueError('Original image identity does not join the staged image')
    if (stage['historical_source_ref'] != original
            or stage['bytes'] != original['bytes']
            or stage['sha256'] != original['sha256']):
        raise ValueError('Staged JPEG differs from the original image reference')
    return dict(path=stage['remote_path'], bytes=stage['bytes'], sha256=stage['sha256'])


def reference(path):
    path = Path(path)
    raw = path.read_bytes()
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def load_source(ref, alias):
    path = Path(ref['path'])
    assert not path.is_symlink() and reference(path) == ref
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def main():
    import resource
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--spec-sha', required=True)
    args = parser.parse_args()
    raw = Path(args.spec).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == args.spec_sha
    spec = json.loads(raw)
    assert spec['operation'] == 'CPU_FIRST_ORIGINAL_SAR_AND_NATIVE100_INPUT_CHECK'
    assert spec['ordinal'] == 0 and spec['image_id'] == 4
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    os.sched_setaffinity(0, {0, 1})
    resource.setrlimit(resource.RLIMIT_CPU, (60, 61))
    out = Path(spec['output_directory'])
    out.mkdir(exist_ok=False)
    stat = Path('/proc/self/stat').read_text()
    report = dict(status='STARTED', schema='SAMRefiner-first-original-input-check-v1',
                  pid=os.getpid(), pgid=os.getpgrp(),
                  start_ticks=int(stat[stat.rfind(')')+2:].split()[19]),
                  boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                  SAM_model_loaded=False, AP=False, training_updates=0,
                  full187_tensor_validation=False, images_loaded=0)
    report_path = out / 'REPORT.json'
    started = time.monotonic()
    c = None
    try:
        owner = load_source(spec['owner_source'], '_SAM_INPUT_CPU_CONTROL')
        import psutil
        c = owner.make_control(spec, psutil)
        c.check()
        for ref in spec['sources']:
            c.closed(ref)
        assert Path(__file__).resolve() == c.closed(spec['worker_source'])
        import torch
        import numpy as np
        import cv2
        from PIL import Image
        import PIL
        import pycocotools.mask as masks
        for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
            assert os.environ.get(name) == '2'
        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)
        assert torch.get_num_threads() == torch.get_num_interop_threads() == 2
        assert not torch.cuda.is_initialized()
        for name, module in (('torch', torch), ('numpy', np), ('psutil', psutil)):
            assert reference(Path(module.__file__).resolve()) == spec['library_sources'][name]
        refs = spec['source_refs']
        entry = load_source(refs['u65_entry'], '_SAM_INPUT_U65_ENTRY')
        aliases = {}
        def source(role, alias):
            return entry.load_closed(c.closed, refs[role], alias, aliases)
        reader_module = source('portable_reader', '_SAM_INPUT_READER')
        geometry = source('image_geometry', '_SAM_INPUT_GEOMETRY')
        body = source('u65_body', '_SAM_INPUT_U65_BODY')
        diagnostic = source('rle_decoder', '_SAM_INPUT_RLE')
        manifest = json.loads(c.closed(spec['manifest_ref']).read_bytes())
        reader = reader_module.prepare(
            torch, c, manifest['entries'], image_geometry=geometry,
            source_refs={k: refs[k] for k in ('dev_reader', 'dev_rgb_reader', 'image_geometry')})
        item, row, originals = reader.load(0)
        body.validate_loaded_metadata(0, item, row, originals, manifest['entries'][0])
        assert row['image_id'] == 4 and row['dataset_index'] == 0
        stage = json.loads(c.closed(spec['image_stage_ref']).read_bytes())
        image_ref = join_original_image(stage, item['identity'])
        image_path = c.closed(image_ref)
        shape = item['identity']['original_shape']
        with Image.open(image_path) as image:
            assert image.format == 'JPEG' and image.size == (shape[1], shape[0])
            orientation = image.getexif().get(274)
            assert orientation is None or (type(orientation) is int and orientation == 1)
            with image.convert('RGB') as rgb:
                pixels = np.array(rgb, dtype=np.uint8, copy=True)
        assert pixels.shape == (shape[0], shape[1], 3)
        png = out / 'original_rgb.png'
        with Image.fromarray(pixels, mode='RGB') as rgb:
            rgb.save(png, format='PNG')
        loaded_pixels = cv2.cvtColor(cv2.imread(str(png), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        assert loaded_pixels.dtype == np.uint8 and np.array_equal(loaded_pixels, pixels)
        assert len(originals) == 100 and [r['query_id'] for r in originals] == item['query_ids']
        assert item['query_ids'] == manifest['entries'][0]['query_ids']
        original_digest = reader_module.metadata_sha256(originals)
        coarse = np.stack([diagnostic._decode(np, masks, r['segmentation'], shape)
                           for r in originals], axis=0).astype(np.uint8, copy=False)
        assert coarse.shape == (100, shape[0], shape[1])
        mask_path = out / 'native100_masks.npy'
        np.save(mask_path, coarse, allow_pickle=False)
        saved = np.load(mask_path, allow_pickle=False)
        assert np.array_equal(saved, coarse)
        candidates_path = out / 'native100_candidates.json'
        candidates_path.write_text(json.dumps(originals, sort_keys=True, indent=2) + '\n')
        assert reader_module.metadata_sha256(json.loads(candidates_path.read_bytes())) == original_digest
        assert reader_module.metadata_sha256(originals) == original_digest
        empty_positions = np.flatnonzero(~coarse.reshape(100, -1).any(axis=1)).tolist()
        c.check()
        assert not torch.cuda.is_initialized()
        report.update(status='FIRST_ORIGINAL_RGB_AND_NATIVE100_INPUT_CHECK_PASSED',
                      ordinal=0, image_id=4, images_loaded=1, queries=100,
                      original_shape=shape, original_JPEG_ref=image_ref,
                      original_identity_joined=True, current_PIL_PNG_OpenCV_pixels_exact=True,
                      historical_Windows_decoder_pixel_equality_claimed=False,
                      original_RGB_SHA256=hashlib.sha256(pixels.tobytes()).hexdigest(),
                      decoded_native_masks_SHA256=hashlib.sha256(coarse.tobytes()).hexdigest(),
                      native_export_order_and_nonmask_fields_preserved=True,
                      native_metadata_SHA256=original_digest,
                      empty_positions=empty_positions,
                      empty_query_ids=[originals[i]['query_id'] for i in empty_positions],
                      nonempty_masks=100-len(empty_positions),
                      PNG_ref=reference(png), masks_ref=reference(mask_path),
                      candidates_ref=reference(candidates_path),
                      torch_version=str(torch.__version__), numpy_version=np.__version__,
                      Pillow_version=PIL.__version__, opencv_version=cv2.__version__,
                      CUDA_initialized=False, GT_accessed=False,
                      author_internal_return_order_verified=False)
        item = reader = coarse = saved = pixels = loaded_pixels = None
        gc.collect()
        c.check()
    except BaseException as exc:
        report.update(status='FAILED_PRESERVED_NO_AUTO_RETRY', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - started
        report['peak_self_RSS_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        if c is not None:
            report['sampled_control'] = c.summary()
        report_path.write_text(json.dumps(report, sort_keys=True, indent=2) + '\n')
        print(json.dumps({k: v for k, v in report.items() if k not in ('empty_positions', 'empty_query_ids')}), flush=True)


if __name__ == '__main__':
    main()
