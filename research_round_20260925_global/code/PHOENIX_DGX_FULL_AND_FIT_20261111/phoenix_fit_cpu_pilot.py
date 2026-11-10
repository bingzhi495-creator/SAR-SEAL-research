"""Run fixed-candidate Phoenix CPU refinement on two HRSID FIT images."""
from argparse import ArgumentParser
from hashlib import sha256
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import re
import runpy
import sys
from time import perf_counter

import numpy as np


AUTHOR_CHECK_SHA256 = '3a1004b498be517d0c113ac9e5e02b77211326e7e88bab79937c564081141085'
AUTHOR_COMMIT = '13ee3996ef3d0292fab144316cc8e03b69c0ecfa'
REQUEST_STATUS = 'PHOENIX_HRSID_FIT114_FIXED_CANDIDATE_REQUEST_V1'
RULES = ('stock_coco', 'all_nonempty')
PROVENANCE_KEYS = {
    'selection_salt', 'fit_bundle_sha256', 'cohort_sha256', 'predictions_sha256',
    'checkpoint_sha256', 'original_export_protocol', 'original_export_terminal_sha256',
    'original_export_acceptance_sha256',
}
ROW_KEYS = {'native_rank', 'query_id', 'score', 'category_id', 'segmentation'}
IMAGE_KEYS = {'image_id', 'image_file', 'encoded_sha256', 'decoded_rgb_sha256', 'candidate_rows'}
MASK_SIZE = (800, 800)
CHUNK_SIZE = 2
REFINE_ITERS = 5
SEED = 5202


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest_file(path):
    digest = sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def is_digest(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def decode_request(request_path, request_sha256):
    from PIL import Image
    from pycocotools import mask as coco_mask

    require(is_digest(request_sha256), 'Expected request SHA-256')
    raw = request_path.read_bytes()
    require(sha256(raw).hexdigest() == request_sha256, 'Request SHA-256 differs')
    request = json.loads(raw)
    require(isinstance(request, dict) and set(request) == {'status', 'provenance', 'images'}
            and request['status'] == REQUEST_STATUS, 'Request schema differs')
    provenance = request['provenance']
    require(isinstance(provenance, dict) and set(provenance) == PROVENANCE_KEYS
            and all(is_digest(value) if key.endswith('sha256') else isinstance(value, str) and bool(value)
                    for key, value in provenance.items()), 'Request provenance differs')
    records = request['images']
    require(isinstance(records, list) and len(records) == 2, 'Exactly two FIT images required')
    images = []
    image_ids = set()
    for ordinal, record in enumerate(records, start=1):
        require(isinstance(record, dict) and set(record) == IMAGE_KEYS, 'Image schema differs')
        image_id = record['image_id']
        require(type(image_id) is int and image_id >= 0 and image_id not in image_ids, 'Image identity differs')
        image_ids.add(image_id)
        basename = f'image_{ordinal:03}.png'
        require(record['image_file'] == basename and is_digest(record['encoded_sha256'])
                and is_digest(record['decoded_rgb_sha256']), 'Image binding differs')
        path = request_path.parent / basename
        require(path.resolve(strict=True) == path and digest_file(path) == record['encoded_sha256'],
                'Encoded image differs')
        with Image.open(path) as image_file:
            require(image_file.format == 'PNG' and image_file.size == MASK_SIZE[::-1], 'FIT image geometry differs')
            rgb = np.array(image_file.convert('RGB'), dtype=np.uint8)
        require(rgb.shape == (*MASK_SIZE, 3)
                and sha256(rgb.tobytes()).hexdigest() == record['decoded_rgb_sha256'], 'Decoded RGB differs')
        rows = record['candidate_rows']
        require(isinstance(rows, list) and 1 <= len(rows) <= 100, 'Expected native candidate rows')
        previous_rank = 0
        query_ids = set()
        masks = []
        for row in rows:
            require(isinstance(row, dict) and set(row) == ROW_KEYS, 'Candidate schema differs')
            rank, query, score = row['native_rank'], row['query_id'], row['score']
            require(type(rank) is int and previous_rank < rank <= 100 and type(query) is int
                    and 0 <= query < 164 and query not in query_ids and type(score) is float
                    and math.isfinite(score) and 0.0 <= score <= 1.0 and type(row['category_id']) is int
                    and row['category_id'] == 1, 'Candidate identity or native score differs')
            previous_rank = rank
            query_ids.add(query)
            segmentation = row['segmentation']
            require(isinstance(segmentation, dict) and set(segmentation) == {'size', 'counts'}
                    and segmentation['size'] == list(MASK_SIZE) and isinstance(segmentation['counts'], str),
                    'Native RLE schema differs')
            mask = coco_mask.decode({'size': list(MASK_SIZE), 'counts': segmentation['counts'].encode('ascii')})
            require(mask.shape == MASK_SIZE and bool(np.all((mask == 0) | (mask == 1))), 'Native RLE grid differs')
            masks.append(mask.astype(bool))
        images.append((image_id, rgb, rows, masks))
    return images


def refine_image(refiner, rows, masks, rule, encode_mask):
    require(rule in RULES and len(rows) == len(masks), 'Unknown rule or mask length differs')
    areas = [int(mask.sum()) for mask in masks]
    eligible = [index for index, (mask, area) in enumerate(zip(masks, areas))
                if area > 0 and (rule == 'all_nonempty' or 100 * area > mask.size)]
    variant = [dict(row) for row in rows]
    empty_input = areas.count(0)
    empty_after_refinement = changed_rows = added_pixels = removed_pixels = 0
    start = perf_counter()
    for offset in range(0, len(eligible), CHUNK_SIZE):
        chunk = eligible[offset:offset + CHUNK_SIZE]
        output = refiner.refine_batch([masks[index] for index in chunk], refine_iters=REFINE_ITERS,
                                      smooth_kernel=1, binarize_input=True, multimask_output=True,
                                      return_logits=False, max_batch=CHUNK_SIZE)
        require(len(output) == len(chunk), 'Author batch returned a different row count')
        for index, mask in zip(chunk, output):
            original = masks[index]
            require(isinstance(mask, np.ndarray) and mask.shape == original.shape and mask.dtype == np.bool_,
                    'Author output is not an original-grid binary mask')
            added = int(np.count_nonzero(mask & ~original))
            removed = int(np.count_nonzero(original & ~mask))
            added_pixels += added
            removed_pixels += removed
            empty_after_refinement += not mask.any()
            if added or removed:
                changed_rows += 1
                variant[index]['segmentation'] = encode_mask(mask)
    summary = dict(candidates=len(rows), eligible=len(eligible), unchanged_rows=len(rows) - changed_rows,
                   changed_rows=changed_rows, empty_input=empty_input,
                   empty_after_refinement=empty_after_refinement,
                   empty_output=empty_input + empty_after_refinement,
                   added_pixels=added_pixels, removed_pixels=removed_pixels,
                   changed_pixels=added_pixels + removed_pixels, seconds=perf_counter() - start)
    return variant, summary


def write_json(path, data):
    raw = (json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf-8')
    with path.open('xb') as target:
        target.write(raw)
    return sha256(raw).hexdigest()


def main():
    parser = ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--author-check-source', type=Path, required=True)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--request-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    paths = (args.source_root, args.author_check_source, args.request, args.output.parent)
    require(all(path.is_absolute() and path.resolve(strict=True) == path for path in paths)
            and args.output.is_absolute() and not os.path.lexists(args.output), 'Canonical inputs and fresh output required')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '-1', 'GPU must be hidden for CPU pilot')
    require(sys.dont_write_bytecode, 'Run with -B to preserve source directories')
    start = perf_counter()
    require(digest_file(args.author_check_source) == AUTHOR_CHECK_SHA256, 'Author smoke source SHA-256 differs')
    author_check = runpy.run_path(str(args.author_check_source))
    sys.addaudithook(author_check['deny_network'])
    os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '1'
    author_check['verify_inputs'](args.source_root)
    images = decode_request(args.request, args.request_sha256)
    input_seconds = perf_counter() - start

    import resource
    import torch
    import FastGeodis
    from pycocotools import mask as coco_mask

    require(importlib.metadata.version('FastGeodis') == '1.0.5'
            and callable(FastGeodis.generalised_geodesic2d), 'Expected native FastGeodis 1.0.5')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    sys.path.insert(0, str(args.source_root))
    import phoenix

    require(Path(phoenix.__file__).resolve() == args.source_root / 'phoenix/__init__.py', 'Wrong author source import')
    checkpoint = args.source_root / 'phoenix_efficientvit_xl1.pt'
    require(not torch.serialization.get_unsafe_globals_in_checkpoint(checkpoint), 'Expected tensor-only official checkpoint')
    tick = perf_counter()
    model = phoenix.build_phoenix(checkpoint=str(checkpoint), encoder='efficientvit_xl1', device='cpu', strict=True)
    refiner = phoenix.PhoenixRefiner(model)
    load_seconds = perf_counter() - tick
    encode_seconds = 0.0
    variants = {rule: [] for rule in RULES}
    results = {rule: [] for rule in RULES}

    def encode_mask(mask):
        rle = coco_mask.encode(np.asfortranarray(mask, dtype=np.uint8))
        return {'size': [int(value) for value in rle['size']], 'counts': rle['counts'].decode('ascii')}

    with torch.inference_mode():
        for image_id, rgb, rows, masks in images:
            tick = perf_counter()
            refiner.set_image(rgb)
            features = refiner.predictor.get_image_embedding()
            require(tuple(features.shape) == (1, 256, 64, 64) and features.device.type == 'cpu'
                    and features.dtype == torch.float32 and bool(torch.isfinite(features).all())
                    and refiner.predictor.original_size == MASK_SIZE, 'Author CPU encoder geometry differs')
            encode_seconds += perf_counter() - tick
            for rule in RULES:
                random.seed(SEED)
                np.random.seed(SEED)
                torch.random.default_generator.manual_seed(SEED)
                variant, metrics = refine_image(refiner, rows, masks, rule, encode_mask)
                variants[rule].append(dict(image_id=image_id, candidate_rows=variant))
                results[rule].append(metrics)
    require(not torch.cuda.is_initialized(), 'CUDA initialized during CPU pilot')
    args.output.mkdir()
    variant_digests = {}
    for rule in RULES:
        variant_digests[rule] = write_json(args.output / f'{rule.upper()}_PRIVATE.json', {
            'rule': rule, 'request_sha256': args.request_sha256, 'images': variants[rule]})
    report = dict(status='PHOENIX_HRSID_FIT114_FIXED_CANDIDATE_CPU_PILOT_COMPLETE',
                  author_commit=AUTHOR_COMMIT, request_sha256=args.request_sha256, candidate_variants_sha256=variant_digests,
                  image_count=len(images), per_rule={rule: {key: sum(image[key] for image in results[rule])
                              for key in results[rule][0]} for rule in RULES},
                  encode_seconds=encode_seconds, load_seconds=load_seconds, input_seconds=input_seconds,
                  wall_seconds=perf_counter() - start, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                  fastgeodis='1.0.5', pytorch=str(torch.__version__), numpy=np.__version__, device='cpu', cpu_threads=2,
                  chunk_size=CHUNK_SIZE, refine_iters=REFINE_ITERS, branch_seed=SEED, cuda_initialized=False,
                  fit_images_only=True, ground_truth_accessed=False, AP=False)
    result_sha256 = write_json(args.output / 'PILOT_SAFE.json', report)
    print(json.dumps(dict(status=report['status'], result_sha256=result_sha256)), flush=True)


if __name__ == '__main__':
    main()
