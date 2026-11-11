"""Independently verify closed FIT114 sparse logits and historical candidate parity."""
from argparse import ArgumentParser
from collections import Counter
from hashlib import sha256
from heapq import nsmallest
import json
from pathlib import Path
import re
from time import perf_counter

import numpy as np


CAPTURE_STATUS = 'P316_HRSID_FIT114_SPARSE_SIGNED_LOGIT_CAPTURE_COMPLETE'
COHORT_SHA = '70e9a0e8fb30f5ba70b48b9067c1b95d07bfec23d6b248d92c5983727acceb6e'
QUERY_SHA = '23c4ab97a3459f77936a95d0d09965db339e7af54e2a4a1b9d0d9f787f8690f7'
ACCEPT_SHA = 'ba1d8a0df0678f7063b73998ddb487941160898a8758d6c02ea6469595768968'
TERM_SHA = '329f4fb71bcaa5fc0d11d684c52d28d0ff605be837d43eb95ffc138a8c5c969c'
CHECKPOINT_SHA = '7824943afc7856cceb247b281f0c2d71dc087d96906d0aed0f28b6f0e34840e1'
OFFSETS = ((0, 0), (-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
SALT = b'P316-HRSID-FIT114-SIGNED-BAND-v1\0'
CAP = 32
SIZE = 800


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pinned(path, digest):
    raw = path.read_bytes()
    require(sha256(raw).hexdigest() == digest, 'Pinned receipt or historical input digest differs')
    return raw


def verify_points(mask, points, encoded_sha, query_id):
    require(isinstance(points, list) and len(points) <= 2 * CAP, 'Sparse candidate point cap exceeded')
    height, width = mask.shape
    foreground_y = np.flatnonzero(mask.any(axis=1))
    foreground_x = np.flatnonzero(mask.any(axis=0))
    if not len(foreground_y):
        require(not points, 'Empty prediction has sparse points')
        return 0, 0
    top = max(0, int(foreground_y[0]) - 1)
    bottom = min(height, int(foreground_y[-1]) + 2)
    left = max(0, int(foreground_x[0]) - 1)
    right = min(width, int(foreground_x[-1]) + 2)
    area = mask[top:bottom, left:right]
    windows = np.lib.stride_tricks.sliding_window_view(np.pad(area, 1), (3, 3))
    inner = area & ~windows.all(axis=(2, 3))
    outer = ~area & windows.any(axis=(2, 3))
    prefix = SALT + bytes.fromhex(encoded_sha) + query_id.to_bytes(2, 'little')
    expected = []
    for inside, band in ((True, inner), (False, outer)):
        coords = [(int(y) + top, int(x) + left) for y, x in np.argwhere(band)]
        if len(coords) > CAP:
            side = prefix + bytes((inside,))
            def rank(point):
                y, x = point
                return sha256(side + (y * width + x).to_bytes(4, 'little')).digest(), point
            coords = sorted(nsmallest(CAP, coords, key=rank))
        expected.extend((y, x, inside) for y, x in coords)
    require(len(points) == len(expected), 'Prediction-only boundary population differs')
    for point, (y, x, inside) in zip(points, expected, strict=True):
        require(isinstance(point, dict) and set(point) == {'y', 'x', 'inside', 'logits'}
                and type(point['y']) is int and point['y'] == y and type(point['x']) is int
                and point['x'] == x and point['inside'] is inside, 'One-pixel boundary sample differs')
        values = point['logits']
        require(isinstance(values, list) and len(values) == len(OFFSETS)
                and all(type(v) in (int, float) and np.isfinite(v) for v in values), 'Sparse logit is malformed')
        require((values[0] > 0) is inside, 'Center logit sign differs from original mask')
        for value, (dy, dx) in zip(values[1:], OFFSETS[1:], strict=True):
            yy, xx = min(height - 1, max(0, y + dy)), min(width - 1, max(0, x + dx))
            require((value > 0) == mask[yy, xx], 'Neighbor logit sign differs from original mask')
    inner_count = min(CAP, int(inner.sum()))
    return inner_count, len(expected) - inner_count


def compare_rows(historical, candidates):
    historical_queries = historical['native_ids']
    previous = {query: rank for query, rank in zip(historical_queries, historical['native_ranks'], strict=True)}
    current_queries = [row['query_id'] for row in candidates]
    overlap = [row for row in candidates if row['query_id'] in previous]
    same_order = historical_queries == current_queries
    ranks = scores = rles = 0
    max_score_difference = 0.0
    for row in overlap:
        query = row['query_id']
        ranks += row['native_rank'] != previous[query]
        score_difference = abs(row['score'] - historical['ship_scores'][query])
        scores += score_difference != 0
        max_score_difference = max(max_score_difference, score_difference)
        rles += row['segmentation'] != historical['all_rles'][query]
    return dict(historical_rows=len(historical_queries), cpu_rows=len(candidates), common_rows=len(overlap),
                missing_old_rows=len(historical_queries) - len(overlap), new_cpu_rows=len(candidates) - len(overlap),
                ordered_query_list_exact=int(same_order), rank_different=ranks, score_different=scores,
                rle_different=rles, complete_row_stream_exact=int(same_order and not (ranks or scores or rles))), max_score_difference


def readback(capture_dir, report_sha, original):
    from pycocotools import mask as coco

    report = json.loads(pinned(capture_dir / 'CAPTURE_SAFE.json', report_sha))
    require(report['status'] == CAPTURE_STATUS and report['device'] == 'cpu' and report['FIT'] is True
            and report['cpu_native_signed_raster_exact'] is True
            and all(report[key] is False for key in ('training', 'DEV', 'TEST', 'AP', 'GT_opened'))
            and (report['images'], report['core_images'], report['stress_images'], report['forward_calls'],
                 report['optimizer_updates']) == (114, 109, 5, 114, 0)
            and tuple(map(tuple, report['logit_offsets_yx'])) == OFFSETS and report['max_points_per_side'] == CAP
            and report['boundary_radius_original_pixels'] == 1 and report['boundary_distance'] == 'Chebyshev'
            and report['edge_neighbor'] == 'clamp_to_original_grid' and report['point_salt'] == SALT[:-1].decode()
            and report['historical_cohort_sha256'] == COHORT_SHA and report['historical_predictions_sha256'] == QUERY_SHA
            and report['historical_export_terminal_sha256'] == TERM_SHA
            and report['source_checkpoint_sha256'] == CHECKPOINT_SHA, 'Capture identity or protocol differs')
    acceptance = json.loads(pinned(original / 'ROOT_CLOSED_EXPORT_PRIVATE.json', ACCEPT_SHA))
    terminal = json.loads(pinned(original / 'EXPORT_CONTROL' / 'TERMINAL_PRIVATE.json', TERM_SHA))
    require(acceptance['terminal'] == terminal and acceptance['resource_closed'] is True
            and acceptance['status'] == 'ORIGINAL_GT_FREE_FIT_Q164_EXPORT_ACCEPTED'
            and terminal['original_OS_wait'] is True and type(terminal['exit']) is int and terminal['exit'] == 0
            and terminal['guard'] is None, 'P192 original export closure differs')
    cohort = json.loads(pinned(original / 'COHORT_PRIVATE.json', COHORT_SHA))['images']
    old = [json.loads(line) for line in pinned(original / 'EXPORT_ACTUAL_PRIVATE' / 'QUERIES_PRIVATE.jsonl', QUERY_SHA).splitlines()]
    require(len(cohort) == len(old) == 114 and len({row['image_id'] for row in cohort}) == 114
            and sum(row['core'] for row in cohort) == len({row['group'] for row in cohort if row['core']}) == 109
            and sum(row['stress'] for row in cohort) == 5 and all(row['core'] != row['stress'] for row in cohort)
            and sum(len(block['native_ids']) for block in old) == 10572, 'Original FIT114 population differs')
    digest = sha256()
    stats = Counter(images=0, empty_candidates=0, inner_points=0, outer_points=0)
    parity = Counter()
    total_bytes = 0
    max_score_difference = 0.0
    with (capture_dir / 'CANDIDATES_PRIVATE.jsonl').open('rb') as stream:
        for line, planned, historical in zip(stream, cohort, old, strict=True):
            digest.update(line)
            total_bytes += len(line)
            block = json.loads(line)
            metadata = ('image_id', 'group', 'encoded_sha256', 'core', 'stress')
            require(set(block) == {*metadata, 'candidates'}
                    and all(block[k] == planned[k] for k in metadata)
                    and historical['image_id'] == planned['image_id'], 'Captured image population/order differs')
            rows = block['candidates']
            require(isinstance(rows, list) and len(rows) <= 100, 'Candidate row cap exceeded')
            last_rank, queries = 0, set()
            for row in rows:
                require(set(row) == {'native_rank', 'query_id', 'score', 'category_id', 'segmentation', 'points'},
                        'CPU native candidate schema differs')
                rank, query, score = row['native_rank'], row['query_id'], row['score']
                require(type(rank) is int and last_rank < rank <= 100 and type(query) is int and 0 <= query < 164
                        and query not in queries and type(score) is float and 0 <= score <= 1
                        and type(row['category_id']) is int and row['category_id'] == 1, 'CPU native candidate identity differs')
                last_rank = rank
                queries.add(query)
                rle = row['segmentation']
                require(set(rle) == {'size', 'counts'} and rle['size'] == [SIZE, SIZE]
                        and isinstance(rle['counts'], str) and rle['counts'].isascii() and bool(rle['counts']),
                        'Original-grid candidate RLE differs')
                mask = coco.decode({'size': rle['size'], 'counts': rle['counts'].encode('ascii')}).astype(bool)
                require(mask.shape == (SIZE, SIZE), 'Decoded candidate dimensions differ')
                inner, outer = verify_points(mask, row['points'], planned['encoded_sha256'], query)
                stats['inner_points'] += inner
                stats['outer_points'] += outer
                stats['empty_candidates'] += not mask.any()
            counters, max_delta = compare_rows(historical, rows)
            parity.update(counters)
            max_score_difference = max(max_score_difference, max_delta)
            stats['images'] += 1
    stream_sha = digest.hexdigest()
    require(stream_sha == report['candidate_stream_sha256'] and total_bytes == report['candidate_stream_bytes'],
            'Closed candidate stream SHA-256 or byte length differs')
    require(all(report['counters'][key] == value for key, value in stats.items())
            and report['counters']['stock_sign_verified_rows'] == 11400
            and dict(parity) == report['historical_gpu_vs_current_cpu']
            and report['historical_gpu_parity_exact'] == (parity['complete_row_stream_exact'] == 114),
            'Independent population or GPU/CPU parity arithmetic differs')
    points = stats['inner_points'] + stats['outer_points']
    return dict(status='P316_FIT114_SPARSE_LOGIT_READBACK_COMPLETE', capture_report_sha256=report_sha,
                candidate_stream_sha256=stream_sha, candidate_stream_bytes=total_bytes,
                counts=dict(stats), core_images=109, stress_images=5, historical_gpu_vs_current_cpu=dict(parity),
                historical_cohort_sha256=COHORT_SHA, historical_predictions_sha256=QUERY_SHA,
                max_historical_common_query_abs_score_difference=max_score_difference,
                sampled_center_sign_checks=points, sampled_clamped_neighbor_sign_checks=8 * points,
                full_cpu_signed_raster_rows_reported_by_capture=11400, mask_neighbor_points_verified=True,
                training=False, FIT=True, DEV=False, TEST=False, AP=False, GT_opened=False)


def main():
    parser = ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--capture-dir', required=True, type=Path)
    parser.add_argument('--capture-report-sha256', required=True)
    parser.add_argument('--original', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    require(re.fullmatch(r'[0-9a-f]{64}', args.capture_report_sha256) is not None, 'Pass original report SHA-256')
    require(not args.output.exists() and not args.output.is_symlink(), 'Use a fresh output path')
    start = perf_counter()
    result = readback(args.capture_dir, args.capture_report_sha256, args.original)
    result['wall_seconds'] = perf_counter() - start
    raw = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    with args.output.open('xb') as stream:
        stream.write(raw)
    print(json.dumps(dict(status=result['status'], result_sha256=sha256(raw).hexdigest())), flush=True)


if __name__ == '__main__':
    main()
