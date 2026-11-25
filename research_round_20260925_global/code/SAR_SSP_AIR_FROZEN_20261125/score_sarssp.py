"""Score closed SAR-SSP B5 arms against the author's native binary ship instances."""

import argparse
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from copy import deepcopy
from hashlib import sha256
from io import BytesIO, StringIO
import json
import math
from pathlib import Path
from time import perf_counter
from types import ModuleType
from zipfile import ZipFile

import numpy as np
from PIL import Image
from pycocotools import mask as codec
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


ARCHIVE_SHA = '23505cd2de8faa3afa9b4cb329397c486724d02cc24482fd1883ba70a01f77a5'
PROTOCOL_SHA = 'f8f6149cff43d5166ddb0cbdb9bea0c0f4957a50b74ef53692c5982494c83947'
EDIT_PROTOCOL_SHA = '6618b0eb2c30a2e3cd090ba55d57eaad39ddfb8b8e680eeb650dc0ff27beda87'
CAPTURE_SHA = 'c40a133d62fafdecb9736d17abbd0ddcbf31ec42672d61eb0cbfc5d5df0b643e'
READBACK_SHA = '5788cd43fdc1e404cf6cc3c611dec47bb8ed30c55b926b22aabb6d3ec80e3711'
HEAD_SHA = '3a60f658149f4262451cd9118902bbe52dc83345d9fbcd23973a11dc473fcdf1'
CHECKPOINT_SHA = '011bd30ed056c871be1d61e14d71b3b2b258330a9d20cf2f164ca4a7c4f23c11'
EVALUATOR_SHA = 'dd8e435976d2331146c8f83d70f368ee4e0954c05520c54fa9a3b1205838d40f'
EVALUATOR_PATH = 'PSEG_TEST232_DFINE_FROZEN_SCORE_20261121/score_test232.py'
HW = [500, 500]
ARMS = {
    'fixed': 'distance_paired',
    'free_matched_count': 'free_matched_count',
    'free_natural': 'free_natural',
    'native_matched_count': 'native_matched_count',
    'attraction_dynamic': 'dynamic_attraction',
    'attraction_static_at_dynamic_count': 'static_at_dynamic_count',
    'learned_at_attraction_count': 'learned_at_attraction_count',
}
FAMILIES = (
    ('distance_paired', 'free_matched_count'),
    ('free_natural', 'native_matched_count'),
    ('dynamic_attraction', 'static_at_dynamic_count', 'learned_at_attraction_count'),
)
PRIMARY = ('free_matched_count', 'distance_paired')
SECONDARY = ('learned_at_attraction_count', 'dynamic_attraction')
COMPARISONS = (PRIMARY, SECONDARY, ('learned_at_attraction_count', 'static_at_dynamic_count'),
               ('dynamic_attraction', 'static_at_dynamic_count'), ('free_natural', 'native_matched_count'))
COMPARISONS += tuple((arm, 'native') for arm in ARMS.values())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pinned(path, expected):
    content = path.read_bytes()
    require(sha256(content).hexdigest() == expected, 'Pinned file differs: ' + path.name)
    return content


def read_stream(path, expected):
    return [json.loads(line) for line in pinned(path, expected).splitlines()]


def compare_name(pair):
    return '__vs__'.join(pair)


def rle_area(rle, hw):
    require(set(rle) == {'size', 'counts'} and rle['size'] == hw
            and isinstance(rle['counts'], str) and rle['counts'].isascii(), 'Original RLE canvas and schema')
    area = int(codec.area(rle))
    require(0 <= area <= hw[0] * hw[1], 'Valid RLE area')
    return area


def signed_counts(native, edited, native_area):
    retained = int(codec.area(codec.merge([native, edited], intersect=True)))
    return int(codec.area(edited)) - retained, native_area - retained


def stream_records(predictions, audits, edits, hw):
    records, names = {}, {}
    edited_counts = Counter()
    for pred, audit, edit in zip(predictions, audits, edits, strict=True):
        image_id = pred['image_id']
        require(type(image_id) is int and image_id == audit['image_id'] == edit['image_id']
                and (not records or image_id > next(reversed(records))), 'Full streams in exact increasing image order')
        require(audit['original_hw'] == hw, 'Original image canvas')
        rows, metadata, ledger = pred['coco'], audit['candidates'], edit['changes']
        ranks = [entry['native_rank'] for entry in metadata]
        require(len(rows) == len(metadata) <= 100 and ranks == audit['native_ranks'] == pred['native_ranks']
                and ranks == sorted(set(ranks)), 'All original mapped candidates in native rank order')
        require(set(ledger) <= {str(position) for position in range(1, len(rows) + 1)}, 'Known original edit positions')
        names[image_id] = (audit['source_image_name'], audit['image_sha256'])
        candidates = []
        previous_score = math.inf
        query_ids = set()
        for position, (row, candidate) in enumerate(zip(rows, metadata, strict=True), 1):
            score = row['score']
            query_id = candidate['query_id']
            require(candidate['position'] == position and row['image_id'] == image_id
                    and row['category_id'] == 1 and math.isfinite(score) and 0 <= score <= 1
                    and score == candidate['score'] and score <= previous_score and query_id not in query_ids,
                    'Exact original score, ordering, class, query and candidate position')
            previous_score = score
            query_ids.add(query_id)
            stock = row['segmentation']
            area = rle_area(stock, hw)
            require(area == candidate['mask_area'], 'Native RLE area matches capture audit')
            edited_counts['mapped_candidates'] += 1
            eligible = 32 <= area < 1024
            edited_counts['eligible_planes'] += int(eligible)
            changes = ledger.get(str(position), {})
            require(set(changes) <= ARMS.keys() and (not changes or eligible), 'Original eligible support and edit arms')
            masks, counts = {}, {}
            for key, value in changes.items():
                arm = ARMS[key]
                rle, swaps = value['rle'], value['swaps']
                require(set(value) == {'rle', 'swaps'} and type(swaps) is int
                        and 0 < swaps <= min(8, area // 10) and rle_area(rle, hw) == area
                        and signed_counts(stock, rle, area) == (swaps, swaps), 'Exact signed counts from RLE pixels')
                masks[arm], counts[arm] = rle, swaps
                edited_counts[key + '_swaps'] += swaps
            for family in FAMILIES:
                require(len({counts.get(arm, 0) for arm in family}) == 1, 'Equal actual signed edits inside each family')
            candidates.append(dict(native=stock, score=score, masks=masks, counts=counts))
        records[image_id] = candidates
    return records, names, dict(edited_counts)


def qualified_records(args):
    root = args.capture_run
    result_name = 'RESULT_SAFE.json'
    edit = json.loads(pinned(root / 'verify-edits' / result_name, args.edit_receipt_sha))
    terminal = json.loads(pinned(root / 'observed_verify-edits/TERMINAL.json', args.edit_terminal_sha))
    require(edit['status'] == 'SAR_SSP_VAL232_B5_EDITS_VERIFIED_V1'
            and terminal['phase'] == edit['phase'] == 'verify-edits' and terminal['actual_exit'] == 0
            and terminal['output_footer_verified'] is True
            and terminal['result_sha256'] == args.edit_receipt_sha
            and terminal['worker_sha256'] == edit['worker_sha256']
            and edit['target_labels_opened'] is False and edit['optimizer_updates'] == edit['detector_forwards'] == 0
            and edit['exact_support_xor_area'] is True and edit['exact_replayed_policy_choices'] is True
            and edit['protocol_sha256'] == EDIT_PROTOCOL_SHA and edit['point_head_sha256'] == HEAD_SHA
            and edit['capture_sha256'] == CAPTURE_SHA and edit['capture_readback_sha256'] == READBACK_SHA,
            'Independently observed exact frozen edits')
    original = json.loads(pinned(root / 'capture' / result_name, CAPTURE_SHA))
    readback = json.loads(pinned(root / 'verify-capture' / result_name, READBACK_SHA))
    require(original['archive_sha256'] == readback['archive_sha256'] == ARCHIVE_SHA
            and original['checkpoint_sha256'] == readback['checkpoint_sha256'] == CHECKPOINT_SHA
            and original['images'] == readback['images'] == edit['counts']['images'] == 232
            and readback['exact_native_parity'] is True
            and original['target_labels_opened'] is False and readback['target_labels_opened'] is False
            and original['model_forward_calls'] == 232 and readback['model_forward_calls'] == 0
            and original['intake_sha256'] == readback['intake_sha256'] == edit['intake_sha256']
            and all(original[key] == readback[key] for key in ('predictions_sha256', 'audit_sha256')),
            'Frozen native capture and independent parity')
    prepared = json.loads(pinned(root / 'prepare-edits' / result_name, edit['prepare_sha256']))
    require(prepared['edits_sha256'] == edit['edits_sha256'] and prepared['counts'] == edit['counts'],
            'Prepared and independently replayed same ledger')
    capture = root / 'capture'
    predictions = read_stream(capture / 'PREDICTIONS_PRIVATE.jsonl', original['predictions_sha256'])
    audits = read_stream(capture / 'CANDIDATE_AUDIT_PRIVATE.jsonl', original['audit_sha256'])
    edits = read_stream(root / 'prepare-edits/EDITS_PRIVATE.jsonl', edit['edits_sha256'])
    records, names, counts = stream_records(predictions, audits, edits, HW)
    require(len(records) == 232 and all(edit['counts'].get(key, 0) == value for key, value in counts.items())
            and all(edit['counts'].get(key + '_swaps', 0) == counts.get(key + '_swaps', 0) for key in ARMS),
            'Full validated roster and aggregate endpoint parity')
    return records, names, counts, original, edit


def exact_ground_truth(package, names):
    manifest = json.loads(package.read('datasets/DATASET_INDEX.json'))
    indexed = {row['path']: (row['bytes'], row['sha256']) for row in manifest['files']}

    def author_bytes(path):
        raw = package.read('datasets/' + path)
        require((len(raw), sha256(raw).hexdigest()) == indexed[path], 'Exact indexed author archive member')
        return raw

    annotation_root = 'SAR-SSP/annotations/panoptic_val2017'
    author = json.loads(author_bytes(annotation_root + '.json'))
    images = {row['id']: row for row in author['images']}
    annotated = {row['image_id']: row for row in author['annotations']}
    categories = {row['id']: row for row in author['categories']}
    require(set(images) == set(annotated) == set(names) and len(images) == len(author['images'])
            == len(author['annotations']) and set(categories) == {0, 1, 2}
            and categories[0]['name'] == 'ship' and categories[0]['isthing'] == 1
            and all(categories[k]['isthing'] == 0 for k in (1, 2)), 'Author exact roster and ship category')
    result = dict(images=[], annotations=[], categories=[dict(id=0, name='ship')], info={})
    groups, edge = {}, set()
    for key, (name, image_sha) in names.items():
        image, row = images[key], annotated[key]
        require(name == image['file_name'] and indexed['SAR-SSP/val2017/' + name][1] == image_sha
                and [image['height'], image['width']] == HW
                and all(image[k] == row[k] for k in ('source_image_no', 'tile_x', 'tile_y')),
                'Captured JPEG and exact author provenance')
        result['images'].append(dict(id=key, width=image['width'], height=image['height']))
        groups[key] = image['source_image_no']
        with Image.open(BytesIO(author_bytes(annotation_root + '/' + row['file_name']))) as png:
            require(png.format == 'PNG' and png.size == tuple(HW[::-1]), 'Native panoptic PNG canvas')
            rgb = np.asarray(png.convert('RGB'), dtype=np.int32)
        ids = rgb[:, :, 0] + 256 * rgb[:, :, 1] + 65536 * rgb[:, :, 2]
        color_ids, pixels = np.unique(ids, return_counts=True)
        observed = dict(zip(color_ids.tolist(), pixels.tolist(), strict=True))
        segments = row['segments_info']
        require(set(observed) - {0} == {s['id'] for s in segments}
                and len(segments) == len({s['id'] for s in segments}), 'Exact author panoptic segment IDs')
        for segment in segments:
            sid, cat = segment['id'], segment['category_id']
            require(cat in categories and observed[sid] == segment['area'], 'Exact author raster area and category')
            if cat != 0:
                continue
            require(segment['iscrowd'] == 0, 'Expected individual noncrowd ship hulls')
            binary = ids == sid
            rle = codec.encode(np.asfortranarray(binary, dtype=np.uint8))
            rle = dict(size=list(rle['size']), counts=rle['counts'].decode('ascii'))
            require(codec.toBbox(rle).tolist() == segment['bbox'], 'Exact author hull bounding box')
            index = len(result['annotations']) + 1
            if binary[0].any() or binary[-1].any() or binary[:, 0].any() or binary[:, -1].any():
                edge.add(index)
            result['annotations'].append(dict(id=index, image_id=key, category_id=0, iscrowd=0,
                                              area=segment['area'], bbox=segment['bbox'], segmentation=rle))
    positive_images = {row['image_id'] for row in result['annotations']}
    require((len(result['annotations']), len(positive_images), len(edge), len(set(groups.values())),
             len({groups[i] for i in positive_images})) == (139, 48, 48, 31, 23), 'Complete prelocked author truth')
    return result, groups, edge


def coco_rows(records, arm):
    return [dict(image_id=image_id, category_id=0, score=row['score'],
                 segmentation=row['masks'].get(arm, row['native']))
            for image_id, candidates in records.items() for row in candidates]


def native_matches(truth, records, expected_ap):
    rows = coco_rows(records, 'native')
    positions = {index: (image_id, position) for index, (image_id, position) in enumerate(
        ((image_id, position) for image_id, candidates in records.items()
         for position in range(1, len(candidates) + 1)), 1)}
    with redirect_stdout(StringIO()):
        ref = COCO()
        ref.dataset = deepcopy(truth)
        ref.createIndex()
        det = ref.loadRes(deepcopy(rows)) if rows else COCO()
        ev = COCOeval(ref, det, 'segm')
        ev.params.imgIds = list(records)
        ev.params.catIds = [0]
        ev.params.maxDets = [1, 10, 100]
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    require(float(ev.stats[0]) == expected_ap, 'Exact native matching parity with the reused evaluator')
    threshold, = np.flatnonzero(np.isclose(ev.params.iouThrs, 0.50))
    matches = {}
    truth_count = 0
    for row in ev.evalImgs:
        if row is None or row['aRng'] != ev.params.areaRng[0] or row['maxDet'] != 100:
            continue
        for tid, did, ignored in zip(row['gtIds'], row['gtMatches'][threshold], row['gtIgnore'], strict=True):
            require(not ignored, 'All noncrowd author hulls entered native matching')
            truth_count += 1
            if did:
                image_id, position = positions[int(did)]
                require(image_id == row['image_id'] and tid not in matches, 'Exact stock COCO candidate pair')
                matches[tid] = (image_id, position)
    require(truth_count == len(truth['annotations']), 'All author hulls accounted for in native matching')
    return matches


def hull_measures(rle, truth, truth_area):
    predicted = int(codec.area(rle))
    hit = int(codec.area(codec.merge([rle, truth], intersect=True)))
    return (100 * hit / (predicted + truth_area - hit), 100 * (predicted - hit) / truth_area,
            100 * (truth_area - hit) / truth_area)


def fixed_geometry(annotations, groups, matches, records):
    names = ['native', *ARMS.values()]
    values = defaultdict(list)
    primary_hulls = 0
    primary_groups = set()
    secondary_hulls = 0
    secondary_groups = set()
    matched = 0
    matched_groups = set()
    for annotation in annotations:
        aid, image_id = annotation['id'], annotation['image_id']
        group = groups[image_id]
        if aid not in matches:
            values[group].append([(0., 0., 100.)] * len(names))
            continue
        iid, position = matches[aid]
        require(iid == image_id, 'Fixed match belongs to its author hull image')
        candidate = records[image_id][position - 1]
        native = candidate['native']
        masks = {name: candidate['masks'].get(name, native) for name in names}
        for pair, changed in ((PRIMARY, primary_groups), (SECONDARY, secondary_groups)):
            left, right = (masks[arm] for arm in pair)
            if left == right:
                continue
            if signed_counts(right, left, int(codec.area(right))) != (0, 0):
                changed.add(group)
                if pair == PRIMARY:
                    primary_hulls += 1
                else:
                    secondary_hulls += 1
        matched += 1
        matched_groups.add(group)
        values[group].append([hull_measures(masks[name], annotation['segmentation'], annotation['area'])
                              for name in names])
    parent_means = np.asarray([np.mean(values[group], axis=0) for group in sorted(values)])
    require(len(parent_means) > 0, 'Paired geometry requires at least one ship-bearing source group')
    parent_draws = np.random.Generator(np.random.PCG64(5217)).integers(
        len(parent_means), size=(4096, len(parent_means)))
    fields = ('iou_points', 'false_positive_area_over_truth_points', 'false_negative_area_over_truth_points')
    comparisons = {}
    for left, right in COMPARISONS:
        delta = parent_means[:, names.index(left)] - parent_means[:, names.index(right)]
        interval = np.percentile(delta[parent_draws].mean(axis=1), [2.5, 97.5], axis=0)
        comparisons[compare_name((left, right))] = {
            field: dict(delta=float(mean), percentile95=bounds.tolist())
            for field, mean, bounds in zip(fields, delta.mean(axis=0), interval.T, strict=True)}
    return dict(hulls=len(annotations), groups=len(values), matched_hulls=matched, matched_groups=len(matched_groups),
                unmatched_hulls=len(annotations) - matched,
                primary_changed_hulls=primary_hulls, primary_changed_groups=len(primary_groups),
                secondary_changed_hulls=secondary_hulls, secondary_changed_groups=len(secondary_groups),
                arms={name: dict(zip(fields, map(float, parent_means.mean(axis=0)[index]), strict=True))
                      for index, name in enumerate(names)}, comparisons=comparisons)


def practical_decision(ap_delta, geometry):
    if geometry['primary_changed_hulls'] < 20 or geometry['primary_changed_groups'] < 8:
        return 'insufficient_exposure'
    lower, upper = geometry['comparisons'][compare_name(PRIMARY)]['iou_points']['percentile95']
    if ap_delta >= 0.30 and lower > 0:
        return 'positive_practical_transfer'
    if ap_delta <= -0.30 and upper < 0:
        return 'negative_practical_transfer'
    return 'mixed_or_subthreshold'


def load_evaluator(path):
    evaluator = ModuleType('pseg_frozen_score')
    exec(compile(pinned(path, EVALUATOR_SHA), str(path), 'exec'), evaluator.__dict__)
    return evaluator.evaluate


def score(truth, groups, edges, records, evaluate):
    fractions = {arm: evaluate(truth, coco_rows(records, arm)) for arm in ('native', *ARMS.values())}
    matches = native_matches(truth, records, fractions['native']['AP'])
    geometry = fixed_geometry(truth['annotations'], groups, matches, records)
    non_edge = fixed_geometry([row for row in truth['annotations'] if row['id'] not in edges], groups, matches, records)
    app = {arm: {key: (100 * value if value >= 0 else None) for key, value in values.items()}
           for arm, values in fractions.items()}
    delta = {}
    for left, right in COMPARISONS:
        delta[compare_name((left, right))] = {
            key: (100 * (value - fractions[right][key]) if value >= 0 and fractions[right][key] >= 0 else None)
            for key, value in fractions[left].items()}
    decision = practical_decision(delta[compare_name(PRIMARY)]['AP'], geometry)
    return dict(decision=decision, images=len(records), candidates=sum(map(len, records.values())),
                all_scene_groups=len(set(groups.values())), ship_free_scene_groups=len(set(groups.values())) - geometry['groups'],
                edge_touch_hulls=len(edges), coco_mask_metric_points=app, coco_mask_deltas_points=delta,
                fixed_native_match_all_truth=geometry, fixed_native_match_nonedge_sensitivity=non_edge)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for label in ('protocol', 'archive', 'capture-run', 'code', 'output'):
        parser.add_argument('--' + label, type=Path, required=True)
    parser.add_argument('--edit-receipt-sha', required=True)
    parser.add_argument('--edit-terminal-sha', required=True)
    args = parser.parse_args()
    require(not args.output.exists() and args.output.parent.is_dir(), 'Fresh isolated score output')
    started = perf_counter()
    pinned(args.protocol, PROTOCOL_SHA)
    evaluate = load_evaluator(args.code / EVALUATOR_PATH)
    records, names, counts, native, edits = qualified_records(args)
    print(json.dumps(dict(stage='label_free_streams_passed', images=len(records), **counts)), flush=True)
    with args.archive.open('rb') as stream:
        archive_hash = sha256()
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            archive_hash.update(block)
        require(archive_hash.hexdigest() == ARCHIVE_SHA, 'Locked author archive SHA-256 differs')
        with ZipFile(stream) as archive:
            truth, groups, edges = exact_ground_truth(archive, names)
    result = score(truth, groups, edges, records, evaluate)
    result.update(status='SAR_SSP_VAL232_NATIVE_RASTER_SCORE_V1', phase='score',
                  protocol_sha256=PROTOCOL_SHA, worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                  evaluator_sha256=EVALUATOR_SHA, archive_sha256=ARCHIVE_SHA,
                  checkpoint_sha256=CHECKPOINT_SHA, point_head_sha256=HEAD_SHA,
                  capture_sha256=CAPTURE_SHA, capture_readback_sha256=READBACK_SHA,
                  native_predictions_sha256=native['predictions_sha256'], native_audit_sha256=native['audit_sha256'],
                  edit_receipt_sha256=args.edit_receipt_sha, edit_terminal_sha256=args.edit_terminal_sha,
                  edits_sha256=edits['edits_sha256'], actual_signed_pair_counts=counts,
                  target_partition='val2017', target_fit_or_selection=False,
                  detector_forwards=0, optimizer_updates=0, scoring_seconds=perf_counter() - started)
    data = (json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    args.output.mkdir()
    with (args.output / 'RESULT_SAFE.json').open('xb') as stream:
        stream.write(data)
    print(json.dumps(dict(status=result['status'], result_sha256=sha256(data).hexdigest())), flush=True)


if __name__ == '__main__':
    main()
