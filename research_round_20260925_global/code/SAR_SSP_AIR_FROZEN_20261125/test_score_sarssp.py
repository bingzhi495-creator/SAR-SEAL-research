"""Synthetic regressions for the prelocked scoring adapter."""
import importlib.util
import sys
import unittest
from hashlib import sha256
from io import BytesIO
import json
from zipfile import ZipFile

from PIL import Image
from pathlib import Path

import numpy as np
from pycocotools import mask as codec

spec = importlib.util.spec_from_file_location("score_sarssp", Path(__file__).with_name("score_sarssp.py"))
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def rect(x, y):
    result = np.zeros((12, 12), dtype=np.uint8)
    result[y:y + 6, x:x + 6] = 1
    return result


class Adapter(unittest.TestCase):
    def setUp(self):
        self.baseline = rect(2, 2)
        self.better = self.baseline.copy()
        self.better[2, 2] = 0
        self.better[2, 8] = 1

    def encoded(self, data):
        raw = codec.encode(np.asfortranarray(data))
        return dict(size=list(raw['size']), counts=raw['counts'].decode('ascii'))

    def test_decode_size_and_area(self):
        self.assertEqual(worker.rle_area(self.encoded(self.baseline), [12, 12]), 36)
        with self.assertRaises(ValueError):
            worker.rle_area(self.encoded(self.baseline), [12, 13])

    def test_signed_counts(self):
        self.assertEqual(worker.signed_counts(self.encoded(self.baseline), self.encoded(self.better), 36), (1, 1))
        add_only = self.baseline.copy()
        add_only[2, 8] = 1
        self.assertEqual(worker.signed_counts(self.encoded(self.baseline), self.encoded(add_only), 36), (1, 0))

    def test_fixed_geometry_complete_truth(self):
        result = worker.fixed_geometry(
            annotations=[dict(id=1, image_id=1, area=36, segmentation=self.encoded(self.better)),
                         dict(id=2, image_id=2, area=36, segmentation=self.encoded(self.baseline))],
            groups={1: 10, 2: 20, 3: 30},
            matches={1: (1, 1)},
            records={1: [dict(native=self.encoded(self.baseline), masks={'free_matched_count': self.encoded(self.better)}, counts={'free_matched_count': 1})]},
        )
        self.assertEqual(result['hulls'], 2)
        self.assertEqual(result['groups'], 2)
        self.assertEqual(result['matched_hulls'], 1)
        self.assertEqual(result['primary_changed_hulls'], 1)
        self.assertEqual(result['primary_changed_groups'], 1)
        self.assertAlmostEqual(result['comparisons']['free_matched_count__vs__distance_paired']['iou_points']['delta'], (1 - 35 / 37) * 50)


    def fixture(self):
        names = ['synthetic-1.jpg', 'synthetic-2.jpg', 'synthetic-3.jpg']
        groups = {1: 10, 2: 20, 3: 30}
        truth = dict(images=[dict(id=i, height=12, width=12) for i in groups],
                     annotations=[dict(id=i, image_id=i, category_id=0, iscrowd=0, area=36,
                                       bbox=codec.toBbox(rle).tolist(), segmentation=rle)
                                  for i, rle in [(1, self.encoded(self.better)), (2, self.encoded(self.baseline))]],
                     categories=[dict(id=0, name='ship')])
        stock = self.encoded(self.baseline)
        edits = [dict(image_id=i, changes={}) for i in groups]
        moved = self.baseline.copy()
        moved[7, 7] = 0
        moved[8, 7] = 1
        edits[0]['changes']['1'] = dict(fixed=dict(swaps=1, rle=self.encoded(moved)),
                                         free_matched_count=dict(swaps=1, rle=self.encoded(self.better)))
        predictions, audit = [], []
        for i in groups:
            has_prediction = i != 2
            score = 0.8 if i == 1 else 0.9
            ranks = [1] if has_prediction else []
            predictions.append(dict(image_id=i, native_ranks=ranks,
                                    coco=[dict(image_id=i, category_id=1, score=score, segmentation=stock)]
                                    if has_prediction else []))
            audit.append(dict(image_id=i, original_hw=[12, 12], native_ranks=ranks, source_image_name=names[i-1],
                              image_sha256='synthetic', candidates=[dict(position=1, native_rank=1, query_id=0,
                              score=score, mask_area=36)] if has_prediction else []))
        return truth, groups, edits, predictions, audit

    def test_full_candidate_scoring_and_empty_image(self):
        truth, groups, edits, predictions, audit = self.fixture()
        records, _, counts = worker.stream_records(predictions, audit, edits, [12, 12])
        self.assertEqual(counts['mapped_candidates'], 2)
        self.assertEqual(counts['fixed_swaps'], 1)
        self.assertEqual(worker.coco_rows(records, 'free_matched_count')[0]['score'], 0.8)
        evaluator = worker.load_evaluator(Path(sys.argv[1]))
        result = worker.score(truth, groups, {1}, records, evaluator)
        self.assertEqual(result['ship_free_scene_groups'], 1)
        self.assertEqual(result['fixed_native_match_all_truth']['unmatched_hulls'], 1)
        self.assertEqual(result['fixed_native_match_nonedge_sensitivity']['matched_hulls'], 0)
        self.assertEqual(result['decision'], 'insufficient_exposure')
        self.assertGreater(result['coco_mask_deltas_points'][worker.compare_name(worker.PRIMARY)]['AP'], 0)
        self.assertLess(result['coco_mask_metric_points']['native']['AP50'], 50)

    def test_lost_stream_incorrect_score_and_count_rejected(self):
        _, _, edits, predictions, audit = self.fixture()
        with self.assertRaises(ValueError):
            worker.stream_records(predictions[:-1], audit, edits, [12, 12])
        audit[0]['candidates'][0]['score'] = 0.4
        with self.assertRaisesRegex(ValueError, 'Exact original score'):
            worker.stream_records(predictions, audit, edits, [12, 12])
        audit[0]['candidates'][0]['score'] = 0.8
        del edits[0]['changes']['1']['fixed']
        with self.assertRaisesRegex(ValueError, 'Equal actual signed edits'):
            worker.stream_records(predictions, audit, edits, [12, 12])


    def test_complete_panoptic_archive_preserves_rgb_ids_edge_flags_and_hulls(self):
        archive_buffer = BytesIO()
        stem = 'SAR-SSP/annotations/panoptic_val2017'
        names, index, images, annotations = {}, [], [], []
        with ZipFile(archive_buffer, mode='w') as archive:
            def add(path, data):
                archive.writestr('datasets/' + path, data)
                digest = sha256(data).hexdigest()
                index.append(dict(path=path, bytes=len(data), sha256=digest))
                return digest

            for iid in range(1, 233):
                group = (iid - 1) % (23 if iid <= 48 else 31)
                file_name = f'{iid:012}.jpg'
                h, w = worker.HW
                images.append(dict(id=iid, file_name=file_name, source_image_no=group,
                                   tile_x=iid * w, tile_y=0, height=h, width=w))
                names[iid] = (file_name, add('SAR-SSP/val2017/' + file_name, str(iid).encode()))
                segments, rgb = [], np.zeros((h, w, 3), dtype=np.uint8)
                if iid <= 48:
                    hulls = [(4, 0, 6, 5), (15, 15, 6, 6)]
                    if iid <= 43:
                        hulls.append((51, 51, 5, 6))
                    for j, (x, y, width, height) in enumerate(hulls):
                        color = [19+j, 186, 79]
                        rgb[y:y+height, x:x+width] = color
                        code = sum(v * 256 ** k for k, v in enumerate(color))
                        segments.append(dict(id=code, category_id=0, area=width*height,
                                             bbox=[x, y, width, height], iscrowd=0))
                mask_name = f'{iid:012}.png'
                mask_buffer = BytesIO()
                Image.fromarray(rgb).save(mask_buffer, format='PNG')
                add(stem + '/' + mask_name, mask_buffer.getvalue())
                annotations.append(dict(image_id=iid, file_name=mask_name, source_image_no=group,
                                        tile_x=iid*w, tile_y=0, segments_info=segments))
            add(stem + '.json', json.dumps(dict(images=images, annotations=annotations,
                categories=[dict(id=i, name=name, isthing=int(i == 0))
                            for i, name in enumerate(['ship', 'land', 'sea'])])).encode())
            archive.writestr('datasets/DATASET_INDEX.json', json.dumps(dict(files=index)))
        with ZipFile(archive_buffer) as archive:
            truth, groups, edges = worker.exact_ground_truth(archive, names)
        self.assertEqual((len(truth['annotations']), len(set(groups.values())), len(edges)), (139, 31, 48))
        self.assertEqual(truth['annotations'][0]['bbox'], [4, 0, 6, 5])
        self.assertEqual(worker.rle_area(truth['annotations'][0]['segmentation'], worker.HW), 30)
        self.assertIn(1, edges)
        self.assertNotIn(2, edges)


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0]])
