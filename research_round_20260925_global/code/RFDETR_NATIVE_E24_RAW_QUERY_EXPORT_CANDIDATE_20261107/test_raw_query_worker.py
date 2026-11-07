import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from raw_query_worker import image_record, observed_postprocess, write_records


def canonical_bytes(value):
    return json.dumps(value, separators=(',', ':'), allow_nan=False).encode('utf-8')


def stock_slot0_rows():
    return [(rank, 99-index, 1.0-rank/201.0, dict(size=[2, 2], counts='fixture'+str(index)))
            for index, rank in enumerate(range(1, 200, 2))]


def accepted_primary(rows):
    retained = [(rank, query, score, rle) for rank, query, score, rle in rows if rank <= 100]
    records = [dict(image_id=37, category_id=1, score=score, segmentation=rle)
               for rank, query, score, rle in retained]
    return dict(image_id=37, image_sha256='a'*64, original_hw=[2, 2],
                retained_native_ranks=[rank for rank, query, score, rle in retained],
                prediction_count=len(records),
                canonical_json_sha256=hashlib.sha256(canonical_bytes(records)).hexdigest())


class RawQueryTests(unittest.TestCase):
    def test_all_query_masks_keep_natural_index_and_primary_stock_rank_witness(self):
        rows = stock_slot0_rows()
        primary = accepted_primary(rows)
        record = image_record(primary, rows)
        self.assertEqual(record['image_id'], 37)
        self.assertEqual(len(record['raw_masks']), 100)
        self.assertEqual(record['raw_masks'][0]['counts'], 'fixture99')
        self.assertEqual(record['raw_masks'][99]['counts'], 'fixture0')
        self.assertEqual(record['slot0_native_ranks'][99], 1)
        self.assertEqual(record['slot0_native_ranks'][0], 199)
        self.assertEqual(record['retained'][0], dict(native_rank=1, query_id=99))
        self.assertEqual(len(record['retained']), 50)
        self.assertEqual(record['primary_canonical_json_sha256'], primary['canonical_json_sha256'])


    def test_primary_score_change_fails_the_admitted_prefix_hash(self):
        rows = stock_slot0_rows()
        primary = accepted_primary(rows)
        rank, query, score, rle = rows[0]
        rows[0] = (rank, query, score - 0.001, rle)
        with self.assertRaisesRegex(ValueError, 'accepted original top100'):
            image_record(primary, rows)

    def test_missing_raw_query_cannot_be_replaced_by_a_duplicate(self):
        rows = stock_slot0_rows()
        primary = accepted_primary(rows)
        rank, query, score, rle = rows[-1]
        rows[-1] = (rank, rows[-2][1], score, rle)
        with self.assertRaisesRegex(ValueError, 'duplicate raw query'):
            image_record(primary, rows)

    def test_stock_observer_passes_native_tuple_by_identity(self):
        selection = object(), object(), object()

        class Native:
            def __init__(self, **options):
                self.options = options

            def _select_topk(self, logits):
                return selection

        worker = observed_postprocess(Native)
        self.assertEqual(worker.options, dict(num_select=200, upsample_masks_to_image_size=True))
        self.assertIs(worker._select_topk('fixture logits'), selection)
        self.assertIs(worker.selection, selection)

    def test_failed_second_hash_keeps_only_the_first_verified_partial_line(self):
        rows = stock_slot0_rows()
        verified = accepted_primary(rows)
        bad = dict(verified, canonical_json_sha256='c'*64)
        calls = []

        def infer(item):
            calls.append(item)
            return rows

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory).resolve() / 'diagnostic'
            with self.assertRaisesRegex(ValueError, 'accepted original top100'):
                write_records(output, list(range(187)), [verified, bad] + [verified]*185, infer)
            self.assertEqual(calls, [0, 1])
            self.assertFalse((output/'RAW_QUERY_SLOT0.jsonl').exists())
            lines = (output/'RAW_QUERY_SLOT0.jsonl.partial').read_bytes().splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[0])['primary_canonical_json_sha256'],
                             verified['canonical_json_sha256'])


if __name__ == '__main__':
    unittest.main()
