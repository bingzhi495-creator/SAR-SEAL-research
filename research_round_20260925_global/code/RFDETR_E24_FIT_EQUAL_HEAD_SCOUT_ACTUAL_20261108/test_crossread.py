"""Synthetic fixed-order mask cross-readout contract."""
from copy import deepcopy
from unittest import TestCase, main

from crossread import build_crossread


def row(image_id, size, counts):
    return dict(image_id=image_id, category_id=1, score=.5,
                segmentation=dict(size=list(size), counts=counts))


class CrossreadTests(TestCase):
    def test_mixed_status_keeps_row_order_scores_and_ignored_original(self):
        original = {4: [row(4, [2, 2], value) for value in ('4', '13', '22')], 3: []}
        edited = {4: [row(4, [2, 2], value) for value in ('31', '22', '4')], 3: []}
        status = {4: ['matched', 'unmatched', 'ignored'], 3: []}
        frozen = deepcopy((original, edited, status))
        result = build_crossread(original, edited, status, [3, 4])

        self.assertEqual(list(result['matched_only']), [3, 4])
        self.assertEqual(result['matched_only'][4], [edited[4][0], original[4][1], original[4][2]])
        self.assertEqual(result['unmatched_only'][4], [original[4][0], edited[4][1], original[4][2]])
        self.assertEqual(result['matched_only'][3], [])
        self.assertEqual(result['unmatched_only'][3], [])
        self.assertEqual(result['summary']['status_counts'], dict(matched=1, unmatched=1, ignored=1))
        self.assertEqual(result['summary']['rle_replacements'], dict(matched_only=1, unmatched_only=1))
        result['matched_only'][4][0]['segmentation']['size'][0] = 99
        self.assertEqual((original, edited, status), frozen)

    def test_empty_and_no_detection_images_preserve_explicit_population(self):
        for ids in ([], [7]):
            with self.subTest(ids=ids):
                empty = {image: [] for image in ids}
                result = build_crossread(empty, empty, empty, ids, require_no_ignored=True)
                self.assertEqual(result['matched_only'], empty)
                self.assertEqual(result['unmatched_only'], empty)
                self.assertEqual(result['summary']['candidate_count'], 0)
                self.assertEqual(result['summary']['image_count'], len(ids))

    def test_rejects_ignored_actual_row_and_changed_score(self):
        original = {4: [row(4, [2, 2], '4')]}
        edited = deepcopy(original)
        with self.assertRaisesRegex(ValueError, 'no ignored'):
            build_crossread(original, edited, {4: ['ignored']}, [4], require_no_ignored=True)
        edited[4][0]['score'] = .51
        with self.assertRaisesRegex(ValueError, 'Frozen detector score'):
            build_crossread(original, edited, {4: ['matched']}, [4])


if __name__ == '__main__':
    main()
