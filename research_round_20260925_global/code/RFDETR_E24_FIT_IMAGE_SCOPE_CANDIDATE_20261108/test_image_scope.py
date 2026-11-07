from unittest import TestCase
from image_scope import split_by_content

class ImageScopeTests(TestCase):
    def test_duplicate_byte_groups_are_atomic_and_annotation_fields_are_ignored(self):
        source = [dict(image_id=i, image_sha256=hex(i // 2)[2:].zfill(64), ignored=i*100) for i in range(12)]
        a = split_by_content(source)
        b = split_by_content([dict(row, ignored=None) for row in reversed(source)])
        self.assertEqual(a, b)
        self.assertEqual(set(a), set(range(12)))
        self.assertEqual(a[0], a[1])
        self.assertEqual(a[10], a[11])
        self.assertEqual({a[i] for i in a}, {'head_fit','calibration','head_holdout'})
