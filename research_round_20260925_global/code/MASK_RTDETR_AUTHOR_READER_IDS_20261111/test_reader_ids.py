import unittest

from reader_ids import bind_reader_ids


class ReaderIdentityTest(unittest.TestCase):
    def test_reader_ids_bind_by_exact_path_not_input_position_or_coco_id(self):
        images = [(900, "last.png"), (2, "first.png"), (413, "middle.png")]
        author_paths = {0: "middle.png", 1: "last.png", 2: "first.png"}
        self.assertEqual(bind_reader_ids(images, author_paths), {0: 413, 1: 900, 2: 2})

    def test_rejects_duplicate_missing_or_unexpected_image(self):
        for images, author_paths in [
            ([(1, "a"), (1, "b")], {0: "a", 1: "b"}),
            ([(1, "a"), (2, "a")], {0: "a", 1: "a"}),
            ([(1, "a"), (2, "b")], {0: "a", 1: "a"}),
            ([(1, "a"), (2, "b")], {0: "a", 1: "c"}),
            ([(1, "a"), (2, "b")], {0: "a"}),
        ]:
            with self.subTest(images=images, author_paths=author_paths):
                with self.assertRaises(ValueError):
                    bind_reader_ids(images, author_paths)
