import copy
import unittest

from first_input_check import join_original_image


class OriginalIdentityTest(unittest.TestCase):
    def setUp(self):
        self.ref = dict(path='/historical/000005.jpg', bytes=38651, sha256='a' * 64)
        self.identity = dict(image_id=4, image=self.ref, original_shape=[20, 30])
        self.stage = dict(ordinal=0, image_id=4, historical_source_ref=self.ref,
                          bytes=38651, sha256='a' * 64,
                          remote_path='/private/original_ordinal000000.jpg')

    def test_exact_copy_can_move_without_rewriting_history(self):
        before = copy.deepcopy((self.stage, self.identity))
        actual = join_original_image(self.stage, self.identity)
        self.assertEqual(actual, dict(path=self.stage['remote_path'],
                                     bytes=38651, sha256='a' * 64))
        self.assertEqual((self.stage, self.identity), before)

    def test_changed_image_bytes_do_not_join_historical_identity(self):
        self.stage['sha256'] = 'b' * 64
        with self.assertRaises(ValueError):
            join_original_image(self.stage, self.identity)

    def test_wrong_image_identity_cannot_borrow_the_correct_file(self):
        self.stage['image_id'] = 5
        with self.assertRaises(ValueError):
            join_original_image(self.stage, self.identity)


if __name__ == '__main__':
    unittest.main()
