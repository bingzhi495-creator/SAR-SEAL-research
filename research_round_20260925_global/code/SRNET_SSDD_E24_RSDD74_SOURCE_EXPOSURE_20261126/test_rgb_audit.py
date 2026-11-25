from io import BytesIO
import unittest

from PIL import Image, ImageOps
from audit_rgb import compare, rgb_key


def fixture(color, size=(3, 2), orient=False):
    stream = BytesIO()
    image = Image.new("RGB", size, color)
    if orient:
        exif = Image.Exif()
        exif[274] = 6
        image.save(stream, format="PNG", exif=exif)
    else:
        image.save(stream, format="PNG")
    return stream.getvalue()


class RgbIdentity(unittest.TestCase):
    def test_overlap_weights_duplicates_and_requires_native_shape(self):
        first = rgb_key(fixture("red"))
        same = rgb_key(fixture("red", (2, 3), orient=True))
        transposed_shape = rgb_key(fixture("red", (2, 3)))
        second = rgb_key(fixture("blue"))
        self.assertEqual(first, same)
        self.assertEqual(first[2], transposed_shape[2])
        self.assertNotEqual(first, transposed_shape)
        self.assertEqual(compare([first, first, second], [first, transposed_shape]),
                         {"unique_source_rgb":2, "unique_target_rgb":2,
                          "matching_rgb_identities":1, "source_images_matching":2,
                          "target_images_matching":1, "cross_pair_count":2})


if __name__ == "__main__":
    unittest.main()

