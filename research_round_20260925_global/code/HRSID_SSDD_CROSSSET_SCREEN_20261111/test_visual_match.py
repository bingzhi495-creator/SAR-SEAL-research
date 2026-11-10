from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from visual_match import build_index, features, retrieval, verify


def textured(seed):
    rng = np.random.default_rng(seed)
    image = np.clip(rng.normal(75, 9, (384, 384)), 0, 255).astype(np.uint8)
    for y in range(18, 384, 25):
        for x in range(18, 384, 25):
            offset = rng.integers(-6, 7, size=2)
            center = (int(x + offset[0]), int(y + offset[1]))
            radius = int(rng.integers(2, 10))
            cv2.circle(image, center, radius, int(rng.integers(115, 246)), 1, cv2.LINE_AA)
            tip = (center[0] + int(rng.integers(-8, 9)), center[1] + int(rng.integers(-8, 9)))
            cv2.line(image, center, tip, int(rng.integers(120, 241)), 1, cv2.LINE_AA)
    return image


class VisualMatchTests(unittest.TestCase):
    def test_working_grid_and_descriptor_cap(self):
        image = cv2.resize(textured(31), (768, 510))
        result = features(image)
        self.assertEqual(result.gray.shape, (255, 384))
        self.assertEqual(result.points.shape, (128, 2))
        self.assertEqual(result.descriptors.shape, (128, 128))
        self.assertEqual(result.descriptors.dtype, np.float32)
        with self.assertRaises(ValueError):
            features(np.zeros((16, 16, 3), dtype=np.uint8))

    def test_shared_crop_resize_and_jpeg_retrieves_and_verifies(self):
        source = textured(17)
        crop = source[38:352, 57:353]
        adapted = cv2.resize(crop, (267, 292), interpolation=cv2.INTER_AREA)
        adapted = np.rint(255 * (adapted.astype(float) / 255) ** 0.78).astype(np.uint8)
        ok, jpeg = cv2.imencode('.jpg', adapted, [cv2.IMWRITE_JPEG_QUALITY, 86])
        self.assertTrue(ok)
        query = features(cv2.imdecode(jpeg, cv2.IMREAD_GRAYSCALE))
        gallery = [features(textured(3)), features(source), features(textured(84))]
        self.assertEqual(retrieval(build_index(gallery), query)[0], 1)
        self.assertEqual(retrieval(build_index(gallery), query), retrieval(build_index(gallery), query))
        result = verify(gallery[1], query)
        self.assertEqual(result['status'], 'strong', result)
        self.assertGreaterEqual(result['inlier_cells'], 4)
        self.assertEqual(result, verify(gallery[1], query))

    def test_similar_geometry_with_unrelated_surroundings_is_not_strong(self):
        original = textured(13)
        query = textured(14)
        query[115:269, 115:269] = original[115:269, 115:269]
        result = verify(features(original), features(query))
        self.assertEqual(result['status'], 'weak', result)
        self.assertIn(result['reason'], ('inlier_support', 'texture_support', 'texture_correlation'))

    def test_disjoint_and_empty_retrieval(self):
        blank = features(np.zeros((60, 70), dtype=np.uint8))
        image = features(textured(51))
        self.assertEqual(blank.descriptors.shape, (0, 128))
        self.assertEqual(retrieval(build_index([blank]), image), [])
        self.assertEqual(retrieval(build_index([image]), blank), [])
        self.assertEqual(verify(image, blank)['status'], 'none')
        self.assertEqual(verify(image, features(textured(52)))['status'], 'none')


if __name__ == '__main__':
    cv2.setNumThreads(1)
    unittest.main()
