import unittest
from types import SimpleNamespace

import torch

from phase_train import batch_identity, tensor_digest


class QualificationIdentityTests(unittest.TestCase):
    def test_supervision_and_image_fingerprints_are_separate(self):
        sample = SimpleNamespace(tensors=torch.ones(1, 3, 4, 4))
        target = dict(image_id=torch.tensor([7]), labels=torch.zeros(1, dtype=torch.int64),
                      masks=torch.zeros(1, 4, 4, dtype=torch.bool), boxes=torch.tensor([[.5, .5, .1, .1]]))
        first = batch_identity(sample, target)
        target["masks"][0, 2, 2] = True
        second = batch_identity(sample, target)
        self.assertEqual(first["image_id"], 7)
        self.assertEqual(first["image_sha256"], second["image_sha256"])
        self.assertEqual(first["boxes_sha256"], second["boxes_sha256"])
        self.assertNotEqual(first["masks_sha256"], second["masks_sha256"])
        self.assertNotEqual(tensor_digest(torch.zeros(2, dtype=torch.bool)),
                            tensor_digest(torch.zeros(2, dtype=torch.uint8)))


if __name__ == "__main__":
    unittest.main()
