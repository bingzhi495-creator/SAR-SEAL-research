from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from ssdd_phase_init import describe_q100, verify_coco_source


class SSDDPhaseInitializationTest(unittest.TestCase):
    def test_unqualified_checkpoint_is_rejected_by_identity_before_model_load(self):
        with TemporaryDirectory() as root:
            source = Path(root).resolve() / "checkpoint.pt"
            source.write_bytes(b"unqualified checkpoint")
            with self.assertRaisesRegex(ValueError, "official COCO Seg Nano size"):
                verify_coco_source(source)

    def test_width_expansion_and_invalid_query_embeddings_are_rejected(self):
        classifier = SimpleNamespace(weight=SimpleNamespace(shape=(2, 256)))
        module = SimpleNamespace(
            model_config=SimpleNamespace(num_queries=100, num_select=100, group_detr=13, num_classes=1),
            model=SimpleNamespace(num_queries=164, class_embed=classifier,
                                  state_dict=lambda: {"refpoint_embed.weight": SimpleNamespace(shape=(1300, 4)),
                                                      "query_feat.weight": SimpleNamespace(shape=(1300, 256))}),
        )
        with self.assertRaisesRegex(ValueError, "Q100"):
            describe_q100(module, lambda values: "0" * 64)
        module.model.num_queries = 100
        good = describe_q100(module, lambda values: "0" * 64)
        self.assertEqual(good["online_tensor_sha256"], "0" * 64)
        self.assertEqual(good["training_queries_per_image"], 1300)
        module.model.state_dict = lambda: {"refpoint_embed.weight": SimpleNamespace(shape=(164 * 13, 4)),
                                           "query_feat.weight": SimpleNamespace(shape=(164 * 13, 256))}
        with self.assertRaisesRegex(ValueError, "Q100"):
            describe_q100(module, lambda values: "0" * 64)

