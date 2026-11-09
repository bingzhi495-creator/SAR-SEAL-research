import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image
import torch
from rfdetr.config import RFDETRSegNanoConfig, SegmentationTrainConfig
from rfdetr.training.module_data import GradAccumAlignedDataset

from ssdd_fit_only import FitOnlyData


class FitOnlyTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.train = self.root / "train"
        self.train.mkdir()
        self.annotation = self.train / "_annotations.coco.json"
        self.model_config = RFDETRSegNanoConfig(
            resolution=576, positional_encoding_size=48, num_queries=100, num_select=100, num_classes=1,
        )
        self.train_config = SegmentationTrainConfig(
            dataset_dir=str(self.root), dataset_file="roboflow", class_names=["ship"], batch_size=1,
            grad_accum_steps=4, num_workers=0, augmentation_backend="torchvision", multi_scale="off",
            square_resize_div_64=True, run_test=False,
        )

    def make_fit(self, count=741):
        pixels = io.BytesIO()
        Image.new("RGB", (16, 16), (12, 34, 56)).save(pixels, "JPEG")
        jpg = pixels.getvalue()
        images, annotations = [], []
        for index in range(1, count + 1):
            name = f"fit-{index}.jpg"
            (self.train / name).write_bytes(jpg)
            images.append(dict(id=index, file_name=name, width=16, height=16))
            annotations.append(dict(id=index, image_id=index, category_id=1, area=64, iscrowd=0,
                                    bbox=[2, 2, 8, 8], segmentation=[[2, 2, 10, 2, 10, 10, 2, 10]]))
        document = dict(images=images, annotations=annotations, categories=[dict(id=1, name="ship")])
        self.annotation.write_text(json.dumps(document), encoding="utf-8")
        return document

    def module(self):
        return FitOnlyData(self.model_config, self.train_config, self.root)

    def test_fit_builds_native_loader_and_reads_one_real_synthetic_sample(self):
        self.make_fit()
        torch.manual_seed(1)
        module = self.module()
        module.setup("fit")
        module.setup("fit")
        loader = module.train_dataloader()
        self.assertIs(type(loader.dataset), GradAccumAlignedDataset)
        self.assertEqual(len(loader), 744)
        self.assertEqual(len(loader.dataset._pad_indices), 3)
        image, target = loader.dataset[0]
        self.assertEqual(tuple(image.shape), (3, 576, 576))
        self.assertEqual(target["image_id"].item(), 1)
        self.assertEqual(target["labels"].tolist(), [0])
        self.assertEqual(tuple(target["masks"].shape), (1, 576, 576))
        self.assertIsNone(module._dataset_val)
        self.assertIsNone(module._dataset_test)

    def test_nonfit_stages_and_loaders_are_denied_before_dataset_access(self):
        module = self.module()
        for stage in (None, "validate", "test", "predict"):
            with self.subTest(stage=stage), self.assertRaisesRegex(RuntimeError, "Only SSDD FIT setup"):
                module.setup(stage)
        for action in (module.val_dataloader, module.test_dataloader, module.predict_dataloader):
            with self.subTest(action=action), self.assertRaisesRegex(RuntimeError, "loaders are forbidden"):
                action()
        self.assertIsNone(module._dataset_train)
        self.assertIsNone(module._dataset_val)
        self.assertIsNone(module._dataset_test)

    def test_train_setup_rejects_a_mixed_layout_before_parsing_annotations(self):
        self.annotation.write_text("deliberately invalid", encoding="utf-8")
        (self.root / "valid").mkdir()
        module = self.module()
        with self.assertRaisesRegex(RuntimeError, "only its own train directory"):
            module.setup("fit")
        self.assertIsNone(module._dataset_train)

    def test_a_valid_json_index_cannot_direct_loader_outside_fit(self):
        data = self.make_fit()
        data["images"][0]["file_name"] = "../valid/example.jpg"
        self.annotation.write_text(json.dumps(data), encoding="utf-8")
        module = self.module()
        with self.assertRaisesRegex(RuntimeError, "annotation names must match"):
            module.setup("fit")
        with self.assertRaisesRegex(RuntimeError, "Complete isolated SSDD FIT setup"):
            module.train_dataloader()

    def test_fit_roster_must_have_741_images(self):
        self.make_fit(740)
        with self.assertRaisesRegex(RuntimeError, "FIT741 file roster"):
            self.module().setup("fit")
