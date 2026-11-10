from hashlib import sha256
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from zipfile import ZipFile

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as codec
from rfdetr.datasets.coco import make_coco_transforms_square_div_64
from rfdetr.models.postprocess import PostProcess
from rfdetr.utilities.tensors import nested_tensor_from_tensor_list

from export_schwegmann import center_row, decode_png, infer_png, manifest_items, native_center_row, read_lock, read_png, validate_image_zip


class SchwegmannExportTest(unittest.TestCase):
    def test_stock_all_slot_rank_then_center_box_selects_even_an_empty_mask(self):
        logits = torch.full((1, 164, 2), -20.)
        logits[0, 0, 0] = -25.
        logits[0, 0, 1] = 12.
        logits[0, 1, 0] = 11.
        logits[0, 2:4, 0] = 9.
        boxes = torch.full((1, 164, 4), .1)
        boxes[0, [0, 2, 3], :2] = .5
        masks = torch.ones((1, 164, 144, 144))
        masks[0, 2] = -1.
        row = native_center_row(
            dict(pred_logits=logits, pred_boxes=boxes, pred_masks=masks),
            image_id=8, torch=torch, np=np, codec=codec,
            postprocess=PostProcess(num_select=100, upsample_masks_to_image_size=True),
        )
        self.assertEqual(row["image_id"], 8)
        self.assertEqual((row["raw_query_count"], row["native_count"], row["eligible_count"]), (164, 100, 2))
        chosen = row["candidate"]
        self.assertEqual(chosen["native_rank"], 3)
        self.assertEqual(chosen["segmentation"]["size"], [101, 101])
        self.assertFalse(codec.decode(chosen["segmentation"]).any())


    def test_manifest_order_exact_zip_and_lossless_gray_png(self):
        png = BytesIO()
        Image.new("RGB", (101, 101), (37, 37, 37)).save(png, format="PNG")
        data = png.getvalue()
        row = {"image_id": 2, "basename": "chip_0002.png", "bytes": len(data), "sha256": sha256(data).hexdigest()}
        manifest = {"status": "P254_SCHWEGMANN_IMAGE_ONLY_V1", "images": [row]}
        items = manifest_items(manifest)
        self.assertEqual(items, [row])
        archive_data = BytesIO()
        with ZipFile(archive_data, "w") as archive:
            archive.writestr(row["basename"], data)
        with ZipFile(archive_data) as archive:
            validate_image_zip(archive, items)
            image = decode_png(read_png(archive, row), np)
            self.assertEqual(image.getpixel((0, 0)), (37, 37, 37))
            with self.assertRaisesRegex(ValueError, "PNG bytes"):
                read_png(archive, dict(row, sha256="0" * 64))
        with ZipFile(archive_data, "a") as archive:
            archive.writestr("other.json", b"{}")
        with ZipFile(archive_data) as archive, self.assertRaisesRegex(ValueError, "member set"):
            validate_image_zip(archive, items)
        wrong_color = BytesIO()
        Image.new("RGB", (101, 101), (37, 38, 37)).save(wrong_color, format="PNG")
        with self.assertRaisesRegex(ValueError, "replicated"):
            decode_png(wrong_color.getvalue(), np)


    def test_stock_val_transform_restores_original_canvas_and_preserves_uncropped_mask(self):
        png = BytesIO()
        Image.new("RGB", (101, 101), (18, 18, 18)).save(png, format="PNG")
        logits = torch.full((1, 164, 2), -20.)
        logits[0, 0, 0] = 8.
        boxes = torch.full((1, 164, 4), .1)
        boxes[0, 0, :2] = .5
        masks = torch.ones((1, 164, 144, 144))

        def synthetic_model(sample):
            self.assertEqual(tuple(sample.tensors.shape), (1, 3, 576, 576))
            self.assertEqual(sample.tensors.dtype, torch.float32)
            self.assertFalse(bool(sample.mask.any()))
            self.assertFalse(torch.is_grad_enabled())
            return dict(pred_logits=logits, pred_boxes=boxes, pred_masks=masks)

        transform = make_coco_transforms_square_div_64(
            "val", 576, multi_scale=False, expanded_scales=False, skip_random_resize=True,
            patch_size=12, num_windows=1, aug_config=None, scale_jitter=True, gpu_postprocess=False,
        )
        with torch.inference_mode():
            row = infer_png(
                png.getvalue(), {"image_id": 3}, model=synthetic_model, transform=transform,
                block_size=12, postprocess=PostProcess(num_select=100, upsample_masks_to_image_size=True),
                torch=torch, np=np, codec=codec, nested_tensor=nested_tensor_from_tensor_list,
            )
        self.assertEqual(row["candidate"]["native_rank"], 1)
        self.assertEqual(codec.decode(row["candidate"]["segmentation"]).shape, (101, 101))
        self.assertTrue(codec.decode(row["candidate"]["segmentation"]).all())


    def test_center_window_is_half_open_and_selection_never_crops_the_mask(self):
        results = {
            "scores": torch.linspace(1., 0., 100),
            "labels": torch.zeros(100, dtype=torch.long),
            "boxes": torch.zeros(100, 4),
            "masks": torch.zeros((100, 1, 101, 101), dtype=torch.bool),
        }
        results["labels"][0] = 1
        results["boxes"][:4] = torch.tensor([
            [45., 45., 55., 55.], [60., 40., 62., 42.], [40., 60., 42., 62.], [39., 39., 41., 41.],
        ])
        results["masks"][3, 0, 0, 0] = True
        selected = center_row(results, image_id=1, torch=torch, np=np, codec=codec)
        self.assertEqual(selected["eligible_count"], 1)
        self.assertEqual(selected["candidate"]["native_rank"], 4)
        mask = codec.decode(selected["candidate"]["segmentation"])
        self.assertEqual((int(mask.sum()), int(mask[0, 0])), (1, 1))

        results["boxes"][3, 1] = 38.
        missed = center_row(results, image_id=1, torch=torch, np=np, codec=codec)
        self.assertEqual(missed["eligible_count"], 0)
        self.assertIsNone(missed["candidate"])


    def test_duplicate_lock_keys_fail_before_source_admission(self):
        data = b'{"sources":{},"sources":{}}'
        with TemporaryDirectory() as temp:
            lock = Path(temp).resolve() / "synthetic-lock.json"
            lock.write_bytes(data)
            with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
                read_lock(lock, sha256(data).hexdigest())


if __name__ == "__main__":
    unittest.main()
