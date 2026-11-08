"""Synthetic archive scope and raster tests under the pinned CPU runtime."""
from io import BytesIO
import json
from pathlib import Path
import sys
import tempfile
import unittest
from zipfile import ZipFile

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hrsid_png_fit_audit import audit_fit


def encoded_png(color, metadata=None):
    out = BytesIO()
    Image.new("RGB", (800, 800), color).save(out, "PNG", pnginfo=metadata)
    return out.getvalue()


def annotation(image_id, ann_id, polygon):
    return {"id": ann_id, "image_id": image_id, "category_id": 1, "iscrowd": 0,
            "segmentation": [polygon], "area": 16.0, "bbox": [2.0, 2.0, 4.0, 4.0]}


def image_record(image_id, name, split):
    return {"id": image_id, "file_name": name + ".png", "width": 800, "height": 800}, {
        "image_id": image_id, "file_name": name + ".jpg", "group": name.split("_")[0], "split": split,
    }


class ObservedZip(ZipFile):
    def __init__(self, path):
        super().__init__(path)
        self.opened = []

    def open(self, name, *args, **kwargs):
        self.opened.append(name if isinstance(name, str) else name.filename)
        return super().open(name, *args, **kwargs)


class ScopeTest(unittest.TestCase):
    def test_reads_only_fit_png_and_original_train_annotations(self):
        fi, fa = image_record(0, "P0001_chip", "fit")
        di, da = image_record(1, "P0002_chip", "dev")
        train = {"images": [fi, di], "annotations": [
            annotation(0, 8, [2, 2, 6, 2, 6, 6, 2, 6]),
            {"id": 9, "image_id": 1, "segmentation": "poison DEV geometry"},
        ], "categories": [{"id": 1, "name": "ship"}]}
        with tempfile.TemporaryDirectory() as out:
            path = Path(out) / "scope.zip"
            with ZipFile(path, "w") as z:
                z.writestr("annotations/train2017.json", json.dumps(train))
                z.writestr("annotations/test2017.json", "poison TEST annotation")
                z.writestr("images/P0001_chip.png", encoded_png((9, 2, 1)))
                z.writestr("images/P0002_chip.png", "poison DEV PNG")
                z.writestr("images/P0003_chip.png", "poison TEST PNG")
            with ObservedZip(path) as z:
                result = audit_fit(z, [fa, da], expected=(2, 2, 1, 1, 1, 1))
                self.assertEqual(z.opened, ["annotations/train2017.json", "images/P0001_chip.png"])
        self.assertEqual(result["decoded_fit_images"], 1)
        self.assertEqual(result["nonempty_fit_rasters"], 1)
        self.assertTrue(result["strict_input_qualified"])


    def test_empty_and_duplicate_rasters_are_reported_without_losing_rows(self):
        from PIL.PngImagePlugin import PngInfo
        rows = [image_record(i, f"P000{i+1}_chip", "fit") for i in range(3)]
        positive = [2, 2, 6, 2, 6, 6, 2, 6]
        outside = [810, 810, 815, 810, 815, 815, 810, 815]
        multi = annotation(2, 8, positive)
        multi["segmentation"].append([10, 10, 14, 10, 14, 14, 10, 14])
        train = {"images": [im for im, _ in rows], "annotations": [
            annotation(0, 6, positive), annotation(1, 7, outside), multi,
        ], "categories": [{"id": 1, "name": "ship"}]}
        plain = encoded_png((1, 2, 3))
        metadata = PngInfo()
        metadata.add_text("fixture", "same pixels, different bytes")
        with tempfile.TemporaryDirectory() as out:
            path = Path(out) / "duplicates.zip"
            with ZipFile(path, "w") as z:
                z.writestr("annotations/train2017.json", json.dumps(train))
                z.writestr("images/P0001_chip.png", plain)
                z.writestr("images/P0002_chip.png", plain)
                z.writestr("images/P0003_chip.png", encoded_png((1, 2, 3), metadata))
            diagnostics = []
            with ObservedZip(path) as z:
                result = audit_fit(z, [r for _, r in rows], expected=(3, 3, 3, 3, 3, 0),
                                   diagnostics=diagnostics)
        self.assertEqual(result["status"], "COMPLETED_FULL_FIT_WITH_EMPTY_RASTERS")
        self.assertTrue(result["full_fit_image_and_raster_attempt_coverage"])
        self.assertFalse(result["strict_input_qualified"])
        self.assertEqual(result["attempted_fit_rasters"], 3)
        self.assertEqual(result["nonempty_fit_rasters"], 2)
        self.assertEqual(result["empty_fit_rasters"], 1)
        self.assertEqual(result["fit_multi_component_rastered_annotations"], 1)
        self.assertEqual(result["encoded_fit_png"]["excess_duplicate_images"], 1)
        self.assertEqual(result["decoded_fit_RGB8"]["excess_duplicate_images"], 2)
        self.assertEqual(diagnostics, [{"image_id": 1, "annotation_id": 7, "kind": "EMPTY_COCO_RASTER"}])


if __name__ == "__main__":
    unittest.main()


