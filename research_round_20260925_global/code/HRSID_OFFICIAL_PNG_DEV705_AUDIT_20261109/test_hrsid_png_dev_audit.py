"""Small synthetic ZIP fixtures for the DEV-only audit contract."""
from io import BytesIO
import json
import tempfile
import unittest
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from PIL import Image

from hrsid_png_dev_audit import audit_dev


class ObservedZip(ZipFile):
    def __init__(self, path):
        super().__init__(path)
        self.opened = []

    def open(self, name, *args, **kwargs):
        self.opened.append(name if isinstance(name, str) else name.filename)
        return super().open(name, *args, **kwargs)


def fixture(path):
    images = [
        {"id": i, "file_name": f"P{i:04d}_chip.png", "width": 800, "height": 800}
        for i in (1, 2)
    ]
    assignment = [
        {"image_id": i, "file_name": f"P{i:04d}_chip.jpg", "group": f"P{i:04d}", "split": role}
        for i, role in ((1, "fit"), (2, "dev"))
    ]
    annotations = [
        {"id": 4, "image_id": 1, "segmentation": "out-of-scope FIT geometry"},
        {"id": 5, "image_id": 2, "category_id": 1, "iscrowd": 0, "area": 4.125,
         "bbox": [2.25, 2.5, 2, 2.25], "segmentation": [[2.25, 2.5, 4.25, 2.5, 4.25, 4.75, 2.25, 4.75]]},
        {"id": 6, "image_id": 2, "category_id": 1, "iscrowd": 0, "area": 25.0,
         "bbox": [810, 810, 5, 5], "segmentation": [[810, 810, 815, 810, 815, 815, 810, 815]]},
    ]
    train = {"images": images, "annotations": annotations, "categories": [{"id": 1, "name": "ship"}]}
    rgb = BytesIO()
    Image.new("RGB", (800, 800), (1, 7, 19)).save(rgb, "PNG")
    with ZipFile(path, "w") as z:
        z.writestr("annotations/train2017.json", json.dumps(train))
        z.writestr("annotations/test2017.json", "poison TEST annotations")
        z.writestr("images/P0001_chip.png", "poison FIT pixels")
        z.writestr("images/P0002_chip.png", rgb.getvalue())
        z.writestr("images/P0003_chip.png", "poison TEST pixels")
    return assignment, images[1], annotations[1:]


class AuditScopeTest(unittest.TestCase):
    def test_reads_only_dev_png_and_keeps_original_float_and_empty_gt(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "scope.zip"
            roles, image, annotations = fixture(path)
            with ObservedZip(path) as archive:
                report, manifest, gt, findings = audit_dev(archive, roles, expected=(2, 3, 1, 1, 1, 1))
                self.assertEqual(archive.opened, ["annotations/train2017.json", "images/P0002_chip.png"])
        self.assertTrue(report["full_dev_input_audit"])
        self.assertEqual(report["decoded_dev_images"], 1)
        self.assertEqual(report["empty_dev_rasters"], 1)
        self.assertEqual(report["fractional_coordinate_annotations"], 1)
        self.assertEqual(gt["images"], [image])
        self.assertEqual(gt["annotations"], annotations)
        self.assertEqual([row["image_id"] for row in manifest], [2])
        self.assertEqual(findings, [{"image_id": 2, "annotation_id": 6, "kind": "EMPTY_COCO_RASTER"}])


    def test_rejects_corrupt_selected_png_by_zip_crc(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "corrupt.zip"
            roles, _, _ = fixture(path)
            with ZipFile(path) as archive:
                info = archive.getinfo("images/P0002_chip.png")
            offset = info.header_offset + 30 + len(info.filename.encode("ascii")) + len(info.extra)
            with path.open("r+b") as stream:
                stream.seek(offset + 100)
                old = stream.read(1)
                stream.seek(offset + 100)
                stream.write(bytes([old[0] ^ 1]))
            with ObservedZip(path) as archive:
                with self.assertRaises(BadZipFile):
                    audit_dev(archive, roles, expected=(2, 3, 1, 1, 1, 1))
                self.assertEqual(archive.opened, ["annotations/train2017.json", "images/P0002_chip.png"])

    def test_reports_decoded_overlap_even_when_encoded_bytes_differ(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "overlap.zip"
            roles, _, _ = fixture(path)
            with ObservedZip(path) as archive:
                _, images, _, _ = audit_dev(archive, roles, expected=(2, 3, 1, 1, 1, 1))
            fit_fingerprints = {"encoded_sha256": {"another encoding"},
                                "decoded_rgb_sha256": {images[0]["decoded_rgb_sha256"]}}
            with ObservedZip(path) as archive:
                report, _, _, _ = audit_dev(archive, roles, expected=(2, 3, 1, 1, 1, 1),
                                            fit_fingerprints=fit_fingerprints)
        self.assertEqual(report["dev_images_matching_fit_fingerprints"],
                         {"encoded_sha256": 0, "decoded_rgb_sha256": 1})

if __name__ == "__main__":
    unittest.main()

