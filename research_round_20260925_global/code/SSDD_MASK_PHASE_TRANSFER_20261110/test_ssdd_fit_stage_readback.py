from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch
from zipfile import ZIP_STORED, ZipFile

import ssdd_fit_stage_readback as subject


class SSDDFitStageReadbackTest(TestCase):
    def test_independent_reader_checks_all_names_against_the_bound_content(self):
        with TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            layout = root / "fit"
            train = layout / "train"
            train.mkdir(parents=True)
            image = b"test bytes for two logical images"
            image_sha = sha256(image).hexdigest()
            (train / "x.jpg").write_bytes(image)
            (train / "y.jpg").write_bytes(image)
            index = dict(images=[dict(id=i, file_name=n, height=5, width=7)
                                 for i, n in ((2, "x.jpg"), (1, "y.jpg"))],
                         annotations=[dict(id=4, image_id=1, category_id=1)], categories=[dict(id=1)])
            index_bytes = json.dumps(index).encode()
            (train / "_annotations.coco.json").write_bytes(index_bytes)
            rows = [dict(dataset_index=i, image_id=row["id"], original_basename=row["file_name"],
                         original_shape=[5, 7], original_image_ref=dict(bytes=len(image), sha256=image_sha),
                         archive_image_entry=f"images/{image_sha}.jpg") for i, row in enumerate(index["images"])]
            manifest = json.dumps(dict(schema=subject.MANIFEST_SCHEMA, rows=rows)).encode()
            archive = root / "fixed.zip"
            with ZipFile(archive, "w", compression=ZIP_STORED) as zipped:
                zipped.writestr("manifest.json", manifest)
                zipped.writestr("annotations/fit.json", index_bytes)
                zipped.writestr(f"images/{image_sha}.jpg", image)
            overrides = dict(FIT_IMAGES=2, FIT_ANNOTATIONS=1, UNIQUE_IMAGES=1, FIT_ARCHIVE_BYTES=archive.stat().st_size,
                             FIT_ARCHIVE_SHA=sha256(archive.read_bytes()).hexdigest(), INDEX_BYTES=len(index_bytes),
                             FIT_INDEX_SHA=sha256(index_bytes).hexdigest(), MANIFEST_BYTES=len(manifest),
                             MANIFEST_SHA=sha256(manifest).hexdigest())
            with patch.multiple(subject, **overrides):
                got = subject.readback_fit_layout(archive, layout)
                self.assertEqual(got["images"], 2)
                self.assertEqual(got["unique_encoded_images"], 1)
                self.assertFalse(got["DEV"] or got["TEST"] or got["model_executed"])
                (train / "y.jpg").write_bytes(image[::-1])
                with self.assertRaisesRegex(ValueError, "JPEG content"):
                    subject.readback_fit_layout(archive, layout)

    def test_mixed_layout_fails_before_opening_the_archive(self):
        with TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            (root / "train").mkdir()
            (root / "valid").mkdir()
            with patch.object(subject, "file_sha256") as digest:
                with self.assertRaisesRegex(ValueError, "FIT-only"):
                    subject.readback_fit_layout(root / "missing.zip", root)
                digest.assert_not_called()

