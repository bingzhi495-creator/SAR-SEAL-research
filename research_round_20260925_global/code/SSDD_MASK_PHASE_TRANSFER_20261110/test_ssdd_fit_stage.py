from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import ssdd_fit_stage as subject


class SSDDFitStageTest(TestCase):
    def test_bound_split_builder_is_called_for_train_only(self):
        with TemporaryDirectory() as parent:
            root = Path(parent).resolve()
            archive = root / "fixed_fit.zip"
            with archive.open("xb") as stream:
                stream.truncate(subject.FIT_ARCHIVE_BYTES)
            destination = root / "isolated"

            def assemble(split, bound, target):
                self.assertEqual((split, bound, target), ("train", archive, destination / "train"))
                target.mkdir()
                return dict(image_ids=list(range(1, 742)), images=741, instances=1661, category_ids=[1],
                            unique_encoded_images=730, annotation_bytes=498760,
                            annotation_sha256=subject.FIT_INDEX_SHA), {"synthetic"}

            with patch.object(subject, "load_api", return_value=SimpleNamespace(assemble_split=assemble)):
                safe = subject.stage_fit_layout(archive, destination)
            self.assertEqual(list(destination.iterdir()), [destination / "train"])
            self.assertEqual(safe["images"], 741)
            self.assertFalse(safe["DEV"] or safe["TEST"])
            self.assertNotIn("image_ids", safe)
            ordered = json.dumps(list(range(1, 742)), separators=(",", ":")).encode("utf8")
            self.assertEqual(safe["native_sorted_id_order_sha256"], sha256(ordered).hexdigest())

    def test_refuses_reusing_layout_and_never_calls_builder(self):
        with TemporaryDirectory() as parent:
            root = Path(parent).resolve()
            archive = root / "fixed_fit.zip"
            with archive.open("xb") as stream:
                stream.truncate(subject.FIT_ARCHIVE_BYTES)
            destination = root / "existing"
            destination.mkdir()
            with patch.object(subject, "load_api") as load:
                with self.assertRaisesRegex(ValueError, "fresh"):
                    subject.stage_fit_layout(archive, destination)
                load.assert_not_called()

