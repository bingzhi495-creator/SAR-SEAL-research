"""Synthetic fixtures for the exact image-only DEV187 materializer."""

from hashlib import sha256
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import materialize_dev187_images as stage


def fixture(root):
    objects = root / "objects" / "sha256"
    objects.mkdir(parents=True)
    payloads = [b"synthetic JPEG bytes " + i.to_bytes(2, "big") for i in range(183)]
    refs = {}
    for payload in payloads:
        digest = sha256(payload).hexdigest()
        (objects / (digest + ".jpg")).write_bytes(payload)
        refs[digest] = payload
    logical = [*payloads, *payloads[:4]]
    rows = []
    for image_id, payload in enumerate(logical):
        rows.append(dict(image_id=image_id, original_hw=[256, 512],
                         image_ref=dict(path=f"/original/valid/im{image_id:03d}.jpg", bytes=len(payload),
                                        sha256=sha256(payload).hexdigest())))
    metadata = dict(schema="Root-RFDETR-DEV187-image-only-manifest-v1", population="internal_DEV187",
                    model_execution=False, AP=False, TEST=False, expected_ids=list(range(187)), items=rows)
    return objects, logical, refs, metadata


def write_manifest(root, metadata):
    path = root / "MANIFEST.json"
    data = json.dumps(metadata).encode()
    path.write_bytes(data)
    return dict(path=path.as_posix(), bytes=len(data), sha256=sha256(data).hexdigest())


class MaterializerTest(unittest.TestCase):
    def test_full_synthetic_cohort_keeps_all_names_and_four_byte_aliases(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            objects, logical, refs, metadata = fixture(root)
            manifest = write_manifest(root, metadata)
            output = root / "ready"
            with patch.multiple(stage, MANIFEST_BYTES=manifest["bytes"], MANIFEST_SHA=manifest["sha256"],
                                UNIQUE_BYTES=sum(map(len, refs.values())), LOGICAL_BYTES=sum(map(len, logical))):
                report = stage.materialize(manifest, objects, output)
            self.assertEqual(set(output.iterdir()), {output / "valid"})
            self.assertEqual(len(list((output / "valid").iterdir())), 187)
            for image_id, payload in enumerate(logical):
                self.assertEqual((output / "valid" / f"im{image_id:03d}.jpg").read_bytes(), payload)
            self.assertEqual((report["images"], report["distinct_jpeg_sha256"], report["image_aliases"]), (187, 183, 4))
            self.assertFalse(report["GT_read"])
            self.assertFalse(report["TEST"])
            self.assertNotIn(str(root), json.dumps(report))


    def test_bad_object_cleans_partial_stage_without_touching_unrelated_files(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            objects, logical, refs, metadata = fixture(root)
            manifest = write_manifest(root, metadata)
            broken = sha256(logical[2]).hexdigest()
            (objects / (broken + ".jpg")).write_bytes(b"x" * len(logical[2]))
            sentinel = root / "existing.txt"
            sentinel.write_bytes(b"keep")
            output = root / "ready"
            with patch.multiple(stage, MANIFEST_BYTES=manifest["bytes"], MANIFEST_SHA=manifest["sha256"],
                                UNIQUE_BYTES=sum(map(len, refs.values())), LOGICAL_BYTES=sum(map(len, logical))):
                with self.assertRaisesRegex(ValueError, "input SHA-256 mismatch"):
                    stage.materialize(manifest, objects, output)
            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".dev187-stage-*")), [])
            self.assertEqual(sentinel.read_bytes(), b"keep")

    def test_windows_casefold_duplicate_refused_before_staging(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            objects, logical, refs, metadata = fixture(root)
            metadata["items"][1]["image_ref"]["path"] = "/original/valid/IM000.jpg"
            manifest = write_manifest(root, metadata)
            output = root / "ready"
            with patch.multiple(stage, MANIFEST_BYTES=manifest["bytes"], MANIFEST_SHA=manifest["sha256"],
                                UNIQUE_BYTES=sum(map(len, refs.values())), LOGICAL_BYTES=sum(map(len, logical))):
                with self.assertRaisesRegex(ValueError, "Distinct portable original JPEG basenames"):
                    stage.materialize(manifest, objects, output)
            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".dev187-stage-*")), [])


if __name__ == "__main__":
    unittest.main()
