import json
from io import BytesIO
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from zipfile import ZipFile

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from materialize_fit_native import materialize


def png_bytes():
    with BytesIO() as stream:
        Image.new("RGB", (800, 800), (12, 34, 56)).save(stream, format="PNG")
        return stream.getvalue()


def fixture_view(name="P1_fit.jpg"):
    return {"info": {"name": "synthetic"}, "images": [{"id": 4, "file_name": name, "height": 800, "width": 800}],
            "categories": [{"id": 1, "name": "ship"}],
            "annotations": [{"id": 9, "image_id": 4, "category_id": 1, "bbox": [1, 2, 3, 4]}]}


class NativeFITLayoutTest(unittest.TestCase):
    def test_materializes_only_fit_and_preserves_all_other_fields(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = png_bytes()
            archive = root / "source.zip"
            with ZipFile(archive, "w") as stream:
                stream.writestr("images/P1_fit.png", image)
                stream.writestr("images/P2_other_split.png", b"must never be decoded")
            out = root / "dataset"
            with ZipFile(archive) as stream:
                result = materialize(stream, fixture_view(), out)
            self.assertEqual(json.loads((out / "train/_annotations.coco.json").read_bytes()),
                             fixture_view("P1_fit.png"))
            self.assertEqual((out / "train/P1_fit.png").read_bytes(), image)
            self.assertEqual({p.name for p in (out / "train").iterdir()}, {"P1_fit.png", "_annotations.coco.json"})
            self.assertFalse((out / "valid").exists())
            self.assertFalse((out / "test").exists())
            self.assertEqual(result["images"][0]["member"], "images/P1_fit.png")



    def test_rejects_unsafe_basename_and_case_collisions_before_creating_output(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "source.zip"
            with ZipFile(archive, "w") as stream:
                stream.writestr("images/P1_fit.png", png_bytes())
            out = root / "dataset"
            with ZipFile(archive) as stream:
                for name in ("../P1_fit.jpg", "C:/P1_fit.jpg", r"folder\P1_fit.jpg"):
                    with self.subTest(name=name), self.assertRaisesRegex(ValueError, "Safe official FIT"):
                        materialize(stream, fixture_view(name), out)
                duplicated = fixture_view()
                duplicated["images"].append(dict(duplicated["images"][0], id=5, file_name="P1_FIT.png"))
                with self.assertRaisesRegex(ValueError, "Duplicate FIT"):
                    materialize(stream, duplicated, out)
            self.assertFalse(out.exists())

    def test_rejects_relative_existing_and_dot_parent_outputs(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "subdir").mkdir()
            archive = root / "source.zip"
            with ZipFile(archive, "w") as stream:
                stream.writestr("images/P1_fit.png", png_bytes())
            with ZipFile(archive) as stream:
                for path in (Path("relative_dataset"), root / "subdir", root / "subdir/../dataset"):
                    with self.subTest(path=path), self.assertRaisesRegex(ValueError, "Fresh, absolute and resolved"):
                        materialize(stream, fixture_view(), path)
            self.assertFalse((root / "dataset").exists())

    def test_rejects_duplicate_selected_archive_member_before_creating_output(self):
        import warnings

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "source.zip"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                with ZipFile(archive, "w") as stream:
                    stream.writestr("images/P1_fit.png", png_bytes())
                    stream.writestr("images/P1_fit.png", png_bytes())
            out = root / "dataset"
            with ZipFile(archive) as stream, self.assertRaisesRegex(ValueError, "exactly once"):
                materialize(stream, fixture_view(), out)
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
