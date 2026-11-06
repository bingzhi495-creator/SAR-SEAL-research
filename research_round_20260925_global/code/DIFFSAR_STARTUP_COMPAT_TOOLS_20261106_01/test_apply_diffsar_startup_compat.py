"""Small CLI checks using text-only temporary source trees; no ML imports."""

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("apply_diffsar_startup_compat.py")
FIXTURES = {
    "detectron2/modeling/backbone/__init__.py": (
        "# unrelated first line\n"
        "from .fpn import FPN, FPN_1p_5p, SCJE_FPN, FPN_1p_4p\n"
        "from .fan import FAN\n"
        "from detectron2_backbone.backbone.bifpn import build_resnet_bifpn_p1_p5_backbone\n"
        "# unrelated last line\n"
    ),
    "diffsarshipinst/loss.py": (
        "# unrelated first line\n"
        "from fvcore.nn import sigmoid_focal_loss_jit, poly_focal_loss_jit\n"
        "# unrelated last line\n"
    ),
    "detectron2/data/transforms/transform.py": (
        "# unrelated first line\n"
        "    def __init__(self, src_rect, output_size, interp=Image.LINEAR, fill=0):\n"
        "# unrelated last line\n"
    ),
}


class CompatCLI(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for relative, text in FIXTURES.items():
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(text.encode("utf-8"))

    def snapshot(self):
        return {p.relative_to(self.root).as_posix(): p.read_bytes()
                for p in self.root.rglob("*") if p.is_file()}

    def cli(self, *extra):
        return subprocess.run(
            [sys.executable, "-B", str(SCRIPT), "--source-root", str(self.root), *extra],
            capture_output=True, text=True, encoding="utf-8", timeout=10,
        )

    def test_preview_reports_exact_scope_without_writing(self):
        before = self.snapshot()
        result = self.cli()
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        report = json.loads(result.stdout)
        self.assertEqual(report["status"], "PREVIEW")
        self.assertEqual(report["change_count"], 5)
        self.assertEqual(len(report["files"]), 3)
        self.assertEqual(self.snapshot(), before)
        for row in report["files"]:
            self.assertEqual(row["before_sha256"], hashlib.sha256(before[row["path"]]).hexdigest())
            self.assertNotEqual(row["before_sha256"], row["after_sha256"])
            self.assertTrue(row["changes"])

    def test_explicit_apply_preserves_other_text_and_reports_actual_hashes(self):
        outside = self.root / "not_a_target.txt"
        outside.write_bytes(b"must remain untouched\r\n")
        # A generic text patch must retain the selected copy's line endings.
        for relative in FIXTURES:
            p = self.root / relative
            p.write_bytes(p.read_bytes().replace(b"\n", b"\r\n"))
        before = self.snapshot()
        preview = self.cli()
        self.assertEqual(preview.returncode, 0, preview.stdout + preview.stderr)
        result = self.cli("--apply")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["status"], "APPLIED")
        self.assertEqual(report["files"], json.loads(preview.stdout)["files"])
        after = self.snapshot()
        self.assertEqual(set(after), set(before))
        self.assertEqual(after["not_a_target.txt"], before["not_a_target.txt"])
        for row in report["files"]:
            actual = after[row["path"]]
            self.assertEqual(hashlib.sha256(actual).hexdigest(), row["after_sha256"])
            self.assertTrue(actual.startswith(b"# unrelated first line\r\n"))
            self.assertTrue(actual.endswith(b"# unrelated last line\r\n"))
            self.assertNotIn(b"\n", actual.replace(b"\r\n", b""))
        backbone = after["detectron2/modeling/backbone/__init__.py"]
        self.assertIn(b"from .fpn import FPN, SCJE_FPN\r\n", backbone)
        self.assertNotIn(b"from .fan", backbone)
        self.assertNotIn(b"detectron2_backbone", backbone)
        self.assertIn(b"from fvcore.nn import sigmoid_focal_loss_jit\r\n",
                      after["diffsarshipinst/loss.py"])
        self.assertIn(b"interp=Image.Resampling.BILINEAR",
                      after["detectron2/data/transforms/transform.py"])

    def test_missing_or_duplicate_original_line_refuses_before_any_write(self):
        last = self.root / "detectron2/data/transforms/transform.py"
        original = last.read_bytes()
        signature = b"    def __init__(self, src_rect, output_size, interp=Image.LINEAR, fill=0):\n"
        for invalid in (original.replace(signature, b"# signature missing\n"), original + signature):
            with self.subTest(content=invalid):
                last.write_bytes(invalid)
                before = self.snapshot()
                result = self.cli("--apply")
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertEqual(json.loads(result.stdout)["status"], "REFUSED")
                self.assertEqual(self.snapshot(), before)

    def test_already_applied_tree_is_refused_without_additional_writes(self):
        first = self.cli("--apply")
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        before = self.snapshot()
        repeated = self.cli("--apply")
        self.assertEqual(repeated.returncode, 2, repeated.stdout + repeated.stderr)
        self.assertEqual(json.loads(repeated.stdout)["status"], "REFUSED")
        self.assertEqual(self.snapshot(), before)


if __name__ == "__main__":
    unittest.main()
