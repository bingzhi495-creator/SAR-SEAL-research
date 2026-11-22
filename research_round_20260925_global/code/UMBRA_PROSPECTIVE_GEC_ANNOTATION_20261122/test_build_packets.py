"""Generated closed-intake and corruption checks for the blind archive builder."""
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from zipfile import ZipFile

from PIL import Image

SRC = Path(__file__).resolve().parent
COHORT = "1" * 64


def digest(raw):
    return sha256(raw).hexdigest()


def put(path, value):
    raw = json.dumps(value).encode()
    path.write_bytes(raw)
    return digest(raw)


class PacketTest(unittest.TestCase):
    def test_only_two_isolated_exact_archives(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            images = root / "images"
            images.mkdir()
            provenance = []
            for i in range(1, 9):
                im = Image.new("L", (1024, 1024), i)
                buf = BytesIO()
                im.save(buf, format="PNG")
                raw = buf.getvalue()
                identifier = f"UMB-{i:03d}"
                (images / (identifier + ".png")).write_bytes(raw)
                provenance.append(dict(window_id=identifier, status="complete", native_shape_yx=[1024, 1024],
                                       native_dtype="|u1", png_sha256=digest(raw),
                                       native_array_sha256=digest(im.tobytes()), private_geometry="EXCLUDED"))
            source = put(root / "PROVENANCE_PRIVATE.json", dict(manifest_sha256=COHORT, results=provenance))
            score = put(root / "RESULT_SAFE.json", dict(manifest_sha256=COHORT, provenance_sha256=source,
                        verified_native_windows=8, decoded_unique_windows=8))
            put(root / "TERMINAL.json", dict(actual_exit=0, output_footer_verified=True, result_sha256=score))
            def build(target, expected):
                cmd = [sys.executable, "-I", "-B", "-X", "utf8", str(SRC / "build_packets.py"), "--intake", str(root),
                       "--terminal", str(root / "TERMINAL.json"), "--template", str(SRC / "annotate.html"),
                       "--instructions", str(SRC / "REVIEWER_GUIDE.md"), "--output", str(root / target),
                       "--result-sha256", expected, "--manifest-sha256", COHORT]
                return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
            failed = build("failed", "0" * 64)
            self.assertNotEqual(failed.returncode, 0)
            self.assertFalse((root / "failed").exists())
            passed = build("passed", score)
            self.assertEqual(passed.returncode, 0, passed.stderr)
            for role in "AB":
                with ZipFile(root / "passed" / f"UMBRA_SAR_REVIEWER_{role}.zip") as z:
                    content = z.read("打开判读.html").decode()
                    self.assertIn(f'data-reviewer="{role}"', content)
                    self.assertNotIn("EXCLUDED", content)
                    self.assertEqual(len(json.loads(z.read("ROSTER.json"))["images"]), 8)
                    self.assertEqual(len(z.namelist()), 11)
                    self.assertIsNone(z.testzip())
                    self.assertEqual(z.read("images/UMB-001.png"), (images / "UMB-001.png").read_bytes())


if __name__ == "__main__":
    unittest.main()

