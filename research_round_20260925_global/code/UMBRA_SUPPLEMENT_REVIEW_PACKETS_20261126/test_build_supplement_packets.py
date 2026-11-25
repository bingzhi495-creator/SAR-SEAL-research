"""Generated-only end-to-end packet fixtures."""
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from zipfile import ZipFile

from PIL import Image


def digest(raw):
    return sha256(raw).hexdigest()


def put(path, content):
    raw = json.dumps(content).encode()
    path.write_bytes(raw)
    return digest(raw)


def make_fixture(folder, count=18, unavailable=()):
    intake = folder / "intake"
    images = intake / "images"
    images.mkdir(parents=True)
    rows = []
    for i in range(count):
        identifier = f"UMB-{101 + i:03d}"
        if 101 + i in unavailable:
            rows.append(dict(window_id=identifier, status="failed"))
            continue
        png = BytesIO()
        gray = Image.new("L", (1024, 1024), i)
        gray.save(png, format="PNG")
        image = png.getvalue()
        (images / f"{identifier}.png").write_bytes(image)
        rows.append(dict(window_id=identifier, status="complete", native_shape_yx=[1024, 1024],
                         native_dtype="|u1", png_sha256=digest(image), native_array_sha256=digest(gray.tobytes()),
                         private_coordinate_record="NEVER_INCLUDE"))
    verified = count - len(unavailable)
    extract_exit = int(bool(unavailable))
    manifest = folder / "cohort.json"
    manifest_sha = put(manifest, {"schema": "umbra_gec_supplement_v1", "items": [{"window_id": r["window_id"]} for r in rows]})
    provenance = put(intake / "PROVENANCE_PRIVATE.json", {"manifest_sha256": manifest_sha, "results": rows})
    intake_result = put(intake / "RESULT_SAFE.json", {
        "requested_windows": count, "verified_native_windows": verified, "decoded_unique_windows": verified,
        "failed_window_count": len(unavailable), "manifest_sha256": manifest_sha, "provenance_sha256": provenance,
        "status": "UMBRA_SUPPLEMENT_NATIVE_WINDOWS_PARTIAL" if unavailable else "UMBRA_SUPPLEMENT_NATIVE_WINDOWS_COMPLETE", "native_window_shape_yx": [1024, 1024],
        "annotation_files": 0, "detector_forwards": 0, "optimizer_updates": 0,
    })
    readback = folder / "readback.json"
    readback_sha = put(readback, {
        "status": "UMBRA_SUPPLEMENT_NATIVE_READBACK_COMPLETE", "manifest_sha256": manifest_sha,
        "extraction_result_sha256": intake_result, "extraction_actual_exit": extract_exit, "requested_windows": count,
        "readback_windows": verified, "unique_pixel_arrays": verified, "matching_uncompressed_ranges": 4 * verified,
        "source_coords_exposed": False, "annotations_read": 0, "detector_forwards": 0, "optimizer_updates": 0, "native_window_shape_yx": [1024, 1024], "eligible_for_detector": verified >= 18, "failed_window_count": len(unavailable),
    })
    lock = {"manifest_sha256": manifest_sha}
    for label, phase in (("intake", "native-window-intake"), ("readback", "native-window-readback")):
        receipt_sha = intake_result if label == "intake" else readback_sha
        worker = folder / f"{label}_worker.py"
        worker.write_text(f"# Synthetic fixture agent for {label}\n", encoding="utf-8")
        worker_sha = digest(worker.read_bytes())
        terminal = folder / f"{label}_terminal.json"
        terminal_sha = put(terminal, {
            "phase": phase, "actual_exit": extract_exit if label == "intake" else 0, "worker_sha256": worker_sha,
            "result_sha256": receipt_sha, "output_footer_verified": True,
            "started_utc": "2026-11-25T17:00:00+00:00" if label == "intake" else "2026-11-25T17:01:00+00:00",
            "ended_utc": "2026-11-25T17:00:05+00:00" if label == "intake" else "2026-11-25T17:01:05+00:00",
        })
        lock[label] = dict(phase=phase, terminal_sha256=terminal_sha, result_sha256=receipt_sha, worker_sha256=worker_sha)
        if label == "intake":
            receipt = json.loads(readback.read_bytes())
            receipt["extraction_terminal_sha256"] = terminal_sha
            readback_sha = put(readback, receipt)
    return folder, lock


class PacketTest(unittest.TestCase):
    def run_builder(self, workspace, lock):
        path = workspace / "ADMISSION.json"
        lock_sha = put(path, lock)
        output = workspace / "packets"
        arguments = {
            "base": BASE, "prelock": SRC / "PRELOCK.md", "admission": path, "admission-sha256": lock_sha,
            "manifest": workspace / "cohort.json", "intake": workspace / "intake",
            "intake-terminal": workspace / "intake_terminal.json", "intake-worker": workspace / "intake_worker.py",
            "readback-result": workspace / "readback.json", "readback-terminal": workspace / "readback_terminal.json",
            "readback-worker": workspace / "readback_worker.py", "output": output,
        }
        command = [sys.executable, "-I", "-B", "-X", "utf8", str(SRC / "build_supplement_packets.py")]
        for key, value in arguments.items():
            command.extend(("--" + key, str(value)))
        return subprocess.run(command, capture_output=True, encoding="utf-8"), output

    def test_exact_cohort_and_separate_roles_and_exports(self):
        with TemporaryDirectory() as temp:
            workspace, lock = make_fixture(Path(temp), count=24)
            done, output = self.run_builder(workspace, lock)
            self.assertEqual(done.returncode, 0, done.stderr)
            result = json.loads((output / "RESULT_SAFE.json").read_bytes())
            footer = json.loads(done.stdout)
            self.assertEqual(footer["result_sha256"], digest((output / "RESULT_SAFE.json").read_bytes()))
            self.assertEqual(result["windows_per_packet"], 24)
            self.assertEqual(result["labels_present"], 0)
            self.assertEqual(result["model_outputs_present"], 0)
            rosters = []
            for role in ("A", "B"):
                archive = output / f"UMBRA_SUPPLEMENT_REVIEWER_{role}.zip"
                with ZipFile(archive) as z:
                    self.assertIsNone(z.testzip())
                    self.assertEqual(set(z.namelist()), {"打开判读.html", "判读说明.md", "ROSTER.json",
                                                       *(f"images/UMB-{i:03d}.png" for i in range(101, 125))})
                    app = z.read("打开判读.html").decode()
                    self.assertIn(f'data-reviewer="{role}"', app)
                    self.assertIn('a.download="SAR_ship_supplement_review_"+reviewer+".json"', app)
                    self.assertNotIn("SAR_ship_review_", app)
                    self.assertNotIn("NEVER_INCLUDE", app)
                    self.assertEqual(z.read("images/UMB-124.png"),
                                     (workspace / "intake/images/UMB-124.png").read_bytes())
                    rosters.append(z.read("ROSTER.json"))
            self.assertEqual(rosters[0], rosters[1])
            self.assertEqual((output / "ROSTER.json").read_bytes(), rosters[0])
            self.assertEqual(digest(rosters[0]), result["anonymous_roster_sha256"])

    def test_partial_closure_keeps_original_omission_and_ids(self):
        with TemporaryDirectory() as temp:
            workspace, lock = make_fixture(Path(temp), count=19, unavailable=(104,))
            done, output = self.run_builder(workspace, lock)
            self.assertEqual(done.returncode, 0, done.stderr)
            self.assertEqual(json.loads((output / "RESULT_SAFE.json").read_bytes())["unavailable_windows"], 1)
            with ZipFile(output / "UMBRA_SUPPLEMENT_REVIEWER_A.zip") as z:
                self.assertNotIn("images/UMB-104.png", z.namelist())
                roster = json.loads(z.read("ROSTER.json"))
                self.assertEqual([r["image_id"] for r in roster["images"]],
                                 [f"UMB-{i:03d}" for i in range(101, 120) if i != 104])

    def test_old_validator_accepts_both_scoped_synthetic_returns(self):
        with TemporaryDirectory() as temp:
            workspace, lock = make_fixture(Path(temp), count=19, unavailable=(104,))
            done, output = self.run_builder(workspace, lock)
            self.assertEqual(done.returncode, 0, done.stderr)
            with ZipFile(output / "UMBRA_SUPPLEMENT_REVIEWER_A.zip") as z:
                roster_bytes = z.read("ROSTER.json")
            roster_path = workspace / "verified_roster.json"
            roster_path.write_bytes(roster_bytes)
            roster = json.loads(roster_bytes)
            for role in ("A", "B"):
                put(workspace / f"SAR_ship_supplement_review_{role}.json", {
                    "schema": "sar-hull-review-v1", "reviewer": role,
                    "images": [{"image_id": r["image_id"], "width": 1024, "height": 1024,
                                "status": "complete_empty", "notes": "", "modes": ["native"], "shapes": []}
                               for r in roster["images"]],
                })
            cmd = [sys.executable, "-I", "-B", "-X", "utf8", str(BASE / "validate_reviews.py"),
                   "agreement", "--roster", str(roster_path),
                   "--a", str(workspace / "SAR_ship_supplement_review_A.json"),
                   "--b", str(workspace / "SAR_ship_supplement_review_B.json")]
            verdict = subprocess.run(cmd, capture_output=True, encoding="utf-8")
            self.assertEqual(verdict.returncode, 0, verdict.stderr)

    def test_tampered_readback_terminal_fails_before_image_access(self):
        with TemporaryDirectory() as temp:
            workspace, lock = make_fixture(Path(temp))
            (workspace / "readback_terminal.json").write_bytes(b"changed")
            (workspace / "intake/images/UMB-101.png").unlink()
            done, output = self.run_builder(workspace, lock)
            self.assertNotEqual(done.returncode, 0)
            self.assertIn("input bytes differ", done.stderr)
            self.assertFalse(output.exists())

    def test_tampered_pixel_file_creates_no_packet(self):
        with TemporaryDirectory() as temp:
            workspace, lock = make_fixture(Path(temp))
            (workspace / "intake/images/UMB-101.png").write_bytes(b"changed")
            done, output = self.run_builder(workspace, lock)
            self.assertNotEqual(done.returncode, 0)
            self.assertIn("input bytes differ", done.stderr)
            self.assertFalse(output.exists())

    def test_below_minimum_creates_no_packet(self):
        with TemporaryDirectory() as temp:
            workspace, lock = make_fixture(Path(temp), count=18, unavailable=(103,))
            done, output = self.run_builder(workspace, lock)
            self.assertNotEqual(done.returncode, 0)
            self.assertIn("eighteen unique verified", done.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    SRC = Path(__file__).resolve().parent
    if len(sys.argv) != 3 or sys.argv[1] != "--base":
        raise SystemExit("Supply --base ORIGINAL_ANNOTATION_CODE")
    BASE = Path(sys.argv.pop()).resolve()
    sys.argv.pop()
    unittest.main()
