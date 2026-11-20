"""Only synthetic ZIP/PNG/COCO inputs; never opens an original dataset."""
import importlib.util
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from zipfile import ZipFile

import numpy as np
import PIL
from PIL import Image

HERE = Path(__file__).resolve().parent


def load_audit():
    spec = importlib.util.spec_from_file_location("test_input_audit", HERE / "test_input_audit.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ScopeTests(unittest.TestCase):
    def test_runtime_records_imported_library_versions(self):
        audit = load_audit()
        self.assertEqual(audit.runtime_versions()["Pillow"], PIL.__version__)
        self.assertEqual(audit.runtime_versions()["numpy"], np.__version__)

    def test_wrong_phase_is_rejected_before_archive_access(self):
        audit = load_audit()
        frozen = dict(status=audit.FREEZE_STATUS, phase="DEV705", endpoint_freeze_complete=True,
                      endpoints_sha256={"center_e24": "1" * 64})
        admission = dict(status=audit.ADMISSION_STATUS, phase=audit.PHASE,
                         endpoint_freeze_sha256="2" * 64, input_audit_only=True)
        with self.assertRaisesRegex(ValueError, "phase"):
            audit.audit_archive(None, None, None, frozen, admission, "2" * 64, {})

    def test_cli_wrong_phase_does_not_open_unavailable_archive(self):
        audit = load_audit()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            freeze = dict(status=audit.FREEZE_STATUS, phase="DEV705", endpoint_freeze_complete=True,
                          endpoints_sha256={"center_e24":"1"*64})
            raw = json.dumps(freeze).encode()
            (root/"freeze.json").write_bytes(raw)
            admission = dict(status=audit.ADMISSION_STATUS, phase=audit.PHASE,
                             endpoint_freeze_sha256=sha256(raw).hexdigest(), input_audit_only=True)
            proof = json.dumps(admission).encode()
            (root/"admission.json").write_bytes(proof)
            command = [sys.executable,"-I","-B","-X","utf8",str(HERE/"test_input_audit.py"),
                       "--freeze",str(root/"freeze.json"),"--freeze-sha256",sha256(raw).hexdigest(),
                       "--admission",str(root/"admission.json"),"--admission-sha256",sha256(proof).hexdigest(),
                       "--archive",str(root/"unavailable.zip"),"--fit-helper",str(root/"unavailable.py"),
                       "--fit-helper-sha256","3"*64,"--bridges",str(root/"unavailable.json"),
                       "--bridges-sha256","4"*64,"--output",str(root/"new_output")]
            result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
            self.assertNotEqual(result.returncode,0)
            self.assertIn("phase",result.stderr)
            self.assertNotIn("FileNotFoundError",result.stderr)
            self.assertFalse((root/"new_output").exists())


def png(value, side=800, compression=6):
    output = BytesIO()
    Image.fromarray(np.full((side, side, 3), value, np.uint8), "RGB").save(
        output, format="PNG", compress_level=compression)
    return output.getvalue()


class SyntheticArchiveTests(unittest.TestCase):
    def setUp(self):
        self.audit = load_audit()
        helper_path = Path(os.environ["HRSID_AUDIT_HELPER"])
        self.helper = self.audit.load_helper(helper_path, sha256(helper_path.read_bytes()).hexdigest())
        self.frozen = dict(status=self.audit.FREEZE_STATUS, phase=self.audit.PHASE,
                           endpoint_freeze_complete=True, endpoints_sha256={"center_e24": "1" * 64})
        self.admission = dict(status=self.audit.ADMISSION_STATUS, phase=self.audit.PHASE,
                              endpoint_freeze_sha256="2" * 64, input_audit_only=True)

    def fixture(self, *, side=800, missing=False, annotations=2, invalid=False, image_count=2):
        images = [dict(id=i, file_name=f"P{i}_synthetic.jpg", width=800, height=800)
                  for i in (101, 102)]
        rows = [dict(id=1, image_id=101, category_id=1, iscrowd=0, area=16, bbox=[1,1,4,4],
                     segmentation=[[1,1,5,1,5,5,1,5]]),
                dict(id=2, image_id=102, category_id=1, iscrowd=0, area=3, bbox=[10,10,3,1],
                     segmentation=[[10,10,11,10,12,10]])]
        gt = dict(images=images[:image_count], annotations=rows[:annotations], categories=[dict(id=1, name="ship")])
        if invalid:
            gt["annotations"][1]["segmentation"] = [[10,10,11,10,12]]
        payloads = [png(17, side), png(23)]
        output = BytesIO()
        with ZipFile(output, "w") as z:
            z.writestr("annotations/test2017.json", json.dumps(gt).encode())
            for i, raw in zip((101,102), payloads):
                if not (missing and i == 102):
                    z.writestr(f"images/P{i}_synthetic.png", raw)
            central_offset = z.start_dir
        raw = output.getvalue()
        central_bytes = len(raw) - central_offset - 22
        contract = self.audit.Contract(len(raw), sha256(raw).hexdigest(), central_offset,
                    central_bytes, sha256(raw[central_offset:central_offset+central_bytes]).hexdigest(),
                    1 if missing else 2, 1, 0, 2, 2, {"FIT":1, "DEV":1})
        references = {
            "FIT":[dict(image_id=201, encoded_sha256=sha256(payloads[0]).hexdigest(),
                        decoded_rgb_sha256=sha256(np.full((side,side,3),17,np.uint8).tobytes()).hexdigest())],
            "DEV":[dict(image_id=202, encoded_sha256=sha256(png(23,compression=0)).hexdigest(),
                        decoded_rgb_sha256=sha256(np.full((800,800,3),23,np.uint8).tobytes()).hexdigest())]}
        return raw, contract, references, gt

    def run_fixture(self, **options):
        raw, contract, references, gt = self.fixture(**options)
        result = self.audit.audit_archive(BytesIO(raw), self.helper, contract, self.frozen,
                                          self.admission, "2"*64, references)
        return result, gt

    def test_complete_real_png_and_polygon_audit_preserves_empty_original(self):
        result, original = self.run_fixture()
        report = result["report"]
        self.assertTrue(report["structural_input_qualified"])
        self.assertEqual((report["decoded_images"], report["raster_attempts"],
                          report["empty_rasters"], report["invalid_rasters"]), (2,2,1,0))
        self.assertEqual(result["original_gt"], original)
        self.assertEqual(report["cross_split_identity"]["FIT"]["encoded_sha256"]["test_images_matched"],1)
        self.assertEqual(report["cross_split_identity"]["DEV"]["encoded_sha256"]["test_images_matched"],0)
        self.assertEqual(report["cross_split_identity"]["DEV"]["decoded_rgb_sha256"]["test_images_matched"],1)
        self.assertEqual(report["model_forward_calls"],0)
        self.assertFalse(report["AP"])

    def test_wrong_decoded_size_fails_qualification_without_changing_original_gt(self):
        result, original = self.run_fixture(side=32)
        self.assertFalse(result["report"]["structural_input_qualified"])
        self.assertEqual(result["report"]["image_check_failures"],1)
        self.assertEqual(result["report"]["raster_attempts"],2)
        self.assertEqual(result["original_gt"],original)

    def test_missing_original_member_is_rejected(self):
        with self.assertRaisesRegex(ValueError,"Missing original TEST PNG"):
            self.run_fixture(missing=True)

    def test_wrong_original_annotation_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError,"original TEST image and annotation counts"):
            self.run_fixture(annotations=1)

    def test_wrong_original_image_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError,"original TEST image and annotation counts"):
            self.run_fixture(image_count=1)

    def test_invalid_polygon_is_reported_separately_from_empty_and_preserved(self):
        result, original = self.run_fixture(invalid=True)
        self.assertFalse(result["report"]["structural_input_qualified"])
        self.assertEqual((result["report"]["invalid_rasters"],result["report"]["empty_rasters"]),(1,0))
        self.assertEqual(result["original_gt"],original)

    def test_changed_whole_archive_rejected_before_member_decoding(self):
        raw, contract, references, _ = self.fixture()
        with self.assertRaisesRegex(ValueError,"whole official archive"):
            self.audit.audit_archive(BytesIO(raw+b"x"),self.helper,contract,self.frozen,
                                     self.admission,"2"*64,references)


if __name__ == "__main__":
    unittest.main()
