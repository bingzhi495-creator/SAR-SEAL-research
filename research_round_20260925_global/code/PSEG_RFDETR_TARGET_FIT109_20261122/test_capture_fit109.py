"""Image-only selection and original candidate-offset unit checks."""
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


spec = spec_from_file_location("fit_capture",Path(__file__).with_name("capture_fit109.py"))
capture = module_from_spec(spec)
spec.loader.exec_module(capture)


class CaptureInputs(unittest.TestCase):
    def test_unique_rgb_representatives_use_minimum_numeric_ids_and_ranked_hashes(self):
        rows = [dict(image_id=i,rgb_sha256=k*64) for i,k in ((30,"a"),(3,"b"),(2,"a"),(5,"c"),(4,"b"))]
        order = sorted((("a",2),("b",3),("c",5)),key=lambda x:
                       sha256(("PSEG-SSDD-FIT109-20261122-v1:"+x[0]*64).encode()).digest())
        selected, groups = capture.select_rgb_groups(rows,2)
        self.assertEqual(groups,3)
        self.assertEqual([x["image_id"] for x in selected],[k[1] for k in order[:2]])
        with self.assertRaisesRegex(ValueError,"Insufficient"):
            capture.select_rgb_groups(rows,4)

    def test_empty_rank_block_keeps_exact_offset_and_image_order(self):
        def infer(item):
            image_id = item[2]
            ranks = [] if image_id == 12 else [5,9]
            rows = [dict(image_id=image_id,category_id=1,score=.9-r/100) for r in ranks]
            return rows, dict(raw_query_count=100,native_count=100,retained_native_ranks=ranks)

        with TemporaryDirectory() as temporary:
            root = Path(temporary)/"exports"
            s = [(None,None,i,(23,37)) for i in (12,8,22)]
            report = capture.write_population(s,infer,output_directory=root,checkpoint_sha256="a"*64,
                                              recipe_sha256="b"*64,event_sink=None)
            manifest = json.loads((root/"IMAGE_MANIFEST.json").read_bytes())
            rows = json.loads((root/"PREDICTIONS.json").read_bytes())
            self.assertEqual(manifest["prediction_count"],4)
            self.assertEqual([x["prediction_start"] for x in manifest["images"]],[0,0,2])
            self.assertEqual([x["image_id"] for x in manifest["images"]],[12,8,22])
            self.assertEqual([x["image_id"] for x in rows],[8,8,22,22])
            self.assertEqual(manifest["predictions_ref"],report["predictions_ref"])

    def test_label_paths_are_rejected(self):
        for name in ("_annotations.coco.json","FIT741_FLOAT_POLYGON_ONLY_PRIVATE.json"):
            with self.assertRaises(PermissionError):
                capture.no_label_files("open",(name,))
        capture.no_label_files("open",("manifest.json",))


if __name__ == "__main__":
    unittest.main()
