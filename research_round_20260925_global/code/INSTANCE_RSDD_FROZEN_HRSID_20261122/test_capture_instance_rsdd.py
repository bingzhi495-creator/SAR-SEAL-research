"""Image-only Instance-RSDD roster and native box contracts on synthetic inputs."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import unittest

import numpy as np

spec = spec_from_file_location("rsdd_capture", Path(__file__).with_name("capture_instance_rsdd.py"))
capture = module_from_spec(spec)
spec.loader.exec_module(capture)


class NativeInputs(unittest.TestCase):
    def test_preserves_original_ids_order_and_dimensions(self):
        def row(id_, name):
            return dict(image_id=id_,source_image_name=name,original_hw=[31,47],
                        image_ref=dict(path="images/"+name,sha256="a"*64,bytes=128),
                        rgb_sha256="b"*64,acquisition_group=12,sensor="GF-3")
        author = {"images":[dict(id=9,file_name="1_2_3.jpg",height=31,width=47),
                            dict(id=0,file_name="1_2_4.jpg",height=31,width=47)]}
        manifest = {"expected_ids":[9,0],"items":[row(9,"1_2_3.jpg"), row(0,"1_2_4.jpg")]}
        self.assertEqual(capture.roster(author,manifest,2),manifest["items"])
        manifest["items"][1]["image_ref"]["path"] = "../1_2_4.jpg"
        with self.assertRaisesRegex(ValueError, "native image path"):
            capture.roster(author,manifest,2)

    def test_boxes_scale_width_and_height_independently(self):
        boxes = np.array([[.5,.5,1.2,.5],[.25,.25,.5,.5]], dtype=np.float32)
        scaled = capture.native_xyxy(boxes,(32,64))
        np.testing.assert_array_equal(scaled,[[0,8,64,24],[0,0,32,16]])


if __name__ == "__main__":
    unittest.main()
