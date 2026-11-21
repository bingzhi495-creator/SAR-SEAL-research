from copy import deepcopy
from hashlib import sha256
from io import BytesIO
import json
import unittest

import numpy as np
from pycocotools import mask as codec

from verify_export import check_stream, check_terminal, EXPORT_SOURCE


def rle(array):
    result = codec.encode(np.asfortranarray(array, dtype=np.uint8))
    return dict(size=result["size"], counts=result["counts"].decode("ascii"))


class ExportReadbackTests(unittest.TestCase):
    def setUp(self):
        self.manifest = [dict(image_id=i, decoded_rgb_sha256=h*64, height=5, width=7)
                         for i,h in [(15,"a"), (2,"b"), (300,"a")]]
        pixels = np.zeros((5,7), np.uint8)
        pixels[1:4,3:5] = 1
        masks = [rle(pixels), rle(pixels*0)]
        self.blocks = []
        for row in self.manifest:
            i = row["image_id"]
            ranks = [] if i==2 else [2,99]
            predictions = [dict(image_id=i, category_id=1, score=.875, segmentation=m) for m in masks[:len(ranks)]]
            self.blocks.append(dict(image_id=i, native_ranks=ranks, native_count=100, raw_query_count=164,
                                    unmapped_slot1_count=100-len(ranks), coco=predictions))

    def scan(self):
        lines = [json.dumps(b, sort_keys=True).encode() + b"\n" for b in self.blocks]
        raw = b"".join(lines)
        expected = dict(prediction_bytes=len(raw), predictions_sha256=sha256(raw).hexdigest(), mapped_rows=4)
        return check_stream(BytesIO(raw), self.manifest, expected)

    def test_full_order_rank_empty_and_duplicate_geometry(self):
        stats = self.scan()
        self.assertEqual(stats["mapped_rows"], 4)
        self.assertEqual(stats["images_without_predictions"], 1)
        self.assertEqual(stats["empty_masks"], 2)
        self.assertEqual(stats["summed_mask_area"], 12)
        self.assertEqual(stats["identical_duplicate_prediction_pairs"], 1)
        self.assertEqual(stats["unmapped_slots"], 296)

    def test_wrong_manifest_image_order_rejected(self):
        self.blocks.reverse()
        with self.assertRaisesRegex(ValueError, "identity"):
            self.scan()

    def test_duplicate_geometry_change_rejected(self):
        self.blocks[-1]["coco"][0]["segmentation"] = self.blocks[-1]["coco"][1]["segmentation"]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.scan()

    def test_invalid_rle_accounting_and_order_rejected(self):
        for name in ("RLE", "ranks", "score", "category"):
            with self.subTest(name=name):
                old = deepcopy(self.blocks)
                first = self.blocks[0]
                if name == "RLE":
                    first["coco"][0]["segmentation"]["counts"] = "0"
                elif name == "ranks":
                    first["native_ranks"] = [99,2]
                elif name == "score":
                    first["coco"][1]["score"] = .99
                else:
                    first["coco"][0]["category_id"] = 0
                with self.assertRaisesRegex(ValueError, name):
                    self.scan()
                self.blocks = old

    def test_live_or_unbound_terminal_rejected(self):
        receipt = dict(status="HRSID_OFFICIAL_TEST_NATIVE_EXPORT_ORIGINAL_TERMINAL", tool_session_handle=88756,
                       original_process_terminal=True, original_exit=0, result_sha256="a"*64,
                       export_source_sha256=EXPORT_SOURCE, admission_sha256="b"*64)
        check_terminal(receipt)
        for key, bad in (("original_process_terminal",False),("original_exit",1),
                         ("tool_session_handle",3),("export_source_sha256","f"*64)):
            with self.assertRaises(ValueError):
                check_terminal(dict(receipt, **{key:bad}))
