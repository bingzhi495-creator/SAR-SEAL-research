"""Synthetic whole-mask native writeback checks."""
import argparse
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import unittest

import numpy as np
from pycocotools import mask as codec


HERE = Path(__file__).resolve().parent
CODE = None


def load_worker():
    spec = spec_from_file_location("umbra_whole_verifier", HERE / "verify_whole.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rle(bitmap):
    value = codec.encode(np.asfortranarray(bitmap, dtype=np.uint8))
    return dict(size=value["size"], counts=value["counts"].decode("ascii"))


class WholeReadback(unittest.TestCase):
    def test_zero_output_and_nonempty_position_holes(self):
        worker = load_worker()
        scorer = worker.load_native_decoder(CODE)
        stock = np.zeros(worker.HW, dtype=np.uint8)
        stock[40:48, 40:48] = 1
        native = rle(stock)
        empty = rle(np.zeros_like(stock))
        request = dict(ordinal=1, image_id=1, native_candidate_count=6, candidates=[
            dict(position=2, segmentation=native), dict(position=6, segmentation=native)])
        returned = dict(ordinal=1, image_id=1, lvis=[
            dict(position=2, segmentation=empty), dict(position=6, segmentation=native)])
        counts = worker.verify_row(scorer, request, returned)
        self.assertEqual(counts["empty_outputs"], 1)
        self.assertEqual(counts["unchanged_outputs"], 1)
        self.assertEqual(counts["drops"], 64)
        self.assertEqual(counts["pass_through"], 4)
        returned["lvis"][0]["position"] = 1
        with self.assertRaisesRegex(ValueError, "positions"):
            worker.verify_row(scorer, request, returned)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code", type=Path, required=True)
    args = parser.parse_args()
    CODE = args.code
    unittest.main(argv=[__file__], verbosity=2)
