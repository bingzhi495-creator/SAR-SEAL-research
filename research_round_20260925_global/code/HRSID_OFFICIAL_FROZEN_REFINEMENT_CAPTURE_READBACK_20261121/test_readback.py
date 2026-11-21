"""Generated probabilities and masks only."""

from collections import Counter
import copy
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import zlib

import numpy as np
from pycocotools import mask as codec


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("synthetic_reader", HERE / "readback.py")
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)


def rle(stock):
    encoded = codec.encode(np.asfortranarray(stock, dtype=np.uint8))
    return dict(size=encoded["size"], counts=encoded["counts"].decode())


def fixture():
    # Two original query slots map to category 0. Others in top-100 map to unmapped slot 1.
    joint = np.zeros((164, 2), dtype="<f4")
    joint[:, 1] = .4
    joint[[3, 8], 0] = .8
    image_id = 91
    stock = np.zeros((800, 800), dtype=bool)
    stock[0, 0:50] = True
    another = np.zeros_like(stock)
    another[0, 1] = True
    rows = [dict(image_id=image_id, category_id=1, score=float(joint[3, 0]), segmentation=rle(a))
            for a in (stock, another)]
    native = dict(image_id=image_id, raw_query_count=164, native_count=100, unmapped_slot1_count=98,
                  native_ranks=[1, 2], coco=rows)
    line = reader.encode(native)
    coords = [[50, 0], [51, 1], [52, 2]]
    small = np.array([[.8, .95], [.98, .65], [.49, .55]], dtype="<f4")
    large = np.array([[.99, .70], [.97, .66], [.91, .53]], dtype="<f4")
    # These choices are determined separately from the code being tested.
    indices = {"SF":[51, 1], "SU=F":[51, 2], "SU":[51, 2, 50, 1], "LW=S":[50, 51, 52, 2]}
    scores = {s:dict(S=small.tobytes().hex(), L=large.tobytes().hex()) for s in reader.SEEDS}
    cand = dict(rank=1, query_id=3, area=int(stock.sum()), score=rows[0]["score"], coordinates=coords,
                edits={}, feature_sha256="a" * 64, point_scores_f32le_hex=scores)
    predictions = {}
    for s in reader.SEEDS:
        cand["edits"][s] = {}
        predictions[s] = {}
        for arm, pixels in indices.items():
            dest = stock.copy()
            dest.flat[pixels] = ~stock.flat[pixels]
            out = copy.deepcopy(native)
            out["coco"][0]["segmentation"] = rle(dest)
            predictions[s][arm] = out
            q = large if arm == "LW=S" else small
            score = {p: float(q[i, j] - np.float32(.5) if j == 0 else np.float32(.5)-q[i, j])
                     for i, pair in enumerate(coords) for j, p in enumerate(pair)}
            gain = [score[p] for p in pixels]
            swaps = [gain[i] + gain[i+1] for i in range(0, len(pixels), 2)]
            actions = gain if arm == "LW=S" else swaps
            drops = sum(p < 50 for p in pixels)
            cand["edits"][s][arm] = dict(flipped=pixels, edits=len(pixels), adds=len(pixels)-drops,
                drops=drops, area_delta=len(pixels)-2*drops, actions=len(actions),
                action_kind="flip" if arm == "LW=S" else "swap",
                nonpositive_actions=sum(g <= 0 for g in actions), nonpositive_flips=sum(g <= 0 for g in gain),
                predicted_gain_sum=sum(gain))
    w = dict(image_id=image_id, mapped=2, supported=1, eligible=1, native_line_sha256=reader.digest(line),
             joint_scores_f32le_hex=joint.tobytes().hex(), candidates=[cand,
             dict(rank=2, query_id=8, area=1, score=rows[1]["score"])])
    times = dict(detector_forward_seconds=.125, native_postprocess_parity_seconds=.25, refinement_cpu_seconds=.5)
    return line, dict(index_zero_based=0, image_id=image_id, stock_block_sha256=reader.digest(line),
                      predictions=predictions, witness=w, timings=times)


class Readback(unittest.TestCase):
    def test_closed_streams(self):
        line, block = fixture()
        packed = zlib.compress(reader.encode(block), 1)
        result = dict(image_row_bytes=len(packed), image_rows_sha256=reader.digest(sha256(packed).digest()),
                      phase_seconds=block["timings"])
        baseline = dict(images=1, mapped_rows=2, prediction_bytes=len(line))
        with TemporaryDirectory(dir=HERE) as name:
            root = Path(name)
            rows = root / "rows"
            rows.mkdir()
            (rows / "00000.json.zlib").write_bytes(packed)
            stock = root / "native.jsonl"
            stock.write_bytes(line)
            out = root / "verified"
            out.mkdir()
            refs, paths, totals, _ = reader.replay(rows, stock, result, baseline, out, native_sha=reader.digest(line))
            self.assertEqual(totals["mapped"], 2)
            for s in reader.SEEDS:
                for arm in reader.ARMS:
                    data = (out / paths[s][arm]).read_bytes()
                    self.assertEqual(refs[s][arm], dict(bytes=len(data), sha256=reader.digest(data)))
                    self.assertEqual(json.loads(data), block["predictions"][s][arm])
            self.assertEqual(totals["5217/LW=S/area_delta"], 2)
            result["image_rows_sha256"] = "0" * 64
            invalid = root / "invalid"
            invalid.mkdir()
            with self.assertRaisesRegex(ValueError, "aggregate"):
                reader.replay(rows, stock, result, baseline, invalid, native_sha=reader.digest(line))

    def test_tampering(self):
        line, original = fixture()
        for label in ("query", "sign", "budget", "raster", "posterior", "rank", "score", "omission", "noop"):
            with self.subTest(label=label):
                block = copy.deepcopy(original)
                c = block["witness"]["candidates"][0]
                if label == "query":
                    c["query_id"] = 5
                elif label == "sign":
                    c["coordinates"][0].reverse()
                elif label == "budget":
                    c["edits"]["5217"]["LW=S"]["edits"] += 1
                elif label == "raster":
                    block["predictions"]["5217"]["SU"]["coco"][0]["segmentation"] = json.loads(line)["coco"][0]["segmentation"]
                elif label == "posterior":
                    c["point_scores_f32le_hex"]["5217"]["S"] = c["point_scores_f32le_hex"]["5217"]["L"]
                elif label == "rank":
                    block["predictions"]["5217"]["SU"]["native_ranks"].reverse()
                elif label == "score":
                    c["score"] += 1e-7
                elif label == "omission":
                    del block["predictions"]["5219"]["SU"]
                else:
                    block["predictions"]["5217"]["SF"]["coco"][1]["segmentation"] = block["predictions"]["5217"]["SU"]["coco"][0]["segmentation"]
                with self.assertRaises(ValueError):
                    reader.validate_image(block, 0, line, Counter())


if __name__ == "__main__":
    unittest.main()
