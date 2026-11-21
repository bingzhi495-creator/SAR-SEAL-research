"""Synthetic-only transfer, capture-stream, terminal and support checks."""
from collections import Counter
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from pycocotools import mask as codec


path = Path(__file__).with_name("replay.py")
spec = importlib.util.spec_from_file_location("official_replay", path)
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


def packed(mask):
    x = codec.encode(np.asfortranarray(mask, dtype=np.uint8))
    return dict(size=x["size"], counts=x["counts"].decode("ascii"))


class ReplayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.args = SimpleNamespace()
        cls.m = r.modules(Path(os.environ["TEST_ORDINARY_CODE"]), cls.args)

    def test_native_rectangular_variable_shape_and_independent_support(self):
        m, a = self.m, self.args
        pair, rank = m.native.load_old(a.pair_math, a.rank_math)
        free = m.legacy.load(a.free, m.old.FREE_SOURCE, "_original_free")
        def deterministic_head(_, f, partner, center, scale):
            out = np.zeros(f.shape[:2], dtype=np.float32)
            out[:,0] = .9
            out[:,1] = .1
            return out
        models = pair,rank,free,SimpleNamespace(probabilities=deterministic_head),None,None,None
        for height,width in ((67,99),(99,67)):
            stock = np.zeros((height,width), dtype=np.uint8)
            stock[24:32,27:35] = 1
            row = dict(image_id=17,category_id=1,score=.9,segmentation=packed(stock))
            detail = dict(native_rank=1,score=.9,mask_area=64,box_xyxy=[20.,20.,40.,40.])
            planes = np.full((1,160,160),.5,dtype=np.float32)
            counts = Counter()
            ledger = r.emit(m,a,dict(image_id=17,original_hw=[height,width]),[row],[detail],planes,
                            np.zeros((height,width,3),np.uint8),models,counts)
            changes = ledger["changes"]
            self.assertEqual(set(changes["1"]), set(r.MODES[1:]))
            self.assertEqual(counts["fixed_swaps"], counts["free_matched_count_swaps"])
            self.assertEqual(counts["free_natural_swaps"], counts["native_matched_count_swaps"])
            independent = Counter()
            r.check_support(rank,[row],[detail],changes,independent)
            self.assertTrue(all(independent[k] == counts[k] for k in independent))
            # Preserve area and flip count but move a single addition outside the box.
            original = stock.astype(bool)
            altered = codec.decode(changes["1"]["fixed"]["rle"]).astype(bool)
            altered.reshape(-1)[np.flatnonzero(altered.ravel() & ~original.ravel())[0]] = False
            altered[0,0] = True
            edits = dict(changes["1"]["fixed"], rle=packed(altered))
            with self.assertRaisesRegex(ValueError, "two-pixel support"):
                r.check_support(rank,[row],[detail],{"1":{"fixed":edits}},Counter())

    def test_receipt_is_test_only_and_population_uses_232(self):
        m = self.m
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            root = Path(temp).resolve()
            h = "2"*64
            rows = [dict(image_id=i, original_hw=[200,300],
                         image_ref=dict(path=f"{i:06d}.jpg",bytes=11,sha256=h)) for i in range(232)]
            mf = root / "manifest.json"
            mf.write_text(json.dumps(dict(TEST=True, AP=False, expected_ids=list(range(232)),items=rows)),
                          encoding="utf-8")
            a = SimpleNamespace(manifest=mf,manifest_sha=sha256(mf.read_bytes()).hexdigest(),
                                accepted=root/"acceptance.json")
            receipt = dict(status=r.CAPTURE_ACCEPTED,manifest_sha256=a.manifest_sha,ema_sha256=r.EMA,
                           images=232,probability_files=232,exact_top100_binary_replay=0,probability_rows=0,
                           exact_author_binary_replay=0,post_nms_rows=0,original_author_postprocess_parity=True,
                           job_drained=True,original_os_exit=0,**r.FLAGS)
            receipt.update({k:h for k in ("predictions_sha256","candidate_audit_sha256","probability_index_sha256",
                                          "export_result_sha256","export_terminal_sha256","original_tool_wait_sha256")})
            for actual in (receipt,dict(receipt,TEST=False,DEV=True),dict(receipt,images=187)):
                a.accepted.write_text(json.dumps(actual),encoding="utf-8")
                a.accepted_sha=sha256(a.accepted.read_bytes()).hexdigest()
                if actual is receipt:
                    self.assertEqual(len(r.admit(m,a)[0]),232)
                else:
                    with self.assertRaisesRegex(ValueError,"official TEST232"):
                        r.admit(m,a)

    def test_frozen_native_top100_stream_and_terminal(self):
        m = self.m
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            capture = Path(temp).resolve()
            raw = np.zeros((20,35),np.uint8)
            row = dict(image_id=7,category_id=1,score=.5,segmentation=packed(raw))
            detail = dict(native_rank=1,score=.5,mask_area=0,box_xyxy=[0.,0.,35.,20.])
            (capture/"PROBABILITY_PRIVATE").mkdir()
            plane_path = capture/"PROBABILITY_PRIVATE/0001.npy"
            np.save(plane_path,np.zeros((100,160,160),dtype="<f4"))
            records = [dict(image_id=7,coco=[row]*101),
                       dict(image_id=7,raw_queries=300,internal_top=300,post_nms=101,
                            candidates=[dict(detail,native_rank=i+1) for i in range(101)]),
                       dict(image_id=7,query_ids=list(range(101)),probability=dict(
                            file="PROBABILITY_PRIVATE/0001.npy",rows=100,bytes=plane_path.stat().st_size,
                            sha256=sha256(plane_path.read_bytes()).hexdigest()))]
            receipt = dict(probability_rows=100,post_nms_rows=101)
            for name,key,record in zip((m.author.PREDICTIONS,m.author.AUDIT,"PROBABILITIES_PRIVATE.jsonl"),
                        ("predictions_sha256","candidate_audit_sha256","probability_index_sha256"),records):
                line = m.author.encoded(record)
                (capture/name).write_bytes(line)
                receipt[key]=sha256(line).hexdigest()
            got = list(m.legacy.blocks(SimpleNamespace(capture=capture),m.author,
                                       [dict(image_id=7,original_hw=[20,35])],receipt,True))
            self.assertEqual(len(got[0][1]),100)
            self.assertEqual(got[0][3].dtype,np.dtype("<f4"))
        terminal=dict(status="PSEG_TEST232_REPLAY_TERMINAL_OBSERVED",original_process_terminal=True,
                      original_exit=0,result_sha256="c"*64,tool_session_handle=2)
        r.check_terminal(terminal,"c"*64)
        with self.assertRaisesRegex(ValueError,"observed replay exit"):
            r.check_terminal(dict(terminal,original_exit=1),"c"*64)


if __name__ == "__main__":
    unittest.main()
