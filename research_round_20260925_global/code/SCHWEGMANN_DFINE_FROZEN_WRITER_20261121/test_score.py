"""Generated-only tests. Inputs are synthetic arrays, with no access to projects' observation data."""

import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("writer_score", HERE / "score.py")
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
reference, intake = (Path(v) for v in sys.argv[1:3])
sys.argv[:] = sys.argv[:1]
historic = s.module(reference / "score_schwegmann.py", s.HISTORIC_SCORE)
lex = s.module(intake, s.INTAKE)


def sample_mask(start=40, size=6):
    mask = np.zeros((101, 101), dtype=bool)
    mask[start:start+size, 42:42+size] = True
    return mask


def encode(mask):
    flat = mask.reshape(-1, order="F")
    changed = np.where(flat[1:] != flat[:-1])[0] + 1
    lengths = np.diff(np.r_[0, changed, len(flat)]).tolist()
    if flat[0]:
        lengths.insert(0, 0)
    codes = []
    for i, value in enumerate(lengths):
        number = value - lengths[i-2] if i > 2 else value
        while True:
            c = number & 31
            number >>= 5
            more = number != (-1 if c & 16 else 0)
            codes.append(chr(48 + (c | (32 if more else 0))))
            if not more:
                break
    return {"size": [101, 101], "counts": "".join(codes)}


def rows(ident, group, mask=None, missing=False):
    item = {"image_id": ident, "record_ordinal": ident, "acquisition_group": group}
    score = .625
    candidates = [dict(box_xyxy=[60, 60, 62, 62], score=.99, mask_area=1)]
    prediction = None
    rank = None
    if not missing:
        if mask is None:
            mask = sample_mask()
        candidates.append(dict(box_xyxy=[40, 40, 40, 40], score=score, mask_area=int(mask.sum())))
        prediction = {"image_id": ident, "score": score, "segmentation": encode(mask)}
        rank = 2
    gpu = dict(image_id=ident, group=group, rank=rank, candidates=candidates, prediction=prediction)
    arms = dict.fromkeys(s.ARMS, None if missing else prediction["segmentation"])
    cpu = dict(image_id=ident, group=group, rank=rank, score=None if missing else score, arms=arms)
    return item, gpu, cpu


class Tests(unittest.TestCase):
    def test_balanced_comparison_and_candidate_immutability(self):
        item, gpu, cpu = rows(3, "acq_001")
        raw = sample_mask()
        changed = raw.copy()
        changed[40,42] = False
        changed[46,42] = True
        for key in s.ARMS[1:]:
            cpu["arms"][key] = encode(changed)
        area, masks, swaps = s.selected_row(item, gpu, cpu, historic.decode_rle)
        self.assertEqual(area, 36)
        self.assertEqual(swaps, {"raw":0, **dict.fromkeys(s.ARMS[1:],1)})
        np.testing.assert_array_equal(masks["fixed"], changed)
        cpu["arms"]["fixed"] = encode(raw)
        with self.assertRaisesRegex(ValueError, "matched action"):
            s.selected_row(item, gpu, cpu, historic.decode_rle)
        cpu["arms"]["fixed"] = encode(raw | changed)
        with self.assertRaisesRegex(ValueError, "Area"):
            s.selected_row(item, gpu, cpu, historic.decode_rle)
        cpu["rank"] = 1
        with self.assertRaisesRegex(ValueError, "center rule"):
            s.selected_row(item, gpu, cpu, historic.decode_rle)

    def test_equal_weight_missing_empty_thresholds_and_bootstrap(self):
        raw = sample_mask()
        pop = [rows(1, "A", raw), rows(2, "A", missing=True), rows(3, "A", np.zeros_like(raw)), rows(4, "B", raw)]
        selected = [s.selected_row(*args, historic.decode_rle) for args in pop]
        truth = {n:raw for n in range(1, 5)}
        report = s.score_panel([p[0] for p in pop], selected, truth)
        fixed = report["arms"]["fixed"]
        for metric in s.METRICS:
            self.assertAlmostEqual(fixed["acquisition_equal"][metric], 2/3)
        self.assertEqual(fixed["counts"]["selected_empty"],1)
        self.assertEqual(fixed["counts"]["missing"],1)
        self.assertEqual(fixed["counts"]["recovered_075"],2)
        for pair in report["paired"].values():
            self.assertEqual(pair["acquisition_cluster_percentile95"]["iou"], [0,0])
        # One of two acquisitions has an exact 0.50 IoU, the other has perfect overlap.
        t = np.zeros_like(raw)
        t[40:43,42:48] = True
        truth[4] = t
        report = s.score_panel([p[0] for p in pop], selected, truth)
        self.assertEqual(report["arms"]["raw"]["counts"]["recovered_050"],2)
        self.assertEqual(report["arms"]["raw"]["counts"]["recovered_075"],1)
        truth[4] = raw.copy()
        truth[4][40:43,42:45] = False
        self.assertEqual(s.score_panel([p[0] for p in pop], selected, truth)["arms"]["raw"]["counts"]["recovered_075"],2)

    def test_known_paired_acquisition_effect_and_bootstrap(self):
        mask = sample_mask()
        shifted = mask.copy()
        shifted[40,42:45] = False
        shifted[46,42:45] = True
        pop = [rows(1,"A",mask),rows(2,"B",mask),rows(3,"B",mask)]
        for arm in ("free_natural", "native_matched_count"):
            pop[0][2]["arms"][arm] = encode(shifted)
        selected = [s.selected_row(*p, historic.decode_rle) for p in pop]
        report = s.score_panel([p[0] for p in pop], selected, {n:mask for n in (1,2,3)})
        contrast = report["paired"]["free_natural_minus_raw"]
        self.assertAlmostEqual(contrast["acquisition_equal_delta"]["iou"], -1/13)
        self.assertAlmostEqual(contrast["acquisition_cluster_percentile95"]["iou"][0], -2/13)
        self.assertEqual(contrast["acquisition_cluster_percentile95"]["iou"][1], 0)
        self.assertEqual(report["arms"]["free_natural"]["counts"]["xor_pixels"],6)

    def test_scorer_only_author_projection_and_group_binding(self):
        class Q:
            @staticmethod
            def acquisition(row):
                return row["fullname"]
            @staticmethod
            def gray_levels(power):
                return power.astype(np.uint8)
        gray = np.ones((101,101),dtype=np.uint8)
        target = sample_mask().astype(int).tolist()
        groups = [f"group_{i%16:02d}" for i in range(1596)]
        rows = [dict(id="s",fullname=g,sensor="fixture",polarisation="HH",patchgt="outside enrolled cohort")
                for g in groups]
        rows[82].update(patchfu=gray.tolist(),patchgt=target)
        sha = s.digest(np.repeat(gray[:,:,None],3,axis=2).tobytes())
        item = {"record_ordinal":83,"acquisition_group":"acq_003","rgb_sha256":sha}
        ledger = "".join(f"{i+1}\tacq_{i%16+1:03d}\n" for i in range(1596)).encode()
        with patch.object(s, "GROUP_INDEX", s.digest(ledger)):
            raw = json.dumps([rows]).encode()
            out = s.author_truth(raw, [item], lex, Q)
            np.testing.assert_array_equal(out[83], np.asarray(target, dtype=bool))
            with self.assertRaisesRegex(ValueError, "acquisition"):
                s.author_truth(raw, [dict(item, acquisition_group="acq_001")], lex, Q)

    def test_closed_terminal_or_corruption_refused(self):
        terminal = dict(status="SCHWEGMANN_DFINE_FROZEN_EDITS_TERMINAL", actual_exit=0,
                        result_sha256="a"*64, actual_process_terminal=True,tool_session_handle=1)
        s.observed_terminal(terminal, "EDITS", "a"*64)
        for k, v in (("actual_exit", False),("actual_process_terminal", False),("tool_session_handle", 0)):
            with self.assertRaisesRegex(ValueError, "process observation"):
                s.observed_terminal(dict(terminal, **{k:v}), "EDITS", "a"*64)
        with self.assertRaisesRegex(ValueError, "process observation"):
            s.observed_terminal(terminal, "EDITS", "f"*64)

    def test_entire_synthetic_prelabel_admission(self):
        with TemporaryDirectory() as d:
            base = Path(d)
            a = SimpleNamespace(reference_code=reference)
            def store(name, data):
                p = base/name
                p.write_bytes(data)
                setattr(a, name, p)
                setattr(a, name+"_sha256", s.digest(data))
                return s.digest(data)
            def obj(name, value):
                return store(name, json.dumps(value).encode())
            protocol = store("protocol", b"fixture protocol")
            data = [rows(i, s.GROUPS[(i-1)%16], missing=(i > 1)) for i in range(1,751)]
            items = [dict(r[0],original_hw=[101,101]) for r in data]
            manifest = obj("manifest", dict(expected_ids=list(range(1,751)),items=items,
                source_json_sha256=s.ARCHIVE,group_index_sha256=s.GROUP_INDEX))
            stream = lambda index: b"".join((json.dumps(r[index])+"\n").encode() for r in data)
            gpu_rows = store("capture_rows",stream(1))
            cpu_rows = store("predictions",stream(2))
            counts = dict(selected=1,missing=749,nonempty=1,eligible=1)
            capture = obj("capture_result", dict(status="SCHWEGMANN_DFINE_FROZEN_CENTER_CAPTURE_COMPLETE",
                manifest_sha256=manifest,protocol_sha256=protocol,labels_opened=False,optimizer_updates=0,
                author_E24_sha256=s.EMA,counts=dict(counts,images=750),model_forwards=750,native_rows_sha256=gpu_rows))
            cpu = obj("edit_receipt", dict(status="SCHWEGMANN_DFINE_FROZEN_CENTER_EDITS_COMPLETE",
                labels_opened=False,manifest_sha256=manifest,images=750,capture_result_sha256=capture,
                head_acceptance_sha256=s.HEAD,model_forwards=0,optimizer_updates=0,edits_sha256=cpu_rows,
                edits_bytes=len(a.predictions.read_bytes()),counts=dict(counts,area_eligible=1,zero_support=1)))
            for name, role, h in (("capture_terminal","CAPTURE",capture),("edit_terminal","EDITS",cpu)):
                obj(name,dict(status=f"SCHWEGMANN_DFINE_FROZEN_{role}_TERMINAL",actual_exit=0,
                    actual_process_terminal=True,result_sha256=h,tool_session_handle=5))
            with patch.object(s,"MANIFEST",manifest), patch.object(s,"PROTOCOL",protocol):
                result = s.admitted(a)
                self.assertEqual(result[-1]["retained_top100_rows"],751)
                self.assertEqual(result[-1]["eligible"],1)
                with a.predictions.open("ab") as f:
                    f.write(b"\n")
                with self.assertRaisesRegex(ValueError,"digest mismatch"):
                    s.admitted(a)


if __name__ == "__main__":
    unittest.main()
