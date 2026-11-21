"""Qualification on generated fixtures only."""
import importlib.util
import json
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np


def load_runner():
    path = Path(__file__).with_name("run_capture109.py")
    spec = importlib.util.spec_from_file_location("capture109", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Qualification(unittest.TestCase):
    def test_four_cache_archives_replay_and_reject_a_changed_archive(self):
        runner = load_runner()
        encode = lambda d: (json.dumps(d, sort_keys=True, separators=(",", ":"))+"\n").encode()
        pilot_file = Path(__file__).resolve().parent.parent/"RFDETR_QUERY_FEATURE_REAL_20261121_04"/"run_core4.py"
        self.assertEqual(sha256(pilot_file.read_bytes()).hexdigest(), runner.CORE4_SHA)
        spec = importlib.util.spec_from_file_location("original_core4", pilot_file)
        pilot = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pilot)
        no_cache = SimpleNamespace(core4_output=None, core4_result_sha256=None, core4_closure=None, core4_closure_sha256=None)
        self.assertEqual(runner.admit_core4(no_cache, [], [], pilot, None, np, None), ([], {}))
        no_cache.core4_output = Path(".")
        with self.assertRaises(ValueError):
            runner.admit_core4(no_cache, [], [], pilot, None, np, None)
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            roster, prepared, entries, expected = [], [], [], []
            for i in range(1, 5):
                item = dict(image_id=i, native_basename=f"toy{i}.png", encoded_bytes=1,
                            encoded_sha256="0"*64, decoded_rgb_sha256="1"*64, group=str(i))
                block = dict(item, image_content_sha256="1"*64, source_prefix=str(i), native_ids=[], native_ranks=[],
                             ship_scores=[], all_rles=[], raw_classifier_logits=[], raw_boxes_cxcywh=[],
                             joint_query_ids=[], joint_labels=[], joint_scores=[],
                             logit_capture=dict(sha256="3"*64, bytes=1, shape=[], candidates=[]))
                roster.append((item, i, block))
                n = int(i != 2)
                coords = np.array([[1,2]], np.int32) if n else np.empty((0,2), np.int32)
                features = np.zeros((n,2,14), np.float32)
                layers = np.zeros((n,2), np.int8)
                points = [dict(rank=2, query_id=4, coordinates=coords, features=features, layers=layers)] if n else []
                prepared.append((item, [], points))
                offsets = np.concatenate(([0], np.cumsum([1]))) if n else np.array([0], np.int64)
                out = BytesIO()
                np.savez(out, image_id=i, ranks=np.array([2] if n else [], int), query_ids=np.array([4] if n else [], int),
                         old14=features, layers=layers, coords=coords, offsets=offsets,
                         projected_queries=np.ones((n,256), np.float32), spatial=np.ones((n,2,256), np.float32))
                expected.append(out.getvalue())
                (root/runner.archive_name(i)).write_bytes(expected[-1])
                entries.append(runner.signature(i, expected[-1]))
            def controls(original, spatial, query, coords, layers, candidate_id):
                feature = np.concatenate((original, np.zeros((len(coords), 2, 16), np.float32)), axis=-1)
                focus = feature.copy()
                focus[:,:,14] = 1.
                return dict(features=dict(zero=feature, ordinary_centered=feature, query_conditioned_permuted=feature,
                                          query_conditioned_centered=focus))
            spot = SimpleNamespace(MIB=1024**2, read_cohort=lambda *_: roster, encode=encode)
            residual = SimpleNamespace(point_feature_controls=controls)
            result = dict(status="RFDETR_FIT_QUERY_BASIS_CORE4_CAPTURE_COMPLETE", FIT=True, DEV=False, TEST=False,
                          GT_opened=False, AP=False, head_updates=0, original_candidate_and_mask_exact=True,
                          exact_original_eligible_logits=True, full_basis_logits_exact=True, exact_dense_sample_witness=True,
                          private_output_sha256=sha256(encode(entries)).hexdigest(), entry_sha256=runner.CORE4_SHA,
                          private_file_count=4, model_forwards=4, sampling_channel_chunk=8, candidate_sets=3,
                          point_pairs=3, no_support_images=1, informative_points=6, moved_points=6)
            result_sha = sha256(encode(result)).hexdigest()
            (root/"RESULT_SAFE.json").write_bytes(encode(result))
            closure = dict(result, status="RFDETR_FIT_QUERY_BASIS_CORE4_INDEPENDENTLY_ACCEPTED", result_sha256=result_sha,
                           original_os_exit=0, job_drained=True, independent_readback_accepted=True)
            accepted = root/"closure.json"
            accepted.write_bytes(encode(closure))
            runner.CORE4_CLOSURE_SHA = sha256(encode(closure)).hexdigest()
            args = SimpleNamespace(root=root, core4_output=root, core4_result_sha256=result_sha,
                                   core4_closure=accepted, core4_closure_sha256=runner.CORE4_CLOSURE_SHA)
            self.assertEqual(runner.admit_core4(args, roster, prepared, pilot, spot, np, residual), (expected, result))
            prepared[0][2][0]["features"][0,0,1] = -0.0
            with self.assertRaisesRegex(ValueError, "old14"):
                runner.admit_core4(args, roster, prepared, pilot, spot, np, residual)
            prepared[0][2][0]["features"][0,0,1] = 0.0
            (root/runner.archive_name(2)).write_bytes(expected[1]+b"x")
            with self.assertRaisesRegex(ValueError, "ordered archive"):
                runner.admit_core4(args, roster, prepared, pilot, spot, np, residual)

    def test_prior_prelabel_must_match_canonical_original_bytes(self):
        runner = load_runner()
        encode = lambda d: (json.dumps(d, sort_keys=True, separators=(",", ":"))+"\n").encode()
        pre = runner.verify_prior_prelabel(runner.COUNTS, runner.FEATURE_SHA, encode)
        self.assertEqual(sha256(pre).hexdigest(), runner.PRELABEL_SHA)
        with self.assertRaises(ValueError):
            runner.verify_prior_prelabel(dict(runner.COUNTS, mapped=10487), runner.FEATURE_SHA, encode)
        with self.assertRaises(ValueError):
            runner.verify_prior_prelabel(runner.COUNTS, "0"*64, encode)

    def test_empty_archive_identity_even_without_points(self):
        runner = load_runner()
        arrays = dict(image_id=np.array(42), ranks=np.empty(0, int), query_ids=np.empty(0, int),
                      offsets=np.array([0], dtype=np.int64), coords=np.empty((0, 2), np.int32),
                      old14=np.empty((0, 2, 14), np.float32), layers=np.empty((0, 2), np.int8),
                      projected_queries=np.empty((0, 256), np.float32),
                      spatial=np.empty((0, 2, 256), np.float32))
        runner.verify_arrays(arrays, 42, [], np)
        with self.assertRaises(ValueError):
            runner.verify_arrays(arrays, 41, [], np)


if __name__ == "__main__":
    unittest.main()
