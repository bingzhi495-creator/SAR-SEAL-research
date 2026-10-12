"""Public-path CPU checks for the opt-in eight-view candidate trace."""

import gzip
import json
import tempfile
import unittest
from pathlib import Path

import torch
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from mmdet.structures import DetDataSample

from carrier.mask_tta import MaskTTAModel
from carrier.same_pool_trace import replay_vote_trace, validate_vote_trace


def sample(mask, score, view, image_id=7):
    inst = InstanceData()
    inst.masks = mask[None]
    inst.scores = torch.tensor([score], dtype=torch.float32)
    inst.labels = torch.tensor([0], dtype=torch.long)
    ds = DetDataSample()
    ds.pred_instances = inst
    ds.set_metainfo(dict(img_id=image_id, ori_shape=tuple(mask.shape),
                         flip=False, flip_direction=None,
                         scale_factor=(1.0, 1.0), trace_view=view))
    return ds


def model(trace_dir=None):
    obj = MaskTTAModel.__new__(MaskTTAModel)
    cfg = dict(merge="vote", vote_iou=0.5, vote_frac=0.5,
               vote_mask="consensus", vote_min_support=2,
               vote_max_support=6, vote_gamma=0.0,
               max_per_img=100, nms=dict(type="nms", iou_threshold=0.6))
    if trace_dir is not None:
        cfg["trace_dir"] = str(trace_dir)
    obj.tta_cfg = ConfigDict(cfg)
    return obj


def two_views():
    a = torch.zeros((16, 16), dtype=torch.bool)
    b = torch.zeros((16, 16), dtype=torch.bool)
    a[3:10, 3:10] = True
    b[3:10, 4:11] = True
    return [sample(a, 0.9, 0), sample(b, 0.8, 1)]


class TraceTest(unittest.TestCase):
    def test_opt_in_writes_original_candidate_and_cluster_trace(self):
        with tempfile.TemporaryDirectory() as temp:
            output = model(temp)._merge_single_sample(two_views()).pred_instances
            path = Path(temp) / "img_00000007.json.gz"
            self.assertTrue(path.is_file())
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                trace = json.load(stream)
            self.assertEqual(trace["image_id"], 7)
            self.assertEqual(len(trace["candidates"]), 2)
            self.assertEqual([row["view_id"] for row in trace["candidates"]], [0, 1])
            self.assertEqual(trace["clusters"][0]["seed_id"], 0)
            self.assertEqual(trace["clusters"][0]["member_ids"], [0, 1])
            self.assertEqual(trace["kept_cluster_indices"], [0])
            self.assertEqual(len(output.masks), 1)

    def test_original_readout_replays_exactly_and_top1_keeps_output_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            original = model(temp)._merge_single_sample(two_views()).pred_instances
            with gzip.open(Path(temp) / "img_00000007.json.gz", "rt", encoding="utf-8") as stream:
                trace = json.load(stream)
            self.assertEqual(validate_vote_trace(trace), [])
            baseline = replay_vote_trace(trace, per_view_top1=False)
            top1 = replay_vote_trace(trace, per_view_top1=True)
            self.assertEqual(len(baseline), len(original.masks))
            self.assertEqual([row["score"] for row in baseline], original.scores.tolist())
            self.assertEqual([row["score"] for row in top1], original.scores.tolist())
            self.assertEqual([row["label"] for row in top1], original.labels.tolist())
            self.assertEqual(len(top1), len(baseline))

    def test_trace_switch_does_not_change_returned_instances(self):
        with tempfile.TemporaryDirectory() as temp:
            plain = model()._merge_single_sample(two_views()).pred_instances
            traced = model(temp)._merge_single_sample(two_views()).pred_instances
            for key in ("masks", "scores", "labels", "bboxes"):
                self.assertTrue(torch.equal(plain[key], traced[key]), key)

    def test_trace_keeps_original_top_k_indices_after_two_clusters(self):
        with tempfile.TemporaryDirectory() as temp:
            first = torch.zeros((16, 16), dtype=torch.bool)
            second = torch.zeros((16, 16), dtype=torch.bool)
            first[1:5, 1:5] = True
            second[10:14, 10:14] = True
            traced_model = model(temp)
            traced_model.tta_cfg.max_per_img = 1
            output = traced_model._merge_single_sample([
                sample(first, 0.9, 0), sample(second, 0.8, 1)]).pred_instances
            with gzip.open(Path(temp) / "img_00000007.json.gz", "rt", encoding="utf-8") as stream:
                trace = json.load(stream)
            self.assertEqual(len(trace["clusters"]), 2)
            self.assertEqual(trace["kept_cluster_indices"], [0])
            self.assertEqual(len(output.scores), 1)
            self.assertEqual(validate_vote_trace(trace), [])


if __name__ == "__main__":
    unittest.main()
