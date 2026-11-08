import math
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_native_match as probe
from probe_native_match import expand_groups, match_counts


class RealMatchInstrumentationTest(unittest.TestCase):
    def test_empty_iabce_keeps_native_zero_visible(self):
        z = torch.tensor([-9.0, -8.0, -6.0], dtype=torch.bfloat16)
        check = probe.empty_iabce_observation(z, 0.0)
        expected = sum((1 / (1 + math.exp(-v))) ** 2 * math.log1p(math.exp(v))
                       for v in (-9.0, -8.0, -6.0))
        self.assertEqual(check["native"], 0.0)
        self.assertAlmostEqual(check["stable_fp32"], expected, delta=1e-12)
        self.assertLess(check["native_minus_stable_fp32"], 0.0)

    def test_cpu_receipt_uses_its_own_protocol(self):
        cpu_protocol, gpu_protocol = "a" * 40, "b" * 40
        with TemporaryDirectory() as root:
            path = Path(root) / probe.INPUT_PACKET
            cases = {"empty": {}, "dense": {}}
            torch.save(dict(audit_sha256=probe.AUDIT_REPORT_SHA, fit_view_sha256=probe.FIT_SHA,
                            original_archive_sha256=probe.ARCHIVE_SHA, cases=cases), path)
            digest = probe.file_digest(path)
            probe.write_json(path.parent / "CPU_SELECTION_SAFE.json", dict(
                status="TWO_REAL_HRSID_FIT_BATCHES_READY", protocol=cpu_protocol,
                p109_sha256=probe.AUDIT_REPORT_SHA, selector_sha256=probe.SELECTOR_SHA,
                fit_view_sha256=probe.FIT_SHA, archive_sha256=probe.ARCHIVE_SHA,
                private_packet_sha256=digest, private_packet_bytes=path.stat().st_size,
                FIT_only=True, DEV_image_opened=False, TEST_member_opened=False))
            self.assertEqual(probe.load_packet(path, digest, input_protocol=cpu_protocol), cases)
            with self.assertRaisesRegex(ValueError, "CPU provenance"):
                probe.load_packet(path, digest, input_protocol=gpu_protocol)


    def test_dense_and_empty_group_readback(self):
        groups = 13
        for width, n, expected in ((100, 163, 100), (164, 163, 163), (100, 0, 0), (164, 0, 0)):
            src = torch.arange(width * groups).reshape(groups, width)[:, :expected].flatten()
            dst = torch.arange(expected).repeat(groups)
            self.assertEqual(match_counts([(src, dst)], width, n), [expected] * groups)

    def test_rejects_repeated_target_in_one_group(self):
        src = torch.arange(1300).reshape(13, 100)[:, :2].flatten()
        dst = torch.arange(2).repeat(13)
        dst[1] = dst[0]
        with self.assertRaisesRegex(ValueError, "uniquely"):
            match_counts([(src, dst)], 100, 2)

    def test_expansion_preserves_per_group_prefix_and_seed(self):
        def model():
            x = SimpleNamespace(num_queries=100, group_detr=13,
                                transformer=SimpleNamespace(num_queries=100, group_detr=13),
                                refpoint_embed=nn.Embedding(1300, 4), query_feat=nn.Embedding(1300, 8))
            with torch.no_grad():
                for embed in (x.refpoint_embed, x.query_feat):
                    embed.weight.copy_(torch.arange(embed.weight.numel()).view_as(embed.weight))
            return x

        a, b = model(), model()
        before = a.query_feat.weight.detach().reshape(13, 100, 8).clone()
        rng = torch.random.get_rng_state()
        expand_groups(a, 164)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        expand_groups(b, 164)
        self.assertEqual(a.num_queries, a.transformer.num_queries)
        self.assertEqual(a.num_queries, 164)
        self.assertTrue(torch.equal(a.query_feat.weight, b.query_feat.weight))
        self.assertTrue(torch.equal(a.query_feat.weight.detach().reshape(13, 164, 8)[:, :100], before))
        self.assertTrue(torch.all(a.refpoint_embed.weight.detach().reshape(13, 164, 4)[:, 100:] == 0))


if __name__ == "__main__":
    unittest.main()
