from hashlib import sha256
from io import BytesIO
import json
import unittest

import numpy as np
from pycocotools import mask as codec

from score_signed_phase import (CONTROLS, choose_points, evaluate_roles, features,
                                fit_heads, point_bank, score_edit, split_roles)


def rectangle(start):
    plane = np.zeros((800, 800), dtype=np.uint8, order="F")
    plane[10:20, start:start+10] = 1
    return codec.encode(plane)


def fixture_image(image_id):
    row = dict(image_id=image_id, group=f"P{image_id}", encoded_sha256=f"{image_id:064x}",
               core=True, stress=False)
    prediction = rectangle(10)
    points = []
    for x, value in ((9, -1.), (10, .1), (19, 1.), (20, -.1)):
        points.append(dict(y=15, x=x, inside=value > 0, logits=[value, -value, *([value]*7)]))
    candidate = dict(native_rank=1, query_id=0, score=.9, category_id=1,
                     segmentation=dict(size=prediction["size"], counts=prediction["counts"].decode("ascii")),
                     points=points)
    empty = codec.encode(np.zeros((800, 800), dtype=np.uint8, order="F"))
    unmatched = dict(native_rank=2, query_id=3, score=.1, category_id=1, points=[],
                     segmentation=dict(size=empty["size"], counts=empty["counts"].decode("ascii")))
    return row, dict(row, candidates=[candidate, unmatched])


class TestSignedPhase(unittest.TestCase):
    def test_zero_padding_support_and_clamped_logits_at_canvas_edges(self):
        # The point has outside-background support and all clamped neighbors
        # are foreground, as on an uninterrupted foreground canvas edge.
        for x, y in ((0, 5), (799, 5), (5, 0), (5, 799), (0, 0)):
            point = dict(x=x, y=y, inside=True, logits=[1.]*9)
            xy, inside, _, logits, _ = point_bank({"points": [point]}, 100, "edge")
            self.assertEqual(xy.tolist(), [[x, y]])
            self.assertEqual(inside.tolist(), [True])
            self.assertEqual(logits.tolist(), [[1.]*9])
        interior = dict(x=5, y=5, inside=True, logits=[1.]*9)
        exterior = dict(x=0, y=5, inside=False, logits=[-1.]*9)
        for invalid in (interior, exterior):
            with self.assertRaisesRegex(ValueError, "boundary band"):
                point_bank({"points": [invalid]}, 100, "not a boundary")

    def test_same_side_budget_and_margin_are_exact(self):
        inside = np.array([True, False, True, False])
        selected = choose_points(np.array([.80, .90, .99, .70]), inside, np.array([1, 2, 3, 4]), 30)
        self.assertEqual(selected.tolist(), [2, 1])
        self.assertEqual(choose_points(np.full(4, .46), inside, np.arange(4), 100).size, 0)
        self.assertEqual(score_edit(8, 10, inside[selected], np.array([False, True])), (9, 9))
        self.assertEqual(score_edit(8, 10, inside[selected], np.array([True, False])), (7, 11))

    def test_phase_permutation_keeps_same_side_pair_and_true_logits(self):
        xy = np.array([[3, 5], [4, 6], [9, 2], [10, 2]])
        inside = np.array([True, True, False, False])
        z = np.arange(36, dtype=np.float64).reshape(4, 9)
        z[~inside, 0] *= -1
        blocks = features(xy, inside, z, "fixture")
        self.assertTrue(np.array_equal(blocks["phase"][:, :10], blocks["permuted"][:, :10]))
        for side in (False, True):
            self.assertEqual(set(map(tuple, blocks["phase"][inside == side, 10:12])),
                             set(map(tuple, blocks["permuted"][inside == side, 10:12])))

    def test_no_groups_cross_fit_held_and_stress_never_trains(self):
        cohort = [dict(image_id=i, group=f"P{i}", encoded_sha256=f"{i:064x}", core=True, stress=False)
                  for i in range(109)]
        cohort += [dict(image_id=109+i, group=f"P{i}", encoded_sha256=f"{i+109:064x}", core=False, stress=True)
                   for i in range(5)]
        roles, _ = split_roles(cohort)
        self.assertEqual(list(roles.values()).count("fit"), 54)
        self.assertEqual(list(roles.values()).count("held"), 55)
        self.assertEqual(list(roles.values()).count("stress"), 5)
        self.assertTrue(all(roles[i] == "held" for i in range(5)))
        cohort[15]["encoded_sha256"] = cohort[0]["encoded_sha256"]
        roles, groups = split_roles(cohort)
        self.assertEqual(roles[15], roles[0])
        self.assertEqual(groups[15], groups[0])

    def test_fit_and_held_use_immutable_matches_and_count_matched_controls(self):
        cohort, rows = zip(*(fixture_image(i) for i in range(1, 4)))
        raw = b"".join((json.dumps(row) + "\n").encode("utf-8") for row in rows)
        stream, digest = BytesIO(raw), sha256(raw).hexdigest()
        roles, groups = {1: "fit", 2: "fit", 3: "held"}, {i: f"P{i}" for i in range(1, 4)}
        truth = {i: [rectangle(11)] for i in roles}
        models, fit = fit_heads(stream, digest, cohort, roles, groups, truth)
        self.assertEqual(fit["sampled_points"], 8)
        counts, scores, _ = evaluate_roles(stream, digest, cohort, roles, groups, truth, models)
        self.assertEqual(counts["held"]["fixed_matches"], 1)
        self.assertEqual(counts["held"]["unmatched_candidates"], 1)
        self.assertAlmostEqual(scores["held"]["raw"]["fixed_iou_sum"], 90/110)
        phase = scores["held"]["phase"]
        self.assertGreater(phase["add"] + phase["delete"], 0)
        for control in CONTROLS:
            self.assertEqual(scores["held"][control]["add"], phase["add"])
            self.assertEqual(scores["held"][control]["delete"], phase["delete"])


if __name__ == "__main__":
    unittest.main()
