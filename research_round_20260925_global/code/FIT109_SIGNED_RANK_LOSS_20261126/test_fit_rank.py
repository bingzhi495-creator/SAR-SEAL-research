from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import torch

spec = spec_from_file_location("fit_rank", Path(__file__).with_name("fit_rank.py"))
fit = module_from_spec(spec)
spec.loader.exec_module(fit)


def test_rank_penalty_respects_original_side_and_weights_equal_targets_zero():
    logits = torch.tensor([[[0., 1.], [1., 0.], [2., -1.], [3., -2.]]])
    targets = torch.tensor([[[0., 1.], [.5, .5], [.8, .1], [1., 0.]]])
    good = float(fit.ranking_loss(logits, targets))
    wrong = float(fit.ranking_loss(-logits, targets))
    shift = float(fit.ranking_loss(logits + torch.tensor([10., -6.]), targets))
    assert good < wrong
    assert abs(good - shift) < 1e-7
    assert float(fit.ranking_loss(logits, torch.zeros_like(targets))) == 0


def test_batch_packets_use_single_candidate_per_4_rows():
    groups = [np.array([0, 1]), np.array([2])]
    offset = np.array([0, 5, 8, 14])
    indices = fit.sample_packets(groups, offset, 5217, 4)
    assert indices.shape == (4, 16, 4)
    assert np.array_equal(indices, fit.sample_packets(groups, offset, 5217, 4))
    for packet in indices.reshape(-1, 4):
        cands = np.searchsorted(offset, packet, side="right") - 1
        assert len(set(cands)) == 1


def test_point_training_uses_equal_initialization_and_distinct_matched_objectives():
    import os
    code = Path(os.environ["FIT109_PUBLIC_CODE"])
    step = fit.load(code / fit.STEP, fit.STEP_SHA)
    trainer = step.load_trainer(code / fit.BASE / "fit109_paired_point/train_point.py")
    random = np.random.default_rng(12)
    n = 15
    features = random.normal(size=(n, 2, 14)).astype(np.float32)
    features[..., 0] = random.uniform(.1, .9, size=(n, 2))
    truth = random.uniform(size=(n, 2))
    arrays = dict(features=features, truth=truth, role=np.full(3, 2, dtype=np.uint8),
                  fold=np.array([0, 1, 2]), group=np.array(["A", "B", "C"]), offset=np.arange(4)*5)
    center, scale, sampled, _ = fit.fold_schedule(arrays, 0, 5217, trainer)
    assert (sampled >= 5).all()
    assert np.allclose(center, features[5:].mean(axis=(0, 1)), atol=1e-7)
    _, first = fit.fit_model(trainer, step, arrays, center, scale, sampled[:3], 5217, "bce")
    _, second = fit.fit_model(trainer, step, arrays, center, scale, sampled[:3], 5217, "bce_rank")
    assert first["initial_sha256"] == second["initial_sha256"]
    assert abs(first["initial_loss"]["bce"] - second["initial_loss"]["bce"]) < 1e-8
    assert abs(second["initial_loss"]["total"] - second["initial_loss"]["bce"]
               - .25 * second["initial_loss"]["within_sign_rank"]) < 1e-7
    assert first["steps"] == second["steps"] == 3 and first["final_sha256"] != second["final_sha256"]


if __name__ == "__main__":
    torch.set_num_threads(1)
    test_rank_penalty_respects_original_side_and_weights_equal_targets_zero()
    test_batch_packets_use_single_candidate_per_4_rows()
    test_point_training_uses_equal_initialization_and_distinct_matched_objectives()
    print("signed ranking, separate held folds, equal initialization and matched optimization: PASS")

