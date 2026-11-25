"""Render the prelocked spatial/nonspatial signed menu without reference labels."""

from argparse import ArgumentParser
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np


HERE = Path(__file__).resolve().parent
LOCK = "925c4d5287af89ddac0b6343d53a0edfc32517cf8a1edef451696deb44293d9c"
FIT_RESULT = "4f80ed3a7d7792ec7a132db3c5203f8faaf47a38efb332259b3077cd29406f1f"
FIT_TERMINAL = "2741b4a4e7d9c2685499cb873eb4d0ae973df186db99a7d87f874e317076d376"
FIT_WORKER = "280cb7547a3d85d02806ab62dd05519c6dd81626d6903e7fb0a2a9862dd47e47"
PAIR_CORE = "1c1c772afd476dcf227c176a2f46706ae83a801b504bab7c3dce0f1952b686c7"
PAIR_GRAPHS = "a904c262f0ce554fb6fc6b6e92864f29566394fdd9f21981bd9a3858cd0791f2"
PAIR_SELECT = "c874cfd8608691203db5aab80a16ff7d15b7ce5d4a51c7a9a3d0c6f22daefb0c"
EXPECTED_SOURCE = "1d9dd5049ee896e09026f0e128b24e623857d1a6fd0416fbd6c2db8701ac6f64"
STATUS = "SPSD_PSEG_SPATIAL_PAIR_RENDER_COMPLETE"


def module(path, expected):
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != expected:
        raise ValueError("Expected byte-exact closed source")
    result = ModuleType(path.stem)
    result.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), result.__dict__)
    return result


def fit(c, root):
    terminal = json.loads(c.pin(root / "fit/observed_fit/TERMINAL.json", FIT_TERMINAL))
    result = json.loads(c.pin(root / "fit/actual/RESULT_SAFE.json", FIT_RESULT))
    c.need(terminal["phase"] == "prepare-edits" and terminal["actual_exit"] == 0
           and terminal["output_footer_verified"] is True and terminal["result_sha256"] == FIT_RESULT
           and terminal["worker_sha256"] == result["worker_sha256"] == FIT_WORKER
           and result["status"] == "SPSD_PSEG_SPATIAL_PAIR_FIT_COMPLETE"
           and result["protocol_sha256"] == LOCK and result["pair_core_sha256"] == PAIR_GRAPHS
           and result["qualified_for_fixed_rerender"] is result["fit_only"] is True
           and result["source_gate_labels_opened"] is result["source_gate_points_opened"] is False
           and result["transfer_labels_opened"] is result["historical_test_opened"] is False,
           "Mandatory actual closed FIT-only paired probability gate")
    payload_sha = result["full_fit_kappa_state_sha256"]
    payload = json.loads(c.pin(root / "fit/actual/KAPPA_PRIVATE.json", payload_sha))
    c.need(payload["status"] == "SPSD_PSEG_SPATIAL_PAIR_KAPPA_FIT" and payload["protocol_sha256"] == LOCK
           and payload["fit_only"] is payload["qualified_for_fixed_rerender"] is True
           and set(payload["kappa"]) == {"pseg", "spsd"}, "Exact frozen FIT coefficients")
    weights = {graph: tuple(payload["kappa"][reference][graph] for reference in ("pseg", "spsd"))
               for graph in ("local", "nonspatial")}
    return weights, payload_sha


def main():
    p = ArgumentParser(description=__doc__, allow_abbrev=False)
    for key in ("code", "helpers", "inputs", "root", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    a = p.parse_args()
    prior_root = a.code / "SPSD_PSEG_EXPECTED_IOU_SIGNED_BUDGET_20261126"
    prior = module(prior_root / "run_expected.py", EXPECTED_SOURCE)
    c = prior.helper(a.helpers)
    c.LOCK = LOCK
    c.pin(HERE / "PRELOCK.md", LOCK)
    c.need(sys.flags.isolated and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
           and a.root.resolve() == HERE and a.output.resolve() == (HERE / "render").resolve()
           and not a.output.exists(), "Fresh closed-study CPU rendering directory")
    weights, payload_sha = fit(c, a.root)
    graph = module(HERE / "pair_iou.py", PAIR_GRAPHS)
    exact = module(HERE / "paired_expectation.py", PAIR_CORE)
    joint = module(HERE / "select_pairs.py", PAIR_SELECT)
    moment = module(prior_root / "expected_iou.py", prior.CORE)
    cfg = c.settings(a.inputs, prior.RENDER_KEYS)
    counts = {name: Counter() for name in weights}
    candidate_changes = Counter()

    def amend(source):
        score = "                decisions=policy.select(sub,incumbent,k,area)"
        receipt = '    raw=(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\\n").encode()'
        c.need(source.count(score) == source.count(receipt) == 1, "Exact verified legacy render hook sites")
        return source.replace(score, "                decisions=policy.select(sub,incumbent,k,area,yy,native.shape[1],f\"{iid}:{rank}\")"
                             ).replace(receipt, "    result.update(extra_facts())\n" + receipt)

    worker = c.load(cfg.base_renderer, prior.BASE_RENDERER, amend)
    original_loader = worker.load

    def policy(path, expected):
        base = original_loader(path, expected)
        if expected != prior.WRITER:
            return base

        def select(p, static, k, area, xy, width, tag):
            legacy = base.select(p, static, k, area)
            old, old_counts = moment.select(base, p, static, k, area)
            coords = np.stack((xy // width, xy % width), axis=-1)
            graphs = graph.graphs(coords, tag)
            local, local_counts = joint.choose(base, moment, exact.prepare, p, static, k, area, graphs, weights)
            for name, stats in local_counts.items():
                c.need(stats["menu_actions"] == old_counts["menu_actions"] == legacy["menu_actions"],
                       "All policies evaluate the same feasible signed action universe")
                counts[name].update(stats)
                for group in ("exact", "fixed"):
                    arm = f"{name}_{group}"
                    x, y = local[arm]["action"], old[f"expectation_{group}"]["action"]
                    candidate_changes[arm] += int(x != y)
                    candidate_changes[arm+"_split"] += int(len(x[0]) != len(y[0]))
            return legacy | old | local

        return SimpleNamespace(ARMS=(*base.ARMS, *prior.EXPECTED, *joint.ARMS),
                               select=select, write_mask=base.write_mask)

    worker.load = policy
    worker.LOCK = LOCK
    worker.STATUS = STATUS
    worker.__file__ = __file__
    worker.extra_facts = lambda: dict(exact_expected_iou_sha256=prior.CORE,
                                     joint_pair_iou_sha256=PAIR_CORE, pair_graphs_sha256=PAIR_GRAPHS,
                                     joint_selector_sha256=PAIR_SELECT, fit_result_sha256=FIT_RESULT,
                                     fit_terminal_sha256=FIT_TERMINAL, fit_kappa_sha256=payload_sha,
                                     no_new_occupancy_updates=True, pair_model_arrays_retained=False,
                                     joint_search={name: dict(values) for name, values in counts.items()},
                                     candidate_differences_vs_independent=dict(candidate_changes))
    cfg.protocol = HERE / "PRELOCK.md"
    cfg.output = a.output
    worker.render(cfg)


if __name__ == "__main__":
    main()
