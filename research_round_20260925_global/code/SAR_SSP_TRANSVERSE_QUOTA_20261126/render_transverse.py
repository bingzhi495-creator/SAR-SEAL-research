"""Label-free replay of the exact frozen SAR-SSP transfer with quota controls."""
from argparse import ArgumentParser
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sys
from types import ModuleType


HERE = Path(__file__).resolve().parent
BASE_WORKER = "83ff61406235b142b86c2bcdbebbc92b3b17a7d3eb48032cd6c6619ec7e6f8a2"
BASE_RENDER = "9cf1753d4818df64717217831d29421f329eae21014b4656e52be83ecefd1701"
BASE_SCORE = "a857f06b37ca65e38205e24c5fd024e837f533dbf1ba6c448fed5d2e4727e837"
BASE_RENDER_TERMINAL = "1ea6b73fc7f977ad7f3cce632c97083fb2d3fa6b8c153ce4480dd212ec67c582"
BASE_SCORE_TERMINAL = "faa2c69b75cb23bfd5077a338a6682a9fd6233e6c66720fc713c1f64108f57c3"
OLD_ARMS = ("fixed_iou", "variable_iou", "near_iou", "bce_fixed_robust")
NEW_ARMS = ("transverse_iou", "area_iou", "longitudinal_iou")
RENDER_STATUS = "SAR_SSP_TRANSVERSE_QUOTA_RENDER_COMPLETE"


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(raw):
    return sha256(raw).hexdigest()


def pin(path, expected):
    raw = path.read_bytes()
    need(digest(raw) == expected, "Unexpected bound input " + path.name)
    return raw


def patch(source, before, after):
    need(source.count(before) == 1, "Exactly one trusted archive seam: " + before)
    return source.replace(before, after, 1)


def adapted(source):
    source = patch(
        source, "                selected=policy.select(p,incumbent,k,area)",
        "                selected,geometry=select_transverse(policy,p,incumbent,k,stock,coords)",
    )
    source = patch(
        source, "                results[str(rank)]=dict(k=k,coordinates=coords.tolist(),arms=edits)",
        "                results[str(rank)]=dict(k=k,coordinates=coords.tolist(),arms=edits,"
        "transverse_geometry=geometry)",
    )
    return patch(
        source,
        '            raw=line(dict(image_id=item["image_id"],rgb_sha256=item["rgb_sha256"],edits=results))',
        '            document=dict(image_id=item["image_id"],rgb_sha256=item["rgb_sha256"],edits=results)\n'
        '            verify_old(ordinal-1,document)\n'
        '            raw=line(document)',
    )


def closed_archive(base):
    receipts = {}
    for name, expected, terminal, phase in (
        ("render", BASE_RENDER, BASE_RENDER_TERMINAL, "capture"),
        ("score", BASE_SCORE, BASE_SCORE_TERMINAL, "verify"),
    ):
        result = json.loads(pin(base / name / "RESULT_SAFE.json", expected))
        observed = json.loads(pin(base / ("observed_" + name) / "TERMINAL.json", terminal))
        need(observed["phase"] == phase and observed["actual_exit"] == 0
             and observed.get("natural_exit", True) is True
             and observed["output_footer_verified"] is True
             and observed["result_sha256"] == expected
             and observed["worker_sha256"] == result["worker_sha256"] == BASE_WORKER,
             "Exact archived capture and observed historical score closure")
        receipts[name] = result
    need(receipts["render"]["gate_labels_opened"] is False
         and receipts["score"]["gate_labels_opened"] is True
         and receipts["score"]["render_result_sha256"] == BASE_RENDER
         and receipts["score"]["original_control_parity"],
         "Original author validation chain")
    return receipts


def verify_old(saved, new):
    need(new["image_id"] == saved["image_id"] and new["rgb_sha256"] == saved["rgb_sha256"]
         and new["edits"].keys() == saved["edits"].keys(),
         "Unchanged image identity and original positive-quota ranks")
    for rank, prior in saved["edits"].items():
        replacement = new["edits"][rank]
        need(replacement["k"] == prior["k"] and replacement["coordinates"] == prior["coordinates"]
             and set(replacement["arms"]) == set((*OLD_ARMS, *NEW_ARMS))
             and set(prior["arms"]) == set(OLD_ARMS)
             and all(replacement["arms"][key] == value for key, value in prior["arms"].items()),
             "All old signed counts, actions, point order and actual RLEs must be exact")


def run(a):
    past = closed_archive(a.base_run)
    archive_rows = [
        json.loads(line)
        for line in pin(a.base_run / "render/RENDER_PRIVATE.jsonl", past["render"]["raw_sha256"]).splitlines()
    ]
    need(len(archive_rows) == 232 and sum(len(row["edits"]) for row in archive_rows) == 10393,
         "Every original positive-quota slot is present in the closed render")

    base_path = a.base_run / "transfer.py"
    bundle = ModuleType("_admitted_transverse_target_parent")
    bundle.__file__ = str(Path(__file__).resolve())
    old_source = pin(base_path, BASE_WORKER)
    exec(compile(adapted(old_source.decode("utf-8")), str(base_path), "exec"), bundle.__dict__)
    need(bundle.NEW_ARMS == OLD_ARMS and bundle.HERE == HERE,
         "Expected archive writer arm layout")
    bundle.NEW_ARMS = (*OLD_ARMS[:-1], *NEW_ARMS, OLD_ARMS[-1])
    bundle.LOCK = a.protocol_sha
    bundle.RENDER_STATUS = RENDER_STATUS
    bundle.verify_old = lambda i, new: verify_old(archive_rows[i], new)

    sys.path.insert(0, str(bundle.OLD / "numpy_1_26_4_only"))
    sys.path.insert(0, str(bundle.OLD / "capture/CODEC_PRIVATE"))
    method = ModuleType("_pinned_transverse_quota")
    method.__file__ = str(a.method)
    exec(compile(pin(a.method, a.method_sha), str(a.method), "exec"), method.__dict__)
    need(tuple(method.NEW_ARMS) == NEW_ARMS, "Frozen renderer and selector arm agreement")

    tally = Counter()
    selected_a = {arm: Counter() for arm in (*OLD_ARMS[:2], *NEW_ARMS)}
    width_comparisons = Counter()
    old_finish = bundle.finish

    def selection(policy, p, static, k, stock, coords):
        decisions, geometry = method.select(policy, p, static, k, stock, coords)
        tally.update(positive_candidates=1, feasible_quotas=geometry["feasible_quotas"])
        tally.update({geometry["disposition"]: 1})
        admitted = geometry["admitted_counts"]
        need(set(admitted) == set(NEW_ARMS) and len(set(admitted.values())) == 1
             and all(k in options for options in geometry["admitted_quotas"].values()),
             "Identical number of choices for the new arms and B retained")
        tally.update(admitted_quotas_each=admitted["transverse_iou"])
        free = decisions["variable_iou"]["action"]
        fixed = decisions["fixed_iou"]["action"]
        for arm, hist in selected_a.items():
            action = decisions[arm]["action"]
            need(sum(map(len, action)) == 2 * k, "Each selected action uses original 2k")
            hist[len(action[0]) - len(action[1])] += 1
            if arm in NEW_ARMS:
                tally["changed_vs_variable_" + arm] += int(action != free)
                tally["changed_vs_fixed_" + arm] += int(action != fixed)
        if geometry["eligible"]:
            width = geometry["width_by_arm"]
            cap = width["fixed_iou"]
            need(method.within_width_cap(width["transverse_iou"], cap) and
                 decisions["transverse_iou"]["score"] >= decisions["fixed_iou"]["score"],
                 "Transverse cap and B guarantee")
            for arm, observed in width.items():
                width_comparisons["widened_vs_fixed_" + arm] += int(not method.within_width_cap(observed, cap))
        return decisions, geometry

    def finish(output, status, started, **outcome):
        need(status == RENDER_STATUS and outcome["gate_labels_opened"] is False
             and tally["positive_candidates"] == 10393 and outcome["counts"]["candidates"] == 23200
             and tally["eligible"] + tally["aspect"] + tally["disconnected"] == 10393,
             "Complete inherited label-free candidate and eligible geometry census")
        need(all(h["adds"] + h["drops"] == 147118 for h in outcome["signed_arm_counts"].values()),
             "All original and new writers preserve 147118 realized flips")
        old_finish(
            output, status, started, **outcome,
            prelock_commit=a.prelock_commit, original_render_sha256=BASE_RENDER,
            original_score_sha256=BASE_SCORE, original_render_terminal_sha256=BASE_RENDER_TERMINAL,
            original_score_terminal_sha256=BASE_SCORE_TERMINAL, original_worker_sha256=BASE_WORKER,
            transverse_core_sha256=a.method_sha, all_archived_actions_rle_and_quota_exact=True,
            transverse_geometry=dict(tally), width_comparisons=dict(width_comparisons),
            net_area_change_histograms={arm: dict(hist) for arm, hist in selected_a.items()},
        )

    bundle.select_transverse = selection
    bundle.finish = finish
    bundle.render(a)


def main():
    parser = ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("phase", choices=("render",))
    for name in ("base-run", "method", "work-root", "output", "protocol"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("method-sha", "protocol-sha", "prelock-commit"):
        parser.add_argument("--" + name, required=True)
    a = parser.parse_args()
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" and sys.flags.isolated
         and sys.dont_write_bytecode and a.output.resolve() == a.work_root.resolve() / a.phase
         and a.work_root.is_dir() and not a.output.exists(),
         "Use a fresh isolated CPU replay output")
    need(re.fullmatch("[0-9a-f]{40}", a.prelock_commit) is not None,
         "Require the verified published protocol commit")
    pin(a.protocol, a.protocol_sha)

    def offline(event, _):
        if event.startswith("socket."):
            raise PermissionError("Offline label-free replay")

    sys.addaudithook(offline)
    run(a)


if __name__ == "__main__":
    main()

