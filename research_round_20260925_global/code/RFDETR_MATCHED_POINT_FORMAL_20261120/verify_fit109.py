"""Check the naturally closed original FIT109 schedule and every frozen S/L archive."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("trained", "tool-wait", "spot-source", "code", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--wait-sha256", required=True)
    args = parser.parse_args()
    sys.path[0:0] = [str(Path(__file__).resolve().parent), str(args.spot_source.resolve(strict=True))]
    import fit_plan
    import fit_step
    import run_spot as spot
    import torch

    torch.set_num_threads(1)
    wait = json.loads(spot.pinned(args.tool_wait, args.wait_sha256, 1 * spot.MIB))
    final = wait["terminal"]
    spot.need(wait["original_session_id"] > 0 and final["exit_code"] == 0 and final.get("session_id") is None,
              "Exact original training handle natural exit")
    footer = json.loads(final["output"].splitlines()[-1])
    source = json.loads(spot.pinned(args.trained / "RESULT_SAFE.json", footer["sha256"], 4 * spot.MIB))
    spot.need(source["status"] == footer["status"] == "RFDETR_MATCHED_POINTS_FIT109_RUN_COMPLETE"
              and source["FIT"] and not any(source[key] for key in ("DEV", "TEST", "AP"))
              and source["images"] == footer["images"] == 109
              and source["head_updates"] == footer["head_updates"] == 7200
              and source["detector_forward_calls"] == 0, "Exact closed RF source-only schedule")
    prelabel = json.loads(spot.pinned(args.trained / "PRELABEL_SAFE.json", source["prelabel_sha256"], 64 * 1024))
    spot.need(prelabel["features"] == source["preparation"] and not prelabel["GT_opened"],
              "Source features were sealed prior to label use")
    trainer = fit_step.load_trainer(args.code / "DFINE_SEG_EXTERNAL_BASELINE_FIT32_E24_20261112/e24/fit109_paired_point/train_point.py")
    expected = [(seed, fold) for seed in fit_plan.SEEDS for fold in (*range(5), None)]
    actual = source["all_initializations"]
    spot.need([(arm["seed"], arm["fold"]) for arm in actual] == expected, "All three seeds and five folds plus refits")
    for item in actual:
        seed, fold = item["seed"], item["fold"]
        endpoint = "all_fit" if fold is None else f"fold_{fold}"
        filename = f"{seed}_{endpoint}.npz"
        spot.need(item["endpoint"] == endpoint and item["relative_weights"] == filename
                  and item["steps"] == 200 and item["batch"] == 64 and item["sampled_pair_rows"] == 12800
                  and item["sampled_point_rows"] == 25600 and item["shared_pairs"] > 0 and item["source_groups"] > 0
                  and set(item["heads"]) == {"S", "L"}, "Exact common training schedule and archive path")
        heads, _, _ = fit_plan.restore_bundle(args.trained / "POINT_HEADS_PRIVATE" / filename,
                                           item["weights_sha256"], trainer, fit_step)
        for name, params in (("S", 1321), ("L", 139265)):
            evidence = item["heads"][name]
            spot.need(evidence["finite_parameters"] and evidence["optimizer_steps"] == [200]
                      and evidence["parameters"] == params
                      and evidence["initial_tensor_sha256"] != evidence["final_tensor_sha256"]
                      and fit_step.tensor_hash(heads[name]) == evidence["final_tensor_sha256"],
                      "Observed head updates and exact final parameter readback")
    result = dict(status="RFDETR_MATCHED_POINTS_FIT109_INDEPENDENT_ACCEPTED",
                  fit_result_sha256=footer["sha256"], run_natural_exit=0, exact_schedule=True,
                  checkpoint_bundles_verified=len(actual), head_updates_verified=7200, model_forward_calls=0,
                  original_tool_wait_sha256=args.wait_sha256, FIT=True, DEV=False, TEST=False, AP=False)
    digest = spot.write(args.output, result)
    print(json.dumps(dict(sha256=digest, **result)), flush=True)


if __name__ == "__main__":
    main()
