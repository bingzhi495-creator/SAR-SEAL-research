"""Evaluate fixed SSDD FIT mask transforms against continuous source-pixel area."""
import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

HERE = Path(__file__).resolve().parent
BASE_REPLAY_SHA = "14b097a19c8b951e7d8e246e75e855b2eb574bb1149aa3945b9cf15b91a2cacf"
INPUT_SUMMARY_SHA = "fc3a895226e0005f46ab9c6fe4d912dfcc2a4463ad52775a6cdacc22d1f774e4"
IDS_SHA = "42b93cbf2424e1c1ffa45c9dc4b32c3537fca6fe3a224e81762fcc339276e4cd"
ARMS = ("native", "center", "one_pass_center")
METRICS = ("iou", "source_y_error", "abs_source_y_error", "fp", "fn", "area_signed_error")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def selected_result(values, selected):
    import numpy as np

    fields = {}
    for name in METRICS:
        sample = values[name][selected]
        present = np.isfinite(sample)
        fields[name] = (int(present.sum()), float(sample[present].sum()))
    fields["nonempty"] = (int(selected.sum()), float((values["area"][selected] > 0).sum()))
    return fields


def record(results, path, values):
    import numpy as np

    small = values["small"]
    truth = ~values["truth_empty"]
    for size, area in (("small", small), ("other", ~small)):
        tags = [(size, area)]
        if path == "crop":
            cut = values["crop_truncated"]
            tags += [(size + "_cropped_support", area & cut), (size + "_intact_support", area & ~cut)]
        for key, selected in tags:
            out = results.setdefault(path + "_" + key,
                                     dict(census=Counter(), by_arm={arm:Counter() for arm in ARMS},
                                          pairs=Counter()))
            valid = selected & truth
            out["census"].update(rows=int(selected.sum()), positive_truth=int(valid.sum()),
                                 truth_empty=int((selected & ~truth).sum()),
                                 original_edge=int((selected & values["original_edge"]).sum()),
                                 crop_truncated=int((selected & values["crop_truncated"]).sum()))
            for arm in ARMS:
                for metric, (count, total) in selected_result(values["by_arm"][arm], valid).items():
                    out["by_arm"][arm][metric + "_count"] += count
                    out["by_arm"][arm][metric + "_sum"] += total
            for name, left, right in (("center_vs_native","center","native"),
                                      ("one_pass_vs_center","one_pass_center","center")):
                delta = values["by_arm"][left]["iou"][valid] - values["by_arm"][right]["iou"][valid]
                require(np.isfinite(delta).all(), "Positive truth requires finite continuous IoU")
                out["pairs"].update({name + "_wins":int((delta > 1e-12).sum()),
                                     name + "_ties":int((np.abs(delta) <= 1e-12).sum()),
                                     name + "_losses":int((delta < -1e-12).sum())})


def summarize(results):
    summary = {}
    for name, group in sorted(results.items()):
        arms = {}
        for arm, fields in group["by_arm"].items():
            values = {}
            for metric in (*METRICS, "nonempty"):
                count, total = fields[metric + "_count"], fields[metric + "_sum"]
                values[metric] = dict(count=count, mean=total / count if count else None)
            arms[arm] = values
        summary[name] = dict(census=group["census"], arms=arms, pairwise_iou=group["pairs"])
    return summary


def replay_module(results):
    from ssdd_pixel_footprint import measure_masks
    import numpy as np

    path = HERE / "ssdd_phase_fit_replay.py"
    raw = path.read_bytes()
    require(sha256(raw).hexdigest() == BASE_REPLAY_SHA, "Exact original FIT replay")
    source = ModuleType("_ssdd_footprint_scoped_original_fit")
    source.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), source.__dict__)
    load = source._load_replay

    def decorated_load():
        replay = load()
        original = replay.replay_view
        point_reference = replay.center_reference
        compose = point_reference.__globals__["_compose"]

        def decorated_view(audit, context, image, prepared, seed):
            observed = []

            def capture(target, *args, **kwargs):
                context.target_summary(target, *args, **kwargs)
                observed.append(target)

            result = original(audit, SimpleNamespace(_torchvision=context._torchvision,
                                                     set_seed=context.set_seed, target_summary=capture),
                              image, prepared, seed)
            require(len(observed) == 2, "Original replay presents exactly two validated targets")
            row_id = "_p198_row_id"
            common, ai, bi = np.intersect1d(observed[0][row_id].numpy(), observed[1][row_id].numpy(),
                                             assume_unique=True, return_indices=True)
            a, b = observed[0]["masks"].numpy()[ai], observed[1]["masks"].numpy()[bi]
            shared = np.any(a, axis=(1,2)) & np.any(b, axis=(1,2))
            require(int(shared.sum()) == result["rows"]["common_nonempty"], "Exact original common live population")
            common, a, b = common[shared], a[shared], b[shared]
            orig = prepared["masks"][common].numpy()
            small = prepared["area"][common].numpy() < 1024
            steps = result["steps"]
            one = point_reference(orig, steps)
            measured = measure_masks(orig, dict(native=a, center=b, one_pass_center=one), small, steps, compose)
            record(results, result["path"], measured)
            return result

        replay.replay_view = decorated_view
        return replay

    source._load_replay = decorated_load
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for field in ("archive", "layout", "historical", "output"):
        parser.add_argument("--"+field, type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0,str(HERE))
    from ssdd_fit_replay_dgx import runtime
    runtime()
    import torch
    from rfdetr import config
    from ssdd_fit_stage_readback import readback_fit_layout
    from ssdd_phase_init import CONFIG, CONFIG_SHA, E1, E1_SHA, RECIPE, RECIPE_SHA, load_api, offline, pinned_bytes
    from ssdd_fit_only import FitOnlyData

    sys.addaudithook(offline)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    require(not torch.cuda.is_initialized(), "GPU stays hidden")
    output = args.output
    require(output.is_absolute() and output.parent.resolve(strict=True) == output.parent
            and not output.exists() and not output.is_symlink(), "Fresh canonical result path")
    require(args.historical.is_absolute() and args.historical.resolve(strict=True)==args.historical,
            "Exact independently closed historical FIT replay path")
    historical_raw = args.historical.read_bytes()
    require(sha256(historical_raw).hexdigest()==INPUT_SUMMARY_SHA, "Exact original closed comparison")
    historical = json.loads(historical_raw)
    layout = args.layout
    pre = readback_fit_layout(args.archive,layout)
    index = json.loads((layout / "train/_annotations.coco.json").read_bytes())
    ids = sorted(row["id"] for row in index["images"])
    require(len(ids)==len(set(ids))==741 and
            sha256(json.dumps(ids,separators=(",",":")).encode()).hexdigest()==IDS_SHA,
            "Exact stock FIT ID order")
    pinned_bytes(RECIPE,RECIPE_SHA)
    model, train = load_api(CONFIG,CONFIG_SHA)._load_configs(RECIPE,layout)
    load_api(E1,E1_SHA).native_recipe(model,train,config)
    data = FitOnlyData(model,train,layout)
    data.setup("fit")
    groups = {}
    baseline = replay_module(groups).replay_fit_population(data._dataset_train,ids)
    require(all(value==historical[key] for key,value in baseline.items()),
            "Same actual original A/B geometry, cohort and paths")
    require(pre==readback_fit_layout(args.archive,layout) and not torch.cuda.is_initialized(),
            "Unchanged FIT inputs and no CUDA initialization")
    result = summarize(groups)
    require(sum(result[p+"_"+s]["census"]["rows"] for p in ("direct","crop") for s in ("small","other"))==
            baseline["membership"]["common_nonempty"], "Complete original common support census")
    safe = dict(status="SSDD_FIT741_CONTINUOUS_PIXEL_FOOTPRINT_REPLAY_COMPLETE",
                protocol_reference_input_sha256=INPUT_SUMMARY_SHA, fit_index_sha256=pre["fit_index_sha256"],
                ordered_fit_ids_sha256=IDS_SHA, views=741, populations=result,
                exact_original_replay_aggregate=True, oracle="clipped_original_boolean_pixel_squares",
                primary="crop_small", weighting="common nonempty A/B annotation rows",
                platform="Linux_aarch64_DGX_CPU", cuda_initialized=False,
                FIT=True, DEV=False, TEST=False, AP=False, forward_calls=0, optimizer_updates=0)
    with output.open("x",encoding="utf8",newline="\n") as stream:
        json.dump(safe,stream,sort_keys=True,indent=2,allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(status=safe["status"],views=safe["views"],common=baseline["membership"]["common_nonempty"]),
                     sort_keys=True),flush=True)


if __name__=="__main__":
    main()

