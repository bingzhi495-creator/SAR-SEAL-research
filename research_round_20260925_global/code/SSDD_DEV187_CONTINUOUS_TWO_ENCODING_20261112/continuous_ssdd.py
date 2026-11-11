"""Compare fixed SSDD DEV187 streams against exact continuous and raster annotation overlaps."""

from argparse import ArgumentParser
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType

REFERENCE_SHA = "9f273cab09b086d7f303cc98a2915349cf69cbfa5f36b03394e481b62f27354a"
INTEGER_SHA = "5657600d6ea198f79bd5d557d2a1aadce5bec031d2ecf72a98716248d1183a55"
FLOAT_SHA = "8d98e6f49200d5390c17bee66697de1f582133d08063259a561ed5f710be7b6c"
SEEDS = ("5201", "5202")
ARMS = ("native", "center")
LABELS = ("integer", "candidate_float")
METRICS = ("AP", "AP50", "AP75", "AP90", "APs", "APm", "APl", "AR100")

def need(condition, message):
    if not condition:
        raise ValueError(message)

def pinned(path, expected):
    raw = path.read_bytes()
    need(sha256(raw).hexdigest() == expected, "Closed input bytes differ: " + path.name)
    return raw

def module(path, expected, name):
    raw = pinned(path, expected)
    obj = ModuleType(name)
    obj.__file__ = str(path)
    sys.modules[name] = obj
    exec(compile(raw, str(path), "exec"), obj.__dict__)
    return obj

def exact_original(actual, original):
    need(all(abs(actual[key] - original[key]) <= 1e-12 for key in original if key != "all12")
         and len(actual["all12"]) == len(original["all12"]) == 12
         and all(abs(a - b) <= 1e-12 for a, b in zip(actual["all12"], original["all12"], strict=True)),
         "Both original raster label scores must reproduce")

def paired_delta(arms):
    return {key: 100 * (arms["center"][key] - arms["native"][key]) for key in METRICS}

def run(args):
    started = perf_counter()
    sys.path.append(str(args.shapely_site))
    import shapely
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" and version("numpy") == "1.26.4"
         and version("pycocotools") == "2.0.11" and shapely.__version__ == "2.0.2"
         and shapely.geos_version_string == "3.11.2", "Exact closed CPU evaluator versions required")
    def offline(event, _):
        if event.startswith("socket."):
            raise PermissionError("Offline closed DEV only")
    sys.addaudithook(offline)
    code = args.code_root
    engine = module(code/"HRSID_DEV705_CONTINUOUS_AP_20261111/full_continuous_ap.py",
                    "cba94ecd0df862b6a303117a56fa889bc76c0a8c017af50fff2cdb9859f94139", "_ssdd_continuous")
    wrapped = module(code/"SSDD_DEV187_POST_PRIMARY_RANK_20261110/ssdd_fixed_match_geometry.py",
                     "5aebe92a3b355874e8a20f24ef19b98ace8c20e3537610ec92a9c0e026b260e3", "_ssdd_continuous_adapter")
    official = wrapped.load_helpers(code, code/"HRSID_MASK_PHASE_REPEAT_DEV705_20261109/dev705_paired_mask_geometry.py")[4]
    shift = module(code/"HRSID_COMMON_DEV705_FIXED_SHIFT_AP_20261111/fixed_shift_ap.py",
                   "68a7ec9e0811cc5b18d2b97a09c13b2471f3908a36b39c3dedd4a50b098f97f2", "_ssdd_continuous_runs")
    vector_path = code/"HRSID_FIT114_VECTOR_LABEL_GEOMETRY_20261110"
    vector = module(vector_path/"vector_label_geometry.py",
                    "4877c48d38a03e58e97163381ba544df3c90a78dde85c9141085fa0b92da3b09", "_ssdd_continuous_vector")
    math_api = module(vector_path/"vector_population_math.py",
                      "0821d2cc3b6ca793d33fdfe35b634f3678aad072fc55b86352fa2501e969fc7e", "_ssdd_continuous_footprint")
    reference = json.loads(pinned(args.reference, REFERENCE_SHA))
    need(reference["status"] == "P370_SSDD_TWO_SEED_FIXED_DEV187_FLOAT_SENSITIVITY_COMPLETE"
         and reference["DEV"] and not reference["TEST"] and reference["regenerated_float_equal"]
         and reference["max_dets"] == [1, 10, 100], "Exact accepted raster sensitivity scope")
    gt = dict(integer=json.loads(pinned(args.integer_gt, INTEGER_SHA)),
              candidate_float=json.loads(pinned(args.float_gt, FLOAT_SHA)))
    a, b = (gt[key] for key in LABELS)
    need(a["images"] == b["images"] and len(a["images"]) == 187 and a["categories"] == b["categories"]
         and len(a["annotations"]) == len(b["annotations"]) == 380, "Exact shared DEV population")
    for x, y in zip(a["annotations"], b["annotations"], strict=True):
        need(x.keys() == y.keys() and all(x[k] == y[k] for k in x if k not in ("segmentation","bbox","area")),
             "Only accepted GT geometry fields may differ")
    polygons, census, neutral = {}, {}, set()
    for label in LABELS:
        polygons[label], unsupported, census[label] = engine.polygon_support(gt[label], vector)
        neutral.update(unsupported)
    need(0 < 380 - len(neutral) <= 380, "No valid common vector population")
    seeds = {}
    for seed in SEEDS:
        prior = reference["seeds"][seed]
        need(prior["primary_exact"] and prior["training_seed"] == int(seed), "Prior independently matched seed required")
        streams = {arm: json.loads(pinned(getattr(args, "seed"+seed+"_"+arm), prior["prediction_sha256"][arm]))
                   for arm in ARMS}
        need(all(len(rows) == prior["candidate_counts"][arm] for arm, rows in streams.items()),
             "Exact accepted full prediction streams required")
        out = {}
        for label in LABELS:
            raster, continuous, audited = {}, {}, {}
            for arm in ARMS:
                rows = streams[arm]
                original, original_rank, _ = engine.score(gt[label], rows, official)
                exact_original(original, prior[label][arm])
                raster[arm], rrank, _ = (engine.score(gt[label], rows, official, neutral) if neutral
                                           else (original, original_rank, {}))
                continuous[arm], crank, audited[arm] = engine.score(gt[label], rows, official, neutral, polygons[label],
                                                                   shift.runs, math_api.run_footprint_metrics)
                need(original_rank == rrank == crank and audited[arm]["dense_geometry_spots"] == 16
                     and audited[arm]["pruned_dense_spots"] == 8, "Require exact candidate ordering and dense checks")
            out[label] = dict(raster_common=raster, continuous_common=continuous,
                              raster_delta_points=paired_delta(raster), continuous_delta_points=paired_delta(continuous),
                              geometry_checks=audited)
            print(json.dumps(dict(event="SSDD_CONTINUOUS_LABEL_COMPLETE", seed=seed, label=label,
                                  raster_delta=out[label]["raster_delta_points"]["AP"],
                                  continuous_delta=out[label]["continuous_delta_points"]["AP"])), flush=True)
        shifts = {mode: {k: out["candidate_float"][mode][k] - out["integer"][mode][k] for k in METRICS}
                  for mode in ("raster_delta_points", "continuous_delta_points")}
        seeds[seed] = dict(cells=out, float_minus_integer_paired_effect_shift_points=shifts)
    result = dict(status="SSDD_DEV187_TWO_ENCODING_FULL_CONTINUOUS_OVERLAP_ACTUAL", images=187,
        annotations=380, valid_common_polygons=380-len(neutral), neutral_common_polygons=len(neutral),
        label_support=census, seeds=seeds, raster_full_exact=True, fixed_original_candidates_and_scores=True,
        max_dets=[1, 10, 100], dev=True, fit=False, test=False, model_forward_calls=0, optimizer_updates=0,
        source_reference_sha256=REFERENCE_SHA, wall_seconds=perf_counter()-started)
    raw = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n").encode()
    with args.output.open("xb") as stream:
        stream.write(raw)
    print(json.dumps(dict(status=result["status"], sha256=sha256(raw).hexdigest())), flush=True)

def main():
    parser = ArgumentParser(description=__doc__)
    for name in ("code-root", "shapely-site", "reference", "integer-gt", "float-gt", "output",
                 "seed5201-native", "seed5201-center", "seed5202-native", "seed5202-center"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    need(args.output.is_absolute() and not args.output.exists(), "Fresh output required")
    run(args)

if __name__ == "__main__":
    main()