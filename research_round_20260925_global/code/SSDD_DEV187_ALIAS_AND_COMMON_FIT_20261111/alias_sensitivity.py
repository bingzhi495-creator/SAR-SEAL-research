"""Score all image-byte duplicate representative choices on the closed SSDD pair."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import product
import json
import os
from pathlib import Path
import sys
from types import ModuleType

P299 = ("SSDD_RETRAINED_DEV187_DIAGNOSTIC_20261111/ssdd_retrained_dev187_diagnostic.py",
        "ebefd0eab1394484219cda2991b45f3c3b6a93462938e3e4b0c8297b16785930")
ARMS = ("native", "center")
REPORT = "SSDD_DEV187_BYTE_ALIAS_SENSITIVITY_COMPLETE"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def module_from(path, digest, name):
    payload = path.read_bytes()
    require(0 < len(payload) <= 1024**2 and sha256(payload).hexdigest() == digest,
            "Exact ordinary scoring source required")
    module = ModuleType(name)
    module.__file__ = str(path)
    exec(compile(payload, str(path), "exec"), module.__dict__)
    return module


def duplicate_groups(items):
    groups = defaultdict(list)
    for row in items:
        groups[row["image_ref"]["sha256"]].append(row["image_id"])
    alias = sorted(tuple(sorted(ids)) for ids in groups.values() if len(ids) > 1)
    unique = {ids[0] for ids in groups.values() if len(ids) == 1}
    return unique, alias


def geometry_signatures(gt, duplicate_ids):
    from pycocotools import mask as codec

    sizes = {im["id"]: (im["height"], im["width"]) for im in gt["images"]}
    sig = {i: Counter() for i in duplicate_ids}
    for row in gt["annotations"]:
        i = row["image_id"]
        if i not in sig:
            continue
        raw = row["segmentation"]
        h, w = sizes[i]
        if isinstance(raw, list):
            rle = codec.merge(codec.frPyObjects(raw, h, w))
        else:
            rle = codec.frPyObjects(raw, h, w) if isinstance(raw["counts"],list) else raw
        counts = rle["counts"]
        if isinstance(counts, str):
            counts = counts.encode("ascii")
        sig[i][(row["category_id"], row["iscrowd"], tuple(rle["size"]), counts)] += 1
    return sig


def prediction_signatures(rows, duplicate_ids):
    sig = {i: [] for i in duplicate_ids}
    for row in rows:
        i = row["image_id"]
        if i in sig:
            sig[i].append({k:v for k,v in row.items() if k!="image_id"})
    return sig


def evaluate_subset(ids, gt, rows, template, official):
    selected = set(ids)
    local = dict(gt,images=[im for im in gt["images"] if im["id"] in selected],
                 annotations=[ann for ann in gt["annotations"] if ann["image_id"] in selected])
    pred = {a:[row for row in rows[a] if row["image_id"] in selected] for a in ARMS}
    metrics = {a:template.evaluate_cell(local,pred[a],ids,official)[0] for a in ARMS}
    return dict(images=len(ids), annotations=len(local["annotations"]), metrics=metrics,
                center_minus_native_points=template.differences(metrics["native"],metrics["center"]))


def diagnose(a):
    p299 = module_from(a.code/P299[0],P299[1],"_p329_p299")
    rec = p299.decode(p299.pinned_bytes(a.accepted,p299.ACCEPTED_SHA256,1024**2,"closed SSDD pair"),dict)
    require(rec["status"]=="P286_SSDD_DEV187_INDEPENDENTLY_ACCEPTED" and rec["DEV"] is True
            and rec["TEST"] is rec["FIT"] is rec["training"] is False and rec["coordinator_closed"] is True
            and rec["coordinator_original_os_wait"] is True and rec["coordinator_original_os_exit"]==0
            and rec["primary_GT"]=="INTEGER_PRIMARY" and rec["floating_gt_sha256"] is None,
            "Original closed integer DEV evidence required")
    g = module_from(a.code/p299.GEOMETRY_WRAPPER,p299.GEOMETRY_SHA256,"_p329_geometry")
    _, _, template, scoring, official, _ = g.load_helpers(a.code,a.code/p299.GEOMETRY_CORE)
    b = p299.pinned_bytes(a.manifest,rec["image_manifest_sha256"],1024**2,"fixed DEV image manifest")
    _, ids, items = scoring.fixed_images(scoring.Artifact(
        dict(path=a.manifest.as_posix(),bytes=len(b),sha256=rec["image_manifest_sha256"]), b))
    gt = p299.decode(p299.pinned_bytes(a.gt,rec["integer_gt_sha256"],1024**2,
                                     "closed integer GT",exact_size=scoring.INTEGER_GT_BYTES),dict)
    scoring.validate_gt(gt,ids,items)
    rows = {mode:p299.decode(p299.pinned_bytes(getattr(a,mode),rec["arms"][mode]["prediction_sha256"],
                                              64*1024**2,mode+" original predictions"),list) for mode in ARMS}
    require(scoring.paired_scores(official,gt,rows,ids)==rec["primary"],"Full primary replay differs")
    unique, alias = duplicate_groups(items)
    require(len(unique)==179 and [len(x) for x in alias]==[2]*4, "Fixed four alias-pair structure differs")
    duplicate_ids = set(ids)-unique
    gt_sig = geometry_signatures(gt, duplicate_ids)
    pred = {mode:prediction_signatures(rows[mode],duplicate_ids) for mode in ARMS}
    selections=[]
    for pick in product(*alias):
        kept = unique | set(pick)
        selections.append(evaluate_subset([i for i in ids if i in kept],gt,rows,template,official))
    dropped = evaluate_subset([i for i in ids if i in unique],gt,rows,template,official)
    key_metrics=("AP","AP75","AP90","APs")
    ranges={}
    for k in key_metrics:
        delta=[row["center_minus_native_points"][k] for row in selections]
        require(all(x is not None for x in delta),"Undefined score in fixed subset")
        ranges[k]=dict(min=min(delta),max=max(delta),positive_choices=sum(x>0 for x in delta))
    return dict(status=REPORT, FIT=False,DEV=True,TEST=False,AP=True,training=False,model_forward_calls=0,
                training_seed=5201,original_closed_pair_sha256=p299.ACCEPTED_SHA256,
                full_primary_scores_exact=True,original_primary_delta_points=rec["primary"]["center_minus_native_points"],
                jpeg_alias_groups=len(alias),logical_alias_images=len(duplicate_ids),unique_singletons=len(unique),
                distinct_jpeg=rec["distinct_jpeg"],representative_choices=len(selections),
                same_gt_raster_multiset_pairs=sum(gt_sig[i]==gt_sig[j] for i,j in alias),
                same_prediction_row_sequence_pairs={a:sum(pred[a][i]==pred[a][j] for i,j in alias) for a in ARMS},
                retained_annotation_min=min(x["annotations"] for x in selections),
                retained_annotation_max=max(x["annotations"] for x in selections),
                representative_delta_ranges_points=ranges, drop_all_aliases=dropped,
                selection_rule="enumerate both choices in each of four exact JPEG byte pairs in ID order",
                posthoc_development_sensitivity=True)


def deny_network(event, _):
    if event.startswith('socket.'):
        raise PermissionError('offline scoring')


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for key in ("code","accepted","manifest","gt","native","center","output"):
        cli.add_argument("--"+key,type=Path,required=True)
    a = cli.parse_args()
    require(os.environ.get("CUDA_VISIBLE_DEVICES")=="-1" and not a.output.exists()
            and a.output.is_absolute() and a.output.parent.resolve(strict=True)==a.output.parent,
            "CPU and fresh canonical output required")
    sys.addaudithook(deny_network)
    raw=(json.dumps(diagnose(a),sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    with a.output.open("xb") as out:
        out.write(raw)
        out.flush()
        os.fsync(out.fileno())
    print(json.dumps(dict(status=REPORT,result_sha256=sha256(raw).hexdigest())),flush=True)


if __name__=="__main__":
    main()
