"""Frozen Instance-RSDD release scoring after independent image-only replay."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from importlib.metadata import version
from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace

import numpy as np


LOCK = "5c0c9410db15710e34fe453271bd3a99a1f1a50eb365bd354f884e40b88c8b85"
AUTHOR_TEST = "ed5be06ac39b4cd5c373c00fdba132bc8f0ad1260023e2001c5e2d24a33f0760"
SCORE_SOURCE = "77a3fbfaa04ac08287eff4ec7bb827ce3914e61aa20dd4f1e6f9c1e3b8ccc9fa"
GEOMETRY_SOURCE = "9ecbac508e1506b6cee2706a31f497eb4a46dd9bb62fbd56340e9e64ec981fdc"
READBACK_STATUS = "INSTANCE_RSDD_TEST740_RFDETR_EDIT_REPLAY_ACCEPTED_V1"
CAPTURE_STATUS = "INSTANCE_RSDD_TEST740_RFDETR_READBACK_COMPLETE_V1"
SOURCE_CKPT = "011bd30ed056c871be1d61e14d71b3b2b258330a9d20cf2f164ca4a7c4f23c11"
ARMS = ("raw", "fixed", "free_matched_count", "free_natural", "native_matched_count",
        "attraction_dynamic", "attraction_static_at_dynamic_count", "learned_at_attraction_count")
PAIRS = (("free_natural","raw"),("free_matched_count","fixed"),("free_natural","native_matched_count"),
         ("learned_at_attraction_count","attraction_dynamic"),
         ("learned_at_attraction_count","attraction_static_at_dynamic_count"))
FIELDS = ("continuous_iou_points","fp_per_truth_area_points","fn_per_truth_area_points")


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(data):
    return sha256(data).hexdigest()


def pinned(path, expected):
    data = path.read_bytes()
    require(digest(data) == expected,"Exact input digest: "+path.name)
    return data


def module(path, sha):
    pinned(path,sha)
    spec = spec_from_file_location(path.stem,path)
    value = module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def records(path, sha):
    return [json.loads(row) for row in pinned(path,sha).splitlines()]


def terminal(path, expected_sha, result_sha, phase):
    value = json.loads(pinned(path,expected_sha))
    require(value["phase"] == phase and value["actual_exit"] == 0 and value["output_footer_verified"] is True
            and value["result_sha256"] == result_sha,"Observed exact accepted label-free output")
    return value


def representatives(blocks, ids):
    chosen = {}
    for image_id in sorted(ids):
        chosen.setdefault(blocks[image_id]["rgb"],image_id)
    return set(chosen.values())


def coco_annotation_ids(gt):
    """COCOeval reserves zero in match arrays. Preserve all other data and ordering."""
    original = [row["id"] for row in gt["annotations"]]
    require(len(set(original)) == len(original) and all(type(i) is int for i in original),"Unique author annotation IDs")
    if all(i > 0 for i in original):
        return gt,False
    annotations = [dict(row,id=i) for i,row in enumerate(gt["annotations"],1)]
    return dict(gt,annotations=annotations),True


def geometry_summary(groups):
    if not groups:
        return None
    values = np.array([np.mean(groups[g],axis=0) for g in sorted(groups)],dtype=np.float64)
    group_ids = np.random.Generator(np.random.PCG64(5217)).integers(len(groups),size=(4096,len(groups)))
    contrasts = {}
    for a,b in PAIRS:
        change = values[:,ARMS.index(a)]-values[:,ARMS.index(b)]
        interval = np.percentile(change[group_ids].mean(axis=1),[2.5,97.5],axis=0)
        contrasts[a+" minus "+b] = {m:dict(delta=float(change[:,i].mean()),interval_95=interval[:,i].tolist())
                                     for i,m in enumerate(FIELDS)}
    return dict(paired_contrasts_points=contrasts,groups=len(groups),
                group_equal_arms={name:dict(zip(FIELDS,values[:,i].mean(axis=0).tolist(),strict=True))
                                  for i,name in enumerate(ARMS)})


def admitted_inputs(a, adapter, geometry):
    terminal(a.edit_terminal,a.edit_terminal_sha,a.edit_receipt_sha,"verify-edits")
    terminal(a.cap_terminal,a.cap_terminal_sha,a.cap_receipt_sha,"verify-capture")
    edit = json.loads(pinned(a.edit_receipt,a.edit_receipt_sha))
    cap = json.loads(pinned(a.cap_receipt,a.cap_receipt_sha))
    require(edit["status"] == READBACK_STATUS and cap["status"] == CAPTURE_STATUS
            and cap["checkpoint_sha256"] == SOURCE_CKPT and edit["protocol_sha256"] == cap["protocol_sha256"] == LOCK
            and edit["GT_opened"] is False and edit["exact_replayed_policy_choices"] is True
            and edit["exact_support_xor_area"] is True and cap["GT_opened"] is False
            and edit["predictions_sha256"] == cap["predictions_sha256"]
            and edit["candidate_audit_sha256"] == cap["audit_sha256"],"Verified archived capture and edits")
    sources = (records(a.predictions,edit["predictions_sha256"]),records(a.audit,edit["candidate_audit_sha256"]),
               records(a.edits,edit["edits_sha256"]))
    require(all(len(s) == 740 for s in sources),"Every author image in all three verified streams")
    blocks = {}
    for pred,audit,curve in zip(*sources,strict=True):
        image_id = pred["image_id"]
        require(image_id == audit["image_id"] == curve["image_id"] and image_id not in blocks,
                "Native identity and bound order")
        masks = adapter.native_masks(geometry,pred,audit,image_id)
        ranks = [x["position"] for x in audit["candidates"]]
        changes = geometry.verify_edits(curve["changes"],masks,512,512)
        require(audit["original_hw"] == [512,512] and 1 <= audit["acquisition_group"] <= 125
                and audit["sensor"] in {"GF-3","TerraSAR-X"},"Admitted acquisition group and sensor")
        blocks[image_id] = dict(stock=pred["coco"],ranks=ranks,rgb=audit["rgb_sha256"],
                                sensor=audit["sensor"],group=audit["acquisition_group"], file=audit["source_image_name"],
                                changes={r:{arm:dict(swaps=k,rle=rle) for arm,(k,rle) in v.items()}
                                         for r,v in changes.items()})
    require(Counter(x["sensor"] for x in blocks.values()) == {"GF-3":583,"TerraSAR-X":157}
            and len({x["group"] for x in blocks.values()}) == 112,"Exact released test population")
    return blocks


def fixed_geometry(core, shapely, associations, polygons, blocks):
    groups, stats, exposed_groups = defaultdict(list), Counter(),set()
    marks = {str(v):{arm:Counter(gained=0,lost=0,retained=0,both_below=0) for arm in ARMS}
             for v in (.5,.75,.9)}
    for gid,(image_id,rank,stock) in associations.items():
        polygon = polygons[gid]
        if not polygon.is_valid:
            stats["invalid_matched_continuous_excluded"] += 1
            continue
        original = stock["segmentation"]
        area = int(core.codec.area(original))
        if not 32 <= area < 1024:
            stats["out_of_area_matched"] += 1
            continue
        frame = blocks[image_id]
        edits = frame["changes"].get(rank,{})
        exposed = edits.get("fixed",{}).get("swaps",0) > 0
        stats["eligible_anchors"] += 1
        if exposed:
            stats["fixed_positive_anchors"] += 1
            exposed_groups.add(frame["group"])
        values,cache = [],{}
        for arm in ARMS:
            rle = edits.get(arm,{}).get("rle",original)
            key = rle["counts"]
            if key not in cache:
                iou = core.geometric_iou(rle,polygon,shapely)
                intersection = iou * (polygon.area+area) / (1.0+iou)
                cache[key] = [100*iou,100*(area-intersection)/polygon.area,100*(polygon.area-intersection)/polygon.area]
            values.append(cache[key])
        groups[frame["group"]].append(values)
        for name,records_ in marks.items():
            before = values[0][0] >= 100*float(name)
            for arm,row in zip(ARMS,values,strict=True):
                after = row[0] >= 100*float(name)
                kind = ("retained" if after else "lost") if before else ("gained" if after else "both_below")
                records_[arm][kind] += 1
    return dict(original_anchors=len(associations),counts=dict(stats),positive_exposure_groups=len(exposed_groups),
                transitions=marks,stats=geometry_summary(groups))


def result_view(gt, keep, blocks, score, core, polygons, shapely):
    chosen = dict(gt,images=[v for v in gt["images"] if v["id"] in keep],
                     annotations=[v for v in gt["annotations"] if v["image_id"] in keep])
    scoped = {k:v for k,v in blocks.items() if k in keep}
    metrics, associations = {},{}
    for arm in ARMS:
        metrics[arm], anchor = score.score_one(chosen,scoped,arm,anchors=arm=="raw")
        if arm == "raw":
            associations = anchor
    deltas = {a+" minus "+b:{m:100*(v-metrics[b][m]) if min(v,metrics[b][m])>=0 else None
                           for m,v in metrics[a].items()} for a,b in PAIRS}
    return dict(images=len(keep), annotations=len(chosen["annotations"]),sensor_groups=len({x["group"] for x in scoped.values()}),
                metrics_fraction=metrics,paired_contrasts_points=deltas,geometry=fixed_geometry(core,shapely,associations,polygons,scoped))


def run(a):
    import shapely
    require(shapely.__version__ == "2.0.2" and shapely.geos_version_string == "3.11.2", "Qualified continuous geometry")
    score = module(a.code/"PSEG_RFDETR_FROZEN_SOURCE_HEAD_20261122/score.py",SCORE_SOURCE)
    adapter = module(a.code/"UEIKAP_RFDETR_CARRIER_20261122/score_rf.py",GEOMETRY_SOURCE)
    core = adapter.load_geometry(a.code/"UEIKAP_RESERVED_HULL_FROZEN_20261122/score.py")
    blocks = admitted_inputs(a,adapter,core)
    # All accepted candidate and edit streams are verified before reading target labels.
    gt = {"info":{},**json.loads(pinned(a.author_test,AUTHOR_TEST))}
    require(len(gt["images"])==740 and len(gt["annotations"])==872 and 1 in {c["id"] for c in gt["categories"]}
            and {row["category_id"] for row in gt["annotations"]} == {1}
            and set(blocks)=={x["id"] for x in gt["images"]} and all(x["width"]==x["height"]==512 and
                 x["file_name"] == blocks[x["id"]]["file"] for x in gt["images"]), "Exact original COCO file-to-image join")
    require(all(len(row["segmentation"]) == 1 for row in gt["annotations"]), "One original ring per instance")
    gt, alias_ids = coco_annotation_ids(gt)
    polygons = {x["id"]:shapely.Polygon(np.asarray(x["segmentation"][0],dtype=float).reshape(-1,2))
                for x in gt["annotations"]}
    require(sum(not p.is_valid for p in polygons.values())==1, "Original single non-simple outline retained in COCO")
    complete = set(blocks)
    gf = {key for key,value in blocks.items() if value["sensor"]=="GF-3"}
    selected = {"gf3":gf,"full":complete,"terrasar_x":complete-gf,
                "gf3_deduplicated":representatives(blocks,gf),"full_deduplicated":representatives(blocks,complete)}
    results, computed = {}, {}
    for title, ids in selected.items():
        key = frozenset(ids)
        if key not in computed:
            computed[key] = result_view(gt,ids,blocks,score,core,polygons,shapely)
        results[title] = computed[key]
    natural = "free_natural minus raw"
    paired = "free_matched_count minus fixed"
    main = results["gf3"]
    ap = main["paired_contrasts_points"][natural]["AP"]
    main_geo = main["geometry"]
    g = main_geo["stats"]
    checks = dict(positive_continuous_lower=bool(g and g["paired_contrasts_points"][natural][FIELDS[0]]["interval_95"][0] > 0),
                  coverage=main_geo["counts"].get("fixed_positive_anchors",0) >= 20 and main_geo["positive_exposure_groups"] >= 8,
                  same_count_ap_positive=main["paired_contrasts_points"][paired]["AP"] > 0,
                  dedup_ap_no_sign_reversal=all(results["gf3_deduplicated"]["paired_contrasts_points"][k]["AP"] >= 0
                                                 for k in (natural,paired)))
    return dict(status="INSTANCE_RSDD_TEST740_FROZEN_HRSID_SCORE_COMPLETE_V1",author_TEST=True,gt_sha256=AUTHOR_TEST,
                evaluator_annotation_ids_rekeyed=alias_ids,
                detector_forwards=0,optimizer_updates=0,non_simple_ap_retained=1,views=results,criteria=checks,
                replicate_5218=ap>0 and all(checks.values()),full_replication_eligible=ap>=1 and all(checks.values()),
                primary_ap_points=ap,protocol_sha256=LOCK,capture_receipt_sha256=a.cap_receipt_sha,
                edit_receipt_sha256=a.edit_receipt_sha)


def main():
    p = argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument("--authority",required=True,type=Path)
    p.add_argument("--authority-sha",required=True)
    p.add_argument("--output",required=True,type=Path)
    a = p.parse_args()
    values = json.loads(pinned(a.authority,a.authority_sha))
    paths = {"protocol","code","author_test","edit_terminal","edit_receipt","cap_terminal","cap_receipt","predictions","audit","edits","shapely_site"}
    hashes = {"edit_terminal_sha","edit_receipt_sha","cap_terminal_sha","cap_receipt_sha"}
    require(set(values)==paths|hashes and not a.output.exists() and sys.flags.isolated and sys.dont_write_bytecode
            and os.environ.get("CUDA_VISIBLE_DEVICES")=="-1" and np.__version__=="1.26.4" and version("pycocotools")=="2.0.11",
            "Fresh pinned CPU scoring scope")
    inputs = SimpleNamespace(**{k:Path(v) if k in paths else v for k,v in values.items()})
    pinned(inputs.protocol,LOCK)
    sys.path.append(str(inputs.shapely_site))
    start = perf_counter()
    result = run(inputs)
    result.update(worker_sha256=digest(Path(__file__).read_bytes()),authority_sha256=a.authority_sha,elapsed_seconds=perf_counter()-start)
    payload = (json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    a.output.mkdir()
    (a.output/"RESULT_SAFE.json").write_bytes(payload)
    print(json.dumps(dict(status=result["status"],result_sha256=digest(payload))),flush=True)


if __name__ == "__main__":
    main()
