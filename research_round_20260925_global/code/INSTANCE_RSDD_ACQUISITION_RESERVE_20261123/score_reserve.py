"""Score the locked 74-image parent reserve after closing all image-only inputs."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import sys
from time import perf_counter
from types import ModuleType, SimpleNamespace

import numpy as np


HISTORIC = "7039d7d0336f6a0cc3b64bfbe4b0c446b38b073cc90764f446a4dc6d09f90d85"
CENSUS = "9454f1de69dc0d9450c30005647333f048dcea184b0561acf5dac518759e7bc6"
TRAIN = "54c92363aafe07dca084382a258190d436005d8df839f5de13995c89bd662b92"
ROSTER = "2614e567e9ef1b2c791c1b357b03adfe7302ece53e911c1963c40e623ad405fc"
STATUS = "INSTANCE_RSDD_RESERVE74_SCORE_COMPLETE_V1"
PHASES = ("intake", "capture", "replay", "source")
RAW, FIXED, MATCHED, NATURAL = ARMS = ("raw", "fixed", "free_matched_count", "free_natural")
PAIRS = ((MATCHED, FIXED), (NATURAL, RAW), (MATCHED, RAW), (FIXED, RAW), (NATURAL, MATCHED))
PRIMARY = MATCHED + " minus " + FIXED
FIELDS = ("continuous_iou_points", "fp_per_truth_area_points", "fn_per_truth_area_points")


def check(ok, message):
    if not ok:
        raise ValueError(message)


def pin(path, expected):
    data = path.read_bytes()
    check(sha256(data).hexdigest() == expected, "Input fingerprint differs: " + path.name)
    return data


def admitted(a, old, core, adapter, census):
    closed = {}
    for phase in PHASES:
        result_sha = getattr(a, phase+"_receipt_sha")
        t = json.loads(pin(getattr(a, phase+"_terminal"), getattr(a, phase+"_terminal_sha")))
        check(t["actual_exit"] == 0 and t["output_footer_verified"] is True and t["result_sha256"] == result_sha,
              "Observed exit and exact result required for " + phase)
        closed[phase] = json.loads(pin(getattr(a, phase+"_receipt"), result_sha))
    intake, cap, rep, src = (closed[p] for p in PHASES)
    check(intake["status"] == "INSTANCE_RSDD_RESERVE74_IMAGE_INTAKE_COMPLETE_V1" and intake["roster_sha256"] == ROSTER
          and intake["images"] == intake["exact_qualified_rf_rgb"] == 74 and intake["target_polygon_rows_opened"] == 0
          and intake["common_test_parents"] == intake["historical_test_rgb_matching_reserve_images"] == 0,
          "Complete image-only intake and prior author-test comparison")
    check(src["status"] == "INSTANCE_RSDD_RESERVE74_SOURCE_IDENTITIES_VERIFIED_V1" and src["passed"] is True
          and src["reserve_roster_sha256"] == ROSTER and src["reserve_manifest_sha256"] == intake["image_manifest_sha256"]
          and src["reserve_intake_result_sha256"] == a.intake_receipt_sha and src["reserve_intake_terminal_sha256"] == a.intake_terminal_sha
          and src["historical_test_manifest_sha256"] == intake["historical_test_manifest_sha256"]
          and (src["source_images"], src["source_fit109"], src["historical_test_images"], src["reserve_images"], src["reserve_parents"]) == (2937,109,740,74,13)
          and src["reserve_rgb_matches_hrsid_fit"] == src["reserve_rgb_matches_historical_test"] == src["reserve_parent_matches_historical_test"] == 0
          and src["confirmed_common_hrsid_fitting_acquisitions"] == 0
          and src["reserve_within_duplicate_images"] == intake["within_reserve_duplicate_extra_images"], "Complete source exposure qualification")
    check(cap["status"] == "INSTANCE_RSDD_RESERVE74_RFDETR_READBACK_COMPLETE_V1" and cap["images"] == 74
          and cap["checkpoint_sha256"] == old.SOURCE_CKPT and cap["exact_native_parity"] is True
          and cap["intake_sha256"] == intake["image_manifest_sha256"] and cap["author_roster_sha256"] == ROSTER
          and cap["intake_result_sha256"] == a.intake_receipt_sha and cap["intake_terminal_sha256"] == a.intake_terminal_sha,
          "Independent native capture verification on the locked reserve")
    check(rep["status"] == "INSTANCE_RSDD_RESERVE74_EDIT_ACCEPTED_V1" and rep["exact_replayed_policy_choices"] is True
          and rep["capture_receipt_sha256"] == a.capture_receipt_sha and rep["predictions_sha256"] == cap["predictions_sha256"]
          and rep["candidate_audit_sha256"] == cap["audit_sha256"] and rep["GT_opened"] is cap["GT_opened"] is False
          and rep["protocol_sha256"] == cap["protocol_sha256"] == a.protocol_sha
          and rep["prelock_commit"] == cap["prelock_commit"] == a.prelock_commit, "Independent frozen edit verification")
    manifest = json.loads(pin(a.manifest, intake["image_manifest_sha256"]))
    items = manifest["items"]
    streams = (old.records(a.predictions, cap["predictions_sha256"]), old.records(a.audit, cap["audit_sha256"]),
               old.records(a.edits, rep["edits_sha256"]))
    check(all(len(s) == 74 for s in (*streams, items)) and manifest["expected_ids"] == [r["image_id"] for r in items],
          "Full author intake, candidate and edit rosters")
    blocks = {}
    for item, pred, audit, edit in zip(items, *streams, strict=True):
        i = item["image_id"]
        check(i == edit["image_id"] == pred["image_id"] == audit["image_id"] and i not in blocks
              and all(item[k] == audit[k] for k in ("source_image_name", "rgb_sha256", "acquisition_group", "sensor", "original_hw"))
              and audit["original_hw"] == [512,512], "Manifest-bound candidate identities")
        masks = adapter.native_masks(core, pred, audit, i)
        changes = {}
        for key, options in edit["changes"].items():
            j = int(key)
            check(key == str(j) and 1 <= j <= len(masks) and options and set(options) <= set(ARMS[1:]), "Active compact candidate policies")
            stock = masks[j-1]
            area = int(core.codec.area(stock))
            check(32 <= area < 1024 and options.get(FIXED, {}).get("swaps",0) == options.get(MATCHED, {}).get("swaps",0),
                  "Same eligible original and same per-candidate fixed/matched budget")
            for update in options.values():
                k = update["swaps"]
                after = core.checked_rle(update["rle"], 512, 512)
                overlap = int(core.codec.area(core.codec.merge([stock,after], intersect=True)))
                check(set(update) == {"swaps","rle"} and type(k) is int and 1 <= k <= 8 and int(core.codec.area(after)) == area
                      and area-overlap == k, "Active signed native additions, deletions and original area")
            changes[j] = options
        blocks[i] = dict(stock=pred["coco"], ranks=[r["position"] for r in audit["candidates"]], changes=changes,
                         rgb=item["rgb_sha256"], group=item["acquisition_group"], sensor=item["sensor"], file=item["source_image_name"])
    counts = Counter((v["group"], v["sensor"]) for v in blocks.values())
    check(counts == {(p["paper_scene"],p["sensor"]):p["images"] for p in census["per_parent"]}
          and len({v["rgb"] for v in blocks.values()}) == intake["rgb_unique"], "Exact parent census and decoded deduplication")
    return blocks, src


TOKENS = re.compile(rb'"(?:\\.|[^"\\])*"|[\[\]{}]')


def container_end(data, start):
    depth = 0
    for token in TOKENS.finditer(data, start):
        char = data[token.start()]
        if char in b"{[":
            depth += 1
        elif char in b"}]":
            depth -= 1
            if depth == 0:
                return token.end()
    raise ValueError("Unterminated author JSON container")


def select_labels(data, blocks):
    """Hash the whole archived file; decode annotation objects only for admitted IDs."""
    head, = re.finditer(rb'"annotations"\s*:\s*\[', data)
    start = head.end()-1
    pos, keep = start+1, []
    boundary = re.compile(rb'\s*(?:,\s*)?([\]{])')
    while True:
        token = boundary.match(data, pos)
        check(token is not None, "Author annotation array framing")
        if token.group(1) == b"]":
            break
        begin = token.start(1)
        pos = container_end(data, begin)
        entry = memoryview(data)[begin:pos]
        image, = re.finditer(rb'(?:\{|,)\s*"image_id"\s*:\s*(\d+)\s*(?=[,}])', entry)
        if int(image.group(1)) in blocks:
            keep.append(json.loads(entry.tobytes()))
    header = json.loads(data[:start]+b"[]"+data[token.end():])
    check(len(header["images"]) == 2810 and 1 in {c["id"] for c in header["categories"]}, "Original author training split")
    images = [r for r in header["images"] if r["id"] in blocks]
    check([r["id"] for r in images] == list(blocks) and all(r["width"] == r["height"] == 512 and r["file_name"] == blocks[r["id"]]["file"] for r in images)
          and all(r["category_id"] == 1 and len(r["segmentation"]) == 1 for r in keep), "Selected author image and polygon identities")
    return {"info": {}, **header, "images":images, "annotations":keep}


def bootstrap(parent_means, keys):
    """Resample whole blocks and weight represented parents by each block's draw multiplicity."""
    labels = sorted(set(keys))
    group = np.asarray([labels.index(k) for k in keys])
    sums = np.stack([parent_means[group == j].sum(axis=0) for j in range(len(labels))])
    sizes = np.bincount(group)
    sample = np.random.Generator(np.random.PCG64(5217)).integers(len(labels), size=(4096,len(labels)))
    means = sums[sample].sum(axis=1) / sizes[sample].sum(axis=1)[:,None]
    return np.percentile(means, (2.5,97.5), axis=0).T.tolist()


def geometry(anchors, polygons, blocks, dates, core, shapely):
    groups = defaultdict(list)
    counts = {p:Counter(original_matches=0, eligible_matches=0, positive_fixed_edits=0,
                        invalid_polygon_matches=0, area_excluded_matches=0) for p in {r["group"] for r in blocks.values()}}
    for gid, (i,j,stock) in anchors.items():
        parent = blocks[i]["group"]
        c = counts[parent]
        c["original_matches"] += 1
        poly = polygons[gid]
        if not poly.is_valid or poly.area <= 0:
            c["invalid_polygon_matches"] += 1
            continue
        original = stock["segmentation"]
        area = int(core.codec.area(original))
        if not 32 <= area < 1024:
            c["area_excluded_matches"] += 1
            continue
        c["eligible_matches"] += 1
        changes = blocks[i]["changes"].get(j, {})
        c["positive_fixed_edits"] += int(changes.get(FIXED, {}).get("swaps",0) > 0)
        values, cache = [], {}
        for arm in ARMS:
            mask = changes.get(arm, {}).get("rle", original)
            key = mask["counts"]
            if key not in cache:
                iou = core.geometric_iou(mask, poly, shapely)
                overlap = iou*(area+poly.area)/(1+iou)
                cache[key] = [100*iou,100*(area-overlap)/poly.area,100*(poly.area-overlap)/poly.area]
            values.append(cache[key])
        groups[parent].append(values)
    parents = sorted(groups)
    exposed = [p for p,c in counts.items() if c["positive_fixed_edits"]]
    result = dict(counts_by_parent=counts, counts=dict(sum(counts.values(),Counter())), represented_parents=len(parents),
                  represented_sensor_date_blocks=len({dates[p] for p in parents}), exposed_parents=len(exposed),
                  exposed_sensor_date_blocks=len({dates[p] for p in exposed}), paired_contrasts_points={}, group_equal_arms={})
    if parents:
        means = np.stack([np.mean(groups[p], axis=0) for p in parents])
        for idx, arm in enumerate(ARMS):
            result["group_equal_arms"][arm] = dict(zip(FIELDS, means[:,idx].mean(axis=0).tolist(), strict=True))
        for left, right in PAIRS:
            d = means[:,ARMS.index(left)] - means[:,ARMS.index(right)]
            ci_date = bootstrap(d, [dates[p] for p in parents])
            ci_parent = bootstrap(d, parents)
            result["paired_contrasts_points"][left+" minus "+right] = {
                f:dict(delta=float(mu), sensor_date_interval_95=di, parent_interval_95=pi)
                for f,mu,di,pi in zip(FIELDS,d.mean(axis=0),ci_date,ci_parent,strict=True)}
    return result


def score_view(gt, ids, blocks, polygons, dates, evaluator, core, shapely):
    chosen = dict(gt, images=[r for r in gt["images"] if r["id"] in ids], annotations=[r for r in gt["annotations"] if r["image_id"] in ids])
    subset = {i:v for i,v in blocks.items() if i in ids}
    metrics, anchors = {}, {}
    for arm in ARMS:
        scores, matches = evaluator.score_one(chosen, subset, arm, anchors=arm == RAW)
        metrics[arm] = {key:float(x) if np.isfinite(x) and x >= 0 else None for key,x in scores.items()}
        if arm == RAW:
            anchors = matches
    contrasts = {a+" minus "+b:{k:100*(v-metrics[b][k]) if v is not None and metrics[b][k] is not None else None
                               for k,v in metrics[a].items()} for a,b in PAIRS}
    g = geometry(anchors, polygons, subset, dates, core, shapely)
    continuous = g["paired_contrasts_points"].get(PRIMARY, {}).get(FIELDS[0])
    ap = contrasts[PRIMARY]["AP"]
    gates = dict(fixed_exposure_20_matches_8_parents=g["counts"].get("positive_fixed_edits",0) >= 20 and g["exposed_parents"] >= 8,
                 matched_ap_positive=ap is not None and ap > 0,
                 sensor_date_paired_iou_lower_positive=continuous is not None and continuous["sensor_date_interval_95"][0] > 0)
    return dict(images=len(ids), annotations=len(chosen["annotations"]), metrics_fraction=metrics, paired_contrasts_points=contrasts,
                geometry=g, criteria=gates, within_recipe_mechanism_criterion=all(gates.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--authority", required=True, type=Path)
    parser.add_argument("--authority-sha", required=True)
    parser.add_argument("--output", required=True, type=Path)
    cmd = parser.parse_args()
    start = perf_counter()
    raw = json.loads(pin(cmd.authority, cmd.authority_sha))
    phase_paths = {p+s for p in PHASES for s in ("_terminal","_receipt")}
    paths = {"protocol","code","census","shapely_site","author_train","manifest","predictions","audit","edits"} | phase_paths
    hashes = {p+"_sha" for p in phase_paths} | {"protocol_sha"}
    check(set(raw) == paths | hashes | {"prelock_commit"} and all(Path(raw[k]).is_absolute() for k in paths)
          and all(re.fullmatch(r"[0-9a-f]{64}",raw[k]) for k in hashes) and re.fullmatch(r"[0-9a-f]{40}",raw["prelock_commit"])
          and sys.flags.isolated and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" and not cmd.output.exists()
          and np.__version__ == "1.26.4" and version("pycocotools") == "2.0.11", "Exact qualified CPU authority and fresh output")
    a = SimpleNamespace(**{k:Path(v) if k in paths else v for k,v in raw.items()})
    pin(a.protocol, a.protocol_sha)
    census = json.loads(pin(a.census, CENSUS))
    source = a.code/"INSTANCE_RSDD_FROZEN_HRSID_20261122/score_rsdd.py"
    old = ModuleType("historical_rsdd_score")
    old.__file__ = str(source)
    exec(compile(pin(source,HISTORIC),str(source),"exec"),old.__dict__)
    evaluator = old.module(a.code/"PSEG_RFDETR_FROZEN_SOURCE_HEAD_20261122/score.py",old.SCORE_SOURCE)
    adapter = old.module(a.code/"UEIKAP_RFDETR_CARRIER_20261122/score_rf.py",old.GEOMETRY_SOURCE)
    core = adapter.load_geometry(a.code/"UEIKAP_RESERVED_HULL_FROZEN_20261122/score.py")
    sys.path.append(str(a.shapely_site))
    import shapely
    check(shapely.__version__ == "2.0.2" and shapely.geos_version_string == "3.11.2", "Original geometry runtime")
    blocks, src = admitted(a,old,core,adapter,census)
    gt, rekeyed = old.coco_annotation_ids(select_labels(pin(a.author_train, TRAIN), blocks))
    polygons = {v["id"]:shapely.Polygon(np.asarray(v["segmentation"][0]).reshape(-1,2)) for v in gt["annotations"]}
    dates = {p["paper_scene"]:(p["sensor"],p["calendar_date"]) for p in census["per_parent"]}
    dated = set(census["date_separated_sensitivity"]["parent_scene_ids"])
    full = set(blocks)
    gf = {i for i,v in blocks.items() if v["sensor"] == "GF-3"}
    subsets = dict(full=full, gf3=gf, terrasar_x=full-gf, date_separated={i for i,v in blocks.items() if v["group"] in dated},
                   full_deduplicated=old.representatives(blocks,full))
    def view(ids):
        return score_view(gt,ids,blocks,polygons,dates,evaluator,core,shapely)
    cache, views = {}, {}
    for name, ids in subsets.items():
        key = frozenset(ids)
        if key not in cache:
            cache[key] = view(ids)
        views[name] = cache[key]
    per_parent = {}
    for p in sorted(dates):
        v = view({i for i,b in blocks.items() if b["group"] == p})
        per_parent[p] = dict(images=v["images"], annotations=v["annotations"], sensor_date=dates[p],
                             metrics_fraction=v["metrics_fraction"], counts=v["geometry"]["counts_by_parent"][p])
    result = dict(status=STATUS, protocol_sha256=a.protocol_sha, prelock_commit=a.prelock_commit, census_sha256=CENSUS, author_train_sha256=TRAIN,
                  authority_sha256=cmd.authority_sha, worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                  receipt_sha256={p:getattr(a,p+"_receipt_sha") for p in PHASES}, source_exposure_qualified=True,
                  hrsid_physical_cross_scene_disjointness_verified=src["hrsid_physical_cross_scene_disjointness_verified"],
                  evaluator_annotation_ids_rekeyed=rekeyed, non_simple_polygons_retained_in_ap=sum(not p.is_valid for p in polygons.values()),
                  point_rend_comparator="deferred; reserve support parity not established", detector_forwards=0, optimizer_updates=0,
                  views=views, by_parent=per_parent, elapsed_seconds=perf_counter()-start)
    data = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()
    cmd.output.mkdir()
    (cmd.output/"RESULT_SAFE.json").write_bytes(data)
    print(json.dumps(dict(status=STATUS,result_sha256=sha256(data).hexdigest())),flush=True)


if __name__ == "__main__":
    main()
