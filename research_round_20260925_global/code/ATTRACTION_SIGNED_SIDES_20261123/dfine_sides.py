"""Recombine frozen signed D-FINE edits, then score their independently closed native masks."""
import argparse
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType

import numpy as np
from pycocotools import mask as codec


ARMS = ("raw", "SS", "LS", "SL", "LL", "DD")
NAMES = {"raw": "raw", "SS": "attraction_static_at_dynamic_count", "LL": "learned_at_attraction_count",
         "DD": "attraction_dynamic"}
PINS = {
    "dev705": dict(images=705, positive=37811, swaps=262802, history="HRSID_DEV705_CLASSICAL_ATTRACTION_ACTUAL_20261122_01",
        capture="HRSID_DFINE_E24_DEV705_PROBABILITY_CAPTURE_20261112_01/CONTROL/EXPORT",
        pred="80382f6f7318ac7a033e4d8169cf278322a9d51731f2d86e11b5f426d318d0ed",
        audit="29258878c41ec3d0cb2b173066d846a440197e2a0f45cd1277455dba0f786389",
        manifest="39a40d089d4a54448fe6efca8ea5cc98ad1eb7c0587815f3dd9a3b097173e78f",
        verified="58e6bc13ac39c3cd2ca1eeaf1e3f0b00157d08eee095cff00f504c0189d897d4",
        scored="85ee1ec662bca27b24b58aff3f5c7e5c6052a4ad9322807bb1e1443e6215a8ea"),
    "pseg232": dict(images=232, positive=11054, swaps=73770, history="PSEG_TEST232_CLASSICAL_ATTRACTION_ACTUAL_20261122_01",
        capture="DFINE_HRSID_E24_TO_PSEG_SSDD_TEST232_CAPTURE_ACTUAL_20261121_02",
        pred="e20db48eb090a053c0f6f28401e1f708700b1df280392e23a03f8e8887a82e05",
        audit="df7b11dcc7b019ebe3455e9be862c3287dc9ca4751aa0355287a1bff7e9b8daa",
        manifest="9c0b9bc07eade8b9de4349f187a80837f6a1c8449e596a2c1234f3827356b65e",
        verified="39ce2d6b0aefeef9a9447342e69d05a250c2cb61beafaaecb39298ad896b53aa",
        scored="25a9d2709a8b53ea24a53d8067a8f22777008b451eec16a41a718c99eca25901")}
BASE = "DFINE_SEG_EXTERNAL_BASELINE_FIT32_E24_20261112/e24/dev705_free_endpoint/"
SRC = {
    "cross": ("INDEPENDENT_SIDE_FACTORIAL_20261123/replay_sides.py", "aa590649fb2fa0a70d7fd239b26e9f865efac2d8c0dfe86c4525933f6720f214"),
    "base": (BASE+"score_endpoint.py", "77159ea493a95973fe422dac1d303eaeba603e6d8af7ae179c72aecc2e4301b0"),
    "polygon": ("MASK_RTDETR_DEV705_CONTINUOUS_AP_20261112/continuous_endpoint.py", "7cad8b51a9230101e2233db8455f84c28163abfb2ee8ddd8cbe8718a2e719bd0"),
    "matcher": ("HRSID_FIT_HULL_CDF_20261112/stock_coco_match.py", "12fd1684b262a3c9b6732b90a0c1e624034c523a7d130919a079e17728f80075"),
    "pseg": ("PSEG_TEST232_DFINE_FROZEN_SCORE_20261121/score_test232.py", "dd8e435976d2331146c8f83d70f368ee4e0954c05520c54fa9a3b1205838d40f"),
    "pgeom": ("PSEG_TEST232_PAIRED_GEOMETRY_20261121/paired_geometry.py", "107f31864bbb7e807a4194e552c7a1778d410f0e564f80437c5717fe6b47098d"),
    "adapter": ("DIFFSAR_NATIVE_CPU_COCO_ADAPTER_DRAFT_20261107_01/worker.py", "f9fc79403d12d8840c3269b04ba22d7584b8eb5e3c592ead2b6359ad7df65748")}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def read(path, expected):
    raw = path.read_bytes()
    need(sha256(raw).hexdigest() == expected, "Digest mismatch: " + path.name)
    return raw


def load(path, expected):
    m = ModuleType(path.stem)
    m.__file__ = str(path)
    exec(compile(read(path, expected), str(path), "exec"), m.__dict__)
    return m


def module(a, name):
    path, expected = SRC[name]
    return load(a.code / path, expected)


def encode(x):
    return (json.dumps(x, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def closed(root, phase, term_sha=None):
    path = root / ("observed_" + phase) / "TERMINAL.json"
    raw = read(path, term_sha) if term_sha else path.read_bytes()
    terminal = json.loads(raw)
    need(terminal["phase"] == phase and terminal["actual_exit"] == 0 and terminal["output_footer_verified"] is True,
         "Real closed worker with verified footer")
    result_sha = terminal["result_sha256"]
    return json.loads(read(root / phase / "RESULT_SAFE.json", result_sha)), result_sha, sha256(raw).hexdigest()


def historical(a, phase):
    p = PINS[a.cohort]
    return closed(a.root / p["history"], phase, p["verified" if phase == "verify-edits" else "scored"])[0]


def rows(a, edit_path, edit_sha):
    p = PINS[a.cohort]
    capture = a.root / p["capture"]
    manifest = json.loads(read(a.manifest, p["manifest"]))
    roster = manifest if a.cohort == "dev705" else manifest["items"]
    need(len(roster) == p["images"], "Exact image manifest")
    paths = [capture / "DEV_MASK_PREDICTIONS_PRIVATE.jsonl", capture / "DEV_CANDIDATE_AUDIT_PRIVATE.jsonl", edit_path]
    digests = [sha256() for _ in paths]
    from contextlib import ExitStack
    with ExitStack() as stack:
        streams = [stack.enter_context(path.open("rb")) for path in paths]
        for item in roster:
            raw = [f.readline() for f in streams]
            need(all(v.endswith(b"\n") for v in raw), "Complete three-stream image blocks")
            for digest, value in zip(digests, raw):
                digest.update(value)
            pred, audit, edit = [json.loads(value) for value in raw]
            identity = item["image_id"]
            need(identity == pred["image_id"] == audit["image_id"] == edit["image_id"] and
                 len(pred["coco"]) == len(audit["candidates"]) == audit["post_nms"] >= 100,
                 "Candidate and image provenance")
            natural, details = pred["coco"][:100], audit["candidates"][:100]
            need(all(x["image_id"] == identity and x["category_id"] == 1 and x["score"] == y["score"]
                     and y["native_rank"] == j for j,(x,y) in enumerate(zip(natural,details),1))
                 and all(natural[j]["score"] >= natural[j+1]["score"] for j in range(99)), "Stock ranks and scores")
            yield item, natural, details, edit
        need(not any(f.read(1) for f in streams) and [h.hexdigest() for h in digests] == [p["pred"],p["audit"],edit_sha],
             "All native identities and candidate stream bytes")


def replay(a):
    ref = historical(a, "verify-edits")
    need(ref["GT_opened"] is False and ref["optimizer_updates"] == ref["detector_forwards"] == 0,
         "Closed historical label-free controls")
    p = PINS[a.cohort]
    cross = module(a, "cross")
    counts, h = Counter(), sha256()
    original = a.root / p["history"] / "prepare-edits/EDITS_PRIVATE.jsonl"
    with (a.output / "EDITS_PRIVATE.jsonl").open("xb") as out:
        for i,(item, pred, audit, old) in enumerate(rows(a, original, ref["edits_sha256"]),1):
            changes = {}
            for key, action in old["changes"].items():
                rank = int(key)
                need(str(rank) == key and 1 <= rank <= 100, "Native rank")
                budgets = [action.get(NAMES[name], {}).get("swaps",0) for name in ("SS","LL","DD")]
                need(len(set(budgets)) == 1, "Equal archived realized per-side budgets")
                k = budgets[0]
                if not k:
                    continue
                area = audit[rank-1]["mask_area"]
                native = pred[rank-1]["segmentation"]
                arms, added, dropped = cross.cross_masks(native,action[NAMES["SS"]],action[NAMES["LL"]],area)
                # Recheck the two new hybrids with independent Boolean algebra on the original sign sets.
                s,l = (codec.decode(action[NAMES[name]]["rle"]).astype(bool) for name in ("SS","LL"))
                n = codec.decode(native).astype(bool)
                result = {}
                for dest, key2 in (("SS","FF"),("LS","IF"),("SL","FI"),("LL","II")):
                    got = arms[key2]["rle"]
                    expected = ((l if dest[0]=="L" else s) & ~n) | ((l if dest[1]=="L" else s) & n)
                    need(np.array_equal(codec.decode(got).astype(bool),expected), "Independent two-sign hybrid reproduction")
                    result[dest] = dict(rle=got, swaps=k)
                result["DD"] = action[NAMES["DD"]]
                d = result["DD"]["rle"]
                need(int(codec.area(d)) == area and int(codec.area(codec.merge([native,d],intersect=True))) == area-k,
                     "Dynamic context count")
                changes[key] = dict(arms=result, displaced_adds=int(added), displaced_drops=int(dropped))
                counts.update(positive=1, swaps=k, displaced_adds=int(added), displaced_drops=int(dropped))
            row = dict(image_id=item["image_id"], changes=changes)
            if a.cohort == "pseg232":
                row["rgb_sha256"] = old["rgb_sha256"]
            raw = encode(row)
            out.write(raw)
            h.update(raw)
            if i % 100 == 0: print(json.dumps(dict(replayed_images=i)),flush=True)
    need(counts["positive"] == p["positive"] and counts["swaps"] == p["swaps"], "Historical edit-count equality")
    return dict(status="DFINE_ATTRACTION_SIGNED_SIDES_REPLAY_COMPLETE", target_labels_opened=False,
                images=p["images"], counts=dict(counts), edits_sha256=h.hexdigest(), archived_edits_sha256=ref["edits_sha256"],
                historical_verify_terminal_sha256=p["verified"], manifest_sha256=p["manifest"],
                predictions_sha256=p["pred"], audit_sha256=p["audit"], signed_sets_and_old_arms_exact=True)


def add_measurements(records, matches, truth, neutral, base, modules):
    measured = {}
    for gt_id,(image_id,rank) in matches.items():
        im = records[image_id]
        area = im["audit"][rank-1]["mask_area"]
        if gt_id in neutral or gt_id not in truth or not 32 <= area < 1024:
            continue
        shape = truth[gt_id][0]
        native = im["native"][rank-1]["segmentation"]
        choice = im["edit"].get(str(rank), {})
        cache, values = {}, []
        for arm in ARMS:
            rle = choice.get("arms",{}).get(arm,{}).get("rle",native)
            key = rle["counts"]
            if key not in cache:
                iou, fp, fn = base.continuous(rle,area,shape,modules)
                cache[key] = (iou,100-fn,fp,fn)
            values.append(cache[key])
        v = np.asarray(values,dtype=np.float64)
        j = v[:,1] * shape.area / 100
        residual = abs(j[4] + j[1] - j[2] - j[3])
        need(residual <= 1e-7, "Four-way area decomposition")
        measured[gt_id] = (image_id, im["group"], v, choice.get("displaced_adds",0)>0,
                           choice.get("displaced_drops",0)>0,residual)
    return measured


def effects(data):
    grouped, exposure = defaultdict(list), [set(),set()]
    n_exposed = [0,0]
    for _,g,v,plus,minus,_ in data:
        grouped[g].append(v)
        for i,flag in enumerate((plus,minus)):
            if flag:
                exposure[i].add(g)
                n_exposed[i] += 1
    avg = np.asarray([np.mean(grouped[g],axis=0) for g in sorted(grouped)])
    draws = np.random.Generator(np.random.PCG64(5217)).integers(len(avg),size=(4096,len(avg)))
    def summary(v,quantiles):
        return dict(points=float(v.mean()), interval=np.percentile(v[draws].mean(axis=1),quantiles).tolist(),
                    groups_improved=int(sum(v>0)),groups_worse=int(sum(v<0)),groups_equal=int(sum(v==0)))
    side = {}
    plus = avg[:,2,1]-avg[:,1,1]
    minus = avg[:,3,1]-avg[:,1,1]
    for i,(name,delta) in enumerate((("additions",plus),("removals",minus))):
        side[name] = dict(**summary(delta,[1.25,98.75]), exposed_matches=n_exposed[i],
                          exposed_groups=len(exposure[i]),adequate=n_exposed[i]>=20 and len(exposure[i])>=8)
    dominance = summary(plus-minus,[2.5,97.5])
    enough = all(x["adequate"] for x in side.values())
    localized = ("additions" if enough and side["additions"]["interval"][1] < 0 and dominance["interval"][1] < 0
                 else "removals" if enough and side["removals"]["interval"][1] < 0 and dominance["interval"][0] > 0
                 else None)
    contrasts = {}
    for lhs,rhs in (("LS","SS"),("SL","SS"),("LL","SS"),("LS","LL"),("SL","LL"),("LL","DD")):
        contrasts[lhs+" minus "+rhs] = summary(avg[:,ARMS.index(lhs),0]-avg[:,ARMS.index(rhs),0],[2.5,97.5])
    return dict(eligible_matches=len(data),groups=len(avg),arm_mean_iou_points=dict(zip(ARMS,avg.mean(axis=0)[:,0].tolist())),
                bootstrap_draws=4096,bootstrap_seed=5217,signed_intersection_per_truth_points=side,
                addition_minus_removal=dominance,localized_dominant_deficit=localized,
                iou_contrasts_points=contrasts,max_interaction_residual_square_pixels=max(x[5] for x in data))


def geometry_parity(result, ref, key):
    for arm, name in NAMES.items():
        need(abs(result["arm_mean_iou_points"][arm]-ref[name][key]) <= 1e-9, "Published group IoU "+arm)


def collect(a,state,records,modules,base,reference):
    if a.cohort == "dev705":
        dev = load(a.root/"HRSID_DFINE_E24_DEV705_DRAFT_20261112_01/dfine_dev705.py",
                   "5be622cdec5c889ca8520aab5547f6f15f01a54568a0263423a816dca7f96626")
        gt = dev.dev_json(a.manifest.parent,"DEV_ORIGINAL_GT_PRIVATE.json")
        empty = dev.verify_labels(gt,dev.dev_json(a.manifest.parent,"DEV_EMPTY_RASTERS_PRIVATE.json"),state)
        truth, neutral, _ = modules.geometry.polygon_support(gt,modules.vector)
        adapter = module(a,"adapter")
        ap = {"primary":{},"sensitivity":{}}
        for arm in ARMS:
            flat = [row for im in gt["images"] for row in materialize(records[im["id"]],arm)]
            measured = dev.score_variants(gt,flat,empty,adapter.evaluate_population)
            for key,value in measured.items():
                if arm in NAMES:
                    need(value == reference["coco_fraction"][key]["arms"][NAMES[arm]], "Published COCO values "+arm)
                ap[key][arm] = value
            print(json.dumps(dict(evaluated=arm)),flush=True)
        matcher = module(a,"matcher")
        anns = defaultdict(list)
        matches = {}
        for ann in gt["annotations"]:
            anns[ann["image_id"]].append(ann)
        for image in gt["images"]:
            iid = image["id"]
            rec = records[iid]
            pool = [dict(rank=i, score=row["score"], rle=row["segmentation"],area=k["mask_area"])
                    for i,(row,k) in enumerate(zip(rec["native"],rec["audit"]),1)]
            with redirect_stdout(StringIO()):
                fixed,_,_ = matcher.freeze_coco(image,anns[iid],pool)
            matches.update({g:(iid,r) for g,r in fixed.items()})
        raw = reference["geometry_points"]
        need(len(matches)==raw["counts"]["stock_matches"], "Historical native matches")
        measured = add_measurements(records,matches,truth,neutral,base,modules)
        geometry = effects(list(measured.values()))
        need((geometry["eligible_matches"],geometry["groups"]) == (1447,27), "Complete DEV geometry population")
        geometry_parity(geometry,raw["metrics"],"group_equal_iou_points")
        return ap, {"primary":geometry}

    scorer, evaluator = module(a,"pseg"),module(a,"pgeom")
    gt = json.loads(read(a.gt,scorer.GT_SHA))
    truth,neutral,_ = modules.geometry.polygon_support(gt,modules.vector)
    need(not neutral and len(truth)==546,"Full author polygon census")
    by_name = {v["name"]:v for v in records.values()}
    records = {im["id"]:by_name[im["file_name"]] for im in gt["images"]}
    official = sorted(records)
    views = {"official":official,"exclude_exact_fit_overlap":
             sorted(i for i in records if records[i]["name"] not in scorer.EXCLUDED)}
    need(list(map(len,views.values()))==[232,222],"Original PSeg populations")
    result,geometries,measured = {},{},None
    for view,ids in views.items():
        wanted = set(ids)
        subset = dict(gt,images=[im for im in gt["images"] if im["id"] in wanted],
                      annotations=[ann for ann in gt["annotations"] if ann["image_id"] in wanted])
        ap = {}
        for arm in ARMS:
            flat = [dict(row,image_id=im["id"],category_id=0) for im in subset["images"]
                    for row in materialize(records[im["id"]],arm)]
            ap[arm] = scorer.evaluate(subset,flat)
            if arm in NAMES:
                need(ap[arm] == reference["views"][view]["metrics_fraction"][NAMES[arm]], "Original PSeg COCO "+arm)
            print(json.dumps(dict(view=view,evaluated=arm)),flush=True)
        matches,_ = evaluator.evaluate_raw(gt,records,ids,ap["raw"])
        if measured is None:
            measured = add_measurements(records,matches,truth,neutral,base,modules)
            full_matches = matches
        else:
            need(matches == {g:r for g,r in full_matches.items() if r[0] in wanted}, "PSeg .50 frozen match identity")
        geom = effects([v for v in measured.values() if v[0] in wanted])
        need((geom["eligible_matches"],geom["groups"]) == ((293,143) if view=="official" else (279,138)),
             "Published PSeg fixed geometry cohort")
        geometry_parity(geom,reference["geometry"][view]["arms"],"continuous_iou_points")
        result[view] = ap
        geometries[view] = geom
    return result,geometries


def materialize(record, arm):
    for rank,row in enumerate(record["native"],1):
        entry = record["edit"].get(str(rank),{}).get("arms",{}).get(arm)
        yield dict(row,segmentation=entry["rle"]) if entry else dict(row)


def score(a):
    p = PINS[a.cohort]
    accepted,prepared_sha,terminal_sha = closed(a.phase_root,"prepare-edits")
    need(accepted["status"]=="DFINE_ATTRACTION_SIGNED_SIDES_REPLAY_COMPLETE" and
         accepted["protocol_sha256"] == a.protocol_sha and accepted["cohort"]==a.cohort and
         accepted["source_sha256"] == sha256(Path(__file__).read_bytes()).hexdigest() and
         accepted["target_labels_opened"] is False and accepted["signed_sets_and_old_arms_exact"] is True,
         "Independent observed label-free worker required")
    base = module(a,"base")
    state,records = [],{}
    for item, pred, detail, edit in rows(a,a.phase_root/"prepare-edits/EDITS_PRIVATE.jsonl",accepted["edits_sha256"]):
        iid = item["image_id"]
        if a.cohort == "dev705":
            name,group = item["file_name"],base.as_group(item)
        else:
            name,group = item["image_ref"]["path"],edit["rgb_sha256"]
        need(iid not in records and isinstance(group,str) and group,"Complete unique native cohort")
        state.append(item)
        records[iid] = dict(native=pred,audit=detail,edit=edit["changes"],name=name,group=group)
    need(len(records)==p["images"],"Complete native cohort before opening evaluation labels")
    sys.path.append(str(a.shapely_site.resolve(strict=True)))
    import shapely
    need(shapely.__version__=="2.0.2" and shapely.geos_version_string=="3.11.2","Original continuous geometry runtime")
    modules = module(a,"polygon").geometry_modules(a.code)
    ref = historical(a,"score")
    coco,geom = collect(a,state,records,modules,base,ref)
    return dict(status="DFINE_ATTRACTION_SIGNED_SIDES_SCORE_COMPLETE",target_labels_opened=True,
                images=p["images"], swaps_per_arm=p["swaps"], positive_budget_candidates=p["positive"],
                historical_score_terminal_sha256=p["scored"],replay_result_sha256=prepared_sha,
                replay_terminal_sha256=terminal_sha, edits_sha256=accepted["edits_sha256"],
                archived_pure_arms_coco_exact=True, archived_pure_arms_iou_within_1e_minus_9=True,
                coco_fraction=coco,geometry=geom)


def main():
    p = argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument("phase",choices=("prepare-edits","score"))
    p.add_argument("--cohort",choices=tuple(PINS),required=True)
    for name in ("root","code","protocol","manifest","phase-root","output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--shapely-site",type=Path)
    p.add_argument("--gt",type=Path)
    a = p.parse_args()
    a.protocol_sha = "c0053d7cdf70372b1a2b8e41093a2cac3800982e7f05b3c3772618fabe7fd15a"
    read(a.protocol,a.protocol_sha)
    need(a.output == a.phase_root/a.phase and not a.output.exists(),"Fresh observed phase destination")
    need((a.phase=="score")==bool(a.shapely_site) and (a.phase=="score" and a.cohort=="pseg232")==bool(a.gt),
         "Supply label/geometry inputs only to scoring")
    a.output.mkdir()
    started = perf_counter()
    result = replay(a) if a.phase=="prepare-edits" else score(a)
    result.update(protocol_sha256=a.protocol_sha,cohort=a.cohort,detector_forwards=0,head_forwards=0,
                  optimizer_updates=0,source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                  seconds=perf_counter()-started)
    raw = (json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    (a.output/"RESULT_SAFE.json").write_bytes(raw)
    print(json.dumps(dict(status=result["status"],result_sha256=sha256(raw).hexdigest())),flush=True)


if __name__=="__main__":
    main()
