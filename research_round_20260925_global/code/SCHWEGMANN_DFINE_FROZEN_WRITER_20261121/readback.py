"""Independent Schwegmann center-score remeasurement from the author record stream and original COCO masks."""
from collections import Counter, defaultdict
from hashlib import sha256
from io import BytesIO
import json
import os
from math import fsum, isclose
from pathlib import Path
import re
from zipfile import ZipFile

import numpy as np
from PIL import Image
from pycocotools import mask as coco


OUT = Path(__file__).resolve().parent
TMP = Path(os.environ["SCHWEGMANN_DIAGNOSTIC_WORK_ROOT"]).resolve(strict=True)
PRIVATE = Path(os.environ["SAR_SEAL_RUNTIME_ROOT"]).resolve(strict=True)
ARCH = PRIVATE / "P254_SCHWEGMANN_EXTERNAL_EVIDENCE_PRIVATE_20261110_01"
INTAKE = TMP / "SCHWEGMANN_FROZEN_WRITER_EVAL_IMAGESET_20261121_01"
OWNER = TMP / "SCHWEGMANN_DFINE_FROZEN_WRITER_20261121_01"
CAP = PRIVATE / "SCHWEGMANN_DFINE_E24_FROZEN_CAPTURE_ACTUAL_20261121_01"
EDIT = PRIVATE / "SCHWEGMANN_DFINE_E24_FROZEN_EDITS_ACTUAL_20261121_01"
SCORE = "42e7703035ac7f8c6c4701a60e0366e31a7d3337e45555759253c384d78f1185"
SOURCE = "78ff85f49eb5102716ae6d39bdbafaf27fc9c63a630be0d86a2c434ab09f053b"
IMAGES = "2485f984db978572e5d446847444f63a8055d868aeebbdede2fe018e3997fc46"
ARMS = ("raw","fixed","free_matched_count","free_natural","native_matched_count")
METRICS = ("iou","dice","recovery_050","recovery_075")
PAIRS = (("free_matched_count","fixed"),("free_natural","raw"),("free_natural","native_matched_count"))
STRUCT = re.compile(rb'"(?:[^"\\]|\\.)*"|[{}]')
FIELDS = re.compile(rb'"(sensor|fullname)"\s*:\s*("(?:[^"\\]|\\.)*")')


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def h(data):
    return sha256(data).hexdigest()


def read(path, digest):
    buf = path.read_bytes()
    require(h(buf) == digest, "Digest mismatch for " + path.name)
    return buf


def open_records(raw):
    depth = 0
    end = 0
    for token in STRUCT.finditer(raw):
        c = raw[token.start()]
        if c == 123:
            if depth == 0:
                delimiter = raw[end:token.start()].strip()
                require((end == 0 and re.fullmatch(rb"\[\s*\[",delimiter)) or delimiter == b",","Record boundary")
                start = token.start()
            depth += 1
        elif c == 125:
            depth -= 1
            require(depth >= 0,"Record nesting")
            if depth == 0:
                end = token.end()
                yield raw[start:end]
    require(depth == 0 and re.fullmatch(rb"\s*\]\s*\]\s*",raw[end:]), "Original nested-array terminator")


def acquisition(row):
    p = row["fullname"].split("_")
    if row["sensor"] == "SEN1":
        require(len(p) == 9 and p[0] == "S1A", "Sentinel original product")
        return "S1A",p[4][:8],p[6],p[7]
    require(row["sensor"] == "RSAT" and p[0] == "RS2" and len(p)==9,"RADARSAT original product")
    return "RS2",p[5],p[6],p[1],p[2],p[3]


def references(items):
    prior = read(ARCH/"COHORT_03/IMAGE_ONLY/IMAGES_ONLY_PRIVATE.zip",
                 "428d696ea4a59a417dccdf26bfcc05f55f9ab20de69045e9f1f5b62c43ab11d7")
    with ZipFile(BytesIO(prior)) as z:
        historical = {h(Image.open(BytesIO(z.read(n))).convert("RGB").tobytes()) for n in z.namelist()}
    require(len(historical) == 32,"32 historical image exclusions")
    objs = {x["image_id"]:x for x in items}
    truth, groupkeys, seen = {}, [], set()
    raw = read(ARCH/"SHIP_POSITIVES_FROZEN_PRIVATE.json", SOURCE)
    for ordinal, b in enumerate(open_records(raw), 1):
        if ordinal in objs:
            item = objs[ordinal]
            row = json.loads(b)
            require(row["polarisation"] in ("HH","VV"),"Original co-polar image")
            power = np.asarray(row["patchfu"],dtype=np.float64)
            require(power.shape==(101,101) and np.isfinite(power).all() and power.min()>=0,"Original eligible intensity")
            decibel = np.clip(10.0 * np.log10(np.maximum(power, .0001)), -40.0, 10.0)
            gray = np.rint((decibel + 40.0) * (255.0 / 50.0)).astype("u1")
            rgb = np.stack((gray, gray, gray), axis=-1).tobytes()
            fingerprint = h(rgb)
            require(fingerprint == item["rgb_sha256"] and fingerprint not in historical and fingerprint not in seen,
                    "Unique unchanged decoded source image held out from the 32")
            seen.add(fingerprint)
            img = item["image_ref"]
            png = read(INTAKE/"IMAGE_ONLY"/img["path"],img["sha256"])
            with Image.open(BytesIO(png)) as im:
                require(im.mode=="RGB" and im.size==(101,101) and im.tobytes()==rgb and len(png)==img["bytes"],
                        "Image-only PNG matches native source pixels")
            label = np.asarray(row["patchgt"])
            require(label.shape == (101,101) and np.logical_or(label==0,label==1).all() and label.any(),
                    "Positive original binary author scatter mask")
            truth[ordinal] = label.astype(np.bool_)
        else:
            row = {key.decode():json.loads(value) for key,value in FIELDS.findall(b)}
        groupkeys.append(acquisition(row))
    groups = {k: f"acq_{n:03d}" for n,k in enumerate(sorted(set(groupkeys)),1)}
    ledger = h("".join(f"{n}\t{groups[g]}\n" for n,g in enumerate(groupkeys,1)).encode("ascii"))
    require(len(groupkeys)==1596 and len(groups)==16 and len(truth)==750 and
            all(groups[groupkeys[i-1]]==x["acquisition_group"] for i,x in objs.items()), "Original groups and cohort")
    return truth, ledger


def compare(expected, actual, path=""):
    if isinstance(expected,dict):
        require(set(expected)==set(actual), "Result fields at "+path)
        for key in expected:
            compare(expected[key], actual[key], path+"/"+key)
    elif isinstance(expected,list):
        require(len(expected)==len(actual),"Result interval size")
        for i,(a,b) in enumerate(zip(expected,actual)):
            compare(a,b,path+f"/{i}")
    elif type(expected) is float:
        require(isclose(expected, actual, abs_tol=2e-15,rel_tol=0),"Numerical mismatch at "+path)
    else:
        require(expected==actual,"Mismatch at "+path)


def observed(path, digest, status, receipt):
    doc = json.loads(read(path,digest))
    require(doc["status"]==status and doc["actual_process_terminal"] is True and
            type(doc["actual_exit"]) is int and doc["actual_exit"]==0 and doc["result_sha256"]==receipt and
            (type(doc.get("tool_session_handle")) is int and doc["tool_session_handle"]>0 or
             bool(re.fullmatch(r"[0-9a-f]+", doc.get("tool_chunk_id",""))) and
             doc.get("observation")=="direct_tool_exit_with_footer"), "Observed successful process closure")


def score_pair(mask, target):
    intersection = int(np.count_nonzero(np.logical_and(mask,target)))
    union = int(np.count_nonzero(np.logical_or(mask,target)))
    denom = int(np.count_nonzero(mask)) + int(np.count_nonzero(target))
    return (intersection/union, 2*intersection/denom, int(2*intersection>=union), int(4*intersection>=3*union))


def main():
    original = json.loads(read(TMP/"SCHWEGMANN_DFINE_FROZEN_WRITER_SCORE_20261121_02/RESULT_SAFE.json", SCORE))
    require(original["source_json_sha256"]==SOURCE and original["manifest_sha256"]==IMAGES, "Pinned author and cohort")
    for name, digest, status, receipt in (
        ("CAPTURE_TERMINAL_PRIVATE.json","capture_terminal_sha256","SCHWEGMANN_DFINE_FROZEN_CAPTURE_TERMINAL","capture_result_sha256"),
        ("WRITER_TERMINAL_PRIVATE.json","edit_terminal_sha256","SCHWEGMANN_DFINE_FROZEN_EDITS_TERMINAL","edit_receipt_sha256")):
        observed(OWNER/name,original[digest],status,original[receipt])
    cap = json.loads(read(CAP/"RESULT_SAFE.json",original["capture_result_sha256"]))
    edit = json.loads(read(EDIT/"RESULT_SAFE.json",original["edit_receipt_sha256"]))
    m = json.loads(read(INTAKE/"IMAGE_MANIFEST_PRIVATE.json",IMAGES))
    items=m["items"]
    require(len(items)==750 and [x["record_ordinal"] for x in items] == [x["image_id"] for x in items]
            == sorted(set(m["expected_ids"]))==m["expected_ids"],"Original positive unique ordinal roster")
    require(cap["manifest_sha256"]==edit["manifest_sha256"]==IMAGES and
            cap["native_rows_sha256"]==original["capture_rows_sha256"] and
            edit["capture_result_sha256"]==original["capture_result_sha256"] and
            edit["edits_sha256"]==original["predictions_sha256"] and
            cap["labels_opened"] is edit["labels_opened"] is False and
            cap["model_forwards"]==750 and edit["model_forwards"]==0, "Original prelabel capture/edit lineage")
    gpu = read(CAP/"CENTER_CANDIDATES_PRIVATE.jsonl",cap["native_rows_sha256"]).splitlines()
    arms_raw = read(EDIT/"ARMS_PRIVATE.jsonl",edit["edits_sha256"])
    require(len(arms_raw)==edit["edits_bytes"] and len(gpu)==750,"Original closed readout populations")
    arms = arms_raw.splitlines()
    reference, ledger = references(items)
    require(ledger==m["group_index_sha256"]==original["group_index_sha256"],"Acquisition group index digest")
    by_group = defaultdict(lambda: defaultdict(list))
    counts = {k: Counter({j:0 for j in "added_pixels changed_images missing recovered_050 recovered_075 removed_pixels selected selected_empty selected_nonempty xor_pixels".split()}) for k in ARMS}
    areas = Counter()
    top_count = 0
    identical_rows = 0
    absent_planes = 0
    for item, line, edits in zip(items,gpu,arms,strict=True):
        row, edited = json.loads(line),json.loads(edits)
        ident, group = item["image_id"],item["acquisition_group"]
        require(edited["image_id"]==row["image_id"]==ident and edited["group"]==row["group"]==group,"Bound image/group")
        candidate_rows = row["candidates"]
        require(len(candidate_rows)<=100 and all(c["native_rank"]==i for i,c in enumerate(candidate_rows,1)),
                "Original top-100 roster")
        top_count += len(candidate_rows)
        rank = next((i for i,c in enumerate(candidate_rows,1) if
            all(80 <= lo + hi < 122 for lo,hi in zip(c["box_xyxy"][:2],c["box_xyxy"][2:]))),None)
        require(rank==row["rank"]==edited["rank"] and set(edited["arms"])==set(ARMS), "Original box-center selection")
        if rank is None:
            require(row["prediction"] is None and edited["score"] is None and all(x is None for x in edited["arms"].values()),
                    "Original missing candidate")
            raw = np.zeros((101,101),dtype=bool)
        else:
            selected = row["prediction"]
            detail = candidate_rows[rank-1]
            require(selected["segmentation"]==edited["arms"]["raw"] and selected["score"]==edited["score"]==detail["score"],
                    "Original selected RLE and score")
            raw = coco.decode(selected["segmentation"]).astype(np.bool_)
            require(raw.shape==(101,101) and int(raw.sum())==detail["mask_area"],"Decoded original area")
            area = int(raw.sum())
            areas["empty" if area==0 else "one_to_31" if area<32 else "32_to_1023" if area<1024 else "1024_plus"] += 1
        if row["plane_offset"] is None and row["plane_sha256"] is None:
            absent_planes += 1
        all_identical = True
        for arm in ARMS:
            if rank is None:
                mask = raw
                counts[arm]["missing"] += 1
            else:
                mask = coco.decode(edited["arms"][arm]).astype(np.bool_)
                counts[arm]["selected"] += 1
                counts[arm]["selected_nonempty" if mask.any() else "selected_empty"] += 1
            up = np.count_nonzero(mask & ~raw)
            down = np.count_nonzero(raw & ~mask)
            require(up==down and (arm=="raw" or int(up)==edited["swaps"][arm]),"Observed equal-area edits")
            counts[arm].update(added_pixels=int(up),removed_pixels=int(down),changed_images=int(up>0),xor_pixels=int(up+down))
            all_identical = all_identical and np.array_equal(mask,raw)
            values=score_pair(mask,reference[ident])
            by_group[group][arm].append(values)
            counts[arm].update(recovered_050=values[2], recovered_075=values[3])
        identical_rows += int(all_identical)
    groups=sorted(by_group)
    image_groups={g:len(by_group[g]["raw"]) for g in groups}
    average=lambda values:fsum(values)/len(values)
    means={arm:{g:[average(v) for v in zip(*by_group[g][arm])] for g in groups} for arm in ARMS}
    arms_summary={arm:dict(counts=dict(counts[arm]),acquisition_equal=dict(zip(METRICS,
        (average(v) for v in zip(*(means[arm][g] for g in groups)))))) for arm in ARMS}
    compare(original["arms"],arms_summary)
    compare(original["group_images"],image_groups)
    require(original["images"]==sum(image_groups.values())==750 and original["acquisitions"]==len(groups)==16,
            "Complete grouped population")
    draw=np.random.Generator(np.random.PCG64(254101)).integers(0,len(groups),size=(4096,len(groups)))
    pairs={}
    for a,b in PAIRS:
        effects={g:[left-right for left,right in zip(means[a][g],means[b][g])] for g in groups}
        metrics=list(zip(*(effects[g] for g in groups)))
        bounds=[np.percentile(np.take(values,draw).mean(axis=1),[2.5,97.5]).tolist() for values in metrics]
        pairs[a+"_minus_"+b]=dict(acquisition_equal_delta=dict(zip(METRICS,map(average,metrics))),
            acquisition_cluster_percentile95=dict(zip(METRICS,bounds)),
            per_acquisition={g:dict(zip(METRICS,effects[g])) for g in groups})
    compare(original["paired"],pairs)
    plane = read(CAP/"SELECTED_PLANES_PRIVATE.bin",cap["selected_planes_sha256"])
    require(absent_planes==750 and len(plane)==0 and areas["32_to_1023"]==0,"Entire panel outside support")
    census=dict(eligible=areas["32_to_1023"],images=750,missing=counts["raw"]["missing"],nonempty=counts["raw"]["selected_nonempty"],
        point_pairs=0,retained_top100_rows=top_count,selected=counts["raw"]["selected"],supported=0,zero_support=0)
    compare(original["candidate_counts"],census)
    require(identical_rows==750,"All five arms yield identical original-grid bits on every chip")
    result=dict(status="SCHWEGMANN_DFINE_FROZEN_CENTER_SCORE_INDEPENDENTLY_ACCEPTED",original_score_sha256=SCORE,
        readback_source_sha256=h(Path(__file__).read_bytes()),source_json_sha256=SOURCE,manifest_sha256=IMAGES,
        group_index_sha256=ledger,images=750,acquisitions=16,prior_image_overlap=0,group_images=image_groups,
        confirmed_candidate_counts=census,selected_nominal_area_bands={k:areas[k] for k in ("empty","one_to_31","32_to_1023","1024_plus")},
        identical_mask_rows=identical_rows,raw=arms_summary["raw"],paired_zero_mean_and_zero_percentile95=True,
        confirmed_contrasts=list(pairs),paired_acquisition_draws=4096,seed=254101,
        original_author_reference_bits_and_pngs_checked=750,model_forwards=0,instance_AP=False)
    raw=(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    with (OUT/"RESULT_SAFE.json").open("xb") as file:
        file.write(raw)
    print(json.dumps(dict(status=result["status"],result_sha256=h(raw),bands=result["selected_nominal_area_bands"],
        raw=arms_summary["raw"]["acquisition_equal"])))


if __name__=="__main__":
    main()
