"""Derive a training-only HRSID mask view with five explicitly excluded zero-area polygons."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile


def need(ok,message):
    if not ok:
        raise ValueError(message)


def training_view(source, image_ids, excluded):
    images=[r for r in source["images"] if r["id"] in image_ids]
    selected=[r for r in source["annotations"] if r["image_id"] in image_ids]
    need(len(images)==len(image_ids) and {r["id"] for r in images}==image_ids
         and len({r["id"] for r in selected})==len(selected)
         and excluded <= {r["id"] for r in selected},"Exact FIT records and FIT-only exclusions")
    annotations=[r for r in selected if r["id"] not in excluded]
    need({r["image_id"] for r in annotations}==image_ids,"Every original FIT image retains mask supervision")
    return {**source,"images":images,"annotations":annotations}


def pinned_json(path,digest):
    b=path.read_bytes()
    need(sha256(b).hexdigest()==digest,"Exact closed input "+path.name)
    return json.loads(b)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("archive","assignment","census","findings","output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--protocol",required=True)
    args=p.parse_args()
    need(len(args.protocol)==40 and all(c in "0123456789abcdef" for c in args.protocol),"Exact prospective protocol")
    census=pinned_json(args.census,"d0e4deff1be184c3037990aed0e0df208f1503b20c079db2539bbd63a29e120d")
    need(census["decoded_fit_images"]==2937 and census["attempted_fit_rasters"]==9265
         and census["empty_fit_rasters"]==5 and census["nonempty_fit_rasters"]==9260
         and census["image_failures"]==census["fit_raster_check_failures_excluding_empty"]==0,
         "Closed complete FIT census")
    roster=pinned_json(args.assignment,"c92b85dc278bc49cc8e34634fd7d354018844ae2a2692ea451b014a082bbaa5a")
    findings=pinned_json(args.findings,"4ecc36d10a329ed93df280734e8867b0b330bb240b7ed084de86f96dc950e64f")
    ids={r["image_id"] for r in roster if r["split"]=="fit"}
    removed={r["annotation_id"]:r["image_id"] for r in findings}
    need(len(ids)==2937 and len(findings)==len(removed)==5
         and all(r["kind"]=="EMPTY_COCO_RASTER" and r["image_id"] in ids for r in findings),"Exact known FIT empties")
    need(args.archive.stat().st_size==5698302315,"Exact archive size")
    with ZipFile(args.archive) as z:
        b=z.read("annotations/train2017.json")
    source_sha=sha256(b).hexdigest()
    need(source_sha=="3e240c38014526e9b93206a946105fb41c808272238932b0d4c2422978d5116f",
         "Original official TRAIN only")
    source=json.loads(b)
    need({r["id"]:r["image_id"] for r in source["annotations"] if r["id"] in removed}==removed,"Exact source exclusion mapping")
    derived=training_view(source,ids,set(removed))
    counts=Counter(r["image_id"] for r in derived["annotations"])
    need(len(derived["images"])==2937 and len(derived["annotations"])==9260
         and max(counts.values())==163 and sum(max(n-100,0) for n in counts.values())==113,
         "Exact derived population and independently observed query bound")
    args.output.mkdir()
    output=(json.dumps(derived,ensure_ascii=False,separators=(",",":"),allow_nan=False)+"\n").encode()
    with (args.output/"FIT_MASK_TRAINING_VIEW_PRIVATE.json").open("xb") as f:
        f.write(output)
    result=dict(status="ACTUAL_HRSID_PNG_FIT_MASK_TRAINING_VIEW_DERIVED",
        protocol=args.protocol,source_annotation_sha256=source_sha,images=2937,
        source_instances=9265,excluded_degenerate_polygons=5,derived_instances=9260,
        derived_bytes=len(output),derived_sha256=sha256(output).hexdigest(),
        retained_rows_unmodified=True,original_annotation_files_unmodified=True,
        images_opened=False,official_TEST_member_opened=False,model_executed=False,AP=False,
        capacity_100_minimum_unassigned_per_matching_group=113)
    with (args.output/"TRAIN_VIEW_SAFE.json").open("x",encoding="utf-8",newline="\n") as f:
        json.dump(result,f,indent=2,sort_keys=True,allow_nan=False)
        f.write("\n")
    print(json.dumps(result,sort_keys=True))


if __name__=="__main__":
    main()

