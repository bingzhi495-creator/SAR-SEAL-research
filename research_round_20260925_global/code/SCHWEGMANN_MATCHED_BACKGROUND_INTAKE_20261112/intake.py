"""Bounded acquisition-group matched author background intake."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha1, sha256
import importlib.util
from io import BytesIO
import json
import math
import os
from pathlib import Path
import re
from time import sleep, perf_counter
from urllib.request import Request, urlopen
from zipfile import ZipFile

import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask


COMMIT = "9decec2a18b9d7d6fd2d057a435808ca67d2bad1"
BASE = "https://gitlab.com/api/v4/projects/sarmaps%2Fsar-ship-dataset/repository/files/json%2F"
SOURCES = {
    "ocean": ("true_negatives", 69729582, "1bf57cca8779bf7d32ea08b507d2bc04eb23341e",
              "3abd70a17cbccd2ae9011545f4a727c055d8eb6030ffb6205ab0014437c76c53", "NEG"),
    "lookalike": ("false_positives", 275346383, "c49b762ea59c0bd55dca9bdfeb087384b75a8fc5",
                  "69ef61e899c0634c4d032d38a398e042888a957897913ed37d27baf1472f3715", "FP"),
}
WINDOW = 256 * 1024
DECODER = json.JSONDecoder()
OBJECT_START = re.compile(r'\{\s*"(?:id|sensor)"\s*:')
LABEL = b"schwegmann-source-matched-background-20261112\0"


def require(ok, reason):
    if not ok:
        raise RuntimeError(reason)


def stable_sha(path, blob=None):
    h = sha256()
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        if blob is not None:
            blob.update(f"blob {before.st_size}\0".encode())
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(data)
            if blob is not None:
                blob.update(data)
        after = os.fstat(stream.fileno())
        require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), "Source changed")
    return h.hexdigest()


def save(path, value):
    raw = (json.dumps(value, sort_keys=True, allow_nan=False, indent=2) + "\n").encode("ascii")
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return sha256(raw).hexdigest()


def load_positive_api(path):
    require(stable_sha(path) == "41704d3d3980b6ae6442d17b92625e37ed782e417e06a585806227d881ce7d79",
            "Original closed P254 source required")
    spec = importlib.util.spec_from_file_location("p254_source", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def iter_rows(path):
    with path.open(encoding="ascii") as stream:
        buffer = stream.read(1024 * 1024)
        initial = re.match(r"\s*\[\s*\[\s*", buffer)
        require(initial is not None, "Expected the original double-list JSON")
        buffer = buffer[initial.end():]
        while True:
            buffer = buffer.lstrip()
            while not buffer or buffer[0] == ",":
                if buffer:
                    buffer = buffer[1:].lstrip()
                else:
                    buffer = stream.read(1024 * 1024)
                    require(buffer, "Unexpected positive EOF")
            if buffer[0] == "]":
                require(re.fullmatch(r"\]\s*\]\s*", buffer + stream.read(1024)) is not None, "Expected original JSON end")
                return
            try:
                row, consumed = DECODER.raw_decode(buffer)
            except json.JSONDecodeError:
                extra = stream.read(1024 * 1024)
                require(extra and len(buffer) <= 2 * 1024 * 1024, "Could not finish source JSON row")
                buffer += extra
                continue
            require(isinstance(row, dict), "Expected an original author row")
            yield row
            buffer = buffer[consumed:]


def point(record):
    return dict(row=record["row"], column=record["column"],
                latitude=record["latitude"], longitude=record["longitude"])


def lock_positive(args):
    api = load_positive_api(args.p254_source)
    blob = sha1()
    require(args.positive_archive.stat().st_size == api.SOURCE_BYTES
            and stable_sha(args.positive_archive, blob) == api.SOURCE_SHA256 and blob.hexdigest() == api.SOURCE_BLOB,
            "Original positive file identity required")
    old = args.cohort
    require(stable_sha(old/"QUALIFICATION_SAFE.json") == "b2f557037236f28221f76e169ecd66da61cfd52bc158bae93748605c1b5d19f2",
            "Closed P254 cohort receipt required")
    report = json.loads((old/"QUALIFICATION_SAFE.json").read_bytes())
    old_zip = old/"IMAGE_ONLY/IMAGES_ONLY_PRIVATE.zip"
    old_mask = old/"SCORER_ONLY/PATCHGT_PRIVATE.json"
    require(stable_sha(old_zip) == report["image_zip_sha256"] and stable_sha(old_mask) == report["ground_truth_sha256"],
            "Closed P254 image and mask exact readback required")
    all_points, product_map, record_map = defaultdict(list), {}, {}
    def records():
        for row in iter_rows(args.positive_archive):
            key = api.acquisition(row)
            name = row["fullname"]
            require(product_map.setdefault(name, key) == key, "Physical group consistency")
            position = point(row)
            all_points[name].append(position)
            record_map[row["id"]] = dict(product=name, **position)
            yield row
    class FixedPositiveRows:
        def __iter__(self):
            return records()

        def __len__(self):
            return 1596

    selected, counts = api.fixed_rows(FixedPositiveRows())
    require(counts["author_records"] == len(record_map) == 1596 and len(selected) == 32, "Exact old cohort expected")
    group_map = {key: f"acq_{i:03}" for i, key in enumerate(sorted(set(product_map.values())), 1)}
    gt = json.loads(old_mask.read_bytes())["images"]
    frozen = []
    with ZipFile(old_zip) as bundle:
        for i, (group, row) in enumerate(selected, 1):
            pos = record_map[row["ident"]]
            previous = gt[i-1]
            rle = previous["segmentation"]
            with Image.open(BytesIO(bundle.read(f"chip_{i:04}.png"))) as image:
                require(np.array_equal(np.asarray(image), np.repeat(row["gray"][:,:,None],3,axis=2)), "Exact old positive pixel identity")
            reference = coco_mask.decode(dict(rle,counts=rle["counts"].encode()))
            require(group == previous["acquisition"] == group_map[row["key"]]
                    and np.array_equal(reference, row["mask"]), "Exact original positive mask and group")
            cropped = row["gray"][25:76,25:76]
            cropped_mask = row["mask"][25:76,25:76]
            frozen.append(dict(group=group,pol=row["polarisation"],input_sha=sha256(cropped.tobytes()).hexdigest(),
                               mask_sha=sha256(cropped_mask.tobytes()).hexdigest(),
                               crop_eligible=bool(cropped_mask.sum()==row["mask"].sum()),**pos))
    result = dict(status="P254_MATCHED_BACKGROUND_POSITIVE_LOCK",p254_image_zip_sha256=report["image_zip_sha256"],
                  p254_mask_sha256=report["ground_truth_sha256"],positives=frozen,
                  products={p:group_map[k] for p,k in product_map.items()},positive_points=all_points)
    digest = save(args.output,result)
    print(json.dumps(dict(status=result["status"],positive_lock_sha256=digest,
                          groups=len(group_map),products=len(product_map),positive_crops=len(frozen),
                          positive_crops_eligible=sum(row["crop_eligible"] for row in frozen))),flush=True)


def fragments(raw):
    text = raw.decode("ascii")
    for match in OBJECT_START.finditer(text):
        try:
            row, end = DECODER.raw_decode(text,match.start())
        except json.JSONDecodeError:
            continue
        if end < len(text) and text[end] not in ",]\r\n\t ":
            continue
        if isinstance(row,dict) and {"sensor","fullname","polarisation","row","column","gttype","patchfu"} <= row.keys():
            yield row


def starts(size, phase):
    a = [i*(size-WINDOW)//31 for i in range(32)]
    b = [(x+y)//2 for x,y in zip(a,a[1:])]
    require(all(y-x >= WINDOW for x,y in zip(sorted(a+b),sorted(a+b)[1:])), "Frozen windows must be disjoint")
    return a if phase==1 else b


def read_range(kind,start,out):
    name,total,blob,file_sha,_ = SOURCES[kind]
    end = start+WINDOW-1
    url = f"{BASE}{name}.json/raw?ref={COMMIT}"
    headers = {"Range":f"bytes={start}-{end}","Accept-Encoding":"identity","User-Agent":"SAR research range reader/1.0"}
    failures=[]
    for attempt in range(2):
        try:
            with urlopen(Request(url,headers=headers), timeout=22) as response:
                require(response.status==206 and response.headers.get("Content-Range")==f"bytes {start}-{end}/{total}"
                        and response.headers.get("X-Gitlab-Blob-Id")==blob
                        and response.headers.get("X-Gitlab-Commit-Id")==COMMIT
                        and response.headers.get("X-Gitlab-Content-Sha256")==file_sha, "Exact HTTP range and GitLab source identity required")
                raw=response.read(WINDOW+1)
            require(len(raw)==WINDOW,"Exact bounded range body size required")
            path=out/f"{kind}_{start:09}.range"
            with path.open("xb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            return dict(start=start,end=end,sha256=sha256(raw).hexdigest(),bytes=len(raw),failures=failures),raw
        except Exception as err:
            failures.append(f"{type(err).__name__}: {err}")
            if isinstance(err,RuntimeError):
                break
            if attempt==0:
                sleep(2)
    return dict(start=start,end=end,bytes=0,failures=failures),None


def distance(p,q):
    lat1,lat2=math.radians(p["latitude"]),math.radians(q["latitude"])
    dl=math.radians(p["longitude"]-q["longitude"])
    a=math.sin((lat1-lat2)/2)**2+math.cos(lat1)*math.cos(lat2)*math.sin(dl/2)**2
    return 12742.*math.asin(min(1.,math.sqrt(a)))


def valid_point(p):
    return type(p["row"]) is type(p["column"]) is int and all(math.isfinite(p[k]) for k in ("latitude","longitude"))


def qualify(rows,lock,api):
    products=lock["products"]
    positive=lock["positive_points"]
    allowed={(row["product"],row["pol"]) for row in lock["positives"] if row["crop_eligible"]}
    positive_hash={r["input_sha"] for r in lock["positives"]}
    group_products=defaultdict(list)
    for prod,group in products.items():
        group_products[group].append(prod)
    encountered=defaultdict(list)
    counters={kind:Counter() for kind in SOURCES}
    for kind,records in rows.items():
        _,_,_,_,tag=SOURCES[kind]
        for row in records:
            c=counters[kind]
            c["decoded_records"]+=1
            prod,pol=row["fullname"],row["polarisation"]
            if row["gttype"]!=tag:
                c["unexpected_author_tag"]+=1
                continue
            if (prod,pol) not in allowed:
                c["unmatched_product_pol"]+=1
                continue
            c["exact_product_pol"]+=1
            p=point(row)
            group=products[prod]
            if not valid_point(p):
                c["invalid_coordinates"]+=1
                continue
            if any(abs(p["row"]-q["row"])<51 and abs(p["column"]-q["column"])<51 for q in positive[prod]):
                c["same_product_positive_overlap"]+=1
                continue
            if any(distance(p,q)<=3. for other in group_products[group] if other!=prod for q in positive[other]):
                c["other_group_product_positive_nearby"]+=1
                continue
            power=np.asarray(row["patchfu"],dtype="<f8")
            if power.shape!=(51,51) or not np.isfinite(power).all() or (power<0).any():
                c["invalid_image_shape_or_values"]+=1
                continue
            gray=api.gray_levels(power)
            inp=sha256(gray.tobytes()).hexdigest()
            if inp in positive_hash:
                c["positive_exact_input_overlap"]+=1
                continue
            rank=sha256(LABEL+kind.encode()+b"\0"+bytes.fromhex(inp)).hexdigest()
            encountered[inp].append(dict(group=group,kind=kind,product=prod,pol=pol,input_sha=inp,rank=rank,
                                          gray=gray,raw=sha256(power.tobytes()).hexdigest(),**p))
    eligible=defaultdict(list)
    for inp,same in encountered.items():
        if len({(r["group"],r["kind"]) for r in same})!=1:
            for r in same:
                counters[r["kind"]]["cross_category_or_group_duplicate"]+=1
            continue
        best=min(same,key=lambda r:(r["raw"],r["product"],r["row"],r["column"]))
        counters[best["kind"]]["repeated_input"]+=len(same)-1
        eligible[(best["kind"],best["group"])].append(best)
    selected=defaultdict(list)
    used=set()
    for (kind,group), vals in sorted(eligible.items()):
        for row in sorted(vals,key=lambda r:r["rank"]):
            key=(row["product"],row["row"],row["column"])
            if key in used:
                counters[kind]["repeat_selected_center"]+=1
                continue
            selected[(kind,group)].append(row)
            used.add(key)
            if len(selected[(kind,group)])==2:
                break
    return selected,counters


def run(args):
    require(stable_sha(args.positive_lock)==args.positive_sha and stable_sha(args.protocol)==args.protocol_sha,
            "Exact prospective protocol and positive lock SHA required before network")
    locked=json.loads(args.positive_lock.read_bytes())
    require(locked["status"]=="P254_MATCHED_BACKGROUND_POSITIVE_LOCK" and len(locked["positives"])==32,
            "The original fixed cohort must be prelocked")
    api=load_positive_api(args.p254_source)
    args.output.mkdir()
    raw_path=args.output/"RANGES_PRIVATE"
    raw_path.mkdir()
    report=[]
    records={kind:[] for kind in SOURCES}
    groups=sorted(set(locked["products"].values()))
    start=perf_counter()
    selected={}
    for phase in (1,2):
        for kind,(_,size,*_) in SOURCES.items():
            if phase==2 and all(len(selected.get((kind,group),[]))==2 for group in groups):
                continue
            with ThreadPoolExecutor(max_workers=4) as pool:
                windows=list(pool.map(lambda x:read_range(kind,x,raw_path),starts(size,phase)))
            total=0
            for (info,raw) in windows:
                matched=[] if raw is None else list(fragments(raw))
                records[kind].extend(matched)
                total+=len(matched)
                report.append(dict(kind=kind,phase=phase,complete_rows=len(matched),**info))
            print(json.dumps(dict(status="MATCHED_BACKGROUND_RANGE_PHASE",kind=kind,phase=phase,
                                  received=sum(r["bytes"] for r,_ in windows),complete_rows=total)),flush=True)
        selected,counters=qualify(records,locked,api)
    receipt=save(args.output/"RANGE_READBACK_PRIVATE.json",report)
    group_counts={group:{kind:len(selected.get((kind,group),[])) for kind in SOURCES} for group in groups}
    chips=[]
    pack=args.output/"MATCHED_BACKGROUND_ONLY_PRIVATE.zip"
    with ZipFile(pack,"x") as bundle:
        for (kind,group),rows in sorted(selected.items()):
            for n,row in enumerate(rows,1):
                name=f"{kind}_{group}_{n:02}.png"
                data=BytesIO()
                Image.fromarray(row["gray"]).save(data,format="PNG",compress_level=6)
                bundle.writestr(name,data.getvalue())
                chips.append(dict(name=name,**{k:v for k,v in row.items() if k!="gray"}))
    with ZipFile(pack) as bundle:
        for row in chips:
            with Image.open(BytesIO(bundle.read(row["name"]))) as image:
                gray=np.asarray(image)
                require(gray.shape==(51,51) and sha256(gray.tobytes()).hexdigest()==row["input_sha"],
                        "Exact acquired chip readback")
    for r in report:
        if r["bytes"]:
            require(stable_sha(raw_path/f"{r['kind']}_{r['start']:09}.range")==r["sha256"],"Exact range raw readback")
    manifest=save(args.output/"CHIPS_PRIVATE.json",chips)
    safe=dict(status="SCHWEGMANN_MATCHED_BACKGROUND_BOUNDED_INTAKE_COMPLETE",source_commit=COMMIT,
              positive_lock_sha256=args.positive_sha,protocol_sha256=args.protocol_sha,range_receipt_sha256=receipt,
              chip_manifest_sha256=manifest,chip_pack_sha256=stable_sha(pack),source_fully_downloaded=False,
              positive_crops_eligible=sum(r["crop_eligible"] for r in locked["positives"]),
              range_requests={k:sum(r["kind"]==k for r in report) for k in SOURCES},
              successful_ranges={k:sum(r["kind"]==k and r["bytes"]>0 for r in report) for k in SOURCES},
              actual_bytes={k:sum(r["bytes"] for r in report if r["kind"]==k) for k in SOURCES},
              group_counts=group_counts,counts={k:dict(v) for k,v in counters.items()},official_TEST=False,GPU=False,
              model_forward_calls=0,scoring=False,seconds=perf_counter()-start)
    digest=save(args.output/"RESULT_SAFE.json",safe)
    print(json.dumps(dict(status=safe["status"],safe_sha256=digest,actual_bytes=safe["actual_bytes"],
                          totals=dict(Counter(r["kind"] for r in chips)))),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="op",required=True)
    lock=sub.add_parser("lock")
    intake=sub.add_parser("intake")
    for entry in (lock,intake):
        entry.add_argument("--p254-source",type=Path,required=True)
        entry.add_argument("--output",type=Path,required=True)
    for name in ("positive-archive","cohort"):
        lock.add_argument("--"+name,type=Path,required=True)
    for name in ("positive-lock","protocol"):
        intake.add_argument("--"+name,type=Path,required=True)
    for name in ("positive-sha","protocol-sha"):
        intake.add_argument("--"+name,required=True)
    args=parser.parse_args()
    (lock_positive if args.op=="lock" else run)(args)


if __name__=="__main__":
    main()
