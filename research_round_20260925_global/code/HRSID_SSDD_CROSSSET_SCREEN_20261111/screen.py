"""Screen closed HRSID FIT2937 and SSDD DEV187 image content for potential overlaps."""
import argparse
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import sys
from time import monotonic
from zipfile import ZipFile

import cv2
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve(strict=True).parent))
import visual_match

HRSID_BYTES = 5698302315
CENTRAL_OFFSET = 5697629273
CENTRAL_BYTES = 672944
CENTRAL_SHA = "4a54303e8cce24f0330cddd2e6d3c12205c8b0002859d0b5bf534efb531a2dce"
FIT_MANIFEST_SHA = "a7e4d2a7ff9e8ff238adbf94a1d0397553d8746e2558c316db4e417f86f8fec9"
DEV_MANIFEST_SHA = "18df007e15e3e838e607f178f14e7a26f2b37b992ee39bf091e0b74cf084fb13"
DEV_ZIP_SHA = "1bf2caae7c7b263174431cbb17e81546dc21cd372a1eb58f2451428542067eda"
CONTROL_SEED = "HRSID_SSDD_CROSSSET_VISUAL_SCREEN_CONTROLS_V1"
MAX_CANDIDATES = 32
POPCOUNT = np.asarray([bin(x).count("1") for x in range(256)], dtype=np.uint8)


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def checked(path, expected, limit):
    with path.open("rb") as file:
        raw = file.read(limit+1)
    require(len(raw) <= limit and sha256(raw).hexdigest() == expected, "Exact closed input: " + path.name)
    return raw


def image(raw, expected_format):
    with Image.open(BytesIO(raw)) as decoded:
        require(decoded.format == expected_format, "Exact original encoding format required")
        return decoded.convert("RGB")


def rgb_digest(array):
    return sha256(array.tobytes()).hexdigest()


def read_fit_member(archive, row):
    require(row["member"] == "images/" + row["native_basename"] and row["member"].endswith(".png"),
            "Only an independently locked FIT PNG member is allowed")
    z = archive.getinfo(row["member"])
    require((z.header_offset, f"{z.CRC:08x}", z.file_size, z.compress_size) ==
            (row["zip_header_offset"], row["zip_crc32"], row["encoded_bytes"], row["compressed_bytes"]),
            "Exact original FIT ZIP member identity")
    raw = archive.read(z)
    require(sha256(raw).hexdigest() == row["encoded_sha256"], "Exact FIT PNG bytes")
    rgb = np.asarray(image(raw, "PNG"))
    require(rgb.shape == (800,800,3) and rgb_digest(rgb) == row["decoded_rgb_sha256"], "Exact FIT decoded RGB")
    return rgb


def phash(gray):
    reduced = cv2.resize(gray, (32,32), interpolation=cv2.INTER_AREA).astype(np.float32)
    coeff = cv2.dct(reduced)[:8,:8].reshape(-1)
    bits = coeff > np.median(coeff[1:])
    bits[0] = False
    return np.packbits(bits)


def phash_windows(gray):
    h,w = gray.shape
    result = [phash(gray)]
    for scale in (.75,.50):
        dh,dw = round(h*scale),round(w*scale)
        for y in (0,(h-dh)//2,h-dh):
            for x in (0,(w-dw)//2,w-dw):
                result.append(phash(gray[y:y+dh,x:x+dw]))
    return np.stack(result)


def nearest_hash(bank, gray):
    distances = POPCOUNT[np.bitwise_xor(bank, phash(gray))].sum(axis=2).min(axis=1)
    return np.argsort(distances, kind="stable")[:MAX_CANDIDATES].tolist()


def jpeg_roundtrip(pic, quality):
    out=BytesIO()
    pic.save(out,format="JPEG",quality=quality)
    return np.asarray(image(out.getvalue(),"JPEG"))


def choose_controls(rows):
    ranked = sorted(range(len(rows)),key=lambda i:sha256(
        (CONTROL_SEED + rows[i]["encoded_sha256"]).encode("ascii")).hexdigest())
    selected = []
    groups = set()
    for index in ranked:
        group = rows[index]["native_basename"].split("_",1)[0]
        require(group.startswith("P") and group[1:].isdigit(),"HRSID source prefix required")
        if group not in groups:
            groups.add(group)
            selected.append(index)
        if len(selected)==24:
            return selected
    raise ValueError("24 independent-prefix controls are required")


def control_variants(rgb, ordinal):
    pic=Image.fromarray(rgb)
    lossless = BytesIO()
    pic.save(lossless,format="PNG",compress_level=1)
    result = dict(identity=np.asarray(image(lossless.getvalue(),"PNG")),
                  resize_jpeg=jpeg_roundtrip(pic.resize((600,420),Image.Resampling.BICUBIC),75))
    rng = np.random.default_rng(80211+ordinal)
    x,y = (int(x) for x in rng.integers(0,236,size=2))
    tile = pic.crop((x,y,x+565,y+565))
    gamma = .8 if ordinal%2 else 1.25
    table = np.rint(np.power(np.arange(256,dtype=float)/255,gamma)*255).astype(np.uint8)
    tile = Image.fromarray(table[np.asarray(tile)])
    result["crop_jpeg_gamma"] = jpeg_roundtrip(tile.resize((420,420),Image.Resampling.BICUBIC),70)
    blocks = rgb.reshape(5,160,5,160,3).transpose(0,2,1,3,4).reshape(25,160,160,3)
    permuted = blocks[rng.permutation(25)].reshape(5,5,160,160,3).transpose(0,2,1,3,4)
    result["shuffle_negative"] = permuted.reshape(800,800,3)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for flag in ("fit-archive","fit-manifest","dev-archive","dev-manifest","output"):
        parser.add_argument("--"+flag,type=Path,required=True)
    args=parser.parse_args()
    require(sys.flags.isolated==1 and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES")=="-1",
            "Isolated CPU process only")
    require(args.output.is_absolute() and not args.output.exists(),"Fresh private output")
    cv2.setNumThreads(1)
    start=monotonic()
    def resource():
        require(monotonic()-start<=650, "Screening worker time bound exceeded")

    fit=json.loads(checked(args.fit_manifest,FIT_MANIFEST_SHA,2**21))
    require(len(fit["images"])==2937 and fit["archive_bytes"]==HRSID_BYTES
            and fit["archive_central_sha256"]==CENTRAL_SHA,"Independent FIT scope")
    rows=sorted(fit["images"],key=lambda x:x["zip_header_offset"])
    require(len({x["member"] for x in rows})==len({x["image_id"] for x in rows})==2937,"Exact distinct FIT rows")
    control_indices=choose_controls(rows)
    features,hashes,control_inputs=[],[],{}
    with args.fit_archive.open("rb") as fp:
        before=os.fstat(fp.fileno())
        require(before.st_size==HRSID_BYTES,"Original HRSID archive size")
        fp.seek(CENTRAL_OFFSET)
        require(sha256(fp.read(CENTRAL_BYTES)).hexdigest()==CENTRAL_SHA,"Exact archive central directory")
        with ZipFile(fp) as zipfile:
            for i,row in enumerate(rows):
                rgb=read_fit_member(zipfile,row)
                gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
                obj=visual_match.features(gray)
                features.append(obj)
                hashes.append(phash_windows(obj.gray))
                if i in control_indices:
                    control_inputs[i]=rgb.copy()
                if i%100==0:
                    resource()
                if (i+1)%600==0:
                    print(json.dumps({"stage":"FIT features","images":i+1}),flush=True)
        after=os.fstat(fp.fileno())
        require((before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino),
                "Closed official archive changed")
    resource()
    index=visual_match.build_index(features)
    bank=np.stack(hashes)
    def query(rgb):
        feat=visual_match.features(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY))
        loc=visual_match.retrieval(index,feat)
        glob=nearest_hash(bank,feat.gray)
        return feat, list(dict.fromkeys(loc+glob))
    controls = {label:Counter() for label in ("identity","resize_jpeg","crop_jpeg_gamma","shuffle_negative")}
    for ordinal,source in enumerate(control_indices):
        for label,rgb in control_variants(control_inputs[source],ordinal).items():
            ctr=controls[label]
            feat,ids=query(rgb)
            ctr["attempts"]+=1
            ctr["source_retrieved"]+=source in ids
            if label=="identity":
                ctr["exact_decoded"]+=rgb_digest(rgb)==rows[source]["decoded_rgb_sha256"]
            if source in ids:
                ctr["source_strong"]+=visual_match.verify(features[source],feat)["status"]=="strong"
            if label=="shuffle_negative":
                ctr["any_strong"]+=any(visual_match.verify(features[i],feat)["status"]=="strong" for i in ids)
        resource()
    passed=(controls["identity"]["exact_decoded"]==24 and controls["resize_jpeg"]["source_strong"]>=22
            and controls["crop_jpeg_gamma"]["source_strong"]>=22 and controls["shuffle_negative"]["any_strong"]==0)
    print(json.dumps({"stage":"controls frozen","passed":passed,"controls":controls}),flush=True)
    manifest=json.loads(checked(args.dev_manifest,DEV_MANIFEST_SHA,2**20))
    target=manifest["items"]
    require(manifest["TEST"] is False and manifest["AP"] is False and manifest["model_execution"] is False
            and manifest["expected_ids"]==[r["image_id"] for r in target] and len(target)==187,"Exact DEV-only population")
    devzip=checked(args.dev_archive,DEV_ZIP_SHA,8*2**20)
    fit_encoded={row["encoded_sha256"] for row in rows}
    fit_decoded={(800,800,row["decoded_rgb_sha256"]) for row in rows}
    grouped={}
    for row in target:
        grouped.setdefault(row["image_ref"]["sha256"],[]).append(row)
    require(len(grouped)==183,"Exact unique DEV JPEG population")
    per_query=[]
    all_pairs=0
    with ZipFile(BytesIO(devzip)) as archive:
        by_id={r["image_id"]: "valid/"+Path(r["image_ref"]["path"]).name for r in target}
        require(set(archive.namelist())==set(by_id.values()) and len(archive.namelist())==187,
                "Exact sealed JPEG-only archive members")
        for ordinal,(expected,items) in enumerate(grouped.items()):
            images=[]
            for item in items:
                raw=archive.read(by_id[item["image_id"]])
                require(len(raw)==item["image_ref"]["bytes"] and sha256(raw).hexdigest()==expected,
                        "Exact DEV original JPEG bytes")
                rgb=np.asarray(image(raw,"JPEG"))
                require(tuple(rgb.shape[:2])==tuple(item["original_hw"]),"Exact DEV JPEG decoded dimensions")
                images.append(rgb)
            rgb=images[0]
            feat,ids=query(rgb)
            results=Counter()
            for i in ids:
                results[visual_match.verify(features[i],feat)["status"]]+=1
            require(set(results)<=set(("strong","weak","none")),"Expected visual verification status")
            all_pairs+=len(ids)
            per_query.append(dict(multiplicity=len(items),encoded=expected in fit_encoded,
                decoded=(*rgb.shape[:2],rgb_digest(rgb)) in fit_decoded,strong=results["strong"],weak=results["weak"],
                ordinal=ordinal+1))
            if ordinal%8==0:
                resource()
            if (ordinal+1)%40==0:
                print(json.dumps({"stage":"DEV queries","images":ordinal+1}),flush=True)
    resource()
    def counts(key):
        return dict(unique=sum(bool(row[key]) for row in per_query),
                    logical=sum(row["multiplicity"] for row in per_query if row[key]))
    summary=dict(status="HRSID_FIT2937_SSDD_DEV187_VISUAL_CONTENT_SCREEN_COMPLETE_V1",
        fit_images=2937,dev_images=187,unique_dev_jpeg=183,source_group_proxy="HRSID filename P prefix",
        fit_manifest_sha256=FIT_MANIFEST_SHA,dev_manifest_sha256=DEV_MANIFEST_SHA,dev_zip_sha256=DEV_ZIP_SHA,
        fit_central_sha256=CENTRAL_SHA,controls=dict(controls),core_controls_passed=passed,
        encoded_overlap=counts("encoded"),decoded_overlap=counts("decoded"),strong_queries=counts("strong"),
        weak_queries=counts("weak"),strong_candidate_pairs=sum(row["strong"] for row in per_query),
        weak_candidate_pairs=sum(row["weak"] for row in per_query),verified_pairs=all_pairs,
        phash_top=MAX_CANDIDATES,sift_top=MAX_CANDIDATES,elapsed_seconds=monotonic()-start,
        cv2_version=cv2.__version__,pillow_version=Image.__version__,numpy_version=np.__version__,
        annotation_reads=0,official_TEST_opened=False,model_forward_calls=0,acquisition_identity_verified=False)
    args.output.mkdir()
    with (args.output/"RESULT_SAFE.json").open("x",encoding="utf-8",newline="\n") as f:
        json.dump(summary,f,sort_keys=True,indent=2)
        f.write("\n")
    with (args.output/"QUERY_COUNTS_PRIVATE.json").open("x",encoding="utf-8",newline="\n") as f:
        json.dump(per_query,f,sort_keys=True,indent=2)
    print(json.dumps(summary,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
