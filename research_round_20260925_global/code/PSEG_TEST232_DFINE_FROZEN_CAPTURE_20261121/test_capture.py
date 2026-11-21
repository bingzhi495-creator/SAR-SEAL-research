"""Test the official roster contract and independent capture census using generated JPEGs and masks."""

from collections import Counter
from copy import deepcopy
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
from io import BytesIO
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from PIL import Image
from pycocotools import mask as codec


def load(path):
    spec = spec_from_file_location(path.stem, path)
    obj = module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def digest(data):
    return sha256(data).hexdigest()


here = Path(__file__).resolve().parent
target = load(here / "target_capture.py")
verify = load(here / "verify_target.py")
refs = Path(sys.argv[1]).resolve(strict=True)
base = target.load_source(refs, "author_export_reference.py", target.BASE_SHA)
sidecar = target.load_source(refs, "probability_sidecar.py", target.SIDECAR_SHA)
target.load_source(refs, "native_capture_reference.py", target.CAPTURE_SHA)
assert digest((here / "target_capture.py").read_bytes()) == verify.WORKER


def generated_jpeg(height, width):
    with BytesIO() as buf:
        Image.fromarray(np.zeros((height,width,3),dtype=np.uint8)).save(buf, format="JPEG")
        return buf.getvalue()


def record(path, value):
    raw = sidecar.canonical_line(value)
    path.write_bytes(raw)
    return digest(raw)


with TemporaryDirectory(dir=here) as name:
    root = Path(name)
    image_dir = root / "jpegs"
    image_dir.mkdir()
    export = root / "synthetic_capture"
    export.mkdir()
    items, expected_lines = [], []
    jpg_names = [f"{i:06}.jpg" for i in range(1,1161) if i % 10 in (1,9)]
    sizes = [(494,300),(192,668)]
    jpegs = [generated_jpeg(*hw) for hw in sizes]
    for ordinal, filename in enumerate(jpg_names):
        image = jpegs[ordinal % 2]
        h,w = sizes[ordinal % 2]
        ref = dict(path=filename, bytes=len(image), sha256=digest(image))
        expected_lines.append("\t".join([filename,str(len(image)),digest(image)]) + "\n")
        items.append(dict(image_id=ordinal, original_hw=[h,w], image_ref=ref))
        if ordinal < 2:
            (image_dir / filename).write_bytes(image)
            assert target.read_rgb(image_dir, items[-1], base).shape == (h,w,3)
    doc = dict(schema=target.SCHEMA, population="official_TEST232", model_execution=False, AP=False,
               TEST=True, GT_opened=False, expected_ids=list(range(232)), items=items)
    manifest = root / "manifest.json"
    md = record(manifest, doc)
    inventory = digest("".join(expected_lines).encode("ascii"))
    assert target.validate_roster(doc, base)[1] == inventory
    try:
        target.manifest_rows(manifest, md, base)
    except ValueError as exc:
        assert "inventory" in str(exc)
    else:
        raise AssertionError("Generated images admitted as real TEST232")
    for defect in ("labels", "width", "role", "repeat_id"):
        broken = deepcopy(doc)
        if defect == "labels":
            broken["annotations"] = []
        elif defect == "width":
            broken["items"][4]["original_hw"][1] = 669
        elif defect == "role":
            broken["TEST"] = False
        else:
            broken["expected_ids"][0] = 1
        try:
            target.validate_roster(broken, base)
        except ValueError:
            continue
        raise AssertionError(defect)

    observed = Counter()
    ph,ah = sha256(),sha256()
    with sidecar.Capture(export) as archive, (export/base.PREDICTIONS).open("wb") as pf, (export/base.AUDIT).open("wb") as af:
        for i,row in enumerate(items):
            h,w=row["original_hw"]
            count = int(i < 2)
            stock = np.zeros((h,w),np.uint8)
            stock[2:6,3:8] = 1
            rle = codec.encode(np.asfortranarray(stock))
            rle["counts"] = rle["counts"].decode()
            pred = dict(image_id=i, coco=[dict(image_id=i,category_id=1,score=.75,segmentation=rle)] if count else [])
            detail = dict(native_rank=1,score=.75,box_xyxy=[3.,2.,8.,6.],mask_area=20,query_id=123,probability_row=0)
            audit = dict(image_id=i,raw_queries=300,internal_top=300,post_nms=count,candidates=[detail] if count else [])
            audit=archive.write(i+1,i,audit,np.zeros((count,160,160),dtype="<f4"))
            assert base.validate_rows(i,[h,w],pred,audit) == (count,0)
            for content,stream,hashobj in ((pred,pf,ph),(audit,af,ah)):
                raw=base.encoded(content)
                stream.write(raw)
                hashobj.update(raw)
    protocol = "c"*64
    result=dict(status=target.STATUS,protocol_sha256=protocol,export_source_sha256=verify.WORKER,
        ema_sha256=target.EMA_SHA,e24_acceptance_sha256=target.ACCEPT_SHA,manifest_sha256=md,
        images=232,model_forwards=232,native_dimensions_verified=232,raw_queries=69600,internal_top_rows=69600,
        exact_author_binary_replay=2,post_nms_rows=2,exact_top100_binary_replay=2,rows_at_100=2,
        original_author_postprocess_parity=True,confidence_threshold=0.,box_nms_iou=.7,mask_threshold=.5,
        keep_ratio=False,rect=False,model_precision="FP32_TF32_disabled",mask_resize_precision="author_CUDA_FP16",
        TEST=True,DEV=False,FIT=False,GT_opened=False,AP=False,historical_target_exposure=True,optimizer_updates=0,
        predictions_sha256=ph.hexdigest(),candidate_audit_sha256=ah.hexdigest(),image_inventory_sha256=inventory,
        empty_post_nms=0,empty_at_100=0,zero_prediction_images=230,**archive.receipt())
    result_sha=record(export/"RESULT_SAFE.json",result)
    terminal=root/"terminal.json"
    terminal_sha=record(terminal, dict(status="DFINE_HRSID_E24_SSDD_TEST232_PROBABILITY_EXPORT_OS_CLOSED_V1",
        original_os_wait=True,original_os_exit=0,job_drained=True,resource_stop=False,
        export_result_sha256=result_sha,protocol_sha256=protocol,original_tool_wait_sha256="d"*64))
    args=["verify_target.py","--export",str(export),"--manifest",str(manifest),"--terminal",str(terminal),
          "--output",str(root/"verified.json"),"--base",str(refs/"author_export_reference.py"),
          "--sidecar-source",str(refs/"probability_sidecar.py"),"--codec-wheel","synthetic","--protocol-sha",protocol,
          "--terminal-sha",terminal_sha,"--result-sha",result_sha,"--manifest-sha",md]
    # The test uses the installed codec. Production still requires the exact original wheel and image inventory.
    original_source = verify.source
    def installed_codec(path, checksum):
        scope=original_source(path,checksum)
        if checksum == verify.BASE:
            scope["prepare_export_codec"]=lambda wheel,destination: None
        return scope
    with patch.object(verify,"source",installed_codec), patch.object(verify,"INVENTORY_SHA",inventory), \
         patch.object(verify,"MANIFEST_SHA",md), \
         patch.object(sys,"argv",args), patch.dict(os.environ,CUDA_VISIBLE_DEVICES="-1"):
        verify.main()
        accepted=json.loads((root/"verified.json").read_bytes())
        assert accepted["probability_rows"] == 2 and accepted["images"] == 232 and accepted["TEST"]
        path = export/"PROBABILITY_PRIVATE"/"0001.npy"
        corrupt=bytearray(path.read_bytes())
        corrupt[-1]^=1
        path.write_bytes(corrupt)
        try:
            verify.main()
        except ValueError as exc:
            assert "probability bytes" in str(exc)
        else:
            raise AssertionError("Corrupt probability plane was accepted")
print("232 generated rows; landscape/portrait JPEGs; variable-size masks; zero and nonzero candidates; corruption rejected")
