
"""Score exact closed private predictions independently of native CUDA inference."""
from contextlib import redirect_stdout
import base64
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
import types

MIB=1024**2

def need(ok,message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read(ref,path,limit):
    p=Path(ref["path"])
    need(p==path and p.is_absolute() and p.resolve(strict=True)==p and not p.is_symlink()
         and 0<p.stat().st_size==ref["bytes"]<=limit,"Exact closed artifact path and size")
    b=p.read_bytes()
    need(len(b)==ref["bytes"] and digest(b)==ref["sha256"],"Exact closed artifact bytes")
    return b


def still_present(birth):
    path=Path("/proc")/str(birth["pid"])/"stat"
    try:
        row=path.read_text().rsplit(")",1)[1].split()
    except (FileNotFoundError,ProcessLookupError):
        return False
    return int(row[19])==birth["start_ticks"]


def check_owner(owner,birth_boot,spec,report_ref):
    need(owner["boot_id"]==birth_boot and owner["returncode"]==0
         and owner["status"]=="CHILD_COMPLETE_PENDING_EXTERNAL_CLOSURE"
         and owner["reaped"] is owner["owned_group_empty"] is owner["staged_sources_unchanged_after"] is True
         and not owner["remaining_owned_group"] and not owner["cleanup_errors"] and not owner["violations"]
         and owner["spec_ref"]==spec and owner["worker_report_ref"]==report_ref
         and not still_present(owner["owner_identity"]) and not still_present(owner["child_identity"]),
         "Closed same-boot owner and original child evidence")


def run(q):
    boot=Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    need(boot==q["boot_id"] and sys.executable==q["python"],"Same boot and admitted venv")
    root=Path(q["output_root"])
    runtime=types.ModuleType("_closed_native_eval_runtime")
    exec(compile(read(q["runtime"],root/"E24/SOURCE/runtime.py",MIB),"runtime.py","exec"),runtime.__dict__)
    need(sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode
         and all(path.resolve(strict=True)==path for path in runtime.SITES),"Isolated pinned CPU scoring runtime")
    sys.path.extend(map(str,runtime.SITES))
    need(version("pycocotools")=="2.0.11","Pinned official COCO library")
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    official=q["official"]
    official_raw=base64.b64decode(official["base64"],validate=True)
    need(len(official_raw)==official["bytes"] and digest(official_raw)==official["sha256"]
         =="f9fc79403d12d8840c3269b04ba22d7584b8eb5e3c592ead2b6359ad7df65748","Pinned COCO adapter")
    adapter=types.ModuleType("_closed_native_eval_official")
    exec(compile(official_raw,"official_coco_adapter.py","exec"),adapter.__dict__)
    gt_ref=q["gt"]
    need(gt_ref["sha256"]=="5657600d6ea198f79bd5d557d2a1aadce5bec031d2ecf72a98716248d1183a55",
         "Exact primary integer GT")
    gt=json.loads(read(gt_ref,Path(gt_ref["path"]),MIB))
    order=[(row["id"],row["height"],row["width"]) for row in gt["images"]]
    ids=[row[0] for row in order]
    need(len(ids)==len(set(ids))==187 and len(gt["annotations"])==380 and [c["id"] for c in gt["categories"]]==[1],
         "Frozen complete integer DEV population")
    cases=q["epoch_cases"]
    need(cases==dict(INDEPENDENT_E24=24,E24=24,E50=50,E100=100),"Fixed endpoint set")
    out={}
    precision=None
    for name,epoch in cases.items():
        case=root/name
        owner=json.loads(read(q["owners"][name],case/"OWNER_RECEIPT.json",MIB))
        spec=q["spec_refs"][name]
        worker_ref=q["workers"][name]
        check_owner(owner,boot,spec,worker_ref)
        r=json.loads(read(worker_ref,case/"RESULT.json",MIB))
        need(r["status"]=="NATIVE_E24_EMA_DEV187_EXPORT_COMPLETE_PENDING_ROOT"
             and not r["blocked_input_opens"] and r["offline_audit"]["passed"] is True
             and r["unique_DEV_images_read"]==187 and r["TEST"] is r["AP"] is r["training"] is False,
             "Closed image-only export with offline provenance")
        result=r["result"]
        expected_epoch_updates=epoch*186
        need(result["fixed_epoch"]==epoch and result["optimizer_updates"]==expected_epoch_updates
             and result["eager_model_state"]["ema_updates"]==expected_epoch_updates
             and result["eager_model_state"]["strict_readback_exact"] is True
             and result["checkpoint_sha256"]==q["archives"][name]["sha256"]
             and result["image_count"]==187 and result["native_count_total"]==18700,
             "Strict EMA identity and entire frozen population")
        numerical=(result["tensor_precision"],result["autocast"],result["float32_matmul_precision"],result["cudnn_tf32"])
        if precision is None:
            precision=numerical
        need(precision==numerical and numerical[:2]==("float32",False),"Matched CUDA evaluation precision")
        pred_ref=result["predictions_ref"]
        index=json.loads(read(result["image_manifest_ref"],case/"ARTIFACTS/IMAGE_MANIFEST.json",MIB))
        need(index["predictions_ref"]==pred_ref and index["checkpoint_sha256"]==q["archives"][name]["sha256"]
             and [(row["image_id"],*row["original_hw"]) for row in index["images"]]==order,
             "Same deterministic GT/image sequence and original coordinate geometry")
        predictions=json.loads(read(pred_ref,case/"ARTIFACTS/PREDICTIONS.json",256*MIB))
        need(len(predictions)==result["prediction_count"]==index["prediction_count"]
             ==result["codec_roundtrip_masks"] and len(predictions)+result["unmapped_slot1_total"]==18700,
             "Full mapped prediction count")
        offset=0
        for row in index["images"]:
            count=row["prediction_count"]
            block=predictions[offset:offset+count]
            need(row["prediction_start"]==offset and len(block)==count and 0<=count<=100
                 and all(item["image_id"]==row["image_id"] for item in block),"Contiguous official scoring rows")
            offset+=count
        need(offset==len(predictions),"Complete scoring rows")
        adapter.validate_predictions(predictions,[dict(image_id=i,height=h,width=w) for i,h,w in order])
        with redirect_stdout(sys.stderr):
            coco=COCO()
            coco.dataset=gt
            coco.createIndex()
            values=adapter.evaluate_population(dict(COCO=COCO,COCOeval=COCOeval),coco,predictions,ids)
        out[name]=dict(epoch=epoch,optimizer_updates=expected_epoch_updates,
                       strict_ema_tensors=result["eager_model_state"]["tensors"],
                       AP=values[0],AP75=values[2],APs=values[3],AR100=values[8],all12=values,
                       mapped_rows=len(predictions),timing_seconds=result["timing_seconds"],
                       cuda_peak_reserved_bytes=result["cuda_peak_reserved_bytes"],
                       checkpoint_sha256=result["checkpoint_sha256"],predictions_sha256=pred_ref["sha256"])
    return dict(status="DGX_FOUR_CLOSED_NATIVE_ENDPOINTS_FIXED_DEV187_OFFICIAL_COCO_SCORED",
                backend="DGX_eager_FP32",population="INTERNAL_DEV187_ALREADY_USED_FOR_DEVELOPMENT_NOT_OFFICIAL_TEST",
                precision=dict(dtype=precision[0],autocast=precision[1],matmul=precision[2],cudnn_tf32=precision[3]),
                images=187,GT_instances=380,category_ids=[1],iou_type="segm",max_dets=[1,10,100],
                original_training_SSH_wait_observed=False,
                new_evaluation_original_owner_and_child_closed=True,
                official_coco_version=version("pycocotools"),training=False,TEST=False,cases=out)


if __name__=="__main__":
    print(json.dumps(run(json.load(sys.stdin)),sort_keys=True,allow_nan=False),flush=True)
