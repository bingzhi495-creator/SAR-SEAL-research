"""Run a bounded native SRNet CPU inference/RLE smoke on one prelocked FIT image."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType
from zipfile import ZipFile

P335_SHA = "c078c51822ceabf107426d32fc500ccacef19cfe857b4c11cc2a6455be58c077"
P335_REPORT_SHA = "81c6f27647f19cdad1013f36620be457887b49713fe9fec425911df005102f68"
R50_SHA = "0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a"
R50_BYTES = 102530333
STATUS = "SRNET_AUTHOR_NATIVE_FIT_MASK_EXPORT_CPU_SMOKE_COMPLETE"


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def read_exact(path, pin, size_limit):
    require(path.is_absolute() and path.resolve(strict=True)==path and path.is_file()
            and 0<path.stat().st_size<=size_limit,"Canonical bounded frozen file required")
    raw=path.read_bytes()
    require(sha256(raw).hexdigest()==pin,"Pinned file SHA256 mismatch")
    return raw


def load_prior(path):
    raw=read_exact(path,P335_SHA,1024**2)
    m=ModuleType("_p336_p335")
    m.__file__=str(path)
    exec(compile(raw,str(path),"exec"),m.__dict__)
    return m


def run(a):
    begin=perf_counter()
    p=load_prior(a.prior)
    prior=json.loads(read_exact(a.prior_report,P335_REPORT_SHA,64*1024))
    require(prior["status"]==p.STATUS and prior["author_parsed_targets"]==1661 and prior["TEST"] is False,
            "Closed actual P335 author FIT loader qualification required")
    for name,pin in p.SOURCE_PINS.items():
        read_exact(a.source/name,pin,1024**2)
    data_path=a.fit/"_annotations.coco.json"
    dataset_raw=json.loads(read_exact(data_path,p.FIT_INDEX_SHA,2*1024**2))
    row=p.choose(dataset_raw["images"])[0]
    read_exact(a.archive,p.FIT_SHA,34*1024**2)
    with ZipFile(a.archive) as z:
        roster=json.loads(z.read("manifest.json"))["rows"]
    item,=[m for m in roster if m["image_id"]==row["id"]]
    require(row["file_name"]==item["original_basename"],"Exact prospective FIT image")
    read_exact(a.fit/row["file_name"],item["original_image_ref"]["sha256"],4*1024**2)
    require(len(read_exact(a.r50,R50_SHA,R50_BYTES))==R50_BYTES,"Exact public ImageNet initializer length")
    sys.path.insert(0,str(a.source))
    import cv2
    import mmcv
    import mmrotate
    import numpy as np
    import torch
    from mmdet.core import encode_mask_results
    from mmrotate.datasets import build_dataset
    from mmrotate.models import build_detector
    from pycocotools import mask as codec

    require(Path(mmrotate.__file__).resolve()==a.source/"mmrotate"/"__init__.py",
            "Exact author package source required")
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.manual_seed(5201)
    np.random.seed(5201)
    cfg=mmcv.Config.fromfile(str(a.source/p.AUTHOR_CONFIG))
    settings=cfg.data.test.copy()
    settings.ann_file=str(data_path)
    settings.img_prefix=str(a.fit)+os.sep
    settings.test_mode=True
    ds=build_dataset(settings)
    idx,=[i for i,m in enumerate(ds.data_infos) if m["id"]==row["id"]]
    observed=ds.prepare_test_img(idx)
    tensor,=observed["img"]
    meta,=observed["img_metas"]
    if not torch.is_tensor(tensor):
        tensor=tensor.data
    meta=meta.data
    hw=(row["height"],row["width"])
    require(tuple(tensor.shape)==(3,608,608) and tuple(meta["ori_shape"][:2])==hw,
            "Author eval transforms applied to the original FIT native canvas")
    cfg.model.backbone.init_cfg.checkpoint=str(a.r50)
    cfg.model.test_cfg.score_thr=0.0
    cfg.model.test_cfg.nms_pre=32
    cfg.model.test_cfg.max_per_img=4
    model=build_detector(cfg.model)
    model.init_weights()
    model.cpu().eval()
    expected=torch.load(str(a.r50),map_location="cpu")
    require(torch.equal(model.backbone.conv1.weight.detach(),expected["conv1.weight"]),
            "Author ResNet50 first convolution must reproduce the exact supplied weights")
    del expected
    with torch.inference_mode():
        item,=model.simple_test(tensor[None], [meta], rescale=True)
    boxes, segm=item
    require(len(boxes)==len(segm)==1 and 0<len(boxes[0])==len(segm[0])<=4
            and boxes[0].shape[1]==6 and bool(np.isfinite(boxes[0]).all()),
            "Bounded nonempty native mask branch returned finite paired detection rows")
    encoded=encode_mask_results(segm)
    require(len(encoded)==1 and len(encoded[0])==len(segm[0]),"Author RLE order changed")
    for mask,rle in zip(segm[0],encoded[0]):
        require(mask.shape==hw and mask.dtype==np.bool_ and np.array_equal(mask,codec.decode(rle)),
                "Author original-grid boolean mask and native COCO RLE must round-trip")
    require(not torch.cuda.is_initialized(),"CPU smoke must not initialize CUDA")
    return dict(status=STATUS,author_commit=p.AUTHOR_COMMIT,fit=True,dev=False,test=False,ap=False,
                fit_archive_sha256=p.FIT_SHA,fit_index_sha256=p.FIT_INDEX_SHA,prior_report_sha256=P335_REPORT_SHA,
                r50_initializer_sha256=R50_SHA,seed=5201,full_detection_checkpoint_loaded=False,
                author_architecture_forward_calls=1,original_hw=list(hw),model_input_hw=[608,608],
                score_threshold=0.0,nms_pre=32,max_per_img=4,det_rows=len(segm[0]),
                nonempty_masks=sum(bool(x.any()) for x in segm[0]),native_rle_roundtrips=len(encoded[0]),
                training_updates=0,elapsed_seconds=perf_counter()-begin)


def deny_network(event,_):
    if event.startswith("socket."):
        raise PermissionError("offline author model smoke")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ("prior","prior-report","source","fit","archive","r50","output"):
        p.add_argument("--"+flag,type=Path,required=True)
    a=p.parse_args()
    require(all(x.is_absolute() and x.resolve(strict=True)==x for x in (a.source,a.fit))
            and a.output.is_absolute() and a.output.parent.resolve(strict=True)==a.output.parent
            and not a.output.exists() and os.environ.get("CUDA_VISIBLE_DEVICES")=="-1",
            "Fresh isolated author CPU output required")
    sys.addaudithook(deny_network)
    raw=(json.dumps(run(a),sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    with a.output.open("xb") as out:
        out.write(raw)
        out.flush()
        os.fsync(out.fileno())
    print(json.dumps(dict(status=STATUS,result_sha256=sha256(raw).hexdigest())),flush=True)


if __name__=="__main__":
    main()
