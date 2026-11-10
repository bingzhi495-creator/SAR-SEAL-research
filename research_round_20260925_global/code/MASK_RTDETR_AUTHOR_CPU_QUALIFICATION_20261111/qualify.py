"""Qualify stock Mask RT-DETR-S dataset/reader and a strict COCO CPU forward."""
import argparse
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import random
import sys
from time import perf_counter


SEED=20261111
SOURCE_COMMIT="b25522a0f4bde8c80603f3ba5e3472059972e3b5"
CONFIG_SHA="fecf97a00f1de72f671472f21f2a7c713144eab8e5900f7983f7d7abc9e585fd"
MANIFEST_SHA="8a0099846bdb530f2250d8c8473014b934a9107bc82f6fd053074b71a77a746a"
WEIGHT_SHA="d71ea6590442d1c3d6118fe0558b4a9fe84283a4832b12dc6d7b5c8cf3e09d9f"


def require(condition,message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()


def safe_write(path,value):
    raw=canonical(value)
    with path.open("xb") as f:
        f.write(raw)
    print(json.dumps(dict(event=value["status"],sha256=sha256(raw).hexdigest()),sort_keys=True),flush=True)


def deny_network(event,args):
    if event in {"socket.connect","socket.getaddrinfo","socket.gethostbyname","socket.gethostbyaddr"}:
        raise RuntimeError("Offline author CPU qualification required")


def check_fit(root):
    path=root/"CLOSED_INPUT_PRIVATE.json"
    raw=path.read_bytes()
    require(sha256(raw).hexdigest()==MANIFEST_SHA,"Expected exact sealed eight-source roster")
    manifest=json.loads(raw)
    rows=manifest["images"]
    require(len(rows)==8 and len({r["sha256"] for r in rows})==8 and manifest["annotation_count"]==16
            and manifest["official_DEV_opened"] is False and manifest["official_TEST_opened"] is False,
            "Expected 8 distinct train images and 16 ship rows")
    for r in rows:
        name=r["basename"]
        require(Path(name).name==name and name.endswith(".jpg"),"Expected exact original JPEG basename")
        p=root/"images"/name
        data=p.read_bytes()
        require(len(data)==r["bytes"] and sha256(data).hexdigest()==r["sha256"],"Expected original FIT image bytes")
    labels=(root/"annotations.json").read_bytes()
    require(sha256(labels).hexdigest()==manifest["annotation_sha256"],"Expected exact FIT annotation bytes")
    return rows


def load_runtime(source):
    require(os.environ.get("CUDA_VISIBLE_DEVICES")=="-1" and os.environ.get("OMP_NUM_THREADS")=="1"
            and sys.version_info[:3]==(3,10,13), "Expected isolated CPU single-thread Python runtime")
    sys.path.insert(0,str(source))
    import numpy as np
    import paddle
    import ppdet
    require(not paddle.is_compiled_with_cuda() and paddle.__version__=="2.6.2","Expected exact native CPU Paddle")
    require(Path(ppdet.__file__).resolve().parent==(source/"ppdet").resolve(),"Expected exact author package")
    paddle.set_device("cpu")
    random.seed(SEED)
    np.random.seed(SEED)
    paddle.seed(SEED)
    return np,paddle


def reader_qualification(cfg,create,root,rows,np):
    cfg["num_classes"]=1
    cfg["TrainDataset"].update(dataset_dir=str(root),image_dir="images",anno_path="annotations.json")
    cfg["TrainReader"]["use_shared_memory"]=False
    dataset=create("TrainDataset")()
    reader=create("TrainReader")(dataset,0)
    require(len(reader)==2 and len(dataset)==8,"Native TrainReader must expose two four-image batches")
    require(dataset.catid2clsid=={1:0},"Stock reader must map the sole ship category to slot zero")
    seen=[]
    batch_rows=[]
    for batch in reader:
        image=batch["image"].numpy()
        ids=batch["im_id"].numpy().reshape(-1).tolist()
        require(image.ndim==4 and image.shape[:2]==(4,3) and len(ids)==4
                and np.isfinite(image).all(),"Expected native finite four-image CHW tensor")
        sizes=[]
        for boxes,classes,masks in zip(batch["gt_bbox"],batch["gt_class"],batch["gt_segm"],strict=True):
            box=boxes.numpy()
            label=classes.numpy()
            mask=masks.numpy()
            n=len(box)
            require(box.shape==(n,4) and label.shape==(n,1) and (label==0).all()
                    and mask.shape==(n,*image.shape[-2:]) and np.logical_or(mask==0,mask==1).all()
                    and np.isfinite(box).all(),"Expected stock single-class box and raster row correspondence")
            sizes.append(dict(retained_rows=n,nonempty_masks=int(np.count_nonzero(mask.reshape(n,-1).any(1))) if n else 0))
        seen.extend(ids)
        batch_rows.append(dict(images=4,output_hw=list(image.shape[-2:]),
                               retained_rows=sum(x["retained_rows"] for x in sizes),
                               nonempty_masks=sum(x["nonempty_masks"] for x in sizes)))
    require(sorted(seen)==sorted(row["image_id"] for row in rows),"Expected each of eight admitted FIT sources once")
    return dict(batches=batch_rows,fit_visits=len(seen),distinct_fit_visits=len(set(seen)),class_slots=[0])


def stock_forward(cfg,create,source,root,row,weights,np,paddle):
    require(weights.stat().st_size==58728806 and sha256(weights.read_bytes()).hexdigest()==WEIGHT_SHA,
            "Exact official COCO endpoint weights required")
    cfg["num_classes"]=80
    cfg["TestDataset"].update(dataset_dir=None,anno_path=None)
    ds=create("TestDataset")()
    ds.set_images([str(root/"images"/row["basename"])],do_eval=False)
    reader=create("TestReader")(ds,0)
    require(len(reader)==1 and len(ds)==1,"Expected one original FIT source in stock unlabeled TestReader")
    model=create(cfg.architecture)
    model.load_meanstd(cfg["TestReader"]["sample_transforms"])
    state=paddle.load(str(weights))
    target=model.state_dict()
    require(set(target)==set(state),"Author COCO checkpoint must cover every exact parameter and buffer")
    require(all(tuple(target[k].shape)==tuple(state[k].shape) and target[k].dtype==state[k].dtype for k in target),
            "Author checkpoint shapes and types must match stock S model")
    model.set_state_dict(state)
    verified=model.state_dict()
    require(all(np.array_equal(verified[k].numpy(),state[k].numpy()) for k in target),
            "Exact official state values must be loaded")
    cpu_functions=sorted({m.ms_deformable_attn_core.__module__ for m in model.sublayers()
                          if hasattr(m,"ms_deformable_attn_core")})
    require(cpu_functions==["ppdet.modeling.transformers.utils"],"Expected native Paddle CPU attention")
    model.eval()
    batch,=list(reader)
    require(tuple(batch["image"].shape)==(1,3,640,640),"Expected stock TestReader resolution")
    with paddle.no_grad():
        pred=model(batch)
    boxes=pred["bbox"].numpy()
    masks=pred["mask"].numpy()
    counts=pred["bbox_num"].numpy().reshape(-1).tolist()
    require(counts==[100] and boxes.shape==(100,6) and masks.shape==(100,*row["original_hw"])
            and np.isfinite(boxes).all() and np.logical_or(masks==0,masks==1).all(),
            "Expected stock 100 candidates and original-grid binary masks")
    require(np.all((boxes[:,0]>=0)&(boxes[:,0]<80)&(boxes[:,0]==np.floor(boxes[:,0]))),
            "Expected actual COCO output class indices")
    return dict(weight_tensors_loaded=len(target),native_attention_modules=cpu_functions,
                input_shape=[1,3,640,640],boxes_shape=list(boxes.shape),masks_shape=list(masks.shape),
                native_top_predictions=counts[0],native_nonempty_masks=int(np.count_nonzero(masks.reshape(100,-1).any(1))),
                optimizer_updates=0,model_forward_calls=1)


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ("source","fit","weights","output","reader-output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--protocol-commit",required=True)
    args=p.parse_args()
    start=perf_counter()
    config=args.source/"configs/mask_rtdetr/mask_rtdetr_hgnetv2_s_6x_coco.yml"
    require(sha256(config.read_bytes()).hexdigest()==CONFIG_SHA,"Expected exact official S YAML")
    rows=check_fit(args.fit)
    np,paddle=load_runtime(args.source)
    from ppdet.core.workspace import create,load_config
    cfg=load_config(str(config))
    cfg["use_gpu"]=False
    cfg["worker_num"]=0
    sys.addaudithook(deny_network)
    result=dict(protocol_commit=args.protocol_commit,upstream_commit=SOURCE_COMMIT,seed=SEED,
                runtime={n:version(n) for n in ("paddlepaddle","numpy","opencv-python","pycocotools","protobuf")},
                fit_manifest_sha256=MANIFEST_SHA,official_DEV_opened=False,official_TEST_opened=False,AP_evaluated=False)
    train=reader_qualification(cfg,create,args.fit,rows,np)
    safe_write(args.reader_output,dict(result,status="MASK_RTDETR_S_NATIVE_CPU_TRAIN_READER_FIT8_COMPLETE_V1",
                                       reader=train,model_forward_calls=0,optimizer_updates=0))
    inference=stock_forward(cfg,create,args.source,args.fit,rows[0],args.weights,np,paddle)
    safe_write(args.output,dict(result,status="MASK_RTDETR_S_NATIVE_CPU_FIT_COCO_FORWARD_COMPLETE_V1",
                                reader=train,forward=inference,coco_weights_sha256=WEIGHT_SHA,
                                elapsed_seconds=perf_counter()-start))


if __name__=="__main__":
    main()

