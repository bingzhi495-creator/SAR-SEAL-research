"""Run the official one-class Mask RT-DETR-S trainer for two fixed CPU FIT updates."""
import argparse
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import sys
from time import perf_counter

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from qualify import CONFIG_SHA, MANIFEST_SHA, SEED, SOURCE_COMMIT, check_fit, deny_network, load_runtime, require, safe_write


BACKBONE_SHA="a37a27dfab3fe923ee4d1095f2f84c48d67af836d4dab2596eadeba6a1213408"


def tensor_hash(value):
    return sha256(value.numpy().tobytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ("source","fit","backbone","output","artifacts"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--protocol-commit",required=True)
    args=p.parse_args()
    start=perf_counter()
    config=args.source/"configs/mask_rtdetr/mask_rtdetr_hgnetv2_s_6x_coco.yml"
    require(sha256(config.read_bytes()).hexdigest()==CONFIG_SHA,"Expected exact S author YAML")
    require(args.backbone.stat().st_size==25457962 and sha256(args.backbone.read_bytes()).hexdigest()==BACKBONE_SHA,
            "Expected official S initialization backbone")
    require(not args.artifacts.exists(),"Expected fresh private training artifacts")
    rows=check_fit(args.fit)
    np,paddle=load_runtime(args.source)
    import psutil
    from ppdet.core.workspace import load_config
    from ppdet.engine import Trainer
    from ppdet.utils.checkpoint import match_state_dict

    cfg=load_config(str(config))
    cfg.update(use_gpu=False,num_classes=1,worker_num=0,epoch=1,save_dir=str(args.artifacts))
    cfg["TrainDataset"].update(dataset_dir=str(args.fit),image_dir="images",anno_path="annotations.json")
    cfg["TrainReader"]["use_shared_memory"]=False
    sys.addaudithook(deny_network)
    trainer=Trainer(cfg,mode="train")
    require(len(trainer.loader)==2 and len(trainer.dataset)==8 and trainer.dataset.catid2clsid=={1:0}
            and trainer.use_ema and not trainer.use_amp,"Expected actual two stock FP32 FIT batches and EMA")
    state=trainer.model.state_dict()
    before={k:tensor_hash(v) for k,v in state.items()}
    checkpoint=paddle.load(str(args.backbone))
    mapping=match_state_dict(state,checkpoint)
    require(bool(mapping) and all(k.startswith("backbone.") for k in mapping),"Backbone-only author initialization")
    trainer.load_weights(str(args.backbone))
    after=trainer.model.state_dict()
    initialized={k:tensor_hash(v) for k,v in after.items()}
    for key,value in after.items():
        if key in mapping:
            expected=mapping[key]
            if isinstance(expected,np.ndarray):
                expected=paddle.to_tensor(expected)
            require(np.array_equal(value.numpy(),expected.astype(value.dtype).numpy()),"Author loaded value mismatch")
        else:
            require(initialized[key]==before[key],"Non-backbone task tensors must stay at native seeded initial values")
    backbone_tensors_loaded=len(mapping)
    del checkpoint,mapping,state,after,before
    cpu_functions=sorted({m.ms_deformable_attn_core.__module__ for m in trainer.model.sublayers()
                          if hasattr(m,"ms_deformable_attn_core")})
    require(cpu_functions==["ppdet.modeling.transformers.utils"],"Expected native Paddle CPU attention path")
    batches=[]
    seen=[]
    counts=dict(optimizer=0,lr=0,ema=0)

    def observe_forward(layer,inputs,outputs):
        data,=inputs
        image=data["image"]
        ids=data["im_id"].numpy().reshape(-1).tolist()
        loss=float(outputs["loss"].numpy().reshape(-1)[0])
        require(len(ids)==4 and np.isfinite(loss),"Finite stock train loss on four genuine FIT inputs")
        num_rows=sum(len(v) for v in data["gt_segm"])
        empty=sum(int(np.count_nonzero(~m.numpy().reshape(len(m),-1).any(1))) if len(m) else 0 for m in data["gt_segm"])
        seen.extend(ids)
        batches.append(dict(size=list(image.shape),loss=loss,gt_rows=num_rows,empty_masks=empty))
        print(json.dumps(dict(event="MASK_RTDETR_S_ACTUAL_TRAIN_FORWARD",ordinal=len(batches),loss=loss),sort_keys=True),flush=True)

    def witness(original,name):
        def call(*items,**kwargs):
            output=original(*items,**kwargs)
            counts[name]+=1
            return output
        return call

    trainer.model.register_forward_post_hook(observe_forward)
    trainer.optimizer.step=witness(trainer.optimizer.step,"optimizer")
    trainer.lr.step=witness(trainer.lr.step,"lr")
    trainer.ema.update=witness(trainer.ema.update,"ema")
    trainer.train(validate=False)
    require(counts==dict(optimizer=2,lr=2,ema=2) and len(batches)==2
            and sorted(seen)==sorted(row["image_id"] for row in rows),"Both true stock training steps must close")
    model_path=args.artifacts/"model_final.pdparams"
    online_path=args.artifacts/"model_final.pdema"
    optimizer_path=args.artifacts/"model_final.pdopt"
    ema=paddle.load(str(model_path))
    online=paddle.load(str(online_path))
    opt=paddle.load(str(optimizer_path))
    require(set(ema)==set(online)==set(initialized) and all(
            np.isfinite(v.numpy()).all() for state in (ema,online) for v in state.values()),
            "Exact native online and EMA checkpoint state must reload finite")
    require(opt["last_epoch"]==1 and len(opt)>2,"Actual native AdamW checkpoint required")
    changed=[k for k,v in online.items() if tensor_hash(v)!=initialized[k]]
    ema_different=sum(tensor_hash(ema[k])!=tensor_hash(online[k]) for k in initialized)
    require(any("score_head" in k for k in changed) and any("mask" in k for k in changed)
            and ema_different>0,"Expected actual class and mask updates and distinct EMA values")
    hashes={p.suffix:dict(bytes=p.stat().st_size,sha256=sha256(p.read_bytes()).hexdigest())
            for p in (model_path,online_path,optimizer_path)}
    memory=psutil.Process().memory_info()
    result=dict(status="MASK_RTDETR_S_NATIVE_CPU_FIT8_TWO_UPDATE_TRAIN_QUALIFICATION_V1",
                upstream_commit=SOURCE_COMMIT,protocol_commit=args.protocol_commit,fit_manifest_sha256=MANIFEST_SHA,
                author_backbone_sha256=BACKBONE_SHA,seed=SEED,train_batches=batches,fit_visits=len(seen),
                class_slots=[0],counts=counts,completed_epochs=1,model_state_tensors=len(initialized),
                backbone_tensors_loaded=backbone_tensors_loaded,
                actual_parameter_or_buffer_changes=len(changed),online_ema_different_tensors=ema_different,
                checkpoint_files=hashes,optimizer_state_fields=len(opt),checkpoint_epoch=opt["last_epoch"],
                native_attention_modules=cpu_functions,elapsed_seconds=perf_counter()-start,
                actual_python_peak_rss_bytes=memory.peak_wset,
                runtime={n:version(n) for n in ("paddlepaddle","numpy","opencv-python","pycocotools")},
                official_DEV_opened=False,official_TEST_opened=False,AP_evaluated=False)
    safe_write(args.output,result)


if __name__=="__main__":
    main()

