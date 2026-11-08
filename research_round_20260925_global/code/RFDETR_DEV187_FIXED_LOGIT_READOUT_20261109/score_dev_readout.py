"""Independent CPU evaluation of an OS-closed, image-only DEV187 export."""
import argparse
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from hashlib import sha256
from importlib.metadata import version
from io import StringIO
import json
import os
from pathlib import Path
import sys
from types import ModuleType

import numpy as np

sys.path.insert(0,str(Path(__file__).parent))
from dev_readout_contract import ARMS,DEV_GT_SHA,E100_AP,NATIVE_SHA,SEEDS,encode_json,unchanged_identities

SCOUT_SHA="69dd7c06e346eab10117ac0ae77df8ba89ebd1b19773b3b9a62211df44f75ed3"
NATIVE=dict(AP=0.6744334056814277,AP75=0.8538575350205851,
            APsmall=0.6731170548617869,AR100=0.7526315789473684,predictions=15434)


def need(ok,message):
    if not ok:
        raise ValueError(message)


def read(path,size,digest):
    need(path.is_absolute() and path.resolve(strict=True)==path and path.stat().st_size==size
         and 0<size<=64*1024**2,"Exact bounded input")
    b=path.read_bytes()
    need(len(b)==size and sha256(b).hexdigest()==digest,"Pinned independent input bytes")
    return b


def signed(pred,truth):
    fp=int(np.count_nonzero(pred&~truth))
    fn=int(np.count_nonzero(~pred&truth))
    area=int(np.count_nonzero(truth))
    return fp,fn,area>0 and 10*fn+9*fp<=area


def geometry(pred,gt,ids,codec,scout):
    from pycocotools.coco import COCO
    coco=COCO()
    coco.dataset=gt
    with redirect_stdout(StringIO()):
        coco.createIndex()
    grouped={}
    for arm,rows in pred.items():
        by_image=defaultdict(list)
        for row in rows:
            by_image[row["image_id"]].append(row["segmentation"])
        grouped[arm]=by_image
    sums={k:Counter() for k in ARMS}
    for iid in ids:
        masks=grouped["native"][iid]
        truth=[coco.annToRLE(a) for a in coco.imgToAnns[iid] if not a.get("iscrowd",0)]
        iou=codec.iou(masks,truth,[0]*len(truth)) if masks and truth else np.zeros((len(masks),len(truth)))
        matched=dict(scout.greedy(iou))
        target=codec.decode(truth).transpose(2,0,1).astype(bool) if truth else None
        original={j:signed(codec.decode(masks[j]).astype(bool),target[t]) for j,t in matched.items()}
        for arm,by_image in grouped.items():
            edited=by_image[iid]
            need(len(edited)==len(masks) and (not masks or np.array_equal(codec.area(masks),codec.area(edited))),
                 "Identical candidate count and per-row original area")
            changed=[j for j,(a,b) in enumerate(zip(masks,edited,strict=True)) if a!=b]
            total=sums[arm]
            total.update(matched=len(matched),original_strict=sum(x[2] for x in original.values()),
                         edited_rows=len(changed),edited_unmatched=sum(j not in matched for j in changed))
            for j in changed:
                n=codec.decode(masks[j]).astype(bool)
                p=codec.decode(edited[j]).astype(bool)
                need(int(np.count_nonzero(p!=n))==8,"Each admitted row has exact four-add four-delete budget")
                if j not in matched:
                    continue
                f,g,z=signed(p,target[matched[j]])
                f0,g0,z0=original[j]
                total.update(fp_delta=f-f0,fn_delta=g-g0,strict_loss=int(z0 and not z),
                             strict_gain=int(z and not z0),both_reduced=int(f<f0 and g<g0))
    return sums


def run(args):
    need(version("pycocotools")=="2.0.11" and not args.output.exists()
         and args.output.parent.resolve(strict=True)==args.output.parent,"Pinned CPU scoring scope")
    terminal=json.loads(read(args.export_terminal,args.export_terminal.stat().st_size,args.export_terminal_sha))
    birth=terminal["worker"]
    need(terminal["exit"]==0 and terminal["original_OS_wait"] and not terminal["resource_stop"]
         and terminal["protocol"]==args.protocol and terminal["seed"]==args.seed
         and not Path(f"/proc/{birth['pid']}").exists(),"Original image worker genuinely closed")
    report_path=args.export/"EXPORT.json"
    report=json.loads(read(report_path,report_path.stat().st_size,terminal["result_sha256"]))
    need(report["status"]=="DEV187_FIXED_LOGIT_IMAGE_ONLY_EXPORT_COMPLETE" and not report["dev_gt_read"]
         and not report["TEST"] and report["native_exact_sha"]==NATIVE_SHA and report["seed"]==args.seed
         and report["protocol"]==args.protocol and report["images"]==187 and report["mapped_rows"]==15434,
         "Qualified image-only prediction artifact")
    def read_one(k):
        ref=report["predictions"][k]
        return read(args.export/(k+".json"),ref["bytes"],ref["sha256"])
    pred={k:json.loads(read_one(k)) for k in ARMS}
    keys=json.loads(read_one("ROW_KEYS"))
    need(len(keys)==15434 and all(unchanged_identities(pred["native"],p) for p in pred.values())
         and all([r["image_id"],0,r["score"].hex()]==[k[0],k[3],k[4]]
                 for r,k in zip(pred["native"],keys,strict=True)), "Fixed complete candidate identities and score bits")
    src=read(args.scout,args.scout.stat().st_size,SCOUT_SHA)
    scout=ModuleType("_sealed_dev_score_api")
    scout.__file__=str(args.scout)
    exec(compile(src,str(args.scout),"exec"),scout.__dict__)
    from pycocotools import mask as codec
    # DEV annotations are opened only after all source, output, and original-exit checks.
    gt=json.loads(read(args.gt,118319,DEV_GT_SHA))
    gt.setdefault("info",{})
    ids=[x["id"] for x in gt["images"]]
    need(len(set(ids))==len(ids)==187 and len(gt["annotations"])==380
         and [c["id"] for c in gt["categories"]]==[1],"Exact DEV GT population")
    metrics={k:scout.fit_stats(gt,p,ids) for k,p in pred.items()}
    need(all(metrics["native"][k]==v for k,v in NATIVE.items()),"Exact independent E24 DEV AP replay")
    geom=geometry(pred,gt,ids,codec,scout)
    need(all(geom[k]["edited_rows"]==report["eligible_rows"] for k in ARMS[1:]),"Exact same action budget")
    native=metrics["native"]
    loc=metrics["local_logit"]
    best=max([E100_AP]+[metrics[k]["AP"] for k in ARMS[:-1]])
    g=loc["AP"]-best
    quality=(loc["AP75"]>=native["AP75"] and loc["AP90"]>native["AP90"]
             and geom["local_logit"]["strict_loss"]*20<=geom["local_logit"]["original_strict"])
    if args.seed==5201:
        decision=("STOP_NONPOSITIVE_OR_QUALITY_GUARD" if g<=0 or not quality else
                  "ALLOW_SECOND_ONLY" if g<.01 else "ALLOW_SECOND_STRONG_FIRST")
    else:
        decision="APPLY_PRELOCKED_ACROSS_SEED_RULE_WITH_PRIOR_VERIFIED_RESULTS"
    out=dict(status="DEV187_FIXED_LOGIT_PROSPECTIVE_SCORE_COMPLETE",protocol=args.protocol,seed=args.seed,
             metrics=metrics,geometry=geom,AP=True,DEV=True,TEST=False,strong_reference_E100_AP=E100_AP,
             primary_gain_fraction=g,quality_guard=quality,decision=decision,
             original_export_terminal_sha256=args.export_terminal_sha,native_predictions_sha256=NATIVE_SHA,
             export_result_sha256=terminal["result_sha256"],training=report["training"],
             updates_each_learned=report["updates_each_learned"],detector_updates=0,
             fit_loss=report["loss"],timing=report["shared_and_incremental_seconds"],
             fit_total_seconds=report["fit_total_seconds"],per_head_training_seconds=report["per_head_training_seconds"],
             cuda_peak_bytes=report["cuda_peak_bytes"],images=187)
    args.output.mkdir(mode=0o700)
    with (args.output/"RESULT.json").open("xb") as f:
        f.write(encode_json(out))
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps(out,sort_keys=True),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("export","export-terminal","scout","gt","output"):
        p.add_argument("--"+n,type=Path,required=True)
    p.add_argument("--protocol",required=True)
    p.add_argument("--export-terminal-sha",required=True)
    p.add_argument("--seed",type=int,choices=SEEDS,required=True)
    run(p.parse_args())

