"""One fixed DEV187 counterfactual for retaining original high-quality masks."""
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
import time
from types import ModuleType

import numpy as np

SCORER_SHA="9a0598fe77ba83379e65333304d78ca6e894bbade2e5c3e272748ce5c94a2aab"
SCORE_SHA="ab4033293a0505e7bc0d51653581a6d7dc23b59731ba25c4900b6b0366856a05"
EXPORT_SHA="770636cc9fec4d5630e58a078b27d7457b158ff81bf6eb6d1585e007536f8141"
SCOUT_SHA="69dd7c06e346eab10117ac0ae77df8ba89ebd1b19773b3b9a62211df44f75ed3"
GT_SHA="5657600d6ea198f79bd5d557d2a1aadce5bec031d2ecf72a98716248d1183a55"
NAMES=("native","local_logit","protected","strict_only","matched_nonstrict_only","unmatched_only")


def need(ok,message):
    if not ok:
        raise ValueError(message)


def substitute(native,edited,positions):
    return [edit if j in positions else stock for j,(stock,edit) in enumerate(zip(native,edited,strict=True))]


def exact(path,digest):
    need(path.is_absolute() and path.resolve(strict=True)==path and 0<path.stat().st_size<64*1024**2,
         "Expected immutable bounded input")
    raw=path.read_bytes()
    need(sha256(raw).hexdigest()==digest,"Exact original digest")
    return raw


def source(path,digest,name):
    module=ModuleType(name)
    module.__file__=str(path)
    sys.modules[name]=module
    exec(compile(exact(path,digest),str(path),"exec"),module.__dict__)
    return module


def run(args):
    started=time.perf_counter()
    need(version("pycocotools")=="2.0.11" and not args.output.exists()
         and args.output.parent.resolve(strict=True)==args.output.parent,"Exact fresh CPU diagnostic scope")
    report=json.loads(exact(args.score_report,SCORE_SHA))
    export=json.loads(exact(args.export/"EXPORT.json",EXPORT_SHA))
    need(report["seed"]==export["seed"]==5201 and report["protocol"]==export["protocol"]
         and not report["quality_guard"] and report["export_result_sha256"]==EXPORT_SHA,"Closed original stopped endpoint")
    api=source(args.scorer,SCORER_SHA,"_retention_dev_score")
    scout=source(args.scout,SCOUT_SHA,"_retention_dev_scout")
    pred={k:json.loads(exact(args.export/(k+".json"),export["predictions"][k]["sha256"]))
          for k in ("native","local_logit")}
    need(api.unchanged_identities(*pred.values()) and len(pred["native"])==15434,"Exact original candidates")
    from pycocotools import mask as codec
    from pycocotools.coco import COCO
    gt=json.loads(exact(args.gt,GT_SHA))
    gt.setdefault("info",{})
    ids=[x["id"] for x in gt["images"]]
    need(len(ids)==187 and len(gt["annotations"])==380,"Exact fixed DEV scope")
    coco=COCO()
    coco.dataset=gt
    with redirect_stdout(StringIO()):
        coco.createIndex()
    grouped={k:defaultdict(list) for k in pred}
    for k,rows in pred.items():
        for r in rows:
            grouped[k][r["image_id"]].append(r)
    streams={k:[] for k in NAMES}
    stats={k:Counter() for k in NAMES}
    lost_slack=[]
    lost_area=[]
    lost_margin=[]
    stream_order=list(dict.fromkeys(r["image_id"] for r in pred["native"]))
    need(set(stream_order)<=set(ids),"All output rows in fixed DEV population")
    for iid in stream_order+[i for i in ids if i not in set(stream_order)]:
        n=grouped["native"][iid]
        e=grouped["local_logit"][iid]
        seg=[r["segmentation"] for r in n]
        g=[coco.annToRLE(a) for a in coco.imgToAnns[iid] if not a.get("iscrowd",0)]
        iou=codec.iou(seg,g,[0]*len(g)) if seg and g else np.zeros((len(seg),len(g)))
        matches=dict(scout.greedy(iou))
        target=codec.decode(g).transpose(2,0,1).astype(bool) if g else None
        strict=set()
        for j,k in matches.items():
            y=target[k]
            f0,n0,z0=api.signed(codec.decode(seg[j]).astype(bool),y)
            if z0:
                strict.add(j)
                f1,n1,z1=api.signed(codec.decode(e[j]["segmentation"]).astype(bool),y)
                if not z1:
                    area=int(y.sum())
                    lost_area.append(area)
                    lost_slack.append(area-10*n0-9*f0)
                    lost_margin.append(10*(n1-n0)+9*(f1-f0))
        active={j for j,(a,b) in enumerate(zip(n,e,strict=True)) if a["segmentation"]!=b["segmentation"]}
        selections=dict(native=set(),local_logit=active,protected=active-strict,strict_only=active&strict,
                        matched_nonstrict_only=(active&set(matches))-strict,unmatched_only=active-set(matches))
        for arm,chosen in selections.items():
            stream=substitute(n,e,chosen)
            streams[arm].extend(stream)
            d=stats[arm]
            d.update(edited_rows=len(chosen),matched=len(matches),original_strict=len(strict))
            for j in chosen&set(matches):
                y=target[matches[j]]
                f0,n0,z0=api.signed(codec.decode(seg[j]).astype(bool),y)
                f1,n1,z1=api.signed(codec.decode(e[j]["segmentation"]).astype(bool),y)
                d.update(strict_loss=int(z0 and not z1),strict_gain=int(z1 and not z0),fp_delta=f1-f0,fn_delta=n1-n0)
    # Reassemble in original export order regardless of GT record order.
    need(all(api.unchanged_identities(pred["native"],v) for v in streams.values()),"No candidate identity changes")
    metrics={k:scout.fit_stats(gt,v,ids) for k,v in streams.items()}
    need(all(metrics[k]==report["metrics"][k] for k in ("native","local_logit"))
         and len(lost_slack)==report["geometry"]["local_logit"]["strict_loss"]==11
         and stats["protected"]["strict_loss"]==0 and stats["native"]["original_strict"]==60,
         "Exact replay and original-strict retention")
    native=metrics["native"]
    p=metrics["protected"]
    delta={arm:{k:m[k]-native[k] for k in ("AP","AP75","AP90","APsmall","AR100")}
           for arm,m in metrics.items() if arm!="native"}
    gate=(p["AP"]-report["strong_reference_E100_AP"]>=.01 and p["AP75"]>=native["AP75"]
          and p["AP90"]-native["AP90"]>=.01)
    bins=Counter((19 if n<19 else 38 if n<38 else 57 if n<57 else 76) for n in lost_slack)
    out=dict(status="DEV187_STRICT_RETENTION_FIXED_CROSSREAD_COMPLETE",protocol=args.protocol,
             original_protocol=report["protocol"],metrics=metrics,geometry=stats,delta_vs_native=delta,
             protected_minus_E100_AP=p["AP"]-report["strong_reference_E100_AP"],mechanism_gate=bool(gate),
             loss_rows=dict(count=len(lost_area),small=sum(v<1024 for v in lost_area),
                            slack_bins=dict(bins),slack_median=float(np.median(lost_slack)),
                            added_signed_cost_median=float(np.median(lost_margin))),
             elapsed_s=time.perf_counter()-started,AP=True,DEV=True,TEST=False,
             uses_GT_for_substitution=True,optimizer_updates=0)
    args.output.mkdir(mode=0o700)
    with (args.output/"RESULT.json").open("xb") as f:
        f.write((json.dumps(out,separators=(",",":"),allow_nan=False)+"\n").encode())
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps(out,sort_keys=True),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("scorer","scout","score-report","export","gt","output"):
        p.add_argument("--"+k,type=Path,required=True)
    p.add_argument("--protocol",required=True)
    run(p.parse_args())

