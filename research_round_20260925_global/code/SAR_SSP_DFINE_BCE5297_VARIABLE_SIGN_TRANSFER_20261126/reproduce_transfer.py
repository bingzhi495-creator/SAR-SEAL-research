"""Path-configurable audit replay of the frozen SAR-SSP transfer."""
from argparse import ArgumentParser
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType, SimpleNamespace
from zipfile import ZipFile

HERE=Path(__file__).resolve().parent
OLD: Path
SOURCE: Path
VAR: Path
CODE: Path
INTAKE: Path
SIGNED: Path
BASE_RENDER: Path
FIT: Path
CORE: Path
SELECTOR: Path
ARCHIVE: Path
LOCK="fdbe685d142f326e08f5f8870fdf4c62bcf52cb4132852415fd5d492e9d8e5c4"
OLD_REPLAY="b76c243b13047d4cc3906ddde5c934850003ced019e9c8116dac2c44359ea15f"
OLD_VERIFY="223715cb6c21a6d9b20927410a914605c51e2fca566df20e44a1ec71596fc853"
OLD_SCORE="77da6f63f83e23c804a8773bb40d432531241bab68b3871bdf7ba936d79e9bc6"
SOURCE_SCORE="f86390623ffd658e91c17e8a7b99e0859bcc3918b9fce0afda2ff332d72d023a"
SOURCE_RENDER="d12a193bf80e6fb24b3c853c8da7d342f6408aa22e9f3920b873f2d5eb3de1e5"
SOURCE_RENDER_WORKER="a0c11c5c37c3a8569e1a54047fbc09598604710861bcd19f4096a0e3abe35959"
FIT_SHA="ae1b165a111a6b88939eb37b5741421259f0f39531c859a7c7663ad9116d1b05"
ARCH_BCE="a2c169f895b1435e689480836ebf4f04fd6e05d84652f51d74e0448ed099faff"
WRITER="fb49070f10d064fbe1bcce89e7baf1fb59d3ae81950863437adb95c37c7c4b81"
SIGNED_SHA="c4841925513aeee64078064d483c73557b48c352594dd0d1edbe67a15be4b5b8"
NEW_ARMS=("fixed_iou","variable_iou","near_iou","bce_fixed_robust")
OLD_ARMS={"attraction_static_at_dynamic_count":"static",
          "attraction_dynamic":"dynamic",
          "learned_at_attraction_count":"legacy_attraction_learned",
          "free_matched_count":"legacy_free_matched",
          "fixed":"legacy_fixed_distance"}
OLD_SCORE_MAP={"static":"static_at_dynamic_count","dynamic":"dynamic_attraction",
               "legacy_attraction_learned":"learned_at_attraction_count",
               "legacy_free_matched":"free_matched_count","legacy_fixed_distance":"distance_paired"}
RENDER_STATUS="SAR_SSP_DFINE_BCE5297_VARIABLE_SIGN_RENDER_COMPLETE"
SCORE_STATUS="SAR_SSP_DFINE_BCE5297_VARIABLE_SIGN_TRANSFER_COMPLETE"


def need(ok,msg):
    if not ok:
        raise ValueError(msg)


def digest(raw):
    return sha256(raw).hexdigest()


def pin(path,expected):
    raw=path.read_bytes()
    need(digest(raw)==expected,f"Exact bound input: {path.name}")
    return raw


def load(path,expected,name):
    m=ModuleType(name)
    m.__file__=str(path)
    exec(compile(pin(path,expected),str(path),"exec"),vars(m))
    return m


def line(value):
    return (json.dumps(value,sort_keys=True,allow_nan=False,separators=(",",":"))+"\n").encode("ascii")


def finish(output,status,started,**outcome):
    data=line(dict(status=status,protocol_sha256=LOCK,worker_sha256=digest(Path(__file__).read_bytes()),
                   source_fit_sha256=FIT_SHA,source_writer_sha256=WRITER, detector_forwards=0,
                   optimizer_updates=0,historically_exposed_cohort=True,new_target_calibration=False,
                   elapsed_seconds=perf_counter()-started,**outcome))
    with (output/"RESULT_SAFE.json").open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(status=status,result_sha256=digest(data))),flush=True)


def setup():
    old=load(OLD/"replay_score.py",OLD_REPLAY,"_closed_sar_ssp")
    args=SimpleNamespace(code=CODE,manifest=INTAKE/"IMAGE_MANIFEST_PRIVATE.json",images=INTAKE/"images")
    m, _, roster, capture=old.configure(args)
    scoring=old.load(CODE/"SAR_SSP_AIR_FROZEN_20261125/score_sarssp.py",old.SCORE,"_sar_ssp_author_raster")
    checked=old.read_phase("verify",OLD_VERIFY)
    old.terminal_at("score",OLD_SCORE)
    previous=json.loads(pin(OLD/"score/RESULT_SAFE.json",OLD_SCORE))
    need(checked["support_verified"] and checked["independent_replay"] and not checked["target_labels_opened"]
         and previous["target_labels_opened"] and previous["prior_verify_sha256"]==OLD_VERIFY,
         "Closed original historical SAR-SSP capture, replay and score")
    var_result=json.loads(pin(VAR/"render/RESULT_SAFE.json",SOURCE_RENDER))
    term=json.loads((VAR/"RENDER_TERMINAL.json").read_bytes())
    source_score=json.loads(pin(SOURCE/"score/RESULT_SAFE.json",SOURCE_SCORE))
    score_term=json.loads((SOURCE/"SCORE_TERMINAL.json").read_bytes())
    need(term["actual_exit"]==score_term["actual_exit"]==0
         and term["output_footer_verified"] is score_term["output_footer_verified"] is True
         and term["result_sha256"]==SOURCE_RENDER and term["worker_sha256"]==SOURCE_RENDER_WORKER
         and score_term["result_sha256"]==SOURCE_SCORE and var_result["worker_sha256"]==SOURCE_RENDER_WORKER,
         "Closed exact source writer and previously observed scoring")
    policy=load(VAR/"variable_writer.py",WRITER,"_source_variable_policy")
    return old,m,args,roster,capture,checked,previous,scoring,policy


def check_side(stock,coords,action,k):
    import numpy as np

    flat=stock.ravel(order="C")
    need(coords.ndim==2 and coords.shape[1]==2 and len(np.unique(coords))==coords.size
         and coords.dtype.kind in "iu" and np.all((0<=coords)&(coords<flat.size))
         and not flat[coords[:,0]].any() and flat[coords[:,1]].all()
         and len(action)==2 and sum(map(len,action))==2*k
         and all(len(side)==len(set(side)) and all(0<=v<len(coords) for v in side) for side in action),
         "Exact per-candidate native pixel positions and budget")


def static_pair(a):
    return tuple(tuple(sorted(int(x) for x in side)) for side in zip(*a,strict=True))


def load_fit():
    import torch

    family=load(SIGNED/"render_gate.py",SIGNED_SHA,"_closed_first_fit_writer")
    original=load(BASE_RENDER,family.RENDER_WORKER,"_first_gate_reader")
    core,models,center,scale,fit,_=family.fitted(SimpleNamespace(fit=FIT,fit_sha=FIT_SHA,core=CORE),original)
    need(fit["train_audit"]["heads"]["bce"]["final_tensor_sha256"]==ARCH_BCE,
         "Exact first source BCE tensor fingerprint")
    selector=load(SELECTOR,family.SELECTOR,"_original_fixed_intersection_policy")
    attraction=load(CODE/"FIT109_CLASSICAL_ATTRACTION_20261121/spatial_attraction.py",
                    family.ATTRACTION,"_original_same_budget_attraction")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    return core,models["bce"],center,scale,selector,attraction


def render(a):
    import numpy as np
    from pycocotools import mask as codec

    started=perf_counter()
    old,m,args,roster,capture,checked,_,scoring,policy=setup()
    core,model,center,scale,selector,attraction=load_fit()
    feature_source,rank_source=m.native.load_old(args.pair_math,args.rank_math)
    adapted,probability=m.native.make_native(feature_source,rank_source,500,500)
    archived=scoring.read_stream(OLD/"prepare/EDITS_PRIVATE.jsonl",checked["edits_sha256"])
    counts,arm_counts=Counter(),{arm:Counter() for arm in NEW_ARMS}
    stream_hash=sha256()
    a.output.mkdir()
    with (a.output/"RENDER_PRIVATE.jsonl").open("xb") as stream:
        for ordinal,(item,coco,audit,planes) in enumerate(m.legacy.blocks(args,m.author,roster,capture,True),1):
            assert len(coco)==len(audit)==len(planes)==100
            previous=archived[ordinal-1]
            image=m.legacy.rgb(args,item)
            need(previous["image_id"]==item["image_id"]
                 and digest(image.tobytes(order="C"))==previous["rgb_sha256"]==item["rgb_sha256"],
                 "Exact prior and current native decoded image")
            gray=feature_source.image_features(image)
            eligible=[]
            for r,(c,d,plane) in enumerate(zip(coco,audit,planes,strict=True),1):
                counts["candidates"]+=1
                previous_edit=previous["changes"].get(str(r),{})
                k=int(previous_edit.get("attraction_static_at_dynamic_count",{}).get("swaps",0))
                area=d["mask_area"]
                if not 32<=area<1024:
                    need(k==0,"Original complete no-op out of area")
                    continue
                counts["area_eligible"]+=1
                stock=codec.decode(c["segmentation"]).astype(bool,order="C")
                received=adapted.paired_points(stock,d["box_xyxy"],area,c["score"],plane,gray,probability)
                if received is None:
                    counts["zero_support"]+=1
                    need(k==0,"Original complete no-op unsupported")
                    continue
                coords,f,_,discord,half=received
                counts.update(supported=1,pairs=len(coords),threshold_discordant=discord,threshold_half=half)
                dynamic,static=attraction.select(stock,coords,min(8,area//10,len(coords)))
                need(len(dynamic)==len(static)==k,"Exact archived dynamic attraction quota")
                if not k:
                    continue
                incumbent=static_pair(static)
                for mode,picks in (("attraction_dynamic",dynamic),("attraction_static_at_dynamic_count",static)):
                    frozen,_=policy.write_mask(stock,coords,static_pair(picks),k,codec)
                    need(frozen==previous_edit[mode]["rle"] and previous_edit[mode]["swaps"]==k,
                         "Exact original dynamic/static RLE")
                eligible.append((r,k,area,stock,coords,f,incumbent))
            sizes=[len(row[4]) for row in eligible]
            predictions=core.predict(model,np.concatenate([row[5] for row in eligible],axis=0),center,scale) if sizes else []
            cursor=0
            results={}
            for n,(rank,k,area,stock,coords,f,incumbent) in zip(sizes,eligible,strict=True):
                p=predictions[cursor:cursor+n]
                cursor+=n
                selected=policy.select(p,incumbent,k,area)
                legacy=selector.select(p[:,0,:],p[:,1,:],*incumbent,k=k)
                choices={arm:selected[arm]["action"] for arm in NEW_ARMS[:-1]}
                choices["bce_fixed_robust"]=legacy["robust"]
                edits={}
                for arm,action in choices.items():
                    rle,(adds,drops)=policy.write_mask(stock,coords,action,k,codec)
                    need(adds+drops==2*k and (arm not in ("fixed_iou","bce_fixed_robust") or adds==drops==k),
                         "Exact signed and total per-candidate counts")
                    check_side(stock,coords,action,k)
                    edits[arm]=dict(rle=rle,action=[list(side) for side in action],counts=[adds,drops])
                    arm_counts[arm].update(adds=adds,drops=drops,expanded=int(adds>drops),shrank=int(adds<drops),
                                           balanced=int(adds==drops))
                results[str(rank)]=dict(k=k,coordinates=coords.tolist(),arms=edits)
                counts.update(edited=1,k=k,actions=selected["menu_actions"])
            counts["images"]+=1
            raw=line(dict(image_id=item["image_id"],rgb_sha256=item["rgb_sha256"],edits=results))
            need(stream.write(raw)==len(raw),"Complete private render output")
            stream_hash.update(raw)
            if ordinal%50==0 or ordinal==232:
                print(json.dumps(dict(images=ordinal,support=counts["supported"])),flush=True)
        stream.flush()
        os.fsync(stream.fileno())
    for key in ("area_eligible","candidates","pairs","supported","threshold_discordant","threshold_half","zero_support"):
        need(counts[key]==checked["counts"][key],"All previous exact original point coverage")
    need(counts["images"]==232 and counts["edited"]==10393 and counts["k"]==73559,
         "Exact fixed opportunity census")
    finish(a.output,RENDER_STATUS,started,counts=counts,signed_arm_counts=arm_counts,gate_labels_opened=False,
           frozen_capture_sha256=old.CAPTURE,old_replay_verify_sha256=OLD_VERIFY,raw_sha256=stream_hash.hexdigest())


def census(geometry,score):
    main=geometry["comparisons"]["variable_iou__vs__fixed_iou"]["iou_points"]
    changed=geometry["primary_changed_hulls"]>=20 and geometry["primary_changed_groups"]>=8
    return dict(changed_exposure_sufficient=changed,ap_delta=score,geometry_delta=main["delta"],
                geometry_interval=main["percentile95"],
                criteria_pass=bool(changed and score>=.30 and main["delta"]>=.15 and main["percentile95"][0]>0))


def score(a):
    import numpy as np
    from pycocotools import mask as codec

    start=perf_counter()
    old,m,args,roster,capture,checked,previous,scoring,policy=setup()
    worker=digest(Path(__file__).read_bytes())
    proof=json.loads((a.work_root/"observed_render/TERMINAL.json").read_bytes())
    render_sha=proof["result_sha256"]
    bound=json.loads(pin(a.work_root/"render/RESULT_SAFE.json",render_sha))
    need(proof["phase"]=="capture" and proof["actual_exit"]==0 and proof["output_footer_verified"] is True
         and proof["worker_sha256"]==bound["worker_sha256"]==worker and bound["protocol_sha256"]==LOCK
         and bound["status"]==RENDER_STATUS and not bound["gate_labels_opened"],"Exact observed label-free render")
    rendered=scoring.read_stream(a.work_root/"render/RENDER_PRIVATE.jsonl",bound["raw_sha256"])
    archived=scoring.read_stream(OLD/"prepare/EDITS_PRIVATE.jsonl",checked["edits_sha256"])
    records={}
    arm_counts={arm:Counter() for arm in NEW_ARMS}
    points=Counter()
    for j, (item,p,_,_) in enumerate(m.legacy.blocks(args,m.author,roster,capture,False)):
        iid=item["image_id"]
        now,old_row=rendered[j],archived[j]
        need(now["image_id"]==old_row["image_id"]==iid and now["rgb_sha256"]==old_row["rgb_sha256"]==item["rgb_sha256"],
             "Exact native record and private writeback join")
        current=[]
        original=old_row["changes"]
        new=now["edits"]
        valid={rank for rank in original if "attraction_static_at_dynamic_count" in original[rank]}
        need(set(new)==valid and len(p)==100,"All original 100 masks, zero-budget pass through and exact k-positive subset")
        for r,c in enumerate(p,1):
            row=original.get(str(r),{})
            masks={OLD_ARMS[name]:obj["rle"] for name,obj in row.items() if name in OLD_ARMS}
            native=c["segmentation"]
            if str(r) in new:
                f=new[str(r)]
                k=f["k"]
                stock=codec.decode(native).astype(bool,order="C")
                coords=np.asarray(f["coordinates"],dtype=np.int32)
                need(k==row["attraction_static_at_dynamic_count"]["swaps"] and set(f["arms"])==set(NEW_ARMS),
                     "Exact reference quota and arm census")
                for arm,payload in f["arms"].items():
                    action=payload["action"]
                    check_side(stock,coords,action,k)
                    independent=stock.copy(order="C").ravel(order="C")
                    adds,drops=map(len,action)
                    independent[coords[np.asarray(action[0],dtype=np.intp),0]]=True
                    independent[coords[np.asarray(action[1],dtype=np.intp),1]]=False
                    rebuilt=codec.encode(np.asfortranarray(independent.reshape(stock.shape),dtype=np.uint8))
                    rebuilt["counts"]=rebuilt["counts"].decode("ascii")
                    need(rebuilt==payload["rle"] and [adds,drops]==payload["counts"]
                         and scoring.signed_counts(native,rebuilt,int(codec.area(native)))==(adds,drops),
                         "Independent exact reconstructed RLE and actual native edits")
                    masks[arm]=rebuilt
                    arm_counts[arm].update(adds=adds,drops=drops,expanded=int(adds>drops),
                                          shrank=int(adds<drops),balanced=int(adds==drops))
                points.update(edited=1,k=k)
            current.append(dict(native=native,score=c["score"],masks=masks))
        records[iid]=current
    need(len(records)==232 and len(rendered)==len(archived)==232 and sum(map(len,records.values()))==23200
         and points==dict(edited=10393,k=73559) and arm_counts==bound["signed_arm_counts"],
         "All prelabel render writeback and budget independently read back")
    print("All native scores, old controls and new native masks independently verified before labels.",flush=True)
    need(old.file_hash(ARCHIVE)==scoring.ARCHIVE_SHA,"Exact originally indexed author zip")
    names={row["image_id"]:(Path(row["image_ref"]["path"]).name,row["image_ref"]["sha256"]) for row in roster}
    with ZipFile(ARCHIVE) as package:
        truth,groups,edges=scoring.exact_ground_truth(package,names)
    arm_order=("native",*OLD_SCORE_MAP,*NEW_ARMS)
    scoring.ARMS={name:name for name in arm_order[1:]}
    scoring.PRIMARY=("variable_iou","fixed_iou")
    scoring.SECONDARY=("variable_iou","bce_fixed_robust")
    scoring.COMPARISONS=(scoring.PRIMARY,scoring.SECONDARY,("variable_iou","static"),
                         ("variable_iou","dynamic"),("variable_iou","near_iou"),
                         ("variable_iou","legacy_attraction_learned"),("variable_iou","legacy_free_matched"),
                         ("fixed_iou","static"),("legacy_free_matched","legacy_fixed_distance"))
    evaluate=scoring.load_evaluator(CODE/scoring.EVALUATOR_PATH)
    measured=scoring.score(truth,groups,edges,records,evaluate)
    geometry=measured["fixed_native_match_all_truth"]
    metrics=measured["coco_mask_metric_points"]
    need((geometry["hulls"],geometry["groups"],geometry["matched_hulls"],geometry["unmatched_hulls"])
         ==(139,23,88,51),"Exact native association and author denominator parity")
    for arm,archived_arm in dict(OLD_SCORE_MAP,native="native").items():
        old_metrics=previous["coco_mask_metric_points"][archived_arm]
        need(all(metrics[arm][key]==number for key,number in old_metrics.items())
             and all(abs(geometry["arms"][arm][key]-number)<1e-12 for key,number in
                     previous["fixed_native_match_all_truth"]["arms"][archived_arm].items()),
             "Exact original control AP, fixed IoU and FP/FN geometry")
    primary=measured["coco_mask_deltas_points"]["variable_iou__vs__fixed_iou"]["AP"]
    a.output.mkdir()
    finish(a.output,SCORE_STATUS,start,scoring=measured,
           primary=census(geometry,primary),signed_arm_counts=arm_counts,
           original_control_parity=True,render_result_sha256=render_sha,
           frozen_capture_sha256=old.CAPTURE,old_replay_verify_sha256=OLD_VERIFY,gate_labels_opened=True)


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("render","score"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--work-root",type=Path,required=True)
    parser.add_argument("--inputs",type=Path,required=True,
                        help="Private JSON mapping logical artifact names to local paths")
    a=parser.parse_args()
    source_paths=json.loads(a.inputs.read_text(encoding="utf-8"))
    required = {'OLD': 'prior_sar_ssp_run', 'SOURCE': 'source_variable_score_run', 'VAR': 'source_variable_render_run', 'CODE': 'released_code_root', 'INTAKE': 'sar_ssp_intake', 'SIGNED': 'first_signed_action_gate_run', 'BASE_RENDER': 'earlier_gate_render_source', 'FIT': 'first_matched_fit_run', 'CORE': 'occupancy_core_source', 'SELECTOR': 'fixed_count_selector_source', 'ARCHIVE': 'author_zip'}
    if set(source_paths) != set(required.values()):
        parser.error("Inputs JSON must contain the eleven keys documented in README.md")
    for symbol,key in required.items():
        path=Path(source_paths[key])
        if not path.is_absolute():
            path=a.inputs.resolve().parent/path
        globals()[symbol]=path.resolve(strict=True)
    a.work_root=a.work_root.resolve(strict=True)
    a.output=a.output.resolve()
    need(os.environ.get("CUDA_VISIBLE_DEVICES")=="-1" and sys.flags.isolated and sys.dont_write_bytecode
         and a.output==a.work_root/a.phase and not a.output.exists(),"Isolated fresh CPU run")
    pin(HERE/"PRELOCK.md",LOCK)
    def offline(event,_):
        if event.startswith("socket."):
            raise PermissionError("Offline closed transfer")
    sys.addaudithook(offline)
    sys.path.insert(0,str(OLD/"numpy_1_26_4_only"))
    sys.path.insert(0,str(OLD/"capture/CODEC_PRIVATE"))
    (render if a.phase=="render" else score)(a)


if __name__=="__main__":
    main()



