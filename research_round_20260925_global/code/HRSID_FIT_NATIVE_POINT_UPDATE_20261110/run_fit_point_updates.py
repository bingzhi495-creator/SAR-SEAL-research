"""Equal-budget updates to RF-DETR's native final mask query projection."""
import argparse
from collections import defaultdict
from copy import deepcopy
from hashlib import sha256
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from statistics import fmean
from time import perf_counter
from types import ModuleType
from zipfile import ZipFile
import json
import math
import os
import sys


PROBE_SHA = "7483c319344f4d57ac69720f0f3e19c489984650572985da813f82d104846810"
MATH_SHA = "0821d2cc3b6ca793d33fdfe35b634f3678aad072fc55b86352fa2501e969fc7e"
TRAIN = ((0, 2), (1, 5), (2, 9), (3, 12), (4, 16), (6, 26), (8, 32), (9, 36))
HELD = ((10, 43, 40), (12, 48, 51), (13, 53, 52), (15, 61, 60),
        (16, 64, 65), (18, 75, 72), (20, 80, 81), (21, 86, 84))
ARMS = ("phase", "uncertainty", "random")
ALL = ("anchor", *ARMS)
METRICS = ("polygon_iou", "raster_iou", "center_iou", "ring_bce")
SIZE = (576, 576)


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def module(path, expected, name):
    raw = path.read_bytes()
    require(path.resolve(strict=True) == path and sha256(raw).hexdigest() == expected, "Pinned source differs: " + path.name)
    item = ModuleType(name)
    item.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), item.__dict__)
    return item


def compare_held(scores):
    require(set(scores) == set(ALL) and all(len(scores[arm]) == len(HELD) for arm in ALL), "Exact held source roster")
    means = {arm: {metric: fmean(x[metric] for x in scores[arm]) for metric in METRICS} for arm in ALL}
    deltas = {}
    for other in ("uncertainty", "random", "anchor"):
        values = [a["polygon_iou"]-b["polygon_iou"] for a,b in zip(scores["phase"],scores[other],strict=True)]
        deltas[other] = dict(paired_polygon_iou=values, mean=fmean(values), positive=sum(v > 0 for v in values))
    guard = means["phase"]["center_iou"]-max(means[arm]["center_iou"] for arm in ("anchor","uncertainty","random"))
    return dict(means=means, phase_minus=deltas, phase_center_guard=guard,
                descriptive_gate=all(deltas[arm]["mean"] >= .005 and deltas[arm]["positive"] >= 5
                                     for arm in ("uncertainty","random"))
                                 and deltas["anchor"]["mean"] >= 0 and guard >= -.002)


def param_digest(projection):
    from cached_final_query import projection_parameters
    return sha256(b"".join(x.detach().numpy().tobytes() for x in projection_parameters(projection))).hexdigest()


def prepare_views(args, probe, pilot, native, np, torch, model, config, matcher):
    from cached_final_query import capture_final_query, sample_integer_cells
    from common_mask_supervision import ROW_ID
    from point_occupancy import bitmap_cell_occupancy
    from point_update_loss import freeze_row_banks
    from rfdetr.utilities.tensors import nested_tensor_from_tensor_list
    from vector_label_geometry import UnsupportedPolygon, final_polygons, polygon_raster, polygon_rings

    audit = native.NativeInputAudit()
    roster = [(o,c,"crop",True) for o,c in TRAIN]
    roster += [(o,c,p,False) for o,crop,direct in HELD for c,p in ((crop,"crop"),(direct,"direct"))]
    views = []
    with args.bundle.open("rb") as stream:
        identity = pilot.fingerprint(os.fstat(stream.fileno()))
        digest = sha256()
        for block in iter(lambda: stream.read(1024**2),b""):
            digest.update(block)
        require(identity[2] == pilot.BUNDLE_BYTES and digest.hexdigest() == pilot.BUNDLE_SHA, "Exact sealed FIT-only packet")
        stream.seek(0)
        with ZipFile(stream) as bundle, torch.no_grad():
            cohort = json.loads(pilot.pinned_member(bundle,"COHORT_PRIVATE.json",pilot.COHORT_SHA,1024**2))
            index = json.loads(pilot.pinned_member(bundle,"FIT114_COCO_PRIVATE.json",pilot.SUBSET_SHA,4*1024**2))
            history = [json.loads(line) for line in pilot.pinned_member(
                bundle,"HISTORICAL_VIEWS_PRIVATE.jsonl",pilot.HISTORY_SHA,1024**2).splitlines()]
            images = cohort["images"]
            names = {row["id"]:row["file_name"] for row in index["images"]}
            labels = defaultdict(list)
            for annotation in index["annotations"]:
                require(annotation["image_id"] in names, "Exact sealed FIT-only annotation parent")
                labels[annotation["image_id"]].append(annotation)
            selected = [images[o] for o in [r[0] for r in TRAIN]+[r[0] for r in HELD]]
            require(cohort["fit_only"] is True and len(images)==len(names)==114 and len(history)==456
                    and len({x["group"] for x in selected}) == len({x["encoded_sha256"] for x in selected}) == 16,
                    "Disjoint fixed source groups and bytes required")
            for ordinal,cursor,path,train_view in roster:
                record,old = images[ordinal],history[cursor]
                ident,name = record["image_id"],record["native_basename"]
                require(record["core"] and names[ident]==name and old["ordinal"]==ordinal
                        and old["path"]==path and old["group"]==record["group"] and old["seed"]==pilot.BASE_SEED+cursor,
                        "Exact selected historical view")
                png=pilot.pinned_member(bundle,"images/"+name,record["encoded_sha256"],4*1024**2)
                image,scales=native.io_utils.decode_pil_image_bytes(png,draft_size=None)
                require(image.size==(800,800) and image.mode=="RGB" and scales==(1.,1.)
                        and len(png)==record["encoded_bytes"]
                        and pilot.digest(np.asarray(image).tobytes())==record["decoded_rgb_sha256"], "Exact decoded FIT image")
                image,target=audit.convert(image,dict(image_id=ident,annotations=labels[ident]))
                source=pilot.source_annotations(labels[ident],target,(800,800),torch)
                pixels,original,b,c,steps=probe.fixed_view(native,audit,image,target,old,torch)
                require(tuple(pixels.shape)==(3,*SIZE),"Exact historical model canvas")
                sample=nested_tensor_from_tensor_list([pixels],block_size=config.patch_size*config.num_windows)
                with capture_final_query(model.segmentation_head) as captured:
                    raw=model(sample)
                cache=captured.seal(raw["pred_masks"])
                torch.manual_seed(252001000+cursor)
                [(queries,positions)]=matcher(raw,[c],group_detr=1)
                matched=dict(zip(positions.tolist(),queries.tolist(),strict=True))
                ids=c[ROW_ID].tolist()
                require(set(matched)==set(range(len(ids))) and len(set(matched.values()))==len(ids),"Fixed native assignment")
                rows=[]
                for pos,row_id in enumerate(ids):
                    annotation=source[row_id]
                    if float(annotation["area"])>=1024:
                        continue
                    hard=c["masks"][pos].numpy()
                    bitmap=original["masks"][row_id].numpy()
                    segmentation=annotation["segmentation"]
                    try:
                        rings,hw=final_polygons(segmentation,(800,800),steps)
                        require(hw==SIZE and np.array_equal(bitmap,polygon_raster(polygon_rings(segmentation),(800,800))),
                                "Exact original polygon and stock bitmap")
                    except UnsupportedPolygon:
                        continue
                    cells,strata=probe.boundary_bank(hard)
                    raster=bitmap_cell_occupancy(bitmap,steps,cells)
                    q=matched[pos]
                    row=dict(query=q,cells=cells,raster=raster,center=hard,rings=rings,bitmap=bitmap)
                    if train_view:
                        x,y=cells.T
                        z=sample_integer_cells(raw["pred_masks"][0,q:q+1],cells,SIZE)[0].numpy()
                        row["banks"]=freeze_row_banks(hard[y,x],b["masks"][pos].numpy()[y,x],raster,z,strata,
                                                     252010000+1000*cursor+row_id)
                    rows.append(row)
                require(rows,"A fixed view has no eligible small rows")
                views.append(dict(ordinal=ordinal,cursor=cursor,path=path,train=train_view,steps=steps,cache=cache,rows=rows))
                print(json.dumps(dict(event="P253_CLOSED_CACHE_PROGRESS",views=len(views),rows=len(rows))),flush=True)
        require(pilot.fingerprint(os.fstat(stream.fileno()))==pilot.fingerprint(args.bundle.stat())==identity,
                "Sealed FIT-only input remained exact")
    require(len(views)==24 and [sum(len(v["rows"]) for v in views if match(v)) for match in (
        lambda v:v["train"],lambda v:not v["train"] and v["path"]=="crop",
        lambda v:not v["train"] and v["path"]=="direct")] == [31,8,9],"Exact 31/8/9 eligible row roster")
    return views


def train_one(arm, original, views, torch, output):
    from cached_final_query import matched_dense_logits, projection_parameters, sample_integer_cells
    from point_update_loss import paired_point_loss

    start=perf_counter()
    head=deepcopy(original).requires_grad_(True)
    params=projection_parameters(head)
    before=param_digest(head)
    require(before==param_digest(original) and sum(x.numel() for x in params)==65792,"Exact native projection initialization")
    opt=torch.optim.AdamW(params,lr=1e-4,weight_decay=1e-4,foreach=False,fused=False)
    events=[]
    for view in views:
        rows=view["rows"]
        opt.zero_grad(set_to_none=True)
        dense=matched_dense_logits(head,view["cache"],[x["query"] for x in rows])
        selected=[]
        for i,row in enumerate(rows):
            bank=row["banks"][arm]
            z=sample_integer_cells(dense[i:i+1],row["cells"][bank.indices],SIZE)[0]
            selected.append((z,torch.tensor(bank.targets.tolist(),dtype=z.dtype)))
        loss=paired_point_loss(selected,bce_weight=1.,dice_weight=0.)
        require(bool(torch.isfinite(loss)),"Finite controlled point loss")
        loss.backward()
        require(all(x.grad is not None and bool(torch.isfinite(x.grad).all()) for x in params),"Finite original-head gradients")
        opt.step()
        events.append(dict(cursor=view["cursor"],rows=len(rows),points=sum(x.numel() for x,_ in selected),
                           loss=float(loss.detach())))
    require(len(events)==8 and all(int(opt.state[x]["step"])==8 for x in params)
            and all(bool(torch.isfinite(x).all()) for x in params) and param_digest(head)!=before,
            "Eight real updates and changed finite native projection required")
    path=output/("PROJECTION_"+arm.upper()+"_PRIVATE.pt")
    with path.open("xb") as stream:
        torch.save(head.state_dict(),stream)
        stream.flush()
        os.fsync(stream.fileno())
    return head,dict(optimizer_updates=8,events=events,initial_sha256=before,parameter_sha256=param_digest(head),
                     private_archive_sha256=sha256(path.read_bytes()).hexdigest(),seconds=perf_counter()-start)


def held_readback(projections, views, np, torch, vector_math):
    from cached_final_query import matched_dense_logits, sample_integer_cells
    from reference_resize import _compose
    from ssdd_pixel_footprint import measure_masks
    from vector_label_geometry import binary_iou

    result={path:{arm:[] for arm in ALL} for path in ("crop","direct")}
    with torch.no_grad():
        for view in views:
            rows=view["rows"]
            values={arm:[] for arm in ALL}
            dense={arm:matched_dense_logits(projections[arm],view["cache"],[x["query"] for x in rows]) for arm in ALL}
            rasterized={arm:(torch.nn.functional.interpolate(z[None],size=SIZE,mode="bilinear",
                                                              align_corners=False)[0]>0).numpy() for arm,z in dense.items()}
            for i,row in enumerate(rows):
                masks={arm:array[i] for arm,array in rasterized.items()}
                p=vector_math.run_footprint_metrics(row["rings"],SIZE,masks)["by_arm"]
                r=measure_masks(row["bitmap"][None],{arm:mask[None] for arm,mask in masks.items()},
                                np.asarray([True],dtype=np.bool_),view["steps"],_compose)["by_arm"]
                for arm in ALL:
                    z=sample_integer_cells(dense[arm][i:i+1],row["cells"],SIZE)[0]
                    target=torch.as_tensor(row["raster"],dtype=z.dtype)
                    loss=torch.nn.functional.binary_cross_entropy_with_logits(z,target)
                    item=dict(polygon_iou=float(p[arm]["iou"]),raster_iou=float(r[arm]["iou"][0]),
                              center_iou=float(binary_iou(masks[arm],row["center"])),ring_bce=float(loss))
                    require(all(math.isfinite(x) for x in item.values()),"Finite fixed held geometry")
                    values[arm].append(item)
            for arm in ALL:
                result[view["path"]][arm].append({key:fmean(x[key] for x in values[arm]) for key in METRICS})
    return result


def run(args):
    start=perf_counter()
    require(sys.version_info[:3]==(3,10,13) and sys.flags.isolated==sys.flags.utf8_mode==1 and sys.dont_write_bytecode
            and os.environ.get("CUDA_VISIBLE_DEVICES")=="-1","Pinned isolated Windows CPU runtime")
    for path in (args.code,args.bundle,args.checkpoint,args.receipts,args.shapely_site):
        require(path.is_absolute() and path.resolve(strict=True)==path,"Canonical sealed input")
    require(args.output.is_absolute() and args.output.parent.resolve(strict=True)==args.output.parent
            and not args.output.exists() and not args.output.is_symlink(),"Fresh private output")
    code=args.code
    probe_dir=code/"HRSID_FIT_FROZEN_POINT_PROBE_20261110"
    vector_dir=code/"HRSID_FIT114_VECTOR_LABEL_GEOMETRY_20261110"
    probe=module(probe_dir/"run_fit_point_probe.py",PROBE_SHA,"_p253_pinned_point_probe")
    pilot=module(vector_dir/"run_fit114_vector_trial.py",probe.PILOT_SHA,"_p253_pinned_geometry_pilot")
    native,np,torch,_=pilot.install_support(code,args.shapely_site)
    sys.path[:0]=[str(Path(__file__).resolve().parent),str(probe_dir)]
    from cached_final_query import projection_parameters
    from rfdetr.models.lwdetr import build_criterion_from_config

    vector_math=module(vector_dir/"vector_population_math.py",MATH_SHA,"_p253_held_vector_math")
    base=probe.accepted_phase_evaluator(code,args.receipts)
    base.RECIPE=code.parent/"configs/RFDETR_SEG_NANO_BASELINE_576_E100_20261107.yaml"
    base.FIT_AUDIT=code/"HRSID_OFFICIAL_PNG_FIT_AUDIT_20261109/hrsid_png_fit_audit.py"
    for name,expected in dict(base.MODEL_SOURCES,**probe.EXTRA_RUNTIME).items():
        require(sha256(Path(import_module(name).__file__).read_bytes()).hexdigest()==expected,"Pinned native runtime: "+name)
    require(version("scipy")=="1.15.3","Exact frozen CPU assignment backend")
    model,config,train,_=base.load_model(args.checkpoint,probe.CHECKPOINT_SHA,torch,args.output.parent,probe.TRAIN_PROTOCOL)
    model.eval().requires_grad_(False)
    criterion,_=build_criterion_from_config(config,train)
    original=model.segmentation_head.query_features_proj
    projection_parameters(original)
    before=param_digest(original)
    args.output.mkdir()
    views=prepare_views(args,probe,pilot,native,np,torch,model,config,criterion.matcher)
    projections={"anchor":original}
    train_views=[v for v in views if v["train"]]
    arm_reports={}
    for arm in ARMS:
        projections[arm],arm_reports[arm]=train_one(arm,original,train_views,torch,args.output)
        print(json.dumps(dict(event="P253_REAL_NATIVE_PROJECTION_UPDATES",arm=arm,updates=8)),flush=True)
    ledgers=[[{k:e[k] for k in ("cursor","rows","points")} for e in arm_reports[arm]["events"]] for arm in ARMS]
    require(ledgers[0]==ledgers[1]==ledgers[2] and sum(e["points"] for e in ledgers[0])==487
            and param_digest(original)==before
            and all(not p.requires_grad and p.grad is None for p in model.parameters())
            and not torch.cuda.is_initialized(),"Equal realized update budget and unchanged frozen carrier")
    measured=held_readback(projections,[v for v in views if not v["train"]],np,torch,vector_math)
    result=dict(status="P253_FIT_POINT_NATIVE_PROJECTION_UPDATES_COMPLETE",FIT=True,DEV=False,TEST=False,AP=False,
                optimizer_held_out_seen_by_frozen_source=True,model_forward_calls=len(views),frozen_match_calls=len(views),
                updated_parameters_per_arm=65792,initial_projection_sha256=before,
                source_checkpoint_sha256=probe.CHECKPOINT_SHA,fit_bundle_sha256=pilot.BUNDLE_SHA,train_roster=TRAIN,
                held_roster=HELD,arms=arm_reports,held_metrics=measured,
                held_comparison={path:compare_held(score) for path,score in measured.items()},wall_seconds=perf_counter()-start)
    raw=(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    with (args.output/"RESULT_SAFE.json").open("xb") as stream:
        require(stream.write(raw)==len(raw),"Complete update report")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(status=result["status"],result_sha256=sha256(raw).hexdigest(),forward_calls=len(views),
                          updates_per_arm=8,wall_seconds=result["wall_seconds"])),flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ("code","bundle","checkpoint","receipts","shapely-site","output"):
        cli.add_argument("--"+name,type=Path,required=True)
    run(cli.parse_args())
