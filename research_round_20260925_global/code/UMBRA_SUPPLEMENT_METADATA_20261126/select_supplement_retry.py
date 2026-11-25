"""Witness a fresh supplemental Umbra roster using only author metadata."""
from argparse import ArgumentParser
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from itertools import combinations
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType
from urllib.error import HTTPError, URLError
from xml.etree.ElementTree import ParseError

HERE=Path(__file__).resolve().parent
PROTOCOL="df1e5cf4cf299917096e3ca9ef65f5740b324775be272aa7f77441e05c6fc059"
PRIOR="f635a1370a15dd9e9d058c73d878d49ff8cc372a3cd9a4080b3b68c9fc9d2bf5"
U12_MANIFEST="24d08f3bf1ecb08212f78e5a3cfa8c709ef9a2194e9d345f5cb04a1f9f48c29e"
U12_RESULT="6d2a7b4b248fce5384ef021545ab5ef9f244fd9510ac59249967f94d2575c751"
U12_TERMINAL="f90cc16528dbacdff260e14290a7ce691a407015b0fe1398e3bdc714c91efbb6"


def need(ok,why):
    if not ok:
        raise ValueError(why)


def sha(b):
    return sha256(b).hexdigest()


def pin(p,wanted):
    raw=p.read_bytes()
    need(sha(raw)==wanted,"Exact historic source or closure: "+p.name)
    return raw


def base_source(path):
    prior=ModuleType("umbra_original_metadata")
    prior.__file__=str(path)
    exec(compile(pin(path,PRIOR),str(path),"exec"),vars(prior))
    return prior


def midpoint_probe(tasks,excluded,cap):
    remaining=[task for task in tasks if task not in excluded]
    count=min(len(remaining),cap)
    return [remaining[((2*i+1)*len(remaining))//(2*count)] for i in range(count)]


def select(ordered,excluded_centers,excluded_footprints,distance,quota=12):
    decisions=Counter()
    retained=[]
    for row,footprint in ordered:
        if any(distance(row["center_lon_lat"],old)<100 for old in excluded_centers):
            decisions["near_prior_center"]+=1
        elif any(footprint.intersects(old) for old in excluded_footprints):
            decisions["intersects_prior_footprint"]+=1
        else:
            retained.append((row,footprint))
    selected,footprints=[],[]
    used=set()

    def seek(records,count):
        for row,footprint in records:
            if count==0:
                break
            collection=row["collect_id"]
            if collection in used or any(distance(row["center_lon_lat"],old["center_lon_lat"])<100 for old in selected):
                continue
            if any(footprint.intersects(other) for other in footprints):
                continue
            selected.append(dict(row,window_id=f"UMB-{101+len(selected):03d}"))
            footprints.append(footprint)
            used.add(collection)
            count-=1

    seek((x for x in retained if x[0]["target_ipr_m"]>=1.),quota)
    seek((x for x in retained if x[0]["target_ipr_m"]<1.),quota)
    seek(retained,2*quota-len(selected))
    decisions["passed_prior_spatial_exclusions"]=len(retained)
    decisions["selected"]=len(selected)
    decisions["retained_but_not_selected"]=len(retained)-len(selected)
    need(len({r["task_id"] for r in selected})==len(selected) and not any(a.intersects(b) for a,b in combinations(footprints,2)),
         "Exact unique tasks and nonintersecting selected footprints")
    return selected,dict(decisions)


def acquire(args,prior):
    started=perf_counter()
    proof=json.loads(pin(args.original_u12/"TERMINAL.json",U12_TERMINAL))
    latest=json.loads(pin(args.original_u12/"RESULT_SAFE.json",U12_RESULT))
    need(proof["phase"]=="metadata-only-cohort" and proof["actual_worker_exit"]==0 and proof["worker_sha256"]==PRIOR
         and proof["safe_result_sha256"]==U12_RESULT and proof["observed_status"]==latest["status"]=="planned_12" and proof["cohort_private_sha256"]==latest["cohort_private_sha256"]==U12_MANIFEST
         and latest["image_reads"]==latest["label_reads"]==latest["model_forwards"]==0,
         "Original twelve-window metadata was separately locked and closed")
    old=json.loads(pin(args.original_u12/"COHORT_PRIVATE.json",U12_MANIFEST))
    need(len(old["items"])==12,"Original U12 exclusion roster")
    old_tasks,old_collects,centers=prior.historic(args.historical_manifest,args.historical_geometry)
    entries=prior.pinned_json(args.historical_geometry,prior.GEOMETRY_SHA)
    old_footprints=[prior.checked_polygon(row["geometry"]) for row in entries["items"]]
    for row in old["items"]:
        old_tasks.add(prior.uuid_text(row["task_id"]))
        old_collects.add(prior.uuid_text(row["collect_id"]))
        centers.append(prior.point(row["center_lon_lat"]))
        old_footprints.append(prior.checked_polygon(dict(type=row["footprint_type"],coordinates=row["footprint_lon_lat"])))
    transport=prior.Transport()
    for receipt_path,pilot_sha in zip(args.pilot_result,prior.PILOT_SHAS,strict=True):
        receipt=prior.pinned_json(receipt_path,pilot_sha)
        need(receipt["status"]=="NATIVE_TECHNICAL_PILOT_COMPLETE" and len(receipt["products"])==1,"Closed pilot receipt")
        uri=receipt["products"][0]["author_url"]
        need(uri.startswith(prior.BASE+prior.PREFIX) and uri.endswith("_GEC.tif"),"Official earlier pixel pilot metadata")
        doc=json.loads(transport.get(uri[:-len("_GEC.tif")]+"_METADATA.json",20*1024))
        need(len(doc["collects"])==1,"Exact prior pilot collect")
        item=doc["collects"][0]
        tid=prior.uuid_text(item["taskId"])
        need(tid==prior.uuid_text(uri.split(prior.PREFIX,1)[1].split("/",1)[0]),"Earlier pilot task identity")
        old_tasks.add(tid)
        old_collects.add(prior.uuid_text(item["id"]))
        centers.append(prior.point(item["sceneCenterPointLla"]["coordinates"]))
        old_footprints.append(prior.checked_polygon(item["footprintPolygonLla"]))
    need(len(centers)==len(old_footprints)==135,"Historical, pilot and original U12 geographic exclusions")
    tasks,page_hashes=prior.all_tasks(transport)
    task_digest=sha(prior.packed(tasks))
    matches_earlier=task_digest==latest["ordered_task_list_sha256"]
    metadata_exclusion=set()
    if matches_earlier:
        reconstructed=midpoint_probe(tasks,set(),64)
        need(sha(prior.packed(reconstructed))==latest["ordered_sample_task_list_sha256"],"Exact prior metadata probes")
        metadata_exclusion.update(reconstructed)
    chosen=midpoint_probe(tasks,old_tasks|metadata_exclusion,128)

    def observe(task):
        try:
            row,reason=prior.candidate(transport,task)
            if row is None:
                return None,reason
            if row["collect_id"] in old_collects:
                return None,"historical_collect"
            dims=(row["declared_gec_rows"],row["declared_gec_cols"])
            if any(type(n) not in (int,type(None)) or isinstance(n,int) and n<1024 for n in dims):
                return None,"native_window_dimensions"
            return row,"qualified"
        except (HTTPError,URLError,TimeoutError):
            return None,"author_metadata_transport_failed"
        except (TypeError,ValueError,KeyError,ParseError):
            return None,"invalid_author_metadata"

    with ThreadPoolExecutor(max_workers=10) as pool:
        seen=list(pool.map(observe,chosen))
    counts=Counter(note for _,note in seen)
    qualified,ids=[],set()
    for row,_ in seen:
        if row is None:
            continue
        if row["collect_id"] in ids:
            counts["duplicate_collect_among_qualified"]+=1
            continue
        ids.add(row["collect_id"])
        polygon=prior.checked_polygon(dict(type=row["footprint_type"],coordinates=row["footprint_lon_lat"]))
        qualified.append((row,polygon))
    result,spatial=select(qualified,centers,old_footprints,prior.distance)
    a_ready=18<=len(result)<=24
    manifest=dict(schema="umbra_gec_supplement_v1",items=result,source_snapshot_sha256=task_digest,
           probe_order_sha256=sha(prior.packed(chosen)),selection_protocol_sha256=PROTOCOL,original_u12_sha256=U12_MANIFEST)
    body=prior.packed(manifest)
    geo=min((prior.distance(r["center_lon_lat"],old) for r in result for old in centers),default=None)
    separation=min((prior.distance(a["center_lon_lat"],b["center_lon_lat"]) for a,b in combinations(result,2)),default=None)
    safe=dict(status="UMBRA_SUPPLEMENT_METADATA_LOCKED" if a_ready else "UMBRA_SUPPLEMENT_METADATA_FEASIBILITY_ONLY",
         protocol_sha256=PROTOCOL,worker_sha256=sha(Path(__file__).read_bytes()),author_metadata_helper_sha256=PRIOR,
         cohort_private_sha256=sha(body),paired_original_request_sha256=U12_MANIFEST,cohort_snapshot_sha256=task_digest,
         provider_listing_page_sha256=page_hashes,provider_task_count=len(tasks),sample_order_sha256=manifest["probe_order_sha256"],
         known_prior_tasks_in_catalog=len(set(tasks)&old_tasks),documented_prior_locations=len(centers),
         catalog_matches_prior=matches_earlier,old_metadata_probe_exclusion_count=len(metadata_exclusion),
         sampled_task_count=len(chosen),qualification=dict(counts),unique_qualified_collects=len(qualified),
         spatial_filter_decisions=spatial,selected_scene_count=len(result),
         selected_bands=dict(Counter(r["class_group"] for r in result)),
         selected_unique_tasks=len({r["task_id"] for r in result}),selected_unique_collects=len({r["collect_id"] for r in result}),
         minimum_new_to_prior_km=geo,minimum_new_pairwise_km=separation,selected_footprint_intersections=0,
         eligible_for_pixel_stage_after_public_witness=a_ready,source_metadata_response_count=transport.count,
         source_metadata_bytes=transport.charged,image_reads=0,label_reads=0,model_forwards=0,
         elapsed_seconds=perf_counter()-started)
    args.output.mkdir()
    for name,data in (("COHORT_PRIVATE.json",body),("RESULT_SAFE.json",prior.packed(safe))):
        with (args.output/name).open("xb") as out:
            out.write(data)
            out.flush()
            os.fsync(out.fileno())
    print(json.dumps(dict(status=safe["status"],result_sha256=sha(prior.packed(safe)))),flush=True)


def main():
    p=ArgumentParser(description=__doc__)
    for name in ("helper","original-u12","historical-manifest","historical-geometry","output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--pilot-result",type=Path,required=True,action="append")
    a=p.parse_args()
    need(sys.flags.isolated and sys.dont_write_bytecode and len(a.pilot_result)==3 and not a.output.exists(),
         "Isolated metadata coordinator with the three prior pilots and a fresh output")
    pin(HERE/"PRELOCK.md",PROTOCOL)
    acquire(a,base_source(a.helper))

if __name__=="__main__":
    main()

