"""Compare author vector contours for byte-identical repeated SSDD FIT images."""

from argparse import ArgumentParser
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import combinations
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from zipfile import ZipFile

ARCHIVE_SHA = "1c8f9b75e79fbc02c983ccd283a70082fcd5ab0d6538c777e8b286dbfdb78100"
MANIFEST_SHA = "5bf021523d1ba3507a5ef7ba6a731da8699c37eb32d6b8ea962c52dcfbaafb47"
INTEGER_SHA = "62966857c78cea1e3b060700c62a7b896942686c2e674d3eb09f516a92250bc1"
FLOAT_SHA = "97162d23a33b9387fe0b47a117f7e5b1f9dde7471033f244ab336cee1ca9039b"

def need(condition, why):
    if not condition:
        raise ValueError(why)

def check(data, digest):
    need(sha256(data).hexdigest()==digest, "Closed FIT bytes differ")
    return data

def iou(a,b):
    cross=a.intersection(b).area
    return cross/(a.area+b.area-cross)

def clipped_polygon(ann, canvas, shapely):
    rings = ann["segmentation"]
    need(isinstance(rings,list) and len(rings)==1 and len(rings[0])>=6 and len(rings[0])%2==0,
         "Expected one closed source vector contour")
    raw = shapely.Polygon(list(zip(rings[0][::2],rings[0][1::2],strict=True)))
    if not raw.is_valid or raw.area<=0:
        return None
    shape=raw.intersection(canvas)
    return shape if shape.is_valid and shape.area>0 else None

def summarize(records, np):
    result={}
    for name in ("float_iou","integer_iou","aligned_float_iou","within_row_truncation_iou",
                 "repeat_centroid_px","truncation_centroid_px","along_centroid_px","across_centroid_px"):
        values=[row[name] for row in records if row[name] is not None]
        if values:
            result[name]={"mean":float(np.mean(values)),"median":float(np.median(values)),
                          "min":float(min(values)),"max":float(max(values)),"n":len(values)}
    return result

def main():
    parser=ArgumentParser(description=__doc__)
    for name in ("archive","float-annotations","shapely-site","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args()
    started=perf_counter()
    need(os.environ.get("CUDA_VISIBLE_DEVICES")=="-1" and args.output.is_absolute() and not args.output.exists(),
         "Fresh closed FIT CPU evaluation required")
    sys.path.append(str(args.shapely_site))
    import numpy as np
    import shapely
    from scipy.optimize import linear_sum_assignment
    from shapely.affinity import translate
    need(shapely.__version__=="2.0.2" and shapely.geos_version_string=="3.11.2",
         "Use previously qualified exact vector geometry runtime")
    def offline(event,_):
        if event.startswith("socket."):
            raise PermissionError("Offline FIT only")
    sys.addaudithook(offline)
    check(args.archive.read_bytes(),ARCHIVE_SHA)
    with ZipFile(args.archive) as archive:
        manifest=json.loads(check(archive.read("manifest.json"),MANIFEST_SHA))
        integer=json.loads(check(archive.read("annotations/fit.json"),INTEGER_SHA))
    floating=json.loads(check(args.float_annotations.read_bytes(),FLOAT_SHA))
    need(len(integer["images"])==741 and len(integer["annotations"])==1661 and
         integer["images"]==floating["images"] and integer["categories"]==floating["categories"],
         "Exact FIT-only census required")
    for a,b in zip(integer["annotations"],floating["annotations"],strict=True):
        need(a.keys()==b.keys() and all(a[key]==b[key] for key in a if key!="segmentation"),
             "Only source polygons can vary")
    groups=defaultdict(list)
    images={item["id"]:item for item in integer["images"]}
    for row, image in zip(manifest["rows"],integer["images"],strict=True):
        need(row["image_id"]==image["id"] and row["original_basename"]==image["file_name"]
             and row["original_shape"]==[image["height"],image["width"]],"Exact content-to-row bridge required")
        groups[row["original_image_ref"]["sha256"]].append(image["id"])
    need(len(groups)==730 and sum(map(len,groups.values()))==741,"Exact closed encoded content groups required")
    duplicate_groups=[sorted(ids) for _,ids in sorted(groups.items()) if len(ids)>1]
    ids={iid for group in duplicate_groups for iid in group}
    by_image=defaultdict(list)
    counts=Counter()
    for a,b in zip(integer["annotations"],floating["annotations"],strict=True):
        iid=a["image_id"]
        if iid not in ids:
            continue
        image=images[iid]
        canvas=shapely.box(0,0,image["width"],image["height"])
        left,right=clipped_polygon(a,canvas,shapely),clipped_polygon(b,canvas,shapely)
        by_image[iid].append((a["id"],left,right))
        counts["annotation_rows_in_duplicate_groups"]+=1
        counts["invalid_integer_rows"]+=int(left is None)
        counts["invalid_float_rows"]+=int(right is None)

    records=[]
    group_reports=[]
    for g, group in enumerate(duplicate_groups,1):
        group_records=[]
        count=Counter()
        for ia,ib in combinations(group,2):
            a=sorted(by_image[ia],key=lambda row:row[0])
            b=sorted(by_image[ib],key=lambda row:row[0])
            count["image_pairs"]+=1
            count["unequal_annotation_count_pairs"]+=int(len(a)!=len(b))
            av=[row for row in a if row[2] is not None]
            bv=[row for row in b if row[2] is not None]
            if av and bv:
                overlaps=np.asarray([[iou(a0[2],b0[2]) for b0 in bv] for a0 in av])
                feasible=overlaps>=.5
                bonus=1+min(len(av),len(bv))
                lefts,rights=linear_sum_assignment(feasible*(bonus+overlaps),maximize=True)
                matches=[(av[i],bv[j],float(overlaps[i,j])) for i,j in zip(lefts,rights,strict=True) if feasible[i,j]]
            else:
                matches=[]
            count["matched_rows"]+=len(matches)
            count["unmatched_rows"]+=len(a)+len(b)-2*len(matches)
            count["equal_float_annotation_multisets"]+=int(len(matches)==len(a)==len(b)
                and all(abs(value-1)<1e-12 for _,_,value in matches))
            for (aid, ai, af),(bid, bi, bf),overlap in matches:
                dx,dy=af.centroid.x-bf.centroid.x,af.centroid.y-bf.centroid.y
                aligned=iou(af,translate(bf,xoff=dx,yoff=dy))
                need(abs(aligned-iou(translate(af,xoff=-dx,yoff=-dy),bf))<1e-10,
                     "Centroid alignment must be symmetric")
                line=np.asarray(af.minimum_rotated_rectangle.exterior.coords)
                edges=line[1:]-line[:-1]
                lengths=np.linalg.norm(edges,axis=1)
                normal_proj=along_proj=None
                if lengths.max()>=2*lengths.min() and lengths.min()>0:
                    tangent=edges[lengths.argmax()]/lengths.max()
                    along_proj=float(abs(tangent @ np.array([dx,dy])))
                    normal_proj=float(abs(np.array([-tangent[1],tangent[0]]) @ np.array([dx,dy])))
                same=[(x,y) for x,y in ((ai,af),(bi,bf)) if x is not None]
                group_records.append(dict(float_iou=overlap,integer_iou=iou(ai,bi) if ai is not None and bi is not None else None,
                    aligned_float_iou=aligned, repeat_centroid_px=af.centroid.distance(bf.centroid),
                    within_row_truncation_iou=float(np.mean([iou(x,y) for x,y in same])) if same else None,
                    truncation_centroid_px=float(np.mean([x.centroid.distance(y.centroid) for x,y in same])) if same else None,
                    along_centroid_px=along_proj,across_centroid_px=normal_proj))
        counts.update(count)
        records.extend(group_records)
        group_reports.append(dict(group=g,image_count=len(group),counts=dict(count),metrics=summarize(group_records,np)))
    summary=summarize(records,np)
    groups_equal={}
    for name in summary:
        measures=[group["metrics"][name]["mean"] for group in group_reports if name in group["metrics"]]
        groups_equal[name]=dict(mean=float(np.mean(measures)),groups=len(measures))
    result=dict(status="SSDD_FIT741_DUPLICATE_VECTOR_LABEL_AUDIT_COMPLETE",fit=True,dev=False,test=False,
        model_forward_calls=0,optimizer_updates=0,images_decoded=0,fit_images=741,fit_annotation_rows=1661,
        duplicate_groups=len(duplicate_groups),duplicate_images=len(ids),counts=dict(counts),
        matches=summary,group_equal=groups_equal,group_summaries=group_reports,seconds=perf_counter()-started)
    raw=(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n").encode()
    with args.output.open("xb") as out:
        out.write(raw)
    print(json.dumps(dict(status=result["status"],sha256=sha256(raw).hexdigest(),groups=result["duplicate_groups"],
                          matches=counts["matched_rows"],groups_equal=groups_equal)),flush=True)

if __name__=="__main__":
    main()