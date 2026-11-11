"""Recompute closed SSDD dev scores for every decoded-RGB representative choice."""
import argparse
from collections import defaultdict
from itertools import product
import json
import os
from pathlib import Path
import sys

RGB_SOURCE_SHA = "534f53359d0f055208704848dbc2ef17a99669019ce406233f7af30350ab8716"
ALIAS_SOURCE_SHA = "88086b3ff6a31d5788805ecb690472b01dbae2f1f92614fb4a946ae342b0f271"
STATUS = "SSDD_DEV187_DECODED_RGB_ALIAS_SENSITIVITY_COMPLETE"


def run(a, rgb, alias):
    p299=alias.module_from(a.code/alias.P299[0],alias.P299[1],"_p332_p299")
    receipt=p299.decode(p299.pinned_bytes(a.accepted,p299.ACCEPTED_SHA256,1024**2,"closed SSDD pair"),dict)
    require=alias.require
    require(receipt["coordinator_closed"] is True and receipt["coordinator_original_os_exit"]==0
            and receipt["DEV"] is True and receipt["TEST"] is False
            and receipt["image_manifest_sha256"]==rgb.DEV_SHA,"Fixed closed SSDD dev receipt required")
    g=alias.module_from(a.code/p299.GEOMETRY_WRAPPER,p299.GEOMETRY_SHA256,"_p332_geometry")
    _,_,template,scoring,official,_=g.load_helpers(a.code,a.code/p299.GEOMETRY_CORE)
    raw=rgb.exact(a.manifest,rgb.DEV_SHA,1024**2)
    _,ids,items=scoring.fixed_images(scoring.Artifact(
        dict(path=a.manifest.as_posix(),bytes=len(raw),sha256=rgb.DEV_SHA),raw))
    gt=p299.decode(p299.pinned_bytes(a.gt,receipt["integer_gt_sha256"],1024**2,"integer dev GT",
                                    exact_size=scoring.INTEGER_GT_BYTES),dict)
    scoring.validate_gt(gt,ids,items)
    rows={m:p299.decode(p299.pinned_bytes(getattr(a,m),receipt["arms"][m]["prediction_sha256"],
                                        64*1024**2,m+" original predictions"),list) for m in alias.ARMS}
    groups=defaultdict(list)
    enc_by_rgb=defaultdict(set)
    for item in items:
        ref=item["image_ref"]
        name=Path(ref["path"]).name
        path=a.dev_root/name
        require(path.resolve(strict=True).parent==a.dev_root,"Fixed original dev image root required")
        arr=rgb.rgb_id(rgb.exact(path,ref["sha256"],4*1024**2),item["original_hw"])
        groups[arr].append(item["image_id"])
        enc_by_rgb[arr].add(ref["sha256"])
    require(len(groups)==182 and sum(len(v)>1 for v in enc_by_rgb.values())==1,
            "Fixed decoded pixel alias census differs")
    unique={v[0] for v in groups.values() if len(v)==1}
    duplicates=sorted(tuple(sorted(v)) for v in groups.values() if len(v)>1)
    pres=[alias.evaluate_subset([i for i in ids if i in (unique|set(chosen))],gt,rows,template,official)
          for chosen in product(*duplicates)]
    excluded=alias.evaluate_subset([i for i in ids if i in unique],gt,rows,template,official)
    names=("AP","AP50","AP75","AP90","APs")
    ranges={}
    for key in names:
        vals=[m["center_minus_native_points"][key] for m in pres]
        require(all(v is not None for v in vals),"Fixed dev metric missing")
        ranges[key]=dict(min=min(vals),max=max(vals),positive_choices=sum(v>0 for v in vals))
    dup_ids=set(ids)-unique
    gsig=alias.geometry_signatures(gt,dup_ids)
    psig={m:alias.prediction_signatures(rows[m],dup_ids) for m in alias.ARMS}
    same_group=lambda signatures,group:all(signatures[x]==signatures[group[0]] for x in group)
    return dict(status=STATUS,DEV=True,TEST=False,FIT=False,AP=True,training=False,model_forward_calls=0,
                training_seed=5201,decoded_rgb_unique=len(groups),logical_images=len(ids),
                duplicate_group_sizes=sorted(map(len,duplicates)),exact_dev_jpeg_manifest_sha256=rgb.DEV_SHA,
                cross_encoded_same_rgb_groups=sum(len(v)>1 for v in enc_by_rgb.values()),
                gt_exact_raster_multiset_alias_groups=sum(same_group(gsig,g) for g in duplicates),
                same_prediction_sequence_alias_groups={m:sum(same_group(psig[m],g) for g in duplicates)
                                                      for m in alias.ARMS},
                representative_choices=len(pres),retained_annotations_min=min(m["annotations"] for m in pres),
                retained_annotations_max=max(m["annotations"] for m in pres),
                representative_delta_ranges_points=ranges,drop_all_aliases=excluded,
                selection_rule="enumerate every one-image choice per shape plus EXIF-oriented RGB group",
                posthoc_development_sensitivity=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("code","rgb-source","alias-source","accepted","manifest","gt","native","center","dev-root","output"):
        p.add_argument("--"+key,type=Path,required=True)
    a=p.parse_args()
    from hashlib import sha256
    from types import ModuleType

    def load(path,pin,name):
        raw=path.read_bytes()
        if sha256(raw).hexdigest()!=pin:
            raise ValueError("Exact fixed diagnostic source required")
        module=ModuleType(name)
        module.__file__=str(path)
        exec(compile(raw,str(path),"exec"),module.__dict__)
        return module

    rgb=load(a.rgb_source,RGB_SOURCE_SHA,"_p332_rgb")
    alias=load(a.alias_source,ALIAS_SOURCE_SHA,"_p332_alias")
    alias.require(a.dev_root.resolve(strict=True)==a.dev_root and a.output.is_absolute()
                  and a.output.parent.resolve(strict=True)==a.output.parent and not a.output.exists()
                  and os.environ.get("CUDA_VISIBLE_DEVICES")=="-1","CPU and fresh output required")
    sys.addaudithook(rgb.deny_network)
    raw=(json.dumps(run(a,rgb,alias),sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    with a.output.open("xb") as out:
        out.write(raw)
        out.flush()
        os.fsync(out.fileno())
    print(json.dumps(dict(status=STATUS,result_sha256=sha256(raw).hexdigest())),flush=True)


if __name__=="__main__":
    main()
