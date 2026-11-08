"""Qualify the empty-mask-only repair on the exact closed HRSID FIT audit cohort."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import sys
from types import ModuleType


INDEX_SHA="e95cf26cc60a2c6fb651e907541380a256f2df7b1e153244a6c0f803b0b19f8f"
AUDIT_SHA="6d1b1b94c14cc8f6c3cf7d5eb70ea8db83a6c0f4c0219c68b4dbf24d773aca14"
GROUPS=32
REPEATS=12


def load_pin(path, expected):
    raw=path.read_bytes()
    if sha256(raw).hexdigest()!=expected:
        raise ValueError("Exact pinned FIT source required")
    return raw


def main(args):
    if os.environ["CUDA_VISIBLE_DEVICES"]!="-1" or sys.version_info[:3]!=(3,10,13):
        raise ValueError("CPU-only original HRSID numeric environment")
    if args.output.exists() or args.output.is_symlink() or args.output.is_relative_to(args.train.parent):
        raise ValueError("Fresh private output outside FIT dataset")
    repair=ModuleType("_prospective_empty_mask_filter")
    repair.__file__=str(args.filter)
    exec(compile(load_pin(args.filter,"3b164cbf8c104539eabee8775c8e1f297d9a1607e37473c4d7a4a9b561f93471"),str(args.filter),"exec"),repair.__dict__)
    source=ModuleType("_closed_native_augmentation_audit")
    source.__file__=str(args.audit)
    exec(compile(load_pin(args.audit,AUDIT_SHA),str(args.audit),"exec"),source.__dict__)
    sys.addaudithook(source.offline)
    import numpy as np
    import torch
    from PIL import Image

    source.need(not torch.cuda.is_available()
                and all(source.version(k)==v for k,v in source.VERSIONS.items())
                and all(sha256(Path(m.__file__).read_bytes()).hexdigest()==v for m,v in source.SOURCE_SHA.items()),
                "Exact previously qualified source and runtime")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    index=json.loads(load_pin(args.train/"_annotations.coco.json",INDEX_SHA))
    anns=defaultdict(list)
    for ann in index["annotations"]:
        anns[ann["image_id"]].append(ann)
    source.need(len(index["images"])==len(anns)==2937 and len(index["annotations"])==9260,
                "Exact official-derived FIT census")
    # Select original images by the smallest retained non-crowd positive COCO area.
    eligible=[(min(a["area"] for a in anns[row["id"]] if a["area"]>0 and not a.get("iscrowd",0)),
               row["id"],row) for row in index["images"]
              if any(a["area"]>0 and not a.get("iscrowd",0) for a in anns[row["id"]])]
    selected=sorted(eligible)[:GROUPS]
    audit=source.NativeInputAudit()
    original=source._torchvision._apply_to_masks
    traces=[]

    def trace_mask_resize(masks, resize):
        out=original(masks,resize)
        vanished=masks.flatten(1).any(1) & ~out.flatten(1).any(1)
        if bool(vanished.any()):
            traces.append(dict(from_shape=list(masks.shape[-2:]),to_shape=list(out.shape[-2:]),
                               newly_empty=int(vanished.sum())))
        return out

    source._torchvision._apply_to_masks=trace_mask_resize
    args.output.mkdir()
    findings=[]
    counts=Counter()
    for ordinal,(_,iid,row) in enumerate(selected):
        name=row["file_name"]
        path=args.train/name
        source.need(Path(name).name==name and path.suffix.lower()==".png"
                    and path.resolve(strict=True)==path and row["height"]==row["width"]==800,
                    "Only a canonical original FIT PNG")
        with Image.open(path) as image:
            im,raw=audit.convert(image.convert("RGB"),dict(image_id=iid,annotations=anns[iid]))
        source.need(not bool(source.mask_empty(raw).any()),"The source retained masks are all nonempty")
        for rep in range(REPEATS):
            seed=520150000+ordinal*REPEATS+rep
            source.set_seed(seed)
            traces.clear()
            image,target=audit.transform(im,raw)
            _,packed=audit.collate([(image,target)])
            target,=packed.to_list("cpu")
            boxes,labels=target["boxes"],target["labels"]
            empties=source.mask_empty(target)
            half=boxes[:,2:]/2
            label_ok=bool((labels==0).all())
            finite=bool(torch.isfinite(boxes).all())
            positive=bool((half>0).all())
            bounded=bool(((boxes[:,:2]-half)>=-1e-6).all() & ((boxes[:,:2]+half)<=1+1e-6).all())
            before=target.copy()
            removed=repair.drop_empty_augmented_masks(target)
            source.need(removed==int(empties.sum()) and not bool(source.mask_empty(target).any()),
                        "Exactly the empty rows are removed")
            for key,value in before.items():
                if removed and key in ("boxes","labels","area","iscrowd","keypoints","masks"):
                    source.need(torch.equal(target[key],value[~empties]),"Every surviving row is exact")
                else:
                    source.need(target[key] is value,"Image metadata and unaffected target values stay identical")
            source.need(label_ok and finite and positive and bounded,"All hard predicates remain qualified")
            counts.update(filtered_rows=removed,unchanged_views=int(removed==0),
                          views=1,retained_targets=len(labels),empty_mask_rows=int(empties.sum()),
                          views_with_empty_masks=int(bool(empties.any())),bad_labels=int(not label_ok),
                          nonfinite_boxes=int(not finite),nonpositive_boxes=int(not positive),bounds=int(not bounded))
            if bool(empties.any()) or not (label_ok and finite and positive and bounded):
                findings.append(dict(image_id=iid,ordinal=ordinal,repeat=rep,seed=seed,empty_rows=int(empties.sum()),
                                     labels_ok=label_ok,finite_boxes=finite,positive_boxes=positive,bounded_boxes=bounded,
                                     resize_witness=list(traces)))
        print(json.dumps(dict(event="HRSID_TARGET_GUARD_AUDIT_GROUP_COMPLETE",ordinal=ordinal)),flush=True)
    source.need(counts["views"]==384 and counts["retained_targets"]==8027
                and counts["filtered_rows"]==counts["empty_mask_rows"]==2,
                "Exact closed two-witness qualification scope")
    safe=dict(status="HRSID_NATIVE_EMPTY_MASK_FILTER_FIT_QUALIFIED",protocol=args.protocol,
              FIT_only=True,DEV=False,TEST=False,AP=False,optimizer_updates=0,groups=GROUPS,repeats=REPEATS,
              counts=dict(counts),distinct_affected_groups=len({x["ordinal"] for x in findings}),
              witnesses_with_resize_erasure=sum(bool(x["resize_witness"]) for x in findings),
              original_failure_identity_recovered=False,
              resize_stage_tallies=dict(Counter(f'{x["from_shape"][0]}->{x["to_shape"][0]}'
                  for v in findings for x in v["resize_witness"])))
    for name,value in (("RESULT.json",safe),("FINDINGS_PRIVATE.json",findings)):
        with (args.output/name).open("x",encoding="utf-8") as f:
            json.dump(value,f,indent=2,sort_keys=True,allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
    print(json.dumps(safe,sort_keys=True),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("train","audit","filter","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--protocol",required=True)
    main(parser.parse_args())
