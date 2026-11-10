"""Check the original HRSID FIT114 views against continuous source-pixel footprints."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import sys
from types import ModuleType
from zipfile import ZIP_STORED, ZipFile


HERE = Path(__file__).resolve().parent
SSDD = HERE.parent / "SSDD_MASK_PHASE_TRANSFER_20261110"
AUDIT = HERE.parent / "HRSID_PNG_FIT_NATIVE_AUG_AUDIT_20261109/audit_hrsid_fit_native_aug.py"
AUDIT_SHA = "6d1b1b94c14cc8f6c3cf7d5eb70ea8db83a6c0f4c0219c68b4dbf24d773aca14"
FIT_SHA = "e95cf26cc60a2c6fb651e907541380a256f2df7b1e153244a6c0f803b0b19f8f"
COHORT_SHA = "70e9a0e8fb30f5ba70b48b9067c1b95d07bfec23d6b248d92c5983727acceb6e"
HISTORY_SHA = "318408a43f2c107b329638b8be3132717f7d366b046e98aa2190318f3aaca053"
BASE_SEED = 520198000
REPEATS = 4


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return sha256(raw).hexdigest()


def pinned_read(bundle, name, expected, limit):
    info = bundle.getinfo(name)
    require(info.compress_type == ZIP_STORED and 0 < info.file_size <= limit, "Exact bounded FIT-only ZIP entry")
    raw = bundle.read(info)
    require(len(raw) == info.file_size and digest(raw) == expected, "Exact sealed FIT-only entry bytes")
    return raw


def baseline_equal(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(actual, dict):
        return actual.keys() == expected.keys() and all(baseline_equal(a, expected[k]) for k, a in actual.items())
    if isinstance(actual, list):
        return len(actual) == len(expected) and all(baseline_equal(a, b) for a,b in zip(actual,expected,strict=True))
    if type(actual) is float:
        return math.isclose(actual,expected,rel_tol=1e-12,abs_tol=1e-10)
    return actual == expected


class CurrentGroup:
    """Route the reused footprint recorder to the current historical FIT group."""
    def __init__(self):
        self.groups = defaultdict(dict)
        self.current = None

    def setdefault(self, key, value):
        return self.groups[self.current].setdefault(key, value)


def aggregate(groups, summarize, arms, metrics):
    merged = {}
    for group in groups:
        for tag, record in group.items():
            bucket = merged.setdefault(tag, dict(census=Counter(), by_arm={arm:Counter() for arm in arms},
                                                pairs=Counter()))
            bucket["census"].update(record["census"])
            bucket["pairs"].update(record["pairs"])
            for arm in arms:
                bucket["by_arm"][arm].update(record["by_arm"][arm])
    output = summarize(merged)
    for tag,bucket in output.items():
        for arm in arms:
            for metric in (*metrics, "nonempty"):
                matched = [g[tag]["by_arm"][arm] for g in groups if tag in g
                           and g[tag]["by_arm"][arm][metric+"_count"] > 0]
                bucket["arms"][arm][metric]["groups"] = len(matched)
                bucket["arms"][arm][metric]["group_equal_mean"] = (
                    sum(g[metric+"_sum"] / g[metric+"_count"] for g in matched)/len(matched) if matched else None)
    return output


def run(args):
    sys.path.insert(0, str(SSDD))
    from ssdd_fit_replay_dgx import runtime
    runtime()
    from ssdd_fit_footprint_dgx import ARMS, METRICS, replay_module, summarize
    import numpy as np
    import torch

    source = AUDIT.read_bytes()
    require(digest(source)==AUDIT_SHA, "Exact original HRSID augmentation audit source")
    audit_module=ModuleType("_hrsid_fit114_closed_augmentation")
    audit_module.__file__=str(AUDIT)
    exec(compile(source,str(AUDIT),"exec"),audit_module.__dict__)
    sys.addaudithook(audit_module.offline)
    require(all(audit_module.version(name)==v for name,v in audit_module.VERSIONS.items())
            and all(digest(Path(m.__file__).read_bytes())==v for m,v in audit_module.SOURCE_SHA.items()),
            "Exact HRSID augmentation runtime")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    require(not torch.cuda.is_initialized(), "GPU stays hidden")

    path, out = args.bundle,args.output
    require(path.is_absolute() and path.resolve(strict=True)==path and 0<path.stat().st_size<=192*1024**2
            and out.is_absolute() and out.parent.resolve(strict=True)==out.parent and not out.exists()
            and not out.is_symlink(), "Original sealed FIT-only input and fresh output")
    with path.open("rb") as f:
        require(file_digest(f)==args.bundle_sha256, "Exact transported private FIT subset")
    start_identity = (path.stat().st_ino,path.stat().st_size,path.stat().st_mtime_ns)
    router=CurrentGroup()
    replay=replay_module(router)._load_replay()
    audit=audit_module.NativeInputAudit()
    baseline_hash=sha256()
    used_views=Counter()
    layer_paths=defaultdict(Counter)
    original_metrics_exact=True
    with ZipFile(path) as bundle:
        cohort=json.loads(pinned_read(bundle,"COHORT_PRIVATE.json",COHORT_SHA,1024**2))
        index=json.loads(pinned_read(bundle,"FIT114_COCO_PRIVATE.json",args.subset_sha256,4*1024**2))
        history=[json.loads(line) for line in pinned_read(bundle,"HISTORICAL_VIEWS_PRIVATE.jsonl",HISTORY_SHA,1024**2).splitlines()]
        images={row["id"]:row["file_name"] for row in index["images"]}
        anns=defaultdict(list)
        for annotation in index["annotations"]:
            require(annotation["image_id"] in images, "Only selected FIT annotation rows")
            anns[annotation["image_id"]].append(annotation)
        rows=cohort["images"]
        expected={"COHORT_PRIVATE.json","FIT114_COCO_PRIVATE.json","HISTORICAL_VIEWS_PRIVATE.jsonl",
                  *("images/"+row["native_basename"] for row in rows)}
        require(len(rows)==len(images)==114 and len(history)==456 and cohort["fit_only"] is True
                and set(bundle.namelist())==expected and len(bundle.namelist())==len(expected),
                "Exact closed FIT114 input roster")
        for ordinal,row in enumerate(rows):
            iid,name=row["image_id"],row["native_basename"]
            require(images[iid]==name and row["core"]!=row["stress"], "Exact historic FIT cohort identity")
            raw=pinned_read(bundle,"images/"+name,row["encoded_sha256"],4*1024**2)
            require(len(raw)==row["encoded_bytes"], "Exact historic encoded size")
            image,scale=audit_module.io_utils.decode_pil_image_bytes(raw,draft_size=None)
            require(image.mode=="RGB" and image.size==(800,800) and scale==(1.,1.)
                    and digest(np.asarray(image).tobytes(order="C"))==row["decoded_rgb_sha256"],
                    "Exact historical decoded pixels")
            prepared_image,target=audit.convert(image,dict(image_id=iid,annotations=anns[iid]))
            audit_module.target_summary(target,800,iid,normalized=False)
            require(not bool(audit_module.mask_empty(target).any()), "Original FIT masks are nonempty")
            target[replay.ROW_ID]=torch.arange(len(target["labels"]),dtype=torch.int64)
            layer="core" if row["core"] else "stress"
            router.current=(layer,row["group"])
            for rep in range(REPEATS):
                cursor=ordinal*REPEATS+rep
                seed=BASE_SEED+cursor
                view=replay.replay_view(audit,audit_module,prepared_image,target,seed)
                observed=dict(view,group=row["group"],layer=layer,ordinal=ordinal,seed=seed)
                require(baseline_equal(observed,history[cursor]), f"Historical FIT view differs at ordinal {cursor}")
                original_metrics_exact=original_metrics_exact and observed==history[cursor]
                baseline_hash.update((json.dumps(observed,sort_keys=True,allow_nan=False)+"\n").encode())
                used_views[layer]+=1
                layer_paths[layer][view["path"]]+=1
            if (ordinal+1)%20==0:
                print(json.dumps(dict(event="HRSID_FIT114_FOOTPRINT_PROGRESS",images=ordinal+1)),flush=True)
    require(used_views=={"core":436,"stress":20} and not torch.cuda.is_initialized()
            and start_identity==(path.stat().st_ino,path.stat().st_size,path.stat().st_mtime_ns),
            "Exact view scope and stable closed data")
    per_layer={layer:aggregate([v for (name,_),v in router.groups.items() if name==layer],summarize,ARMS,METRICS)
               for layer in ("core","stress")}
    report=dict(status="HRSID_FIT114_CONTINUOUS_PIXEL_FOOTPRINT_REPLAY_COMPLETE",views=456,images=114,
                views_by_layer=used_views,paths_by_layer=layer_paths,fit_index_sha256=FIT_SHA,
                cohort_sha256=COHORT_SHA,subset_sha256=args.subset_sha256,old_view_ledger_sha256=HISTORY_SHA,
                reproduced_view_ledger_sha256=baseline_hash.hexdigest(),
                exact_original_ledger_bytes=baseline_hash.hexdigest()==HISTORY_SHA,
                exact_original_numeric_view_values=original_metrics_exact,
                exact_old_scope_trace_and_membership=True,primary="core.crop_small",
                oracle="clipped_original_boolean_pixel_squares",populations=per_layer,
                platform="Linux_aarch64_DGX_CPU",cuda_initialized=False,
                FIT=True,DEV=False,TEST=False,AP=False,forward_calls=0,optimizer_updates=0)
    with out.open("x",encoding="utf8",newline="\n") as f:
        json.dump(report,f,sort_keys=True,allow_nan=False,indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def file_digest(stream):
    value=sha256()
    for block in iter(lambda:stream.read(1024**2),b""):
        value.update(block)
    return value.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument("--bundle",required=True,type=Path)
    p.add_argument("--bundle-sha256",required=True)
    p.add_argument("--subset-sha256",required=True)
    p.add_argument("--output",required=True,type=Path)
    run(p.parse_args())


if __name__=="__main__":
    main()

