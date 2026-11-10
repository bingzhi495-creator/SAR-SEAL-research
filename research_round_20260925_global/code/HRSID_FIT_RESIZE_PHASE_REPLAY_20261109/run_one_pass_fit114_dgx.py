"""Audit the actual single-pass HRSID FIT loader on the sealed historical cohort."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from types import ModuleType
from zipfile import ZipFile


HERE=Path(__file__).resolve().parent
SSDD=HERE.parent/"SSDD_MASK_PHASE_TRANSFER_20261110"


def run(args):
    sys.path[:0]=[str(HERE),str(SSDD)]
    from run_continuous_fit_dgx import (AUDIT,AUDIT_SHA,BASE_SEED,COHORT_SHA,FIT_SHA,HISTORY_SHA,
                                        REPEATS,digest,file_digest,pinned_read,require)
    from ssdd_fit_one_pass_dgx import audit_view,load_audit_helpers,summarize_audits
    from ssdd_fit_replay_dgx import runtime
    runtime()
    import numpy as np
    import torch

    source=AUDIT.read_bytes()
    require(digest(source)==AUDIT_SHA,"Exact original HRSID augmentation audit")
    native=ModuleType("_hrsid_fit114_one_pass_audit_stock")
    native.__file__=str(AUDIT)
    exec(compile(source,str(AUDIT),"exec"),native.__dict__)
    sys.addaudithook(native.offline)
    require(all(native.version(k)==v for k,v in native.VERSIONS.items())
            and all(digest(Path(m.__file__).read_bytes())==v for m,v in native.SOURCE_SHA.items()),
            "Exact original HRSID augmentation runtime")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    require(not torch.cuda.is_initialized(),"GPU stays hidden")

    bundle_path,out=args.bundle,args.output
    require(bundle_path.is_absolute() and bundle_path.resolve(strict=True)==bundle_path
            and 0<bundle_path.stat().st_size<=192*1024**2 and out.is_absolute()
            and out.parent.resolve(strict=True)==out.parent and not out.exists() and not out.is_symlink(),
            "Exact sealed FIT input and fresh independent output")
    with bundle_path.open("rb") as f:
        require(file_digest(f)==args.bundle_sha256,"Exact transported HRSID FIT114 data")
    def identity():
        st=bundle_path.stat()
        return st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns
    before=identity()
    out.mkdir()
    audit=native.NativeInputAudit()
    helpers=load_audit_helpers()
    entries=defaultdict(list)
    old_paths=defaultdict(Counter)
    evidence=sha256()
    with ZipFile(bundle_path) as bundle,(out/"VIEWS_PRIVATE.jsonl").open("xb") as ledger:
        cohort=json.loads(pinned_read(bundle,"COHORT_PRIVATE.json",COHORT_SHA,1024**2))
        index=json.loads(pinned_read(bundle,"FIT114_COCO_PRIVATE.json",args.subset_sha256,4*1024**2))
        history=[json.loads(line) for line in pinned_read(bundle,"HISTORICAL_VIEWS_PRIVATE.jsonl",HISTORY_SHA,1024**2).splitlines()]
        images={row["id"]:row["file_name"] for row in index["images"]}
        anns=defaultdict(list)
        for row in index["annotations"]:
            require(row["image_id"] in images,"Only exact selected FIT annotations")
            anns[row["image_id"]].append(row)
        rows=cohort["images"]
        expected={"COHORT_PRIVATE.json","FIT114_COCO_PRIVATE.json","HISTORICAL_VIEWS_PRIVATE.jsonl",
                  *("images/"+row["native_basename"] for row in rows)}
        require(len(rows)==len(images)==114 and len(history)==456 and cohort["fit_only"] is True
                and set(bundle.namelist())==expected and len(bundle.namelist())==len(expected),
                "Exact isolated FIT114 population")
        for ordinal,row in enumerate(rows):
            iid,name=row["image_id"],row["native_basename"]
            require(images[iid]==name and row["core"]!=row["stress"],"Exact historic cohort identity")
            raw=pinned_read(bundle,"images/"+name,row["encoded_sha256"],4*1024**2)
            image,scale=native.io_utils.decode_pil_image_bytes(raw,draft_size=None)
            require(len(raw)==row["encoded_bytes"] and image.mode=="RGB" and image.size==(800,800) and scale==(1.,1.)
                    and digest(np.asarray(image).tobytes(order="C"))==row["decoded_rgb_sha256"],
                    "Exact original encoded and decoded PNG identity")
            prepared,targets=audit.convert(image,dict(image_id=iid,annotations=anns[iid]))
            native.target_summary(targets,800,iid,normalized=False)
            require(not bool(native.mask_empty(targets).any()),"Historical prepared masks are nonempty")
            layer="core" if row["core"] else "stress"
            for rep in range(REPEATS):
                cursor=ordinal*REPEATS+rep
                seed=BASE_SEED+cursor
                old=history[cursor]
                view=audit_view(native._torchvision,audit.transform,prepared,targets,seed,
                                set_seed=native.set_seed,helpers=helpers)
                verify_historical(view,old,ordinal,seed,layer,row["group"],require)
                entries[layer].append(view)
                old_paths[layer][old["path"]]+=1
                data=(json.dumps(dict(view,ordinal=ordinal,seed=seed,layer=layer,image_id=iid),
                                 sort_keys=True,allow_nan=False)+"\n").encode("utf-8")
                ledger.write(data)
                evidence.update(data)
            if (ordinal+1)%20==0:
                print(json.dumps(dict(event="HRSID_FIT114_ONE_PASS_AUDIT_PROGRESS",images=ordinal+1)),flush=True)
        ledger.flush()
        os.fsync(ledger.fileno())
    require(len(entries["core"])==436 and len(entries["stress"])==20 and before==identity()
            and not torch.cuda.is_initialized(),"Exact original view census and stable FIT data")
    scopes={layer:summarize_audits(entries[layer]) for layer in ("core","stress")}
    result=dict(status="HRSID_FIT114_ONE_PASS_STOCK_AUDIT_COMPLETE",images=114,views=456,views_by_layer={"core":436,"stress":20},
                paths_by_layer=old_paths,fit_index_sha256=FIT_SHA,cohort_sha256=COHORT_SHA,
                historical_ledger_sha256=HISTORY_SHA,actual_audit_ledger_sha256=evidence.hexdigest(),scope_results=scopes,
                same_historical_center_population_and_trace=True,platform="Linux_aarch64_DGX_CPU",
                FIT=True,DEV=False,TEST=False,AP=False,cuda_initialized=False,forward_calls=0,optimizer_updates=0)
    with (out/"RESULT_SAFE.json").open("x",encoding="utf-8",newline="\n") as f:
        json.dump(result,f,sort_keys=True,indent=2,allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def verify_historical(view,old,ordinal,seed,layer,group,require):
    require(old["ordinal"]==ordinal and old["seed"]==seed and old["layer"]==layer and old["group"]==group,
            "Exact historical FIT view ordinal, seed and scope")
    require(view["path"]==old["path"] and view["flip"]==old["flip"] and view["steps"]==old["steps"],
            "Exact historical per-view augmentation trace")
    require(view["raw_rows"]==old["rows"]["raw_rows"] and view["b_present"]==old["rows"]["center_present"]
            and view["b_nonempty"]==old["rows"]["center_nonempty"],
            "Exact historical stepwise-center target census")


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument("--bundle",required=True,type=Path)
    p.add_argument("--bundle-sha256",required=True)
    p.add_argument("--subset-sha256",required=True)
    p.add_argument("--output",required=True,type=Path)
    run(p.parse_args())


if __name__=="__main__":
    main()

