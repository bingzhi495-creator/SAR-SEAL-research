"""Re-read original four FIT captures without model forwards or labels."""

import argparse
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import runpy
from types import SimpleNamespace

import numpy as np
import torch
from pycocotools import mask as codec


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("root","code","capture","worker"):
        p.add_argument("--"+name,type=Path,required=True)
    for name in ("result-sha","receipt-sha"):
        p.add_argument("--"+name,required=True)
    a=p.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES")=="-1"
    worker_sha="b49b9cd10a906c9141caceeb68ace08499305505986b7ea705579bb7b604b8a0"
    assert sha256(a.worker.read_bytes()).hexdigest()==worker_sha
    core=runpy.run_path(str(a.worker))
    pinned=core["pinned"]
    result=json.loads(pinned(a.capture/"RESULT_SAFE.json",a.result_sha))
    tool=json.loads(pinned(a.capture/"ORIGINAL_TOOL_WAIT_PRIVATE.json",a.receipt_sha))
    assert tool["terminal"]["exit_code"]==0 and tool["launch"]["session_id"]==58711
    assert tool["observed_footer"]["result_sha256"]==tool["result_sha256"]==a.result_sha
    assert result["status"]=="RFDETR_FIT_QUERY_BASIS_CORE4_CAPTURE_COMPLETE" and result["entry_sha256"]==worker_sha
    assert result["model_forwards"]==result["private_file_count"]==4 and not torch.cuda.is_initialized()
    assert result["FIT"] and not any(result[x] for x in ("DEV","TEST","AP","GT_opened","head_updates"))
    for k in ("original_candidate_and_mask_exact","exact_original_eligible_logits","full_basis_logits_exact","exact_dense_sample_witness"):
        assert result[k] is True
    prelabel=json.loads(pinned(a.capture/"PRELABEL_SAFE.json",result["prelabel_sha256"]))
    assert prelabel["FIT"] and not any(prelabel[x] for x in ("DEV","TEST","AP","GT_opened"))
    cap,spot,fit_step,feat=[core["load"](a.code/name,digest,"_core4_verify_"+str(i))
                            for i,(name,digest) in enumerate(core["PINS"].items()) if i<4]
    residual=core["load"](a.code/"RFDETR_QUERY_CHANNEL_RESIDUAL_20261120/channel_residual.py",
                          core["RESIDUAL_SHA"],"_core4_verify_residual")
    posterior=fit_step.load(a.code/"RFDETR_MATCHED_POINT_REFINEMENT_20261120/adapter.py",
                            spot.ADAPTER_SHA,"_core4_verify_posterior")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    old=cap.load_core(a.root)
    rec=SimpleNamespace(capture=a.root/core["CAPTURE_DIR"], manifest=a.root/core["MANIFEST_DIR"]/"IMAGE_ONLY_MANIFEST_PRIVATE.json",
                        acceptance=a.root/core["MANIFEST_DIR"]/"INDEPENDENT_READBACK_SAFE.json",fit_images=old.LAYOUT/"train",code=a.code)
    baserows=spot.read_cohort(rec,lambda images,count: images[:count])
    rows, counts, feature_digest=spot.prepare(rec,baserows,fit_step,feat,posterior,np,torch,codec)
    assert feature_digest==prelabel["feature_sha256"] and counts==prelabel["original_feature_counts"] and len(rows)==4
    files=[]
    no_support=pairs=candidates=changed=informative=0
    for i,(item,_,point_sets) in enumerate(rows,1):
        name=f"NATIVE_ENDPOINT_BASIS_{i:02d}_PRIVATE.npz"
        data=(a.capture/name).read_bytes()
        files.append(dict(ordinal=i,sha256=sha256(data).hexdigest(),bytes=len(data)))
        with np.load(BytesIO(data),allow_pickle=False) as z:
            assert set(z.files)=={"image_id","ranks","query_ids","coords","old14","layers","spatial","projected_queries","offsets"}
            assert int(z["image_id"])==item["image_id"]
            ranks,qids,offset=z["ranks"],z["query_ids"],z["offsets"]
            assert ranks.tolist()==[row["rank"] for row in point_sets] and qids.tolist()==[row["query_id"] for row in point_sets]
            assert offset.tolist()==[0,*np.cumsum([len(row["coordinates"]) for row in point_sets]).tolist()]
            c,f,t,s,q=[z[n] for n in ("coords","old14","layers","spatial","projected_queries")]
            k=int(offset[-1])
            assert (c.shape,f.shape,t.shape,s.shape,q.shape)==((k,2),(k,2,14),(k,2),(k,2,256),(len(point_sets),256))
            assert c.dtype.kind==t.dtype.kind=="i" and f.dtype==s.dtype==q.dtype==np.float32
            assert np.isfinite(s).all() and np.isfinite(q).all()
            for j,row in enumerate(point_sets):
                part=slice(offset[j],offset[j+1])
                assert np.array_equal(c[part],row["coordinates"]) and np.array_equal(t[part],row["layers"])
                assert f[part].dtype==row["features"].dtype and f[part].tobytes()==row["features"].tobytes()
                controls=residual.point_feature_controls(f[part],s[part],q[j],c[part],t[part],
                             candidate_id=(item["image_id"],row["rank"],row["query_id"]))["residuals"]
                side=controls["query_conditioned_centered"]
                changed+=int(np.any(side!=controls["query_conditioned_permuted"],axis=-1).sum())
                informative+=int(np.any(side!=controls["ordinary_centered"],axis=-1).sum())
            no_support+=int(not point_sets)
            pairs+=k
            candidates+=len(point_sets)
    aggregate=sha256(spot.encode(files)).hexdigest()
    assert aggregate==result["private_output_sha256"]
    assert (pairs,candidates,no_support,changed,informative)==tuple(result[k] for k in
           ("point_pairs","candidate_sets","no_support_images","moved_points","informative_points"))
    assert not torch.cuda.is_initialized()
    receipt=dict(status="RFDETR_FIT_QUERY_BASIS_CORE4_INDEPENDENTLY_ACCEPTED",result_sha256=a.result_sha,
                 entry_sha256=worker_sha,original_os_exit=0,job_drained=True,private_output_sha256=aggregate,
                 private_file_count=4,model_forwards=4,independent_readback_accepted=True,
                 original_terminal_receipt_sha256=a.receipt_sha,old_features_reproduced=True,
                 candidate_sets=candidates,point_pairs=pairs,no_support_images=no_support,
                 informative_points=informative,moved_points=changed,FIT=True,DEV=False,TEST=False,GT_opened=False,AP=False,
                 independent_forward_calls=0,independent_head_updates=0)
    digest=spot.write(a.capture/"INDEPENDENT_READBACK_SAFE.json",receipt)
    print(json.dumps(dict(sha=digest,receipt=receipt),ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()

