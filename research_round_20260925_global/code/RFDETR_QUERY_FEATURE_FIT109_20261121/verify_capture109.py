"""Independently read back a closed FIT109 endpoint capture without detector passes or labels."""

import argparse
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

WORKER = "9f25e0bfcac25d44258b0b75ec94ad33f03a65feb4a20e896fb0381a3d21dc20"
CORE4 = "b49b9cd10a906c9141caceeb68ace08499305505986b7ea705579bb7b604b8a0"
CORE4_CLOSURE = "896963c572cfb482dbb005aaf01b721ffc156f88666ecd1377cbf3bb2d96e1fa"
FORMAL = "9f989c3a976642d9e790d65199dda43da4ae6b946d649fa13d1106258e99f84e"
ORIGINAL_ACCEPTANCE = "7f3f6dd112d287623360c8e3a174bae3a12a039c7f5505e4bf7c5d76b2e5ed6f"
PRELABEL = "33fc0e63fa5deff68aa384e03a3f00fd78430e63a7680b7a5a8025e33422acf4"
FEATURES = "1cea85eb8c30b11bfa847292c55d2231fb4d16dd2a551c3da52ddffdd8d3830e"
COMPLETE = "RFDETR_FIT109_QUERY_BASIS_CAPTURE_COMPLETE"
CONTROL_KEYS = {"zero", "ordinary_centered", "query_conditioned_centered", "query_conditioned_permuted"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bounded(path, size):
    with path.open("rb") as stream:
        raw = stream.read(size+1)
    require(0 < len(raw) <= size, "Bounded " + path.name)
    return raw


def pinned(path, digest):
    raw = bounded(path, 1024**2)
    require(sha256(raw).hexdigest() == digest, "Original " + path.name)
    return raw


def closed_tool(receipt, session, digest):
    require(type(session) is int and type(receipt["launch"]["session_id"]) is int
            and receipt["launch"]["session_id"] == session
            and type(receipt["terminal"]["exit_code"]) is int and receipt["terminal"]["exit_code"] == 0
            and receipt["observed_footer"]["status"] == COMPLETE
            and receipt["observed_footer"]["result_sha256"] == receipt["result_sha256"] == digest,
            "Original tool session, terminal and footer must all identify successful capture")


def same(actual, expected, name):
    require(actual.shape == expected.shape and actual.dtype == expected.dtype
            and actual.tobytes() == expected.tobytes(), "Byte-identical " + name)


def replay_image(raw, image_id, points, residual, np):
    keys = {"image_id","ranks","query_ids","offsets","old14","layers","coords","spatial","projected_queries"}
    with np.load(BytesIO(raw), allow_pickle=False) as store:
        require(len(store.files) == len(keys) and set(store.files) == keys, "Exact native archive schema")
        arrays = {k:store[k] for k in keys}
    same(arrays["image_id"], np.asarray(image_id), "image_id")
    for key, attr in (("ranks","rank"),("query_ids","query_id")):
        same(arrays[key], np.asarray([p[attr] for p in points], int), key)
    offsets = (np.concatenate(([0],np.cumsum([len(p["coordinates"]) for p in points]))) if points
               else np.array([0], dtype=np.int64))
    same(arrays["offsets"], offsets, "offsets")
    n = int(offsets[-1])
    for key, suffix in (("coords",(2,)),("layers",(2,)),("old14",(2,14)),("spatial",(2,256))):
        require(arrays[key].shape == (n,*suffix), "Complete shape for "+key)
    require(arrays["projected_queries"].shape == (len(points),256), "Exact supported query count")
    for key in ("old14","spatial","projected_queries"):
        require(arrays[key].dtype == np.float32 and bool(np.isfinite(arrays[key]).all()), "Finite frozen "+key)
    if not points:
        require(arrays["coords"].dtype == np.int32 and arrays["layers"].dtype == np.int8, "Native empty archive dtypes")
    moved = informative = 0
    for i, row in enumerate(points):
        cut = slice(offsets[i], offsets[i+1])
        for key, source in (("coords","coordinates"),("layers","layers"),("old14","features")):
            same(arrays[key][cut], row[source], key)
        k = len(row["coordinates"])
        require(1 <= k <= 32, "Sealed nonempty candidate support")
        controls = residual.point_feature_controls(row["features"], arrays["spatial"][cut], arrays["projected_queries"][i],
                   row["coordinates"], row["layers"], candidate_id=(image_id,row["rank"],row["query_id"]))
        concat, added = controls["features"], controls["residuals"]
        require(set(concat) == set(added) == CONTROL_KEYS, "Complete four controls")
        for key in CONTROL_KEYS:
            require(concat[key].shape == (k,2,30) and concat[key].dtype == np.float32
                    and bool(np.isfinite(concat[key]).all()), "Finite reconstructed control")
            same(concat[key][:,:,:14], row["features"], "old14 control prefix")
            same(concat[key][:,:,14:], added[key], "appended projection")
        focus, perm = added["query_conditioned_centered"], added["query_conditioned_permuted"]
        idx = controls["permutation_source_flat_indices"].reshape(-1)
        positions = np.arange(2*k)
        tags = row["layers"].reshape(-1)
        require(bool((added["zero"] == 0).all()) and np.array_equal(np.sort(idx),positions)
                and np.array_equal(idx%2,positions%2) and np.array_equal(tags[idx],tags), "Label-free control permutation scope")
        same(perm, focus.reshape(-1,16)[idx].reshape(k,2,16), "permuted projection")
        moved += int(np.any(perm != focus,axis=-1).sum())
        informative += int(np.any(added["ordinary_centered"] != focus,axis=-1).sum())
    return dict(candidate_sets=len(points), point_pairs=n, no_support_images=int(not points),
                moved_points=moved, informative_points=informative)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("root","code","capture","original-capture","source-cohort","acceptance","worker","core4-source",
                 "core4-closure","residual","protocol","receipt"):
        ap.add_argument("--"+name, required=True, type=Path)
    for name in ("protocol-sha","receipt-sha","result-sha"):
        ap.add_argument("--"+name, required=True)
    ap.add_argument("--session-id", required=True, type=int)
    a = ap.parse_args()
    require(sys.flags.isolated and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1",
            "Original CPU-only isolated process")
    require(a.session_id == 60587, "Sole original FIT109 capture tool session")
    closed_tool(json.loads(pinned(a.receipt,a.receipt_sha)), a.session_id, a.result_sha)
    result = json.loads(pinned(a.capture/"RESULT_SAFE.json",a.result_sha))
    require(result["status"] == COMPLETE and result["entry_sha256"] == WORKER
            and result["model_forwards"] == 105 and result["reused_core4"] == 4
            and result["images"] == result["private_file_count"] == 109
            and result["sampling_channel_chunk"] == 8 and result["head_updates"] == 0
            and all(result[k] is True for k in ("FIT","full_basis_logits_exact","exact_original_eligible_logits",
                                                "original_candidate_and_mask_exact","exact_dense_sample_witness"))
            and all(result[k] is False for k in ("GT_opened","AP","DEV","TEST")), "Accepted production capture scope")
    require(result["accepted_capture_sha256"] == ORIGINAL_ACCEPTANCE and result["feature_sha256"] == FEATURES
            and result["prelabel_sha256"] == PRELABEL and result["core4_source_sha256"] == CORE4
            and result["core4_closure_sha256"] == CORE4_CLOSURE and result["protocol_sha256"] == a.protocol_sha,
            "Qualified capture sources and protocol")
    pinned(a.worker,WORKER)
    pinned(a.protocol,a.protocol_sha)
    previous = json.loads(pinned(a.core4_closure,CORE4_CLOSURE))
    require(previous["result_sha256"] == result["core4_result_sha256"]
            and previous["independent_readback_accepted"] is True and previous["original_os_exit"] == 0
            and previous["job_drained"] is True and previous["model_forwards"] == 4, "Four accepted original archives")
    def offline(event, _):
        if event in ("socket.connect","socket.getaddrinfo"):
            raise PermissionError("Only local source images and sealed capture archives")
    sys.addaudithook(offline)
    pilot = ModuleType("_closed_pilot")
    pilot.__file__ = str(a.core4_source)
    exec(compile(pinned(a.core4_source,CORE4),pilot.__file__,"exec"),pilot.__dict__)
    require(result["basis_source_sha256"] == pilot.BASIS_SHA and result["residual_source_sha256"] == pilot.RESIDUAL_SHA,
            "Qualified frozen basis and projected controls")
    import numpy as np
    import torch
    from pycocotools import mask as codec
    require(sys.version_info[:3] == (3,10,13) and torch.__version__ == "2.10.0+cu130"
            and np.__version__ == "1.26.4" and not torch.cuda.is_initialized(), "Original CPU numerical runtime")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cap, spot, step, feat = [pilot.load(a.code/name,digest,f"_readback_prior_{i}")
                             for i,(name,digest) in enumerate(pilot.PINS.items()) if i < 4]
    formal = pilot.load(a.code/"RFDETR_MATCHED_POINT_FORMAL_20261120/run_fit109.py",FORMAL,"_readback_formal")
    residual = pilot.load(a.residual,pilot.RESIDUAL_SHA,"_readback_residual")
    posterior = step.load(a.code/"RFDETR_MATCHED_POINT_REFINEMENT_20261120/adapter.py",spot.ADAPTER_SHA,"_readback_post")
    core = cap.load_core(a.root.resolve(strict=True))
    source = SimpleNamespace(code=a.code, capture=a.original_capture, acceptance=a.acceptance,
                             acceptance_sha256=ORIGINAL_ACCEPTANCE, source_cohort=a.source_cohort,
                             fit_images=core.LAYOUT/"train")
    cohort, old_accept = formal.read_population(source,spot)
    prepared, counts, feature_digest = spot.prepare(source,cohort,step,feat,posterior,np,torch,codec)
    require(len(cohort) == len(prepared) == 109 and len({i["image_id"] for i,_,_ in cohort}) == 109
            and feature_digest == FEATURES and counts == result["original_feature_counts"]
            and counts["mapped"] == old_accept["native_mapped_rows"]
            and counts["eligible"] == counts["exact_original_masks"] == old_accept["same_device_raster_exact"],
            "Original FIT109 feature bytes and complete eligible count")
    reconstructed = spot.encode(dict(status="RFDETR_MATCHED_POINTS_FIT109_PRELABEL", features=counts,
        feature_sha256=feature_digest, accepted_capture_sha256=ORIGINAL_ACCEPTANCE,
        FIT=True, DEV=False, TEST=False, GT_opened=False))
    require(reconstructed == pinned(a.capture/"PRELABEL_SAFE.json",PRELABEL), "Exact original canonical prelabel")
    names = [f"NATIVE_ENDPOINT_BASIS_{i:02d}_PRIVATE.npz" for i in range(1,110)]
    require({file.name for file in a.capture.glob("NATIVE_ENDPOINT_BASIS_*_PRIVATE.npz")} == set(names),
            "Exactly 109 closed endpoint archives")
    totals = Counter()
    files = []
    for (image,ordinal,_),(same_image,_,points) in zip(cohort,prepared,strict=True):
        require(image == same_image and ordinal == len(files)+1, "Complete original prepared order")
        raw = bounded(a.capture/names[ordinal-1],32*spot.MIB)
        files.append(dict(ordinal=ordinal,sha256=sha256(raw).hexdigest(),bytes=len(raw)))
        totals.update(replay_image(raw,image["image_id"],points,residual,np))
    private_digest = sha256(spot.encode(files)).hexdigest()
    require(sha256(spot.encode(files[:4])).hexdigest() == previous["private_output_sha256"]
            and private_digest == result["private_output_sha256"], "Four original and all 109 ordered archive hashes")
    require(all(result[k] == v for k,v in totals.items()) and totals["candidate_sets"] == counts["candidate_point_sets"]
            and totals["point_pairs"] == counts["pairs"] and totals["informative_points"] > 0
            and totals["moved_points"] > 0 and not torch.cuda.is_initialized(), "Complete projected controls readback")
    receipt = dict(status="RFDETR_FIT109_QUERY_BASIS_INDEPENDENTLY_ACCEPTED", independent_readback_accepted=True,
                   FIT=True, DEV=False, TEST=False, GT_opened=False, AP=False, images=109, original_os_exit=0,
                   job_drained=True, model_forwards=105, reused_core4=4, head_updates=0, independent_forward_calls=0,
                   independent_head_updates=0, old_features_reproduced=True, entry_sha256=WORKER,
                   private_file_count=len(files), private_output_sha256=private_digest, result_sha256=a.result_sha,
                   original_terminal_receipt_sha256=a.receipt_sha, protocol_sha256=a.protocol_sha, prelabel_sha256=PRELABEL,
                   feature_sha256=feature_digest, accepted_capture_sha256=ORIGINAL_ACCEPTANCE,
                   core4_result_sha256=previous["result_sha256"], core4_closure_sha256=CORE4_CLOSURE,
                   independent_source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(), original_feature_counts=counts,
                   **dict(totals))
    digest = spot.write(a.capture/"INDEPENDENT_READBACK_SAFE.json",receipt)
    print(json.dumps(dict(status=receipt["status"],result_sha256=digest)),flush=True)


if __name__ == "__main__":
    main()
