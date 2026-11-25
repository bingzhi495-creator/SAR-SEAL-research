"""Fit clean matched heads with or without static-relative action supervision."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
from time import perf_counter
from types import ModuleType

import numpy as np
from pycocotools import mask
from scipy.optimize import linear_sum_assignment
import torch


PRIOR_FIT = "c00138dae34e620fe0617336309e2737ea7b247914fc8dc49b558382e278077e"
ACTION_CORE = "5e7deb9fb35a2b22f0d86dc4a4a1daca24a87d5134a7d87dacfb1af9824c5463"
OCCUPANCY_CORE = "0f1537f12ed4890bb60f71957e5d81926aeaab4760c8b01291383e4b5c53ee0d"
ATTRACTION = "8eb162a3bd544929843d7f2aeef174f1f1ba03a2c593b6bcda49289cd94ad3c6"
PROTOCOL = "baea6fc15caa8a053ab8cdda747ca37dc1b1dd347138c6b4dee1c1c9a67a08ce"
STATUS = "SPSD_PSEG_BCE_SIGNED_ACTION_MATCHED_FIT_COMPLETE"


def need(flag, reason):
    if not flag:
        raise ValueError(reason)


def digest(blob):
    return sha256(blob).hexdigest()


def pin(path, expected):
    raw = path.read_bytes()
    need(digest(raw)==expected, "Exact bound input " + path.name)
    return raw


def module(path, expected):
    mod = ModuleType(path.stem)
    mod.__file__ = str(path)
    exec(compile(pin(path,expected),str(path),"exec"),mod.__dict__)
    return mod


def canonical(rows):
    return tuple(tuple(sorted(x.tolist())) for x in np.asarray(rows,dtype=np.intp).reshape((-1,2)).T)


def neighbor(ranked, base):
    retained = set(base)
    out = next(i for i in reversed(ranked) if i in retained)
    enter = next(i for i in ranked if i not in retained)
    retained.remove(out)
    retained.add(enter)
    return tuple(sorted(retained))


def action_bank(coords, feature, stock, quota, original, attraction, key):
    n = len(coords)
    k = len(original)
    if not 0 < k < n:
        return ()
    area = int(stock.sum())
    dynamic, raw_static = attraction.select(stock,coords,min(8,area//10,n))
    need(len(raw_static) == len(dynamic) == k and np.array_equal(raw_static,original) and k==quota,
         "Exact label-blind dynamic/static same-k replay")
    static = canonical(raw_static)
    ranked = (np.argsort(-feature[:,0,0], kind="stable").tolist(),
              np.argsort(feature[:,1,0], kind="stable").tolist())
    native = tuple(tuple(sorted(side[:k])) for side in ranked)
    entropy = int.from_bytes(sha256(("FIT-signed-action-v1|" + key).encode("ascii")).digest()[:16],"big")
    rng = np.random.Generator(np.random.PCG64(entropy))
    shuffled = tuple(tuple(sorted(rng.permutation(n)[:k].tolist())) for _ in range(2))
    a,d = (neighbor(order,side) for order,side in zip(ranked,static,strict=True))
    proposals = [canonical(dynamic),native,shuffled,(a,static[1]),(static[0],d),(a,d)]
    unique = [q for q in dict.fromkeys(proposals) if q != static]
    need(bool(unique), "Unchanged native geometry must admit a distinct same-count action")
    weights = np.zeros((len(unique),n,2),dtype=np.int8)
    for idx, plan in enumerate(unique):
        for side, sign in enumerate((1,-1)):
            need(len(plan[side])==len(set(plan[side]))==k and all(0<=x<n for x in plan[side]),
                 "Complete matching action quota")
            weights[idx,list(plan[side]),side] += sign
            weights[idx,list(static[side]),side] -= sign
    need(np.all(weights.sum(axis=1)==0), "All signed actions preserve both edit counts")
    return weights


def fit_packets(args, api, action_core, attraction):
    authored, bindings, index, native = api.read_inputs(args)
    matches = [api.assignment(gt,native) for gt in authored]
    access = [defaultdict(list),defaultdict(list)]
    for rows, doc in zip(access,authored,strict=True):
        for a in doc["annotations"]:
            rows[a["image_id"]].append(a)
    counts = Counter(original_image_records=len(index), pseg_native_match=len(matches[0]), spsd_native_match=len(matches[1]))
    packets = []
    packet_digest = sha256()
    for item in index:
        iid, (h,w) = item["image_id"], item["original_hw"]
        labels = [table[iid] for table in access]
        masks = [[a["segmentation"] for a in rows] for rows in labels]
        overlap = mask.iou(*masks,[0]*len(masks[1]))
        shared = {}
        left,right = linear_sum_assignment(-overlap)
        counts["reference_pairs"] += len(left)
        for i,j in zip(left,right,strict=True):
            if overlap[i,j] < .50:
                continue
            counts["reference_pairs_ge050"] += 1
            original_p = matches[0].get(labels[0][i]["id"])
            original_s = matches[1].get(labels[1][j]["id"])
            if original_p is not None and original_p==original_s:
                need(original_p[0]==iid and original_p[1] not in shared,"Exact one-to-one matched FIT owner")
                shared[original_p[1]] = (masks[0][i],masks[1][j])
        counts["shared_native_match"] += len(shared)
        filename = f"POINTS_PRIVATE/{item['native_archive_ordinal']:04}.npz"
        need(filename==item["filename"],"Original FIT-only point filename")
        raw = pin(args.points/"actual_retry/fit"/filename,item["sha256"])
        need(len(raw)==item["bytes"], "Exact FIT point packet length")
        with np.load(BytesIO(raw),allow_pickle=False) as point_file:
            rank,offsets,coords,features,starts,static = (point_file[key] for key in
                ("native_ranks","point_offsets","signed_coordinates","features","static_offsets","static_side_indices"))
        n = len(rank)
        need(n==item["supported_candidates"] and len(set(rank))==n
             and offsets.shape==starts.shape==(n+1,) and offsets[0]==starts[0]==0
             and offsets[-1]==len(coords)==item["paired_sites"]
             and starts[-1]==len(static)==item["realized_static_exchanges"]
             and features.shape==(len(coords),2,14) and coords.shape==(len(features),2)
             and np.isfinite(features).all() and np.all((coords>=0)&(coords<h*w)),
             "Full FIT-only feature and point arrays")
        group = bindings[iid]["decoded_group_sha256"]
        for j,r in enumerate(rank):
            rid = int(r)
            if rid not in shared:
                continue
            lo,hi = (int(v) for v in offsets[j:j+2])
            qlo,qhi = (int(v) for v in starts[j:j+2])
            k = qhi-qlo
            counts["match_supported_hulls"] += 1
            if not 0 < k < hi-lo:
                counts["no_positive_or_free_quota_hulls"] += 1
                continue
            raw_stock = native[iid][rid-1]["segmentation"]
            stock = mask.decode(raw_stock).astype(bool,order="C")
            point = coords[lo:hi]
            feature = features[lo:hi].copy()
            weights = action_bank(point,feature,stock,k,static[qlo:qhi],attraction,
                                  f"{item['native_archive_ordinal']}|{rid}")
            need(weights.shape[1:]==(hi-lo,2),"Exact signed delta shape")
            target = np.stack([mask.decode(rle).ravel(order="C")[point] for rle in shared[rid]],axis=-1)
            need(target.shape==(hi-lo,2,2) and np.all((target==0)|(target==1)),"Native dual reference target")
            packet_digest.update(f"{int(iid)}|{rid}|{k}|{len(weights)}\n".encode("ascii"))
            for tensor in (feature,target,weights):
                packet_digest.update(tensor.tobytes(order="C"))
            packets.append(action_core.CandidatePacket(feature,target,weights,k,group,iid))
            counts.update(eligible_candidate_packets=1, native_endpoint_pairs=hi-lo, actions=len(weights),
                          disagreeing_points=int(np.count_nonzero(target[:,:,0]!=target[:,:,1])))
    counts["fit_groups_with_supervision"] = len({x.group_id for x in packets})
    counts["fit_images_with_supervision"] = len({x.image_id for x in packets})
    need(counts["reference_pairs"]==1575 and counts["reference_pairs_ge050"]==1488
         and counts["shared_native_match"]==1100 and counts["match_supported_hulls"]==848
         and counts["eligible_candidate_packets"]==len(packets)>0, "Prior frozen FIT association parity")
    return packets,dict(counts),packet_digest.hexdigest()


def write(path,result):
    data=(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return digest(data)


def main():
    p = ArgumentParser(description=__doc__)
    for name in ("protocol","capture","points","labels","prior-fit","action-core","source-core","attraction","out"):
        p.add_argument("--"+name,type=Path,required=True)
    args = p.parse_args()
    start = perf_counter()
    need(os.environ.get("CUDA_VISIBLE_DEVICES")=="-1","Admitted CPU-only training")
    pin(args.protocol,PROTOCOL)
    pin(args.source_core,OCCUPANCY_CORE)
    previous = module(args.prior_fit,PRIOR_FIT)
    core = module(args.action_core,ACTION_CORE)
    engine = module(args.attraction,ATTRACTION)
    packets,counts,packets_sha = fit_packets(args,previous,core,engine)
    args.out.mkdir()
    admitted = write(args.out/"INPUT_SAFE.json",dict(status="FIT_ACTION_PACKETS_ADMITTED",counts=counts,
                   packet_tensor_sha256=packets_sha, fit_only=True, label_blind_candidate_action_bank=True,
                   source_gate_labels_opened=False, historical_test_opened=False))
    models,center,scale,audit = core.train_matched_action(packets,source_core=args.source_core)
    heads = audit["heads"]
    need(set(heads)==set(models)=={"bce","bce_signed_action"}
         and all(v["optimizer_updates"]==1000 and v["parameters"]==1346 for v in heads.values())
         and len({v["initial_tensor_sha256"] for v in heads.values()})==1, "Actual equal capacity and updates")
    path=args.out/"FITTED_WEIGHTS_PRIVATE.pt"
    with path.open("xb") as stream:
        torch.save(dict(models={k:v.state_dict() for k,v in models.items()},center=torch.from_numpy(center),
                        scale=torch.from_numpy(scale),seed=5297),stream)
        stream.flush()
        os.fsync(stream.fileno())
    result=dict(status=STATUS, protocol_sha256=PROTOCOL, worker_sha256=digest(Path(__file__).read_bytes()),
                action_core_sha256=ACTION_CORE, training_core_sha256=OCCUPANCY_CORE,
                input_receipt_sha256=admitted, fit_packet_sha256=packets_sha, counts=counts,
                source_gate_labels_opened=False, historical_test_opened=False, detector_forwards=0,
                total_optimizer_updates=2000, output_checkpoint_sha256=digest(path.read_bytes()), train_audit=audit,
                elapsed_seconds=perf_counter()-start)
    checksum=write(args.out/"RESULT_SAFE.json",result)
    print(json.dumps(dict(status=STATUS,result_sha256=checksum)),flush=True)


if __name__=="__main__":
    main()
