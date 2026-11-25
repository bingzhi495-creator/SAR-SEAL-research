"""FIT-only two-reference dependence calibration on the existing grouped OOF BCE."""
from argparse import ArgumentParser
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType, SimpleNamespace

import numpy as np
import torch


PARENT = "ae1b165a111a6b88939eb37b5741421259f0f39531c859a7c7663ad9116d1b05"
PARENT_WORKER = "011914954482cd505343a6421296ae754b1ea3098f5ad3fe37cdaa14e3c05fd3"
PARENT_TERMINAL = "b0d837ca9fdeb4b21bb6c4bb83ecda130b86696d751b4cad309a41379887991b"
BANK = "4872b63ee24c8021e09612314161c691006607dceb3ea92a2ce27690016836e4"
BANK_WORKER = "5419c9179f1ca82bf21436049450e623725c7e45b1cb2b7b19c2541e57e26fb7"
BANK_TERMINAL = "52e572c8916cafff8297f2308b95aa364fd6de7329e747f1e54275a32664548d"
BANK_READBACK = "8b280c6777d53e7125fb1c1f342d3eb7c33811ee18c700bf0f7be580a06f8af1"
REPEAT_PACKET = "de25b5d0c76eceb75d20bcedf4ec779b8cabc0eec31fee59d7305d4c4e51d785"
REPEAT_PROB = "04031bcfb6ed916d4e43ff3e362e45ecda866cbaf723e03ce1dba42f5d97aae7"
REPEAT_FOLD = "217c7b6e13e5739ee13ffe21da3caa6fa010c716372f430cee6d80d7bb5d4308"
ARGS = ("capture","points","labels","prior_fit","previous","action_core","source_core","attraction","writer","bank")
REFS = ("pseg","spsd")
GRAPHS = ("local","nonspatial")
METHODS = ("independent","local","nonspatial")
STATUS = "SPSD_PSEG_SPATIAL_PAIR_FIT_COMPLETE"


def need(check, reason):
    if not check:
        raise ValueError(reason)


def digest(data):
    return sha256(data).hexdigest()


def pin(path, expected):
    data = path.read_bytes()
    need(digest(data) == expected, "Exact prior SHA: " + path.name)
    return data


def load(path, expected, transform=lambda txt: txt):
    module = ModuleType(path.stem)
    module.__file__ = str(path)
    exec(compile(transform(pin(path,expected).decode("utf-8-sig")), str(path), "exec"), vars(module))
    return module


def write(path, obj):
    data = (json.dumps(obj, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()
    with path.open("xb") as stream:
        stream.write(data)
    return digest(data)


def closed_fit(root, name, safe_sha, terminal_sha, worker_sha):
    receipt = json.loads(pin(root/"actual/RESULT_SAFE.json",safe_sha))
    terminal = json.loads(pin(root/"observed_fit/TERMINAL.json",terminal_sha))
    pin(root/name,worker_sha)
    need(terminal["phase"]=="prepare-edits" and terminal["actual_exit"]==0 and
         terminal["output_footer_verified"] is True and terminal["result_sha256"]==safe_sha and
         terminal["worker_sha256"]==receipt["worker_sha256"]==worker_sha and
         receipt["source_gate_labels_opened"] is receipt["historical_test_opened"] is False,
         "Successful, exact closed FIT authority")
    return receipt


def fit_only_reader(source):
    # Preserve the exact combined archive receipt and decode FIT ordinal records only.
    anchor = "raw = [json.loads(line) for line in pinned("
    end = 'acceptance["predictions_sha256"]).splitlines()]'
    need(source.count(anchor)==source.count(end)==1,"Prior FIT-only native reader anchors")
    source=source.replace(anchor,"raw = [json.loads(line) if j in fit_ordinals else None for j,line in enumerate(pinned(")
    source=source.replace(end,'acceptance["predictions_sha256"]).splitlines())]')
    return source.replace("    raw = [json.loads(line) if j",
                          '    fit_ordinals = {r["native_archive_ordinal"]-1 for r in index}\n    raw = [json.loads(line) if j',1)


def bind_native_coordinates(source):
    anchor='            packet_digest.update(f"{int(iid)}|{rid}|{k}|{len(weights)}\\n".encode("ascii"))'
    need(source.count(anchor)==1,"Original FIT capture at the exact same matched label/coordinate slices")
    return source.replace(anchor,"            remember(iid, rid, point, w, target)\n"+anchor)


def reconstruct(config):
    settings = json.loads(config)
    need(set(settings)==set(ARGS),"Exact admitted FIT path configuration")
    a=SimpleNamespace(**{key:Path(settings[key]).resolve(strict=True) for key in ARGS})
    original=closed_fit(a.previous,"fit_action.py",PARENT,PARENT_TERMINAL,PARENT_WORKER)
    bank_stats=closed_fit(a.bank,"fit_banks.py",BANK,BANK_TERMINAL,BANK_WORKER)
    readback=json.loads(pin(a.bank/"READBACK_SAFE.json",BANK_READBACK))
    need(readback["fit_result_sha256"]==BANK and readback["terminal_sha256"]==BANK_TERMINAL and
         bank_stats["source_gate_points_opened"] is False, "Independently closed original OOF bank readback")
    old=load(a.previous/"fit_action.py",PARENT_WORKER,bind_native_coordinates)
    prior=load(a.prior_fit,old.PRIOR_FIT,fit_only_reader)
    action=load(a.action_core,old.ACTION_CORE)
    occupancy=load(a.source_core,old.OCCUPANCY_CORE)
    attraction=load(a.attraction,old.ATTRACTION)
    bank=load(a.bank/"fit_banks.py",BANK_WORKER)
    data=[]
    def remember(iid, rank, point, width, target):
        coordinates=np.stack((point//width,point%width),axis=-1).astype(np.int32,copy=False)
        data.append((iid,rank,coordinates,target))
    old.remember=remember
    packets,counts,packet_sha=old.fit_packets(a,prior,action,attraction)
    need(counts==original["counts"]==bank_stats["counts"] and
         packet_sha==original["fit_packet_sha256"]==bank_stats["fit_packet_sha256"]==REPEAT_PACKET and
         len(packets)==len(data)==848 and counts["fit_groups_with_supervision"]==439 and
         counts["fit_images_with_supervision"]==447 and counts["native_endpoint_pairs"]==26983,
         "Exact original eligible common dual-mask FIT packets")
    probabilities,audit,fold_sha,probability_sha=bank.crossfit_occupancy(packets,action,occupancy,a.source_core)
    folds=bank.fixed_folds([p.group_id for p in packets])
    need(fold_sha==bank_stats["out_of_fold_group_sha256"]==REPEAT_FOLD==digest(folds.tobytes()) and
         probability_sha==bank_stats["out_of_fold_probabilities_sha256"]==REPEAT_PROB and
         len(audit)==5 and all(v["overlap_groups"]==v["overlap_images"]==0 for v in audit) and
         sum(v["train_head"]["optimizer_updates"] for v in audit)==5000,
         "Exact prior five-group-fold predictions and schedule")
    return packets,probabilities,data,folds,counts


def weighted_pair_rows(groups, folds, sizes, group_ids):
    group_counts=Counter(groups)
    group_rows=np.repeat([group_ids[g] for g in groups],sizes)
    fold_rows=np.repeat(folds,sizes)
    weight=np.repeat([1/(group_counts[g]*n) for g,n in zip(groups,sizes,strict=True)],sizes)
    return group_rows,fold_rows,weight


def lexicographic_edges(edges):
    a=np.sort(edges,axis=1)
    return a[np.lexsort((a[:,1],a[:,0]))]


def prepare_blocks(packets, probabilities, data, folds, core):
    joined={key:dict(p=[],y=[],dist=[]) for key in GRAPHS}
    groups, fold_ids, sizes=[],[],[]
    type_counts=Counter()
    paired_vertex_groups=Counter()
    input_hasher=sha256()
    changed=0
    keys=set()
    for packet,prob,(iid,rank,points,target),fold in zip(packets,probabilities,data,folds,strict=True):
        key=f"{iid}:{rank}"
        need(iid==packet.image_id and np.array_equal(packet.targets,target) and
             points.shape==prob.shape==target.shape and points.shape[1:]==(2,2) and key not in keys,
             "Exact aligned native coordinates, probability and two reference labels")
        keys.add(key)
        edge=core.graphs(points,key)
        for name in GRAPHS:
            values=edge[name]
            need(values.ndim==2 and values.shape[1]==2 and np.issubdtype(values.dtype,np.integer) and
                 np.all((values>=0)&(values<points.shape[0]*2)) and len(np.unique(values))==values.size,
                 "Disjoint valid graph endpoint slots")
        local,nonspatial=(edge[name] for name in GRAPHS)
        need(np.array_equal(np.sort(local,axis=None),np.sort(nonspatial,axis=None)) and
             np.array_equal(np.sort((local%2).sum(axis=1)),np.sort((nonspatial%2).sum(axis=1))),
             "Nonspatial control preserves every paired vertex and the count of every sign type")
        input_hasher.update((key+"\n").encode("ascii"))
        input_hasher.update(points.astype("<i4",copy=False).tobytes())
        for name in GRAPHS:
            input_hasher.update(edge[name].astype("<i4",copy=False).tobytes())
        if not len(local):
            continue
        coord=points.reshape(-1,2)
        groups.append(packet.group_id)
        fold_ids.append(fold)
        sizes.append(len(local))
        paired_vertex_groups[packet.group_id]+=len(local)
        for j,n in enumerate(np.bincount((local%2).sum(axis=1),minlength=3)):
            type_counts[j]+=int(n)
        all_local={tuple(row) for row in lexicographic_edges(local)}
        changed+=sum(tuple(row) not in all_local for row in lexicographic_edges(nonspatial))
        for name in GRAPHS:
            ij=edge[name]
            arr=joined[name]
            arr["p"].append(prob.reshape(-1,2)[ij])
            arr["y"].append(target.reshape(-1,2)[ij])
            arr["dist"].append(np.abs(coord[ij[:,0]]-coord[ij[:,1]]).sum(axis=1))
    need(sizes and sum(sizes)==sum(paired_vertex_groups.values()),"At least one native spatial pair")
    group_ids={g:i for i,g in enumerate(sorted(set(groups)))}
    group_rows,fold_rows,weights=weighted_pair_rows(groups,fold_ids,sizes,group_ids)
    graph_summaries={}
    for name in GRAPHS:
        joined[name]={k:np.concatenate(v,axis=0) for k,v in joined[name].items()}
        dist=joined[name]["dist"]
        graph_summaries[name]=dict(mean_manhattan=float(dist.mean()), median_manhattan=float(np.median(dist)),
                                   max_manhattan=int(dist.max()), distance_one_edges=int(np.count_nonzero(dist==1)))
    need(graph_summaries["local"]["distance_one_edges"]==sum(sizes),"Exact four-neighbor local contrasts")
    geometry=dict(total_common_candidates=len(packets),candidates_with_pairs=len(sizes),
                  candidates_without_pairs=len(packets)-len(sizes),groups_with_pairs=len(group_ids),
                  groups_with_at_least_two_pairs=sum(v>=2 for v in paired_vertex_groups.values()),
                  pairs_per_graph=sum(sizes), changed_nonspatial_edges=changed,
                  changed_nonspatial_fraction=changed/sum(sizes), same_vertices_and_signed_type=True,
                  signed_type_pairs={key:type_counts[j] for j,key in enumerate(("outer_outer","outer_inner","inner_inner"))},
                  graph_distance=graph_summaries, geometry_sha256=input_hasher.hexdigest())
    return joined,group_rows,fold_rows,weights,geometry


def metrics(core, joined, groups, folds, weight, geometry):
    group_count=geometry["groups_with_pairs"]
    draws=np.random.Generator(np.random.PCG64(5297)).integers(group_count,size=(4096,group_count))
    references, final={},{}
    for ref_idx, ref in enumerate(REFS):
        fit={g:{kind:joined[g][kind][...,ref_idx] for kind in ("p","y")} for g in GRAPHS}
        group_brier={g:{method:np.zeros(group_count) for method in METHODS} for g in GRAPHS}
        fold_audit=[]
        for f in range(5):
            held=folds==f
            train=~held
            k={"independent":0.0,**{g:float(core.fit_kappa(fit[g]["p"][train],fit[g]["y"][train],weight[train]))
                                   for g in GRAPHS}}
            fold_audit.append(dict(fold=f,train_group_count=len(set(groups[train])),
                                   held_group_count=len(set(groups[held])),held_pair_count=int(held.sum()), kappa=k))
            for g in GRAPHS:
                grp=groups[held]
                q,y=fit[g]["p"][held],fit[g]["y"][held]
                for method in METHODS:
                    loss=np.asarray(core.brier(q,y,k[method]),dtype=np.float64)
                    need(loss.shape==grp.shape and np.all(np.isfinite(loss)) and np.all(loss>=0),
                         "One proper Brier score per original pair")
                    group_brier[g][method]+=np.bincount(grp,weights=loss*weight[held],minlength=group_count)
        def summary(values):
            return dict(mean=float(values.mean()),
                        percentile95=np.quantile(values[draws].mean(axis=1),[.025,.975]).tolist())
        held={g:{method:summary(value) for method,value in group_brier[g].items()} for g in GRAPHS}
        contrasts={g:{f"local_minus_{other}":summary(group_brier[g]["local"]-group_brier[g][other])
                       for other in ("independent","nonspatial")} for g in GRAPHS}
        final[ref]={g:float(core.fit_kappa(fit[g]["p"],fit[g]["y"],weight)) for g in GRAPHS}
        references[ref]=dict(held_scoring=held, held_local_minus_controls=contrasts,
                             folds=fold_audit, all_fit_kappa=final[ref],
                             local_held_strictly_better=all(x["mean"]<0 for x in contrasts["local"].values()))
    return references,final


def main():
    parser=ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ("inputs","core","protocol","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    for name in ("inputs-sha256","core-sha256","protocol-sha256"):
        parser.add_argument("--"+name,required=True)
    a=parser.parse_args()
    start=perf_counter()
    need(sys.flags.isolated and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES")=="-1" and
         all(os.environ.get(v)=="1" for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS")) and
         torch.get_num_threads()==1 and not a.output.exists(),"Fresh isolated original single-threaded FIT runtime")
    pin(a.protocol,a.protocol_sha256)
    core=load(a.core,a.core_sha256)
    packets,predictions,data,folds,counts=reconstruct(pin(a.inputs,a.inputs_sha256))
    joint,groups,fold_rows,weight,geometry=prepare_blocks(packets,predictions,data,folds,core)
    references,final=metrics(core,joint,groups,fold_rows,weight,geometry)
    qualified=(geometry["groups_with_at_least_two_pairs"]>=150 and geometry["changed_nonspatial_fraction"]>=.25 and
               all(row["local_held_strictly_better"] for row in references.values()))
    a.output.mkdir()
    fit_state=dict(status="SPSD_PSEG_SPATIAL_PAIR_KAPPA_FIT",kappa=final,protocol_sha256=a.protocol_sha256,
                   qualified_for_fixed_rerender=qualified, fit_only=True)
    state_sha=write(a.output/"KAPPA_PRIVATE.json",fit_state)
    report=dict(status=STATUS, protocol_sha256=a.protocol_sha256, worker_sha256=digest(Path(__file__).read_bytes()),
                pair_core_sha256=a.core_sha256, fit_inputs_sha256=a.inputs_sha256, fit_packet_sha256=REPEAT_PACKET,
                prior_action_fit_result_sha256=PARENT, prior_bank_result_sha256=BANK, prior_bank_readback_sha256=BANK_READBACK,
                oof_probability_sha256=REPEAT_PROB,oof_fold_sha256=REPEAT_FOLD, prior_counts=counts, fit_only=True,
                historical_test_opened=False, transfer_labels_opened=False, source_gate_labels_opened=False,
                source_gate_points_opened=False,native_detector_forwards=0,oof_optimizer_updates=5000,
                nested_base_heads=False,group_bootstrap_draws=4096,geometry=geometry,references=references,
                full_fit_kappa_state_sha256=state_sha, qualified_for_fixed_rerender=qualified,
                elapsed_seconds=perf_counter()-start)
    result_sha=write(a.output/"RESULT_SAFE.json",report)
    print(json.dumps(dict(status=STATUS,result_sha256=result_sha)),flush=True)


if __name__=="__main__":
    main()
