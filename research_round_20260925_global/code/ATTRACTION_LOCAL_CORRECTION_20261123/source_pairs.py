"""Materialize FIT109 source-held-out static/learned disagreements and their signed fractional labels."""
import argparse
from collections import Counter
from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType, SimpleNamespace

import numpy as np
from pycocotools import mask as codec


PILOT = ("FIT109_BOUNDARY_SIMPLE_POINT_PILOT_20261121/pilot.py",
         "d5f62c0996b67ee6b3becbea12ab8bc52db75d88bc28d6d91622b7fc6c11dbe9")
COMPARISON = ("FIT109_CLASSICAL_ATTRACTION_20261121/compare.py",
              "acb7e00e7dfd2c60e5afc4397e073a6fabcb6a4ed977f1c8eaf89115724b52ed")
ATTRACTION = ("FIT109_CLASSICAL_ATTRACTION_20261121/spatial_attraction.py",
              "8eb162a3bd544929843d7f2aeef174f1f1ba03a2c593b6bcda49289cd94ad3c6")
CLASSICAL_RESULT = "f78d431c749c4a58c3a14dd3ad2e248c5580093a6cbe59e47c4e2c35d6f7e7e6"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pinned(path, expected):
    data = path.read_bytes()
    require(sha256(data).hexdigest() == expected, "Digest mismatch for " + path.name)
    return data


def load(path, expected, name):
    obj = ModuleType(name)
    obj.__file__ = str(path)
    sys.modules[name] = obj
    exec(compile(pinned(path, expected), str(path), "exec"), obj.__dict__)
    return obj


def supervised(rows, coordinates, truth, static, learned):
    """Validate pair membership independently and attach truth utility on the native cell."""
    require(set(rows) == {"add", "drop"} and len(static) == len(learned), "Signed disagreement schema and count")
    result = {}
    for side, key in enumerate(("add", "drop")):
        static_pts = {int(coordinates[p[side],side]) for p in static}
        learned_pts = {int(coordinates[p[side],side]) for p in learned}
        require(len(static_pts) == len(learned_pts) == len(static), "Distinct equally sized signed endpoint sets")
        expected = list(zip(sorted(static_pts - learned_pts), sorted(learned_pts - static_pts), strict=True))
        observed = [(int(s),int(l)) for s,l,_ in rows[key]]
        require(observed == expected, "Pairing must preserve both sorted exclusive native endpoint sets")
        labels = {int(p):float(q) for p,q in zip(coordinates[:,side],truth[:,side],strict=True)}
        sign = 1 if side == 0 else -1
        pairs = []
        for s,l,x in rows[key]:
            x = np.asarray(x,dtype=np.float64)
            require(x.ndim == 1 and x.size and np.isfinite(x).all(), "Finite signed-pair feature vector")
            y = sign*(labels[int(l)]-labels[int(s)])
            require(abs(y) <= 1+1e-12, "Unit-cell signed fractional label")
            pairs.append([int(s),int(l),x.tolist(),y])
        total = sign*(sum(labels[p] for p in learned_pts)-sum(labels[p] for p in static_pts))
        require(abs(sum(v[3] for v in pairs)-total) <= 1e-10, "Exclusives recover the full signed intersection change")
        result[key] = pairs
    return result


def run(args):
    import torch
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" and not torch.cuda.is_initialized(), "Source CPU run required")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    os.environ["SEAL_FIT109_SOURCE_ROOT"] = str(args.root.resolve(strict=True))
    pilot = load(args.code/PILOT[0], PILOT[1], "_fit109_qualified_pilot")
    compare = load(args.code/COMPARISON[0], COMPARISON[1], "_fit109_qualified_classical_compare")
    gate = load(Path(__file__).with_name("gate_model.py"), args.gate_sha, "_source_disagreement_gate")
    for package, version in pilot.VERSIONS.items():
        require(importlib.metadata.version(package) == version, "Original package version for "+package)
    a = SimpleNamespace(private_root=args.root,code=args.code,helper=args.code/ATTRACTION[0],helper_sha256=ATTRACTION[1])
    m = pilot.sources(a)
    m.common.MODES = m.gray.MODES = m.pair.MODES = compare.MODES
    m.pair.GEOMETRY = (*compare.MODES, "oracle")
    ref = json.loads(pinned(args.code.parent/"results/FIT109_CLASSICAL_ATTRACTION_ACTUAL_SAFE_20261121.json",CLASSICAL_RESULT))
    history = json.loads(pilot.pinned(a.historical,pilot.INPUTS["historical"][1]))
    prior = json.loads(pilot.pinned(a.previous,pilot.INPUTS["previous"][1]))
    _,blocks = m.base.read_core(a)
    receipt,planned = m.pair.sealed_points(a,m.base,m.math,blocks)
    require(receipt["points_manifest_sha256"] == pilot.INPUTS["points_manifest"][1], "Original native endpoint inventory")
    gt,anns = m.pair.read_labels(a,m.base,blocks)
    slots,scope = m.pair.fixed_slots(a,m.base,blocks,planned,gt,anns)
    original, score_and_rank, observed, by_features, new_actions, count_table = compare.observe(slots,m)
    evidence, q_hash = {},sha256()
    raw_count = 0

    def capture(model, features, context, center, scale):
        nonlocal raw_count
        q = score_and_rank(model,features,context,center,scale)
        i = by_features[id(features)]
        slot,block = slots[i],blocks[i//100]
        image_id,rank,fold = int(block["image_id"]),i%100+1,int(slot["fold"])
        require(q.shape == features.shape[:2] and np.isfinite(q).all() and np.all((q>=0)&(q<=1))
                and slot["row"]["image_id"] == image_id and slot["group"] == block["group"]
                and planned[i//100]["fold"] == fold, "Original held-out candidate position and prediction")
        q_hash.update(np.asarray([image_id,rank,fold,len(q)],dtype="<i8").tobytes())
        q_hash.update(np.ascontiguousarray(q,dtype="<f4").tobytes())
        raw_count += len(q)
        if slot["truth"] is None:
            return q
        coordinates = slot["data"]["coordinates"]
        mask = codec.decode(slot["row"]["segmentation"]).astype(bool)
        y,x = np.divmod(coordinates,mask.shape[1])
        attraction = m.topology._initial(mask,y,x)/float(m.topology._WEIGHTS.sum())
        static,learned = new_actions[compare.STATIC][i],new_actions[compare.LEARNED][i]
        prepared = gate.pair_rows(coordinates,features,q,attraction,static,learned)
        vectors = np.concatenate((attraction[...,None],q[...,None],features),axis=2)
        require(vectors.shape==(len(q),2,16), "Static, OOF probability and 14 original point features")
        for side,key in enumerate(("add","drop")):
            ordinal = {int(cell):k for k,cell in enumerate(coordinates[:,side])}
            for s,l,xv in prepared[key]:
                expected = vectors[ordinal[int(l)],side]-vectors[ordinal[int(s)],side]
                expected = -expected if side else expected
                require(np.shape(xv)==(16,) and np.allclose(xv,expected,atol=2e-7,rtol=1e-6),
                        "Disagreement vector must use the same signed cells and original held-out probability")
        require(i not in evidence, "One supervised source prediction per native candidate")
        evidence[i] = supervised(prepared,coordinates,slot["truth"],static,learned)
        return q

    m.learner.probabilities = capture
    try:
        actions,training,usage = m.free.fit(slots,m.learner,m.math,prior)
    finally:
        m.learner.probabilities = original
    require(training == history["training"] == ref["training"] and scope == history["match_scope"] == ref["fixed_original_matches"]
            and usage == history["usage"] and len(observed) == len(by_features) == receipt["counts"]["supported_candidates"]
            and raw_count == receipt["counts"]["pairs"], "Five held-out source heads, training trace and complete endpoint scores")
    count_table[(0,0)] += len(slots)-len(observed)
    require({(r["learned_natural"],r["dynamic"]):r["candidates"] for r in ref["count_table"]} == dict(count_table),
            "Candidate-by-budget frequency table differs from the original source comparison")
    actions.update(new_actions)
    transitions,witness = m.common.threshold(slots,actions)
    witness_sha = sha256(b"".join(m.base.encoded(r) for r in witness)).hexdigest()
    require(transitions == ref["strict90_transitions"] and witness_sha == ref["transition_witness_sha256"],
            "Exact checksum of each historical GT/rank identity, edit count and threshold transition")
    for key in (compare.STATIC,compare.DYNAMIC,compare.LEARNED):
        n = [len(v) for v in actions[key]]
        require(sum(n) == ref["budgets"][key]["swaps"] and sum(bool(k) for k in n) == ref["budgets"][key]["edited"],
                "Original signed-ranking policy budget")

    counts,fold_counts,groups = Counter(),Counter(),set()
    h = sha256()
    with (args.output/"SOURCE_PAIRS_PRIVATE.jsonl").open("xb") as stream:
        for i,slot in enumerate(slots):
            if slot["raw"] is None:
                continue
            require((i in evidence) == (slot["truth"] is not None), "Unsupported native match or missing held-out readout")
            group,fold = slot["group"],int(slot["fold"])
            side = evidence.get(i,{"add":[],"drop":[]})
            row = dict(group=group,fold=fold,image_id=int(slot["row"]["image_id"]),native_rank=i%100+1,**side)
            data = m.base.encoded(row)
            stream.write(data)
            h.update(data)
            groups.add(group)
            fold_counts[str(fold)] += 1
            counts.update(rows=1,add_pairs=len(row["add"]),drop_pairs=len(row["drop"]),
                          add_candidates=bool(row["add"]),drop_candidates=bool(row["drop"]))
    require(counts["rows"] == scope["fixed_valid"] == 412 and len(groups) == 93 and len(fold_counts) == 5,
            "Complete source matched-candidate population")
    return dict(status="ATTRACTION_LOCAL_CORRECTION_FIT109_PAIRS_COMPLETE",FIT=True,DEV=False,TEST=False,
                target_labels_opened=False,source_fractional_labels_opened=True,detector_forwards=0,
                point_optimizer_updates=1000,source_images=len(blocks),source_groups=len(groups),
                source_candidates=len(slots),oof_scored_candidates=len(observed),oof_scored_endpoint_pairs=raw_count,
                supervised_endpoint_pairs=sum(len(slots[i]["truth"]) for i in evidence),
                rows_by_fold=dict(fold_counts),counts=dict(counts),source_pairs_sha256=h.hexdigest(),
                oof_q_float32_sha256=q_hash.hexdigest(),historical_412_match_witness_sha256=witness_sha,
                historical_training_reproduced=True,historical_transition_witness_exact=True,
                historical_attraction_source_result_sha256=CLASSICAL_RESULT,gate_sha256=args.gate_sha,
                input_sha256={k:v[1] for k,v in pilot.INPUTS.items()},points=receipt["counts"])


def main():
    p = argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ("root","code","protocol","output"):
        p.add_argument("--"+name,type=Path,required=True)
    for name in ("protocol-sha","gate-sha"):
        p.add_argument("--"+name,required=True)
    a = p.parse_args()
    pinned(a.protocol,a.protocol_sha)
    require(sys.flags.isolated and sys.dont_write_bytecode and not a.output.exists(),"Isolated fresh source-only phase")
    a.output.mkdir()
    start = perf_counter()
    result = run(a)
    result.update(protocol_sha256=a.protocol_sha,worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                  seconds=perf_counter()-start)
    raw = (json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n").encode()
    with (a.output/"RESULT_SAFE.json").open("xb") as stream:
        stream.write(raw)
    print(json.dumps(dict(status=result["status"],result_sha256=sha256(raw).hexdigest())),flush=True)


if __name__=="__main__":
    main()
