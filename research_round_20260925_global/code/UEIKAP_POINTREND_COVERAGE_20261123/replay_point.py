"""Prepare and independently rerender the original coarse and PointRend whole-RoI pastes on UEIKAP."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType, SimpleNamespace


PROTOCOL = "e0cad345351d1cfc23b6777d84abfeffd9fc7e2c82d69242813356af33715225"
FIT_WEIGHT = "7401de08367ef31b909ea9c92daa27be02d4efb04460acbf96eab33648fdd7cc"
OLD = ("ATTRACTION_SIGNED_SIDES_20261123/replay_ueikap.py",
       "05d8b47f3d4b9351c1ffd12ac32b3f4f6e91b05daf87a1c6d148536e67f97492")
CAPTURE = ("UEIKAP_RFDETR_CARRIER_20261122/capture.py",
           "89c3294d79e1d69530401a4bf08e3400d52fbc6b77bc7967029cbd1fbbdbb116")
RENDER = ("INSTANCE_RSDD_POINTREND_TRANSFER_20261122/replay_pointrend.py",
          "9368956a7d47e502c005a6adc1ab39edaca18cd5b3780002073ba672732a2b2f")
PREPARED = "UEIKAP342_POINTREND_COVERAGE_EDITS_PREPARED"
VERIFIED = "UEIKAP342_POINTREND_COVERAGE_EDITS_VERIFIED"
BASIS = "UEIKAP_POINTREND_COVERAGE_BASIS_COMPLETE"
BASIS_WORKER = "c898fc228c2ccc1bc4babefd17fd77e121c7a683d541d2f96d4c8e0e95ca3d51"
NAMES = {"C":"coarse_paste", "P":"point_paste"}
EDITS = "EDITS_PRIVATE.jsonl"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def pin(path, value):
    raw = path.read_bytes()
    require(sha256(raw).hexdigest() == value, "Changed input " + path.name)
    return raw


def source(root, spec):
    path, digest = spec
    target = root / path
    module = ModuleType(target.stem)
    module.__file__ = str(target)
    exec(compile(pin(target, digest), str(target), "exec"), module.__dict__)
    return module


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",",":"), allow_nan=False)+"\n").encode()


def guard():
    def audit(event, args):
        if event in {"socket.connect","socket.getaddrinfo","socket.gethostbyname"}:
            raise PermissionError("Offline mask replay")
        if event == "open" and isinstance(args[0], (str,bytes)):
            p = Path(os.fsdecode(args[0]))
            if (p.suffix.lower() in {".jpg",".jpeg",".png",".tif",".tiff"} or
                p.name.lower() in {"labels_roster.json","train.json","test.json"} or
                any(part.lower() in {"labels","annotations","annotation_coco"} for part in p.parts)):
                raise PermissionError("Image and label bodies excluded from mask replay")
    sys.addaudithook(audit)


def observed(directory, path, phase, status):
    tb = path.read_bytes()
    obs = json.loads(tb)
    require(obs["phase"] == phase and type(obs["actual_exit"]) is int and obs["actual_exit"] == 0 and
            obs["output_footer_verified"] is True, "Expected closed observer: " + phase)
    digest = obs["result_sha256"]
    result = json.loads(pin(directory/"RESULT_SAFE.json", digest))
    require(result["status"] == status and result["protocol_sha256"] == PROTOCOL and
            obs["worker_sha256"] == result["worker_sha256"], "Worker or protocol binding: " + phase)
    return result, digest, sha256(tb).hexdigest()


def load_inputs(a):
    old = source(a.code, OLD)
    lineage, counts, _ = old.old_inputs(a.old_root)
    result, basis_sha, basis_term = observed(a.basis, a.basis_terminal, "basis", BASIS)
    require(result["GT_opened"] is result["target_labels_opened"] is False and result["images"] == 342
            and result["worker_sha256"] == BASIS_WORKER and result["groups"] == 13
            and result["counts"] == counts and result["checked_all_query_planes"] == 342*164
            and result["model_forwards"] == 342 and result["optimizer_updates"] == 0
            and result["exact_original_candidates_and_masks"] is result["exact_original_eligible_logits"]
            is result["exact_all_query_basis_contraction"] is True, "Exact label-free target basis required")
    for key, earlier in (("original_capture_sha256","capture_result_sha256"),
                         ("verified_capture_sha256","capture_readback_result_sha256"),
                         ("original_capture_terminal_sha256","capture_terminal_sha256"),
                         ("verified_capture_terminal_sha256","capture_readback_terminal_sha256"),
                         ("predictions_sha256","predictions_sha256"),("audit_sha256","audit_sha256"),
                         ("checkpoint_sha256","checkpoint_sha256")):
        require(result[key] == lineage[earlier], "Basis native provenance differs: " + key)
    items = json.loads(pin(a.old_root/"intake/images_roster.json", lineage["manifest_sha256"]))
    index = json.loads(pin(a.basis/"INDEX_PRIVATE.json", result["index_sha256"]))
    rows = old.records(a.old_root/"capture/CANDIDATE_AUDIT_PRIVATE.jsonl", lineage["audit_sha256"])
    native = old.records(a.old_root/"capture/PREDICTIONS_PRIVATE.jsonl", lineage["predictions_sha256"])
    population = []
    # Fully exhaust both streams (including their SHA checks) before rendering any row.
    require(items["expected_ids"] == list(range(1,343)) and len({item["acquisition_group"] for item in items["items"]})==13,
            "Original full 342-image/13-family roster")
    for i, (item, ref, row, pred) in enumerate(zip(items["items"], index, rows, native, strict=True),1):
        require(item["image_id"] == ref["image_id"] == row["image_id"] == pred["image_id"] == i and
                ref["ordinal"] == i and ref["rgb_sha256"] == row["rgb_sha256"] == item["rgb_sha256"] and
                row["image_sha256"] == item["image_ref"]["sha256"] and row["original_hw"] == item["original_hw"] == [512,512]
                and row["source_image_name"] == f"{i:06d}.png" and ref["eligible_rows"] == len(row["logit_ranks"])
                and ref["spatial"]["path"] == f"spatial/{i:06d}.npy", "Aligned native image, logits and projected basis")
        population.append((row,pred,ref))
    require(len(population) == 342 and counts["mapped"] == 33975 and counts["eligible"] == 17537,
            "Archived population and area eligibility")
    return population, counts, dict(old_lineage=lineage, basis_result_sha256=basis_sha, basis_terminal_sha256=basis_term,
                                    basis_index_sha256=result["index_sha256"], fit_weight_sha256=FIT_WEIGHT,
                                    render_source_sha256=RENDER[1])


def initialize(code, native_dir, basis_dir, weights):
    global worker
    guard()
    import numpy as np
    import torch
    require(sys.version_info[:3] == (3,10,13) and np.__version__ == "1.26.4" and torch.__version__ == "2.10.0+cu130"
            and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" and not torch.cuda.is_initialized(),
            "Original qualified single-thread CPU numerical runtime")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    old = source(code, RENDER)
    deps = {Path(k).stem:source(code,(k,v)) for k,v in old.PINNED.items()}
    adapter = deps["roi_point_head"]
    head = old.restore_head(adapter, pin(weights,FIT_WEIGHT))
    seconds = Counter()
    def timed(fn, label):
        def call(*args, **kwargs):
            started = perf_counter()
            value = fn(*args, **kwargs)
            seconds[label] += perf_counter()-started
            return value
        return call
    adapter.coarse_roi_logits = timed(adapter.coarse_roi_logits, "shared_coarse_query_sampling")
    subdivide = deps["pointrend_subdivision"]
    subdivide.subdivide_binary_mask = timed(subdivide.subdivide_binary_mask, "point_subdivision")
    infer = deps["infer_point_policies"]
    paste_rle = infer._paste_rle
    def timed_paste(adapter, logits, *args):
        hw = tuple(logits.shape[-2:])
        require(hw in ((7,7),(224,224)), "Original coarse and refined RoI chart dimensions")
        key = "coarse_paste" if hw == (7,7) else "point_paste"
        return timed(paste_rle,key)(adapter,logits,*args)
    infer._paste_rle = timed_paste
    worker = SimpleNamespace(old=old, io=source(code, CAPTURE), native=native_dir, basis=basis_dir,
            infer=infer, adapter=adapter, subdivide=subdivide, budget=deps["matched_budget"], head=head,
            seconds=seconds)


def render(work):
    import numpy as np
    from pycocotools import mask as codec

    row, pred, ref = work
    worker.seconds.clear()
    iid = row["image_id"]
    hw = tuple(row["original_hw"])
    ranks = row["native_ranks"]
    qids = row["query_ids"]
    require(pred["native_ranks"] == ranks == sorted(set(ranks)) and len(set(qids)) == len(qids)
            and all(1 <= r <= 100 for r in ranks), "Exact native top-100 identities")
    planes = worker.io.read_array(worker.native, row["logits_ref"])
    spatial = worker.io.read_array(worker.basis,ref["spatial"])
    require(planes.shape == (len(row["logit_ranks"]),144,144) and spatial.shape == (256,144,144),
            "All eligible raw query planes and detached projected spatial basis")
    selected, positions, eligible = [], [], []
    totals = Counter(images=1, mapped_candidates=len(ranks), area_eligible=0, rendered=0, degenerate=0,
                     area_below_32=0, area_at_least_1024=0, point_evaluations=0)
    for name in NAMES.values():
        for field in ("candidates","adds","drops","empty"):
            totals[name+"_"+field] = 0
    empty_coords = np.empty((0,2), dtype=np.int64)
    for pos, (rank,qid,cand,detection) in enumerate(zip(ranks,qids,row["candidates"],pred["coco"],strict=True),1):
        rle = detection["segmentation"]
        area = int(codec.area(rle))
        require(cand["position"] == pos and cand["native_rank"] == rank and cand["query_id"] == qid and
                detection["image_id"] == iid and detection["category_id"] == 1 and detection["score"] == cand["score"]
                and area == cand["mask_area"] and rle["size"] == list(hw), "Native compact position/score/geometry")
        if not 32 <= area < 1024:
            totals["area_below_32" if area < 32 else "area_at_least_1024"] += 1
            continue
        slot = len(eligible)
        eligible.append((rank,qid))
        totals["area_eligible"] += 1
        box = np.asarray(cand["box_xyxy"], dtype=np.float32)
        require(box.shape == (4,) and np.isfinite(box).all() and (box>=0).all() and (box<=512).all() and
                (box[2:]>=box[:2]).all(), "Original clipped native RoI")
        if (box[2:] == box[:2]).any():
            totals["degenerate"] += 1
            continue
        # The C/P path uses only raw query logits and projected features. Endpoint
        # projections F/B/J are separate readouts and do not enter either paste.
        selected.append(dict(image_id=iid,rank=rank,query_id=qid,rle=rle,box_xyxy01=box/np.float32(512),
                             raw_logits=planes[slot],coordinates=empty_coords,k=0))
        positions.append(pos)
    require(eligible == list(zip(row["logit_ranks"],row["logit_query_ids"],strict=True)),
            "Original eligible native logit rank ordering")
    out, summary = worker.infer.infer_image(worker.head,spatial,selected,adapter=worker.adapter,render=worker.subdivide,
                                           budget=worker.budget,image_hw=hw,chunk_size=12)
    require(summary["fallback"] == 0 and len(out) == summary["rendered"] == len(selected)
            and summary["point_evaluations"] == 3136*len(selected), "Original PointRend four-stage census")
    totals["rendered"] = len(selected)
    totals["point_evaluations"] = summary["point_evaluations"]
    changes = {}
    for pos, item in zip(positions, selected, strict=True):
        choices = {}
        stock = codec.decode(item["rle"]).astype(bool)
        outputs = out[(iid,item["rank"],item["query_id"])]
        for alias,name in NAMES.items():
            paste = outputs[alias]
            adds, drops, empty = worker.old.counts_for_paste(stock,paste,outputs["diagnostics"][alias])
            if adds or drops:
                choices[name] = dict(rle=paste,adds=adds,drops=drops)
                totals[name+"_candidates"] += 1
                totals[name+"_adds"] += adds
                totals[name+"_drops"] += drops
                totals[name+"_empty"] += int(empty)
        if choices:
            changes[str(pos)] = choices
    return dict(image_id=iid,changes=changes), totals, dict(worker.seconds)


def run(a):
    pin(a.protocol, PROTOCOL)
    pin(a.fit_weights, FIT_WEIGHT)
    population, original_counts, lineage = load_inputs(a)
    prepare = None
    prepare_sha = prepare_term = None
    if a.phase == "verify-edits":
        prepare, prepare_sha, prepare_term = observed(a.prepare, a.prepare_terminal, "prepare-edits", PREPARED)
        require(all(prepare.get(k) == v for k,v in lineage.items()) and
                prepare["worker_sha256"] == sha256(Path(__file__).read_bytes()).hexdigest() and
                prepare["target_labels_opened"] is False and prepare["optimizer_updates"] == 0,
                "Independent verifier requires this exact preparation and lineage")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    totals, seconds, digest = Counter(), Counter(), sha256()
    a.output.mkdir()
    mode = ((a.prepare/EDITS).open("rb") if prepare else (a.output/EDITS).open("xb"))
    with mode as stream, ProcessPoolExecutor(max_workers=a.workers, initializer=initialize,
            initargs=(a.code,a.old_root/"capture",a.basis,a.fit_weights)) as pool:
        for i,(row,counters,timings) in enumerate(pool.map(render,population,chunksize=2),1):
            require(row["image_id"] == i, "Ordered render output")
            raw = encoded(row)
            if prepare:
                require(stream.readline() == raw, "Independent per-image re-render differs")
            else:
                stream.write(raw)
            digest.update(raw)
            totals.update(counters)
            seconds.update(timings)
            if i%50 == 0:
                print(json.dumps(dict(rendered_images=i)), flush=True)
        if prepare:
            require(stream.read(1) == b"" and prepare["edits_sha256"] == digest.hexdigest() and
                    prepare["counts"] == dict(totals), "Independent ledger bytes or complete census differ")
    require(totals["images"] == 342 and totals["mapped_candidates"] == original_counts["mapped"]
            and totals["area_eligible"] == original_counts["eligible"] == totals["rendered"] + totals["degenerate"]
            and totals["mapped_candidates"] == totals["area_eligible"] + totals["area_below_32"] + totals["area_at_least_1024"]
            and totals["point_evaluations"] == 3136*totals["rendered"], "All original candidates accounted for")
    return dict(status=VERIFIED if prepare else PREPARED, phase=a.phase, images=342, groups=13,
        arms=["N","C","P"], counts=dict(totals), edits_sha256=digest.hexdigest(),
        coarse_chart_hw=[7,7], refined_chart_hw=[224,224], summed_worker_operation_seconds=dict(seconds),
        prepare_result_sha256=prepare_sha, prepare_terminal_sha256=prepare_term, target_labels_opened=False,
        GT_opened=False, optimizer_updates=0, detector_forwards=0, exact_independent_render=bool(prepare),
        protocol_sha256=PROTOCOL, worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest(), **lineage)


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("phase",choices=("prepare-edits","verify-edits"))
    for key in ("code","old-root","basis","basis-terminal","fit-weights","protocol","output"):
        p.add_argument("--"+key,type=Path,required=True)
    for key in ("prepare","prepare-terminal"):
        p.add_argument("--"+key,type=Path)
    p.add_argument("--protocol-sha256",required=True)
    p.add_argument("--workers",type=int,choices=range(1,9),default=4)
    a = p.parse_args()
    require(a.protocol_sha256 == PROTOCOL and not a.output.exists(), "Locked protocol and fresh output directory")
    require((a.prepare is not None)==(a.phase=="verify-edits") and
            (a.prepare_terminal is not None)==(a.phase=="verify-edits"), "Supply preparation only for verification")
    for key,path in vars(a).items():
        if isinstance(path,Path):
            setattr(a,key,path.resolve(strict=key!="output"))
    guard()
    started = perf_counter()
    result = run(a)
    result["wall_seconds"] = perf_counter()-started
    data = encoded(result)
    (a.output/"RESULT_SAFE.json").write_bytes(data)
    print(json.dumps(dict(status=result["status"],result_sha256=sha256(data).hexdigest())),flush=True)


if __name__ == "__main__":
    main()
