"""Fit an anchored quota-only calibration on grouped out-of-fold FIT rows."""
from argparse import ArgumentParser
from collections import Counter
import json
import os
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import HERE, LOCK, QUOTA, WRITER, closed, closeout, load, need, pin, settings, write
import numpy as np
import torch
from pycocotools import mask

PREVIOUS_FIT = "ae1b165a111a6b88939eb37b5741421259f0f39531c859a7c7663ad9116d1b05"
PREVIOUS_WORKER = "011914954482cd505343a6421296ae754b1ea3098f5ad3fe37cdaa14e3c05fd3"
PREVIOUS_EXIT = "b0d837ca9fdeb4b21bb6c4bb83ecda130b86696d751b4cad309a41379887991b"
BANK = "5419c9179f1ca82bf21436049450e623725c7e45b1cb2b7b19c2541e57e26fb7"
BANK_RESULT = "4872b63ee24c8021e09612314161c691006607dceb3ea92a2ce27690016836e4"
BANK_EXIT = "52e572c8916cafff8297f2308b95aa364fd6de7329e747f1e54275a32664548d"
KEYS = ("capture","points","labels","prior_fit","previous","action_core","source_core","attraction","writer","bank")


def fit_reader(source):
    decode = "raw = [json.loads(line) for line in pinned("
    need(source.count(decode) == 1, "FIT-only reader insertion")
    source = source.replace(decode, "raw = [json.loads(line) if j in fit_ordinals else None for j,line in enumerate(pinned(")
    end = 'acceptance["predictions_sha256"]).splitlines()]'
    need(source.count(end)==1, "Exact prelabel combined candidate stream")
    source = source.replace(end,'acceptance["predictions_sha256"]).splitlines())]')
    return source.replace("    raw = [json.loads(line) if j", '    fit_ordinals = {r["native_archive_ordinal"]-1 for r in index}\n    raw = [json.loads(line) if j', 1)


def instrument_fit(source):
    anchor = '            packet_digest.update(f"{int(iid)}|{rid}|{k}|{len(weights)}\\n".encode("ascii"))'
    need(source.count(anchor)==1, "Exact qualified dual author association before recording geometry")
    callback = '            measure(iid, rid, stock, raw_stock, shared[rid], static[qlo:qhi], target)'
    return source.replace(anchor,callback+"\n"+anchor)


def run(cfg, out):
    started = perf_counter()
    orig = cfg.previous/"fit_action.py"
    parent,_ = closed(cfg.previous/"actual/RESULT_SAFE.json",cfg.previous/"observed_fit/TERMINAL.json",
                      orig,PREVIOUS_FIT,PREVIOUS_EXIT,PREVIOUS_WORKER)
    bank, bank_sha = closed(cfg.bank/"actual/RESULT_SAFE.json",cfg.bank/"observed_fit/TERMINAL.json",
                            cfg.bank/"fit_banks.py",BANK_RESULT,BANK_EXIT,BANK)
    need(parent["source_gate_labels_opened"] is False and parent["historical_test_opened"] is False
         and bank["source_gate_points_opened"] is bank["source_gate_labels_opened"] is bank["historical_test_opened"] is False,
         "Closed training-only source parents")
    old = load(orig,PREVIOUS_WORKER,instrument_fit)
    retry = load(cfg.prior_fit,old.PRIOR_FIT,fit_reader)
    action = load(cfg.action_core,old.ACTION_CORE)
    occupancy = load(cfg.source_core,old.OCCUPANCY_CORE)
    attraction = load(cfg.attraction,old.ATTRACTION)
    cache = []
    def measure(iid, rank, stock, native, truths, static, target):
        areas = np.array([int(mask.area(t)) for t in truths])
        hits = np.array([int(mask.area(mask.merge([native,t],intersect=True))) for t in truths])
        area = int(stock.sum())
        need(area==int(mask.area(native)) and np.all((0<hits)&(hits<=np.minimum(area,areas))),
             "Native and full reference geometry")
        cache.append((iid,rank,area,areas,hits,old.canonical(static),target))
    old.measure = measure
    packets,counts,packet_sha = old.fit_packets(cfg,retry,action,attraction)
    need(counts==parent["counts"]==bank["counts"] and packet_sha==parent["fit_packet_sha256"]==bank["fit_packet_sha256"]
         and len(cache)==len(packets)==848,"The original 848 exact packet tensors and labels")
    bank_core = load(cfg.bank/"fit_banks.py",BANK)
    predictions, folds, fold_sha, prediction_sha = bank_core.crossfit_occupancy(packets,action,occupancy,cfg.source_core)
    need(fold_sha==bank["out_of_fold_group_sha256"] and prediction_sha==bank["out_of_fold_probabilities_sha256"],
         "Previously admitted five-fold BCE byte parity")

    q = load(HERE/"quota.py",QUOTA)
    writer = load(cfg.writer,WRITER)
    group_images = {g:{p.image_id for p in packets if p.group_id==g} for g in {p.group_id for p in packets}}
    image_counts = Counter(p.image_id for p in packets)
    data = [[],[]]
    advance = Counter()
    ys, weights = [],[]
    for packet,p,(iid,rank,area,areas,hits,static,target) in zip(packets,predictions,cache,strict=True):
        need(iid==packet.image_id and np.array_equal(target,packet.targets),"Exact candidate area and label order")
        rows = q.panel(writer,p,static,packet.k,area)
        old_choice = writer.select(p,static,packet.k,area)
        need(rows[0]["action"]==old_choice["fixed_iou"]["action"]
             and max(rows,key=lambda r:r["score"])["action"]==old_choice["variable_iou"]["action"],
             "Exact original fixed and raw variable FIT quota choices")
        if len(rows)==1:
            advance["fixed_only"]+=1
            continue
        raw,x = q.features(rows,packet.k,area,False)
        raw2,control = q.features(rows,packet.k,area,True)
        need(np.array_equal(raw,raw2) and np.array_equal(x[:,1:],control[:,1:]),"Matched raw offset and exact geometry features")
        data[0].append(x[1:])
        data[1].append(control[1:])
        ys.append((q.true_iou(rows,area,areas,hits,target.astype(np.int64))-raw)[1:])
        count = len(rows)-1
        weights.append(np.full(count,1/(len(group_images[packet.group_id])*image_counts[iid]*count)))
        advance.update(fit_candidates=1,nonzero_splits=count,
                       shuffled_feature_changed=int(np.sum(x[:,0]!=control[:,0])))
    residual,weight = np.concatenate(ys),np.concatenate(weights)
    models = {name:q.fit(np.concatenate(rows),residual,weight)
              for name,rows in zip(("calibrated_quota","reversed_quota"),data,strict=True)}
    out.mkdir()
    model_sha = write(out/"CALIBRATION_PRIVATE.json",models)
    closeout(out/"RESULT_SAFE.json",Path(__file__),"SPSD_PSEG_QUOTA_FIT_COMPLETE",
             fit_only=True,source_gate_labels_opened=False,source_gate_points_opened=False,historical_target_opened=False,
             historical_test_opened=False,detector_forwards=0,calibration_parameters_per_arm=4,ridge_lambda=.02,
             oof_optimizer_updates=5000,oof_fit_count=len(folds),oof_group_sha256=fold_sha,oof_probabilities_sha256=prediction_sha,
             fit_packet_sha256=packet_sha,prior_fit_result_sha256=PREVIOUS_FIT,bank_fit_result_sha256=bank_sha,
             calibrator_sha256=model_sha,fit_counts=counts,training_rows=dict(advance),
             elapsed_seconds=perf_counter()-started)


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument("--inputs",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    need(sys.flags.isolated and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES")=="-1"
         and all(os.environ.get(v)=="1" for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"))
         and torch.get_num_threads()==1 and not a.out.exists(), "Fresh single-threaded CPU FIT")
    pin(HERE/"PRELOCK.md",LOCK)
    run(settings(a.inputs,KEYS),a.out)

if __name__ == "__main__":
    main()
