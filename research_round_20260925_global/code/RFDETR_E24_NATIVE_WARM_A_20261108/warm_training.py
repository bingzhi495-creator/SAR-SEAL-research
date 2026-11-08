"""Shared training tape and actual equal-budget RF-DETR warm-scout loop."""
import hashlib
import json
import random
import time
from contextlib import nullcontext

import numpy as np
import torch

from risk_criterion import pixel_center_bce


def training_tape(seed, aligned_positions):
    if type(seed) is not int or aligned_positions<=0 or aligned_positions%4:
        raise ValueError("Expected integer seed and complete native accumulation windows")
    return [
        [(int(slot),seed*100000+1000*epoch+pos) for pos,slot in enumerate(
            np.random.Generator(np.random.PCG64(seed+epoch)).permutation(aligned_positions))]
        for epoch in range(2)
    ]


def train_epochs(model,criterion,native_api,point_sample,optimizer,ema,fetch,tape,arm,seed,device):
    parameters=[p for group in optimizer.param_groups for p in group["params"]]
    if len({id(p) for p in parameters})!=len(parameters) or set(map(id,parameters))!={
            id(p) for p in model.parameters() if p.requires_grad}:
        raise ValueError("Native groups must cover every trainable parameter exactly once")
    ctx=nullcontext()
    if arm!="A":
        generator=torch.Generator(device=device).manual_seed(seed*100000+90000)
        ctx=pixel_center_bce(criterion,native_api,point_sample,arm,generator)
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    start=time.perf_counter()
    visit_digest=hashlib.sha256()
    completed=updates=0
    epoch_stats=[]
    with ctx:
        for epoch,items in enumerate(tape):
            mean_loss=torch.zeros((),device=device,dtype=torch.float32)
            targets_seen=0
            for position,(slot,micro_seed) in enumerate(items):
                random.seed(micro_seed)
                np.random.seed(micro_seed)
                torch.manual_seed(micro_seed)
                samples,targets,index=fetch(slot)
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    outputs=model(samples,targets)
                    losses=criterion(outputs,targets)
                    loss=torch.stack([v*criterion.weight_dict[k] for k,v in losses.items()
                                      if k in criterion.weight_dict]).sum()
                (loss/4).backward()
                mean_loss+=loss.detach().float()
                targets_seen+=len(targets[0]["labels"])
                visit_digest.update(f"{epoch}/{position}/{slot}/{index}/{micro_seed}\n".encode())
                completed+=1
                if (position+1)%4==0:
                    torch.nn.utils.clip_grad_norm_(parameters,0.1,error_if_nonfinite=True)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    ema.update_parameters(model)
                    updates+=1
                    if updates%32==0:
                        print(json.dumps(dict(event="WARM_BD_TRAINING_PREFIX",arm=arm,updates=updates,
                            microbatches=completed,train_seconds=time.perf_counter()-start)),flush=True)
            row=dict(epoch=epoch+1,microbatches=completed,updates=updates,targets_after_augmentation=targets_seen,
                     mean_total_loss=float(mean_loss.item()/len(items)))
            epoch_stats.append(row)
            print(json.dumps(dict(event="WARM_BD_EPOCH_COMPLETE",arm=arm,**row)),flush=True)
    torch.cuda.synchronize(device)
    count=int(ema.n_averaged.item())
    if completed!=1488 or updates!=372 or count!=updates:
        raise RuntimeError("Incomplete actual tape, optimizer or EMA")
    return dict(microbatches=completed,updates=updates,ema_updates=count,epoch_stats=epoch_stats,
        visit_sha256=visit_digest.hexdigest(),train_seconds=time.perf_counter()-start,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
