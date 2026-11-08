"""Native optimizer-state restoration and first-update evidence."""
from collections import Counter
from hashlib import sha256
import json
import math
import os
from pathlib import Path

import torch


START_STEP = 4464
UPDATES = 372
END_STEP = START_STEP + UPDATES
ORDER_SHA256 = "e4a6acd66823612a0dcabcd95b380e388da013adef52ea09c6158eda6c16a189"
GRADIENT_LIMIT = 256 * 1024**2


def need(ok, message):
    if not ok:
        raise ValueError(message)


def tensor_hash(digest, key, value):
    if value is None:
        digest.update(json.dumps([key, None]).encode() + b"\n")
        return
    cpu = value.detach().cpu().contiguous()
    digest.update(json.dumps([key, list(cpu.shape), str(cpu.dtype)]).encode() + b"\n")
    digest.update(memoryview(cpu.reshape(-1).view(torch.uint8).numpy()))


def tensor_dict_sha256(values):
    digest = sha256()
    for name, value in values.items():
        tensor_hash(digest, name, value)
    return digest.hexdigest()


def file_ref(path, maximum):
    digest = sha256()
    with path.open("rb") as stream:
        a = os.fstat(stream.fileno())
        need(0 < a.st_size <= maximum, "File exceeds admitted limit")
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
        b = os.fstat(stream.fileno())
        need((a.st_size, a.st_mtime_ns) == (b.st_size, b.st_mtime_ns), "File changed during hash")
    return dict(path=path.as_posix(), bytes=a.st_size, sha256=digest.hexdigest())


class ContinuationEMA:
    """Give the unchanged P77 loop a local count while the native EMA sees its E24 clock."""

    def __init__(self, average, initial_step):
        need(int(average.n_averaged.item()) == initial_step, "Exact archived EMA start count")
        self.average = average
        self.initial_step = initial_step

    @property
    def n_averaged(self):
        return self.average.n_averaged - self.initial_step

    def update_parameters(self, model):
        self.average.update_parameters(model)


def install_optimizer_history(model, optimizer, archive, arm):
    """Load the same E24 group metadata into both arms and old m/v/step into S only."""
    need(arm in ("S", "F"), "Expected explicit preserve or fresh arm")
    names = {id(p): n for n, p in model.named_parameters()}
    saved, = archive["optimizer_states"]
    scheduler, = archive["lr_schedulers"]
    live_groups = optimizer.param_groups
    empty = optimizer.state_dict()
    saved_groups = saved["param_groups"]
    need(len(live_groups) == len(saved_groups) == 28 and not empty["state"]
         and scheduler["last_epoch"] == START_STEP, "Exact group and scheduler admission")
    mapping = {}
    groups = []
    for index, (live, original, fresh) in enumerate(zip(live_groups, saved_groups, empty["param_groups"], strict=True)):
        need(original["params"] == fresh["params"]
             and set(original) == set(fresh) | {"initial_lr"} and "param_names" not in original,
             "Exact native positional optimizer groups")
        need(all(original[key] == val for key, val in fresh.items() if key not in ("params", "lr"))
             and original["initial_lr"] == fresh["lr"] == scheduler["base_lrs"][index]
             and original["lr"] == scheduler["_last_lr"][index], "Exact archived optimizer options and E24 LR")
        group_names = []
        for slot, param in zip(original["params"], live["params"], strict=True):
            need(slot not in mapping, "Duplicated optimizer slot")
            mapping[slot] = param
            group_names.append((slot, names[id(param)], list(param.shape), str(param.dtype)))
            old = saved["state"].get(slot)
            if old is not None:
                need(set(old) == {"step", "exp_avg", "exp_avg_sq"} and float(old["step"]) == START_STEP
                     and all(old[k].shape == param.shape and old[k].dtype == param.dtype for k in ("exp_avg", "exp_avg_sq")),
                     "Exact old AdamW step and moment shapes")
        groups.append((original["initial_lr"], original["weight_decay"], group_names))
    order = sha256(json.dumps(groups, separators=(",", ":")).encode()).hexdigest()
    need(order == ORDER_SHA256 and set(mapping) == set(range(544))
         and len(set(map(id, mapping.values()))) == 544
         and len(saved["state"]) == 543 and set(saved["state"]) < set(mapping),
         "E24 audited ordered parameter-name binding")
    history = saved["state"] if arm == "S" else {}
    optimizer.load_state_dict(dict(state=history, param_groups=saved_groups))
    for slot, param in mapping.items():
        stored = history.get(slot)
        loaded = optimizer.state.get(param)
        if stored is None:
            need(loaded is None, "An originally empty optimizer position acquired state")
        else:
            need(loaded is not None and set(loaded) == set(stored)
                 and all(torch.equal(loaded[k].detach().cpu(), v) for k, v in stored.items()),
                 "Optimizer state did not restore exact moment/step values")
    need(len(optimizer.state) == (543 if arm == "S" else 0)
         and [g["lr"] for g in optimizer.param_groups] == scheduler["_last_lr"],
         "Exact installed history scope and shared applied E24 LR")
    absent, = set(mapping) - set(saved["state"])
    return dict(parameter_group_order_sha256=order, initial_state_slots=len(optimizer.state),
                initial_adam_step=START_STEP if arm == "S" else 0, archived_unused_slot=absent,
                applied_group_lrs=[g["lr"] for g in optimizer.param_groups])


class FirstStepTrace:
    """Observe the real first accumulated update after native gradient clipping."""

    def __init__(self, model, optimizer, arm, initial_step, gradient_output=None):
        self.parameters = dict(model.named_parameters())
        self.arm = arm
        self.initial_step = initial_step
        self.gradient_output = gradient_output
        self.input_digest = sha256()
        self.input_count = 0
        self.first = None
        self.before = None
        self.terminal = None
        self.pre_handle = optimizer.register_step_pre_hook(self._before)
        self.post_handle = optimizer.register_step_post_hook(self._after)

    def observe_input(self, samples, targets, index):
        if self.input_count >= 4:
            return
        self.input_digest.update(json.dumps([self.input_count, index]).encode() + b"\n")
        tensor_hash(self.input_digest, "samples.tensors", samples.tensors)
        tensor_hash(self.input_digest, "samples.mask", samples.mask)
        need(len(targets) == 1, "Expected one realized FIT input per microbatch")
        for key, value in sorted(targets[0].items()):
            tensor_hash(self.input_digest, key, value)
        self.input_count += 1

    def _before(self, optimizer, _args, _kwargs):
        need(self.input_count == 4 and self.first is None, "Complete real first accumulation window")
        gradients = {name: p.grad.detach().to(device="cpu", copy=True)
                     for name, p in self.parameters.items() if p.grad is not None}
        need(bool(gradients) and all(g.is_floating_point() and bool(torch.isfinite(g).all()) for g in gradients.values()),
             "Finite actual first clipped gradients")
        history = Counter(int(v["step"].item()) for v in optimizer.state.values())
        need((self.arm == "S" and set(history) == {self.initial_step}) or (self.arm == "F" and not history),
             "Actual first-step optimizer history differs from arm")
        self.before = {name: self.parameters[name].detach().to(device="cpu", copy=True) for name in gradients}
        self.first = dict(arm=self.arm, first_microbatches=self.input_count,
                          first_input_sha256=self.input_digest.hexdigest(),
                          clipped_grad_sha256=tensor_dict_sha256(gradients),
                          clipped_grad_l2=math.sqrt(math.fsum(float(g.double().square().sum()) for g in gradients.values())),
                          clipped_grad_tensors=len(gradients), applied_group_lrs=[g["lr"] for g in optimizer.param_groups],
                          adam_counts_before={str(k): v for k, v in sorted(history.items())})
        if self.gradient_output is not None:
            need(sum(g.numel() * g.element_size() for g in gradients.values()) < GRADIENT_LIMIT - 1024**2,
                 "Bounded private first clipped gradient tensor bytes")
            with self.gradient_output.open("xb") as stream:
                torch.save(dict(schema="RFDETR_FIRST_CLIPPED_GRAD_V1", tensors=gradients), stream)
                stream.flush()
                os.fsync(stream.fileno())
            self.first["gradient_ref"] = file_ref(self.gradient_output, GRADIENT_LIMIT)
        self.pre_handle.remove()

    def _after(self, optimizer, _args, _kwargs):
        need(self.first is not None and self.before is not None, "First optimizer update ran without pre-step evidence")
        l2_squared = {}
        for name, before in self.before.items():
            after = self.parameters[name].detach().cpu()
            l2_squared[name] = float((after.double() - before.double()).square().sum())
        history = Counter(int(v["step"].item()) for v in optimizer.state.values())
        self.terminal = dict(self.first, raw_update_l2=math.sqrt(math.fsum(l2_squared.values())),
                             segmentation_head_update_l2=math.sqrt(math.fsum(v for name, v in l2_squared.items()
                                                                            if name.startswith("segmentation_head."))),
                             adam_counts_after={str(k): v for k, v in sorted(history.items())})
        need(self.terminal["raw_update_l2"] > 0 and math.isfinite(self.terminal["raw_update_l2"]),
             "First real AdamW step did not change weights")
        self.before = None
        self.post_handle.remove()

    def result(self):
        need(self.terminal is not None, "The first actual AdamW step did not complete")
        return self.terminal


def compare_first(saved, fresh):
    need(saved["arm"] == "S" and fresh["arm"] == "F", "Expected first traces from opposite arms")
    same = all(saved[k] == fresh[k] for k in ("first_input_sha256", "first_microbatches", "applied_group_lrs"))
    return dict(same_initial_forward_conditions=same, same_clipped_gradient=saved["clipped_grad_sha256"] ==
                fresh["clipped_grad_sha256"] and saved["clipped_grad_tensors"] == fresh["clipped_grad_tensors"])
