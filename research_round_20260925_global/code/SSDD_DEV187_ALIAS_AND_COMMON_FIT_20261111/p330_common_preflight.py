"""Qualify the fixed common-support SSDD A/B dataset path on CPU only."""

import argparse
from hashlib import sha256
from importlib import import_module
from io import StringIO
import json
import os
from pathlib import Path
import random
import sys
from time import perf_counter
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


STATUS = "P330_SSDD_FIXED_COMMON_CPU_PREFLIGHT_COMPLETE"
SEED, FIT_COUNT, VISITS = 5201, 741, 32
REPLAY = "HRSID_FIT_RESIZE_PHASE_REPLAY_20261109"
TRANSFER = "SSDD_MASK_PHASE_TRANSFER_20261110"
PINS = {
    "entry": (f"{TRANSFER}/ssdd_windows_phase_entry.py",
              "e29edeb0a3aff7c871123deee7c0c198ef5913fd33e7c016434a671d186796d3"),
    "trace_resize": (f"{REPLAY}/trace_resize.py",
                     "902a0e3640a319ba729172847f49ce582f3c0e3954593862ed94971727dc309f"),
    "common_mask_supervision": (f"{REPLAY}/common_mask_supervision.py",
                                "59ea82e24353eeb4a14115058d5115333a4e7ea517a0abef581bee137a9a3a57"),
    "paired_fit_transform": (f"{REPLAY}/paired_fit_transform.py",
                             "64f3f63c6ac89b66578f0ada42fa3eb8b4db3b2f1ae2ea9113debe6ab9b270e0"),
    "common_fit_binding": (f"{REPLAY}/common_fit_binding.py",
                           "2bfc4d6039b28a632693131126120cc2502c144b7cf6564de5dc4d765a9e4fba"),
}


def need(condition, why):
    if not condition:
        raise ValueError(why)


def prospective_positions():
    return tuple(random.Random(SEED).sample(range(FIT_COUNT), VISITS))


def compare_common(native, center):
    need(native["visits"] == center["visits"] == VISITS and native["ledger"] == center["ledger"],
         "Exact common ordered-consumption ledger required in both arms")
    need(native["changed"] == center["changed"] > 0 and native["chosen_sha"] != center["chosen_sha"],
         "Common support must include real differing native/center masks")
    need(native["crop"] == center["crop"] and 0 < native["crop"] < VISITS,
         "Exact common replay must exercise both crop and direct resize")
    return True


def pinned(code, key):
    relative, digest = PINS[key]
    path = code / relative
    need(path.is_absolute() and path.resolve(strict=True) == path and 0 < path.stat().st_size <= 64*1024,
         "Canonical exact ordinary released source required")
    raw = path.read_bytes()
    need(sha256(raw).hexdigest() == digest, "Exact ordinary released dependency digest required: "+key)
    mod = ModuleType("_p330_"+key)
    mod.__file__ = str(path)
    exec(compile(raw,str(path),"exec"),mod.__dict__)
    return mod


def validate_selected(torch, paired, image, target, audit, mode):
    need(image.device.type == "cpu" and image.dtype == torch.float32 and image.shape == (3,576,576)
         and set(target) == paired.PREPARED_FIELDS and target["masks"].dtype == torch.bool
         and target["labels"].dtype == torch.int64 and target["boxes"].dtype == torch.float32
         and target["masks"].shape == (audit["common_nonempty"],576,576)
         and target["boxes"].shape == (audit["common_nonempty"],4)
         and len(target["labels"]) == audit["common_nonempty"]
         and int(target["image_id"].item()) == audit["image_id"]
         and bool(target["masks"].flatten(1).any(dim=1).all()), "Exact common original-row CPU batch fields")
    actual_image = paired._tensor_digest((("image",image),))
    actual_nonmask = paired._tensor_digest((k,target[k]) for k in sorted(paired.PREPARED_FIELDS-{"masks"}))
    actual_masks = paired._training_mask_digest(target["masks"])
    need((actual_image,actual_nonmask,actual_masks) ==
         (audit["image_sha256"],audit["common_nonmask_sha256"],audit[mode+"_common_masks_sha256"]),
         "Actual returned CPU batch bytes must equal paired image, common fields and selected masks")
    return actual_masks


def run_arm(*, mode, positions, ids, layout, model, train, deps, torch, binding, paired, historical):
    from rfdetr.datasets import _torchvision, coco
    from rfdetr.training import module_data

    historical._seed_cpu(SEED)
    data = deps.fit_only_data(model,train,layout)
    written = StringIO()
    ledger = binding.ConsumedFitLedger(written)
    selected = sha256()
    crop = changed_crop = changed_direct = 0

    def original_guard(_active,_model,batch,_batch_idx):
        need(deps.empty_guard(batch[1][0]) == 0,
             "Original stock training empty-mask guard may not remove common-support rows")

    observer = SimpleNamespace(on_train_batch_start=original_guard)
    binding.bind_common_observer(observer,ledger)
    trainer = SimpleNamespace(current_epoch=0)
    with binding.common_fit_dataset(module_data,coco.CocoDetection,layout/"train",FIT_COUNT,
                                    _torchvision,mode,ledger):
        data.setup("fit")
        dataset = data._dataset_train
        need(type(dataset) is coco.CocoDetection and tuple(dataset.ids) == tuple(ids)
             and data._dataset_val is data._dataset_test is None,
             "Exact original isolated SSDD FIT-only native dataset")
        for batch,position in enumerate(positions):
            image,target = dataset[position]
            audit, = ledger.pending
            need(audit["image_id"] == ids[position], "Expected prelocked FIT visit image ID")
            mask_digest = validate_selected(torch,paired,image,target,audit,mode)
            observer.on_train_batch_start(trainer,None,(image,[target]),batch)
            need(validate_selected(torch,paired,image,target,audit,mode) == mask_digest,
                 "Original empty-mask guard must preserve every actual common batch field")
            selected.update(bytes.fromhex(mask_digest))
            crop += audit["crop"]
            changed_crop += bool(audit["crop"] and audit["common_changed_masks"])
            changed_direct += bool(not audit["crop"] and audit["common_changed_masks"])
    stats = ledger.finish(VISITS)
    need(stats["prefetched_unconsumed"] == 0 and stats["common_targets_total"] > 0,
         "All exact CPU visits must be consumed with common nonempty targets")
    return dict(ledger=written.getvalue(),chosen_sha=selected.hexdigest(),crop=crop,visits=VISITS,
                changed=stats["changed_mask_visits"],changed_crop=changed_crop,changed_direct=changed_direct,
                stats=stats)


def run(args):
    started = perf_counter()
    entry = pinned(args.code_root,"entry")
    lock = dict(operation="replay",code=str(args.code_root),overlay=str(args.overlay),
                fit_layout=str(args.fit_layout),fit_archive={"path":str(args.fit_archive)},
                source_manifest_sha256=entry.WORKER_MANIFEST_SHA)
    torch, init, _, deps = entry.admit_windows_runtime(lock)
    need(not torch.cuda.is_initialized(), "CPU preflight must never initialize CUDA")
    layout, ids = entry.read_fit_population(lock)
    config = init.load_api(init.CONFIG,init.CONFIG_SHA)
    init.pinned_bytes(init.RECIPE,entry.RECIPE_SHA)
    model, train = config._load_configs(init.RECIPE,layout)
    deps.e1.native_recipe(model,train,import_module("rfdetr.config"))
    need(train.seed == SEED and train.num_workers == 0, "Exact old SSDD seed and serial stock recipe")
    historical = import_module("ssdd_phase_fit_replay")
    positions = prospective_positions()
    helpers = {name:pinned(args.code_root,name) for name in ("trace_resize","common_mask_supervision")}
    with patch.dict(sys.modules,helpers):
        paired = pinned(args.code_root,"paired_fit_transform")
        binding = pinned(args.code_root,"common_fit_binding")
        with patch.dict(sys.modules,{"paired_fit_transform":paired}):
            results = {mode:run_arm(mode=mode,positions=positions,ids=ids,layout=layout,model=model,
                train=train,deps=deps,torch=torch,binding=binding,paired=paired,historical=historical)
                for mode in ("native","center")}
    a,b = (results[mode] for mode in ("native","center"))
    compare_common(a,b)
    need(entry.read_fit_population(lock) == (layout,ids) and not torch.cuda.is_initialized()
         and torch.get_num_threads() == 1, "Exact FIT layout must remain unchanged and CPU-only")
    common = a["ledger"].encode("utf-8")
    raw = dict(status=STATUS, FIT=True, DEV=False, TEST=False, AP=False, training=False,
        model_forward_calls=0, optimizer_updates=0, cuda_initialized=False, torch_threads=1,
        initial_cpu_rng_seed=SEED, fit_images=FIT_COUNT, visits_per_arm=VISITS, crops_per_arm=a["crop"],
        direct_per_arm=VISITS-a["crop"], common_changed_mask_visits=a["changed"],
        common_changed_crop_visits=a["changed_crop"], common_changed_direct_visits=a["changed_direct"],
        ordered_common_rows_nonmask_image_rng_trace_exact=True, original_empty_filter_removals=0,
        common_ledger_sha256=sha256(common).hexdigest(),
        native_selected_mask_stream_sha256=a["chosen_sha"],center_selected_mask_stream_sha256=b["chosen_sha"],
        selection="python random.Random(5201).sample(range(741),32) in stock sorted dataset index order",
        prospective_order_sha256=sha256(entry.compact([ids[i] for i in positions])).hexdigest(),
        fit_archive_sha256=entry.FIT_ARCHIVE_SHA,fit_index_sha256=entry.FIT_INDEX_SHA,
        stock_worker_manifest_sha256=entry.WORKER_MANIFEST_SHA,source_sha256={k:v[1] for k,v in PINS.items()},
        common_targets_total=a["stats"]["common_targets_total"],common_empty_visits=a["stats"]["common_empty_visits"],
        elapsed_seconds=perf_counter()-started)
    result = (json.dumps(raw,sort_keys=True,indent=2,allow_nan=False)+"\n").encode("utf-8")
    args.output.mkdir()
    for name,payload in (("CONSUMED_COMMON_PRIVATE.jsonl",common),("RESULT_SAFE.json",result)):
        with (args.output/name).open("xb") as file:
            need(file.write(payload) == len(payload), "Complete closed preflight output required")
            file.flush()
            os.fsync(file.fileno())
    entry.emit(dict(status=STATUS,result_sha256=sha256(result).hexdigest()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ("code-root","overlay","fit-layout","fit-archive","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    args = parser.parse_args(argv)
    for path in (args.code_root,args.overlay,args.fit_layout,args.fit_archive):
        need(path.is_absolute() and path.resolve(strict=True)==path, "Canonical immutable closed input path required")
    need(args.output.is_absolute() and args.output.parent.resolve(strict=True)==args.output.parent
         and not args.output.exists() and not args.output.is_symlink()
         and not args.output.is_relative_to(args.fit_layout) and not args.fit_layout.is_relative_to(args.output),
         "Fresh canonical output outside the isolated FIT layout required")
    run(args)


if __name__ == "__main__":
    main()
