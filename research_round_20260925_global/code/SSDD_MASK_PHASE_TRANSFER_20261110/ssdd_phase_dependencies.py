"""Assemble the isolated SSDD phase worker from exact ordinary native sources."""

from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import sys
from unittest.mock import patch

from ssdd_phase_init import CODE, CONFIG_SHA, E1, E1_SHA, RECIPE_SHA, STATE_SHA, load_api, pinned_bytes, require


HERE = Path(__file__).resolve().parent
PHASE_DIR = CODE / "HRSID_FIT_RESIZE_PHASE_TRAIN_20261109"
LEGACY = {
    "e100": (CODE / "RFDETR_NATIVE_UNINTERRUPTED_E100_WORKER_CANDIDATE_20261107/e100_worker.py",
             "ecc8c67bf67e4e58d0d697bf4ac48b9ff4a339ff1d162019ce95b73a0fed0144"),
    "e1": (E1, E1_SHA),
    "boundary": (CODE / "RFDETR_NATIVE_E1_CHECKPOINT_API_RECOVERY_20261107/e1_resume.py",
                 "014e087379ffe6e8d32329c337d6b625b98dfc2d235cb321becebfdf3832b013"),
    "trace_resize": (CODE / "HRSID_FIT_RESIZE_PHASE_REPLAY_20261109/trace_resize.py",
                     "902a0e3640a319ba729172847f49ce582f3c0e3954593862ed94971727dc309f"),
    "phase_patch": (PHASE_DIR / "phase_patch.py", "811eccf2b42319272aea8b3f50a71bc029734d006bba43260f66117e9421948e"),
    "phase": (PHASE_DIR / "phase_train.py", "8550b43791a6cb6f53736dc7e64b9c5c6204723cf4c07c3b112491ba1bd74043"),
    "empty_guard": (CODE / "HRSID_NATIVE_EMPTY_MASK_FILTER_20261109/filter_empty_masks.py",
                    "3b164cbf8c104539eabee8775c8e1f297d9a1607e37473c4d7a4a9b561f93471"),
}
CANDIDATE = ("ssdd_phase_init.py", "ssdd_fit_only.py", "ssdd_phase_train.py")


def load_worker(candidate_pins):
    """Return an exact native arm callable, its dependencies, and a source identity."""
    require(type(candidate_pins) is dict and set(candidate_pins) == set(CANDIDATE),
            "Exact prospective worker, loader and initialization source pins required")
    for name in CANDIDATE:
        pinned_bytes(HERE / name, candidate_pins[name])
    require(version("rfdetr") == "1.12.0" and version("pytorch-lightning") == "2.6.6"
            and version("torch") == "2.10.0+cu130" and version("torchvision") == "0.25.0+cu130",
            "Exact SSDD RF-DETR, Lightning and Torch package versions required")
    from rfdetr.datasets import _torchvision, coco
    from rfdetr.training import module_data

    runtime = (
        (_torchvision, "03c12e881f733d4f6b469f784187e461753b8775b7e35e57c1d8a6c35c46ef56"),
        (coco, "be64b20a178d9cde41db69e01e17e1296e36eee791e837294c3f930ae56f03b9"),
        (module_data, "624bc77f7028bc53e0ca57dcbf5be1a7d334685147dbfab65e9acaa063d715b5"),
    )
    for module, expected in runtime:
        pinned_bytes(Path(module.__file__).resolve(strict=True), expected, 1024**2)
    originals = {key: load_api(*LEGACY[key]) for key in ("e100", "e1", "boundary", "empty_guard", "trace_resize")}
    with patch.dict(sys.modules, {"trace_resize": originals["trace_resize"]}):
        phase_patch = load_api(*LEGACY["phase_patch"])
        with patch.dict(sys.modules, {"phase_patch": phase_patch}):
            phase = load_api(*LEGACY["phase"])
    init, fit, train = (load_api(HERE / name, candidate_pins[name]) for name in CANDIDATE)
    dependencies = train.TrainingDependencies(init.initialize_q100, fit.FitOnlyData, originals["e100"],
                                               originals["e1"], originals["boundary"], phase,
                                               originals["empty_guard"].drop_empty_augmented_masks, _torchvision)
    pins = dict(candidate_pins, recipe=RECIPE_SHA, config=CONFIG_SHA, state_digest=STATE_SHA,
                dependency_assembly=sha256(Path(__file__).read_bytes()).hexdigest())
    pins.update({key: digest for key, (_, digest) in LEGACY.items()})
    pins.update({module.__name__: digest for module, digest in runtime})
    manifest = sha256(json.dumps(pins, sort_keys=True, separators=(",", ":")).encode("ascii")).hexdigest()
    return train.run_fit_arm, dependencies, manifest

