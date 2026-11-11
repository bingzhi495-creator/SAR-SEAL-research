"""Verify only naturally closed original floating/common E24 arms at downstream seed 5202."""
from hashlib import sha256
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parent.parent
BASE_VERIFIER = ROOT / "SSDD_WINDOWS_FLOAT_COMMON_E24_INDEPENDENT_DRAFT_PRIVATE_20261112_02/accept_float_e24.py"
BASE_VERIFIER_SHA = "16f3508c8c4add5ed68e89e14e1cec4a43b64cc22b730c7fc38bff3dfb067219"
BRIDGE = ROOT / "SSDD_WINDOWS_FLOAT_COMMON_REPEAT5202_E24_DRAFT_PRIVATE_20261112_01/repeat5202_float_e24.py"
BRIDGE_SHA = "e469f36f2d57017ee2a75d618e5fbc077343eee7c9ec0a480bc3b8d729ef5cdf"
SHORT_PAIR_SHA = "6bef2983c978f3614fb610089ce173713dfc3892641a1890139fe560517692f8"
ARM_STATUS = "SSDD_WINDOWS_FLOAT_COMMON_REPEAT5202_E24_ARM_INDEPENDENTLY_ACCEPTED_V1"
PAIR_STATUS = "SSDD_WINDOWS_FLOAT_COMMON_REPEAT5202_E24_PAIR_INDEPENDENTLY_ACCEPTED_V1"
INITIALIZATION_SEED = 5201
TRAINING_SEED = 5202


def require(ok, why):
    if not ok:
        raise ValueError(why)


def source_text(path, expected):
    raw = path.read_bytes()
    require(path.is_absolute() and path.resolve(strict=True) == path and sha256(raw).hexdigest() == expected,
            "Exact canonical closed reader source " + path.name)
    return raw.decode("utf8")


def patched(source, edits):
    for before, after, count in edits:
        require(source.count(before) == count, "Exact immutable seed and receipt adapter seam")
        source = source.replace(before, after)
    return source


def module(path, source, name):
    loaded = ModuleType(name)
    loaded.__file__ = str(path)
    exec(compile(source, str(path), "exec"), loaded.__dict__)
    return loaded


def split_stock_seed_source(source):
    return patched(source, (
        ("initialization_seed=SEED", "initialization_seed=INITIALIZATION_SEED", 2),
        ('phase.get("initialization_seed") == SEED', 'phase.get("initialization_seed") == INITIALIZATION_SEED', 1),
    ))


def split_verifier_source(source):
    return patched(source, (
        ("mask_resize_mode=mode,seed=5201,", "mask_resize_mode=mode,seed=5202,", 1),
        ("result=dict(status=ARM_STATUS,phase=phase,",
         "result=dict(status=ARM_STATUS,phase=phase,initialization_seed=5201,training_seed=5202,", 1),
        ("dict(status=ARM_STATUS,qualification_pair_sha256=m.SHORT_ACCEPTANCE_SHA,",
         "dict(status=ARM_STATUS,qualification_pair_sha256=m.SHORT_ACCEPTANCE_SHA,initialization_seed=5201,training_seed=5202,", 1),
        ("qualification_pair_sha256=m.SHORT_ACCEPTANCE_SHA,seed=5201,fit_images=741,",
         "qualification_pair_sha256=m.SHORT_ACCEPTANCE_SHA,seed=5202,initialization_seed=5201,training_seed=5202,fit_images=741,", 1),
    ))


def closed_verifier():
    source = split_verifier_source(source_text(BASE_VERIFIER, BASE_VERIFIER_SHA))
    verifier = module(BASE_VERIFIER, source, "_exact_float_common_seed5202_closed_verifier")

    def bridge():
        runtime = module(BRIDGE, source_text(BRIDGE, BRIDGE_SHA), "_exact_executed_repeat5202_bridge")
        training = runtime.bridge()
        require(training.SHORT_ACCEPTANCE_SHA == SHORT_PAIR_SHA, "Exact sole accepted repeat qualification")
        original = training.module

        def checked_stock(path, expected, name):
            if name != "_exact_stock_ssdd_e24_acceptance":
                return original(path, expected, name)
            require(expected == verifier.ACCEPT_API, "Exact stock acceptance API")
            source = split_stock_seed_source(source_text(path, expected))
            api = module(path, source, name)
            require(api.SEED == INITIALIZATION_SEED, "Exact original stock initialization seed")
            api.SEED, api.INITIALIZATION_SEED = TRAINING_SEED, INITIALIZATION_SEED
            return api

        training.module = checked_stock
        return training

    verifier.BRIDGE = BRIDGE
    verifier.BRIDGE_SHA = BRIDGE_SHA
    verifier.ARM_STATUS, verifier.PAIR_STATUS = ARM_STATUS, PAIR_STATUS
    verifier.bridge = bridge
    return verifier


if __name__ == "__main__":
    closed_verifier().main()
