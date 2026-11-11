"""Owner-callable SSDD short repeat, fixed original COCO state and downstream seed 5202."""

import ast
from hashlib import sha256
import json
from pathlib import Path
import re
from types import ModuleType, SimpleNamespace


INITIALIZATION_SEED = 5201
TRAINING_SEED = 5202
AUTHORITY = "SSDD_MASK_PHASE_REPEAT_5202_SHORT_PROSPECTIVE_V1"
PAIR_STATUS = "SSDD_MASK_PHASE_REPEAT_5202_QUALIFICATION_PAIR_ACCEPTED_V1"
MODES = ("native", "center")
SUBDIR = "SSDD_MASK_PHASE_TRANSFER_20261110"
TRAIN = ("ssdd_phase_train.py", "6785b1f9ffe23b6b2c1da5939d10d8b5877b93127316f535af5890ca05f4653c")
QUALIFIER = ("ssdd_phase_qualification.py", "8b76b42d59fff857dfb0c0fc9b67664817c72a494966f5f0ec808156776cdaec")
WINDOWS = ("ssdd_windows_phase_entry.py", "e29edeb0a3aff7c871123deee7c0c198ef5913fd33e7c016434a671d186796d3")
WINDOWS_MANIFEST = "046ddaa300e428140952a09931269f6914c789a63fabdc94f98484c3f2efaa91"
DGX_MANIFEST = "6218d274eaff924ae3ea05903b19517a31a3717efae6f7bb6a006ffe3b516221"
OLD_WINDOWS = "22e4976a4536c908921b4756d928c5ba9b2f1ffd"
OLD_PROTOCOLS = {OLD_WINDOWS, "5d69d04342e241e32d56c219d99213aed0242e62",
                 "6e324a4339312c582dbeb8b2203246796954fb88", "02d65ce5f01d2239a68c55bd84bf8e8739c11f29"}
Q100_SHA = "40524fc63797d2fcab7b50b6ee1f041f9429df73e505cb9e647a5a8cdf6f2e2f"


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def hex_value(value, length):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{%d}" % length, value) is not None


def digest(data):
    return sha256(data).hexdigest()


def compact(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")


def read_pinned_source(path, expected):
    require(path.is_absolute() and path.resolve(strict=True) == path, "Exact ordinary source path required")
    with path.open("rb") as stream:
        raw = stream.read(64*1024+1)
    require(0 < len(raw) <= 64*1024 and digest(raw) == expected, "Exact source digest required")
    return raw


class DownstreamSeed(ast.NodeTransformer):
    """Bind the downstream seed in named functions, preserving the init witnesses."""

    def __init__(self):
        self.replacements = 0
        self.kept_initialization = 0

    def visit_keyword(self, node):
        if node.arg == "initialization_seed" and isinstance(node.value, ast.Name) and node.value.id == "SEED":
            self.kept_initialization += 1
            return node
        return self.generic_visit(node)

    def visit_Name(self, node):
        if node.id == "SEED" and isinstance(node.ctx, ast.Load):
            self.replacements += 1
            return ast.copy_location(ast.Constant(TRAINING_SEED), node)
        return node


def bind_module(source, filename, functions):
    tree = ast.parse(source, filename=filename)
    for name, expected in functions.items():
        matches = [item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == name]
        require(len(matches) == 1, "Exact source seed sites require a unique function")
        binding = DownstreamSeed()
        binding.visit(matches[0])
        require((binding.replacements, binding.kept_initialization) == expected, "Exact downstream and init seed sites differ")
    module = ModuleType("_p328_repeat_" + Path(filename).stem)
    module.__file__ = filename
    exec(compile(ast.fix_missing_locations(tree), filename, "exec"), module.__dict__)
    require(module.SEED == INITIALIZATION_SEED, "Original initialization seed constant differs")
    return module


def load_programs(code_root):
    """Use the stock code and isolated, source-pinned downstream seed bindings."""
    code = Path(code_root) / SUBDIR
    trainer_path, qualifier_path = code / TRAIN[0], code / QUALIFIER[0]
    trainer = bind_module(read_pinned_source(trainer_path, TRAIN[1]), str(trainer_path), {"run_fit_arm": (7, 1)})
    qualifier = bind_module(read_pinned_source(qualifier_path, QUALIFIER[1]), str(qualifier_path),
                            {"validate_arm": (5, 2), "verify_pair": (1, 0)})
    require((trainer.SHORT_VISITS, trainer.SHORT_UPDATES, trainer.Q100_SHA, qualifier.SHORT_VISITS, qualifier.UPDATES,
             qualifier.Q100_SHA) == (32, 8, Q100_SHA, 32, 8, Q100_SHA), "Original short budgets or Q100 differ")
    return SimpleNamespace(worker=trainer, qualifier=qualifier)


def repeat_initializer(original_initializer):
    def initialize(fit_layout, output, coco_path):
        model, train, module, witness = original_initializer(fit_layout, output, coco_path)
        require(type(train.seed) is int and train.seed == INITIALIZATION_SEED and module.train_config is train,
                "Exact shared original TrainConfig required after stock Q100 construction")
        require(witness["initialization_seed"] == INITIALIZATION_SEED
                and witness["online_tensor_sha256"] == Q100_SHA and witness["state_tensors"] == 545,
                "Exact untouched official COCO Q100 state required")
        train.seed = TRAINING_SEED
        require(module.train_config is train and train.seed == TRAINING_SEED,
                "Actual repeat seed must reach the original shared TrainConfig")
        return model, train, module, witness
    return initialize


def make_identity(*, training_protocol, implementation_commit, adapter_sha256, original_manifest_sha256):
    require(all(hex_value(v, 40) and v not in OLD_PROTOCOLS for v in (training_protocol, implementation_commit))
            and training_protocol != implementation_commit and hex_value(adapter_sha256, 64)
            and original_manifest_sha256 in (WINDOWS_MANIFEST, DGX_MANIFEST),
            "Distinct prospective protocol, implementation and known stock platform provenance required")
    adapter_path = Path(__file__).resolve(strict=True)
    read_pinned_source(adapter_path, adapter_sha256)
    scientific_sources = dict(adapter_sha256=adapter_sha256, stock_train_sha256=TRAIN[1],
                              stock_qualifier_sha256=QUALIFIER[1], original_manifest_sha256=original_manifest_sha256)
    return dict(authority_identity=AUTHORITY, training_protocol=training_protocol,
                implementation_commit=implementation_commit, training_seed=TRAINING_SEED,
                initialization_seed=INITIALIZATION_SEED, stage="qualification", **scientific_sources,
                source_manifest_sha256=digest(compact(scientific_sources)))


def require_identity(identity):
    require(type(identity) is dict, "Exact seed-5202 short authority identity required")
    expected = make_identity(
        training_protocol=identity.get("training_protocol"), implementation_commit=identity.get("implementation_commit"),
        adapter_sha256=identity.get("adapter_sha256"), original_manifest_sha256=identity.get("original_manifest_sha256"))
    require(identity.keys() == expected.keys()
            and all(type(identity[key]) is type(value) and identity[key] == value for key, value in expected.items()),
            "Exact seed-5202 short authority identity required")


def run_windows_owned_short(*, code_root, identity, mode, stock_scope_lock, stock_scope_sha256, owner_admit):
    """Called inside a new Windows owner reservation, before any actual input is opened."""
    require_identity(identity)
    require(mode in MODES and identity["original_manifest_sha256"] == WINDOWS_MANIFEST
            and hex_value(stock_scope_sha256, 64) and callable(owner_admit)
            and owner_admit(identity, mode, stock_scope_lock, stock_scope_sha256) is True,
            "Exact separately admitted seed-5202 original owner required before FIT")
    code_root = Path(code_root)
    programs = load_programs(code_root)
    source = code_root / SUBDIR / WINDOWS[0]
    entry = ModuleType("_p328_windows_stock_admission")
    entry.__file__ = str(source)
    exec(compile(read_pinned_source(source, WINDOWS[1]), str(source), "exec"), entry.__dict__)
    lock = entry.read_lock(Path(stock_scope_lock), stock_scope_sha256, "short")
    require(lock["protocol"] == OLD_WINDOWS and lock["mode"] == mode and lock["source_manifest_sha256"] == WINDOWS_MANIFEST
            and Path(lock["code"]) == code_root,
            "Old stock lock is only the known FIT/platform scope of this arm")
    torch, _, _, deps = entry.admit_windows_runtime(lock)
    layout, ids = entry.read_fit_population(lock)
    entry.admit_host()
    free, total, fraction = entry.admit_cuda_short(torch.cuda)
    entry.emit(dict(event="SSDD_WINDOWS_Q100_REPEAT_5202_GPU_ADMITTED_V1",
                    protocol=identity["training_protocol"], mode=mode, cuda_free_bytes=free,
                    cuda_total_bytes=total, cuda_allocator_fraction=fraction))
    deps = deps._replace(initialize_q100=repeat_initializer(deps.initialize_q100))
    return programs.worker.run_fit_arm(stage="qualification", mode=mode, output=Path(lock["output"]), fit_layout=layout,
        coco_path=Path(lock["coco"]["path"]), expected_fit_ids=ids, expected_initial_sha256=Q100_SHA,
        protocol=identity["training_protocol"], recipe_sha256=entry.RECIPE_SHA,
        source_manifest_sha256=identity["source_manifest_sha256"], dependencies=deps, event_sink=entry.emit)


def qualify_closed_pair(*, code_root, identity, native, center, expected_fit_ids):
    """Accept only raw two-arm witnesses. Original OS closure stays with the owner."""
    require_identity(identity)
    qualifier = load_programs(code_root).qualifier
    result = qualifier.verify_pair(qualifier.QualificationArm(*native), qualifier.QualificationArm(*center),
        protocol=identity["training_protocol"], source_manifest_sha256=identity["source_manifest_sha256"],
        expected_fit_ids=expected_fit_ids)
    require(result["seed"] == TRAINING_SEED and result["checkpoint_saved"] is False,
            "Expected an independently closed repeat qualification")
    return qualifier.encode_receipt(dict(result, status=PAIR_STATUS, initialization_seed=INITIALIZATION_SEED,
        authority_identity=AUTHORITY, implementation_commit=identity["implementation_commit"],
        adapter_sha256=identity["adapter_sha256"], original_manifest_sha256=identity["original_manifest_sha256"]))
