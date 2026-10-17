"""UNBOUND B1 initialization draft; no Torch/NumPy import or execution on import.

Future Root caller must source-close the seven modules/references and separately
authorize CPU construction. This is not a trainer, checkpoint loader or B0 pass.
Only the named, source-reviewed common operations may share state. Everything
else retains its own seed202610173101 constructor draw and original inactive boundary.
"""

import ast
import hashlib
import json
from pathlib import Path


SEED = 202610173101
SOURCE_SHA256 = {
    'full_legacy': 'c210b318be29aca7d26516e9f47c8849e2c6a76642d4049fe7b133726ee809b9',
    'full_dsconv': 'b4b612edf9f87cf29dc48bb675d501658fdec5dbee1ea363a7f02d86d197e5f0',
    'full_head': '6caba966ac7601b611fd1b36fdbd1d758875c5d029758fefe586738c8009f16e',
    'full_feature': 'f7953780580449dd246cbdf1803652183c1dd178f2fd60b37adb238a6d2b6a5a',
    'plain_legacy': '902e13e5c60b1579459854c6bf4393ad8f268092a73a1d851f0fa640c810f48d',
    'plain_feature': 'd93ff973a57ec3abfde98bb7b9ab4fcf5a3443445a512c2acd16460fce5bfbad',
    'instance_adapter': '9ad221170f69c028474f76700c991cb76dfecf19b11f5198767843b8e378d8b3',
}
BLOCKS = {
    'stage_0': (3, 64, False),
    'stage_2': (128, 256, False),
    'CONV0_1': (128, 64, True),
    'CONV0_2': (192, 64, True),
    'CONV0_3': (256, 64, True),
    'CONV0_4': (320, 64, True),
}
UP_EDGES = {
    'upsample_3_1': (1024, 512),
    'upsample_2_1': (512, 256), 'upsample_2_2': (512, 256),
    'upsample_1_1': (256, 128), 'upsample_1_2': (256, 128), 'upsample_1_3': (256, 128),
    'upsample_0_1': (128, 64), 'upsample_0_2': (128, 64),
    'upsample_0_3': (128, 64), 'upsample_0_4': (128, 64),
}
COMMON_ROOTS = tuple(BLOCKS)+tuple(UP_EDGES)


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def _class(tree, name):
    nodes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name]
    need(len(nodes) == 1, 'Exactly one original class '+name)
    return nodes[0]


def _declarations(tree):
    model = _class(tree, 'UnetPlusPlus')
    constructors = [node for node in model.body if isinstance(node, ast.FunctionDef) and node.name == '__init__']
    need(len(constructors) == 1, 'Single original model constructor')
    found = {}
    for node in ast.walk(constructors[0]):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == 'self':
                if target.attr in COMMON_ROOTS:
                    need(target.attr not in found, 'Duplicate common declaration '+target.attr)
                    found[target.attr] = node.value
    need(set(found) == set(COMMON_ROOTS), 'Every explicitly reviewed operation must be declared')
    return found


def verify_operation_sources(full_bytes, plain_bytes):
    """Pure stdlib source check. No class import or state-shape inference by NN."""
    for role, raw in (('full_legacy', full_bytes), ('plain_legacy', plain_bytes)):
        need(type(raw) is bytes and hashlib.sha256(raw).hexdigest() == SOURCE_SHA256[role],
             'Exact reviewed original source: '+role)
    full, plain = ast.parse(full_bytes), ast.parse(plain_bytes)
    common_class = ast.dump(_class(full, 'ContinusParalleConv'), include_attributes=False)
    need(common_class == ast.dump(_class(plain, 'ContinusParalleConv'), include_attributes=False),
         'Original ordinary block constructor AND forward operations must be identical')
    declarations = [_declarations(tree) for tree in (full, plain)]
    rows = []
    for root in COMMON_ROOTS:
        a, b = (ast.dump(found[root], include_attributes=False) for found in declarations)
        need(a == b, 'Same named/positioned operation must have identical constructor expression: '+root)
        rows.append(dict(root=root, constructor_AST_equal=True,
                         constructor_AST_sha256=hashlib.sha256(a.encode()).hexdigest()))
    return dict(status='STATIC_SOURCE_OPERATIONS_MATCH', roots=rows,
                ordinary_block_AST_sha256=hashlib.sha256(common_class.encode()).hexdigest(),
                tensor_copy_executed=False, actual_shapes_verified=False)


def static_state_contract():
    """Exact leaf paths derived from the reviewed Sequential declarations.

    FP32 is explicitly required by the future constructor. BN counters are
    int64 scalar buffers. ConvTranspose2d has [in,out,kH,kW] weight ordering.
    These are expectations, not a claim that actual objects have been tested.
    """
    result = {}

    def add(name, shape, kind='parameter', dtype='torch.float32'):
        need(name not in result, 'Duplicate static state path')
        result[name] = dict(shape=list(shape), kind=kind, dtype=dtype)

    def bn(prefix, channels):
        for name in ('weight', 'bias'):
            add(prefix+'.'+name, (channels,))
        for name in ('running_mean', 'running_var'):
            add(prefix+'.'+name, (channels,), 'buffer')
        add(prefix+'.num_batches_tracked', (), 'buffer', 'torch.int64')

    def conv(prefix, in_channels, out_channels):
        add(prefix+'.weight', (out_channels, in_channels, 3, 3))
        add(prefix+'.bias', (out_channels,))

    for root, (inputs, outputs, pre_bn) in BLOCKS.items():
        prefix = root+'.Conv_forward.'
        if pre_bn:
            bn(prefix+'0', inputs)
            conv(prefix+'2', inputs, outputs)
            bn(prefix+'3', outputs)
            conv(prefix+'5', outputs, outputs)
        else:
            conv(prefix+'0', inputs, outputs)
            bn(prefix+'1', outputs)
            conv(prefix+'3', outputs, outputs)
            bn(prefix+'4', outputs)
    for root, (inputs, outputs) in UP_EDGES.items():
        add(root+'.weight', (inputs, outputs, 4, 4))
        add(root+'.bias', (outputs,))
    need(len(result) == 104
         and sum(row['kind'] == 'parameter' for row in result.values()) == 68,
         'Reviewed 68 parameter tensors plus 36 BN buffers')
    return result


def _module_signature(torch, root, ordinary_class):
    rows = []
    for name, module in root.named_modules():
        if type(module) is ordinary_class:
            kind, attrs = 'ContinusParalleConv', ('in_channels', 'out_channels')
        elif type(module) is torch.nn.Sequential:
            kind, attrs = 'Sequential', ()
        elif type(module) is torch.nn.Conv2d:
            kind, attrs = 'Conv2d', ('in_channels', 'out_channels', 'kernel_size', 'stride', 'padding',
                                    'dilation', 'groups', 'padding_mode')
        elif type(module) is torch.nn.ConvTranspose2d:
            kind, attrs = 'ConvTranspose2d', ('in_channels', 'out_channels', 'kernel_size', 'stride',
                'padding', 'output_padding', 'dilation', 'groups', 'padding_mode')
        elif type(module) is torch.nn.BatchNorm2d:
            kind, attrs = 'BatchNorm2d', ('num_features', 'eps', 'momentum', 'affine', 'track_running_stats')
        elif type(module) is torch.nn.ReLU:
            kind, attrs = 'ReLU', ('inplace',)
        else:
            raise RuntimeError('Unreviewed operation in common subtree: '+name+' '+type(module).__name__)
        values = {attr:getattr(module, attr) for attr in attrs}
        if kind in ('Conv2d', 'ConvTranspose2d'):
            values['bias_present'] = module.bias is not None
        rows.append((name, kind, values, tuple(module._modules)))
    return rows


def _snapshot(model):
    return dict(modules=tuple((name, id(value), value.training) for name, value in model.named_modules()),
                parameters=tuple((name, id(value), value.requires_grad) for name, value in model.named_parameters()),
                buffers=tuple((name, id(value)) for name, value in model.named_buffers()))


def _digest(value):
    # Only called by the future explicitly authorized constructor. No current
    # Torch/NumPy numerical execution, existing weights or images are involved.
    value = value.detach().cpu().contiguous()
    h = hashlib.sha256(json.dumps([str(value.dtype), list(value.shape)], separators=(',', ':')).encode())
    h.update(value.numpy().tobytes(order='C'))
    return h.hexdigest()


def build_pair(torch, modules, source_refs):
    """Future CPU-only construction -> checked common copy -> paired adapter.

    modules/source_refs contain exactly SOURCE_SHA256's seven role keys. Modules
    must already have been loaded by Root from these closed references; this
    function never imports arbitrary modules or loads checkpoints. Returns
    ({'PLAIN': adapter, 'FULL': adapter}, actual_initialization_receipt).
    It performs no forward/backward, optimizer creation, data I/O or training.
    """
    need(set(modules) == set(source_refs) == set(SOURCE_SHA256), 'Exact seven source-closed component roles')
    need(not torch.cuda.is_initialized() and torch.get_default_dtype() == torch.float32,
         'Cold CUDA and explicit FP32 default before CPU construction')
    contents = {}
    for role, expected in SOURCE_SHA256.items():
        reference = source_refs[role]
        path = Path(reference['path'])
        need(path.is_absolute() and Path(modules[role].__file__).resolve() == path.resolve(),
             'Loaded module path must equal the admitted reference: '+role)
        raw = path.read_bytes()
        need(len(raw) == reference['bytes'] and reference['sha256'] == expected
             and hashlib.sha256(raw).hexdigest() == expected, 'Exact source bytes: '+role)
        contents[role] = raw
    proof = verify_operation_sources(contents['full_legacy'], contents['plain_legacy'])
    need(modules['full_legacy'].DSConv is modules['full_dsconv'].DSConv
         and modules['full_legacy'].CoordConv is modules['full_head'].CoordConv,
         'Original full imports must resolve to the exact admitted dependency objects')

    # Same numeric seed does not by itself give identical shared tensors:
    # different constructors consume random draws in different orders.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(SEED)
        plain = modules['plain_legacy'].UnetPlusPlus(num_classes=1, deep_supervision=False)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(SEED)
        full = modules['full_legacy'].UnetPlusPlus(num_classes=1, deep_supervision=False,
            use_x2_transformer=False, input_size=400, instantiate_x2_transformer=False,
            include_legacy_x4_transformer=False)
    need(full.encoder_x2 is None and not hasattr(full, 'encoder_x4'), 'No allocated Transformer variant')
    models = {'PLAIN':plain, 'FULL':full}
    snapshots = {arm:_snapshot(model) for arm, model in models.items()}
    for model in models.values():
        need(all(value.device.type == 'cpu' and value.dtype == torch.float32 for value in model.parameters()),
             'Original parameter objects must be CPU FP32')
    wrappers = {
        'PLAIN':modules['plain_feature'].make_plain_feature_backbone(torch, plain, input_size=400),
        'FULL':modules['full_feature'].make_feature_backbone(torch, full),
    }
    inactive = {arm:wrapper.inactive_parameter_names() for arm, wrapper in wrappers.items()}
    expected = static_state_contract()
    for arm in models:
        need(type(inactive[arm]) is tuple
             and not set('legacy.'+name for name in expected).intersection(inactive[arm]),
             'Mapped parameters must not cross either original inactive boundary')

    for root in COMMON_ROOTS:
        plain_signature = _module_signature(torch, getattr(plain, root), modules['plain_legacy'].ContinusParalleConv)
        full_signature = _module_signature(torch, getattr(full, root), modules['full_legacy'].ContinusParalleConv)
        need(plain_signature == full_signature, 'Actual module tree and operation attributes must match: '+root)
    states = {arm:model.state_dict() for arm, model in models.items()}
    for arm, model in models.items():
        actual_common = {name for name in states[arm] if name.split('.', 1)[0] in COMMON_ROOTS}
        need(actual_common == set(expected), 'Exact actual common state names, with no partial match: '+arm)
        params, buffers = dict(model.named_parameters()), dict(model.named_buffers())
        for name, item in expected.items():
            value = states[arm][name]
            need(list(value.shape) == item['shape'] and str(value.dtype) == item['dtype'],
                 'Exact actual shape/dtype for '+arm+':'+name)
            need(name in (params if item['kind'] == 'parameter' else buffers), 'Parameter/buffer kind: '+name)
    untouched_before = {arm:{name:_digest(value) for name, value in state.items() if name not in expected}
                        for arm, state in states.items()}
    copied = []
    with torch.no_grad():
        for name, item in expected.items():
            # Canonical donor is the original PLAIN seed202610173101 draw. Preserve the
            # recipient Parameter/Buffer object and every architectural path.
            states['FULL'][name].copy_(states['PLAIN'][name])
            need(torch.equal(states['FULL'][name], states['PLAIN'][name]), 'Actual equal copied tensor: '+name)
            copied.append(dict(name=name, **item, sha256=_digest(states['PLAIN'][name])))
    for arm, model in models.items():
        need(_snapshot(model) == snapshots[arm], 'No modules/parameters/buffers/modes/flags replaced: '+arm)
        need(all(_digest(states[arm][name]) == digest for name, digest in untouched_before[arm].items()),
             'Every unmapped original state retains its own constructor draw/buffer: '+arm)
        need(wrappers[arm].inactive_parameter_names() == inactive[arm], 'Original inactive boundary unchanged')

    adapters, adapter_digests = {}, {}
    for arm in ('PLAIN', 'FULL'):
        # Reset only within forked CPU RNG; do not inherit the number of draws
        # consumed by either backbone or by construction of the other adapter.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(SEED)
            adapters[arm] = modules['instance_adapter'].make_adapter(torch, wrappers[arm],
                'legacy.CONV0_4', 64, query_channels=256, hidden=32)
        adapter = adapters[arm]
        adapter_digests[arm] = {name:_digest(value) for name, value in adapter.state_dict().items()
                               if not name.startswith('backbone.')}
        need(bool((adapter.head[-1].weight == 0).all()) and bool((adapter.head[-1].bias == 0).all()),
             'Original shared adapter has its actual zero final head')
        need(_snapshot(models[arm]) == snapshots[arm], 'Adapter registration preserves legacy objects and flags')
    need(adapter_digests['PLAIN'] == adapter_digests['FULL'], 'All shared adapter initial tensors match exactly')
    need(not torch.cuda.is_initialized(), 'CPU initialization must not initialize CUDA')
    receipt = dict(schema='U34-common-operation-initialization-v1', status='INITIALIZED_CPU_ONLY',
        seed=SEED, adapter_seed=SEED, full_input_size=400, image_batch_size=1,
        source_refs=source_refs, source_operation_proof=proof,
        common_state= copied, common_parameter_tensors=68, common_BN_buffers=36,
        canonical_common_donor='PLAIN fresh seed202610173101; copy in place to FULL',
        differential_states='Each original constructor fresh seed202610173101, unmapped values left unchanged',
        unmapped_state_sha256=untouched_before,
        inactive_parameter_names={arm:list(names) for arm, names in inactive.items()},
        adapter_state_sha256=adapter_digests['PLAIN'], old_UNet_weights_loaded=False,
        CUDA_initialized=False, forward_executed=False, backward_executed=False,
        optimizer_created=False, training_started=False, B0_pass_claimed=False)
    return adapters, receipt
