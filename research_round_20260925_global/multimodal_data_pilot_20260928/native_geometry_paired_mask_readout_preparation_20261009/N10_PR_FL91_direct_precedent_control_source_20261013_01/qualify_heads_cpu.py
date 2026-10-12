"""Unbound SOURCE for ROOT-owned synthetic C/PR point-head CPU qualification.

This entry never loads a dataset, GT, cached sample, model or checkpoint. A
single admitted cold-CUDA CPU leaf checks only small private synthetic inputs,
the exact bound head factories, algebra, band support and two AdamW steps.
Passing cannot establish task efficacy, scientific novelty, fair real-data
training, GPU equivalence, convergence or generalization. ROOT must separately
join the current original held process's actual terminal receipt.

All 28 normal original RAWs and three explicitly preserved failed1 RAWs are
parsed and terminal-gated with standard-library code before the first input
SHA, external import or numerical operation. No retry/fallback is implemented.
"""
import argparse
import ast
import gc
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time
import types


SCHEMA = 'ROOT-N10-C-PR-FL91-synthetic-head-CPU-spec-v1'
Q_SCHEMA = 'ROOT-N10-C-PR-FL91-synthetic-head-CPU-qualification-v1'
STATUS = 'ROOT_N10_C_PR_FL91_SYNTHETIC_HEAD_CPU_QUALIFIED'
OWNER_SHA = 'ef9a9f17d4fb1bc3464936095ef799021dd880f8b15488d558a871c08b6e1924'
RAW_KEYS = ('receipt', 'phase', 'admission', 'phase_binding', 'bootstrap')
THREAD_ENV = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
              'NUMEXPR_NUM_THREADS')
REF_KEYS = ('entry_source', 'original_point_heads_source', 'PR_point_heads_source')
REQUIRED_SPEC_KEYS = ('schema', 'phase', 'sources', 'RAW_inputs',
    'prior_failed_CPU_receipt', 'additional_failed_RAW_inputs',
    'CPU_admission_file', 'Q_path', 'deadline_seconds') + REF_KEYS
C_NAMES = ('hidden1.weight', 'hidden1.bias', 'hidden2.weight', 'hidden2.bias',
           'final.weight', 'final.bias')
PR_NAMES = ('hidden1.weight', 'hidden1.bias', 'hidden2.weight', 'hidden2.bias',
            'hidden3.weight', 'hidden3.bias', 'final.weight', 'final.bias')
C_SHAPES = ((64, 562), (64,), (64, 64), (64,), (1, 64), (1,))
PR_SHAPES = ((91, 257), (91,), (91, 92), (91,), (91, 92), (91,), (1, 92), (1,))
SYNTHETIC_RECIPE = dict(factory_seed=701, private_input_seed=713,
    private_probe_seed=719, CPU_FP32=True, sampled_points=24,
    AdamW_steps_per_head=2, AdamW_lr=0.001, AdamW_weight_decay=0.0,
    AdamW_betas=[0.9, 0.999], AdamW_eps=1e-8, AdamW_foreach=False,
    final_zero_initialization=True, band_limit=2.0, LN_eps=1e-5,
    detached_native_inputs=True, synthetic_targets='all one, no real labels',
    bitwise_signed_zero_identity_claimed=False,
    numerical_comparison='same CPU arithmetic path, exact torch.equal')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    def reject(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), parse_constant=reject)


def ref_shape(ref):
    require(type(ref) is dict and set(ref) == {'path', 'bytes', 'sha256'} and
        type(ref['path']) is str and Path(ref['path']).is_absolute() and
        type(ref['bytes']) is int and ref['bytes'] >= 0 and
        type(ref['sha256']) is str and len(ref['sha256']) == 64 and
        set(ref['sha256']) <= set('0123456789abcdef'),
        'Malformed absolute byte/SHA reference')


def reference(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def bound(ref):
    ref_shape(ref)
    require(reference(ref['path']) == ref, 'Actual pinned input changed: ' + ref['path'])
    return Path(ref['path'])


def deadline(started, seconds):
    require(time.monotonic() - started < seconds,
        'Prelocked synthetic CPU deadline exceeded; no fallback or retry')


def terminal(ref, phase, code=0):
    """Original RAW JSON only; all terminal gates precede SHA/import."""
    ref_shape(ref)
    path = Path(ref['path'])
    require(path.name == 'PARENT_RECEIPT.json' and type(phase) is str and phase and
        path.parent.name == phase, 'Actual original RAW/phase locator required')
    raw = read(path)
    require(raw['status'] == 'ACTUAL_OWNED_TERMINAL' and
        type(raw['returncode']) is int and raw['returncode'] == code and
        type(raw['original_identity']) is dict and raw['original_identity'] and
        raw['root_wait_reaped'] is True and
        raw['held_handle_exit'] == dict(wait_state='WAIT_OBJECT_0', actual_exit_code=code) and
        type(raw['held_handle_exit']['actual_exit_code']) is int and
        raw['numeric_birth_absence']['original_birth_absent'] is True and
        raw.get('error') is None and raw.get('abort') is None and
        raw['wait_errors'] is False and raw['GPU_observation_errors'] == [],
        'Original held/reaped terminal%d required' % code)
    if 'phase' in raw:
        require(type(raw['phase']) is str and raw['phase'] == phase,
            'Original RAW phase differs')
    sha, peak = raw['admission_sha256'], raw['whole_GPU_observed_peak_MiB']
    require(type(sha) is str and len(sha) == 64 and
        set(sha) <= set('0123456789abcdef') and type(peak) in (int, float) and
        math.isfinite(peak) and 0 <= peak <= 8192 and
        type(raw['whole_GPU_used_limit_MiB']) is int and
        raw['whole_GPU_used_limit_MiB'] == 8192 and
        raw['whole_leaf_VRAM_certified'] is False and
        raw['GPU_scope'] == 'whole GPU observed, not per-process WDDM allocation',
        'Original RAW owner/resource scope differs')
    return raw


def owner_join(row, raw, closure):
    owner, pb = read(bound(row['admission'])), read(bound(row['phase_binding']))
    bootstrap = bound(row['bootstrap'])
    runtime = Path(row['receipt']['path']).parent
    require(Path(row['phase_binding']['path']) == runtime/'PHASE_BINDING.json' and
        bootstrap == runtime/'bootstrap.py' and
        owner['status'] == 'ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE' and
        owner['phase'] == row['phase'] and owner['driver_sha256'] == OWNER_SHA and
        owner['automatic_retry'] is False and owner['allowed_cpus'] == [0, 1] and
        owner['cuda_visible_devices'] in ('', '0') and
        owner['whole_GPU_used_limit_MiB'] == 8192 and
        raw['admission_sha256'] == row['admission']['sha256'] == pb['admission_sha256'] and
        pb['bootstrap_sha256'] == row['bootstrap']['sha256'] and
        pb['argv'][-1] == str(bootstrap) and pb['CPU_affinity'] == owner['allowed_cpus'] and
        pb['CUDA_visible_devices'] == owner['cuda_visible_devices'] and
        pb['automatic_retry'] is False, 'Original actual owner/PB/bootstrap differs')
    pins = owner['bound_source_sha256']
    entry, args = owner['entry'], owner['entry_args']
    require(type(entry) is str and type(args) is list and entry in closure and
        pins.get(entry) == closure[entry]['sha256'], 'Original entry absent or unbound')
    tree = ast.parse(bootstrap.read_text(encoding='utf-8'))
    argv = [ast.literal_eval(node.value) for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Attribute) and
            isinstance(target.value, ast.Name) and target.value.id == 'sys' and
            target.attr == 'argv' for target in node.targets)]
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and
        isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and
        node.func.value.id == 'runpy' and node.func.attr == 'run_path']
    require(argv == [[entry]+args] and len(calls) == 1 and
        ast.literal_eval(calls[0].args[0]) == entry and
        {item.arg: ast.literal_eval(item.value) for item in calls[0].keywords} ==
            {'run_name': '__main__'}, 'Actual original bootstrap entry/argv differs')
    if '--spec' in args:
        require(args.count('--spec') == 1, 'Ambiguous ancestor SPEC')
        locator = args[args.index('--spec')+1]
        require(locator in closure and pins.get(locator) == closure[locator]['sha256'],
            'Actual original SPEC absent or unbound')
    if 'scientific_spec' in owner:
        ref = owner['scientific_spec']
        require(ref in closure.values() and pins.get(ref['path']) == ref['sha256'],
            'Original scientific SPEC absent or unbound')
    return owner


def source_gate(path, spec, started):
    require(all(key in spec for key in REQUIRED_SPEC_KEYS) and
        spec['schema'] == SCHEMA and type(spec['phase']) is str and spec['phase'] and
        type(spec['RAW_inputs']) is list and len(spec['RAW_inputs']) == 28 and
        all(type(row) is dict and set(RAW_KEYS) <= set(row) for row in spec['RAW_inputs']),
        'Complete ROOT-authored 28-ancestor synthetic CPU SPEC required')
    rows = spec['RAW_inputs']
    for row in rows:
        ref_shape(row['receipt'])
    receipt_paths = [row['receipt']['path'] for row in rows]
    require(len(set(receipt_paths)) == 28, 'Distinct actual normal RAWs required')
    raws = [terminal(row['receipt'], row['phase']) for row in rows]
    failed_refs = [spec['prior_failed_CPU_receipt']]
    ref_shape(failed_refs[0])
    terminal(failed_refs[0], Path(failed_refs[0]['path']).parent.name, 1)
    additional = spec['additional_failed_RAW_inputs']
    require(type(additional) is list and len(additional) == 2,
        'Two additional preserved failed1 RAWs required')
    for row in additional:
        require(type(row) is dict and set(row) == {'receipt', 'phase', 'returncode'} and
            type(row['returncode']) is int and row['returncode'] == 1,
            'Explicit preserved failed1 locator required')
        terminal(row['receipt'], row['phase'], 1)
        failed_refs.append(row['receipt'])
    require(len({ref['path'] for ref in failed_refs}) == 3 and
        not set(receipt_paths) & {ref['path'] for ref in failed_refs},
        'Failed1 cannot qualify a normal ancestor')
    # ALL 28 normal and 3 failed1 terminal gates precede first SHA/import.
    require(type(spec['sources']) is list and spec['sources'], 'Complete input closure required')
    for ref in spec['sources']:
        ref_shape(ref)
    closure = {ref['path']: ref for ref in spec['sources']}
    require(len(closure) == len(spec['sources']), 'Unique complete input closure required')
    required = [spec[key] for key in REF_KEYS] + failed_refs + [row[key]
        for row in rows for key in RAW_KEYS if key != 'phase']
    require(all(ref in closure.values() for ref in required),
        'Actual primary refs absent from complete sources')
    require(type(spec['deadline_seconds']) is int and spec['deadline_seconds'] > 0 and
        os.environ.get('CUDA_VISIBLE_DEVICES') == '' and
        all(os.environ.get(key) == '2' for key in THREAD_ENV),
        'Prelocked CPU environment/deadline differs')
    for ref in spec['sources']:
        deadline(started, spec['deadline_seconds'])
        bound(ref)
    source, spec_ref = reference(__file__), reference(path)
    require(source == spec['entry_source'], 'Actual CPU entry SOURCE differs')
    owners = [owner_join(row, raw, closure) for row, raw in zip(rows, raws)]
    admission_path = Path(spec['CPU_admission_file'])
    require(admission_path.is_absolute(), 'Absolute own CPU admission required')
    admission_ref = reference(admission_path)
    owner = read(admission_path)
    require(owner['status'] == 'ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE' and
        owner['driver_sha256'] == OWNER_SHA and owner['phase'] == spec['phase'] and
        owner['entry'] == source['path'] and
        owner['entry_args'] == ['--spec', str(path), '--owner-execute'] and
        owner['scientific_spec'] == spec_ref and owner['allowed_cpus'] == [0, 1] and
        owner['cuda_visible_devices'] == '' and owner['automatic_retry'] is False and
        owner['whole_GPU_used_limit_MiB'] == 8192 and
        type(owner['qualified_leaf_deadline_seconds']) is int and
        owner['qualified_leaf_deadline_seconds'] == spec['deadline_seconds'] and
        owner['bound_source_sha256'].get(str(path)) == spec_ref['sha256'] and
        all(owner['bound_source_sha256'].get(ref['path']) == ref['sha256']
            for ref in spec['sources']), 'Actual current CPU owner or complete closure differs')
    require(spec['original_point_heads_source']['path'] != spec['PR_point_heads_source']['path'],
        'Distinct bound original C and PR SOURCE entries required')
    target = Path(spec['Q_path'])
    require(target.is_absolute() and target.parent.is_dir() and not target.exists() and
        not target.with_suffix(target.suffix+'.tmp').exists(), 'Fresh absolute Q output required')
    return dict(closure=closure, owner=owner, source=source, spec_ref=spec_ref,
                admission_ref=admission_ref, normal_owner_joins=len(owners))


def load(ref, name):
    path = bound(ref)
    require(path.suffix == '.py', 'Bound Python SOURCE required')
    source = path.read_bytes()
    require(len(source) == ref['bytes'] and hashlib.sha256(source).hexdigest() == ref['sha256'],
        'SOURCE changed after binding')
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(source, str(path), 'exec'), module.__dict__)
    return module


def resources(psutil, torch):
    require(psutil.Process().cpu_affinity() == [0, 1] and
        torch.get_num_threads() == torch.get_num_interop_threads() == 2 and
        os.environ.get('CUDA_VISIBLE_DEVICES') == '' and
        all(os.environ.get(key) == '2' for key in THREAD_ENV) and
        not torch.cuda.is_initialized(), 'Actual CPU01/threads2/cold CUDA differs')


def imported_files(psutil, torch, gate):
    modules = dict(psutil=psutil, psutil_platform=psutil._psplatform,
        psutil_binary=psutil._psplatform.cext, torch=torch, torch_binary=torch._C,
        torch_tensor=sys.modules['torch._tensor'],
        torch_autograd=sys.modules['torch.autograd'], torch_random=torch.random)
    refs = {name: reference(module.__file__) for name, module in modules.items()}
    objects = dict(torch_Module=torch.nn.Module, torch_Linear=torch.nn.Linear,
        layer_norm=torch.nn.functional.layer_norm,
        binary_cross_entropy_with_logits=torch.nn.functional.binary_cross_entropy_with_logits,
        AdamW=torch.optim.AdamW, Optimizer=torch.optim.Optimizer)
    refs.update({name: reference(inspect.getsourcefile(obj)) for name, obj in objects.items()})
    require(all(ref in gate['closure'].values() and
        gate['owner']['bound_source_sha256'].get(ref['path']) == ref['sha256']
        for ref in refs.values()), 'Actual imported CPU library/binary/optimizer absent or unbound')
    return refs


def cpu_finite(torch, value, shape, message):
    require(isinstance(value, torch.Tensor) and value.device.type == 'cpu' and
        value.dtype == torch.float32 and value.layout == torch.strided and
        tuple(value.shape) == shape and bool(torch.isfinite(value).all()), message)


def parameter_check(torch, head, arm):
    names = C_NAMES if arm == 'C' else PR_NAMES
    shapes = C_SHAPES if arm == 'C' else PR_SHAPES
    expected_count = 40257 if arm == 'C' else 40497
    parameters = tuple(head.named_parameters())
    require(tuple(name for name, _ in parameters) == names and not tuple(head.buffers()),
        'Exact synthetic head parameter names/buffer population differs: ' + arm)
    for (name, parameter), shape in zip(parameters, shapes):
        cpu_finite(torch, parameter, shape, 'Finite CPU FP32 parameter differs: ' + arm + '/' + name)
        require(parameter.requires_grad is True, 'All head parameters must be trainable')
    require(sum(parameter.numel() for _, parameter in parameters) == expected_count,
        'Synthetic head parameter count differs: ' + arm)
    return expected_count


def independent(torch, head, F, e, L, arm):
    """Explicit arithmetic counterpart; no head.forward call here."""
    weights = dict(head.named_parameters())
    F = torch.nn.functional.layer_norm(F.detach(), (256,), weight=None, bias=None, eps=1e-5)
    native = L.detach()
    coarse = native.clamp(-8, 8).unsqueeze(1) / 8
    if arm == 'C':
        e = torch.nn.functional.layer_norm(e.detach(), (256,), weight=None, bias=None, eps=1e-5)
        inputs = torch.cat((F, e, coarse, torch.zeros((len(L), 49), dtype=torch.float32)), dim=1)
        hidden = torch.relu(torch.nn.functional.linear(inputs, weights['hidden1.weight'], weights['hidden1.bias']))
        hidden = torch.relu(torch.nn.functional.linear(hidden, weights['hidden2.weight'], weights['hidden2.bias']))
        residual = torch.nn.functional.linear(hidden, weights['final.weight'], weights['final.bias']).squeeze(-1)
    else:
        hidden = F
        for name in ('hidden1', 'hidden2', 'hidden3'):
            hidden = torch.relu(torch.nn.functional.linear(torch.cat((hidden, coarse), dim=1),
                weights[name+'.weight'], weights[name+'.bias']))
        residual = torch.nn.functional.linear(torch.cat((hidden, coarse), dim=1),
            weights['final.weight'], weights['final.bias']).squeeze(-1)
    band = native.abs() <= 2
    return dict(logits=torch.where(band, native+residual, native), residual=residual, band=band)


def output_check(torch, output, L, expected, cold):
    require(type(output) is dict and set(output) == {'logits', 'residual', 'band'},
        'Exact head output ABI differs')
    count = len(L)
    for name in ('logits', 'residual'):
        cpu_finite(torch, output[name], (count,), 'Finite CPU output shape differs: ' + name)
        require(torch.equal(output[name], expected[name]), 'Independent synthetic arithmetic differs: ' + name)
    require(output['band'].device.type == 'cpu' and output['band'].dtype == torch.bool and
        tuple(output['band'].shape) == (count,) and torch.equal(output['band'], L.detach().abs() <= 2) and
        torch.equal(output['band'], expected['band']), 'Original native |L|<=2 band differs')
    band = output['band']
    require(torch.equal(output['logits'][~band], L.detach()[~band]) and
        torch.equal(output['logits'][band], (L.detach()+output['residual'])[band]),
        'Native outside-band preservation or inside-band residual formula differs')
    if cold:
        require(torch.equal(output['logits'], L.detach()) and
            int(torch.count_nonzero(output['residual'])) == 0,
            'Zero-final synthetic head must degenerate numerically to native logits')


def synthetic_inputs(torch):
    private = torch.Generator(device='cpu')
    private.manual_seed(SYNTHETIC_RECIPE['private_input_seed'])
    F = torch.randn((24, 256), generator=private, dtype=torch.float32)
    e = torch.randn((24, 256), generator=private, dtype=torch.float32)
    L = torch.tensor([-8., -2.0001, -2., -1.9, -0.5, -0.0, 0.0, 0.5,
        1.9, 2., 2.0001, 8.] * 2, dtype=torch.float32)
    patch = torch.rand((24, 49), generator=private, dtype=torch.float32)
    return F, e, L, patch


def input_unchanged(torch, before, after):
    require(all(torch.equal(left, right.detach()) for left, right in zip(before, after)),
        'Synthetic original operands were modified')


def abi_rejection_check(torch, head, F, e, L, patch, arm):
    bad_cases = [(F[:, :255], e, L, patch, arm),
        (F, e[:, :255], L, patch, arm), (F, e, L[:-1], patch, arm),
        (F, e, L, patch[:, :48], arm), (F.double(), e, L, patch, arm),
        (F, e, L, patch+2., arm), (F, e, L, patch, 'INVALID')]
    nonfinite = F.clone()
    nonfinite[0, 0] = float('nan')
    bad_cases.append((nonfinite, e, L, patch, arm))
    for values in bad_cases:
        try:
            head(*values[:4], arm=values[4])
        except (TypeError, ValueError):
            continue
        raise RuntimeError('Invalid synthetic input was accepted: ' + arm)
    return len(bad_cases)


def structural_probe(torch, factory, arm, F, e, L, patch):
    """Nonzero residual parameters prevent trivial cold-head independence."""
    head = factory.make_point_head(torch, SYNTHETIC_RECIPE['factory_seed'])
    parameter_check(torch, head, arm)
    private = torch.Generator(device='cpu')
    private.manual_seed(SYNTHETIC_RECIPE['private_probe_seed'])
    with torch.no_grad():
        for name, parameter in head.named_parameters():
            values = torch.randn(tuple(parameter.shape), generator=private, dtype=torch.float32)*0.025
            parameter.copy_(values+0.05 if name.endswith('.bias') else values)
        head.final.bias.fill_(0.25)
        baseline = head(F, e, L, patch, arm=arm)
        output_check(torch, baseline, L, independent(torch, head, F, e, L, arm), False)
        changed_e = head(F, -e, L, patch, arm=arm)
        changed_patch = head(F, e, L, 1.-patch, arm=arm)
        changed_F = head(-F, e, L, patch, arm=arm)
        require(int(torch.count_nonzero(baseline['residual'])) > 0,
            'Nonzero synthetic structure probe required')
        require(torch.equal(baseline['logits'], changed_patch['logits']) and
            torch.equal(baseline['residual'], changed_patch['residual']),
            'Both C/PR must ignore the supplied raw patch in the residual function')
        e_equal = torch.equal(baseline['residual'], changed_e['residual'])
        require(e_equal if arm == 'PR' else not e_equal,
            'PR e independence / C e-path synthetic contrast differs')
        require(not torch.equal(baseline['residual'], changed_F['residual']),
            'Active F path synthetic probe differs')
        maximum_e_delta = float((baseline['residual']-changed_e['residual']).abs().max())
        maximum_F_delta = float((baseline['residual']-changed_F['residual']).abs().max())
    del head
    return dict(nonzero_parameters_used=True, ignored_patch_exact=True,
        ignored_e_exact=bool(e_equal), C_e_path_active_in_this_probe=arm == 'C',
        F_path_active_in_this_probe=True, maximum_residual_e_delta=maximum_e_delta,
        maximum_residual_F_delta=maximum_F_delta,
        no_claim_that_every_parameter_is_active=True)


def gradient_rows(torch, head):
    rows = []
    for name, parameter in head.named_parameters():
        gradient = parameter.grad
        require(gradient is not None, 'Synthetic parameter gradient missing: ' + name)
        cpu_finite(torch, gradient, tuple(parameter.shape), 'Finite CPU gradient differs: ' + name)
        rows.append(dict(parameter=name, scalars=parameter.numel(),
            nonzero_gradient_scalars=int(torch.count_nonzero(gradient)),
            maximum_abs_gradient=float(gradient.abs().max())))
    return rows


def backward_update_check(torch, head, arm, F, e, L, patch):
    parameters_before = {name: value.detach().clone() for name, value in head.named_parameters()}
    operands = [value.detach().clone().requires_grad_(True) for value in (F, e, L, patch)]
    originals = [value.detach().clone() for value in operands]
    optimizer = torch.optim.AdamW(head.parameters(), lr=SYNTHETIC_RECIPE['AdamW_lr'],
        betas=tuple(SYNTHETIC_RECIPE['AdamW_betas']), eps=SYNTHETIC_RECIPE['AdamW_eps'],
        weight_decay=SYNTHETIC_RECIPE['AdamW_weight_decay'], foreach=False)
    targets = torch.ones_like(L)
    stages = []
    for step in (1, 2):
        optimizer.zero_grad(set_to_none=True)
        result = head(*operands, arm=arm)
        output_check(torch, result, operands[2], independent(torch, head, *operands[:3], arm), step == 1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(result['logits'], targets, reduction='mean')
        require(loss.device.type == 'cpu' and loss.dtype == torch.float32 and
            loss.numel() == 1 and bool(torch.isfinite(loss)), 'Finite synthetic point BCE required')
        loss.backward()
        gradients = gradient_rows(torch, head)
        require(all(value.grad is None for value in operands), 'Native operands must remain detached')
        if step == 1:
            require(all(row['nonzero_gradient_scalars'] == 0 for row in gradients
                if not row['parameter'].startswith('final.')) and
                all(row['nonzero_gradient_scalars'] > 0 for row in gradients
                    if row['parameter'].startswith('final.')),
                'Cold zero-final head must expose only nonzero final-layer gradients')
        else:
            require(all(row['nonzero_gradient_scalars'] > 0 for row in gradients),
                'Warm synthetic backward must reach every head parameter tensor')
        if arm == 'C':
            require(int(torch.count_nonzero(head.hidden1.weight.grad[:, 513:])) == 0,
                'C zero-patch columns must have no data gradient')
        before_step = {name: value.detach().clone() for name, value in head.named_parameters()}
        optimizer.step()
        changed = [name for name, value in head.named_parameters()
            if not torch.equal(value.detach(), before_step[name])]
        require('final.weight' in changed and 'final.bias' in changed,
            'Both final tensors must update on each synthetic AdamW step')
        if step == 1:
            require(set(changed) == {'final.weight', 'final.bias'},
                'With synthetic weight_decay=0 cold hidden parameters must remain unchanged')
        else:
            require(len(changed) == len(tuple(head.parameters())),
                'Warm synthetic AdamW must change every head parameter tensor')
        parameter_check(torch, head, arm)
        states = []
        for name, parameter in head.named_parameters():
            state = optimizer.state[parameter]
            require(set(state) == {'step', 'exp_avg', 'exp_avg_sq'} and
                isinstance(state['step'], torch.Tensor) and state['step'].device.type == 'cpu' and
                state['step'].numel() == 1 and float(state['step']) == step,
                'Actual synthetic AdamW per-tensor step differs')
            for key in ('exp_avg', 'exp_avg_sq'):
                cpu_finite(torch, state[key], tuple(parameter.shape), 'Finite AdamW state differs: ' + name + '/' + key)
            states.append(dict(parameter=name, step=float(state['step'])))
        stages.append(dict(step=step, loss=float(loss.detach()), gradients=gradients,
            changed_parameter_tensors=changed, optimizer_parameter_steps=states,
            detached_input_gradients_all_none=True))
        input_unchanged(torch, originals, operands)
    with torch.no_grad():
        final_result = head(F, e, L, patch, arm=arm)
        output_check(torch, final_result, L, independent(torch, head, F, e, L, arm), False)
        require(not torch.equal(final_result['logits'][final_result['band']], L[final_result['band']]),
            'At least one in-band logit must update in this synthetic trajectory')
    changed_total = [name for name, value in head.named_parameters()
        if not torch.equal(value.detach(), parameters_before[name])]
    del optimizer
    for value in operands:
        value.grad = None
    return dict(backward_calls=2, AdamW_updates=2, stages=stages,
        changed_parameter_tensors=changed_total, native_input_gradient_path_absent=True,
        C_unused_patch_column_data_gradients_zero=arm == 'C',
        actual_training_recipe_replayed=False, two_synthetic_updates_only=True)


def head_checks(torch, factory, arm, F, e, L, patch, started, seconds):
    deadline(started, seconds)
    before_rng = torch.random.get_rng_state().clone()
    head = factory.make_point_head(torch, SYNTHETIC_RECIPE['factory_seed'])
    repeat = factory.make_point_head(torch, SYNTHETIC_RECIPE['factory_seed'])
    count = parameter_check(torch, head, arm)
    parameter_check(torch, repeat, arm)
    require(torch.equal(before_rng, torch.random.get_rng_state()) and
        all(torch.equal(value, dict(repeat.named_parameters())[name])
            for name, value in head.named_parameters()),
        'Factory must preserve external CPU RNG and reproduce its private seed')
    require(int(torch.count_nonzero(head.final.weight)) == 0 and
        int(torch.count_nonzero(head.final.bias)) == 0, 'Exactly zero final layer required')
    originals = [value.clone() for value in (F, e, L, patch)]
    with torch.no_grad():
        result = head(F, e, L, patch, arm=arm)
        output_check(torch, result, L, independent(torch, head, F, e, L, arm), True)
        empty = head(F[:0], e[:0], L[:0], patch[:0], arm=arm)
        output_check(torch, empty, L[:0], independent(torch, head, F[:0], e[:0], L[:0], arm), True)
        invalid_count = abi_rejection_check(torch, head, F, e, L, patch, arm)
    input_unchanged(torch, originals, (F, e, L, patch))
    structure = structural_probe(torch, factory, arm, F, e, L, patch)
    deadline(started, seconds)
    backward = backward_update_check(torch, head, arm, F, e, L, patch)
    require(torch.equal(before_rng, torch.random.get_rng_state()) and not torch.cuda.is_initialized(),
        'Synthetic checks must preserve CPU RNG and cold CUDA')
    input_unchanged(torch, originals, (F, e, L, patch))
    del head, repeat
    gc.collect()
    return dict(parameters=count, parameter_tensors=6 if arm == 'C' else 8,
        factory_same_seed_repeat_exact=True, factory_external_CPU_RNG_unchanged=True,
        zero_final_native_identity_checked=True, original_band_including_exact_boundary_checked=True,
        outside_band_preservation_checked=True, independent_formula_checked=True,
        empty_point_batch_checked=True, invalid_input_rejections=invalid_count,
        structural_probe=structure, synthetic_backward=backward)


def atomic_write(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    require(path.parent.is_dir() and not path.exists() and not temporary.exists(),
        'Fresh output file required')
    with temporary.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
    temporary.replace(path)


def qualify(spec_path):
    started = time.monotonic()
    path = Path(spec_path)
    require(path.is_absolute(), 'Absolute ROOT-authored CPU SPEC required')
    path = path.resolve(strict=True)
    spec = read(path)
    gate = source_gate(path, spec, started)
    import psutil
    require(psutil.Process().cpu_affinity() == [0, 1],
        'Actual CPU01 required before numerical imports')
    import torch
    require(torch.__version__ == '2.0.1+cu118' and not torch.cuda.is_initialized(),
        'Original Torch and cold CUDA required')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    resources(psutil, torch)
    libraries = imported_files(psutil, torch, gate)
    external_rng = torch.random.get_rng_state().clone()
    C_factory = load(spec['original_point_heads_source'], '_N10_bound_original_C_head_factory')
    PR_factory = load(spec['PR_point_heads_source'], '_N10_bound_PR_FL91_head_factory')
    require(callable(getattr(C_factory, 'make_point_head', None)) and
        callable(getattr(PR_factory, 'make_point_head', None)), 'Both bound factories required')
    F, e, L, patch = synthetic_inputs(torch)
    require(int(torch.count_nonzero(L.abs() <= 2)) == 16 and
        bool((L == -2.).any()) and bool((L == 2.).any()) and bool((L.abs() > 2).any()),
        'Synthetic probes must include both exact band boundaries and out-of-band points')
    checks = {}
    for arm, factory in (('C', C_factory), ('PR', PR_factory)):
        checks[arm] = head_checks(torch, factory, arm, F, e, L, patch,
            started, spec['deadline_seconds'])
        resources(psutil, torch)
        print(json.dumps(dict(N10_synthetic_head_checked=arm,
            parameters=checks[arm]['parameters'])), flush=True)
    require(torch.equal(external_rng, torch.random.get_rng_state()) and
        not torch.cuda.is_initialized() and 'mmdet' not in sys.modules,
        'No native model import, external RNG change or CUDA initialization allowed')
    for ref in spec['sources']:
        deadline(started, spec['deadline_seconds'])
        bound(ref)
    require(reference(path) == gate['spec_ref'] and reference(__file__) == gate['source'] and
        reference(spec['CPU_admission_file']) == gate['admission_ref'] and
        imported_files(psutil, torch, gate) == libraries,
        'Final SPEC/SOURCE/admission/library closure changed')
    resources(psutil, torch)
    deadline(started, spec['deadline_seconds'])
    result = dict(schema=Q_SCHEMA, status=STATUS, phase=spec['phase'],
        source=gate['source'], spec=gate['spec_ref'], CPU_admission=gate['admission_ref'],
        original_point_heads_source=spec['original_point_heads_source'],
        PR_point_heads_source=spec['PR_point_heads_source'], RAW_inputs=spec['RAW_inputs'],
        prior_failed_CPU_receipt=spec['prior_failed_CPU_receipt'],
        additional_failed_RAW_inputs=spec['additional_failed_RAW_inputs'],
        normal_actual_terminal0_owner_joins=gate['normal_owner_joins'],
        preserved_failed1_original_terminal_gates=3,
        C_parameters=40257, PR_parameters=40497, parameter_difference=240,
        parameter_count_difference_does_not_establish_information_or_compute_matching=True,
        synthetic_cold_identity_and_band_checked=True,
        synthetic_backward_update_checked=True, synthetic_CPU_only=True,
        synthetic_independent_formula_checked=True, synthetic_private_CPU_RNG_unchanged=True,
        synthetic_input_independence_nonzero_parameter_probe_checked=True,
        synthetic_recipe=SYNTHETIC_RECIPE, synthetic_checks=checks,
        no_task_efficacy_claim=True, dataset_loaded=False, GT_loaded=False,
        cached_data_loaded=False, model_loaded=False, checkpoint_loaded=False,
        native_teacher_forwards=0, synthetic_head_backward_calls=4,
        synthetic_head_AdamW_updates=4, real_training_updates=0,
        scientific_novelty_established=False, convergence_established=False,
        generalization_established=False, GPU_equivalence_established=False,
        actual_imported_files=libraries, CPU_affinity=[0, 1], threads=2,
        CUDA_initialized=False, NumPy_explicitly_used=False,
        elapsed_seconds=time.monotonic()-started, deadline_seconds=spec['deadline_seconds'],
        automatic_retry=False, current_owned_CPU_terminal_self_certified=False,
        whole_leaf_VRAM_certified=False,
        evidence_limits=[
            'Ancestor RAW/owner/entry/SPEC/PB/bootstrap identities are joined; unused historical numerical payloads are not replayed or reopened.',
            'Only small private synthetic CPU FP32 operands and the two exact bound head factories are tested.',
            'No real data, GT, cache, teacher, checkpoint, task metric, native loss or training trajectory is loaded or replayed.',
            'Zero-final degeneration uses numerical equality; signed-zero bitwise identity is not asserted.',
            'Nonzero probes test e/patch independence and active F paths for these operands; every scalar need not be active.',
            'C has zero-patch columns and PR has different information/depth; nearby parameter counts do not match compute or information.',
            'Two synthetic AdamW steps use weight_decay=0 to separate cold data gradients from parameter decay; they are not the real training recipe.',
            'A passing synthetic qualification establishes no task benefit, innovation, convergence, generalization or GPU numerical equivalence.',
            'ROOT separately closes and checks the actual original held process; the current leaf cannot certify its own terminal RAW.'
        ])
    qpath = Path(spec['Q_path'])
    atomic_write(qpath, result)
    print(json.dumps(dict(status=STATUS, Q=reference(qpath)), allow_nan=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--owner-execute', action='store_true')
    args = parser.parse_args()
    require(args.owner_execute, 'Explicit single ROOT-owned CPU execution required')
    qualify(args.spec)
