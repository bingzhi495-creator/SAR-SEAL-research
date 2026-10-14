"""UNBOUND N14 single context-memory addon, equal-budget E24 warm fine-tune.
No scientific imports/file access at module import. Exact candidate/cache/score
policy stays inherited LOG; new stage24 uses reset AdamW in both arms.
"""
import argparse
import copy
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

OWNER_SHA="ef9a9f17d4fb1bc3464936095ef799021dd880f8b15488d558a871c08b6e1924"
RAW_KEYS=("receipt","phase","admission","phase_binding","bootstrap")
THREAD_ENV=("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS")
ARMS=("BASE_CONTINUATION","MEMORY_ADDON")
SCHEMA="ROOT-N14-LOG-E24-memory-paired-warm-stage-GPU-spec-v1"
STATUS="N14_MEMORY_PAIRED_STAGE_LEAF_COMPLETE_PENDING_INDEPENDENT_QUALIFICATION"

def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def api(module, name):
    value = getattr(module, name, None)
    require(callable(value), 'ROOT must adapt/review missing external ABI: ' + name)
    return value


def read(path):
    def reject(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), parse_constant=reject)


def ref_shape(ref):
    require(type(ref) is dict and set(ref) == {'path', 'bytes', 'sha256'} and
            type(ref['path']) is str and Path(ref['path']).is_absolute() and
            type(ref['bytes']) is int and ref['bytes'] >= 0 and type(ref['sha256']) is str and
            len(ref['sha256']) == 64 and set(ref['sha256']) <= set('0123456789abcdef'),
            'Malformed absolute pinned reference')


def reference(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def bound(ref):
    ref_shape(ref)
    require(reference(ref['path']) == ref, 'Pinned input changed: ' + ref['path'])
    return Path(ref['path'])


def closed(closure, ref):
    ref_shape(ref)
    require(closure.get(ref['path']) == ref, 'Missing exact complete SOURCE reference: ' + ref['path'])
    return bound(ref)


def load(ref, name):
    path = bound(ref)
    source = path.read_bytes()
    require(len(source) == ref['bytes'] and hashlib.sha256(source).hexdigest() == ref['sha256'], 'SOURCE changed before import')
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(source, str(path), 'exec'), module.__dict__)
    return module


def terminal_raw_only(ref, phase, code):
    # Reads only original RAW JSON; no SOURCE hash/import or process probe.
    ref_shape(ref)
    path = Path(ref['path'])
    require(path.name == 'PARENT_RECEIPT.json' and type(phase) is str and phase and path.parent.name == phase,
            'Original RAW locator required')
    raw = read(path)
    require(raw['status'] == 'ACTUAL_OWNED_TERMINAL' and type(raw['returncode']) is int and raw['returncode'] == code and
            type(raw['original_identity']) is dict and raw['original_identity'] and raw['root_wait_reaped'] is True and
            raw['held_handle_exit'] == dict(wait_state='WAIT_OBJECT_0', actual_exit_code=code) and
            type(raw['held_handle_exit']['actual_exit_code']) is int and
            raw['numeric_birth_absence']['original_birth_absent'] is True and raw.get('error') is None and
            raw.get('abort') is None and raw['wait_errors'] is False and raw['GPU_observation_errors'] == [],
            'Actual held exit/reap/original birth absence required before any SHA/import')
    if 'phase' in raw:
        require(raw['phase'] == phase, 'Original RAW phase differs')
    sha, peak = raw['admission_sha256'], raw['whole_GPU_observed_peak_MiB']
    require(type(sha) is str and len(sha) == 64 and set(sha) <= set('0123456789abcdef') and
            type(peak) in (int, float) and math.isfinite(peak) and 0 <= peak <= 8192 and
            raw['whole_GPU_used_limit_MiB'] == 8192 and raw['whole_leaf_VRAM_certified'] is False and
            raw['GPU_scope'] == 'whole GPU observed, not per-process WDDM allocation',
            'Preserved RAW resource/admission scope differs')
    return raw


def check_pin(pin, raw, phase, code):
    require(type(pin) is dict and set(pin) == {'phase', 'returncode', 'original_identity', 'admission_sha256'} and
            pin['phase'] == phase and type(pin['returncode']) is int and pin['returncode'] == code and
            pin['original_identity'] == raw['original_identity'] and pin['admission_sha256'] == raw['admission_sha256'],
            'Original actual owner differs from its immutable identity/admission pin')


def ninth_failed_raw_only(spec):
    extra = spec['additional_preserved_independent_CPU_failure']
    require(type(extra) is dict and set(extra) == {'RAW_ref', 'phase', 'pin', 'returncode'} and
            type(extra['returncode']) is int and extra['returncode'] == 1,
            'Independent failed9 needs exactly RAW_ref/phase/pin/returncode1')
    raw = terminal_raw_only(extra['RAW_ref'], extra['phase'], 1)
    check_pin(extra['pin'], raw, extra['phase'], 1)
    require(type(spec['total_failed_RAW_count']) is int and spec['total_failed_RAW_count'] == 9 and
            type(spec['total_terminal_identity_pin_count']) is int and spec['total_terminal_identity_pin_count'] == 104 and
            extra['RAW_ref']['path'] not in spec['terminal_identity_pins'],
            'Failed9/pins103 must remain separate from the original pins102')
    return raw


def pre_gate(spec, minimum):
    # Inherited RAW/source contract of the original GPU pre_gate, with
    # minimum94 supplied here. Additional failed9 has ALREADY passed RAW-only.
    rows = spec['RAW_inputs']
    expected = spec['normal_RAW_count']
    require(type(expected) is int and expected >= minimum and type(rows) is list and len(rows) == expected and
            len({row['receipt']['path'] for row in rows}) == expected, 'All original normal RAW ancestors required')
    normals = [terminal_raw_only(row['receipt'], row['phase'], 0) for row in rows]
    failed_ref = spec['prior_failed_CPU_receipt']
    failed_phase = Path(failed_ref['path']).parent.name
    failed = terminal_raw_only(failed_ref, failed_phase, 1)
    extra = spec['additional_failed_RAW_inputs']
    require(type(extra) is list and len(extra) == 7 and
            all(type(row) is dict and set(row) == {'receipt', 'phase', 'returncode'} and
                type(row['returncode']) is int and row['returncode'] == 1 for row in extra),
            'Do not alter the original extra failed7 list or preserved failed8')
    failures = [failed] + [terminal_raw_only(row['receipt'], row['phase'], 1) for row in extra]
    failed_paths = [failed_ref['path']] + [row['receipt']['path'] for row in extra]
    require(len(set(failed_paths)) == 8 and not {row['receipt']['path'] for row in rows}.intersection(failed_paths),
            'Original normal94 and failed8 owners must remain distinct')
    all_rows = [dict(row, returncode=0) for row in rows] + [dict(receipt=failed_ref, phase=failed_phase, returncode=1)] + extra
    pins = spec['terminal_identity_pins']
    require(type(pins) is dict and set(pins) == {row['receipt']['path'] for row in all_rows},
            'Original normal-plus-failed8 pins must remain unchanged')
    for row, raw in zip(all_rows, normals + failures):
        check_pin(pins[row['receipt']['path']], raw, row['phase'], row['returncode'])
    require(expected == 95 and len(pins) == 103, 'Current normal95/legacy pins103 required')
    # FIRST HASH: failed9 and all original normal94/failed8 RAW checks passed.
    closure = {ref['path']: ref for ref in spec['sources']}
    require(len(closure) == len(spec['sources']), 'Unique SOURCE closure required')
    for ref in spec['sources']:
        bound(ref)
    require(all(row[key] in closure.values() for row in rows for key in RAW_KEYS if key != 'phase') and
            failed_ref in closure.values() and all(row['receipt'] in closure.values() for row in extra),
            'Original owner metadata and failed8 receipts must be sources-bound')
    return normals, closure



def imported_files(psutil,torch,np,gate):
    modules=dict(psutil=psutil,psutil_platform=psutil._psplatform,psutil_binary=psutil._psplatform.cext,
        torch=torch,torch_binary=torch._C,torch_tensor=sys.modules['torch._tensor'],torch_autograd=sys.modules['torch.autograd'],
        torch_random=torch.random,numpy=np,numpy_binary=np.core._multiarray_umath)
    result={name:reference(module.__file__) for name,module in modules.items()}
    objects=dict(torch_Module=torch.nn.Module,torch_Linear=torch.nn.Linear,AdamW=torch.optim.AdamW,
        Optimizer=torch.optim.Optimizer,point_math=torch.nn.functional.binary_cross_entropy_with_logits,
        torch_serialization=torch.load)
    result.update({name:reference(inspect.getsourcefile(obj)) for name,obj in objects.items()})
    require(all(ref in gate['closure'].values() and gate['admission']['bound_source_sha256'].get(ref['path'])==ref['sha256']
        for ref in result.values()),'Actual imported library/source/binary is not fully pinned')
    return result


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def digest(value):
    a = value.detach().cpu().contiguous().numpy()
    return dict(shape=list(a.shape), dtype=str(a.dtype),
                sha256=hashlib.sha256(a.tobytes()).hexdigest())


def cpu_tree(torch, value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(torch, v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(torch, v) for v in value)
    return value


def gate_inputs(path, spec):
    require(spec['schema'] == SCHEMA and spec['first_stage_epoch'] == 1 and
            spec['last_stage_epoch'] == 8 and spec['fixed_stage_epochs'] == 24 and
            spec['optimizer_reset_both'] is True and spec['seed'] == 202610120701 and
            spec['module_seed'] == 202610140801 and spec['order_seed'] == 202610140801 and
            spec['deadline_seconds'] == 1800 and spec['allocator_limit_MiB'] == 4096 and
            spec['whole_GPU_limit_MiB'] == 8192 and spec['automatic_retry'] is False,
            'Only prelocked paired new-stage1--8 of stage24 is admitted')
    ninth_failed_raw_only(spec)
    normals, closure = pre_gate(spec, 95)
    closed(closure, spec['additional_preserved_independent_CPU_failure']['RAW_ref'])
    for k in ('entry_source', 'memory_source', 'carrier_spec', 'carrier_evaluation_source',
              'carrier_curve_qualified_summary', 'protocol'):
        closed(closure, spec[k])
    carrier = read(bound(spec['carrier_spec']))
    require(carrier['seed'] == spec['seed'] and carrier['head_endpoint_epoch'] == 24 and
            carrier['entry_source'] == spec['carrier_evaluation_source'],
            'Common qualified first701 LOG E24 carrier required')
    curve = read(closed(closure, spec['carrier_curve_qualified_summary']))
    curve_report = read(closed(closure, curve['evaluation_GPU_report']))
    require(curve['status'] == 'QUALIFIED_E24_CURVE_NOT_E48_PRIMARY_CLAIM' and
            curve['single_seed_AP_qualified'] is True and
            curve['evaluation_GPU_actual_returncode'] == curve['evaluation_CPU_actual_returncode'] == 0 and
            curve_report['spec'] == spec['carrier_spec'] and curve_report['seed'] == spec['seed'] and
            curve_report['head_endpoint_epoch'] == 24,
            'Qualified E24 curve must link to this actual common carrier')
    context = dict(carrier)
    for k in ('sources', 'RAW_inputs', 'normal_RAW_count', 'prior_failed_CPU_receipt',
              'additional_failed_RAW_inputs', 'terminal_identity_pins', 'entry_source',
              'phase', 'GPU_admission_file'):
        context[k] = spec[k]
    lineage = load(carrier['endpoint_lineage_source'], '_N14_actual_carrier_lineage')
    gate = lineage.own_gate(path, context, __file__, closure, normals, cpu=False)
    endpoint = lineage.endpoint_gate(context, closure, gate['owners'])
    gate.update(endpoint)
    gate.update(context=context, lineage=lineage, cache_Q=endpoint['cache_q'],
                parent=endpoint['parent'])
    require(len(gate['cache_report']['packet_refs']) == 741 and
            gate['cache_report']['raw_GT'] == 1661,
            'Complete qualified FIT741/1661 input required')
    return gate


def packet(torch, gate, index):
    report = gate['cache_report']
    require(report['packet_refs'][index] == gate['cache_Q']['packet_refs'][index],
            'Qualified immutable packet order required')
    return gate['common'].load_packet(torch, report, index)


def forward(torch, head, gate, value, g, device):
    F, e, L, y, flag = gate['common'].instance_inputs(torch, value, g, 'LOG_PR', device)
    result = gate['common'].region_logits(torch, head, F, e, L, flag)
    require(torch.equal(result['logits'][~flag], L[~flag]), 'LOG exterior changed')
    loss = torch.nn.functional.binary_cross_entropy_with_logits(result['logits'], y)
    require(bool(torch.isfinite(loss)), 'Nonfinite point loss')
    return loss, result, (F, e, L, y, flag)


def preflight(torch, gate, heads):
    value = packet(torch, gate, 0)
    F, e, L, y, flag = gate['common'].instance_inputs(torch, value, 0, 'LOG_PR', 'cpu')
    patch = L.new_zeros((L.numel(), 49))
    outside = torch.random.get_rng_state().clone()
    a = heads[ARMS[0]](F, e, L, patch, arm='PR')
    b = heads[ARMS[1]](F, e, L, patch, arm='PR')
    require(torch.equal(a['residual'], b['residual']) and torch.equal(a['logits'], b['logits']),
            'Zero addon must exactly reproduce common warm parent')
    perm = torch.arange(L.numel()-1, -1, -1, dtype=torch.int64)
    c = heads[ARMS[1]](F[perm], e[perm], L[perm], patch[perm], arm='PR')
    require(torch.equal(b['residual'][perm], c['residual']),
            'Memory must be point-permutation equivariant at initialization')
    test_head = copy.deepcopy(heads[ARMS[1]])
    for p in test_head.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(test_head.parameters(), lr=3e-4, weight_decay=1e-4)
    gradient = []
    for step in range(2):
        opt.zero_grad(set_to_none=True)
        loss, _, _ = forward(torch, test_head, gate, value, 0, 'cpu')
        loss.backward()
        require(all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                    for p in test_head.parameters()), 'Present finite addon/base gradients required')
        gradient.append({n: bool((p.grad != 0).any()) for n, p in test_head.named_parameters()})
        opt.step()
    require(any(gradient[0][n] for n in gradient[0] if n.endswith('output.weight')),
            'Zero projection receives no first-step learning signal')
    require(any(gradient[1][n] for n in gradient[1] if 'memory_' in n or 'query.' in n),
            'Retrieval receives no next-step learning signal')
    require(torch.equal(outside, torch.random.get_rng_state()) and not torch.cuda.is_initialized(),
            'Preflight changed global RNG or initialized CUDA')
    return dict(real_FIT_image=0, real_GT_index=0, initial_exact_parent_identity=True,
                initial_permutation_equivariance=True, exterior_identity=True,
                two_step_gradient_nonzero=gradient,
                scope='Behavior preflight on cloned head only; not AP efficacy')


def measure(torch, gate, heads, started, spec):
    sums = {arm: [] for arm in ARMS}
    for head in heads.values():
        head.eval()
    with torch.no_grad():
        for index in range(741):
            require(time.monotonic()-started < spec['deadline_seconds'], 'Endpoint deadline exceeded')
            value = packet(torch, gate, index)
            for g in range(len(value['query_ids'])):
                for arm in ARMS:
                    loss, _, _ = forward(torch, heads[arm], gate, value, g, 'cuda:0')
                    sums[arm].append(float(loss.detach().cpu()))
            del value
    for head in heads.values():
        head.train()
    require(all(len(v) == 1661 for v in sums.values()), 'Complete endpoint1661 required')
    return {arm: math.fsum(v)/1661 for arm, v in sums.items()}


def run(path):
    started = time.monotonic()
    path = Path(path).resolve(strict=True)
    spec = read(path)
    gate = gate_inputs(path, spec)
    import psutil
    import torch
    import numpy as np
    require(psutil.Process().cpu_affinity() == [0, 1] and not torch.cuda.is_initialized(),
            'Actual affinity and cold CUDA required')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    require(torch.get_num_threads() == torch.get_num_interop_threads() == 2,
            'Exactly two torch threads required')
    imported = imported_files(psutil, torch, np, gate)
    evaluator = load(spec['carrier_evaluation_source'], '_N14_original_parent_loader')
    old_heads, states, proofs = evaluator.load_heads(torch, gate['context'], gate)
    factory = load(gate['context']['point_heads_source'], '_N14_original_PR_factory')
    base = old_heads.pop('LOG_PR')
    memory_base = factory.make_point_head(torch, spec['seed'])
    memory_base.load_state_dict(states['LOG_PR'], strict=True)
    memory_module = load(spec['memory_source'], '_N14_memory_module')
    outside_CPU = torch.random.get_rng_state().clone()
    memory = memory_module.make_context_memory_head(torch, memory_base,
        seed=spec['module_seed'], tokens=8, dim=8)
    heads = dict(BASE_CONTINUATION=base, MEMORY_ADDON=memory)
    require(torch.equal(outside_CPU, torch.random.get_rng_state()) and not torch.cuda.is_initialized(),
            'Private module construction changed outside RNG/CUDA')
    require(sum(p.numel() for p in base.parameters()) == 40497 and
            sum(p.numel() for p in memory.parameters()) == 42698,
            'Prelocked40497/42698 parameter counts required')
    for head in heads.values():
        for p in head.parameters():
            p.requires_grad_(True)
    initial = {arm: {n: digest(v) for n, v in head.state_dict().items()}
               for arm, head in heads.items()}
    behavior = preflight(torch, gate, heads)
    output = Path(spec['output_directory'])
    require(not output.exists(), 'Never overwrite old science output')
    output.mkdir()
    write_new(output/'START.json', dict(schema=SCHEMA, phase=spec['phase'],
        source=gate['source'], spec=gate['spec_ref'], GPU_admission=gate['admission_ref'],
        parent=spec['carrier_spec'], common_LOG_parent_proofs=proofs['LOG_PR'],
        initial_state_digests=initial, behavior=behavior, stage_epochs=24,
        first_stage_epoch=1, last_stage_epoch=8, optimizer_reset_both=True,
        original_parent_epochs=24, continuous_parent_plus_stage=False,
        native_weights_updated=False, no_DEV=True, no_TEST=True, no_AP=True))
    del old_heads, states, proofs, memory_base, base, memory
    torch.cuda.set_device(0)
    require(torch.cuda.device_count() == 1, 'Exactly CUDA0 required')
    torch.cuda.set_per_process_memory_fraction(4096*1024**2/torch.cuda.get_device_properties(0).total_memory, 0)
    torch.cuda.reset_peak_memory_stats(0)
    outside_GPU = torch.cuda.get_rng_state(0).clone()
    opts = {}
    for arm in ARMS:
        heads[arm].to('cuda:0').train()
        opts[arm] = torch.optim.AdamW(heads[arm].parameters(), lr=3e-4, weight_decay=1e-4)
        require(len(opts[arm].state) == 0, 'Both new stage optimizers must be fresh')
    order_rng = torch.Generator(device='cpu')
    order_rng.manual_seed(spec['order_seed'])
    history, orders = [], []
    activity = {arm: {n: dict(present=0, nonzero=0) for n, _ in heads[arm].named_parameters()}
                for arm in ARMS}
    with (output/'UPDATE_TRACE.jsonl').open('x', encoding='utf-8', newline='\n') as trace:
        for epoch in range(1, 9):
            order = torch.randperm(741, generator=order_rng).tolist()
            order_sha = hashlib.sha256(json.dumps(order, separators=(',', ':')).encode('ascii')).hexdigest()
            orders.append(dict(stage_epoch=epoch, indices=order, sha256=order_sha))
            count, losses = 0, {arm: [] for arm in ARMS}
            for index in order:
                require(time.monotonic()-started < spec['deadline_seconds'], 'Training leaf deadline exceeded')
                value = packet(torch, gate, index)
                for g in range(len(value['query_ids'])):
                    row = dict(stage_epoch=epoch, image_index=index, GT_index=g,
                        annotation_id=value['identity']['ann_ids'][g], query_id=int(value['query_ids'][g]),
                        stage_update=(epoch-1)*1661+count+1, BCE={})
                    for arm in ARMS:
                        head, opt = heads[arm], opts[arm]
                        opt.zero_grad(set_to_none=True)
                        loss, result, operands = forward(torch, head, gate, value, g, 'cuda:0')
                        require(loss.requires_grad, 'Loss must connect to both learned branches')
                        loss.backward()
                        for name, p in head.named_parameters():
                            require(p.grad is not None and bool(torch.isfinite(p.grad).all()), 'Missing or nonfinite gradients')
                            activity[arm][name]['present'] += 1
                            activity[arm][name]['nonzero'] += int(bool((p.grad != 0).any()))
                        opt.step()
                        opt.zero_grad(set_to_none=True)
                        require(all(bool(torch.isfinite(p).all()) for p in head.parameters()), 'Nonfinite model state')
                        observed = float(loss.detach().cpu())
                        row['BCE'][arm] = observed
                        losses[arm].append(observed)
                    trace.write(json.dumps(row, allow_nan=False)+'\n')
                    count += 1
                del value, loss, result, operands
            require(count == 1661 and all(len(x) == 1661 for x in losses.values()),
                    'Both arms must complete1661 updates per epoch')
            endpoint = measure(torch, gate, heads, started, spec)
            h = dict(stage_epoch=epoch, paired_image_order_sha256=order_sha, updates_per_arm=1661,
                cumulative_stage_updates=epoch*1661,
                online_BCE={arm: math.fsum(v)/1661 for arm, v in losses.items()},
                full_FIT_BCE=endpoint)
            history.append(h)
            trace.flush()
            print(json.dumps(h), flush=True)
    checkpoints = {}
    for arm in ARMS:
        head, opt = heads[arm], opts[arm]
        require(len(opt.state) == len(tuple(head.parameters())) and
                all(int(opt.state[p]['step']) == 13288 for p in head.parameters()),
                'Actual AdamW must record13288 new stage updates for every parameter')
        meta = dict(schema='N14-context-memory-stage-checkpoint-v1', arm=arm,
            source=gate['source'], spec=gate['spec_ref'], GPU_admission=gate['admission_ref'],
            phase=spec['phase'], parent_spec=spec['carrier_spec'], parent_head_epochs=24,
            common_parent_arm='LOG_PR', stage_epochs_completed=8, stage_updates=13288,
            fixed_stage_epochs=24, optimizer_reset_at_stage_start=True,
            memory_source=spec['memory_source'], head_factory_source=gate['context']['point_heads_source'],
            parameters=sum(p.numel() for p in head.parameters()), initial_state_digests=initial[arm],
            history=history, epoch_orders=orders, gradient_activity=activity[arm],
            cache_CPU_qualification=gate['context']['cache_CPU_qualification'],
            natural_query_scores_changed=False, support_changed=False, no_DEV=True, no_TEST=True,
            no_AP=True, native_weights_updated=False, independent_qualified=False)
        cp = dict(meta=meta, state_dict=cpu_tree(torch, head.state_dict()),
            optimizer_state_dict=cpu_tree(torch, opt.state_dict()),
            private_order_state=order_rng.get_state().clone())
        target = output/('HEAD_'+arm+'_STAGE8.pth')
        with target.open('xb') as stream:
            torch.save(cp, stream)
        replay = torch.load(str(target), map_location='cpu')
        require(all(torch.equal(v, replay['state_dict'][n]) for n, v in cp['state_dict'].items()) and
                replay['meta'] == meta, 'Exact own checkpoint readback required')
        checkpoints[arm] = reference(target)
        del cp, replay, opt, head
    require(torch.equal(outside_CPU, torch.random.get_rng_state()) and
            torch.equal(outside_GPU, torch.cuda.get_rng_state(0)), 'Outside RNG changed')
    peak = torch.cuda.max_memory_allocated(0)
    for head in heads.values():
        head.cpu()
    del head
    opts.clear()
    heads.clear()
    gc.collect()
    torch.cuda.synchronize()
    torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache()
    require(torch.cuda.memory_allocated(0) == 0, 'Live GPU aliases remain')
    require(psutil.Process().cpu_affinity() == [0, 1] and
            torch.get_num_threads() == torch.get_num_interop_threads() == 2 and
            'mmdet' not in sys.modules, 'Final thread/affinity/no native model contract differs')
    require(imported_files(psutil, torch, np, gate) == imported, 'Imported library closure changed')
    for ref in spec['sources']:
        bound(ref)
    require(reference(path) == gate['spec_ref'] and reference(__file__) == gate['source'] and
            reference(spec['GPU_admission_file']) == gate['admission_ref'], 'Own execution binding changed')
    report = dict(status=STATUS, schema=SCHEMA, source=gate['source'], spec=gate['spec_ref'],
        GPU_admission=gate['admission_ref'], phase=spec['phase'], checkpoints=checkpoints,
        original_parent_epochs=24, new_stage_epochs=8, fixed_stage_epochs=24,
        updates_per_arm=13288, history=history, behavior_preflight=behavior,
        initial_state_digests=initial, gradient_activity=activity, imported_library_refs=imported,
        trace=reference(output/'UPDATE_TRACE.jsonl'), start=reference(output/'START.json'),
        elapsed_seconds=time.monotonic()-started, peak_torch_allocated_bytes=peak,
        own_terminal_self_certified=False, independent_CPU_qualified=False,
        no_DEV=True, no_TEST=True, AP_evaluated=False, capacity_confound_not_yet_controlled=True,
        scope='Exploratory single-module training only; FIT loss cannot establish mask AP gain or innovation')
    require(report['elapsed_seconds'] < 1800, 'Whole leaf deadline exceeded')
    write_new(output/'REPORT.json', report)
    print(json.dumps(dict(status=STATUS, report=reference(output/'REPORT.json'))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--owner-execute', action='store_true')
    args = parser.parse_args()
    require(args.owner_execute, 'One explicit ROOT-owned attempt required')
    run(args.spec)


