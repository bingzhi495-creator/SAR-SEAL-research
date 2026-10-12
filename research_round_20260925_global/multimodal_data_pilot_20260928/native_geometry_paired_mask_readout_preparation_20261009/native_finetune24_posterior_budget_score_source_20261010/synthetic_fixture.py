"""SOURCE-only preparation of a tiny scoring fixture; do not execute at authoring."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import types

SCHEMA = 'ROOT-nativeE48-posterior-budget-synthetic-SOURCE-spec-v1'
STATUS = 'NATIVE_E48_POSTERIOR_BUDGET_SYNTHETIC_CPU_PASS'
EXECUTION = dict(device='cpu', cpu_affinity=[0, 1], torch_threads=2,
                 torch_interop_threads=2, cuda_visible_devices='')


def check(value, label):
    if not value:
        raise AssertionError(label)


def source_gate(spec, out, owner):
    check(owner and spec.is_absolute() and out.is_absolute(), 'owner/absolute paths required')
    check(not out.exists() and out.parent.is_dir(), 'fresh output required')
    manifest = json.loads(spec.read_text(encoding='utf-8'))
    check(set(manifest) == {'schema', 'inputs', 'sources', 'execution'} and
          manifest['schema'] == SCHEMA and manifest['inputs'] == 'SOURCE_ONLY' and
          manifest['execution'] == EXECUTION and set(manifest['sources']) == {'scorer', 'fixture'}, 'manifest differs')
    here, refs, code = Path(__file__).resolve(), {}, {}
    for role, path in (('fixture', here), ('scorer', here.parent / 'score_control.py')):
        ref = manifest['sources'][role]
        check(set(ref) == {'path', 'bytes', 'sha256'} and Path(ref['path']).is_absolute() and
              Path(ref['path']).resolve() == path, 'SOURCE path differs')
        code[role] = path.read_bytes()
        refs[role] = dict(path=str(path), bytes=len(code[role]), sha256=hashlib.sha256(code[role]).hexdigest())
        check(ref['bytes'] == refs[role]['bytes'] and ref['sha256'] == refs[role]['sha256'], 'SOURCE pin differs')
    check(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and 'torch' not in sys.modules, 'SOURCE/resources before Torch')
    return refs, code


def close(torch, actual, expected, label):
    check(actual.shape == expected.shape and torch.allclose(actual, expected, rtol=2e-6, atol=2e-6), label)


def numerical_checks(torch, score):
    rng = torch.get_rng_state().clone()
    logits = torch.tensor([[[-2., 1., 3.], [0., -1., 2.]], [[-1., -2., 1.], [2., 0., 3.]],
                           [[-3., -2., -1.], [-4., -1., -2.]], [[2., -1., 0.], [1., -2., 3.]]], requires_grad=True)
    support = torch.tensor([[[0, 0, 0], [0, 0, 0]], [[1, 1, 1], [1, 1, 1]],
                            [[1, 0, 1], [0, 1, 0]], [[0, 1, 0], [1, 0, 1]]], dtype=torch.bool)
    classes = torch.tensor([.25, .5, .75, 1.], requires_grad=True)
    originals = [v.detach().clone() for v in (logits, support, classes)]
    actual = score(torch, logits, support, classes)
    check(set(actual) == {'posterior_budget', 'class_times_posterior_budget'}, 'score keys')
    expected, transformed = [], []
    for row_logits, row_support in zip(logits.detach().tolist(), support.tolist()):
        p = [1. / (1. + math.exp(-v)) for row in row_logits for v in row]
        b = [int(v) for row in row_support for v in row]
        intersection, area = sum(v * m for v, m in zip(p, b)), sum(b)
        union = area + sum(v * (1 - m) for v, m in zip(p, b))
        expected.append(intersection / (union + 1e-6))
        sdc_eps = 2 * intersection / (area + sum(p) + 1e-6)
        transformed.append(sdc_eps / (2 - sdc_eps))
    expected = torch.tensor(expected)
    close(torch, actual['posterior_budget'], expected, 'independent scalar budget')
    close(torch, actual['posterior_budget'], torch.tensor(transformed), 'smoothed SDC transform')
    close(torch, actual['class_times_posterior_budget'], expected * classes.detach(), 'class times budget')
    check(float(actual['posterior_budget'][0]) == 0. and float(actual['posterior_budget'][2]) > 0., 'empty/supplied support')
    thresholded = score(torch, logits, logits > 0, classes)
    check(float(thresholded['posterior_budget'][2]) == 0., 'actual support test not discriminating')
    for value in actual.values():
        check(value.dtype == torch.float32 and not value.requires_grad and
              bool(torch.isfinite(value).all()) and bool(((value >= 0) & (value <= 1)).all()), 'finite bounded no_grad scores')
    hundred_logits, hundred_support, hundred_class = logits.detach().repeat(25, 1, 1), support.repeat(25, 1, 1), classes.detach().repeat(25)
    hundred = score(torch, hundred_logits, hundred_support, hundred_class)
    order = torch.arange(99, -1, -1)
    permuted = score(torch, hundred_logits[order], hundred_support[order], hundred_class[order])
    for key in actual:
        check(hundred[key].shape == (100,), 'all100 retained')
        close(torch, permuted[key], hundred[key][order], 'actual query permutation')
        chunks = [score(torch, hundred_logits[i:i + 7], hundred_support[i:i + 7], hundred_class[i:i + 7])[key]
                  for i in range(0, 100, 7)]
        close(torch, torch.cat(chunks), hundred[key], 'query chunk invariance')
    p = torch.tensor([[[.8, .1]], [[.8, .9]]])
    same_mask = torch.tensor([[[1, 0]], [[1, 0]]], dtype=torch.bool)
    ranked = score(torch, torch.log(p / (1 - p)), same_mask, torch.tensor([.75, .75]))
    check(float(ranked['posterior_budget'][0]) > float(ranked['posterior_budget'][1]), 'outside posterior mass penalty')
    for value, before in zip((logits, support, classes), originals):
        check(torch.equal(value, before) and (not value.requires_grad or value.grad is None), 'input mutation/gradient')
    check(torch.equal(torch.get_rng_state(), rng), 'external CPU RNG changed')
    return dict(scalar_formula=True, smoothed_SDC_transform=True, actual_support_not_CPU_threshold=True,
                unchanged_inputs=True, empty_mask_score_zero=True, all100_query_order_and_chunks=True,
                outside_mass_penalty=True, no_grad_and_RNG_unchanged=True)


def run(spec, out, owner):
    refs, code = source_gate(spec, out, owner)
    import psutil
    psutil.Process().cpu_affinity([0, 1])
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    check(sorted(psutil.Process().cpu_affinity()) == [0, 1] and torch.get_num_threads() == 2 and
          torch.get_num_interop_threads() == 2 and not torch.cuda.is_initialized(), 'CPU resources/CUDA')
    module = types.ModuleType('_synthetic_budget_scorer')
    exec(compile(code['scorer'], refs['scorer']['path'], 'exec'), module.__dict__)
    checks = numerical_checks(torch, module.posterior_budget_scores)
    check(not torch.cuda.is_initialized(), 'CUDA initialized')
    result = dict(status=STATUS, sources=refs, execution=EXECUTION, synthetic_execution_performed=True,
                  checks=checks, AP_computed=False, calibration_guaranteed=False, expected_IoU_guaranteed=False,
                  novelty_claimed=False, native_integration_qualified=False,
                  limits=['Synthetic logits/support only; no native packets/RLE/GT/CP or COCO consumed.',
                          'Known soft-IoU scoring formula; no training or calibration guarantee.',
                          'No GPU/CPU bitwise rendering guarantee; actual ROOT terminal is separate evidence.'])
    with out.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, allow_nan=False, indent=2)
        handle.write('\n')
    return dict(status=STATUS, out=str(out))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--owner-execute', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args.spec, args.out, args.owner_execute)))
