"""Future ROOT-only authoring of a small fixed-endpoint factorial CPU manifest.

No science, launch, AUTH, model/CP loading or evaluation. Original evaluation
and all ancestor raws must be terminal0 before the first hash or Q read.
ROOT supplies its separately observed actual preceding CPU session terminal0.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import types


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load_runner(path):
    """Compilation is not performed until the local independent raw bootstrap."""
    code = path.read_bytes(); module = types.ModuleType('_ROOT_factorial_manifest_source_ONLY')
    module.__file__ = str(path); exec(compile(code, str(path), 'exec'), module.__dict__)
    return module, dict(path=str(path), bytes=len(code), sha256=hashlib.sha256(code).hexdigest())


def raw_bootstrap(spec_path):
    # This SOURCE-only duplicate intentionally avoids importing/hashing runner first.
    import math
    def read(path):
        require(Path(path).is_absolute(), 'Absolute raw locator required')
        return json.loads(Path(path).read_text(encoding='utf-8-sig'))
    def terminal(raw, phase):
        require(raw.get('status') == 'ACTUAL_OWNED_TERMINAL' and raw.get('original_identity') is not None and
            type(raw.get('returncode')) is int and raw['returncode'] == 0 and raw.get('root_wait_reaped') is True and
            raw.get('held_handle_exit') == dict(wait_state='WAIT_OBJECT_0', actual_exit_code=0) and
            raw.get('numeric_birth_absence', {}).get('original_birth_absent') is True and
            all(k in raw and raw[k] is None for k in ('abort','error')) and raw.get('wait_errors') is False and
            raw.get('GPU_observation_errors') == [], 'STOP before SHA/Q/SOURCE: strict actual raw0 required')
        peak = raw.get('whole_GPU_observed_peak_MiB')
        require(raw.get('phase',phase) == phase and type(peak) in (int,float) and math.isfinite(peak) and
            0 <= peak <= 8192 and raw.get('whole_GPU_used_limit_MiB') == 8192 and
            raw.get('whole_leaf_VRAM_certified') is False and
            raw.get('GPU_scope') == 'whole GPU observed, not per-process WDDM allocation', 'Raw resource scope differs')
    def gated(raw_path, phase):
        raw_path = Path(raw_path)
        require(raw_path.is_absolute() and raw_path.parent.name == phase, 'Absolute raw locator/phase directory differs')
        terminal(read(raw_path), phase)
    spec = read(spec_path)
    require(spec.get('endpoint_kind') in ('joint24','warm48'), 'One fixed qualified evaluation endpoint required')
    receipt = spec_path.parent/'finite_point_cache_execution_20261005_v2'/'runs'/spec['phase']/'PARENT_RECEIPT.json'
    gated(receipt, spec['phase'])
    train = read(spec['joint_training_spec']['path']); tiny = read(train['joint_preflight_spec']['path'])
    modes = ('native0','global','spatial_halves','spatial_reversed')
    require(set(tiny['training_receiptmap']) == set(tiny['training_phasemap']) == set(modes), 'Original four raw locators required')
    priors = [(tiny['training_receiptmap'][m], tiny['training_phasemap'][m]) for m in modes]
    for row in tiny['prior_receipts']:
        require(set(row) == {'receipt','phase'}, 'Explicit restoration/technical prior raw required')
        priors.append((row['receipt'],row['phase']))
    priors.append((train['joint_preflight_receipt'],train['joint_preflight_phase']))
    if spec['endpoint_kind'] == 'warm48':
        priors.append((train['first24_receipt'],train['first24_phase']))
    priors.append((spec['joint_training_receipt'],spec['joint_training_phase']))
    for ref, phase in priors: gated(ref['path'], phase)
    return spec, receipt


def build(args):
    spec_path, q_path, target, output = [Path(value) for value in
        (args.evaluation_spec,args.qualification,args.manifest,args.out)]
    require(all(p.is_absolute() for p in (spec_path,q_path,target,output)) and target.parent.is_dir() and
        output.parent.is_dir() and target != output and not target.exists() and not output.exists() and
        not target.with_name(target.name+'.tmp').exists() and not output.with_name(output.name+'.tmp').exists(),
        'Explicit original locators and distinct fresh absolute manifest/report required')
    require(type(args.qualified_cpu_session) is int and args.qualified_cpu_session > 0,
        'ROOT must supply the actual preceding CPU tool session after observing exit0')
    spec, receipt = raw_bootstrap(spec_path)
    # All original/current ancestry terminal gates above precede SOURCE reads, hashes and Q access.
    here = Path(__file__).resolve(strict=True).parent
    runner, runner_ref = load_runner((here/'run_diagnostic_cpu.py').resolve(strict=True))
    require(runner.reference(runner_ref['path']) == runner_ref and 'torch' not in sys.modules, 'Verified stdlib-only runner required')
    require(runner.read(spec_path) == spec, 'Original SPEC changed after raw bootstrap')
    q_ref = runner.reference(q_path)
    _, refs, q, _, exports, _ = runner.qualified_inputs(spec_path,spec,q_ref)
    sources = dict(consumer_source=runner_ref, crossed_predictions_source=runner.reference(here/'crossed_predictions.py'),
        manifest_builder_source=runner.reference(__file__))
    helper = runner.load(sources['crossed_predictions_source'],'_ROOT_factorial_public_helper_ONLY')
    require(callable(helper.assemble_crossed_predictions) and callable(helper.summarize_factorial_AP) and
        all(ref not in spec['sources'] for ref in sources.values()) and 'torch' not in sys.modules,
        'Separate pure helper public APIs/provenance required; original owner closure is immutable')
    manifest = dict(schema=runner.SCHEMA, inputs=runner.INPUTS, evaluation_spec=refs['spec'],
        evaluation_CPU_qualification=q_ref, evaluation_GPU_receipt=runner.reference(receipt),
        evaluation_GPU_admission=refs['GPU_admission'], qualified_CPU_reader=spec['CPU_reader'],
        qualified_CPU_tool_terminal=dict(session_id=args.qualified_cpu_session,exit_code=0),
        official_evaluation_CPU_reader=spec['official_evaluation_CPU_reader'], DEV=spec['DEV'], exports=exports,
        execution=runner.EXECUTION, COCO_configuration=runner.COCO_CONFIGURATION, output_report=str(output), **sources)
    require(set(manifest) == runner.KEYS and manifest['evaluation_GPU_receipt'] == refs['parent_receipt'] and
        q['spec'] == manifest['evaluation_spec'], 'Exact qualified input/manifest contract differs')
    for key in ('evaluation_spec','evaluation_CPU_qualification','evaluation_GPU_receipt','evaluation_GPU_admission',
        'qualified_CPU_reader','consumer_source','crossed_predictions_source','manifest_builder_source','official_evaluation_CPU_reader','DEV'):
        runner.bound(manifest[key])
    for value in exports.values():
        for ref in value.values(): runner.bound(ref)
    require(runner.read(spec_path) == spec and 'torch' not in sys.modules, 'Final authoring inputs changed or science imported')
    runner.write_fresh(target,manifest)
    print(json.dumps(dict(status='ROOT_FACTORIAL_CPU_MANIFEST_BUILT_NOT_EXECUTED',manifest=runner.reference(target),
        endpoint_kind=spec['endpoint_kind'],phase=spec['phase'],effective_native_epoch=q['effective_native_epoch'],
        effective_B_readout_epoch=q['effective_B_readout_epoch'],output_report=str(output),
        source=manifest['manifest_builder_source'],CPU_tool_terminal_metadata_not_independent_OS_proof=True,
        science_executed=False,diagnostic_launched=False),sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation-spec',required=True); parser.add_argument('--qualification',required=True)
    parser.add_argument('--qualified-cpu-session',type=int,required=True)
    parser.add_argument('--manifest',required=True); parser.add_argument('--out',required=True)
    parser.add_argument('--owner-author',action='store_true'); args = parser.parse_args()
    require(args.owner_author,'Explicit future ROOT manifest authoring required'); build(args)


if __name__ == '__main__':
    main()
