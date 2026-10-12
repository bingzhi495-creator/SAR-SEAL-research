"""Future single ROOT-owned N09 FIT-only scalar calibration on the N07 cache.

SOURCE preparation does not admit or execute this entry. Every declared normal
RAW and the preserved historical failed CPU RAW must be terminal before the
first input SHA or non-stdlib import. The complete current input closure and
actual owner are then verified. Synthetic operator checks precede cache report,
qualification and packet reads. Only cached native_logits/targets enter the
FP64 fit; all 1661 raw GT rows and their original 1024 sampled points are kept.
This is neither a native teacher replay nor an AP/model-efficacy qualification.

Required SPEC fields: schema, phase, sources, RAW_inputs,
prior_failed_CPU_receipt, FIT, cache_report, cache_CPU_qualification, cache_dir,
offset_source, test_source, output_dir, CPU_admission_file, deadline_seconds.
Refs use the exact {path, bytes, sha256} ABI. RAW rows use receipt, phase,
admission, phase_binding and bootstrap. This entry's own ref is in sources;
the current SPEC ref is separately bound by the current CPU admission.
"""
import argparse
import ast
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time
import types


SCHEMA = 'ROOT-N09-FIT-global-band-offset-CPU-v1'
REPORT_SCHEMA = 'N09-FIT-global-band-offset-report-v1'
STATUS = 'N09_FIT_GLOBAL_BAND_OFFSET_CPU_COMPLETE_PENDING_ROOT_QUALIFICATION'
OWNER_SHA = 'ef9a9f17d4fb1bc3464936095ef799021dd880f8b15488d558a871c08b6e1924'
CACHE_REPORT_SCHEMA = 'N07-frozen-E24-point-cache-report-v1'
CACHE_STATUS = 'N07_FROZEN_E24_FIT_POINT_CACHE_GPU_COMPLETE_PENDING_CPU'
CACHE_Q_SCHEMA = 'ROOT-N07-frozen-E24-original-detail-point-cache-CPU-qualification-v1'
CACHE_Q_STATUS = 'ROOT_N07_FROZEN_E24_ORIGINAL_DETAIL_POINT_CACHE_CPU_QUALIFIED'
PACKET_SCHEMA = 'N07-frozen-E24-original-detail-point-cache-packet-v1'
FIT = dict(images=741, GT=1661, points_per_GT=1024, total_points=1700864)
RAW_KEYS = ('receipt', 'phase', 'admission', 'phase_binding', 'bootstrap')
THREAD_ENV = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')


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
            set(ref['sha256']) <= set('0123456789abcdef'), 'Malformed absolute input reference')


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


def terminal(ref, phase, code=0):
    """Read only the original RAW before any hash, import or cache payload."""
    ref_shape(ref)
    path = Path(ref['path'])
    require(type(phase) is str and phase and path.name == 'PARENT_RECEIPT.json' and
            path.parent.name == phase, 'Original absolute RAW/phase locator required')
    raw = read(path)
    require(raw['status'] == 'ACTUAL_OWNED_TERMINAL' and type(raw['returncode']) is int and
            raw['returncode'] == code and type(raw['original_identity']) is dict and
            raw['original_identity'] and raw['root_wait_reaped'] is True and
            raw['held_handle_exit'] == dict(wait_state='WAIT_OBJECT_0', actual_exit_code=code) and
            type(raw['held_handle_exit']['actual_exit_code']) is int and
            raw['numeric_birth_absence']['original_birth_absent'] is True and
            raw.get('error') is None and raw.get('abort') is None and raw['wait_errors'] is False and
            raw['GPU_observation_errors'] == [], 'Actual held/reaped terminal%d required' % code)
    if 'phase' in raw:
        require(type(raw['phase']) is str and raw['phase'] == phase, 'Original RAW phase differs')
    sha, peak = raw['admission_sha256'], raw['whole_GPU_observed_peak_MiB']
    require(type(sha) is str and len(sha) == 64 and set(sha) <= set('0123456789abcdef') and
            type(peak) in (int, float) and math.isfinite(peak) and 0 <= peak <= 8192 and
            type(raw['whole_GPU_used_limit_MiB']) is int and raw['whole_GPU_used_limit_MiB'] == 8192 and
            raw['whole_leaf_VRAM_certified'] is False and
            raw['GPU_scope'] == 'whole GPU observed, not per-process WDDM allocation',
            'Original RAW owner/resource scope differs')
    return raw


def owner_join(row, raw, closure):
    """N07 SOURCE-derived RAW/admission/PB/bootstrap join, using stdlib only."""
    owner, pb = read(bound(row['admission'])), read(bound(row['phase_binding']))
    bootstrap = bound(row['bootstrap'])
    runtime = Path(row['receipt']['path']).parent
    require(Path(row['phase_binding']['path']) == runtime/'PHASE_BINDING.json' and
            bootstrap == runtime/'bootstrap.py' and
            owner['status'] == 'ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE' and
            owner['phase'] == row['phase'] and owner['driver_sha256'] == OWNER_SHA and
            owner['automatic_retry'] is False and owner['allowed_cpus'] == [0, 1] and
            owner['cuda_visible_devices'] in ('', '0') and owner['whole_GPU_used_limit_MiB'] == 8192 and
            raw['admission_sha256'] == row['admission']['sha256'] == pb['admission_sha256'] and
            pb['bootstrap_sha256'] == row['bootstrap']['sha256'] and pb['argv'][-1] == str(bootstrap) and
            pb['CPU_affinity'] == owner['allowed_cpus'] and
            pb['CUDA_visible_devices'] == owner['cuda_visible_devices'] and
            pb['automatic_retry'] is False, 'Original owner/admission/PB/bootstrap differs')
    entry, args = owner['entry'], owner['entry_args']
    require(type(entry) is str and type(args) is list and entry in closure and
            owner['bound_source_sha256'].get(entry) == closure[entry]['sha256'],
            'Original actual entry absent or unbound')
    tree = ast.parse(bootstrap.read_text(encoding='utf-8'))
    argv = [ast.literal_eval(node.value) for node in ast.walk(tree)
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and
                target.value.id == 'sys' and target.attr == 'argv' for target in node.targets)]
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and
             isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and
             node.func.value.id == 'runpy' and node.func.attr == 'run_path']
    require(argv == [[entry]+args] and len(calls) == 1 and
            ast.literal_eval(calls[0].args[0]) == entry and
            {item.arg: ast.literal_eval(item.value) for item in calls[0].keywords} == {'run_name': '__main__'},
            'Original bootstrap actual entry/argv differs')
    if '--spec' in args:
        require(args.count('--spec') == 1, 'Ambiguous original SPEC')
        locator = args[args.index('--spec')+1]
        require(locator in closure and owner['bound_source_sha256'].get(locator) == closure[locator]['sha256'],
                'Original actual SPEC absent or unbound')
    if 'scientific_spec' in owner:
        ref = owner['scientific_spec']
        require(ref in closure.values() and owner['bound_source_sha256'].get(ref['path']) == ref['sha256'],
                'Original scientific SPEC absent or unbound')
    return owner


def deadline(started, seconds):
    require(time.monotonic()-started < seconds, 'Prelocked N09 CPU deadline exceeded; no fallback or retry')


def source_gate(path, spec, started):
    require(spec['schema'] == SCHEMA and type(spec['phase']) is str and spec['phase'] and
            type(spec['RAW_inputs']) is list and len(spec['RAW_inputs']) >= 1 and
            all(type(row) is dict and set(RAW_KEYS) <= set(row) for row in spec['RAW_inputs']) and
            len({row['receipt']['path'] for row in spec['RAW_inputs']}) == len(spec['RAW_inputs']),
            'Distinct actual normal RAW ancestors required')
    raws = [terminal(row['receipt'], row['phase']) for row in spec['RAW_inputs']]
    failed = spec['prior_failed_CPU_receipt']
    terminal(failed, Path(failed['path']).parent.name, 1)
    require(failed['path'] not in {row['receipt']['path'] for row in spec['RAW_inputs']},
            'Preserved failed CPU attempt cannot qualify a normal ancestor')
    # Every declared normal terminal and failed1 precedes the first SHA/import.
    require(type(spec['sources']) is list and spec['sources'], 'Complete actual input refs required')
    for ref in spec['sources']:
        ref_shape(ref)
    closure = {ref['path']: ref for ref in spec['sources']}
    require(len(closure) == len(spec['sources']), 'Unique complete actual input closure required')
    roles = ('cache_report', 'cache_CPU_qualification', 'offset_source', 'test_source', 'prior_failed_CPU_receipt')
    required = [spec[key] for key in roles]
    required += [row[key] for row in spec['RAW_inputs'] for key in RAW_KEYS if key != 'phase']
    require(all(ref in closure.values() for ref in required), 'All actual primary refs must be sources-bound')
    require(spec['FIT'] == FIT and type(spec['deadline_seconds']) is int and spec['deadline_seconds'] > 0 and
            os.environ.get('CUDA_VISIBLE_DEVICES') == '' and
            all(os.environ.get(key) == '2' for key in THREAD_ENV), 'Locked FIT population/CPU environment differs')
    for ref in spec['sources']:
        deadline(started, spec['deadline_seconds'])
        bound(ref)
    source, spec_ref = reference(__file__), reference(path)
    require(closure.get(source['path']) == source, 'Actual N09 entry SOURCE absent or changed')
    owners = [owner_join(row, raw, closure) for row, raw in zip(spec['RAW_inputs'], raws)]
    admission_path = Path(spec['CPU_admission_file'])
    require(admission_path.is_absolute(), 'Absolute current CPU admission required')
    admission_ref = reference(admission_path)
    admission = read(admission_path)
    require(admission['status'] == 'ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE' and
            admission['driver_sha256'] == OWNER_SHA and admission['phase'] == spec['phase'] and
            admission['scientific_spec'] == spec_ref and admission['entry'] == source['path'] and
            admission['entry_args'] == ['--spec', str(path), '--owner-execute'] and
            admission['allowed_cpus'] == [0, 1] and admission['cuda_visible_devices'] == '' and
            admission['automatic_retry'] is False and admission['whole_GPU_used_limit_MiB'] == 8192 and
            type(admission['qualified_leaf_deadline_seconds']) is int and
            admission['qualified_leaf_deadline_seconds'] == spec['deadline_seconds'] and
            admission['bound_source_sha256'].get(str(path)) == spec_ref['sha256'] and
            all(admission['bound_source_sha256'].get(ref['path']) == ref['sha256'] for ref in spec['sources']),
            'Actual current CPU owner or complete closure differs')
    return closure, owners, source, spec_ref, admission_ref, admission


def load(ref, name):
    path = bound(ref)
    require(path.suffix == '.py', 'Pinned Python SOURCE required')
    source = path.read_bytes()
    require(len(source) == ref['bytes'] and hashlib.sha256(source).hexdigest() == ref['sha256'],
            'SOURCE bytes changed after binding')
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(source, str(path), 'exec'), module.__dict__)
    return module


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def imported_files(psutil, torch, np, closure, admission):
    """Actual files ROOT must additionally bind; no version-only qualification.

    Windows psutil affinity uses its platform SOURCE/cext. Torch packet loading
    and fixture backward use serialization, _tensor and autograd SOURCE; Torch
    and NumPy numerical binaries are also bound. No native/MMDet/decoder import.
    """
    paths = dict(psutil=Path(psutil.__file__), psutil_platform=Path(psutil._psplatform.__file__),
                 psutil_binary=Path(psutil._psplatform.cext.__file__),
                 torch=Path(torch.__file__), torch_binary=Path(torch._C.__file__),
                 torch_serialization=Path(inspect.getsourcefile(torch.load)),
                 torch_tensor=Path(sys.modules['torch._tensor'].__file__),
                 torch_autograd=Path(sys.modules['torch.autograd'].__file__),
                 numpy=Path(np.__file__), numpy_binary=Path(np.core._multiarray_umath.__file__))
    refs = {}
    for role, path in paths.items():
        ref = reference(path)
        require(closure.get(ref['path']) == ref and
                admission['bound_source_sha256'].get(ref['path']) == ref['sha256'],
                'Actual imported CPU library absent or unbound: ' + role)
        refs[role] = ref
    return refs


def resources(psutil, torch):
    require(psutil.Process().cpu_affinity() == [0, 1] and
            torch.get_num_threads() == torch.get_num_interop_threads() == 2 and
            os.environ.get('CUDA_VISIBLE_DEVICES') == '' and
            all(os.environ.get(key) == '2' for key in THREAD_ENV) and not torch.cuda.is_initialized(),
            'Actual CPU01/threads2/cold CUDA resources differ')


def cache_metadata(spec, closure, owners):
    """Verify admitted N07 metadata and actual successful GPU/CPU owner joins.

    Do not reopen the inherited native training/image input closure or repeat
    the CPU label/JPEG replay. The bound pre-existing qualification carries that
    evidence; this entry checks its metadata join and the cached fit operands.
    """
    report = read(bound(spec['cache_report']))
    cache_dir = Path(spec['cache_dir'])
    require(cache_dir.is_absolute() and Path(spec['cache_report']['path']) == cache_dir/'REPORT.json' and
            report['schema'] == CACHE_REPORT_SCHEMA and report['status'] == CACHE_STATUS and
            report['images'] == FIT['images'] and report['raw_GT'] == FIT['GT'] and
            report['point_count'] == FIT['total_points'] and
            report['recipe']['images'] == 741 and report['recipe']['raw_GT'] == 1661 and
            report['recipe']['native_queries'] == 100 and report['recipe']['points_per_GT'] == 1024 and
            report['recipe']['band_abs_logit'] == 2. and
            report['all_raw_GT_retained'] is True and report['native610_unchanged_SOURCE_observed'] is True and
            report['optimizer_created'] is False and report['weights_updated'] is False and
            report['AP_computed'] is False and report['no_DEV'] is True and report['no_TEST'] is True and
            report['cleanup']['allocated_after'] == 0, 'Actual complete frozen N07 FIT cache report differs')
    refs = [report[key] for key in ('source', 'spec', 'GPU_admission', 'start', 'trace')]
    require(type(report['packet_refs']) is list and type(report['rows']) is list and
            len(report['packet_refs']) == len(report['rows']) == 741 and
            len({ref['path'] for ref in report['packet_refs']}) == 741 and
            all(ref in closure.values() for ref in refs+report['packet_refs']) and
            Path(report['start']['path']) == cache_dir/'START.json' and
            Path(report['trace']['path']) == cache_dir/'TRACE.jsonl', 'Complete actual cache metadata/packet closure required')
    gpu = [(row, owner) for row, owner in zip(spec['RAW_inputs'], owners)
           if row['phase'] == report['phase'] and row['admission'] == report['GPU_admission'] and
           owner['entry'] == report['source']['path'] and owner['scientific_spec'] == report['spec'] and
           owner['entry_args'] == ['--spec', report['spec']['path'], '--owner-execute']]
    require(len(gpu) == 1 and gpu[0][1]['cuda_visible_devices'] == '0',
            'Actual successful frozen cache GPU owner differs')
    q = read(bound(spec['cache_CPU_qualification']))
    require(q['schema'] == CACHE_Q_SCHEMA and q['status'] == CACHE_Q_STATUS and
            q['source'] == report['source'] and q['cache_spec'] == report['spec'] and
            q['cache_report'] == spec['cache_report'] and q['cache_GPU_admission'] == report['GPU_admission'] and
            q['cache_GPU_RAW'] == gpu[0][0]['receipt'] and q['cache_phase'] == report['phase'] and
            q['packet_refs'] == report['packet_refs'] and q['images'] == 741 and q['raw_GT'] == 1661 and
            q['point_count'] == 1700864 and q['hard_targets_CPU_checked'] is True and
            q['indices_JPEG_patches_CPU_checked'] is True and q['point_head_synthetic_CPU_qualified'] is True and
            all(q[key] in closure.values() for key in ('reader_source', 'spec', 'CPU_admission')),
            'Bound pre-existing independent CPU cache qualification differs')
    cpu = [(row, owner) for row, owner in zip(spec['RAW_inputs'], owners)
           if row['phase'] == q['phase'] and row['admission'] == q['CPU_admission'] and
           owner['entry'] == q['reader_source']['path'] and owner['scientific_spec'] == q['spec'] and
           owner['entry_args'] == ['--spec', q['spec']['path'], '--owner-execute']]
    require(len(cpu) == 1 and cpu[0][1]['cuda_visible_devices'] == '',
            'Actual successful cache CPU qualification owner differs')
    return report, dict(cache_GPU_RAW=gpu[0][0]['receipt'], cache_CPU_RAW=cpu[0][0]['receipt'],
                       cache_CPU_qualification=spec['cache_CPU_qualification'],
                       CPU_qualification_metadata_join_checked=True, native_teacher_replayed=False,
                       prior_CPU_label_and_JPEG_checks_replayed=False)


def tensor_digest(value):
    array = value.detach().cpu().contiguous().numpy()
    return dict(shape=list(array.shape), dtype=str(array.dtype), sha256=hashlib.sha256(array.tobytes()).hexdigest())


def packet_population(torch, np, value, row, ref, report, index):
    """SOURCE-derived N07 packet ABI; preserve raw GT order and sample positions."""
    identity = value['identity']
    G = len(identity['ann_ids'])
    require(value['schema'] == PACKET_SCHEMA and
            value['dataset_index'] == identity['dataset_index'] == row['dataset_index'] == index and
            identity == row['identity'] and identity['image_id'] == row['image_id'] and row['packet'] == ref and
            value['source'] == report['source'] and value['spec'] == report['spec'] and
            value['phase'] == report['phase'] and 1 <= G <= 100 and G == row['raw_GT'] and
            identity['ann_ids'] == row['raw_GT_ids'] and row['points'] == G*1024 and
            value['replacement'] is True and value['duplicates_preserved'] is True and
            value['GT_selected_points'] is False, 'Exact cached raw-GT/point/lineage row differs')
    shapes = dict(native_logits=(G, 1024), targets=(G, 1024), point_indices=(G, 1024), query_ids=(G,))
    for name, shape in shapes.items():
        tensor = value[name]
        dtype = torch.float32 if name == 'native_logits' else (torch.bool if name == 'targets' else torch.int64)
        require(isinstance(tensor, torch.Tensor) and tensor.device.type == 'cpu' and
                tensor.layout == torch.strided and tensor.dtype == dtype and tuple(tensor.shape) == shape and
                not tensor.requires_grad and (dtype != torch.float32 or bool(torch.isfinite(tensor).all())) and
                tensor_digest(tensor) == row['cache_digests'][name], 'Qualified cached operand differs: ' + name)
    queries = value['query_ids'].tolist()
    require(queries == row['query_ids'] and len(set(queries)) == G and
            all(type(query) is int and 0 <= query < 100 for query in queries), 'Original matched query order differs')
    shape = identity['original_shape']
    require(type(shape) is list and len(shape) == 2 and all(type(n) is int and n > 0 for n in shape),
            'Original raster shape differs')
    indices = value['point_indices']
    require(bool(((indices >= 0) & (indices < shape[0]*shape[1])).all()) and
            value['band_counts'] == row['band_counts'] and len(value['band_counts']) == G and
            all(type(count) is int and 0 <= count <= shape[0]*shape[1] for count in value['band_counts']) and
            len(row['duplicate_points_by_GT']) == G and
            all(type(count) is int and 0 <= count < 1024 for count in row['duplicate_points_by_GT']),
            'Original fixed sample metadata differs')
    L, y = value['native_logits'].numpy(), value['targets'].numpy()
    band = np.abs(L) <= 2.
    require(y.sum(axis=1).tolist() == row['sampled_foreground_by_GT'], 'Cached foreground population differs')
    for g, count in enumerate(value['band_counts']):
        require(bool(band[g, :768].all()) if count else not bool(band[g].any()),
                'SOURCE-observed native-band sampling rule differs')
    summary = dict(dataset_index=index, image_id=identity['image_id'], raw_GT=G, raw_GT_ids=identity['ann_ids'],
                   query_ids=queries, point_count=G*1024, sampled_band_points_by_GT=band.sum(axis=1).tolist(),
                   sampled_foreground_by_GT=row['sampled_foreground_by_GT'],
                   original_raster_band_counts_SOURCE_observed=row['band_counts'],
                   duplicate_points_by_GT_cache_metadata=row['duplicate_points_by_GT'], packet=ref)
    return L, y, summary


def run(spec_path):
    started = time.monotonic()
    path = Path(spec_path)
    require(path.is_absolute(), 'Absolute ROOT-authored SPEC required')
    path = path.resolve(strict=True)
    spec = read(path)
    closure, owners, source, spec_ref, admission_ref, admission = source_gate(path, spec, started)
    output = Path(spec['output_dir'])
    require(output.is_absolute() and not output.exists(), 'Fresh absolute output directory required; no overwrite')
    import psutil
    require(psutil.Process().cpu_affinity() == [0, 1], 'Actual CPU01 required before numerical imports')
    import torch
    require(torch.__version__ == '2.0.1+cu118' and not torch.cuda.is_initialized(), 'Original Torch/cold CUDA required')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    import numpy as np
    imported = imported_files(psutil, torch, np, closure, admission)
    resources(psutil, torch)
    operator = load(spec['offset_source'], '_N09_FIT_offset_math')
    fixtures = load(spec['test_source'], '_N09_FIT_offset_tests')
    require(operator.FIT_SHAPE == (1661, 1024) and operator.BAND_LIMIT == 2. and
            operator.ROOT_WIDTH == 1e-8 and operator.MAX_STEPS == 64, 'Reviewed scalar math contract differs')
    deadline(started, spec['deadline_seconds'])
    output.mkdir()
    write(output/'START.json', dict(schema='N09-FIT-global-band-offset-start-v1', source=source, spec=spec_ref,
          CPU_admission=admission_ref, phase=spec['phase'], FIT=FIT, cache_report=spec['cache_report'],
          cache_CPU_qualification=spec['cache_CPU_qualification'], cache_dir=spec['cache_dir'],
          offset_source=spec['offset_source'], test_source=spec['test_source'], imported_files=imported,
          CPU_affinity=[0, 1], threads=2, CUDA_initialized=False, deadline_seconds=spec['deadline_seconds'],
          no_DEV=True, no_TEST=True, AP_computed=False, scientific_result_present=False))
    # The synthetic operator checks occur before any FIT cache REPORT/Q/packet read.
    tests = fixtures.run_tests(np, torch, operator)
    require(tests['status'] == 'SYNTHETIC_MATH_CHECKS_PASSED_NOT_SCIENTIFIC_QUALIFICATION' and
            tests['CPU_RNG_unchanged'] is True and tests['CUDA_initialized'] is False and
            tests['real_FIT_or_DEV_accessed'] is False, 'Synthetic operator checks did not pass')
    resources(psutil, torch)
    deadline(started, spec['deadline_seconds'])
    report, lineage = cache_metadata(spec, closure, owners)
    native_logits = np.empty((1661, 1024), dtype=np.float32)
    targets = np.empty((1661, 1024), dtype=np.bool_)
    population_rows = []
    cursor = 0
    for index, (ref, row) in enumerate(zip(report['packet_refs'], report['rows'])):
        deadline(started, spec['deadline_seconds'])
        require(Path(ref['path']) == Path(spec['cache_dir'])/('PACKET_%04d.pth' % index),
                'Actual SOURCE-defined N07 packet locator differs')
        value = torch.load(str(bound(ref)), map_location='cpu')
        L, y, summary = packet_population(torch, np, value, row, ref, report, index)
        G = summary['raw_GT']
        require(cursor+G <= 1661, 'Cached GT population exceeds declared FIT')
        native_logits[cursor:cursor+G] = L
        targets[cursor:cursor+G] = y
        cursor += G
        population_rows.append(summary)
        del value, L, y
    require(cursor == 1661 and len(population_rows) == 741 and native_logits.size == targets.size == 1700864,
            'All original FIT741/GT1661/1700864 point positions required')
    resources(psutil, torch)
    deadline(started, spec['deadline_seconds'])
    fitted = operator.fit_offset(np, native_logits, targets)
    require(fitted['fixture_mode'] is False and fitted['GT_instances'] == 1661 and
            fitted['points_per_GT'] == 1024 and fitted['total_points'] == 1700864 and
            fitted['native_band_closed_abs_logit'] == 2. and fitted['band_renormalized'] is False and
            fitted['regularizer_or_offset_cap'] is None and type(fitted['delta']) is float and
            math.isfinite(fitted['delta']), 'Exact FIT-only scalar fit contract differs')
    del native_logits, targets
    resources(psutil, torch)
    for ref in spec['sources']:
        deadline(started, spec['deadline_seconds'])
        bound(ref)
    require(reference(path) == spec_ref and reference(__file__) == source and
            reference(spec['CPU_admission_file']) == admission_ref, 'Final SPEC/SOURCE/admission changed')
    elapsed = time.monotonic()-started
    deadline(started, spec['deadline_seconds'])
    result = dict(schema=REPORT_SCHEMA, status=STATUS, source=source, spec=spec_ref, CPU_admission=admission_ref,
                  phase=spec['phase'], start=reference(output/'START.json'), delta=fitted['delta'], FIT_fitted=fitted,
                  FIT=FIT, cache_report=spec['cache_report'], cache_CPU_qualification=spec['cache_CPU_qualification'],
                  cache_dir=spec['cache_dir'], offset_source=spec['offset_source'], test_source=spec['test_source'],
                  operator_tests=tests, cache_lineage=lineage, population_rows=population_rows,
                  images=741, raw_GT=1661, point_count=1700864, packet_count=741,
                  all_raw_GT_retained=True, raw_GT_order_preserved=True, all_cached_sample_positions_retained=True,
                  duplicates_preserved=True, point_deduplication_performed=False, GT_point_selection_changed=False,
                  cached_duplicate_point_positions_SOURCE_observed=sum(sum(row['duplicate_points_by_GT_cache_metadata'])
                      for row in population_rows), fit_operands=['native_logits', 'targets'],
                  fit_input_dtypes=dict(native_logits='float32', targets='bool'), fit_arithmetic_dtype='float64',
                  equal_GT_fullpoint_mean_BCE=True, native_band_closed_abs_logit=2., no_DEV=True, no_TEST=True,
                  AP_computed=False, DEV_selection_performed=False, native_teacher_replayed=False,
                  native_model_loaded=False, native_weights_updated=False, optimizer_created=False,
                  independent_AP_or_model_efficacy_qualified=False, current_owned_terminal_closed_by_ROOT=False,
                  imported_files=imported, imported_versions=dict(torch=torch.__version__, numpy=np.__version__, psutil=psutil.__version__),
                  CPU_affinity=[0, 1], threads=2, CUDA_initialized=False, cuda_visible_devices='',
                  elapsed_seconds=elapsed, deadline_seconds=spec['deadline_seconds'], automatic_retry=False,
                  whole_GPU_used_limit_MiB=8192, whole_leaf_VRAM_certified=False,
                  evidence_limits=[
                      'One global native-band intercept fitted only on the admitted fixed N07 FIT cached-point population.',
                      'Existing CPU label/JPEG qualification is metadata-bound, not repeated; no native teacher replay or full-raster verification.',
                      'Sampling uses replacement. All cached positions retain their original multiplicity and full1024-point per-GT weight.',
                      'Synthetic checks establish the CPU operator contract, not CUDA/rendering equality or scientific efficacy.',
                      'FP64 root interval is practical finite arithmetic, not a certified real-arithmetic enclosure.',
                      'No DEV/TEST/AP read, tuning or comparison is performed. Evaluation and this producer terminal require separate ROOT closure.'
                  ])
    write(output/'REPORT.json', result)
    print(json.dumps(dict(status=STATUS, report=reference(output/'REPORT.json')), allow_nan=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--owner-execute', action='store_true')
    args = parser.parse_args()
    require(args.owner_execute, 'Explicit single ROOT-owned execution required')
    run(args.spec)
