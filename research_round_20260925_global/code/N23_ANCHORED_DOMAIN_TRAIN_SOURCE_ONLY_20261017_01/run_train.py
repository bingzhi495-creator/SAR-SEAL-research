"""One owned N23 anchored Omega/C point-head training stage; no DEV/AP."""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path


def need(ok, message):
    if not ok:
        raise AssertionError(message)


def validate_prelock(pre, spec):
    need(pre['schema'] == 'N23-anchored-Omega-C-training-prelock-v1'
         and pre['no_new_training'] is False and all(pre[k] is True for k in ('no_DEV', 'no_TEST', 'no_AP'))
         and pre['expected_images'] == 741 and pre['expected_real_GT'] == 1661
         and pre['original_draws_per_image'] == 4096
         and pre['population_filter'] is None
         and pre['dataset_indices'] == list(range(741))
         and spec['selected_dataset_indices'] == pre['dataset_indices'],
         'Prelocked complete original FIT scope only')
    image_ids = pre['fit_image_ids']
    groups = pre['fit_ann_ids_by_image']
    annotations = [v for group in groups for v in group]
    need(len(image_ids) == len(set(image_ids)) == len(groups) == 741
         and all(len(group) > 0 for group in groups)
         and len(annotations) == len(set(annotations)) == 1661,
         'Complete distinct original image and real-GT identities')


def reference(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


class Control:
    def __init__(self, spec):
        import psutil
        self.psutil = psutil
        self.process = psutil.Process()
        self.spec = spec
        self.started = time.monotonic()
        self.peak_rss = 0
        self.minimum_available = None
        self.checked = set()
        self.allowed = {str(Path(r['path']).resolve()).lower(): r for r in spec['sources']}
        self.output = Path(spec['output_directory']).resolve()

    def check(self):
        rss = self.process.memory_info().rss
        ram = self.psutil.virtual_memory().available
        self.peak_rss = max(self.peak_rss, rss)
        self.minimum_available = ram if self.minimum_available is None else min(self.minimum_available, ram)
        need(self.process.cpu_affinity() == [0, 1] and
             os.environ.get('CUDA_VISIBLE_DEVICES') == self.spec['cuda_visible_devices'] and
             all(os.environ.get(k) == '2' for k in
                 ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')) and
             rss <= 16 * 1024 ** 3 and ram >= 16 * 1024 ** 3 and
             time.monotonic() - self.started < 1800,
             'Owned CPU/thread/CUDA/RSS/RAM/deadline contract')

    def closed(self, ref):
        self.check()
        need(type(ref) is dict and set(ref) == {'path', 'bytes', 'sha256'} and
             Path(ref['path']).is_absolute(), 'Exact closed reference')
        key = str(Path(ref['path']).resolve()).lower()
        need(self.allowed.get(key) == ref, 'Input not in own direct scope: ' + ref['path'])
        if key not in self.checked:
            need(reference(ref['path']) == ref, 'Pinned direct input changed: ' + ref['path'])
            self.checked.add(key)
        return Path(ref['path'])

    def ref(self, path):
        self.check()
        path = Path(path)
        resolved = path.resolve()
        need(self.output in resolved.parents or str(resolved).lower() in self.allowed,
             'Reference inside own output or explicitly bound input')
        return reference(path)

    def write_json(self, path, value):
        self.check()
        path = Path(path)
        need(path.is_absolute() and self.output in path.resolve().parents and
             not path.exists() and not path.is_symlink(), 'New output inside own scope')
        raw = (json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
        with path.open('xb') as stream:
            stream.write(raw)
        need(path.read_bytes() == raw, 'Output JSON readback')
        return reference(path)


def run(spec_path):
    spec_path = Path(spec_path)
    need(spec_path.is_absolute(), 'Absolute spec')
    spec = json.loads(spec_path.read_bytes())
    need(spec['schema'] == 'ROOT-N23-direct-owned-anchored-training-spec-v1' and
         spec['operation'] == 'GPU_ANCHORED_TRAIN' and
         spec['owner_index'] == spec['normal_RAW_count'] and
         spec['total_terminal_identity_pin_count'] == spec['normal_RAW_count'] + spec['total_failed_RAW_count'] and
         spec['automatic_retry'] is False and spec['deadline_seconds'] == 1800 and
         spec['whole_GPU_limit_MiB'] == 8192 and spec['allocator_limit_MiB'] == 4096 and
         spec['allowed_cpus'] == [0, 1] and spec['thread_limit'] == 2 and
         spec['cuda_visible_devices'] == '0' and
         spec['maximum_process_RSS_bytes'] == spec['minimum_available_RAM_bytes'] == 16 * 1024 ** 3 and
         spec['no_new_training'] is False and spec['no_DEV'] is True and spec['no_TEST'] is True and
         spec['no_AP'] is True, 'Own fixed diagnostic and resource scope')
    need(reference(__file__) == spec['entry_source'], 'Actual entry identity')
    ad = json.loads(Path(spec['admission_file']).read_bytes())
    need(len(ad) == 13 and ad['scientific_spec'] == reference(spec_path) and
         ad['entry'] == spec['entry_source']['path'] and
         ad['entry_args'] == ['--spec', str(spec_path), '--owner-execute'] and
         ad['phase'] == spec['phase'] and ad['allowed_cpus'] == [0, 1] and
         ad['thread_limit'] == 2 and ad['cuda_visible_devices'] == spec['cuda_visible_devices'] and
         ad['whole_GPU_used_limit_MiB'] == 8192 and ad['qualified_leaf_deadline_seconds'] == 1800 and
         ad['automatic_retry'] is False and
         ad['bound_source_sha256'] == {r['path']: r['sha256'] for r in spec['sources'] + [reference(spec_path)]},
         'Exact own admission')
    control = Control(spec)
    control.execution_spec_ref = ad['scientific_spec']
    control.execution_admission_ref = reference(spec['admission_file'])
    control.check()
    need(len(control.allowed) == len(spec['sources']), 'Unique direct inputs')
    pre = json.loads(control.closed(spec['prelock']).read_bytes())
    validate_prelock(pre, spec)
    output = Path(spec['output_directory'])
    need(output.is_absolute() and output.parent.is_dir() and not output.exists() and
         not output.is_symlink() and control.psutil.disk_usage(str(output.parent)).free >= 20 * 1024 ** 3,
         'Fresh owned output and storage margin')
    import torch
    import numpy as np
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # The accepted initializer must build and strictly load on CPU before
    # CUDA is initialized. The capture body applies the allocator limit only
    # after that CPU construction, immediately before its first CUDA move.
    need(not torch.cuda.is_initialized(), 'CPU construction precedes CUDA initialization')
    control.check()
    body_path = control.closed(spec['body_source'])
    module_spec = importlib.util.spec_from_file_location('N23_OWNED_ANCHORED_TRAIN_BODY', body_path)
    body = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(body)
    result = body.run(torch, np, spec, control)
    control.check()
    need(type(result) is dict and output.is_dir(), 'Completed body output')
    peak = torch.cuda.max_memory_allocated(0)
    gc.collect()
    torch.cuda.synchronize(0)
    torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(0)
    cleanup = dict(allocated_after=torch.cuda.memory_allocated(0),
                   reserved_after=torch.cuda.memory_reserved(0), peak_allocated_bytes=peak)
    need(cleanup['allocated_after'] == cleanup['reserved_after'] == 0 and peak <= 4096 * 1024 ** 2,
         'Complete GPU cleanup including cuBLAS workspaces')
    for source in (spec['entry_source'], spec['body_source']):
        need(reference(source['path']) == source, 'Source unchanged')
    gc.collect()
    q = dict(schema='N23-owned-pending-training-result-v1',
             status='COMPLETED_PENDING_ORIGINAL_ACTUAL0_AND_ROOT_ACCEPTANCE',
             operation=spec['operation'], phase=spec['phase'], source=spec['entry_source'],
             spec=reference(spec_path), admission=reference(spec['admission_file']),
             prelock=spec['prelock'], result=result, owner_index=spec['owner_index'],
             normal_RAW_count=spec['normal_RAW_count'], total_failed_RAW_count=spec['total_failed_RAW_count'],
             total_terminal_identity_pin_count=spec['total_terminal_identity_pin_count'],
             legacy_pin_count=spec['legacy_pin_count'], qualified=False,
             original_actual0_self_certified=False, innovation_established=False, goal_complete=False,
             elapsed_seconds=time.monotonic() - control.started,
             peak_RSS_bytes=control.peak_rss, minimum_available_RAM_bytes=control.minimum_available,
             cleanup=cleanup,
             new_training=True, DEV=False, TEST=False, AP=False)
    qref = control.write_json(output / 'Q.json', q)
    print(json.dumps(dict(status=q['status'], Q=qref, qualified=False)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', required=True)
    parser.add_argument('--owner-execute', action='store_true')
    args = parser.parse_args()
    need(args.owner_execute, 'Root-owned single scientific leaf only')
    run(args.spec)
