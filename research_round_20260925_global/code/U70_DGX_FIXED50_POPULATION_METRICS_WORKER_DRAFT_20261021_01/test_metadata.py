"""Synthetic JSON/source/import-boundary stubs only. Never import real ML libraries."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent


def load_worker():
    spec = importlib.util.spec_from_file_location('u70_worker', HERE/'worker.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ref(path, raw=b'x'):
    return dict(path=path, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def metadata_fixture():
    roles = ('population_contract', 'eval_body', 'official', 'cocoeval', 'mask_api', 'geometry_diagnostic', 'population_metrics')
    modules = {k: ref('/synthetic/'+k+'.py') for k in roles}
    libraries = {k: ref('/synthetic/lib/'+k+'.py') for k in ('numpy', 'psutil', 'pycocotools', 'coco_loader', 'mask_extension')}
    inputs = {k: ref('/synthetic/'+k+'.json') for k in
              ('context', 'legacy_context', 'legacy_result', 'native_predictions', 'R_predictions', 'DEV')}
    request = dict(worker_source=ref('/synthetic/worker.py'), core_source=ref('/synthetic/core.py'),
        module_sources=modules, library_sources=libraries, input_refs=inputs,
        shard_refs=[ref('/synthetic/shard'+str(i)+'.json') for i in range(4)], output_directory='/synthetic/OUT')

    def old(value):
        return dict(value, path='E:\\historical\\'+value['path'].split('/')[-1])

    common = dict(native_predictions=old(inputs['native_predictions']), R_predictions=old(inputs['R_predictions']),
                  legacy_ROOT_result=old(inputs['legacy_result']), image_ids=list(range(187)))
    context = dict(deepcopy(common), legacy_context=old(inputs['legacy_context']))
    legacy = dict(deepcopy(common), DEV=old(inputs['DEV']), COCO_source=old(modules['official']))
    full = dict(stats=[0.0]*12, metrics={'synthetic': 0}, iouType='segm', maxDets=[1, 10, 100],
        original_GT_preserved=True, score_filter=None, official_COCO_crowd_ignore_semantics=True)
    result = dict(control_results={'native_official': {'COCO': deepcopy(full)},
                                  'strong_fixed_E24_Omega': {'COCO': deepcopy(full)}})
    native = [dict(image_id=i, query_id=q, category_id=1, score=0., segmentation={'synthetic': True})
              for i in range(187) for q in reversed(range(100))]
    documents = dict(context=context, legacy_context=legacy, legacy_result=result,
                     native_predictions=native, R_predictions=deepcopy(native))
    endpoint = dict(schema='U58-U57-DGX-fixed-endpoint-metadata-v1', mode='FORMAL', formal_endpoint=True,
                    cursor=dict(global_updates=37050, completed_epochs=50, within_epoch=0))
    shards = [dict(schema='U66-DGX-fixed50-DEV-shard-pending-v1', status='PREDICTIONS_COMPLETE_PENDING_ROOT_ACCEPTANCE',
        shard_index=i, dataset_range=list(span), endpoint=deepcopy(endpoint),
        source_refs={k: deepcopy(modules[k]) for k in ('population_contract', 'mask_api')})
        for i, span in enumerate(((0,47), (47,94), (94,141), (141,187)))]
    return request, documents, shards


class MemoryPath:
    def __init__(self, path, raw):
        self.path, self.raw = path, raw

    def __str__(self):
        return self.path

    def read_bytes(self):
        return self.raw


class Control:
    def __init__(self, objects):
        self.objects, self.visits, self.checks = objects, [], 0

    def closed(self, value):
        self.visits.append(value['path'])
        raw = self.objects[value['path']]
        if ref(value['path'], raw) != value:
            raise ValueError('synthetic source byte mismatch')
        return MemoryPath(value['path'], raw)

    def check(self):
        self.checks += 1

    def summary(self):
        return dict(synthetic_only=True, samples=self.checks)


def runtime_fixture(request, control):
    names = {'numpy': ('library_sources', 'numpy'), 'psutil': ('library_sources', 'psutil'),
        'pycocotools': ('library_sources', 'pycocotools'), 'pycocotools.coco': ('library_sources', 'coco_loader'),
        'pycocotools._mask': ('library_sources', 'mask_extension'), 'pycocotools.mask': ('module_sources', 'mask_api'),
        'pycocotools.cocoeval': ('module_sources', 'cocoeval')}
    values = {}
    for name, (key, role) in names.items():
        value = ModuleType(name)
        value.__file__ = request[key][role]['path']
        values[name] = value
    values['numpy'].__version__ = '1.26.4'
    state = ('synthetic', [1, 2], 0, 0, 0.)
    values['numpy'].random = SimpleNamespace(get_state=lambda: deepcopy(state), set_state=lambda value: None)
    values['numpy'].array_equal = lambda a, b: a == b
    values['psutil'].__version__ = '6.1.1'
    values['psutil'].Process = lambda: SimpleNamespace(cpu_affinity=lambda: [0, 1])
    values['pycocotools.mask'].decode = lambda value: None
    values['pycocotools.coco'].COCO = lambda *args: None
    values['pycocotools.cocoeval'].COCOeval = lambda *args: None
    control.psutil = values['psutil']
    return values


def io_fixture(worker, fail=False, category=1):
    request, docs, shards = metadata_fixture()
    objects = {r['path']: b'x' for r in worker._references(request)}
    for role in ('population_contract', 'eval_body', 'official', 'geometry_diagnostic', 'population_metrics'):
        names = {'population_contract': ['merge_predictions'], 'eval_body': ['official_metrics'],
                 'official': ['coco_metrics'], 'geometry_diagnostic': ['diagnose'],
                 'population_metrics': ['relabel_geometry','_delta_stats']}[role]
        raw = ('# synthetic harmless '+role+'\n'+''.join('def '+name+'(*a,**k):pass\n' for name in names)).encode()
        objects[request['module_sources'][role]['path']] = raw
        request['module_sources'][role] = ref(request['module_sources'][role]['path'], raw)
    core = '''
def evaluate_u66_population(np,c,*,shards,native,strong,reference_COCO,coco,ids,modules):
    c.core_calls += 1
    assert len(shards)==4 and len(native)==len(strong)==18700 and len(ids)==187
    assert set(modules)=={'population_contract','eval_body','official','cocoeval','mask_api','geometry_diagnostic','population_metrics'}
    assert 'metrics' in reference_COCO['R'] and len(reference_COCO['R']['stats'])==12
    if FAIL:raise RuntimeError('synthetic core failure')
    return {'summary':{'status':'METRICS_COMPLETE_PENDING_ROOT_ACCEPTANCE','qualified':False,
        'ROOT_actual_terminal_self_certified':False,'AP':True,'DEV':True,'TEST':False,'new_training':False,
        'formal_training_budget_credit':0,'synthetic_only':True},'private_pair_records':[{'synthetic_only':True}]}
'''.replace('if FAIL:', 'if '+repr(fail)+':').encode()
    request['core_source'] = ref('/synthetic/core.py', core)
    objects['/synthetic/core.py'] = core
    annotation = dict(images=[{'id': i} for i in range(187)], annotations=[{'id': i} for i in range(380)], categories=[{'id': category}])
    docs['DEV'] = annotation
    for role in ('native_predictions', 'R_predictions', 'legacy_result', 'DEV'):
        raw = json.dumps(docs[role]).encode()
        request['input_refs'][role] = ref(request['input_refs'][role]['path'], raw)
        objects[request['input_refs'][role]['path']] = raw
    for name in ('context', 'legacy_context'):
        for a, b in (('native_predictions','native_predictions'), ('R_predictions','R_predictions'), ('legacy_ROOT_result','legacy_result')):
            docs[name][a] = dict(request['input_refs'][b], path='E:\\never-open\\'+b)
    docs['legacy_context']['DEV'] = dict(request['input_refs']['DEV'], path='E:\\never-open\\DEV')
    docs['legacy_context']['COCO_source'] = dict(request['module_sources']['official'], path='E:\\never-open\\official')
    for name in ('legacy_context', 'context'):
        if name == 'context':
            docs[name]['legacy_context'] = dict(request['input_refs']['legacy_context'], path='E:\\never-open\\legacy')
        raw = json.dumps(docs[name]).encode()
        request['input_refs'][name] = ref(request['input_refs'][name]['path'], raw)
        objects[request['input_refs'][name]['path']] = raw
    for i, shard in enumerate(shards):
        shard['source_refs'] = {k: deepcopy(request['module_sources'][k]) for k in ('population_contract','mask_api')}
        raw = json.dumps(shard).encode()
        request['shard_refs'][i] = ref(request['shard_refs'][i]['path'], raw)
        objects[request['shard_refs'][i]['path']] = raw
    control = Control(objects)
    control.core_calls = 0
    runtime = runtime_fixture(request, control)

    def coco(path):
        value = json.loads(objects[path])
        return SimpleNamespace(imgs={i['id']: i for i in value['images']},
                               anns={i['id']: i for i in value['annotations']}, cats={i['id']: i for i in value['categories']})

    runtime['pycocotools.coco'].COCO = coco
    pins = {k: {v: request['module_sources'][k][v] for v in ('bytes','sha256')} for k in worker.SOURCE_PINS}
    return request, control, runtime, pins


class MetadataTests(unittest.TestCase):
    def test_relocation_checks_content_without_following_original_Windows_path(self):
        worker = load_worker()
        original = ref('E:\\historical\\never-open.json')
        actual = ref('/synthetic/actual.json')
        value = worker.relocate('native_predictions', original, actual)
        self.assertEqual(value, dict(role='native_predictions', original=original, actual=actual))
        with self.assertRaisesRegex(ValueError, 'content'):
            worker.relocate('native_predictions', original, ref('/synthetic/actual.json', b'different'))

    def test_required_context_bridge_references_and_complete_COCO_projection(self):
        worker = load_worker()
        request, documents, shards = metadata_fixture()
        joined = worker.join_inputs(request, documents, shards)
        self.assertEqual(joined['ids'], list(range(187)))
        self.assertEqual(joined['reference_COCO']['R'], documents['legacy_result']['control_results']['strong_fixed_E24_Omega']['COCO'])
        self.assertTrue(any(v['role'] == 'context.legacy_context' for v in joined['relocations']))
        documents['context']['legacy_context']['sha256'] = 'a'*64
        with self.assertRaisesRegex(ValueError, 'context.legacy_context'):
            worker.join_inputs(request, documents, shards)
        documents['context']['legacy_context'] = dict(request['input_refs']['legacy_context'])
        documents['legacy_context']['image_ids'].reverse()
        with self.assertRaisesRegex(ValueError, 'image_ids'):
            worker.join_inputs(request, documents, shards)

    def test_exact_finite_request_pins_and_output_separation(self):
        worker = load_worker()
        request, documents, shards = metadata_fixture()
        request['core_source'].update(worker.CORE_PIN)
        for role, pin in worker.SOURCE_PINS.items():
            request['module_sources'][role].update(pin)
        request['library_sources']['pycocotools'] = ref('/synthetic/lib/pycocotools.py', b'')
        worker.validate_request(request)
        refs = worker.direct_references(request)
        self.assertEqual(refs, sorted(refs, key=lambda value: value['path']))
        self.assertEqual(len(refs), 24)
        request['module_sources']['official'] = ref('/synthetic/encoder.py', b'wrong-encoder')
        with self.assertRaisesRegex(ValueError, 'official'):
            worker.validate_request(request)
        request['module_sources']['official'].update(worker.SOURCE_PINS['official'])
        request['output_directory'] = '/synthetic'
        with self.assertRaisesRegex(ValueError, 'output'):
            worker.direct_references(request)
        request['output_directory'] = '/synthetic/OUT'
        request['input_refs']['DEV'] = request['library_sources']['pycocotools']
        with self.assertRaisesRegex(ValueError, 'reference'):
            worker.validate_request(request)

    def test_runtime_really_uses_canonical_import_boundary_and_CPU_hidden_GPU(self):
        worker = load_worker()
        request, documents, shards = metadata_fixture()
        control = Control({r['path']: b'x' for r in worker._references(request)})
        values = runtime_fixture(request, control)
        imports = []

        def import_module(name):
            imports.append(name)
            return values[name]

        with patch('importlib.import_module', side_effect=import_module), \
             patch('platform.system', return_value='Linux'), patch('platform.machine', return_value='aarch64'), \
             patch('platform.python_version', return_value='3.12.14'), \
             patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2'}):
            loaded = worker.import_runtime(control, request)
            self.assertIs(loaded['numpy'], values['numpy'])
            self.assertEqual(set(imports), set(values))
            values['numpy'].__file__ = '/wrong/numpy.py'
            with self.assertRaisesRegex(ValueError, 'numpy'):
                worker.import_runtime(control, request)

    def test_full_JSON_IO_loads_original_modules_calls_core_once_and_splits_outputs(self):
        worker = load_worker()
        request, control, runtime, pins = io_fixture(worker)
        with tempfile.TemporaryDirectory(dir=HERE) as folder:
            output = Path(folder)/'OUT'
            actual_path = lambda value: output if str(value) == request['output_directory'] else Path(value)
            prior = ModuleType('previous_official')
            with patch.object(worker, 'Path', side_effect=actual_path), \
                 patch.object(worker, '__file__', request['worker_source']['path']), \
                 patch.object(worker, 'SOURCE_PINS', pins), \
                 patch.object(worker, 'CORE_PIN', {k: request['core_source'][k] for k in ('bytes','sha256')}), \
                 patch('importlib.import_module', side_effect=lambda name: runtime[name]), \
                 patch('platform.system', return_value='Linux'), patch('platform.machine', return_value='aarch64'), \
                 patch('platform.python_version', return_value='3.12.14'), \
                 patch.dict('sys.modules', {'_U70_official': prior}), \
                 patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2'}):
                report = worker.run_worker(control, request, report={})
                import sys
                self.assertIs(sys.modules['_U70_official'], prior)
            self.assertEqual(control.core_calls, 1)
            self.assertEqual(report['status'], 'COMPLETE_PENDING_ROOT_ACCEPTANCE')
            self.assertFalse(report['qualified'])
            self.assertTrue(report['source_aliases_restored'])
            self.assertTrue(report['annotation_JSON_parsed'])
            self.assertEqual(json.loads((output/'SUMMARY.json').read_text())['synthetic_only'], True)
            self.assertEqual(json.loads((output/'FIXED_NATIVE_PAIRS_PRIVATE.json').read_text()), [{'synthetic_only': True}])
            self.assertNotIn('private_pair_records', json.loads((output/'SUMMARY.json').read_text()))
            self.assertEqual(json.loads((output/'REPORT.json').read_text()), report)
            self.assertFalse(any(p.startswith('E:') for p in control.visits))

    def test_core_failure_keeps_visible_stack_and_unknown_AP_scope_despite_cleanup_error(self):
        worker = load_worker()
        request, control, runtime, pins = io_fixture(worker, fail=True)
        original_check = control.check

        def check():
            if control.core_calls:
                raise ValueError('synthetic final control failure')
            original_check()

        control.check = check
        report = {}
        with tempfile.TemporaryDirectory(dir=HERE) as folder:
            output = Path(folder)/'OUT'
            actual_path = lambda value: output if str(value) == request['output_directory'] else Path(value)
            with patch.object(worker, 'Path', side_effect=actual_path), \
                 patch.object(worker, '__file__', request['worker_source']['path']), \
                 patch.object(worker, 'SOURCE_PINS', pins), \
                 patch.object(worker, 'CORE_PIN', {k: request['core_source'][k] for k in ('bytes','sha256')}), \
                 patch('importlib.import_module', side_effect=lambda name: runtime[name]), \
                 patch('platform.system', return_value='Linux'), patch('platform.machine', return_value='aarch64'), \
                 patch('platform.python_version', return_value='3.12.14'), \
                 patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2'}):
                with self.assertRaisesRegex(RuntimeError, 'synthetic core failure'):
                    worker.run_worker(control, request, report=report)
            saved = json.loads((output/'REPORT.json').read_text())
            self.assertEqual(saved['status'], 'FAILED')
            self.assertIn('evaluate_u66_population', saved['error']['traceback'])
            self.assertTrue(saved['annotation_JSON_parsed'])
            self.assertTrue(saved['evaluation_started'])
            self.assertIsNone(saved['AP'])
            self.assertEqual(saved['AP_computation_status'], 'STARTED_COMPLETION_UNKNOWN')
            self.assertIn('final control failure', saved['cleanup_errors'][0]['message'])
            self.assertIn('resource_observations', saved)
            self.assertTrue(saved['source_aliases_restored'])
            self.assertFalse((output/'SUMMARY.json').exists())

    def test_bad_bridge_precedes_import_and_actual_category_ID_precedes_metrics(self):
        worker = load_worker()
        for case in ('bridge', 'category'):
            request, control, runtime, pins = io_fixture(worker, category=2 if case == 'category' else 1)
            if case == 'bridge':
                path = request['input_refs']['context']['path']
                context = json.loads(control.objects[path])
                context['legacy_context']['sha256'] = 'a'*64
                raw = json.dumps(context).encode()
                control.objects[path] = raw
                request['input_refs']['context'] = ref(path, raw)
            report = {}
            with tempfile.TemporaryDirectory(dir=HERE) as folder:
                output = Path(folder)/'OUT'
                actual_path = lambda value: output if str(value) == request['output_directory'] else Path(value)
                with patch.object(worker, 'Path', side_effect=actual_path), \
                     patch.object(worker, '__file__', request['worker_source']['path']), \
                     patch.object(worker, 'SOURCE_PINS', pins), \
                     patch.object(worker, 'CORE_PIN', {k: request['core_source'][k] for k in ('bytes','sha256')}), \
                     patch('importlib.import_module', side_effect=lambda name: runtime[name]) as imports, \
                     patch('platform.system', return_value='Linux'), patch('platform.machine', return_value='aarch64'), \
                     patch('platform.python_version', return_value='3.12.14'), \
                     patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2'}):
                    with self.assertRaisesRegex(ValueError, 'context.legacy_context' if case == 'bridge' else 'categoryID1'):
                        worker.run_worker(control, request, report=report)
                    if case == 'bridge':
                        imports.assert_not_called()
                self.assertEqual(control.core_calls, 0)
                self.assertFalse(report['evaluation_started'])
                self.assertFalse((output/'SUMMARY.json').exists())
                self.assertEqual(json.loads((output/'REPORT.json').read_text())['status'], 'FAILED')


if __name__ == '__main__':
    unittest.main()
