"""U74-only stdlib/stub wiring checks. No real science, data or processes."""
from copy import deepcopy
import ast
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location('_u74_test_'+name, HERE/(name+'.py'))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def ref(path, raw=b'SYNTHETIC'):
    return dict(path=path, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


class MetadataPath(PurePosixPath):
    """Only the original worker's metadata output-path guard, no filesystem I/O."""
    def resolve(self, strict=False): return self
    def is_symlink(self): return False
    def is_dir(self): return True


class MemoryPath:
    def __init__(self, path, raw): self.path, self.raw = path, raw
    def __str__(self): return self.path
    def read_bytes(self): return self.raw


class Closer:
    def __init__(self, values): self.values, self.visits = values, []
    def __call__(self, value):
        self.visits.append(value['path'])
        raw = self.values[value['path']]
        if ref(value['path'], raw) != value: raise ValueError('Synthetic closure mismatch')
        return MemoryPath(value['path'], raw)


def fixture(entry, raw):
    original = ModuleType('_U74_TEST_FROZEN_U73_METADATA')
    original.__file__ = '/source/U73.py'
    with patch('pathlib.Path', MetadataPath):
        exec(compile(raw, original.__file__, 'exec'), original.__dict__)
    request = dict(worker_source=ref('/source/U73.py', raw),
        prediction_core_source=dict(ref('/source/U66.py'), sha256=original.CORE_SHA256),
        source_refs={k:ref('/source/'+k+'.py') for k in original.SOURCE_ROLES},
        library_sources={k:ref('/library/'+k+'.py', b'' if k=='pycocotools' else b'SYNTHETIC') for k in original.LIBRARIES},
        endpoint_refs={k:ref('/endpoint/'+k+'.json') for k in ('qualified','spec','pair_commit','order')},
        checkpoint_refs={k:ref('/endpoint/'+k+'.pth') for k in ('PLAIN','FULL')},
        manifest_ref=None, shard_index=1, dataset_range=[47,94], output_directory='/output/WORKER')
    for role, pin in original.SOURCE_PINS.items():
        request['source_refs'][role]['sha256'] = pin
    request['source_refs']['fit_portable_reader']['bytes'] = 19618
    request['source_refs']['encoder'].update(bytes=entry.ENCODER_BYTES, sha256=entry.ENCODER_SHA256)
    entries = [dict(dataset_index=i,image_id=i+1000,query_ids=list(range(99,-1,-1)),
        package_ref=ref('/input/DEV_FULL100_'+str(i)+'.pt')) for i in range(187)]
    manifest = json.dumps({'entries':entries}).encode('utf-8')
    request['manifest_ref'] = ref('/input/MANIFEST.json', manifest)
    spec = dict(schema=entry.SCHEMA,operation=entry.OPERATION,phase='U74_SYNTHETIC_ONLY',
        entry_source=ref('/source/entry.py'),owner_source=ref('/source/linux_owner.py'),request=request,
        limits=deepcopy(entry.LIMITS),output_directory='/output/ENTRY',owner_output_directory='/output/OWNER')
    refs = [spec['entry_source'],spec['owner_source'],request['worker_source'],request['prediction_core_source'],request['manifest_ref']]
    for key in ('source_refs','library_sources','endpoint_refs','checkpoint_refs'):
        refs += list(request[key].values())
    refs += [v['package_ref'] for v in entries[47:94]]
    spec['sources'] = sorted(refs,key=lambda v:v['path'])
    close = Closer({'/source/U73.py':raw,'/input/MANIFEST.json':manifest})
    return spec, entries, close


class VirtualPath:
    def __init__(self, fs, path): self.fs, self.logical = fs, PurePosixPath(str(path))
    def __str__(self): return str(self.logical)
    def __eq__(self, other): return str(self)==str(other)
    def __truediv__(self, name): return self.fs.path(str(self.logical/name))
    @property
    def parent(self): return self.fs.path(str(self.logical.parent))
    @property
    def local(self): return self.fs.root.joinpath(*self.logical.parts[1:])
    def is_absolute(self): return self.logical.is_absolute()
    def resolve(self, strict=False):
        if strict and not self.local.exists(): raise FileNotFoundError(str(self))
        return self
    def is_symlink(self): return False
    def exists(self): return self.local.exists()
    def is_dir(self): return self.local.is_dir()
    def is_file(self): return self.local.is_file()
    def mkdir(self): self.local.mkdir()
    def read_bytes(self): return self.local.read_bytes()
    def stat(self): return self.local.stat()
    def open(self,*args,**kwargs): return self.local.open(*args,**kwargs)


class VirtualFS:
    def __init__(self, root):
        self.root = Path(root).resolve()
        (self.root/'output').mkdir()
    def path(self, value): return VirtualPath(self, value)
    def write_json(self, path, value):
        raw = (json.dumps(value,sort_keys=True,separators=(',',':'))+'\n').encode('utf-8')
        obj = self.path(path)
        obj.local.write_bytes(raw)
        return ref(path,raw)


def owned_fixture(entry, fs, *, closure=True, fail=False):
    # Deliberately harmless metadata/worker definitions, not a genuine U73 proof.
    roles = ('training_state','training_entry','u36_helpers','u39_training_ops','shared_initialization',
        'full_legacy','full_dsconv','full_head','full_feature','plain_legacy','plain_feature','instance_adapter',
        'restore_state','portable_reader','image_geometry','prediction_core','encoder','mask_api',
        'population_contract','dev_reader','dev_rgb_reader','fit_portable_reader')
    raw = ('SOURCE_ROLES='+repr(roles)+'\nSOURCE_PINS={k:"a"*64 for k in SOURCE_ROLES if k!="encoder"}\n'
        +'CORE_SHA256="b"*64\nLIBRARIES=("torch","numpy","psutil","pycocotools","mask_extension")\n').encode()
    spec, entries, close = fixture(entry, raw)
    class Control:
        def __init__(self): self.checks, self.worker_entered = 0, False
        def closed(self, value):
            return close(value) if value['path'] in close.values else MemoryPath(value['path'], b'SYNTHETIC_UNREAD_INPUT')
        def check(self): self.checks += 1
        def summary(self): return dict(schema='U74-Linux-child-control-observations-v1',samples=self.checks,
            peak_RSS_bytes=1024,minimum_available_RAM_bytes=32*1024**3,violations=[],elapsed_seconds=0.1)
    control = Control()
    worker = ModuleType('_U74_SYNTHETIC_WORKER_BOUNDARY')
    worker.__file__ = spec['request']['worker_source']['path']
    def run(c, request, *, report):
        c.worker_entered = True
        fs.path(request['output_directory']).mkdir()
        report.update(source_refs=deepcopy(request['source_refs']), original_restore_source_closure_verified=closure)
        if fail:
            report.update(status='FAILED', AP=False, partial_prediction_evidence={'synthetic_only': True})
            fs.write_json(request['output_directory']+'/REPORT.json', report)
            raise RuntimeError('synthetic U73 prediction failure')
        refs21 = {k:v for k,v in request['source_refs'].items() if k!='fit_portable_reader'}
        prediction = dict(schema='U66-DGX-fixed50-DEV-shard-pending-v1',status='PREDICTIONS_COMPLETE_PENDING_ROOT_ACCEPTANCE',
            shard_index=1,dataset_range=[47,94],images=47,queries=4700,fixed_epoch=50,updates_per_arm=37050,query_batch=4,
            endpoint=dict(schema='U58-U57-DGX-fixed-endpoint-metadata-v1',mode='FORMAL',formal_endpoint=True,
                cursor=dict(global_updates=37050,completed_epochs=50,within_epoch=0),checkpoint_refs=request['checkpoint_refs']),
            source_refs=refs21,arms={a:dict(status='PREDICTIONS_COMPLETE_PENDING_ROOT_ACCEPTANCE') for a in ('PLAIN','FULL')},
            qualified=False,AP=False,DEV=True,DEV_technical_input=True,TEST=False,new_training=False,formal_training_budget_credit=0,
            whole_population_predicted=False,DEV_metrics_computed=False,GT_used_for_prediction=False,GT_rasterization_executed=False,
            remaining_model_objects=0,cleanup=dict(allocated_after=0,reserved_after=0),model_BN_Torch_RNG_unchanged=True,
            original_full_S0_class_order_preserved=True,source_bytes_rechecked_after=True)
        prediction_ref = fs.write_json(request['output_directory']+'/U66_SHARD.json',prediction)
        report.update(schema='U73-single-shard-worker-report-v1',status='COMPLETE_PENDING_ROOT_ACCEPTANCE',
            qualified=False,AP=False,DEV=True,DEV_technical_input=True,TEST=False,new_training=False,formal_training_budget_credit=0,
            ROOT_actual_terminal_self_certified=False,automatic_retry=False,cleanup_errors=[],shard_index=1,dataset_range=[47,94],
            worker_source=request['worker_source'],prediction_core_source=request['prediction_core_source'],prediction_ref=prediction_ref,
            direct_input_refs=[r for r in spec['sources'] if r not in (spec['entry_source'],spec['owner_source'])],
            source_aliases_restored=True,source_and_metadata_bytes_rechecked=True,outer_runtime_states_restored=True,elapsed_seconds=0.1)
        fs.write_json(request['output_directory']+'/REPORT.json',report)
        return report
    worker.run_worker = run
    return spec, control, worker


class Tests(unittest.TestCase):
    def test_dedicated_contract_keeps_original_gpu_and_encoder_limits(self):
        entry = load('entry')
        self.assertEqual(entry.SCHEMA, 'ROOT-U74-DGX-fixed50-single-shard-supervision-spec-v1')
        self.assertEqual(entry.OPERATION, 'GPU_U74_DGX_FIXED50_SINGLE_SHARD_PREDICTION')
        self.assertEqual(entry.LIMITS, dict(allowed_cpus=[0,1], threads=2, allocator_bytes=4*1024**3,
            maximum_RSS_bytes=16*1024**3, minimum_available_RAM_bytes=16*1024**3, deadline_seconds=1800))
        self.assertEqual(entry.ENCODER_BYTES, 61452)
        self.assertEqual(entry.ENCODER_SHA256, '49c6dbc37434be372d8bb5b7e841a519491db8122f1eacef314a9e87f29a94e7')

    def test_frozen_U73_metadata22_union_rejects_old_worker_wrong_pin_and_extra(self):
        entry=load('entry')
        worker_path=HERE.parent/'U73_DGX_FIXED50_PREDICTION_SOURCE_CLOSURE_DRAFT_20261021_01/worker.py'
        raw=worker_path.read_bytes()  # Ordinary frozen code, never any science input.
        spec,entries,close=fixture(entry,raw)
        with patch('pathlib.Path',MetadataPath):
            actual=entry.close_source_union(spec,close)
        self.assertEqual(actual,{v['path']:v for v in spec['sources']})
        self.assertEqual(len(spec['request']['source_refs']),22)
        self.assertIn(spec['request']['source_refs']['fit_portable_reader']['path'],actual)
        self.assertIn(entries[47]['package_ref']['path'],actual)
        self.assertNotIn(entries[46]['package_ref']['path'],actual)
        for sha in ('46211627ffdf3692e6d8e322013eabbe6c5e3d943d677bb076f46c7b0c056fa7','f'*64):
            bad=deepcopy(spec);bad['request']['worker_source']['sha256']=sha
            before=len(close.visits)
            with self.assertRaisesRegex(ValueError,'U73'):
                entry.close_source_union(bad,close)
            self.assertEqual(len(close.visits),before)
        bad=deepcopy(spec);bad['sources'].append(ref('/unadmitted/EXTRA.pt'))
        bad['sources'].sort(key=lambda v:v['path'])
        with patch('pathlib.Path',MetadataPath),self.assertRaisesRegex(ValueError,'union'):
            entry.close_source_union(bad,close)
        self.assertEqual(set(close.visits),{'/source/U73.py','/input/MANIFEST.json'})

    def test_owned_gpu_launch_and_eight_original_lifecycle_guards(self):
        owner = load('linux_owner')
        entry = load('entry')
        self.assertEqual((owner.SCHEMA, owner.OPERATION, owner.LIMITS), (entry.SCHEMA, entry.OPERATION, entry.LIMITS))
        with patch('subprocess.Popen', return_value='SYNTHETIC_NO_PROCESS') as spawn:
            self.assertEqual(owner.launch_child({'entry_source': {'path': '/source/entry.py'}},
                Path('/source/SPEC.json'), 'a'*64, object(), object()), 'SYNTHETIC_NO_PROCESS')
        args, kwargs = spawn.call_args
        self.assertEqual(args[0][-4:], ['--spec', str(Path('/source/SPEC.json')), '--spec-sha', 'a'*64])
        self.assertTrue(kwargs['start_new_session'])
        self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], '0')
        self.assertEqual(kwargs['env']['CUBLAS_WORKSPACE_CONFIG'], ':4096:8')
        self.assertEqual([kwargs['env'][k] for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')], ['2']*3)
        guards = ('supervise_child','parse_proc_stat','proc_identity','birth_absent',
                  '_group_members','_kill_owned_group','_prepare_child','boot_id')
        old = HERE.parent/'U69_DGX_FIXED50_SINGLE_SHARD_SUPERVISION_DRAFT_20261021_01/linux_owner.py'
        trees = [ast.parse(path.read_text(encoding='utf-8')) for path in (old, HERE/'linux_owner.py')]
        maps = [{n.name: ast.dump(n, include_attributes=False) for n in t.body if isinstance(n, ast.FunctionDef)} for t in trees]
        for name in guards:
            self.assertEqual(maps[0][name], maps[1][name], name)

    def test_failed_original_restore_source_proof_cannot_become_entry_success(self):
        entry=load('entry')
        with tempfile.TemporaryDirectory(dir=HERE) as root:
            fs=VirtualFS(root)
            spec,control,worker=owned_fixture(entry,fs,closure=False)
            with patch.object(entry,'U73_SHA256',spec['request']['worker_source']['sha256']), \
                 patch.object(entry,'Path',fs.path), patch.object(entry,'close_source_union',return_value={}):
                with self.assertRaisesRegex(ValueError,'restore source'):
                    entry.execute_owned(spec,ref('/metadata/SPEC.json'),control,worker,report={})

    def test_original21_prediction_is_persisted_with_outer22_proof(self):
        entry=load('entry')
        with tempfile.TemporaryDirectory(dir=HERE) as root:
            fs=VirtualFS(root)
            spec,control,worker=owned_fixture(entry,fs)
            report={}
            with patch.object(entry,'U73_SHA256',spec['request']['worker_source']['sha256']), \
                 patch.object(entry,'Path',fs.path), patch.object(entry,'close_source_union',return_value={}):
                self.assertIs(entry.execute_owned(spec,ref('/metadata/SPEC.json'),control,worker,report=report), report)
            saved=json.loads(fs.path('/output/ENTRY/REPORT.json').read_bytes())
            prediction=json.loads(fs.path('/output/WORKER/U66_SHARD.json').read_bytes())
            self.assertEqual(len(prediction['source_refs']),21)
            self.assertNotIn('fit_portable_reader',prediction['source_refs'])
            self.assertEqual(len(saved['worker_report']['source_refs']),22)
            self.assertEqual(saved['prediction_ref'],saved['worker_report']['prediction_ref'])
            self.assertEqual(saved['prediction_metadata']['source_refs'],prediction['source_refs'])
            self.assertFalse(saved['AP']);self.assertFalse(saved['qualified'])

    def test_U73_failure_keeps_partial_outer22_and_original_error(self):
        entry=load('entry')
        with tempfile.TemporaryDirectory(dir=HERE) as root:
            fs=VirtualFS(root)
            spec,control,worker=owned_fixture(entry,fs,fail=True)
            original_check=control.check
            def check():
                if control.worker_entered: raise ValueError('synthetic final RAM failure')
                original_check()
            control.check=check
            control.summary=lambda:dict(schema='U74-Linux-child-control-observations-v1',samples=2,
                peak_RSS_bytes=1024,minimum_available_RAM_bytes=16*1024**3-1,violations=['free_RAM'],elapsed_seconds=0.1)
            with patch.object(entry,'U73_SHA256',spec['request']['worker_source']['sha256']), \
                 patch.object(entry,'Path',fs.path), patch.object(entry,'close_source_union',return_value={}):
                with self.assertRaisesRegex(RuntimeError,'U73 prediction failure'):
                    entry.execute_owned(spec,ref('/metadata/SPEC.json'),control,worker,report={})
            saved=json.loads(fs.path('/output/ENTRY/REPORT.json').read_bytes())
            self.assertEqual(saved['status'],'FAILED')
            self.assertIn('U73 prediction failure',saved['error']['traceback'])
            self.assertEqual(len(saved['worker_report']['source_refs']),22)
            self.assertTrue(saved['worker_report']['partial_prediction_evidence']['synthetic_only'])
            self.assertEqual(saved['resource_observations']['violations'],['free_RAM'])
            self.assertFalse(saved['AP']);self.assertFalse(saved['qualified'])


if __name__ == '__main__':
    unittest.main()
