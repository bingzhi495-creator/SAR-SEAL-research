"""One finite, image-only, native RF-DETR E24 DEV187 export original."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import types


SCHEMA = 'Root-RFDETR-native-closed-endpoint-DEV187-export-spec-v1'
SUCCESS = 'NATIVE_E24_EMA_DEV187_EXPORT_COMPLETE_PENDING_ROOT'
FAILURE = 'NATIVE_E24_EMA_DEV187_EXPORT_FAILED_PRESERVED'
FILES = dict(entry='entry.py', runtime='runtime.py', worker='e24_dev_export_worker.py',
             long='e100_worker.py', boundary='e1_resume.py', pilot='ema_pilot_worker.py',
             raw_export='e1_export.py', canonical='canonical_export.py', config_entry='config_entry.py',
             yaml='config.yaml', audit='offline_audit.py', input_manifest='dev_image_manifest.json')
PINS = dict(
    runtime='544b21396f5b854afc55641f6880c80800e5158f37360a9246918d8f8f2b7194',
    worker='7bf88b0880a360f52de9263b215bfda30d34ab14d91183dc33a13a13b8aa26f5',
    long='ecc8c67bf67e4e58d0d697bf4ac48b9ff4a339ff1d162019ce95b73a0fed0144',
    boundary='014e087379ffe6e8d32329c337d6b625b98dfc2d235cb321becebfdf3832b013',
    pilot='e9fd227e098742e5406833877d13a91dced5bf64a12b0a338c98ce24d91c594f',
    raw_export='4e1fe8e825f31152380ee4191adb92987a828669eb14189058622ac2d22c4039',
    canonical='b2e99e581acb0236f6cee5542321bb3e01b2c620f2c727062b28b71eedfd1820',
    config_entry='f0642e8f33e75a2440eac0968dd76f270d445498ba88e812543902e7d0ad8453',
    yaml='ba9508d9acab5ff68b10db8a1aff5d9001868d6dda00e39f299d6f5cc1b8011b',
    audit='e64116cdc39c6bf46d9eab59670124431b2a5ca6b713c3e0f43af3017335078e',
    input_manifest='18df007e15e3e838e607f178f14e7a26f2b37b992ee39bf091e0b74cf084fb13',
)
URLLIB_SHA = '2633bbdb69731e5ccb5cf4e4afd65605d86c7979cc5633126f50c92d5ad74a74'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha_string(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def exact(value, keys):
    need(type(value) is dict and set(value) == set(keys), 'Exact scoped input fields')


def canonical(path, *, directory=False):
    need(type(path) is str, 'Explicit selected path')
    p = Path(path)
    need(p.is_absolute() and p.as_posix() == path and p.resolve(strict=True) == p and not p.is_symlink()
         and (p.is_dir() if directory else p.is_file()), 'Canonical selected path')
    return p


def read_source(ref, path, expected=None):
    exact(ref, ('path', 'bytes', 'sha256'))
    p = canonical(ref['path'])
    need(p == path and type(ref['bytes']) is int and 0 < ref['bytes'] <= 1024**2
         and p.stat().st_size == ref['bytes'] and sha_string(ref['sha256']), 'Bounded exact source reference')
    data = p.read_bytes()
    need(len(data) == ref['bytes'] and digest(data) == ref['sha256']
         and (expected is None or ref['sha256'] == expected), 'Exact original source bytes')
    return data


def load(ref, source, role):
    m = types.ModuleType('_root_rf_e24_export_' + role)
    m.__file__ = ref['path']
    sys.modules[m.__name__] = m
    exec(compile(source, m.__file__, 'exec'), m.__dict__)
    return m


def install_guard(dataset, image_paths, archive, denied, seen):
    def hook(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes)):
            return
        path = Path(os.path.abspath(os.fsdecode(args[0])))
        mode, flags = args[1:3]
        writing = ((isinstance(mode, str) and any(c in mode for c in 'wax+'))
                   or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT |
                                                          os.O_TRUNC | os.O_APPEND)))
        data = path.is_relative_to(dataset.parent)
        weight = path.suffix.lower() in {'.pt', '.pth', '.ckpt', '.safetensors'}
        protected = data or weight or path == archive
        admitted = path == archive or path in image_paths
        if ((protected and (not admitted or writing))
                or path.is_relative_to('/data/REQUIRES_QUALIFIED_SAR_COCO_SPLITS')):
            denied.append(str(path))
            raise PermissionError('Outside admitted read-only DEV image and E24 checkpoint roles')
        if path in image_paths:
            seen.add(path)
    sys.addaudithook(hook)


def run(spec, out, stage):
    receipt = dict(schema='Root-RFDETR-native-E24-DEV187-export-result-v1', status=FAILURE,
                   qualified=False, terminal_self_certified=False, AP=False, TEST=False, training=False,
                   blocked_input_opens=[], offline_audit=None, last_completed_phase='startup', result=None)
    audit = None
    epoch = spec['endpoint_epoch']
    steps = epoch * 186
    need(type(epoch) is int and epoch in (24,50,100), 'Fixed closed endpoint epoch')
    try:
        refs = spec['sources']
        exact(refs, {*FILES, 'urllib3'})
        src = {role: read_source(refs[role], stage / filename, PINS.get(role))
               for role, filename in FILES.items()}
        runtime = load(refs['runtime'], src['runtime'], 'runtime')
        urllib3 = runtime.BASE / 'urllib3/util/connection.py'
        read_source(refs['urllib3'], urllib3, URLLIB_SHA)
        receipt['versions'] = runtime.isolated_runtime()
        auditor = load(refs['audit'], src['audit'], 'audit')
        audit = auditor.install_offline_audit(refs['urllib3'])
        items_doc = json.loads(src['input_manifest'])
        exact(items_doc, ('schema', 'population', 'expected_ids', 'items', 'model_execution', 'AP', 'TEST'))
        need(items_doc['schema'] == 'Root-RFDETR-DEV187-image-only-manifest-v1'
             and items_doc['population'] == 'internal_DEV187'
             and items_doc['model_execution'] is items_doc['AP'] is items_doc['TEST'] is False
             and len(items_doc['items']) == len(items_doc['expected_ids']) == 187, 'Frozen ordered DEV metadata')
        dataset = canonical(spec['dataset_root'], directory=True)
        paths = {canonical(row['image_ref']['path']) for row in items_doc['items']}
        need(len(paths) == 187 and all(p.parent == dataset/'valid' and p.suffix == '.jpg' for p in paths),
             'Complete admitted DEV image paths')
        archive_ref = spec['archive_ref']
        archive = canonical(archive_ref['path'])
        need(archive.name == f'checkpoint_{epoch - 1}.ckpt' and out not in archive.parents and dataset.parent not in out.parents,
             'Original independently accepted E24 archive and fresh export output')
        seen = set()
        install_guard(dataset, paths, archive, receipt['blocked_input_opens'], seen)
        receipt['last_completed_phase'] = 'exact_scoped_sources_ready'
        runtime.reference(archive_ref, archive, 1024**3)

        import torch

        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)
        need(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Expected single CUDA')
        device = torch.cuda.get_device_properties(0)
        cap = min(8*1024**3, int(device.total_memory))
        torch.cuda.set_per_process_memory_fraction(cap/device.total_memory, 0)
        config_entry = load(refs['config_entry'], src['config_entry'], 'config_entry')
        model, train = config_entry._load_configs(stage/'config.yaml', dataset)
        api = {k: load(refs[k], src[k], k) for k in ('worker','long','boundary','pilot','raw_export','canonical')}
        api['worker'].EPOCH = epoch
        api['worker'].E24_STEPS = steps
        api['worker'].ARCHIVE_NAME = archive.name
        receipt['last_completed_phase'] = 'native_configs_ready'

        def event(value):
            print(encoded(value).decode('utf-8'), end='', flush=True)

        artifact = out/'ARTIFACTS'
        result = api['worker'].run_e24_dev_export(archive_ref, items_doc['items'], items_doc['expected_ids'], model, train,
                    recipe_sha256=spec['recipe_sha256'], validate_archive_progress=api['long'].validate_archive_progress,
                    boundary_api=api['boundary'], pilot_api=api['pilot'],
                    export_e1_image=api['raw_export'].export_e1_image,
                    canonical_export=api['canonical'].export_coco_mask_predictions,
                    output_directory=artifact.as_posix(), event_sink=event)
        receipt['result'] = result
        need(result['status'] == SUCCESS and result['qualified'] is result['terminal_self_certified'] is False
             and result['AP'] is result['TEST'] is result['GT_read'] is result['training'] is False
             and result['full_DEV'] is True and result['fixed_epoch'] == epoch and result['optimizer_updates'] == steps
             and result['checkpoint_sha256'] == archive_ref['sha256'] and result['recipe_sha256'] == spec['recipe_sha256']
             and result['image_count'] == 187 and result['native_count_total'] == 18700 and result['prediction_count'] > 0
             and result['prediction_count'] == result['codec_roundtrip_masks']
             and result['prediction_count'] + result['unmapped_slot1_total'] == 18700 and seen == paths
             and result['cuda_peak_reserved_bytes'] <= cap, 'Complete original no-GT DEV187 export')
        runtime.reference(result['predictions_ref'], artifact/'PREDICTIONS.json', 256*1024**2)
        runtime.reference(result['image_manifest_ref'], artifact/'IMAGE_MANIFEST.json', 1024**2)
        need(set(artifact.iterdir()) == {artifact/'PREDICTIONS.json',artifact/'IMAGE_MANIFEST.json'}, 'Exact two private output files')
        runtime.reference(archive_ref, archive, 1024**3)
        need(all(read_source(refs[k], stage/FILES[k], PINS.get(k)) == src[k] for k in FILES), 'Original staged sources unchanged')
        receipt.update(status=SUCCESS, unique_DEV_images_read=len(seen), last_completed_phase='e24_dev_export_artifacts_rechecked')
    except BaseException as error:
        receipt['error'] = dict(type=type(error).__name__, message=str(error)[:4000])
    finally:
        receipt['offline_audit'] = audit.receipt() if audit else None
        if not receipt['offline_audit'] or receipt['offline_audit']['passed'] is not True or receipt['blocked_input_opens']:
            receipt['status'] = FAILURE
        with (out/'RESULT.json').open('xb') as f:
            f.write(encoded(receipt));f.flush();os.fsync(f.fileno())
    return receipt['status']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec', required=True)
    p.add_argument('--spec-sha256', required=True)
    args = p.parse_args()
    path = canonical(args.spec)
    need(path.stat().st_size <= 32*1024, 'Bounded original spec')
    raw = path.read_bytes()
    need(sha_string(args.spec_sha256) and digest(raw) == args.spec_sha256, 'Exact original spec bytes')
    spec = json.loads(raw)
    exact(spec, ('schema','output_dir','sources','dataset_root','archive_ref','recipe_sha256','endpoint_epoch'))
    need(spec['schema'] == SCHEMA and sha_string(spec['recipe_sha256']), 'Exact selected archive recipe')
    out = canonical(spec['output_dir'], directory=True)
    stage = canonical((out/'SOURCE').as_posix(), directory=True)
    need(path == stage/'spec.json' and Path(__file__).resolve(strict=True) == stage/'entry.py'
         and not (out/'RESULT.json').exists() and not (out/'ARTIFACTS').exists(), 'Original fresh isolated worker scope')
    status = run(spec, out, stage)
    return 0 if status == SUCCESS and path.read_bytes() == raw else 1


if __name__ == '__main__':
    raise SystemExit(main())
