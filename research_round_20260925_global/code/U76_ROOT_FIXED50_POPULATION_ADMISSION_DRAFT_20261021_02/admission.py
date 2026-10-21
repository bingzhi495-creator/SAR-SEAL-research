"""UNBOUND Root-only population admission math; no I/O, launch or science."""
from copy import deepcopy
import hashlib
import json
from pathlib import PurePosixPath, PureWindowsPath
from types import ModuleType

ACCEPTANCE_PIN=dict(bytes=34428,sha256='fd7ee03ded035ced8f3aea9c74afbf018b7e6ef73a25b5a463018d9c3b4f41c2')
RANGES=((0,47),(47,94),(94,141),(141,187))


def need(ok,message):
    if not ok:
        raise ValueError(message)


def signature(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)


def strict_ref(value,*,allow_empty=False):
    need(type(value) is dict and set(value)=={'path','bytes','sha256'}
         and type(value['path']) is str and value['path']
         and type(value['bytes']) is int and value['bytes']>=(0 if allow_empty else 1)
         and type(value['sha256']) is str and len(value['sha256'])==64
         and set(value['sha256'])<=set('0123456789abcdef'),'Exact explicit file reference')
    path=value['path']
    parts=PurePosixPath(path).parts if path.startswith('/') else PureWindowsPath(path).parts
    need((PurePosixPath(path).is_absolute() if path.startswith('/') else PureWindowsPath(path).is_absolute())
         and '..' not in parts,'Explicit absolute artifact path')
    return value


def decode_artifact(reference,raw):
    """Bind actual original bytes before parsing; never guess a reencoding hash."""
    strict_ref(reference)
    need(type(raw) is bytes and len(raw)==reference['bytes']
         and hashlib.sha256(raw).hexdigest()==reference['sha256'],'Actual original artifact bytes/SHA')
    value=json.loads(raw)
    signature(value)
    return value


def close_inputs(input_refs,raw_inputs,expected_refs):
    """Exact finite caller-provided byte set; no file reads or nested ref chasing."""
    need(type(expected_refs) is list and expected_refs and type(input_refs) is list
         and type(raw_inputs) is dict,'Explicit input reference and raw-byte lists')
    expected={}
    for value in expected_refs:
        strict_ref(value)
        need(value['path'] not in expected,'Distinct input role paths required')
        expected[value['path']]=value
    need(input_refs==[expected[k] for k in sorted(expected)]
         and set(raw_inputs)==set(expected),'Exact actual input refs/raw keys; no missing, extra or replaced path')
    for path,value in expected.items():
        raw=raw_inputs[path]
        need(type(raw) is bytes and len(raw)==value['bytes']
             and hashlib.sha256(raw).hexdigest()==value['sha256'],'Actual original input bytes/SHA')
    return dict(raw_inputs)


def shard_image_ids(prediction,span):
    """Derive range-order IDs only after pinned U75 validated each full shard.

    U75 owns full100/two-arm/proof/cost/Omega checks. U76 only joins their
    already-validated identity blocks and tests cross-shard distinct coverage.
    """
    return [prediction['identity_predictions'][100*offset]['image_id']
            for offset in range(span[1]-span[0])]


def admit_population(input_refs,raw_inputs,*,proof_refs,prediction_refs,endpoint,cpu_request,acceptance_source_ref):
    """Bind four explicit original artifacts; never launch or certify CPU work."""
    need(type(proof_refs) is list and len(proof_refs)==4
         and type(prediction_refs) is list and len(prediction_refs)==4,'Four ordered proof and original prediction refs')
    need(type(cpu_request) is dict and cpu_request['shard_refs']==prediction_refs,
         'CPU original shard refs must match each accepted original ordinal')
    need(set(cpu_request)=={'worker_source','core_source','module_sources','library_sources','input_refs','shard_refs','output_directory'},
         'Original U70 seven-key CPU request')
    need(type(endpoint) is dict and endpoint['schema']=='U58-U57-DGX-fixed-endpoint-metadata-v1'
         and endpoint['mode']=='FORMAL' and endpoint['formal_endpoint'] is True
         and signature(endpoint['cursor'])==signature(dict(global_updates=37050,completed_epochs=50,within_epoch=0))
         and type(endpoint['checkpoint_refs']) is dict and set(endpoint['checkpoint_refs'])=={'PLAIN','FULL'},
         'Original complete FORMAL E50 endpoint scope and two checkpoints')
    for value in endpoint['checkpoint_refs'].values(): strict_ref(value)
    strict_ref(acceptance_source_ref)
    need(all(acceptance_source_ref[k]==v for k,v in ACCEPTANCE_PIN.items()),'Exact frozen U75 acceptance source')
    actual=close_inputs(input_refs,raw_inputs,[acceptance_source_ref]+proof_refs+prediction_refs)
    validator=ModuleType('_U76_CLOSED_U75')
    validator.__file__=acceptance_source_ref['path']
    exec(compile(actual[acceptance_source_ref['path']],validator.__file__,'exec'),validator.__dict__)
    need(callable(getattr(validator,'validate_accepted_proof',None)),'Original U75 pure accepted-proof validator')
    proofs,predictions,image_ids=[],[],[]
    for index,(proof_ref,prediction_ref,span) in enumerate(zip(proof_refs,prediction_refs,RANGES)):
        proof=decode_artifact(proof_ref,actual[proof_ref['path']])
        prediction=decode_artifact(prediction_ref,actual[prediction_ref['path']])
        validator.validate_accepted_proof(proof,prediction)
        need(proof['prediction_ref']==prediction_ref,'Accepted proof must bind the exact original prediction path/bytes/SHA')
        need(signature(proof['endpoint'])==signature(endpoint)==signature(prediction['endpoint']),
             'Same whole accepted endpoint, original checkpoints and source identity')
        need(type(proof['source_refs']) is dict and len(proof['source_refs'])==21
             and signature(proof['source_refs'])==signature(prediction['source_refs']), 'Original21 source identities')
        need(type(proof['outer_source_refs']) is dict and set(proof['outer_source_refs'])==set(proof['source_refs'])|{'fit_portable_reader'}
             and all(proof['outer_source_refs'][k]==v for k,v in proof['source_refs'].items()), 'Exact outer22 source identities')
        need(type(proof['library_sources']) is dict
             and set(proof['library_sources'])=={'torch','numpy','psutil','pycocotools','mask_extension'},
             'Original selected prediction library sources')
        for value in proof['outer_source_refs'].values(): strict_ref(value)
        for role,value in proof['library_sources'].items(): strict_ref(value,allow_empty=role=='pycocotools')
        strict_ref(proof['evidence_refs']['manifest'])
        if proofs:
            need(proof['evidence_refs']['manifest']==proofs[0]['evidence_refs']['manifest'],
                 'Identical actual original187 manifest ref across all four accepted shards')
            need(all(signature(proof[key])==signature(proofs[0][key])
                     for key in ('source_refs','outer_source_refs','library_sources')),
                 'Identical actual source and selected library refs across four shards')
        for role in ('population_contract','mask_api'):
            need(proof['source_refs'][role]==cpu_request['module_sources'][role],
                 'Original actual population/mask source ref must equal CPU request')
        for role in ('pycocotools','mask_extension'):
            need(proof['library_sources'][role]==cpu_request['library_sources'][role],
                 'Original actual shared mask library ref must equal CPU request; not an ABI proof')
        for value in (proof,prediction):
            need(type(value['shard_index']) is int and value['shard_index']==index
                 and signature(value['dataset_range'])==signature(list(span)), 'Fixed ordered original shard ranges')
            for key,expected in (('images',span[1]-span[0]),('queries',100*(span[1]-span[0])),
                                 ('fixed_epoch',50),('updates_per_arm',37050)):
                need(type(value[key]) is int and value[key]==expected,'Complete original shard population and budget')
        proofs.append(proof);predictions.append(prediction)
        image_ids += shard_image_ids(prediction,span)
    need(len(image_ids)==len(set(image_ids))==187,'Complete187 distinct original image identities in range order')
    return dict(schema='Root-U76-fixed50-population-admission-material-v1',
        status='FOUR_ROOT_ACCEPTED_SHARDS_BOUND_FOR_FUTURE_CPU',
        external_shard_admission_refs=deepcopy(proof_refs),shard_refs=deepcopy(prediction_refs),
        endpoint=deepcopy(endpoint),cpu_request_metadata_sha256=hashlib.sha256(signature(cpu_request).encode('utf-8')).hexdigest(),
        input_refs=deepcopy(input_refs),acceptance_source_ref=deepcopy(acceptance_source_ref),
        images=sum(p['images'] for p in predictions),queries=sum(p['queries'] for p in predictions),image_ids=image_ids,
        qualified=False,ROOT_actual_terminal_self_certified=False,CPU_execution_performed=False,
        GPU_execution_performed=False,AP=False,DEV=False,DEV_technical_input=True,TEST=False,new_training=False,
        formal_training_budget_credit=0,automatic_CPU_launch=False,
        source_module_self_certifies_original_observations=False)
