"""UNBOUND U54 full-FIT portable CPU reader; no numerical imports or I/O on import."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import weakref


SCHEMA='U54-full-FIT-portable-manifest-v1'
POPULATION=dict(images=741,GT=1661,tensor_population_revalidated=False)
FIELDS=('rgb400','e','M','L','targets')
ORIGINAL_READER_PINS={
    'rgb_matched_reader':'678ce3781872e21e961b24a7b0a4d0ecc69f0f8a08b8feb7eb0b2e8624d57025',
    'matched_reader':'82565caa9c42b6c2ed501e13af9554c6ef47a99b6b525ccfce0bcef620534870',
    'image_geometry':'65db107de80c5f1d4dade7cb2ffd3f241fd6bf7f9b233cb1aa9c90cbfd00abf7',
}
EXPECTED_BATCHES=(list(range(32)),list(range(32,64)),list(range(64,741)))


def need(ok,message):
    if not ok:raise ValueError(message)


def validate_coverage(indices,declared_instances):
    """Declared original metadata population only; never a tensor/GT revalidation."""
    need(type(indices) is list and all(type(index) is int for index in indices)
         and indices==list(range(741)) and type(declared_instances) is int
         and declared_instances==1661,'Exactly original indices0..740 once in order and1661 declared matched instances')
    return dict(POPULATION)


def _reference(value):
    need(type(value) is dict and set(value)=={'path','bytes','sha256'}
         and type(value['path']) is str
         and (PureWindowsPath(value['path']).is_absolute() or PurePosixPath(value['path']).is_absolute())
         and type(value['bytes']) is int and value['bytes']>=0,'Strict original/local absolute reference')
    _sha(value['sha256'])
    return value


def _sha(value):
    need(type(value) is str and len(value)==64 and set(value)<=set('0123456789abcdef'),'Lowercase SHA256')
    return value


def _primitive(value):
    if value is None or type(value) in (str,int,bool):return
    if type(value) is float:need(math.isfinite(value),'Finite primitive metadata')
    elif type(value) in (list,tuple):
        for child in value:_primitive(child)
    elif type(value) is dict:
        need(all(type(key) is str for key in value),'String metadata keys')
        for child in value.values():_primitive(child)
    else:raise ValueError('Primitive manifest/package metadata only, no tensors or arbitrary objects')


def validate_export(value,export_index):
    """Exact historical batch/schema declaration; actual Root acceptance is external."""
    need(type(export_index) is int and 0<=export_index<3 and type(value) is dict
         and set(value)=={'root_qualified_ref','root_qualified_schema','package_schema','dataset_indices'},
         'Exactly three explicitly described historical export batches')
    generation='U51' if export_index<2 else 'U53'
    need(value['root_qualified_schema']=='ROOT-'+generation+'-matched-CPU-export-qualified-v1'
         and value['package_schema']==generation+'-matched-CPU-portable-item-v1'
         and type(value['dataset_indices']) is list and all(type(index) is int for index in value['dataset_indices'])
         and value['dataset_indices']==EXPECTED_BATCHES[export_index],
         'Exact original0..31/32..63 U51 and intended64..740 U53; no U49/two-image shortcut')
    _reference(value['root_qualified_ref'])
    return deepcopy(value)


def _shape(value):
    need(type(value) in (list,tuple) and len(value)==2 and all(type(v) is int and v>0 for v in value),
         'Positive original H,W metadata')
    return list(value)


def _proofs(fields,G,original):
    need(type(fields) is dict and set(fields)==set(FIELDS),'Exactly five original matched field proofs')
    H,W=original
    shapes=dict(rgb400=[1,3,400,400],e=[G,256],M=[G,200,200],L=[G,H,W],targets=[G,H,W])
    for name,shape in shapes.items():
        proof=fields[name];logical=4
        for dimension in shape:logical*=dimension
        need(type(proof) is dict and set(proof)=={'shape','dtype','logical_bytes','source_sha256','sha256',
             'bitexact','independent_storage','compact_storage'} and type(proof['shape']) is list
             and all(type(v) is int for v in proof['shape']) and proof['shape']==shape
             and proof['dtype']=='torch.float32' and type(proof['logical_bytes']) is int
             and proof['logical_bytes']==logical and _sha(proof['source_sha256'])==_sha(proof['sha256'])
             and all(proof[k] is True for k in ('bitexact','independent_storage','compact_storage')),
             'Exact shape/dtype/byte/compactness export proof: '+name)
    return shapes


def validate_row(row,export,index):
    """One supplied original record + accepted-export/transfer metadata; no files."""
    need(type(index) is int and 0<=index<741 and type(row) is dict
         and set(row)=={'dataset_index','record','package_ref','original_package_ref','export_index','summary'}
         and type(row['dataset_index']) is int and row['dataset_index']==index
         and type(row['export_index']) is int,'Original manifest row and explicit export membership')
    validate_export(export,row['export_index'])
    need(index in export['dataset_indices'],'Original ordinal belongs to its precise historical batch')
    record=row['record']
    need(type(record) is dict and set(record)=={'dataset_index','image_id','G','query_ids','identity'}
         and type(record['dataset_index']) is int and record['dataset_index']==index
         and type(record['image_id']) is int and type(record['G']) is int and 1<=record['G']<=100,
         'General original G, never hardcoded pilot images')
    identity=record['identity'];G=record['G'];queries=record['query_ids']
    need(type(identity) is dict and identity['image_id']==record['image_id']
         and type(queries) is list and len(queries)==len(set(queries))==G
         and all(type(q) is int and 0<=q<100 for q in queries)
         and type(identity['ann_ids']) is list and len(identity['ann_ids'])==len(set(identity['ann_ids']))==G
         and all(type(a) is int for a in identity['ann_ids']),'Original GT/query one-to-one order')
    _primitive(identity);_reference(identity['image'])
    original=_shape(identity['original_shape']);content=_shape(identity['actual_img_shape']);longest=max(original)
    need(identity['batch_input_shape']==[800,800]
         and content==[(2*dimension*800+longest)//(2*longest) for dimension in original],
         'Original round-half-up keep-ratio content, canvas800 and branch400')
    local=_reference(row['package_ref']);origin=_reference(row['original_package_ref'])
    need(local['bytes']>0 and (local['bytes'],local['sha256'])==(origin['bytes'],origin['sha256']),
         'Transferred local file must retain exact accepted original package bytes/SHA')
    summary=row['summary']
    need(type(summary) is dict and set(summary)=={'images','matched_instances','fields','package_bytes',
         'package_sha256','weights_only_CPU_roundtrip_verified'} and type(summary['images']) is int
         and summary['images']==1 and type(summary['matched_instances']) is int and summary['matched_instances']==G
         and type(summary['package_bytes']) is int and summary['package_bytes']==origin['bytes']
         and summary['package_sha256']==origin['sha256'] and summary['weights_only_CPU_roundtrip_verified'] is True,
         'Original accepted per-package summary, not a newly inferred GT or tensor qualification')
    _proofs(summary['fields'],G,original)
    return deepcopy(record)


def validate_manifest(manifest):
    """Validate full metadata, not the741 payloads or Root terminal/qualification history."""
    need(type(manifest) is dict and set(manifest)=={'schema','population_metadata','original_context',
         'original_reader_sources','exports','rows'} and manifest['schema']==SCHEMA
         and manifest['population_metadata']==POPULATION
         and manifest['population_metadata']['tensor_population_revalidated'] is False,
         'Explicit U54 full-FIT manifest metadata scope')
    _primitive(manifest)
    context=manifest['original_context'];sources=manifest['original_reader_sources']
    need(type(context) is dict and set(context)=={'FIT_context','N13_qualified','N13_report'}
         and type(sources) is dict and set(sources)==set(ORIGINAL_READER_PINS),'Original source/context lineage')
    for reference in context.values():_reference(reference)
    for role,pin in ORIGINAL_READER_PINS.items():
        need(_reference(sources[role])['sha256']==pin,'Original immutable reader/geometry source: '+role)
    exports=manifest['exports'];rows=manifest['rows']
    need(type(exports) is list and len(exports)==3 and type(rows) is list and len(rows)==741,
         'Three accepted export descriptions and741 original metadata rows')
    for export_index,export in enumerate(exports):validate_export(export,export_index)
    records=[];paths=set()
    for index,row in enumerate(rows):
        need(type(row) is dict and type(row.get('export_index')) is int and 0<=row['export_index']<3,'Declared export index')
        records.append(validate_row(row,exports[row['export_index']],index))
        path=row['package_ref']['path'];key=path.replace('\\','/')
        if PureWindowsPath(path).is_absolute():key=key.casefold()
        need(key not in paths,'One distinct local package path per original image');paths.add(key)
    validate_coverage([record['dataset_index'] for record in records],sum(record['G'] for record in records))
    return deepcopy(records)


def _metadata(meta,row,manifest):
    record=row['record'];identity=record['identity']
    need(type(meta) is dict and set(meta)=={'dataset_index','image_id','G','query_ids','ann_ids','identity',
         'rgb_receipt','geometry','provenance','population_metadata'},'Exact U51/U53 matched metadata ABI')
    _primitive(meta)
    need(type(meta['dataset_index']) is int and meta['dataset_index']==record['dataset_index']
         and type(meta['image_id']) is int and meta['image_id']==record['image_id']
         and type(meta['G']) is int and meta['G']==record['G'] and meta['query_ids']==record['query_ids']
         and meta['identity']==identity and meta['ann_ids']==identity['ann_ids']
         and meta['population_metadata']==POPULATION
         and meta['population_metadata']['tensor_population_revalidated'] is False,
         'Exact original identity/GT/query order and declared population metadata; never reorder/rematch')
    geo=meta['geometry'];original=_shape(identity['original_shape']);content=_shape(identity['actual_img_shape'])
    need(type(geo) is dict and set(geo)=={'original_shape','actual_img_shape','canvas_size','branch_size'}
         and geo['original_shape']==original and geo['actual_img_shape']==content
         and geo['canvas_size']==800 and geo['branch_size']==400,'Original reviewed geometry parameters')
    rgb=meta['rgb_receipt']
    need(type(rgb) is dict and rgb['image']==identity['image'] and rgb['original_shape']==original
         and rgb['actual_img_shape']==content and rgb['canvas_size']==800 and rgb['branch_size']==400
         and rgb['decoded_mode']=='RGB' and rgb['auto_orientation'] is False
         and (rgb['exif_orientation'] is None or type(rgb['exif_orientation']) is int and rgb['exif_orientation']==1)
         and rgb['teacher_photometric_reconstruction'] is False
         and rgb['input_convention']=='RGB uint8 -> FP32 /255; common reviewed geometry'
         and rgb['baseline']=='N16 native_L selected by unchanged original GT-matched query ids'
         and rgb['image_source_control'] in ('base','separate_explicit_admitted_control'),
         'Original stored RGB/native-L receipt, no changed photometry or strong-R substitution')
    provenance=meta['provenance']
    need(type(provenance) is dict and set(provenance)=={'context','source_refs','input_refs'}
         and provenance['context']==manifest['original_context']
         and provenance['source_refs']==manifest['original_reader_sources']
         and type(provenance['input_refs']) is list and provenance['input_refs'],
         'Original context and source refs preserved; Windows lineage is not reopened on DGX')
    seen={}
    for reference in provenance['input_refs']:
        _reference(reference);key=reference['path'].replace('\\','/').casefold()
        need(key not in seen or seen[key]==reference,'No contradictory original input provenance')
        seen[key]=reference
    need(identity['image'] in provenance['input_refs']
         and all(reference in provenance['input_refs'] for reference in provenance['context'].values()),
         'Actual original image and context remain in inherited export provenance')


def _closed(c,reference):
    _reference(reference);path=Path(c.closed(reference))
    need(path.is_absolute() and path.resolve()==Path(reference['path']).resolve(),'Exact admitted local file path')
    return path


def _digest(value):
    return hashlib.sha256(value.detach().contiguous().numpy().tobytes(order='C')).hexdigest()


class PortableReader:
    """U39 consumed ABI only; no F, persistent items, GT reconstruction or CUDA work."""
    def __init__(self,torch,c,manifest,image_geometry):
        self.records=validate_manifest(manifest)
        self._manifest=deepcopy(manifest)
        self._torch,self._c,self._geometry=torch,c,image_geometry
        self._live=[]
        self.visits=[]
        self.load_attempts=0
        self.failed_loads=0
        self.last_audit=None

    def load(self,index):
        c,torch=self._c,self._torch
        need(type(index) is int and 0<=index<741,'Original full-FIT dataset index')
        need(all(reference() is None for reference in self._live),
             'Release previous returned tensor objects before loading another image; no persistent image cache')
        row=self._manifest['rows'][index]
        need(self.records[index]==row['record'],'Public records must retain the exact original metadata')
        export=self._manifest['exports'][row['export_index']]
        self.load_attempts+=1
        package=tensors=value=item=geometry=None
        self.last_audit=dict(dataset_index=index,status='LOADING')
        try:
            c.check();path=_closed(c,row['package_ref'])
            package=torch.load(str(path),map_location='cpu',weights_only=True)
            need(type(package) is dict and set(package)=={'schema','metadata','tensors','equivalence','selected_indices',
                 'images','qualified','new_training','AP','DEV','TEST','whole_population_exported'}
                 and package['schema']==export['package_schema'] and package['selected_indices']==export['dataset_indices']
                 and all(type(i) is int for i in package['selected_indices'])
                 and type(package['images']) is int and package['images']==1
                 and all(package[k] is False for k in ('qualified','new_training','AP','DEV','TEST','whole_population_exported')),
                 'Exact manifest-declared pending U51/U53 single-image package, never U49')
            _metadata(package['metadata'],row,self._manifest)
            tensors=package['tensors'];proofs=package['equivalence'];record=row['record'];G=record['G']
            need(type(tensors) is dict and type(proofs) is dict and set(tensors)==set(proofs)==set(FIELDS)
                 and proofs==row['summary']['fields'],'Exact five fields and original Root-accepted export proof')
            shapes=_proofs(proofs,G,record['identity']['original_shape']);pointers=set();checks={}
            for name in FIELDS:
                c.check();value=tensors[name]
                need(isinstance(value,torch.Tensor) and value.device.type=='cpu' and value.dtype==torch.float32
                     and value.layout==torch.strided and tuple(value.shape)==tuple(shapes[name])
                     and value.storage_offset()==0 and value.is_contiguous() and not value.requires_grad
                     and value.grad_fn is None and bool(torch.isfinite(value).all()),
                     'Finite detached compact CPU FP32 field: '+name)
                logical=int(value.numel()*value.element_size());pointer=value.untyped_storage().data_ptr()
                need(value.untyped_storage().nbytes()==logical and pointer not in pointers,
                     'Independent compact loaded storage across the five fields: '+name)
                pointers.add(pointer);digest=_digest(value)
                need(digest==proofs[name]['sha256']==proofs[name]['source_sha256'],
                     'Transferred field bytes match original export, not a reconstructed tensor: '+name)
                checks[name]=dict(shape=list(value.shape),dtype=str(value.dtype),logical_bytes=logical,sha256=digest)
            value=None
            need(bool(((tensors['rgb400']>=0)&(tensors['rgb400']<=1)).all()),'Stored RGB400 remains in[0,1]')
            need(bool(((tensors['targets']==0)|(tensors['targets']==1)).all()),'Original binary per-instance targets')
            geo=package['metadata']['geometry']
            geometry=self._geometry.make_geometry(torch,ori_shape=tuple(geo['original_shape']),
                content_shape=tuple(geo['actual_img_shape']),canvas_size=800,branch_size=400)
            item=dict(tensors,G=G,query_ids=deepcopy(record['query_ids']),identity=deepcopy(record['identity']),geometry=geometry)
            audit=dict(dataset_index=index,status='LOADED',G=G,source_package=deepcopy(row['package_ref']),
                original_package=deepcopy(row['original_package_ref']),export_qualified=deepcopy(export['root_qualified_ref']),
                tensor_checks=checks,original_metadata=deepcopy(package['metadata']),
                original_source_authentication='Root-accepted export and authenticated manifest/transfer required; not rederived',
                qualified=False,new_training=False,AP=False)
            c.check()
            self._live=[weakref.ref(tensors[name]) for name in FIELDS]
            self.visits.append(index)
            self.last_audit=audit
            return item
        except BaseException as error:
            self.failed_loads+=1
            self.last_audit.update(status='FAILED',error=dict(type=type(error).__name__,message=str(error)))
            raise
        finally:
            package=tensors=value=item=geometry=None


def prepare(torch,c,manifest,*,image_geometry,source_refs):
    """Future owned invocation only. Caller already closed the manifest JSON.

    Source refs are exact local portable_reader/image_geometry files; original
    Windows contexts/export Q refs in the manifest remain provenance only.
    The separate actual DGX protocol must admit every local741 package ref.
    """
    need(callable(getattr(c,'closed',None)) and callable(getattr(c,'check',None))
         and type(source_refs) is dict and set(source_refs)=={'portable_reader','image_geometry'},
         'Explicit owned local Source closure and resource controller')
    c.check()
    for role,reference in source_refs.items():
        path=_closed(c,reference);raw=path.read_bytes()
        need(len(raw)==reference['bytes'] and hashlib.sha256(raw).hexdigest()==reference['sha256'],
             'Actual admitted immutable reader/geometry bytes: '+role)
        expected=Path(__file__) if role=='portable_reader' else Path(image_geometry.__file__)
        need(path.resolve()==expected.resolve(),'Exact source-backed module object: '+role)
    need(source_refs['image_geometry']['sha256']==ORIGINAL_READER_PINS['image_geometry']
         and callable(getattr(image_geometry,'make_geometry',None)),'Same original800/400 numerical geometry')
    reader=PortableReader(torch,c,manifest,image_geometry)
    c.check()
    return reader
