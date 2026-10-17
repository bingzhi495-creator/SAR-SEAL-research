"""Unbound fixed-population dense DEV reader; no tensor imports at import time.

Only original native low-resolution fields and original GPU-captured L are
loaded. No image, annotation, original teacher weight or GT matcher is used.
"""
import copy
import hashlib
import json
from pathlib import Path

INHERITED=('native_predictions','R_predictions','encoder_source','legacy_ROOT_result')


def need(ok,message):
    if not ok:raise RuntimeError(message)


def validate_ref(value):
    need(type(value) is dict and set(value)=={'path','bytes','sha256'}
         and type(value['path']) is str and Path(value['path']).is_absolute()
         and type(value['bytes']) is int and value['bytes']>=0
         and type(value['sha256']) is str and len(value['sha256'])==64
         and set(value['sha256'])<=set('0123456789abcdef'),'Strict closed reference required')


def prepare_records(context,legacy):
    """Check canonical187 metadata; preserve each original arbitrary query order."""
    need(context['schema']=='ROOT-N28-fixed24-dense-native-DEV-context-v1'
         and legacy['schema']=='ROOT-N23-qualified-cache-DEV-context-v1'
         and context['image_ids']==legacy['image_ids']
         and len(context['image_ids'])==len(set(context['image_ids']))==187
         and all(type(v) is int for v in context['image_ids'])
         and len(context['rows'])==len(legacy['rows'])==187,'Complete canonical DEV population')
    need(all(context[key]==legacy[key] for key in INHERITED),'Original prediction/export/qualification sources unchanged')
    references=[context['legacy_context']]+[context[key] for key in INHERITED]
    for index,(row,old) in enumerate(zip(context['rows'],legacy['rows'])):
        queries=row['qids']
        need(row['dataset_index']==index and type(row['dataset_index']) is int
             and row['image_id']==context['image_ids'][index]
             and type(queries) is list and len(queries)==100
             and all(type(q) is int for q in queries) and set(queries)==set(range(100))
             and {k:v for k,v in row.items() if k!='dense_packet'}==old,
             'Original row, native100 query permutation and export order unchanged')
        references.extend(row[key] for key in ('bundle','original_capture_manifest','dense_packet'))
    unique={}
    for value in references:
        validate_ref(value)
        key=value['path'].replace('\\','/').casefold()
        need(key not in unique or unique[key]==value,'Conflicting path identity')
        unique[key]=value
    return copy.deepcopy(context['rows']),copy.deepcopy(list(unique.values()))


def _read(c,value):
    return json.loads(Path(c.closed(value)).read_bytes())


def _digest(value):
    array=value.detach().cpu().contiguous().numpy()
    return dict(shape=list(array.shape),dtype=str(array.dtype),
                sha256=hashlib.sha256(array.tobytes()).hexdigest())


def _tensor(torch,value,shape,label):
    need(isinstance(value,torch.Tensor) and value.dtype==torch.float32
         and value.device.type=='cpu' and tuple(value.shape)==tuple(shape)
         and not value.requires_grad and bool(torch.isfinite(value).all()),
         label+': finite original detached CPU FP32 shape')
    return value


class _Reader:
    def __init__(self,torch,c,context,records,references):
        self._torch,self._c,self._context=torch,c,context
        self.records,self.input_references=records,references
        self._manifests={}

    def load(self,index):
        """One original complete image, natural tensor indices plus export qids.

        This path has not been executed on real DEV in the draft version.
        Its first owned GPU execution must calibrate all original native RLEs.
        """
        need(type(index) is int and 0<=index<187,'Canonical DEV ordinal')
        torch,c,context=self._torch,self._c,self._context
        row=self.records[index];c.check()
        reference=row['original_capture_manifest']
        if reference['path'] not in self._manifests:
            self._manifests[reference['path']]=_read(c,reference)
        manifest=self._manifests[reference['path']]
        original=manifest['rows'][index-manifest['dataset_ordinal_range'][0]]
        need(original['dataset_index']==index and original['image_id']==row['image_id']
             and original['bundle']==row['bundle'] and original['qids']==row['qids']
             and original['shared_actual_image_inputs']==row['shared_locator']
             and original['native_packet']==row['dense_packet']
             and manifest['capture_binding']==row['capture_binding'],
             'Original manifest/bundle/full-field packet and member association')
        bundle=torch.load(str(c.closed(row['bundle'])),map_location='cpu')
        need(bundle['schema']=='N16-actual-GPU-image-capture-bundle-v1'
             and bundle['binding']==row['capture_binding']
             and bundle['dataset_index']==index and bundle['image_id']==row['image_id'],
             'Original actual-GPU image bundle identity')
        shared=bundle['members'][row['shared_locator']]
        packet=torch.load(str(c.closed(row['dense_packet'])),map_location='cpu')
        identity=packet['identity'];H,W=identity['original_shape']
        need(packet['schema']=='N07-original-detail-all100-native-fields-v1'
             and packet['dataset_index']==index and packet['prediction_query_ids']==row['qids']
             and packet['teacher_checkpoint']==context['teacher_checkpoint']
             and shared['schema']=='N16-actual-GPU-image-shared-inputs-v1'
             and shared['native_packet']==row['dense_packet'] and shared['identity']==identity
             and identity['image_id']==row['image_id'] and [H,W]==row['original_shape']
             and identity['batch_input_shape']==[800,800]
             and len(identity['actual_img_shape'])==2
             and all(type(v) is int and 0<v<=800 for v in identity['actual_img_shape'])
             and shared['actual_GPU_F_captured'] is True and shared['CPU_F_rerendered'] is False
             and shared['dataset_index']==index and shared['image_id']==row['image_id']
             and shared['natural_query_ids']==list(range(100))
             and shared['native_export_query_ids']==row['qids'],
             'Same teacher, image/native query order and original/crop geometry')
        fields=packet['native_fields']
        need(set(fields)=={'F256','e0','M200','final_class_logits'},'All original dense fields, no sampled-F substitution')
        shapes={'F256':(1,256,200,200),'e0':(1,100,256),'M200':(1,100,200,200),
                'final_class_logits':(1,100,2)}
        for name,shape in shapes.items():_tensor(torch,fields[name],shape,name)
        need({name:_digest(value) for name,value in fields.items()}==packet['field_digests'],
             'Every original field tensor digest exact')
        native_L=_tensor(torch,shared['native_L'],(100,H,W),'Original GPU native_L')
        embeddings=_tensor(torch,shared['e0'],(100,256),'Original natural e0')
        need(torch.equal(embeddings,fields['e0'][0]),'Captured and original dense-packet natural embeddings bitexact')
        result=dict(F=fields['F256'],e=fields['e0'][0],M=fields['M200'][0],L=native_L,
                    identity=copy.deepcopy(identity),query_ids=list(row['qids']))
        # References to unrelated saved point scouts/F_unique do not escape.
        del bundle,shared,packet,fields,embeddings,native_L
        c.check()
        return result


def prepare(torch,c,context):
    legacy=_read(c,context['legacy_context'])
    records,references=prepare_records(context,legacy)
    accepted=_read(c,context['legacy_ROOT_result'])
    need(accepted['qualified'] is True,'Original DEV cache Root accepted')
    return _Reader(torch,c,context,records,references)
