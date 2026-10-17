"""UNBOUND per-image dense FIT reader; no scientific imports or I/O at import.

Reuses qualified original full pixel-decoder F carriers, natural-query e/L and
true original matcher pairs. No teacher forward, point-F interpolation, JPEG,
DEV, quality-head inputs or weights are loaded. Caller releases each returned
image before requesting the next; this reader retains only metadata and COCO.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def _read(c, reference):
    return json.loads(Path(c.closed(reference)).read_bytes())


def _population_records(fit, qualified, report, report_ref):
    """Pure metadata check; never derives matches from GT overlap or point draws."""
    need(qualified['status'] == 'ROOT_N13_SAMEK_REGION_FIT_CACHE_CPU_QUALIFIED_V2'
         and qualified['GPU_report'] == report_ref
         and report['schema'] == 'N13-sameK-region-FIT-cache-report-v1'
         and qualified['teacher_checkpoint'] == report['teacher_checkpoint'] == fit['teacher_checkpoint']
         and qualified['images'] == report['images'] == fit['images'] == 741
         and qualified['raw_GT'] == report['raw_GT'] == fit['GT'] == 1661
         and qualified['validation_carrier_refs'] == report['validation_carrier_refs']
         and fit['dataset_indices'] == list(range(741)),
         'Original qualified complete dense-F population and same frozen teacher')
    need(all(len(fit[key]) == 741 for key in
             ('identities','image_ids','existing_packet_refs','existing_native_L_refs'))
         and len(report['rows']) == len(report['validation_carrier_refs']) == 741,
         'All 741 original image/container references are required')
    records=[];all_annotations=set();all_images=set();total=0
    for index,row in enumerate(report['rows']):
        identity=fit['identities'][index]
        image=fit['image_ids'][index];annotations=identity['ann_ids'];G=len(annotations)
        queries=row['query_ids']
        need(type(image) is int and image not in all_images
             and row['dataset_index'] == index and row['image_id'] == identity['image_id'] == image
             and row['identity'] == identity and row['raw_GT'] == G
             and row['validation_carrier'] == report['validation_carrier_refs'][index]
             and 1 <= G <= 100 and len(set(annotations)) == G
             and all(type(a) is int and a not in all_annotations for a in annotations)
             and type(queries) is list and len(queries) == len(set(queries)) == G
             and all(type(q) is int and 0 <= q < 100 for q in queries),
             'Canonical image, full GT order, distinct real queries and exact carrier association')
        for key in ('original_shape','actual_img_shape'):
            shape=identity[key]
            need(type(shape) in (list,tuple) and len(shape) == 2
                 and all(type(value) is int and value > 0 for value in shape),
                 'Original/crop grid metadata must remain intact')
        need(all(value <= 800 for value in identity['actual_img_shape'])
             and identity['batch_input_shape'] == [800,800], 'Original padded 800 canvas')
        all_images.add(image);all_annotations.update(annotations);total+=G
        records.append(dict(dataset_index=index,image_id=image,identity=deepcopy(identity),
                            G=int(G),query_ids=list(queries)))
    need(total == len(all_annotations) == 1661, 'Every true original FIT GT retained exactly once')
    return records


def _tensor(torch, value, dtype, shape, label):
    need(isinstance(value,torch.Tensor) and value.device.type == 'cpu'
         and value.dtype == dtype and tuple(value.shape) == tuple(shape)
         and not value.requires_grad and bool(torch.isfinite(value).all()),
         label+': original finite detached CPU tensor and exact dtype/shape required')
    return value


def _digest(value):
    array=value.detach().cpu().contiguous().numpy()
    return dict(shape=list(array.shape),dtype=str(array.dtype),
                sha256=hashlib.sha256(array.tobytes()).hexdigest())


def _input_references(context, fit, report):
    """Enumerate exact refs for Source binding, without opening those files."""
    values=[context[key] for key in ('FIT_context','N13_qualified','N13_report')]+[fit['FIT']]
    values+=report['validation_carrier_refs']+fit['existing_packet_refs']+fit['existing_native_L_refs']
    result=[];seen={}
    for value in values:
        need(type(value) is dict and set(value) == {'path','bytes','sha256'}
             and type(value['path']) is str and Path(value['path']).is_absolute()
             and type(value['bytes']) is int and value['bytes'] >= 0
             and type(value['sha256']) is str and len(value['sha256']) == 64
             and set(value['sha256']) <= set('0123456789abcdef'), 'Strict direct-input reference')
        key=value['path'].replace('\\','/').casefold()
        need(key not in seen or seen[key] == value, 'Contradictory direct-input references')
        if key not in seen:
            seen[key]=value;result.append(deepcopy(value))
    return result


class _Reader:
    def __init__(self, torch, np, c, context, fit, report, records, coco):
        self._torch,self._np,self._c=torch,np,c
        self._fit,self._report,self._records,self._coco=fit,report,records,coco
        self.records=deepcopy(records)
        self.input_references=_input_references(context,fit,report)

    def load(self, index):
        need(type(index) is int and 0 <= index < 741, 'Exact FIT image ordinal required')
        torch,np,c=self._torch,self._np,self._c
        c.check()
        fit,report=self._fit,self._report
        record=self._records[index];row=report['rows'][index]
        identity=record['identity'];G=record['G'];queries=record['query_ids']
        H,W=identity['original_shape'];annotations=identity['ann_ids']
        carrier=torch.load(str(c.closed(report['validation_carrier_refs'][index])),map_location='cpu')
        packet=torch.load(str(c.closed(fit['existing_packet_refs'][index])),map_location='cpu')
        native=torch.load(str(c.closed(fit['existing_native_L_refs'][index])),map_location='cpu')
        need(carrier['schema'] == 'N13-sameK-region-FIT-validation-carrier-v1'
             and packet['schema'] == 'N16-all100-FIT-action-sampled-cache-body-v1'
             and native['schema'] == 'N16-actual-GPU-natural100-L-carrier-v1'
             and all(value['dataset_index'] == index and value['identity'] == identity
                     for value in (carrier,packet,native)), 'Original image/container schema and identity')
        need(all(carrier[key] == report[key] for key in ('source','spec','phase','teacher_checkpoint'))
             and carrier['teacher_checkpoint'] == packet['teacher_checkpoint'] == fit['teacher_checkpoint']
             and packet['fixed_PR_checkpoint'] == fit['fixed_PR_checkpoint']
             and packet['original_cache_packet'] == row['original_cache_packet']
             and packet['matching_loss_seed'] == row['matching_loss_seed']
             and packet['spec'] == fit['N16_GPU_spec']
             and all(native[key] == packet[key] for key in ('source','spec','GPU_admission','phase'))
             and packet['natural_native_L_carrier'] == fit['existing_native_L_refs'][index],
             'Same frozen teacher, PR, original N07 parent and natural-L source bindings')
        qids=_tensor(torch,carrier['query_ids'],torch.int64,(G,),'Original GT-ordered queries')
        need(qids.tolist() == queries, 'Original GT/query order cannot be changed')
        F=_tensor(torch,carrier['native_F256'],torch.float32,(1,256,200,200),'Full original F')
        Mfull=_tensor(torch,carrier['matched_M200'],torch.float32,(1,G,200,200),'Original matched M')
        e0=_tensor(torch,packet['native_e0'],torch.float32,(1,100,256),'Natural all100 e0')
        Lall=_tensor(torch,native['native_L'],torch.float32,(100,H,W),'Natural original-grid all100 L')
        need({key:_digest(carrier[key]) for key in ('query_ids','native_F256','matched_M200')}
             == row['carrier_digests'], 'Original exact full-F/M/query tensor digests')
        assignment=packet['assignment']
        mapping=_tensor(torch,assignment['raw_GT_index_by_query'],torch.int64,(100,),'Original matcher')
        mapping_values=mapping.tolist()
        need(packet['natural_query_ids'] == assignment['natural_query_ids'] == list(range(100))
             and packet['raw_GT_ids'] == annotations
             and all(-2 <= g < G for g in mapping_values)
             and sorted(g for g in mapping_values if g >= 0) == list(range(G))
             and [mapping_values[q] for q in queries] == list(range(G))
             and assignment['raw_GT_annotation_id_by_query'] ==
                 [annotations[g] if g >= 0 else None for g in mapping_values],
             'Every raw GT exactly once through the original matcher; ignored/background never targets')
        image=self._coco.imgs[record['image_id']]
        need(image['height'] == H and image['width'] == W, 'Annotation and original L grids must agree')
        masks=[]
        for annotation_id in annotations:
            annotation=self._coco.anns[annotation_id]
            need(annotation['id'] == annotation_id and annotation['image_id'] == record['image_id']
                 and annotation['category_id'] == 1, 'Original GT annotation identity/category')
            mask=np.ascontiguousarray(self._coco.annToMask(annotation) != 0,dtype=np.bool_)
            need(mask.shape == (H,W), 'Complete original-grid instance GT required')
            masks.append(mask)
        raw=np.ascontiguousarray(np.stack(masks),dtype=np.bool_)
        need(_digest(torch.from_numpy(raw)) == packet['raw_GT_digest'],
             'Original GT pixel stack digest and raw instance order exact')
        selected=torch.tensor(queries,dtype=torch.int64,device='cpu')
        e=e0[0].index_select(0,selected).contiguous()
        M=Mfull[0].contiguous()
        L=Lall.index_select(0,selected).contiguous()
        targets=torch.from_numpy(raw.astype(np.float32,copy=True)).contiguous()
        _tensor(torch,e,torch.float32,(G,256),'Matched e')
        _tensor(torch,M,torch.float32,(G,200,200),'Matched coarse M')
        _tensor(torch,L,torch.float32,(G,H,W),'Matched original L')
        _tensor(torch,targets,torch.float32,(G,H,W),'True complete GT targets')
        c.check()
        # Advanced selection copies L/e: returning G queries does not retain the
        # all100 full-resolution L container. No tensor is retained on self.
        return dict(F=F,e=e,M=M,L=L,targets=targets,G=int(G),query_ids=list(queries),
                    identity=deepcopy(identity))


def prepare(torch, np, c, context):
    """Open only qualified metadata/annotation; load dense tensors one image at a time."""
    need(type(context) is dict and all(key in context for key in
         ('FIT_context','N13_qualified','N13_report')), 'Three original FIT dense-input metadata refs')
    c.check()
    fit=_read(c,context['FIT_context'])
    qualified=_read(c,context['N13_qualified'])
    report=_read(c,context['N13_report'])
    need(fit['schema'] == 'ROOT-N26-quality-input-context-v1', 'Original complete FIT context schema')
    records=_population_records(fit,qualified,report,context['N13_report'])
    from pycocotools.coco import COCO
    coco=COCO(str(c.closed(fit['FIT'])))
    expected_annotations={a for record in records for a in record['identity']['ann_ids']}
    need(len(coco.imgs) == 741 and len(coco.anns) == 1661 and set(coco.cats) == {1}
         and set(coco.imgs) == {record['image_id'] for record in records}
         and set(coco.anns) == expected_annotations, 'Complete original FIT741/1661 annotation population')
    c.check()
    return _Reader(torch,np,c,context,fit,report,records,coco)
