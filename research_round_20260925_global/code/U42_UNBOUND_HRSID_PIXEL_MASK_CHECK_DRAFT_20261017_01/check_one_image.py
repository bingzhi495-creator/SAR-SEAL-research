"""UNBOUND single-image HRSID pixel/COCO-raster compatibility check.

No numerical libraries are imported. All real decoding/raster work is pending a
future owned caller injecting genuine modules and exact source/resource checks.
This does not establish semantic label alignment, blind TEST or model utility.
"""
from copy import deepcopy
import hashlib
import math
from pathlib import Path


def _need(ok,message):
    if not ok:raise ValueError(message)


def _failure(error,stage):
    return dict(stage=stage,type=type(error).__name__,message=str(error))


def _number(value):
    return type(value) in (int,float) and math.isfinite(value)


def _annotation_rows(annotations,image_id):
    """Retain original instances, including crowd and invalid metadata rows."""
    rows=[];seen=set()
    for ordinal,annotation in enumerate(annotations):
        row=dict(ordinal=ordinal,annotation_id=None,status='NOT_ATTEMPTED',
                 component_count=None,iscrowd=None,error=None,
                 out_of_canvas_coordinate_count=None,out_of_canvas_component_count=None)
        try:
            _need(type(annotation) is dict and type(annotation['id']) is int,'Original integer annotation ID')
            row['annotation_id']=annotation['id']
            _need(annotation['id'] not in seen,'Duplicate annotation ID within this original image')
            seen.add(annotation['id'])
            _need(type(annotation['image_id']) is int and annotation['image_id']==image_id
                  and type(annotation['category_id']) is int and annotation['category_id']==1,
                  'Original image association and HRSID ship category1')
            _need(type(annotation['iscrowd']) is int and annotation['iscrowd'] in (0,1),
                  'Original crowd value is retained, never used to discard an instance')
            segmentation=annotation['segmentation']
            _need(type(segmentation) is list and segmentation,'Nonempty original COCO polygon-component list')
            out_coordinates=out_components=0
            for component in segmentation:
                _need(type(component) is list and len(component)>=6 and len(component)%2==0
                      and all(_number(value) for value in component),
                      'Finite original polygon coordinates with valid component structure')
                outside=sum(value<0 or value>800 for value in component)
                out_coordinates+=outside;out_components+=int(outside>0)
            _need(_number(annotation['area']) and annotation['area']>=0,'Finite nonnegative original area metadata')
            bbox=annotation['bbox']
            _need(type(bbox) is list and len(bbox)==4 and all(_number(value) for value in bbox)
                  and bbox[2]>0 and bbox[3]>0,'Finite original bbox with positive width/height')
            row.update(component_count=len(segmentation),iscrowd=annotation['iscrowd'],
                       declared_area=annotation['area'],declared_bbox=list(bbox),
                       out_of_canvas_coordinate_count=out_coordinates,
                       out_of_canvas_component_count=out_components)
        except Exception as error:
            row.update(status='FAILED',error=_failure(error,'ANNOTATION_METADATA'))
        rows.append(row)
    return rows


def geometry_metadata_difference(annotation,raster_area,raster_bbox):
    """Pure scalar bookkeeping, not a raster or a validity/semantic decision."""
    _need(type(raster_area) is int and raster_area>=0,'Actual measured integer raster area')
    _need(type(raster_bbox) is list and len(raster_bbox)==4 and all(_number(v) for v in raster_bbox),
          'Actual measured finite xywh raster bbox')
    _need(_number(annotation['area']) and type(annotation['bbox']) is list
          and len(annotation['bbox'])==4 and all(_number(v) for v in annotation['bbox']),
          'Original finite area/bbox metadata')
    return dict(declared_area=annotation['area'],raster_area=raster_area,
        area_difference_raster_minus_declared=raster_area-annotation['area'],
        declared_bbox=list(annotation['bbox']),raster_bbox=list(raster_bbox),
        bbox_difference_raster_minus_declared=[a-b for a,b in zip(raster_bbox,annotation['bbox'])],
        metadata_equality_required=False)


def check_one_image(Image,ImageFile,np,masks,closed_ref,check,image_ref,image_record,annotations):
    """Check exactly one stored JPEG, preserving its original annotation groups.

    closed_ref(strict_ref)->absolute path must independently validate source bytes;
    check() is the injected original resource/abort controller. No discovery,
    resize, EXIF transpose, annotation repair, model or AP is performed.
    """
    _need(callable(closed_ref) and callable(check),'Inject original source/resource check callables')
    check()
    _need(ImageFile.LOAD_TRUNCATED_IMAGES is False,'Truncated-image tolerance is prohibited; global flag is not changed')
    _need(type(image_ref) is dict and set(image_ref)=={'path','bytes','sha256'}
          and type(image_ref['path']) is str and Path(image_ref['path']).is_absolute()
          and type(image_ref['bytes']) is int and image_ref['bytes']>=0
          and type(image_ref['sha256']) is str and len(image_ref['sha256'])==64
          and set(image_ref['sha256'])<=set('0123456789abcdef'),'Exact original JPEG source reference')
    _need(type(image_record) is dict and type(image_record['id']) is int
          and type(image_record['width']) is int and type(image_record['height']) is int
          and (image_record['width'],image_record['height'])==(800,800)
          and type(image_record['file_name']) is str
          and Path(image_record['file_name']).name==Path(image_ref['path']).name,
          'Original image identity and 800x800 declared canvas')
    _need(type(annotations) is list,'One original image annotation list; empty is legal')
    path=Path(closed_ref(image_ref))
    _need(path.resolve()==Path(image_ref['path']).resolve(),'Original closed JPEG path, no remapping')
    report=dict(schema='U42-single-image-pixel-mask-compatibility-pending-v1',status='PENDING',
        image_id=image_record['id'],image_ref=deepcopy(image_ref),declared_shape=[800,800],
        annotation_count=len(annotations),annotations=_annotation_rows(annotations,image_record['id']),pixel_decode_complete=False,
        EXIF_orientation=None,orientation_policy=None,EXIF_transform_applied=False,
        image_error=None,new_training=False,AP=False,model_executed=False,qualified=False,
        split_not_inferred_from_numeric_ids=True,formal_dataset_or_TEST_qualification=False,
        semantic_alignment_proven=False,population_qualified=False,
        scope='One stored JPEG and original per-annotation official COCO raster; no semantic/model claim')
    pixels=None;image=None;stage='JPEG_OPEN'
    try:
        with Image.open(path) as image:
            _need(image.format=='JPEG' and image.mode=='RGB' and tuple(image.getbands())==('R','G','B')
                  and tuple(image.size)==(800,800),'Exact stored 800x800 RGB JPEG; no implicit conversion')
            stage='EXIF_ORIENTATION'
            exif=image.getexif()
            orientation=exif.get(274)
            report['EXIF_orientation']=orientation if type(orientation) is int else None
            report['EXIF_orientation_value_type']=type(orientation).__name__
            if 274 not in exif:
                report['orientation_policy']='absent_no_transform_applied'
            else:
                _need(type(orientation) is int and orientation==1,'Declared non-1/invalid EXIF orientation rejected; no transpose')
                report['orientation_policy']='declared_1_no_transform_applied'
            stage='JPEG_PIXEL_LOAD'
            image.load()  # verify() is deliberately never a substitute for entropy/pixel decode.
            stage='DECODED_RGB_ARRAY'
            pixels=np.asarray(image)
            _need(pixels.dtype==np.uint8 and tuple(pixels.shape)==(800,800,3),
                  'Actual fully decoded original uint8 RGB800 pixels, without implicit conversion')
            report.update(pixel_decode_complete=True,decoded_shape=list(pixels.shape),
                decoded_dtype=str(pixels.dtype),decoded_bytes=int(pixels.nbytes),
                decoded_RGB_sha256=hashlib.sha256(pixels.tobytes(order='C')).hexdigest())
    except Exception as error:
        report['status']='FAILED';report['image_error']=_failure(error,stage)
    finally:
        pixels=None
        if image is not None:
            image.close()
        image=None
    check()
    attempted=passed=failed=0
    if report['pixel_decode_complete'] and report['image_error'] is None:
        for annotation,row in zip(annotations,report['annotations']):
            check()  # Controller exceptions propagate, never relabelled as corrupt annotations.
            if row['status']=='FAILED':
                continue  # Explicit retained ANNOTATION_METADATA failure, never dropped from output.
            bitmap=rles=rle=area_value=bbox_value=None
            stage='COCO_FR_PY_OBJECTS';attempted+=1
            try:
                # One original annotation is one GT. All its components are
                # merged together, including crowd=1; no component becomes GT.
                rles=masks.frPyObjects(annotation['segmentation'],800,800)
                _need(type(rles) is list and len(rles)==row['component_count'],
                      'One official RLE per original polygon component')
                stage='COCO_MERGE_COMPONENTS'
                rle=masks.merge(rles)
                _need(type(rle) is dict and list(rle['size'])==[800,800],
                      'Merged original instance retains the full original canvas')
                stage='COCO_DECODE'
                bitmap=masks.decode(rle)
                _need(bitmap.dtype==np.uint8 and tuple(bitmap.shape)==(800,800)
                      and bool(((bitmap==0)|(bitmap==1)).all()),'Official full-canvas binary uint8 mask')
                pixel_area=int(bitmap.sum(dtype=np.int64))
                area_value=float(masks.area(rle))
                _need(math.isfinite(area_value) and area_value.is_integer(),
                      'Official RLE area must be a finite scalar integer')
                row['raster_area']=pixel_area
                _need(int(area_value)==pixel_area,'Official RLE area and decoded-pixel sum agree')
                _need(pixel_area>0,'Empty raster is a retained annotation failure, never silently skipped')
                stage='COCO_RASTER_GEOMETRY'
                bbox_value=masks.toBbox(rle)
                _need(tuple(bbox_value.shape)==(4,),'Official raster bbox has four xywh values')
                bbox=[float(value) for value in bbox_value.tolist()]
                _need(all(math.isfinite(value) for value in bbox) and bbox[0]>=0 and bbox[1]>=0
                      and bbox[2]>0 and bbox[3]>0 and bbox[0]+bbox[2]<=800 and bbox[1]+bbox[3]<=800,
                      'Nonempty official raster bbox stays on the original pixel canvas')
                row.update(geometry_metadata_difference(annotation,pixel_area,bbox))
                row.update(status='PASSED',raster_shape=[800,800],raster_dtype=str(bitmap.dtype),
                    raster_sha256=hashlib.sha256(bitmap.tobytes(order='C')).hexdigest(),
                    RLE_area_matches_pixel_sum=True,error=None)
                passed+=1
            except Exception as error:
                row.update(status='FAILED',error=_failure(error,stage));failed+=1
            finally:
                bitmap=rles=rle=area_value=bbox_value=None
            check()
    metadata_failed=sum(row['error'] is not None and row['error']['stage']=='ANNOTATION_METADATA'
                        for row in report['annotations'])
    not_attempted=sum(row['status']=='NOT_ATTEMPTED' for row in report['annotations'])
    report.update(raster_attempted_annotations=attempted,raster_passed_annotations=passed,
        raster_failed_annotations=failed,annotation_metadata_failed_count=metadata_failed,
        not_attempted_annotations=not_attempted,raw_pixels_or_masks_returned=False,
        pixel_array_released=True,per_annotation_arrays_released=True)
    report['status']=('PASSED' if report['pixel_decode_complete'] and report['image_error'] is None
        and passed==len(annotations) and failed==metadata_failed==not_attempted==0 else 'FAILED')
    check()
    return report
