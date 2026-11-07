"""Stock COCO segmentation AP on the frozen 111-image FIT selector holdout."""
from copy import deepcopy
from math import isfinite


HOLDOUT_IMAGES = 111
CATEGORY_ID = 1
MAX_DETS = (1, 10, 100)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def holdout_gt(integer_gt, image_ids):
    require(type(integer_gt) is dict and all(type(integer_gt.get(key)) is list
            for key in ('images', 'annotations', 'categories')), 'Expected a COCO GT document')
    categories = integer_gt['categories']
    require(len(categories) == 1 and type(categories[0].get('id')) is int
            and categories[0]['id'] == CATEGORY_ID, 'Expected only original ship category 1')
    requested = set(image_ids)
    images = {}
    for image in integer_gt['images']:
        image_id = image['id']
        if image_id not in requested:
            continue
        require(type(image_id) is int and image_id not in images
                and all(type(image[key]) is int and image[key] > 0 for key in ('height', 'width')),
                'Invalid or repeated holdout GT image')
        images[image_id] = image
    require(set(images) == requested, 'Every prescribed holdout image needs original GT metadata')
    annotations = []
    seen = set()
    for annotation in integer_gt['annotations']:
        image_id = annotation['image_id']
        if image_id not in requested:
            continue
        aid = annotation['id']
        require(type(image_id) is type(aid) is type(annotation['category_id']) is int
                and aid not in seen and aid > 0 and annotation['category_id'] == CATEGORY_ID
                and type(annotation.get('iscrowd')) is int and annotation['iscrowd'] in (0, 1)
                and 'segmentation' in annotation and 'area' in annotation,
                'Invalid or repeated original holdout annotation')
        seen.add(aid)
        annotations.append(annotation)
    # Keep source annotations and their order; COCOeval sets internal fields on a copy.
    return dict(info=integer_gt.get('info', {}), images=[images[i] for i in image_ids],
                categories=categories, annotations=annotations)


def prediction(row, image):
    require(type(row) is dict and set(row) == {'image_id', 'category_id', 'score', 'segmentation'},
            'Expected canonical mask result with no bbox')
    require(type(row['image_id']) is type(row['category_id']) is int and row['image_id'] == image['id']
            and row['category_id'] == CATEGORY_ID, 'Prediction image/category does not match the frozen population')
    score = row['score']
    require(type(score) in (int, float) and 0 <= score <= 1 and isfinite(score), 'Invalid frozen detector score')
    rle = row['segmentation']
    require(type(rle) is dict and set(rle) == {'size', 'counts'}, 'Expected a compressed mask RLE')
    size, counts = rle['size'], rle['counts']
    require(type(size) is list and len(size) == 2 and all(type(v) is int for v in size)
            and size == [image['height'], image['width']] and type(counts) is str and counts
            and all(48 <= ord(ch) <= 111 for ch in counts), 'Expected an original-sized compressed ASCII mask RLE')
    return dict(image_id=row['image_id'], category_id=row['category_id'], score=score,
                segmentation=dict(size=list(size), counts=counts))


def result_lists(original_by_image, edited_by_image, images):
    prescribed = {image['id'] for image in images}
    for population in (original_by_image, edited_by_image):
        require(type(population) is dict and all(type(key) is int for key in population)
                and set(population) == prescribed, 'Every holdout image must have an explicit prediction list')
    original, edited = [], []
    images_without_predictions = 0
    for image in images:
        rows = original_by_image[image['id']]
        changes = edited_by_image[image['id']]
        require(type(rows) is type(changes) is list and len(rows) == len(changes) <= MAX_DETS[-1],
                'Native prediction count or ordered population changed')
        images_without_predictions += not rows
        for row, change in zip(rows, changes):
            before = prediction(row, image)
            after = prediction(change, image)
            a, b = before['score'], after['score']
            require(type(a) is type(b) and (a.hex() == b.hex() if type(a) is float else a == b),
                    'Frozen score or prediction order changed')
            original.append(before)
            edited.append(after)
    return original, edited, images_without_predictions


def stock_stats(dataset, results, image_ids, coco_class, cocoeval_class):
    ground_truth = coco_class()
    ground_truth.dataset = deepcopy(dataset)
    ground_truth.createIndex()
    if results:
        detections = ground_truth.loadRes(results)
    else:
        # Stock loadRes indexes its first row. An indexed empty COCO result has the same semantics.
        detections = coco_class()
        detections.dataset = dict(images=deepcopy(dataset['images']),
                                  categories=deepcopy(dataset['categories']), annotations=[])
        detections.createIndex()
    evaluator = cocoeval_class(ground_truth, detections, 'segm')
    params = evaluator.params
    require(params.iouType == 'segm' and params.useCats == 1 and tuple(params.maxDets) == MAX_DETS,
            'Expected stock COCO segmentation defaults')
    params.imgIds = list(image_ids)
    params.catIds = [CATEGORY_ID]
    # Stock COCO sorts image IDs and uses stable mergesort for equal scores.
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    stats = [float(value) for value in evaluator.stats]
    require(len(stats) == 12 and all(isfinite(value) and (value == -1 or 0 <= value <= 1) for value in stats),
            'Expected all 12 raw COCO summary values')
    return stats


def evaluate_holdout(integer_gt, original_by_image, edited_by_image, image_ids, *,
                     coco_class, cocoeval_class):
    """Evaluate two complete, row-aligned holdout populations with official pycocotools.

    The caller owns selection identity, edit-budget qualification, and runtime pins.
    Per-image input row order is preserved. An all-zero RLE is still a detection.
    """
    require(type(image_ids) in (list, tuple) and len(image_ids) == HOLDOUT_IMAGES
            and all(type(i) is int for i in image_ids) and len(set(image_ids)) == HOLDOUT_IMAGES,
            'Expected exactly 111 distinct prescribed holdout image IDs')
    dataset = holdout_gt(integer_gt, image_ids)
    original, edited, empty_images = result_lists(original_by_image, edited_by_image, dataset['images'])
    result = dict(image_count=HOLDOUT_IMAGES, category_id=CATEGORY_ID, iou_type='segm',
                  max_dets=list(MAX_DETS))
    for arm, predictions in (('original', original), ('edited', edited)):
        stats = stock_stats(dataset, predictions, image_ids, coco_class, cocoeval_class)
        result[arm] = dict(AP=stats[0], AP75=stats[2], stats=stats, prediction_count=len(predictions),
                           images_without_predictions=empty_images)
    result['delta'] = {
        key: None if -1 in (result['original'][key], result['edited'][key])
        else result['edited'][key] - result['original'][key]
        for key in ('AP', 'AP75')
    }
    return result
