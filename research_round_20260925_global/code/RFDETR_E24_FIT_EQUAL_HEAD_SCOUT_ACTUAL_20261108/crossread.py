"""GT-conditioned cross-readout of already materialized FIT scout predictions."""


ROW_FIELDS = {'image_id', 'category_id', 'score', 'segmentation'}
STATUS = ('matched', 'unmatched', 'ignored')
ARMS = ('matched_only', 'unmatched_only')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_row(row, image_id):
    require(type(row) is dict and set(row) == ROW_FIELDS, 'Expected a canonical COCO mask row')
    require(type(row['image_id']) is type(row['category_id']) is int
            and row['image_id'] == image_id and row['category_id'] == 1,
            'Wrong frozen image or category')
    score = row['score']
    require(type(score) in (int, float) and 0 <= score <= 1, 'Invalid frozen detector score')
    rle = row['segmentation']
    require(type(rle) is dict and set(rle) == {'size', 'counts'}, 'Expected whole compressed RLE')
    size, counts = rle['size'], rle['counts']
    require(type(size) is list and len(size) == 2 and all(type(n) is int and n > 0 for n in size)
            and type(counts) is str and counts and counts.isascii(), 'Invalid original-grid RLE')
    return rle


def build_crossread(original_by_image, edited_by_image, match_status_by_image, image_ids, *,
                    require_no_ignored=False):
    """Swap whole RLEs by original match status while retaining every native row.

    The caller owns original and edited row identity, edit-budget verification, and
    the exact source match assignment. This helper does not rematch or score.
    """
    require(type(image_ids) in (list, tuple) and all(type(i) is int for i in image_ids)
            and len(set(image_ids)) == len(image_ids), 'Expected distinct prescribed image IDs')
    require(type(require_no_ignored) is bool, 'Expected explicit boolean ignored policy')
    ids = set(image_ids)
    for population in (original_by_image, edited_by_image, match_status_by_image):
        require(type(population) is dict and all(type(i) is int for i in population)
                and set(population) == ids, 'All inputs must cover exactly the prescribed images')

    output = {arm: {} for arm in ARMS}
    counts = dict.fromkeys(STATUS, 0)
    replacements = dict.fromkeys(ARMS, 0)
    for image_id in image_ids:
        before, after, statuses = (values[image_id] for values in
                                   (original_by_image, edited_by_image, match_status_by_image))
        require(type(before) is type(after) is type(statuses) is list
                and len(before) == len(after) == len(statuses) <= 100,
                'Prediction and match-status rows must stay aligned')
        for arm in ARMS:
            output[arm][image_id] = []
        original_size = None
        for original, edited, status in zip(before, after, statuses):
            require(type(status) is str and status in STATUS, 'Unknown original match status')
            require(not require_no_ignored or status != 'ignored', 'Actual cohort must have no ignored rows')
            base, edit = verify_row(original, image_id), verify_row(edited, image_id)
            if original_size is None:
                original_size = base['size']
            require(base['size'] == edit['size'] == original_size, 'Original mask grid changed')
            a, b = original['score'], edited['score']
            require(type(a) is type(b) and (a.hex() == b.hex() if type(a) is float else a == b),
                    'Frozen detector score or candidate order changed')
            counts[status] += 1
            for arm, admitted_status in zip(ARMS, STATUS[:2]):
                rle = edit if status == admitted_status else base
                replacements[arm] += rle != base
                output[arm][image_id].append(
                    dict(original, segmentation=dict(size=list(rle['size']), counts=rle['counts'])))
    return dict(**output, summary=dict(image_count=len(image_ids), candidate_count=sum(counts.values()),
                                       status_counts=counts, rle_replacements=replacements,
                                       gt_conditioned=True))
