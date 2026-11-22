"""Exact single-candidate counterfactuals for frozen-score, all-area mask AP."""

import numpy as np


IOU_LEVELS = np.linspace(.50,.95,10)
RECALL_LEVELS = np.linspace(0,1,101)


def check_inputs(iou, crowd, thresholds):
    if (iou.ndim != 2 or iou.shape[1] != len(crowd) or crowd.ndim != 1
            or crowd.dtype != np.bool_ or not np.isfinite(iou).all()
            or ((iou < 0) | (iou > 1)).any() or not np.array_equal(thresholds, IOU_LEVELS)):
        raise ValueError("Original COCO all-area mask overlaps and crowd flags required")


def truth_priority(iou, crowd):
    # Nonignored positives precede crowds. An equal-IoU tie prefers the later
    # ground-truth annotation in its original class order, as in COCOeval.
    return np.lexsort((-np.arange(len(crowd)), -iou, crowd))


def match_one(iou, crowd, order, occupied, threshold):
    for k in order:
        if iou[k] >= threshold:
            if crowd[k]:
                return -(int(k)+1), occupied
            bit = 1 << int(k)
            if not occupied & bit:
                return int(k)+1, occupied | bit
    return 0, occupied


class FrozenImage:
    """Hold the baseline matcher state in native confidence order for one image."""

    def __init__(self, iou, crowd):
        self.iou = np.array(iou, dtype=np.float64, copy=True)
        self.crowd = np.array(crowd, copy=True)
        check_inputs(self.iou,self.crowd,IOU_LEVELS)
        self.order = [truth_priority(row,self.crowd) for row in self.iou]
        self.baseline = np.zeros((len(IOU_LEVELS),len(iou)),dtype=np.int64)
        self.prefix = []
        for j,threshold in enumerate(IOU_LEVELS):
            occupied = 0
            states = []
            for i,row in enumerate(self.iou):
                states.append(occupied)
                self.baseline[j,i],occupied = match_one(row,self.crowd,self.order[i],occupied,threshold)
            self.prefix.append(tuple(states))

    def substitute(self, position, candidate):
        """Replace one overlap row and recompute only the affected greedy suffix."""
        iou = np.asarray(candidate,dtype=np.float64)
        if not 0 <= position < len(self.iou) or iou.shape != self.crowd.shape:
            raise ValueError("Original confidence position and all original ground truths required")
        check_inputs(iou[None],self.crowd,IOU_LEVELS)
        order = truth_priority(iou,self.crowd)
        result = self.baseline.copy()
        for j,threshold in enumerate(IOU_LEVELS):
            value,occupied = match_one(iou,self.crowd,order,self.prefix[j][position],threshold)
            if value == self.baseline[j,position]:
                continue
            result[j,position] = value
            for i in range(position+1,len(self.iou)):
                if occupied == self.prefix[j][i]:
                    break
                result[j,i],occupied = match_one(self.iou[i],self.crowd,self.order[i],occupied,threshold)
        return result


def ap_curve(statuses, positives):
    """Return COCO 101-grid AP for ten thresholds from globally score-sorted rows."""
    rows = np.asarray(statuses)
    if rows.ndim != 2 or rows.shape[0] != len(IOU_LEVELS) or type(positives) is not int or positives <= 0:
        raise ValueError("Ten-threshold globally ordered rows and positive denominator required")
    ap = np.zeros(len(IOU_LEVELS),dtype=np.float64)
    for j,row in enumerate(rows):
        tp = np.cumsum(row > 0,dtype=np.float64)
        fp = np.cumsum(row == 0,dtype=np.float64)
        if not len(tp):
            continue
        recall = tp / positives
        precision = tp / (tp+fp+np.spacing(1))
        if recall[-1] > 1:
            raise ValueError("A nonignored truth was matched against multiple candidates")
        best = np.maximum.accumulate(precision[::-1])[::-1]
        position = np.searchsorted(recall,RECALL_LEVELS,side="left")
        valid = position < len(row)
        ap[j] = best[position[valid]].sum()/len(RECALL_LEVELS)
    return ap


def marginal_ap(baseline, global_indices, updated_image_rows, positives):
    """Keep confidence order and replace only one image's candidate outcomes."""
    base = np.asarray(baseline)
    indices = np.asarray(global_indices)
    updated = np.asarray(updated_image_rows)
    if (indices.ndim != 1 or indices.dtype.kind not in "iu" or len(indices) != len(set(indices))
            or np.any((indices < 0) | (indices >= base.shape[-1]))
            or updated.shape != (len(IOU_LEVELS),len(indices))):
        raise ValueError("Unique global positions aligned with the image's frozen order required")
    alternative = base.copy()
    alternative[:, indices] = updated
    return ap_curve(alternative,positives)-ap_curve(base,positives)
