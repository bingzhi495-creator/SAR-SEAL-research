"""RF-DETR native CPU posterior and matched-count readouts; no model loader."""
from copy import deepcopy
from math import isfinite


def need(ok, message):
    if not ok:
        raise ValueError(message)


def native_probability(torch, logits, original_hw, stock=None):
    """Original author bilinear logit rendering; never interpolate probabilities."""
    need(logits.device.type == 'cpu' and logits.dtype == torch.float32
         and logits.ndim == 3 and all(n > 0 for n in logits.shape)
         and bool(torch.isfinite(logits).all()), 'Finite CPU FP32 N,H,W logits')
    need(len(original_hw) == 2 and all(type(n) is int and n > 0 for n in original_hw),
         'Positive original integer H,W')
    with torch.no_grad():
        rendered = torch.nn.functional.interpolate(logits[:, None], size=original_hw,
                                                   mode='bilinear', align_corners=False)[:, 0]
        mask = rendered > 0
        if stock is not None:
            need(stock.dtype == torch.bool and stock.device.type == 'cpu'
                 and torch.equal(mask, stock), 'Exact native stock parity')
        return rendered.sigmoid(), mask


def replace_segmentation(record, segmentation):
    """Keep original class, score, rank, query and all other fields verbatim."""
    need('segmentation' in record, 'Original segmentation field required')
    copied = deepcopy(record)
    copied['segmentation'] = deepcopy(segmentation)
    return copied


def readouts(stock, pairs, q_small, q_large, q_native, cap):
    """Flat native boolean raster and (add,drop) indices in frozen pair order.

    Each q is a same-order sequence of (add posterior, drop posterior).
    Fixed/independent ties use original pair order, as in the published D-FINE
    pair_math.choose_pairs/free_pairing.ranked_endpoints. LW uses native pixel
    index for its joint list and removes add/drop quotas. No label is an input.
    """
    need(type(cap) is int and 0 <= cap <= min(8, len(pairs)), 'Fixed bounded cap')
    need(all(type(v) is bool for v in stock), 'Flat native bool raster')
    endpoints = [v for pair in pairs for v in pair]
    need(all(len(p) == 2 for p in pairs) and len(set(endpoints)) == len(endpoints)
         and all(type(i) is int and 0 <= i < len(stock) for i in endpoints), 'Disjoint valid endpoints')
    need(all(not stock[a] and stock[d] for a, d in pairs), 'Original add/drop orientation')
    for q in (q_small, q_large, q_native):
        need(len(q) == len(pairs) and all(len(row) == 2 for row in q)
             and all(isfinite(v) and 0 <= v <= 1 for row in q for v in row), 'Finite aligned posteriors')

    def fixed(q, count=None):
        order = sorted(range(len(pairs)), key=lambda j: (-(q[j][0]-q[j][1]), j))
        chosen = ([j for j in order if q[j][0] > q[j][1]][:cap]
                  if count is None else order[:count])
        return [(pairs[j][0], pairs[j][1]) for j in chosen]

    def free(q, count=None):
        adds = sorted(range(len(pairs)), key=lambda j: (-q[j][0], j))
        drops = sorted(range(len(pairs)), key=lambda j: (q[j][1], j))
        rows = [(pairs[a][0], pairs[d][1], q[a][0]-q[d][1]) for a,d in zip(adds,drops)]
        chosen = ([row for row in rows if row[2] > 0][:cap] if count is None else rows[:count])
        return [(a,d) for a,d,_ in chosen]

    sf, su = fixed(q_small), free(q_small)
    kf, ks = len(sf), len(su)
    choices = {'N': [], 'SF': sf, 'SU=F': free(q_small,kf), 'SU': su,
               'LF=S': fixed(q_large,ks), 'LU=S': free(q_large,ks),
               'NP=S': free(q_native,ks)}
    flat = {key: [i for pair in selected for i in pair] for key,selected in choices.items()}
    gains = {index: (q_large[j][side]-.5 if side == 0 else .5-q_large[j][side])
             for j,pair in enumerate(pairs) for side,index in enumerate(pair)}
    flat['LW=S'] = sorted(endpoints, key=lambda index: (-gains[index], index))[:2*ks]
    out = {}
    for key, indices in flat.items():
        mask = list(stock)
        for index in indices:
            mask[index] = not mask[index]
        adds = sum(not stock[index] for index in indices)
        drops = len(indices)-adds
        out[key] = dict(mask=mask, flipped=indices, edits=len(indices), adds=adds,
                        drops=drops, area_delta=adds-drops)
        if key == 'LW=S':
            out[key]['nonpositive_flips'] = sum(gains[index] <= 0 for index in indices)
    need(out['SU=F']['edits'] == out['SF']['edits'] and
         all(out[name]['edits'] == 2*ks for name in ('SU','LF=S','LU=S','LW=S','NP=S')),
         'Exact per-candidate realized counts')
    return out
