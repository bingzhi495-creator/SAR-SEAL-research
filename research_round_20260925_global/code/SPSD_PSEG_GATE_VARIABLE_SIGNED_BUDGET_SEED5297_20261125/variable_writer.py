"""Finite variable-allocation search at a strict original 2k signed edit count."""
import operator

import numpy as np


ARMS = ("fixed_iou", "variable_iou", "variable_mean_iou", "variable_raw_min", "near_iou")


def need(flag, reason):
    if not flag:
        raise ValueError(reason)


def ranking(probabilities, largest):
    mixtures = np.arange(11,dtype=np.float64)[:,None]
    values = mixtures*probabilities[None,:,0] + (10-mixtures)*probabilities[None,:,1]
    return np.argsort(-values if largest else values, axis=1, kind="stable")


def side_sets(orders, count, old):
    choices = [old] if count == len(old) else []
    choices.extend(tuple(s) for s in np.sort(orders[:,:count],axis=1).tolist())
    return tuple(dict.fromkeys(choices))


def side_sum(values, sets):
    return values[np.asarray(sets,dtype=np.intp)].sum(axis=1)


def iou(intersection, mask_area, reference_area):
    union = mask_area + reference_area - intersection
    need(np.all(union > 0) and np.all(intersection >= -1e-10)
         and np.all(intersection <= np.minimum(mask_area,reference_area)+1e-10),
         "Valid estimated reference and predicted intersection")
    return intersection / union


def select(posterior, static, k, area):
    """Return fixed and variable maximizing actions plus gate diagnostics."""
    k, area = operator.index(k), operator.index(area)
    p = np.asarray(posterior,dtype=np.float64)
    n = len(p)
    need(p.shape == (n,2,2) and 0 < k <= n <= area and area > 2*k
         and np.isfinite(p).all() and np.all((0 <= p) & (p <= 1)), "Original supported candidate geometry and probabilities")
    add, drop = p[:,0,:], p[:,1,:]
    incumbent = tuple(tuple(sorted(map(operator.index,side))) for side in static)
    need(len(incumbent)==2 and all(len(side)==len(set(side))==k and all(0 <= index < n for index in side)
         for side in incumbent), "Original exact two-sided incumbent")
    outside = area-n
    estimated_g = outside + p.sum(axis=(0,1))
    base_i = outside + drop.sum(axis=0)
    static_i = base_i + side_sum(add,(incumbent[0],))[0] - side_sum(drop,(incumbent[1],))[0]
    static_iou = iou(static_i,area,estimated_g)
    ranked_add, ranked_drop = ranking(add,True),ranking(drop,False)
    offsets = sorted(range(max(0,2*k-n),min(n,2*k)+1),key=lambda a:(abs(a-k),a))
    winners = {}
    evaluated = 0
    for a in offsets:
        d = 2*k-a
        adds = side_sets(ranked_add,a,incumbent[0])
        drops = side_sets(ranked_drop,d,incumbent[1])
        intersection = base_i + side_sum(add,adds)[:,None,:] - side_sum(drop,drops)[None,:,:]
        change = iou(intersection,area+a-d,estimated_g) - static_iou
        intersection_change = intersection - static_i
        eligible = dict(variable_iou=change.min(axis=-1),
                        variable_mean_iou=change.mean(axis=-1),
                        variable_raw_min=intersection_change.min(axis=-1))
        if a == k:
            eligible["fixed_iou"] = eligible["variable_iou"]
        if abs(a-k)<=1:
            eligible["near_iou"] = eligible["variable_iou"]
        for arm, objective in eligible.items():
            index = np.unravel_index(int(objective.argmax()),objective.shape)
            score = float(objective[index])
            if arm not in winners or score>winners[arm]["score"]:
                winners[arm] = dict(action=(adds[index[0]],drops[index[1]]), score=score,
                    iou_gain=change[index].tolist(), intersection_gain=intersection_change[index].tolist())
        evaluated += len(adds)*len(drops)
    need(tuple(sorted(winners)) == tuple(sorted(ARMS)) and all(winners[k]["score"] >= -1e-12 for k in ARMS)
         and winners["variable_iou"]["score"]>=winners["near_iou"]["score"]>=winners["fixed_iou"]["score"],
         "Full enumerated menu dominates each subset including static")
    winners["menu_actions"] = evaluated
    return winners


def write_mask(native_plane, coordinates, action, quota, codec):
    state = np.asarray(native_plane,dtype=bool,order="C")
    coords = np.asarray(coordinates)
    n = len(coords)
    need(coords.shape == (n,2) and coords.dtype.kind in "iu"
         and np.all((0<=coords)&(coords<state.size)) and len(set(coords.ravel().tolist())) == 2*n
         and all(len(side)==len(set(side)) and all(0 <= v < n for v in side) for side in action)
         and sum(map(len,action)) == 2*quota, "Disjoint exact support and total quota")
    flat = state.ravel(order="C")
    need(not flat[coords[:,0]].any() and flat[coords[:,1]].all(), "Original signed support membership")
    out = state.copy(order="C")
    result = out.ravel(order="C")
    result[coords[np.asarray(action[0],dtype=np.intp),0]] = True
    result[coords[np.asarray(action[1],dtype=np.intp),1]] = False
    observed = int(np.count_nonzero(result&~flat)), int(np.count_nonzero(flat&~result))
    need(observed == tuple(map(len,action)), "Exact realized native edits per sign")
    rle = codec.encode(np.asfortranarray(out,dtype=np.uint8))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle, observed
