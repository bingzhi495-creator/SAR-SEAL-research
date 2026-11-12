"""Exact native-pixel deletion ordering and continuous fixed-match geometry."""
import numpy as np


def ordered_deletions(coords, probabilities, count):
    order = np.lexsort((coords, probabilities))
    return coords[order[:count]].tolist()


def continuous_after_drop(area, truth_area, original_intersection, fractions):
    """Return IoU, FP and FN in percentage points for every deletion prefix."""
    removed = np.arange(len(fractions)+1,dtype=np.float64)
    predicted = area-removed
    intersection = original_intersection-np.concatenate(([0.],np.cumsum(fractions,dtype=np.float64)))
    cap = np.minimum(predicted,truth_area)
    if (predicted<=0).any() or (intersection < -1e-8).any() or (intersection>cap+1e-8).any():
        raise ValueError("Exact valid continuous deletion geometry")
    intersection = np.clip(intersection,0,cap)
    return 100*np.column_stack((intersection/(predicted+truth_area-intersection),
                                (predicted-intersection)/truth_area,(truth_area-intersection)/truth_area))


def ridge(features, targets, weights, penalty=.02):
    weights = weights/weights.sum()
    center = weights @ features
    spread = np.sqrt(weights @ np.square(features-center))
    spread[spread==0.] = 1.
    design = (features-center)/spread
    intercept = float(weights @ targets)
    weighted_design = design * weights[:,None]
    lhs = design.T @ weighted_design + penalty*np.eye(design.shape[1])
    rhs = weighted_design.T @ (targets-intercept)
    slopes = np.linalg.solve(lhs,rhs)

    def predict(x):
        return intercept + ((x-center)/spread) @ slopes
    return predict


def choose_prefix(predicted_y, area):
    actions = np.arange(1,len(predicted_y)+1,dtype=np.float64)
    gain = np.concatenate(([0.],(actions/area)*predicted_y))
    return int(np.argmax(gain))