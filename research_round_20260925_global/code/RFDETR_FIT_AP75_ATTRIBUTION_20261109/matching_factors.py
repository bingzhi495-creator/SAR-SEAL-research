"""AP75 edge eligibility and IoU ordering for diagnostic COCO replay."""
import numpy as np

THRESHOLD=.75


def mix_edges(eligibility_source,priority_source):
    eligibility=np.asarray(eligibility_source)
    priority=np.asarray(priority_source)
    if eligibility.shape!=priority.shape:
        raise ValueError("Same detections and annotations required")
    if not eligibility.size:
        return eligibility.copy()
    if eligibility.ndim!=2 or not (np.isfinite(eligibility).all() and np.isfinite(priority).all()):
        raise ValueError("Finite detection by annotation matrices required")
    mixed=np.zeros(eligibility.shape,dtype=np.float64)
    for i,(allowed,score) in enumerate(zip(eligibility>=THRESHOLD,priority,strict=True)):
        if allowed.any():
            _,ranks=np.unique(score[allowed],return_inverse=True)
            mixed[i,allowed]=THRESHOLD+(1-THRESHOLD)*(ranks+1)/(score.size+1)
    return mixed
