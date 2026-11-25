"""Keep the endpoint choices fixed while comparing signed quota scores."""
import numpy as np


def panel(writer, p, static, k, area):
    p = np.asarray(p, dtype=np.float64)
    add, drop = p[:, 0], p[:, 1]
    n = len(p)
    base_i = area - n + drop.sum(axis=0)
    g = area - n + p.sum(axis=(0, 1))
    static_i = base_i + writer.side_sum(add, (static[0],))[0] - writer.side_sum(drop, (static[1],))[0]
    static_utility = writer.iou(static_i, area, g)
    best = []
    orders_add = writer.ranking(add, True)
    orders_drop = writer.ranking(drop, False)
    legal = sorted(range(max(0,2*k-n), min(n,2*k)+1), key=lambda a: (abs(a-k),a))
    for a in legal:
        d = 2*k-a
        adds = writer.side_sets(orders_add,a,static[0])
        drops = writer.side_sets(orders_drop,d,static[1])
        i = base_i + writer.side_sum(add,adds)[:,None,:] - writer.side_sum(drop,drops)[None,:,:]
        gain = writer.iou(i,area+a-d,g)-static_utility
        scores = gain.min(axis=-1)
        row,col = np.unravel_index(int(scores.argmax()),scores.shape)
        best.append(dict(adds=a, score=float(scores[row,col]), action=(adds[row],drops[col]),
                         iou_gain=gain[row,col].tolist(), intersection_gain=(i[row,col]-static_i).tolist()))
    if best[0]["adds"] != k:
        raise ValueError("The original fixed split must be feasible")
    return best


def features(rows,k,area,reverse):
    raw = np.array([row["score"]-rows[0]["score"] for row in rows])
    q = np.array([(row["adds"]-k)/k for row in rows])
    x = np.column_stack((raw,q,np.abs(q),2*k*q/area))
    if reverse:
        for mask in (q<0,q>0):
            x[mask,0] = raw[mask][::-1]
    return raw,x


def fit(x,residual,weight):
    weight = np.asarray(weight,dtype=np.float64)
    weight = weight / weight.sum()
    scale = np.sqrt(weight @ (x*x))
    scale = np.where(scale>0,scale,1.)
    z = x / scale
    gram = z.T @ (weight[:,None]*z)
    rhs = z.T @ (weight*residual)
    coef = np.linalg.solve(gram+0.02*np.eye(x.shape[1]),rhs)
    return dict(scale=scale.tolist(),coefficients=coef.tolist())


def choose(rows,raw,x,model):
    score = raw+(x/np.asarray(model["scale"])) @ model["coefficients"]
    if score[0]!=0 or not np.isfinite(score).all():
        raise ValueError("Expected finite, zero-anchored quota scores")
    return rows[int(score.argmax())]


def true_iou(rows,area,reference_area,native_hit,labels):
    values = []
    for row in rows:
        adds,drops = row["action"]
        hit = native_hit + labels[list(adds),0].sum(axis=0) - labels[list(drops),1].sum(axis=0)
        values.append(hit/(area+len(adds)-len(drops)+reference_area-hit))
    iou = np.asarray(values)
    return (iou-iou[0]).min(axis=1)
