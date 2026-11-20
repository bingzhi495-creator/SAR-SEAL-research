"""CORE36 native RF posterior, published point features, sparse exact readouts.

No data/model loader, source selection by labels, training, or scoring.
Numerical modules and original published math are explicit caller dependencies.
"""
from types import SimpleNamespace


def need(ok, message):
    if not ok:
        raise ValueError(message)


def select_images(rows, count=4):
    """Select original manifest objects by decoded RGB SHA then source image ID."""
    need(type(count) is int and 1 <= count <= 4 and len(rows) >= count,
         'Nonempty bounded image selection')
    for row in rows:
        digest = row['decoded_rgb_sha256']
        need(type(digest) is str and len(digest) == 64
             and all(ch in '0123456789abcdef' for ch in digest)
             and type(row['image_id']) is int, 'Original digest and integer source identity')
    need(len({row['image_id'] for row in rows}) == len(rows), 'No duplicate source IDs')
    return sorted(rows, key=lambda row: (row['decoded_rgb_sha256'],row['image_id']))[:count]


def render_posterior(torch, np, posterior_adapter, raw144, stock800):
    """One original query, CPU144 logit -> author native800 parity -> FP32 p."""
    need(raw144.shape==(144,144) and raw144.dtype==np.float32
         and stock800.shape==(800,800) and stock800.dtype==np.bool_, 'Exact RF native query shapes')
    # Independent tiny clones prevent read-only/mmap NumPy alias surprises.
    raw = torch.from_numpy(np.array(raw144,dtype=np.float32,copy=True,order='C'))[None]
    stock = torch.from_numpy(np.array(stock800,dtype=np.bool_,copy=True,order='C'))[None]
    probability, _ = posterior_adapter.native_probability(torch,raw,(800,800),stock)
    return probability[0].numpy()


def extract_candidate(np, pair_math, rank_math, stock, box_xyxy, score, probability, image_features):
    """Reuse original fourteen features, with native RF FP32 posterior sampling.

    No target/match is accepted. Caller pins the two original source modules.
    Original paired_points is explicitly an 800-canvas implementation.
    """
    need(stock.shape == (800,800) and stock.dtype == np.bool_, 'Native bool 800 canvas')
    need(probability.shape == stock.shape and probability.dtype == np.float32
         and np.isfinite(probability).all() and np.all((probability >= 0)&(probability <= 1)),
         'Native finite FP32 rendered posterior, no half cast')
    need(image_features.shape == (5,800,800) and image_features.dtype == np.float32
         and np.isfinite(image_features).all(), 'Original five image features')
    box = np.asarray(box_xyxy, dtype=np.float32)
    need(box.shape == (4,) and np.isfinite(box).all() and box[2]>box[0] and box[3]>box[1]
         and np.isfinite(score) and 0 <= score <= 1, 'Finite original positive author box and score')
    area = int(np.count_nonzero(stock))
    need(32 <= area < 1024, 'Caller must preserve ineligible candidates unchanged')

    def sample_probability(_unused, coords, width):
        need(width == 800 and np.all((coords >= 0)&(coords < 800*800)), 'Original native point indices')
        return probability.reshape(-1)[coords]

    adapted = SimpleNamespace(editable_pools=rank_math.editable_pools,
                              sample_probability=sample_probability)
    result = pair_math.paired_points(stock,box,area,score,probability,image_features,adapted)
    if result is None:
        return None
    coordinates, features, layers, discord_ge, half = result
    need(features.shape == (len(coordinates),2,14) and features.dtype == np.float32
         and np.isfinite(features).all(), 'Exact original finite fourteen features')
    # p=.5 is not a replacement definition of RF strict-logit stock. Preserve
    # original diagnostic, and separately name >.5 comparison without rethresholding.
    signs = np.asarray([False,True])[None,:]
    diagnostic = dict(area=area, pairs=len(coordinates), posterior_half_points=int(half),
                      posterior_ge_half_discordance=int(discord_ge),
                      posterior_gt_half_discordance=int(np.count_nonzero((features[:,:,0]>.5)!=signs)),
                      probability_dtype='float32', probability_resampled=False)
    return dict(coordinates=coordinates,features=features,layers=layers,diagnostics=diagnostic)


def readouts_sparse(np, stock, coordinates, qS, qL, qNative, cap):
    """Return eight O(K) edit plans, not eight O(HW) mask copies.

    Fixed/free ties retain original pair ordinals. LW joint ties use native
    row-major pixel index. Pair-policy actions are swaps; LW actions are flips.
    nonpositive_flips is comparable across all rows, actions are explicitly typed.
    """
    need(stock.ndim == 2 and stock.dtype == np.bool_, 'Native 2D bool raster')
    coords = np.asarray(coordinates)
    need(coords.ndim == 2 and coords.shape[1] == 2 and coords.dtype.kind in 'iu'
         and coords.size <= 64 and stock.size <= np.iinfo(np.int32).max,
         'At most 32 aligned integer endpoint pairs')
    coords = coords.astype(np.int32,copy=False)
    k = len(coords)
    need(np.all((coords>=0)&(coords<stock.size)) and len(np.unique(coords)) == coords.size,
         'Disjoint bounded original endpoint identities')
    flat = stock.reshape(-1)
    need(not flat[coords[:,0]].any() and flat[coords[:,1]].all(), 'Native add/drop orientation')
    need(type(cap) is int and 0 <= cap <= min(8,int(np.count_nonzero(stock))//10,k),
         'Exact candidate-area action cap')
    qs = []
    for q in (qS,qL,qNative):
        q = np.asarray(q)
        need(q.shape == (k,2) and q.dtype == np.float32 and np.isfinite(q).all()
             and np.all((q>=0)&(q<=1)), 'Aligned FP32 posterior matrix')
        qs.append(q)
    qS,qL,qNative = qs
    ordinals = np.arange(k,dtype=np.int32)

    def fixed(q,count=None):
        gain = q[:,0]-q[:,1]
        order = np.lexsort((ordinals,-gain))
        chosen = order[gain[order]>0][:cap] if count is None else order[:count]
        return np.column_stack((chosen,chosen)).astype(np.int32)

    def free(q,count=None):
        add = np.lexsort((ordinals,-q[:,0]))[:cap]
        drop = np.lexsort((ordinals,q[:,1]))[:cap]
        marginal = q[add,0]-q[drop,1]
        count = int(np.count_nonzero(marginal>0)) if count is None else count
        return np.column_stack((add[:count],drop[:count])).astype(np.int32)

    sf,su = fixed(qS),free(qS)
    kf,ks = len(sf),len(su)
    choices = {'N':(qS,np.empty((0,2),dtype=np.int32)), 'SF':(qS,sf),
               'SU=F':(qS,free(qS,kf)), 'SU':(qS,su),
               'LF=S':(qL,fixed(qL,ks)), 'LU=S':(qL,free(qL,ks)),
               'NP=S':(qNative,free(qNative,ks))}
    result = {}

    def summarize(flipped,flip_gain,action_gain,kind):
        add = int(np.count_nonzero(~flat[flipped]))
        drop = len(flipped)-add
        return dict(flipped=flipped.astype(np.int32,copy=False),edits=len(flipped),adds=add,drops=drop,
                    area_delta=add-drop,action_kind=kind,actions=len(action_gain),
                    nonpositive_actions=int(np.count_nonzero(action_gain<=0)),
                    nonpositive_flips=int(np.count_nonzero(flip_gain<=0)),
                    predicted_gain_sum=float(np.sum(flip_gain,dtype=np.float64)))

    for name,(q,actions) in choices.items():
        add,drop = actions[:,0],actions[:,1]
        flipped = np.column_stack((coords[add,0],coords[drop,1])).reshape(-1)
        flip_gain = np.column_stack((q[add,0]-np.float32(.5),np.float32(.5)-q[drop,1])).reshape(-1)
        result[name] = summarize(flipped,flip_gain,q[add,0]-q[drop,1],'swap')
    gains = np.column_stack((qL[:,0]-np.float32(.5),np.float32(.5)-qL[:,1])).reshape(-1)
    endpoints = coords.reshape(-1)
    selected = np.lexsort((endpoints,-gains))[:2*ks]
    result['LW=S'] = summarize(endpoints[selected],gains[selected],gains[selected],'flip')
    need(result['SF']['edits'] == result['SU=F']['edits'] == 2*kf
         and all(result[name]['edits']==2*ks for name in ('SU','LF=S','LU=S','LW=S','NP=S'))
         and all(row['area_delta']==0 for name,row in result.items() if name!='LW=S'),
         'Exact realized edits and required area conservation')
    return result


def apply_sparse(np, stock, edit):
    """Materialize only one policy mask, independently verify XOR and area."""
    need(stock.ndim==2 and stock.dtype==np.bool_, 'Original bool mask')
    indices = np.asarray(edit['flipped'])
    need(indices.ndim==1 and indices.dtype.kind in 'iu' and len(np.unique(indices))==len(indices)
         and np.all((indices>=0)&(indices<stock.size)), 'Unique bounded sparse flips')
    result = stock.copy(order='C')
    result.reshape(-1)[indices] = ~stock.reshape(-1)[indices]
    adds = int(np.count_nonzero(result & ~stock))
    drops = int(np.count_nonzero(stock & ~result))
    need(adds==edit['adds'] and drops==edit['drops'] and adds+drops==edit['edits']==len(indices)
         and adds-drops==edit['area_delta']
         and int(np.count_nonzero(result))-int(np.count_nonzero(stock))==edit['area_delta'],
         'Actual mask XOR, sign counts and area match sparse metadata')
    return result
