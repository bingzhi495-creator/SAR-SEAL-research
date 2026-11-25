"""Exact model expected IoU over the existing fixed-total signed mask menu."""
import numpy as np

ARMS = ("expectation_exact", "expectation_fixed")
TOL = 5e-12


def reciprocal(probabilities, area):
    """Reciprocal moment of the integer Poisson-binomial union increment."""
    deterministic = int(np.count_nonzero(probabilities == 1))
    random = probabilities[(probabilities > 0) & (probabilities < 1)]
    mass = np.ones(1,dtype=np.float64)
    for probability in random:
        mass = np.convolve(mass,(1-probability,probability))
    return float(mass @ (1/(area+deterministic+np.arange(len(mass)))))


def exact_utility(p,action,area,numerator):
    add,drop = action
    omitted = np.ones(len(p),dtype=bool)
    omitted[np.asarray(add,dtype=np.intp)] = False
    omitted_endpoint_p = np.concatenate((p[omitted,0,:],p[np.asarray(drop,dtype=np.intp),1,:]),axis=0)
    return numerator*np.array([reciprocal(col,area) for col in omitted_endpoint_p.T])


def moment_screen(numerator,variance,mean_union,area,max_sum):
    plug_in = numerator/mean_union
    common = numerator*variance/mean_union**2
    return plug_in+common/(area+max_sum), plug_in+common/area, common/area


def select(writer, p, static, k, area):
    p = np.asarray(p,dtype=np.float64)
    n = len(p)
    writer.need(p.shape==(n,2,2) and 0<k<=n<=area and area>2*k
                and np.isfinite(p).all() and np.all((0<=p)&(p<=1)),"Original finite signed endpoint scope")
    add, drop = p[:,0],p[:,1]
    var_add,var_drop = add*(1-add),drop*(1-drop)
    base_i = area-n+drop.sum(axis=0)
    static_i = base_i + writer.side_sum(add,(static[0],))[0] - writer.side_sum(drop,(static[1],))[0]
    ref_area = area-n+p.sum(axis=(0,1))
    static_u = exact_utility(p,static,area,static_i)
    rank_add,rank_drop = writer.ranking(add,True),writer.ranking(drop,False)
    quotas = sorted(range(max(0,2*k-n),min(n,2*k)+1),key=lambda a:(abs(a-k),a))
    largest_lower, fixed_lower, max_gap, summed_gap = -np.inf,-np.inf,0.,0.
    menu_actions = 0
    blocks = []
    for a in quotas:
        d = 2*k-a
        adds = writer.side_sets(rank_add,a,static[0])
        drops = writer.side_sets(rank_drop,d,static[1])
        i = base_i + writer.side_sum(add,adds)[:,None,:]-writer.side_sum(drop,drops)[None,:,:]
        edited = area+a-d
        mean_u = edited + ref_area-i
        variance = var_add.sum(axis=0)-writer.side_sum(var_add,adds)[:,None,:]+writer.side_sum(var_drop,drops)[None,:,:]
        variance = np.maximum(variance,0.)
        lo,hi,gap = moment_screen(i,variance,mean_u,edited,n-a+d)
        lower = (lo-static_u).min(axis=-1)
        upper = (hi-static_u).min(axis=-1)
        if a==k:
            fixed_lower = float(lower.max())
        largest_lower = max(largest_lower,float(lower.max()))
        max_gap = max(max_gap,float(gap.max()))
        summed_gap += float(gap.max(axis=-1).sum())
        blocks.append((a,edited,adds,drops,i,lower,upper))
        menu_actions += len(adds)*len(drops)

    winners = {}
    evaluated = 0
    for a,edited,adds,drops,i,lower,upper in blocks:
        could_win = upper>=largest_lower-TOL
        could_win_fixed = upper>=fixed_lower-TOL if a==k else np.zeros_like(could_win)
        for row,col in np.argwhere(could_win|could_win_fixed):
            action = (adds[row],drops[col])
            gain = exact_utility(p,action,edited,i[row,col])-static_u
            score = float(gain.min())
            writer.need(lower[row,col]-TOL<=score<=upper[row,col]+TOL,"Exact convolution inside moment bounds")
            for arm,admitted in zip(ARMS,(could_win[row,col],could_win_fixed[row,col]),strict=True):
                if admitted and (arm not in winners or score>winners[arm]["score"]):
                    winners[arm] = dict(action=action,score=score,iou_gain=gain.tolist(),
                                        intersection_gain=(i[row,col]-static_i).tolist())
            evaluated += 1
    writer.need(set(winners)==set(ARMS) and winners[ARMS[0]]["score"]+TOL>=winners[ARMS[1]]["score"]>=-TOL
                and 0<evaluated<=menu_actions,"Finite nested exact decision coverage")
    audit = dict(evaluated=evaluated,menu_actions=menu_actions,pruned=menu_actions-evaluated,
                 max_gap_bound=max_gap,sum_action_max_gap_bound=summed_gap)
    return winners,audit
