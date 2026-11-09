"""Exact paired action regret for the fixed four-add/four-delete budget."""
import torch


def iou_table(intersection,area_sum):
    """Count fixed COCO thresholds for each possible number of correct flips."""
    h=torch.arange(9,device=intersection.device)
    m=torch.arange(10,20,device=intersection.device)
    updated=intersection[:,None]+h[None,:]-4
    return ((20+m[None,None,:])*updated[:,:,None]>=m[None,None,:]*area_sum[:,None,None]).sum(-1)


def prefix_sums(scores,valid):
    chosen=torch.sort(scores.masked_fill(~valid,-torch.inf),dim=-1,descending=True,stable=True).values[...,:4]
    if chosen.shape[-1]<4:
        chosen=torch.nn.functional.pad(chosen,(0,4-chosen.shape[-1]),value=-torch.inf)
    zero=scores.new_zeros((*scores.shape[:-1],1))
    return torch.cat((zero,chosen.cumsum(dim=-1)),dim=-1)


def joint_scores(scores,labels,valid):
    good=prefix_sums(scores,labels&valid)
    bad=prefix_sums(scores,~labels&valid)
    sides=good+bad.flip(-1)
    return sides[:,0,:,None]+sides[:,1,None,:]


def loss_for_cost(score,cost,feasible,active):
    upper=(score+cost).amax((1,2))
    target=score.masked_fill(~(feasible&(cost==0)),-torch.inf).amax((1,2))
    upper=torch.where(active,upper,0)
    target=torch.where(active,target,0)
    return (upper-target).clamp_min(0)/8


def regrets(scores,labels,valid,utility):
    """Return threshold and equal-range linear-hit losses on identical rows.

    Scores and masks have shape [candidate, side, point]. Padded points are invalid.
    Utility has shape [candidate, 9], one nondecreasing threshold count per hit count.
    """
    actions=joint_scores(scores,labels,valid)
    feasible=torch.isfinite(actions)
    q=torch.arange(5,device=scores.device)
    h=q[:,None]+q[None,:]
    u=utility[:,h]
    most=u.masked_fill(~feasible,-1).amax((1,2))
    least=u.masked_fill(~feasible,11).amin((1,2))
    span=(most-least).clamp_min(0)
    active=span>0
    hmax=h.expand_as(actions).masked_fill(~feasible,-1).amax((1,2))
    hmin=h.expand_as(actions).masked_fill(~feasible,9).amin((1,2))
    threshold=(most[:,None,None]-u).to(scores.dtype)
    linear=(span[:,None,None]*(hmax[:,None,None]-h[None,:,:])).to(scores.dtype)
    linear=linear/(hmax-hmin).clamp_min(1)[:,None,None]
    return (loss_for_cost(actions,threshold,feasible,active),
            loss_for_cost(actions,linear,feasible,active))
