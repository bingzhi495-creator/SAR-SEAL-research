"""FIT-only global native-logit offset: pure injected-library mathematics.

No top-level NumPy/Torch imports, files, scientific manifests or execution entry.
Default fit inputs are caller-qualified FIT[1661,1024] FP32 L/bool y. Synthetic
positive two-dimensional inputs require fixture=True. Each GT contributes its
full P-point mean BCE equally; outside-band terms remain in the reported loss.
The scalar has no regularizer/cap and uses no DEV or AP criterion. Population,
lineage, CPU/GPU rendering and efficacy are not qualified by this pure module.
SOURCE preparation is not numerical verification or permission to launch.
"""
import math

FIT_SHAPE=(1661,1024)
BAND_LIMIT=2.
ROOT_WIDTH=1e-8
MAX_STEPS=64


def _sigmoid(np,values):
    """Stable FP64 sigmoid without eager overflow in an unused branch."""
    result=np.empty_like(values,dtype=np.float64); nonnegative=values>=0
    result[nonnegative]=1./(1.+np.exp(-values[nonnegative]))
    exponent=np.exp(values[~nonnegative]); result[~nonnegative]=exponent/(1.+exponent)
    return result


def _full_BCE(np,logits,targets,band,delta):
    z=np.where(band,logits+delta,logits)
    losses=np.maximum(z,0.)-z*targets+np.log1p(np.exp(-np.abs(z)))
    value=float(losses.mean(axis=1,dtype=np.float64).mean(dtype=np.float64))
    if not math.isfinite(value): raise ValueError('full-point BCE must be finite')
    return value


def fit_offset(np,native_logits,targets,*,fixture=False):
    """Return one deterministic mixed-band FP64 root and full-point losses.

    Root interval is [logit(a)-2, logit(a)+2], where a is the weighted band
    foreground fraction. Equal positive P per GT makes a the ordinary band
    foreground count fraction. Crucially, loss/derivative denominators remain
    G*P, never per-GT band sizes. The closed band is abs(original native L)<=2.
    A mixed band gives a unique finite root; all-one-class bands raise instead
    of inventing a finite MLE/cap. An empty band is flat and returns delta=0.
    Bisection width<=1e-8/max64 is fixed before data. This is a practical FP64
    calculation, not a certified real-arithmetic root enclosure.
    """
    if type(fixture) is not bool: raise TypeError('fixture must be bool')
    if not isinstance(native_logits,np.ndarray) or native_logits.dtype!=np.float32:
        raise TypeError('native_logits must be an FP32 NumPy array')
    if not isinstance(targets,np.ndarray) or targets.dtype!=np.bool_:
        raise TypeError('targets must be a bool NumPy array')
    if native_logits.ndim!=2 or targets.shape!=native_logits.shape or any(n<=0 for n in native_logits.shape):
        raise ValueError('same positive two-dimensional logits/target shape required')
    if not fixture and tuple(native_logits.shape)!=FIT_SHAPE:
        raise ValueError('qualified FIT shape must be exact (1661,1024)')
    if not bool(np.isfinite(native_logits).all()): raise ValueError('native_logits must be finite')
    logits=native_logits.astype(np.float64,copy=True); labels=targets.astype(np.float64,copy=True)
    band=np.abs(logits)<=BAND_LIMIT; count=int(np.count_nonzero(band)); foreground=int(np.count_nonzero(targets & band)); background=count-foreground
    G,P=logits.shape; denominator=G*P; before=_full_BCE(np,logits,labels,band,0.)
    result=dict(delta=0.,fullpoint_BCE_before=before,fullpoint_BCE_after=before,GT_instances=G,points_per_GT=P,total_points=denominator,
                band_points=count,outside_band_points=denominator-count,band_foreground=foreground,band_background=background,
                band_positive_fraction=foreground/count if count else None,root_interval=[0.,0.],initial_root_interval=None,
                derivative_interval=[0.,0.],derivative_residual=0.,steps=0,unique_mixed_band_root=False,
                fixture_mode=fixture,fit_input_dtype='float32',fit_arithmetic_dtype='float64',native_band_closed_abs_logit=2.,
                instance_weights='equal GT weights; each full P-point mean, outside constants included',band_renormalized=False,
                regularizer_or_offset_cap=None,selection_rule='one full caller-provided FIT cached-point BCE root; no DEV or AP selection',
                no_DEV_selection_by_operator=True,population_or_lineage_independently_qualified=False,
                solver=dict(width=ROOT_WIDTH,max_steps=MAX_STEPS,certified_root_enclosure=False))
    if not count: return result
    if not foreground or not background:
        raise ValueError('degenerate single-class band: no finite global-offset MLE; do not add an implicit cap')
    selected=logits[band]

    def derivative(delta):
        # This is the derivative of equal-instance/full-point BCE. Summing only
        # nonzero derivative terms does not renormalize the optimization loss.
        value=(float(_sigmoid(np,selected+delta).sum(dtype=np.float64))-foreground)/denominator
        if not math.isfinite(value): raise ValueError('offset derivative must be finite')
        return value

    logit_fraction=math.log(foreground)-math.log(background)
    lower,upper=logit_fraction-BAND_LIMIT,logit_fraction+BAND_LIMIT
    initial=[lower,upper]; d_lower,d_upper=derivative(lower),derivative(upper)
    # Endpoint equalities can round by FP64 summation. This allowance neither
    # widens/truncates delta nor supplies a certified enclosure or data tuning.
    roundoff=64.*float(np.finfo(np.float64).eps)
    if d_lower>roundoff or d_upper < -roundoff: raise RuntimeError('mixed-band analytic bracket failed')
    steps=0
    while upper-lower>ROOT_WIDTH and steps<MAX_STEPS:
        middle=lower+(upper-lower)/2.; value=derivative(middle)
        if value<=0.: lower=middle
        else: upper=middle
        steps+=1
    if upper-lower>ROOT_WIDTH: raise RuntimeError('offset root width did not converge within64 steps')
    delta=lower+(upper-lower)/2.; after=_full_BCE(np,logits,labels,band,delta)
    if after>before+64.*float(np.finfo(np.float64).eps)*max(1.,before):
        raise RuntimeError('fitted full-point BCE increased beyond FP64 arithmetic allowance')
    result.update(delta=delta,fullpoint_BCE_after=after,root_interval=[lower,upper],initial_root_interval=initial,
                  derivative_interval=[derivative(lower),derivative(upper)],derivative_residual=derivative(delta),steps=steps,
                  unique_mixed_band_root=True,derivative_roundoff_allowance=roundoff)
    return result


def apply_offset(torch,native_logits,delta):
    """Return a new tensor: L+delta only where the original abs(L)<=2.

    No input detach/in-place change, query filtering, score handling or GT is
    performed. Caller retains original native scores and all query identities.
    Allow empty tensors and preserve their connected graph. Delta is a finite
    scalar fitted on FIT, not a per-query/per-image parameter or residual head.
    FP32/FP64 callers apply the scalar in their own dtype and must independently
    qualify that finite-dtype/rendering path; this is not CUDA bitwise proof.
    """
    if not isinstance(native_logits,torch.Tensor) or native_logits.layout!=torch.strided or native_logits.dtype not in (torch.float32,torch.float64):
        raise TypeError('native_logits must be a strided FP32/FP64 tensor')
    if type(delta) not in (int,float): raise TypeError('delta must be a non-bool Python real scalar')
    if not math.isfinite(delta) or not bool(torch.isfinite(native_logits).all()): raise ValueError('delta and native logits must be finite')
    value=native_logits.new_tensor(float(delta))
    if not bool(torch.isfinite(value)): raise ValueError('delta must stay finite in the native tensor dtype')
    band=native_logits.abs()<=BAND_LIMIT
    result=torch.where(band,native_logits+value,native_logits)
    if not bool(torch.isfinite(result).all()): raise ValueError('offset output must be finite; no implicit clipping is allowed')
    return result
