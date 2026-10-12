"""Future ROOT-only synthetic tests; SOURCE preparation does not run them.

Call run_tests(np, torch, offset_module) with explicitly supplied libraries and
the verified SOURCE module. No library imports, files, GT, images, model or
scientific execution entry is provided here. Tests have not been executed.
"""
import math


def _require(ok,message):
    if not ok: raise AssertionError(message)


def _raises(kind,fragment,function):
    try:
        function()
    except kind as error:
        _require(fragment in str(error),'Wrong rejection message: '+str(error))
    else:
        raise AssertionError('Expected '+kind.__name__+' containing '+fragment)


def _scalar_sigmoid(value):
    if value>=0: return 1./(1.+math.exp(-value))
    exp_value=math.exp(value); return exp_value/(1.+exp_value)


def _scalar_BCE(value,target):
    # Independently select softplus(-z) for y=1 or softplus(z) for y=0.
    signed=-value if target else value
    return math.log1p(math.exp(signed)) if signed<=0 else signed+math.log1p(math.exp(-signed))


def _oracle(logits,targets,delta):
    rows=[]; derivatives=[]
    for values,labels in zip(logits.tolist(),targets.tolist()):
        losses=[]; gradient=[]
        for value,target in zip(values,labels):
            band=abs(value)<=2.; z=value+delta if band else value
            losses.append(_scalar_BCE(z,target)); gradient.append((_scalar_sigmoid(z)-int(target)) if band else 0.)
        rows.append(math.fsum(losses)/len(values)); derivatives.append(math.fsum(gradient)/len(values))
    return math.fsum(rows)/len(rows),math.fsum(derivatives)/len(derivatives)


def run_tests(np,torch,offset_module):
    """Synthetic CPU-only numerical execution belongs to ROOT, not authoring."""
    _require(not torch.cuda.is_initialized(),'Synthetic CPU must start without initialized CUDA')
    cpu_rng=torch.random.get_rng_state().clone(); checked=[]

    def arrays(logits,targets):
        return np.array(logits,dtype=np.float32),np.array(targets,dtype=np.bool_)

    def check_fit(logits,targets,expected_delta,label):
        before_L=logits.copy(); before_y=targets.copy()
        result=offset_module.fit_offset(np,logits,targets,fixture=True)
        _require(np.array_equal(logits,before_L) and np.array_equal(targets,before_y),'Fitting mutated synthetic inputs')
        _require(abs(result['delta']-expected_delta)<=1e-8,'Independent analytic root differs: '+label)
        lo,hi=result['root_interval']; _require(hi-lo<=1e-8 and 0<=result['steps']<=64,'Locked interval/iteration rule differs')
        oracle_before,_=_oracle(logits,targets,0.); oracle_after,oracle_grad=_oracle(logits,targets,result['delta'])
        _require(abs(result['fullpoint_BCE_before']-oracle_before)<=1e-12 and abs(result['fullpoint_BCE_after']-oracle_after)<=1e-12 and
                 abs(result['derivative_residual']-oracle_grad)<=1e-12 and oracle_after<=oracle_before+1e-12,'Scalar BCE/derivative oracle differs')
        _require(result['fixture_mode'] is True and result['no_DEV_selection_by_operator'] is True and
                 result['population_or_lineage_independently_qualified'] is False,'Pure-math evidence scope differs')
        checked.append(label); return result

    L,y=arrays([[0.,0.,0.,0.],[0.,0.,0.,0.]],[[True,False,True,False],[False,True,False,True]])
    check_fit(L,y,0.,'zero mixed-band root')
    L,y=arrays([[0.,0.,0.,0.]],[[True,True,True,False]])
    check_fit(L,y,math.log(3.),'positive root')
    L,y=arrays([[0.,0.,0.,0.]],[[True,False,False,False]])
    check_fit(L,y,-math.log(3.),'negative root')

    # Two GT rows have different band counts. Full P-point means give 2FG:1BG
    # and root log(2); normalizing each row by its own band size would give 0.
    L,y=arrays([[0.,0.,3.,-3.],[0.,3.,3.,-3.]],[[True,True,False,True],[False,False,False,True]])
    result=check_fit(L,y,math.log(2.),'different GT band counts keep instance weights')
    _require(result['band_foreground']==2 and result['band_background']==1 and result['band_points']==3 and
             result['outside_band_points']==5 and abs(result['delta'])>.5,'Band counts or implicit band renormalization differs')
    # Alter only outside labels: the fitted delta stays fixed, full BCE changes.
    different=y.copy(); different[0,2:]=~different[0,2:]; different[1,1:]=~different[1,1:]
    other=check_fit(L,different,math.log(2.),'outside BCE is constant with respect to offset')
    _require(abs(result['fullpoint_BCE_after']-other['fullpoint_BCE_after'])>.1,'Outside points were silently dropped from reported BCE')
    L,y=arrays([[2.,2.,2.]],[[True,False,False]])
    check_fit(L,y,-math.log(2.)-2.,'closed native band includes +2')
    L,y=arrays([[-2.,-2.,-2.]],[[True,True,False]])
    check_fit(L,y,math.log(2.)+2.,'closed native band includes -2')

    L,y=arrays([[3.,-3.],[4.,-4.]],[[True,False],[False,True]])
    empty=offset_module.fit_offset(np,L,y,fixture=True); before,_=_oracle(L,y,0.)
    _require(empty['delta']==0. and empty['root_interval']==[0.,0.] and empty['steps']==0 and empty['band_points']==0 and
             empty['unique_mixed_band_root'] is False and abs(empty['fullpoint_BCE_before']-before)<=1e-12 and
             empty['fullpoint_BCE_after']==empty['fullpoint_BCE_before'],'Empty-band identity differs')
    checked.append('empty band returns zero without invented unique root')
    for target in (True,False):
        L,y=arrays([[0.,2.,3.]],[[target,target,not target]])
        _raises(ValueError,'degenerate',lambda:offset_module.fit_offset(np,L,y,fixture=True))
    checked.append('single-class band has no finite estimate')

    tensor=torch.tensor([[-3.,-2.,-.5,0.,2.,3.]],dtype=torch.float32,requires_grad=True)
    original=tensor.detach().clone(); version=tensor._version
    identity=offset_module.apply_offset(torch,tensor,0.)
    _require(identity is not tensor and torch.equal(identity,original),'Zero offset must return native values in a new tensor')
    output=offset_module.apply_offset(torch,tensor,.25)
    expected=torch.tensor([[-3.,-1.75,-.25,.25,2.25,3.]],dtype=torch.float32)
    _require(torch.equal(output,expected) and torch.equal(output[tensor.abs()>2],original[tensor.abs()>2]) and
             tensor._version==version and torch.equal(tensor.detach(),original),'Original-band or nonmutation rule differs')
    output.sum().backward(); _require(torch.equal(tensor.grad,torch.ones_like(tensor)) and bool(torch.isfinite(tensor.grad).all()),'Pure apply lost native input gradient')
    zero=torch.empty((0,3),dtype=torch.float32,requires_grad=True); z=offset_module.apply_offset(torch,zero,.25)
    z.sum().backward(); _require(z.shape==zero.shape and zero.grad is not None and zero.grad.numel()==0,'Empty tensor apply must stay connected')
    checked.append('apply identity, endpoints, outside, gradients and empty tensor')

    L,y=arrays([[0.,1.]],[[True,False]])
    _raises(ValueError,'1661',lambda:offset_module.fit_offset(np,L,y))
    _raises(TypeError,'FP32',lambda:offset_module.fit_offset(np,L.astype(np.float64),y,fixture=True))
    _raises(TypeError,'bool',lambda:offset_module.fit_offset(np,L,y.astype(np.int64),fixture=True))
    _raises(ValueError,'finite',lambda:offset_module.fit_offset(np,np.array([[np.nan]],dtype=np.float32),np.array([[True]],dtype=np.bool_),fixture=True))
    _raises(ValueError,'positive',lambda:offset_module.fit_offset(np,np.empty((0,2),dtype=np.float32),np.empty((0,2),dtype=np.bool_),fixture=True))
    _raises(TypeError,'delta',lambda:offset_module.apply_offset(torch,tensor,True))
    _raises(ValueError,'finite',lambda:offset_module.apply_offset(torch,tensor,float('inf')))
    _require(torch.equal(cpu_rng,torch.random.get_rng_state()) and not torch.cuda.is_initialized(),'Synthetic operations changed CPU RNG or initialized CUDA')
    return dict(status='SYNTHETIC_MATH_CHECKS_PASSED_NOT_SCIENTIFIC_QUALIFICATION',checks=checked,
                CPU_RNG_unchanged=True,CUDA_initialized=False,real_FIT_or_DEV_accessed=False,
                actual_FIT_population_or_lineage_verified=False,model_efficacy_verified=False)
