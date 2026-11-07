"""UNBOUND caller-owned actual tensor loading helper; no ML imports."""
from copy import deepcopy
from collections.abc import Mapping

PREFIX='backbone.bottom_up.'


def need(ok,message):
    if not ok:raise ValueError(message)


def _metadata(value):
    return dict(dtype=str(value.dtype),shape=list(value.shape),numel=value.numel())


def _tensor(torch,value,key):
    need(type(value) is torch.Tensor,'Ordinary torch.Tensor required: '+key)
    need(value.device.type=='cpu' and value.layout==torch.strided and not value.is_quantized,
         'CPU strided unquantized tensor required: '+key)
    need(torch.isfinite(value).all().item() is True,'All tensor values must be finite: '+key)


def _state(torch,state,label):
    need(isinstance(state,Mapping) and all(type(key) is str for key in state),'Named tensor mapping: '+label)
    for key,value in state.items():_tensor(torch,value,label+':'+key)


def _parameters(model):
    rows=list(model.named_parameters(remove_duplicate=False))
    need(all(type(key) is str and type(value.requires_grad) is bool for key,value in rows),
         'Named parameter identity and trainability')
    need(len({key for key,value in rows})==len(rows),'Unique named parameter keys')
    return {key:(id(value),value.requires_grad) for key,value in rows}


def load_backbone_verified(torch, model, source_state, coverage):
    """Load only a complete explicit R101 backbone into a caller-owned model."""
    rng_before=torch.get_rng_state().detach().clone()
    parameters_before=_parameters(model)
    state=model.state_dict()
    _state(torch,source_state,'source')
    _state(torch,state,'model_before')
    need(set(parameters_before)<=set(state),'All named parameters must be present in full state')
    buffer_rows=list(model.named_buffers(remove_duplicate=False))
    need(len({key for key,value in buffer_rows})==len(buffer_rows),'Unique named buffer keys')
    _state(torch,dict(buffer_rows),'buffers_before')
    buffers_before={key:value.detach().clone() for key,value in buffer_rows}
    source_meta={key:_metadata(value) for key,value in source_state.items()}
    target_meta={key:_metadata(value) for key,value in state.items() if key.startswith(PREFIX)}
    plan=coverage.check_coverage(source_meta,target_meta)
    need(plan['mapped_key_count']==520 and plan['mapped_numel']==42605504 and
         plan['target_key_count']==586,'Complete fixed R101 coverage plan')
    before={key:value.detach().clone() for key,value in state.items()}
    expected_source={key:value.detach().clone() for key,value in source_state.items()}
    payload={key:value.detach().clone() for key,value in before.items()}
    mapped={row['target']:row['source'] for row in plan['mapping']}
    need(len(mapped)==520,'Unique complete backbone targets')
    need(len(set(mapped.values()))==520 and sum(expected_source[key].numel() for key in mapped.values())==42605504,
         'Unique actual source tensors and full state element count')
    for dest,source in mapped.items():payload[dest]=expected_source[source].detach().clone()
    with torch.no_grad():
        result=model.load_state_dict(payload,strict=True,assign=False)
    need(not result.missing_keys and not result.unexpected_keys,'Strict complete state loading result')
    after=model.state_dict();need(set(after)==set(before),'Complete state keys unchanged')
    _state(torch,after,'model_after')
    for key,value in after.items():
        expected=expected_source[mapped[key]] if key in mapped else before[key]
        need(str(value.dtype)==str(expected.dtype),'Tensor dtype changed after loading: '+key)
        need(tuple(value.shape)==tuple(expected.shape) and value.numel()==expected.numel(),
             'Tensor shape or numel changed after loading: '+key)
        need(torch.equal(value,expected),'Tensor values differ after loading: '+key)
    need(set(source_state)==set(expected_source),'Source tensor keys changed during loading')
    _state(torch,source_state,'source_after')
    for key,value in source_state.items():
        need(_metadata(value)==_metadata(expected_source[key]) and torch.equal(value,expected_source[key]),
             'Source tensor changed during loading: '+key)
    buffer_rows_after=list(model.named_buffers(remove_duplicate=False))
    need(len(buffer_rows_after)==len(buffers_before) and
         {key for key,value in buffer_rows_after}==set(buffers_before),'Named buffer keys changed')
    _state(torch,dict(buffer_rows_after),'buffers_after')
    for key,value in buffer_rows_after:
        expected=expected_source[mapped[key]] if key in mapped else buffers_before[key]
        need(_metadata(value)==_metadata(expected) and torch.equal(value,expected),
             'Named buffer values or dtype changed: '+key)
    need(_parameters(model)==parameters_before,'Named parameter identity or requires_grad changed')
    need(torch.equal(torch.get_rng_state(),rng_before),'CPU Torch RNG changed during loading')
    unloaded={key:value for key,value in after.items() if key not in mapped}
    unloaded_dtypes={}
    for key,value in unloaded.items():
        row=unloaded_dtypes.setdefault(str(value.dtype),dict(keys=0,numel=0))
        row['keys']+=1;row['numel']+=value.numel()
    return dict(schema='R101-caller-owned-actual-tensor-load-observation-v1',
                actual_tensor_load_checked=True,mapped_keys=len(mapped),
                mapped_state_numel=plan['mapped_numel'],stages=deepcopy(plan['stages']),
                unloaded_state_keys=len(after)-len(mapped),
                unloaded_state_numel=sum(value.numel() for value in unloaded.values()),
                unloaded_state_dtypes=unloaded_dtypes,
                unloaded_target_backbone_offset_keys=len(plan['target_unloaded_plan']),
                mapped_values_equal_to_independent_source_snapshot=True,
                unloaded_values_equal_to_independent_model_snapshot=True,
                full_state_keys_unchanged=True,all_source_and_model_state_values_finite=True,
                parameter_identity_and_requires_grad_unchanged=True,CPU_Torch_RNG_unchanged=True,
                source_tensor_values_unchanged=True,
                named_buffer_values_checked_including_nonpersistent=True,
                source_provenance_verified=False,model_identity_verified=False,
                equality_semantics='torch.equal plus explicit dtype/shape/numel; not bitwise byte equality',
                forward_executed=False,optimizer_created=False,qualified=False,AP=False)
