"""Public numerical behavior check; Torch is injected into the callable test."""
import importlib.util
from pathlib import Path

def verify_cpu_components(torch, objective):
    z = torch.tensor([[-2., 0., 2., .5], [1., -1., .25, -3.]], dtype=torch.float64, requires_grad=True)
    y = torch.tensor([[False, True, True, False], [True, False, True, False]])
    parts = objective.point_loss_components(torch, z, y)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(z, y.to(z.dtype))
    torch.testing.assert_close(parts['uniform'], expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(parts['foreground']+parts['background'], expected, rtol=1e-12, atol=1e-12)
    actual = torch.autograd.grad(parts['uniform'], z, retain_graph=True)[0]
    expected_gradient = (torch.sigmoid(z.detach())-y.to(z.dtype))/z.numel()
    torch.testing.assert_close(actual, expected_gradient, rtol=1e-12, atol=1e-12)
    actual_balanced = torch.autograd.grad(parts['balanced'], z)[0]
    positive = y.sum(1).clamp_min(1).to(z.dtype)
    negative = (~y).sum(1).clamp_min(1).to(z.dtype)
    expected_balanced = (-y.to(z.dtype)*(1-torch.sigmoid(z.detach()))/positive[:,None]
                        +(~y).to(z.dtype)*torch.sigmoid(z.detach())/negative[:,None])/(2*z.shape[0])
    torch.testing.assert_close(actual_balanced, expected_balanced, rtol=1e-12, atol=1e-12)
    return {'status':'PUBLIC_POINT_OBJECTIVE_VALUES_AND_ANALYTIC_GRADIENTS_PASS',
            'dtype':'float64','native_model_executed':False,'CUDA_used':False}

if __name__ == '__main__':
    path = Path(__file__).with_name('point_objectives.py')
    module_spec = importlib.util.spec_from_file_location('_new_public_point_objectives', path)
    objective = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(objective)
    # A RED missing implementation stops above, before any scientific import.
    import json
    import torch
    print(json.dumps(verify_cpu_components(torch, objective)))

