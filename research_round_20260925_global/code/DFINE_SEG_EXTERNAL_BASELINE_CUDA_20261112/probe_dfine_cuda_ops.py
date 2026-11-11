"""Synthetic CUDA operator qualification for the pinned author D-FINE-seg runtime."""
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
from sysconfig import get_path
from time import perf_counter


ROOT = Path(__file__).resolve().parent
AUTHOR = ROOT.parent / "AUTHOR_SOURCE_VERIFIED"
SOURCE_COMMIT = "b782ece206487792f36c165ca65548154cd12094"


def need(condition, why):
    if not condition:
        raise ValueError(why)


def main():
    need(sys.version_info[:3] == (3,11,15) and sys.flags.isolated == sys.flags.utf8_mode == 1
         and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "Exact isolated single CUDA runtime")
    versions = {name: version(name) for name in ("dfine-seg", "torch", "torchvision")}
    need(versions == {"dfine-seg":"0.7.0", "torch":"2.13.0+cu130", "torchvision":"0.28.0+cu130"}, "Exact verified author and PyTorch packages")
    installed = Path(get_path("purelib")) / "dfine_seg"
    count = 0
    for path in (AUTHOR / "dfine_seg").rglob("*"):
        if path.is_file() and path.suffix in {".py", ".yml", ".yaml"}:
            need((installed/path.relative_to(AUTHOR/"dfine_seg")).read_bytes() == path.read_bytes(), "Exact original author Python/config")
            count += 1
    need(count == 70, "Exact complete installed author Python/config roster")

    import torch
    from torchvision.ops import nms
    from dfine_seg.model.arch.utils import deformable_attention_core_func_v2

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    need(torch.version.cuda == "13.0" and torch.cuda.is_available() and torch.cuda.device_count() == 1,
         "Actual CUDA13 RTX must be available")
    dev = torch.device("cuda:0")
    gpu = torch.cuda.get_device_name(dev)
    capability = torch.cuda.get_device_capability(dev)
    free, total = torch.cuda.mem_get_info(dev)
    need("RTX 3090 Ti" in gpu and capability == (8,6) and free >= 12 * 1024**3, "Actual free local RTX3090Ti")
    torch.cuda.set_per_process_memory_fraction(min(0.1, 2*1024**3 / total), dev)

    def attention(device):
        value = [
            torch.arange(12, dtype=torch.float32, device=device).reshape(1,1,2,6).div(10).requires_grad_(),
            torch.arange(8, dtype=torch.float32, device=device).reshape(1,1,2,4).div(7).requires_grad_(),
        ]
        points = torch.tensor([[[[[.21,.29],[.67,.41],[.24,.82],[.94,.08]]],
                                [[[.12,.60],[.84,.21],[.48,.55],[.65,.38]]]]],
                              device=device, dtype=torch.float32, requires_grad=True)
        weights = torch.tensor([[[[.1,.2,.3,.4]],[[.4,.3,.2,.1]]]], device=device,
                               dtype=torch.float32, requires_grad=True)
        output = deformable_attention_core_func_v2(value,[(2,3),(2,2)],points,weights,[2,2])
        gradients = torch.autograd.grad(output.square().sum(), (*value,points,weights))
        need(output.shape == (1,2,2) and all(item.device.type == torch.device(device).type
                                          and torch.isfinite(item).all().item() for item in (output,*gradients)),
             "Original author deformable attention output and all gradients are finite on actual device")
        return tuple(item.detach().cpu() for item in (output,*gradients))

    begin = perf_counter()
    dense = torch.tensor([[1.,2.],[3.,4.]],device=dev)
    product = dense @ dense
    need(product.device.type == "cuda" and product.cpu().tolist() == [[7.,10.],[15.,22.]], "Actual CUDA dense tensor")
    cpu = attention("cpu")
    cuda = attention(dev)
    abs_diffs = [float((a-b).abs().max().item()) for a,b in zip(cpu,cuda,strict=True)]
    need(all(err <= 1e-5 for err in abs_diffs), "Original author CUDA attention matches bounded CPU forward/backward reference")
    boxes = torch.tensor([[0.,0.,10.,10.],[1.,1.,11.,11.],[20.,20.,30.,30.]], device=dev)
    confidence = torch.tensor([.9,.8,.7], device=dev)
    kept = nms(boxes,confidence,0.5)
    need(kept.device.type == "cuda" and kept.cpu().tolist() == [0,2], "Actual torchvision CUDA native NMS")
    torch.cuda.synchronize(dev)
    result = dict(status="DFINE_SEG_CUDA_SYNTHETIC_NATIVE_OPERATORS_ACTUAL_PASS_V1", source_commit=SOURCE_COMMIT,
                  installed_exact_author_files=count, versions=versions, cuda_runtime=torch.version.cuda,
                  gpu_name=gpu, compute_capability=list(capability), free_gpu_bytes_at_admission=free,
                  total_gpu_bytes=total, max_memory_allocated_bytes=torch.cuda.max_memory_allocated(dev),
                  dense_cuda_product=product.cpu().tolist(), author_attention_cuda_output_shape=list(cuda[0].shape),
                  max_abs_cpu_cuda_attention_output_diff=abs_diffs[0],
                  max_abs_cpu_cuda_gradient_diffs=abs_diffs[1:], accepted_abs_tolerance=1e-5,
                  cuda_nms_kept_indices=kept.cpu().tolist(), gpu_operations_seconds=perf_counter()-begin,
                  FIT=False, DEV=False, TEST=False, AP=False, training=False, weights_opened=False,
                  optimizer_updates=0, model_forward_calls=0)
    raw = (json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    with (ROOT/"CUDA_OPERATOR_SMOKE_SAFE.json").open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(status=result["status"], result_sha256=sha256(raw).hexdigest(),
                          max_abs_cpu_cuda_diff=max(abs_diffs), cuda_nms=result["cuda_nms_kept_indices"]),sort_keys=True),flush=True)


if __name__ == "__main__":
    main()
