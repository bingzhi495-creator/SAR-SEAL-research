# GB10 保存张量与 CUDA 额度：未来性能试点可行性

2026-10-21；仅普通公开源码静态核查与官方资料。性能试点 **UNEXECUTED**，未读绑定 SOURCE、实际运行链、数据或权重，未改动当前正式训练。

1. **存在可测假设，尚未定位瓶颈。** [U39 resource_probe.py:185](code/U39_FULL_METHOD_CROSS_IMAGE_RESOURCE_DRAFT_20261017_01/resource_probe.py#L185) 仅在 `adapter.encode(x)` 周围保存中间张量到 CPU；同文件 :194–217 对每个原候选分别 backward、保留共享图，可能重复取回编码器保存值。另有逐参数梯度与 Adam 有限性检查、标量取回（[U36 probe_body.py:269](code/U36_REAL400_CPU_SAVED_ACTIVATIONS_DRAFT_20261017_01/probe_body.py#L269)、:287），不能把整段耗时全部归于 offload。

2. **官方机制支持尝试减少 offload，但不支持承诺倍数。** [PyTorch 2.10 save_on_cpu](https://docs.pytorch.org/docs/2.10/autograd.html#torch.autograd.graph.save_on_cpu) 明确以反向取回开销交换 GPU 内存；[v2.10.0 实现](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/autograd/graph.py) 的非 pinned 分支调用 `tensor.cpu()`，取回调用 `tensor.to(device, non_blocking=pin_memory)`。[NVIDIA](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/overview.html) 说明 CPU/GPU 共享 128GB LPDDR5x；硬件支持共享不等于该框架路径已免除复制、分配或同步，实际成本须测。128GB 不能推出 FP32 速度；[allocator fraction](https://docs.pytorch.org/docs/2.10/generated/torch.cuda.memory.set_per_process_memory_fraction.html) 是内存上限，超限报 OOM，不是算力限速。单独放宽额度未必变快。

3. **原接口没有运行时关闭开关。** [U57 training_body.py:31](code/U57_DGX_ORIGINAL_UPDATE_TRAINING_DRAFT_20261018_01/training_body.py#L31)、:317/:390 固定复用原 helper/update；[U36 :355](code/U36_REAL400_CPU_SAVED_ACTIVATIONS_DRAFT_20261017_01/probe_body.py#L355) 硬设 pin=True，:393 强核实际 CPU 保存。此前“当前 pin=False”是未经核实的简写，已撤回，不能作为运行事实；当前 GB10 绑定执行的该参数未核验，本次未读取绑定源码，不能推断存在矛盾。未来减少 offload 必须形成独立代码版本与资源配置；原固定 E50 的配方、源码身份和连续状态不允许中途替换。

4. **最少三个独立条件才能区分两因素。** A=原 4GiB+原 offload；B=另行核准的较大额度+相同 offload；C=与 B 同额度+关闭 encoder offload。保持 FP32/query1/CPU01/双线程、相同输入顺序和双方相同更新数。每个条件从同一隔离起点恢复完整 model（含 BN/闲置分支/训练模式）、Adam moments/step/参数组及 Python/NumPy/Torch CPU/CUDA RNG；预热后也重新恢复。先各做含小图与高 G/大面积图的相同三次连续更新，再决定是否值得短块重复计时。逐步比较 loss/梯度/完整状态，精确不等即记录，不能事后放宽容差宣布原轨迹等价；试点产物不写回正式父 checkpoint。

5. **以反证与资源边界收尾。** 同时记录含检查的每更新墙钟、同步后的 encode/readout/backward/Adam 时间、pack/unpack 次数与字节、CPU/GPU 时间线；读取/哈希/恢复/保存另外列时，不能只用 CUDA Events 代表端到端。[PyTorch](https://docs.pytorch.org/docs/2.10/notes/cuda.html#asynchronous-execution) 要求正确同步计时；[NVIDIA](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/optimization.html#profiling) 建议先分析时间线。保留 RAM≥16GiB、RSS/时限等已核约束，另核共享 RAM、CUDA allocated/reserved 峰值与 swap；较大额度须结合并存内存测量确定，不能拿全部“可用 RAM”作为 CUDA 预算。若 B 无收益而 C 稳定加快且状态检查通过，才支持减少 offload；若 C 仍无收益或时间主要落在检查/CPU/重复 backward，削弱该假设。OOM、资源越界或状态差异均是有效失败结果。本次未执行上述试点。
