# SAMRefiner单图功能与资源探测已合格：外部AP仍待

2026-10-25。v03清理修正后的独立作业已实际退出0，原控制端、远端命令及worker均为0，并通过Root外部来源、输出及进程回收核验。合格范围限于共享GPU条件下的一张图、100个预测候选、ViT-H FP32、作者默认5轮的功能与资源探测；尚无同协议AP或全187图结论。详见[安全结果](results/U57_SAMREFINER_CLEANUP_FIX_ONE_IMAGE_SAFE_RESULT_20261025.json)。

## 本次实际证据

真实作者路径的image encoder前/后钩子各1次，prompt encoder和mask decoder各5次，并已CUDA同步。钩子是调用记录，不能独立证明轮次语义、作者内部返回顺序或精修质量；候选内部对应与历史输入等价范围仍有未闭合部分。本次没有使用GT或产生训练更新。

| 项目 | 实际观察 |
|---|---:|
| 清理前CUDA allocated | 9,568,256 B |
| 清理前CUDA reserved | 10,026,483,712 B |
| 修正清理后allocated / reserved | **0 / 0 B** |
| CUDA分配器峰值 | 7,011,852,288 B |
| worker采样峰值RSS | 6,125,678,592 B |
| 父监督采样峰值所属进程组RSS | 6,182,375,424 B |
| 父监督采样最低系统可用RAM | 64,506,085,376 B |
| worker耗时 | 19.590439919秒 |

耗时含准备、推理及清理，是共享设备上本次作业的观察，不是标准推理延迟或吞吐基准。worker共记录2599次资源采样，未记录资源越界；CUDA分配器峰值也不等于整卡或全部原生库内存。

Root在原正常终态之后核对38项唯一准入来源与输出字节，同一启动周期下所属子进程出生身份消失、所属进程组为空，父进程出生信息此次已记录并确认消失。该来源足迹限选定准入文件，不是完整Python、原生扩展或动态库的二进制闭包。

## 清理修正支持什么解释

v03在推理结束后的清理阶段加入进程内工作区清理，并配合分配器缓存释放与同步，随后allocated与reserved均为0。PyTorch v2.10.0的[CUDA绑定](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/csrc/cuda/Module.cpp#L1481)提供工作区清理入口，[cuBLAS/cuBLASLt工作区池](https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/cuda/CublasHandlePool.cpp#L98)维护两个映射并实现清空。因此，本次前后观察支持当前配置下后端工作区持有分配的解释。

这里没有独立拆开每个清理步骤，不能证明先前残留全部来自cuBLAS，也不能据一次0/0给出通用Tensor泄漏排除证明。0/0限框架分配器计数，不代表CUDA上下文、所有原生库或整张GPU物理内存为零。共享背景负载亦未被锁定，不能据该耗时解释其他训练速度。

此前[v02原始失败](results/U57_SAMREFINER_SHARED_PROBE_V02_SAFE_FAILURE_20261025.json)永久保留，其不合格输出没有作为父产物，本次正常终态不追溯改写它。[v02失败分析与待检验假说](U57_SAMREFINER_FIRST_REAL_MODEL_FAILURE_AND_CLEANUP_HYPOTHESIS_20261025.md)保留原记录；本页另行记录新增证据。

这一步闭合了真实模型在当前单图配置下的功能与资源探测。外部比较仍需固定候选内部对应、完整人口运行、统一COCO评价及成本记录；本页不构成精度提升、核心创新、充分外部对比或TGRS就绪证据。
