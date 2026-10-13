# 固定PR原栅格区域装配：接口与有限合成检查

更新：2026-10-14。状态：**UNBOUND_DRAFT / SYNTHETIC_ONLY**。本项为真实诊断准备必要接口，没有执行SAR实验，也不属于核心创新。

[装配源码](code/N13_fixed_PR_native_domain_assembly_draft.py)只接收显式Torch、native_L及作者边界函数回调，不加载模型、数据或作者文件，不接收G、权重、K或最终选集合。原栅格L须为非空有限FP32 CPU/CUDA二维张量；native_P=L>0，old_LOG=abs(L)≤2。传给回调的是独立、连续的CPU FP32 [1,H,W]预测副本，boundary_width固定为2；回调输出须为CPU int64同形标签0/1/2。

使用回调返回的标签1得到native_width2，再令new_halo=native_width2 AND NOT old_LOG、Omega=old_LOG OR new_halo。五个输出域均为L设备上的独立布尔存储。计数只是逻辑域大小，不是实际计算量、时耗或资源测量。这里不实现作者公式，也不以一般距离≤2替代width2；显式提供一个回调不认证它是实际作者函数。[作者定义与本地文字核对](N13_FIXED_PR_ACTUAL_SOURCE_INTERFACE_REVIEW_20261014.md)仍仅支持既有接口规则，真实域须另按来源资格核验。

## 已完成的有限行为检查

按tdd的一条主路径先出现预期未实现错误（RED实际子进程退出1，3.798秒），再完成最小装配实现。RED是预期研发测试失败，不是科学执行失败或模型效用结果。GREEN实际子进程退出0、已回收，3.182秒、stderr为空；[便携检查](code/N13_fixed_PR_native_domain_assembly_hand_check.py)的49项显式核对及13项拒绝通过。没有重跑已有适配器或oracle自测。

主例是非连续L=[[-3,-2,0,2],[3,-1,1,-4]]；回调返回独立字面标签[[[1,0,1,2],[0,0,1,0]]]。其标签故意作为装配输入，不是作者函数数值输出。手算与实际计数一致：

| 域 | flat索引 | 点数 |
|---|---|---:|
| native_P | 3,4,6 | 3 |
| old_LOG | 1,2,3,5,6 | 5 |
| native_width2（字面回调） | 0,2,6 | 3 |
| new_halo | 0 | 1 |
| Omega | 0,1,2,3,5,6 | 6 |
| outside | 4,7 | 2 |

另一个L=[[-3,0,3]]、字面标签[[[0,1,2]]]无新增halo，Omega=old_LOG且点数1。检查包括严格零阈值、±2包含、非空旧域/新增域/域外、无halo、回调输入被破坏时L/native_P不变、返回标签与域输出之间的存储隔离；错误返回形状/dtype/device/标签和非有限/错误native_L被显式拒绝。

环境为host Python3.10.13、Torch2.2.2+cpu、NumPy1.26.4、CPU亲和[0,1]、Torch1/1线程、四线程环境值1、CUDA空且未初始化、-B、30秒内。**未验证作者函数数值、真实L/函数身份、CUDA结果、科学Torch2.0.1+cu118、完整SAR域或大栅格资源。**不能称width2已经在真实数据上重建通过；作者回调仍需由ROOT按绑定来源提供，并以独立计数核验。

## 复现

仓库根目录，需Torch、NumPy、psutil及可用CPU0/1；命令只执行字面回调的合成检查：

```powershell
$env:CUDA_VISIBLE_DEVICES = ""
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
python -B research_round_20260925_global/code/N13_fixed_PR_native_domain_assembly_hand_check.py --assembly research_round_20260925_global/code/N13_fixed_PR_native_domain_assembly_draft.py
```

公开两份源码按字节复制，SHA256分别为001d3f06976bf4b55adf587939e5ae1e03c8ff39209e95d4398fa37d06e82963（装配）和fa465f8a282f2d54a58174ee2255a37f46e45c5574eaeb0b070382024b1765a7（检查）。RED/GREEN私有运行记录和隐藏辅助运行器本地保留，不纳入发布范围。没有为发布读取运行中的科学载荷或绑定SOURCE。

[诊断连接层](N13_FIXED_PR_PIPELINE_SYNTHETIC_INTEGRATION_CHECK_20261014.md)可接收此API输出的old_LOG/new_halo；这仅说明接口对应，未以真实输入执行组合。仍先完成剩余预锁E24曲线，再登记固定首种子701 LOG E48的374匹配/6未匹配GT诊断。当前三种子E48联合配方停止决定、无新AP、无核心创新/正式TEST/跨集/TGRS依据的状态保持。
