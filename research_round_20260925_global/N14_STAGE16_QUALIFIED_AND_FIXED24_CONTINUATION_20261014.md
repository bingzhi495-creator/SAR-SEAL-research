# N14：stage16 核验完成，继续固定 stage24

状态日期：2026-10-14。此文是完成执行后的独立普通总结；不会为同步读取运行中的科学载荷、绑定源码或资格链。

## 固定配方与实际状态

研究候选是已有点精修头上的 F+c 上下文记忆残差。BASE_CONTINUATION 与 MEMORY_ADDON 均从预锁第一701 LOG E24 父端点开始，保持共同数据、点采样、监督、初始父参数和图像顺序；新阶段起点同时重建 AdamW，阶段内部连续恢复参数、优化器与私有顺序。两个头分别40,497与42,698参数，容量因素尚未排除。原24轮＋新24轮不能称为连续48轮。

第9—16轮GPU原执行3518、独立CPU原执行75870均实际退出0。ROOT原读取69384实际0，在任何科学输出读取前通过全部99个常规成功及9个保留失败的RAW终止门，再核验完整10,029项来源、CPU stdout唯一资格引用与报告/详情连接。失败数量仍9；无自动重试。

两臂各累计26,576次AdamW更新；模型/优化器FP32形状与有限性、每参数step、完整两段配对trace、16个741图私有顺序及最终随机状态通过独立检查。此范围不包括逐梯度优化器轨迹重放或每实例GPU端点损失完全相等。

## 训练集固定端点，不是验证AP

| 臂 | GPU FIT BCE | CPU FIT BCE | 平均绝对差 |
|---|---:|---:|---:|
| BASE_CONTINUATION | 0.3255756508923356 | 0.32557564988763144 | 1.0047e-9 |
| MEMORY_ADDON | 0.32129877591236683 | 0.32129877614929764 | 2.3693e-10 |

第16轮MEMORY相对BASE的训练端点损失降低约1.31%；第9—16轮并非逐轮都占优，不能由训练损失宣称掩膜AP涨点、泛化、机制成立或TGRS水平。首次FIT实例的训练后校正最大绝对值0.0008997917，1024点均非零；点置换误差0，256个LOG域外点严格继承原生。该行为检查仅覆盖该实例。

新增CPU平均损失数值兼容协议是在GPU完成后、CPU运行前独立锁定，atol=rtol=1e-4。不是GPU训练前登记的效用判据；范围限同一缓存输入的末端平均值。旧诊断容差未改变。

## 下一步已执行

第17—24轮GPU原14941已单次启动，沿用冻结的 continuation GPU v2 源码。继续同一新阶段模型/AdamW状态与私有顺序，不重置优化器。准入继承10,039项来源、99个常规成功和9个保留失败共108身份。CPU[0,1]/2线程、CUDA0、1800秒、4096MiB分配器、8192MiB整卡观察、最低16GiB RAM和无自动重试规则保留；GPU整卡观察不是每进程显存认证。末端预定各累计39,864次新增更新，未实际结束前不报告完成。

只在固定第24轮比较完整DEV187图/380 GT的官方COCO mask AP，保持全部100个query和原生分数，不用第8/16轮AP择优。第一组停止/追加种子规则见[预锁协议](N14_CONTEXT_MEMORY_SINGLE_MODULE_EXPLORATION_PROTOCOL_20261014.md)。当前没有N14 DEV/AP、正式TEST、跨数据集、重复种子、容量或直接先例机制证据。

## 同步研发

[独立CPU continuation草稿](code/N14_context_memory_continuation_CPU_qualification_draft.py)与[合同](N14_CONTINUATION_CPU_QUALIFICATION_CONTRACT_20261014.md)支持stage16/24累计检查；stage16已实际执行，stage24未执行。

[固定DEV同GPU输入adapter](code/N14_fixed_DEV_same_GPU_input_adapter_draft.py)及[说明](N14_FIXED_DEV_SAME_GPU_INPUT_ADAPTER_20261014.md)已通过独立静态审阅，仅是普通UNBOUND helper，完整runner/gate/sink及实际AP尚未完成。它保存实际GPU采样特征与原生L，CPU据同输入独立重算两臂mask，避免此前CPU/GPU全图特征渲染差异影响解释。相同输入仍不能保证近零sign一致；如sign/RLE核验失败，保留失败而不事后放宽。

GLDet的完整实例分割能力、模块消融、500轮配方及移植约束见[原文评议](GLDET_PRIMARY_PAPER_REVIEW_AND_SEGMENTATION_TRANSFER_20261014.md)。UCMNet借鉴只限上下文检索，不等于完整复现或SAR专属创新。
