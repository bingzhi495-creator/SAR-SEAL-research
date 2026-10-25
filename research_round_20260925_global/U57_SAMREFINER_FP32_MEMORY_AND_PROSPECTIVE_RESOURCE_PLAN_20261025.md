# SAMRefiner FP32显存推导与独立试跑资源方案

2026-10-25。证据为固定作者提交的源码静态检查与算术推导，尚无完整模型执行或实际OOM，不是精度结论。主实验PLAIN/FULL的4GiB allocator、既有资源与验收规则均不改变。

## 先排除人为资源拦截

[作者ViT-H构造](https://github.com/linyq2117/SAMRefiner/blob/4bb7f95738c7a2e805f189210fb09e02a2197557/segment-anything/segment_anything/build_sam.py#L14)设置宽度1280、32块、16头、1024输入和16patch，产生4096个图像token。仅各块qkv/proj/MLP权重的FP32下界为32×12×1280²×4字节，即2.34375GiB，尚未计其余参数。一次全局attention的[16,4096,4096] FP32张量为1GiB。[显式attention与非原位softmax](https://github.com/linyq2117/SAMRefiner/blob/4bb7f95738c7a2e805f189210fb09e02a2197557/segment-anything/segment_anything/modeling/image_encoder.py#L231)在普通eager路径有新旧大张量重叠；权重加两张attention的保守条件下界已为4.34375GiB。因此不能用主实验4GiB限额下的OOM充当外部方法无效或精度更差的证据。

[作者精修入口](https://github.com/linyq2117/SAMRefiner/blob/4bb7f95738c7a2e805f189210fb09e02a2197557/sam_refiner.py#L92)只编码一次图像，4MiB embedding在默认5轮共享，并非为100个mask编码100次。但[decoder](https://github.com/linyq2117/SAMRefiner/blob/4bb7f95738c7a2e805f189210fb09e02a2197557/segment-anything/segment_anything/modeling/mask_decoder.py#L126)对N100提示产生各400MiB的稠密prompt、重复image和PE，以及约800MiB上采样特征；三mask到1024²的插值约1200MiB。不同临时张量不必全部同时存活，不能相加假装实测峰值，也不能简单乘5轮；实际首个失败位置与峰值仍待执行。

## 新的独立前瞻方案，尚未绑定

保持完整作者ViT-H、FP32、无AMP、原100候选及默认5轮。拟为外部单图资源/功能试跑单独设置16GiB PyTorch allocator，CPU0/1及各线程2，RSS上限16GiB、运行可用RAM下限16GiB、1800秒、最多一次。启动前要求MemAvailable至少48GiB，即16GPU+16RSS+16余量的保守和；它不是测得的峰值保证，RSS与统一内存可能有重叠，CUDA上下文/驱动也不能靠allocator字段完全覆盖。

资源cap与启动阈值必须由未来SPEC明确给出，无默认自动放宽。仅在原科学任务全部回收、完整权重与作者源码快照重新核验、新的Root独占资格及外部监督实际具备后执行。该方案不替换主方法配方，不减少基线精度、模型、N100或迭代数，也不把不同预训练与计算成本隐藏为相同条件。当前权重仍在有界长传输，16GiB是否足够尚未实测。

## 实际试跑还需什么

[候选驱动](code/U57_SAMREFINER_REAL_ONE_IMAGE_TRIAL_DRAFT_20261025_01/README.md)仅通过3项纯标准库元数据测试、语法检查及独立静态审阅，未加载模型。它在构造模型前要求独立Root核验的完整权重/归一化阶段证书/新lease，保留构造、载入、转CUDA、encoder/prompt/decoder Python pre/post及完整作者返回的持久记录。Python post-hook不等于GPU已经完成，decoder次数不等于已完成精修轮数；另需CUDA同步、真实原工具/SSH/worker退出与外部进程组为空。

首图已核当前PIL→PNG→OpenCV像素及100原候选，但当前PIL通过原几何重建rgb400与历史缓存是否相同尚未测量。未来应量化这一差异，不能把资源/功能试跑写成公平精度对比。完整187评价、AP、跨集、稳定性与创新结论均未完成。
