# N26 完整匹配R训练输入捕获：执行前源码

状态：源码冻结，最终同团队静态审阅未发现确定启动阻塞；真实GPU执行尚未开始。5项标准库scope/配对/尾块及ABI检查、相同质量模块7项合成检查通过，不代表模型或真实数据已经通过。

固定741图/1661真实匹配，分两片[0,371)、[371,741)串行实际运行。每图一次原冻结teacher loss观察，仅重放原matcher取得配对；全100 e0/class/native L对旧缓存逐字节校准，旧抽样F中心校准。沿原teacher训练SPEC构造610张量/451参数，强R原E24头8张量/40497参数，权重及grad状态保持；不调用旧过时RAW计数入口。

实际重新取得teacher F，对匹配query完整Ω升序、2048分块和真实尾块计算FP32 L+u严格>0；域外保留P，K0不改。抽样F仅用于校准，不能替代完整R。低质和empty匹配保留，未匹配query不伪标0。

小型输入为e0+预测mask自身bbox裁剪784特征；目标为完整原图mask与原matcher GT的IoU，GT不进入特征。私有AUDIT含压缩完整R RLE和摘要，回读校验后供独立CPU验证；不保存新全量F/L，不发布RLE、目标、缓存或权重。真实任务终止/Root文件接收和独立数值验证分开，后者未通过不得训练。

两项未执行代码阶段的ABI错误已修：Control实际字段execution_admission_ref；旧N07 packet仅有source/spec/phase/teacher_checkpoint，无GPU_admission。纯标准库AST表达式回归覆盖这些真实接口，原冻结SOURCE未改。

CPU0/1、线程2、冷CUDA后allocator4GiB、整卡观测8GiB、RSS/可用RAM各16GiB、每leaf1800秒、无自动重试。实际CUDA清理包含cuBLAS及allocated/reserved零检查。仍依赖未发布的既有模型/图像/绑定，不是一键复现包。

[接口合同](CONTRACT.md)、[原训练计划](../../../refine-logs/N26_QUERY_MASK_QUALITY_BASELINE_20261017_01/EXPERIMENT_PLAN.md)、[数据阶段](../../../refine-logs/N26_QUERY_MASK_QUALITY_BASELINE_20261017_01/DATA_CAPTURE_PLAN.md)、[当前实际进展](../../N26_DATA_CAPTURE_PROGRESS_20261017.md)。
