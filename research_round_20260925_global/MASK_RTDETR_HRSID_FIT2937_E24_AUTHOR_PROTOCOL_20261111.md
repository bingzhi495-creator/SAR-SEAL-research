# Mask RT-DETR-S 的 HRSID FIT2937 同预算 E24 预锁

本项先完成一个作者模型外部对照，再验证固定共同监督下的掩膜取样效果能否出现在第二个训练载体。RF-DETR 的既有整数主口径、连续敏感性及严格轮廓失败分别保留。两种架构的查询数量、增强、优化器和推理流程分别遵循各自规定，统一 FIT 样本、访问和更新预算以及后续 COCO 评分规则。 [RF-DETR 固定共同行及评估](HRSID_MASK_PHASE_COMMON_SUPPORT_AND_SSDD_TRANSFER_DESIGN_20261109.md#共同行-e24-的固定-dev705-实测) · [Mask RT-DETR-S 作者配置](https://github.com/PaddlePaddle/PaddleDetection/blob/b25522a0f4bde8c80603f3ba5e3472059972e3b5/configs/mask_rtdetr/mask_rtdetr_hgnetv2_s_6x_coco.yml)

## 已验收的初始化、数据和八步资格

- **官方 COCO 转移。** 作者 PaddleDetection v2.9.0 固定提交 `b25522a0f4bde8c80603f3ba5e3472059972e3b5`，S 型 80 类 COCO 资产 58,728,806 字节，SHA-256 `d71ea6590442d1c3d6118fe0558b4a9fe84283a4832b12dc6d7b5c8cf3e09d9f`。CPU 原生加载复制 646 个状态张量中的 643 个，其中骨干 210、颈部 264、Transformer 169。两个分类头张量及去噪类嵌入因形状变化保留种子初始化。保留正常一类形状，未选择或切片某个 COCO 分类槽。原进程退出 0。 [官方资产](https://paddledet.bj.bcebos.com/models/mask_rtdetr_hgnetv2_s_6x_coco.pdparams) · [单类加载实测](results/MASK_RTDETR_HRSID_COCO_ONE_CLASS_INIT_ACTUAL_SAFE_20261111.json)
- **FIT 数据与计划。** 逐个核对 2,937 张 800² PNG 的编码摘要、长度和图头，共 3,698,804,355 字节。原生 COCODataSet 解析 2,937 个有效样本及 9,260 条框和多边形标注，船类 ID 1 映射到槽 0。首次核验在误把数据工厂当数据对象时退出 1，新的原进程按作者工厂调用后退出 0。未解码 PNG 或访问 DEV、TEST。作者 `drop_last=True` 的完整四图批次为 734。给单卡原生 shuffle 的末批补同轮前 3 个位置，可保留 2,937 个唯一图像的原始排列，并将末批补齐。真实作者 TrainReader 注入后验收 735 批。逐轮比较 24 份原生顺序，合计计划 70,560 次访问、72 次补齐、17,640 次更新。计划测试未执行模型。 [数据与学习率核验](results/MASK_RTDETR_HRSID_FIT2937_AUTHOR_ADMISSION_ACTUAL_SAFE_20261111.json) · [原生读取器复核](results/MASK_RTDETR_HRSID_FIT2937_NATIVE_READER_PLAN_ACTUAL_SAFE_20261111.json) · [采样代码及测试](code/MASK_RTDETR_HRSID_FIT2937_SAMPLER_20261111/)
- **真实 CUDA 短窗。** 相同种子 20261111、初值和 32 次有序 FIT 访问，两臂各完成 8 次 AdamW、调度及 EMA，为每轮 735 批中的前 8 批。两臂的实际图像、框、类别、实例顺序、非掩膜字段及前向前 CPU／CUDA 随机状态经核对。139 条保留掩膜中有 138 条实际不同，原生、未保护中心和最终中心均无空掩膜。两臂各执行 48 次作者 GPU `grid_sample` 回退。原等待约 11.58、11.55 秒且均退出 0。短窗未走到尾批，也未完成一轮或计算 AP。 [真实配对汇总](results/MASK_RTDETR_HRSID_FIT2937_CUDA_SHORT_PAIR_ACTUAL_SAFE_20261111.json)

## 长训及固定评价

1. A 用原作者训练掩膜。B 只在作者批次位图缩放中改变取样。若同一实例的原生和中心掩膜均非空，用中心取样。其他情形沿用原生位图。记录未保护中心的空掩膜和差异、最终实际支持及掩膜变化。图像、框、类别、顺序及原生增强不改。该支持保留规则在八步实测前已锁定。
2. 两臂分别从相同的官方 COCO 转移及种子 20261111 启动。Paddle 2.6.2 CUDA、FP32 张量、AMP 关闭、作者注意力回退，单卡批量 4，读入仅有 FIT2937。沿用作者 AdamW、裁剪和学习率构建器。基础学习率 0.0001，原 2,000 步线性预热。实际构建的第 0、1、1,999、2,000、17,639 步学习率分别为 0.0000001、0.00000014995、0.00009995005、0.0001、0.0001。24 轮均在原里程碑前。
3. 本机先运行 A，只有原进程及档案验收后才接 B。记录两臂真实的逐轮访问、补齐、学习率、优化器和 EMA 计数、GPU/主存资源、端点档案及原等待。每 8 轮保留一次作者档案，E8／E16 仅用于诊断和恢复依据。E24 EMA 为唯一评分权重。超时观测不触发重启。出现资源或实现失败时保留原终态，并区分原数据顺序与重试路径。
4. 实际训练闭合后，固定 HRSID DEV705 的独立影像导出及原标签评分。主导出采用作者 CUDA FP32 推理、固定 640² 读入、原 PostProcess 前 100 和掩膜阈值 0.5，输出原图 800² 的掩膜及原分数。保持原 COCO `maxDets=[1,10,100]`，报告 Mask AP、AP75、AP90、APs／APm／APl、AR100、原始候选数及实际成本。导出和计分分别验收；正式 TEST 继续隔离。固定同一初值、单个配对种子与开发集限制了结论。
5. A 为主外部对照，B−A 检验支持保持取样在当前作者载体与预算的效用。按固定端点完整保留，包括负结果。技术创新由独立方法及先例差异决定，当前实验用作强对照和取样作用的复验。
