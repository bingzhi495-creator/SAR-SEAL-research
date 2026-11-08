# 优化器状态分支的原文源码核验

[固定短程 A 的全查询诊断](RFDETR_WARM_RAW100_COVERAGE_ACTUAL_20261108.md)显示，短热启动后 37 个原有严格覆盖丢失中，36 个在全部 100 个原始查询中也找不到达标几何。[FIT 教师实测](RFDETR_FIT_TEACHER_BOUNDARY_ACTUAL_20261108.md)未通过软边界分支准入。连续 E24/E50/E100 的实际端点已闭合，四端点同 DGX 的固定前向已启动。两轮 A 使用旧 EMA 作为在线初始化、全新 AdamW 与固定短程学习率。它和同运行连续端点的对比同时包含在线权重来源、优化器状态、数据位置和预算差异。

| 直接先例 | 原文与实际源码 | 对本项目的约束 |
| --- | --- | --- |
| Dremov 等，TMLR 2025 | 附录 F 在同任务、同冷却前端点、固定冷却曲线比较保留与重置 AdamW。两种曲线的验证困惑度各仅差 0.01。代码将优化器恢复与 scheduler、EMA 和 worker 随机状态分别处理。 | 同检查点的 AdamW 保留与重置已有明确先例。它在 SAR 上可作为机制控制。必须实际核对优化器 param groups 的逐步学习率。 [作者原文](https://arxiv.org/abs/2508.01483) · [固定检查点加载源码](https://github.com/alexdremov/wsd-cooldown-dynamics/blob/a8fcea75c90a735d96dcad5ad6e1d5e0bb8c1563/src/optim/utils.py#L272-L303) · [数据游标与随机状态入口](https://github.com/alexdremov/wsd-cooldown-dynamics/blob/a8fcea75c90a735d96dcad5ad6e1d5e0bb8c1563/src/optim/base.py#L100-L138) |
| Bu 等，NeurIPS 2024 | 附录 B.3 在 CIFAR100 ViT-Base 从 AdamW 切换到 DP-AdamW 时分别重置一阶矩、二阶矩和步数。重置效果依赖哪一部分改变。 | 全保留对全重置只能识别联合效应；若要解释某一个矩的作用，需要额外控制步数和实际更新范数。作者报告的任务含梯度机制切换。 [原文](https://proceedings.neurips.cc/paper_files/paper/2024/file/ac04e54e0a2d1927d60709019e4e7870-Paper-Conference.pdf) |
| Switch EMA | 作者将 EMA 周期性复制回在线模型，COCO 实例分割有 Mask R-CNN 与 Cascade Mask R-CNN 表格。检测钩子直接执行权重复制。 | 在线参数被改动时，不能把增益或回落归因于 AdamW 状态。当前 A 的 EMA 作为在线初始化需要单列。 [原文](https://openreview.net/pdf?id=rwNzSB3sDt) · [固定作者钩子](https://github.com/Westlake-AI/SEMA/blob/45811267deafdc90d32401d4dab4d84ed2a29d51/detection/mmdet_custom/core/hook/switch_ema.py#L80-L129) |

若连续端点评估及严格几何证据仍支持追踪短启动退化，下一步固定同一个闭合 E24 的在线参数、EMA 参数与计数、每组实际 LR、随机状态及增强数据序列，再比较 AdamW 全恢复与全重置。第一步要验证前向、梯度与应用 LR 在两臂相同，记录更新范数和实际 EMA 次数。RF-DETR 1.12.0 的 EMA 在 fit 开始时会依据训练器 global step 调整恢复保护，优化器 per-parameter step 重置不应重置训练器与 EMA 时钟。统一报告实际预算、原始查询 IoU≥0.90、一对一覆盖、原生 COCO mask AP 与耗时。它回答短热启动回落是否与状态重建有关，最终研究方向需由实际端点及同预算对照支持。 [RF-DETR 固定版本 EMA 源码](https://github.com/roboflow/rf-detr/blob/639eac4555ceb89ef399ccda3fde716bbcfe34cb/src/rfdetr/training/callbacks/ema.py#L308-L312) · [数学、物理与雷达方法来源索引](PRIMARY_METHODS_SOURCE_AUDIT_CONTINUATION_20261108.md)
