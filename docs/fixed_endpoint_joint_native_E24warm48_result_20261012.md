# 固定端点主结果与当前配方决策（2026-10-12）

本轮首组固定端点的主评价已完成独立 CPU 核验。预先指定的 E24 native0 参考在同一 DEV、原始实例标注、全 100 query 与同一 COCO helper 下重算，其全部 12 项 COCO 统计与历史结果完全相同。

| 固定结果 | mask AP（百分制） | AP75（百分制） |
| --- | ---: | ---: |
| A_native_official | 64.686168 | 80.792285 |
| B_posterior_official | 66.797068 | 81.735328 |
| 预选强参考 E24 native0 | 67.071057 | 83.842665 |

B−A = +2.110899 AP 百分点；B−max(A,E24) = −0.273989 AP 百分点。按原合同，结束当前配方扩展，不启动该配方的种子 2/3 或 PointRend 风格 C 控制。该结论仅针对本轮配方；不宣称所有残差读出方法失效、核心创新成立、胜过 SOTA 或达到 TGRS 投稿水平。长期研究目标仍未完成。

## 已验证范围

- 本轮 A/B 都来自各自 first24→各自 additional48 的合格状态；每个追加阶段实际 48 轮、35568 微步、17784 更新、79728 GT。原生初始化 E24 成本另计；阶段重建优化器和随机过程，不是恢复优化器的连续 72 轮。
- 固定内部 DEV：187 图、380 原始 GT。所有每图 100 query、原始 GT crowd/ignore 与空预测保留；此 DEV 历史上已接触，不能称盲测或正式官方 TEST。
- 当前六项控制的独立 CPU 几何重建与 GPU 导出二值掩膜全部一致。主要 B 的 CPU 重建 AP 为 0.6679554649701489，GPU 导出 AP 为 0.6679706764420889，差 −0.000015211471939946541；保留该分数数值差异，不宣称 GPU 位精确重放。决策使用预锁 GPU 导出主列。
- E24 本次只重算固定导出 COCO；没有重新运行模型、重构掩膜几何或重算 mask-quality。它补齐同一固定参考的本轮 CPU 执行凭据，不创造未见测试证据。
- 可复用 query 的 best100 GT 几何统计仅作诊断，不是一对一召回、部署匹配或唯一因果证明。保持 B-native 几何的评分控制已解释相当部分 AP 增益，不能把主差值全部归因于几何修复。

## 实际成本

| 作业 | 实测耗时 | 范围 |
| --- | ---: | --- |
| 首 24 轮 A/B 成对训练 | 26657.859 秒（约 7.40 小时） | 当前单次作业，不含 E24 初始化 |
| 追加 48 轮 A/B 成对训练 | 52255.906 秒（约 14.52 小时） | 当前单次作业 |
| 固定 GPU 导出 | 214.093 秒 | 两臂导出，不是单模型统一延迟基准 |
| 当前独立 CPU 评价核验 | 1814.171 秒 | 完整核验开销，不是部署延迟 |
| E24 固定导出 COCO 重算 | 2.5 秒 | 不含模型推理与几何重建 |

上述成本不是 DGX 测速结果。下一批仅在本机已合格环境规划执行；DGX 原生环境尚未完成科学资格验证。

## 下一条待检验假设

额外原图均匀点 BCE 可能改变原生优化中的有效监督/梯度配比，使续训预测发生有害漂移；现有残差头可能主要缓解该副作用。最强竞争解释是阶段重建、学习率和续训本身导致退化。FP 与 FN 同时变差并不支持简单“背景让掩膜收缩”的既成结论。

先在预定 FIT 小批次检验监督质量及梯度，再决定是否开展有限四臂诊断。类别平衡 BCE 是已知诊断干预，不作为新核心创新。详细依赖见 [下一批任务](../refine-logs/NEXT_RESEARCH_BATCH.md)。

## 原始证据入口

- 当前独立 CPU 核验：E:\SAR\seal\research_round_20260925_global\multimodal_data_pilot_20260928\goal_continuation_20261005\ROOT_QUERY_PIXEL_JOINT_NATIVE_E24WARM_ADDITIONAL48_EVALUATION_CPU_QUALIFICATION_20261011_02.json
- E24 固定导出重算：E:\SAR\seal\research_round_20260925_global\multimodal_data_pilot_20260928\goal_continuation_20261005\ROOT_E24_NATIVE0_FIXED_EXPORT_COCO_REFERENCE_CPU_QUALIFICATION_20261012_01.json
- E24 本次实际执行记录：E:\SAR\seal\research_round_20260925_global\multimodal_data_pilot_20260928\goal_continuation_20261005\finite_point_cache_execution_20261005_v2\runs\E24_native0_export_COCO_reference_recompute_20261012_01\PARENT_RECEIPT.json
- E24 来源与执行绑定：E:\SAR\seal\research_round_20260925_global\multimodal_data_pilot_20260928\goal_continuation_20261005\finite_point_cache_execution_20261005_v2\runs\E24_native0_export_COCO_reference_recompute_20261012_01\PHASE_BINDING.json
- E24 重算协议：E:\SAR\seal\docs\E24_reference_export_COCO_recompute_protocol_20261012.md

ROOT 在全部 14 项本轮与祖先实际退出记录通过后，核对输入/source、执行绑定与 Q；实际工具 7d46e5 退出 0。完整原始记录和本轮六项控制保留，未事后更换主列。
