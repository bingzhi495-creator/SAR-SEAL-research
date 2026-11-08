# 统一严格反向路径的 AdamW 状态对照

两次首窗口复验已实测完整梯度一致。新 F/S 实验只检验保留 AdamW 历史是否缓解短程热启动后的严格掩膜覆盖下降。两组都使用刚验证的相同 Windows 反向路径。 [首窗口复验](RFDETR_E24_STRICT_BACKWARD_REPLAY_ACTUAL_20261109.md) · [原 F/S 协议和停止规则](RFDETR_E24_OPTIMIZER_PRESERVATION_PROTOCOL_20261108.md)

- 沿用同一 E24 在线权重与 EMA、历史 EMA 计数 4464、固定 28 组学习率、RF-DETR 原生匹配、损失和点采样。F 的 AdamW 从空状态开始，S 恢复同一 E24 的一阶矩、二阶矩和每参数步数。固定种子 5201、同一 FIT741 数据和增强序列、两轮每轮 744 个微批次、累积 4。两组分别进行 372 次新 AdamW 和 EMA 更新，实际 EMA 最终均为 4836。
- 原生 CUDA 前向保留原确定性设置和 BF16。严格确定性仅包围每次反向，网格采样和普通双线性上采样使用实测通过的 CPU 反向。首次 AdamW 更新前，两组都必须通过四次严格反向、36/8 次 CPU 路由计数及首步剪裁后梯度摘要的精确核对。全程各 1,488 次严格反向，逐形状核对 CPU 路由前后计数。每组墙钟上限 30 分钟，计算预算仍固定为 372 次更新。任何失败保留原终止状态。
- 两个原进程真实退出后，独立核验闭合权重、同起点和数据序列、首步全梯度及更新计数。合格后在同一台 Windows 机器按原协议对独立 E24、F、S 做固定整数 DEV187 评价。S 须同时满足 IoU≥0.90 一对一保留掩膜覆盖比 F 至少多 5 个、mask AP 高于 F 且不低于 E24。固定规则决定停止或进一步复验。
- 首步严格反向与旧 Windows CUDA 反向的梯度存在约 0.82% 到 1.00% 的相对 L2 差异。结果范围限定在本路径、单种子和内部开发集。原失败记录保留，TEST 不参与。

四项固定运行时 CPU 测试通过，同一起点和学习率的依赖检查通过。已启动实际 F/S 顺序训练，F 已完成首步梯度门槛及 32 次更新。固定评价只在独立验收两组端点后启动。 [新训练入口](code/RFDETR_E24_STRICT_OPTIMIZER_PAIR_20261109/run_strict_optimizer_arm.py)

固定 DEV 入口适配了新端点路径。独立 CPU 复核读取原始进程终止回执，重新计算 COCO 全 12 项和严格一对一覆盖。四项合成检查通过，尚无实际评分。 [评价入口](code/RFDETR_E24_STRICT_OPTIMIZER_PAIR_20261109/evaluate_optimizer_pair.py) · [独立核验](code/RFDETR_E24_STRICT_OPTIMIZER_PAIR_20261109/verify_optimizer_eval_cpu.py)
