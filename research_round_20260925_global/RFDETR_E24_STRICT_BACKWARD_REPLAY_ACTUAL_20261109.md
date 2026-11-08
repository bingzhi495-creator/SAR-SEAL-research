# RF-DETR 首步严格反向复验实测

固定的两次 F 组复验各运行四个真实 FIT 微批次。两个原进程均退出 0，未触发资源保护。每组实际完成 36 次网格采样 CPU 反向、8 次普通双线性上采样 CPU 反向和 4 次严格反向，优化器和 EMA 更新均为 0。F0/F1 的原生前向、匹配、采样点、损失及 RNG 摘要与各自原运行完全相同。 [执行协议](RFDETR_E24_EFFICIENT_ATTENTION_BACKWARD_ACTUAL_20261108.md)

独立 CPU 回读显示，F0/F1 剪裁后梯度在 543 个张量的 33,882,021 个元素上完全一致，相对 L2 为 0，余弦为 1。两组剪裁前全局范数均为 193.11116。严格反向累计 2.625/2.610 秒，两个完整进程 19.453/18.906 秒。 [独立回读结果](results/RFDETR_E24_STRICT_BACKWARD_REPLAY_ACTUAL_20261109_SAFE.json) · [核验源码](code/RFDETR_E24_DETERMINISTIC_BACKWARD_REPLAY_20261108_02/verify_deterministic_replay_cpu.py)

与原 CUDA 全反向相比，两组剪裁后梯度相对 L2 仍差 0.8175% 和 1.0021%。任何后续 F/S 差异均限定在共同的新 Windows 反向路径及同一 E24 起点。当前证据只覆盖固定的首个累积窗口。新的真实 F/S 训练需在首次更新前核对首步梯度，并独立复核训练计数和端点后才可打开固定 DEV 评价。原 F/S 失败记录保留。
