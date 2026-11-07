# RF-DETR E2 原生断点接续实测

2026-11-07 在 DGX Spark 上，以已独立核验的原生 E1 归档运行一次 E2 更新探针。沿用 RF-DETR Seg Nano 1.12.0、576 像素、seed 5201、BF16、741 张 FIT 图和 100 轮调度。使用原生 `Trainer.fit(ckpt_path=..., weights_only=True)`，起点为零基 epoch 1、全局更新 186。实际访问 4 张不同 FIT 图，完成四个微批次和一次 AdamW 更新。全局更新、LambdaLR 和 EMA 到达 187 后退出。 [续训工作体](code/RFDETR_NATIVE_E2_RESUME_WORKER_CANDIDATE_20261107/e2_worker.py) · [Lightning 固定版本](https://github.com/Lightning-AI/pytorch-lightning/blob/2.6.6/src/lightning/pytorch/trainer/trainer.py#L478)

工作体计时约 9.31 秒，监督进程约 20.77 秒。SSH 和原工作进程均退出 0，独立核验同一启动周期内的进程组已退出，原断点字节保持一致。新输出目录无断点。此次没有验证、测试或 AP，也没有可用于从 187 再续训的归档。

接续实验证明当前配方的原生断点可恢复并完成下一次更新。固定版 RF-DETR 在 `on_fit_start` 会再次设种子，该调用晚于单设备的归档回调恢复。采样器与迭代器初始化也可能先行消耗随机状态。为获得清晰的长训证据，下一条强基线从同一 COCO 初始权重在同一 100 轮调度下连续运行。未来如需证明续训等价，单独配对核对采样顺序、增强与更新后的参数。 [RF-DETR 原生设种源码](https://github.com/roboflow/rf-detr/blob/639eac4555ceb89ef399ccda3fde716bbcfe34cb/src/rfdetr/training/module_model.py#L585-L599) · [Lightning 归档恢复顺序](https://github.com/Lightning-AI/pytorch-lightning/blob/2.6.6/src/lightning/pytorch/trainer/trainer.py#L943-L970)

原生推理的 100 行分类及掩膜转换接口另已通过三个合成检查。真实 EMA 模型、图像和编码器的端到端验证属于后续实验。 [导出候选源码及范围](code/RFDETR_NATIVE_E1_EXPORT_CANDIDATE_20261107/README.md) · [实测安全汇总](results/RFDETR_NATIVE_E2_ACTUAL_ACCEPTED_SAFE_20261107.json)
