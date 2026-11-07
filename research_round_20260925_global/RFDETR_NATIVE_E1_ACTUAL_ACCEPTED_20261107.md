# RF-DETR 原生 BF16 E1 已验收

2026-11-07 在 DGX Spark 上，RF-DETR Seg Nano 1.12.0 使用公开 COCO 初始化，seed 5201，576 像素，单卡 BF16，批量 1，累积 4。固定 SSDD 内部训练集 741 图。原生调度保持 100 轮、18,600 次计划更新、558 次预热。本次运行只完成首轮。内部 DEV187 用于初始化数据模块，训练期间禁用验证和测试。 [原始 RF-DETR 训练源码](https://github.com/roboflow/rf-detr/blob/639eac4555ceb89ef399ccda3fde716bbcfe34cb/src/rfdetr/training/trainer.py) · [固定配置](configs/RFDETR_SEG_NANO_BASELINE_576_E100_20261107.yaml)

| 实测项目 | 结果 |
| --- | ---: |
| 完成 epoch / 优化器更新 | 1 / 186 |
| 微批次 / 独立 FIT 图 / 补齐访问 | 744 / 741 / 3 |
| 增强后实例访问 | 1,441 |
| 分类和掩膜参数的梯度观察 | 各 186 次，最大范数非零且有限 |
| 原生训练阶段 | 231.49 秒 |
| 监督进程 | 247.12 秒，981 次资源采样 |
| 原生 checkpoint | 543,063,540 字节，CPU 安全加载回读及独立字节核验通过 |

断点检查覆盖原生模型参数、EMA、AdamW 状态、学习率游标、训练循环与四类随机状态记录。分类器指定参数的最大绝对变化为 0.005122，原生和 EMA 分类器指定参数差值为 0.000369。该检查限于已指定的参数证据。原始 SSH 和子进程退出均为 0。独立核验确认同一启动周期内对应进程组已退出，无资源越限。

第一次执行在 Lightning 回调的属性访问处退出。首次修正后又在旧依赖指纹处退出。两次均为 0 次更新。最终执行使用新独立请求，检查完整依赖指纹后完成 E1。 [两次原始失败及修正](RFDETR_NATIVE_E1_FIRST_ATTEMPT_ACTUAL_FAILURE_20261107.md)

下一步用同一断点走原生恢复接口，核对 E2 起点、四个微批次和第 187 次 AdamW、调度及 EMA 更新。E1 未运行验证、AP 或 TEST，恢复和不间断轨迹等价性仍待独立验证。固定 50 轮的 PLAIN/FULL 结论不使用本次训练计数。

E2 候选沿用 100 轮调度和原生 `Trainer.fit(ckpt_path=...)`。观察到第四个微批次、一次实际 AdamW 返回及更新 187 后设置停止。候选要求此阶段不生成新断点，不执行验证或测试。三项合成检查通过，真实断点接续尚未执行。 [E2 候选源码](code/RFDETR_NATIVE_E2_RESUME_WORKER_CANDIDATE_20261107/e2_worker.py) · [合成检查](code/RFDETR_NATIVE_E2_RESUME_WORKER_CANDIDATE_20261107/test_e2_worker.py)

[安全汇总](results/RFDETR_NATIVE_E1_ACTUAL_ACCEPTED_SAFE_20261107.json) · [长训与诊断计划](RFDETR_BASELINE_AND_DIAGNOSTIC_PLAN_20261107.md)
