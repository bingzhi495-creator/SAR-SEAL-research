# RF-DETR 连续强基线与首个 E24 端点

预先固定 RF-DETR Seg Nano 1.12.0、公开 COCO 初值、576 像素、seed 5201、单 GPU BF16、741 张 FIT 图、批量 1、梯度累积 4。学习率调度仍按 **100 轮、18,600 次更新和 558 次预热**建立。首个连续运行的目标为 E24 **4,464 次更新、17,856 个微批次**，保留完整原生归档。训练不读 DEV 影像，不执行验证、AP 或 TEST。先做 E24 端点，随后以固定的整集 DEV 评价检查可定位误差。E24 的预算与旧主线固定 E50 不同。 [固定配方](configs/RFDETR_SEG_NANO_BASELINE_576_E100_20261107.yaml) · [工作体及合成检查](code/RFDETR_NATIVE_UNINTERRUPTED_E100_WORKER_CANDIDATE_20261107/e100_worker.py)

E2 原生接续已验证单步。为避免继承未验证的 RNG 与数据轨迹，E24 从同一 COCO 初值开始，在同一次原生 `fit` 内执行。归档后核验真实更新数、EMA、调度、输入人口和退出。另有两图 EMA 导出候选，可检查原始后处理与 COCO RLE 的端到端接口，结果范围限于接口。 [E2 实际证据](RFDETR_NATIVE_E2_ACTUAL_ACCEPTED_20261107.md) · [双图导出候选](code/RFDETR_NATIVE_E1_EMA_PILOT_WORKER_CANDIDATE_20261107/README.md)

实际执行已启动。独立同一启动周期观察核对到一个与本次请求相符的 DGX 训练进程，原始进程返回及 E24 断点待核验。按 E1 实测时长估计约 93 分钟，实际时间以运行记录为准。
