# RF-DETR 原生 SSDD 数据入口实际结果

2026-11-07。在主实验实际接收到36,352次更新后，使用已通过离线依赖、原生 CLI 和合成编解码检查的同一 RF-DETR 1.12.0 独立环境，原生 RFDETRDataModule.setup("fit") 实际完成。工具、SSH 和子进程均返回0，原父进程与子进程组已独立观察闭合。随后主实验接续。

- 原生 train、valid 各自的全部图像 ID 与冻结的 FIT741、DEV187 一致，训练类别映射为 COCO 船舶1 → 训练标签0 → COCO 船舶1。DEV 仍为内部开发验证集。
- 每组读取事前固定的一个正例。原图以原始尺寸解码，原生实例掩膜与按原 COCO polygon/RLE 直接栅格化逐像素一致。两张图均经过官方 576 变换，得到有限的 3×576×576 图像张量、合法的归一化框和非空布尔掩膜。两组样本均从原始一个实例保留为变换后一个实例。该掩膜对照核验的是数据读取与变换链，使用同一 pycocotools 算法。
- 两份原标注、两张固定图像和布局凭据在执行前后均通过字节及 SHA-256 检查。审计只观察到预期被拒绝的 IPv6 回环探测。实际数据读取限制在已批准文件。

适用范围为全部划分 ID、类别映射和两个固定图像的读取与变换。全体 JPEG 原字节由[布局验收](RFDETR_SSDD_NATIVE_INPUT_LAYOUT_ACTUAL_20261107_01.md)单独覆盖。模型前向、真实优化器更新、完整训练和 SAR AP 仍需实际实验。 [数据探针](RFDETR_REAL_SSDD_DATA_QUALIFICATION_DRAFT_20261107_01/probe.py) · [绑定入口](RFDETR_REAL_SSDD_DATA_QUALIFICATION_DRAFT_20261107_01/entry.py)
