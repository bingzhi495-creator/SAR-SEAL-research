# HRSID 原生 RF-DETR 首个完整 FIT 轮次

使用已独立读回的原生 COCO FIT 目录。2,937 张官方 PNG 和 9,260 条有效训练标注保持固定，正式评价在单独协议中执行。实测每组 164 个查询覆盖稠密图全部 163 个实例，两种宽度下空图反向均完成。首轮记录固定宽度下全量训练流的输入、更新和资源。 [数据资格](HRSID_FIT_NATIVE_COCO_LAYOUT_PROTOCOL_20261109.md) · [原生 CUDA 诊断](HRSID_REAL_FIT_NATIVE_MATCH_PROTOCOL_20261109.md)

| 项目 | 固定条件 |
| --- | --- |
| 初始化 | SSDD E24 在线权重严格加载。按 13 组逐组保留原 100 行，新增 64 个零参考点和用独立种子 5201164 生成的查询特征。HRSID 训练从新 AdamW、调度器及 EMA 时钟开始。 |
| 模型和输入 | RF-DETR 1.12.0 Nano 分割配置，576，训练用 13×164 查询，`num_select=100`。原生 BF16 自动混合精度、单图批次、累积 4、原生增强、掩膜点监督及 IA-BCE。 |
| 原生调度 | 固定 100 轮及 73,500 个计划更新，3 轮预热至 2,205 更新，余弦下降至基准学习率的 0.05。只在完整首轮回调核验后停止，首轮末的学习率系数为 1/3。 |
| 实际计数 | 2,937 个唯一 FIT 图加 3 个由原生加载器按固定规则补采的访问。预定 2,940 次 microbatch、735 次真实 AdamW 返回、735 次 EMA 更新。 |
| 资格和资源 | 单 GPU，6 GiB 分配上限，本机可用 RAM 下限 16 GiB，进程 RSS 上限 10 GiB。每次访问检查目标数、有效掩膜、坐标及有限训练信号。无目标裁剪保留并单计。 |
| 可复现端点 | 只保存一个首轮完整原生检查点。独立按 CPU `weights_only` 回读真实循环和优化器时钟、调度器、EMA、两张宽度为 13×164 的查询表及分类参数变化。 |

由专用 FIT 数据模块保留原生训练增强、对齐、取样和批处理，物理目录无验证或测试输入。验证和评分回调从本轮 FIT 执行剥离，原生训练优化与完整模型归档保持。当前初始化和指标范围属于目标域训练适配。COCO-only 初始化及未来相同宽度、初始化与预算的方案对照需另行固定。 [原生数据模块](https://github.com/roboflow/rf-detr/blob/639eac4555ceb89ef399ccda3fde716bbcfe34cb/src/rfdetr/training/module_data.py) · [Lightning Trainer](https://lightning.ai/docs/pytorch/stable/core-api/trainer)
Windows 原生训练模块加载时会导入官方 `faster-coco-eval`。在独立目录使用 PyPI 的 1.8.0 CPython 3.10 x64 wheel，SHA-256 为 `3eba19bf8e2c0365e698bcc51a88797fef8b13320d333f3accfd7684de59051d`。启动核验 wheel 和解出的全部 30 个文件。原固定环境保持原样，训练不执行评估回调。进度中返回的损失采用 Lightning 梯度累积后的缩放口径。 [官方 wheel](https://pypi.org/project/faster-coco-eval/1.8.0/#files) · [训练工作体](code/HRSID_NATIVE_FIT_E1_20261109/train_hrsid_native_e1.py) · [独立归档回读](code/HRSID_NATIVE_FIT_E1_20261109/verify_hrsid_native_fit_e1_archive.py)
