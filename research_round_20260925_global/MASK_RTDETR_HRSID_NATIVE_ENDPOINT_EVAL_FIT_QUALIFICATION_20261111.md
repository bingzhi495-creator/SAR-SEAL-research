# Mask RT-DETR 固定端点评价的 FIT 接口实测

固定 HRSID FIT2937 长训方案指定 E24 EMA 权重、作者 CUDA FP32、640² 输入、800² 原图掩膜、每图原生前 100 个候选和固定 DEV705 评分。以下三项实测处理权重、图像身份和面积分层这三个接口。训练、模型导出和分割评分分别验收。当前记录均基于 FIT 或关闭的 FIT 产物，正式评分尚待双臂 E24 退出与资格确认。 [固定长训及评价协议](MASK_RTDETR_HRSID_FIT2937_E24_AUTHOR_PROTOCOL_20261111.md)

## 三项已闭合实测

| 实测 | 结果 | 适用范围 |
| --- | --- | --- |
| 作者原生读取器与导出 | 原生读取器 ID 0、1、2 分别映射回原 COCO ID 3641、0、1832。三次独立批次前向各输出 100 条，300 张掩膜均可逐像素解码，非空 99、97、76。此前独立单图导出的 100 条框与掩膜在这次完整复现。 | 固定 CPU 官方 COCO 单类转移，批量 1，FIT 图像清单按 ID 排序后预选末位、首位、中位。未加载标签或运行 AP。 |
| 作者评价权重加载 | 关闭的 A 臂 FIT 八步短窗中，直接调用作者 `load_weight` 加载 `.pdparams` 后，646 个参数与缓冲逐位等于 EMA。与旁置的 `.pdema` 在线状态相比，555 个张量不同。 | 同一作者 CPU 评价载体和完整单类状态。未加载优化器、执行模型前向或打开标注。 |
| 独立多图评分 | 29 张 FIT 图、2,900 条输入的 12 项 COCO 摘要及 AP90 与直接 PyCOCO 逐值相等。颠倒图块而保持图内顺序，三类精度、召回与分数张量也一致。误将框字段合入分割评分时，705 条面积分层变化，尺度摘要的最大偏移为 1.699 AP 或 AR 点。 | 一张作者原生导出，加 28 张 RF-DETR 真实 FIT 压力输入。用来测评分接口。 |

[多图原生导出实测](results/MASK_RTDETR_AUTHOR_FIT_MULTI_IMAGE_EXPORT_ACTUAL_SAFE_20261111.json) · [EMA 加载实测](results/MASK_RTDETR_AUTHOR_FIT_EMA_LOAD_ACTUAL_SAFE_20261111.json) · [跨图评分和等分边界](MASK_RTDETR_CROSS_IMAGE_SCORE_STRESS_RESULT_20261111.md) · [作者权重存取](https://github.com/PaddlePaddle/PaddleDetection/blob/b25522a0f4bde8c80603f3ba5e3472059972e3b5/ppdet/utils/checkpoint.py#L342-L355)

首次外层启动缺少进程观察依赖，未创建科学执行或启动子进程。使用已具备该依赖的同版本宿主后，原多图工作体实际用时 12.210 秒，原等待约 12.890 秒，作业树闭合且两级退出 0。原等待摘要 SHA-256 为 `fc0f6302c4a646104afc7acaa1dcd611905c8c09b3f36a543765c228ae4e8f3e`。跨图评分的两次失败分别修正输入摘要抄录和 NumPy 整数写出，最终退出 0。合并前保留原生逐图候选顺序。同图同分的不同掩膜只凭类别、分数和图像无法反推出原配对，因此使用同一次原子导出，并保留原候选顺序。 [读图身份映射及测试](code/MASK_RTDETR_AUTHOR_READER_IDS_20261111) · [COCO 的面积和稳定排序](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py#L327-L352)

## 长训可回答的科学问题

PyTorch 的中心最近邻和 MONAI 的组合变换后单次标签重采样已有先例。RSI4EO 作者在 HRSID 局部留出上报告，修正推理细化的半像素偏移使 mask AP 从 56.4 到 60.7。CoFailMed 附录 B.3 测得约 0.47 输出像素的质心偏移，并用中心最近邻重训四项医学分割任务的 1×1 头。MatchDown 则在整数步幅下考察粗标签的实例存留。当前 Mask RT-DETR 配对要检验的是动态非整数缩放的**训练标签干预**。两臂共同监督实例和原生增强相同，掩膜变化仍可能影响模型内部的匹配。先用 E24 自然候选的固定原网格 AP 衡量效用，再解释它能否跨作者载体、训练种子和连续多边形口径复现。中心最近邻本身属于既有技术。 [PyTorch 源码](https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/native/UpSample.h#L290-L343) · [MONAI 论文](https://doi.org/10.1016/j.cmpb.2024.108422) · [RSI4EO 附录](https://arxiv.org/html/2610.18845v2#A1.SS3) · [CoFailMed 附录](https://arxiv.org/pdf/2610.41790v1#page=23) · [MatchDown](https://arxiv.org/html/2610.39880v1#S4.SS3)
