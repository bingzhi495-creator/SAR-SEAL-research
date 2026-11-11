# Mask RT-DETR 固定端点的逐图导出与独立评分

作者高层 `Trainer.predict()` 在返回结果中保留整批数据的原始 NumPy 预测。按 705 图、每图 100 条、800² 和 32 位掩膜算，单臂的未压缩掩膜约 168 GiB。这里继续使用作者的单图 TestReader、模型、后处理和 `get_infer_results`。每图检查原始 COCO ID、预测分数顺序和 800² 掩膜尺寸，成对写入压缩框／掩膜记录后释放稠密预测。完整写出后发布流文件摘要。独立评分进程核对原退出、顺序、行数和摘要，向 COCO 仅传掩膜字段。图内并列候选不重新排序。 [作者高层推理](https://github.com/PaddlePaddle/PaddleDetection/blob/b25522a0f4bde8c80603f3ba5e3472059972e3b5/ppdet/engine/trainer.py#L1215-L1235) · [作者后处理](https://github.com/PaddlePaddle/PaddleDetection/blob/b25522a0f4bde8c80603f3ba5e3472059972e3b5/ppdet/modeling/post_process.py#L614-L636)

## FIT 真实出口验收

- 使用已闭合的原生八步训练 EMA，作者加载后 646 个参数与缓冲逐位相同。沿用前次验证固定三张 FIT 图和相同原始 ID 顺序，实际 CPU 前向、导出 300 条候选。其中 6 条非空，保留其余空掩膜。GPU 和标注在此阶段均未参与。
- 原导出工作体约 6.930 秒，其中同步边界定义的 CPU 前向累计 6.362 秒。独立评分从闭合预测流和原 FIT 标注重新计算，三张图有 16 条标注。在主口径与仅排除空栅格的口径中，12 项 COCO 摘要和 AP90 都与直接 PyCOCO 逐值一致。当前三图没有空 GT，正式开发集的四条空 GT 处理仍依照已锁定协议。
- 三个接口测试覆盖完整配对顺序、并列分数、截断及变更摘要。首次独立评分时，FIT 标注索引超过给定读取上限，未打开标签内容。修订使用既定索引的真实 30,076,663 字节上限，同一导出与同一标注通过。两阶段最终均实际退出 0。

[逐图导出及测试](code/MASK_RTDETR_FIXED_ENDPOINT_STREAM_20261111) · [闭合 FIT 导出](results/MASK_RTDETR_FIXED_ENDPOINT_STREAM_FIT_EXPORT_ACTUAL_SAFE_20261111.json) · [闭合 FIT 评分](results/MASK_RTDETR_FIXED_ENDPOINT_STREAM_FIT_SCORE_ACTUAL_SAFE_20261111.json)

## 固定 DEV705 运行边界

正式开发评价只使用预锁的 HRSID DEV705 图像清单和原标注。两臂都必须先独立验收原 E24 退出、完整计数和 EMA，导出阶段保持标注隔离。每臂保留全部 70,500 条作者候选。两个导出会话分别验收后，CPU 使用已有公共 COCO 评分函数，主口径保留 1,782 条原始标注，敏感性只排除预登记四条空栅格。报告每臂前向、导出和训练实测成本。固定 E24 对照使用 Paddle CUDA FP32，当前真实验证仅涉及 CPU FIT，本机 CUDA 导出与 CPU 双臂评分入口已通过语法和字段核对，CUDA 前向及开发集端点评价待实际运行。 [预锁正式评价](MASK_RTDETR_HRSID_FIT2937_E24_AUTHOR_PROTOCOL_20261111.md) · [评价接口前序资格](MASK_RTDETR_HRSID_NATIVE_ENDPOINT_EVAL_FIT_QUALIFICATION_20261111.md)
