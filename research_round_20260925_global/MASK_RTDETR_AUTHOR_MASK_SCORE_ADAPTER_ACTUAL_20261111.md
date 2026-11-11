# Mask RT-DETR 作者原生掩膜评分接口的 FIT 验收

对已结束的作者原生 CPU 导出进行独立评分。固定使用一个 HRSID FIT 图像的 13 个标注和 100 条有序掩膜候选，其中 97 条非空。保持作者导出的类别、分数和顺序，在 COCO 掩膜评分中只送入独立掩膜流，由 RLE 面积决定面积分层。直接运行 PyCOCO 2.0.11 与项目已发布的 COCO 评分路径，12 项标准指标及 AP90 逐值相等。诊断合并框字段的入口时，100 条的面积都会改变，15 条会更换大小区间。该诊断只比较面积，没有运行错误口径的评分。 [原导出](MASK_RTDETR_HRSID_FIT_ONE_CPU_EXPORT_ACTUAL_20261111.md) · [验收摘要](results/MASK_RTDETR_AUTHOR_MASK_SCORE_ADAPTER_ACTUAL_SAFE_20261111.json)

接口与三个单元测试覆盖分数及排序、RLE 尺寸和错误输入。首次隔离调用缺少评分脚本目录，第二次评分检查完成后写出摘要时遇到 NumPy 整数序列化错误。两次均未形成合格摘要。独立修订将诊断计数显式转成 Python 整数，最终退出 0。未重新前向、更新优化器、调用 GPU 或访问 DEV、TEST。单图分数只用于两条评分路径一致性，不作为正式性能。 [接口与测试](code/MASK_RTDETR_AUTHOR_MASK_ROWS_20261111)
