# HRSID 原生 COCO FIT 目录资格

原生模型已在两张实际 FIT 图上确认 164 查询宽度可覆盖 163 个目标。为完整 FIT 首轮训练准备独立 COCO 布局。仅从已核验的 HRSID 官方 PNG 归档复制派生 FIT 视图选定的 2,937 个成员，保留 9,260 条标注。只为原生加载器把 `images[].file_name` 的后缀统一为实际的 PNG，其他字段逐项重读。验证原归档目录和成员，输出 PNG 字节与原成员一致，并保存私有逐图像素指纹。训练布局只有 `train/`，不构建 DEV 和 TEST。 [原生 CUDA 实测](HRSID_REAL_FIT_NATIVE_MATCH_PROTOCOL_20261109.md)

四项合成测试通过。实际提取与独立核验待执行。下一阶段按固定 576、单图批量、累计 4、13×164 查询和原生 BF16 记录 2,940 次访问及 735 次实际更新，包括原生累积对齐的 3 个补采索引。用较长调度范围冻结首轮资格，保留真正的优化器、EMA 及检查点状态，另外报告 E24 迁移初始化与 COCO 初始化的适用范围。当前阶段只有训练数据资格。 [布局源码](code/HRSID_FIT_NATIVE_COCO_LAYOUT_20261109/materialize_fit_native.py) · [合成测试](code/HRSID_FIT_NATIVE_COCO_LAYOUT_20261109/test_materialize_fit_native.py)
