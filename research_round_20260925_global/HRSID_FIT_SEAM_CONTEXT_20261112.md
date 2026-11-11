# HRSID 同一船舶的近远裁片边缘对照

研究问题是，同一完整船体在靠近裁片边缘时，原生候选匹配与轮廓是否变差，且变化是否超过同来源、同位移的两侧远边缘对照。HRSID 由母幅标注和重叠裁片构成，先检查共享原始图像证据。已有研究对重叠裁片做特征一致性，ReSAM 对遥感实例做弱强视图对齐，遥感检测也已有切片与全图融合。这里先用同一真实 SAR 回波、整数格点位移及独立双侧远边缘对照验证问题。 [HRSID 作者论文](https://xplorestaging.ieee.org/document/9127939/) · [重叠裁片一致性先例](https://openaccess.thecvf.com/content/ICCV2025/papers/Cai_Unsupervised_Histopathological_Image_Semantic_Segmentation_with_Overlapping_Patches_Consistency_Constraint_ICCV_2025_paper.pdf) · [ReSAM](https://arxiv.org/abs/2511.21606) · [遥感切片检测对照](https://www.mdpi.com/2072-4292/18/20/3587)

原生 FIT 2,937 张中有 2,867 个明确裁片矩形，4,112 个正重叠的整数位移候选对，来自 108 个来源前缀。首次按固定哈希抽取 36 个不同来源，每组一对。72 个 PNG 在重叠区的 366 万个 RGB 像素全部相同。单独开启标注后，232 条原标注中只有 14 个完整船体配对来自 8 组，近边缘≤16 像素与远边缘≥64 像素的配对仅 3 个、3 组。低于原门槛，第一次试点停止，未前向。全部位移在原图 800、模型输入 576、48 补丁格及 144 掩膜格均为整数。绝对位置嵌入仍变化。 [像素实测](results/HRSID_FIT_OVERLAP_PHASE_CENSUS_ACTUAL_SAFE_20261112.json) · [第一轮标注准入](results/HRSID_FIT_OVERLAP_TARGET_ADMISSION_ACTUAL_SAFE_20261112.json)

另锁定按标签几何富集的独立 FIT 试点，排除初始 36 个前缀。用两视图完整轮廓在母幅坐标下的连续 IoU≥.95、双向匹配与至少 .10 次优间距。56 个来源各有一个近远边缘配对，其中 40 组另有同来源、同位移绝对值、双侧距边≥64 像素的配对。冻结名单内 89 个不同裁片对、176 个不同 PNG 通过像素核验，共享区 1,599 万个 RGB 像素逐值相同。初始随机试点的负结论保留。来源前缀来自模型训练 FIT，标签只用于建立预先选择的诊断对象。 [几何富集实测](results/HRSID_FIT_OVERLAP_ENRICHED_CENSUS_ACTUAL_SAFE_20261112.json)

模型诊断已预锁只使用同时具备两类配对的 40 个来源。预计一个固定模型对 144 个唯一图块各前向一次，主评分保留每来源两类配对，共 80 个完整船。前向不接收标注；独立闭合后按原 COCO .50 一对一匹配，漏检记零，对比连续 IoU、IoU≥.75 达标数，以及双侧远边缘的自然变化。近远条件的差值仍可能包含绝对位置作用，只有通过实测门槛才进一步检验同位置上下文干预或等成本训练。 [固定模型诊断协议](code/HRSID_FIT_SEAM_CONTEXT_20261112/PROTOCOL.md)
