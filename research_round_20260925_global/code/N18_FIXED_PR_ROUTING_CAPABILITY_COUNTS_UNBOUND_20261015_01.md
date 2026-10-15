# N18 固定 PR 路由/能力计数合同（UNBOUND）

仅普通开发、纯 Python 标准库；只用手工布尔数据测试，未读取科学文件、GT、权重或运行缓存，未训练、未执行实际诊断，不作资格认证。

- `count_query(P_C, GT_C, A_C, K, selections) -> dict`。输入来自**原固定匹配**的一条 query，在同一个已核验升序 C 的坐标顺序。支持一维 NumPy bool 数组（仅调用 tolist，不导入 NumPy）或元素严格为内建 bool 的 list/tuple；拒绝整数索引、数值 0/1 和二维输入。
- `selections` 恰含 `E24_LOCAL_atmost`、`E24_CENTER_atmost`、`Omega_strong_R`，值为 BOOL[C] 实际物理支持，含 noop；前两者支持数≤K，强R支持数=K。K 为非 bool 的整数，0≤K≤|C|。选择、GT、匹配、坐标、来源和真实支持真实性由外层核验，GT 不得参与构造 S。
- E=|P≠G|（仅 C 内原始错误），δ=(A≠P)，B=|E∩δ|（可修），H=|非 E∩δ|（有害）。每策略输出选中支持、选中 B/H、实际翻转（B+H）、选中 noop，以及分母 min(K,B) 和相应分数；这是已保存动作的计数描述，不是 oracle 上界、IoU/AP 或 AP 因果解释。
- **K=0**：即使 C 非空也必须 A_C=None，三支持全 False；E 仍可观察，B/H、能力 E 分母和预算分母为 None（JSON null/NA），不能推断固定 PR 能力。正 K 但 B=0 的预算分母为 0，分数 NA；E=0 的能力分数同样 NA。
- `aggregate_counts(rows) -> dict` 对 count_query 返回值求 micro 整数和，不平均逐实例比率；`E_all_queries` 包含 K0，而 `PR_capability_E_denominator` 仅正 K。无任何正 K 时能力计数和分母为 NA；逐实例结果由外层保留。6 条无匹配 query 的 GT 不得虚构为空 mask，不输入此函数，由外层单列。

原始固定群体/GT 读法只经普通 N15 GPU 源码核对：`tmp/N15_FIXED_ACTION_OPPORTUNITY_GPU_UNBOUND_DRAFT_20261014_01.py` L385–394 将 fixed.rows 逐值绑定到原 carrier CPU Q 的 details 文件中 `fixed_native_query_GT_pairs['LOG_PR_seed202610120701_E24_fixed_native_score']` 唯一 threshold=.5 的 `GT_rows`：374 个 `native_matched_fixed_query`、6 个 `native_FN`；键为 image_id/gt_id/query_id/status/native/refined，gt_id 是 COCO annotation ID。这里未见固定 population schema 字面锁，不能猜造，strictref 由 ROOT 提供。

同文件 L772–774 的原 GT 解码为 `masks.decode(reader.rle_bytes(coco.annToRLE(coco.anns[pair['gt_id']]))).astype(np.bool_)`，先核 annotation.image_id，再核原形状和非空。新 N17 bundle 不含 GT，亦无强R新 profile。N17 adapter `decode_image(...).query(q)` 的 `common_PR_chunks` 六键为 flat_indices/sampled_F/e0/native_L/GPU_raw_residual/GPU_combined，按升序 C 拼接后 A=(GPU_combined>0)；`read_profile(24, arm, 'atmost')` 给实际选中索引/PRchunks/mask/RLE。坐标 flat=y*W+x，P=(L>0)，Ω=(abs(L)≤2)，K=|Ω|。强R从已资格旧预测 RLE 解码，必须先确认其 Ω 上真实 after-bit 与本次 common-C A 在同坐标**逐点 exact**，才能把同 A 的 Ω 支持计数称作该真实强R路由；否则报告不一致，不能默默用共同 A 替换原强R结果。

验证：一个手算 fixture（E=2/B=2/H=2/翻转4）先因未实现 RED，再 GREEN；同项补 K0、无错误/无可修、micro 分母及严格输入拒绝边界。无实际 SAR 运行。SOURCE/RAW/owner/终态/完整 374/6/GT 身份资格均由 ROOT 外层另行处理。
