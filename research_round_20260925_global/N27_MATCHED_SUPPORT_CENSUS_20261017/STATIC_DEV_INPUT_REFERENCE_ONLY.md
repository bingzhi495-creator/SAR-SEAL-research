**范围：** 本文仅保留早期DEV读取接口参考，不是当前N27主诊断。当前预锁只用FIT1661真配对，不读取DEV；本文未执行数值。

<!-- encoding: utf-8 -->
# N27 支持域 census：静态读取契约

状态：**SOURCE_ONLY / NOT_NUMERICALLY_VERIFIED**。仅核对下列公开普通源码；没有打开真实 bundle、预测、GT、权重或原绑定 Source，没有导入 Torch、执行数值或改变 N26。以下是字段契约，不证明未来传入文件已经满足它。

公开代码根：`E:/SAR/seal/github_release_20261011/research_round_20260925_global/code/`。下列 G/A/D/C 均相对此根：

- **G**：`N16_FIXED_E24_ALL100_SCOUT_FULLDEV_GPU_UNBOUND_20261014_05.py`
- **A**：`N16_FIXED_E24_ALL100_SCOUT_FULLDEV_CPU_ADAPTER_UNBOUND_20261014_02.py`
- **D**：`N23_FOURHEAD_FIXED_E24_DEV_SOURCE_ONLY_20261017_02/dev_body.py`
- **C**：同 D 目录的 `N23_FOURHEAD_DEV_CORE_UNBOUND_20261017_01.py`

**1. 物理文件与内部定位。** D:72、83–105 的已存在 consumer 使用 `ctx['rows']`（187图 canonical order）。每 row 至少提供 `dataset_index,image_id,original_shape,qids,original_capture_manifest,bundle,shared_locator,capture_binding`。先闭合原 manifest strictref，再用 `manifest['rows'][index-manifest['dataset_ordinal_range'][0]]` 精确关联 bundle/图身份/qids/共享 locator；manifest 的 `capture_binding` 必须相等。随后闭合 `row.bundle` 并 CPU load。物理包 schema 为 `N16-actual-GPU-image-capture-bundle-v1`；`binding/dataset_index/image_id` 精确对应 row。共享值取 `bundle['members'][row['shared_locator']]`。locator 是字符串，**不是另一个文件 strictref**；标准形式 `i%03d/SHARED_ACTUAL_INPUTS.pth`（G:754–756、795–814）。各物理父 manifest 使用自己的起点，不把 global ordinal 当局部下标。

**2. 原图 L/P/Ω。** 共享 schema `N16-actual-GPU-image-shared-inputs-v1`；`native_L` 是实际 GPU 渲染后保存的 CPU FP32 `[100,H_original,W_original]`，要求 finite/no-grad，原图 shape 与 row/identity 对应（G:824–849、867–888；D:107–114）。`natural_query_ids==list(range(100))`，但 `native_export_query_ids==row.qids` 是原导出排列，必须保留。没有独立 `native_P` 或 `Omega` 张量字段：逐自然 query **q** 取 `L[q]`，完整原图 `P=L[q]>0`、`Omega=abs(L[q])<=np.float32(2)`。Ω flat indices 为 row-major `np.flatnonzero` 升序，数量必须等 `shared['K_counts'][q]`（C:4–20；G:1011–1022）。不使用 sigmoid 阈值近似，不裁至 C/F union，不删除 K0 或低质量 query。

`shared` 另有 `e0:FP32[100,256]`、`union_flat_indices:INT64[U]`、`F_unique:FP32[U,256]`、`C_indices[q]/C_counts/K_counts`（G:883–887、1003–1008）。纯 Ω 内外错误计数无需使用 F/e/PR chunks 或重跑 teacher。注意现格式 `torch.load` 会物化整份物理 bundle；逻辑只用 L **不等于**已经实现只读 L 的磁盘优化。

**3. query/ordinal/RLE 校准。** 自然 q 是 `native_L` 第0维索引；导出 ordinal 通过 `enumerate(row.qids)` 对应 q，不把 ordinal 当 q。原生与 R 参考均按 `(image_id,query_id)` 建唯一字典，保留 category=1、全部18700项（D:35–43、121–128）。用原 encoder role `ctx['encoder_source']` 的 `segmentation(np,masks,(L[q]>0).astype(np.uint8))`，须逐 query **精确等** `ctx['native_predictions']` 的原 segmentation。G:835–840 展示原 class/S0/query/RLE 整字典校准；A:71–87 展示从 Ω 原动作重建 R 后整字典精确相等；D:153–159 展示同规则。新 census 可直接闭合已合格 `ctx['R_predictions']` 并解码原 RLE，无需再运行 PR。原 score/category/q/order 都不重算或重排。

**4. 可复用最小接口与边界。** G 的 `lookup_CPU_bundle_member(torch,bundle,member_id,expected_binding,member_use=None)`（1040–1058）核包 binding、member ID/registry；`validate_shared_for_CPU(torch,np,cv2,shared)`（997–1027）可重核完整 C/K/union，但也扫描 F，且明确不证明实际 Source/owner。Root 仍负责物理 strictref/历史资格关联。C 的 `query_plan` 可重核 Ω/K/union；仅 Ω census 可直接按上式由 L 重建。GT mask/DEV query→GT 配对**不是 shared 内已有 matcher 监督**；后续 FP/FN 必须另锁完整官方380 GT的配对/未匹配处理，不把未匹配 query 当真实背景。该契约不提供 Ω 覆盖实测、结构瓶颈、可修复上界或模型因果结论。
