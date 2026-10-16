<!-- encoding: utf-8 -->
# N27 FIT 真 matcher 支持域：静态读取契约

**SOURCE_ONLY / NOT_NUMERICALLY_VERIFIED**。本页仅核普通发布源码；未读真实 carrier、packet、AUDIT、GT、权重或资格文件，未导入 Torch/执行数值。主人口固定完整 FIT741 图 /1661 真 GT-matched query；不按 IoU、K、Ω覆盖或预测非空筛选。DEV GT-best-query oracle 不属于此契约。

公开根 `E:/SAR/seal/github_release_20261011/research_round_20260925_global/code/`；下文 **N**=`N26_FIT_MATCHED_R_CAPTURE_SOURCE_ONLY_20261017_01/capture_body.py`，**B**=`N16_cache_body.py`，**W**=`N16_cache_GPU.py`。

**1. 741份完整原图 L。** N:122–150、285–296 给现成定位链：N26 context 的 `N16_GPU_spec/N16_GPU_report` 和 `existing_packet_refs/existing_native_L_refs/identities/image_ids`。对 index=0..740，必须等于 N16 REPORT 的同序 `packet_refs[index]/native_L_refs[index]`，并等于 `rows[index].packet/natural_native_L_carrier`。逐 ref 经 Root 闭包验证后 CPU load：packet schema `N16-all100-FIT-action-sampled-cache-body-v1`；**独立 L carrier** schema `N16-actual-GPU-natural100-L-carrier-v1`，字段 `native_L,identity,source,spec,GPU_admission,phase,dataset_index`。`native_L` 是 finite/no-grad CPU FP32 `[100,H,W]`，H/W=`identity.original_shape`，第0维直接是自然 query q。packet/载体/REPORT 的 source/spec/GPU_admission/phase 和完整 identity 相等；packet 的 `natural_native_L_carrier` 等该 L ref。W:503–515 明确将完整 `natural_native_L` 从 packet 移出，另存 `NATIVE_L_%04d.pth`。不能从 N07 的1024个 sampled logits 或 N16 的4096样本重建完整 L。

**2. 真 GT→q；不伪造背景。** packet 的 `assignment.raw_GT_index_by_query` 是 CPU INT64[100]；g>=0 指 `packet.raw_GT_ids[g] == identity.ann_ids[g]` 的原始 GT（0-based），-1 是历史新加 background proxy，-2 是 ignored，二者均**不进入主诊断**。字段还包括 `raw_GT_annotation_id_by_query`、正/负/ignore列表及 native labels/weights（B:113–149）。每图所有真 GT 必须各恰有一个 q，按原 GT 顺序保留；不能改成 max-IoU query。N:184–194 精确核旧10-stage matcher/原GT映射和 assignment；N:151–153 核原 matching seed；N:165–168 核 raw GT shape、ann_id顺序、转换映射与栈 digest。

**3. P/Ω及计数。** 对每个真实配对(g,q)，在完整原图取 `P=L[q]>0`、`Omega=abs(L[q])<=FP32(2)`，row-major升序 flat；Ω数须等 `packet.K_counts_all100[q]` 及 REPORT row 同字段（N:211–215）。注意 FIT字段叫 `K_counts_all100`，不是 DEV shared 的 `K_counts`。N16 packet 同时有 `C_counts_all100`，但 `samples.actual_PR_residual/actual_combined/native_P/targets` 仅抽样点，不是全图 R（B:292–305）。N:198–205 的 N26 replay 已要求全100 e/class/L与旧载体 byteexact；此静态复核不代替 Root 对该已执行证据的关联。

**4. 与现有 N26 完整 R 的直接连接。** 每个 scope 的 CAPTURE_REPORT 有 `inputs`/`audit` strictrefs；取其 `AUDIT.json`（schema `N26-complete-R-mask-matched-quality-audit-v1`）的 `records`。按五字段 `(dataset_index,image_id,GT_index,GT_annotation_id,query_id)` 精确 join 上述 assignment，不按数组碰巧位置猜；合并两 scope 后完整1661、GT/query不重不漏。`images` 每图 audit 还直接保存 `native_packet/native_L_carrier/original_N07_packet`（N:259–276、363–368）。

每 record 的 `R_mask_RLE={size,counts_ASCII}` 是完整原图 R；另有 `R_mask_digest/R_mask_area/crop_yxyx_halfopen/empty/native_mask_digest/matched_GT_digest/Omega_points/quality_target_FP64`。RLE解码回 CPU bool 后核 shape/area/digest；digest算法为 contiguous ndarray 的 shape、dtype、`SHA256(tobytes())`（N:69–72、103–114）。由 L 得的 P应等 `native_mask_digest`；原 canonical GT完整 raster 应等 `matched_GT_digest`，且 ann_id/index一致。N:247–250要求 R在Ω外逐像素等 P（K0同样保留）。`FIT_QUALITY_INPUTS.pth.records` **特意删掉 R_mask_RLE**，不能拿784维maskfeature替代完整mask；RLE必须从 AUDIT读取（N:348–365）。

**最小角色与边界。** Root需闭合已接受 N26两scope report/AUDIT、原 N26 context→N16 REPORT及各图 L/packet refs、原 FIT COCO annotations和既有相同 raster/GT转换规则。无需新 teacher/PR forward、无需读取 F或权重。FP/FN按固定真 matcher 的同一 GT在Ω内外计数；低IoU/empty/K0保留。不得由有限支持域覆盖宣称可达AP上界、唯一因果或已选定新结构；历史 background proxy不等同真GT背景，1661配对诊断也不代表全部100 query的误报质量。
