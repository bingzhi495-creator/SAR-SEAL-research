# N14 固定 stage24 DEV 评价适配草稿

状态：**UNBOUND / 未执行 / helper 已写，ROOT actual ABI 和资格适配待完成**。本稿未读取科学绑定 SOURCE、RAW、SPEC、Q、缓存或权重，没有运行科学任务，不修改原 N13 评价源及实际 N14 训练源。只读取 ROOT 指定的两份 ordinary closed-source 摘录及 N14 普通源码草稿。它不是可直接启动的完整 runner，也不是实际资格证书。

## 固定比较与接口风险

唯一主端点是共同旧 LOG E24 父权重之后的新独立 **stage24**：`BASE_CONTINUATION` 40497 参数与 `MEMORY_ADDON` 42698 参数。新的三段各 8 轮之间继续 optimizer/private order/history，完整阶段每分支 39864 次更新；旧 E24 是初始化出处，不能称连续 48 轮。当前阶段尚未完成；本稿不选择中间最佳 epoch，不改变 stage24 主端点，不根据 FIT loss 或中间 DEV 结果选模型。

评价仍是旧历史 **DEV187 / 原始 GT380 / 每图全部 100 native queries / 每 control 18700 个预测**。保留同一个 frozen teacher、原始 packet、原 renderer、`M=eF` 精确投影检查、原 class/query order、quality 与 natural score、原 official COCO adapter。所有输出的 category、score、query_id、image_id、顺序完全保持。native 空 mask 的零分亦保留；不做 GT guided matching、NMS、阈值或 TEST。374 个已配对 GT 只能用于另一个诊断，不能替代此次 18700 query 评价。

从获准旧摘录核实：`evaluate_image`、`refine`、selector reuse 骨架可适配；旧 `source_gate`/`endpoint_gate`/`load_heads` 将 PR 锁定为 8 张量/40497，并绑定旧 LOG/GEO 训练资格，**不能作为 memory 的资格门**。新 helper 用 `load_stage24_heads` 验证原 base 的 8 张量、memory 的 14 张量（新增 6、base 名字前缀 `base.`），但其前置 ROOT stage24 GPU+CPU gate 是必须实现的 adapter 边界。不要用几项布尔值替代真实资格链。

## 最小运行路线与待 ROOT 绑定的实际 ABI

1. 完成真实 stage17–24 GPU 与独立 CPU 端点资格；冻结实际证据。新评价自身也须单独 owner/admission/full SOURCE/RAW 门。首 SHA 之前先核全部正常和失败 actual RAW、held/reap/birth absence，数量按实际累计正常数与旧 8 + 单独 failed9 递增，不照抄旧 41 或训练首叶 95。
2. ROOT 新 gate 应闭合 stage24 的 GPU source/spec/admission/RAW/report、两个 checkpoint、三段 UPDATE trace、原 stage START、完整 24 个 history/order 与 gradient activity，以及独立 CPU source/spec/admission/RAW/Q/details。CPU Q 必须确证同一 GPU report、同一对 checkpoint、39864 exact AdamW steps、累计状态与 trace。GPU owner join 使用 CUDA `0`，CPU 使用空字符串，须连接真实 entry/spec/admission；不能只看 Q.status。原 E24 与 first-stage 零初始化/RNG/cold proofs 保留为初始化出处，已训练 memory 不重新要求输出等于 BASE。
3. `stage24_adapter_contract()` 定义的是 **本稿新增 canonical adapter 输出**，不是声明未知 actual Q 的字段。ROOT 必须根据实际 stage24 CPUv2 ABI 实现 `root_stage24_gate(spec)`；绑定原始 report/Q 的 ref 与 decoded report，不凭重命名假装闭合。CPU 端点数值协议 JSON 与原 GPU training protocol 分别绑定；不要将 GPU 后、CPU 前预锁协议写成原 GPU 训练前已有。
4. 在科学 import/CUDA 前验证上述 source、库、checkpoint与 native packet/selector资格；CPU factory 严格加载已 qualified 的 stage24 状态，保持 RNG 和 cold CUDA。GPU 评价 no_grad/read-only；原 no-score-change/状态/RNG/内存/cleanup/最终 SOURCE 与实际 terminal 门由完整 ROOT runner 保留。
5. 每图沿用原 `packet_check`/renderer/export/selector verifier：旧 exact GPU `L` 与 immutable selector 逐字相等后，唯一支持为 **`abs(actual original GPU native L) <= 2`**。不调用 GEO 构造，不重新排序、不 top-K。两 arm 顺序评价、chunk2048，仍只有一份 refined100 raster，region 外原 L 逐字保持；原 `n10.fixed_score_rows` 导出。
6. 新 CPU 独立 replay 使用此次 GPU 保存的真实点输入，而非 CPU 重新 resize/rerender F。全187图全部 query重算，精确 RLE与原 official COCO12 readback后，再按原规则闭合 CPU实际 RAW/SOURCE。GPU报告不得自称完成独立CPU或证明自己 terminal。

除实际 N14 CPUv2/stage24 gate，ROOT 还需绑定旧 N10 helpers、original renderer/export/reader/COCO、selector parent + manifest 与原 native population/DEV refs。helper不实现或猜测这些未知实际 spec locator。`evaluate_image`的`packet_selector_gate`保留原 native packet/selector helper所需ABI，**与load_stage24_heads返回的canonical训练资格gate分开**，不能直接互换。`original_selector_packet` 只能使用旧 **reuse** verifier；原旧综合 source/head gate不得调用。`ImageCapture.sink` 是新 adapter 接口：fresh write + reference/hash + exact readback，read 再 bound；需要在实际 owner 下实现，不是现存隐藏接口。

## actual GPU point capture 与独立 CPU mask readback

N13 image113 已提示 CPU 重渲染 F 可触发公式门差。这里把 CPU审计范围明确为 **同一实际 GPU point input 下的已训练 head、external oldLOG gate、全 query mask 与 official COCO**，不宣称 CPU独立复现 GPU feature renderer。

每图按 natural q=0..99 保存以下字段，空支持/空mask/未匹配 query仍有记录：

|范围|必须字段与来源|
|---|---|
|图级 shared|image/dataset index、identity原H/W、native packet ref、selector row+carrier ref、natural q0..99和export order、原 baseline100 RLE/score/class、own evaluation source/spec/admission/phase与stage24 GPU report/CPU Q绑定|
|共享实际点 F|支持并集的 sorted INT64 `union_flat_indices[U]`，从此次 GPU `original_feature` **直接 gather** 的 FP32 `sampled_F[U,256]`；按2048分块复制到 CPU，避免额外整图 GPU F或整个并集GPU F副本|
|原 e|此次 GPU原 `e0[1,100,256]` 的 CPU FP32副本；虽当前 base/memory 对 e 未表现有效预测作用，仍保持原 ABI，不填假 e|
|query input|q、K、shared ref、sorted INT64 `flat_indices[K]`、此次 GPU FP32 `native_L[K]`；只 oldLOG 点，不再次保存全100 L|
|每 arm query output|arm、inputs ref、此次 GPU FP32 `residual[K]` 和 **实际同次 `native_L+residual`** `refined_logits[K]`；并保存此次实际 fixed-score GPU100 query export，供独立 RLE精确比较|

`F` 在100 queries间按图共享；query用 searchsorted 映射其 flat indices到并集。原完整100 L selector保留原 ref并复用，无 CPU F重渲染。两arm共享点输入，但各自实际 GPU输出保存；code仍先完整BASE、再完整MEM，减少额外 raster峰值。外部 ROI对残差直接 gating，不以 head.logits 偷换更新规则。

CPU replay核所有100支持集合恰为原 selector `abs L<=2`，并集精确、native L字节精确、FP32形状/有限性、同样chunk2048。使用 captured F/e/L调用严格加载的8/14张量head，再算 `L+residual`。残差/combined容差需由ROOT在CPU执行前预锁并绑定本GPU完成report；建议沿原 refined support `atol=rtol=1e-4`，但本草稿不替实际预锁。本稿明确设计的新DEV CPU protocol adapter keys是 `same_actual_GPU_point_inputs`、`point_chunk`、`mask_RLE_exact`、`locked_before_CPU_execution`、`bound_completed_GPU_evaluation_report`、`atol`、`rtol`；它们不是声称旧CPU协议已有的key。`completed_GPU_evaluation_ref`参数须由CPU own gate闭合实际GPU评价report（含所有capture refs）后提供；训练端点report不能冒充评价report。**即使数值allclose，任一点sign变化仍失败精确mask门，不能自动放宽近零边界**。从原baselineRLE复制full query mask，仅在对应flat indices替换sign；与actual GPU fixed-score RLE逐query精确一致，再用原 official COCO全18700读取（segm、maxDets[1,10,100]、无score filter、原crowd/ignore）。`n10.stats_exact`保留原COCO12规则（原数值协议的stats atol1e-12）；不另发明评价器。

## 保存与内存预算

设第i图原像素 `N_i=H_i W_i`，所有100 queries支持点总数 `T_i=sum_q K_iq`，空间支持并集 `U_i`。**准确的新增原始 tensor payload** 为：

`B_i = 4*256*U_i + 8*U_i + 4*100*256 + (8+4+2*(4+4))*T_i = 1032*U_i + 102400 + 28*T_i bytes`。

其中 `T_i <= 100*N_i`、`U_i <= N_i`。若只从已闭合原selector metadata得到T，尚无U，可预锁 conservative disk bound：`1032*sum_i N_i + 28*sum_i T_i + 187*102400`；不知道T则最大 `3832*sum_i N_i + 19148800`。旧完整 selector本身是 `400*sum_i N_i bytes`，**复用，非新增capture**。没有读取实际尺寸/K，不能给虚构的GB总数。ROOT应从终止闭合资料读取实际 N/T锁出精确上界、磁盘可用量与资源；serialization、RLE/JSON/reference元数据与读回临时文件需另算，建议保留原至少2GiB free-space余量，但这不是tensor精确公式的一部分。

每图一份新共享tensor的payload是 `1032*U_i+102400`，处理一个query的输入/两arm输出逻辑payload不超过 `28*K_iq`；sink如果整文件exact readback，瞬时会多一份共享payload，不能漏记。GPU helper为保留旧骨架，仍持有全部100个LOG indices数组，合计 `8*T_i` CPU bytes（上界 `800*N_i`），不能按仅一个query算总RAM。分块 F是 `4*256*2048=2097152 bytes`（2MiB），实际GEMM/head、query GPU indices、CPU expected indices/searchsorted位置、native/refined rasters、库与文件读回另计。full原L selector CPU解码是 `400*N_i`，原GPU F是 `1024*N_i`，原GPU L/单arm refined各 `400*N_i`，原export native masks是 `100*N_i` byte量级；均不可把新增payload公式当总进程RAM/GPU峰值。union bool工作区 `N_i` bytes，CPU/native支持计算还有临时bool/gather与numpy索引。sink要逐图、逐query保存/读回并释放，不缓存全DEV F或全query采样F；任何超预算必须在实际启动前被锁定资源门拒绝。

## 可支持与不可支持的结论

只完整stage24、单seed、该固定DEV与配方的BASE/MEM AP差及计算/保存预算。追加8/24阶段和评价/capture成本均报告。即使DEV AP提升，也不能自动归因于memory routing而排除额外2201参数/容量，不能当独立TEST或外部SOTA/创新/TGRS充分证据。容量control和必要routing机制对照仍按先前顺序在收益后开展；本稿不启动新模块。

## 静态核验与当前交付

已准备Python definitions only helper：`load_stage24_heads`、budget accounting、实际GPU F共享capture、顺序oldLOG refine、adapted `evaluate_image`、CPU同输入全query mask replay与原officialCOCO readback。没有执行模型、加载数据/权重或运行CPU/GPU科学评价。完整runner/真实source/spec/admission/RAW/CPU Q适配由ROOT后续完成，只有实际终止与资格证据才能标为可执行/已完成。

2026-10-14 独立静态审阅（`/root/final703_e24_reader_static_review_20261014`）已读本普通稿主体：未发现确定普通Python阻断；全100 CPU mask重建基本数据字段齐全；原始tensor预算 `1032U+28T+102400` 正确，all100索引 `8T` 与sink读回副本另计。其两项必要风险均保留：same-GPU-input不保证CPU/GPU近零sign完全一致，失败应记为资格失败而不事后放宽；原packet/selector的native metadata ABI须由ROOT实际适配，不能用canonical stage24训练gate顶替。此意见是静态审阅，不是实际资格或科学结果。仅普通草稿与说明进入常规发布范围；未来实际F/L/capture缓存属于科学载荷，不随持续同步自动发布。
