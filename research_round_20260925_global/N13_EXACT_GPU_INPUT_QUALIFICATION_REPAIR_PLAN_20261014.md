# UNBOUND：捕获真实 GPU 点输入并用独立 CPU 公式复核

状态：**未绑定、未执行、待 ROOT 适配外部 ABI 与审查**。本计划只依据允许的 UNBOUND 草稿和 ROOT 提供的已闭合结果；未读取私有 RAW、SPEC、Q、数据或权重，也未触碰任何实际科学 SOURCE。

## 实际问题及证据范围

原 CPU v01 在完整检查的第 26 张图（dataset_index 25、image_id 113）失败。该图完整包含 (GT39, query1) 与 (GT40, query89)，并非挑选成功样本。ROOT 提供的四路探针显示：GT40/query89 的 addedHalo 274 点中，CPU 渲染特征＋CPU PR 最大 logit 差约 0.0002875328，1 点超过原 1e-4 atol/rtol；GPU 渲染特征转 CPU＋CPU PR 最大差约 4.29e-6，无超差；GPU 特征＋GPU PR 与原存储值 bit-exact；CPU 特征转 GPU＋GPU PR 最大差约 0.0002889633，同 1 点超差。四路均无符号差异。

这支持该图的不兼容来自 CPU/GPU 特征重渲染差异；不说明 PR 方法失败，也不证明全部 374 对已经兼容。原失败保留，不能通过放宽容差或重跑旧 CPU 源将其改写为成功。

## 最小接口改动

1. 创建独立新 GPU SOURCE，保持原 GPU fixed-PR 全 DEV187、固定 native .50 配对 374＋6、冻结 LOG E48 权重、精确 selector GPU L、域、分区、输出 C 与统计流程。只在候选适配器实际 raw PR 调用处增加输入捕获。
2. 每个实际 oldLOG/addedHalo 的升序 2048 点块，先将实际收到的 GPU FP32 `F_points[N,256]`、精确 GPU 原始 `L_points[N]` 与该块升序 `flat_indices[N]` 复制为 CPU FP32/int64，再调用原 GPU PR 公式。复制不改变 dtype、值、分区或数学运算。
3. 捕获器核对实际调用输入与原 renderer 特征、精确 frozen GPU L 的原升序采样完全一致，记录原 raw PR 返回值与 adapter 最终保存 residual/logit 的精确一致性；候选结果只增补 `exact_GPU_point_inputs` 字段。不得将 G、S 或 K 传入 PR。
4. 创建独立新 CPU SOURCE，删除数值复核中的 CPU `F_original` 重渲染路径。CPU 使用捕获的同一 GPU 点输入与冻结八权重，独立 functional LN/linear/ReLU 复核 residual/logit。`atol=rtol=1e-4` 保持不变。
5. CPU 的域仍来自精确 frozen GPU L，C 仍是实际 GPU 输出。保留原整数 width2、OpenCV 棋盘距离、完整原 LOG binary、固定人口 pixel counts、逐对及全人口 exact JSON 检查。统计/oracle 源复用，不能宣称排序代码独立实现。

新增草稿是可审阅的辅助实现。ROOT 必须将它们接入**新的独立完整 runner**、审查其实际调用链及资格 gate，之后才能绑定执行；它们自身不启动科学计算，也不构成 Q。

## 每对、每分区捕获字段

新增 `candidates.exact_GPU_point_inputs`：

- `schema = N13-fixed-PR-exact-GPU-point-inputs-v1`；`point_chunk_size=2048`；`refined_binary_rule=strict_gt_zero`。
- `provenance`：新捕获 SOURCE、当前 GPU spec/admission/phase、image/GT/query identity、native_packet、selector、renderer SOURCE、N10 SOURCE、冻结 LOG E48 checkpoint。GT ID 仅为 identity 标签；捕获函数不接收 GT 栅格。
- `originalLOG` 与 `addedHalo` 各含：`flat_indices` CPU int64 [N]、`F_points` CPU FP32 [N,256]、`native_logit_points` CPU FP32 [N]、固定 `chunk_count`、每块起止 ordinal 与实际调用 ordinal，以及三个张量的 shape/dtype＋原始字节 SHA-256。
- `capture_contract`：actual GPU raw-PR call observed、实际 sampled inputs 与原 F/L exact、原 PR 数学及 candidate 输出未改、CPU copy 无 dtype 转换；这些标志必须由实际运行检查形成，不能替代外部 SOURCE/owner 资格。

原 candidates 的 residual/logit/binary/merged_C 与原 diagnostic JSON 保留。新增捕获字段不能覆盖它们。CPU 先验证原 frozen L、域和索引，再验证捕获 digest、L 点值、分块记录及来源，之后才进行独立公式复核。

## 数值与 binary 规则

- Formula 比较始终使用原 `1e-4` atol/rtol。CPU raw residual 和 refined logit 均对实际 GPU 存储值检查，不依据失败结果更改阈值。
- 符号差异继续使用原 practical allowance：`tau=refined_support_atol + refined_support_rtol*abs(CPU_refined_logit)`；只有 CPU abs(logit)>tau 的符号差异才拒绝。范围内逐块、逐对及总数记录，不新增绝对 near-zero 上限或 count cap，不称为 certified error bound。
- 先精确核对 stored GPU binary=stored GPU logit>0，并从两分区重建 stored merged_C；CPU binary 仅作检查和差异计数。所有科学统计仍使用实际 frozen GPU C。
- 原 LOG 全 binary 从 A 的 oldLOG 位置替换原 GPU C，并对原输出 full-raster exact；不将 CPU 重算值写入输出或统计。

## 资格链与失败保留

新 spec **保留** `RAW_inputs` normal94、`normal_RAW_count=94`、legacy failed8 字段与 `terminal_identity_pins` 原 pins102；不改写、不追加到该历史字典。

- 独立 `additional_preserved_independent_CPU_failure={RAW_ref,phase,pin,returncode:1}`，保存原 CPU v01 实际 failed9。任何 SOURCE SHA/导入之前先完成该 RAW-only exit1/reap/birth-absence/pin 检查。
- 独立 `successful_compatibility_probe={RAW_input,pin,source,spec,GPU_admission,report,ROOT_qualification}`，其中 `RAW_input` 是标准 receipt/phase/admission/phase_binding/bootstrap 五键行。它代表 actual successful normal95，而不是把旧 failed9 改成成功。
- ROOT 对成功 probe 完成独立 RAW-only actual0/reap/birth-absence/pin 检查，然后运行 legacy `pre_gate(spec,94)` 的完整 RAW-only 部分及首个 SOURCE hash。此后 bound 检查两个新增引用与所有来源。
- 当前 GPU own_gate 使用由 ROOT **新建**的 context：legacy normal94＋成功 probe normal95；context 的 normal pins 为原 pins102＋probe pin（103）；failed9 继续单列，所有实际 ancestor pins 合计104。原 spec 的 legacy 字典保持原样。
- 新 GPU capture 实际0关闭并经 ROOT 审计后，CPU spec 另含 `exact_input_capture_GPU={RAW_input,pin,source,spec,GPU_admission,report}`。CPU 当前 context 的 normals 为96，legacy failed8 pins 加两个新成功 owner 为104，单列 failed9 后所有 actual pins105。
- `owner_ref_join` 对 GPU carrier、原诊断 GPU、成功 probe GPU、捕获 GPU 使用尾参 `'0'`；对历史 CPU carrier 使用空串。当前 full context 的 prefix 与独立 spec 里的 legacy94均要核对，不能用旧 CPU 的失败作为正常 owner。
- ROOT 要核验实际 lineage/common 对额外字段的 ABI，并将新增 normal owner 的 source/spec/admission 和 report/资格引用加入 closure。辅助草稿不伪造 own_gate，不替代 actual held handles 或 ROOT 回收。

成功 probe 的 `ROOT_qualification` 实际 schema/status 由 ROOT 填入明确 descriptor，草稿不猜。数值失败原 SOURCE、原 RAW、原 stdout、旧 Q 的存在状态和失败原因保持原件；新修复有新的 source/spec/phase/output 与真实 attempt。

## SOURCE 扩展与发布范围

GPU spec 至少增补新完整 runner SOURCE、捕获辅助 SOURCE、保留 failed9 RAW、新 probe 的 RAW/admission/phase_binding/bootstrap/source/spec/report/ROOT qualification。仍保留全部既锁 carrier、原 diagnostic、selector、native packet、renderer/N10、权重、运行库及历史资格来源引用。

GPU 每个新 candidate packet 含实际捕获输入，故其新 refs/SHA 与旧 packet 不同。CPU spec 将新 GPU report、RAW/admission/owner 元数据、全部 374 个捕获 candidate packets 和原 summary/pair/population/image-check refs 加入完整 closure。不得复用旧 packet SHA 或伪称旧 report 已含点特征。原 candidate 文件保持冻结。

捕获存储的最低 tensor 内容为每点 256*4＋4＋8=1036 bytes；实际序列化 overhead 和重复原 candidate 内容需 ROOT 根据已锁 `logical_candidate_points` 预估，并预锁 free-space 要求。CPU 点特征属于真实科学载荷，不能作为普通文档在运行期间复制、哈希或发布。仅在 actual owner 终止并完成原资格规则核验后，整理发布范围内的源码和结果摘要；不发布数据、权重、完整标注或运行缓存。

## 新 CPU 仍需要完成的检查

ROOT 当前 Source/owner 闭包 → 所有捕获 refs 与 exact frozen L → 独立公式全部 candidate 点 → 原 GPU binary/C 自洽 → 原 LOG full binary 与原固定 pair counts → 整数 width2 与独立 OpenCV 原域 → 374 对 exact diagnostic JSON → 固定 374＋6 全 population exact JSON。适用范围与证据等级保持原限定，不因新同步或资格通过宣称 AP、因果、部署、总体效力或创新成立。

当前文件：

- `tmp/N13_FIXED_PR_EXACT_GPU_INPUT_CAPTURE_UNBOUND_DRAFT_20261014_01.py`：实际 raw-call 输入捕获辅助实现及 legacy94＋独立 failed9/probe95 的 context 接口。
- `tmp/N13_FIXED_PR_EXACT_GPU_INPUT_CPU_UNBOUND_DRAFT_20261014_01.py`：同输入独立 CPU 数值与冻结 C 复核辅助实现，提供接入原 pair/population 检查的接口。

所有内容仍为 UNBOUND；ROOT 的实际外部 ABI、完整 runner、资源/时间、SOURCE 绑定、owner 控制和执行后资格审计尚未实施。
