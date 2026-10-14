# N14 stage24 全 DEV 独立 CPU replay 草稿合同

状态：普通 UNBOUND、未绑定、未导入执行、未科学执行。入口已完整形成；只做 AST 语法检查。编写仅读取获准普通 GPU/CPU 草稿、v02 capture adapter 和普通旧 selector 摘录，没有读取 actual SCI、运行输出、RAW/SPEC/Q、GT、cache 或权重。ROOT 负责实际来源、外部 ABI、资格链及新 admission 的最终绑定。

入口：`N14_FIXED_STAGE24_FULL_DEV_INDEPENDENT_CPU_REPLAY_UNBOUND_DRAFT_20261014_01.py`。必须由 ROOT 使用新独立 CPU schema `ROOT-N14-stage24-fixed-fullDEV-CPU-spec-v1`、`--spec` 和 `--owner-execute` 明确准入，草稿形成不构成运行许可。

先读独立失败9的 RAW/pin，再读全部 normal 与原 legacy failed8 的 RAW/pin。所有 held actual exit、root reap、original birth absence 和原 identity/admission pin通过之后才首次 SHA、源码导入或载荷读取。`expected_normal_RAW_count == normal_RAW_count >= 102`，legacy pins 为 normal+8；失败9另列，total failed9、total pins normal+9。completed GPU DEV 的 normal 前缀比当前 CPU 少一，新增最后一行须由 `owner_ref_join(...,'0')` 对应同一 GPU DEV source/spec/admission/RAW。CPU 当前自己的终止记录不作为输入。

复用普通完整 GPU runner 的 `source_gate(path,spec,cpu=True)` 和 `canonical_stage24_gate(torch,gate,spec)`，但 CPU 入口独立核 `entry_source == __file__`、自己的 `CPU_admission_file`、ROOT 实际资源字段映射、CUDA visibility 空、真实 affinity[0,1]、四个线程环境变量为2、Torch两线程、1800秒且无重试。Torch CUDA 始终保持 cold；不初始化 GPU，不重新渲染 F，不创建 optimizer、不训练。common 与独立 CPU imported-files 门覆盖实际库/源码/binary，官方 COCOeval 源码单独闭合。

继承 GPU runner 的全部 carrier/native/selector 和 stage24 字段。新增 CPU spec 必须闭合以下真实 ref：

- `GPU_runner_source`，与 `completed_GPU_DEV_source` 相等；以及 `completed_GPU_DEV_spec/report/admission/RAW`。
- `CPU_DEV_numeric_protocol`：绑定已实际完成 GPU DEV report 的新 JSON；不可使用 training CPU endpoint numeric protocol替代。
- completed GPU report 的真实 `capture_manifest`、187个 `capture_refs`、3个 prediction JSON、START、TRACE、IMAGE_DIAGNOSTICS；原 native/selector/teacher/DEV 文件和完整 stage24 GPU、独立训练 CPU24 Q/details/checkpoints/三段训练 trace 与 origin START，均在同一 current SOURCE 闭包内。

新的 DEV numeric JSON 必须恰有8键：`schema`、`same_actual_GPU_point_inputs`、`point_chunk`、`mask_RLE_exact`、`locked_before_CPU_execution`、`bound_completed_GPU_evaluation_report`、`atol`、`rtol`。schema=`N14-fixed-DEV-same-GPU-point-input-CPU-numeric-protocol-v1`；三项 true、chunk2048、report 等于上述 completed GPU DEV report、atol=rtol=1e-4。ROOT 应在 GPU DEV actual terminal/report 审计后、CPU attempt 前形成并锁入源码/admission闭包；不声称该锁在 GPU DEV前已存在。

capture 是每图一个真实 physical whole-image `.pth`，没有虚拟 member refs。CPU一次读取 shared GPU sampled F、e 与100个 query的 flat indices/native L/两分支 residual+combined；原完整100个 natural GPU L从既有 `selector_row.carrier` 读回，并核原 selector source/spec/GPU admission、identity/query/renderer 与 native_L digest。只使用原 `abs(L)<=2` oldLOG支持；包括空、未匹配query。全部187图、GT380、每图100个query、每分支18700预测，score/class/query/export order保持原样。BASE40497与MEMORY42698分别加载独立8/14个冻结 FP32 state，完整 stage24累计39864步资格由 canonical gate检查，不能套旧8tensor gate于 memory。

每个分支都必须同时通过 residual 与 `native_L+residual` 固定1e-4 allclose、strict `>0` sign逐点相同、完整 query RLE与GPU实际导出逐项相同。两分支及native按原 official COCO、segm、maxDets[1,10,100]、无score cutoff、原crowd/ignore语义复算全部12项，并核完成GPU结果。精确mask标准比数值容差更强，近零sign差异仍失败，不事后放宽。

CPU spec 新 `CPU_replay_budget` 必须恰有8个非负整数字段：`maximum_image_capture_file_bytes`、`maximum_image_capture_tensor_payload_bytes`、`maximum_selector_tensor_bytes`、`required_available_RAM_bytes`、`maximum_process_RSS_bytes`、`auxiliary_output_budget_bytes`、`minimum_free_output_bytes`、`minimum_available_RAM_margin_bytes`。前两项是完成GPU manifest真实最大值；selector=400×最大图原pixel数；RAM margin至少1GiB，可用RAM下限包含capture+selector+margin。每图加载前及每2048点call核deadline/RSS；采样RSS不冒充完整过程峰值证书。不会再保存F或selector副本。

成功只在全部检查与最终 Source/admission/library/resource 核验后输出 DETAILS 和 QUALIFICATION；Q从写出字节计算stdout ref，不读回运行时自身Q。Q仍明确 `current_owned_CPU_terminal_self_certified=False`，ROOT须在原attempt实际退出0/reap/birth absent之后独立资格。失配不输出成功Q；三类数值/sign/RLE差异记录 `CPU_COMPUTATIONAL_PORTABILITY_FAILURE`，其他绑定或执行错误为 `CPU_DEV_PROTOCOL_OR_EXECUTION_FAILURE` 并退出1。继承失败9不被覆盖；本次若失败，其新实际RAW须由ROOT另行保留。

本CPU Q仅支持这批真实 GPU输入在固定stage24、单种子历史DEV187的计算可移植性与完整读回。CPU失败不自动判GPU AP无效，也不推断方法失效；GPU AP有效性仍须独立审计。没有blind TEST、外部SOTA、公平容量对照、机制归因、收敛、部署或创新结论。
