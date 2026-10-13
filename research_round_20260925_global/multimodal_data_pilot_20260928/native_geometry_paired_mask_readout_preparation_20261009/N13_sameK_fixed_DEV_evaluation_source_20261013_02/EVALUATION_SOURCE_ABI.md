# N13 full187 paired fixed-epoch evaluation SOURCE ABI v1

状态：**未绑定、仅静态实现，尚无数值执行或 AP**。ROOT 是唯一科学执行者；完成一次聚焦独立静态审阅、实际端点/存储/全来源绑定后才能运行。

## 固定科学范围

一个预锁 seed pair（202610120701/702/703），完整 DEV187 / GT380 / 每control18700 natural100查询。controls 为 native_official、LOG_PR_seed{seed}_E{epoch}_fixed_native_score、GEO_PR_seed{seed}_E{epoch}_fixed_native_score；epoch 只允许 8、24、48。8/24仅曲线、48唯一主端点，不按DEV选择best或停止种子。无分片实现，无新增方法臂、训练规则或指标选择。

GPU 按原 N10 已资格化 helper 重放 all100 M=eF、原 renderer、topk100 sorted=False、原 all100质量和S0；全部18700 native预测与既有强E24 COCO12精确相等后才细化。只改 segmentation，原 image/query/category/score/order 和空mask规则逐项不变，无NMS、过滤、score重算或GT选query。

SciPy按 saved actual GPU L 的 P、|L|≤2、Chebyshev距异类中心及lex(D,|L|,flat)构造每query相同K；每臂flat升序、chunk2048、F renderer channel16。PR只使用 ungated residual，在实际arm ROI内写 L+u；没有旧 head['logits'] 内部门控。K0保留原mask。

CPU以 saved **exact GPU L** 独立用 OpenCV DIST_C mask3 和排序重建所有18700区域、digest、K/fallback；独立 PR functional公式作用于全部实际ROI点。CPU近似M→L allclose仅诊断，不选区域。原生 support 和 saved L>0精确一致；新导出改变必须在 exact arm ROI内。refined CPU/GPU support 只允许预锁 near-zero rtol/atol1e-4，不给region边界额外允许区；不是CUDA位等价/认证误差界。

官方COCO12与重匹配使用GPU导出预测，CPU近似mask不替换。固定原生.50的374 pair/6 FN另表，所有380GT保留；.75/.90人口依原生对应匹配，不硬编码374。历史DEV不是盲TEST、机制因果、充分收敛、创新、SOTA或TGRS证据。

## 来源与入口

文件：endpoint_lineage.py（stdlib资格）、evaluate.py（GPU）、qualify_evaluation_cpu.py（CPU）。入口均 `--spec ABS --owner-execute`，当前 driver SHA ef9a9f17d4fb1bc3464936095ef799021dd880f8b15488d558a871c08b6e1924；当前leaf不能自认证actual terminal。

schema：

- GPU SPEC ROOT-N13-sameK-fixed-fullDEV-GPU-spec-v1；REPORT N13-sameK-fixed-fullDEV-GPU-report-v1；状态 N13_SAMEK_FIXED_FULLDEV_GPU_COMPLETE_PENDING_CPU。
- CPU SPEC ROOT-N13-sameK-fixed-fullDEV-CPU-spec-v1；Q ROOT-N13-sameK-fixed-fullDEV-CPU-qualification-v1；状态 ROOT_N13_SAMEK_FIXED_FULLDEV_CPU_QUALIFIED_V1。
- selector manifest N13-sameK-native-all100-selector-manifest-v1；packet N13-sameK-native-all100-exact-GPU-L-v1。

RAW_inputs保持ROOT实际完整normal0有序链（GPU minimum41，CPU minimum42），加既有primary failed1及exact额外7 failed1；8失败不可删除或改为成功，历史训练/恢复记录仍保留其原7失败事实。terminal_identity_pins精确覆盖全部normal+failed。每次所有RAW实终端、持有handle/reaped/birthabsent/resource/identity/admission检查在第一SHA、源码加载、载荷读取与数值import前完成。随后完整sources逐件校验及RAW→admission→PB→bootstrap→entry/spec连接；不把当前尾索引当独立分支身份。失败后需要新版本/明确链扩展，不能自动重启旧leaf。

实际数值库版本与原环境：Torch2.0.1+cu118、NumPy1.23.5、OpenCV4.8.0、SciPy1.10.1、原pycoco与psutil；全实际PY/PYD/DLL SOURCE由ROOT闭合，加载模块的实际源路径也逐件检查。

## Endpoint descriptor

`endpoint`直接沿用v04 typed resume mapping，base keys为type、training_entry_source、training_spec、training_GPU_RAW、training_GPU_admission、CPU_entry_source、CPU_spec、CPU_qualification、CPU_RAW、CPU_admission、checkpoints（两臂绝对byte/SHA refs）。

- QUALIFIED_SUCCESSFUL_GPU_ENDPOINT增加training_report：只允许实际成功v3或v4训练/CPU-Q schema，GPU0+CPU0独立进程与exact父prefix相连，checkpoint/proof/全部9产物及来源完整。cold v3只能702/703 E08；评价本身不允许E16/E32等未预锁curve端点。
- CPU_ADOPTED_FAILED_GPU_ENDPOINT增加recovery_report和ROOT_adoption_record：原GPU/SPEC/CP为v2且GPU1；实际CPU recovery source/spec/report/Q是v3且CPU0；ROOT adoption schema=v1、status ROOT_ACTUAL_CPU0_COMPLETE_E08_ADOPTED_ORIGINAL_GPU_EXIT1_PRESERVED，11137 bytes/SHA a53834223f6a59b925a11d2be1e0aa63d57febbb4a9084220be30c86e1043b32。原GPU REPORT缺失始终保留，仅使用标注artifact-derived兼容view；不伪造成功GPU或原REPORT。仅701 E08。

旧ROOT记录adopted_for_continuation_only=True不改写。ROOT另明确批准新SPEC/admission的 `recovered_endpoint_use='ROOT_EXPLICIT_FIXED_E08_CURVE_EVALUATION'`；successful分支此字段必须None。这只是固定学习曲线资格，不据此改变预算或认定方法有效。

评价SPEC同时绑定与endpoint相同的training_recipe、cache/relationship/factory/validation字段；源码将eval范围转换为纯元数据context来使用相同v04 parent checks，不执行其训练source_gate或数值训练函数。CP只加载finite state并核对proof、meta、seed/epoch、40497参数、CPU-Q引用；optimizer轨迹不重新执行。

## Shared selector

`selector_mode='produce'`只允许首个701 E08，selector_parent=None；每图一次torch.save CPU FP32 contiguous L[100,H,W]，保留actualproducer source/spec/admission、naturalIDs、native export qids、image/crop identity、原packet/renderer refs及L byte digest；立即独立readback检查。载荷只保存L，不复制F或权重。manifest含187rows，含全部carrier refs及总像素字节。

`selector_mode='reuse'`必须给exact十key selector_parent：evaluation_entry_source、evaluation_spec、evaluation_report、evaluation_GPU_RAW、evaluation_GPU_admission、CPU_entry_source、CPU_spec、CPU_qualification、CPU_RAW、CPU_admission。它必须连接首个produce实际GPU0和独立CPU0 Q，manifest与全部carrier完整sources-bound；新GPU仍逐图all100原生重放并精确核对当前L digest/值与冻结selector。缺失时不静默重算，无重复九份数据。

CPU SPEC继承GPU所有评价与training primary字段，另给evaluation五refs、Q_path、details_path。sources包含新GPU START/TRACE/diagnostics/全部预测、selector manifest/187carriers、父endpoint/全部ancestor闭合。具体absolute路径、SHA及RAW pins由ROOT实际终止后生成，作者未读取科学载荷。

## 有界运行与输出

每叶deadline1800s、CPU[0,1]/2threads、RAM可用≥16GiB、GPU allocator4096MiB/整卡观察8192MiB，hidden单次driver，无retry。SOURCE先闭合再SCI，current report/Q终端字段false，ROOTactual GPU0/CPU0独立确认后才可引用qualified AP。

ROOT先按实际H,W锁定selector_uncompressed_tensor_bytes=400×sum(HW)。minimum_free_output_bytes产生时至少该载荷+2GiB，复用时至少2GiB，实际可用磁盘必须足够。原图尺寸不是预处理≤800的推论。既有N10真实耗时支持完整187候选，不保证新排序/I/O仍能满足deadline；失败保留而非减覆盖。

GPU输出 START.json、TRACE.jsonl、IMAGE_DIAGNOSTICS.json、3组预测、REPORT.json，produce另187载荷与manifest；REPORT记录exact native、pair端点、共享及各臂耗时、GPU allocator峰值/cleanup、全来源及未建立科学结论。CPU输出独立COCO12、新matching、固定人口像素差异、全18700区域/formula/nearzero检查、完整metadata closure及Q。运行缓存与全量标注不进GitHub，只发布SOURCE和准确结果摘要。


## Independent execution repair v02 (2026-10-13)

Original v01 seed701 E08 evaluation exited 1 at its first diagnostic JSON write: a NumPy bool_ from the immutable gpu_regions helper was stored without conversion. v01, its owned RAW/admission/SPEC/logs and partial outputs are preserved; no original complete report or qualified AP exists.

The only prediction-path edit is bool(fallback) when appending the diagnostic list. It occurs after ROI construction; truth value, region indices, K, head weights/residuals, native queries, scores, thresholds and numerical tolerances remain unchanged. The actual qualified CPU independent_regions already returns Python bool; its numerical code is unchanged.

All three evaluation entry/lineage RAW-first guards retain the complete actual normal ancestry supplied by each admission and now require exactly eight actual preserved failures (seven extra rows and the primary). The first repaired admission has 75 normal ancestors; the original minimum41/minimum42 bounds are unchanged. Every original identity and admission SHA remains pinned, and all failures remain distinct from normal ancestors. Training and recovery parents retain their exact historical first-six extra-failure prefix, while the original seed701 training failure is located at extra index5 and checked against the exact historical receipt/phase/returncode. The appended evaluation failure is never mistaken for that training failure. This is a historical-prefix interface repair, not permission to remove, relabel or replace a failed row. New destinations and a separately reviewed single-attempt admission are required.

The failed partial selector is never reused. The repaired first seed701 E08 evaluation must generate a fresh complete selector and pass actual GPU0, independent CPU0 and ROOT closure before any E48 or other E08/E24 reuse. Fixed48 remains the sole primary endpoint; the evaluation recipe and 187-image/18,700-query population remain fixed. This source is statically prepared, not numerical qualification, AP efficacy, innovation or TGRS evidence.
