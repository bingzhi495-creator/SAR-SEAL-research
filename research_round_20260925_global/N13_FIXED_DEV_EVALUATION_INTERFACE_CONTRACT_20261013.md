# N13 固定完整 DEV 评价：接口契约与最小改动方案

日期：2026-10-13。状态：**仅静态接口设计，尚无评价源码、实际评价或 AP 结果**。本文由独立发布副本中的 N10 评价源码、N13 v2 头/门控/缓存资格源码、cold v3 源码及锁定 N13 实验方案静态推导；没有读取科学载荷、标注、SPEC、Q、RAW，没有哈希、导入或执行科学源码。实际端点资格与运行统计由 ROOT 提供并核验。

## 1. 评价对象与不可变条件

- 每次只评价一个预锁种子的 LOG_PR / GEO_PR 配对端点；seed 属于 202610120701、202610120702、202610120703。两臂必须是同一累计 epoch。8、24 是学习曲线，48 是唯一主端点；不得按曲线 AP 停止种子、改预算或选 best。
- 完整历史 DEV 为 187 图、380 GT；每臂及原生对照均保留 18,700 个预测。该 DEV 已用于选型，不能表述为官方盲 TEST。
- controls 恰为 native_official、LOG_PR_seed{seed}_E{epoch}_fixed_native_score、GEO_PR_seed{seed}_E{epoch}_fixed_native_score。主效用需要最终三个 seed 的完整 48 结果，单叶结果不替代主表。
- 冻结 E24、原始 all100 自然查询渲染、query 排序、类别、分数及原生空掩膜规则必须精确复现。不得 NMS、筛分数、删空预测或按 GT 选 query。
- 两臂使用相同 PR-FL-91（8 张量 / 40,497 参数）；只有支持域及训练点分布不同。GT 不参与推理区域。比较是“区域与监督分布的联合方案”，不独立归因区域。
- 评价不创建 optimizer、不 backward、不更新头/教师，不解码 JPEG，不加载或重新前向教师，不访问 TEST。

## 2. 端点资格采用显式分支，不能伪造成功 GPU 祖先

评价 SPEC 使用一个显式 endpoint_descriptor，双方共同指向同一 qualified endpoint；描述符种类必须在源码中白名单锁定，不按缺失字段自动回退。

|分支|必要实际来源|端点状态要求|
|---|---|---|
|normal_training_v3|实际 v3 train SOURCE / SPEC / REPORT / GPU RAW / admission；实际 v3 CPU reader / SPEC / Q / RAW / admission；ROOT 终端确认；两臂 checkpoint / proof|GPU0 与 CPU0 为不同真实原进程；cold v3 只接受 702/703 E08；CP schema 为 v3；每臂 13,288 updates|
|normal_training_v4|实际 v4 train 与 CPU reader 的同组字段；前一合格端点的完整分支描述；两臂 checkpoint / proof|精确 v4 schema 待实现作者提供；连续 optimizer / 私有 order RNG / 累计访问，累计更新为 1661 × epoch；不能仅凭 schema 后缀接受|
|recovered_failed_gpu_v2|原 v2 train SOURCE / SPEC / **GPU1** RAW / admission / PB / bootstrap / ROOT 故障记录 / 原 9 件产物；实际 **v3** recovery CPU SOURCE / SPEC / REPORT / Q / **CPU0** RAW / admission；ROOT adoption **v1** 记录|仅 701 E08；原 CP metadata 保留 v2 source/spec/admission；原 GPU1 始终在失败链；原 REPORT 缺失不由 synthetic REPORT 填充；CPU0 仅资格化既有端点，不改称原 GPU0|

统一检查：两臂 CP 的真实源、seed、累计 epoch、缓存 Q、teacher、头工厂、有限 state、8/40497、proof、无待处理梯度，与相应 CPU Q 的 checkpoint_results 精确相连。加载后的 state digest 与 START 一致，评价前后 state 不变。无需在评价中重放完整训练轨迹；不能据此声称梯度或每步 AdamW 已独立重演。

**实际版本连接已由 ROOT 澄清**：原 GPU 与 CP 是 v2；已经实际 CPU0 的 recovery SOURCE/report/Q 是 v3；ROOT adoption 为 ROOT-N13-completed8-failedGPU-endpoint-adoption-v1，状态 ROOT_ACTUAL_CPU0_COMPLETE_E08_ADOPTED_ORIGINAL_GPU_EXIT1_PRESERVED，11137 bytes、SHA a53834223f6a59b925a11d2be1e0aa63d57febbb4a9084220be30c86e1043b32。没有 recovery v2 Q。旧 ROOT record 的 adopted_for_continuation_only=True 保留；ROOT 另授权在新评价 SPEC/admission 中显式 recovered_endpoint_use='ROOT_EXPLICIT_FIXED_E08_CURVE_EVALUATION'，仅作为固定 E08 曲线，不据 AP 调整训练预算。v4 精确分支复用当前未绑定源码的白名单 schema 和 parent_gate。

## 3. RAW-first 与完整来源连接

- 第一次 SOURCE / payload SHA、源码加载或数值 import 之前，按 ROOT 当前完整链读取所有 normal RAW（actual0）及全部 7 个 preserved failed RAW（actual1），检查原持有 handle 的 WAIT_OBJECT_0、真实 exit、reaped、原 birth 已不存在、实际 identity/admission pins、资源及无 wait/观测异常。
- normal_RAW_count 与 terminal_identity_pins 由 ROOT 在具体 SPEC 中锁定；不要沿用 N10 的 31 或固定成功索引。禁止将 701 原 GPU1 放入 normal 列表。
- 当前全链可以包含多个独立训练分支。对每个父 SPEC 保留它自己的原 RAW 顺序、失败历史前缀和 identity pins；在当前 closure 中完整包含它，不把旧 SPEC 的计数重写成当前计数。
- RAW → admission → PB → bootstrap → entry + exact args → SPEC → REPORT/Q/CP 的连接均显式核验；不只检查 REPORT 的布尔标记。
- sources 绑定被使用的全部父来源和产物、原生 N07/N09/N10 资格链、actual libraries / PYD / DLL、当前 evaluation / CPU source。复用历史范围的闭合不是删减来源的理由。
- 当前叶输出不能认证自己的终端；ROOT 另核验实际 receipt 后才允许 CPU 资格及下一项。deadline 超时只观察原 handle，不自动重启。

## 4. GPU 计算流程：原生精确复现后才做相同 K 细化

静态可复用 N10 evaluate.py 的 packet_check、original_feature、original_export_kernels、fixed_score_rows 和原始 COCO 规则；source_gate、head_endpoints、refine、report 必须改为 N13 契约，不能仅改常量名称。

1. 从已资格化原 all100 packet 读取 F256[1,256,200,200]、e0[1,100,256]、M200[1,100,200,200]、class logits[1,100,2]。核对字段 digest、dataset_index、identity 和原生 query/export 顺序。
2. 在 GPU 精确核对 M = einsum(e,F)；按原核的 topk100 sorted=False 获取既有顺序。用原始 natural all100 renderer 得到 FP32 L[100,H,W]，再仅为原生导出重排；质量/score 仍按原生 all100 路径计算。18,700 个原生预测字典及原 COCO12 均必须与原资格化参考精确相等，不能以 CPU 近似替换。
3. natural L[q] 与 e[0,q] 保持 query ID 对齐。将**当前真实 GPU L** 保存为只读 CPU FP32 selector carrier；P、LOG/GEO 区域、K、fallback、区域 RLE/索引及 digest 绑定该载荷，不能从近似 CPU M→L 重新选区域。
4. 对每个 q，使用原 region_common.gpu_regions：P=L>0，LOG 为 |L|≤2，K 为 LOG 像素数；GEO 为域内异类 Chebyshev 距离 lex(D,|L|,flat) 前 K，输出 flat 升序。无异类时既有 uncertainty fallback；K0 两臂均不编辑。区域对任何端点、seed、arm 都不依赖 GT。
5. F256 使用同一原始 renderer，channel chunk 固定 16，构成 F_original[256,H,W]。每臂仅按其 ROI 的升序 flat indices，以 point chunk 2048 采样 F 与真实 L。e 仅 ABI，patch 为零49。
6. 调用 PR head 后只能取 **result['residual']**，不能用 result['logits']，后者含旧 |L|≤2 内部门控。统一外部门控 L'=where(arm_region, L+u, L)，未选像素保持原 L；GEO 选中旧 band 外像素仍必须应用 u。两臂精确 K 一致，chunk 调用数按各自 K 记录。
7. 只覆盖每个原预测字典的 segmentation；image_id、query_id、category_id、score、字典顺序均逐项精确保持。原生空变非空仍留原零分；细化后变空仍留原正分；不重新算 mask quality 或 rezero。
8. 逐图记录区域计算耗时、两臂 head 耗时、渲染、导出、I/O 与总时间，以及全部 100 query 的 K/fallback/区域 digest、增加/删除像素、空预测变化及 chunk 数。共同成本与各臂增量成本分开列示；相同 K 不等于排序开销相同。

## 5. Saved GPU L 的存储和有界叶

- 100 × H × W × 4 是每张 selector 的无压缩 FP32 载荷字节数；总量为 400 × sum(H_i W_i)。即使所有 187 图恰为 800×800，也约为 **44.6 GiB**（47.872 GB），不是 47 GiB。
- N10 静态源码仅限定预处理 actual_img_shape≤800；没有证明原图 H,W≤800。因此以上数值不是已核验上界，不能据此承诺剩余空间足够。ROOT 应在完整 RAW/source 门之后按身份元数据统计实际像素量、最大单图 RAM/GPU 峰值和预留空间。
- 推荐只生产一套冻结 native L selector；保存 natural query 顺序、image/crop 形状、native packet ref/digest、原始 renderer source、actual GPU source/spec/admission、L 形状/dtype/digest及严格只读载荷。经独立 CPU selector 资格和 ROOT actual0 后，跨 seed / E08、E24、E48 复用。后续 GPU 评价仍复现 all100 原生路径，并对载荷逐位核对，不重新存九份。
- 生产和复用必须是显式预锁模式，不能遇到文件缺失后静默重算；如采用无损压缩，解码后字节仍精确且资源/I/O另记。无法得到相同载荷时保留失败，不放宽选择规则。
- 每叶 CPU affinity[0,1]、2 threads、RAM最低16GiB、allocator4096MiB、整卡观察8192MiB、deadline1800秒、hidden python、无自动 retry。内存不足必须改有界执行分段，不能降低 query/像素覆盖。
- 仅凭静态源码不能证明完整187配对GPU或CPU叶可在1800秒结束。ROOT 提供的已核验 N10 普通状态记录：GPU五controls339.50秒、整卡5491MiB；CPU四heads1128.937秒，每头4,573,049点/19,439chunks。它支持完整187的有界候选，不保证新几何排序和I/O仍满足deadline。本次批准只实现完整187叶，不实现任意分片；若新执行资源失败，保留失败后另登记分段，不缩小覆盖。
- 若需要分片，锁定所有区间、全187 image identity、共享 selector/端点状态及完整性规则；每片 GPU0 + CPU0 完成后，单独有界 CPU assembly 检查 disjoint、无缺图、无重图、187×100 完整顺序，再计算全 DEV COCO。局部片的 AP 不能作为全 DEV 结果，失败片不能略过；无需重复独立 replay 已合格片。

## 6. 独立 CPU 资格：精确选择，独立头公式与原始 COCO

- 从 saved **exact GPU L** 独立重建 LOG/GEO/K/fallback。采用 N13 qualify_cache_cpu.independent_regions 的 OpenCV DIST_C mask3，独立 lex 排序；不调用生产 SciPy helper。检查所有187×100 query 的 mask/RLE/index/digest/K，两臂同 K 精确，输入自然 query 身份不变。
- CPU 原始 M→L 渲染可继续提供 class/quality/近似几何诊断，不能作为区域选择器，不能宣称全 L CPU allclose 已通过。主原生 masks/scores 精确复现依赖 GPU 全100路径与已有资格链。
- 独立 CPU 按原 renderer/channel16 重建 F；对各臂 ROI 全部像素执行 LN(F)、coarse clamp(-8,8)/8、三个 linear/ReLU/重注入 coarse、最后 linear 的显式 functional 算式，以 **exact GPU L** 为 coarse 与基底，再用 exact arm region 外部门控。不调用 head.forward 或 region_logits。
- 可以复用 N13 train CPU independent_logits(torch,state,F,L,flag) 的函数体，但来源必须精确绑定其实际独立源码；不能把 N10 independent_logits 的旧 native-band门控照搬给 GEO。
- 对 refined GPU 导出 support 的数值一致性沿用预锁 practical near-zero rtol=1e-4、atol=1e-4（tau=atol+rtol×|CPU refined logit|），不声称位等价或认证误差界。**不再给 region boundary 额外 near-band 允许区**，因为选择由 exact GPU L 精确重建。exported support 的任何 native→refined 改变若在 exact arm ROI 外，直接失败。
- CPU 近似掩膜永不替换已导出的预测用于指标。class probability 与原生 quality 的既有 practical 检查继承 class1e-6、quality1e-4、eps1e-6；遇到新失败保存并诊断，不放宽规则。
- 独立 CPU 对三个 controls 重新计算官方 COCO segm 12 stats、maxDets[1,10,100]、原始 crowd/ignore/areaRange 语义、score_filter=None，保持未定义 stats=-1。GPU若全187计算stats，则 CPU 按预锁 atol1e-12核对；shard模式仅 assembly 报全局stats。
- 主结果附全部实际新匹配 one-to-one（.50/.75/.90）与像素统计。原生 .50 的固定374 pair/6 FN 群体另作附表，所有380 GT含 FN/ignore完整保留；.75/.90人数按原生对应匹配，不硬编码374。固定 query 的像素 FP/FN/IoU/Dice 与新 COCO 重匹配分开，不称全局像素风险或因果分解。
- Q 明确自终端资格为 false、完整梯度/AdamW轨迹重放为 false、创新/SOTA/TGRS/泛化/充分收敛均未由此建立。ROOT actual CPU0 后才能发布 qualified AP。

## 7. 最小源码改动范围与尚缺字段

建议新目录 N13_sameK_fixed_DEV_evaluation_source_20261013_01，不改任何已绑定原件。

|文件|可复用部分|必要新内容|
|---|---|---|
|endpoint_lineage.py（仅 stdlib）|RAW/owner/PB/bootstrap 与 closure 检查|三种显式端点分支、实际 source/schema 表、ROOT成功或adoption记录、独立分支全链；不凭 normal 索引猜身份|
|evaluate.py|N10 native packet/render/order/quality/fixed-score 导出/COCO|单 seed 两臂、8/24/48、SciPy同K ROI、exact GPU L carrier、ungated residual外部门控、区域/耗时与readonly状态核验|
|qualify_evaluation_cpu.py|N10 COCO/matching/fixed-native-pairs/practical-support骨架|OpenCV exact-L独立选择、N13 independent PR formula、去掉旧 band allowance、三control及新元数据核验|
|assemble_evaluation_cpu.py（本轮不实现）|原 COCO和匹配纯 helper|如将来资源证据要求分段，另预登记，不在本轮引入任意分片框架|
|EVALUATION_SOURCE_ABI.md|无|以上协议、schema、primary refs、资源/成本和结论限制|

执行源码前 ROOT 需要锁定/提供：

1. 实际 701 **原GPU/CP v2、CPU recovery v3、ROOT adoption v1** 的各项refs；v4 source/spec/report/CP/CPU-Q 白名单精确schema及允许端点范围。原 GPU1 与 CPU0均需保留唯一 identity，禁止虚构 training_GPU_RAW0。
2. endpoint_descriptor、checkpoint/proof/CPU checkpoint_results 与 teacher/cache/factory的精确连接字段；当前 normal_RAW_count、7失败、terminal_identity_pins及每个父SPEC的前缀。
3. 原生 N07/N09/N10 的 report/Q/actual RAW/admission、native_predictions、native score factors、image_rows、all100 packet refs、renderer/export/COCO/pure reader source，以及强E24 12stats引用。DEV全文不会进入公开包。
4. selector carrier manifest/schema/produce或reuse模式、actual selector GPU/CPU资格链与per-image完整references；actual像素存储计数及空间预留。共享carrier不能在运行中为发布同步而读取或复制。
5. 全187或预锁分片的具体范围、每叶deadline/完整输出名称、全局assembly规则；预锁CPU numeric_protocol。没有足够耗时依据时，不承诺单叶全187或GPU-hours。

上述是实现前明确的小范围契约，不扩大 N13 方法臂、训练预算或创新主张。其目的为尽快得到公平、完整、可复核的真实 AP；BCE 接近或几何 oracle 收益不替代该结果。
