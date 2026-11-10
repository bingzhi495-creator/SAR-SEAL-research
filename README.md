# SEAL：公共 SAR 船舶实例分割研究

<!-- U57_U60_ACTUAL_20261021_01 -->

## 当前状态（2026-11-10）

- [TGRS 投稿准备度与待验证门槛](research_round_20260925_global/TGRS_SUBMISSION_READINESS_20261109.md)。
- [HRSID E24 冻结模型到 SSDD DEV187 的跨库诊断](research_round_20260925_global/HRSID_TO_SSDD_FROZEN_DEV187_PROTOCOL_20261110.md)。六臂评分与独立复算已完成。固定 SSDD DEV187 上三组平均 +0.376 mask AP 点，未达到预设跨库支持门槛。 等量候选的 IoU 0.75 最大覆盖三组增加 5、20、22 个，IoU 0.90 增加 1、0、0 个。 固定共同匹配的 IoU 平均 +0.519 点，纵向方向组合仅两组通过。
- [Schwegmann 中心船散射掩膜辅助验证](research_round_20260925_global/HRSID_TO_SCHWEGMANN_CENTRAL_SHIP_PROTOCOL_20261110.md)。32 张图、16 个采集组的六臂评分与独立复算完成。中心船平均 IoU +3.906 点，采集数低于预设门槛，保留为描述性结果。
- [固定点目标值试点](research_round_20260925_global/HRSID_MASK_PHASE_COMMON_SUPPORT_AND_SSDD_TRANSFER_DESIGN_20261109.md#固定监督点目标值试点实测)。同一 487 点、三臂各 8 次真实更新，单次复合中心相对逐步中心在裁剪留出上 +0.065 IoU 点，低于预设 0.5 点。连续值对照精确复现。
- [冻结模型的点监督分配试点](research_round_20260925_global/HRSID_MASK_PHASE_COMMON_SUPPORT_AND_SSDD_TRANSFER_DESIGN_20261109.md#冻结模型点监督诊断预锁)。固定八张源图上的分歧富集为同配额随机期望的 1.87 倍。随后同预算 8 次原生投影更新的轮廓指标未达到晋级门槛。
- [HRSID 训练标签的多边形优先对照](research_round_20260925_global/HRSID_MASK_PHASE_COMMON_SUPPORT_AND_SSDD_TRANSFER_DESIGN_20261109.md#hrsid-fit-only-多边形优先对照)。456 个固定训练视图已完成。核心裁剪中单次中心相对逐步中心在两种参考下提升 +6.45、+6.27 个几何 IoU 点。模型效用待同预算实测。
- [距离–多普勒迁移的适用范围和验证设计](research_round_20260925_global/RANGE_DOPPLER_TRANSFER_SCOPE_20261109.md)。已验证一份公开复数训练样本可接入方位 FFT。先识别方位时间、方位频率与径向速度轴。论文候选以同景局部多普勒特征辅助空间轮廓，加入复数空间与幅度派生频谱对照。海面 RD 船艇定位单独评价，模型收益待实测。

主实验 PLAIN/FULL 每组已验收 **37050 次正式更新**。整数栅格下 FULL 的 mask AP 低 0.166 个百分点，依预设规则停止自动扩训。候选浮点栅格下差值为 +0.786 个百分点，列为评估口径敏感性。374 个固定配对的共同区总误差差值为 +0.047 点，未达到预设 1 点门槛。RF-DETR 原生 BF16 强基线已按完整 E100 调度连续完成 **E24、4,464 次更新**。固定 DEV187 整数 GT 的 mask AP 为 **67.443**，AP75 为 **85.386**。候选浮点 GT 另报。

- [374 对共同区域的实际结果及误差解释](research_round_20260925_global/MAIN_E50_DUAL_GT_FIXED_PAIR_ACTUAL_20261107.md)
- [HRSID 原生 E24 训练与 DEV705 实测](research_round_20260925_global/HRSID_COCO_E24_DEV705_NATIVE_EVALUATION_PROTOCOL_20261109.md)。原训练独立验收 24 轮、17,640 次 AdamW 和 EMA 更新，三处固定归档通过。固定 DEV705 的 mask AP **50.427**、AP75 **54.466**、AP90 **1.466**。HRSID 官方 TEST 继续隔离。[固定候选误差普查](research_round_20260925_global/HRSID_DEV705_NATIVE_CANDIDATE_COVERAGE_PROTOCOL_20261109.md)独立复算确认 1,778 个有效 GT 仅 104 个有 0.90 达标候选，原顺序达到最大一对一匹配数。
- [HRSID 历史 DEV705 固定掩膜平移](research_round_20260925_global/HRSID_DEV705_NATIVE_TRANSLATION_PROTOCOL_20261109.md)。1,049 个近严格配对中，±2 像素真值辅助平移恢复 159 个，361 个受面积比限制。固定上移新增 82 个并损失 48 个既有严格配对。
- [SSDD DEV187 历史原生平移诊断](research_round_20260925_global/SSDD_DEV187_NATIVE_TRANSLATION_PROTOCOL_20261109.md)。同规则下 248 个近严格配对中 29 个可由 ±2 像素平移恢复，72 个受面积比限制。固定上移新增 13 个并损失 37 个，与 HRSID 的净值方向相反。
- [HRSID FIT 的原生 164 查询与候选保留对照](research_round_20260925_global/HRSID_FIT_NATIVE_QUERY_RETENTION_PROTOCOL_20261109.md)。实测全查询严格匹配 37→39，等量非空保持 37。密集压力图在 0.50 多 36 个，0.90 仍为零。独立复算后按预设门槛停止筛选分支。
- [HRSID FIT 固定查询轮廓几何](research_round_20260925_global/HRSID_FIT_FIXED_CONTOUR_GEOMETRY_PROTOCOL_20261109.md)。主层放宽整体恢复 51/234，局部同预算恢复 37/234，密集层分别为 0/59 与 19/59。按预设门槛优先验证整体平移的预测信号。
- [HRSID 同一查询的框掩膜中心偏差](research_round_20260925_global/HRSID_FIT_BOX_SHIFT_PROTOCOL_20261109.md)。实测框差、掩膜差与常量的严格匹配均为 51，原生为 37；拟合组均失守 15/37，框差未过门。原生掩膜重采样有合成像素中心偏移证据，真实增强重放待检验。
- [HRSID FIT 训练标签采样相位](research_round_20260925_global/HRSID_FIT_RESIZE_PHASE_REPLAY_PROTOCOL_20261109.md)。114 图、456 个配对视图实测，小目标纵向偏差 +0.876→−0.120 原图像素，参考 IoU 0.795→0.952。保留行变动 3 失 1 增，独立复算通过，进入配对训练资格。
- [HRSID DEV705 等量候选覆盖实测](research_round_20260925_global/HRSID_MASK_PHASE_COMMON_SUPPORT_AND_SSDD_TRANSFER_DESIGN_20261109.md#关闭模型的候选覆盖与图内分配诊断)。逐图非空候选等量后，三种子的 IoU 0.90 最大匹配数分别增加 98、70、113。[5203 独立读回](research_round_20260925_global/HRSID_MASK_PHASE_COMMON_SUPPORT_AND_SSDD_TRANSFER_DESIGN_20261109.md#5203-实测及源域与外部域比较)。
- [HRSID DEV705 跨图分数交叉对照](research_round_20260925_global/HRSID_MASK_PHASE_COMMON_SUPPORT_AND_SSDD_TRANSFER_DESIGN_20261109.md#等量候选的跨图分数排序对照)。两种供体分数固定后，中心候选 mask AP 仍高 4.526 至 5.062 点。固定候选换分数的最大绝对变动为 0.298 点。
- [HRSID 共同检出船舶的掩膜方向实测](research_round_20260925_global/HRSID_MASK_PHASE_COMMON_SUPPORT_AND_SSDD_TRANSFER_DESIGN_20261109.md#固定-iou-050-实例的掩膜几何与方向核验)。两个种子的 y 质心偏差分别由 +0.696、+0.517 收敛到 −0.079、−0.098 像素。原生掩膜上移的收益在中心臂消失。
- [单次中心的真实 FIT 审计](research_round_20260925_global/HRSID_MASK_PHASE_COMMON_SUPPORT_AND_SSDD_TRANSFER_DESIGN_20261109.md#dgx-实测与共同行资格)。DGX 原始 SSDD 741 视图和 HRSID 456 视图闭合并独立复算。直缩 3,571 条共同行掩膜一致。裁剪在 SSDD 改变 639／644 条，HRSID 主层改变 635／638 条。自然 C 在 HRSID 多 3 条非空监督。新的共同行加载器通过 4 个真实 FIT 视图。短训监督入口已通过源码预检并在当前占用下拒绝启动。直缩复用候选经 456 个真实 FIT 视图双臂逐值验收；16 个固定视图的开启审计计时中，主层与压力层直缩耗时中位比减少 59.0% 和 72.0%。
- [独立外部数据核验](research_round_20260925_global/HRSID_CROSSSET_EVIDENCE_BOUNDARY_20261108.md#2026-11-10-独立外部数据入口与标签边界)。实际读回 Schwegmann 的 23 条影像掩膜配对，核对 UEIKAP 来源预留和 AIR-SARShip 标签边界。
- [HRSID 漏分的面积与触边分层实测](research_round_20260925_global/HRSID_MASK_PHASE_COMMON_SUPPORT_AND_SSDD_TRANSFER_DESIGN_20261109.md#面积加权漏分的失败范围)。两个种子的全体像素错误合计下降，大型贴边小样本组的净错误均增加。
- [HRSID 训练掩膜相位的配对训练](research_round_20260925_global/HRSID_FIT_RESIZE_PHASE_PAIRED_TRAIN_PROTOCOL_20261109.md)。同机两臂各通过 32 次访问、8 次 AdamW 和 EMA 的真实资格。图像与框配对一致，29 批掩膜不同。首组两臂各通过 24 轮、17,640 次 AdamW 和 EMA 更新及 E1/E8/E24 档案验收。DEV705 原始 GT 的 mask AP 为 A **50.434**、B **55.551**，相对较强原生参考 **+5.117 点**，独立复算通过。[后续种子 5202、5203 已预锁](research_round_20260925_global/HRSID_FIT_RESIZE_PHASE_REPEAT_SEEDS_PROTOCOL_20261109.md)，四个真实短窗已通过独立验收。5202 两臂均独立验收 E24，访问顺序相同。固定 DEV705 的 A 为 **51.333**、B 为 **56.131**，首个确认种子提升 **+4.799 点**，正式 TEST 未开启。 首组与首个确认组均为正，5203 两臂各独立验收 E24，固定 DEV705 的 A 为 **50.661**、B 为 **55.411**。三组平均提升 **4.888 点**，通过预锁的开发集效用门。三种子汇总规则的 9 项合成测试通过。 [共同行对照、方向合成实测及 SSDD 预案](research_round_20260925_global/HRSID_MASK_PHASE_COMMON_SUPPORT_AND_SSDD_TRANSFER_DESIGN_20261109.md)。 共同行训练、短窗与 E24 验收、DEV705 准入及配对几何入口通过 56 项合成测试。真实共同行短窗已验收。每臂 32 次访问、8 次更新，29 次访问中实际入训掩膜不同。原生臂 E24 已独立验收，共 70,560 次访问、17,640 次 AdamW 和 EMA。原中心臂在第 22 轮中断，已核对 21 个完整轮次，原退出回执缺失。新中心臂从相同 COCO 初值重新运行，已记录 100 次更新，E24 验收与配对评价待完成。B/C 长训、固定 DEV705 准入、配对几何和关闭计划均已通过各自合成测试。真实 B/C 短窗待资源交接。 [DEV705 组构成敏感性实测](research_round_20260925_global/HRSID_MASK_PHASE_COMMON_SUPPORT_AND_SSDD_TRANSFER_DESIGN_20261109.md#dev705-文件名前缀组敏感性实测)显示，5201 与 5202 的 AP 和 AP75 在全部 2,048 次抽样与逐一移除 28 组时均为正。 [纵向三臂 FIT 实测](research_round_20260925_global/HRSID_FIT_VERTICAL_DIRECTION_REPLAY_PROTOCOL_20261109.md)通过方向门槛，原生与反射相对纵向中心分别为 +0.988 与 −0.718 原图像素。
- [SSDD 相位迁移入口和 Q100 初始化实测](research_round_20260925_global/HRSID_MASK_PHASE_COMMON_SUPPORT_AND_SSDD_TRANSFER_DESIGN_20261109.md#ssdd-迁移入口与-q100-初始化实测)。已核验完整初始张量，FIT741 隔离目录实际完成并独立读回。长训验收、导出和评分候选已联测，DGX 已完成 741 个真实增强视图，两臂保留同一 1,445 条非空监督，裁剪小目标纵向偏移 +0.582→−0.026 原图像素。DGX Q100 差异定位至一项位置编码，独立副本已恢复原完整初值摘要。正式装配的 DGX CPU 初值验收已通过，GPU 短窗等待显存准入。独立纵向三臂通过反转门槛，连续面积下裁剪小目标 IoU 为原生 0.845、逐步中心 0.933、单次中心 0.955。HRSID FIT114 原账本复现，主层裁剪小目标同指标为 0.687→0.814→0.882。 [DGX 实测回执](research_round_20260925_global/results/SSDD_FIT741_DGX_CPU_REPLAY_ACTUAL_SAFE_20261110.json)。Windows CPU 复验正常关闭，21 个共享字段数值一致，未执行模型前向或评分。[Windows 实测回执](research_round_20260925_global/results/SSDD_FIT741_WINDOWS_CPU_REPLAY_ACTUAL_SAFE_20261110.json)
- [SSDD 训练匹配的采样归因](research_round_20260925_global/RFDETR_E24_FIT_MATCHER_COST_ATTRIBUTION_PROTOCOL_20261109.md)。DGX 30 图精确重放，分层降幅 0.321 点，按预设门槛停止；后续诊断高 IoU 候选。
- [SSDD 双相位掩膜几何](research_round_20260925_global/RFDETR_E24_FIT_DUAL_PHASE_MASK_GEOMETRY_PROTOCOL_20261109.md)。首组 42/55 缺严格候选，最优标量阈值双相位恢复 3/42，按门槛停线。
- [SSDD 冻结空间基底的嵌套掩膜拟合](research_round_20260925_global/RFDETR_E24_FIT_NESTED_MASK_BASIS_ORACLE_PROTOCOL_20261109.md)。固定基底的系数拟合双相位恢复 18/42，标量对照 3/42。
- [SSDD 原生 BF16 系数回放](research_round_20260925_global/RFDETR_E24_FIT_NATIVE_COEFFICIENT_REPLAY_PROTOCOL_20261109.md)。保持共享偏置，仅改系数恢复 19/42，系数相对范数偏大，下一项检验有限改动。
- [SSDD 有界系数实测](research_round_20260925_global/RFDETR_FIT_BOUNDED_COEFFICIENT_ORACLE_PROTOCOL_20261109.md)。迭代 20/42、一步 15/42、偏置 3/42，进入冻结原生推理的同预算全候选试验。
- [SSDD 冻结系数读出学习对照](research_round_20260925_global/RFDETR_FIT_FROZEN_COEFFICIENT_SCOUT_PROTOCOL_20261109.md)。各 484 步，内外侧差 mask AP 70.902，原生 72.563。交叉读出显示严格损失来自原匹配行的扩张。
- [SSDD 原图有界系数教师](research_round_20260925_global/RFDETR_FIT_ORIGINAL_GRID_TEACHER_PROTOCOL_20261109.md)。学生 AP 72.495，原生 72.563；真值保护未过门槛。训练点拟合消除 60.52% 教师差距，恒等目标仍有漂移。
- [SSDD 教师修正与严格保持的局部梯度诊断](research_round_20260925_global/RFDETR_FIT_TEACHER_DIRECTION_PROBE_PROTOCOL_20261109.md)。两半方向均冲突。普通投影保留 61.84% 拟合改善并新增 1 个严格失守，未过门。
- [SSDD 固定候选的有界掩膜收益诊断](research_round_20260925_global/RFDETR_FIT_BOUNDED_ACTION_HEADROOM_PROTOCOL_20261109.md)。111 图真值引导拟合将 AP 72.563 提到 80.659，统一偏置为 73.081，保留全部原严格配对。仅用于确定下一条无真值预测机制。
- [SSDD 边界双侧局部证据](research_round_20260925_global/RFDETR_FIT_LOCAL_SIGNED_EVIDENCE_PROTOCOL_20261109.md)。五组各 96 次实际更新。114 张历史校准图上局部 logits 的固定匹配净裕量领先，法线配方按规则停止。
- [SSDD 局部 logits 全候选编辑与弃权](research_round_20260925_global/RFDETR_FIT_SELECTIVE_LOGIT_ACTION_PROTOCOL_20261109.md)。预设风险门未找到合格阈值。全强制次要对照较原生 +1.621 AP 点，[固定归因](research_round_20260925_global/RFDETR_FIT_FORCED_LOGIT_CROSSREAD_PROTOCOL_20261109.md)中仅原匹配 +1.686 点，进入前瞻强对照准备。
- [SSDD DEV187 固定局部 logits 前瞻实测](research_round_20260925_global/RFDETR_DEV187_FIXED_LOGIT_PROSPECTIVE_PROTOCOL_20261109.md)。有序局部 logits 较 E24 +1.127 AP 点，较 E100 +0.405 点，固定配方停线。[严格保护诊断](research_round_20260925_global/RFDETR_DEV187_STRICT_RETENTION_CROSSREAD_PROTOCOL_20261109.md)恢复 AP90，但相对 E100 的 AP 差仍低于 1 点。
- [SSDD 同八像素预算的补删比例](research_round_20260925_global/RFDETR_FIT_ASYMMETRIC_EDIT_PROTOCOL_20261109.md)。真值比例机会门通过，预测分配 AP 低 0.178 点。[固定归因](research_round_20260925_global/RFDETR_FIT_ASYMMETRIC_CROSSREAD_PROTOCOL_20261109.md)显示下降主要出现在原匹配行。
- [SSDD 候选条件的配额结果估计](research_round_20260925_global/RFDETR_FIT_QUOTA_OUTCOME_PROTOCOL_20261109.md)。多阈值头停线。[同预算机会诊断](research_round_20260925_global/RFDETR_FIT_RANK_QUOTA_ORACLE_AP_PROTOCOL_20261109.md)中真值排序较四补四删 +3.016 AP 点，[同预算点前缀实测](research_round_20260925_global/RFDETR_FIT_PREFIX_RANK_LOSS_PROTOCOL_20261109.md) 主组相对普通成对排序 AP +0.366 点，相对 BCE AP90 +1.081 点、AP75 −0.557 点，未通过前瞻扩展门槛。[固定 COCO 匹配归因](research_round_20260925_global/RFDETR_FIT_AP75_MATCH_ATTRIBUTION_PROTOCOL_20261109.md) 已实际完成。主组 AP75 的 −0.557 点来自达标边变化，只换 IoU 优先顺序的差值为零。[失守来源和拟合普查](research_round_20260925_global/RFDETR_FIT_AP75_ANCHOR_CENSUS_PROTOCOL_20261109.md) 的五条直接失守边均仅 BCE 后达标，拟合侧对应可控支持为 7 个配对、4 个内容组。[484 图双头五折对照](research_round_20260925_global/RFDETR_FIT_AP75_BROAD_SUPPORT_PROTOCOL_20261109.md) 中，扩展训练只多救回 1 个固定配对，难例支持筛查未通过。[固定多阈值动作实测](research_round_20260925_global/RFDETR_FIT_COCO_ACTION_UTILITY_OOF_PROTOCOL_20261109.md) 略高于 BCE，低于全样本前四排序，未过保持与推进门槛。[候选分侧秩实测](research_round_20260925_global/RFDETR_FIT_SIDE_CONTEXT_RANK_PROTOCOL_20261109.md) 在 96 步与原输入高阈值总数持平，384 步少 13 次，未推进。[翻转对齐实测](research_round_20260925_global/RFDETR_FIT_ALIGNED_FLIP_EVIDENCE_PROTOCOL_20261109.md) 增加 117 个正确翻转，高阈值通过较原 38 维头少 2 次，未通过推进门。固定配对中新增 38 次、失守 40 次，两动作真值选择的上限较原头多 38 次。[嵌套五折选择实测](research_round_20260925_global/RFDETR_FIT_NESTED_VIEW_SELECTION_PROTOCOL_20261109.md) 中高阈值通过比原头少 4 次，未过安全与收益门。
- [SSDD 原生掩膜网格诊断](research_round_20260925_global/RFDETR_E24_FIT_NATIVE_GRID_CANCELLATION_PROTOCOL_20261109.md)。真实错分抵消下降 2.249 个百分点，低于预设 5 点准入。
- [HRSID 原生 FIT 首轮](research_round_20260925_global/HRSID_NATIVE_FIT_FIRST_EPOCH_PROTOCOL_20261109.md)。已完成并独立验收 735 次更新和完整原生检查点。
- [SSDD 预测轮廓的切向对应](research_round_20260925_global/RFDETR_E24_FIT_CONTOUR_CORRESPONDENCE_PROTOCOL_20261109.md)。对齐未胜出，按预设五项图像配对门槛停线。
- [SSDD 小船的有符号错分诊断](research_round_20260925_global/RFDETR_E24_FIT_SIGNED_BOUNDARY_EVIDENCE_20261109.md)。两种秩候选均未过预设门槛，保留补删分项和实际覆盖。
- [等像素预算的实例间分配](research_round_20260925_global/RFDETR_E24_FIT_EQUAL_EDIT_ALLOCATION_PROTOCOL_20261109.md)。三组各 14,864 次翻转，估计交并比组低于原生与置信度，按规则停止。
- [严格掩膜排名实测与 FIT 预锁检验](research_round_20260925_global/RFDETR_FIXED_MASK_RANK_WITNESS_AND_FIT_PROTOCOL_20261109.md)。学习秩较原生 AP 低 0.217 个百分点，按规则停止。固定前缀的完美 16 像素编辑上限仅覆盖约四成小目标候选。
- [HRSID 真实空目标与稠密 FIT 输入资格](research_round_20260925_global/HRSID_REAL_FIT_NATIVE_MATCH_PROTOCOL_20261109.md)。空目标与稠密目标的反向均通过，164 查询逐组覆盖全部 163 个实例。
- [HRSID 原生 COCO FIT 目录资格](research_round_20260925_global/HRSID_FIT_NATIVE_COCO_LAYOUT_PROTOCOL_20261109.md)。2,937 张 FIT PNG 与 9,260 条派生标注已落盘并独立核验。
- [HRSID E24 的 DEV705 固定评分](research_round_20260925_global/HRSID_COCO_E24_DEV705_NATIVE_EVALUATION_PROTOCOL_20261109.md)。完成 E24 后使用冻结预测和原始 GT 评价。
- [HRSID DEV705 输入审计](research_round_20260925_global/HRSID_OFFICIAL_PNG_DEV705_AUDIT_PROTOCOL_20261109.md)。705 图和 1,782 条原始标注已核验，保留 4 条空栅格记录。
- [HRSID 官方 PNG FIT 审计实测](research_round_20260925_global/HRSID_OFFICIAL_PNG_FIT_PIXEL_LABEL_AUDIT_PROTOCOL_20261109.md)。全量 FIT 固定增强后有 8,505 个非空目标和 99 个无目标视图。
- [AdamW 历史状态对照实测](research_round_20260925_global/RFDETR_E24_STRICT_OPTIMIZER_PAIR_ACTUAL_20261109.md)。F/S 各 372 次更新，首梯度完全一致。S 严格覆盖净增 10，AP 低 0.573 个百分点，按规则停线。
- [严格反向路径的 AdamW 等预算对照](research_round_20260925_global/RFDETR_E24_STRICT_OPTIMIZER_PAIR_PROTOCOL_20261109.md)。固定 F/S 首步门槛、每组 372 次更新和 DEV187 分支规则。
- [严格反向首步实测](research_round_20260925_global/RFDETR_E24_STRICT_BACKWARD_REPLAY_ACTUAL_20261109.md)。两组真实 FIT 复验全梯度逐元素一致，允许推进同反向路径的公平优化器对照。
- [DINO 高效注意力反向实测与确定性控制](research_round_20260925_global/RFDETR_E24_EFFICIENT_ATTENTION_BACKWARD_ACTUAL_20261108.md)。骨干外部反向边界精确一致，内部高效注意力节点局部复现分歧。 首次严格反向复验维持原 cuBLAS 环境。
- [HRSID 跨数据集证据范围](research_round_20260925_global/HRSID_CROSSSET_EVIDENCE_BOUNDARY_20261108.md)。官方 PNG 归档全量字节和成员 CRC 已核验，保留 JPG、空栅格和历史测试接触记录。
- [双算子联合反向控制实测](research_round_20260925_global/RFDETR_E24_COMBINED_SAMPLING_BACKWARD_CONTROL_ACTUAL_20261108.md)。已观测骨干首特征边界逐字节稳定，patch 入梯度仍分叉，完整配对暂缓。
- [同图反向节点及同梯度 AdamW 首步实测](research_round_20260925_global/RFDETR_E24_SAME_GRAPH_NODE_AND_ADAMW_STEP_ACTUAL_20261108.md)。定位 9 个网格采样和 2 个双线性上采样分叉。固定梯度后历史状态改变首步幅度和方向。
- [双线性采样反向控制实测](research_round_20260925_global/RFDETR_E24_GRID_BACKWARD_CONTROL_ACTUAL_20261108.md)。原生 CUDA 前向逐字节一致，全部受控反向交给 CPU 后仍观察到 1.096% 梯度差。
- [四次首窗复放与反向差异](research_round_20260925_global/RFDETR_E24_FIRST_GRADIENT_REPLAY_ACTUAL_20261108.md)。同臂差异落在跨臂范围内，已核验的前向及损失一致。下一项只控制采样反向。
- [AdamW 配对实测及首梯度资格失败](research_round_20260925_global/RFDETR_E24_OPTIMIZER_FIRST_GRADIENT_GATE_ACTUAL_20261108.md)。每臂 372 次真实更新，首梯度差异集中在骨干，停止 DEV 评价并锁定首窗复放。
- [固定 E24 的 AdamW 历史状态对照协议](research_round_20260925_global/RFDETR_E24_OPTIMIZER_PRESERVATION_PROTOCOL_20261108.md)。等在线权重、EMA、学习率和增强序列，预锁首梯度资格及短程停走规则。
- [DGX 四端点保留掩膜几何实测](research_round_20260925_global/RFDETR_DGX_FOUR_ENDPOINTS_GEOMETRY_ACTUAL_20261108.md)。E100 严格覆盖较同运行 E24 净增 17 个，较独立 E24 少 19 个。
- [DGX 连续 E100 四端点实际评分](research_round_20260925_global/RFDETR_NATIVE_E100_DGX_FOUR_ENDPOINTS_ACTUAL_20261108.md)。同运行 E24/E50/E100 的 mask AP 为 67.866/67.729/68.165。E100 相对 E24 的 AP75 +1.648，小目标 AP −0.339。
- [优化器状态分支的原文及源码](research_round_20260925_global/RFDETR_OPTIMIZER_STATE_PRIMARY_PRIOR_20261108.md)。固定检查点保留与重置已有先例，可用于排查短热启动的几何回落。
- [FIT 冻结教师边界诊断实测](research_round_20260925_global/RFDETR_FIT_TEACHER_BOUNDARY_ACTUAL_20261108.md)。3/26 符合可配对几何回落，净改善见于 2 个图组，依预注册规则停止固定软教师分支。
- [DGX 评估资源预检与重新准入](research_round_20260925_global/RFDETR_NATIVE_E100_DGX_ALT_EVAL_RESOURCE_ADMISSION_20261108.md)。首次预检退出 1，未前向。新运行按长期服务共存与更低资源上限准入。
- [连续端点 DGX 同机补充评估协议](research_round_20260925_global/RFDETR_NATIVE_E100_DGX_ALT_EVAL_PROTOCOL_20261108.md)。受限于归档传输速率，四个端点另用同一 DGX 数值路径，端点选择不变。
- [DGX 连续 E100 闭合与三端点评估协议](research_round_20260925_global/RFDETR_NATIVE_E100_REMOTE_CLOSED_EVAL_PROTOCOL_20261108.md)。已完成 18,600 次更新。远端 OS 退出与归档通过独立核对，保留本机原 SSH 回执缺口。
- [FIT 增强图冻结教师边界可行性协议](research_round_20260925_global/RFDETR_FIT_TEACHER_BOUNDARY_PROTOCOL_20261108.md)。预锁真实增强、掩膜配对和继续门槛。
- [DGX 首次 E100 闭合及新 E100 启动](research_round_20260925_global/RFDETR_NATIVE_E100_FAILURE_RECOVERY_ACTUAL_20261108.md)。首次运行最后记录 E32，已独立闭合。修复进程扫描后，从相同初始权重新启动并核对 GPU 进程。
- [数学、物理、雷达与近期分割方法原文源码续查](research_round_20260925_global/PRIMARY_METHODS_SOURCE_AUDIT_CONTINUATION_20261108.md)。包含可微分水岭、复极化和去斑、结构预训练、可复训分割对照，以及可证伪的候选重叠诊断。
- [RF-DETR E24 邻船重叠实测](research_round_20260925_global/RFDETR_E24_FIT_CANDIDATE_OVERLAP_ACTUAL_20261108.md)。516 张 FIT 拟合图上，857 个低 IoU 匹配实例中有 33 个涉及邻船预测重叠，已用独立算法复算。
- [等点数边界采样诊断协议](research_round_20260925_global/RFDETR_E24_FIT_POINT_SAMPLING_RISK_PROTOCOL_20261108.md)。先核查原生覆盖和同一均匀 BCE 目标的估计方差，达到预锁门槛后再进入真实训练式验证。
- [等点数边界采样固定 E24 实测](research_round_20260925_global/RFDETR_E24_FIT_POINT_SAMPLING_RISK_ACTUAL_20261108.md)。96 图、178 个匹配实例。小实例 357/744 次原生采样未命中边界错分，分层方差通过门槛，进入训练式验证。
- [E24 训练式点覆盖复核协议](research_round_20260925_global/RFDETR_E24_TRAIN_STYLE_POINT_SAMPLING_PROTOCOL_20261108.md)。采用真实训练增广、五层匹配和 13 个查询组，单列成本与训练效用。
- [E24 真实训练式点覆盖与方差结果](research_round_20260925_global/RFDETR_E24_TRAIN_STYLE_POINT_SAMPLING_ACTUAL_20261108.md)。2,535 次匹配验证降方差门槛。39 个增强后 GT 在并集上均有错分点命中。
- [等点数采样成本协议与实现](research_round_20260925_global/RFDETR_E24_POINT_SAMPLER_FULL_STEP_COST_PROTOCOL_20261108.md)。测真实前反向并检查 Dice 一致，计入边界准备成本。
- [分层采样 V2 前反向成本实测](research_round_20260925_global/RFDETR_E24_POINT_SAMPLER_COST_V2_ACTUAL_20261108.md)。Dice 完全一致，时延略超预锁上限，V2 停在成本验证。
- [分层采样 V3 批量抽样与复测协议](research_round_20260925_global/RFDETR_E24_POINT_SAMPLER_COST_V3_PROTOCOL_20261108.md)。按真值分组批量抽样，沿用 V2 时延门槛。
- [分层采样 V3 前反向成本实测](research_round_20260925_global/RFDETR_E24_POINT_SAMPLER_COST_V3_ACTUAL_20261108.md)。同会话配对中位数通过成本门槛、Dice 完全一致。整轮训练耗时和 AP 待配对训练。
- [E24 等目标 B/D 热启动配对训练协议](research_round_20260925_global/RFDETR_E24_EQUAL_RISK_WARM_BD_SCOUT_PROTOCOL_20261108.md)。每臂 372 次更新，固定同机 E24 和 DEV187 评价。
- [B/D 首次导入退出与实际启动](research_round_20260925_global/RFDETR_E24_WARM_BD_IMPORT_RECOVERY_ACTUAL_20261108.md)。恢复版通过依赖检查并进入实际更新。
- [E24 等目标 B/D 实训与 DEV187 固定评价](research_round_20260925_global/RFDETR_E24_EQUAL_RISK_WARM_BD_SCOUT_ACTUAL_20261108.md)。D 相对同预算 B 的 AP 增加 1.091 点，相对同机 E24 低 0.188 点，按预锁规则停止此短训配方扩展。
- [E24 原生损失 A 热启动对照协议](research_round_20260925_global/RFDETR_E24_NATIVE_WARM_A_CONTROL_PROTOCOL_20261108.md)。固定原预算与数据，验证高 IoU 覆盖是否也在原生损失下回落。
- [原生 A 实测与固定共同查询交叉](research_round_20260925_global/RFDETR_E24_NATIVE_WARM_A_CONTROL_ACTUAL_20261108.md)。A 的 AP 与 D 接近，严格覆盖也回落，预锁解释落在混合区间。
- [全部 100 个原始查询覆盖协议](research_round_20260925_global/RFDETR_WARM_RAW100_COVERAGE_PROTOCOL_20261108.md)。区分候选筛选遗漏与原始掩膜覆盖缺失。
- [全部 100 个原始查询实测](research_round_20260925_global/RFDETR_WARM_RAW100_COVERAGE_ACTUAL_20261108.md)。A/B/D 严格覆盖丢失分别有 36/37、39/39、41/42 在原始查询中也缺失。
- [FIT 匹配状态交叉读出实测](research_round_20260925_global/RFDETR_E24_FIT_MATCH_CROSSREAD_ACTUAL_20261108.md)。秩的匹配候选独立编辑未改变 AP 或 AP75，停止判定保持。
- [FIT 等结构选择器实测与停止判定](research_round_20260925_global/RFDETR_E24_FIT_EQUAL_HEAD_SCOUT_ACTUAL_STOP_20261108.md)。同预算秩相对排序略好，校准后未达到锁定的 AP/AP75 门槛，停止进入 DEV。
- [完整 FIT741 CUDA 原生采集实测](research_round_20260925_global/RFDETR_E24_FIT741_NATIVE_CUDA_CAPTURE_ACTUAL_20261108.md)。741 图已独立验收，60,881 个船舶候选、19,917,672 个带内 logits；固定匹配与等结构头协议已锁。
- [同图 CUDA 接线验收与 FIT 主采集设备](research_round_20260925_global/RFDETR_E24_FIT_NATIVE_WINDOWS_CUDA_ACTUAL_20261108.md)。本机实际 CUDA 原件通过独立复核，完整 CUDA 采集锁为选择器的主输入。
- [第一张真实 FIT 图的原生捕获已独立验收](research_round_20260925_global/RFDETR_E24_FIT_NATIVE_CPU_PILOT_ACTUAL_20261108.md)。90 个船舶候选、19,363 个原图带内 logits；完整 FIT 采集路径已冻结。
- [FIT 选择器的冻结划分与原生采集协议](research_round_20260925_global/RFDETR_E24_FIT_NATIVE_CAPTURE_AND_SCOPE_20261108.md)。FIT741 分为 516/114/111 张，同字节图像成组；候选采集与等支持灰度特征已通过合成测试。
- [E24 实际预测边界带的限额编辑上限](research_round_20260925_global/RFDETR_E24_PRODUCTION_BOUNDARY_ORACLE_ACTUAL_20261108.md)。K=16 新增 90/377 双达标，超过预设的 20/377 机会门，下一步进入 FIT 选择器试验。
- [E24 双标注限额修边上限实测](research_round_20260925_global/RFDETR_E24_SIGNED_EDIT_ORACLE_ACTUAL_20261107.md)。每实例最多改 16 或 32 像素，GT 边界带联合编辑分别新增 53 或 103 个双达标实例，下一项锁定实际预测支持。
- [DGX 连续 E100 已启动与等预算修边诊断方案](research_round_20260925_global/RFDETR_E100_ORIGINAL_START_AND_SIGNED_EDIT_ORACLE_PLAN_20261107.md)。E100 固定为主要端点，限额编辑上限的整数计算已通过合成穷举对照，实际数据待执行。
- [E24 双标注边界误差定位实测](research_round_20260925_global/RFDETR_E24_BOUNDARY_LOCALIZATION_ACTUAL_20261107.md)。224 个双标注严格阈值未达标实例中，2 像素共同边界带聚集 64.69% 的漏分与 71.96% 的溢出。
- [RF-DETR E24 的真实 DEV187 双标注评价与误差诊断](research_round_20260925_global/RFDETR_E24_DEV187_ACTUAL_EVALUATION_20261107.md)
- [全部原始查询覆盖实测与高 IoU 缺口](research_round_20260925_global/RFDETR_E24_RAW_QUERY_GEOMETRY_ACTUAL_20261107.md)
- [E24 原始查询诊断首次资源失败及后续核验](research_round_20260925_global/RFDETR_E24_RAW_QUERY_FIRST_ORIGINAL_FAILURE_20261107.md)
- [RF-DETR 连续 E24 实测与原生断点](research_round_20260925_global/RFDETR_NATIVE_CONTINUOUS_E24_STATUS_20261107.md)
- [E24 固定 DEV187 的原生导出与 CPU 评价协议](research_round_20260925_global/RFDETR_E24_DEV187_EVALUATION_CHAIN_20261107.md)
- [HRSID 与既有像素核验的输入字节对齐及跨集重复检查](research_round_20260925_global/HRSID_U43_V04_INPUT_IDENTITY_ACTUAL_20261107.md)
- [RF-DETR E2 原生接续，第 187 次更新已验收](research_round_20260925_global/RFDETR_NATIVE_E2_ACTUAL_ACCEPTED_20261107.md)
- [RF-DETR BF16 E1 实测与原生断点验收](research_round_20260925_global/RFDETR_NATIVE_E1_ACTUAL_ACCEPTED_20261107.md)
- [RF-DETR E1 的两次实际退出、根因和完整指纹修正](research_round_20260925_global/RFDETR_NATIVE_E1_FIRST_ATTEMPT_ACTUAL_FAILURE_20261107.md)

- [完整外部基线的初始化和预算设计：教师上游成本不能省略](research_round_20260925_global/U57_DIFFSAR_FULL_BASELINE_BUDGET_INITIALIZATION_DESIGN_20261107.md)
- [原生完整预测的CPU COCO薄适配器：源码候选，尚未执行AP](research_round_20260925_global/code/DIFFSAR_NATIVE_CPU_COCO_ADAPTER_DRAFT_20261107_01/README.md)
- [CPU COCO适配器有限独立审阅：运行仍待](research_round_20260925_global/U57_DIFFSAR_NATIVE_CPU_COCO_ADAPTER_INDEPENDENT_REVIEW_20261107.md)
- [DEV187候选完整内容覆盖：187逻辑/183原字节对象，仍待Root准入](research_round_20260925_global/DEV187_ORIGINAL_IMAGE_CANDIDATE_COMPOSITION_STATUS_20261107_03.md)
- [DiffSAR端点预测有限独立审阅：未确认具体阻断，运行仍待](research_round_20260925_global/U57_DIFFSAR_ENDPOINT_PREDICTION_INDEPENDENT_REVIEW_20261107.md)
- [DiffSAR完整端点预测代码候选：仅源码与有限测试，未执行](research_round_20260925_global/U57_DIFFSAR_ENDPOINT_PREDICTION_SOURCE_STATUS_20261107.md)
- [GLDet作者复现来源：500轮设置、初始化与可训练代码仍未取得](research_round_20260925_global/U57_GLDET_AUTHOR_REPRODUCTION_PRIMARY_REVIEW_20261107.md)
- [DiffSAR正式512：实际完整状态接续，尚无端点AP](research_round_20260925_global/U57_DIFFSAR_FORMAL_CONTINUATION512_ACTUAL_20261107.md)
- [DiffSAR原始来源：初值未知、500训练／1000测试／maxDets100](research_round_20260925_global/U57_DIFFSAR_PAPER_INITIALIZATION_PRIMARY_REVIEW_20261107.md)
- [HRSID标注版本差异：净差18涉及26张、原划分保留](research_round_20260925_global/HRSID_ANNOTATION_PROTOCOL_DIFF_STATUS_20261107_05.md)
- [外部DiffSAR首256次正式更新：完整状态已保存，尚无AP](research_round_20260925_global/U57_DIFFSAR_FORMAL_FIRST256_ACTUAL_20261107.md)
- [DiffSAR恢复状态与原生数值方差：旧失败保留、同起点重复实证](research_round_20260925_global/U57_DIFFSAR_STATE_AND_NUMERICAL_VARIATION_ACTUAL_20261107.md)
- [完整外部训练的新前向协议及证据边界](research_round_20260925_global/U57_DIFFSAR_FORMAL50_STATE_CONTINUITY_FORWARD_PROTOCOL_20261107.md)
- [HRSID5604图本地原字节、CRC、header与跨划分重复聚合](research_round_20260925_global/HRSID_JPG_INPUT_AUDIT_STATUS_20261107_04.md)
- [DiffSAR真实32步原生训练健康结果：第6步激活掩膜信号](research_round_20260925_global/U57_DIFFSAR_NATIVE_HEALTH32_ACTUAL_20261107.md)
- [完整FIT输入已传DGX的安全聚合](research_round_20260925_global/U57_FIT_DGX_INPUT_TRANSFER_20261107.md)与[作者原优化器/采样/学习率来源核对](research_round_20260925_global/U57_DIFFSAR_NATIVE_RECIPE_SOURCE_REVIEW_20261107.md)
- [原DEV187人口/尺寸/顺序核验：真实读取原GT，仅输入范围](research_round_20260925_global/DEV187_ORIGINAL_GT_POPULATION_BRIDGE_STATUS_20261107_04.md)
- [原DEV187字节已私有暂存DGX：数据准入不等于预测](research_round_20260925_global/DEV187_DGX_RAW_INPUT_STAGE_STATUS_20261107_06.md)
- [质量评分与标注约定直接先例：Mask Scoring R-CNN及边界文献](research_round_20260925_global/MASK_QUALITY_ANNOTATION_CONVENTION_PRIMARY_REVIEW_20261107_01.md)
- [N26冻结排名两GT诊断：预绑定源码，真实评价尚未运行](research_round_20260925_global/code/N26_FROZEN_RANKING_GT_DIAGNOSTIC_DRAFT_20261107_01/README.md)
- [N26诊断有限独立静态审阅](research_round_20260925_global/N26_FROZEN_RANKING_GT_DIAGNOSTIC_INDEPENDENT_REVIEW_20261107_01.md)
- [N26首次实际诊断失败：可选bbox检查错误，0个COCO案例，无新AP](research_round_20260925_global/N26_FROZEN_DUALGT_ATTEMPT01_FAILURE_STATUS_20261107.md)
- [N26冻结输入已私有暂存DGX：字节阶段没有数值JSON解析](research_round_20260925_global/N26_FROZEN_DUALGT_DGX_INPUT_STAGE_STATUS_20261107_02.md)
- [N26诊断02最小实现修复：原输入/评分/评价门槛保持，实际未执行](research_round_20260925_global/code/N26_FROZEN_RANKING_GT_DIAGNOSTIC_DRAFT_20261107_02/README.md)
- [排序、边界与栅格化三项原始来源短记：单一条件性负对照](research_round_20260925_global/MASK_QUALITY_RASTERIZATION_PRIMARY_RESEARCH_NOTE_20261107_09.md)
- [N26两GT真实8例结果：排序增益整数+1.438、浮点+1.602，非几何创新](research_round_20260925_global/N26_FROZEN_DUALGT02_ACTUAL_RESULT_20261107.md)
- [N26两GT有限独立结果/完整性范围复核：指定Codex审阅待补](research_round_20260925_global/N26_DUALGT02_LIMITED_RESULT_CLAIM_REVIEW_20261107.md)
- [N26结果后研究取舍：先检验排序/ignore解释，不追加尺度模块](research_round_20260925_global/N26_POST_DUALGT_RESEARCH_DECISION_20261107_01.md)
- [质量后校准直接先例：2024 mask-IoU校准与2025折衷记录](research_round_20260925_global/SCALE_CONDITIONED_MASK_QUALITY_PRIMARY_REVIEW_20261107_01.md)
- [官方COCO尺度分组的竞争解释：未执行的新诊断设计](research_round_20260925_global/COCO_SCALE_GROUP_RANKING_ALTERNATIVE_REVIEW_20261107_01.md)
- [RF-DETR首次真实检查：原始strict失败、资源已闭合、主训接回](research_round_20260925_global/RFDETR_RAW_STRICT_PROBE_ACTUAL_RESULT_20261107_01.md)
- [RF-DETR限定schema兼容与真实CPU前向结果](research_round_20260925_global/RFDETR_SCHEMA_COMPAT_CPU_FORWARD_ACTUAL_RESULT_20261107_02.md)
- [形状、拓扑、OT与等变性六项原文及作者源码](research_round_20260925_global/SHAPE_TOPOLOGY_OT_EQUIVARIANCE_PRIMARY_SUPPLEMENT_20261107_01.md)
- [单类迁移的boat分类行、强初始化与公平对照](research_round_20260925_global/RFDETR_SINGLE_CLASS_TRANSFER_INITIALIZATION_REVIEW_20261107_01.md)
- [canonical预测导出候选：原GT、类别与top100顺序](research_round_20260925_global/code/RFDETR_CANONICAL_EXPORT_DRAFT_20261107_01/README.md)
- [完整训练环境实际清单、两次解析失败与离线准备](research_round_20260925_global/RFDETR_FULL_TRAIN_PREPARATION_ACTUAL_STATUS_20261107_01.md)
- [主实验 36,864 传输失败及有界恢复](research_round_20260925_global/U57_36864_SSH_CAPTURE_FAILURE_AND_RECOVERY_20261107.md)
- [RF-DETR 原生 BF16 两步候选](research_round_20260925_global/RFDETR_TWO_UPDATE_BF16_QUALIFICATION_DRAFT_20261107_01/README.md)
- [RF-DETR 真实数据两步 FP32 优化器实测](research_round_20260925_global/RFDETR_NATIVE_FP32_TWO_UPDATE_ACTUAL_20261107_01.md)
- [RF-DETR 真实数据两次 FP32 优化器更新候选](research_round_20260925_global/RFDETR_TWO_UPDATE_FP32_QUALIFICATION_DRAFT_20261107_01/README.md)
- [RF-DETR 原生 SSDD DataModule 两样本实测](research_round_20260925_global/RFDETR_NATIVE_SSDD_DATA_ENTRY_ACTUAL_20261107_01.md)
- [RF-DETR SSDD 原生目录实际字节验收](research_round_20260925_global/RFDETR_SSDD_NATIVE_INPUT_LAYOUT_ACTUAL_20261107_01.md)
- [RF-DETR 原生 CLI 首次实测与修正](research_round_20260925_global/RFDETR_CLI_CODEC_ACTUAL_RESULT_20261107_01.md)
- [RF-DETR 官方 CLI 与编解码检查候选](research_round_20260925_global/RFDETR_CLI_CODEC_QUALIFICATION_DRAFT_20261107_01/README.md)
- [576训练资格接口：原生加载、更新计数、EMA及canonical导出](research_round_20260925_global/RFDETR_TRAINING_QUALIFICATION_INTERFACE_DRAFT_20261107_01/README.md)
- [完整train/CLI依赖候选与实际执行缺口](research_round_20260925_global/RFDETR_FULL_TRAIN_CLI_DEPENDENCY_READINESS_20261107_01.md)
- [RF-DETR v02空schema兼容、离线诊断与待执行真实验证](research_round_20260925_global/RFDETR_V02_CANDIDATE_STATUS_20261107_02.md)
- [RF-DETR实际基础依赖安装、数据接口与待执行CPU探针](research_round_20260925_global/RFDETR_RUNTIME_PREPARATION_STATUS_20261107_02.md)
- [2026-11-07数学、物理与雷达方法溯源、30条原文源码索引和后续实验](research_round_20260925_global/PRIMARY_METHODS_SOURCE_AUDIT_AND_NEXT_TEST_20261107.md)
- [完整双标注复评：FULL相对PLAIN从−0.166点变为+0.786点](research_round_20260925_global/MAIN_E50_DEV187_DUAL_GT_ACTUAL_RESULT_20261107.md)
- [固定 374 个配对的共同区误差诊断预分析](research_round_20260925_global/MAIN_E50_DUAL_GT_FIXED_PAIR_PREANALYSIS_20261107.md)
- [固定50轮整集DEV187实测：PLAIN 68.423，FULL 68.256，停止FULL自动扩训](research_round_20260925_global/MAIN_E50_DEV187_ACTUAL_RESULT_20261107.md)
- [分片恢复和逐步验收记录](research_round_20260925_global/MAIN_E50_FIRST_SHARD_CORRECTED_PREDICTION_20261107.md)
- [主线固定50轮已验收，首片原始失败和根因记录](research_round_20260925_global/MAIN_E50_ENDPOINT_AND_FIRST_SHARD_IMPORT_FAILURE_20261107.md)
- [主线恢复段和 RF-DETR 原生BF16两步历史记录](research_round_20260925_global/U57_RECOVERY36864_AND_RFDETR_BF16_ACTUAL_20261107.md)
- [2026-11-07研究决策：数学/物理/雷达/最新模型原始证据与RF-DETR新基线](research_round_20260925_global/RESEARCH_DECISION_20261107.md)
- [RF-DETR五块公平实验计划：100轮候选、诊断和停止规则（未运行）](research_round_20260925_global/RFDETR_BASELINE_AND_DIAGNOSTIC_PLAN_20261107.md)
- [R101真实加载与CPU模型factory候选：代码冻结，实际执行仍待](research_round_20260925_global/R101_REAL_LOADING_CANDIDATES_STATUS_20261107.md)
- [R101逐键覆盖候选：520项完整规范、14项有限合成测试，未加载模型](research_round_20260925_global/code/R101_EXPLICIT_BACKBONE_COVERAGE_DRAFT_20261107_01/README.md)
- [R101覆盖候选有限独立审阅：metadata通过不授真实加载资格](research_round_20260925_global/R101_BACKBONE_COVERAGE_CANDIDATE_REVIEW_20261107.md)
- [已接收19段的两臂部分成本：5.59小时及限定ETA](research_round_20260925_global/MAIN_E50_PARTIAL_PAIRED_COST_20261107_01.md)
- [外部DiffSAR公开预训练新入口：ImageNet/COCO与迁移边界](research_round_20260925_global/DIFFSAR_INITIALIZATION_GAP_NEW_EVIDENCE_20261107_02.md)
- [R101 V1候选已私有取得：仅字节核验，未加载模型](research_round_20260925_global/PUBLIC_R101_INITIALIZER_BYTE_INTAKE_STATUS_20261107.md)
- [31232严格接收：旧UNKNOWN保留、新同算法观察通过，无重训](research_round_20260925_global/U57_STRICT_CENSUS_UNKNOWN_AND_FRESH_OBSERVATION_20261107.md)
- [主E50评价资源衔接候选：有限验证完成，真实运行待](research_round_20260925_global/MAIN_E50_ENDPOINT_RESOURCE_CLOSURE_SOURCE_STATUS_20261107.md)
- [原参考数据人口/顺序/角色桥：真实I/O通过，无新AP](research_round_20260925_global/MAIN_E50_REFERENCE_ORDER_ROLE_BRIDGE_STATUS_20261107.md)
- [最后186次更新的普通源码静态合同核查](research_round_20260925_global/MAIN_E50_FINAL186_STATIC_CONTRACT_REVIEW_20261107.md)
- [回答“跟其他方法比”：实测表、论文值、尚未完成的对照](research_round_20260925_global/OTHER_METHODS_COMPARISON_USER_BRIEF_20261107.md)
- [当前与其他方法的对比：已测数字、不同预算和外部缺口](research_round_20260925_global/U57_EXTERNAL_METHOD_COMPARISON_20261107.md)
- [DiffSAR两次限定梯度观察：原失败保留与解释边界](research_round_20260925_global/U57_DIFFSAR_MASK_GRADIENT_DIAGNOSTICS_20261107.md)
- [完整FIT原图字节恢复](research_round_20260925_global/U57_FIT_RAW_RECOVERY_20261107.md)与[私有输入包组装聚合](research_round_20260925_global/U57_FIT_INPUT_ASSEMBLY_20261107.md)
- [完整DiffSAR首FIT实际执行：梯度覆盖失败与待定位范围](research_round_20260925_global/U57_FULL_DIFFSAR_SINGLE_FIT_FAILURE_ACTUAL_20261106.md)
- [作者运行环境、真实D2算子及完整模型准备的实际结果](research_round_20260925_global/U57_AUTHOR_RUNTIME_AND_D2_OPERATOR_ACTUAL_20261106.md)
- [作者Detectron2 CUDA扩展：实际构建与资格边界](research_round_20260925_global/U57_AUTHOR_D2_BUILD_ACTUAL_20261106.md)
- [11月6日重新连接与一次限定恢复](research_round_20260925_global/U57_RECONNECT_AND_BOUNDED_RECOVERY_20261106.md)
- [CARAFE真实CUDA输出与梯度检查](research_round_20260925_global/CARAFE_ACTUAL_OPERATOR_PROBE_20261106.md)
- [标注版本敏感性的实际结果：R优势反转](research_round_20260925_global/U57_INTEGER_FLOAT_GT_SENSITIVITY_ACTUAL_20261025.md)
- [DiffSAR作者实现的CARAFE依赖与实际mapper](research_round_20260925_global/U57_DIFFSAR_CARAFE_AND_MAPPER_EXECUTION_GAPS_20261025.md)
- [全380实例、all-100 oracle覆盖诊断的实际结果](research_round_20260925_global/U57_ALL100_ORACLE_COVERAGE_ACTUAL_20261025.md)
- [固定native几何诊断：实际FP/FN与解释范围](research_round_20260925_global/U57_FIXED_NATIVE_GEOMETRY_DIAGNOSIS_ACTUAL_20261025.md)
- [最新精修直接先例：基础SAM、++与局部保留](research_round_20260925_global/U57_HIGH_IOU_REFINEMENT_PRIOR_UPDATE_20261025.md)
- [实际DEV187标注来源补核](research_round_20260925_global/U57_DEV187_GT_PROVENANCE_SUPPLEMENT_20261025.md)
- [完整187图SAMRefiner强对照：实际结果与统一评价](research_round_20260925_global/U57_SAMREFINER_FULL187_COMPARISON_EXECUTION_20261025.md)
- [其他方法对比：实测范围、论文值与复现缺口](research_round_20260925_global/U57_COMPARISON_EVIDENCE_SNAPSHOT_20261025.md)
- [20736原会话采集失败与一次限定接续](research_round_20260925_global/U57_20736_CAPTURE_RESET_AND_BOUNDED_RECOVERY_20261025.md)：严格前缀保持20480；失败额外计算不作父产物，新接续尚未接收。

- [SAMRefiner真实单图功能/资源探针已合格](research_round_20260925_global/U57_SAMREFINER_ONE_IMAGE_FUNCTIONAL_PROBE_ACCEPTED_20261025.md)：原工具/SSH/worker均0，清理后allocated/reserved=0/0；仅首图100候选，无AP或全187图结论，v02原失败保留。

- [SAMRefiner首次真实模型返回与清理失败](research_round_20260925_global/U57_SAMREFINER_FIRST_REAL_MODEL_FAILURE_AND_CLEANUP_HYPOTHESIS_20261025.md)：100候选的真实FP32精修已返回，原作业清理未通过，失败保留且无AP；[v03清理候选](research_round_20260925_global/code/U57_SAMREFINER_REAL_ONE_IMAGE_CLEANUP_FIX_DRAFT_20261025_03/README.md)尚未执行。

- [SAM权重已到达DGX及共享GPU单图探针范围](research_round_20260925_global/U57_SAMREFINER_SHARED_GPU_PROBE_SCOPE_20261025.md)：完整字节与独立全摘要已验证；已有推理服务保留，真实SAM精修尚未执行。

- [RGB400首图真实比较与首端点前输入敏感性补充规则](research_round_20260925_global/U57_PROSPECTIVE_RGB400_INPUT_SENSITIVITY_POLICY_20261025.md)：比较完成但数值与字节不相等；不推定AP影响，不换主输入。

- [与其他方法的最新对比快照](research_round_20260925_global/U57_COMPARISON_EVIDENCE_SNAPSHOT_20261025.md)：内部实测、PointRend结构适配及外部方法复现缺口分别列明；[19456步独立安全标量](research_round_20260925_global/results/U57_FORMAL19456_ACTUAL_SCALARS_20261021.json)。尚无外部同协议完整AP。

- [15616→15872单次恢复的严格接收结果](research_round_20260925_global/results/U57_CAPTURE_RECOVERY15616_15872_SAFE_RESOLUTION_20261025.json)：失败原段保留，新恢复从合格父状态执行；通信根因和物理修复仍未证明。
- [FastGeodis固定CPU/CUDA算子检查源码与结果](research_round_20260925_global/code/U57_FASTGEODIS_CPU_CUDA_SMOKE_20261025_01/README.md)：两个9×9样例、四次明确调用实际通过；软件/合成检查独立记账，不计完整训练、重复种子或AP。
- [官方ViT-H权重获取状态](research_round_20260925_global/results/U57_SAM_VITH_OFFICIAL_ASSET_ACQUISITION_SAFE_STATUS_20261025.json)：文件已在本机获取并记录本地摘要，尚未加载模型；权重不上传Git。
- [SAMRefiner真实单图驱动候选](research_round_20260925_global/code/U57_SAMREFINER_REAL_ONE_IMAGE_TRIAL_DRAFT_20261025_01/README.md)：3项纯元数据测试与静态审阅通过，完整模型未执行；[独立资源方案](research_round_20260925_global/U57_SAMREFINER_FP32_MEMORY_AND_PROSPECTIVE_RESOURCE_PLAN_20261025.md)尚未绑定。
- [GLDet原PDF指标核查](research_round_20260925_global/U57_GLDET_ORIGINAL_PDF_METRICS_AUDIT_20261025.md)：确有实例分割，作者mask AP与本项目内部DEV不直接排名。
- [SAMRefiner首张原图与100候选输入核对](research_round_20260925_global/code/U57_SAMREFINER_FIRST_INPUT_CHECK_20261025_01/README.md)：真实CPU单图、像素及候选顺序核验通过；没有SAM推理或AP。
- [SAMRefiner实际运行依赖与未闭合条件](research_round_20260925_global/U57_SAMREFINER_RUNTIME_READINESS_AND_BUILD_GAPS_20261025.md)：作者定制SAM包已隔离安装并通过GPU隐藏的真实入口导入；原生GPU算子仅在两固定样例通过，SAM模型及AP仍待。
- [SAMRefiner作者API桥接候选源码](research_round_20260925_global/code/U57_SAMREFINER_REAL_BRIDGE_DRAFT_20261025_01/README.md)：7项小NumPy边界测试及独立源码审阅通过；按作者图像路径接口批量调用，真实作者函数/模型/图像、内部候选顺序及AP仍未执行或验证。
- [SAMRefiner固定候选适配器候选源码](research_round_20260925_global/code/U57_SAMREFINER_FIXED_CANDIDATE_ADAPTER_DRAFT_20261025_01/README.md)：13项有限标准库测试及独立静态审阅通过；真实SAM桥接、推理与AP尚未完成。
- [SAMRefiner同粗掩膜强对照的实际接口核查](research_round_20260925_global/U57_SAMREFINER_PREDICTED_MASK_INTERFACE_AUDIT_20261025.md)：默认5轮、共享编码、候选/评分及空mask适配边界；未运行。
- [本地候选归档与整数标签转换的实际补证](research_round_20260925_global/U57_INTERNAL_DATASET_PROTOCOL_TRACE_20261024.md) · [首端点前新增的标签敏感性规则](research_round_20260925_global/U57_PROSPECTIVE_ANNOTATION_SENSITIVITY_POLICY_20261025.md)
- [DiffSARShipInst作者配置的实际执行缺口](research_round_20260925_global/U57_DIFFSAR_AUTHOR_CONFIG_EXECUTION_GAPS_20261025.md) · [单次诊断恢复的合格结果](research_round_20260925_global/results/U57_CAPTURE_RECOVERY13824_14080_SAFE_RESOLUTION_20261025.json)
- [External method comparison: actual results, literature scope and missing fair controls (2026-10-25)](research_round_20260925_global/U57_EXTERNAL_METHOD_COMPARISON_STATUS_20261025.md)
- [Original E24 actual training roster audit](research_round_20260925_global/results/U57_E24_TRAINING_ROSTER_SAFE_AUDIT_20261025.json) · [Latest capture failure and unchanged formal prefix](research_round_20260925_global/results/U57_13824_CAPTURE_FAILURE_SAFE_STATUS_20261025.json)

- [Current internal dataset lineage and protocol evidence (2026-10-24)](research_round_20260925_global/U57_INTERNAL_DATASET_PROTOCOL_TRACE_20261024.md)
- [Conditional ordinary capacity control: design only, not implemented](research_round_20260925_global/U57_CONDITIONAL_PLAIN_CAPACITY_CONTROL_DESIGN_20261024.md)
- [Trained metrics and comparisons with other models (2026-10-24)](research_round_20260925_global/TRAINED_METRICS_COMPARISON_20261024.md)
- [Current progress and complete method explanation (2026-10-24)](research_round_20260925_global/U57_CURRENT_PROGRESS_AND_METHOD_EXPLANATION_20261024.md)
- [原始终态丢失、ESRCH失败与最小修复](research_round_20260925_global/U57_OWNER_ESRCH_FAILURE_AND_FIX_20261021.md) · [修复候选源码与测试范围](research_round_20260925_global/code/U57_OWNER_ESRCH_FIX_DRAFT_20261021_01/README.md)
- [本轮真实训练、CPU恢复及证据边界](research_round_20260925_global/U57_U60_ACTUAL_TRAINING_AND_CPU_RESTORE_20261021.md) · [实测标量](research_round_20260925_global/results/U57_U60_ACTUAL_SCALARS_20261021.json)
- [研究路线与证据范围核对](research_round_20260925_global/U57_EXPERIMENT_LINEAGE_AND_EVIDENCE_SCOPE_20261021.md)：旧24＋48实验已闭环，当前E50属于独立结构实验，总目标及证据要求保持不变。
- [GCIE-Net直接先例补记](research_round_20260925_global/U57_GCIE_DIRECT_PRIOR_DELTA_LIMITED_20261021.md)：补充SAR跨尺度门控先例，保留全文/代码未核边界。
- [GCIE查询读出与H/4特征窄补证](research_round_20260925_global/GCIE_QUERY_READOUT_EVIDENCE_LIMITED_20261021.md)：出版社索引§3.2式1–2显示同任务已有该组合，不单列首创；全文/代码未核，保留整体差异和效用待证。
- [RSI4EO新预印本协议核查](research_round_20260925_global/U57_RSI4EO_PREPRINT_PROTOCOL_CHECK_20261021.md)：区分作者报告与内部实验，检查几何启示，不直接排名。
- [GB10 CPU offload性能可行性备忘](research_round_20260925_global/U57_GB10_OFFLOAD_PERFORMANCE_FEASIBILITY_20261021.md)：仅未来瓶颈假设，尚未开展性能试验，不改变正式E50配方或训练。
- [CUDA初始化失败与定向缓存观察](research_round_20260925_global/U57_CUDA_INIT_FAILURE_AND_TARGETED_CACHE_OBSERVATION_20261021.md)：22:29:44 UTC的pending为历史观察，后续接续已严格接收4352；原0更新失败、MemFree/MemAvailable差异及因果不确定性保留，未证明速度提升。
- [U60恢复检查源码](research_round_20260925_global/code/U60_DGX_SMOKE_RESTORE_ENTRY_DRAFT_20261020_01/README.md)：7项有限测试及静态审阅后另经上述真实执行。目录UNEXECUTED标签为成稿时状态；E50恢复、完整预测及AP仍待验证。
- [U61完整候选导出入口](research_round_20260925_global/code/U61_DEV_FULL100_SINGLE_EXPORT_DRAFT_20261021_01/README.md)已真实通过单图100候选导出与CPU重载；已与U63拼齐187图；DGX完整CPU读取已另经U65通过，零残差校准仍待完成。[查新与直接先例](research_round_20260925_global/U57_NEAREST_PRIOR_NOVELTY_CHECK_20261018.md)。

- [U62剩余186图导出入口](research_round_20260925_global/code/U62_DEV_REMAINING186_EXPORT_DRAFT_20261021_01/README.md)：有限测试和静态审阅通过，尚未绑定或执行；此前资源观察未达到其准入门槛，旧版未执行；后续独立U63已完成剩余导出。

- [U63独立CPU8G导出配置](research_round_20260925_global/code/U63_DEV_REMAINING186_CPU8G_EXPORT_DRAFT_20261021_01/README.md)：已实际通过剩余186图18600候选的导出、保存后重载及Root验收；与首图拼成完整187图索引。独立U65完整CPU读取已通过，零残差RLE校准仍待验证。

- 既有读取证据：187/187包最终远端字节inventory已核验，原传输任务真实退出0；[U65完整CPU读取入口](research_round_20260925_global/code/U65_DGX_DEV187_CPU_READER_VALIDATION_DRAFT_20261021_01/README.md)随后独立实际读取并释放187图/18700候选，198项直接来源闭合、失败与残留均0，已由Root严格接收，摘要另行stage成功。它不证明E50恢复、模型forward、RLE或AP，历史Windows原始来源未重验。8192阶段历史原训练与输入验证管线DGX账户38正常/3失败/41总计包含独立CPU验证和更新前失败，不是纯训练计数。旧2048磁盘故障的额外512次更新及未知SSH/owner退出保持未合格；成功重算不按loss/AP选优，也不宣称逐bit复现或独立种子。
- [U65纯metadata接收器](research_round_20260925_global/code/U65_ROOT_ACCEPTANCE_METADATA_DRAFT_20261021_01/README.md)：原5项有限标准库测试及静态审阅之外，本次另有真实退出、187条audit和来源/日志字节核验并完成Root接收；替身测试本身不构成执行证据。
- [U66固定E50薄预测核心](research_round_20260925_global/code/U66_DGX_FIXED50_PREDICTION_CORE_DRAFT_20261021_01/README.md)：4项有限标准库替身测试、AST及独立静态审阅通过，修正两处真实ABI接线；仍未绑定、未执行，不构成GPU/reader/恢复/forward、owner、CPU指标或AP资格。
- [U67固定E50完整人口薄指标核心](research_round_20260925_global/code/U67_DGX_FIXED50_POPULATION_METRICS_CORE_DRAFT_20261021_01/README.md)：7项有限标准库JSON/替身测试、AST及有限静态审阅通过；尚未绑定或实际读取数据/运行COCO/AP，没有owner或运行资格，不替代真实完整人口评价。
- [U68固定E50单shard薄worker](research_round_20260925_global/code/U68_DGX_FIXED50_SINGLE_SHARD_WORKER_DRAFT_20261021_01/README.md)：3源码文件已发布并核验，6项有限标准库/替身接线测试及静态审阅通过；真实GPU/模型路径未执行；成稿时encoder来源待核的本机起源层已另经U69追溯确认，外部GPU监督/终态与运行资格仍待核，不授owner、预测或AP资格。后续补核确认旧U68/U69有限来源并集漏掉恢复所需FIT reader，旧源码的有限审阅不表示完整预测就绪。
- [U69原encoder起源追溯](research_round_20260925_global/U69_ORIGINAL_DEV_ENCODER_PROVENANCE_LIMITED_20261021.md)：仅确认本机原DEV元数据/读出源码身份，不是teacher权重、特征新提取或运行资格。
- [U69监督入口源码](research_round_20260925_global/code/U69_DGX_FIXED50_SINGLE_SHARD_SUPERVISION_DRAFT_20261021_01/README.md) · [未来评价源码就绪边界](research_round_20260925_global/U69_FUTURE_EVALUATION_SOURCE_READINESS_LIMITED_20261021.md)：五份目录文件冻结，6+7项有限替身测试及静态集成审阅通过；五份未来运行源码已独立DGX暂存并核字节，原mask pin通过独立overlay保留、原环境未改。运行spec未绑定，源码/真实库/模型未执行；全部21个科学源码角色闭合、E50、标定、四分片和CPU COCO及外部Root资格仍待完成。
- [U70总体评价I/O追溯](research_round_20260925_global/U70_POPULATION_EVALUATION_IO_TRACE_AND_NEXT_ADAPTER_20261021.md) · [薄CPU worker源码](research_round_20260925_global/code/U70_DGX_FIXED50_POPULATION_METRICS_WORKER_DRAFT_20261021_01/README.md)：3份源码/说明文件已冻结、同步并核验；7项有限标准库/合成JSON及import边界替身测试、AST与Root/独立静态审阅通过，未见具体阻断。7份未来科学源码、6份原JSON（14004851字节输入）及五文件库候选已独立DGX字节暂存；未绑定、未import、未真实运行或取得ABI/AP/Root资格；实际CPU监督资格及四份真实E50分片仍缺，U71另补未执行监督源码。未来AP三态为源码约定，合成结果不是真AP。
- [U71薄CPU监督入口](research_round_20260925_global/code/U71_DGX_FIXED50_CPU_METRICS_SUPERVISION_DRAFT_20261021_01/README.md)：4文件冻结，7项有限标准库/替身测试、AST及作者外独立与Root静态审阅通过，未见具体阻断；未来entry/owner两份源码已DGX字节暂存。未绑定、未导入或真实运行，不授ABI/AP/Root资格，四份真实E50分片及实际外部门禁仍待。
- [U71窄范围直接先例补查](research_round_20260925_global/U71_TARGETED_DIRECT_PRIOR_UPDATE_20261021.md)：补记9月27日UnfoldCRF预印本对容量/计算控制和个体退化解释的约束；不是10月17日后新发表，不更改当前主判据或增训练臂，也不证明本方法创新或外部代码已复现。
- [U72–U74来源闭合缺口与修复](research_round_20260925_global/U72_U73_PREDICTION_SOURCE_CLOSURE_GAP_AND_FIX_20261021.md) · [U73 worker](research_round_20260925_global/code/U73_DGX_FIXED50_PREDICTION_SOURCE_CLOSURE_DRAFT_20261021_01/README.md) · [U74监督入口](research_round_20260925_global/code/U74_DGX_FIXED50_SINGLE_SHARD_SUPERVISION_DRAFT_20261021_01/README.md)：旧U68/U69遗漏的原FIT代码引用现有独立22角色接口，保留原12整ref/14关闭门，投影原21来源及19模块给U66。作者5+6项有限新测试与独立集成静态审阅未见剩余具体阻断；7份旧候选源码及3份新源码已字节暂存，14份原路径候选仍来自非最终6144端点、须E50重核。未绑定、未执行、无真实恢复/预测/AP资格；训练和模型数学不改。

- [U75/U76外部接收与人口准入边界](research_round_20260925_global/U75_U76_EXTERNAL_ACCEPTANCE_AND_POPULATION_ADMISSION_LIMITED_20261021.md) · [U75单片接收](research_round_20260925_global/code/U75_ROOT_FIXED50_SHARD_ACCEPTANCE_DRAFT_20261021_01/README.md) · [U76修订02完整人口准入](research_round_20260925_global/code/U76_ROOT_FIXED50_POPULATION_ADMISSION_DRAFT_20261021_02/README.md)：源码冻结，作者7+8项有限合成测试及独立集成静态审阅未见剩余具体阻断；01漏共同manifest的源码门已在独立02最小修复，01保留。六份新源码与本备忘已另经远端核验；新helper本轮未DGX字节暂存，未绑定或真实调用；成稿时Root结果保存/CAS及U71 CPU prelaunch语义门尚未接线；后续U78/U79已补源码调用点，真实E50运行仍未接通，不授预测、ABI、Root或AP资格。
- [U77—U79实际调用点与证据范围](research_round_20260925_global/U77_U79_ROOT_INTEGRATION_LIMITED_20261021.md)：保留U77旧01快照缺口及02通用零字段兼容问题，独立03保持原owned身份严格性；新有限测试为U77五项+两次各新增一项、U78六项、U79四项，未重复旧suite。03的进程观察函数已仅用于本次U57终态补强并真实通过；未执行真实E50 COLLECT/ENVELOPE、预测接收/CAS或CPU评价。
- [U80—U81原启动与已保存配置消费](research_round_20260925_global/U80_U81_ORIGINAL_LAUNCH_AND_SAVED_SPEC_LIMITED_20261021.md)：共享真实预检/Popen/wait/二进制日志源码，CPU原样消费U78保存的SPEC；保留01的三处缺口，02最小修复后与U81真共享实现合成联调通过。有限测试为U80六项＋02新增三项、U81五项；未实际启动真实E50预测或CPU评价。Windows与POSIX混合宿主须可信RootAPI的显式字节reader，默认同宿主CLI不能称即用；实际部署和后续post/active回收/接收仍待落实。
- 连接与本机资源（8192阶段历史）：此前7168→7424的连接失败后已恢复并接收；本次8192→8448的连接中断也已恢复，原配置检查通过后仅启动一次，尚未接收。05:59 UTC本机历史空闲32535512KiB低于原门槛，整卡1329MiB，未启动本机训练或新增科学失败。

<details>
<summary>历史阶段记录（以顶部当前状态为准）</summary>

<!-- U57_PROGRESS_20261020_01 -->

### 2026-10-20 阶段记录

DGX PLAIN/FULL 已各完成并接收 **768/37050 次正式更新**（完整一轮后再27图）；本机各2816次保持独立，不能相加。最新本机成对资源检查未全部达标，未放宽限额。**尚无最终AP、已证实涨点或TGRS就绪结论。**

- [训练进展、成本、评估缺口与下一步](research_round_20260925_global/U57_TRAINING_PROGRESS_AND_EVALUATION_READINESS_20261020.md) · [实测标量记录](research_round_20260925_global/results/U57_DGX_PROGRESS_ACTUAL_20261020.json)
- [U58模型恢复草稿](research_round_20260925_global/code/U58_DGX_FIXED50_RESTORE_DRAFT_20261018_01/README.md) · [U59全候选输入草稿](research_round_20260925_global/code/U59_DGX_DEV_FULL_QUERY_PACKAGE_DRAFT_20261018_01/README.md)：18项有限合成测试及独立静态审阅通过，真实恢复、序列化、搬运和零残差校准仍未执行。
- [直接先例查新](research_round_20260925_global/U57_NEAREST_PRIOR_NOVELTY_CHECK_20261018.md) · [风险与否证设计](research_round_20260925_global/U57_FIXED_QUERY_REFINEMENT_RISK_AND_DIRECT_PRIORS_20261018.md)



<!-- U57_ACTUAL_FORMAL_20261018_01 -->

### 2026-10-18 首段历史记录

全741图输入已在DGX实际校验通过；两段接续测试通过；独立正式训练已接收**PLAIN/FULL各256/37050步**。本机保持**各2816/37050步**，当前资源不满足原准入门槛。两条训练记录独立，不能相加。**无新增AP或已证实涨点，尚无TGRS投稿就绪结论。**

- [本轮真实结果、限制及下一步](research_round_20260925_global/U57_ALL741_DGX_TRAINING_ACTUAL_20261018.md) · [结构化实测摘要](research_round_20260925_global/results/U57_DGX_ORIGINAL_TRAINING_ACTUAL_20261018.json)
- [直接先例查新](research_round_20260925_global/U57_NEAREST_PRIOR_NOVELTY_CHECK_20261018.md) · [固定query风险与否证设计](research_round_20260925_global/U57_FIXED_QUERY_REFINEMENT_RISK_AND_DIRECT_PRIORS_20261018.md)
- [训练实现与协议](research_round_20260925_global/code/U57_DGX_ORIGINAL_UPDATE_TRAINING_DRAFT_20261018_01/README.md)：源码成稿时的“未执行”标签保留为历史；当前实际执行状态以上述结果为准。



<!-- U57_NEAREST_PRIOR_20261018_01 -->

[最近直接先例查新（含2026已发表来源）](research_round_20260925_global/U57_NEAREST_PRIOR_NOVELTY_CHECK_20261018.md)：保留5项直接相关先例，区分原文/作者代码/仅索引可得；U形网络+实例条件及图像+粗mask精修不能作为宽泛首创。当前整合效用仍待公平训练，容量/计算匹配为优先解释对照；未核线索明确保留。

<!-- ROOT_U54_U57_ACCEPTANCE_CODE_20261018_01 -->

[全量输入接收元数据草稿](research_round_20260925_global/code/U54_ROOT_ACCEPTANCE_METADATA_DRAFT_20261018_01/README.md) / [DGX训练接收元数据草稿](research_round_20260925_global/code/U57_ROOT_ACCEPTANCE_METADATA_DRAFT_20261018_01/README.md)：辅助将真实退出、回收与逐项字节核验组织为下一段输入；合计5项微型标准库测试通过，尚未用于真实接收。函数不观察外部进程、不读取数据、不自行推进训练，不替代真实执行证据。

<!-- U57_ORIGINAL_UPDATE_CODE_20261018_01 -->

[DGX 独立原更新训练源码草稿](research_round_20260925_global/code/U57_DGX_ORIGINAL_UPDATE_TRAINING_DRAFT_20261018_01/README.md)：保留原数值更新，新增全状态断点接续和配对提交；5项微型标准库测试及独立静态审查通过，修复输入摘要字段不一致。尚未运行模型/训练，必须先真实接收全741图验证，再执行一次独立SMOKE与全新E50；不计入本机2816步，不构成涨点或速度结论。

<!-- U57_STATIC_TRAINING_PLAN_20261018_01 -->

[DGX 原更新全量成对训练设计](research_round_20260925_global/U57_DGX_ORIGINAL_UPDATE_FULL_FIT_TRAINING_PLAN_20261018.md)：独立新环境从零计数，沿用已有结构与原数学更新；先验收741图实际读取，再完成一次接续全链检查，随后固定E50。当前仅设计，训练实现与执行尚未完成。

<!-- U54_ACTUAL_INPUTS_AND_BASELINE_CONTROL_20261018_01 -->

**2026-10-18 最新状态：** 全741图/1661实例的本机CPU便携导出已通过；DGX全量传输与实际读取尚未完成。本机接续因RAM低于原限额实际停止，正式进度仍为2816/37050步/臂，严格失败资格存在原输出捕获缺口。DGX严格诊断中PLAIN原法自重放通过，候选参数超出预锁容差；FULL遇到不支持的确定性反向，未完成比较。候选未采纳，无新增AP。见[最新真实结果及下一步](research_round_20260925_global/U56_ACTUAL_STRICT_CONTROL_AND_MAIN_RESOURCE_STOP_20261018.md)、[全量导出汇总](research_round_20260925_global/results/U53_ALL741_CPU_EXPORTS_ACTUAL_20261018.json)和[严格诊断摘要](research_round_20260925_global/results/U56_DGX_STRICT_EQUIVALENCE_ACTUAL_FAILURE_20261018.json)。


<!-- U53_INPUT_PREPARATION_20261018_01 -->

**2026-10-18 当前状态：** 主训练已核验2816步，下一段中断且退出未知；已保留原件、不计完成。DGX试跑6次真实更新；便携输入64/741图已核验。U53合并导出及U52数值比较尚未执行；无新增AP。见[中断记录及输入准备](research_round_20260925_global/U40_INTERRUPTION_AND_FULL_INPUT_EXPORT_20261018.md)。新增内容远端待同步。


<!-- U51_ACTUAL_AND_U40_RESOURCE_RESUMED_CURRENT_20261018_01 -->

**2026-10-18 当前进度：** 本机已按原限额接续2816→3072，最后已核验端点仍2816/37050（7.60%预算），新段运行中。DGX已实际完成两臂各3次更新的兼容性试跑；全量输入首批32/741图（43匹配实例，约117.78MB）也已导出核验。[实际记录及后续](research_round_20260925_global/U50_DGX_PAIRED_PILOT_ACTUAL_AND_FULL_INPUT_NEXT_20261018.md)、[首批数值](research_round_20260925_global/results/U51_MATCHED_FIT_0000_0032_EXPORT_ACTUAL_20261018.json)。[保留结构的训练计算候选](research_round_20260925_global/code/U52_SHARED_ENCODER_GRADIENT_ACCUMULATION_DRAFT_20261018_01/README.md)已保存源码，梯度等价和速度尚未实测，当前训练未替换。尚无新增AP或创新成立结论；GitHub传输错误仍待恢复，不能称已同步。


<!-- U50_DGX_ACTUAL_ACCEPTED_CURRENT_20261018_01 -->

**2026-10-18 DGX已实际跑通训练：** FULL/PLAIN各完成3次真实更新，共6次，2图/28不同匹配实例；约45秒，Torch分配峰值2.17GiB、进程组RSS11.43GiB，原任务退出0并核验接收。[实际结果及下一步](research_round_20260925_global/U50_DGX_PAIRED_PILOT_ACTUAL_AND_FULL_INPUT_NEXT_20261018.md)、[标量摘要](research_round_20260925_global/results/U50_DGX_PAIRED_PILOT_ACTUAL_20261018.json)。这只是兼容性与资源试点，无AP或加速结论；主训练仍每臂2816/37050（预算7.60%）。[分批导出入口](research_round_20260925_global/code/U51_MATCHED_CPU_EXPORT_DRAFT_20261018_01/README_ENTRY.md)已完成源码，尚未实际导出；下一步按批准备全部741图输入。普通源码及摘要已本地保存，GitHub仍待传输恢复和远端核验。


<!-- U49_ACTUAL_EXPORT_ACCEPTED_CURRENT_20261018_01 -->

**2026-10-18 DGX输入导出已实测：** 2图/28匹配实例已完成原值、紧凑存储和CPU重载核验；私有输入约43.37 MB，worker4.609秒。仅导出，不是训练或AP；[核验摘要](research_round_20260925_global/results/U49_MATCHED_CPU_EXPORT_ACTUAL_20261018.json)。主训练保持每臂2816/37050（预算7.60%）；资源满足原门限后续训。DGX实际运行尚未完成，GitHub传输错误下新增内容仍待远端同步。


<!-- U40_2816_ACTUAL_ACCEPTED_20261018_01 -->

**2026-10-18 主训练已接收2816：** 原2560→2816任务实际退出0并核验接收，两臂各256次更新、各561次匹配实例训练访问；主训现2816/37050（预算7.60%），无新增AP。原任务1060.828秒，资源未越限、检查点恢复及重载核验通过。FULL/PLAIN平均总训练损失0.03616286/0.03624759，仅为优化监测，不能称精度提升。[本段结果及下一步](research_round_20260925_global/U40_U02560_02816_ACCEPTED_AND_DGX_EXPORT_STATUS_20261018.md)、[正式数值](research_round_20260925_global/results/U40_FORMAL_U02560_02816_ACTUAL_20261018.json)。[U49导出入口](research_round_20260925_global/code/U49_MATCHED_CPU_EXPORT_DRAFT_20261018_01/README.md)与[U50 DGX试点核心](research_round_20260925_global/code/U50_DGX_PAIRED_PILOT_BODY_DRAFT_20261018_01/README.md)已完成普通源码，尚未实际导出/远端运行；新环境试点不计U40。此前“本段运行中/2560”记录是历史状态。新改动已保留本地，GitHub传输故障仍待恢复。


<!-- U40_POST_U48_ACTUAL_RESUMED_20261018_01 -->

2026-10-18 接续实况更新：09:09:57 UTC 与09:11:48 UTC两次资源观察均达标，free RAM分别38350995456/38238998528 B、整卡2065/1963 MiB。Root已实际启动并登记原配对2560→2816接续任务，17份训练来源及原配方/限额不变，恢复原2560检查点。当前为运行中，尚无该段终态或新配对预算；有效端点仍2560/37050。此前资源不足描述保留为当时观察，不能据启动记为已完成。 DGX的[独立配对试点可行性](research_round_20260925_global/DGX_PAIRED_PILOT_PORT_FEASIBILITY_20261018.md)已保存：matched-only导出可避开完整教师环境，仍需实际导出与核验；尚无DGX数值试点或速度结论。本轮本地提交待传，GitHub连接传输故障尚未恢复。


<!-- U48_NATIVE_SINGLE_IMAGE_ACTUAL_20261018_01 -->

**2026-10-18 单图原生推理实测：** 原 E24 教师在预选 HRSID TRAIN 图像的全部 100 查询推理已实际退出 0 并接收；原生掩膜/分数/标签校准通过，26.453 秒，分配器峰值约 2.11GiB，显存已释放。原 2GiB 后处理 OOM 失败单独保留，独立修正版只把分配预算改为 3GiB。无新增训练、AP 或全人口资格；主训仍每臂2560/37050，资源余量未达接续门。[实测与下一步](research_round_20260925_global/U48_SINGLE_HRSID_NATIVE_CAPTURE_ACTUAL_AND_NEXT_STEP_20261018.md)、[成功摘要](research_round_20260925_global/results/U48_HRSID_TRAIN0_NATIVE_CAPTURE_3G_ACTUAL_20261018.json)、[失败摘要](research_round_20260925_global/results/U48_HRSID_TRAIN0_NATIVE_CAPTURE_2G_OOM_ACTUAL_20261018.json)、[资源修正版](research_round_20260925_global/code/U48_EXTERNAL_NATIVE_CAPTURE_GPU3G_DRAFT_20261018_01/README.md)。[完整查询覆盖假设](research_round_20260925_global/U40_FIXED50_QUERY_COVERAGE_ERROR_HYPOTHESIS_20261018.md)仅为E50后检验设计。GitHub 连接传输仍失败，本轮及此前待传改动尚未核验远端同步。


**2026-10-18教师恢复实测：** U47原任务退出0并核验接收，原E24教师完整610项状态/451个参数与加载证明一致；worker21.5秒、采样RSS峰值约1.52GiB，CUDA未初始化、模型及anchor已释放。无图像前向或外部AP。[实测及剩余接线](research_round_20260925_global/U47_TEACHER_CPU_RESTORE_ACTUAL_AND_DGX_STATUS_20261018.md)、[恢复代码](research_round_20260925_global/code/U47_EXTERNAL_TEACHER_CPU_RESTORE_DRAFT_20261018_01/README.md)、[数值摘要](research_round_20260925_global/results/U47_EXTERNAL_TEACHER_CPU_RESTORE_ACTUAL_20261018.json)。DGX已实际只读连接；其既有Torch2.10/Python3.12环境不同于原U40锁定环境，不能直接拼接训练，尚无迁移或速度结论。主训有效端点仍2560，下面的历史“teacher未加载”描述以本条更新为准；新图预处理来源、无GT实际前向和E50全人口评价仍待完成。新增本地改动因GitHub连接传输异常标记待同步，未报告远端成功。

<!-- U40_RESOURCE_RECOVERY_CURRENT_20261018_01 -->

**2026-10-18最新状态：** [2560,2816)独立接续尝试因整卡显存8398 MiB超过8192 MiB受控退出；可用内存未低于16 GiB，800.078秒未到时限。[中断记录](research_round_20260925_global/results/U40_U02560_02816_WHOLE_GPU_ABORT_ACTUAL_20261018.json)已核验保存。有效配对进度仍为2560/37050。主训需两次相隔30秒观察整卡基线≤3584 MiB、可用RAM≥32 GiB，原运行限额保持，目前未重启。原版CPU实测确认morph0两个端点AD/FD不一致；独立三行候选U46两方向局部VJP/FD均符合完整累计预期，两次actual0均已核验接收。[实际结果与下一步](research_round_20260925_global/U45_U46_DSC_COORDINATE_ACTUAL_FINDING_AND_NEXT_STEP_20261018.md)。均无真实影像、全网训练或新AP，且端点完整累计已有作者先例，不作为新采样方法或创新成立。

**2026-10-18资源中断补记：** 第11段[2560,2816)原任务因可用RAM低于原16GiB下限受控退出，实际终态与失败记账已接收；整卡峰值8165MiB没有超过8192MiB，775.453秒也未到1800秒期限。正常配对进度仍为每臂2560，失败段不计入已接收预算。原检查点与部分输出保留，未将单臂部分结果当成配对完成。触发低内存的具体进程尚不能归因；并行全人口合成测试亦不能排除干扰，后续训练期间仅做小fixture验证。[实际失败](research_round_20260925_global/results/U40_U02560_02816_RESOURCE_ABORT_ACTUAL_20261018.json)。用户已授权资源充足时按原限额独立接续；总50轮/37050、17科学源、共同顺序和实例读出配方不改。

**外部推理函数层已补齐，均未科学执行：** [E50严格CPU恢复](research_round_20260925_global/code/U44_EXTERNAL_FIXED50_RESTORE_DRAFT_20261018_01/README.md)、[三模型小窗口轮换](research_round_20260925_global/code/U44_EXTERNAL_STREAM_WINDOW_DRAFT_20261018_01/README.md)、[原TEST全人口合并与逐method COCO](research_round_20260925_global/code/U44_EXTERNAL_SPLIT_METRICS_DRAFT_20261018_01/README.md)。分别2/2/3项标准库元数据行为测试通过，Root静态审与窗口独立互接审未发现确定阻断；真实CP恢复、无GT图像前向、资源稳态及外部AP均未运行。teacher实际构造/加载、外部owner入口、来源准入与原片接收仍未实现；强R缺项仍须显式处理，不因这些函数存在宣称跨集可运行。


**2026-10-18实际进展：** 自研完整UNet++与普通对照已接收每臂2560次更新（3轮+337图），[2560,2816)后续原尝试已资源中断，模型进度仍为2560；固定50轮/37050预算不变。第10段每臂614次匹配GT训练访问，训练标量均有限、两臂loss接近，尚无新增实例AP。[第10段实测](research_round_20260925_global/results/U40_FORMAL_U02304_02560_ACTUAL_20261017.json)、[匹配训练监测](research_round_20260925_global/results/U40_U02304_02560_TRAINING_LOSS_MONITOR_20261017.json)。HRSID路径修复后已完整访问5604图/16969实例：全图RGB解码成功，16956实例非空栅格通过、13实例为空（train9/test4），原非空门未满足，actual1失败已保留；未删图或改标注。[实际失败](research_round_20260925_global/results/U43_CANONICAL_POPULATION_EMPTY_RASTER_FAILURE_ACTUAL_20261018.json)、[COCO协议核查](research_round_20260925_global/U43_EMPTY_RASTER_COCO_PROTOCOL_REVIEW_20261018.md)。

**外部单图接线已保存：** [U44薄桥](research_round_20260925_global/code/U44_NEW_IMAGE_ITEM_BRIDGE_DRAFT_20261017_01/README.md)将U41自然e/M/L、原100查询及S0接到U40逐图残差预测，沿原RGB400与几何；两项标准库元数据测试和独立静态ABI审通过，未运行真实像素、张量、窗口、模型或外部AP。窗口/恢复与全TEST COCO的函数层已在本轮补充，均未科学执行；teacher实际加载、完整owner与来源准入仍待接线，强R缺项须显式处理。旧187图包装不能直接用于外部人口。


**完整自主UNet++主线：** 按用户要求以复合编码、H/4语义/双方向融合及完整嵌套解码作为主体，新增每图共享一次的实例读出；[方案与最少对照](refine-logs/UNETPP_FULL_METHOD_20261017_01/EXPERIMENT_PLAN.md)、[执行跟踪](refine-logs/UNETPP_FULL_METHOD_20261017_01/EXPERIMENT_TRACKER.md)、[接口草稿](research_round_20260925_global/code/UNETPP_FULL_INSTANCE_ADAPTER_DRAFT_20261017_01/README.md)。已核历史语义正信号：HRSID单seed F1约+2.40点、SSDD三seed均值约+1.03点；这些不归因给full/new1全部模块或实例AP。共享接口三项tiny真实nn CPU合约已实际通过并经Root接收；[原完整主干feature wrapper](research_round_20260925_global/code/UNETPP_FULL_FEATURE_BACKBONE_DRAFT_20261017_01/README.md)及[真实full CPU64入口](research_round_20260925_global/code/U30_FULL_FEATURE_CPU_DRAFT_20261017_01/README.md)已实际通过原完整卷积主体特征精确等价和eval反向两项；245个有效参数张量梯度均有限，四个旧语义头零调用。真实100查询M/F已实际补采并接收；U36两臂均通过400真实影像两步更新，U38完整FIT读取和共同初始化也已通过；U39跨图资源已通过，正式50轮主训练已启动：前十段两臂各累计2560次已接收（3轮+337图），第11段资源中断已保留，待充足余量独立接续；尚无新增实例AP。[实际结果](research_round_20260925_global/U30_REAL_FULL_FEATURE_CPU64_ACTUAL_RESULT_20261017.md)。[本轮实测与范围](research_round_20260925_global/UNETPP_SHARED_ADAPTER_CPU_RESULT_AND_FULL_FEATURE_STATUS_20261017.md)。

**真实400主干资源试点已通过：** U36两臂统一使用FP32中间激活CPU暂存，普通与自主FULL均完成真实9实例影像两次AdamW更新；FULL第二步245个有效参数张量梯度均有限且非零，峰值GPU分配约1.97GiB、采样RSS约13.00GiB。小模型另做普通/暂存各两步，梯度与Adam状态精确相同。合计8次一次性试点更新，不计入主训练。此前U33/U35显存失败保留；单图通过不代表741图稳定、AP改善或加速。[实际结果与范围](research_round_20260925_global/U36_REAL400_TRAINING_RESOURCE_ACTUAL_RESULT_20261017.md)、[数值摘要](research_round_20260925_global/results/U36_REAL400_CPU_SAVED_ACTIVATIONS_ACTUAL_20261017.json)。

**全量输入与共同初始化已通过：** U38原任务退出0且外部接收，完成全部FIT741图/1661匹配实例的原JPEG、EXIF、RGB400/整数几何及原缓存/GT关联检查；实际完成公共104状态与共享实例头同源初始化。CUDA未初始化，无训练或AP。原尺寸572种、G范围1–29，不能以此前9实例单图代替全部负载。[实际结果](research_round_20260925_global/U38_FULL_FIT_RGB_AND_COMMON_INITIALIZATION_ACTUAL_RESULT_20261017.md)、[脱敏摘要](research_round_20260925_global/results/U38_FULL_METHOD_CPU_READER_INIT_ACTUAL_20261017.md)。后续[U39跨图检查](research_round_20260925_global/code/U39_FULL_METHOD_CROSS_IMAGE_RESOURCE_DRAFT_20261017_01/DESIGN.md)已实际完成并接收；正式训练状态见下段，DEV全100候选输入与评价准备并行推进。

**跨图训练实测与正式实现：** [U39原始试点](research_round_20260925_global/results/U39_CROSS_IMAGE_RESOURCE_ACTUAL_20261017.md)实际退出0并接收：两臂各3次连续更新，覆盖27实例图，状态图间保留；6次丢弃更新不计正式预算。[主方法衔接](research_round_20260925_global/U39_CROSS_IMAGE_RESOURCE_AND_FINAL_METHOD_CONTINUATION_20261017.md)继续沿用已有自定义UNet++卷积骨架。[U40连续训练代码](research_round_20260925_global/code/U40_FULL_METHOD_CONTINUOUS_TRAIN_DRAFT_20261017_01/DESIGN.md)已进入正式训练，当前两臂各累计2560次更新已接收，[2560,2816)原尝试已资源终止并保留；总预算每臂50轮/37,050更新，完整保存恢复模型、Adam、BN、RNG及共同顺序。尚无新增AP。

**正式训练接线：** 原首三段[0,256)、[256,512)、[512,768)均实际退出0并接收；[首段](research_round_20260925_global/results/U40_FORMAL_U00000_00256_ACTUAL_20261017.json)、[第二段](research_round_20260925_global/results/U40_FORMAL_U00256_00512_ACTUAL_20261017.json)、[第三段](research_round_20260925_global/results/U40_FORMAL_U00512_00768_ACTUAL_20261017.json)。第四段[768,1024)也已正常结束，两臂各累计1024更新（第1轮+283图），worker模型/Adam/BN/RNG恢复与保存回读检查通过；[第四段实测](research_round_20260925_global/results/U40_FORMAL_U00768_01024_ACTUAL_20261017.json)。第五段[1024,1280)已实际退出0并接收，每臂604次匹配GT训练访问、worker恢复和回读检查通过；[第五段实测](research_round_20260925_global/results/U40_FORMAL_U01024_01280_ACTUAL_20261017.json)。第六段[1280,1536)已实际退出0并接收，每臂568次匹配GT训练访问、worker恢复和保存回读检查通过；[第六段实测](research_round_20260925_global/results/U40_FORMAL_U01280_01536_ACTUAL_20261017.json)。第七段[1536,1792)已实际退出0并接收，每臂563次匹配GT训练访问、worker恢复和保存回读检查通过；[第七段实测](research_round_20260925_global/results/U40_FORMAL_U01536_01792_ACTUAL_20261017.json)。第八段[1792,2048)已实际退出0并接收，每臂574次匹配GT训练访问、worker恢复和保存回读检查通过；[第八段实测](research_round_20260925_global/results/U40_FORMAL_U01792_02048_ACTUAL_20261017.json)。第九段[2048,2304)已实际退出0并接收，每臂550次匹配GT训练访问、worker完整恢复和保存回读检查通过；[第九段实测](research_round_20260925_global/results/U40_FORMAL_U02048_02304_ACTUAL_20261017.json)。第十段[2304,2560)也已实际退出0并接收，每臂614次匹配GT训练访问、恢复和保存回读检查通过；[第十段实测](research_round_20260925_global/results/U40_FORMAL_U02304_02560_ACTUAL_20261017.json)。该段原尝试已受控资源终止，完成数仅计2560；下一次须在原限额及充足余量下独立接续。固定每臂50轮/37050更新才完整DEV评价；不以中途AP选型，不额外加pilot。

**训练拟合监测：** [前1280更新摘要](research_round_20260925_global/results/U40_FIRST1280_TRAINING_LOSS_MONITOR_20261017.json)由第5段实际终止并接收后、下一任务启动前，对五段已接收原JSON trace作标准库标量汇总，未重放tensor/图像/GT/模型。loss/BCE/softDice均有限；第五段同图平均loss普通0.038416、自主0.038289，差异很小。它只反映匹配训练实例的拟合，不能判定AP、全部候选假阳性、收敛或创新，不据此选轮次或改配方。第6段[匹配训练loss监测](research_round_20260925_global/results/U40_U01280_01536_TRAINING_LOSS_MONITOR_20261017.json)同图均值为普通0.035898、自主0.035758，仍非常接近；全部有限，没有开发评价或配方调整。第7段[训练拟合监测](research_round_20260925_global/results/U40_U01536_01792_TRAINING_LOSS_MONITOR_20261017.json)同图均值普通0.037304、自主0.037189，仍接近，未用于改配方或选择端点。第8段[训练拟合监测](research_round_20260925_global/results/U40_U01792_02048_TRAINING_LOSS_MONITOR_20261017.json)同图均值普通0.035317、自主0.035126，全部有限且差距小，没有新增AP或选型。第9段[训练拟合监测](research_round_20260925_global/results/U40_U02048_02304_TRAINING_LOSS_MONITOR_20261017.json)同图均值普通0.037823、自主0.037617，仍只支持匹配训练拟合、没有AP或收敛结论。

**固定端点评价研发：** [E50完整DEV评价设计](research_round_20260925_global/U40_FIXED50_DEV_EVALUATION_DESIGN_20261017.md)已有[预测核心](research_round_20260925_global/code/U40_FIXED50_DEV_PREDICTION_DRAFT_20261017_01/DESIGN.md)和[完整人口CPU评价核心](research_round_20260925_global/code/U40_FIXED50_DEV_METRICS_DRAFT_20261017_01/README.md)草稿，各3项纯元数据测试通过；[数值外层入口](research_round_20260925_global/code/U40_FIXED50_DEV_ENTRY_DRAFT_20261017_01/README.md)也已实现，另3项纯元数据测试与独立静态接口审阅通过，均未实际数值执行；[RGB读取桥](research_round_20260925_global/code/U40_FIXED50_DEV_RGB_READER_IMPLEMENTATION_DRAFT_20261017_01/README.md)已完成源码、静态与3项纯元数据测试，真实读取和数值路径尚未执行。仅在每臂37,050更新完成并接收后评价187图/18,700查询；保留原查询映射、S0及全部候选。187个原JPEG候选引用已从四份原JSON采集清单精确提取，桥的真实metadata函数已通过；[准备摘要](research_round_20260925_global/results/U40_FIXED50_DEV_METADATA_PREPARATION_ACTUAL_20261017.json)。尚未读取JPEG/tensor/GT/权重或执行NN/COCO，实际文件/数值准入仍待固定端点；不补teacher、不用中途AP选型。

**直接先例更新：** [O2Former与DACF-net查新](research_round_20260925_global/U40_DIRECT_PRIOR_UPDATE_20261017.md)区分正式发表与arXiv证据、离岸/近岸与整体AP，以及结构/骨干/后处理版本；方向、跨层调制和残差融合已有直接先例，当前结构收益与差异仍待同预算和机制对照。[固定query精修与点提示比较范围](research_round_20260925_global/U40_FIXED_QUERY_REFINEMENT_AND_PROMPT_PROTOCOL_PRIORS_20261017.md)补核ReSAM正式索引/v1/作者代码及CascadePSP先例，不改正在运行的配方。

**新图采集接口研发：** [U41单图原生采集草稿](research_round_20260925_global/code/U41_UNBOUND_NEW_IMAGE_NATIVE_CAPTURE_DRAFT_20261017_01/README.md)复用原观察、渲染、评分函数，区分最终e与旧pre-hook输入h；纯影像流程沿实际库data_samples=None分支保留原training=True预处理/固定补边，不读取标注或伪造GT。3项标准库接口测试与独立静态审阅通过，尚未执行像素/NN或取得新人口资格；HRSID1962、SSDD232及模型接触来源仍待独立核验，不改当前U40。[U42单图像素/COCO栅格检查](research_round_20260925_global/code/U42_UNBOUND_HRSID_PIXEL_MASK_CHECK_DRAFT_20261017_01/README.md)也已形成未绑定草稿，3项标准库接口/失败注入测试与独立静态审阅通过；保留原实例分组、连续/栅格面积差异和有限越界坐标，没有真实解码、GT栅格或人口资格。来源/数据边界见[跨集准备度](research_round_20260925_global/N16_FORMAL_TEST_AND_CROSS_DATASET_EVIDENCE_READINESS_20261014.md)。

**HRSID全集输入检查研发：** [固定执行计划](research_round_20260925_global/U43_HRSID_FULL_PIXEL_MASK_INPUT_CHECK_PLAN_20261017.md)、[U43完整人口核心](research_round_20260925_global/code/U43_HRSID_POPULATION_PIXEL_MASK_CHECK_DRAFT_20261017_01/README.md)与[CPU入口/工作体](research_round_20260925_global/code/U43_HRSID_POPULATION_CHECK_ENTRY_DRAFT_20261017_01/README.md)已完成6项标准库元数据/失败路径测试和一次独立静态互接审。固定原train3642/11047、test1962/5922，全5604图/16969原实例调用U42，不合并ID命名空间、不去重或过滤失败项；全程不导入Torch、不执行模型。首次原任务已actual1并完成Root失败保留：在PIL库引用核验处4.844秒提前退出，未像素解码/原split记录解析/COCO栅格，0逐图记录；入口此前已核编码Source字节。[实际失败](research_round_20260925_global/results/U43_HRSID_POPULATION_LIBRARY_PATH_FAILURE_ACTUAL_20261017.json)、[路径原因](research_round_20260925_global/results/U43_LIBRARY_CANONICAL_PATH_DIAGNOSTIC_20261017.json)。十二库的物理路径与入口别名不同、bytes/SHA相同；[独立修复计划](research_round_20260925_global/U43_CANONICAL_LIBRARY_REFERENCE_REPAIR_PLAN_20261017.md)保留原代码与严格门，尚未绑定或重新执行。没有取得完整输入资格、外部AP或盲测证明。

**外部推理接线设计：** [U44固定HRSID外部路线](research_round_20260925_global/U44_HRSID_FIXED_EXTERNAL_PREDICTION_DESIGN_20261017.md)采用固定小CPU窗口、teacher/普通/自主主干逐臂轮换上卡，仅保存最终RLE及必要身份，避免全1962图dense logit缓存。仍待主E50效用、真实输入检查及新图接口闭合，不改主训练或宣称已运行外部AP；强R缺项不得临时用内部RLE冒充。

**教师初始化来源核对：** 已在第7段实际终止接收后、下一段启动前，沿当前FIT/N13/N16原JSON加载回执核对，五处native0 E24 teacher引用一致，并追到原COCO initializer。其本地完整SHA与绑定ref相等、官方文件名校验前缀/响应长度/首4MiB SHA一致，MD5字串与ETag相同；[实际多指纹摘要](research_round_20260925_global/results/M2F_COCO_INITIALIZER_ENDPOINT_FINGERPRINT_LINKAGE_20261017.json)。[官方全流读取尝试](research_round_20260925_global/results/M2F_OFFICIAL_COCO_REMOTE_BYTES_WITNESS_20261017_01.json)90秒仅读4MiB，仍保留未完成状态；ETag不被假定为服务端MD5，未完成全量远端SHA256，也不声称独立训练谱系或消除历史TEST暴露。[范围与原始来源](research_round_20260925_global/N16_FORMAL_TEST_AND_CROSS_DATASET_EVIDENCE_READINESS_20261014.md)。

**U38执行源码：** [CPU入口与工作体](research_round_20260925_global/code/U38_FULL_METHOD_CPU_ENTRY_DRAFT_20261017_01/DESIGN.md)和[组合资格函数](research_round_20260925_global/code/U38_FULL_METHOD_CPU_QUALIFIER_DRAFT_20261017_01/DESIGN.md)已实际执行并接收；完整原JPEG引用有序摘要一致、公共初始化已实际构造/核对后释放，Root接收没有重放像素或神经计算。

**完整主干后续安排：** [U38原最小验证方案](research_round_20260925_global/U38_FULL_METHOD_READER_AND_RESOURCE_GATE_DESIGN_20261017.md)的CPU部分已实际完成；U39保持相同FP32/400/chunk1/CPU暂存与AdamW，验证跨图状态保留及资源。普通和自主完整主干仍拟同50轮/37050更新，不从试点loss选型、不增加训练不需要的全FIT候选补采。

**完整主干的真实输入（U32已完成）：** 冻结模型在一张9实例FIT影像上真实取得100查询M/F，原L、查询、类别、GT/匹配、分数及输出顺序与既有锚点精确一致。实际退出0并接收；无优化器、DEV、TEST或AP。U36已完成普通/自主400主干每臂两次一次性更新；仅该真实单图范围，见上方最新结果。[结果和具体试点](research_round_20260925_global/U32_NATIVE100_CAPTURE_ACTUAL_AND_U33_REAL400_PLAN_20261017.md)。

**最新实测（N26，三预锁种子完成）：** 固定R掩膜，仅学习质量乘分；AP为69.423613 / 69.637142 / 69.551345，均值**69.537367 ± 0.107449**（样本SD），相对原R 68.099385平均**+1.437982 AP点**。各624次更新及完整187图/380GT/18700query COCO均实际结束并经Root接收，qhat1原COCO12精确复现。**这是已有质量回归适配的评分效用，掩膜未改善，核心创新尚未成立。** [完整结果](research_round_20260925_global/N26_FIXED_R_QUALITY_THREE_SEED_RESULT_20261017.md)、[全指标与成本](research_round_20260925_global/results/N26_FIXED_R_QUALITY_THREE_SEED_AGGREGATE_20261017.json)、[主张评议](research_round_20260925_global/N26_RESULT_TO_CLAIM_20261017/findings.md)、[执行前计划](refine-logs/N26_QUERY_MASK_QUALITY_BASELINE_20261017_01/EXPERIMENT_PLAN.md)、[训练评价源码](research_round_20260925_global/code/N26_QUALITY_TRAIN_EVAL_SOURCE_ONLY_20261017_01/README.md)。

**当前结构研发（N28，固定E24完整DEV已完成）：** 两臂各17784更新；187图/380GT/18700查询统一COCO已实际完成并接收。SELECTIVE **67.021271 AP**，WHOLE_NODE **66.655257**、强R **68.099385**；配对+0.366014，预锁强参考门槛Delta−2.516096。Root决定停止本路由配方扩展，不加种子/预算/融合；[结果、几何与主方法接续](research_round_20260925_global/N28_FIXED24_FULLDEV_RESULT_AND_FULL_METHOD_CONTINUATION_20261017.md)、[全部指标](research_round_20260925_global/results/N28_FIXED24_COMPLETE187_COCO_ACTUAL_20261017.json)。该试验不代替完整自研UNet++主干验证；主方法已从每臂1792更新接续，固定50轮预算不变。

**N28查新与评价研发：** [五项直接先例核查](research_round_20260925_global/N28_DIRECT_PRECEDENT_REVIEW_20261017.md)明确分支门控已有先例；[固定24轮评价设计](research_round_20260925_global/N28_FIXED24_DEV_EVALUATION_DESIGN_20261017.md)及[评价源码](research_round_20260925_global/code/N28_FIXED24_DEV_EVAL_DRAFT_20261017_01/README.md)已用于原固定端点评价。四个GPU分片与完整CPU COCO、几何诊断均已实际完成并接收，范围及停止判断见上段；文件名中的DRAFT保留原源码身份，不表示本次仍未运行。

**N28恢复预锁与源码：** [恢复预锁](refine-logs/N28_DENSE_ROUTING_20261017_01/RESOURCE_RECOVERY_PLAN.md)、[CPU检查源码](research_round_20260925_global/code/N28_CHECKPOINT_RECOVERY_SOURCE_ONLY_20261017_01/README.md)与[单轮修订训练源码](research_round_20260925_global/code/N28_SINGLE_EPOCH_TRAIN_SOURCE_ONLY_20261017_01/README.md)均在执行前发布。CPU恢复检查及第6至24轮原GPU训练已实际结束并接收；原8轮任务的超时仍保留为失败。训练接续成功不代表方法已经涨点。

**最新机制诊断（N27，完整FIT741/1661已实际完成）：** R剩余错误73.8521%在固定Ω外。IoU0.90/0.95当前未达标实例中，即使域内用GT完美纠正，仍无法达标的分别为424/581、1101/1304。该结果支持检验具备域外写入能力的结构，不证明门控有效、AP上界或核心创新。对应N28路由两臂保持同输入、同写入权限、同预算，固定第24轮完整DEV已完成；SELECTIVE相对WHOLE_NODE +0.366014AP点，但低于强参考，按规则停止该路由配方。[结果及下一项约束](research_round_20260925_global/N27_MATCHED_SUPPORT_CENSUS_RESULT_20261017.md)、[全部阈值与成本](research_round_20260925_global/results/N27_MATCHED_SUPPORT_CENSUS_AGGREGATE_20261017.json)、[预锁计划](refine-logs/N27_MATCHED_SUPPORT_CENSUS_20261017_01/EXPERIMENT_PLAN.md)、[源码](research_round_20260925_global/code/N27_MATCHED_SUPPORT_CENSUS_SOURCE_ONLY_20261017_01/README.md)。正式TEST、跨集和完整结构验证仍待完成。

**UNet++ 模块复核（2026-10-17）：** 已重新核对自定义 CBAM_block 内部的卷积、注意力、残差及 SimAM 增强，H/4 方向融合和 up_x 传输路径；补核发现解码块前置 BN/ReLU 结果被 `conv1(x)` 覆盖，已纠正此前的顺序描述并记录独立修复对照方案。这是实际源码确认，尚非完整公平训练证据。当前 N26 未直接训练这套 UNet++。[具体改动与验证边界](research_round_20260925_global/UNETPP_MODULE_MODIFICATIONS_CONFIRMED_20261017.md)。

**前轮结构训练（N23，当前配方STOP）：** 三臂均完成固定新增24轮、每臂39864次更新；完整187图/380GT/18700query统一COCO已实际完成并经Root接收。B mask AP **67.807324**，A_COMMON **67.788929**、A_ACTIVE **67.603337**，强R **68.099385**；主Delta **−0.292061 AP点**，不追加本配方种子/预算/调参/融合。原native/R的全部预测及COCO12精确复现。[完整结果与范围](research_round_20260925_global/N23_FIXED24_FULLDEV_RESULT_AND_STOP_20261017.md)、[全部指标与成本](research_round_20260925_global/results/N23_FIXED24_FOURHEAD_COCO_AGGREGATE_20261017.json)、[结果到主张](research_round_20260925_global/N23_RESULT_TO_CLAIM_20261017/findings.md)。

**最新诊断（N24已完成）：** 完整187图/380GT/18700query、五组预测的原COCO12全部精确复现。所有已检阈值/top10/100下均无贪心匹配数量差；强R在IoU0.90/0.95仅112/4个GT有达标候选，确认高精度轮廓空间，但不能排除score排序损失。[完整结果与限制](research_round_20260925_global/N24_FULLDEV_ERROR_CENSUS_RESULT_20261017.md)、[五组聚合](research_round_20260925_global/results/N24_FULLDEV_ERROR_CENSUS_AGGREGATE_20261017.json)、[预锁原文](refine-logs/N24_FULLDEV_ERROR_CENSUS_20261017_01/EXPERIMENT_PLAN.md)、[源码](research_round_20260925_global/code/N24_FULLDEV_ERROR_CENSUS_SOURCE_ONLY_20261017_01/README.md)。

**最新诊断（N25已完成，GT oracle）：** 五组×三评分完整COCO已actual0并Root接收；R固定mask的S0×GT质量由68.099385到73.396406（+5.297021 AP点），仅GT质量为3.380706。全部几何不变，原5组COCO12精确复现。**这是不可部署诊断，没有新模型涨点。** [完整结果](research_round_20260925_global/N25_FIXEDMASK_SCORE_ORACLE_RESULT_20261017.md)、[15组指标](research_round_20260925_global/results/N25_FIXEDMASK_SCORE_ORACLE_AGGREGATE_20261017.json)、[判断范围](research_round_20260925_global/N25_RESULT_TO_CLAIM_20261017/findings.md)。



**已保存的研发与证据：** [固定训练方案](refine-logs/N23_ANCHORED_DOMAIN_TRAIN_20261017_01/EXPERIMENT_PLAN.md)、[训练源码](research_round_20260925_global/code/N23_ANCHORED_DOMAIN_TRAIN_SOURCE_ONLY_20261017_01/README.md)、[固定评价方案](refine-logs/N23_FIXED_E24_FOURHEAD_DEV_20261017_01/EXPERIMENT_PLAN.md)、[评价源码](research_round_20260925_global/code/N23_FOURHEAD_FIXED_E24_DEV_SOURCE_ONLY_20261017_02/README.md)、[N22实际机会诊断](research_round_20260925_global/N22_FIT_FULLC_QUALIFIED_DIAGNOSTIC_RESULT_20261017.md)、[真实全FIT覆盖](research_round_20260925_global/N23_REAL_FULLFIT_CACHE_COVERAGE_RESULT_20261017.md)、[2026最新直接先例及监督边界](research_round_20260925_global/N23_DIRECT_SUPERVISION_AND_2026_STRUCTURAL_PRIOR_CHECK_20261017.md)。

论文、方法源码、协议和研究记录的工作快照，更新至 **2026-10-17**。研究仍在进行；当前没有可直接投稿的最终论文，也没有经过正式测试和跨数据集验证的 SOTA 结论。



**自主设计与作者基线归属补充：** 团队自定义UNet++按完整结构评估；历史D目录的单个FPN文件与DiffSARShipInst作者固定版本统一换行后完全一致，SCJE-FPN及其中采用的CARAFE/ACmix须按作者基线引用。两份本地R535配置声明启用SCJE，不能仅改builder构造普通FPN对照；这不认证当前E/N20调用。[归属、实际路由与对照要求](research_round_20260925_global/UNETPP_OWN_DESIGN_AND_DIFFSARSHIPINST_BASELINE_ATTRIBUTION_20261016.md)。

**自定义 UNet++ 历史训练证据：** full 版本有明确训练入口；SamplingBoundary 的 HRSID、SSDD 配置与各 5 轮、每轮 64 批的历史日志对应。未确认历史模型源码及权重身份，也不据此推断实例 AP 收益。[入口、短训记录与证据边界](research_round_20260925_global/UNETPP_HISTORICAL_TRAINING_ROUTE_EVIDENCE_20261017.md)。

**当前状态（N20）：** 第二种子全187图统一COCO及独立Root接收均已完成，阶段结果冻结为`STOP_CURRENT_RECIPE`。固定主终点E24-atmost下，seed201 LOCAL/CENTER/强R仍为68.297126/67.978791/68.099385 AP，seed202仍为67.991722/67.697705/68.099385 AP；平均主差值为+0.045039 AP点（精确值+0.04503879895506202），未达到预锁1点门槛，不启动seed203、追加预算或融合。恢复原执行任务实际退出0，exactheld已保存，fresh221 FIRST实际退出0后，Root外部接收实际完成；N20接收时为221正常终态+27保留失败=248、legacy pins229。原post19115实际退出1作为failure27保留，恢复只修复metadata的JSON键比较，没有重训或重算十组COCO。独立N21几何结果见下，不改变本阶段停止结论。旧独立审阅仍为WARN/provisional partial；本阶段不证明总体胜过强R、创新成立或TGRS就绪，总目标未完成。[两seed全部结果与阶段判定](research_round_20260925_global/N20_TWO_SEED_MAIN_RESULTS_20261017.md)。

**已完成（N21）：** 预锁32张FIT图×64个唯一整数中心、共2048组的理想插值几何诊断已实际0并完成Root外部接收。主分母2002/2048满足50%覆盖要求，同一原生2×2单元为0/2002，按原80%判据为`SHARED_FOUR_PREMISE_REJECTED`；主分母的理想系数矩阵均rank9。撤回“普遍同四角→低秩瓶颈”前提，不据此启动原生邻域训练。当前全局222正常+27失败=249、legacy pins230；不更改N20停止规则，不推实际F/LN rank、训练点分布、AP或总体模型有效性，总目标未完成。[完整结果](research_round_20260925_global/N21_NATIVE_SUPPORT_GEOMETRY_RESULT_20261017.md)、[结果前预锁原文](research_round_20260925_global/N21_NATIVE_SUPPORT_GEOMETRY_PRELOCK_20261017.md)、[机器可读摘要](research_round_20260925_global/results/N21_NATIVE_SUPPORT_GEOMETRY_SUMMARY_20261017.json)。接收前的失败及恢复时间线保留在两seed主结果说明和Git历史中。

**复核开销测量：** 第二批外部接收实际完成分段计时，共297.468秒，其中原来源及保存证明核验293.453秒；不能进一步归因于某一具体检查，也不作为模型FPS或加速收益。当前32范围、科学入口与核验规则保持。[实测分项与后续工程候选](research_round_20260925_global/N20_CPU_REVIEW_COST_AND_FUTURE_PROTOCOL_20261016.md)。

**首种子201的全量汇总状态：** 已按32份固定范围执行全187图统一COCO，并完成独立post与Root实际终态核验；静态审阅仅保留为执行前历史。[静态审阅记录](research_round_20260925_global/N20_FINAL32_ENTRY_STATIC_READINESS_REVIEW_20261016.md)。

**跨集数据准备：** 本地5604个JPG的train/test实例标注，与已核官方PNG对应split按文件stem、ID及全部标注字段精确相等；完整文本SHA不同。5604文件编码哈希/头部检查实际0，均800×800三分量JPEG，尚未解码像素/mask。发现两组跨train/test相同编码图、一组test内部重复，其标注多重集均不同；原协议保留，未来另预锁去重敏感性补充。合并标注16951实例与split合计16969不等，不能互换。官方JPG归档身份、掩膜空间对齐和母景独立未确认，不恢复盲TEST；不继续等待PNG下载，不改变N20。[本地数据完整补充](research_round_20260925_global/HRSID_LOCAL_JPG_METADATA_AND_VERSION_BOUNDARY_20261015.md)。

**增量查新：** [O²Former与DACF-net先例核查](research_round_20260925_global/DIRECT_SAR_INSTANCE_SEGMENTATION_INCREMENTAL_PRIOR_CHECK_20261015_01.md)分别标记全文和官方域索引证据，不与当前DEV跨协议排名；方向、多尺度和上下文模块已有直接先例；已补记v1实现需澄清的形状与对齐边界，不将自行适配称作完整作者复现。

**采样与精修直接先例补查：** LIIF已包含原生3×3、相对坐标及解码后局部集成；I-MedSAM展示先插值到目标网格再作连续精修的另一条路线。条件rank上界不是本项目实测瓶颈；原生邻域本身不能算新原语。七项有限证据区分代码、正文与索引摘录，不宣称穷尽2026研究。[有限补查与设计边界](research_round_20260925_global/N20_NATIVE_GRID_VS_RENDERED_F9_DIRECT_PRIOR_REVIEW_20261016_01.md)。

**自主UNet++与历史资产扩展审阅：** 按用户说明记录自主设计，已还原13组实际前向结构；重点保留选择性跨尺度传输、H/4语义与方向互补、采样重建协同三条候选。扩展登记1009个源码/配置/示例项；D内260个模型候选中222个关联数值训练记录，另20个pth是评价缓存。Mamba100轮/MSCK160轮等检测成绩与实例mask AP分开；旧full骨架与clean50轮不能混同，来源未接上不等于未训练。[扩展总评与下一步](research_round_20260925_global/UNETPP_EXPANDED_MODEL_ASSET_REVIEW_AND_NEXT_DESIGN_20261015.md)、[自主结构还原](research_round_20260925_global/UNETPP_SELF_DESIGNED_STRUCTURE_RECONSTRUCTION_20261015.md)、[全部源码目录](research_round_20260925_global/UNETPP_COMPLETE_MODEL_SOURCE_CATALOG_20261015.md)、[训练资产目录](research_round_20260925_global/UNETPP_COMPLETE_TRAINED_ASSET_CATALOG_20261015.md)、[有界先例核对](research_round_20260925_global/UNETPP_SELF_STRUCTURE_BOUNDED_PRIOR_CONTEXT_20261015.md)。三条均尚未确认学术新颖性或当前实例效用，不改N20端点。

**UNet++模块内部改造复核（10月17日）：** 已重新核到编码与解码复合注意力、抗混叠下采样、门控细节上采样四组内部设计及实际forward；full有`trainer_coco.py:5/62`训练入口。full的未使用BlurPool结果、前置BN旁路及DSConv批次索引问题仍为静态证据，尚未张量回归；当前N20不是这套UNet++整网试验。[具体实现、问题与验证范围](research_round_20260925_global/UNETPP_MODULE_INTERNAL_CHANGES_SOURCE_CHECK_20261016.md)。

**UNet++末端门控机制补充：** 本地SimAM核心公式与作者公式相同，外层 `h + SimAM(h)` 构成带恒等项的空间选择增益；需用固定增益等对照区分选择作用与整体放大，尚未执行验证。[公式、适用条件与候选对照](research_round_20260925_global/UNETPP_SIMAM_RESIDUAL_MECHANISM_STATIC_NOTE_20261016.md)。

**采样候选直接先例补查（10月16日）：** 已核BlurPool、FADE、DySample、FreqFusion，以及2025遥感双向重采样和2026抗混叠分割先例；模块内部改造已确认，联合机制与实例效用待验。记录聚合/相位、普通残差容量、外部skip及共享BN的替代解释；不改当前N20。[有限查新与可否定假设](research_round_20260925_global/UNETPP_SAMPLING_DIRECT_PRIOR_TARGETED_REVIEW_20261016.md)。

**自主传输路由的近年先例补查（10月16日）：** 新读ICUnet++（2023）、RPAU-Net++（2025）及两篇遥感分割作者稿，补充直接门控位置对照；不以“UNet+++CBAM”称新。Root确认原model_new1为先门控后上采样，后续需区分运算顺序与被门控支路，所列对照尚未执行。[有界查新与源码顺序校核](research_round_20260925_global/UNETPP_UPSAMPLE_TRANSPORT_RECENT_DIRECT_PRIOR_20261016.md)。

**旧自主结构的权重来源补查：** 222已定位writer/reader代码声明；111/222/两份ALL_NEW完成只读ZIP与pickle操作码元数据登记，未加载模型或tensor。不同前向、开关与完整训练身份仍需核对；部分旧AP50/AP75实际是整幅掩膜IoU阈值通过指标，不能用于COCO mask AP排名。[来源与结构线索](research_round_20260925_global/UNETPP_LEGACY_PROVENANCE_AND_CONTAINER_EVIDENCE_20261015.md)、[有界只读工具](tools/legacy_weight_metadata_intake.py)、[两项合成回归](tests/test_legacy_weight_metadata_intake.py)。

**已完成（N16）：** 同24轮配对训练及完整187图/380GT/18700自然query/12策略GPU、独立数值CPU、统一全量COCO与Root资格均完成。主Utility AP **67.666732**，同训Error **67.895932**，强参考 **68.099385**；Delta **−0.432653** AP点，按预锁STOP，不补旧162/163。原生67.071057的局部收益不能替代强参考。原受控截止/接口失败均保留，不改主端点；单首初始化内部DEV，没有跨集、核心创新或TGRS结论。[完整结果和范围](research_round_20260925_global/N16_FULLDEV_QUALIFIED_FIXED24_RESULT_AND_STOP_20261015.md)、[独立质疑](research_round_20260925_global/N16_RESULT_TO_CLAIM_AND_LOCAL_CONTEXT_INDEPENDENT_CHALLENGE_20261015.md)。

**最近已完成的单模块结果（N14）：** 固定新stage24训练与完整DEV187/all100 GPU、独立CPU及ROOT资格均通过（每臂39,864更新，全18,700 query同GPU输入sign/RLE/COCO12一致）。BASE AP67.686612、MEMORY67.683479；模块−基线−0.003132点、相对原较强E24端点−0.415905点，C1未通过，按预锁规则停止该配方追加seed/预算和融合。训练BCE下降约1.41%没有转为AP收益；不推断方法总体失效或充分收敛。[完整合格结果与停止决定](research_round_20260925_global/N14_FIXED24_FAIR_COMPARISON_QUALIFIED_RESULT_AND_STOP_20261014.md)、[首次AP前规则](research_round_20260925_global/N14_FIXED24_FIRST_SEED_DECISION_CLARIFICATION_20261014.md)。下一轮[固定强E24行动机会诊断](refine-logs/N15_FIXED_ACTION_OPPORTUNITY_20261014_01/EXPERIMENT_PLAN.md)已于UTC04:31:14预登记，[真实GPU、独立CPU与ROOT诊断现已全部合格](research_round_20260925_global/N15_FIXED_E24_ACTION_OPPORTUNITY_QUALIFIED_RESULT_AND_NEXT_STEP_20261014.md)：同K的C行动oracle相对Omega平均+1.461603 IoU点，286/374>=.5点，高IoU实例增加39，预锁机会门通过。当前是GT理想行动选择上界，没有学习选择器的AP结果，[N16唯一公平scout协议现已锁定](research_round_20260925_global/N16_FAIR_UTILITY_SELECTOR_LOCKED_TRAINING_DESIGN_20261014.md)，已核原matcher负query不受mask监督，双方共同扩展背景代理target并纠正分层采样比例；完整缓存已通过GPU、独立CPU和ROOT核验，两scout已完成相同固定24轮训练及E8/E24端点资格；两个实现失败和原641/741受控时限失败均保留。固定无GT候选C、相同K预算和374匹配条件人口，区分实际PR可修补机会、避免伤害及理想GT上界。GO门提前锁定，没有新AP或核心创新结论。原[下一假设](research_round_20260925_global/N15_ACTION_UTILITY_SUPPORT_SELECTION_NEXT_HYPOTHESIS_20261014.md)保留提出时状态。

**当前研发（N17已完成）：** 同预算24轮LOCAL/CENTER、全187图/380GT/18700query、五GPU与七数值CPU、统一全量官方COCO及post实际退出后Root资格均完成。主E24-atmost LOCAL **67.786531**、CENTER **68.103370**、强R **68.099385**，Δ **−0.316840 AP点**，按预锁STOP当前配方，不补172/173、不融合。正常136/失败21/157终态身份；没有运行中的N17科学任务。N18诊断已实际完成并核验，见下一段；没有创新/TGRS就绪结论。[完整结果](research_round_20260925_global/N17_FULLDEV_QUALIFIED_FIXED24_RESULT_AND_STOP_20261015.md)、[全部10端点原始stats](research_round_20260925_global/results/N17_FIXED24_ALL10_COCO_STATS_20261015.json)、[claim判定](research_round_20260925_global/N17_RESULT_TO_CLAIM_20261015/findings.md)。

**N18已核诊断：** 原374固定配对，LOCAL可修选中4574/有害3651，强Ω6564/8528；LOCAL同时减少纠正和误改，漏选率58.140%高于Ω39.929%。PR可修11146/65423（17.037%）与N15一致，不能据此认定主要AP瓶颈。原CPU实际0/472.219秒，正常137/失败21/158身份，无科学任务运行。下一项先研究FP/FN的非对称修补取舍，再锁定公平强对照；没有新训练/AP/创新结论。[完整统计和下一假设](research_round_20260925_global/N18_QUALIFIED_ROUTING_CAPABILITY_RESULT_AND_NEXT_HYPOTHESIS_20261015.md)、[像素数与AP边界](research_round_20260925_global/N18_PIXEL_COUNTS_VERSUS_AP_INTERPRETATION_20261015.md)。

**最新完成（N19）：** 固定1/2方向评分DIR AP **67.726509**，普通偏置67.816800、反向偏置67.789369、强R68.099385；主Δ **−0.372876点**，当前方案STOP，不补评分参数/质量头/种子。全187图/18700query及Root资格完成，正常138/失败21/159身份，无科学任务运行。后续真实feature邻域同容量/同预算检验N20已完成缓存资格并启动原训练任务，见顶部当前状态。[全部结果与范围](research_round_20260925_global/N19_FIXED_HALF_FULLDEV_QUALIFIED_RESULT_AND_STOP_20261015.md)、[原始stats12](research_round_20260925_global/results/N19_FIXED_HALF_ALL6_COCO_STATS_20261015.json)。

**机会与指标的边界：** [条件IoU机会不能保证全量AP](research_round_20260925_global/N17_CONDITIONAL_IOU_OPPORTUNITY_VERSUS_FULL_AP_20261015.md)给出结果后最小诊断：区分实际选点漏掉可修位置与固定PR本身不可修；N18统计已完成，不改变主端点/停止规则，不把未匹配query当真实背景。

**新增直接先例（GLDet）：** 用户提供的TGRS2025论文含实例分割，原文SSDD/HRSID mask AP71.3/58.6、整网500轮。已核全文/表格；完整作者代码尚未确认。空间扫描/邻域卷积/方向pool需要新空间载体，当前点头不能原样接入；N14无空间memory首试已按负结果停止，GLDet的完整空间结构迁移仍未执行，不直接比较不同划分AP。[原论文综述](research_round_20260925_global/GLDET_PRIMARY_PAPER_REVIEW_AND_SEGMENTATION_TRANSFER_20261014.md)、[代码现况和单项局部信息检验设计](research_round_20260925_global/GLDET_CODE_RECHECK_AND_MINIMAL_LOCAL_INFORMATION_TEST_20261015.md)。其局部信息检验已演变为N17：完整FIT邻域缓存和固定24轮训练已合格，DEV全量验证完成且主结果负收益停止；这是低成本信息检验，尚非完整GLDet复现或新创新。不能使用DEV捕获训练。

**局部特征与细化的直接先例：** 新增[五项定向核查](research_round_20260925_global/N20_DIRECT_REFINEMENT_PRECEDENT_FOCUSED_REVIEW_20261015_01.md)，覆盖RefineMask、Mask Transfiner、SegRefiner、SAMRefiner（ICLR 2025）与UnfoldCRF（2026预印本）。普通3×3真实邻域即使涨点，也先支持当前配方的信息效用；稀疏点交互、多阶段细化和通用条件修正均有先例，尚不能宣称结构创新或SOTA。

**当前已核验结果（N13）：** 三种子固定E48 LOG/GEO均值67.768660/67.579076 AP，配对差−0.189583点，三组均负，已停止该GEO联合配方扩展。全部E24曲线已核验，LOG/GEO均值67.992117/67.781883。真实诊断CPU失败9保留；image113四路探针实际0并ROOT核验，仅支持此例CPU/GPU特征渲染差异，同GPU F的CPU PR满足原容差，全374对尚未qualified。[局部探针结果](research_round_20260925_global/N13_IMAGE113_NUMERICAL_PROBE_RESULT_20261014.md)。无正式TEST、跨集或核心创新/TGRS结论。


新[CPU恢复入口](research_round_20260925_global/multimodal_data_pilot_20260928/native_geometry_paired_mask_readout_preparation_20261009/N13_completed_epoch_artifact_recovery_source_20261013_03/ABI.md)已实际通过独立端点检查；原GPU失败与缺失报告保留。

后续[冷启动训练源码](research_round_20260925_global/multimodal_data_pilot_20260928/native_geometry_paired_mask_readout_preparation_20261009/N13_qualified_endpoint_training_source_20261013_03/ABI.md)已静态审阅并保存，702/703首段训练及独立CPU核验已通过；当前入口只允许702/703首段。后续[连续48轮接续源码](research_round_20260925_global/multimodal_data_pilot_20260928/native_geometry_paired_mask_readout_preparation_20261009/N13_continuous48_adopted_endpoint_training_source_20261013_04/ABI.md)已通过静态审阅，三个种子第9–16轮、第17–24轮、第25–32轮及第33–40轮均已通过核验，三个固定E48均已通过独立CPU和ROOT核验，原固定DEV评价失败保留，修复版GPU/CPU及ROOT已通过，701 E08曲线及三条E48主端点均已合格；三个E08曲线已完成独立CPU和ROOT核验，全部三条E24已完成独立CPU/ROOT核验，真实固定PR诊断GPU原59604实际0，独立CPU原4404实际1，数值故障定位中。见[训练与接续状态](research_round_20260925_global/N13_SEED702_COLD_TRAINING_AND_CONTINUOUS48_SOURCE_STATUS_20261013.md)及[完整DEV评价接口](research_round_20260925_global/N13_FIXED_DEV_EVALUATION_INTERFACE_CONTRACT_20261013.md)。

新[原栅格固定PR适配器草案](research_round_20260925_global/code/N13_original_raster_fixed_PR_adapter_draft.py)实现已知F/L算术及独立候选分块；无GT、K或最终选集合输入。已有CPU合成验证；新[非零残差/非连续栅格逐点检查](research_round_20260925_global/code/N13_original_raster_fixed_PR_nonzero_strided_check.py)实际通过，真实权重映射、渲染/导出及固定配对的源码规则已完成[文字核对](research_round_20260925_global/N13_FIXED_PR_ACTUAL_SOURCE_INTERFACE_REVIEW_20261014.md)，真实输入/端点绑定、域定义、科学运行时及大栅格资源资格仍待核验。见[诊断方案和验证范围](research_round_20260925_global/N13_FIXED_PR_COVERAGE_REPAIRABILITY_DIAGNOSTIC_DRAFT_20261014.md)。 已补入条件IoU诊断进入完整AP检验的两个必要条件：自然全部query的GT无关策略，以及明确固定候选/exact-K或拒绝写入的行动范围；当前未执行新方法评价。新增[纯诊断统计](research_round_20260925_global/code/N13_fixed_PR_diagnostic_statistics_draft.py)及[独立手算检查](research_round_20260925_global/code/N13_fixed_PR_diagnostic_statistics_independent_hand_check.py)已通过host合成检查，仅SYNTHETIC_ONLY。 已核对外部作者width2函数的严格邻域计数规则及空/满/孤点静态含义；本地作者函数及N12 v02调用/计数规则已在任务间隙完成ROOT文字核对，真实诊断输入/域重建和运行资格仍待核验。

新[候选—原LOG重建—统计连接层](research_round_20260925_global/N13_FIXED_PR_PIPELINE_SYNTHETIC_INTEGRATION_CHECK_20261014.md)已保存；72项首次合成检查与独立106项halo/域外补查均实际退出0。新增域真实翻转、非空域外保持和完整C冒充原参考的统计前拒绝已验证，仅SYNTHETIC_ONLY，未取得真实SAR诊断资格。

新[显式区域装配接口](research_round_20260925_global/N13_FIXED_PR_NATIVE_DOMAIN_ASSEMBLY_SYNTHETIC_CHECK_20261014.md)连接native L、CPU作者回调与互斥旧域/halo；49项字面回调检查及13项拒绝实际通过，尚未验证真实作者数值、来源、CUDA或SAR域。

新[诊断接口补齐](research_round_20260925_global/N13_FIXED_PR_PRECOMPUTED_POPULATION_AND_DISTANCE_INTERFACE_CHECK_20261014.md)复用同一人口汇总，允许保留逐对JSON后释放栅格，并在原LOG重建后透传显式距离分层。59项/10拒绝和35项有限检查实际通过；不认证预计算数值、距离来源或真实SAR效果。

</details>

## 从这里开始

- [GLDet附件定向复核与loss边界](research_round_20260925_global/GLDET_ATTACHMENT_TARGETED_RECHECK_AND_LOSS_SCOPE_20261015_01.md)：确认实例分割直接先例；式(7)10/3/10属于标签分配cost，不能写成联合训练loss权重。
- [HRSID官方PNG编码文件获取试点](research_round_20260925_global/HRSID_OFFICIAL_PNG_ENCODED_IMAGE_BYTES_PILOT_20261015_01.md)：两文件真实字节/SHA/ZIP及PNG容器CRC/IHDR通过；未解码像素、未取得完整数据集或评价模型，原header解析失败保留。
- [HRSID官方PNG同包TRAIN/TEST实核](research_round_20260925_global/HRSID_PNG_OFFICIAL_ARCHIVE_DIRECTORY_AND_TEST_METADATA_20261015_01.md)：Root独立核TRAIN3642图/11047实例、TEST1962图/5922实例，文件名无交集且覆盖5604 PNG目录；数值ID跨split复用，不能裸ID合并。仅元数据，尚无影像/mask/AP资格。
- [N20渲染F9条件性信息分析](research_round_20260925_global/N20_RENDERED_F9_CONDITIONAL_INFORMATION_AND_CONTROL_SCOPE_20261015_01.md)：明确±1为最终栅格像素；仅在同一有效双线性单元等条件下有四基表示，尚无实际秩/覆盖率或效用结论。
- [HRSID作者固定版本标注实文件核验](research_round_20260925_global/HRSID_OFFICIAL_ANNOTATION_INTAKE_20261015_01.md)：取得3573图/9870实例的Git候选JSON并独立核对，215图无标注关联；该候选的配套TEST及母景独立未认证，与同包PNG的具体差异见上方实核，不能混配。

- [YOLO11n路口微调实测](docs/edge_traffic/xavier_nx_20261015/training_20261015/TRAINING_REPORT.md)：完成1022/128/128、30轮训练；四类mAP50 17.53%→48.12%，行人AP50仅16.33%；公开视频仍有小车误报公交/广告牌误报卡车，不具备五类高精度部署结论；NX与电动自行车未验证。

- [N20真实F邻域实验](refine-logs/N20_TRUE_FEATURE_NEIGHBOR_20261015_01/EXPERIMENT_PLAN.md)：固定24轮E8/E24实验方案；首种子（201）GPU清理失败后的保存产物恢复过程见[恢复核验](research_round_20260925_global/N20_SAVED_E8_E24_CPU_RECOVERY_QUALIFIED_STATUS_20261015.md)。各个种子的统一评价与重复实验进展见[执行状态](refine-logs/N20_TRUE_FEATURE_NEIGHBOR_20261015_01/EXPERIMENT_TRACKER.md)。

- [N20当前32份核验合同](research_round_20260925_global/code/N20_FIXEDDEV_FEATURE_ERROR_CPU_FRONT_UNBOUND_CONTRACT_20261015_02.md)与[全187统一评价合同](research_round_20260925_global/code/N20_FIXEDDEV_FULL187_COCO_CPU_FRONT_UNBOUND_CONTRACT_20261015_02.md)：规定分批核验、全量统一评价及独立复核流程；实际进展见[执行状态](refine-logs/N20_TRUE_FEATURE_NEIGHBOR_20261015_01/EXPERIMENT_TRACKER.md)。旧八份合同保留为历史。主端点与继续规则不变，合成检查不是实际AP或发表依据。

- [N20真实F邻域结构原型与训练设计](research_round_20260925_global/N20_TRUE_FEATURE_NEIGHBOR_SCOUT_DESIGN_AND_IMPLEMENTATION_STATUS_20261015.md)：18,851参数适配器的结构原型与训练设计，文档保留提出时状态。普通DW结构仅为信息基线，不列为新创新；统一评价与重复实验进展见[执行状态](refine-logs/N20_TRUE_FEATURE_NEIGHBOR_20261015_01/EXPERIMENT_TRACKER.md)。

- [N20原生网格取点先例与诊断候选](research_round_20260925_global/N20_NATIVE_GRID_SAMPLING_DIRECT_PRIOR_CANDIDATE_20261016.md)：补充LIIF、Deformable DETR、IFA与Implicit PointRend的直接先例，区分单次四角条件与两次插值、LN后的适用范围。仅为尚未预锁或执行的诊断候选，不改变当前实验。

- [路口五类检测数据集核查（2026-10-15）](docs/edge_traffic/xavier_nx_20261015/DATASET_SHORTLIST.md)：Intersection-Flow-5K、MIO-TCD、BDD100K与电动自行车专项候选；明确类别/视角/许可/获取状态及部分标注问题。

- [Xavier NX 路口预训练检测选型与公开实测（2026-10-15）](docs/edge_traffic/xavier_nx_20261015/REPORT.md)：三模型各完整719帧；权重来源、可复跑工具、误报失败例与NX待验收条件。尚未满足电动自行车独立识别/五类高准确率/NX30FPS。

- [N19固定方向代价探针](refine-logs/N19_FIXED_HALF_DIRECTIONAL_READOUT_20261015_01/EXPERIMENT_PLAN.md)：只用已有同头概率、固定1/2评分及普通/反向偏置对照，协议已事前锁定；body手算与outer接口检查通过，源码及一次独立定向审阅完成。原CPU实际0/426.812秒，三份完整AP及Root资格完成，主DIR67.726509，Δ−0.372876点，按预锁STOP；[完整结果](research_round_20260925_global/N19_FIXED_HALF_FULLDEV_QUALIFIED_RESULT_AND_STOP_20261015.md)。[查新](research_round_20260925_global/N19_IOU_DIRECTIONAL_ACTION_RISK_TARGETED_PRIOR_20261015.md)判为低新颖性/效用待检验，保持旧STOP，不称新理论或质量估计效应。

- [N18最小机制诊断计划](refine-logs/N18_FIXED_PR_ROUTING_CAPABILITY_20261015_01/EXPERIMENT_PLAN.md)：固定374配对复用已存动作，区分路由漏选与PR纠错能力；纯计数手算及单项接口检查已通过；新读取源码完成一次独立静态审查。真实CPU诊断实际退出0并通过Root整数/来源核对；结果见[N18诊断](research_round_20260925_global/N18_QUALIFIED_ROUTING_CAPABILITY_RESULT_AND_NEXT_HYPOTHESIS_20261015.md)。

- [N17局部粗mask、错误估计和选点直接先例](research_round_20260925_global/N17_LOCAL_NATIVE_L_ERROR_SCOUT_TARGETED_PRIOR_CHECK_20261015.md)：SEED/QuBER/MagNet/PointRend重叠与具体差异，现阶段只做信息探针，不确认创新。

- [N16预锁主结果解释器](research_round_20260925_global/code/N16_FIRST_SEED_RESULT_INTERPRETER_UNBOUND_DRAFT_20261014_01.md)：仅E24主端点触发复验/停止；6项合成检查通过，真实AP解释待完整独立资格。

- [N16四CPU证明后的最终完整核验合同](research_round_20260925_global/code/N16_FIXED_E24_ALL100_SCOUT_CPU_NUMERIC_SHARDS_QUALIFIER_CONTRACT_20261014_01.md)：代码/最小合成/静态审阅已准备，尚未实际执行，不替代完整资格。

- [N15下一单一公平选择器草案](research_round_20260925_global/N15_SINGLE_FAIR_UTILITY_SELECTOR_MINIMAL_UNBOUND_DESIGN_20261014.md)：相同容量/输入的修补效用与错误概率，两者共有真实目标/负query及成本规则；ROOT指出均匀all100采样的稀释风险，尚未正式登记或训练。

- [N15完整机会诊断合格结果与下一步](research_round_20260925_global/N15_FIXED_E24_ACTION_OPPORTUNITY_QUALIFIED_RESULT_AND_NEXT_STEP_20261014.md)：实际固定头存在扩域修补机会；GT oracle、学习能力和部署AP分开，3个提前门通过。

- [N15行动效用的五项直接先例查新](research_round_20260925_global/N15_ACTION_UTILITY_TARGETED_PRIMARY_PRIOR_CHECK_20261014.md)：PointRend、MagNet、Mask Transfiner、PromptMoE与SAM-Veteran；已知效用监督与新的工程假设分开，当前N15机会诊断已合格，不能等同AP或选择器效用。
- [N15固定E24行动机会预登记](refine-logs/N15_FIXED_ACTION_OPPORTUNITY_20261014_01/EXPERIMENT_PLAN.md)：同C动作、同K、原强预测核对、实际GPU输入CPU回放及预锁GO门。

- [真实诊断CPU数值失败与单图定位](research_round_20260925_global/N13_REAL_FIXED_PR_CPU_COMPATIBILITY_FAILURE_AND_SINGLE_IMAGE_PROBE_20261014.md)：第9实际失败保留，不放宽原容差，探针区分特征重渲染与算术，原GPU33452单次执行中。

- [全部三种子E24曲线与真实固定PR诊断启动](research_round_20260925_global/N13_FIXED24_THREE_SEED_CURVE_RESULT_AND_REAL_DIAGNOSTIC_START_20261014.md)：全部固定曲线核验完成，当前701 LOG E48的374＋6真实诊断GPU原59604实际0，独立CPU原4404实际1，数值故障定位中，尚无核验后新效用。

- [合格曲线与三种子主结果](refine-logs/N13_SAMEK_PR_20261013_01/findings.md)：三条E48已核验并执行当前联合配方停止决定；三个E08仅作固定学习曲线，全部三条E24已完成独立CPU/ROOT核验，真实固定PR诊断GPU原59604实际0，独立CPU原4404实际1，数值故障定位中。

- [固定DEV执行错误与独立修复版](research_round_20260925_global/N13_FIXED_DEV_SERIALIZATION_REPAIR_STATUS_20261013.md)：原实际失败及未完成产物保留；修复版已通过GPU、独立CPU和ROOT核验，完整三种子主结果及当前曲线见最新状态。

- [固定48轮训练资格与稳定性结果](research_round_20260925_global/N13_FIXED48_TRAINING_QUALIFIED_RESULT_20261013.md)：六头各79,728更新，全部端点核验通过；六头43–48轮平均可编辑BCE较37–42轮均下降超过1%，尚不支持充分收敛；训练资格与实际AP评价分别核验。
- [HRSID官方协议与发布范围核查](research_round_20260925_global/HRSID_OFFICIAL_PROTOCOL_AND_RELEASE_SCOPE_CHECK_20261013.md)：原文支持65/35裁片协议；补核固定作者提交与未回应许可issue，场景独立与数据再发布许可仍未确认；未打开数据包。
- [Poly-BRBLE 原论文全文核查](research_round_20260925_global/N13_POLY_BRBLE_FULLTEXT_CHECK_20261013.md)：直接先例已包含边界buffer、不确定选点与点精修；明确当前sameK差异、联合处理限制和后续强对照。
- [SAR提示式基础模型全文核查](research_round_20260925_global/SAR_PROMPTED_FOUNDATION_MODELS_FULLTEXT_REVIEW_20261013.md)：核对监督范围、指标与可复现性，并明确它与当前残差点头实验的可比边界。
- [学习选择器的监督目标与完整模型边界](research_round_20260925_global/N13_LEARNED_SELECTOR_TARGETS_AND_FINE_FEATURE_PRIOR_CHECK_20261014.md)：Mask Transfiner的GT缩放损失目标与固定作者重实现、RefineMask的可训练P2/SFM及Simple版本；明确后续强对照与创新措辞。
- [固定K二值IoU oracle与概率代理边界](research_round_20260925_global/N13_FIXED_K_BINARY_IOU_ORACLE_AND_PROBABILISTIC_PROXY_LIMIT_20261014.md)：更正真实标签与预测概率的适用范围，给出证明、人工反例及固定权重强对照；不是新SAR实验。
- [固定预算行动选择的两项直接先例查漏](research_round_20260925_global/N13_FIXED_BUDGET_ACTION_UTILITY_DIRECT_PRIOR_GAPS_20261014.md)：MagNet候选置信排序、PromptMoE执行质量监督，以及待登记的错误预测/实际翻转、MagNet式排序和固定权重强对照；当前实验不改。
- [预计算人口与距离分层接口补齐](research_round_20260925_global/N13_FIXED_PR_PRECOMPUTED_POPULATION_AND_DISTANCE_INTERFACE_CHECK_20261014.md)：保留原数组路径、避免重复逐对统计；公开有限复现，科学资格仍未认证。
- [固定PR显式区域装配及有限检查](research_round_20260925_global/N13_FIXED_PR_NATIVE_DOMAIN_ASSEMBLY_SYNTHETIC_CHECK_20261014.md)：固定CPU FP32作者回调接口和标签1域，不实现或替换作者width2；只有合成装配证据。
- [固定PR连接层及新增域/域外独立合成补查](research_round_20260925_global/N13_FIXED_PR_PIPELINE_SYNTHETIC_INTEGRATION_CHECK_20261014.md)：必需原预测参考、只写旧域的完整重建和统计前拒绝；公开便携源码及四路径复现命令，尚无真实SAR效用。
- [固定PR候选的覆盖、纠正能力与选择余量诊断草案](research_round_20260925_global/N13_FIXED_PR_COVERAGE_REPAIRABILITY_DIAGNOSTIC_DRAFT_20261014.md)：四项嵌套oracle、原输出重建、独立候选及exact-K强制伤害门；纯布尔代码人工穷举已通过，真实SAR适配/执行尚未登记。
- [固定48轮三种子完整主结果与下一诊断](research_round_20260925_global/N13_FIXED48_THREE_SEED_PRIMARY_RESULT_AND_NEXT_DIAGNOSTIC_20261014.md)：配对均值为负，按原规则停止当前联合配方；保留所有指标、成本、失败与适用范围，下一行动oracle只是草案。
- [实测区域成本的归因边界](research_round_20260925_global/N13_MEASURED_REGION_COST_ATTRIBUTION_PLAN_20261014.md)：聚合计时含两臂选择、排序校验和摘要；列明遗漏的传输/导出成本及主评价后最小补测，未认定子阶段瓶颈或取得加速。
- [共享投影与成本公平性审阅](research_round_20260925_global/N13_SHARED_PROJECTION_COST_FAIRNESS_REVIEW_20261013.md)：稠密强对照应同样共享特征投影；列明计算、内存和数值等价的边界，实际加速与质量收益仍待检验。
- [当前研究目标与执行要求](RESEARCH_GOAL.md)：问题、公平比较、重复实验、机制验证和停止规则。
- [修补行动收益与计算分配的直接先例](research_round_20260925_global/N13_ACTION_UTILITY_AND_ALLOCATION_DIRECT_PRIOR_ADDENDUM_20261013.md)：六项原始工作、近期SAR有限补核、像素收益与IoU的区别，以及待检验的质量与成本方案；出版核验限制明确保留，当前训练不改。
- [论文版本索引](docs/PAPER_INDEX.md)：各代正文、补充材料与历史稿的关系。
- 可阅读历史修订稿：[正文 PDF](paper_tgrs/revision_tgrs_20260905/main.pdf)、[补充 PDF](paper_tgrs/revision_tgrs_20260905/supplement.pdf)、[说明](paper_tgrs/revision_tgrs_20260905/README_CN.md)。它们属于旧融合主线，未重写成 10 月当前研究；PDF 与当前源码的对应关系未重新编译核验。
- [数据获取与版本说明](docs/DATA_ACCESS.md)、[上游数据索引](docs/upstream_data_index.json)：PSeg-SSDD 与 HRSID 的官方入口。
- [掩膜残差读出与校准先例核查](research_round_20260925_global/CALIBRATION_PRECEDENT_REVIEW_20261012.md)：PointRend、Implicit PointRend、质量评分及 SAR 直接先例；机制假设仍待检验。
- [完整实验计划](refine-logs/EXPERIMENT_PLAN.md)、[实验跟踪](refine-logs/EXPERIMENT_TRACKER.md)、[下一研究批次](refine-logs/NEXT_RESEARCH_BATCH.md)。这些记录保留历史合同；有日期的最新状态优先。

## 当前证据和决定

所有下表结果均来自历史内部 **DEV187 / 380 GT**，不是官方 TEST。AP 使用百分制；“+1 AP 点”表示 67 到 68。没有把本项目的内部指标与外部论文数值直接排名。

| 固定比较 | mask AP | 能支持的结论 |
|---|---:|---|
| 预选强 E24 原生参考 | 67.071057 | 同协议固定参考 |
| A：原生模型 + 同样点监督，24 轮再追加 48 轮 | 64.686168 | 当前续训配方的端点 |
| B：联合残差读出，同样预算 | 66.797068 | B 高于 A，但低于强 E24；结束该配方扩展 |
| C：冻结 E24，粗 F/e/L 点头，两个下游种子均值 | 68.166136 | 相对 E24 +1.095079 AP 点的开发集效用 |
| R：相同点头加真实 JPEG 7×7 patch，两个下游种子均值 | 68.153211 | 与 C 几乎持平；未支持新增 patch 的作用 |

详见 [固定长预算 A/B 结果](docs/fixed_endpoint_joint_native_E24warm48_result_20261012.md)与 [N07 全部种子及固定端点评价](docs/N07_fixed8_fullDEV_point_refinement_result_20261012.md)。N07 每头固定 8 轮、13,288 更新；两个下游种子共享同一个教师初始化，不能替代多教师重复或充分收敛证据。R−C 配对均值为 −0.012924 AP 点，未满足结果前锁定的跟进规则；保留效用，停止当前 patch 配方扩展。

N09已完成简单全局logit偏移控制：仅在FIT缓存点拟合、只在原生 |L|≤2 内应用、保持query和原生分数。首个DEV评价在原生精确一致性检查处退出1；已保留[历史失败状态](docs/N09_global_offset_runtime_state_20261012.md)，完整修复与合格结果如下。

2026-10-13 已修复原始 all100 评分路径；完整 DEV 的 GPU 评价和独立 CPU 核验均真实退出 0，原生 18,700 候选输出精确复现。全局偏移 AP 为 **67.1466401**，比原生仅高 **0.0755834** 个百分点、比 C 均值低 **1.0194955**；未达到预登记近似替代规则。这只限定本次全局偏移，不证明新颖性或 F/e 因果作用。[完整核验结果](research_round_20260925_global/N09_FIXED_GLOBAL_OFFSET_RESULT_20261013.md)。已进入 [PointRend 式公平结构对照](research_round_20260925_global/FAIR_POINT_HEAD_CONTROL_DESIGN_20261013.md)，实际执行状态见下文 N10 更新。

当前核心创新、正式 TEST、跨集收益和 TGRS 投稿依据仍未成立。失败结果和停止原因保留，不将局部提升包装为最终成功。

## 目录导航

| 目录 | 内容 |
|---|---|
| `paper/`、`paper_tgrs/` | 多代论文 LaTeX / BibTeX、作者说明、历史正文和补充 PDF |
| `carrier/`、`mne/`、`seeding/`、`losses/` | 历史模型扩展、配置与候选组件 |
| `protocol/`、`benchmark/`、`diagnostics/` | 数据转换、评价和误差诊断源码 |
| `research_round_20260925_global/` | 当前及历史训练、固定评价和独立核验 SOURCE |
| `docs/`、`refine-logs/` | 审阅、先例分析、实验计划和研究范围 |
| `autodl/`、`tools/` | 历史环境准备与辅助工具 |

最新 SOURCE 位于 `research_round_20260925_global/multimodal_data_pilot_20260928/native_geometry_paired_mask_readout_preparation_20261009/` 下：

- `N07_frozen_E24_original_detail_point_refinement_source_20261012_01/`：FIT 缓存、点头训练、GPU 评价、独立 CPU 核验。
- `N09_FIT_only_global_logit_offset_control_source_20261012_01/`：全局偏移数学、FIT 求解及独立 CPU 核验；历史首个评价失败保留。
- `N09_FIT_only_global_logit_offset_control_source_20261013_02/`：恢复原始 all100 评分路径的完整 DEV 评价器。
- `N06_fixed_E24_one_to_one_error_audit_source_20261012_01/`、`N01_uniform_point_supervision_gradient_probe_source_20261012_01/`：固定误差和监督诊断。

## 数据与复现范围

本仓库包含数据获取说明和转换/审计源码，**不包含原始数据、全量标注、权重及运行缓存**，不能称为完整数据镜像。官方 SSDD 的 PSeg 多边形与 BBox/RBox 检测框版本不同；内部 FIT741/DEV187 不等于官方 TEST232。HRSID 的 JPG/PNG、场景协议和纯背景补充版本也必须分别记录。数据及其许可入口见 [DATA_ACCESS.md](docs/DATA_ACCESS.md)。

历史环境记录为 PyTorch 2.0.1+cu118、torchvision 0.15.2、MMEngine 0.10.4、MMCV 2.1.0、MMDetection 3.3.0，并依赖 NumPy、SciPy、OpenCV、Pillow、pycocotools。最新本机资格环境与历史环境不可直接混用；没有声称全仓库在新机器完成重跑。

历史入口示例：`python protocol/pseg_to_coco.py`（需配置数据路径）及 `python protocol/seal_eval.py --preds ... --gt ...`。当前科学入口采用 `--spec ABS --owner-execute`，依赖本地冻结数据、初始化和来源资格链。本次保存源码，不包含全部绑定输入，**尚不能下载后一键复现最新实验**；不得删除检查以冒充已验证的可移植版本。



**2026-10-13 / N10 已闭合：** 四头各固定8轮/13,288次更新，完整DEV GPU评价及独立CPU核验均正常结束并完成ROOT核验。C均值AP **68.166136**，PointRend结构适配PR均值 **67.927378**；C−PR两组均正，平均 **+0.238757 AP点**，未达到预锁+0.30开发优先线，不自动延长此结构。两臂均高于原生67.071057，但不能据此确认创新或充分收敛。原生IoU≥.90固定配对上四臂平均IoU均小幅下降，下一步检验可编辑区域的误差覆盖与修复上限。见 [完整合格结果](docs/N10_fixed8_C_PR_fullDEV_qualified_result_20261013.md)、[全DEV状态](research_round_20260925_global/N10_FULLDEV_STATUS_20261013.md)、[固定协议](refine-logs/N10_PR_FL91_20261013_01/EXPERIMENT_PLAN.md)、[实际状态](refine-logs/N10_PR_FL91_20261013_01/EXPERIMENT_TRACKER.md)及 [研发代码](research_round_20260925_global/multimodal_data_pilot_20260928/native_geometry_paired_mask_readout_preparation_20261009/N10_PR_FL91_direct_precedent_control_source_20261013_01/)。

当前C实例条件偏置属于已知原语，不能单独宣称创新；定向先例与最小反证方案见 [查新备忘录](research_round_20260925_global/C_CONDITIONED_POINT_HEAD_NOVELTY_MEMO_20261013.md)。

下一方向依据 [既有误差证据](docs/N01_N06_N07_N09_ERROR_EVIDENCE_FOR_NEXT_DECISION_20261013.md)与 [固定区域/采样风险定向查新](research_round_20260925_global/FIXED_BAND_SAMPLING_RISK_PRECEDENT_MEMO_20261013.md)。区域监督、MIS和控制变量均有先例；候选全图几何目标尚未执行或确认，不能用新名称替代实证。

**2026-10-13 / N11：** GPU及独立CPU均已实际退出0并完成ROOT资格。原生.50固定374对中，当前区域只覆盖 **22.53%错分像素**；仅改带内损失/容量无法覆盖多数排除错误。见 [限定范围的完整结果](docs/N11_original_editable_band_qualified_result_20261013.md)，下一方向优先强几何区域对照与可否证的新选择规则；尚未建立新方法效用或创新。见 [当前状态与范围](research_round_20260925_global/N11_EDITABLE_REGION_COVERAGE_STATUS_20261013.md)及 [诊断SOURCE](research_round_20260925_global/multimodal_data_pilot_20260928/native_geometry_paired_mask_readout_preparation_20261009/N11_native_editable_region_coverage_source_20261013_01/SOURCE_ABI.md)。

## 持续同步

本仓库是后续研究的持续记录入口。总结、综述、方法设计、研发源码与配置、论文更新及实质对话研究总结形成后，在当前工作任务内及时提交；实验成功、失败和未完成的验证均按实际状态保留。只有核验远端提交与目标文件后才记为已同步，失败则标为待同步；应用关闭或离线时不承诺自动上传。具体范围与流程见 [持续同步规则](docs/CONTINUOUS_SYNC.md)。

## 发布范围与许可

首次发布快照及完整性验证见 [publication_manifest_20261012.json](docs/publication_manifest_20261012.json)（限定于其中记录的提交；后续版本以 Git 提交历史为准），历史工作区盘点见 `docs/workspace_inventory_20261011.csv.gz`（元数据，非原始数据）。[发布范围](PUBLICATION_SCOPE.md)明确哪些内容未随仓库提供。

作者、单位、日期和其他提交元数据保留占位符。自有项目尚未指定统一开源许可；第三方实现使用原始来源链接与各自许可证，不把第三方代码改授为本项目许可。外部下载的文献全文、第三方参考仓库、凭据、运维配置、无关工程以及整个工作区 Git 历史不随本快照发布。

**以下为历史形成记录；当前训练进度以顶部和最新状态段为准。**

当时下一项仅作为待检验假设：同一冻结模型、点头、编辑与训练预算下，先比较几何边界选择与不确定性选择，完整训练采用至少三个配对种子。强先例、反证条件及执行边界见 [区域选择设计备忘录](research_round_20260925_global/N11_REGION_CHOICE_NEXT_HYPOTHESIS_20261013.md)；尚未新增训练或确认创新。

**N12 实验前登记：** [同预算区域覆盖诊断协议](refine-logs/N12_REGION_COVERAGE_20261013_01/EXPERIMENT_PLAN.md)已锁定主比较与反证规则；[执行跟踪](refine-logs/N12_REGION_COVERAGE_20261013_01/EXPERIMENT_TRACKER.md)记录实际状态：首个GPU运行在依赖文件来源核验处退出1，尚未开始区域计算；修正实际包树绑定后，GPU及独立CPU均真实退出0并完成ROOT核验。失败源码与修正版均保存，见[运行状态](docs/N12_region_diagnostic_runtime_status_20261013.md)。只有真实GPU及独立CPU核验完成后才报告新覆盖/距离结果；并非新方法效用。


**N12 已核验诊断：** 相同K下，主固定374配对的错误覆盖22.53%→24.67%，条件oracle IoU仅增加0.000728650；.75固定集合覆盖略降，两臂oracle≥.90/.95主槽位计数未增加。通过预登记的有限跟进条件，将登记匹配FIT缓存与至少三个配对种子的公平训练，尚无新训练提升或创新。见[完整限定结果](docs/N12_sameK_region_qualified_result_20261013.md)。


**历史快照（首轮缓存核验失败时，后续进展见顶部当前状态）：** 下一项[N13训练准备](research_round_20260925_global/N13_REGION_TRAINING_PREPARATION_20261013.md)已完成缓存/门控静态盘点：旧FIT采样缓存不足以构造新区域，旧头内部门控必须统一外移。采用相同PR骨架和三个冷初始化配对种子，已在新缓存计算前登记48轮固定预算、评价和停止规则；新缓存和训练源码均通过一次独立静态审阅；GPU缓存已正常退出并完成生成阶段ROOT核验；首个独立CPU核验在第4图原生L重算检查处退出1，正在调查；缓存尚未资格，未开始训练。失败原件保留，不改判据追求通过。缓存源码已经保存，见[执行状态与代码入口](docs/N13_sameK_region_execution_status_20261013.md)。见[固定协议](refine-logs/N13_SAMEK_PR_20261013_01/EXPERIMENT_PLAN.md)和[实际跟踪](refine-logs/N13_SAMEK_PR_20261013_01/EXPERIMENT_TRACKER.md)。

N13核验失败后，已在新的数值执行前保存[固定单图诊断登记](research_round_20260925_global/N13_NATIVE_L_REPLAY_DIAGNOSTIC_PLAN_20261013.md)及[原生L数值验证审阅](research_round_20260925_global/N13_NATIVE_L_NUMERIC_VALIDATION_REVIEW_20261013.md)；新[诊断源码](research_round_20260925_global/multimodal_data_pilot_20260928/native_geometry_paired_mask_readout_preparation_20261009/N13_cache_L_replay_diagnostic_source_20261013_01/ABI.md)仅用于定位重算差异，不改变容差或放行未资格缓存。

**N13有限诊断已结束：** 第4图原FP32重算有1/536571个值超过原容差，FP64有3个；该图P与原边界区域未变。原判据仍失败，未产生缓存Q或开始训练。见[限定调试观察](research_round_20260925_global/N13_NATIVE_L_REPLAY_DIAGNOSTIC_OBSERVATION_20261013.md)；不以单图数值差或源码审阅宣称方法效用、创新或全缓存资格。

- [N13 全样本 GPU 关系与 CPU v2 资格修订](research_round_20260925_global/N13_CANONICAL_GPU_RELATION_AND_CPU_QUALIFICATION_AMENDMENT_20261013.md)：GPU关系与CPU v2资格实际通过，原失败及固定训练配方保留。

- [N13 CPU v2 与固定48轮训练源码接口](research_round_20260925_global/multimodal_data_pilot_20260928/native_geometry_paired_mask_readout_preparation_20261009/N13_sameK_region_qualified_relation_training_source_20261013_02/SOURCE_ABI.md)：CPU v2实际资格通过，种子701配对训练1–8轮已启动。

- [N13 全体冻结 M→L GPU 关系实际通过](research_round_20260925_global/N13_CANONICAL_GPU_RELATION_OBSERVATION_20261013.md)：741图/1661query逐字节一致，原执行157.109秒；已获CPU v2资格并启动首段配对训练。

- [N13 直接先例定点更新（六项）](research_round_20260925_global/N13_POINT_REGION_DIRECT_PRECEDENT_UPDATE_20261013.md)：含2026年SeSAM近邻；同K几何区域是待检验差异，已有采样/点头/带外继承不算独立创新。当前训练配方保持不变。

- [N13 CPU v2实际合格结果与训练准入范围](research_round_20260925_global/N13_CACHE_CPU_V2_ACTUAL_QUALIFIED_RESULT_20261013.md)：全部区域/监督/采样强门通过；原full-L CPU容差在6图失败保留为诊断。种子701首段配对训练运行中。


**本轮研发更新：** [完整DEV评价源码](research_round_20260925_global/multimodal_data_pilot_20260928/native_geometry_paired_mask_readout_preparation_20261009/N13_sameK_fixed_DEV_evaluation_source_20261013_01/EVALUATION_SOURCE_ABI.md)已静态审阅，尚未实际评价；[几何与点精修定向查新](research_round_20260925_global/N13_GEOMETRIC_POINT_REFINEMENT_TARGETED_PRIOR_UPDATE_20261013.md)补充五项论文及更接近的Poly-BRBLE技术披露，创新主张需以具体支持分配差异和强对照效用为依据。

- [后续候选：预测实际修补的正负收益](research_round_20260925_global/N13_LEARNED_EDIT_UTILITY_NEXT_HYPOTHESIS_20261013.md)：与同容量错误概率选择器比较；待登记、未执行，不改变当前三种子48轮实验。像素收益不等于IoU/AP提升。

- [N13三个首段端点完整核验及连续48轮进度](research_round_20260925_global/N13_ALL_THREE_E08_ENDPOINTS_AND_CONTINUOUS48_STATUS_20261013.md)：701原报告失败保留；三个固定E48均已核验（各79,728更新/臂），首个固定DEV评价执行失败待修复。完整DEV实际载荷约10.95GiB，尚无新AP。

单图探针元数据原44062实际0（9,975来源、94成功、原8失败及新增1失败、总103身份）；ROOT准入后，原GPU33452已单次启动。比较原容差不变，不生成Q／AP，数值原因尚待实际结束和ROOT读回。

最新执行状态：[N16全FIT缓存已实际完成、待独立CPU核验](research_round_20260925_global/N16_FULLFIT_CACHE_EXECUTION_STATUS_20261014.md)。正常退出0，741图/3035136点/611.937秒；完整独立CPU核验及ROOT最终准入均已实际通过；固定24轮配对scout训练已原实际退出0，独立端点核验与ROOT最终资格已通过，尚无新增AP。

[独立机制与结果解释矩阵](research_round_20260925_global/N16_FIXED_PROTOCOL_MECHANISM_RESULT_INTERPRETATION_20261014.md)：净纠错不等于AP；背景代理监督、dense-flip强对照及完整部署成本的主张边界。训练端点与部署选点的独立CPU核验源码已保存为未绑定草稿，合成通过不等于实际资格。

**N16正式训练准入：** GPU和独立CPU全部741图/3035136点实际通过，combined最大绝对误差1.52587890625e-5、实际修正符号精确；ROOT闭合通过后已启动固定24轮Utility/Error配对训练任务。E24主评价尚未执行。

**固定24轮已完成：** 两臂各17784更新、72843264点暴露，原GPU任务正常退出0；端点独立CPU核验已启动，完整DEV AP尚未执行。训练完成不等于方法涨点或投稿资格。

**固定端点完整资格通过：** E8/E24模型、实际AdamW状态与计数、私有图序、完整训练记录和独立CPU前向/固定损失均通过，原GPU/CPU/ROOT均实际0。当前准备完整DEV187/all100的固定E24主评价；尚无新AP、部署选点或SOTA/TGRS结论。

[固定24轮完整DEV评价协议与状态](research_round_20260925_global/N16_FIXED_E24_FULLDEV_EVALUATION_PROTOCOL_AND_STATUS_20261014.md)

[完整DEV固定分段执行补充](research_round_20260925_global/N16_FIXED_FULLDEV_EXECUTION_PARTITION_SUPPLEMENT_20261014.md)


[正式TEST与跨集证据准备度](research_round_20260925_global/N16_FORMAL_TEST_AND_CROSS_DATASET_EVIDENCE_READINESS_20261014.md)：定位既有测试暴露与HRSID旧TEST选型历史，区分固定配方外部评价与独立确认性证据；实际包、真mask、ID/场景与许可覆盖仍须执行前闭合。本次仅普通历史/元数据审阅，未新增科学验证。

[补充跨集候选：Instance-RSDD（尚未入场）](research_round_20260925_global/INSTANCE_RSDD_PUBLIC_DATA_CANDIDATE_AND_MASK_SCOPE_20261016.md)：已核对论文的人工逐实例像素标注说明与作者入口，并实际看到 `annotation_coco` 目录及标称1KB的 `readme.txt`；目录名不认证COCO格式，包内编码、实际划分、历史使用、独立性与许可仍待核验。未新增跨集AP，不改变N20。

[历史TEST用途台账补充（2026-10-16）](research_round_20260925_global/HISTORICAL_TEST_USE_LEDGER_TEXT_EVIDENCE_20261016.md)：从已发布实验稿定位HRSID融合设置/BPR种子与PSeg-SSDD对比checkpoint的测试选型披露，补齐具体已知用途；原始运行、权重与时间链未重新核验，不推及当前N20，不恢复盲测身份。


[scout条件重复的初始化编号登记](research_round_20260925_global/N16_SCOUT_INITIALIZATION_REPEAT_PROSPECTIVE_LOCK_20261014.md)：在ROOT查看本轮完整AP前，固定后续202610140162/163；只改变配对scout初始化，原停止门不变，没有开始新增训练，结论限固定载体下初始化稳健性。

[后续最少必要实验与验证边界](research_round_20260925_global/N17_MINIMAL_SCIENTIFIC_EVIDENCE_NEXT_ACTIONS_20261015.md)：先完成主AP，有收益按锁复验，无收益先分析已有全量错误；只在新改动、失败或具体疑点出现时重复工程检查。

[统一COCO后的独立后置ROOT入口](research_round_20260925_global/code/N17_FIXEDDEV_FULL187_POST_ACTUAL_ROOT_CPU_UNBOUND_DRAFT_20261015_01.py)：已冻结/接口fixture通过，独立静态审阅READY、真实执行未开始；不重复计算COCO。


[保留第21次失败的统一COCO03](research_round_20260925_global/code/N17_FIXEDDEV_FULL187_CPU_COCO_FINAL_ROOT_UNBOUND_DRAFT_20261015_03.py) · [独立后置ROOT02](research_round_20260925_global/code/N17_FIXEDDEV_FULL187_POST_ACTUAL_ROOT_CPU_UNBOUND_DRAFT_20261015_02.py)：均已冻结/接口检查通过，待五段独立CPU实际证据和原终态后执行，不是新AP。

[独立CPU重复计算的静态观察](research_round_20260925_global/N17_CPU_REDUNDANT_COMPUTATION_STATIC_OBSERVATION_20261015.md)：两个只读缓存候选，未测耗时、未实现，不改变当前核验。


N20后续未执行源码：[独立CPU端点核验入口](research_round_20260925_global/code/N20_PAIRED_TRAIN_CPU_ENTRY_UNBOUND_DRAFT_20261015_01.py)、[真实邻域全候选评分](research_round_20260925_global/code/N20_ALL100_LOCAL_CENTER_FEATURE_ERROR_SELECTOR_EVAL_BODY_UNBOUND_DRAFT_20261015_01.py)、[分段DEV入口前门](research_round_20260925_global/code/N20_FIXEDDEV_FEATURE_ERROR_GPU_FRONT_UNBOUND_DRAFT_20261015_01.py)。CPU入口只通过语法检查；评分body通过语法和一次独立静态核对，尚未数值执行；DEV前门尚需接真实legacy/prior carrier hook，不可直接启动。已有common-C精修仅作固定参考，新选中点仍fresh计算并保留原数值/符号门，不把缓存读取当部署计时。

原正常GPU0路径下的CPU141/23和DEV142前门保留为未执行历史草稿；收尾失败24后当前改用CPU恢复140/24/164，原GPUactual0保持False。后续DEV需消费新恢复资格，不能直接套用旧前门或补写缺失GPU报告。

已另存[消费恢复资格的DEV前门v02](research_round_20260925_global/code/N20_FIXEDDEV_FEATURE_ERROR_GPU_FRONT_UNBOUND_DRAFT_20261015_02.py)：GPU失败只保留为失败记录，四个模型从原快照取，后续首DEV141/24；仍需真实legacy/priorcarrier hooks和CPU恢复实际资格，未绑定、未执行。

**完整主线接线更新：** 已选plain64原结构作更贴近的普通对照；新几何/plain包装已实现仅AST，FIT全部查询M100真实补采源已完成但尚未执行。历史clean32/GN、plain64与full64分别注明，不把它们混成同一基线。[当前研发与输入缺口](research_round_20260925_global/U31_MAIN_CONTROL_AND_NATIVE100_INPUT_PROGRESS_20261017.md)。

- 2026-10-25：新增[U57首次端点AP前效用与复验规则](research_round_20260925_global/U57_PROSPECTIVE_UTILITY_AND_REPEAT_POLICY_20261025.md)；首次训练途中明确，不追溯预锁。对比表补入固定R 68.099 AP和N26质量排序三种子均值69.537 AP，历史参考预算与本轮分列。

- 2026-10-25：[SegS-YOLO官方原文增量核查](research_round_20260925_global/U57_DIRECT_PRIOR_INCREMENT_20261025.md)：确为SAR实例分割与上下文/多尺度/动态蛇形结构先例；LS-SSDD仅定性，不改变当前训练或已有晋级规则。

- 2026-10-25：[最近直接先例差异表已补核SegS-YOLO](research_round_20260925_global/U57_NEAREST_PRIOR_NOVELTY_CHECK_20261018.md)，保留原访问日期并另标本轮补核日期；当前未形成外部复现或新AP结论。

- 2026-10-25：[教师身份与数据接触核查](research_round_20260925_global/U57_TEACHER_IDENTITY_AND_DATA_EXPOSURE_SCOPE_20261025.md)：当前原生教师为项目Mask2Former/R50 E24，纠正与DiffSAR外部先例混称；[官方COCO初始化完整摘要关联](research_round_20260925_global/results/U57_OFFICIAL_COCO_INITIALIZER_FULL_STREAM_AUDIT_20261025.json)已补齐，DEV排除仍需实际历史清单核查。历史D目录FPN归属记录不等于当前载体身份。

2026-10-25接续更新：原18688→18944的TCP连接失败保持失败、0更新；新的单次30秒连接接续已通过原工具/SSH/worker0、771项字节和严格进程核验，正式前缀18944。下一18944→19200实际运行中。不同预算/预训练与数据划分的外部数字不直接排名，SAM完整模型候选仍未执行。
