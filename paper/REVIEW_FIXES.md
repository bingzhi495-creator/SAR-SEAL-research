# 论文评审修正清单（工作流 w5g4y381e 对抗评审，2026-06-18）

56 findings (7 blocker, 23 major)。下列 blocker 已**对原始产物核验属实**——是真问题，必须诚实修正。完整评审见 paper/review_summary.md。

## 已核验属实的 blocker

### B2（致命，核验属实）：60/40 oracle 分解过度主张
- 草稿/我声称"边界 oracle 替换 GT 像素测恢复 AP75、小目标更高"——**此实验从未做过**。
- 实际只有 `runs/a0_pseg_seed0/mask_quality_oracle.json`：近岸 det_rate 0.657、recoverable_ap75_frac 0.233。60/40 = 0.343/0.576 未检测 vs 0.233/0.576 已检测但 IoU∈[0.5,0.75)。这是**计数占比，非 AP 可加分解，非标签替换 oracle**。
- "小目标更高"被数据**反驳**：近岸 S frac=0.192 < M=0.333（反而更低）。
- **✅已修正（做了②真 oracle）**：diagnostics/oracle_decomp.py 真 AP-可加 oracle（裁决 runs/refine_gate/B2_真oracle分解裁决.md）：近岸 **51% 掩码分辨率 / 49% 检测覆盖**（base 0.341→mask_oracle 0.637→full 0.921），远海 87% 掩码。**掩码反而是更大的一半**，强化论文主张。论文改"~60/40"为"真 oracle 约各半(~51%掩码)"，删"小目标更高"(未验证)。

### B3（致命，核验属实）：G1 负结果选择性诚实
真实 G1 PSeg 近岸 AUROC：
| 方法 | 近岸 | all |
|---|---|---|
| **mne_full**(融合,未上线) | **0.9597 最高** | 0.9882 |
| ca_cfar / ts_cfar | 0.9515 / 0.9496 | 0.979/0.986 |
| **mne_E**(上线) | 0.9277 | 0.9734 |
| box_z(最弱) | 0.8833 | 0.9756 |
- 我的"±0.002"是 all-split mne_E vs 最弱 box_z；但**近岸** mne_E 输 ts_cfar −2.2pt，而 mne_full **反超所有**(+1.0 vs ts_cfar, +7.6 vs box_z)。HRSID all: box_raw 0.9915 最高、mne_E 0.986 中游。
- **修正**：报对最强 ts_cfar 的比较；披露 mne_full 近岸 +7.6pt；解释为何仍上线单通道（零参/跨任务共享/margin 低于预注册 +0.03 门/价值在召回非分类）。这才是真诚实，非对稻草人诚实。呼应用户"不过度注重分类"。

### B5（核验属实）：HRSID"掩码清洁度泛化/无害"自相矛盾
- 草稿claim与我自己的机制分析(runs/refine_gate/细化头机制分析.md)**相反**：机制分析说 HRSID per-instance IoU −0.027、聚合靠召回补偿非质量改善。
- **修正**：改写 HRSID 段落=机制分析口径（分辨率杠杆不迁移、per-instance 质量轻降、聚合靠召回）。split/merge 降记为计数变化、明确标注 per-instance IoU 未升。

### B1（致命）：等参数对照 → ✅已做，结果=**容量不是分辨率**（重大修正）
- a0r_caps(conv@1/4,同参数,不提分辨率) seed0 **AP75 0.643 ≡ A0R-refine 0.643(+2.8)**；bilinear(纯分辨率)仅 +1.3。
- **评审B1确认：+2.8 来自新增 conv 容量、非分辨率提升**。"分辨率轴"叙事推翻→重构为"轻量学习细化容量"。诊断(掩码51%可恢复)仍成立，修复机制改名。
- 等参数对照**从软肋变贡献**(做了别人跳过的容量对照)。caps 可能比 refine 更优(同AP75/更好AP_m,AP_l/无OOM)。
- 裁决:runs/refine_gate/B1_等参数对照裁决.md。待 seed1(训练中)+HRSID caps 确认。

### B4/B6/B7（trivial）：引用 key 不匹配
- cheng2021boundaryiou→cheng2021boundary；mansour2026zeroshotsar→mansour2026zeroshot；shao2023sisnet→shao2023scaleinscale；psegssdd_note 删或补。组装时全局替换。

## major（23 条，择要）
- M1: +2.6 双seed σ~0.004 是 1-d.o.f. 估计→已入队第3seed。
- M2: bilinear +1.3/+1.5 分解单seed→已入队 bilinear_seed1。
- M3/M4: HRSID 单seed、借 PSeg σ 判定→改为"无方差估计、不做显著性"，对称处理 +0.5/−1.8/−0.2(AP_l 改"consistent with"非"dispels")。

## 行动状态
- ✅ 已建 a0r_caps 配置 + 第3seed 入队(B1/M1/M2)。
- ⏳ 待改写(B2/B3/B5/M3-4)：组装 seal.tex 时按上述诚实口径，不照搬草稿。
- ⏳ 可选实做：真边界 patch oracle(B2②)、3seed/bilinear_seed1 收敛后回填。
- 审计工作流(w2vpyqn52)独立复核中，会交叉佐证。

## 新增诚实修正（2026-06-19 机制分析）
- **检测播种=定位精化非覆盖**（runs/multiseed/检测播种机制分析.md）：AR+1.85 全来自高 IoU(recall@0.9 +6.2pt)，松阈值(0.5)覆盖反降−3.5pt(近岸)。真实证据种子精度17.6%加不了覆盖、只精化定位。论文检测段改"召回booster"→"高IoU定位精化器，覆盖瓶颈仍开放"。**论文工作流w1c8mnl54用了旧框架，组装时改**。
- **细化头=容量非分辨率**(B1双seed)、**oracle 51/49**(非60/40)、**G1对ts_cfar+披露mne_full**(B3) —— 这些已在 CORRECTED 块传给重写工作流。
