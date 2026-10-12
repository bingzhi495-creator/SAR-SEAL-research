# _RESTRUCTURE_MAP.md — 27页→12页重构地图(依据 REVIEW_LIUQK.md 27条批注)

生成: 2026-08-07。数字权威: docs/ALL_TABLES.md + tab_univ.tex。风格权威: STYLE_GUIDE.md + 反AI规则。
备份: _v27_backup/(内容移动的**唯一取材源**, 避免agent间时序依赖)。
铁律: 零幻觉——任何数字必须能在 ALL_TABLES.md / tab_univ.tex / 现有正文中逐字找到。

## 0. 现状盘点(词数近似页数; 27页版, 正文约24.5页+参考文献约2.5页)

| 文件 | 词数 | 占比 | 现估页数(含浮动体) | 目标页数 |
|---|---|---|---|---|
| sec_intro.tex(含摘要) | 2089 | 10.5% | ~2.6 | 1.9(摘要0.5+Intro1.4) |
| sec_related.tex | 811 | 4.1% | ~1.0 | 1.0 |
| sec_method.tex | 3989 | 20.1% | ~5.0 | 3.2 |
| sec_exp.tex(+tab_univ/fafn) | ~11800 | 59.5% | ~13.5 | 5.8 |
| sec_disc.tex | 1836 | 9.3% | ~2.4 | 1.0 |
| 合计 | ~19800 | | ~24.5 | **12.9(硬上限13, 目标12)** |

浮动体现状: 16图+11表 → 目标 **9图(保底11)+7表**。

---

## 1. 新章节骨架

### Abstract(~230词, 重写; 批注#1#2)
按 STYLE_GUIDE §2 模板: 1句背景 → 2句现有方法困难(近岸25-30 AP落差; 专用架构路线未隔离预训练变量) → 提案句组: (i)共识门控mask-IoU投票融合算子(soup+宽TTA只作载体一笔带过, **算子是创新**, 批注#6), (ii)两个自纠错级联(BPR边界修补+学习重排, 皆用模型自身训练集错误, 零新标注), (iii)预训练体制发现一句 → 末尾1-2句简报结果: HRSID 44.5/59.4/70.1(近岸AP75 48.3), SSDD 62.1/67.5/69.8(近岸/官方/远岸), 6载具5家族全部提升 → 收束句"demonstrate the effectiveness and robustness"。零引用、缩写全展开、不自称SOTA。maritime surveillance句只出现一次(#2)。
Keywords按字母序更新, 建议: Consensus fusion, instance segmentation, self-correction, ship detection, synthetic aperture radar (SAR), test-time augmentation (TTA)。

### I. Introduction(4段式+3条contributions+组织段, ~1150词, 第2页左栏收尾; #3#4#5#6)
- P1 任务重要性: SAR全天候→船舶实例分割供给船体几何; HRSID基准。(现L19压缩)
- P2 现有方法+问题: 专用架构路线(O2Former/DiffSAR/Gauss-heatmap/LFG-Net/SARSAM一句带过各自机制)+两体制分裂(近岸落差25-30点); 无人隔离预训练变量; 融合环节无人研究——"现有方法问题"落点在: 近岸残差被当成容量问题、聚合(fusion)自由度被忽视。(现L21+L25压缩合并)
- P3 我们做什么+实验效果: 一段讲完三级级联(共识融合算子为核心的training-free恢复→BPR→学习重排, 皆不动主模型), 效果只给头条: HRSID近岸44.5 AP/48.3 AP75超500-epoch最强发表架构(43.1/45.9), SSDD 62.1, 6载具5家族66格全升。COCO-init 17.9→27.7一句作为动机陈述。(现L23/L27/L29/L31/L33/L35共6段压成1段, 七方向sweep细节全部下放IV-E/V)
- P4 contributions引入句+3条(每条~50词≈5行, 方法创新为主, 不报实验过程; #5#6):
  1. **共识门控mask-IoU投票融合算子**: 以按mask-IoU聚类+跨视图一致性门控的投票融合为核心的training-free恢复框架(权重soup与宽多尺度翻转TTA为其载体); 融合算子而非视图数决定激进TTA成败(matrix-NMS崩至4.1/多数投票蚀远岸)。
  2. **自纠错推理级联**: 两个只写不相交字段的小修正器——±4像素带护栏的边界修补器(只写mask像素)与14特征共识重排器(只写分数), 均以冻结模型在训练集上的自身错误为监督, 零新标注, 且经泄漏审计。
  3. **预训练体制研究+实验效果**(末条可含实验, #5): COCO初始化单独值9.8点近岸; 七方向架构/训练侧对照全部flat/negative/absorbed; 完整配方使HRSID近岸达44.5 AP/48.3 AP75, 超最强500-epoch专用架构, 并在6个载具×5个架构家族×两数据集全部三split上无一例外提升。
- 组织段(固定句式)。
- 现5条贡献中的"negative-results ledger"独立条目取消, 并入第3条一句。

### II. Related Work(3小节, 每小节两段·段2≈4行gap句, 长度相近, 共~850词; #7)
- A. SAR Ship Instance Segmentation: 段1=现有专用架构(现L12前半); 段2=不足: 全部from-scratch协议、无人隔离预训练变量、无人研究推理侧聚合→我们关注点。
- B. Test-Time Augmentation, Weight Ensembling, and Inference-Side Recovery: 段1=EMA/soup/TTA/BPR/DiffusionDet(现L20压缩, DiffusionDet留1句); 段2=gap: 强checkpoint微调域上weight averaging证据最薄、融合算子问题无人回答。
- C. Adaptation and Priors on Strong Bases(合并现PEFT+Anisotropic两小节): 段1=Mona/Frozen-DETR/LayerScale/strip算子一段讲完; 段2=gap: preservation必要不充分, 先验须匹配目标体制(我们的strip负迁移为控制证据)。
- 每小节末尾按 STYLE_GUIDE 加一句"批评+指向本文"。

### III. Method(A-D四小节, 文字~1700词; #8#10#11#9#16-19#12)
- **A. Overview**(~350词): 问题设定(1-2句+必要记号, 删现Eq.(1)输出集合与Eq.(2)readout推导——基座是标准Mask2Former, 引文即可); 两阶段+三级级联总图(Fig.1); **逐条点名哪级是contribution**(#10): 第1级=融合算子(III-B), 第2/3级=自纠错器(III-C/III-D); 预训练体制约束(17.9→27.7, preservation constraint)压成~100词; soup并入III-B。基座张量形状清单、55ep/100872it等**全部移出**(→IV-B)。
- **B. Consensus-Gated Mask-IoU Vote Fusion(含载体)**(~700词): 开头1段=载体: soup(Eq.soup+same-basin两条件+一句"+1.9近岸"; 距离几何30.16/29.81/30.18等recital删, 脚本名删)与宽TTA(Eq.views; eq:pool删为文字; 6视图+7.0/加宽+2.0各1句)——TTA明确写成"common practice we adopt as the carrier"(#6)。主体=融合算子: 失效模式段(压缩)、Eq.(iou)(cluster)(fuse)(score)、Algorithm 1、agreement band机理段、Fig.(figm_fusion)。iou_matrix_meta.json句删。
- **C. Boundary-Patch Refinement**(~300词): 残差诊断1-2句(stride-4上采样halo)、BPR范式引用、训练来源(训练集自身错误, IoU≥0.5配对)1句、band guard Eq.(guard)、"只写mask像素/分数不动"安全性质2句。Eq.(bprloss)删("binary cross-entropy plus Dice loss"一句+引文); 19min/patch64/2×等→IV-B; AP75指纹分析→IV-E。
- **D. Learned Consensus Re-Ranking**(~350词): 动机(AP是排序积分; γ=0处fusion已计算共识量却弃用)、14特征向量Eq.(rrfeat/rrblocks可合并为一式)、回归目标与blend Eq.(rrblend)、"只写分数, mask逐字节不变"2句。α选择流程/行数51781/Spearman/permutation importance数值→IV-B与IV-E; 脚本名删。
- **删除小节**: 现III-A(并入A)、III-C soup(并入B)、III-D TTA(并入B)、III-H Complexity(全部→IV-B一段"Inference cost": 8×实测、K≤800 Gram、修正器CPU开销)。
- 保留公式: soup, views, iou, cluster, fuse, score, guard, rrfeat(+blocks合并), rrblend ≈9个。删除: output, readout, pool, bprloss ≈4个。
- 图注修理(#9#12#16-19): Fig.1注删绘图比例尺/脚本/manifest语句, Color key文字用\colorbox底色对应图色; figm_fusion/figm_bpr注删"×8放大/1.6mm"类绘制说明, 保留信息性内容; 全部注自足(STYLE_GUIDE §6)。

### IV. Experiments(A-G, 文字~2600词; #20-27)
- **A. Dataset**(~120词): HRSID 5604/3642/1962, 369近岸/1593远岸; PSeg-SSDD 928/232, 46/186(#20)。
- **B. Implementation Details**(~280词, 吸收方法节全部实现细节; #11#22): c0协议(COCO-init M2F-R50, 800², AdamW 2.5e-5, batch2×2累积, 100872it=55ep, 3090Ti); SSDD 8352it=36ep与TTA尺度重标{416,512,608,704}; MaskIoU头禁用λ=0一句; BPR训练(patch64, 2×, BCE+Dice, ~19min, 20ep on SSDD); 重排器拟合(HGB 300轮lr0.08, HRSID 51781行/SSDD 5122行, 4s+19s CPU; α∈{0.3,0.5,1.0}按4:1图像划分留出选择, HRSID取0.5/SSDD取1.0); 推理开销8×一句; Cascade mask头TTA修复脚注压成一句放此处。
- **C. Evaluation Setting**(~150词, 吸收种子纪律; #21): COCO mask AP×100三split; 三种子s0/s2/s3、σ≈±0.3噪声带、soup/TTA/融合皆确定性(现IV-L Robustness小节并入此处+E1一句, 小节删除); 同GT同评估器; 对手数字来源约定(published rows未复评)。
- **D. Comparison With State-of-the-Art Methods**(主实验, ~950词; #24#25#27#26):
  - D1 HRSID发表方法对比: **合并近岸+远岸为一张table***(见§2表清单); 每个方法名后加引用(#26, 需补bib); 分析段압缩: 头条句(44.5/48.3/42.0全列领先, +1.4/+2.4/+2.1 over O2Former; training-free档AP50 75.6 vs 69.5)+认输句(远岸70.1 vs 71.7/70.9; training-free档绝对近岸AP不领先)+操作点caveat段(500ep-scratch vs 55ep-COCO, 17.9 vs 27.7)压至~120词。
  - D2 SSDD自训六方法对比: tab:ssdd_cmp(保留)+分析压缩(~200词: 通用四架构38.9-41.2 vs SAR模块52.9/53.2 vs 我们62.1; OAEM/DenoDet为我们的复现的声明1句+选点披露1句); 检测级分析并入此处一段(~120词)+fig_pr(PR曲线); **tab_fafn删**, 保留文字: Pd 0.983 vs 最强竞品0.924, F1* 0.902 vs 0.882, 远岸饱和平局一句。
  - D3 定性对比(#27): fig8(HRSID六列, 66/34 vs 23/32与28/44的counts保留2-3个关键数, 逐panel全量清单删)、fig6(自家阶梯定性, 压缩至88船港口+近饱和远岸两条叙述)、fig7b(SSDD九列; "46场景中仅1场景严格正增益"的诚实段保留但压半)。场景选取规则各保留一句(无打分公式、无脚本名)。
- **E. Ablation Study**(~700词; #23#24):
  - E1 递进阶梯(原"Main Results"): tab:main+fig2(缩为单栏); 近加性(+1.9/+6.9→36.5预测vs36.1实测)与3模型ensemble对照句保留; 三种子TTA鲁棒性一句(+6.9/+7.4/+6.7)。
  - E2 融合算子消融: tab:fusion+机理段压缩(每失效模式1-2句); 单调性句保留。
  - E3 修正器分析: **tab:bpr与tab:rerank合并为一张表**; BPR指纹段(AP50+0.7 vs AP75+10.9)+7向量泄漏审计压成~3句(无GT访问/带内confinement 90of90/分数0改动/确定性/split卫生等择要); 重排审计压成~3句(只训练集拟合/mask逐字节不变/bootstrap 100%正[+0.61,+1.23]); 大目标AP-1.4诚实句保留。
  - E4 架构/训练侧对照: tab:arch+机理段压半(每方向1句); Swin-S与200ep等预算null各1短段; fig4删(推荐)。
- **F. Cross-Dataset Transfer and Universality**(合并压缩, ~350词):
  - F1 SSDD迁移: **tab:ssdd删**(其4行已逐字含于tab:ssdd_cmp下块), 文字给阶梯: 51.9/60.9/64.8→54.0/63.3/66.9→60.0/66.3/68.8→62.1/67.5/69.8(近岸/官方/远岸)与近岸AP75 54.9→72.5→74.6; 每算子号保持一致1句; 与O2Former同协议位置(差1.1/2.1)1句。
  - F2 通用性: tab:univ+fig_univ_qual(缩至2行); 21端点全升+两处中间负读数诚实句+42.3头条句; 现两大机理段(方差轴收缩/边界轴反向, ~900词)压成~150词一段(Spearman -0.77/-0.11与CondInst例外各1句)。
- **G. What Did Not Help**(半页~400词, 正文段落不设表):
  散斑边缘化(从sec_disc V-B移入, 压至~150词: 两变体37.1/37.3 vs 38.8, 子集吸收机理1句, arbiter预测存活1句); miss-rescue verifier(R12) 1句(43.5 vs 43.6 null); 16特征深度一致性(R10) 1句(44.5/59.3/70.0平局); 最小足迹变体1-2句(28.2→36.4被栈吸收); 收束句"每条轴恰有一个remedy"。

### V. Discussion(压缩~450词)+VI. Conclusion(~200词)
- V-A Preservation regularity: 保留但压~40%(七方向清单不再复述——IV-E已表格化, 只留regularity表述+两体制反号例证+Mona外证+strip先验教训+两修正器与regularity一致的读法)。
- V-B What-did-not-help整小节删除(内容已移IV-G)。
- V-C Limitations: 4条保留, 压至~200词(每条1-2句; 大段数字复述删)。
- VI Conclusion: 按STYLE_GUIDE §7重写为单段过去时: 总起→逐组件"困难+动作+效果"→实验一句→消融一句→未来工作一句(极简保留头条数字44.5/48.3与62.1, 其余数字删; proposed/introduced/devised轮换)。

---

## 2. 删减清单(到具体对象)

### 2.1 表: 11 → 7
| 对象 | 动作 | 去向/理由 |
|---|---|---|
| tab:main(递进阶梯) | 保留, 移至IV-E1 | #23: 实为ablation |
| tab:sota_in + tab:sota_off | **合并**为一张table*(方法×{近岸AP/AP50/AP75/APs, 远岸AP/AP50/AP75/APs}, footnotesize) | #24建议; 省一浮动体; 每方法名加\cite(#26, 需补bib: rtmdet, hqisnet, cascade mask r-cnn; solov2用于tab_univ正文) |
| tab:fusion | 保留(IV-E2) | 核心声称 |
| tab:arch | 保留(IV-E4) | |
| tab:bpr + tab:rerank | **合并**为一张表"Effect of the Two Trained Correctors": 行=split×{full stack, +BPR, +re-ranking}(9行), 列=AP/AP50/AP75/APs | 两表现有50%行重复(refiner行两处都出现) |
| tab:ssdd(SSDD阶梯) | **删** | 4行数据逐字重复于tab:ssdd_cmp下块; 近岸AP75阶梯(54.9/72.5/74.6)入正文F1 |
| tab:ssdd_cmp | 保留(IV-D2); 脚注中被排除方法(SOLO发散/PointRend/InstaBoost分辨率/FreqFusion崩)压成一句 | 主实验 |
| tab:univ | 保留(IV-F2), **不手改**(自动生成) | 通用性核心66格 |
| tab:fafn | **删**(删\input{tab_fafn}) | 与fig_pr二选一(任务指定); 关键数以2-3句入IV-D2; PR曲线免阈值争议且更省版面 |
| R12/R10 | 无表(本来就无), 各1句入IV-G | #任务要求 |

### 2.2 图: 16 → 11(硬性), 推荐9
硬删5张:
| 对象(编译号) | 动作 | 理由 |
|---|---|---|
| figm_soup(Fig.2) | **删** | soup降级为III-B内一段; 三角形几何图信息密度低, 距离数字1句正文即可(任务"fig_soup并入overview或删"→判: 删, overview图已含soup方块) |
| figm_tta(Fig.3) | **删** | TTA是载体非创新(#6), 扇出图装饰性; Fig.1已画8视图级; 删图本身就是对#6的排版级回应 |
| figm_rerank(Fig.6) | **删**(判断: **不并入BPR图**) | 三个panel中(a)特征清单=正文可述, (b)三个importance数字=正文一句, (c)"字节相同"是断言不是可视化; 若并入figm_bpr会把两个写字段不相交的独立修正器视觉耦合, 且撑爆一个通栏浮动体。删净比缝合好 |
| fig5_bpr(Fig.10) | **删** | 6个数字的柱图, 纯复绘合并后修正器表; "指纹"一句正文承载 |
| fig7_qual_ssdd6(Fig.13) | **删**(fig7b与fig13二选一→留fig7b) | fig7b(Fig.14)九列是其方法超集, 且行(a)(c)与fig7行(b)(c)同场景; fig7独有的10船港口叙述随之删 |
| fig_scaling / fig_speckle_stats | 不引入 | 正文从未\ref |
推荐再删2张(执行后9图, 用于把IV压进预算; 若版面充裕可回补):
| fig3_fusion(Fig.8) | 推荐删 | tab:fusion的18个数字柱状复绘; 崩溃对比由表+文字承载(样本刊消融一律用表) |
| fig4_arch(Fig.9) | 推荐删 | tab:arch的delta复绘; IV-E4本就压缩 |
留下的9(11)张: fig1_framework, figm_fusion, figm_bpr; fig2_progression(缩单栏), (fig3_fusion), (fig4_arch), fig6_qual_hrsid(**缩至2行**: 88船港口+远岸), fig8_qual_hrsid_multi, fig7b_qual_ssdd8, fig_univ_qual(**缩至2行**: MaskRCNN-HRSID+自家M2F), fig_pr_ssdd(改单栏两panel竖排)。
图片重制(缩行/单栏)是figs/生成脚本任务, 不阻塞文字agent; 文字按新版图行数写注。

### 2.3 正文段落删/压清单(按文件行号, 行号对应当前版=_v27_backup同构)
**sec_method.tex**
- L11后半"The figures of this section draw measured tensors…manifest.json": 删(#9#16-19)。
- L20-27(III-A): 并入Overview; Eq.(1)(2)删。
- L29-38: 张量形状清单段删; 55ep/100872it/lr/GPU→IV-B; 体制段压至~100词。
- L42-46: soup几何recital(30.16/29.81/30.18、17.27…、1.6e-7)压为1句"三checkpoints两两等距、平均即质心"; measure_soup_geometry.py句删。
- L57-77: eq:pool删; rescale=True等实现细节→IV-B; "deliberation budget"修辞删。
- L107尾"iou_matrix_meta.json"句删。
- L152-161: Eq.(bprloss)删; 19min/patch64→IV-B。
- L174-205: α流程/行数/Spearman/importance数值→IV-B与IV-E; measure_rerank_importance.py句删; L205安全性质压至2句。
- L207-209(Complexity小节): 整删, 内容→IV-B。
**sec_exp.tex**
- L20 roadmap段: 按A-G重写。
- L28种子段→IV-C; L26协议段→IV-B(吸收方法节细节)。
- L73/L96-99来源脚注保留但压缩; L128 caveat段压~40%。
- L160-162失效机理段: 每模式1-2句。
- L200-204: 每方向机理1句; L204 200ep段压半。
- L239 七向量审计段→3句; L269 六向量审计段→3句。
- L289两null段→IV-G各1句。
- L293-295定性counts叙述压半; L299-300: 逐panel counts全量清单删(留2-3关键数), 场景选取打分规则删(留1句"programmatic and reproducible"), SOLOv2旁述压成一从句。
- L305: tab:ssdd删后阶梯数字入正文; L381 fig7(已删图)相关叙述删, 其(a)行10船港口例子删。
- L410 OAEM/DenoDet段压半; L412 fig7b逐panel清单删、"仅1场景严格正增益"段压半。
- L416脚注(CascadeRoIHead修复)→IV-B一句; L418+L420两机理段→~150词。
- L426-430(detection-level)→IV-D2一段~120词; tab_fafn的\input(L398)删。
- L432-434(Robustness小节): 并入IV-C+E1一句, 小节删。
**sec_disc.tex**
- V-B(L16-21)整小节删(散斑~150词版移IV-G)。
- V-A(L12-14)压~40%; V-C(L26)压至~200词; VI(L31)按§7重写。
**sec_intro.tex**: 全文重写(见§1); 现L29七方向段、L31三算子长段、L33-35两rung长段的内容大幅下放III/IV。
**其他**: sec_method_v3.tex为旧稿, 不入编译, 任何agent不得动它。

---

## 3. 页数预算表(正文≤13, 目标12)

| 节 | 文字页 | 浮动体页 | 小计 |
|---|---|---|---|
| 标题+摘要+关键词 | 0.5 | — | 0.5 |
| I Intro | 1.4 | — | 1.4(第2页左栏收尾, #5) |
| II Related | 1.0 | — | 1.0 |
| III Method | 1.7 | 1.5(fig1 0.5+figm_fusion 0.4+figm_bpr 0.35+Alg.1 0.25) | 3.2 |
| IV Experiments | 2.6 | 3.2(表7张≈1.9: main/sota合/fusion/arch/修正器合/ssdd_cmp/univ; 图≈1.3: fig2单栏0.2+fig6二行0.3+fig8 0.35+fig7b 0.35+univ二行0.25+pr单栏0.25——按9图方案) | 5.8 |
| V Disc+VI Concl | 1.0 | — | 1.0 |
| **合计正文** | 8.2 | 4.7 | **12.9** |
| References(~26+新增4条) | | | ~2.0 |
| 全文 | | | ~15 ≤ 16-18上限 ✓ |

超支应急序(集成agent执行, 编译>13页时依次砍): ①fig4_arch(若保留了) ②fig3_fusion(若保留了) ③fig2_progression(阶梯已全在tab:main) ④fig6缩至1行或并入fig8 ⑤IV-G压至0.3页。

---

## 4. 改写agent边界划分(一人一文件, 禁越界)

| Agent | 文件(独占写权) | 任务 | 输入 |
|---|---|---|---|
| A1-intro | sec_intro.tex | 摘要+Intro按§1重写; 3条contributions; 不新增引用键 | 本地图§1, ALL_TABLES.md, STYLE_GUIDE §2-3 |
| A2-related | sec_related.tex | 4小节→3小节、每节两段+gap句 | 本地图§1, STYLE_GUIDE §3尾 |
| A3-method | sec_method.tex | 重构为A-D; §2.3方法删单; fig1/figm_fusion/figm_bpr图注修理(#9#12, \colorbox色标) | 本地图§1§2.3; 被移出内容**不负责落位**(只删) |
| A4-exp | sec_exp.tex, refs.bib(只增不改), 移除\input{tab_fafn} | 重排A-G; 两处表合并(sota合表、修正器合表)内联于sec_exp; 吸收实现细节(**取材自_v27_backup/sec_method.tex**, 与A3无时序依赖); 补bib: rtmdet/hqisnet/cascade mask r-cnn/solov2(条目须逐字核实, 零幻觉); IV-G撰写(**散斑内容取材自_v27_backup/sec_disc.tex**) | 本地图§1§2, ALL_TABLES.md, tab_univ.tex |
| A5-disc | sec_disc.tex | V压缩(删V-B)+VI重写 | 本地图§1; 散斑段放心删——IV-G落位由A4负责 |
| A6-integrate | main.tex + 全文件只读校对权(机械修改权) | 最后执行: MiKTeX编译(E:/miktex/miktex/bin/x64/), 页数核对, 浮动参数调整, \ref顺序/未引用浮动体检查, 反AI清扫(正文无bullet、禁词、furthermore全文≤3、数字优先、article不paper、美式拼写、缩写首现展开), 每个数字对照ALL_TABLES.md抽查, 执行§3应急序 | 全部 |

并行规则: A1/A2/A3/A5与A4可全并行(A4取材一律用_v27_backup快照); A6串行最后。引用键: 只有A4可新增bib条目, 其他agent只用现有键(见refs.bib现有26键)。图片重制(fig6/univ_qual缩行、fig2/fig_pr单栏)另开figs任务, 不在本轮文字agent范围; A4按目标版图写注并在注中不描述绘制过程。
遗留决策(需用户拍板, agent不得自作主张): ①标题是否随#6重心调整(现标题"Less Architecture, More Recovery…Training-Free…Pretraining-Regime Study"仍以TTA-free框架为重心; 可选改为突出consensus fusion+self-correcting cascade); ②fig3/fig4是否执行推荐删(9图方案) 或保11图。

## 5. 收敛诊断(2026-08-07实测, 6/7 agent后)
编译17页✓上限, 正文~15.3 vs 预算12.9, 超+2.4。分布: Intro1.3✓/Related1.0✓/Method3.0✓/**Exp~8.0(+2.2)**/**Disc+Concl~2.0(+1.0)**。浮动体按计划(7表5图), 超支全在散文。
散文大头=先前从caption搬入正文的核验背诵(如fig7b段84个数字/逐格计数recital)+Detection-Level散文+Disc未压足。
### 收敛指令(compile-converge执行)
1. sec_exp: 逐格计数背诵删至代表值(每图保留1-2个说明性计数, 整列8方法×3值的recital删——数据在图面板上已印, 批注#9精神'图是示意用真实结果'同样适用于'数据在图里不必正文复读'); Detection-Level段压半; 通用性段已压过再挤0.2页; IV-G确认≤0.4页。目标-2.0页。
2. sec_disc: 限制每条一句+结论6句, 压到合计1.0页。目标-0.7页。
3. 数字纪律修订: 因页限硬约束, 允许删除"图面板上已印刷的计数"的正文复读(此为唯一豁免); 其余数字仍不可丢。
4. 每轮编译量页, 收敛到正文≤13(措施1-2应足够, 不足再动: fig6缩1行)。
