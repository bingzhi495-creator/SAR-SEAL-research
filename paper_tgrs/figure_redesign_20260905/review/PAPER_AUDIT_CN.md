# 面向 TGRS 的论文结构与方法证据审阅

审阅日期：2026-09-05。对象：`C:/Users/houbingzhi/Downloads/main_submit.pdf` 全部 15 页、`E:/SAR/seal/docs/PROMPT_FOR_FIGURE_AI.md`，以及本文列出的本地实现、配置和已有文本日志。

**结论：研究方向值得继续，当前稿件尚不适合直接以“完全未修改的强基座 + 已隔离的预训练研究 + 保证安全的共识融合”作为主张投稿。首要工作是修复方法身份与验证协议，再将稿件集中为“冻结已训练载体上的候选融合、局部边界修正与质量排序”。** 已有表格中的提升值得研究，但本次没有运行训练、推理、COCOeval、bootstrap 或读取模型 checkpoint，因此不构成对 AP 数值、实验复现或投稿就绪状态的认证。

本文标记区分：**[已见]** 表示在论文、代码或现存文本日志中直接读到；**[推断]** 表示由这些材料得到的逻辑判断；**[待验证]** 表示需要另行实验或审计，不能当作已完成证据。

## 1. 论文真正有价值的结构

论文最有潜力的中心问题是：**同一已训练实例分割模型，在近岸密集目标上得到的多个候选应如何保留、合并，并分别修复轮廓和排序？** 这是与 SAR 应用实际困难直接相关的问题，且可以用固定候选池实验隔离。

建议按照三个输出层面组织证据：

| 层面 | 具体残差 | 现有处理 | 最直接的证据 |
|---|---|---|---|
| 实例候选 | 相邻船只被 box-NMS 混淆；跨尺度重复实例占据输出名额 | mask-IoU 贪心聚类、条件投票/保留种子 | 同一候选池的合并对照、邻船误删率、重复率、漏检 |
| 空间轮廓 | 输出分辨率与局部边界不准确 | 独立 BPR 网络、形态学边界带写回 | Boundary AP/IoU、轮廓距离、按目标尺寸的提升及失败 |
| 质量排序 | 检测分数与掩码质量错序 | 训练集误差监督的 GBDT 重评分 | 固定实例/掩码下的 AP、质量校准、排序变化及大目标损失 |

权重平均、八视图 TTA 是产生候选的既有方法；BPR 的基本构思也是既有工作。论文应把新意集中在 **SAR 密集场景下的有条件融合决策、与后续校正的具体接口，以及严格可复核的适用条件**。TGRS 官方范围强调有意义的方法推进，并要求完整说明实验方法与条件；仅把几个现成组件串起来、再用更高最终 AP 支撑，风险较高。[TGRS 官方范围](https://www.grss-ieee.org/publications/transactions-on-geoscience-remote-sensing/)

## 2. 最高优先级：正文中的“未修改基座”与训练记录冲突

**[已见] 论文第 1 页摘要、第 3 页 III-A、第 7 页 IV-B、第 14 页讨论/结论反复写 unmodified Mask2Former、未改变层/损失/训练日程，辅助 mask-IoU 头以推理时 λ=0 为依据称为 disabled。** 然而现存训练链路是：

1. `carrier/soup.py:39` 起定义 `BASE = .../runs/c0_maskiou_v2`，成员是 `s0/s2/s3/iter_100872.pth`；`run_bpr_queue.sh:23` 起的 BASECFG、TTACFG、CKPT 使用这条链路。
2. 三个保存的训练配置 `runs/c0_maskiou_v2/s0/c0_maskiou_v2_m2f_hrsid.py:186`，以及 s2、s3 同名文件相同行，均为 `maskiou_weight=0.5`，检测头类型为 `MSIoUM2FHead`。
3. 保存日志 `runs/c0_maskiou_v2/s0/20260706_104731/20260706_104731.log:2345` 实际记录迭代 100/100872 的 `loss_maskiou: 0.3336`；s2 的 `20260711_153205.log:2345` 为 0.3204；s3 的 `20260712_065351.log:2345` 为 0.3271。这里仅引用损失记录证明辅助监督被运行，没有新增任何实验指标。
4. `carrier/maskiou_head.py:78` 起的辅助质量头读取 query feature 和可微分 mask statistics，没有 detach；`:118` 起计算辅助输出；`:168` 起在权重大于零时把 `loss_maskiou` 加入训练损失。

**[推断] 把 `infer_iou_lambda` 设为零，只关闭推理评分中的质量混合，不能撤销训练时辅助损失对共享表示的作用。** 因此“同一未修改模型”“架构和损失从未改动”“所有触碰权重的尝试均无贡献”不能从当前材料成立。这也影响将 +11.1 AP 全部解释为未修改 stock Mask2Former 的输出恢复，以及用载体的预训练差异作因果结论。

本次未读取权重或逐张校验 dump 的 checkpoint hash，故精确数字到 checkpoint 的最终绑定仍需补齐；但保存的配置、损失日志和部署脚本已经形成需要立即解释的实质矛盾。

可执行的两条修正路径：

- **保留现有训练结果**：如实命名为“COCO-initialized Mask2Former carrier trained with an auxiliary mask-quality objective”；写清该头在最终推理中关闭，冻结发生在完成训练之后。将 stock、aux-trained/λ=0、aux-trained/λ>0 三种状态分开。
- **坚持 unmodified 主张**：需要真正的 stock/no-aux 训练 checkpoint 与同一完整恢复流程对照。该实验未完成前，不能通过把 λ 改为零来沿用 unmodified 标签。

框架图必须体现 **offline fitting → freeze → inference**，不能用“never updated”描述一开始就经 55 轮微调的模型；权重汤是离线参数操作，不能画成接收某张图像预测后再执行的后处理节点。

## 3. 决定可信度的验证协议问题

### 3.1 HRSID 的合并设置直接在测试集选择

**[已见] 第 7 页 IV-C 的 Hyperparameter selection 明确说明 τ、Smax、γ、八视图尺度和合并算子由三个 HRSID 测试子集上的 sweep 选择，没有独立验证集。** 这不是隐瞒泄漏；作者已作披露，但披露不等于消除选择偏差。三个子集也不是三次独立测试：official 包含 inshore 与 offshore。

“Smax 的变化只有 0.4 AP”不能界定全部选择偏差，因为被选择的不止 Smax，还有算子、阈值、尺度、评分策略和多个失败方向。“训练集重扫不能改善选择”也不能支持测试集选择合理：这些图像已经训练过基座，预测错误分布比真正未见数据更简单，稿件自己承认训练图像上的门控很少触发。

**必要的新增证据 [待验证]**：以原始大景/港口/获取批次为分组，从训练数据分出基座也未见的开发集；在其上选固定规则；锁定后在未用于任何开发的测试数据上一次性报告。旧 HRSID 结果可作为 exploratory results 保留，独立确认必须来自新的未参与开发样本或预先锁定协议。已有 PSeg-SSDD 参数迁移是正面证据，但不足以“洗掉”HRSID 的测试选择，也不能仅因算子没重调就称整个训练与比较协议完全独立。

### 3.2 辅助模型的验证与训练残差分布

**[已见] `carrier/bpr_refiner.py:167` 起用固定随机种子对 patch 编号打乱并按 5% 划分；没有按 image_id 或原始大景分组。** 同一实例/图像的重叠补丁可能同时出现在训练与验证中。它不是测试集标签泄漏的直接证据，但不能把这 1850 个补丁当作 1850 个独立泛化样本；选 checkpoint 的验证误差可能偏乐观。

`carrier/scic_rerank.py:204` 起确实按 image_id 将候选行分成 4:1，优于逐行随机划分；但这些图像都被基座看过，所以它验证的是该误差监督器在“基座训练图像”上的泛化，未必代表真实测试时错例。建议保留训练集自校正范式，同时补充 **out-of-fold 基座预测**或真正未见的开发集对照，量化 in-sample residual 与部署 residual 的差异。

第 11 页承认最终 BPR seed0 是 3 个种子中表现最好的一个，提升 0.3–0.5 AP；继续堆叠 re-ranker 的结果也继承该选择。应报告完整链路的预指定种子及多次训练均值/方差，不能仅以推理 deterministic 代替训练不确定性。

### 3.3 “同一协议”与预训练因果推断

**[已见] 第 8–9 页 PSeg-SSDD 竞品按 best-on-test 选 checkpoint，初始化包含 ImageNet 与 COCO、训练预算不同；作者已经透明披露。** 主表更准确的说法是 “same dataset split and evaluator”，而非笼统 one protocol。竞品 best-on-test 确实可能使竞品偏乐观，但不能说明所有其它不公平因素都抵消：更强预训练、三种子汤、八视图以及独立纠错器的监督和计算仍需单列。

第 1 页 17.9→27.7 比较跨论文、跨日程且载体身份有上述辅助损失问题，支持“不同训练协议下的观察差异”，不支持“独立隔离 COCO 初始化贡献 +9.8”。要声称隔离预训练，至少需要同实现、相同 head、损失、数据、日程和评估下的 ImageNet-backbone/random-head 与 COCO-init 配对对照，最好跨多个训练预算形成交互表。

Swin-S + stack 是单种子，而 R50 full method 带 3 个成员的汤（第 12 页 Table VII），不能把所有差异归因于表示保留或模型容量。200 epoch 的早衰减/重启与选 best-on-test 也不是单一变量。建议将这些行降为探索性训练控制，分别匹配“单模型对单模型”和“同种子预算汤对汤”。

### 3.4 分割、统计与运行预算

- **数据分割 [待验证]**：仅文件名交集为零不能排除同一原始大景的相邻/重叠裁块。应发布 image_id、文件名、父场景 ID、裁块坐标、传感器、获取时间/港口，以及 split lists 的版本/哈希。本次没有断言已经发生空间泄漏。
- **bootstrap [已见/推断]**：`tools/bootstrap_ap.py` 对固定预测的图像重采样是有用的不确定性估计；它不包含基座/纠错器种子与开发选择的不确定性。如果裁片在原始大景内相关，按大景 block bootstrap 比按 chip 独立采样更合适。official/inshore/offshore 区间不能当成三份独立复现实验。
- **PSeg-SSDD [已见]**：第 12 页只有 46 张近岸图像，base→stack 与 BPR→rerank 的 95% 区间跨零。可报告正点估计及精确区间，不应概括每一级在每个数据集都显著有效。
- **推理成本 [待验证]**：视图个数不等于准确的运行时间倍率；分辨率变化、Gram 乘法、BPR patch 数量、CPU/GPU 传输都影响耗时。按 HRSID 四个尺度平方和计算，八视图输入像素量约为单个 800² 输入的 9.8432 倍；这只是解析输入工作量，不能替代实测时延。应给单视图/4-view/8-view 的 batch、精度、warmup、GPU 同步、平均及 P95 ms/chip、显存和各阶段耗时。
- **训练成本 [已见/推断]**：Table IV 的 soup 行“0 training”只能表示已有 checkpoint 的平均操作无需训练，不应隐去另外两次 55-epoch 训练。第 2 页“55 epochs”和第 14 页“one third”需统一成每成员 55、总计 3×55，再列纠错器和全部开发成本。

## 4. 合并算子的精确定义与论断边界

### 4.1 簇大小不是不同视图支持度

**[已见] Eq. (4) 与 `carrier/mask_tta.py:300` 只要求同类别、与种子 mask-IoU≥τ、未消耗；没有限制每个视图最多一个候选。** `:319` 明确另算 `torch.unique(view_ids[m_idx]).numel()`，说明代码也区分 `cluster_size` 和 `n_views_present`。门控 `:361` 实际使用前者。

因此 `|C|/V` 不是一般意义上位于 [0,1] 的 cross-view consensus ratio，簇有七个候选也不保证七个不同视图支持。第 4 页“一票就是一个视图”的解释，只在特定图例中成员视图唯一时成立。

当前实现可诚实命名 **cluster-support-gated mask voting**。若要真正声称 cross-view-gated，需在匹配中保留 view ID、每视图唯一成员或显式视图归一化权重，再用 distinct-view support 控制门；这会改变算法，必须另行对照，不能只在正文换符号而保留旧数值。

### 4.2 这是启发式门，不是质量或安全定理

簇支持数不是形式化方差估计。多个视图可能一致地预测码头为船；少数视图也可能恰好给出准确边界。现有统计并未证明高支持必为干净离岸目标、中支持必为近岸高方差目标。

“高支持簇的平均只能侵蚀”“门控让宽 TTA 安全”均过强。图例最多说明某个候选簇的投票使掩码面积变小，**若未比较 GT，面积减少不等于精度下降**。严谨措辞是“在本研究设置下，有条件保留种子降低了无条件投票的退化”；给出 gate 分桶后的 seed vs vote ΔIoU、失败比例、GT 匹配纯度、按船长/距离岸线/密度分层结果。

投票边界也需明确：代码 `mean >= 0.5` 是“至少一半票”，偶数票平局保留前景；|C|=2 时是并集，不是通常理解的严格多数。贪心簇只要求每成员与种子相似，不要求成员两两相似，且依赖评分顺序，不能写成传递闭包或全局聚类最优解。

### 4.3 哪一部分贡献了提升

按论文 Table VI 的已报告数值，box-NMS→seed vote 提升近岸 1.1；seed vote→gated vote 额外 0.6；两者合计 1.7。前一项主要属于 mask 空间聚类/代表选择，后一项才是门控投票的增量。不要把全部 1.7 都写成门控本身的贡献。

此外，无条件 majority 与 probability mean 数字相同，不能自动推出二者数学等价。`mask_tta.py:366` 起有独立浮点概率路径、缺失概率的 fallback，以及高支持分支控制。需要公布当次 resolved config、实际浮点数据类型、是否有非 0/1 值及像素级差异证据；当前审阅不能验证论文的 bitwise 相同说法。

Matrix-NMS 的 4.1 AP 异常差，应先作为失败配置诊断。代码 `mask_tta.py:208` 起有 `filter_thr` 和 `max_num`；注释 `:214` 甚至指出某次 `filter_thr=-1` 导致重复候选保留，当前默认阈值为 0.15。因此正文“Matrix-NMS deletes nothing”不能表述为该算法固有属性。表格应标清当次 σ/kernel/filter_thr/后处理配置，并给出合理验证集调节后的强对照。

### 4.4 Mask voting 已有明确先例

第 3 页“Beyond box-space ... no study asks ... mask-level hypotheses”过于宽泛。MNC 的 CVPR 2016 正文 §6 已明确介绍 NMS 后将相似实例掩码逐像素加权平均再二值化；FCIS CVPR 2017 也采用 mask voting。不能把 mask-level voting 本身作为未被研究的空白。[MNC 原论文 §6](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Dai_Instance-Aware_Semantic_Segmentation_CVPR_2016_paper.pdf)、[FCIS 原论文](https://openaccess.thecvf.com/content_cvpr_2017/papers/Li_Fully_Convolutional_Instance-Aware_CVPR_2017_paper.pdf)

应增设相关工作对照维度：相似度是 box/mask、候选来自单模型多 proposal/多 TTA view、是否跨视图唯一匹配、如何加权、是否条件保留种子、是否输出质量评分。**新颖性尚未通过穷尽检索认证**；当前发现足以要求收缩“首次”或“无人研究”措辞，但不代表已经证明具体门控方案没有新意。

## 5. BPR 与重排器：可证明的性质和不能推出的性质

### 5.1 BPR 的边界带与概率写回

`carrier/bpr_apply.py:42` 使用 3×3 核执行 r 次 dilation/erosion 后 XOR。精确对象是离散形态学带，不能默认成 Euclidean 距离 ≤4 的圆形带；请将 Eq. (7) 写成真实形态学定义，或明确采用的像素邻域距离。

`:73` 先 sigmoid，`:74` 用 INTER_AREA 缩小概率，`:76` 在重叠补丁中平均概率，`:79` 只在 band 且有覆盖的位置阈值写回；`:81` 为空掩码回退。第 5 页“overlapping patch logits are averaged”与实现不符，应为 probabilities。图中可以准确标出带外像素不变和输入/输出分数相同。

“只能修改边界”不保证拓扑不变：在细小目标上带可能覆盖大半甚至整个目标，带内编辑可能产生断裂、孔洞或局部连通变化。可保证的是 **检测记录数不变、带外像素不变、空结果回退、分数不变**，不能升级为物理船体身份或几何质量永远不受损。

第 11 页 r=2/4/8/无限带都给出相同 AP，支持“实现的编辑集中于边界附近”，不支持“±4 guard 导致 +4.8 AP”。应将 guard 定位为部署约束，将 BPR 的监督能力定位为精度来源。

### 5.2 “应用不需要 GT”与脚本接收 annotation 的区别

`refine_instance` 本体只读取图像、预测和 refiner，未使用 GT，这是可从静态代码支持的性质。但 `bpr_apply.py` 外层确实要求 `--ann`、加载 COCO，并用图像元信息和类别及评估器；`run_bpr_queue.sh` 也传入测试 annotation。稿件“apply path receives no annotation”需要指向真正独立的 no-GT 审计脚本和产物，不能仅根据这份 wrapper 直接复述。

建议工程接口拆为 `infer(image_manifest, predictions, weights) -> results` 与 `evaluate(results, annotations)`；使用只含图像 ID/尺寸/路径的 manifest 即可明确消除歧义。本次没有发现 GT 被送进 refiner 的证据，也没有把接收评估文件等同于预测泄漏。

### 5.3 互不覆盖字段，不等于互不影响任务

**[已见] `scic_rerank.py:275` 起保持检测记录和掩码，仅改变 score；但 `:287` 起把 BPR 后的掩码与原始 stack 的掩码按 IoU 贪心重匹配，`:305` 起从原始 stack 读取特征，包括其几何量。** 因此：

- BPR 改变掩码会改变特征归属匹配；执行顺序不是理论上可交换。
- 即使特征固定，AP 中 GT 匹配与 score 排序仍与 mask 质量耦合。两个模块修改不同字段，不意味着一个不可能抵消另一个的 AP 收益。
- 第 11–12 页“pure re-ordering ... must be [positive at every threshold]”错误。反例是把一个 FP 排在 TP 前；固定 mask 集下 AP 即可下降。
- max-IoU regression 的目标与 COCO 一对一匹配不同：两个重复候选都可有高 max-IoU 标签，但 COCO 会把其中一个记为重复误检。需要讨论目标近似以及融合后残余重复候选的影响。
- 训练 target 和 geometry 来自 BPR 前 masks，应用最终 AP 来自 BPR 后 masks，质量关系有分布差异。可以保留，但需明确，最好作 raw-target/refined-target 对照。

推荐保留原始 `instance_id/cluster_id` 跨级联传递，避免不必要的 IoU 重新关联。若确实只有 JSON/RLE 中间文件，则报告 match/unmatch 和潜在错误关联率。建议把性质重写为：“The refiner preserves scores, and the reranker preserves masks. Their empirical combination is evaluated sequentially.”

`scic_rerank.py:230` 评估 α=0 对照但 `:232` 的选优循环不包含 0，意味着会强制从非零 α 选择，即便不优于原始分数；可把 0 纳入验证集选择，但这属于未来修正，需要重新锁定协议，不应悄悄改代码沿用旧数值。

## 6. “方差、容量、物理上限”的解释应降为可检验假设

**[已见] 论文多处将 TTA 提升归为 variance、七方向负结果归为 architecture capacity 不需要、辐射一致性 +0 归为 sensor ceiling。** 这些材料支持启发式诊断，不支持完整因果分解：

- TTA 改变目标采样大小、感受野、插值、候选召回及筛选；其 AP 提升并不等价于训练方差减少。
- “mask 在 stride 4 预测”使边界误差解释合理，但不能仅凭分辨率断言所有边界残差都来自固定 halo。需要分辨率控制、边界偏移分布以及专门的边界指标。[Boundary IoU 原论文](https://openaccess.thecvf.com/content/CVPR2021/html/Cheng_Boundary_IoU_Improving_Object-Centric_Image_Segmentation_Evaluation_CVPR_2021_paper.html)
- 若想讨论 bias/variance，可预先定义像素概率的跨视图方差或重复训练方差、误检/漏检与支持度关系，并验证其是否解释 gate 的收益；不能把 support count 直接称为 variance estimator。
- 七个有限、且若干单种子与预算不完全匹配的失败方向，不足以推出“保留程度越高越有效”的规律。尤其实际载体采用了辅助质量监督，削弱该叙事。
- 第 13 页 CondInst 解释为“最 view-sensitive”、SOLOv2 轮廓差来自 grid kernels，属于待验证机制假设；支持/反例样本都很少，且剔除 CondInst 后相关性改变很大，不宜由 n=6/7 的不显著相关作一般规律。
- 单幅已发布灰度幅度图确有限制，但一个 radiometric probe 的 +0 不能证明 Bayes 误差或物理信息上限；需改为“当前输入与方法下仍未恢复的误差”。HRSID 官方库说明存在多种分辨率、极化和海况，稿件第 14 页“两套 single-sensor”还应核对；使用单通道图像也不能自动证明原数据为 single-look。未见原始成像元数据时，建议写“single-channel SAR intensity/amplitude products”。[HRSID 官方仓库](https://github.com/chaozhong2010/HRSID)

## 7. 最有价值的补实验顺序

这里列的是下一阶段设计，**本轮未执行**。先修方法身份，再做确认性实验，避免继续堆更多模块。

| 顺序 | 要回答的问题 | 最小充分设计 | 决策作用 |
|---|---|---|---|
| 1 | 结果究竟来自哪个载体？ | 建立 stock / aux-trained λ=0 / aux-trained λ>0 的训练配置、日志、checkpoint 和预测关联表 | 决定标题、方法名、预训练叙事是否保留 |
| 2 | 融合收益是否受测试选择影响？ | 未见开发集选择全部设置、固定后独立确认；旧结果保留探索标签 | 支撑投稿主结果可信度 |
| 3 | 门控的独立增量是什么？ | 固定候选池：box-NMS、mask-NMS/seed、经典加权 mask voting、无条件投票、当前 gate、distinct-view gate | 把 mask 聚类与门控收益分开 |
| 4 | gate 对哪些簇有效/失效？ | 按支持数、distinct views、尺寸、邻船密度分桶，比较 seed 与 vote 的 GT IoU | 将启发式变成有证据的 SAR 机制 |
| 5 | BPR 是否只修边界？ | 原始 BPR 风格无带/有带，Boundary AP/IoU、拓扑变化和失败样本 | 区分既有 BPR 能力与新约束 |
| 6 | 重排与 BPR 是否互补？ | 固定实例 ID 的 2×2：原始、仅 BPR、仅 rerank、两者；所有设置 train-val 冻结 | 支撑串联设计，不声称数学无干扰 |
| 7 | 实际部署代价是否合理？ | single/4-view/8-view 各级完整时延与显存、每图目标数和 patch 数分层 | 用 4-view 近似点建立精度—代价曲线 |
| 8 | 迁移结论是否稳健？ | 每载体固定训练/验证协议，纠错器明确需重训；至少完整最终链多种子 | 以 applicability 替代 universality |

无需为追逐容量增加复杂 backbone 或注意力模块。相较继续搜索第八种结构，修复载体身份、外部验证和候选合并机制的证据回报更高。

## 8. 建议的标题、贡献与正文骨架

推荐主标题：

**Consensus-Gated Mask Fusion and Post-Hoc Correction for SAR Ship Instance Segmentation**

若维持当前基于 |C| 的 gate 而不加入视图唯一匹配，技术上更精确的备选：

**Cluster-Support-Gated Mask Fusion for SAR Ship Instance Segmentation**

不建议以 Less Architecture, More Recovery 作为主要论断：当前辅助训练事实与不匹配负结果使其过度对立；Inference-Only Learning、Training-Free Full Pipeline、Self-Supervised 也不准确。纠错器使用已有人工标注的训练 GT，是监督学习，只是没有新增标注。

可用的三条贡献草案（均需完成相应验证才能最终定稿）：

1. **问题与算子**：研究密集近岸 SAR 实例在多视图候选聚合中的重复抑制和轮廓折损，构建以真实 mask-IoU 匹配、按候选支持条件选择投票或种子保留的聚合算子，并用相同候选池隔离其效果。
2. **校正接口**：构建冻结载体后的边界修正和质量重评分流程，利用既有训练标注生成监督，显式限定 mask 与 score 的写入范围；说明 BPR 继承的结构与新增接口约束，提供记录级可复核不变量。
3. **受控证据**：在预先锁定设置的数据划分和多个载体上评估贡献、成本、失败模式及训练变异，分析哪些误差由候选聚合恢复、哪些需要边界与排序校正。COCO 预训练观察及失败结构探索放在辅助研究，不作为已隔离因果结论。

建议正文结构：

- **I Introduction**：具体 SAR 场景问题 → 多候选聚合残差 → 三个层面的设计 → 范围清晰的贡献。删除“问题几乎全部只存在近岸”等泛化判断。
- **II Related Work**：SAR 实例分割与预训练协议；mask voting/候选聚合；BPR 与质量评分。将架构/adapter 大段背景缩短，补 MNC/FCIS 等直接前作。
- **III Method**：实际载体与离线训练；坐标反变换和稳定候选 ID；精确门控融合；BPR；重排；复杂度及能保证的不变量。Fig. 1 表示真实数据流与训练流，Fig. 2 解释门的两个分支，Fig. 3 解释可写区域。
- **IV Protocol**：数据/大景分组/开发选择规则；各模型预算与初始化；种子和指标；预先指定主终点与代码来源。
- **V Results**：受控主结果 → 同候选融合对照 → BPR/重排的 2×2 → 跨载体 → 时延与失败例。文献转引值单独成表，以参考而非同条件胜负定位。
- **VI Discussion and Conclusion**：明确适用范围、负结果和未解决的密集/极小/大目标问题。将七方向失败日志、汤配对、更多案例和实现审计细节放补充材料。

## 9. 可立即修正文句与图表问题

| 位置 | 当前问题 | 修正方向 |
|---|---|---|
| p1/p3/p14 | unmodified / no layer or loss modified | 按真实辅助损失训练身份重写；冻结限于拟合完成后的推理 |
| p2/p14 | 55 epochs 与 one third 混用 | 写每成员 55、总 3×55；另计纠错器和开发成本 |
| p3 | no study merges mask hypotheses | 增补 MNC/FCIS，收窄到当前 SAR 候选支持门控问题 |
| p4 Eq5 | majority 但偶数票平局保前景 | 明确 at least half，包括 \|C\|=2 为 union |
| p4 Eq6 | \|C\|/V 称跨视图比率 | 说明 candidate support 与 distinct-view count 不同 |
| p5 Eq7 | 欧氏距离不明；平均 logits | 改为真实形态学带；平均 sigmoid probabilities |
| p6 | disjoint writes → cannot undo each other | 改为字段不覆盖；性能互补必须由实验验证 |
| p7 | train-side re-sweep could not improve choice | 删除合理化测试调参的推论；解释真实未见开发集需求 |
| p8–9 Table II | one protocol | same split/evaluator；分列初始化、模型选择和计算 |
| p11 Table VI | Matrix-NMS deletes nothing | 报配置特定失败，补合理参数对照 |
| p11–12 | pure rerank must improve every threshold | 删除必然性；只报告观察到的正负变化 |
| p12–13 | universality / every endpoint | 改 tested carriers；区分 stack、BPR、rerank 覆盖范围及负结果 |
| p14 | single-sensor、single-look、sensor ceiling | 查原始成像资料；未证实则改成输入产品与当前证据限制 |
| p15 Ref.[10] | BPR 页码 13921–13930 | CVF 官方是 13926–13935 |
| p1/p14 | XXX affiliation、Manuscript date、GitHub/XXX | 定稿前补齐真实作者信息、机构与可访问代码入口 |

BPR 页码核对来自 [CVF 官方页面](https://openaccess.thecvf.com/content/CVPR2021/html/Tang_Look_Closer_To_Segment_Better_Boundary_Patch_Refinement_for_Instance_CVPR_2021_paper.html)。该论文也已明确提出冻结任意分割模型后的边界补丁精修；本稿需突出实际新增的约束与组合证据，而不重复申领其基本构思。

图件应使用可追溯真实素材，但真实素材本身不能跨阶段冒用：single-view mask、TTA fused mask、BPR output 和 reranked output 需在 manifest 中各自标明。`pred_masks.png` 的文件名不足以证明它来自最终融合；总体图若只能获得单视图真实输出，caption 必须限定其角色。投票“会侵蚀”的反事实图需说明比较对象是 seed，若没有 GT 质量对照就不要图示为精度定论。各图中的训练冻结状态、权重汤位置和两条纠错监督路径，比增加 3D 张量细节更关键。

TGRS 当前作者页要求图注能够独立解释图表，并说明 2026 年提交稿超过 10 个印刷页会产生超页费。15 页并不等于无法投稿，但建议通过移出探索性大表和重复案例提高信息密度；收费与最终页数以编辑部确认为准。[TGRS 作者说明](https://www.grss-ieee.org/publications/author-resources/tgrs-information-for-authors/)

## 10. 当前能与不能给出的判断

能确认：全文有一条可整理为研究问题的输出恢复路线；核心算子可写成确定程序；BPR 核心的局部写回和 reranker 的 score-only 操作有静态实现支撑；作者披露了不少不利事实，这些应继续保留。

不能确认：结果独立复现、训练与数据完全无泄漏、具体 gate 已充分新颖、44.5 等数值可归属于完全未修改载体、每级方法必定安全或性能互补、已经达到 TGRS 可直接提交状态。**最优先修正是辅助质量监督的真实训练身份；其次是测试集参与设计选择；再次是门控新意与机制验证。** 修复后，论文可以围绕明确问题、精确算法和诚实的适用边界建立更有说服力的 TGRS 叙事。
