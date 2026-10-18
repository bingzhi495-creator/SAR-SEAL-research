# U57 固定 query 精修：未匹配项风险的有界查新与最小否证设计

访问日期：2026-10-18。**状态：未证实假设；没有读取实验载荷、状态、资格链、图像、标注或权重，没有运行模型、SSH 或 AP。** 本文为独立于同期正式训练的普通研究笔记，使用 research-lit 流程，未改变当前训练或评估协议。

## 首先需要纠正的推断

**matched-only mask supervision 本身不是已证实缺陷。** 常见 set prediction 方法本来就只监督匹配项的 mask，未匹配 query 通过 no-object 分类约束；不能把“所有未匹配 mask 没有空掩膜监督”直接诊断成错误。本项目更窄的待检验点，是共享精修器学习时只接触 matched 条件，推理时却修改全部固定 query 的 mask，而教师分数及候选规则保持不变。

**FP mask 面积增加不等于 COCO AP 必降。** 官方 COCO evaluator 的 `evaluateImg` 按分数匹配，`accumulate` 使用匹配、ignore 与分数累计 TP/FP。因此，若同一候选排序下，各 IoU/area 设置的 TP/FP 和 ignore 向量都未变化，AP 不随 mask 面积变化。已是 FP 的 mask 变大且仍为 FP，可能只增加背景像素污染；AP 损失必须落实到 TP→FP、重复匹配竞争、ignore 或导出候选变化等。上述条件结论来自 [COCO 官方 `cocoeval.py` 的 `evaluateImg` / `accumulate`](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py)，不是本项目已出现此现象的证据。

未匹配训练项、原生评估 FP、重复候选、低分候选及空 mask 是不同集合，不能互换名称。GT 可用于离线诊断，不能在部署时决定哪个 query 需要精修。零初始化 head 只保证初始化时的恒等输出，不保证训练后不退化。

## 最直接先例及重叠边界

| 一手依据与具体位置 | 已有机制；对本项目主张的限制 |
| --- | --- |
| [Mask2Former 官方 `criterion.py`](https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/criterion.py)，`loss_labels` / `loss_masks`；[IAUNet 作者稿 §3.4](https://arxiv.org/html/2508.01928v1) | 前者 mask 按匹配索引取样，分类包含 no-object；后者采用分类/BCE/Dice 与 no-object 权重 0.1。补 query 负例不能宽泛称首次；IAUNet 还已有 U-Net 与实例 query 联合设计。本项目差异是固定外部 query/分数和加性 logit 残差，不是 no-object 思想。 |
| [SAMRefiner 作者稿 §3.2–3.3、附录 C.4](https://arxiv.org/html/2502.06756v2)；[ICLR 正式索引](https://proceedings.iclr.cc/paper_files/paper/2025/hash/90800f46f84381b7891e1378ee850013-Abstract-Conference.html) | 已从粗 mask 自动生成正负点、框和连续 mask 提示，并用 IoU 预测选多候选；++ 适配排序。提示可靠性和质量选择并非空白，但该机制不是本项目的稠密残差，也不能推出任意粗 mask 都不会被损害。 |
| [Price 等 PromptMoE，CVPR 2026 正式索引](https://openaccess.thecvf.com/content/CVPR2026/html/Price_PromptMoE_A_Segmentation_Refinement_Framework_Leveraging_Mixture_of_Experts_for_CVPR_2026_paper.html)；[作者 `PromptMoE.py`](https://github.com/sprice134/PromptMoE/blob/master/PromptMoE.py)，`PairRouter.gating_weights` / `eval_sam_prompt` | 已有 image-mask 条件动态专家及提示选择。作者代码的 `eval_sam_prompt` 在 SAM 框提示结果与输入 mask 的 IoU<0.5 时回退输入；**这是某专家图的防护，不能扩大为整个最终精修结果的质量保证。** 因而“动态信任门控/不可信则回退”的广义主张重叠明显。 |
| [SegFix，ECCV 2020 官方摘要](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/1633_ECCV_2020_paper.php) | 已有只修不可靠边界、借用内部像素预测的保守空间修正。限制修正区域不能直接当作新原理；边界定位可靠也不等于实例 FP 已受控。 |
| [Mask Scoring R-CNN，CVPR 2019 官方索引](https://openaccess.thecvf.com/content_CVPR_2019/html/Huang_Mask_Scoring_R-CNN_CVPR_2019_paper.html)；[Mask2Former 官方 `instance_inference`](https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/maskformer_model.py) | 前者已有 mask IoU 质量学习；后者将分类分数乘 mask 平均前景概率。因此 mask 质量与排序失配是旧问题。改分数会改变本项目当前固定分数的比较条件，应作为独立预锁控制，不能与 mask 精修收益混报。 |

PromptMoE 指 **Price、Cote、Rundensteiner** 的 CVPR 2026 工作，不是同名异常检测预印本。本文没有穷尽保守学习、拒绝预测或所有 2026 文献；正式正文部分访问受阻，PromptMoE 具体回退行为仅由作者公开代码核实，未认定与最终论文实现完全一致。

引文勘误：已有普通 U57 查新文档把 IAUNet 页码写为 4739–4748；[CVF 正式 HTML/BibTeX](https://openaccess.thecvf.com/content/CVPR2025W/CVMI/html/Prytula_IAUNet_Instance-Aware_U-Net_CVPRW_2025_paper.html) 为 **4778–4787**。同步版本已据此纠正该项页码；不改变方法比较结论。

## 一条可以被否定的最小假设

**H（待检验）：在既定完整训练终点，原生评估为 FP 的 query 上，统一精修增加了背景像素污染，且这组 query 的精修对固定候选 AP 有净负贡献。** 这不预设该现象存在，也不预设它由缺少负例监督造成；训练分布失配只是可能解释。

先用既定完整终点的原生教师、PLAIN、FULL 三份输出做一次共同诊断，不额外训练、不看中途 AP 选模型。严格沿用原划分、分数、query ID、阈值、几何变换、maxDets、同分稳定排序及 crowd/ignore 规则。先核验空 mask 是否在导出时被跳过，禁止精修前后悄悄改变空 mask 过滤、NMS、top-k 或候选数量。GT 只进入诊断和官方评估。

| 最少记录 | 固定定义与判据 |
| --- | --- |
| 原 query 标签 | 每图保留全部原 query ID、教师分数/排序、原生是否空、导出状态。训练匹配 ID 若未来可取得则另列；评估 FP 不伪装成训练未匹配项。 |
| TP/FP 转移 | 从官方匹配输出取每个 IoU 阈值的 TP→TP、TP→FP、FP→TP、FP→FP，另列 ignore 转移；不能只用 `maxIoU` 代替 COCO 匹配。 |
| 重复匹配 | 记录原生/精修匹配 GT ID；对有足够 IoU 却因已有高分候选占用 GT 而未匹配的项单列。TP→TP 但换 GT 也保留，排查匹配连锁效应。 |
| 像素污染 | 在同一有效画布上统计 mask 面积及相对全部有效 GT 并集的新增背景像素，按原生 FP/TP/空 mask 与固定分数排名分层；crowd/ignore 区域单列。重复船体覆盖不等于背景污染。 |
| 只作诊断的回退 | 预先以原生 COCO 匹配在 IoU=0.50、area=all 的 FP query 定义固定集合 S，再把精修输出中 S 的 mask 换回原生，其他内容完全不变，重跑相同 AP/AP50/AP75。集合不按精修好坏重选。GT 参与选择，故此为 **oracle 反事实诊断，不能当可部署方法或论文主结果**。 |

若 S 的背景污染未增加，H 的空间部分不成立；若回退的 AP 没有改善，H 的 AP 部分不成立，即使 mask 看起来变脏也不能称它导致 AP 下降。若 FP→TP 的收益超过损害，应肯定统一精修的修复作用，不能为了支持负例方案忽略它。原生 TP 的 TP→FP 若是主损失，应优先研究过修正或域外边界问题，而不是强行归因于未匹配 query。

首次端点统计只作探索；固定图像配对置信区间描述数据不确定性，不能代替重复训练种子。跨种子/独立公共集确认后才可推广。测试端点暴露的问题不得通过反复测试集调门槛来“修复”。

## 只有风险成立之后才值得增加的实验

先不加新模块。若上述风险具实质影响，可预锁一个新配方的两臂约束对照：A 保留 matched-only；B 在相同结构、共同初始化、图像顺序和完整更新预算上，对预定抽样的训练未匹配 query 增加向原生 mask 的恒等一致性约束。**不要把全部未匹配项的真值设为零**，它们可能覆盖真实船舶或重复候选；保留原生也可能限制 FP→TP 修复，这就是需要同时检验的反例。B 的附加 query 计算与监督量如实报告，不宣称等计算或更快。

B 若没有稳定改善 AP，或仅减少背景面积却伤害召回，应否定其任务效用；B 即使有效，也只证明该明确配方的约束有用，不自动证明训练分布失配是唯一机制，更不是新颖性认证。当前最有价值的下一步仍是完成原配方公平长训并做上表诊断，而不是在没有错误证据时补出一个“第三创新模块”。
