# N26：query-mask条件质量回归基线执行前计划

日期：2026-10-17。状态：**PLANNED_NOT_CAPTURED_NOT_TRAINED**。采集器、strictrefs与执行资格未实现，不宣称已开训。N20、N23原STOP不变；本轮不为旧方法增加预算或改写结论。

## 问题与可检验主张

Root已接收N25：强R原S0 AP 68.09938494745545；S0×GTq为73.39640612271616，差+5.29702117526071个百分点；q-only AP为3.380706129，较原S0低64.71867881839333个百分点。它只展示固定mask下两种GT排序函数的回收空间，不是可学习收益或数学AP上界。

唯一基线是MaskIoU风格的query-mask条件IoU回归，非完整作者复现、非创新。检验仅凭e0和R自身mask，能否预测质量并改善排序。主评分只用S0×qhat；qhat=1须精确恢复原S0。S0为原完整导出分数，不重算、不冒称纯分类概率。

## FIT监督与特征

FIT固定741图、1661个原冻结真实matcher配对，保留原image/query/GT身份及顺序。目标是该query的**强R最终原图shape二值mask与原matcher所配真实GT**的mask IoU；L2回归。不是N25的max-any-GT，不能另选更高IoU的GT。低质、原生IoU0和R最终empty匹配均保留；empty预测对非空GT的目标为0。GT无效或union异常应停止。未匹配query不伪标0、不当真实背景、不改matcher。

输入1040维：
- 原FP32 e0共256维，固定、无归一化。
- R自身二值mask按非零像素最小轴对齐bbox裁剪，无padding；含最大像素，转半开区间。转FP32，双线性缩放28×28，align_corners=False，固定行优先flatten共784维。empty时784维全0。

GT只能生成回归目标，不能进入bbox/crop/resize、e0或输入特征。不用原图ROI纹理、GT框/面积或人工判读。teacher、强R、原score与mask冻结，只更新质量头。

## 数据前置：禁止把采样精修当完整R

N16 FIT旧缓存有full native_L/e0/matcher，但仅4096 sampled F，不能精确计算强R完整Ω。禁止用sampled PR、采样mask、补零F或旧近似作为完整R标签。

必须新实现固定teacher F获取，仅对1661个matched query执行完整R readout。执行前另锁源码、teacher/R身份、FIT列表、原配对、输入strictrefs与输出schema。逐图/query核验新native e0/L及ID与旧缓存一致，不能只查形状。沿既有FP32 Ω、选择/预算、chunk2048、实际tail长度、插值和原图还原规则；不用补齐尾块改变算术，严格>0得到最终二值mask。身份、native数值或R规则不一致即停在数据阶段。

保存原图尺寸、image/query/GT id、原ordinal、配对来源、R mask与e0身份、crop坐标/empty标志、784维特征及匹配IoU目标的可核记录。特征与GT目标生成分离；独立验证低质/empty保留、bbox边界、插值和IoU目标后，再绑定唯一训练输入。采集/独立验证源码及strictrefs目前尚未实现。

DEV固定原187图、官方380GT、18700自然query；以已有qualified R RLE和原native e0生成同样特征，不前向更换mask。DEV GT只供冻结终点评价，禁止训练、选epoch、crop或配方。所有100 query均预测qhat，不按matcher筛选；未匹配query的泛化风险保留。

## 唯一模型与预算

MLP：1040→256 ReLU→128 ReLU→1 Sigmoid，均Linear，无dropout/LN。FP32训练，loss为batch内逐样本平方误差mean；无质量筛选、重采样或额外权重。每seed标准随机初始化新头，记录框架版本、Linear默认初始化、初始参数身份及private RNG状态，不加载其他质量头。

private seeds完整整数依次为202610172601、202610172602、202610172603；主seed2601指第一个完整值。锁定原样本列表，每轮private RNG无放回打乱，各样本访问一次；用独立CPU torch.Generator完整整数seed初始化，不静默截成32位。恢复须保留模型、optimizer、轮次及private RNG，不能重置后冒称连续训练。

固定24 epochs，batch64，drop_last=False；每轮26步、尾batch61，共624 optimizer steps/seed。每seed新建AdamW：lr=3e-4，weight_decay=1e-4，betas=(0.9,0.999)，eps=1e-8；无scheduler、预热或early stopping。强R不更新，不能继承其optimizer。主终点固定E24，禁止DEV选epoch或挑随机初始化。采集、训练与评价成本分记，当前不编造GPU时长。

## 评价与停止规则

主评价仅E24 S0×qhat，沿官方完整DEV segm COCO、maxDets=[1,10,100]、原面积/crowd/ignore语义。固定100自然query、类别、ordinal、RLE与原S0；只改score，不过滤、不NMS、不改mask。并列保持原ordinal，不借GT身份排序。

先用qhat=1逐项精确恢复S0及原COCO12；新预测除score外原字典完全一致，按query/GT双轴核对top100几何不变。保存逐query S0/qhat/新score，报告完整COCO12、十阈值AP及AP百分点差；top10组成与AR10可变并单列。回归误差仅描述，不据DEV选型；不追加qhat-only或搜索评分函数。

- 主seed gain≥1.0 AP百分点：同配方补2602、2603。
- 0<gain<1.0：最多先补2602；仅双seed平均gain≥1.0且各自为正，才补2603，否则STOP本适配。
- 首seed gain≤0：STOP本适配，不调预算、crop、head宽度或评分函数救结果，不额外追加seed。

保留全部seed、失败和恢复记录，不挑最好seed。单seed仅支持当前预算与初始化；正式三seed统计和跨数据集是后续阶段，不自动授权TEST或跨集训练。即使过门槛，也只证明基线可恢复效用，不证明新机制、SOTA或TGRS成功。

负结果仅限制**e0+自身mask、无ROI图像特征**的当前目标/适配/预算，不能否定完整Mask Scoring R-CNN或整个质量评分路线。核心创新仍需另有直接先例差异、公平强对照及独立证据。

直接先例：[Mask Scoring R-CNN原论文](https://openaccess.thecvf.com/content_CVPR_2019/papers/Huang_Mask_Scoring_R-CNN_CVPR_2019_paper.pdf)、[作者代码](https://github.com/zjhuang22/maskscoring_rcnn)。沿用本轮已核入口，不声称复现其ROI特征、完整训练配方或外部结果。

