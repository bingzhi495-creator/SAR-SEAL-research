# SSDD 全候选的局部 logits 选择性编辑

114 图的固定配对诊断中，局部 logits 优于法线与中心点。在部署中，匹配身份不可用，9,081 个未匹配候选必须纳入风险检查。本轮只检验一个问题。用方向证据选择是否进行固定配对编辑，能否在相同修改预算下优于直接按原分数和原生局部不确定度选择。 [上一轮局部证据](RFDETR_FIT_LOCAL_SIGNED_EVIDENCE_PROTOCOL_20261109.md)

保持原 SSDD E24 EMA、候选和分数。从原 484 图的 479 个编码组用锁定哈希划分点读出拟合、开关拟合及阈值校准，按 3/5、1/5、1/5 分配。三组和 111 张历史内部留出互斥。114 张此前校准角色只用于发现上一轮证据，本轮不读取。111 图曾用于内部实验与真值诊断，且已由载体训练覆盖，故此处只作为预先锁定的新读出内部筛查。

沿用上一轮 38 维纯 logits 输入、1,281 参数头和 96 次全批 AdamW。每候选由原掩膜支持给出最多每侧 128 个点。原生局部活跃并至少有四内四外支持才可产生动作。选分最高的四个前景和四个背景位置，交换其标签以保持预测面积。没有足够支持或开关拒绝时原掩膜逐位保留。点标签来自拟合组原匹配的真值。阈值和开关两组读取所有候选，并把原匹配整数裕量为正且不新增严格失守的动作标记为可用。未匹配作为此开关标签的保守负例，另统计其真实编辑数，不预断其 COCO 影响。

两个候选开关都用同宽七维、线性逻辑回归、同样 96 次更新。共享原分数、位次、掩膜面积、两侧支持数。主要开关额外看所选两侧的点读出均值，控制看同位置原始绝对 logit 的两侧均值。阈值校准取通过以下条件的最大完整分数前缀，至少 30 个动作、15 个编码组，动作可用比例至少 60%，未匹配不超过 20%，匹配整数裕量总和为正且零原严格失守。等分动作一起进退。无合格阈值则全部保留原掩膜。该规则属于经验筛查。

固定 111 图及原生 COCO 原分数，原生输出须精确复现 9,082 候选和已核验指标。对比原生、全量强制编辑、主要选择性编辑、同每图编辑数的原置信度选择、同每图编辑数的等容量控制开关。每个选中候选都应用同一固定四加四删方案。评价时先用纯预测生成所有动作和固定分数输出，再用原匹配核算 FP/FN、严格达标与未匹配编辑负担。主开关相对原生 mask AP 至少 +0.5 点，相对两种等预算对照各至少 +0.2 点，AP75 不降，AP90 至少 +1 点，原严格失守不超过 3 个且严格净增，才推进重复和跨集。未到门槛停止本固定配方。无 DEV187、正式 TEST 或新载体训练。

PointRend、Mask Transfiner 已选择待精修位置。Learn then Test 与 Conformal Risk Control 已研究风险校准，本轮样本与有符号 AP 收益条件不支持对应理论保证。 [PointRend](https://openaccess.thecvf.com/content_CVPR_2020/html/Kirillov_PointRend_Image_Segmentation_As_Rendering_CVPR_2020_paper.html) · [Mask Transfiner](https://openaccess.thecvf.com/content/CVPR2022/html/Ke_Mask_Transfiner_for_High-Quality_Instance_Segmentation_CVPR_2022_paper.html) · [Learn then Test](https://arxiv.org/abs/2110.01052) · [Conformal Risk Control](https://openreview.net/forum?id=33XGfHLtZg)

[实验入口](code/RFDETR_FIT_SELECTIVE_LOGIT_ACTION_20261109/selective_action.py) · [编辑与经验阈值](code/RFDETR_FIT_SELECTIVE_LOGIT_ACTION_20261109/selective_math.py)

