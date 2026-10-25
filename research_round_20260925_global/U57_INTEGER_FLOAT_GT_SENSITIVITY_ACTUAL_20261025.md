# 冻结预测的整数／本地浮点GT敏感性：实际结果

2026-10-25。**新增事后诊断已真实完成并通过Root接收。固定R相对原生模型的优势在两种GT之间反转，原正收益必须限定于原整数协议。** 当前FULL尚无端点AP，本结果不构成新方法成功或TGRS依据。

## 同一预测、两个明确标签版本

内部DEV187图/380实例、每方法18700原候选、原S0/类别/顺序均不变，使用同一官方COCO mask评价。native/R/SAM基础默认版三套冻结预测不重新训练、不调分数或参数、不删样本。整数native/R/SAM完整结果均与此前合格结果精确相同。

浮点版本来自已固定的本地候选COCO训练成员。按已声明唯一规则（逐坐标int截断、顶点极值差bbox、鞋带area）严格唯一对应380/380，再对整数DEV完整深拷贝，仅恢复源segmentation/bbox/area三个字段；图像、类别、annotation身份/顺序及其他字段保留。它不认证官方整包、历史converter、许可或物理船体真值。完整标注未发布。

| 方法 | 整数GT AP | 本地候选浮点GT AP | 浮点−整数 | 整数AP75 | 浮点AP75 |
|---|---:|---:|---:|---:|---:|
| native Mask2Former/R50 E24 | 67.071057 | 68.443523 | +1.372466 | 83.842665 | 83.547432 |
| 固定R | 68.099385 | 67.219522 | −0.879862 | 85.213468 | 82.384393 |
| SAMRefiner基础默认，ViT-H FP32/5轮 | 42.770654 | 46.260621 | +3.489967 | 27.947056 | 36.784847 |

R−native：整数+1.028328 AP点，浮点−1.224000点，差异方向反转。SAM−native：整数−24.300403点，浮点−22.182902点，仍明显较低。SAM基础默认版不代表++或作者最强配置，额外SAM预训练/计算亦不同。N26及当前FULL/PLAIN尚未得到本次浮点AP，不拼接不同GT排名。

## GT栅格实际变化

官方annToRLE与mask解码在原网格逐一比较全部380实例：366个变化、14个相同，两版空GT均0。逐实例GT IoU均值0.902626，中位数0.906488；pooled IoU为0.946000。XOR像素合计21869、中位数40.5。整数/浮点GT像素面积总和394039/394056，仅差17；面积比中位数1。

面积总量近似相同并不说明位置和轮廓相同，也不能把全部AP差异唯一归因某一种像素机制。此次polygon、bbox及连续area共同恢复，可能涉及IoU匹配和面积分组变化；没有完成唯一因果分解或物理标注质量判断。两版均非空排除了“此前两个全候选零IoU仅因GT为空”的具体解释，其他原因未确定。

## 实际执行与主张范围

本机仅标准库派生实际退出0、380唯一映射、源输入前后不变；未解码图像或mask、未算AP。随后DGX独立CPU作业原工具/SSH/worker均0；22个worker来源、7个parent来源后核通过，同boot出生身份及所属组为空。六次COCO评价和全部380 GT栅格统计完成；parent约8.498秒、RSS峰值148832256字节、可用RAM最低78159372288字节，违规为空。共享环境诊断不当作FPS或整机独占基准，也不认证全部二进制来源。

这是看到SAM基础默认负结果后新增的**posthoc附加诊断**，不是原启动预注册。原[首E50 FULL/PLAIN标注敏感性政策](U57_PROSPECTIVE_ANNOTATION_SENSITIVITY_POLICY_20261025.md)不变，尚未由本次结果兑现。主训练、原整数主评价、预算和停止规则均不改，不选更有利GT替代原结果。

当前可以说R在原整数协议下有局部收益；不能说其收益跨标注版本稳定，也不能据此断言全部方法无效。接下来首先完成当前固定E50端点，两GT同时评价FULL/PLAIN及相应共同参考，再依据既定效用规则决定重复与最少机制控制。完整外部方法、容量/计算、重复种子和跨集仍缺。

[完整COCO12、逐IoU、差值及GT栅格安全结果](results/U57_GT_VERSION_SENSITIVITY_ACTUAL_20261025.json)；[实际执行源码及范围](code/SAMRefiner_FULL187_GT_SENSITIVITY_DRAFT_20261025_01/README.md)。

## 一项撤回的代码疑问

执行前曾怀疑COCOeval就地修改segmentation会污染后续polygon变化计数。继续追到固定helper的第629行deepcopy及第631行COCOeval调用后，确认修改的是内部副本。该问题假设撤回，未修改候选、未生成v02或新增模拟回归，未把怀疑写成已发生的科学失败。


Limited independent result-to-claim review: PARTIAL / MEDIUM. Actual backend GPT-5.6-sol xhigh, not the unavailable specified GPT-5.4 Codex MCP. It independently checked the saved arithmetic and claim scope; it did not SSH, rerun science or read GT/prediction payloads. Root retains responsibility for actual execution qualification. [Safe review](results/U57_GT_SENSITIVITY_RESULT_CLAIM_REVIEW_20261025.json).
