# HRSID FIT 同一查询的框与掩膜中心偏差

上一轮在 234 个近严格固定配对上观察到，单独真值选平移可恢复 50 个，单独调整厚度恢复零。此次只检验一项预测信号。同一 RF-DETR 查询的检测框中心相对掩膜外接框中心的偏差，能否在留出图组上预测有用的整体平移。冻结 HRSID COCO-only E24 EMA、576 预处理、FP32、原生两槽 top100、候选分数及图像清单。GT-free GPU 导出为每条原生船舶候选增加同查询的原始归一化 cxcywh。原掩膜和候选结果须与已闭合 P192 的预测流去掉新增字段后逐字节一致。 [先前轮廓对照](HRSID_FIT_FIXED_CONTOUR_GEOMETRY_PROTOCOL_20261109.md)

主分析使用相同 109 个 FIT 图组。在前轮 234 个近严格固定配对上加入 37 个原生 0.90 严格配对作为保护层。五张密集图只做留出前缀的压力描述，不参与拟合或门槛。采用 800×800 原图像素中心坐标。原生框中心取 cxcywh 的 cx、cy 乘对应边长。掩膜外接框中心为占据像素的最小和最大坐标之和加一再除二。掩膜质心取前景像素中心的算术均值。两个特征分别为检测框中心减掩膜外接框中心、掩膜质心减掩膜外接框中心，均用 y、x 顺序。

训练标签对原严格配对设为零，其他配对枚举两轴 −2 到 2 的整数平移，排除会裁掉前景的动作。以 IoU 最大选择，平分优先更少异或像素，再以更小 L1 位移和固定字典序裁决。标签只进入各训练折。固定 109 个母景代理前缀，按 SHA-256(UTF-8("HRSID-P196-v1|" + 前缀)) 升序循环分到五折。对每一折，只使用其余四折主样本拟合。压力层所在前缀与该折主层一起留出。比较零位移、训练集平均位移、框偏差及掩膜偏差四种规则。除零位移外，拟合均按图组等总权重。后两种逐轴拟合截距和一个标准化斜率，以 0.25 岭惩罚拟合。标准化只用拟合折。所有实数位移按半值远离零取整、裁到 ±2。平移会使前景出界时回退到零。

固定候选和配对报告 0.75、0.85、0.90 达标数、严格新增与失守、FP/FN、异或像素及执行非零动作数。主门槛要求框偏差相对其余三个对照各净增至少 5 个 0.90 达标配对，且分别在至少 5 个图组净增，图组配对重采样区间下界均大于零，原 37 个严格配对失守不超过 1 个，0.75 达标数不降。用种子 5217 和 4,096 次重采样，109 组均进入分母。只有过门才锁定全候选评价。真值平移上限、留出配对统计和部署效用分开解释。本轮无 AP、模型权重训练、DEV 或官方 TEST。

RF-DETR 已共同预测框和掩膜。Mask DINO 从掩膜生成更好的初始框并联合匹配，BoxInst 用框标注监督掩膜，MaskLab 将语义和方向特征与检测框结合。本轮使用冻结模型的两个现有输出做位移信号对照。普通框掩膜几何和校准不足以单独成立学术创新。 [RF-DETR 原始版本](https://github.com/roboflow/rf-detr/blob/639eac4555ceb89ef399ccda3fde716bbcfe34cb/src/rfdetr/models/postprocess.py) · [Mask DINO](https://openaccess.thecvf.com/content/CVPR2023/html/Li_Mask_DINO_Towards_a_Unified_Transformer-Based_Framework_for_Object_Detection_CVPR_2023_paper.html) · [BoxInst](https://openaccess.thecvf.com/content/CVPR2021/html/Tian_BoxInst_High-Performance_Instance_Segmentation_With_Box_Annotations_CVPR_2021_paper.html) · [MaskLab](https://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_MaskLab_Instance_Segmentation_CVPR_2018_paper.pdf)

## 原导出中止与原生坐标恢复

第一原进程退出 1，无资源中止，在 2 张图写出后因导出检查误把原始归一化 cxcywh 限定在 [0,1] 而停止。冻结 RF-DETR 的 `bbox_reparam` 路径用框增量与参考框计算中心，用指数项计算宽高，没有将原始归一化中心截在画布内。原生后处理在转换为像素端点后才裁剪。新运行沿用锁定的未裁剪中心定义，只要求四个值有限且宽高非负。原运行、源码和记录独立保留。

修正进程原退出 0，114 张图导出 10,572 条原生候选框，其中 364 条中心超出单位画布。去掉新增框字段后，全部候选、分数和掩膜逐字节复现原 P192 流。导出未打开 GT，CPU 评分在确认原导出关闭及完整摘要后执行，也取得原退出 0。GPU 导出工作体 26.49 秒，峰值分配 439.94 MiB，CPU 评分 1.13 秒。八项合成测试通过，另一进程用独立栅格平移、精确分数选取和加权法方程复算，计数和配对区间一致。 [模型坐标公式](https://github.com/roboflow/rf-detr/blob/639eac4555ceb89ef399ccda3fde716bbcfe34cb/src/rfdetr/models/lwdetr.py#L546-L556) · [修正后的导出](code/HRSID_FIT_BOX_SHIFT_20261109/box_export_unclipped.py) · [完整聚合](results/HRSID_FIT_BOX_SHIFT_ACTUAL_SAFE_20261109.json)

## 五折留出主层实测

| 规则 | IoU 0.75 | IoU 0.85 | IoU 0.90 | 严格新增 | 原严格失守 | 异或编辑像素 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 零位移 | 271 | 104 | 37 | 0 | 0 | 0 |
| 图组等权常量 | 254 | 153 | 51 | 29 | 15 | 24490 |
| 掩膜自身几何 | 252 | 153 | 51 | 29 | 15 | 24700 |
| 框掩膜中心差 | 252 | 153 | 51 | 29 | 15 | 24490 |
| 真值选平移上限 | 271 | 200 | 87 | 50 | 0 | 23442 |

原生 37 个严格配对与 234 个近严格配对均保留在分母。框偏差相对零位移严格净增 14 个，配对区间 [+0.761, +9.580] 个百分点。相对等权常量和同容量掩膜对照，每个图组的严格匹配差均为零。框偏差 261 次有效动作中 258 次为 y = −1、x = 0。常量组 261 次有效动作均如此。框偏差比常量多 42 个 FP 和 42 个 FN，二者总编辑数相同。框偏差把 271 个 0.75 达标配对降到 252 个，严格保护损失 15/37，停止此配方，未进入全候选评价。

密集压力层 59 个配对来自五张图和四个前缀组，未用于拟合。框偏差、掩膜偏差和常量的 0.90 恢复均为零，0.75 分别为 35、35、33，原生为 59。真值最优有非零位移的主层 196 对中，框差和最优位移点积为正分别有 122 对，掩膜差为 74 对。在真值可严格恢复的 50 对中分别为 35 和 14。该补充统计只描述原始方向的关联，模型允许负斜率，不能据此宣称额外效用。 [交叉拟合实现](code/HRSID_FIT_BOX_SHIFT_20261109/box_shift_oof.py) · [固定计分](code/HRSID_FIT_BOX_SHIFT_20261109/score_box_shift.py)

## 下一条机制的算子证据

向上一像素的常量动作促使检查训练的掩膜采样。锁定运行时的 torchvision 原生二值掩膜 Resize 与 `nearest` 一致，图像采用双线性，掩膜解码使用 `align_corners=False` 的双线性。实际 CPU 合成探针中，800→576 的单轴 576 个采样位置有 391 个在两种最近邻实现中选择了不同源像素。相对于半像素中心源坐标，原生采样平均偏 −0.679 源像素，中心对齐版本在此比例上的均值约为零。有限精度使原生在 7 个整数边界处与直接整数下取整的理想公式差一格。

同一合成矩形经过 800→576 掩膜缩放与固定双线性还原，原生质心移动 (1.027, 0.523) 原图像素，中心对齐最近邻为 (−0.028, −0.022)，顺序为 y、x。该算子证据和现有模型向上校正的方向相容，但尚未证明训练错位是模型偏移的原因。下一项冻结真实 FIT 图、随机增强和裁剪参数，分别重放原生最近邻及中心对齐最近邻，量化目标偏移、保留行、边界与空掩膜，保持图像与框的相同几何。原生训练在 Resize 或裁剪后可执行水平翻转，可能翻转横向误差的符号，因此两阶段增强须按各自路径实测。锁定的 torchvision 0.25 对 Mask 类型强制使用 NEAREST，候选实现需显式在普通掩膜张量上重采样。nearest-exact 在整数和半像素平分时也可能有相位偏差，不能从 800→576 推及全部训练尺度。RSI4EO 已在 HRSID 作者的检测框裁剪掩膜细化中报告半像素修正，其具体算子未公开，本文将其作为近邻先例。 [PyTorch 插值定义](https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.interpolate.html) · [合成探针](code/HRSID_FIT_BOX_SHIFT_20261109/probe_resize_phase.py) · [RSI4EO 作者报告](https://arxiv.org/html/2610.18845v2#A1.SS3) · [锁定 torchvision 掩膜分发](https://github.com/pytorch/vision/blob/v0.25.0/torchvision/transforms/v2/functional/_geometry.py#L377-L395)

框与掩膜不一致作为推理可靠性信号已有 MITS 先例。Deep-MAC 把框内坐标的掩膜放回预测框，GCBANet 在 SAR 中做框和掩膜的特征对齐。与本轮冻结原图掩膜的整数位移相比，它们的输出坐标和使用阶段不同。项目仅报告该具体信号配方的负结果。 [MITS](https://openaccess.thecvf.com/content/ICCV2023/papers/Xu_Integrating_Boxes_and_Masks_A_Multi-Object_Framework_for_Unified_Visual_ICCV_2023_paper.pdf) · [Deep-MAC](https://openaccess.thecvf.com/content/ICCV2021/html/Birodkar_The_Surprising_Impact_of_Mask-Head_Architecture_on_Novel_Class_Segmentation_ICCV_2021_paper.html) · [GCBANet](https://www.mdpi.com/2072-4292/14/9/2165)
