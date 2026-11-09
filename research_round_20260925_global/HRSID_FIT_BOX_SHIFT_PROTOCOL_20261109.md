# HRSID FIT 同一查询的框与掩膜中心偏差

上一轮在 234 个近严格固定配对上观察到，单独真值选平移可恢复 50 个，单独调整厚度恢复零。此次只检验一项预测信号。同一 RF-DETR 查询的检测框中心相对掩膜外接框中心的偏差，能否在留出图组上预测有用的整体平移。冻结 HRSID COCO-only E24 EMA、576 预处理、FP32、原生两槽 top100、候选分数及图像清单。GT-free GPU 导出为每条原生船舶候选增加同查询的原始归一化 cxcywh。原掩膜和候选结果须与已闭合 P192 的预测流去掉新增字段后逐字节一致。 [先前轮廓对照](HRSID_FIT_FIXED_CONTOUR_GEOMETRY_PROTOCOL_20261109.md)

主分析使用相同 109 个 FIT 图组。在前轮 234 个近严格固定配对上加入 37 个原生 0.90 严格配对作为保护层。五张密集图只做留出前缀的压力描述，不参与拟合或门槛。采用 800×800 原图像素中心坐标。原生框中心取 cxcywh 的 cx、cy 乘对应边长。掩膜外接框中心为占据像素的最小和最大坐标之和加一再除二。掩膜质心取前景像素中心的算术均值。两个特征分别为检测框中心减掩膜外接框中心、掩膜质心减掩膜外接框中心，均用 y、x 顺序。

训练标签对原严格配对设为零，其他配对枚举两轴 −2 到 2 的整数平移，排除会裁掉前景的动作。以 IoU 最大选择，平分优先更少异或像素，再以更小 L1 位移和固定字典序裁决。标签只进入各训练折。固定 109 个母景代理前缀，按 SHA-256(UTF-8("HRSID-P196-v1|" + 前缀)) 升序循环分到五折。对每一折，只使用其余四折主样本拟合。压力层所在前缀与该折主层一起留出。比较零位移、训练集平均位移、框偏差及掩膜偏差四种规则。除零位移外，拟合均按图组等总权重。后两种逐轴拟合截距和一个标准化斜率，以 0.25 岭惩罚拟合。标准化只用拟合折。所有实数位移按半值远离零取整、裁到 ±2。平移会使前景出界时回退到零。

固定候选和配对报告 0.75、0.85、0.90 达标数、严格新增与失守、FP/FN、异或像素及执行非零动作数。主门槛要求框偏差相对其余三个对照各净增至少 5 个 0.90 达标配对，且分别在至少 5 个图组净增，图组配对重采样区间下界均大于零，原 37 个严格配对失守不超过 1 个，0.75 达标数不降。用种子 5217 和 4,096 次重采样，109 组均进入分母。只有过门才锁定全候选评价。真值平移上限、留出配对统计和部署效用分开解释。本轮无 AP、模型权重训练、DEV 或官方 TEST。

RF-DETR 已共同预测框和掩膜。Mask DINO 从掩膜生成更好的初始框并联合匹配，BoxInst 用框标注监督掩膜，MaskLab 将语义和方向特征与检测框结合。本轮使用冻结模型的两个现有输出做位移信号对照。普通框掩膜几何和校准不足以单独成立学术创新。 [RF-DETR 原始版本](https://github.com/roboflow/rf-detr/blob/639eac4555ceb89ef399ccda3fde716bbcfe34cb/src/rfdetr/models/postprocess.py) · [Mask DINO](https://openaccess.thecvf.com/content/CVPR2023/html/Li_Mask_DINO_Towards_a_Unified_Transformer-Based_Framework_for_Object_Detection_CVPR_2023_paper.html) · [BoxInst](https://openaccess.thecvf.com/content/CVPR2021/html/Tian_BoxInst_High-Performance_Instance_Segmentation_With_Box_Annotations_CVPR_2021_paper.html) · [MaskLab](https://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_MaskLab_Instance_Segmentation_CVPR_2018_paper.pdf)
