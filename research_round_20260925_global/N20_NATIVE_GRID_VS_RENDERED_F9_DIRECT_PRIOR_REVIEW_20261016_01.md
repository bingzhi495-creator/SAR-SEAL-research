<!-- encoding: UTF-8 -->
# 原生特征格邻域与渲染像素域F9：直接先例有限补查

2026-10-16。普通研究笔记，供Root另行发布。使用research-lit，sources限定web；只读原论文官方页面、arXiv正文和作者仓库，没有下载PDF、运行模型或读取本项目活动Source/数据/资格链。新笔记路径已按GPU186保护快照（2534230B/SHA256 `c580f273b6aa8d6a6fad1de07e7a5c03a0251448dc6e0d399e01b276ffa86f96`）排除原路径及解析路径。

## 本轮能回答什么

**原生网格邻域和渲染后像素邻域一般不是同一输入，但前者并不因此必然更有效。** LIIF的官方实现已经明确采用原生3×3展开、相对坐标、cell大小及四anchor非线性预测ensemble；I-MedSAM则明确先把特征双线性插值到目标网格，再加坐标编码。因此，“implicit”“连续预测”不自动等于“原生网格取邻域”，也不能把任何原生邻域/offset/ensemble直接列为本项目新原语。

当前N20按Root提供的已知设计，是冻结实例载体上的渲染像素域F9与中心复制的同容量、固定24轮对照；这不是重新核验其活动实现。**第二种子AP仍pending；真实rank、共享cell比例、表达瓶颈及原生邻域收益均未测。** 本轮没有新实验，不改变当前配方、停止规则或选型。

## 七项最相关原始证据

证据标签：**C**＝本轮web实际返回作者代码；**M**＝实际返回原论文HTML方法正文；**V**＝原官方元数据；**I**＝原CVF PDF的索引方法摘录，直接open失败，不能称通读原文。表中机制与本研究效用严格分开。

| 直接先例与发表状态 | 特征域、网格与坐标：实际核到的机制 | 与N20像素F9的实质差别；可否直接作为新颖点 |
| --- | --- | --- |
| **PointRend**，Alexander Kirillov、Yuxin Wu、Kaiming He、Ross Girshick，**CVPR 2020**；V+C，方法描述另有I。[官方论文页](https://openaccess.thecvf.com/content_CVPR_2020/html/Kirillov_PointRend_Image_Segmentation_As_Rendering_CVPR_2020_paper.html)；[作者point_features.py](https://raw.githubusercontent.com/facebookresearch/detectron2/main/projects/PointRend/point_rend/point_features.py) | 直接从CNN特征格，在连续点坐标用双线性采样取得fine-grained特征；结合实例的粗预测，迭代细分并对不确定点预测。代码`point_sample`及`point_sample_fine_grained_features`明确图像/box坐标转换与多feature-map采样接口。 | 直接按坐标读底层特征，与先生成像素F图再取固定3×3不是同一操作顺序；**双线性点采样、fine/coarse融合、自适应点精修本身均非新**。源码接口支持多层，不表示论文每项配置都用了所有层。 |
| **Implicit PointRend**，Bowen Cheng、Omkar Parkhi、Alexander Kirillov，载于 **Pointly-Supervised Instance Segmentation，CVPR 2022**，不是ICCV2021；V+I，作者仓库另C确认类的存在。[官方论文页](https://openaccess.thecvf.com/content/CVPR2022/html/Cheng_Pointly-Supervised_Instance_Segmentation_CVPR_2022_paper.html)；[CVF原PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Cheng_Pointly-Supervised_Instance_Segmentation_CVPR_2022_paper.pdf)；[官方PointSup头](https://raw.githubusercontent.com/facebookresearch/detectron2/main/projects/PointSup/point_sup/mask_head.py) | 原方法§4.3索引摘录明确256维FPN-p2细粒特征、相对box中心的random Fourier位置编码；每个实例产生点函数参数，采用单一点mask loss。作者代码的点监督头继承`ImplicitPointRendMaskHead`；本轮未独立核清其完整当前基类实现。 | 实例条件函数/动态参数与显式坐标是直接先例，不是F9位置无关空间残差。**不能以“query条件+连续点MLP”重命名为新机制**；本轮不能把代码基类未读部分列为独立确认。 |
| **LIIF**，Yinbo Chen、Sifei Liu、Xiaolong Wang，**CVPR 2021**；V+C。[官方论文页](https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Learning_Continuous_Image_Representation_With_Local_Implicit_Image_Function_CVPR_2021_paper.html)；[作者liif.py](https://raw.githubusercontent.com/yinboc/liif/master/models/liif.py) | `feat_unfold`先对encoder原生格做3×3展开；围绕查询点取四个anchor的nearest latent，各自拼接按特征尺寸缩放的相对坐标及可选cell大小，分别MLP预测，最后面积加权ensemble。代码L44–46、65–102可直接定位。 | 是**先保留原生邻域再非线性解码**，而非把双线性混合后的9个目标像素当9份独立原生证据；且`f(加权特征)`一般不等于`加权f(各特征,offset)`。但任务是RGB连续表示/超分，**不是实例mask AP证明**。原生3×3、相对offset、local ensemble和cell conditioning都已有明确先例。 |
| **I-MedSAM**，Xiaobao Wei、Jiajun Cao、Yizhu Jin、Ming Lu、Guangyu Wang、Shanghang Zhang，**ECCV 2024**（arXiv首版2023、所读v3为2024）；V+M。[原正文§3.4–3.5](https://arxiv.org/html/2311.17081v3#S3.S4)；[作者ECCV仓库](https://github.com/ucwxb/I-MedSAM) | 式(5)及正文明确：编码特征**先bilinear Interp至target resolution**，再拼接目标坐标的高频位置编码；coarse INR产生预测和特征，MC-dropout不确定度筛Top-K进入fine INR。 | 是与渲染后特征更接近的近期先例，说明低分辨率插值特征仍可与坐标/非线性精修组合，**不能由插值低rank直接判无效**。它没有证明原生九点优于像素F9；医学提示分割及编码器适配条件不同，不能挪其指标到SAR实例AP。 |
| **AIRM：Leveraging Adaptive Implicit Representation Mapping for Ultra High-Resolution Image Segmentation**，Ziyu Zhao、Xiaoguang Li、Pingping Cai、Canyu Zhang、Song Wang，**arXiv:2407.21256v1，2024预印本**；V+M，未独立核实正式venue。[原元数据](https://arxiv.org/abs/2407.21256)；[正文§3.1、4.1–4.3](https://arxiv.org/html/2407.21256v1#S4) | 图像与粗mask共同编码；式(2)/(5)对查询q周围的特征z_p、相对坐标p−q分别解码后按面积聚合，hypernetwork由全图F生成映射参数；另有affinity监督和多分辨率迭代。 | 局部邻点/offset/ensemble已直接进入mask精修，扩大感受野和全局条件也已有先例。**正文不足以确认p的具体实现分辨率及邻点数，不称其“已核原生九点”**。所报BIG/VOC IoU/mBA不是实例COCO AP；不能直接计算本研究增益或新颖性。 |
| **SAM2Refiner：Towards Fine-grained Interactive Segmentation in Images and Videos**，Yuan Yao、Qiushi Yang、Miaomiao Cui、Liefeng Bo，**ICCV 2025**；V+M。[CVF正式元数据](https://openaccess.thecvf.com/content/ICCV2025/html/Yao_Towards_Fine-grained_Interactive_Segmentation_in_Images_and_Videos_ICCV_2025_paper.html)；[所读arXiv v1 §3.2](https://arxiv.org/html/2502.09660v1#S3.S2) | 对原图和四个放大的图像crop重新编码，global/local cross-attention引入更细图像内容；mask refinement融合对象embedding与encoder多尺度特征，UNet式上采样拼接，再融入低层图像特征。 | 提供**重新看局部原图和独立高分辨率/多层特征**，不只是同一双线性场加密采样；计算与输入证据明显更多。局部/多尺度/低层边界信息本身已有先例。互动/视频设置、额外encoder成本与当前冻结载体不同，不能直接称强于或弱于N20。v1与CVF描述分辨率有变化，本笔记不锁未复核的stride数值。 |
| **ContourFormer: Real-Time Contour-Based End-to-End Instance Segmentation Transformer**，Weiwei Yao、Chen Li、Minjun Xiong、Wenbo Dong、Hao Chen、Xiong Xiao，**arXiv:2501.17688v1，2025预印本**；V+M+C仓库身份，未确认正式venue。[原方法§3.1–3.2](https://arxiv.org/html/2501.17688v1#S3)；[作者仓库](https://github.com/talebolano/Contourformer) | 多尺度backbone/encoder；预测子轮廓的box决定deformable cross-attention采样范围，逐层更新query；另对轮廓坐标offset分布做残差精修。 | 是**轮廓条件自适应取特征与坐标修正**，不是固定9个邻近像素分类，也不是独立模型无关后处理。说明“用offset追边界”早有近期直接实例先例；换成该机制不会自动产生本项目创新。标题中的实时不构成与N20资格采集耗时的公平FPS对照。 |

作者代码链接为本次访问到的官方分支页面，并非已冻结、已复现代码版本；未下载或执行。论文报告的改善仅用于理解机制适用条件，本表不搬运数值排名。RefineMask/Mask Transfiner等先前已读的多阶段细粒特征/边界稀疏精修路线不在此重复展开，也不能因本表省略便认为不存在。

## 关于“单cell rank≤4”的准确读法

这是**条件代数推断，不是文献或本项目实测结果**：九点的raw sampled F9都在同一层特征的同一双线性cell内、共享四corner，且未因边界/padding处理改变该采样关系；在尚未加入归一化、激活或其他非线性处理时，令九个输出位置组成R，则R=A Z，A至多四个有效列，因此rank(R)≤4。这个上界不能直接套到LayerNorm等非线性归一化后的张量。它不要求rank等于4，也不表示九点等于中心一点：9次混合可能提供中心单次混合无法识别的方向/梯度；还可能跨cell取得更多corner。不能把条件上界外推为真实全DEV或所有F9的实际rank、信息增益为零、AP上界或当前瓶颈。

原生3×3可能访问更多源网格节点及更大的物理感受野；若坐标/非线性/层级/crop也改变，收益不能统归“原生采样”。LIIF的ensemble是**解码后**加权，不是凭双线性插值生成了新图像证据。更高维位置编码可提高解码函数表达能力，但不证明恢复了源特征不存在的真实纹理。以上原论文均未给出与本项目同teacher/query/训练/预算下“原生网格对渲染F9”的直接公平比较。

## 只保留为候选设计依据的最小结论

1. 后续若解释N20，先限定当前实测相对表现；rank/cell覆盖/可辨识性需另作描述性核验。当前缓存不自动意味着可读取原生grid，不能现在就宣称机制成立或提前选新模型。
2. 若未来获准比较，至少把**中心复制、现像素F9、原生邻域**的容量/初始化/训练和坐标约定固定；另匹配物理感受野，区分“跨更多源节点”与“窗口变大”。坐标/ensemble、多尺度或额外图像分支分别隔离，计入全部查询、decoder和输入采集成本。此处是控制要求，不是已安排或执行实验。
3. 原生邻域、相对坐标、Fourier编码、local ensemble、偏移采样、多尺度边界特征都不能单独算新原语。贡献候选只能是针对清楚误差机制、与这些直接先例有具体差异的整合，并由公平强对照、重复种子、适用数据范围证实。不能重命名已失败的N16 utility学习，不能借本次查新替代N20第二AP或TGRS证据。

## 检索边界

本次有限检索覆盖PointRend/Implicit PointRend/LIIF，以及2024–2026与implicit mask、local features、refinement、coordinate/offset相关的原始来源；保留七项影响本问题的证据即停止。2026结果中出现跨任务/非对应机制条目，未纳入；**没有完成2026或最近六个月的穷尽查新，也不宣称“世界最新/无人做过”**。CVF直接PDF open多次返回错误，涉及I级内容已降级标注；arXiv HTML及列明官方代码实际返回，不能混称全部全文已核。本轮跨模型正式review未执行，实验资格与结论仍由Root实际终态流程单独处理。STOP。
