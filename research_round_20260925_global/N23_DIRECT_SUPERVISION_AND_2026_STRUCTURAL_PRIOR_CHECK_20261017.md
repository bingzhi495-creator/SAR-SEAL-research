# N23监督先例与2026结构方向核查

访问日期：2026-10-17。本文为自写定向综述，只使用原论文、出版社内容和作者仓库；未复制全文，未独立复现外部结果。不预选下一配方，不以其他协议的数字判断本项目胜负。

## N23首先是适配修复控制

PointRend作者[配置](https://raw.githubusercontent.com/facebookresearch/detectron2/main/projects/PointRend/point_rend/config.py)第27–34行设oversample=3、importance ratio=0.75。[采样器](https://raw.githubusercontent.com/facebookresearch/detectron2/main/projects/PointRend/point_rend/point_features.py)的get_uncertain_point_coords_with_randomness从随机候选中取不确定性排名点，再补均匀随机点；范围为每个RoI，不是全图。75%是排名比例，不等于固定abs(L)<=2的闭带。

作者[点头及损失](https://raw.githubusercontent.com/facebookresearch/detectron2/main/projects/PointRend/point_rend/point_head.py)中，StandardPointHead.forward返回绝对point logits，coarse预测作为输入；roi_mask_point_loss对全部非ignore采样点计算BCE，只有标签-1被忽略。未见本项目L+1_Omega*u这种输出门控；随机点不会因为粗预测自信而自动失去点头梯度。

RefineMask作者[BARCrossEntropyLoss.forward](https://raw.githubusercontent.com/zhanggang001/RefineMask/master/mmdet/models/losses/cross_entropy_loss.py)第284–319行：idx<=start_stage时全格BCE；后续在前级预测边界与前级GT边界的并集监督，除以活跃边界数。该分母是批内区域计数，不能等同本项目逐GT归一化；区域也不只依赖预测置信。

因此，扩大合法点监督、活跃区域归一化已有明确先例。N23若提升，首先支持固定载体及经验池下修复硬门控限制的效用；不能仅据此称SAR新机制、证明头是主要瓶颈或达到投稿标准。标准PointRend输出绝对logit，本项目学习加性残差，两者完整系统仍有差异；本轮控制不冒充作者方法复现。 N23四头推理仍固定在相同Omega；域外补充监督只能经共享参数影响Omega内预测，本轮不是全C部署的检验，负结果也不能据此否定扩大实际修复域。

## 最新结构先例：DACF-Net

[出版社原文](https://www.tandfonline.com/doi/full/10.1080/17538947.2026.2696657)，2026-09-08上线，是真SAR船舶实例分割。HRSID按13:7划分，测试369近岸/1593离岸图；PSeg-SSDD按8:2，测试46近岸/186离岸图。论文采用不同输入尺寸与端到端训练，不能与内部DEV直接比较。

正文基础DACF-Net报告HRSID/PSeg-SSDD分割AP为60.2/64.2；摘要另报DACF-Net‡为63.3/65.7。两者须区分配置，本轮未完整核‡变体对应表，不把最高数值替代基础行。文中还注明部分比较结果摘自其他论文，不能统一称作者独立复现。

其MSFFM含多分支大核/条带深度卷积及Hadamard门控残差，CLFIM用高层语义调制低层特征，另有几何感知提案。因此，“方向融合”“高层引导低层”“选择性跨层传输”已有直接相近先例。UNet++需要具体比较采样坐标、门控位置、嵌套路径及信息来源，不能只靠名称区分。

论文声明[作者仓库](https://github.com/yang199809/DACF-Net)将开源；本轮通过[作者仓库根目录API](https://api.github.com/repos/yang199809/DACF-Net/contents)核到仅.gitignore、LICENSE、README.md三文件，没有可运行模型/训练代码。开源声明与当前可复现材料应分开记录。

补充：[DiffSARShipInst原论文](https://www.sciencedirect.com/science/article/pii/S0924271625000887)（2025）已包含CARAFE/ACmix增强FPN及多非对称卷积、动态实例mask表示；[作者代码](https://github.com/XiaowoXuSAR/DiffSARShipInst)有训练入口。SSDD原文明确8:2，但本轮未核完整划分ID及HRSID注册映射。其方向特征与内容感知重组同样是UNet++结构定位必须比较的先例。

## 基础模型路线的指标边界

[Prompting Foundation Models原文](https://arxiv.org/html/2604.17920v1)（2026）采用SAR框监督的YOLOv11检测器提示现成SAM2，产生逐框实例mask。SSDD测试232图，46近岸/186离岸；分割报告mean IoU=0.637、Dice等，COCO mAP描述的是检测器，不能记为mask AP。SAM2无SAR像素标签微调，不等于整条流水线零SAR监督。

原文第59行声明[hybrivision代码](https://github.com/IslamAlam/hybrivision)公开；本轮仓库访问失败，未核实际内容。该路线可提供另一范式的标注及成本比较，不直接证明UNet++内部传输结构的优劣。

## GLDet沿用已有原PDF证据

[GLDet出版社入口](https://ieeexplore.ieee.org/document/10962196/)及[DOI](https://doi.org/10.1109/TGRS.2025.3559551)身份已核，本轮原站返回418或空页，未重新读取全文。Root已有原PDF审阅，应以[既有GLDet原文审阅与迁移分析](GLDET_PRIMARY_PAPER_REVIEW_AND_SEGMENTATION_TRANSFER_20261014.md)保留其任务、模块及实验证据。本轮访问受限不覆盖既有判断，更不能仅因Detector标题将其判为没有实例分割实验。

后续只在真实误差支持后比较一条具体结构假设。技术整合可以是贡献候选，但仍需直接先例差异、强控制、重复、正式测试、跨集及成本证据。
