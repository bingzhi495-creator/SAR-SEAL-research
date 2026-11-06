# DiffSARShipInst 原始来源核查：论文初值仍未知，训练与推理 proposal 数不同

核查日期：2026-11-07。本次仅联网读取出版方公开内容、ISPRS 学会信息、作者公开仓库及相关原始技术披露；未读取本项目运行 SOURCE、权重或执行载荷，未 SSH、导入科学运行库、加载 checkpoint、改变训练或计算 AP。源码证据固定于作者 commit `bdc3362372213bc8c3a1f6d5b7b74dc224d1b871`；公开代码默认配置不能代替作者论文实际运行记录。

**结论：论文是否使用 ImageNet/COCO 或其他预训练、具体初始 checkpoint 及加载覆盖率，本次仍未证实。** `MODEL.WEIGHTS` 留空只说明所检查的公开 YAML 没有给出该文件，不能据此认定论文从随机初值训练。保持当前已锁定训练配方；后续完整参考需继续披露论文初值未知及与本项目预训练教师的成本差异。

## 文献与论文中可核的范围

正式题名为 *DiffSARShipInst: Diffusion model for ship instance segmentation from synthetic aperture radar imagery*，刊于 *ISPRS Journal of Photogrammetry and Remote Sensing*，223 卷，2025 年 5 月，440–455 页，DOI `10.1016/j.isprsjprs.2025.02.030`。ISPRS 学会当期目录列出 Xiaowo Xu、Xiaoling Zhang、Shunjun Wei、Jun Shi、Wensi Zhang、Tianwen Zhang、Xu Zhan、Yanqin Xu、Tianjiao Zeng 九位作者。[出版方页面](https://www.sciencedirect.com/science/article/abs/pii/S0924271625000887)、[ISPRS 当期目录](https://www.isprs.org/news/newsletter/2025-02/index.html)。

出版方可检索的 Data 节摘要明确 SSDD 共1160图、训练/测试比例8:2；摘要报告 offshore SSDD/HRSID mask AP 为70.6/70.9，inshore 为56.2/42.6。这些是场景分组指标，不能当作两个数据集各一个整体 AP，更不能与本项目 DEV187 直接相减。**全文直接获取返回403，未取得论文实现段全文**，因此论文实际预训练、总预算、精确划分文件、测试阈值与重复种子均保留未知。[论文 Data / Quantitative results 可检索章节摘要](https://www.sciencedirect.com/science/article/abs/pii/S0924271625000887)。

另一个明确标记模型为 DiffSARShipInst、含部分共同作者/机构的原始技术披露 CN119741496A，在 Description 的数据准备步骤分别写 SSDD 8:2、HRSID 13:7，并给出 deformable ResNet-101、AdamW 与 COCO maskAP。它可作相关技术披露的旁证，**不是本次读到的论文实现段**，也没有闭合精确文件身份、论文初值或训练步数。[CN119741496A 原始技术披露](https://patents.google.com/patent/CN119741496A/en)。

## 作者公开默认配方：已证实与未知分开

| 项目 | 固定代码中可核内容 | 仍未证明的事项 |
|---|---|---|
| 初始化入口 | SSDD/HRSID YAML 的 `MODEL.WEIGHTS` 均为空；README 测试命令要求自行指定模型路径；DefaultTrainer 支持加载指定权重或 resume 已有 checkpoint | 论文到底加载过什么、是否通过 CLI 覆盖、加载覆盖率与训练谱系 |
| 结构与输入 | deformable ResNet-101 / SCJE-FPN，6个头；SSDD输入/裁剪512，HRSID为1024；RGB、polygon mask | 这些公开默认是否逐项等同最终论文运行 |
| 训练默认 | 两数据集 YAML 均 batch1、AdamW、base LR3.5e-5、wd1e-4、18560更新，衰减14848/17632、gamma0.1；warmup10/factor1；seed40244023；EMA与AMP关闭 | 论文硬件/实际曝光、实际收敛、总GPU小时；不能将未知成本记为0 |
| 训练/评价调用 | train_net 使用原生 Trainer，走 COCOEvaluator；默认每928更新评价与落盘 | 论文是否选 best、端点还是多次运行均值 |
| 数据身份 | YAML 名称仍为 `coco_2017_train/val`；builtin.py 对应通用COCO路径 | 实际 SAR JSON、影像映射、原始划分清单与哈希；名字不能证明用COCO图训练，也不能证明绑定了何种SAR划分 |
| 权重公开情况 | 本次固定完整 Git 树没有常见 `.pth/.pt/.pkl/.ckpt/.safetensors` 权重文件或PDF；查询时 releases API 返回空列表，README未给模型下载链接 | 不能据此证明作者从未在其他位置或私下提供权重 |

源码定位：[SSDD train 配置 L19–33 输入、L55 骨架、L138 R101、L321 权重、L330–361 求解器](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/configs/DiffSARShipInst-SSDD-train.yaml#L321)、[HRSID train 配置](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/configs/DiffSARShipInst-HRSID-train.yaml#L19)、[Trainer 加载语义 L398–416](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/detectron2/engine/defaults.py#L398)、[作者 README](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/README.md)、[数据别名 L43–44](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/detectron2/data/datasets/builtin.py#L43)、[固定 Git 树](https://api.github.com/repos/XiaowoXuSAR/DiffSARShipInst/git/trees/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871?recursive=1)、[releases 查询](https://api.github.com/repos/XiaowoXuSAR/DiffSARShipInst/releases)。

## 必须纠正的三个不同数量

| 数量 | 角色与源码证据 |
|---|---|
| 训练 `NUM_PROPOSALS=500` | 训练阶段的扩散框/目标填充数量；SSDD与HRSID train.yaml L85相同。[SSDD train L85](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/configs/DiffSARShipInst-SSDD-train.yaml#L85)。 |
| 作者测试 `NUM_PROPOSALS=1000` | 两个 test.yaml L85均为1000；与各自 train.yaml 逐行比较，唯一不同就是该值。它决定模型自身生成/处理的候选数量。[SSDD test L85](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/configs/DiffSARShipInst-SSDD-test.yaml#L85)、[HRSID test L85](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/configs/DiffSARShipInst-HRSID-test.yaml#L85)。 |
| 最终 COCO `maxDets=100` | COCOEvaluator默认最大评价检测数为[1,10,100]；输出AP/AP50/AP75等。不是给模型的100个扩散框。[评价器 L109–115](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/detectron2/evaluation/coco_evaluation.py#L109)、[原 Trainer L88–100](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/train_net.py#L88)。 |

**只读源码支持500训练状态用于1000候选推理，不要求另训1000行的学习型query embedding。** detector L245将数量保存为普通属性，L390/L400按该数量新采样 `[batch,N,4]` 高斯框；L375传给 DynamicHead 的 `init_features=None`。head L264/L388从输入框取得数量，L422–479按运行时 `nr_boxes` 做 RoI特征、attention与输出reshape；头部线性层维度依据 hidden_dim、类别数、动态卷积通道等定义，没有按proposal数量建立的学习型表。loss.py 中也未见按 `NUM_PROPOSALS` 构造的参数或缓冲。[detector L245](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/diffsarshipinst/detector.py#L245)、[采样 L390](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/diffsarshipinst/detector.py#L390)、[调用 L375](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/diffsarshipinst/detector.py#L375)、[动态头 L388](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/diffsarshipinst/head.py#L388)、[loss 构造](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/diffsarshipinst/loss.py#L125)。这是静态支持，**实际checkpoint严格加载及1000候选真实前向尚未执行，本报告不授予运行通过资格**。

还有计算成本细节：detector.inference L744按 `self.num_proposals` 取 top-k，L760–781在NMS后为保留实例生成mask；该路径没有用 `TEST.DETECTIONS_PER_IMAGE` 提前截成100。故不应为省显存或“对齐100查询”而在作者模型内部先截100，再称忠实作者测试。[作者推理 L733–781](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/diffsarshipinst/detector.py#L733)。detector L255另有1000个 diffusion schedule timesteps，这又是不同概念；公开测试的 `SAMPLE_STEP=1`，不等于做1000次推理去噪。[schedule L255](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/diffsarshipinst/detector.py#L255)。

## 对当前研究的操作建议与证据边界

1. 保持已锁定训练500、固定FIT741/50轮37050更新等配方，不因本次查证重写原轨迹。18560在928张训练图、batch1且每图一次访问的理想换算下为20轮；在741张FIT上约25.05轮。两者都只是曝光换算，不证明作者实际顺序或收敛。
2. 在读取任何最终AP之前，另行预锁完整外部评价：忠实作者测试1000候选、SAMPLE_STEP1、原NMS/后处理和COCO maxDets100；端到端方法保留自身候选生成，内部固定100候选精修另列。不要在500/1000等多个配置结果中事后挑最高AP。实际加载/前向需按Root资源与执行资格完成，不能由静态检查代替。
3. 外部成本按1000候选原生整套推理计，预训练/教师缓存/上游项目训练分列。当前公开构造初值参考只能称固定数据与曝光下的公开代码适配；论文初值未知时，不能将其低分解释为击败论文强方法。

本报告是自有查证与建议，保存的只有摘要、定位和链接，不包含第三方全文、原始数据或权重；不修改主实验、旧失败或现有科学结论。
