# 面向公共 SAR 船舶实例分割的基础模型与多模态路线：原文、源码及数据可行性核查

核查日期：2026-11-07。定位：研究决策资料，非实验结果。本文由独立子任务形成；未访问 SSH、运维配置、活动 SOURCE、项目数据或权重，未开展训练、推理或科学载荷执行。仅查阅联网一手论文、作者仓库、模型卡和数据发布页，并保存本原创审阅文档。仓库 commit 是核查时固定的源码快照，不代表本机已经安装、权重已获批或方法已经复现。

## 1. 结论先行

1. **当前最可执行的是 SAR 单图推理的强实例模型适配，辅以经过必要性检验的语义/空间先验。** SAM 3 的类别文本提示原生输出实例框、掩膜与分数；DINOv3 可提供密集特征。它们值得进入基线/教师候选，但尚无本项目可比训练结果，不能提前说能涨点。DINOv3 SAT493M 是 Maxar 光学 RGB 预训练，并非 SAR 预训练。
2. **真实光学—SAR 融合不是在 SSDD/HRSID 上即插即用的路线。** 当前公共灰度裁片不自带对应光学、双极化、相位或可靠逐图传感器元数据。灰度复制三通道、幅度/梯度/频谱三支路属于输入适配或同图多表征；不能称作独立传感器模态。
3. **DOFA-CLIP 有必须先处理的预训练重叠风险。** 原论文附录明确 SAR-ship-caption 来源包括 HRSID、SSDD 和 AIR-SARShip-2.0。没有逐样本预训练清单与下游测试排除证据，不能将它作为“未见过 SSDD/HRSID 的通用模型”证明泛化。这里确认的是来源重叠风险，未确认具体测试图全部被使用。
4. **已找到的新船舶跨模态数据均不能直接替代当前实例掩膜基准。** MOS-Ship 是真实但异步的光学/SAR 定向检测；OS-Ship-1K 是 CycleGAN 合成的伪配对检测；HOSS ReID 是同船身份检索，非逐像素配准分割。三者的检测/检索 mAP 都不能放入本项目 COCO mask AP 表。
5. **普通特征融合、视觉教师蒸馏、生成缺失模态均已有直接先例。** DOFA-CLIP 已有多个视觉教师及模态相关蒸馏；AnySat 已有空间对齐多模态表征；TerraMind 的 TiM 已生成缺失模态 token 后再融合。因此创新候选必须落在可诊断的 SAR 错误机制、明确设计差异与公平实验证据上。

上述结论分别由下文固定源码、原始论文和发布页支持。本轮覆盖 8 类基础模型及 3 个专门船舶数据项目；属于有明确筛选标准的重点核查，不宣称穷尽全球全部最新方法。

## 2. 四类路线必须分别表述

| 路线 | 真实信息来源及可用条件 | 当前能否直接落地 | 论文应如何描述 |
| --- | --- | --- | --- |
| 真双传感器融合 | 同区域实际光学与 SAR，明确空间配准、时间差和可见性；训练/测试均有合法输入 | SSDD/HRSID 现有裁片没有这种配对；MOS-Ship 可研究定向检测，但缺实例掩膜 | optical–SAR sensor fusion；报告错配、时间差和缺失模态鲁棒性 |
| 图像 + 类文本语义先验 | 灰度 SAR 与固定的类提示，如 ship；不要求人工逐图 caption | SAM 3 原生接口可行；DOFA-CLIP 要额外审计预训练污染、输入编码和许可 | vision–language prior；固定文本可能只是闭集类别先验，需验证其必要性 |
| 同图多表征 | 同一灰度图的强度、梯度、频域、局部统计、基础模型特征 | 可行，无新增采集输入；只增加确定性变换或可学习分支 | multi-representation / multi-view，不能声称多传感器或恢复实测相位 |
| train-only optical 知识迁移 | 光学预训练教师或合法训练侧光学数据；部署仅用 SAR | 可行，但无真配对时不能使用同船逐像素对应监督 | cross-domain knowledge transfer / distillation；披露额外预训练与数据、教师算力 |

生成光学由 SAR 条件推断产生，属于模型先验或合成表征；不会凭空提供一份独立观测。作为工程方案可以检验，但其物理真实性、独立信息增益及新颖性须单独论证。

## 3. 基础模型清单：输入、原始任务、权重及当前适用性

所有“公开入口”表示核实到作者发布位置；本轮没有下载或校验权重。除明确 gated 项外，未运行身份认证/下载流程；不可把网页可见等同于本机现已可执行。代码、权重和底层数据许可分别记录。

| 模型与原始依据 | 原生模态、空间尺度及输出 | 代码 / 权重发布与许可 | 对当前灰度 SAR 实例分割的判断 |
| --- | --- | --- | --- |
| **DINOv3**，Siméoni 等，arXiv:2508.10104，2025；[原文](https://arxiv.org/abs/2508.10104) | 自监督视觉特征；SAT493M 为 493M 张 Maxar RGB 正射影像，512×512，0.6 m；ViT patch16；非原生实例头 | [作者仓库](https://github.com/facebookresearch/dinov3)；[SAT493M ViT-L 模型卡](https://huggingface.co/facebook/dinov3-vitl16-pretrain-sat493m)。DINOv3 自定义许可；模型页明确需同意共享信息，**gated**。权重未获批/下载核验 | 优先级高的预训练特征/教师候选。灰度复制 RGB 只能称输入适配；须对比普通自然图像权重与 SAT 权重，不能预设卫星光学对 SAR 一定更优 |
| **SAM 3 / SAM 3.1**，SAM 3: Segment Anything with Concepts，arXiv:2511.16719；[原文](https://arxiv.org/abs/2511.16719) | 图像/视频 + 文本或视觉提示；概念级多实例框/掩膜/置信分数；不是仅 GT 框提示的旧 SAM 用法 | [仓库](https://github.com/facebookresearch/sam3)、[3 权重](https://huggingface.co/facebook/sam3)、[3.1 权重](https://huggingface.co/facebook/sam3.1)；SAM 自定义许可；两个权重入口需要访问条件，3.1 明确 gated | 优先做固定 ship 提示的真实端到端基线及受控适配。3.1 官方发布主要是 Object Multiplex 视频多对象跟踪，不能据版本号宣称静态 SAR 分割更好。不得用测试 GT 框/点构造部署性能 |
| **AnySat**，Astruc 等，CVPR 2025，arXiv:2412.14123；[原文方法](https://arxiv.org/html/2412.14123v2) | 多分辨率、多时相、多传感器统一表征；GeoPlex 包含 0.2–250 m 输入；特定模态 projector 与共享编码器；tile/patch/dense 特征，非实例头 | [仓库](https://github.com/gastruc/AnySat)、[权重卡](https://huggingface.co/g-astruc/AnySat)；代码及 HF metadata 为 MIT。hub 当前只公开 base 的预训练 URL；tiny/small 构造器存在，但预训练映射仍注明 coming soon | 适合真实多时相 S1/S2 或 VHR 配对实验。S1 预期 VV/VH（以及指定 ratio）、日期与 10 m 原生语义；当前单灰度不能假冒完整 S1 输入。学习新单通道 projector 是可检验适配，非原模型无改动复现 |
| **CROMA**，Fuller、Millard、Green，NeurIPS 2023；[原文](https://papers.neurips.cc/paper_files/paper/2023/hash/11822e84689e631615199db3b75cd0e4-Abstract-Conference.html) | 空间/时间对齐 S1 双极化 SAR + 12 通道光学，对比学习 + 掩码重建，X/2D-ALiBi；可单 SAR、单光学或融合输出特征 | [仓库](https://github.com/antofuller/CROMA)、[HF 权重](https://huggingface.co/antofuller/CROMA)；代码与模型页 metadata 为 MIT。README 提供 base/large 直接权重路径；未实际下载 | SAR-only 接口真实存在，是合理跨传感器预训练对照候选；但单灰度与双极化/辐射统计不匹配，需披露适配。默认 120 像素、patch8，不应直接以大分辨率输入假定廉价；全局注意力开销要实测 |
| **DOFA**，Xiong 等，arXiv:2403.15356；[原文](https://arxiv.org/html/2403.15356v2) | 按各通道元数据生成 patch 卷积核，变通道共享 ViT；多源预训练；公开主要分类/语义分割接口 | [作者仓库](https://github.com/zhu-xlab/DOFA)：MIT；[旧官方卡 earthflow/DOFA](https://huggingface.co/earthflow/DOFA)：CC-BY-4.0。最新 README 指向 [XShadow/DOFA](https://huggingface.co/XShadow/DOFA)，本轮该卡访问失败，最新权重许可未确认 | 比固定 2 极化接口更能容纳单通道，但必须说明输入元数据及版本匹配。SSDD/HRSID 不应被统一假定有已知波长/传感器；作者示例 wvs 数值不能未经单位核查视为物理波长。需要实例解码器，不是现成船舶实例模型 |
| **TerraMind**，Jakubik 等，ICCV 2025，arXiv:2504.11171；[原文](https://openaccess.thecvf.com/content/ICCV2025/html/Jakubik_TerraMind_Large-Scale_Generative_Multimodality_for_Earth_Observation_ICCV_2025_paper.html) | 多模态生成基础模型，TerraMesh 约 9M 空间/时间对齐样本；S1GRD/RTC、S2、RGB、DEM 等；TiM 预测缺失模态 token 再融合 | [IBM 作者仓库](https://github.com/IBM/terramind)、实际实现 [TerraTorch](https://github.com/torchgeo/terratorch)、[base 权重卡](https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-base)；代码与 HF metadata Apache-2.0 | S1 patch embed 原生 2 通道 VV/VH；可新增单通道输入层，但新增层不是已预训练的 SAR stem。TiM 要额外 tokenizer/generation 开销。更适合缺失模态与表征研究，当前先不作为最快 mask 提升主线 |
| **Prithvi-EO-2.0**，Szwarcman 等，arXiv:2412.02732，2024；[原文](https://arxiv.org/abs/2412.02732) | 时空 3D patch MAE；4.2M NASA HLS V2 样本，30 m，6 反射率波段；TL 版本另有时间/经纬度编码 | [项目仓库](https://github.com/NASA-IMPACT/Prithvi-EO-2.0)：MIT；实际 backbone 在 TerraTorch；[300M 权重卡](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M)：Apache-2.0；有公开模型入口 | 原生不是 SAR 输入。元数据可缺失并不解决光谱不匹配；灰度复制 6 通道不等于拥有 HLS 波段。作为跨域预训练补充对照可行，当前优先级低于直接实例模型 |
| **DOFA-CLIP（原 GeoLangBind）**，Xiong 等，arXiv:2503.06312v2，2025；[原文](https://arxiv.org/html/2503.06312v2) | 动态通道视觉—语言骨干；GeoLangBind-2M 六类 EO 模态；VECT 多视觉教师、MaKA 模态相关蒸馏；公开主要零样本分类/密集特征研究 | [作者仓库](https://github.com/xiong-zhitong/DOFA-CLIP)；README 说 MIT，但固定 commit 的根 LICENSE 实为 Apache-2.0；[权重卡](https://huggingface.co/earthflow/GeoLB-ViT-14-SigLIP-so400m-384-EO) metadata CC-BY-NC-4.0、正文 CC-BY4.0，存在冲突，按更严格条件保留并核清 | 语义先验有研究价值；**预训练明确含 SSDD/HRSID 来源，未排除测试交叉前不作干净泛化证据**。当前 demo 不是实例分割端到端脚本，需额外 dense/instance 集成与返回结构核验 |

补充：AnySat 作者仓库将工作标为 CVPR 2025；SAM 3 的后续会议状态由总审阅单独核查，此处不据二手汇总添加未经本子任务验证的录用信息。DOFA/DOFA-CLIP/Prithvi 采用所读原文版本，不擅自补充未核实期刊归属。

## 4. 已读到函数的源码证据

下面的 SHA、路径与函数是复现入口，而非仅列仓库首页。固定 SHA 下的源文件已经逐项静态阅读；本轮没有执行这些函数，没有证明依赖兼容、张量正确或数值有效。

| 项目及固定 commit | 已核查 API/路径 | 读码结论与集成影响 |
| --- | --- | --- |
| DINOv3 `6876159a11b4df116f30f667f8c9888617df0751` | [hub/backbones.py](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/hub/backbones.py)：`_make_dinov3_vit`、`Weights.SAT493M`；[vision_transformer.py](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/models/vision_transformer.py)：`forward_features`、`get_intermediate_layers` | 可拿 patch token/中间层空间图；hub 默认 3 输入通道、strict state load。不能把全局 CLS 直接当实例掩膜输出。SAT 专用归一化在固定 README 中单独提供 |
| SAM 3 `f1647e8d0c1d280991f4d1b7d974c2704537fe66` | [sam3_image_processor.py](https://github.com/facebookresearch/sam3/blob/f1647e8d0c1d280991f4d1b7d974c2704537fe66/sam3/model/sam3_image_processor.py)：`Sam3Processor.set_image`、`set_text_prompt`、`_forward_grounding`；[3.1 发布说明](https://github.com/facebookresearch/sam3/blob/f1647e8d0c1d280991f4d1b7d974c2704537fe66/RELEASE_SAM3p1.md) | `set_text_prompt` 调 `forward_text`，无几何提示时使用 dummy prompt；grounding 输出 `pred_boxes/pred_logits/pred_masks`。最终分数是类别 sigmoid × presence sigmoid；默认 threshold 0.5 会先过滤实例。用该演示输出算 COCO AP 时要检查阈值截断召回、保留连续分数，预先统一后处理 |
| AnySat `5f6f475e1a22ce5e3a56a5b18f4ed6d24eca2a4a` | [hubconf.py](https://github.com/gastruc/AnySat/blob/5f6f475e1a22ce5e3a56a5b18f4ed6d24eca2a4a/hubconf.py)：`AnySat.from_pretrained/forward`；[Any_multi.py](https://github.com/gastruc/AnySat/blob/5f6f475e1a22ce5e3a56a5b18f4ed6d24eca2a4a/src/models/networks/encoder/Any_multi.py)：`forward_release` | hub 检查正方形与跨模态相同物理覆盖范围；时序输入 5D。core 从 `*_dates` 读取日期，经各模态 projector、共享空间编码器、token concat 与 combiner。dense 输出拼接融合 patch token 与指定模态 subpatch，再还原空间图；不是原生 instance mask。hub 调用 dense 应显式指定 output_modality，避免隐含选择 |
| CROMA `59505a6bcadbf36ba20767270154bf9f3067c5e7` | [use_croma.py](https://github.com/antofuller/CROMA/blob/59505a6bcadbf36ba20767270154bf9f3067c5e7/use_croma.py)：`PretrainedCROMA.__init__/forward`、`BaseTransformerCrossAttn`、`get_2dalibi` | `SAR_channels=2`、`optical_channels=12`、patch8；forward 支持 SAR-only。cross attention 用 SAR query 和光学 context；相对二维距离 bias 是 patch-pair 矩阵。默认 120×120 输入有 225 patch；800×800 时为 10000 patch，注意力 pair 数比默认约 1975 倍，仅为理论 token 对数量，不是实测耗时 |
| DOFA `0cfb7e1099f4d4c4022946ff7862c7cd7b8411b9` | [dofa_v1.py](https://github.com/zhu-xlab/DOFA/blob/0cfb7e1099f4d4c4022946ff7862c7cd7b8411b9/dofa_v1.py)：`OFAViT.forward_features`；[wave_dynamic_layer.py](https://github.com/zhu-xlab/DOFA/blob/0cfb7e1099f4d4c4022946ff7862c7cd7b8411b9/wave_dynamic_layer.py)：`Dynamic_MLP_OFA.forward` | `wave_list` 转 tensor 后交给动态 patch embed；`wvs*1000` 正余弦编码、残差层与 Transformer weight generator 生成核，再 `conv2d`。当前简版 OFAViT 返回 pooled/CLS，做 mask 要保留空间 token 或接正式 dense backbone。README 特别提醒新权重需要最新动态层，禁止跨版本随意混用 |
| TerraMind 示例仓库 `53d61da09213c2c3629931b6322f94cce617886a`；实际 TerraTorch `83c3a928416fcf97deed924cfb99a59a817730bf` | [terramind_tim.py](https://github.com/torchgeo/terratorch/blob/83c3a928416fcf97deed924cfb99a59a817730bf/terratorch/models/backbones/terramind/model/terramind_tim.py)：`build_tim_modality_embeddings`、`TerraMindTiM.forward`；[modality_info.py](https://github.com/torchgeo/terratorch/blob/83c3a928416fcf97deed924cfb99a59a817730bf/terratorch/models/backbones/terramind/model/modality_info.py)；[terramind_vit.py](https://github.com/torchgeo/terratorch/blob/83c3a928416fcf97deed924cfb99a59a817730bf/terratorch/models/backbones/terramind/model/terramind_vit.py) | TiM.forward 对输入 tokenization，初始化缺失 target，构造生成 schedule，在 `no_grad` 下 sampler.generate，再把生成 token 加回输入并拼接编码。S1GRD/RTC 的原生 embed 都是 2 通道。TiM 使用随机种子生成，实验应固定并记录生成随机性与额外成本；它不是读取真实缺失观测 |
| Prithvi 项目 `758311280f935d4775c05b26d792768bf1fda841`；实际 TerraTorch 同上 | [prithvi_vit.py](https://github.com/torchgeo/terratorch/blob/83c3a928416fcf97deed924cfb99a59a817730bf/terratorch/models/backbones/prithvi_vit.py)：`_create_prithvi`、`prithvi_eo_v2_300`；[prithvi_mae.py](https://github.com/torchgeo/terratorch/blob/83c3a928416fcf97deed924cfb99a59a817730bf/terratorch/models/backbones/prithvi_mae.py)：`PatchEmbed.forward`、encoder `forward/forward_features` | factory 根据 model_bands 设置输入通道数，不提供时假设原始波段顺序；Conv3d patch embed，T=1 可将 4D 自动扩为 5D；时间/经纬度可为空。可改输入通道不代表 SAR 与 HLS 预训练物理量兼容 |
| DOFA-CLIP `be0f7c704eef2afbcc2adae831131e78391ceca4` | [dofa_clip.py](https://github.com/xiong-zhitong/DOFA-CLIP/blob/be0f7c704eef2afbcc2adae831131e78391ceca4/dofa_clip.py)：`encode_image`；[dofa_patch_embed.py](https://github.com/xiong-zhitong/DOFA-CLIP/blob/be0f7c704eef2afbcc2adae831131e78391ceca4/open_clip/src/open_clip/dofa_patch_embed.py)：动态层 `forward`；[实际 LICENSE](https://github.com/xiong-zhitong/DOFA-CLIP/blob/be0f7c704eef2afbcc2adae831131e78391ceca4/LICENSE) | demo 调 fork 的 `model.visual.trunk(image,wvs)`，另调文本 encoder、sigmoid 相似度；demo 有 tuple 解包，而 README 示例直接 normalize 返回，集成前需确认具体 trunk 返回结构。代码主要是分类使用入口，不能报告现成 ship mask 复现 |

### 4.1 三个精读方法的设计启发与边界

**AnySat。** 原方法先以相同地面范围划分跨传感器 patch，用专属 projector 与共享空间编码器容纳不同采样尺度，combiner 融合，训练结合跨模态对比与 EMA 教师的 JEPA 目标。源码的日期、物理尺寸检查和 subpatch 输出与此一致。因此可借鉴“局部细节 + 统一语义”的结构，但当前无真实配对时，不可沿用跨传感器正样本对应假设；还要量化细长小船在 token 化中损失多少。[原方法](https://arxiv.org/html/2412.14123v2)、[实际融合/稠密输出](https://github.com/gastruc/AnySat/blob/5f6f475e1a22ce5e3a56a5b18f4ed6d24eca2a4a/src/models/networks/encoder/Any_multi.py)。

**DOFA。** 关键是“按通道条件生成投影参数”，不是简单增减卷积输入通道；动态编码器、重建解码器与共享骨干共同适应不同来源。当前可考虑学习明确标为未知传感器的一通道适配，但不能编造波长或把强度变换当新谱段。作者 SAR 示例 `wave_list=[5.405,5.405]` 的数字接口已核到；其单位/归一化语义本轮未在原文中找到足够明确说明，故不解释为真实微米波长，实装应追到该版本数据预处理核对。[原方法](https://arxiv.org/html/2403.15356v2)、[动态卷积源码](https://github.com/zhu-xlab/DOFA/blob/0cfb7e1099f4d4c4022946ff7862c7cd7b8411b9/wave_dynamic_layer.py)、[作者用法](https://github.com/zhu-xlab/DOFA/blob/0cfb7e1099f4d4c4022946ff7862c7cd7b8411b9/README.md)。

**TerraMind。** 原方法是异构模态的统一生成建模，TiM 已经把预测出的辅助模态作为额外上下文。源码确认“生成—回填—拼接”的过程。若提出 SAR→光学幻化→融合，应把它作为直接先例对比，而非首创；还必须用同一生成器配普通 concat/平均融合，证明特定可靠性机制超过生成器本身。当前新一通道 stem 的重训需求、生成额外成本以及图像域差距，使其优先级低于简单可执行的实例适配。[原方法](https://openaccess.thecvf.com/content/ICCV2025/html/Jakubik_TerraMind_Large-Scale_Generative_Multimodality_for_Earth_Observation_ICCV_2025_paper.html)、[TiM 代码](https://github.com/torchgeo/terratorch/blob/83c3a928416fcf97deed924cfb99a59a817730bf/terratorch/models/backbones/terramind/model/terramind_tim.py)。

CROMA 的对比+联合重建及 X/2D-ALiBi 已通过原论文摘要与完整 use_croma.py 静态核对；本轮其 NeurIPS PDF 多次超时、arXiv 未提供 HTML，因此不把对全文的细节推断称作精读完成。

## 5. 船舶跨模态数据：真实配对不等于实例掩膜

| 数据 / 一手论文 | 真正提供什么 | 标注、可下载性与源码 | 当前研究中的合理用途与禁止混同 |
| --- | --- | --- | --- |
| **MOS-Ship**；A Multi-Modal Approach for Robust Oriented Ship Detection: Dataset and Methodology，Remote Sensing 18(2):274，2026-01-14；[出版社原文](https://www.mdpi.com/2072-4292/18/2/274) | 作者报告亚米级、空间对齐但异步光学/SAR；更侧重停泊船减少时差影响；附合成云雾数据。逐传感器精确 GSD/时差分布本轮未从可核页面完整获取，保留未知 | DOTA 旋转四点框 + 类别/难度，**非 mask**；[官方仓库](https://github.com/yyy7777777/MOS-Ship-Dataset) `9e5abe03452a0b14ca7a0e0e3d2dbf6d26f6efd5` 只有 README 与 Apache-2.0 LICENSE，以及百度网盘入口；当前快照未发现 MOS-DETR 方法源码 | 是真实传感器融合检测支线，不能冒充 HRSID 的对应光学来源。若拓展成实例分割需已有公开 mask 或另行标注，与当前避免新增人工标注要求冲突。仓库 Apache 许可不能自动推断商业卫星原数据获同等再分发授权 |
| **OS-Ship-1K**；作者仓库当前题名 A Pseudo-Paired Optical–SAR Dataset for Multimodal Ship Detection，标注 PRCV 2026 录用；[作者发布](https://github.com/1998Jiang/OS-Ship-1K) | **1000 对光学和 CycleGAN SAR-style 合成图**，非第二台 SAR 传感器实际成像；Ship/Ships 两类 | 仓库 `aafad57b7a73b5f2b870b4a7e66aa9513f0c4ec0`；README 有网盘下载、14 单模态/6 融合检测器基准；检测标注，未验证实例 mask；repo MIT，底层数据权利需另核 | 可研究合成域、风格差异和伪配对鲁棒性；不能证明实测光学/SAR 互补或物理融合。精确 GSD/传感器及文件完整性未检查，不能以外观推出 |
| **HOSS ReID / TransOSS**；Wang 等，ICCV 2025，7873–7883；[原文](https://arxiv.org/html/2506.22027v1) | 真实 Jilin-1 光学 0.75 m、TY-MINISAR SAR 1 m；异时异角度同船身份；13 序列共 43 帧，从中裁出 1832 图，449 身份 + 163 干扰目标 | Market-1501 风格身份检索；框裁片、ID/相机/序列/模态信息，**无配准 mask**。SAR TIF 单通道 float32；[Zenodo DOI 10.5281/zenodo.15860212](https://zenodo.org/records/15860212) HOSS.zip 82.9 MB，另有约 7.3 GB 的 SEN1-2/DFC23 预训练包；未下载。[官方代码](https://github.com/Alioth2000/Hoss-ReID) `bd9b9028e8690c0929ada06beead500acde19aca` 为 MIT；Zenodo 可解析页面 license 栏为空，数据许可未确认 | 可学同船跨传感器语义，或作为检索支线。身份正样本不能当逐像素对应，mAP/R1 不是 COCO AP。其分辨率和物理尺寸可启发建模，但当前公共裁片无逐图 GSD 时不能凭空添加真实尺寸输入 |

MOS-Ship 论文的存储表述为 PNG 且 SAR 单通道 float32；标准 PNG 存储与这一表述存在需要核对的格式疑点。本轮未打开数据，故只记录作者说法，不保证归档内容真为 float32 PNG，也不以此指控数据错误。核查时出版社随后返回 429；本轮首轮检索已获得其方法/数据段，后续复读受限。

HOSS 实际源码 [vit_transoss.py](https://github.com/Alioth2000/Hoss-ReID/blob/bd9b9028e8690c0929ada06beead500acde19aca/model/backbones/vit_transoss.py) 的 `TransOSS.forward_features(x,camera_id,img_wh)` 按 modality 分流两个 patch embed，后续共享 Transformer；可添加尺度 token，最终默认返回 CLS。它逐图提取检索特征，并非把同船两幅图按像素融合。这一边界在原方法和源码中一致。

### 5.1 现有公共 SAR mask 的信息边界

HRSID/SSDD 的普通发布灰度图不是原始复数回波。不能由此合法获得真实 phase、VV/VH 双极化、散射矩阵、独立光学观测或精确时序。HRSID 自带公开实例标注，是当前无需新增判读的合适 mask 主基准；SSDD 各种扩展/重标版本的 mask 许可、来源与版本仍须沿用本项目既有证据链，不因出现“SSDD”字样就假定标准原版均有实例 mask。任何更改 mask 来源、划分或读取路径属于后续独立实验资格工作，本轮没有执行。

## 6. 新颖性风险：哪些已经有人做，哪些仍可检验

| 拟主张 | 查到的直接先例/反例 | 合理改写或验证措施 |
| --- | --- | --- |
| “多模态基础模型首次用于这里，因此创新成立” | CROMA、AnySat、DOFA、TerraMind 均已有多模态表示；任务搬用单独不说明机制贡献 | 明确 SAR 小目标/近岸误检/粘连的实际误差，提出局部机制及相对最接近先例的技术差异 |
| “多个视觉教师 + 模态蒸馏是新贡献” | DOFA-CLIP 的 VECT/MaKA 已有 SigLIP/DINOv2/ViT 教师及条件化蒸馏 | 不能仅换成 DINOv3/SAM3 改名；需针对局部可靠性、边界带/背景污染的不同监督目标，并有同教师普通蒸馏对照 |
| “SAR 生成光学再融合是首创” | TerraMind TiM 已生成缺失模态 token 参与后续任务 | 可以做任务专用可靠性和小目标保护，但至少与同生成器 + concat/TiM/无辅助生成比较 |
| “ship 文本带来多模态理解” | 单类固定文本可能仅等价于一个额外可学习 query | 固定 ship、空提示/可学习闭集 query、无文本三组；如语义无独立贡献，删除文字论断 |
| “迁移模型一定避免下游监督泄漏” | DOFA-CLIP 附录明确训练字幕来自 HRSID/SSDD | 无排除样本清单前标高风险；不靠相似哈希对活动数据擅自扫描。本轮仅核论文来源，后续需经授权的独立审计流程 |
| “图像变成三通道就是多模态” | 三通道均为同一灰度的确定性函数，没有独立观测 | 按同图多表征命名；与普通卷积/参数匹配支路对比，检验变换机制而非通道数 |

DOFA-CLIP 的重叠依据可直接定位于[原文附录 SAR-ship-caption and IR-ship-caption](https://arxiv.org/html/2503.06312v2)。这不等于宣布作者实验不当，也不等于具体某个公开权重已确定看过本项目全部测试图；它足以要求下游研究保留污染风险、明确证据范围。

## 7. 建议主线：强实例基线上的 SAR 局部结构与基础语义互补

**工作假设，尚未实现/验证：** 基础模型的类别/上下文特征可能有助于拒绝港口亮散射伪目标，而小型细长船、相邻船缝隙和边界定位可能依赖更高分辨率的 SAR 局部结构。应先用基线预测与已有 mask 验证这两个错误群是否存在；不能用一般领域直觉替代误差统计。

**建议的一条主路线：** 以可执行且可公平训练的强实例分割模型 B 为承载，优先完成原生 SAM 3 固定 ship 提示或当前强模型的标准微调基线，再做一个可单独消融的局部残差适配器。部署始终只有 SAR 图；不引入缺失光学/极化/相位。DINOv3 SAT 与自然图预训练、SAM 3 语义等作为候选先验逐个进入，先不用多教师叠加。

- **局部结构支路 L：** 从原始 SAR 灰度图提取原分辨率局部特征；可试已有调整过的 UNet++ 中确实有训练证据的局部/跳连结构，但必须保存原先收益的划分与预算边界。若用梯度/局部统计，只称同图表征；强度和归一化参数只在训练集确定。
- **全局语义 S：** 优先保留 B 的预训练骨干或冻结教师密集特征，先做线性投影/普通蒸馏，避免先承担另一个完整大模型的推理成本。SAM 类文本若作为额外先验，必须与可学习闭集 query 做必要性对照。
- **唯一待检验的新机制候选：** 让局部与全局的分歧决定残差注入的空间位置/强度，目标是保留细长目标边界，同时抑制局部亮点对分类的错误强化。采用近零初始化残差；监督来自已有训练 mask/框，不借测试 GT 提示。必须与普通同容量融合、简单边界辅助监督对照。该候选有既有 reliability gating、adapter、boundary modeling 先例，尚未完成面向直接先例的独立查新，因此不可命名为已证实原创模块。
- **训练侧迁移是备选阶段：** 仅当简单局部分支达到可重复增益再考虑冻结 DINOv3/SAM3 教师；保持教师、输入、损失权重搜索和训练预算一致，检验是否超过普通蒸馏。没有必要先做光学图生成。

这一主线的优点是无需新标注、输入条件真实、每一项改动可与简单对照分离。风险是预训练先验可能对 SAR 无效、模块仅改变置信排序、额外容量解释全部收益，或者短预算优化速度与长期收敛排序不同；均需要下列实验而非文字消解。

### 7.1 最强简单替代对照及最小实验矩阵

**最强简单替代：同一个 B，采用标准端到端微调或普通 LoRA/adapter，使用与候选方法相同预训练、trainable 参数/训练配方与搜索资源。** 若候选使用第二个教师，替代组必须获得同一个教师和普通线性/concat 蒸馏；否则无法区分“多拿一个模型”与新机制。

| 实验 | 唯一用途 | 数据与公平约束 |
| --- | --- | --- |
| B0 原生强基线 | 确定当前真实起点 | 同数据版本、划分、输入尺度、初始化、增强、更新数和评估；推理无测试 GT prompt |
| B1 标准微调/普通 adapter | 证明是否仅为更强训练或容量收益 | 参数量/训练可学习部分匹配；共享超参搜索机会 |
| B2 B + L 的普通残差/concat | 证明局部特征本身的作用 | 与候选用同支路、同宽度、同分辨率，避免引入容量差 |
| B3 B + 简单边界辅助损失 | 排除已有 mask 衍生边界监督足以解释收益 | 同已有 mask；同 loss 调参资源；不用额外人工边缘 |
| B4 B + L + 分歧控制残差 | 检验唯一候选机制 | 与 B2 控制容量/计算量；同时报告 mask 几何与分数变化 |
| 若再用教师：T0 普通蒸馏 / T1 候选教师约束 | 检验迁移机制是否超过教师本身 | 相同教师、训练图、蒸馏步数和额外计算量；先排预训练重叠 |
| 若再用文本：固定 ship / 可学习 query / 去文本 | 检验语言语义是否必要 | 不使用逐测试图人工 caption、GT 数量/框/掩膜 |

先用所有组一致短预算筛选，再将 B0、最强简单替代与候选进行**相同完整预算**训练。短预算负差只否定该预算配方下的优越性；不能直接定性模块总体无效。长训报告学习曲线与收敛检查，至少 3 个预先固定种子，并说明是否相同搜索机会。若现有资源不能支持全部组，保留 B0/B1/B2/B4 的完整比较，减少教师数量，不能缩短对照而长训本方法。

### 7.2 结果应支持什么，不能支持什么

- 主指标用预先锁定的 COCO mask AP@[.50:.95]；附 AP50/AP75、AR、尺寸段、近岸/离岸、相邻船/细长目标分组。分组规则来自已存在图像/标注和预注册算法，不能以测试收益倒推分组。
- 将候选 mask 几何固定、只替换 score 的对照单独列出；若主要收益来自置信排序，应称为校准/排序收益，不能全归因边界机制。
- 报告训练和推理显存、吞吐/延迟、参数与 FLOPs（能可靠测量的部分），包含教师或生成器开销；不以 128 GB 容量当作算力等同证据。
- 跨数据集先限定：用源训练、源验证选择模型，在目标测试集直接测试；若目标重新微调，另称跨数据集微调，不能写零样本泛化。mask 语义和尺度差异需核查，严禁转换框填充掩膜充当 GT。
- 若只有一个数据集一个种子微小增益，只能列观察结果；不足以宣称 SOTA 或 TGRS 投稿级成功。真实可重复的核心改进通常比凑三个模块叙述更有说服力。

## 8. 访问、许可与待办边界

1. DINOv3 与 SAM3/3.1 访问 gated 已由模型页明确核实；本轮未尝试绕过或使用非官方再分发权重。正常获批及取得匹配权重是执行前提，不能报告已就绪。
2. DOFA 新 XShadow 权重模型卡本轮访问失败；旧 earthflow 卡可读且 CC-BY4.0。不得把旧许可直接套到新权重。DOFA-CLIP 的仓库 README/许可证及权重卡内部有冲突，采用前保留原记录并核实；不擅自更改第三方授权。
3. 本机无代理的 HF API 元数据只读请求超时；没有得到模型文件 SHA/gated JSON。作者网页和固定源码已核；所有权重“未下载/未执行”状态保持，不以联网问题推断仓库不存在。
4. MOS-Ship 网盘归档内容、OS-Ship-1K 归档标注及 HOSS 原始数据许可未实际下载核验；公开论文和 repo license 不自动授权重发商业卫星图或底层数据。只保留官方入口，不上传原数据/权重/第三方全文。
5. 若下一步执行，先沿用主项目当前资格链完成独立依赖/接口验证；本审阅不授权读取活动科学载荷，也不对源数据/权重作哈希。优先解决基线、权重访问和误差机制，避免为了“最新”一次性引入全部模型。

## 9. 主要一手资源导航

- DINOv3：[论文](https://arxiv.org/abs/2508.10104)、[固定仓库](https://github.com/facebookresearch/dinov3/tree/6876159a11b4df116f30f667f8c9888617df0751)、[SAT 模型卡](https://huggingface.co/facebook/dinov3-vitl16-pretrain-sat493m)。
- SAM3：[论文](https://arxiv.org/abs/2511.16719)、[固定仓库](https://github.com/facebookresearch/sam3/tree/f1647e8d0c1d280991f4d1b7d974c2704537fe66)、[3.1 release](https://github.com/facebookresearch/sam3/blob/f1647e8d0c1d280991f4d1b7d974c2704537fe66/RELEASE_SAM3p1.md)。
- AnySat：[原文](https://arxiv.org/html/2412.14123v2)、[作者页](https://gastruc.github.io/anysat)、[固定仓库](https://github.com/gastruc/AnySat/tree/5f6f475e1a22ce5e3a56a5b18f4ed6d24eca2a4a)。
- CROMA：[会议论文页](https://papers.neurips.cc/paper_files/paper/2023/hash/11822e84689e631615199db3b75cd0e4-Abstract-Conference.html)、[固定仓库](https://github.com/antofuller/CROMA/tree/59505a6bcadbf36ba20767270154bf9f3067c5e7)。
- DOFA：[原文](https://arxiv.org/html/2403.15356v2)、[固定仓库](https://github.com/zhu-xlab/DOFA/tree/0cfb7e1099f4d4c4022946ff7862c7cd7b8411b9)、[旧官方权重卡](https://huggingface.co/earthflow/DOFA)。
- TerraMind：[会议论文页](https://openaccess.thecvf.com/content/ICCV2025/html/Jakubik_TerraMind_Large-Scale_Generative_Multimodality_for_Earth_Observation_ICCV_2025_paper.html)、[示例仓库](https://github.com/IBM/terramind/tree/53d61da09213c2c3629931b6322f94cce617886a)、[实际实现固定仓库](https://github.com/torchgeo/terratorch/tree/83c3a928416fcf97deed924cfb99a59a817730bf)。
- Prithvi：[技术报告](https://arxiv.org/abs/2412.02732)、[项目仓库](https://github.com/NASA-IMPACT/Prithvi-EO-2.0/tree/758311280f935d4775c05b26d792768bf1fda841)、[300M 模型卡](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M)。
- DOFA-CLIP：[原文和附录](https://arxiv.org/html/2503.06312v2)、[固定仓库](https://github.com/xiong-zhitong/DOFA-CLIP/tree/be0f7c704eef2afbcc2adae831131e78391ceca4)、[权重卡](https://huggingface.co/earthflow/GeoLB-ViT-14-SigLIP-so400m-384-EO)。
- MOS-Ship：[出版社论文](https://www.mdpi.com/2072-4292/18/2/274)、[固定仓库](https://github.com/yyy7777777/MOS-Ship-Dataset/tree/9e5abe03452a0b14ca7a0e0e3d2dbf6d26f6efd5)。
- OS-Ship-1K：[固定作者仓库与当前题名/数据说明](https://github.com/1998Jiang/OS-Ship-1K/tree/aafad57b7a73b5f2b870b4a7e66aa9513f0c4ec0)。本轮未获得与当前修订标题完全对应的可读完整 proceedings 正文，因此不把作者 README 的全部方法细节称为独立论文全文核验。
- HOSS ReID：[原文](https://arxiv.org/html/2506.22027v1)、[作者仓库](https://github.com/Alioth2000/Hoss-ReID/tree/bd9b9028e8690c0929ada06beead500acde19aca)、[数据 DOI](https://zenodo.org/records/15860212)。

状态：本地研究文档完成；未运行科学实验；未证实候选涨点或新颖性；GitHub 同步由主任务按既有发布范围处理，本子任务不宣称已同步。
