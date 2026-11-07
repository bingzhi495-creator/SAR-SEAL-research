# DiffSAR公开预训练对照：新增可核入口及加载范围

本页为子代理静态来源核查快照；后续Root已从已有缓存完成[独立V1候选字节取得](PUBLIC_R101_INITIALIZER_BYTE_INTAKE_STATUS_20261107.md)，仍未执行转换/张量或模型加载。下文“未下载/未核验”限定该静态查证任务。

2026-11-07。本次新证据是可为**另一条公开预训练初始化轨迹**选择官方来源，不是取得DiffSAR论文的实际初值。论文实现段、预训练文件和实际总训练预算仍未知。现有随机构造初值的正式512轨迹保持，不中途加载新权重、不改写为预训练复现。本次没有下载/打开权重、运行模型、SSH或读取运行SOURCE、数据、资格链，也没有新AP。

此前普通预算说明已提出需要可核预训练条件，但没有列出以下具体权重、stride差异及加载覆盖检查。本次只保存新入口和静态风险。

## 1. 优先候选：显式ImageNet-1K R101骨干初始化

Torchvision官方 `ResNet101_Weights.IMAGENET1K_V1` 的权重入口为[resnet101-63fe2227.pth](https://download.pytorch.org/models/resnet101-63fe2227.pth)，其身份和ImageNet-1K范围见[官方版本化源码](https://github.com/pytorch/vision/blob/v0.13.0/torchvision/models/resnet.py#L391)及[官方模型说明](https://docs.pytorch.org/vision/2.0/models/generated/torchvision.models.resnet101.html)。不要用会随版本改变的DEFAULT替代明确的V1。尚未下载，完整SHA和实际内容未核验。

官方Detectron2的[转换器](https://github.com/facebookresearch/detectron2/blob/247691a1c292654867deb35cabb27f18d75a1363/tools/convert-torchvision-to-d2.py#L8)说明支持ResNet18/34/50/101/152，并将stage、norm与shortcut名称转为Detectron2格式，保存matching_heuristics标记。其配套RGB、0–255量纲下均值123.675/116.280/103.530、标准差58.395/57.120/57.375、stride-in-1x1=False，与[DiffSAR公开SSDD配置](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/configs/DiffSARShipInst-SSDD-train.yaml#L19)相符。分类器224裁剪的推理示例不应被误用来替换原SAR训练mapper；实际像素量纲/重复归一化仍须在未来加载验证时核查。

另有省转换步骤的官方入口：DiffusionDet在[Getting Started第32–52行](https://github.com/ShoufaChen/DiffusionDet/blob/1efb36d4b0b721f2d509bc72656235a94d73ab02/GETTING_STARTED.md#L32)提供[torchvision-R-101.pkl](https://github.com/ShoufaChen/DiffusionDet/releases/download/v0.1/torchvision-R-101.pkl)；[v0.1 release元数据](https://api.github.com/repos/ShoufaChen/DiffusionDet/releases/tags/v0.1)记录178666900字节，作者称与Detectron2兼容。该声明不等于我们已经加载成功，也不认证这个已转换文件究竟对应哪一版Torchvision权重；如需要完整可追溯性，显式V1原文件及固定转换器更清楚。文件名、大小和发布链接都不能代替下载后的字节身份。

不优先直接采用MSRA R101。[Detectron2官方模型目录](https://github.com/facebookresearch/detectron2/blob/247691a1c292654867deb35cabb27f18d75a1363/MODEL_ZOO.md#L50)确有公开ImageNet R101，但[官方catalog注释](https://github.com/facebookresearch/detectron2/blob/247691a1c292654867deb35cabb27f18d75a1363/detectron2/checkpoint/catalog.py#L14)说明MSRA采用stride-in-1x1=True，旧catalog模型还要求FrozenBN与unit STD；这与当前False及RGB标准差条件不同。权重张量尺寸相同不代表下采样和预处理语义相同，不应为加载它而悄悄改变原配方。

## 2. 可选的COCO信息参考：DiffusionDet官方COCO-R101

[官方README第19行](https://github.com/ShoufaChen/DiffusionDet/blob/1efb36d4b0b721f2d509bc72656235a94d73ab02/README.md#L19)提供[diffdet_coco_res101.pth](https://github.com/ShoufaChen/DiffusionDet/releases/download/v0.1/diffdet_coco_res101.pth)；同一release元数据记录519674553字节。其[R101配置](https://github.com/ShoufaChen/DiffusionDet/blob/1efb36d4b0b721f2d509bc72656235a94d73ab02/configs/diffdet.coco.res101.yaml)为80类、500 proposals、stride-in-1x1=False，并指向[RGB原base配置](https://github.com/ShoufaChen/DiffusionDet/blob/1efb36d4b0b721f2d509bc72656235a94d73ab02/configs/Base-DiffusionDet.yaml)。这是作者公开COCO检测权重入口，不是SAR或实例掩膜预训练；未下载，不证明实际可用内容或迁移覆盖。

不能把该整包当DiffSAR可严格加载的同构初值。[DiffusionDet分类层](https://github.com/ShoufaChen/DiffusionDet/blob/1efb36d4b0b721f2d509bc72656235a94d73ab02/diffusiondet/head.py#L226)对应80类，DiffSAR为1类；DiffSAR另有SCJE-FPN、MACSA/IAMR及全stage调制DCN。若后续只迁移骨干，必须称COCO骨干初始化；若另迁移同形FPN或动态回归头，必须在看到结果前固定白名单及逐键映射，不能因模块同名就默认其功能/位置相同。[SCJE-FPN仍使用bottom_up与部分fpn_lateral/fpn_output名称](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/detectron2/modeling/backbone/fpn.py#L292)尤其容易造成“同名即同义”的误判。

DiffusionDet项目[许可](https://github.com/ShoufaChen/DiffusionDet/blob/1efb36d4b0b721f2d509bc72656235a94d73ab02/LICENSE)为CC BY-NC 4.0；本页只链接官方入口，不重新分发资产。Detectron2模型目录另声明其列出的模型许可，不将不同发布者的许可混为同一条。当前只讨论科研参考，不增加数据或权重发布范围。

## 3. 真正开新轨迹前需要证明的加载范围

1. 固定原始权重完整字节身份、官方来源、数据声明和转换代码版本；列出每一对源/目标key、dtype、shape及载入后逐张量相等性。单列stem、res2–5、FrozenBN参数/缓冲、FPN、动态头和掩膜分支的key数及numel覆盖，保留missing/unexpected/shape-mismatch的精确允许名单。不能仅凭strict=False没有抛错或matching_heuristics日志就写“成功预训练”。DiffSAR的[加载器](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/detectron2/checkpoint/detection_checkpoint.py#L114)及[后缀匹配逻辑](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/detectron2/checkpoint/c2_model_loading.py#L207)会做名称适配，真实覆盖仍须实际证据。
2. 未载入的DCN offset/mask、SCJE和IAMR保留预先声明的构造初始化，并验证未被意外覆盖。特别是[DeformBottleneck源码第259–312行](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/detectron2/modeling/backbone/resnet.py#L259)：offset卷积权重/偏置初始为0，经sigmoid后mask为0.5。因此复制普通conv2权重的尺寸可匹配性，不等于DCN初始化在函数上与普通卷积等价；不为追求等价而擅自乘2或改mask，这会成为另一个实验条件。
3. 通过真实加载/输入/前向与原资源资格后，才从新正式update0建立独立预锁轨迹，分层记录ImageNet/COCO上游信息及可核成本。当前随机初值512轨迹不能接上新初始化或被覆盖。可以称“公开ImageNet骨干初始化的同协议适配”或准确限定的COCO部分迁移，仍不能称DiffSAR论文复现，也不自动与项目COCO分割教师＋E24＋refiner的完整信息/成本匹配。

本次以上均为官方文本、配置及源码的静态证据和待执行建议。论文初值仍未知；没有权重加载资格、训练正收益或SOTA结论。
