# GLDet作者复现原始来源核查：候选仓库未提供实现，初值仍未知

核查日期：2026-11-07（最后公开查询为2026-11-06 20:21 UTC）。本轮只读用户提供的原始20页论文、出版方登记元数据、GitHub公开仓库及有界定向检索。没有SSH、读取运行SOURCE/权重/资格链或原始GT，没有导入Torch、训练、预测或计算AP；当前主实验与外部参考配方均未修改。

**结论：本轮没有取得能够认证为作者发布、可直接训练的GLDet代码、预训练初值或训练后权重。** 找到两个同名候选仓库，但一个只有8字节README，另一个为空；作者身份也没有闭合。该结论仅覆盖下面实际检查的来源，不证明作者从未在其他位置提供实现。GLDet仍可列作论文报告的相关强方法，不能据此声称已建立完整作者复现或与本地内部DEV的公平排名。

## 原论文能直接确认的实验条件

题名为 *GLDet: Real-Time SAR Ship Detector Based on Global Semantic Information Enhancement and Local Gradient Information Mining*，IEEE TGRS 63（2025），文章5209020，DOI `10.1109/TGRS.2025.3559551`。原PDF首页记录2025年4月10日在线发表、4月21日当前版本。Crossref的出版方登记记录指向IEEE文献10962196；本轮IEEE网页要求JavaScript验证，方法与实验事实来自用户提供的原PDF，不来自搜索摘要。[论文DOI](https://doi.org/10.1109/TGRS.2025.3559551)、[IEEE文献入口](https://ieeexplore.ieee.org/document/10962196/)、[出版方登记元数据](https://api.crossref.org/works/10.1109%2FTGRS.2025.3559551)。

原PDF第10页、IV-A的实验说明可核：

| 项目 | 论文实际写明 | 尚未闭合的执行细节 |
|---|---|---|
| 平台 | CUDA12.1、PyTorch2.1.2、OpenMMLab算法系统 | 具体OpenMMLab各库版本、源码提交与环境锁定 |
| 优化与预算 | AdamW、基础学习率0.004、batch16；作者称全部比较和消融训练500 epochs | 学习率调度、有效全局batch/梯度累积、drop-last、实际步数、收敛、重复种子与端点/best选择 |
| SSDD/HRSID输入 | 训练和检测800×800、COCO格式 | 实际resize/pad/插值与掩膜转换管线、增强、原始JSON/影像版本身份 |
| 阈值 | 文中称rating threshold为0.01 | 无代码证明其确切应用位置；不能擅自等同NMS阈值或COCO maxDets |
| 评价定义 | COCO IoU0.50至0.95、步长0.05，并报告AP50/AP75及尺度指标 | 实际评价器源码/版本、maxDets、忽略/crowd处理与候选后处理 |
| 初始化 | 本轮未找到具体初始checkpoint、预训练来源或加载覆盖率的明确声明 | 是否ImageNet/COCO/SAR预训练、是否全部从头、每种比较方法的初始化与上游成本 |
| 数据划分 | 指定SSDD/HRSID用于实例分割比较 | 本轮未取得准确train/test比例、数量、清单或文件摘要；不能替换成项目内部FIT741/DEV187后仍称论文原协议 |

500轮只是作者报告的训练预算，不能直接与当前50轮精修系统按epoch标签比较；还应分列教师预训练、项目教师训练、缓存、下游训练、图像/实例访问和整套推理成本。论文中的平台与公平性声明也不是本项目独立复现实证。

**第17页的SARATR-X预训练表述属于后续融合建议。** 该段先对比GLDet的几何归纳偏置与SARATR-X的大规模预训练特征，再提出用SARATR-X预训练权重初始化GLDet等混合架构的方向。它没有给出本论文GLDet已加载该权重的执行证据，不能被当作实际预训练来源。初始化在当前记录中仍为未知，不是零成本。

原PDF20页的文本及外部链接检查未发现代码仓库或模型下载地址；PDF链接指向六位作者的ORCID。没有论文/作者页面对下列候选仓库的显式关联，因此不能只凭用户名与Hao Chang相似就认证为官方。

## 实际检查的同名仓库

普通搜索未定位可确认的代码入口后，执行了一次GitHub公开仓库API的`GLDet`名称检索，返回7个名称匹配结果，其中有两个下列候选；其余名称匹配不能视为该论文实现。连接工具不支持该仓库搜索端点，本次使用无认证的GitHub公开API取得列表；后续仓库/分支/内容/树/发布信息用已连接GitHub读取工具核查。[有界仓库查询](https://api.github.com/search/repositories?q=GLDet&per_page=30)。

| 候选 | 本轮实际公开状态 | 可用于复现的内容 |
|---|---|---|
| `changhao123/GLDet` | 默认且唯一返回分支main；最新提交`5927838337271bc07703b0d1bbfda355c08dc5f7`，2023-12-07；初始提交无父节点；完整未截断树只有一个8字节README | README仅为`# AIRSAR`，没有模型、配置、训练器、数据协议、权重或论文身份说明；releases为空 |
| `changhao123/GLDet1` | 仓库元数据size0，创建/最后push为2024-05-11；内容API明确返回repository empty；releases为空 | 没有取得已提交的代码、配置或模型权重 |

第一个候选的固定根树为`d77d6bc540987f574a3b15904f72bb3acb776508`，唯一README blob为`89bc8ee13ffe6516349c49cba3d8035f0a7f46b9`。固定提交比动态main更适合作本轮证据；GitHub提交签名有效不等于论文作者身份已认证。[固定提交](https://github.com/changhao123/GLDet/commit/5927838337271bc07703b0d1bbfda355c08dc5f7)、[固定README](https://github.com/changhao123/GLDet/blob/5927838337271bc07703b0d1bbfda355c08dc5f7/README.md)、[完整固定树](https://api.github.com/repos/changhao123/GLDet/git/trees/d77d6bc540987f574a3b15904f72bb3acb776508)、[GLDet分支查询](https://api.github.com/repos/changhao123/GLDet/branches?per_page=100)、[GLDet发布查询](https://api.github.com/repos/changhao123/GLDet/releases?per_page=100)、[GLDet1仓库](https://github.com/changhao123/GLDet1)、[GLDet1发布查询](https://api.github.com/repos/changhao123/GLDet1/releases?per_page=100)。

本轮未找到可认证的作者模型下载入口；上述两个候选的状态不能扩大为整个互联网或所有作者私有发布均无代码/权重的结论。ORCID网页依赖JavaScript，本次未取得可证明仓库归属的作者主页关联。Crossref另登记正式作者回应DOI，但对应Web of Science评审入口本轮不可读取，未将未取得的回应内容用于初始化判断。

## 对当前实验的行动建议

1. 主PLAIN/FULL固定E50与已锁定DiffSAR参考继续按原协议执行，不因同名空仓库停止主实验，也不修改旧初始化、预算或失败记录。
2. GLDet暂分列为“作者论文报告，当前尚无本项目完整作者复现”。在认证代码、实际初始化、数据版本/划分和评价规则闭合前，不启动一个随意拼出的GLDet结构后以其低分声称战胜原方法；原论文的mask结果也不能与内部DEV187直接排名。
3. 若后续按原论文重新实现，应明确命名“paper-guided reimplementation / 同协议重实现”，逐项列出重建与未知设置；先锁预算、初始化、评价和成本，再训练。重实现与完整作者复现必须保留区别，短预算比较同样给对照同预算，最终效用仍需充分训练证据。
4. 该方法与当前项目共同涉及全局/局部特征组织和轮廓信息，适合作为先例边界讨论；本轮代码可用性核查没有产生AP、新模块有效性或创新性证据。继续优先推进可核作者源码的完整外部参考及现有固定端点，不反复搜索空仓库代替实验。

本页只保存自有摘要、可复查定位与证据边界，不分发第三方论文全文/表格图、代码副本、原始数据、权重、凭据或运维配置。
