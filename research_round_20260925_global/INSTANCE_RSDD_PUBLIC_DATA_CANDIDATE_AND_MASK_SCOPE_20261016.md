# 补充跨集候选：Instance-RSDD 与掩膜标签边界（2026-10-16）

**状态：候选来源已核对，数据尚未入场。** 本轮定向检索 HRSID、PSeg-SSDD 之外的公共 SAR 船舶实例掩膜，保留 Instance-RSDD 作为值得继续核验的一项候选。尚未获取数据包、核验真实标注编码或计算任何 AP；当前 N20 的数据、方法、预算与停止规则均未改变。

## 已取得的具体来源

- **作者入口：** [XIDIAN212Lab/SRNet](https://github.com/XIDIAN212Lab/SRNet) 的 Instance-RSDD 段明确说明对 RSDD-SAR 作实例级重标，并提供[百度公开入口](https://pan.baidu.com/s/1_hIyZKErRDvvceiKDhRmvA)，公开提取码为 `2023`。本次直接读取作者 README 成功；浏览工具未能读取百度目录，没有证明链接失效，也没有证明当前可完整获取。
- **原论文：** Yang et al., *Ship Instance Segmentation Based on Rotated Bounding Boxes for SAR Images*, Remote Sensing 2023, 15(5), 1324，[DOI/出版页面](https://doi.org/10.3390/rs15051324)。§4.1.2 报告从 RSDD 选择 3555 幅较清晰图像，人工逐像素标注船舶实例，2840 幅训练、715 幅评价；原 RSDD 的 5000/2000 是旋转框检测划分，不能混用。§4.1.3 描述基于实例 mask IoU 的 AP。
- **读取边界：** 原论文信息由出版页面的搜索索引正文抽取核对，直接打开 MDPI 正文返回 429，未下载 PDF 或核对图表原图。可复现检索为 `"Ship Instance Segmentation Based on Rotated Bounding Boxes" "2840"`，限定 `mdpi.com`。这些数量与标注说明是作者报告，尚非本地入场计数。

作者说明其影像源自 RSDD-SAR（[原集 DOI](https://doi.org/10.12000/JR22007)）。本次不能据名称或来源声明推定它与 HRSID、SSDD 的母景、港区、采集时间或图像互斥；“较清晰图像子集”也限定了后续结论的覆盖范围。

## 必须继续核验的缺口

| 项目 | 当前已知 / 真正缺口 |
|---|---|
| 标注类型 | 作者报告人工逐实例像素标注；包内究竟为 JSON、多边形、实例 ID 图或逐目标文件，尚未读取 |
| 官方划分与数量 | 论文报告 2840/715；实际文件清单、逐船身份、类别、空图、缺失与忽略规则尚未核验，不能把原检测划分套上来 |
| 坐标与图像关联 | 原图、裁剪、尺寸、掩膜坐标及实例关联尚未核验，不能凭论文的 mask AP 自动认定包内是标准 COCO |
| 独立性与历史使用 | 未核查母景/切片重叠，也未检查本机是否已有或曾用于训练、选型；两项均保持未知，不能称未接触盲测 |
| 授权 | 尚未取得完整上游与外链包许可；不从论文开放许可推导影像再分发权，也不由许可缺口推断禁止研究使用 |
| 实际获取 | 仅确认作者公开入口；目录与包内容尚未获取。本轮未下载、镜像或上传任何影像、标注及第三方全文 |

下一项最小操作是独立取得包内说明/许可、目录、划分与少量原始标注，先确认逐实例身份、轮廓类型、数量、图像关联与来源映射，再决定是否获取全包。数据入场、历史使用与重叠核查之后，才能预锁其评价角色；零调参直接迁移与目标训练集固定配方重训必须分别声明。当前不据这份候选记录启动新模型或修改在途实验。

## 排除一个容易误用的 mask 入口

[OpenSARShip-Ship-Detection-Dataset 作者数据卡](https://huggingface.co/datasets/sylviaHoch/OpenSARShip-Ship-Detection-Dataset) 的 Dataset Overview 明确二值掩膜来自旋转框，WebDataset 表也将 `mask.tif` 标为旋转框掩膜。它不能据此作为真实船体轮廓的实例掩膜真值用于本项目的 mask AP。此判断针对该数据卡描述的标签，不否定其检测或弱监督用途。

本轮定向检查了 2024–2026 年相关数据线索，并回溯到上述 2023 年候选；不是对所有数据集的穷尽证明。已有 HRSID 仍保留原准备状态，见 [本地 JPG 元数据核查](HRSID_LOCAL_JPG_METADATA_AND_VERSION_BOUNDARY_20261015.md)；历史 TEST 暴露见 [用途台账](HISTORICAL_TEST_USE_LEDGER_TEXT_EVIDENCE_20261016.md)。新增候选不恢复任何旧 TEST 的未接触身份，也不证明跨数据集验证已完成。
