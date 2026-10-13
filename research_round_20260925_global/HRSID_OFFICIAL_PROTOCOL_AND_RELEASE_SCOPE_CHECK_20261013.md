# HRSID 官方协议与发布范围定向核查（2026-10-13）

范围：先核对独立发布副本的 DATA_ACCESS.md 与 upstream_data_index.json，仅补协议缺口。已获取原论文公开全文文本，定向阅读 §II-A～C；IEEE 入口受访问验证限制，未取得 PDF 或逐页图像核验。论文为 Wei 等，IEEE Access 8，120234–120254，2020。[正式 DOI](https://doi.org/10.1109/ACCESS.2020.3005861)、[公开原论文文本](https://www.researchgate.net/publication/342541002_HRSID_A_High-Resolution_SAR_Images_Dataset_for_Ship_Detection_and_Instance_Segmentation)。

新增原文依据：摘要（120234）由136幅全景生成5604张800×800裁片，§II-C（120240）按图像数量给65/35训练测试比例；相关段落没有说明按原全景互斥划分。§II-A（120238）文本写双向步长600，却称重叠20%，摘要称25%；按单轴窗口定义推算为25%，但此文本差异尚未经 PDF 图像复核。该节称筛出400张纯背景裁片用于鲁棒性测试，没有交代全部无船图的筛除规则。§II-B（120239，图4/5）先在全景标多边形，再裁出实例掩膜并生成 COCO 标注和框；这不能证明训练测试不存在同源或重叠裁片。[原论文 §II](https://www.researchgate.net/publication/342541002_HRSID_A_High-Resolution_SAR_Images_Dataset_for_Ship_Detection_and_Instance_Segmentation)。

作者说明将 PNG 称为占空间更多的高保真版本，将近岸/离岸版本描述为新增场景标记，并另列400张纯背景测试图。它没有明确保证各包图像、标注和划分逐项等价；本次未打开任何数据包，现有版本差异记录继续有效。[作者数据说明](https://github.com/chaozhong2010/HRSID#dataset-download-link)。

结论仍有边界：65/35裁片比例不足以支持“原场景独立”，也不足以证明实际泄漏。先在全景统一标注能减少重复标注冲突，不能替代划分独立性检查。纯背景补充应单列测试用途和空预测规则，不能无记录并入训练或改动官方基准。

许可仍未解决：仓库 [LICENSE](https://github.com/chaozhong2010/HRSID/blob/master/LICENSE) 为 GPL-3.0，论文注明 CC BY 4.0；现有说明未明确把这些许可映射到外链影像和全量标注。可下载不等于已确认可再发布；当前仅发布入口与自写协议摘要。

后续冻结前必须检验：选定包及图像编码；核对图像 ID、实例 mask/框、训练测试清单与版本对应；建立原场景映射并检查跨划分同源、重叠及同船重复，官方划分和自建场景划分分别命名；单列纯背景补充；扩大发布前取得明确覆盖影像和标注的许可证据。本次核查未读取本地数据、科学载荷或绑定 SOURCE，未执行实验或修改48轮方案；普通摘要的公开同步由主任务完成。
