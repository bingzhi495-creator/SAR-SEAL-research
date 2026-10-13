# HRSID 官方协议与发布范围定向核查（2026-10-13）

范围：先核对独立发布副本的 DATA_ACCESS.md 与 upstream_data_index.json，仅补协议缺口。已获取原论文公开全文文本，定向阅读 §II-A～C；IEEE 入口受访问验证限制，未取得 PDF 或逐页图像核验。论文为 Wei 等，IEEE Access 8，120234–120254，2020。[正式 DOI](https://doi.org/10.1109/ACCESS.2020.3005861)、[公开原论文文本](https://www.researchgate.net/publication/342541002_HRSID_A_High-Resolution_SAR_Images_Dataset_for_Ship_Detection_and_Instance_Segmentation)。

新增原文依据：摘要（120234）由136幅全景生成5604张800×800裁片，§II-C（120240）按图像数量给65/35训练测试比例；相关段落没有说明按原全景互斥划分。§II-A（120238）文本写双向步长600，却称重叠20%，摘要称25%；按单轴窗口定义推算为25%，但此文本差异尚未经 PDF 图像复核。该节称筛出400张纯背景裁片用于鲁棒性测试，没有交代全部无船图的筛除规则。§II-B（120239，图4/5）先在全景标多边形，再裁出实例掩膜并生成 COCO 标注和框；这不能证明训练测试不存在同源或重叠裁片。[原论文 §II](https://www.researchgate.net/publication/342541002_HRSID_A_High-Resolution_SAR_Images_Dataset_for_Ship_Detection_and_Instance_Segmentation)。

作者说明将 PNG 称为占空间更多的高保真版本，将近岸/离岸版本描述为新增场景标记，并另列400张纯背景测试图。它没有明确保证各包图像、标注和划分逐项等价；本次未打开任何数据包，现有版本差异记录继续有效。[作者数据说明](https://github.com/chaozhong2010/HRSID#dataset-download-link)。

结论仍有边界：65/35裁片比例不足以支持“原场景独立”，也不足以证明实际泄漏。先在全景统一标注能减少重复标注冲突，不能替代划分独立性检查。纯背景补充应单列测试用途和空预测规则，不能无记录并入训练或改动官方基准。

许可仍未解决：仓库 [LICENSE](https://github.com/chaozhong2010/HRSID/blob/master/LICENSE) 为 GPL-3.0，论文注明 CC BY 4.0；现有说明未明确把这些许可映射到外链影像和全量标注。可下载不等于已确认可再发布；当前仅发布入口与自写协议摘要。

后续冻结前必须检验：选定包及图像编码；核对图像 ID、实例 mask/框、训练测试清单与版本对应；建立原场景映射并检查跨划分同源、重叠及同船重复，官方划分和自建场景划分分别命名；单列纯背景补充；扩大发布前取得明确覆盖影像和标注的许可证据。本次核查未读取本地数据、科学载荷或绑定 SOURCE，未执行实验或修改48轮方案；普通摘要的公开同步由主任务完成。


## 2026-10-14：固定作者仓库补核，两个缺口仍未解决

独立代理实际核对作者master固定[提交2d682fe0c023df03a8fe1b1bf2e2ac76811e43b5](https://github.com/chaozhong2010/HRSID/tree/2d682fe0c023df03a8fe1b1bf2e2ac76811e43b5)、[README](https://github.com/chaozhong2010/HRSID/blob/2d682fe0c023df03a8fe1b1bf2e2ac76811e43b5/README.md)与[GPL-3.0 LICENSE](https://github.com/chaozhong2010/HRSID/blob/2d682fe0c023df03a8fe1b1bf2e2ac76811e43b5/LICENSE)。没有找到明确映射到外链影像与全量实例标注的条款；不能断言该LICENSE只适用于代码，也不能据此推定覆盖所有外链数据。新增primary索引是作者仓库[Dataset License issue #3](https://github.com/chaozhong2010/HRSID/issues/3)：2025-04-09有人询问数据许可，独立代理于2026-10-13T17:06:54Z实际查询仍open、comments=0且评论响应为[]；主任务随后打开页面核对请求仍未关闭、页面无作者回答。该缺失证据不证明不存在许可，只说明当前未得到作者确认。

独立代理读取[该提交对应递归树](https://api.github.com/repos/chaozhong2010/HRSID/git/trees/f8dc521078221ab0a0eba16a0f57a543e0564bfe?recursive=1)，20项、truncated=false，未见单独的原场景到裁片映射或划分脚本。主任务与独立代理均实际打开[data/readme.txt](https://github.com/chaozhong2010/HRSID/blob/2d682fe0c023df03a8fe1b1bf2e2ac76811e43b5/data/readme.txt)：它描述旧ISSID，数字与当前HRSID说明不同，不能用来认证当前包版本或场景划分。没有打开标注正文、影像/PNG/近岸包，不能声称数据包内缺少映射；疑似坐标文件名前缀也不能自动当原场景ID。

既定下一步不变：继续公开官方入口；65/35按裁片协议描述，场景独立与实际同源重叠须在以后选定版本的数据核验中分别确认。尚未确认许可和场景互斥不能表述为实际违规或泄漏，也不阻止当前内部N13固定评价继续执行。本次仅补primary来源及证据边界，没有发issue、联系作者、读本地数据或取得跨数据集结果。
