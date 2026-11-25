# 公开船体实例参考的资格复核

## 公开来源的可用性

| 来源 | 已核实的公开资产 | 与原生船体对照相关的限制 |
| --- | --- | --- |
| SISS-YOLO / OpenSARShip 2.0 | 论文报告对 24 景 Sentinel-1A 切片绘制了 LabelMe 多边形。交大原始影像 ZIP 有公开响应。 | 论文数据声明指向原始 OpenSARShip 2.0，论文明确说明原包缺船体掩膜。检索范围未发现作者新增的原始 LabelMe 文件或采集映射。[论文数据章节](https://www.mdpi.com/2072-4292/17/17/3118#sec3dot1-remotesensing-17-03118) · [交大原始数据入口](https://opensar.sjtu.edu.cn/DataAndCodes.html) |
| SARShip-4i | 论文报告有自采像元标注和 HRSID 派生标签。作者公开代码有同名图像和掩膜加载器。 | 匿名查看和预览返回 401，官方下载跳转登录页，未取得归档字节。已核对的加载器输出船舶前景语义标签，原始逐船 ID 和图块至采集映射未核实。[论文和数据声明](https://www.mdpi.com/2072-4292/15/13/3304) · [固定版本加载器](https://github.com/imiuupload/MSG-FN/blob/2e2a35eca3a6d38ba017250484540875c6c162b5/dataset.py) |
| OSSDD | 原始 SAR TIFF、二值 TIFF、对齐和旋转检测框均由作者列为数据组成。 | 作者用旋转检测框填充二值标签，属于检测参考。[论文第 4 节](https://arxiv.org/html/2608.01963v1#S4) · [数据仓库](https://huggingface.co/datasets/sylviaHoch/OpenSARShip-Ship-Detection-Dataset) |
| PsSAR-v1.0 | 发布海陆和船舶的联合全景标注。 | 船体轮廓沿用 HRSID，新增标注属于海陆背景。[作者仓库](https://github.com/PerSARption/PsSAR-v1.0-Unlocking-SAR-Panoptic-Segmentation) |
| UEIKAP | 当前公开档案仍为 v1.0，SAR ZIP 摘要与项目已核验档案一致。 | 现有原始 108 图和补充 342 图已经使用符合既定条件的采集家族。[发行档案](https://zenodo.org/records/18391027) |
| SMART-Ship / FSS-Seg | SMART-Ship 论文报告多模态多边形。FSS-Seg 作者的 PID-SAM 归档可取，包含代码和名单。 | SMART-Ship 尚缺可核验的 SAR 原生标注归档和采集映射。PID-SAM 归档缺掩膜和原景映射。[SMART-Ship](https://arxiv.org/html/2508.02384v2#S3.SS2) · [FSS-Seg](https://www.ijcai.org/proceedings/2026/161) · [作者归档](https://github.com/Fm336/PID-SAM/blob/2764fab108304ad5da62fbb5e6c4e65aeb4ece0a/PID.zip) |

本轮未取得可以直接用于新冻结船体评分的作者掩膜。SISS-YOLO 的未增强 LabelMe 标注及切片到原采集的映射最接近缺口。它们需要通过正式的作者共享取得。Umbra 的原 12 景及补充 24 景按已锁计划分别等待独立参考。

## 补查的公开影像与原作者参考

- **FUSAR-Ship 人工描边。** Al Hinai 与 Guida 的方法节和表 3 报告 30 张高分三号 FUSAR-Ship 切片的 CVAT 人工轮廓，并报告 OpenSARShip GRD 和 SLC 各 30 张的对照。FUSAR 原包有复旦官方入口。已检查的期刊和 Surrey 档案页均无 CVAT 导出、这 30 张切片的精确名单或母景映射。论文的分类划分表无法提供轮廓切片的采集独立性。此来源目前只能记录可能取得的已有参考。即使取得，也需先核验原切片、母景和许可，再查看正编辑匹配覆盖。[论文方法和表 3](https://www.mdpi.com/2072-4292/17/1/127) · [Surrey 归档](https://openresearch.surrey.ac.uk/esploro/outputs/journalArticle/Confidence-Aware-Ship-Classification-Using-Contour-Features/991146894402346) · [复旦发行入口](https://emwlab.fudan.edu.cn/resources/list.htm)
- **Capella Maritime 影像。** 匿名读取官方 STAC 的 201 个条目链接和其中 69 个 GEC/GEO 元数据。按官方采集 ID 去重，每次采集优先保留 GEC，得到 35 次采集。筛选地距和方位分辨率均不超过 1 m 的 19 次，依次按较差分辨率、采集 UTC、采集 ID 排序。以 STAC 地理包围框无正面积相交为加入条件，贪心构造了 15 次采集的可行子集。15 个所选原生 TIFF 均有匿名 HTTP HEAD 200 并声明支持字节范围。未读取影像。供应商开放计划与对应许可链接指向 CC BY 4.0。15 次为民用范围及历史覆盖筛查之前的元数据结果，尚无船体轮廓或官方分割划分。当前未纳入冻结评价。[官方 STAC](https://capella-open-data.s3.us-west-2.amazonaws.com/stac/capella-open-data-by-use-case/capella-open-data-maritime/collection.json) · [AWS 登记](https://registry.opendata.aws/capella_opendata/) · [供应商许可](https://support.capellaspace.com/what-is-the-licensing-for-capellas-open-data)


## Instance-RSDD 74 图保留集的当时记录

GitHub 的冻结协议记录发表于 2026-11-22 18:33:19 UTC，核对提交树中协议的 Git Blob 与本地逐字节一致。这早于 18:36:43 的数据接收、18:38:56 的检测器捕获和 18:44:36 的首次评分。捕获、读回、编辑及编辑核验的自然退出按顺序完整。评分代码在独立 RGB 重复核验和冻结编辑核验之后解码选定作者标注。原始作者几何在更早阶段做过数据资格检查。记录支持这一保留集在首次评分前固定了比较方案。[冻结提交](https://github.com/bingzhi495-creator/SAR-SEAL-research/commit/8ea9023ffb48abbd71e683d223a44087f3021d5b) · [原协议](../../INSTANCE_RSDD_PARENT_RESERVE_PRELOCK_20261123.md) · [原评分](../../INSTANCE_RSDD_PARENT_RESERVE_ACTUAL_20261123.md)

74 图来自历史 740 图以外的 13 个作者父景。原核验中，74 图与 2,937 个 HRSID 检测器训练输入及历史 740 图均无完全相同解码影像，FIT109 也沿原训练序列核查。这把证据限定到原作者发行版、冻结载体及已核实的父景关系。HRSID 和 RSDD 的完整物理采集身份交叉映射仍未核实。

原公开元数据记录 5 个保留父景与历史测试共享一种传感器及日历日期，另有 5 个与同传感器的历史父景共享已发布的四舍五入中心，两类交集为 2。并用这两项筛查会排除 8 个父景，只余 **17 图、17 个原正编辑量匹配、5 个父景**，低于原预锁的 **20 个匹配和 8 个父景**。两项筛查基于历史元数据与已验收的匹配计数，未读取新几何或计算新的效果。剩余子集的物理地理或过境独立性仍需更完整的产品元数据。[原元数据汇总](../../results/INSTANCE_RSDD_PARENT_RESERVE_CENSUS_SAFE_20261123.json) · [本轮核验结果](RESULT_SAFE.json)

## 补查供应商原始产品身份

日期与中心是两种不同的筛查。逐行连接作者附表后，保留集与历史 740 图之间没有父景对同时共享传感器、公布日期和舍入中心。两个同时被两类规则标记的保留父景，各自对应的同日期和同中心历史父景不同。仅筛日期后仍保留的 42 图中，25 图来自三个 TerraSAR-X 父景。三者与 21 个历史父景共享同一公布中心，均为同规格 StripMap，日期不同。这类重复地点需要原产品覆盖核对，日期不同不足以确认地理分离。五个日期标记均为高分三号，这些来源对应需要高分三号的产品目录。[原始论文附表](https://rhhz.xml-journal.net/article/doi/10.12000/JR22007) · [Instance-RSDD 原作者发行](https://github.com/XIDIAN212Lab/SRNet#the-instance-rsdd-dataset)

实际测试了 ESA 两个 TerraSAR-X 目录服务。每个服务分别提交一个不限定时空的对照请求和三个带筛选条件的请求。筛选取原作者日期及前后各一天，取公布中心经纬各 ±0.10°。EO-CAT 四次均为请求超时。ESA MAAP 四次均返回 HTTP 200 的 GeoJSON，含零条要素，未限定时空的请求也为空。ESA 的档案说明其 TerraSAR-X 项涵盖 ESA 支持项目取得的产品，因此这组结果无法排除原产品存在。Airbus 的归档规范列出精确 UTC、采集 ID、轨道和足迹，并要求授权请求。此次没有确认任何产品对应、获取凭据或下载影像。[EO-CAT 规范](https://eocat.esa.int/eo-catalogue/features.html) · [ESA TerraSAR-X 档案范围](https://eocat.esa.int/eo-catalogue/collections/TerraSAR-X?httpAccept=text%2Fhtml) · [ESA MAAP 规范](https://catalog.maap.eo.esa.int/doc/stac.html) · [Airbus 归档](https://api.oneatlas.airbus.com/guides/radar/g-radar-api/#12-find-archive-imagery) · [实测汇总](ESA_CATALOG_SAFE.json)

中国资源卫星应用中心的已公开检索页在新的匿名会话中显示登录提示。中国遥感卫星地面站的匿名搜索表单和卫星选项均正常返回，实际显示 17 个传感器选项，未提供高分三号。该表单的通用 SAR 选项属于 ERS-2。未向该表单提交五个高分三号父景的时空查询，匹配记录数保持未知。[资源卫星应用中心](https://data.cresda.cn/#/2dMap) · [中国遥感卫星地面站检索](https://www.noda.ac.cn/rsgs/datasharing/imageFilmSearch)

当前无需重传 SAR 影像。解决已标记对应只需覆盖 39 个相关父景的作者元数据，范围为 8 个标记保留父景和 31 个涉及的历史父景。每行提供作者父景编号、原产品 ID 或文件名、原始采集 ID、UTC 起止、绝对轨道及过境方向、供应商足迹。处理未标记对应需要 125 个切片父景的完整映射。验证与 HRSID 的同地点分离还需要其对应母景表。若主张具体图块地理不重叠，另需图块偏移与像元到地理坐标变换。先锁定相同采集、确认独立和未解决的划分规则，以 UTC 与采集 ID 判定来源身份，以足迹单独判定地点隔离。覆盖计数继续使用原门槛。已评分的原 74 图和嵌套视图原样保留，任何后来取得的更严格子集只作回顾性敏感性分析。

复算 ESA 元数据探针可先用 `pdftotext -layout` 从原始 RSDD-SAR 作者 PDF 提取英文附表，再执行 `python probe_esa_parent_identity.py --appendix-text APPENDIX.txt --output ESA_CATALOG_SAFE.json`。探针只发布服务状态和条目数，原论文座标与目录返回记录不保存。目录可用性会随时间变化。[原作者 PDF](https://files.sciengine.com/pdfs/2095-283X/23A1C428928844EC809FB12C4F5A7A6B.pdf) · [复算程序](probe_esa_parent_identity.py)

复核程序 `audit_reserve_scope.py` 读取上述已关闭的评分和事件回执及原始数据资格汇总。重新执行得到字节一致的安全汇总。报告和输出不含影像、坐标清单、标注和候选。
