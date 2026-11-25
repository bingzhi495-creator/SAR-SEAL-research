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

## Instance-RSDD 74 图保留集的当时记录

GitHub 的冻结协议记录发表于 2026-11-22 18:33:19 UTC，核对提交树中协议的 Git Blob 与本地逐字节一致。这早于 18:36:43 的数据接收、18:38:56 的检测器捕获和 18:44:36 的首次评分。捕获、读回、编辑及编辑核验的自然退出按顺序完整。评分代码在独立 RGB 重复核验和冻结编辑核验之后解码选定作者标注。原始作者几何在更早阶段做过数据资格检查。记录支持这一保留集在首次评分前固定了比较方案。[冻结提交](https://github.com/bingzhi495-creator/SAR-SEAL-research/commit/8ea9023ffb48abbd71e683d223a44087f3021d5b) · [原协议](../../INSTANCE_RSDD_PARENT_RESERVE_PRELOCK_20261123.md) · [原评分](../../INSTANCE_RSDD_PARENT_RESERVE_ACTUAL_20261123.md)

74 图来自历史 740 图以外的 13 个作者父景。原核验中，74 图与 2,937 个 HRSID 检测器训练输入及历史 740 图均无完全相同解码影像，FIT109 也沿原训练序列核查。这把证据限定到原作者发行版、冻结载体及已核实的父景关系。HRSID 和 RSDD 的完整物理采集身份交叉映射仍未核实。

原公开元数据记录 5 个保留父景与历史测试共享一种传感器及日历日期，另有 5 个与同传感器的历史父景共享已发布的四舍五入中心，两类交集为 2。并用这两项筛查会排除 8 个父景，只余 **17 图、17 个原正编辑量匹配、5 个父景**，低于原预锁的 **20 个匹配和 8 个父景**。两项筛查基于历史元数据与已验收的匹配计数，未读取新几何或计算新的效果。剩余子集的物理地理或过境独立性仍需更完整的产品元数据。[原元数据汇总](../../results/INSTANCE_RSDD_PARENT_RESERVE_CENSUS_SAFE_20261123.json) · [本轮核验结果](RESULT_SAFE.json)

复核程序 `audit_reserve_scope.py` 读取上述已关闭的评分和事件回执及原始数据资格汇总。重新执行得到字节一致的安全汇总。报告和输出不含影像、坐标清单、标注和候选。
