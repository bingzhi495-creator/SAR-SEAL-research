# HRSID 官方标注实文件获取与元数据检查

日期：2026-10-15。**取得作者固定提交中的 train 候选JSON实文件；未确认完整原版HRSID划分，测试标注尚未取得。** 本次只用标准库网络读取、JSON 与元数据检查，未读任何图像像素、本机既有 HRSID 样本、科学载荷或绑定来源；未导入 Torch/NumPy/OpenCV/pycocotools，未解码 mask、训练或计算 AP。它补充实际文件证据，不替代原协议和许可边界记录。

## 来源与固定身份

先完整阅读三份独立发布文档：`docs/HRSID_split_provenance_review_20261012.md`、`docs/HRSID_external_protocol_readiness_20261012.md`、`research_round_20260925_global/HRSID_OFFICIAL_PROTOCOL_AND_RELEASE_SCOPE_CHECK_20261013.md`。Root 已确认这三个普通路径不在当前绑定集合；本次没有访问该集合或沿文档引用读取本机科学文件。

使用作者仓库既有固定提交 **`2d682fe0c023df03a8fe1b1bf2e2ac76811e43b5`**，而不是声称已确认当前 master。未登录 GitHub API 获取最新提交返回403 rate limit；没有使用凭据或代理。下列实文件从该提交的官方 raw URL直接取得：

| 文件 | 实际字节 | SHA-256 |
|---|---:|---|
| [annotations/train2017.json](https://raw.githubusercontent.com/chaozhong2010/HRSID/2d682fe0c023df03a8fe1b1bf2e2ac76811e43b5/annotations/train2017.json) | 11,592,317 | `4ac69031accd771b784f90e128f206373f7332293a0130324333154d723541fc` |
| [README.md](https://raw.githubusercontent.com/chaozhong2010/HRSID/2d682fe0c023df03a8fe1b1bf2e2ac76811e43b5/README.md) | 4,315 | `c709d2ee306dec35ab798e3de8298d3ae0243cbc9ab6f74bfe51c0b4252e5ed8` |
| [LICENSE](https://raw.githubusercontent.com/chaozhong2010/HRSID/2d682fe0c023df03a8fe1b1bf2e2ac76811e43b5/LICENSE) | 35,149 | `3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986` |

仅隔离保存在 `E:/SAR/seal/tmp/HRSID_OFFICIAL_ANNOTATION_INTAKE_20261015_01/`；其中`DOWNLOAD_RECEIPTS.json`记录URL/字节/SHA与失败请求，`ANNOTATION_METADATA_RESULT.json`保存本次元数据计数。**不发布原始JSON、README/许可证副本或其他下载文件，不把此目录加入当前科学Source链。** 上表官方响应合计11,631,781B；另读取目录HTML206,328B及不超过131,072B的官方Drive页面HTML，合计上限11,969,181B，未下载图像或完整数据压缩包，满足本轮100MB上限。

## JSON 实际结构与计数

顶层恰为`categories`、`images`、`annotations`，三项均为list。以下数字来自所取得文件，不从论文比例推算：

| 检查 | 实际结果 |
|---|---|
| train图像 / annotation / category | **3,573 / 9,870 / 1**；类别`id=1,name=ship` |
| 被annotation引用的不同image_id / 无annotation的images | **3,358 / 215**；只说明本JSON的引用关系，不证明真实无船、漏标或原包过滤原因 |
| image/annotation/category ID类型与重复 | 全为实际int；三类ID各自重复均0 |
| 文件名重复 / annotation孤立image或category引用 | 均0 |
| images字段 | 全部为`file_name,height,id,ins_file_name,seg_file_name,width` |
| annotations字段 | 全部为`area,bbox,category_id,category_name,id,image_id,iscrowd,segmentation` |
| 图像尺寸元数据 | 3,573条均有正整数width/height，全部800×800 |
| 图像文件名后缀 | **全部`.png`**；未读图像，未核实任何文件是否存在 |
| segmentation格式 | 9,870条均为非空list形式的COCO polygon集合，共9,881个轮廓；没有RLE dict |
| polygon结构检查 | 非list轮廓、奇数坐标长度、少于3顶点、非有限/非数坐标、越出对应图像闭区间范围的轮廓，均0 |
| 其余有限结构检查 | iscrowd全部0；bbox均为4个有限数值，没有缺失/结构异常 |

polygon自身没有RLE式`size`字段，解码画布可由关联image的完整800×800元数据获得；这**不是实际mask尺寸、面积、非空性或拓扑正确的证明**。数值面积与解码面积、实际像素对应、轮廓合法性及COCO实现行为仍须后续独立mask decode资格核验。本轮没有把bbox作为mask。

## 尚未取得的测试文件与协议边界

- 按Root要求只追加一次[固定提交annotations目录网页](https://github.com/chaozhong2010/HRSID/tree/2d682fe0c023df03a8fe1b1bf2e2ac76811e43b5/annotations)核查：标准库请求实际HTML200、206,328B，SHA-256为`d12d6b2ec9860e5b4e25f6a569121a2853acb4cc35a6e4c468296d5e0dc4cfc3`。页面内两处GitHub结构化目录payload均给出`totalCount=1`，唯一条目`annotations/train2017.json`；对应blob链接也只有这一项。因此本固定提交的**该目录**没有另名`instances_test2017.json`或`instances_val2017.json`。这不等于全仓或外链RAR内没有TEST；未盲猜批量下载其他文件。
- [同提交的annotations/test2017.json直链](https://raw.githubusercontent.com/chaozhong2010/HRSID/2d682fe0c023df03a8fe1b1bf2e2ac76811e43b5/annotations/test2017.json)实际返回**HTTP404**。这只说明该固定路径不可取，不能证明官方整包没有测试标注。
- README给出的[原版Google Drive入口](https://drive.google.com/file/d/1BZTU8Gyg20wqHXtBPFzRazn_lEdvhsbE/view)通过禁用代理的标准库请求实际返回HTML200，页面标题为`HRSID_JPG.rar - Google Drive`。本轮没有下载RAR，也未取得其内部目录或测试JSON；不能认证其内容、大小、校验和或与上述PNG文件名清单等价。
- **test图像/annotation/category数量以及train/test image-ID、annotation-ID、file_name交集全部未知，不能填0，也不由总数相减补造test。** 本仓库单JSON的3573图与论文5604图、约65/35叙述不能直接拼成已核的同版train/test；需确认仓库标注对应的PNG发布版本，以及其与原版JPG包的逐项关系。不得擅自换后缀、补图或重切划分。
- 已完整阅读该固定README。它只另列占用更多空间的高保真PNG发布物，**没有说明JPG/PNG转换流程、该JSON对应哪个包、3573计数，或样本缺失/剔除原因**；不能仅据后缀把此JSON认证为某一外链PNG包的完整官方划分。
- 没有取得逐切片到原始采集/母场景的权威映射；文件名前缀不能自动充当采集身份。母场景互斥和实际跨侧重叠仍未知，既不证明独立泛化，也不证明已发生泄漏。

许可证实文件已取得并读取，确认GPL-3.0标题及其通用定义条款；文件未给出HRSID、dataset或annotations的项目专属适用范围声明。GPL通用定义本身并非“仅源码”的证明，**不能断言只覆盖代码，也不能据此断言必然涵盖全部外链影像与标注**；README同样没有解决这一映射。可下载不等于已确认可再发布，继续仅发布官方入口与自写摘要。此次新路径与下载不恢复盲TEST：本轮未审计项目历史接触、选型、训练/预训练来源，不能据此认证模型从未见过HRSID。后续在取得并锁定同一版本的两侧清单、实际影像和mask decode资格、核查历史使用并冻结外部方案前，仍无HRSID外部AP或零调参泛化结论。

## Root独立复核与用途

Root在独立隔离文件上重新计算实际字节与SHA-256，并用内置JSON解析独立核对3573图/9870标注/1类、3358有引用+215无标注、全部PNG与800×800元数据、ID/文件名唯一性、引用完整性和9881条polygon轮廓结构/有限坐标范围，结果与上表一致。未读图像、未解mask、未加入当前科学Source或给N20增加训练/评价数据。这里只补足“该固定候选JSON实际是什么”的证据；其与图像包和完整两侧划分的对应关系仍待确认。
