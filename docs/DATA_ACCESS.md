# 数据获取、标注版本与发布范围

更新：2026-10-15；初次公开说明核对为2026-10-12，HRSID同包标注实核见下方新记录。此源码快照提供作者下载入口与本项目转换代码，**不包含 SSDD/HRSID 原始影像或全量标注，不等于数据集已上传**。10月15日已独立读取并核验隔离HRSID官方PNG目录和TRAIN/TEST JSON的字节、CRC、结构与文件名关系；没有读取影像像素、解码mask或计算跨集AP，也未把它们加入运行中的N20 Sources。

## PSeg-SSDD

从 [Official-SSDD 作者仓库](https://github.com/TianwenZhang0825/Official-SSDD) 获取：[Google Drive](https://drive.google.com/file/d/1glNJUGotrbEyk43twwB9556AdngJsynZ/view?usp=sharing)；[百度网盘](https://pan.baidu.com/s/1Lpg28ZvMSgNXq00abHMZ5Q)，提取码 `2021`。链接来自作者 README，未在本次审阅中重新下载整个压缩包。

实例分割需要 **PSeg-SSDD 的船舶多边形**。BBox-SSDD 是水平检测框，RBox-SSDD 是旋转框；检测框不能冒充真实船体轮廓。官方版本提供 VOC/COCO 组织形式，公开训练/测试为 928/232 张，共 1160 张；文件编号末位为 1 或 9 的图像用于官方测试。详见 [原始数据论文 §3.3、§4、§6.1 与表6](https://www.mdpi.com/2072-4292/13/18/3690)。本研究的 FIT741/DEV187 是训练侧内部划分，不能标为官方 TEST232。

仓库 [LICENSE](https://github.com/TianwenZhang0825/Official-SSDD/blob/main/LICENSE) 为 Apache-2.0。核对到的 GitHub 仓库本身只有 README 与 LICENSE；数据压缩包由外部网盘提供。本次未核对压缩包内的权利说明，因此**尚未确认该仓库许可对每个外部影像文件的覆盖范围**，也不将论文的 CC-BY 许可自动套用到数据影像。当前采用上游获取方式，保留来源与许可入口；这不是“作者禁止再分发”的结论。

本仓库已有准备入口：

1. 下载并选择 PSeg 的 VOC 版本，在独立工作副本中设置 [protocol/datasets.py](../protocol/datasets.py) 的 `PSEG_ROOT`。该解析器要求每个 XML 对象有 `<segm><point>x,y</point>…</segm>`，至少三个点；输入目录含 `JPEGImages_train`、`JPEGImages_test`、`Annotations_train`、`Annotations_test` 及测试近岸/远海目录。
2. 从仓库根目录执行 `python protocol/pseg_to_coco.py`，生成 `data/pseg_ssdd/annotations/` 下的 train/test 及近岸/远海 COCO 文件，影像仍留在原位置。入口见 [转换器](../protocol/pseg_to_coco.py)。
3. 转换器会跳过缺失 XML，并忽略无有效多边形的对象。必须核验实际输出数量、漏标与图像路径；转换完成本身不证明完整性。此历史转换器不生成当前 FIT741/DEV187 的锁定清单，不能用随意重划分代替原协议。

## HRSID

从 [HRSID 作者仓库](https://github.com/chaozhong2010/HRSID) 获取；作者说明原始基准有5604张图像、16951个实例，并另行发布不同数据版本。这是作者报告口径。2026-10-15实读同包PNG TRAIN/TEST得到合计5604图、16969实例，不能把报告数字与实际版本混写；差异原因尚未确认。所选版本应在实验中明确。

| 版本 | 作者下载入口 | 使用说明 |
|---|---|---|
| JPG 基准包 | [Google Drive](https://drive.google.com/open?id=1BZTU8Gyg20wqHXtBPFzRazn_lEdvhsbE)；[百度网盘](https://pan.baidu.com/s/1z0-E3Lwdkg14feu1FwRf3g)，提取码 `bs64` | 不能与 PNG 包混称相同影像编码 |
| 增加近岸/远海标记的版本 | [Google Drive](https://drive.google.com/file/d/1NY3ovgc-woDlNoQdyqzRB3t9McOBH5Ms/view?usp=sharing) | 标记版本与划分分别记录 |
| PNG 版本 | [Google Drive](https://drive.google.com/file/d/1xgXi8KC3MDWuu7Yp4n2J-LOPYLRszHAc/view?usp=sharing) | 作者单独提供的影像版本 |
| 400 张纯背景补充图 | [Google Drive](https://drive.google.com/file/d/1U0Sj1SHoq-2VjXXUKwpXae6rBI3YjyDP/view?usp=sharing) | 独立鲁棒性补充，不能静默并入原基准 |

以上均来自 [作者 README](https://github.com/chaozhong2010/HRSID/blob/master/README.md)。引用数据论文：[Wei et al., HRSID, IEEE Access (2020)](https://ieeexplore.ieee.org/document/9127939)。

HRSID仓库有[GPL-3.0 LICENSE](https://github.com/chaozhong2010/HRSID/blob/master/LICENSE)。所核[固定作者提交的annotations目录](https://github.com/chaozhong2010/HRSID/tree/2d682fe0c023df03a8fe1b1bf2e2ac76811e43b5/annotations)只列[train2017.json](https://github.com/chaozhong2010/HRSID/blob/2d682fe0c023df03a8fe1b1bf2e2ac76811e43b5/annotations/train2017.json)，未同时列test2017.json。2026-10-15已实际读取此候选：11592317B、3573图/9870标注，215图无标注引用；它不等于同包PNG TRAIN。测试标注已另从官方PNG包完整取得并核CRC，不把Git目录当完整镜像。仓库标注引用保留其GPL来源，外部包许可覆盖仍另行核验。

仓库许可与**外部网盘影像压缩包**的具体覆盖关系，在本次查到的 README/LICENSE 中没有单独说明；本次未核验压缩包内文件级说明。因此此快照链接上游标注与影像，不附加本项目自定许可，也不声称完整原图再分发已获核实。上游入口的机器可读索引见 [upstream_data_index.json](upstream_data_index.json)。

已有准备与检查入口：

- [protocol/datasets.py](../protocol/datasets.py) 使用 `annotations/test2017.json` 与 `JPEGImages1/`；历史 [HRSID 模型配置](../carrier/configs/a0_m2f_r50_hrsid.py) 则期望本地 `official_train.json`、`official_test.json`。使用时应从同一所选官方包准备这些输入、保留其图像 ID/标注关系并改好路径；这里没有提供经过当前实验验证的一键准备器。
- [protocol/audit_hrsid.py](../protocol/audit_hrsid.py) 根据芯片文件名检查场景来源和跨划分重叠，并提出场景划分；需先设置其 `ROOT`。脚本注释中的历史数字不作为本次重新核验的结论。
- [protocol/hrsid_scene_split.py](../protocol/hrsid_scene_split.py) 使用前一步生成的场景清单，合并时偏移测试 ID 后另写 `scene_train.json`/`scene_test.json`。**它不会创建或重写 official_train/test.json**。场景级自建协议需独立命名、独立报告，不替代官方协议。
- [tools/make_train_holdout.py](../tools/make_train_holdout.py) 是历史训练侧子集工具，不自动产生当前 PSeg 锁定 FIT/DEV；其历史训练已见子集也不能冒充独立留出集。

## 此次 GitHub 数据范围

提交获取说明、转换/审计源码和协议说明；保留本地原图、标注、权重与运行缓存。未将“数据有官方公开链接”表述为“数据已经随仓库上传”。后续如建立原图镜像，需要针对所选实际压缩包核对许可范围、保留原许可与署名，并记录版本与校验清单。

大文件也需采用适当载体：GitHub 普通 Git 对超过 50 MiB 的文件告警，阻止超过 100 MiB 的文件；超限文件需 Git LFS 或其他合适分发方式。依据 [GitHub 官方大文件说明](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)，不能直接把大数据压缩包或整个实验缓存当普通源码推送。

## 2026-10-13 / HRSID原论文与作者说明补核

[定向核查记录](../research_round_20260925_global/HRSID_OFFICIAL_PROTOCOL_AND_RELEASE_SCOPE_CHECK_20261013.md)补充原论文公开文本 §II-A～C 与作者README依据。65/35指图像数量，尚未确认原场景互斥；公开文本的20%/25%重叠表述差异仍待PDF图像复核，不据此断言实际泄漏。JPG/PNG、近岸/离岸标记和400纯背景图须按所选版本逐项核对。仓库GPL-3.0与论文CC BY 4.0未明确映射至外链影像和全量标注，暂保持官方获取入口，不新增数据镜像。此补核未读取任何本地数据包或执行实验。


## 2026-10-15：官方PNG同包TRAIN/TEST实际版本

[完整来源、失败记录与Root独立核验](../research_round_20260925_global/HRSID_PNG_OFFICIAL_ARCHIVE_DIRECTORY_AND_TEST_METADATA_20261015_01.md)。同一作者PNG归档的完整目录包含5604个PNG；完整TRAIN/TEST经过范围/长度/SHA、raw-deflate/CRC与JSON结构核验，未下载整包或打开图像像素。

| 成员 | 图像 | 标注 | 原文件字节 | CRC32 | SHA256 |
|---|---:|---:|---:|---|---|
| annotations/train2017.json | 3642 | 11047 | 15641078 | 18c7bae6 | 3e240c38014526e9b93206a946105fb41c808272238932b0d4c2422978d5116f |
| annotations/test2017.json | 1962 | 5922 | 8608088 | 18a3bf81 | eb611754afc74c34222fa52cc04a500d1b84f956782fcbcc0e91d9766c4ac937 |

文件名交集0、并集5604，逐项对应同包images/目录；均800×800、单类ship。TRAIN无标注关联图0、TEST为1。这里只确认标注结构与目录关系，没有核mask面积/拓扑或像素。

**跨split ID复用必须处理。** imageID交集1962、annotationID交集5922，不能据ID相同判图像重复或按裸ID直接合并。分别加载两个split；若另建合并数据，按(split,id)建立明确映射并同步annotation.image_id，保存旧ID，不改原JSON。文件名无交集也不证明母景或像素独立。

Git候选与PNG TRAIN共同文件名3572，候选独有1、PNG TRAIN独有70，候选与PNG TEST文件名交集0。不是简单子集，不能按差69补图、改后缀/前缀或混配标注。历史解析器JPEGImages1/与本PNG包images/也不能静默当相同路径或编码。

本机一次有界历史目录盘点仅确认358个.jpg普通文件、总66018309B，与官方PNG全文件名精确匹配0；未打开/hash图像。它不能认证完整基准或像素版本，也不构成全磁盘不存在其他影像的结论。仍需取得或定位配套原影像、实际mask解码/转换核验、审计历史TEST与预训练/选型使用，并明确母景及许可范围。没有跨集AP或盲测资格恢复。

原JSON的licenses条目name/url均null；这不补足外部影像/标注的具体再发布许可。此仓库只保存自写获取说明与元数据校验信息，不上传原JSON、完整归档、图像、下载私有配置或临时签名URL。


## 2026-10-15：官方PNG编码文件获取试点

[两文件试点](../research_round_20260925_global/HRSID_OFFICIAL_PNG_ENCODED_IMAGE_BYTES_PILOT_20261015_01.md)已从同包TRAIN/TEST各固定首名取得完整PNG编码文件，Root独立核文件SHA、ZIP和PNG全部chunk CRC及800×800 RGB8 IHDR。没有解码IDAT/像素、核真实mask对齐、运行模型或取得全部5604图；与上方历史358个JPG目录盘点分开记录。不用两个试点替代完整跨集基准，不把本次获取恢复成盲测资格。原PNG不上传Git。


## 2026-10-15：完整影像获取仍未完成

[实际传输及停止记录](../research_round_20260925_global/HRSID_PNG_OFFICIAL_ARCHIVE_DIRECTORY_AND_TEST_METADATA_20261015_01.md)：单请求8MiB范围控制通过，Root核body/SHA/关键HTTP头与计数；随后fresh整包获取两块完整（16MiB有效前缀），第三块在45秒超时，原会话退出1，失败partial不accepted。该批共3请求/18491347B实际body，按规则停止、无自动重试。没有取得完整ZIP或解压新增影像，也没有像素/mask/AP资格；前述两个PNG文件的容器字节核验仍仅限两文件。原归档及所有分块/私有配置不上传Git。


## 2026-10-15：另立规则02续取，完整数据仍pending

[新任务及首块实证](../research_round_20260925_global/HRSID_PNG_OFFICIAL_ARCHIVE_DIRECTORY_AND_TEST_METADATA_20261015_01.md)保留旧超时和失败partial。仅重核并复用两完整块；新规则每8MiB允许300秒/owned310秒，最多678新请求、5681525099B与12小时，无自动重试。首个新8MiB块实际198.735秒完成且Root核物理SHA/长度，任务仍在运行，尚无完整ZIP、全成员CRC或新增影像/mask/AP资格。延长超时减少这一块被过早截断，不证明提速，也不改变N20。


## 2026-10-15：规则02实际停止与官方备用候选

[完整获取记录](../research_round_20260925_global/HRSID_PNG_OFFICIAL_ARCHIVE_DIRECTORY_AND_TEST_METADATA_20261015_01.md)：规则02共7新请求/57712075B body，有效连续前缀64MiB；第7新块在300秒超时，只收到7380427B，失败partial不accepted。Root核终态及全部新body和64MiB前缀；整包、目录复核及全CRC仍pending，未启动第三轮整包获取。固定作者README另列百度vxrs候选但未明确PNG/同版；本机唯一公开GET只返回302至提取码页，未识别包名。bs64入口明确为JPG，不能替代本PNG版本；所有链接与公开提取码的来源见上文记录。没有登录、绕验证码或使用非公开API，也没有恢复盲测/跨集资格。


## 用户指定历史工作区中的本地JPG候选（2026-10-15）

已定位5604个JPG文件名，与train3642图/11047实例、test1962图/5922实例的原COCO polygon逐字完整对应；Root另核两JSON SHA与名称集合。原合并标注5604图/16951实例不等两split合计16969，按文件名和几何字段忽略重编号ID有22项仅split/4项仅merged，不能互换或静默纠正。详见[本地数据评议](../research_round_20260925_global/HRSID_LOCAL_JPG_METADATA_AND_VERSION_BOUNDARY_20261015.md)。

本次没有读取影像/归档字节或解码mask，实际尺寸/完整性/空间对齐、官方JPG归档身份、许可/母景及历史测试使用仍须核验。可以另锁JPG跨集协议，但不能冒充此前PNG身份或恢复盲TEST；当前N20数据不变。旧语义模型与跨集测试使用历史见[资产总评](../research_round_20260925_global/UNETPP_HISTORICAL_ASSETS_SYNTHESIS_AND_REUSE_PLAN_20261015.md)。只发布自写说明，不发布原图、标注或权重。


## 本地JPG的后续证据更新（2026-10-15）

本地train/test与官方PNG split的全部实例标注字段按stem/ID精确一致；5604文件编码/头部检查已实际完成，均800×800三分量JPEG，但未解码像素或mask。完整编码SHA发现两组train/test重复和一组test内部重复，各组标注多重集不同。原数据及划分不改；另行预锁去重敏感性补充。不能据此认证JPG官方归档身份、像素/mask对齐、母景独立或N20影响。[可核对的补充记录](../research_round_20260925_global/HRSID_LOCAL_JPG_METADATA_AND_VERSION_BOUNDARY_20261015.md)。
