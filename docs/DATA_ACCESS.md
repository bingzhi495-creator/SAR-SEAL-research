# 数据获取、标注版本与发布范围

核对日期：2026-10-12。此源码快照提供作者下载入口与本项目转换代码，**不包含 SSDD/HRSID 原始影像或全量标注，不等于数据集已上传**。本次只审阅公开说明与源码，没有读取本地科学数据、资格记录或运行结果。

## PSeg-SSDD

从 [Official-SSDD 作者仓库](https://github.com/TianwenZhang0825/Official-SSDD) 获取：[Google Drive](https://drive.google.com/file/d/1glNJUGotrbEyk43twwB9556AdngJsynZ/view?usp=sharing)；[百度网盘](https://pan.baidu.com/s/1Lpg28ZvMSgNXq00abHMZ5Q)，提取码 `2021`。链接来自作者 README，未在本次审阅中重新下载整个压缩包。

实例分割需要 **PSeg-SSDD 的船舶多边形**。BBox-SSDD 是水平检测框，RBox-SSDD 是旋转框；检测框不能冒充真实船体轮廓。官方版本提供 VOC/COCO 组织形式，公开训练/测试为 928/232 张，共 1160 张；文件编号末位为 1 或 9 的图像用于官方测试。详见 [原始数据论文 §3.3、§4、§6.1 与表6](https://www.mdpi.com/2072-4292/13/18/3690)。本研究的 FIT741/DEV187 是训练侧内部划分，不能标为官方 TEST232。

仓库 [LICENSE](https://github.com/TianwenZhang0825/Official-SSDD/blob/main/LICENSE) 为 Apache-2.0。核对到的 GitHub 仓库本身只有 README 与 LICENSE；数据压缩包由外部网盘提供。本次未核对压缩包内的权利说明，因此**尚未确认该仓库许可对每个外部影像文件的覆盖范围**，也不将论文的 CC-BY 许可自动套用到数据影像。当前采用上游获取方式，保留来源与许可入口；这不是“作者禁止再分发”的结论。

本仓库已有准备入口：

1. 下载并选择 PSeg 的 VOC 版本，在独立工作副本中设置 [protocol/datasets.py](../protocol/datasets.py) 的 `PSEG_ROOT`。该解析器要求每个 XML 对象有 `<segm><point>x,y</point>…</segm>`，至少三个点；输入目录含 `JPEGImages_train`、`JPEGImages_test`、`Annotations_train`、`Annotations_test` 及测试近岸/远海目录。
2. 从仓库根目录执行 `python protocol/pseg_to_coco.py`，生成 `data/pseg_ssdd/annotations/` 下的 train/test 及近岸/远海 COCO 文件，影像仍留在原位置。入口见 [转换器](../protocol/pseg_to_coco.py)。
3. 转换器会跳过缺失 XML，并忽略无有效多边形的对象。必须核验实际输出数量、漏标与图像路径；转换完成本身不证明完整性。此历史转换器不生成当前 FIT741/DEV187 的锁定清单，不能用随意重划分代替原协议。

## HRSID

从 [HRSID 作者仓库](https://github.com/chaozhong2010/HRSID) 获取；作者说明原始基准有 5604 张图像、16951 个实例，并另行发布不同数据版本。所选版本应在实验中明确。

| 版本 | 作者下载入口 | 使用说明 |
|---|---|---|
| JPG 基准包 | [Google Drive](https://drive.google.com/open?id=1BZTU8Gyg20wqHXtBPFzRazn_lEdvhsbE)；[百度网盘](https://pan.baidu.com/s/1z0-E3Lwdkg14feu1FwRf3g)，提取码 `bs64` | 不能与 PNG 包混称相同影像编码 |
| 增加近岸/远海标记的版本 | [Google Drive](https://drive.google.com/file/d/1NY3ovgc-woDlNoQdyqzRB3t9McOBH5Ms/view?usp=sharing) | 标记版本与划分分别记录 |
| PNG 版本 | [Google Drive](https://drive.google.com/file/d/1xgXi8KC3MDWuu7Yp4n2J-LOPYLRszHAc/view?usp=sharing) | 作者单独提供的影像版本 |
| 400 张纯背景补充图 | [Google Drive](https://drive.google.com/file/d/1U0Sj1SHoq-2VjXXUKwpXae6rBI3YjyDP/view?usp=sharing) | 独立鲁棒性补充，不能静默并入原基准 |

以上均来自 [作者 README](https://github.com/chaozhong2010/HRSID/blob/master/README.md)。引用数据论文：[Wei et al., HRSID, IEEE Access (2020)](https://ieeexplore.ieee.org/document/9127939)。

HRSID 仓库有 [GPL-3.0 LICENSE](https://github.com/chaozhong2010/HRSID/blob/master/LICENSE)。通过 [官方目录元数据 API](https://api.github.com/repos/chaozhong2010/HRSID/contents/annotations?ref=master) 核对，annotations 目录仅列出 [train2017.json](https://github.com/chaozhong2010/HRSID/blob/master/annotations/train2017.json)（11,592,317 字节）；并没有同时列出 test2017.json。本次没有读取该标注载荷，测试标注需从所选官方包取得。仓库标注引用应保留其 GPL 来源，不能写成“没有许可证”，也不能将该目录称为完整训练/测试数据镜像。

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
