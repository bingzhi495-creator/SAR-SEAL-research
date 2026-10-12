# HRSID 外部实例分割协议准备度（仅公开元数据）

日期：2026-10-12。范围：作者论文、作者仓库 README / 目录 / LICENSE 和作者给出的下载入口。此处未下载或读取影像、标注 JSON、权重；未训练、未评价。目的仅是为未来 B4 建立可执行的入场检查，不改变当前主实验。

## 核查结论

| 问题 | 已确认的一手证据 | 当前边界 |
|---|---|---|
| 是否有官方实例掩膜 | [原论文 II.B–C](https://ieeexplore.ieee.org/document/9127939/)明确称逐船标注含 category、bbox、segmentation mask，采用 MS COCO 格式；[作者 README](https://github.com/chaozhong2010/HRSID/blob/master/README.md)明确列出 instance segmentation 并展示 instance-ID / 彩色掩膜示例。 | 官方意图和标注类型明确；本轮没有读取 JSON，不能断言每条 `segmentation` 的实际编码、完整率或质量。 |
| 训练/测试接口 | 论文称 5,604 个 800×800 切片、16,951 船实例，按**切片图像数**约 65%/35% 分为 train/test，格式为 COCO。[作者仓库目录元数据](https://api.github.com/repos/chaozhong2010/HRSID/contents/annotations)目前只列出 [`annotations/train2017.json`](https://github.com/chaozhong2010/HRSID/blob/master/annotations/train2017.json)（11,592,317 B）。 | 作者仓库未列出 `test2017.json`，README 未给测试 JSON 的具体路径；它是否包含在作者下载压缩包中以及命名如何，须实际取得官方包后核验，不能现在把第三方镜像的目录当成官方接口。 |
| 影像下载版本 | [作者 README 下载区](https://github.com/chaozhong2010/HRSID/blob/master/README.md#dataset-download-link)给原版 JPG 的 [Google Drive](https://drive.google.com/open?id=1BZTU8Gyg20wqHXtBPFzRazn_lEdvhsbE) / [百度网盘](https://pan.baidu.com/s/1z0-E3Lwdkg14feu1FwRf3g)（提取码 `bs64`），另列带近岸/离岸附加标注的更新 JPG、高保真 PNG、400 张纯背景样本。 | JPG、更新 JPG、PNG、纯背景是不同发布物；主评价须锁定其中一个具体包、版本及校验和，不得混入附加样本。PNG 仍是影像栅格发布，不能据此称有 SLC/复数相位。 |
| 使用与再分发 | 仓库有 [GPL-3.0 LICENSE](https://github.com/chaozhong2010/HRSID/blob/master/LICENSE)；[IEEE 论文页](https://ieeexplore.ieee.org/document/9127939/)称**文章**为开放许可。README 的下载段没有写明影像及标注的单独授权范围。 | **影像/标注的可用及再分发许可未明确。** 不能由仓库通用 LICENSE 或论文 CC 自动推断第三方 SAR 影像、衍生切片和标注均可打包再分发。未来代码包仅放获取说明与转换代码；需发表/再分发时保留来源并向作者或上游权利方核实。 |
| 母场景/采集来源与独立性 | 原论文 II.A 给出 136 幅原始大场景（99 Sentinel-1B、36 TerraSAR-X、1 TanDEM-X），25% 重叠裁切；II.C / Table 2 按 serial number 汇总相同成像参数。README 示例文件名如 `P0094_0_800_3000_3800.png`，其前缀和坐标**可能**有助于追踪裁切来源。 | 已读论文与 README 没有逐切片→唯一采集 ID/地理区域的公开映射，也没有可复核的“按大场景分组”划分规则。Table 2 的参数组不等于唯一母场景或地理隔离。重叠切片跨 train/test 是**待查风险**，不是已证实泄漏；仅凭 `P####` 前缀也不能证明独立采集。 |

## 未来 B4 的最小执行接口与停止门

1. 从作者 README 的**原版 JPG**入口取得一次发布包，记录获取 URL、时间、原始压缩包 SHA-256 和解包后的目录清单；先只验证归档、影像与标注的可读性。附加版本单独登记，不与原版混合。
2. 在离线转换器中把作者 COCO JSON 的 `images[].id/file_name`、`annotations[].image_id/category_id/bbox/segmentation/iscrowd` 原样映射到项目数据接口；按 `file_name` 精确定位影像，保留实例多边形/RLE语义，禁止把 bbox 充当 mask。先检查类别、影像/标注计数、文件名唯一性、缺失文件、解码后 mask 面积和 800×800 尺寸，再生成只读转换清单。`test2017.json` 路径、实际字段与 `segmentation` 编码必须以作者压缩包内容为准；若测试实例标签缺失，**停止 mask AP 评价**。
3. 基于已取得的官方 train/test 文件名做一次元数据审计：提取 `P####` 候选母场景前缀和裁切坐标，统计两侧前缀交集、同前缀内裁切框的重叠；如需声称按独立采集/地理区域外推，必须再取得作者映射或独立证据把前缀关联到唯一采集 ID，并证明训练/测试来源隔离。若不能完成，只能称“在 HRSID 官方切片划分上的跨数据集零调参评价”，不能称母场景独立泛化。
4. 固定本项目模型权重、预处理、阈值和评价实现后再一次性运行 HRSID test；若因 mask 缺失、版本不明、许可限制或来源重叠疑点无法满足预定声明，分别降级为可复核的格式/可用性说明或受限外部 benchmark，不以结果倒改协议。当前阶段**尚无 HRSID 外部 AP 或泛化证据**。

一手入口：[IEEE 论文](https://doi.org/10.1109/ACCESS.2020.3005861)（作者原文 II.A–C、Table 1–2）、[作者 README](https://github.com/chaozhong2010/HRSID/blob/master/README.md)、[作者仓库标注目录元数据](https://api.github.com/repos/chaozhong2010/HRSID/contents/annotations)、[仓库 LICENSE](https://github.com/chaozhong2010/HRSID/blob/master/LICENSE)。作者仓库 `data/readme.txt` 目前为旧版 ISSID 发布预告，不应拿来确认 HRSID 的测试文件或授权。
