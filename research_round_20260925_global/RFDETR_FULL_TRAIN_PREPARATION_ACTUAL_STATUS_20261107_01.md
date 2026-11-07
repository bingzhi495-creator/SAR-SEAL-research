# RF-DETR 完整训练环境：实际清单、解析结果与离线准备

2026-11-07。已完成官方候选包核验、DGX 原生离线解析及独立目录安装尝试。包安装和元数据检查已通过。原 COCO Nano312 的限定 schema 兼容及 CPU 前向已通过，未重跑。[CPU 检查证据](RFDETR_SCHEMA_COMPAT_CPU_FORWARD_ACTUAL_RESULT_20261107_02.md)

## 完整基础环境清单

实际 Python 3.12.14、Linux aarch64、glibc 2.39。一次独立标准库/包元数据检查退出0：可见发行版100项，按实际路径优先级选中97项，无科学模块导入、包安装或权威状态修改。重复项为 pip、setuptools、wheel；实际选中的分别是25.0.1、80.9.0、0.45.1，不能按最高版本推断。

核心实际版本为 Torch2.10.0+cu130、torchvision0.25.0+cu130、NumPy1.26.4、Triton3.6.0。连同 CUDA bindings/pathfinder 与15项 NVIDIA 包，共21项核心保护。新环境显式只读继承两处现有库目录，训练新增包仅写独立目录。完整清单留在执行侧，仓库仅保存可发布的版本与状态摘要。

当前拟解析16项明确的直接目标，沿用97项基线约束，唯一预定已有发行版覆盖是独立训练目录中的 typer 0.25.1。原 typer 0.28.0 不满足所选 Roboflow 的 `<0.26`；现有 typer-slim 0.24.0 可与候选交集相容，实际安装和 API 尚未验证。当前还存在普通 OpenCV 4.11.0.86，而计划新增同版本 headless，二者共享 `cv2` 模块名；真实导入时必须核查来源与 ABI。[候选包与来源](RFDETR_FULL_TRAIN_CLI_DEPENDENCY_READINESS_20261107_01.md)

## 两次真实解析，均未安装或训练

| 尝试 | 原始工具／SSH／子进程 | pip | 实际观察 |
|---|---|---|---|
| 02 | 1／1／1 | 2 | pip 的可选 User-Agent 发行版信息探测拟调用 `lsb_release`，先打开 `/dev/null` 时被本项目写入限制阻止；未发出HTTP请求。 |
| 03 | 1／1／1 | 1 | 仅改用固定 HTTP User-Agent，保留原生解析算法。发送46次包索引或元数据请求，观察到22个 wheel 元数据候选，最终在当前访问条件下未得到 aiohttp 候选；没有生成完整解析报告、没有获取完整 wheel、没有其他守卫拒绝。 |

两次子进程均已等待结束、所属进程组已空，完整发行版元数据前后相同；没有导入模型库、读取真实 SAR、安装或训练。03 的父进程耗时278.623秒，含传输的本机观察耗时296.235秒，没有超时或终止信号。错误是本次原生解析未找到候选，确切的远端索引／访问原因未证实。官方 [aiohttp 3.14.5 发布](https://pypi.org/project/aiohttp/3.14.5/#files) 有 CPython3.12 的 manylinux ARM64 wheel；已核官方文件元数据，不能据失败推断架构不支持或依赖不存在。

早期仅允许获取元数据的版本未执行。查固定 [pip 25.0.1 解析器](https://github.com/pypa/pip/blob/25.0.1/src/pip/_internal/resolution/resolvelib/resolver.py#L166-L171) 和[报告写入顺序](https://github.com/pypa/pip/blob/25.0.1/src/pip/_internal/commands/install.py#L359-L379) 后确认，原生 dry-run 仍会先补全 wheel 下载。后续声明长度检查仅覆盖普通 HTTP200 wheel 响应的头部预留，64/192 MiB 不代表全量网络字节或所有中间响应体硬上限。[边界说明](RFDETR_BINARY_DRY_RUN_DECLARED_BYTE_BOUNDARY_NOTE_20261107_01.md)

当前改按本机取得官方文件、逐项核对大小/SHA、再传输至 DGX 的路线。根据已观察候选和版本元数据准备的候选池不等于已解析安装计划；必须由目标环境的原生离线 pip 实际闭合，并复核所有选择及21项核心。解析成功、独立目录安装、真实 CLI/编解码器导入、真实 576 SAR 训练资格分别记录。没有新的 SAR AP 或方法收益。

## 离线实际结果

两次在线尝试保留。候选池包括其已观察的22项和 aiohttp 闭包8项；PyArrow 补齐缺失字节后，30个整包全部以官方 SHA-256 核验，共99,643,725字节。传到 DGX 后再次逐包核验。文件传输的实际工具与 SSH 均退出0，无包安装或模型执行。

DGX 以 pip 25.0.1 的原生解析器、仅本地目录与二进制包实际执行 dry-run。工具、SSH、父管控下子进程和 pip 均退出0，计划选中30项，已按官方来源和确切本地文件验收；没有网络请求，没有包变更，97项选中基础元数据与100项可见基础元数据前后一致，21项核心受保护。

第一次安装时，pip install 和 pip check 均实际返回0，观察到30项安装，但隔离守卫拒绝一次 socket.bind，原任务返回1，未接受该环境。首次记录没有调用者证据。查 [pip 25.0.1 vendored urllib3 原始实现](https://github.com/pypa/pip/blob/25.0.1/src/pip/_vendor/urllib3/util/connection.py#L121-L148)，为第二个独立目录仅增加精确调用点的识别与拒绝。第二次在独立目录按同一逐包 SHA 锁实际安装30项，pip install、pip check 均退出0，进程与隔离检查均闭合。记录并拒绝了一次 pip vendored urllib3 的 IPv6 回环能力探测，没有允许绑定或联网，也未把第二次观测倒推给第一次。21项原核心版本和发行版路径保持一致；静态查找确认 typer 与 cv2 来自新目录，后者仍待真实导入和 ABI 核验。

这些是环境证据。真实 576 SAR 数据流、优化器更新、精度和分割评估仍需单独执行；没有新 AP。

## 实际训练资格和公共数据准备

[训练接口草稿](RFDETR_TRAINING_QUALIFICATION_INTERFACE_DRAFT_20261107_01/README.md)记录原生模型、数据、损失、优化器、EMA 和 canonical 导出顺序。首步 warmup 可使学习率为零；需要按实际梯度、计数和后续参数变化判断，不能改 `max_steps` 后假称原100轮调度。当前 batch1 掩膜路由按源码应走 CPU SciPy 分配，梯度累积4不会改变即时 batch。以后若进入合格 GPU 分配路径，其 JIT 条件和可能子进程见[源码补录](RFDETR_GPU_MATCHER_JIT_SOURCE_NOTE_20261107_01.md)，不能在未观测前推断必然发生。

另已定位已有 HRSID JPG 两份归档及配套 train/test 标注，优先复用[已有完整 JPG 字节/CRC 准备](HRSID_JPG_INPUT_AUDIT_STATUS_20261107_04.md)与[标注差异记录](HRSID_ANNOTATION_PROTOCOL_DIFF_STATUS_20261107_05.md)。本次定位没有读取图像或 GT 正文。正式启用前仍须核像素和轮廓对齐、数据身份、跨划分相同图像及评价范围；不把 JPG 等同官方 PNG，也不把仓库许可证自动延伸到外部影像归档。[作者获取入口](https://github.com/chaozhong2010/HRSID#dataset-download-link)

[CLI 与编解码合成 CPU 检查](RFDETR_CLI_CODEC_QUALIFICATION_DRAFT_20261107_01/README.md) 已完成候选源码和静态接口检查，待真实执行。检查从官方 CLI 到原生 RLE、JPEG、PNG、OpenCV 和 faster_coco_eval，合成样本与真实 SAR 指标分别报告。

[首次实际 CLI 检查的失败原因与第二版修正](RFDETR_CLI_CODEC_ACTUAL_RESULT_20261107_01.md) 已记录。首次原任务在解析阶段退出，编解码未执行。

原 PLAIN/FULL 已各接受35,840/37,050次正式更新，下一35,840→36,096原任务已接续。研究总目标继续进行。
