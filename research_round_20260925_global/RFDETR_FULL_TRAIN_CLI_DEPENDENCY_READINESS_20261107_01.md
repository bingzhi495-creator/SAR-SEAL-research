# RF-DETR 1.12 full-train / CLI dependency readiness

后续实际更新：[完整基础发行版清单已取得，两次在线解析失败保留，转本机取包后离线解析](RFDETR_FULL_TRAIN_PREPARATION_ACTUAL_STATUS_20261107_01.md)。下文的“未取得清单”属于早先静态准备时刻；完整依赖安装、CLI与SAR训练仍待。

2026-11-07。状态为有界静态依赖准备，真实 DGX 完整 inventory、完整 train resolver 和实际 CLI parser 均待取得。目标上下文由 Root 给定为 Linux aarch64、CPython3.12.14、Torch2.10.0+cu130、torchvision0.25.0、NumPy1.26.4。本轮读取固定普通源码、候选 YAML、已关闭的基础安装结果，联网只查询官方 PyPI JSON 与官方库文档。以下版本为待 resolver 检验的候选，训练就绪须在真实检查完成后判定。

## 1. 实际入口与 extra

固定提交 `639eac4555ceb89ef399ccda3fde716bbcfe34cb` 的 `pyproject.toml` 中，基础 dependencies 只覆盖 requests/numpy/torch/torchvision/tqdm/transformers/pydantic/supervision/pyDeprecate 等推理路径。`train` 与 `cli` 是两个独立 extra。完整官方训练命令所需范围为 **train + cli**。`cli` 单独只增加 `jsonargparse[signatures]>=4.46.0`，它没有递归包含 train。`pytorch_lightning` 和 `lightning` 的 PyPI distribution 名也应区分，本源码直接依赖并导入前者。[固定 pyproject](https://github.com/roboflow/rf-detr/blob/639eac4555ceb89ef399ccda3fde716bbcfe34cb/pyproject.toml)

`rfdetr` console script → `rfdetr.cli:main`，`python -m rfdetr` 也进入相同函数。`src/rfdetr/cli/__init__.py:15–20` 再延迟导入 `rfdetr.training.cli.main`。训练 CLI 文件26–33行导入 Lightning、模型、数据模块和 trainer，572–578行构造 RFDETRCli。344–373行使用 `set_parsing_settings(subclasses_enabled=[ModelConfig, TrainConfig])`，其明确下限为 jsonargparse4.46.0。375–386行把 model 配置链接给 data，450–529行由 train_config 和 trainer overrides 构建 trainer。已有 YAML 的 ModelConfig/TrainConfig 子类语法因此需要 signatures extra 及该版本 API。[固定 CLI](https://github.com/roboflow/rf-detr/blob/639eac4555ceb89ef399ccda3fde716bbcfe34cb/src/rfdetr/training/cli.py)

当前 YAML 使用 torchvision augmentation、faster_coco_eval、bf16，关闭 compile/cuda_graphs/fused_optimizer 和 tensorboard/wandb/mlflow。首轮依赖范围据此限定为 train+cli。流式视频、WebDataset、自定义增强、LoRA之外的额外部署平台和测试/dev extra 均有各自额外依赖，当前配置没有请求这些入口。特别是 `cuda` extra 的 transformer-engine requirement 有 Linux x86_64 marker，aarch64不会因为勾选该extra自动获得可用 Transformer Engine。[固定 optional dependencies](https://github.com/roboflow/rf-detr/blob/639eac4555ceb89ef399ccda3fde716bbcfe34cb/pyproject.toml)

## 2. 有界直接依赖候选

下表覆盖 train 的12个直接依赖及 cli 的1个直接依赖。现有 DGX 中哪些已经满足，须以新完整 inventory 确认。纯 Python wheel 只解决分发架构门槛，API与传递依赖仍交给真实 resolver 和运行检查。

| 直接 requirement | 已查 PyPI 候选 | Py3.12 / aarch64 分发依据与关键约束 |
| --- | --- | --- |
| peft | 0.21.2 | py3-none-any，Python≥3.10。引入 accelerate≥0.21、Torch≥1.13、Transformers、HF Hub≥0.25 |
| pytorch_lightning≥2.6,<3，排除2.6.2/2.6.3 | 2.6.6 | py3-none-any，Python≥3.10、Torch≥2.1。实际 CLI/import 仍需验证 |
| torchmetrics[detection]≥1.8.2,<1.9 | 1.8.2 | py3-none-any；detection extra 还需 pycocotools>2 和 torchvision≥0.15.1。保留1.8私有API上限 |
| hotcoco≥1.2.1,<1.3 | 1.2.1 | cp310-abi3 manylinux2.17 aarch64，2,514,622B；要求 NumPy≥1.22 |
| faster-coco-eval≥1.7.2 | 1.8.0 | cp312 manylinux2.26/2.28 aarch64，578,095B；是当前 YAML 指定后端 |
| ultrafast-pycocotools≥0.1.13,<0.2 | 0.1.14 | cp312 manylinux2.17 aarch64，723,032B；要求 NumPy≥1.21 |
| vernier≥0.6.0,<0.7 | 0.6.0 | cp310-abi3 manylinux2.17 aarch64，2,265,197B；要求 NumPy≥1.26、pyarrow≥15 |
| pycocotools | 2.0.11 | cp312-abi3 manylinux2.17/2.28 aarch64，398,434B；要求 NumPy |
| scipy | 实际满足约束的继承版本待完整 inventory | 基础模型可能已有 SciPy。本轮没有取得其实际完整 metadata，不自行指定替换版本 |
| simplejpeg≥1.9,<2 | 1.9.0 | cp312 manylinux2.17/2.28 aarch64，448,908B；要求 NumPy≥1.19.3 |
| torch-hungarian==0.1.0rc0 | 0.1.0rc0 | py3-none-any，16,894B；Linux metadata 无条件引入 triton，还需 Torch≥1.12、SciPy≥1.6 |
| roboflow | 1.7.3 | py3-none-any；引入 opencv-python-headless≥4.10、pillow-avif-plugin<2，以及 typer≥0.12,<0.26 |
| jsonargparse[signatures]≥4.46.0 | 4.46.0 | py3-none-any，Python≥3.9；signatures 引入 docstring-parser≥0.17、typeshed-client≥2.8.2。当前PyPI更新版本4.53.0仍需单独验证私有API，首个候选可固定声明下限 |

上表的发布文件及 Requires-Dist 来自对应版本的官方 JSON。Lightning/metrics 的 test、dev、all 等 extra 中出现的其他 NumPy/Torch要求，不属于本轮已选的 extra，不能把这些未激活 marker 误判为当前冲突。[peft0.21.2](https://pypi.org/pypi/peft/0.21.2/json)、[Lightning2.6.6](https://pypi.org/pypi/pytorch-lightning/2.6.6/json)、[TorchMetrics1.8.2](https://pypi.org/pypi/torchmetrics/1.8.2/json)、[jsonargparse4.46.0](https://pypi.org/pypi/jsonargparse/4.46.0/json)

编译型 evaluator / decoder 的具体分发依据如下。这里确认的是 wheel 文件存在及声明的 Python/NumPy需求，目标主机 glibc、动态链接和实际导入还待核验。[hotcoco1.2.1](https://pypi.org/pypi/hotcoco/1.2.1/json)、[faster-coco-eval1.8.0](https://pypi.org/pypi/faster-coco-eval/1.8.0/json)、[ufcoco0.1.14](https://pypi.org/pypi/ultrafast-pycocotools/0.1.14/json)、[vernier0.6.0](https://pypi.org/pypi/vernier/0.6.0/json)、[pycocotools2.0.11](https://pypi.org/pypi/pycocotools/2.0.11/json)、[simplejpeg1.9.0](https://pypi.org/pypi/simplejpeg/1.9.0/json)

即使 YAML 选 faster_coco_eval，完整 `rfdetr[train]` metadata 仍包含上述其他 evaluator；安装层不能把“未选为运行后端”自动解释成“该依赖不存在”。若后续为资源另做手工精简依赖，应另行命名、说明偏离并核完整 import 路径。

## 3. 当前优先处理的四个解析点

**OpenCV 与保护的 NumPy1.26.4。** Roboflow1.7.3要求 opencv-python-headless≥4.10，当前 PyPI最新5.0.0.93在Python≥3.9时要求NumPy≥2。候选 `opencv-python-headless==4.11.0.86` 的Python3.12 metadata要求NumPy≥1.26.0，直接边上可与1.26.4相容，并有 `cp37-abi3-manylinux_2_17_aarch64` wheel，29,487,060B。这是应显式加入的候选约束，整体可解析性仍待实际报告。[Roboflow1.7.3](https://pypi.org/pypi/roboflow/1.7.3/json)、[OpenCV4.11.0.86](https://pypi.org/pypi/opencv-python-headless/4.11.0.86/json)、[OpenCV5.0.0.93](https://pypi.org/pypi/opencv-python-headless/5.0.0.93/json)

**typer 上限。** 本机基础镜像清单曾准备 typer0.28.0，该字节清单本身不能证明实际DGX当前版本。Roboflow1.7.3要求typer<0.26。Transformers5.1.0依赖typer-slim，而已核typer-slim0.24.0只要求typer≥0.24.0，因此候选交集为[0.24,0.26)，已查 `typer==0.25.1` 为纯Python wheel。若真实现有inventory为0.28.0，resolver会提出辅助包降版本，必须在新训练环境的变更清单中显式接受或另选受审Roboflow版本，不能把整环境原封不动的声明与此变化并用。HF Hub1.33.0的直接依赖为click等，其本身没有typer requirement。[Transformers5.1.0](https://pypi.org/pypi/transformers/5.1.0/json)、[typer-slim0.24.0](https://pypi.org/pypi/typer-slim/0.24.0/json)、[typer0.25.1](https://pypi.org/pypi/typer/0.25.1/json)、[HF Hub1.33.0](https://pypi.org/pypi/huggingface-hub/1.33.0/json)

**Triton 与 CUDA栈继承。** torch-hungarian0.1.0rc0的Linux requirement是无架构细分的triton。已查Triton3.6.0存在cp312 manylinux2.27/2.28 aarch64 wheel，176,128,243B。它超过先前镜像下载脚本的120MB单文件上限。是否已经由目标Torch环境提供、实际版本/local tag和Torch依赖是否匹配，必须由完整inventory确认。compile/fused_optimizer=false不会移除这条包metadata依赖。候选冻结应保留满足依赖的现有Triton；缺失或冲突时先停止并重审，不能让resolver顺带替换已验证的Torch/CUDA栈。[torch-hungarian0.1.0rc0](https://pypi.org/pypi/torch-hungarian/0.1.0rc0/json)、[Triton3.6.0](https://pypi.org/pypi/triton/3.6.0/json)

**Arrow 与额外二进制。** vernier0.6.0引入pyarrow≥15。已查候选PyArrow26.0.1有cp312 manylinux2.28 aarch64 wheel，51,818,506B，Python≥3.11。Roboflow的pillow-avif-plugin<2也有1.6.0 cp312 aarch64 wheel，约3.96MB；peft的accelerate候选1.15.0为纯Python。完整解析报告还会展开Lightning的fsspec[http]/lightning-utilities/PyYAML、jsonargparse signatures等。只记录有限关键项，不把这些元数据检查冒充完整传递锁文件。[PyArrow26.0.1](https://pypi.org/pypi/pyarrow/26.0.1/json)、[pillow-avif-plugin1.6.0](https://pypi.org/pypi/pillow-avif-plugin/1.6.0/json)、[Accelerate1.15.0](https://pypi.org/pypi/accelerate/1.15.0/json)

## 4. 解析和资格检查方案

1. 在已有基础环境之外准备独立训练依赖上下文。先取得该上下文的**真实完整** `pip inspect`、`pip list --format=json`/freeze、Python/platform/glibc兼容标签和所有继承distribution来源。记录Torch、torchvision、NumPy、Triton及现有NVIDIA相关包的精确版本，包括local tag。用文件/来源证据确认继承的固定RF代码与构建metadata。原CPU探针的九包或十包选择性receipt只覆盖所声明范围。
2. 写核心constraints，目标至少固定Torch2.10.0+cu130、torchvision实际精确版本及NumPy1.26.4。Torchvision题设0.25.0若实际带local tag，应使用实际精确字符串。保留现有满足要求的Triton和NVIDIA栈。将已核实际Transformers5.1.0、HF Hub1.33.0等基础版本纳入候选约束，辅助包变化另列，避免无边界升级。
3. 以固定源码的 `rfdetr[train,cli]==1.12.0` metadata 和上表候选建直接requirements。首个可检验候选可固定Lightning2.6.6、TorchMetrics[detection]1.8.2、jsonargparse[signatures]4.46.0、peft0.21.2、hotcoco1.2.1、faster-coco-eval1.8.0、ufcoco0.1.14、vernier0.6.0、pycocotools2.0.11、simplejpeg1.9.0、torch-hungarian0.1.0rc0、Roboflow1.7.3，并加入OpenCV4.11.0.86、typer0.25.1约束。SciPy优先保持实际合格版本。此集合须由真实resolver决定能否闭合。
4. 在已准备并核验的aarch64 wheelhouse上做离线、binary-only dry-run，保存完整report，任何缺包/冲突直接保留失败。示意未来命令为 `python -m pip install --dry-run --report train_cli_plan.json --no-index --find-links APPROVED_WHEELHOUSE --only-binary=:all: -c CORE_CONSTRAINTS.txt -r TRAIN_CLI_CANDIDATES.txt`。不要加`--ignore-installed`或无条件`--upgrade`，否则可能把本来继承的CUDA构建当成待重新求解对象。dry-run可能读取/下载metadata，完整wheelhouse和binary-only约束应先成立。[pip选项与安装报告](https://pip.pypa.io/en/stable/cli/pip_install/)
5. Root先读report，要求所有计划安装/替换项都有版本、官方来源、SHA256、正确wheel标签和已声明的变更理由。对保护核心包出现替换提案即停止。依赖安装应另走精确whitelist和既定原始终端/闭合资格；实际安装后取得pip check与完整post-inventory，并对受保护来源做前后核验。
6. 依赖层闭合后，先进行有界的CLI帮助和配置解析检查，再进行模型/Trainer构造。官方CLI的`--print_config`可用于导出解析配置，候选未来形式为 `python -m rfdetr fit --config QUALIFIED_BASELINE.yaml --print_config`。实际固定入口须确认其在构造类前退出。**LightningCLI(run=False)仍可能实例化模型/Trainer，不能把它命名为纯parser检查。** 该阶段会导入训练栈，须由Root另给执行资格。核对model→data的配置链接、双seed5201、576/PE48、batch1×accum4、E100端点EMA、bf16、faster后端/maxDets100，以及显式trainer overrides。[Lightning CLI配置文档](https://lightning.ai/docs/pytorch/stable/reference/cli/lightning_cli_intermediate)
7. YAML仍含数据占位符，预训练文件名/路径与已核官方checkpoint的对应性也须在具体新候选中明确。parser通过后，另需训练入口实际加载、原始tensor/schema证据、优化器/损失与BF16检查及canonical导出资格。只有相应真实步骤完成后才报告该层“就绪”。

## 5. 已有证据与未完成项

已关闭的 `RFDETR_OFFLINE_BASE_ENV_PREPARE_ORIGINAL_RESULT_PRIVATE_20261107_03.json` 显示基础离线解析/安装、pip check、freeze及metadata子步骤均returncode0，最后状态为ACTUAL_BASE_DEPENDENCIES_READY_NO_MODEL_IMPORT，来自镜像29个文件。该汇总只给子日志路径，没有把实际完整freeze/metadata内容内嵌在stdout。本轮没有取得DGX actual fullinventory。旧九包CPU receipt同样不能替代完整freeze；本报告也没有从日志文件名推断各包实际版本。

本轮完成普通源码/配置读取和官方元数据核查。完整train依赖解析、CLI parser、真实compiled extension导入、CUDA/BF16训练及资源资格均待执行。本轮没有运行pip安装或resolver，没有下载包wheel、导入Torch或操作DGX/Ops。

普通输入的SHA256供后续追溯。pyproject.toml30511B为`a5f58f05dd80547e7c9b29999ea5ea35a1d00025801a2c9296445b967eb61530`；training/cli.py25775B为`3641e533aaaeff600b52d79f2520369a717851db3b2dd987b8e9be19ced1574c`；BASELINE_YAML1815B为`ba9508d9acab5ff68b10db8a1aff5d9001868d6dda00e39f299d6f5cc1b8011b`；已关闭基础安装结果1920B为`837156d175761164ad3c981cda5028503932fbbf39d57eeb29f43551a33537dd`。官方PyPI元数据查询日期为2026-11-07，未作论文性能或运行兼容性的外推。
