# D-FINE-seg-S 外部对照的 HRSID 单图掩膜验证

D-FINE-seg-S 是通用实例分割模型，拟作为独立架构对照。锁定作者 v0.7.0、提交 b782ece206487792f36c165ca65548154cd12094。作者发布的 212 个归档源文件逐一对照 Git blob 通过；独立运行使用的 70 个 Python／配置文件与作者固定版本一致。官方 COCO 分割 S 检查点 48,119,167 字节，已下载并核对作者 LFS 摘要。作者源码和该检查点均为 Apache-2.0。 [作者预印本](https://arxiv.org/abs/2602.23043) · [源码版本](https://github.com/ArgoHA/D-FINE-seg/releases/tag/v0.7.0) · [作者检查点](https://huggingface.co/ArgoSA/D-FINE-seg) · [核验汇总](results/DFINE_SEG_EXTERNAL_BASELINE_INTAKE_SAFE_20261112.json)

新建独立 Python 3.11.15 CPU 环境，安装作者要求的 Torch 2.13.0+cpu 和 torchvision 0.28.0+cpu。首次安装因 Windows 临时构建路径过长，在 antlr4-python3-runtime 构建环节退出。仅缩短临时路径后，同一依赖解析结果完成安装，既有项目环境保持原状态。 [版本要求](https://github.com/ArgoHA/D-FINE-seg/blob/b782ece206487792f36c165ca65548154cd12094/pyproject.toml)

原工具进程自然退出 0。固定一张已闭合的 800×800 HRSID FIT PNG，逐字节核对图像及解码摘要。保留官方 80 类 COCO 权重、640×640 输入、作者 NMS 0.70 与掩膜阈值 0.5。单图接口仅为输出验证将置信度门槛设为 0.0。原生接口返回 **84 个 800×800 uint8 二值掩膜**，均非空。分数范围约 0.0772–0.3284，按原 0.5 接口分数门槛该图不会保留候选。单线程真实前向和后处理 0.874 秒，另有作者构造器的一次合成预热。没有开启 FIT 标注或 DEV／TEST，也没有训练、AP 或船舶正确性判断。 [实际安全结果](results/DFINE_SEG_EXTERNAL_BASELINE_FIT1_ACTUAL_SAFE_20261112.json) · [独立读回](results/DFINE_SEG_EXTERNAL_BASELINE_FIT1_INDEPENDENT_SAFE_20261112.json) · [原执行程序](code/DFINE_SEG_EXTERNAL_BASELINE_FIT1_20261112/probe_dfine_native_fit1.py)

接下来进入真实单类训练前，需锁定 HRSID FIT 读取和类别重映射，核验 COCO 初值的共享层及重建分类层，并完成两次真实优化更新及原图掩膜导出。作者默认入口每轮验证并以验证表现保存最佳权重，正式对照应使用固定轮次 EMA，训练过程的核验不使用 DEV705 或正式 TEST。模型内部 top300、原生筛选与评估 maxDets100 分开记录。作者训练对变换后多边形顶点取整并填充，跨模型结果需要单列监督栅格化差异。 [原生掩膜接口](https://github.com/ArgoHA/D-FINE-seg/blob/b782ece206487792f36c165ca65548154cd12094/dfine_seg/infer/torch_model.py#L276-L337) · [训练入口](https://github.com/ArgoHA/D-FINE-seg/blob/b782ece206487792f36c165ca65548154cd12094/dfine_seg/dl/train.py#L1008-L1019) · [栅格化](https://github.com/ArgoHA/D-FINE-seg/blob/b782ece206487792f36c165ca65548154cd12094/dfine_seg/dl/utils.py#L1181-L1195)

## 两次单类 FIT 更新与 EMA 掩膜导出

沿用同一作者 v0.7.0、官方 COCO 检查点和独立 CPU 环境。预先锁定两张 FIT 图及访问顺序，单图批次，FP32 640×640，关闭 Mosaic 和多尺度，其余作者增强用固定种子。直接调用作者原生 COCO 多边形读取、准则、Muon 加 Adan 优化器和 OneCycle，执行名义 55 轮计划的前两次更新。入口不建验证或正式测试数据加载器。载入 806 个同形 COCO 张量，按单类结构重新初始化 9 个分类相关张量。两图原标注分别 13 和 5 条，实际增强后为 13 和 4 条。 [预锁](results/DFINE_SEG_EXTERNAL_BASELINE_FIT2_PRELOCK_SAFE_20261112.json)

两次真实优化器、EMA 和调度器均返回并计数，原进程自然退出 0。总损失分别 36.9318、35.1633，掩膜 BCE 0.5935、0.5477，掩膜 Dice 损失 0.5958、0.3222。每次都有有限非零的掩膜梯度和实际权重变化。两次使用不同训练图，损失只作为执行依据。另一个进程重新载入更新后归档，验证全部 815 个 EMA 张量的摘要和掩膜权重更新轨迹。 [实测汇总](results/DFINE_SEG_EXTERNAL_BASELINE_FIT2_ACTUAL_SAFE_20261112.json) · [独立验收](results/DFINE_SEG_EXTERNAL_BASELINE_FIT2_INDEPENDENT_SAFE_20261112.json)

用该归档原生导出第一张 FIT 图，得到 281 个非空的 800×800 uint8 二值掩膜，模型已是单类。置信度门槛设为 0，仅检验训练到导出的链路。没有使用 DEV／TEST 或计算 AP，也没有记录可与正式训练比较的单步耗时。下一个正式资格是独立 CUDA 环境及固定预算端点验证。 [原执行程序](code/DFINE_SEG_EXTERNAL_BASELINE_FIT2_20261112/run_author_fit2.py)

## 独立 CUDA 算子核验

沿用已核验的 Python 3.11.15 解释器新建独立环境，安装官方 Torch 2.13.0+cu130 与 torchvision 0.28.0+cu130。主 Torch wheel 1,915,488,103 字节，与官方索引摘要一致。torchvision 官方索引未给出摘要，两个官方端点的下载逐字节一致。原 CUDA 进程在本机 RTX 3090 Ti 上自然退出 0，另完成只读核验。

在同一合成输入上，作者原生可变形注意力相对 CPU 的前向最大绝对差为 5.96×10⁻⁸，梯度最大绝对差为 2.38×10⁻⁷。CUDA 张量乘法与参考一致，作者依赖的 CUDA NMS 保留了预定索引。此步骤没有打开权重、图像或标注，没有训练。训练级显存峰值和吞吐仍待实际批次核验。 [CUDA 实测](results/DFINE_SEG_EXTERNAL_BASELINE_CUDA_ACTUAL_SAFE_20261112.json) · [独立读回](results/DFINE_SEG_EXTERNAL_BASELINE_CUDA_INDEPENDENT_SAFE_20261112.json) · [核验源码](code/DFINE_SEG_EXTERNAL_BASELINE_CUDA_20261112/probe_dfine_cuda_ops.py) · [官方 Torch wheel 索引](https://download.pytorch.org/whl/cu130/torch/)

## 真实 GPU 批次和固定 E24 入训准备

在 RTX 3090 Ti、作者 640 输入和实际四图批次上，从相同官方 COCO 权重重新初始化。按预锁的 32 张不同 FIT 图完成 8 次前向、优化器、EMA 和调度器更新，86 条原始多边形在这一短窗均保留为非空监督。CUDA FP32 训练峰值分配 4,738,208,768 字节（约 4.41 GiB），峰值保留 5,316,280,320 字节（约 4.95 GiB）。8 个训练步的实测区间合计 4.062 秒，原工作进程耗时 36.141 秒，含加载和导出。更新后的单类 EMA 重新载入并输出 261 个非空原图掩膜。置信度门槛固定为 0 供链路核验。原进程正常退出，独立复核时重读 815 个 EMA 张量并核对计数。未使用 DEV／TEST 或 AP。 [短训预锁](results/DFINE_SEG_EXTERNAL_BASELINE_FIT32_PRELOCK_SAFE_20261112.json) · [GPU 实测](results/DFINE_SEG_EXTERNAL_BASELINE_FIT32_ACTUAL_SAFE_20261112.json) · [独立读回](results/DFINE_SEG_EXTERNAL_BASELINE_FIT32_INDEPENDENT_SAFE_20261112.json)

全量原生 FIT 标签经过作者 COCO 读取器和独立逐记录核对。2,937 张图、9,260 个原多边形均成功读取，类别、框和所有 float32 多边形顶点一致，未跳过目标。该阶段未解码图像。第一次读取器核验因人为设置的 16 MiB 上限退出。第二次只改用已知精确字节数和原摘要，按同一读取规则正常结束。 [全量读取器验收](results/DFINE_SEG_EXTERNAL_BASELINE_FIT2937_READER_SAFE_20261112.json)

固定 E24 使用 RF-DETR 原生种子 5202 的实际图像访问序列。原账本独立复算通过，各轮 2,937 张唯一图加 3 次原补采，按原顺序分成 735 个四图批次，共 70,560 次访问和 17,640 次计划更新。D-FINE 保留作者四图物理批次、640 输入和原掩膜栅格化，使用 BF16 前向、FP32 损失、原 Muon 加 Adan、作者 OneCycle 名义 55 轮时钟，固定第 24 轮 EMA。与 RF-DETR 的单图累计、增强、栅格化和优化器不同。此处只锁定访问顺序和训练预算，不把跨架构比较当作因果消融。固定 DEV705 评价须等原 E24 独立验收。 [访问账本核验](results/DFINE_SEG_EXTERNAL_BASELINE_E24_REPLAY_ACTUAL_SAFE_20261112.json) · [独立读回](results/DFINE_SEG_EXTERNAL_BASELINE_E24_REPLAY_INDEPENDENT_SAFE_20261112.json) · [E24 预锁](code/DFINE_SEG_EXTERNAL_BASELINE_FIT32_E24_20261112/e24/PROTOCOL.md)

## 固定 E24 已启动并通过首轮计数

同一原进程从官方 COCO 单类初值启动，重新核对全部 2,937 张训练图。前 8 个实际 BF16 前向、FP32 损失、优化器和 EMA 均返回，32 次访问按已接受的 RF-DETR 顺序。掩膜梯度有限非零，调度与两个独立参数时钟一致。E1 已从原进程输出确认 2,940 次访问和 735 次优化器、EMA、调度更新，9,296 条保留的非空监督，耗时 261.740 秒，CUDA 峰值分配 4,032,860,160 字节、保留 4,812,963,840 字节。原进程已进入 E2。此时仍在训练，E24 独立验收及固定 DEV705 尚未开始。 [固定训练源码](code/DFINE_SEG_EXTERNAL_BASELINE_FIT32_E24_20261112/e24/dfine_e24_worker.py) · [闭合验收入口](code/DFINE_SEG_EXTERNAL_BASELINE_FIT32_E24_20261112/e24/accept_dfine_e24.py)

### 原进程进入第四轮，固定 DEV705 入口就绪

原训练继续到 E3 自然完成，累计 8,820 次 FIT 访问和 2,205 次优化器、EMA、调度器更新，E3 用时 265.754 秒。随后同一原进程在 E4 已观测到第 600 个实际更新。训练继续，不选中间模型。固定 DEV705 的作者掩膜导出、独立读回、COCO 评分已做无数据合成测试，包含空预测、AP90 及四个空栅格标注的敏感性。真正导出及评分等待 E24 原进程关闭和独立验收。 [固定评价入口](code/DFINE_SEG_EXTERNAL_BASELINE_FIT32_E24_20261112/e24/dev705/PROTOCOL.md)
