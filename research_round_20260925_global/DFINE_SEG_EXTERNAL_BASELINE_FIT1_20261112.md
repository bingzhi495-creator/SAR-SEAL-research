# D-FINE-seg-S 外部对照的 HRSID 单图掩膜验证

D-FINE-seg-S 是通用实例分割模型，拟作为独立架构对照。锁定作者 v0.7.0、提交 b782ece206487792f36c165ca65548154cd12094。作者发布的 212 个归档源文件逐一对照 Git blob 通过；独立运行使用的 70 个 Python／配置文件与作者固定版本一致。官方 COCO 分割 S 检查点 48,119,167 字节，已下载并核对作者 LFS 摘要。作者源码和该检查点均为 Apache-2.0。 [作者预印本](https://arxiv.org/abs/2602.23043) · [源码版本](https://github.com/ArgoHA/D-FINE-seg/releases/tag/v0.7.0) · [作者检查点](https://huggingface.co/ArgoSA/D-FINE-seg) · [核验汇总](results/DFINE_SEG_EXTERNAL_BASELINE_INTAKE_SAFE_20261112.json)

新建独立 Python 3.11.15 CPU 环境，安装作者要求的 Torch 2.13.0+cpu 和 torchvision 0.28.0+cpu。首次安装因 Windows 临时构建路径过长，在 antlr4-python3-runtime 构建环节退出。仅缩短临时路径后，同一依赖解析结果完成安装，既有项目环境保持原状态。 [版本要求](https://github.com/ArgoHA/D-FINE-seg/blob/b782ece206487792f36c165ca65548154cd12094/pyproject.toml)

原工具进程自然退出 0。固定一张已闭合的 800×800 HRSID FIT PNG，逐字节核对图像及解码摘要。保留官方 80 类 COCO 权重、640×640 输入、作者 NMS 0.70 与掩膜阈值 0.5。单图接口仅为输出验证将置信度门槛设为 0.0。原生接口返回 **84 个 800×800 uint8 二值掩膜**，均非空。分数范围约 0.0772–0.3284，按原 0.5 接口分数门槛该图不会保留候选。单线程真实前向和后处理 0.874 秒，另有作者构造器的一次合成预热。没有开启 FIT 标注或 DEV／TEST，也没有训练、AP 或船舶正确性判断。 [实际安全结果](results/DFINE_SEG_EXTERNAL_BASELINE_FIT1_ACTUAL_SAFE_20261112.json) · [独立读回](results/DFINE_SEG_EXTERNAL_BASELINE_FIT1_INDEPENDENT_SAFE_20261112.json) · [原执行程序](code/DFINE_SEG_EXTERNAL_BASELINE_FIT1_20261112/probe_dfine_native_fit1.py)

接下来进入真实单类训练前，需锁定 HRSID FIT 读取和类别重映射，核验 COCO 初值的共享层及重建分类层，并完成两次真实优化更新及原图掩膜导出。作者默认入口每轮验证并以验证表现保存最佳权重，正式对照应使用固定轮次 EMA，训练过程的核验不使用 DEV705 或正式 TEST。模型内部 top300、原生筛选与评估 maxDets100 分开记录。作者训练对变换后多边形顶点取整并填充，跨模型结果需要单列监督栅格化差异。 [原生掩膜接口](https://github.com/ArgoHA/D-FINE-seg/blob/b782ece206487792f36c165ca65548154cd12094/dfine_seg/infer/torch_model.py#L276-L337) · [训练入口](https://github.com/ArgoHA/D-FINE-seg/blob/b782ece206487792f36c165ca65548154cd12094/dfine_seg/dl/train.py#L1008-L1019) · [栅格化](https://github.com/ArgoHA/D-FINE-seg/blob/b782ece206487792f36c165ca65548154cd12094/dfine_seg/dl/utils.py#L1181-L1195)

## 两次单类 FIT 更新与 EMA 掩膜导出

沿用同一作者 v0.7.0、官方 COCO 检查点和独立 CPU 环境。预先锁定两张 FIT 图及访问顺序，单图批次，FP32 640×640，关闭 Mosaic 和多尺度，其余作者增强用固定种子。直接调用作者原生 COCO 多边形读取、准则、Muon 加 Adan 优化器和 OneCycle，执行名义 55 轮计划的前两次更新。入口不建验证或正式测试数据加载器。载入 806 个同形 COCO 张量，按单类结构重新初始化 9 个分类相关张量。两图原标注分别 13 和 5 条，实际增强后为 13 和 4 条。 [预锁](results/DFINE_SEG_EXTERNAL_BASELINE_FIT2_PRELOCK_SAFE_20261112.json)

两次真实优化器、EMA 和调度器均返回并计数，原进程自然退出 0。总损失分别 36.9318、35.1633，掩膜 BCE 0.5935、0.5477，掩膜 Dice 损失 0.5958、0.3222。每次都有有限非零的掩膜梯度和实际权重变化。两次使用不同训练图，损失只作为执行依据。另一个进程重新载入更新后归档，验证全部 815 个 EMA 张量的摘要和掩膜权重更新轨迹。 [实测汇总](results/DFINE_SEG_EXTERNAL_BASELINE_FIT2_ACTUAL_SAFE_20261112.json) · [独立验收](results/DFINE_SEG_EXTERNAL_BASELINE_FIT2_INDEPENDENT_SAFE_20261112.json)

用该归档原生导出第一张 FIT 图，得到 281 个非空的 800×800 uint8 二值掩膜，模型已是单类。置信度门槛设为 0，仅检验训练到导出的链路。没有使用 DEV／TEST 或计算 AP，也没有记录可与正式训练比较的单步耗时。下一个正式资格是独立 CUDA 环境及固定预算端点验证。 [原执行程序](code/DFINE_SEG_EXTERNAL_BASELINE_FIT2_20261112/run_author_fit2.py)
