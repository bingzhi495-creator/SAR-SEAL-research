# SRNet 在固定 PSeg-SSDD FIT741 上的原生入口实测

SRNet 是船舶旋转框与实例掩膜联合建模的直接外部对照。作者 PSeg-SSDD 官方测试 mask AP、AP50、AP75 为 67.1、94.9、85.4。项目的 FIT741 与 DEV187 位于官方训练侧内。后续比较使用相同 FIT 数据和相同 DEV187 评分规则。此次原生作者源码固定为提交 `2e56c7208f217ff1824f1d4f3f3ea242a5f78bc6`，实际执行前后工作树干净。 [SRNet 原文](https://www.mdpi.com/2072-4292/15/5/1324) · [固定作者源码](https://github.com/XIDIAN212Lab/SRNet/tree/2e56c7208f217ff1824f1d4f3f3ea242a5f78bc6) · [PSeg-SSDD 官方划分](https://www.mdpi.com/2072-4292/13/18/3690)

## 原生数据入口

作者从原多边形经整数坐标、OpenCV 最小外接矩形和 `le90` 转换得到旋转框，原多边形仍进入掩膜。两类监督在同一索引中可复建。以原 FIT741 COCO 和图像布局替换作者配置的训练数据位置，保留其数据集类、重复倍数和增强。作者实际解析 741 张图及全部 1,661 条多边形，过滤图与标注均为 0。按元数据摘要预选的 8 张图在原栅格、等比缩放及最终填充后均保留 20 条实例。独立用 COCO 栅格化核对原尺寸位图逐像素一致。原 CPU 工作体 4.51 秒，退出 0。 [加载器](https://github.com/XIDIAN212Lab/SRNet/blob/2e56c7208f217ff1824f1d4f3f3ea242a5f78bc6/mmrotate/datasets/SSDD.py#L88-L135) · [作者增强配置](https://github.com/XIDIAN212Lab/SRNet/blob/2e56c7208f217ff1824f1d4f3f3ea242a5f78bc6/configs/_base_/datasets/SSDD.py) · [实测源码](code/SRNET_SSDD_FIT741_BASELINE_NATIVE_20261111/srnet_fit_native.py) · [汇总](results/SRNET_SSDD_FIT741_NATIVE_DATA_ACTUAL_SAFE_20261111.json)

Windows CPython 3.10.13 隔离环境使用 PyTorch 1.12.0 CPU、torchvision 0.13.0 CPU、MMCV Full 1.6.0、MMDetection 2.28.2、NumPy 1.26.4、OpenCV 4.11.0.86、pycocotools 2.0.11、Pillow 9.5.0 和 e2cnn 0.2.3。首次执行缺少作者注册树传递依赖 e2cnn，原进程退出 1。补装后保持同一源码重跑。零偏移可变形卷积与普通卷积逐值相同，三个框的旋转 NMS 保留两个预期框。 [PyTorch 官方包](https://download.pytorch.org/whl/cpu) · [MMCV 官方包](https://download.openmmlab.com/mmcv/dist/cpu/torch1.12.0/index.html) · [e2cnn](https://pypi.org/project/e2cnn/0.2.3/)

## 原生模型前向和 RLE

从 PyTorch 官方入口取得 102,530,333 字节的作者指定 ImageNet ResNet-50 初值，核对完整 SHA-256。原作者模型按本地该文件初始化，第一层卷积权重与官方文件逐值一致。FPN、框头和掩膜头按作者配置初始化。原作者 `simple_test` 在一张预选 FIT 图上执行一次，掩膜恢复到 317×501 原网格。为实际走过非空分支，固定 `score_thr=0`、`nms_pre=32`、`max_per_img=4`。得到 4 条有限框及 4 张非空布尔掩膜，原尺寸 COCO RLE 全部逐像素往返一致。CPU 工作体 9.33 秒，原进程退出 0。单独重放除时钟外的全部安全汇总字段一致，实际 Python 子进程的 Windows 峰值工作集为 1,060,839,424 字节。原启动器采样只覆盖 Windows venv 的转接进程，内存以子进程独立核验为准。本轮权重范围为 ImageNet 主干及新初始化的任务头。PSeg 训练、固定 DEV 评分和正式测试待执行。 [官方 ImageNet 初值](https://download.pytorch.org/models/resnet50-0676ba61.pth) · [实测源码](code/SRNET_SSDD_FIT741_BASELINE_NATIVE_20261111/srnet_native_mask_smoke.py) · [输出汇总](results/SRNET_SSDD_NATIVE_MASK_EXPORT_CPU_SMOKE_ACTUAL_SAFE_20261111.json) · [子进程资源核验](results/SRNET_SSDD_NATIVE_CHILD_RSS_ACTUAL_SAFE_20261111.json)

## 接续条件

- 原作者配置为 36 轮、每轮三次数据重复。同预算复现按 RF-DETR 的 FIT741 E24 实际访问量和更新量对齐。先锁定同一原标注训练范围及配方调整，按实际有效访问、优化器更新及显存核对预算。同预算 SRNet 比较用同一 DEV187 原图 RLE 和 COCO 评估规则，另记录作者默认的训练配方。
- CUDA 11.6 版 PyTorch 1.12.0 和相应 Windows MMCV Full 1.6.0 安装文件的头部均已实测返回正确 ZIP 标记。训练还需实际验证本机 CUDA 的作者可变形卷积、旋转 IoU 与原生损失反传。作者掩膜损失显式调用 CUDA。NVIDIA 的 Ada 兼容性说明支持较低次版本的同主版本 cubin，是否适用于整条依赖链要用真实短窗确定。 [PyTorch CUDA 包](https://download.pytorch.org/whl/cu116/torch/) · [MMCV CUDA 包](https://download.openmmlab.com/mmcv/dist/cu116/torch1.12.0/index.html) · [NVIDIA Ada 兼容性](https://docs.nvidia.com/cuda/ada-compatibility-guide/) · [作者掩膜损失](https://github.com/XIDIAN212Lab/SRNet/blob/2e56c7208f217ff1824f1d4f3f3ea242a5f78bc6/mmrotate/models/dense_heads/SRNet_head.py#L888-L913)

## 按层目标配对的原生训练图检查

固定 ImageNet 主干初值、作者新初始化任务头及预选 FIT 样本，在训练模式执行真实特征提取、两级预测和精化阶段的正样本分配。CPU 诊断仅将掩膜损失的设备字面值从 CUDA 改为 CPU，保留作者运算。粗阶段的 SATSS 损失因 CPU 算子限制跳过，精化损失和掩膜调用经过原始路径。8 张固定核心图有 23 个正样本，均在第 0 层，没有错配。另将全部 3 张满足预定混合尺度条件的 FIT 图加入压力核对，得到 11 个正样本，其中一个跨层样本取到了其他目标的掩膜。按相同层索引取目标后，该条由空掩膜变为 297 个前景格点，297 个格点变化。34 个正样本中的这一个案例证明条件可触发，其频率仍受初始化和所选图影响。 [固定掩膜损失](https://github.com/XIDIAN212Lab/SRNet/blob/2e56c7208f217ff1824f1d4f3f3ea242a5f78bc6/mmrotate/models/dense_heads/SRNet_head.py#L953-L984) · [诊断源码](code/SRNET_SSDD_FIT741_BASELINE_NATIVE_20261111/mask_target_pairing.py) · [实际汇总](results/SRNET_MASK_TARGET_PAIRING_ACTUAL_SAFE_20261111.json)

源码按层选择 RoI 后，仍用全部正样本目标列表的前缀生成监督。正确目标需要使用与 RoI 相同的层索引。单层或目标身份恰巧一致时可以相同。此次没有优化器更新或任务 AP，BCE 的变化只描述共享特征与目标改变时的损失差。后续原版与修复版的真实训练将检验影响。

## 全 FIT 的裁剪与回贴四组几何实测

作者 28×28 训练采样位置为内部点，推理使用等宽格的中心。两者满足中心保持、两边长乘 28／29 的缩放关系。全 FIT 的 1,661 个真值旋转框中，有 1,121 个的最大节点差超过半个原图像素，464 个超过一个像素。按作者函数与独立几何公式复算，最大坐标误差低于 0.001 像素。作者的 `finest_scale=56` 用于选择特征层，实际掩膜头仍为 28×28。56×56 的附加核对仅作为几何参照。 [采样实现](https://github.com/XIDIAN212Lab/SRNet/blob/2e56c7208f217ff1824f1d4f3f3ea242a5f78bc6/mmrotate/models/dense_heads/SRNet_utils.py#L28-L47) · [网格核对](results/SRNET_FIT741_TRAIN_TEST_GRID_ACTUAL_SAFE_20261111.json)

先在固定 8 图的 20 个目标上做四组比较，再扩到全部 FIT741、1,661 条原多边形。真值框、真值位图、阈值和作者回贴代码相同，只组合内部点／格点中心裁剪与原框／中心保持收缩框回贴。收缩框边长固定乘 28／29，使其格点中心与原训练内部点重合，全部目标的最大坐标差为 6.10×10⁻⁵ 像素。全量原进程正常退出，工作体 50.13 秒，原 20 目标的四组结果逐值复现。两项合成测试和实际作者函数的合成样本核对通过。 [四组源码](code/SRNET_SSDD_FIT741_BASELINE_NATIVE_20261111/crop_paste_factorial.py) · [全量源码](code/SRNET_SSDD_FIT741_BASELINE_NATIVE_20261111/full_fit_crop_paste.py) · [全量汇总](results/SRNET_FIT741_CROP_PASTE_FACTORIAL_ACTUAL_SAFE_20261111.json)

| 真值位图的裁剪采样 | 原框回贴的平均 IoU | 收缩框回贴的平均 IoU |
| --- | ---: | ---: |
| 原训练内部点 | 96.414% | 97.988% |
| 等宽格中心 | 98.736% | 95.310% |

内部点的匹配回贴提高 1.574 个 IoU 百分点。中心配原框相对内部点配原框提高 2.321 点，相对内部点配匹配回贴提高 0.747 点。若回贴框保持收缩，改用中心裁剪反而降低 2.679 点，1,468 个目标下降、193 个持平。结果支持裁剪与逆映射配对是重要几何因素。剩余差值仍包含栅格、采样支撑与二值阈值作用，尚未识别单一来源。

内部点配原框的像素 FP／FN 为 70,622／1,896，内部点配匹配回贴为 11,275／28,714，中心配原框为 17,584／14,993。四组均有 1,661／1,661 个目标达到 IoU 0.90，空真值与空预测均为 0。大小分组分别有 1,324、332、5 条，细长目标 717 条。大目标和贴边组样本较少。这里直接从真值位图裁剪并使用真值框，描述的是几何重建。真实预测 AP、训练收益及全体训练正样本的错配频率待测。 [20 目标先导结果](results/SRNET_FIT28_CROP_PASTE_FACTORIAL_ACTUAL_SAFE_20261111.json) · [全量分层](results/SRNET_FIT741_CROP_PASTE_FACTORIAL_ACTUAL_SAFE_20261111.json)

## 真实训练的接续配方与环境

已在独立环境安装 PyTorch 1.12.0+cu116、torchvision 0.13.0+cu116、MMCV Full 1.6.0 CUDA 11.6，以及与 CPU 核对一致的 MMDetection 2.28.2、NumPy 1.26.4 等依赖。安装退出 0，离线导入和依赖一致性检查通过，Torch 与 MMCV 的构建 CUDA 均为 11.6。本项尚未执行 CUDA 算子或训练反传。

同预算候选配方使用作者批量 4，将训练数据重复次数设为 1，运行 24 轮。若原生加载保留 FIT741，作者分组采样每轮补齐到 744 次访问，共 17,856 次访问、4,464 次更新。保留作者 SGD，拟将 36 轮的学习率下降点按比例移到 16、22 轮，预热设为 112 次更新，作为明确标注的同预算适配。先进行真实全损失两步短训，再锁定长训端点。三项必要对照依次为原版、仅修复按层目标、修复目标并同时改特征与掩膜的训练采样位置，使用相同初始化、顺序、增强和配方。原版训练后的冻结回贴校准也需比较。RF-DETR 与 SRNet 的骨干、初值、优化器、EMA 和输入处理不同，跨架构结果需同时报告这些条件。 [作者数据配置](https://github.com/XIDIAN212Lab/SRNet/blob/2e56c7208f217ff1824f1d4f3f3ea242a5f78bc6/configs/_base_/datasets/SSDD.py#L40-L50) · [分组采样器](https://github.com/open-mmlab/mmdetection/blob/v2.28.2/mmdet/datasets/samplers/group_sampler.py)

原版 CUDA 两步入口已完成。它预选两张真实 FIT 图，使用作者完整 `forward_train` 的粗阶段 SATSS、精化与掩膜五组损失，计划执行两次原生 SGD 更新，保存各组损失、有限梯度和参数变化。一个 CPU 合成回归已在拟执行的 PyTorch 1.12.0+cu116 环境中通过。该回归验证损失聚合和更新辅助逻辑，真实 CUDA 反传仍待运行。当前 HRSID 原所有者持有本机 GPU 预留，按原接续规则进入该短训。 [两步草稿](code/SRNET_SSDD_FIT741_BASELINE_NATIVE_20261111/srnet_author_gpu_short.py) · [回归](code/SRNET_SSDD_FIT741_BASELINE_NATIVE_20261111/test_srnet_author_gpu_short.py) · [CUDA 环境离线导入](results/SRNET_CUDA116_ENV_OFFLINE_IMPORT_ACTUAL_SAFE_20261111.json)

## 作者原版 CUDA 两步实测

2026-11-11 在 RTX 3090 Ti 上运行原作者 SRNet，使用已核验的 ImageNet R50 初始化和两张预选 FIT 图。输入 608×608，批量 1，两个实际批次分别含 10 和 1 个目标。两次前向与反向均执行粗阶段 SATSS、粗分类与回归、精化分类与回归及掩膜损失。五项损失和全部观测梯度有限，完成两次 SGD 更新，三个任务头的抽查参数均改变。原进程退出 0，进程树清空，资源保护未触发。CUDA 峰值分配 1.798 GiB，峰值保留 1.924 GiB。并行运行的端到端耗时为 22.33 秒。 [实际结果与原进程核验摘要](results/SRNET_AUTHOR_CUDA_TWO_UPDATE_ACTUAL_SAFE_20261111.json)

独立进程在 HRSID 原协调器仍训练时运行。启动可用显存 16,753 MiB，观测最低 13,984 MiB，宿主可用内存最低 25,344,385,024 字节。复核同一 HRSID 协调器继续运行，单次复合中心推进到第 10 轮。此前等待 GPU 预约释放的判断适用于需要改写同一 HRSID／SSDD 账本的运行。此次用独立的 Windows 进程树和资源保护接入 SRNet。

本次确认原版算子、五项损失和两步更新可执行，范围为批量 1 的 FIT 短窗。作者批量 4 的原生加载器与优化器预算、完整 E24 训练和固定 DEV187 AP 仍待实测。 [实测入口](code/SRNET_SSDD_FIT741_BASELINE_NATIVE_20261111/srnet_author_gpu_short.py)

## 作者批量 4 原生训练短窗

在同一 Windows CUDA 环境调用作者 `mmrotate.apis.train_detector`。使用完整 FIT741、原增强和分组采样，保留 FP32、SGD 0.0025、动量 0.9、权重衰减 0.0001、梯度范数上限 35。拟匹配的 E24 配方使用批量 4、数据重复次数 1、Windows 加载进程数 0，按预算折算为 112 次预热和第 16、22 轮学习率衰减。原作者脚本的异常检测在本配方关闭，与批量 1 实测相同。短窗使用相同 E24 配置，从完整训练的首两次优化器更新处结束。

两次 SGD 实际返回，消费 8 次访问，均获得五项有限损失，三个任务头的抽查参数发生变化。原进程退出 0，作业树清空，CUDA 峰值分配 5.22 GiB、峰值保留 5.50 GiB，独立保护未触发。本次未完成整轮或生成权重，E24 的 17,856 次实际访问和 4,464 次更新仍需逐轮核验。训练入口区分采样器计划图像与增强后实际消费图像。 [实测与原进程摘要](results/SRNET_AUTHOR_BATCH4_SHORT_ACTUAL_SAFE_20261111.json) · [训练入口](code/SRNET_SSDD_FIT741_BASELINE_NATIVE_20261111/srnet_author_batch4_train.py)

固定 DEV187 以作者原始推理设置为主，框分数阈值为 0.3、旋转 NMS 阈值 0.1，并用相同整数真值和 COCO `maxDets=[1,10,100]` 评分。作者阈值会在 NMS 前截去低分框，预先单列阈值 0.001 的敏感性结果，保留其他设置与固定 E24 权重。映射保存原图 RLE、逻辑图像身份、作者分数和同分顺序，旋转框另存旁证。三个合成检查已通过，实际权重导出与评分待执行。 [固定评分接线](code/SRNET_SSDD_FIT741_BASELINE_NATIVE_20261111/srnet_dev187_mapping.py)

2026-11-11 11:06（UTC+8）核对同一原 HRSID 协调器仍在训练，SRNet 原生批量 4 的 E24 独立原进程已到第 1 轮第 100 批。完整计划为 24 轮、17,856 次访问与 4,464 次 SGD 更新，最终权重和 DEV187 评分待训练闭合。并行状态下保留宿主内存和显存保护，本次运行耗时受并行负载影响。

## 批量 4 长训的显存分配中断

原版批量 4 长训完成前 3 轮，在第 4 轮完成第 112 批后中止。原训练账本记录连续 670 次 SGD 更新和 2,680 次访问。下一批在原作者掩膜分支的 256 通道特征采样 `grid_sample` 前向中申请 2.00 MiB 失败。闭合回溯对应作者函数第 40 行，后续真值位图采样位于第 43 行。PyTorch 报告进程分配 4.37 GiB、保留和允许各 6.00 GiB，当时 CUDA 报告设备可用 15.36 GiB。原工人退出 1、原所有者退出 5，进程树已清空，外部资源保护未触发，未写最终检查点。该记录支持进程配额下的分配问题，保留为训练中断。 [实测中断与计数](results/SRNET_AUTHOR_BATCH4_E24_ALLOCATOR_OOM_ACTUAL_SAFE_20261111.json)

下一次重跑从同一初始化开始，保留原 FP32 批量 4 配方、训练数据与优化器预算。仅在新进程给原 CUDA 分配器设置 `max_split_size_mb:128`。在当前 Torch 1.12.0+cu116 上实际读到 `max_split_size=134217728`，该探测没有模型前向或优化器更新。当前并行的 HRSID 与 SSDD 训练仍在消费 GPU，资源允许时再运行完整重试，验证能否越过第 4 轮原失败批。 [PyTorch 1.12 CUDA 分配器](https://github.com/pytorch/pytorch/blob/v1.12.0/docs/source/notes/cuda.rst#memory-management)

DEV187 的作者原生 CPU FP32 导出已完成 4 项合成核对，含原输出形状、权重闭合约束和两种预锁阈值。固定作者 NMS 创建 CPU 标签，CPU 路径保留作者实现，推理耗时以实际设备记录。待验收的同一 E24 权重分别用于 0.3 主结果和 0.001 敏感性。 [原生导出入口](code/SRNET_SSDD_FIT741_BASELINE_NATIVE_20261111/srnet_dev187_export.py)

## 原生两档阈值的真实 FIT CPU 实测

在之前预选的首张 FIT 图上，固定作者的 CPU 推理和 ImageNet R50 初始化。任务头使用未经训练的作者初值。输入 608×608，原图 317×501，旋转 NMS IoU 0.1，`nms_pre=2000`、`max_per_img=2000`，掩膜二值门槛 0.5。两次前向只改变框分数门槛。0.3 与 0.001 各有 242 个回贴候选，均为非空掩膜，各自 242 条 RLE 解码与原作者输出逐像素一致。单次前向及回贴分别用时 8.20、7.33 秒。5 GiB 提交内存限额下进程树峰值 2.94 GiB，原工人和外层均退出 0，进程树清空。闭合后复核检测器、掩膜头、采样工具三份作者源码及固定 CPU 依赖，均匹配原固定版本。 [实测入口](code/SRNET_SSDD_FIT741_BASELINE_NATIVE_20261111/srnet_fit_threshold_preflight.py) · [结果和原等待摘要](results/SRNET_FIT_AUTHOR_THRESHOLDS_CPU_PREFLIGHT_ACTUAL_SAFE_20261111.json)

本次验证范围是一张训练图和两档未训练任务头的作者原生回贴及输出编码。两档候选数量相同，输出跨档逐行等价未比较。训练后权重、完整 DEV187 导出、最坏候选数内存和 AP 待实测。本次未打开 DEV 或 TEST。

## 原 E24 分配器重试实际启动

固定 DEV187 评分资源交接后，用独立 Windows 作业从同一初始权重启动 SRNet 作者 FP32、batch 4 的 E24 重跑。唯一运行时变化是既定的 CUDA 分配器 `max_split_size_mb:128`。启动时可用 CUDA 显存 22,325 MiB，宿主可用内存约 34.70 GiB。已观察到第一轮第 50／186 批的有限损失与梯度范数。历史运行在第 4 轮、累计 670 次 SGD 更新后触及进程显存配额。原失败保留，本次仍待真实越过原失败位置和完成固定评价。

同一原作业现已完整完成第 4 轮，累计 744 次 SGD 更新，并进入第 5 轮。前 4 轮的配方与预算保持原定，已越过上次累计 670 次更新后的分配失败位置。E24 原始关闭、最终权重与固定 DEV187 评分仍待完成。

## 原 E24 退出与最终权重验收

同一分配器重跑的原协调器和 GPU 工作进程均退出 0，作业树清空，资源保护未触发。训练完成 24 轮、17,856 次实际访问和 4,464 次 SGD 更新，逐轮更新计数为 186。增强遇到空目标时按原数据入口重新取样，日志记录了 54 次拒绝尝试。最终权重 `epoch_24.pth` 的 364,691,693 字节与原完成报告所列 SHA-256 相符，原终态、训练账本和权重均通过独立读回。训练本体耗时 3,013.65 秒，原所有者全程 3,016.09 秒。CUDA 峰值分配 5.61 GiB、峰值保留 6.00 GiB，作业树峰值 RSS 4.64 GiB。这些是该硬件及配方的运行成本。

先按预锁作者阈值 0.3 导出，再在同一权重和图像上做 0.001 的低阈值敏感性。接口原先把报告中的权重文件名字段当成绝对路径字段，已在新的评估入口按原报告的文件名和实际同目录归属修正。训练报告与权重保持原字节。CPU 主导出已启动，固定 DEV187 AP 和原始设置的推理耗时等待该次导出及评分实际退出。正式 TEST 继续隔离。

## 同一 E24 权重的固定 DEV187 评分实测

两档作者原生 CPU FP32 导出各完成 187 次前向。两次原所有者和工作进程均退出 0，作业树清空。随后固定整数 GT 的 COCO 独立评分进程正常退出，实际计算 1.532 秒。两档共用同一 E24 权重、187 张图和 380 条标注。作者的 0.3 为预定主结果，0.001 只改变 NMS 前的候选阈值。评分统一使用 COCO mask IoU 和 `maxDets=[1,10,100]`。 [SRNet 实测汇总](results/SRNET_AUTHOR_BATCH4_E24_FIXED_DEV187_ACTUAL_SAFE_20261111.json)

| 同一整数 DEV187 口径 | Mask AP | AP50 | AP75 | AP90 | APs | AR100 | 导出候选 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SRNet E24 作者阈值 0.3 | 60.540 | 91.579 | 76.723 | 0.556 | 62.241 | 63.974 | 384 |
| SRNet E24 低阈值 0.001 | 61.134 | 93.196 | 77.309 | 0.556 | 62.762 | 64.816 | 24,155 |
| RF-DETR E24 原生，5202 | 67.708 | 96.927 | 84.736 | 8.662 | 67.781 | 75.079 | 13,597 |
| RF-DETR E24 中心，5202 | 68.514 | 96.231 | 85.497 | 8.544 | 69.418 | 76.237 | 14,313 |

SRNet 阈值降低后候选增加 23,771 条，AP 提高 0.594 点，AP90 数值相同。作者阈值有 2 张图无候选，低阈值全部有候选。两档 CPU 原生导出主体分别耗时 466.618 秒和 635.241 秒，包含掩膜及 RLE 输出。RF-DETR 推理使用 GPU，其运行成本单独报告。表中 RF-DETR 的候选为映射到项目船类的原导出，包含空掩膜。不同架构保留各自的初始化、输入网格、精度、训练器、候选生成与阈值。两者的训练预算均为 FIT741、E24、17,856 次访问和 4,464 次更新。表中差异说明这些完整流程在同一内部口径下的效用，不能从中归因单个模块或外推到正式 TEST。 [RF-DETR 固定口径](results/SSDD_WINDOWS_REPEAT5202_FIXED_DEV187_INDEPENDENT_ACTUAL_SAFE_20261111.json)

## 固定候选预算的严格轮廓覆盖

复用四份已关闭的同批整数 DEV187 预测。低阈值输出中严格大于 FP32 0.3 的 384 条，与作者阈值原输出逐图同序、分数及 RLE 逐项一致。两份流中恰等于该阈值的行均为零。以下先按每个模型原生分数取每图前 100，再排除空掩膜。另在每图固定原生 SRNet、RF-DETR A 与 B 的非空数量最小值，四个模型都按原序取相同配额。共同配额合计 384 条，2 张图的共同配额为零。对 380 条同一整数真值按阈值做最大一对一匹配，覆盖数忽略入选后分数。 [实测汇总](results/SRNET_RFDETR_DEV187_FIXED_CANDIDATE_COVERAGE_ACTUAL_SAFE_20261111.json)

| 预测 | 原前 100 非空数量 | 原前 100 覆盖 .50／.75／.90 | 等量后覆盖 .50／.75／.90 |
| --- | ---: | ---: | ---: |
| SRNet E24 作者阈值 | 384 | 352／301／20 | 352／301／20 |
| SRNet E24 低阈值 | 12,093 | 359／305／20 | 352／301／20 |
| RF-DETR E24 原生，5202 | 10,076 | 376／350／90 | 356／324／67 |
| RF-DETR E24 中心，5202 | 10,538 | 375／349／109 | 352／326／69 |

候选增加主要恢复了较宽松阈值的 SRNet 覆盖，IoU 0.90 下仍为 20 条。逐图等量时 RF-DETR 原生及中心的严格覆盖仍为 67 和 69 条，表明候选总数不足以解释固定端点间的严格差距。不同模型的图内排序仍决定哪些候选入选，覆盖统计也无法拆分各网络设计的因果贡献。原 CPU 进程退出 0，作业树清空，耗时 1.313 秒。本诊断没有新增模型前向、AP 或正式 TEST 访问。
