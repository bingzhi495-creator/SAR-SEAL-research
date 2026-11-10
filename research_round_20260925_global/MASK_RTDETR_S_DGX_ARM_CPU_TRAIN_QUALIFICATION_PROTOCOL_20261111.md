# Mask RT-DETR-S 的 DGX ARM CPU 两步资格

Windows 原生作者读取和 COCO 前向已通过。Windows 两步训练只观察到首批前向并受资源保护停止，原训练更新次数未验收。本轮预锁为独立 DGX ARM CPU 资格，沿用 8 张 FIT 图、16 条原始整数多边形、单类别、种子 20261111、原生四图批次及自然结束的一轮。作者 S 配方的原始 ImageNet B1 骨干加载方式、增强、匹配、损失、AdamW、学习率及 EMA 保留。Windows 与 DGX 具有不同运行时和资源，两个运行分别报告。 [Windows 记录](MASK_RTDETR_AUTHOR_CPU_FIT_QUALIFICATION_PROTOCOL_20261111.md) · [作者 S 配置](https://github.com/PaddlePaddle/PaddleDetection/blob/b25522a0f4bde8c80603f3ba5e3472059972e3b5/configs/mask_rtdetr/mask_rtdetr_hgnetv2_s_6x_coco.yml)

固定 DGX aarch64 CPython 3.12.3，PaddlePaddle 2.6.2 CPU wheel SHA-256 `8f1d922d0afad50352150f6caaf8781cb9850734ef5f15e96347be048fd46741`，OpenCV 4.5.5.64、pycocotools 2.0.8、NumPy 1.26.4 和作者提交 `b25522a0f4bde8c80603f3ba5e3472059972e3b5`。隐去 CUDA，CPU 固定为 2、3，双线程，低调度优先级。开始要求主机可用内存至少 52 GiB。进程树 RSS 限 12 GiB，运行中保留至少 40 GiB 主机可用内存，墙钟限 420 秒。先达到的保护条件与原等待退出分开记录，保护只作用于本次进程组。DGX 上的 CUDA 即时可用量属于独立 GPU 入训条件。 [官方 ARM CPU wheel](https://pypi.org/project/paddlepaddle/2.6.2/#files)

须观察每次真实的优化器、调度和 EMA 调用前及返回后的记录，独立复核两次有限前向、每图一次、单类槽、网络可训练分类与掩膜头张量变化、保存后在线与 EMA 状态、原生 AdamW 和调度张量，以及原进程等待和进程组清空。两步在作者预热段内，范围只限训练资格。模型 AP 与跨域效用仍由预锁的同预算长训判断。

## ARM 依赖与作者源验收

DGX 原生 ARM CPU wheel 直连下载在 120 秒超时，已收 33,687,256／74,273,370 字节。以本机从 PyPI 获取的 70 个 ARM 与通用 wheel 及已核对资产重新封装。传输原进程正常退出，DGX 确认 332,023,782 字节、SHA-256 `a77c560a2d16d57d151bd70a7afdaadbfc46f3bf375bd71ae6dec29f52a92d64`。外部资产和原始数据保留在独立私有目录。Windows Git 归档曾转换行尾，传输后根据关闭的官方提交清单在独立目录恢复并逐个核验 2,114 个源码文件，其中 2,011 个有行尾转换。首次 ARM 导入发现 CPython 3.12 venv 缺少 Paddle 运行时所需的 setuptools。离线补入 PyPI `setuptools==75.8.2` 后，依赖检查、Paddle CPU 实际张量运算和完整作者 `ppdet` 导入均通过。失败尝试发生在科学训练前。 [Paddle ARM wheel](https://pypi.org/project/paddlepaddle/2.6.2/#files) · [setuptools wheel](https://pypi.org/project/setuptools/75.8.2/#files)

## 实际两步与独立验收

DGX 原生训练器完整运行一轮，包含 8 个固定 FIT 访问，每张恰一次。两批各四图，原生网格为 512² 与 672²，16 条源标注经实际增强保留 15 条，均为非空掩膜且落在类别槽位 0。作者 B1 骨干实际加载 210 个张量，其他张量保持种子 20261111 的作者初始化。实际注意力绑定为 CPU 回退模块。两次前向损失为 256.9460、255.1535，分别见证两次原生 AdamW、学习率调度与 EMA 调用进入并返回。两步位于原作者预热段。 [实际工作进程汇总](results/MASK_RTDETR_S_NATIVE_ARM_CPU_FIT8_TWO_UPDATE_ACTUAL_SAFE_20261111.json)

训练后 1 个可训练分类头张量和 6 个可训练掩膜头张量发生变化，保存后的在线模型与训练器恢复的在线状态逐值一致。在线与 EMA 各有 646 个有限状态张量，557 项不同。优化器归档的 1,468 个张量有限，原调度器相对初始推进两步，完成轮次为 1。线上 `.pdema`、EMA `.pdparams` 与 AdamW `.pdopt` 三个端点在原进程结束后另行逐字节摘要核对。原进程实际等待退出 0，进程组已清空。墙钟 125.401 秒，监督抽样树 RSS 峰值 9.35 GiB，进程自身峰值 9.70 GiB，主机最少可用 43.59 GiB。未触发预锁资源保护。 [原关闭与独立读回](results/MASK_RTDETR_S_NATIVE_ARM_CPU_FIT8_ORIGINAL_CLOSURE_SAFE_20261111.json)

该端点通过 ARM CPU 两步训练资格，支持完整官方读取、前反向、优化和档案流程可用。Windows 资源中断原记录单列。当前 ARM 执行尚无开发集 AP 或跨域模型比较。正式 HRSID 原训练及同预算 SSDD RF-DETR 复验优先级更高。 [ARM 训练入口](code/MASK_RTDETR_AUTHOR_CPU_QUALIFICATION_20261111/arm_two_steps.py) · [ARM 环境边界](code/MASK_RTDETR_AUTHOR_CPU_QUALIFICATION_20261111/arm_runtime.py)

## 第二载体的配对短窗预锁

两臂分别从同一份 ImageNet B1 权重和种子 20261111 重新初始化，使用固定 PSeg-SSDD FIT 的 8 个 JPEG 内容组及原整数多边形。原生 A 保留作者 `BatchRandomResize` 中的位图最近邻，中心 B 只把同一次缩放的位图改为中心取样。两臂均沿用原训练器、四图批次、匹配、损失、调度、AdamW 及 EMA。要求各完成 8 次实际访问、2 次优化器／调度／EMA 返回。比较全状态初值、同序样本、实际图像张量、框、类别、裁剪保留的有序标注及增强随机状态。需观察到共同掩膜改变。任一臂出现空掩膜则保留证据并拒绝短窗验收，另行设计共同非空条件。基于位图差异引起的匹配或损失变化属于此次训练干预的一部分。 [作者批次缩放](https://github.com/PaddlePaddle/PaddleDetection/blob/b25522a0f4bde8c80603f3ba5e3472059972e3b5/ppdet/data/transform/batch_operators.py#L137-L150) · [配对入口](code/MASK_RTDETR_PAIRED_SHORT_20261111/paired_short.py)

配对入口已通过三项合成测试，尚未运行真实配对。实际 CPU 作业沿用至少 52 GiB 启动可用主存、40 GiB 运行下限、12 GiB 进程树 RSS 和 420 秒墙钟。每臂单独原始退出后再复核。后续如正式扩展，先固定同一平台上的 FIT741 及作者四图批次、尾批处理、E24 更新次数、固定原始整数 DEV187 和作者推理设置。两步的损失、合成几何及短窗资格不作为 AP 效用结果。
