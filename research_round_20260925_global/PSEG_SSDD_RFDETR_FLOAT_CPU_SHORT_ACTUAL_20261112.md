# PSeg-SSDD 小数标注的 RF-DETR 真实 CPU 配对短训

用固定 FIT741 小数多边形、同一 COCO Q100 初值和种子 5201，以单线程 CPU FP32 实际训练原生及逐步中心两臂。每臂用作者 RF-DETR 模型、训练步骤、AdamW、调度和 EMA。两臂实际使用同一批非空监督实例，完整保留空访问。CPU 的随机数消费与 CUDA 存在差别，本次只验证 CPU 训练入口和真实监督传递。 [小数配对训练方案](PSEG_SSDD_OFFICIAL_FLOAT_COMMON_TRAIN_PROTOCOL_20261111.md) · [配对实测回执](results/PSEG_SSDD_RFDETR_FLOAT_CPU_SHORT_ACTUAL_SAFE_20261112.json)

| 实际量 | 原生 | 中心 |
| --- | ---: | ---: |
| FIT 访问 | 32 | 32 |
| AdamW／EMA 更新 | 8／8 | 8／8 |
| 共同非空实例 | 57 | 57 |
| 无共同实例访问 | 2 | 2 |
| 训练主体耗时 | 245.19 秒 | 243.25 秒 |
| 原进程峰值工作集 | 3.38 GB | 3.39 GB |

独立读回核对两臂所有 32 批的有序图像、框、类别、随机状态和前后监督计数。30 批的实际掩膜不同，两臂没有独有的非空实例，原空掩膜保护未再删行。损失和抽查梯度有限，两臂分类头参数相对共同初值均发生变化。原生正常退出后才启动中心。两个工作进程及协调进程均退出 0，未保存端点权重、计算 AP 或读取 DEV／TEST。首次协调启动因隔离训练解释器缺少 psutil 而在训练前退出。后续仅更换监控解释器，保留相同的训练解释器和配方。

DGX 原 CUDA BF16 短训适配已完成代码和读取验收。以单个数据集实例的标注入口覆盖小数多边形，保留原图像和全部非掩膜字段。741 图、1,661 条记录与直接加载小数文件的结果一致。32 张图的 43 条标注另通过原生 COCO 张量精确核对。当前尚无 DGX CUDA 短训或小数训练的正式 E24 分数。 [CUDA 入口与验收说明](code/SSDD_DGX_FLOAT_COMMON_SHORT_20261111/README.md)

## 可移植复现入口

发布源码把已执行版本的本机路径改为显式参数。单臂入口的 `--preflight` 已实际读回完整 FIT741，协调入口的参数解析已验证。完整两臂结果来自先前验收的原执行版本。复算时，Windows 监控解释器需有 psutil，训练解释器用固定 RF-DETR 1.12.0 环境。协调入口接收代码根、FIT 专用布局及回执、依赖覆盖、官方 COCO 初值、训练解释器、工作脚本摘要和独立输出目录。 [协调入口](code/SSDD_FLOAT_RFDETR_CPU_SHORT_20261112/owner.py) · [单臂入口](code/SSDD_FLOAT_RFDETR_CPU_SHORT_20261112/cpu_short.py) · [原预锁](code/SSDD_FLOAT_RFDETR_CPU_SHORT_20261112/PROTOCOL.md)

## DGX 原训练请求实际预检

已部署 GitHub 固定版本的 CUDA 入口和本次训练请求。原 DGX CPU 预检正常退出，核对原训练器绑定、全部 FIT 图像与输入目录、741 个样本、1,661 条实例及 1,606 条小数顶点变更。预检未创建训练输出目录。随后同一固定环境测得 CUDA 即时可用 2.91 GiB，主机可用约 44.74 GiB。未达到原训练入口的 12 GiB CUDA 可用条件。探针报告 GB10 原生 BF16 可用，当前 PyTorch 构建的架构表未直接列出 sm_121，实际训练算子有待验证。原服务保持运行，未启动 CUDA 配对。 [实际预检与资源回执](results/SSDD_DGX_FLOAT_SOURCE_FIT_PREFLIGHT_ACTUAL_SAFE_20261112.json)
