# E24 同图 CUDA 接线验收与 FIT 主采集设备

2026-11-08。本机独立 Windows 环境已完成真实 CUDA、TorchVision NMS、JPEG 与 COCO 掩膜编解码冒烟测试。完整安装的 76 个 wheel 均按发布方官方摘要核验。初次启动导入 PyTorch 时，Conda 应用目录的旧 C++ 运行库造成 DLL 初始化失败。隔离启动器沿用同字节 Python 3.10.13 和同一组已核验依赖，改用系统现有 C++ 运行库后，原真实 CUDA 检查实际退出 0。系统和旧环境文件保持原样。

原 E24 EMA 的 545 个张量、4,464 次更新及原检查点 SHA-256 保持一致。同一张预先固定的 FIT 图在 Windows RTX 3090 Ti 实际执行原生前向并退出 0，工作体约 6.047 秒。独立复核按同一原图重算带、索引、RLE 和 logit 正号，并复现灰度位图摘要。 [前一轮 CPU 接线](RFDETR_E24_FIT_NATIVE_CPU_PILOT_ACTUAL_20261108.md) · [安全结果](results/RFDETR_E24_FIT_WINDOWS_CUDA_ONE_IMAGE_ACTUAL_SAFE_20261108.json)

| 同一 FIT 图 | DGX CPU | Windows CUDA |
| --- | ---: | ---: |
| 原生 top100 中船舶候选 | 90 | 89 |
| 原图预测带 logits | 19,363 | 17,003 |
| 有效的两侧灰度邻域 | 17,769 | 15,503 |

两端共有 87 个原始查询，其中 59 个的原图掩膜完全相同，16 个的原生排名完全相同。数据和灰度输入身份一致。CPU 和 CUDA 的计算后端、硬件及浮点矩阵乘法策略不同，这个同图观察无法分解各因素作用。它证明两套原生预测和稀疏 logits 必须分开绑定。

## 完整采集与后续比较

按上一轮预登记启动的 DGX CPU 原进程已独立确认在 E100 之外的两枚 CPU 核运行，启动时已观察到 25 张完整图像的进度事件。保留该原始运行及其 CPU 口径。同时，在查看任何 FIT 标签或选择器成绩之前，将**完整 Windows CUDA 采集**锁定为下一项小型选择器试验的主输入。这个决定基于本机 CUDA 通过真实接线验收及正式预测使用 GPU 的设置。CPU 结果作为独立设备记录。所有拟合、校准与留出候选在各自设备内来自同一次固定载体采集。Windows 完整采集沿用已发布的 741 图工作体、原生 top100、原 576 变换和 E24 原权重，FP32 eager，矩阵乘法设为 high。继续独立核验完整输出与终态。 [完整工作体](code/RFDETR_E24_FIT_NATIVE_CAPTURE_CANDIDATE_20261108/native_fit_population_worker.py) · [固定的图像划分](RFDETR_E24_FIT_NATIVE_CAPTURE_AND_SCOPE_20261108.md)

当前只获得图像侧原生输入。选择器的同预算收益、伤害和 AP 待完整捕获通过验收后实测。
