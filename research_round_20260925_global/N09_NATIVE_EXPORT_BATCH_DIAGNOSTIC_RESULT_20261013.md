# N09 原生输出批处理差异诊断

北京时间 2026-10-13。状态：一次新受限 GPU 诊断真实退出 0，ROOT 已核对原始所有者、输入引用和结果。范围仅为首个规范 DEV 图像 image_id=4 的全部 100 个候选，没有全 DEV 评价或新的 AP。

## 实际观察

同一归档 M200 与最终类别 logits，采用原 exporter 表达式，比较原始自然顺序 all100 渲染后重排和先取四候选再渲染的 chunk4 路径。

| 相对归档的精确比较 | 原始 all100 | chunk4 |
|---|---:|---:|
| 候选顺序不一致 | 0 | 0 |
| 类别概率不一致 | 0 | 0 |
| RLE / 二值支持不一致的候选 | 0 / 0 | 0 / 0 |
| mask quality 不一致的候选 | 0 | 36 |
| S0 分数不一致的候选 | 0 | 29 |
| 最大 S0 绝对差 | 0 | 1.1920928955078125e-7 |

两条路径的全部 100 份渲染 FP32 logit 张量也逐元素相同，差异像素为 0；原始 all100 的完整预测字典与归档相同。首图差异发生在分块 mask quality / S0 计算，尚未证明是哪一个 CUDA kernel 或归约算法造成。没有引入数值容差或改用其他分数，也没有把这次运行计为偏移控制的 AP。

## 决定与范围

新 SOURCE 修复恢复原始自然顺序全100渲染、候选重排和全100质量分数计算；先完整精确验证原生输出，再在同一原生 logit 的小块上施加未改变的 FIT-only δ 和带约束。旧失败尝试、源码和输入保持原样。全187图、全18,700候选仍须通过原生精确复现及后续独立 CPU 核验，才能引用新的控制 AP。

首图诊断是评价实现回归证据，不证明全 DEV 已修复，不证明 C 需要空间条件机制，也不构成新方法。当前目标仍为检验普通全局偏移能解释多少 C 的固定协议收益；FIT δ 不重新拟合，DEV 不做阈值扫描。

## 执行依据

- 新 phase：N09_first_image_native_export_batch_diagnostic_GPU_20261013_01。
- 原始 PID：70256；实际退出码 0；原始句柄确认终止并已回收，无资源中断或等待错误。
- 所有者用时 19.578 s，整卡观测峰值 2,092 MiB；这不是训练成本或每进程完整 VRAM 证明。
- 诊断报告：550,283 B，SHA-256 d50ca39b131075abf589261c0ba5040eda31d689ae230ee7e9759fccdff7df89。
- ROOT 审阅记录：3,659 B，SHA-256 9b9c96b79eb2f6c69cf370d4ef19b6bac10574b14fd0938abbc7860e56eee586；原始科学执行资料保留本机，普通摘要同步仓库。
- [本次实际诊断 SOURCE](multimodal_data_pilot_20260928/native_geometry_paired_mask_readout_preparation_20261009/N09_FIT_only_global_logit_offset_control_source_20261012_01/diagnose_native_export_batches.py)：45,568 B，SHA-256 013068a066312680425079658ddc80747334aef2b06a49362795ea04c5ccc2e6。
- [当前直接先例核查](CALIBRATION_PRECEDENT_REVIEW_20261012.md)：PointRend 与 Implicit PointRend 的竞争性解释仍待公平实测。
