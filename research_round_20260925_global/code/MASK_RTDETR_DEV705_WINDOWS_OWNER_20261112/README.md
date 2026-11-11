# Mask RT-DETR DEV705 的原始 CUDA 导出与独立 CPU 计分

入口沿用固定的 P392 导出体和计分体。两臂保持 E24 EMA、原生 Paddle CUDA FP32、640 输入、800 原网格、0.5 掩膜阈值和每图 100 条原候选。原 DEV705 标签在独立 CPU 评分时读取，主口径为 1,782 条原标签，敏感性排除四条冻结空栅格。正式 TEST 继续隔离。

## 先决证据

待 B 原工具等待和 Windows 作业对象自然关闭后，使用 `verify_closed_e24.py --arm center --receipt <original-wait> --output <fresh-receipt>` 独立验收。使用原 `e24_owner.py --accept --protocol-commit 437fa55f14644eab15156dc96092903bfde147da` 验收完整 A/B 配对。需要两份独立 E24 回执及其 SHA-256 和配对验收 SHA-256。A 和 B 均须完成 24 轮及 17,640 次更新。

## 入口与外部原等待

`p392_windows_dev705_owner.py --check-sources --p392 <original-p392-directory> --stock-owner <original-cuda-owner.py>` 只核对原导出、评分及所有者源码。

执行 `--run` 需传入同样的 `--p392` 和 `--stock-owner`，另提供 `--work-root`、`--native-receipt`、`--native-receipt-sha256`、`--center-receipt`、`--center-receipt-sha256`、`--paired-receipt-sha256`、`--fit-manifest` 和 `--fit-images`。工作目录应为原执行根目录下以 `P392_MASK_RTDETR_WINDOWS_DEV705_` 开头的新目录。FIT 文件及目录使用原 P392 输入，同样的一张固定 FIT 图由图像清单 SHA-256 `a7e4d2a7ff9e8ff238adbf94a1d0397553d8746e2558c316db4e417f86f8fec9` 约束。原所有者先确认 GPU 排他交接、12 GiB GPU 和 18 GiB 主存可用，再在同一共享预留下依次监管四个原始 Windows 作业。

1. 加载接受的 A 臂 E24 EMA，对预锁的一张 FIT 图实际执行原生 CUDA 读图、推理和 100 条框掩膜导出。要求 GPU 张量、六次作者注意力回退和原始退出成功。
2. GPU 隐藏，校验及暂存固定 DEV705 原 PNG。保留独立图像阶段报告。
3. A 臂执行完整作者 CUDA FP32 导出。
4. B 臂执行完整作者 CUDA FP32 导出。

记录原 `--run` 外层工具的保留会话号、四条原子进程关闭事件与最终原始工具等待，保存 `status=P392_MASK_RTDETR_DEV705_ORIGINAL_CUDA_OWNER_TOOL_WAIT`、`original_session_id`、原始 `exact_tool_wait` 与原始 `combined_stdout`。累计外层标准输出应完整保留四条闭合事件和最终事件。每个子进程的原退出、资源保护、作业树清空、源文件附着及标准输出由所有者独立落盘。

评分使用 `--score` 及同样的原源码、工作目录、A/B 和配对引用，再提供 `--original-wait`、`--original-wait-sha256`、`--score-python` 和 `--score-output`。CPU 解释器必须有 NumPy 1.26.4 与 pycocotools 2.0.11。评分前逐项复核外层和四个子进程闭合、E24 EMA、图像暂存及配对标识，随后以 GPU 隐藏的独立进程执行原 `author_dev_score.py`。读取两臂完整预测流及摘要后才打开固定 GT。

## 已完成的检查

源码预检和两项模拟闭合检查通过。仍需实际 B 原等待与独立验收，以及实际 CUDA FIT 烟测和完整 DEV 导出；当前未访问 B 或 DEV 有效载荷，未运行 GPU 或评分。

