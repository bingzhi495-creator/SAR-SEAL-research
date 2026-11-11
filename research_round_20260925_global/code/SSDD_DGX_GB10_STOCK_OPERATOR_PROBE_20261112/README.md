# DGX GB10 的最小 CUDA 算子核验

原锁定 PyTorch 2.10.0+cu130 发出 GB10 `sm_121` 超出构建范围的提示。实际 `32×32` BF16 GEMM 与 CPU FP32 参考逐值一致。`grid_sample` 在 BF16 autocast 下实际输出 FP32，前向和输入梯度与 CPU FP32 参考逐值相同，采样网格梯度最大绝对误差为 2.38×10⁻⁷。原进程正常退出，独立进程核对未发现残留 CUDA 上下文。 [聚合实测](RESULT_SAFE.json)

实测 CUDA 可用 2.68 GiB。完整训练的原门槛为 12 GiB，本项只建立以上小算子的实际运行证据。未启动 RF-DETR 模型或优化器。

复现使用相同离线 CUDA 环境，在 `-I -B -X utf8` 下执行本目录的 `probe_gb10_stock_torch.py`。通过两个 `--lease-file` 提供项目现有 GPU 锁的规范路径，通过 `--allow-existing-compute-basename` 声明已单独确认的非科研服务。程序只读取锁状态和进程信息。可移植版本把原执行中的主机路径和既有进程名改为参数，仅通过无 CUDA 的语法及命令行核对。 [复现协议](PROTOCOL.json)

