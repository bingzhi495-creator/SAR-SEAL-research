# DINO 高效注意力反向的实际局部证据

按[固定残余定位方案](RFDETR_E24_COMBINED_SAMPLING_BACKWARD_CONTROL_ACTUAL_20261108.md)运行一次真实 FIT 原生前向和同图双反向。原进程退出 0。固定输入与完整前向和原 F0 的首微批次逐字节一致。各反向使用相同 RNG；一次前向实际路由 9 个网格采样及 2 个普通双线性上采样，双反向全部交由 CPU 原生算子。实际优化器和 EMA 更新均为 0。

4 个 DINO 编码器输出及投影特征收到的梯度均精确相等；位置编码与掩膜不求导。patch 投影输出收到的梯度仍不等。在选定且全部可比较的 128 个节点中，观察到一个同入梯度、异出梯度节点，位于 DINO 编码器的 `ScaledDotProductEfficientAttentionBackward0`。单图两次叶子梯度的相对 L2 为 1.282011%。此证据范围为一个真实微批次及所观察节点。 [实际安全汇总](results/RFDETR_E24_EFFICIENT_ATTENTION_BACKWARD_NODE_ACTUAL_SAFE_20261108.json) · [精确节点观察源码](code/RFDETR_E24_COMBINED_BACKWARD_NODE_TRACE_20261108/combined_backward_node_trace.py)

## 下一项确定性反向定位控制

Windows 运行时实际为 PyTorch 2.10.0+cu130，Git 源码 `449b1768`。作者源码中，高效注意力 CUDA 反向在严格确定性模式下将键分块数设为 1。仅警告模式沿用原分块。 [PyTorch 固定源码](https://github.com/pytorch/pytorch/blob/449b1768410104d3ed79d3bcfe4ba1d65c7f22c0/aten/src/ATen/native/transformers/cuda/attention_backward.cu#L853-L865)

仍保留已验证的网格采样和普通双线性上采样 CPU 反向。F0/F1 固定各四个真实增强 FIT 微批次，进程启动时设 cuBLAS 工作区，原生 CUDA 前向期间保持原确定性模式，严格模式只包围实际反向并在退出时恢复。遇到不支持的算子保留失败，暂不更换实现或放宽条件。严格核对原有前向、匹配和损失摘要、真实两类路由计数、四次严格反向以及零更新。只在全梯度相对 L2≤10⁻⁵、余弦≥0.999999且其余范围匹配时，通过新的数值控制资格。前述失败 F/S 的端点仍封存。
