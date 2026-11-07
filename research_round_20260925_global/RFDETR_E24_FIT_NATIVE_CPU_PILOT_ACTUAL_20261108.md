# E24 真实 FIT 采集接线与完整 CPU 路径

2026-11-08。预登记后，DGX CPU 已对固定拟合组中的第一张图完成真实原生推理。严格读回已验收 E24 EMA 的 545 个模型张量和 4,464 次更新。原生 100 行中保留 90 个船舶候选，全部原生 top100 掩膜的原图网格正号与原后处理一致。存下来的 19,363 个预测带 logits 在独立 Windows 端用正式 COCO 解码器和不同的裁剪边界计算逐项复核，含 8,584 个正值和 10,779 个负值。原灰度位图摘要一致，两侧邻域对 17,769 个候选位置有效。第一张图的全部像素数与候选数仅用于接口核验。原 SSH 和 CPU 子进程实际退出 0，工作体约 11.19 秒，CUDA 未初始化。 [预登记与划分](RFDETR_E24_FIT_NATIVE_CAPTURE_AND_SCOPE_20261108.md) · [安全结果](results/RFDETR_E24_FIT_CPU_ONE_IMAGE_ACTUAL_SAFE_20261108.json) · [独立复核代码](code/RFDETR_E24_FIT_NATIVE_CAPTURE_CANDIDATE_20261108/capture_post.py)

已将已闭合的原 E24 检查点完整复制到本机，543,050,422 字节的 SHA-256 与原件相同。另在本机独立环境从官方摘要核验了 PyTorch 2.10.0 CUDA 13.0 及依赖，安装及依赖检查完成。实际导入在 Windows 运行库初始化处报 WinError 1114，保留失败日志。Windows GPU 入口须通过独立导入和 CUDA 冒烟检查后再绑定。 [固定的官方 PyTorch 版本](https://pytorch.org/get-started/previous-versions/#v2100)

由于 Windows 导入尚未通过，完整 FIT741 采集预先指定已通过一图接线的 DGX CPU。沿用同一原始 E24 检查点、原 741 个逻辑样本、同一预定划分和预测带。一次加载模型并按原图顺序完成 741 个前向，保留全部原生 top100 的正号核对以及每图独立的 RLE、候选索引和 FP32 logits。CPU 原任务使用两枚与 E100 监督进程不同的核、GPU 隐藏、较低优先级及资源保护。原 E100 继续按自身绑定运行。若 CPU 完整采集退出异常，保留阶段产物及原因后重新判断。若本机 CUDA 稍后可用，先做同一张图的独立接口检查。完整采集与两端输出的数值一致性仍需实测。 [完整采集工作体与边界测试](code/RFDETR_E24_FIT_NATIVE_CAPTURE_CANDIDATE_20261108/native_fit_population_worker.py)

当前为图像侧接线结果。采集不读标注、DEV 或 TEST，无新增优化器更新或 AP。冻结载体之后，才按预留的拟合、校准和留出图执行同头、同候选和同 16 像素上限的选择器比较。
