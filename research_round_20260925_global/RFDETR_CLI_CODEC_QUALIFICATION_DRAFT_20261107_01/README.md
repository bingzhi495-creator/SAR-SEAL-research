# RF-DETR 原生 CLI 和编解码检查

截至 2026-11-07，自有 probe.py 已通过语法和接口核对，等待在已安装依赖的 DGX 独立 CPU 环境中执行。公开副本保存诊断内容和原生入口，实际数据和执行绑定保留在执行侧。

- 调用 rfdetr.cli.main() 走 fit --config … --print_config，核对冻结 576 配方的模型配置、类别、查询数、评估后端和训练轮数。实例化哨兵拦截模型、Trainer 或数据模块构造。该模式在应用参数链接和最终验证前退出，结论范围为实际配置打印与指定字段检查。[RF-DETR 固定源码](https://github.com/roboflow/rf-detr/blob/639eac4555ceb89ef399ccda3fde716bbcfe34cb/src/rfdetr/training/cli.py#L572) · [jsonargparse 4.46.0 打印分支](https://github.com/omni-us/jsonargparse/blob/v4.46.0/jsonargparse/_core.py#L340-L355)
- 构造 8×13 RGB 合成图，调用 simplejpeg 快路径、原生 PNG 解码和 headless OpenCV 色彩转换。快路径要求实际返回有效结果。PNG 与输入逐像素一致。OpenCV 检查实际来源和运算结果。
- 同一合成矩形分别编码为 ASCII 压缩 RLE 和 polygon，核对原生解码后两张掩膜逐像素一致及面积18。随后用实际 faster_coco_eval 执行原生 bbox、segm 指标，完美匹配的预期值为1。记录实际 evaluator 模块来源。[固定评价实现](https://github.com/roboflow/rf-detr/blob/639eac4555ceb89ef399ccda3fde716bbcfe34cb/src/rfdetr/training/coco_map.py#L467)

准备执行的环境隐藏 CUDA，限制权重和真实数据读取，并在实际原任务退出后核对进程闭合。真实数据入口、原生前向及优化器更新需要另外验证。
