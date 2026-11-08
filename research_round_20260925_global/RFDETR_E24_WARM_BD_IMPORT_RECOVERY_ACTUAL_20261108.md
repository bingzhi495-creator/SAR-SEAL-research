# RF-DETR 等目标 B/D 的首次导入退出与实际启动

首个本机 B 原进程在导入原生训练包时退出 1。训练包入口同时导入评价回调，触发本机推理环境未安装的 `faster_coco_eval`。该原件无 FIT 前向、优化器更新或模型端点，保留实际失败记录。

恢复版从固定的 RF-DETR 1.12 源码直接加载三个原生模块，沿用参数分组、累积补齐和 EMA。三份本机源码与官方固定提交的原字节摘要一致。CPU 依赖检查通过，没有打开 FIT 图像或初始化 CUDA。恢复版未改变实验协议或采样算法。

新的配对原件已启动，首个 B 工作进程经同出生时间复查仍在运行，实际观察到 32 次优化器更新、128 个微批。D 在 B 的技术端点闭合后按预锁顺序执行。最终效用由三组统一 DEV187 评分决定。 [配对协议](RFDETR_E24_EQUAL_RISK_WARM_BD_SCOUT_PROTOCOL_20261108.md) · [固定训练包导入](https://github.com/roboflow/rf-detr/blob/639eac4555ceb89ef399ccda3fde716bbcfe34cb/src/rfdetr/training/__init__.py)
