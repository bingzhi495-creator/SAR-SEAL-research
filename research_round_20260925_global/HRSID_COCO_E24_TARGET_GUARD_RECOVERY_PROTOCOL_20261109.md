# HRSID COCO 原生 E24 原运行中止及恢复资格

原运行通过官方 COCO 权重初始化和原生 164 查询资格检查，完整完成 4 轮，记录 2,940 次 AdamW 与 EMA 更新。进入第 5 轮后，最后公开进度为第 3,100 次更新。原进程实际退出 1，工作体 4,082.53 秒，无资源中止，未产生正式 E24 端点，未运行 DEV705。异常为 `on_train_batch_start` 的合并掩膜及边框断言。原异常未保留具体失败图像、行、步号或失败子条件，第 3,100 步是已见日志下界。只保留原 E1 归档与已完成账本。 [原运行失败聚合](results/HRSID_COCO_NATIVE_E24_ORIGINAL_FAILURE_SAFE_20261109.json)

原生 torchvision Resize 通过不同路径缩放框和掩膜。布尔掩膜的最近邻重采样可能清空原本非空的细小掩膜，几何框仍可合法。Crop 会同步剔除刚被裁空的行，但保留裁前已为空的框行。这是具体的候选机制。先在同一已封存 FIT 布局做 CPU 复现，将类别、掩膜非空、框有限、框正面积和边界拆开统计，记录原 resize 的消失阶段。 [原生 resize/crop 源码](https://github.com/roboflow/rf-detr/blob/639eac4555ceb89ef399ccda3fde716bbcfe34cb/src/rfdetr/datasets/_torchvision.py)

从 2,937 张 FIT 图中按每图最小正面积非 crowd 标注排序取 32 图，同面积按原图编号排序。每图固定 12 个种子，用原 576、torchvision 双分支增强和同一类别映射。使用真实 PNG 与原派生训练标注，不读 DEV 或 TEST，不创建模型或优化器。复现只说明原机制可出现，不能定位失败原行。若复现的合法框空掩膜有原生 resize 直接证据，设计后变换单次同步过滤，同时验证所有有效行字节不变、图像和原采样配方不变。类别、非有限或非法框仍独立中止。后续新启动或检查点接续必须单独锁定和验收。 [复现源码](code/HRSID_NATIVE_TARGET_GUARD_AUDIT_20261109/audit_target_guard.py)

