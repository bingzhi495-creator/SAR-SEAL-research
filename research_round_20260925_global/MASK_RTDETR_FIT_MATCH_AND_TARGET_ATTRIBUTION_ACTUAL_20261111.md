# Mask RT-DETR 掩膜取样与查询匹配的 FIT 实测

## 作者发布配置与实际分类损失

固定 PaddleDetection v2.9.0 提交 `b25522a0` 的 YAML 配置写有 `use_vfl=True`、`vfl_iou_type='mask'`。原生工作体实例化实际得到 `use_mal=True`、`use_vfl='mask'`、`vfl_iou_type='bbox'`。MaskDINOLoss 以位置传递两个参数，父类在 2025 年 DEIM 合入时新增了前置 `use_mal`，旧调用因此发生位置偏移。分类代码优先选择 MAL，主层质量来自框 IoU。2024 年的 S／M／X 作者发布版本父类还没有该参数，原始发布文档明确写明掩膜 IoU 与置信度对齐。现有资产入口已见于当年的发布记录，权重训练使用的精确提交未从资产中独立取得。 [当前配置](https://github.com/PaddlePaddle/PaddleDetection/blob/b25522a0f4bde8c80603f3ba5e3472059972e3b5/configs/mask_rtdetr/_base_/mask_rtdetr_r50vd.yml#L61-L74) · [父类参数新增](https://github.com/PaddlePaddle/PaddleDetection/commit/16198d4f343adb94462b4323a33a798392be11a0) · [当前子类调用](https://github.com/PaddlePaddle/PaddleDetection/blob/b25522a0f4bde8c80603f3ba5e3472059972e3b5/ppdet/modeling/losses/detr_loss.py#L650-L661) · [早期 S 模型发布](https://github.com/PaddlePaddle/PaddleDetection/commit/96483f21bcecae8cf460b792963ff24393e23a5e)

两臂原长训同用这个固定版本。保留两臂的原实现以回答这一载体下掩膜采样干预的相对效用。恢复早期掩膜 IoU 分类的效果需要另行锁定同预算对照。该实现差异也不能用于回推作者公开 COCO 成绩。

## 冻结单批的四格结果

以已闭合的 A 臂八步 FIT EMA 为初始化，使用作者增强及读取器复放预定首批四图。原图输入、框、类别、变换后形状、原生与中心掩膜逐项等于先前 CUDA 短训所记录的输入指纹。六条掩膜均改变，监督实例及其非空状态一致。一次 CPU 训练态无梯度前向冻结实际最终普通层的 300 个查询，然后用原生损失重放匹配 A／B 与标签 A／B 四格。各格的 12,544 个每图匹配采样点逐位相同，固定匹配时不确定性损失采样坐标也相同。 [正式配对协议](MASK_RTDETR_HRSID_FIT2937_E24_AUTHOR_PROTOCOL_20261111.md) · [作者匹配器](https://github.com/PaddlePaddle/PaddleDetection/blob/b25522a0f4bde8c80603f3ba5e3472059972e3b5/ppdet/modeling/transformers/matchers.py#L124-L177)

| 匹配来源／损失标签 | 分类 MAL | 框 L1 | GIoU | 掩膜 BCE | Dice |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原生／原生 | 8.182596 | 1.072278 | 1.609098 | 0.229720 | 1.694290 |
| 原生／中心 | 8.182596 | 1.072278 | 1.609098 | 0.231558 | 1.874098 |
| 中心／原生 | 8.182596 | 1.072278 | 1.609098 | 0.229720 | 1.694290 |
| 中心／中心 | 8.182596 | 1.072278 | 1.609098 | 0.231558 | 1.874098 |

六条 GT 的匹配查询都相同。中心标签使最终层加权 BCE 增加 0.001838、Dice 增加 0.179808，分类质量目标逐值相同。本批观察到的变化来自固定匹配下的掩膜监督。两套标签定义的损失差只描述这批冻结预测的局部敏感性，不能据此推断训练后轮廓或 AP。范围限于四张 FIT 图和最终普通层的一个早期状态。 [实测摘要](results/MASK_RTDETR_FIT_MATCH_AND_TARGET_ATTRIBUTION_ACTUAL_SAFE_20261111.json)

原预锁文件保持原样。前两次在前向前拦下 YAML 与运行时分支差异，另存解释后按原生实际语义重放。下一次因把 Paddle 注册的归一化移动统计当作可训练权重检查而退出。独立修订核实了 106 个变化均为停止梯度的 BatchNorm 移动统计，所有可训练参数精确保持。最终工具原会话退出 0，工作体 117.060 秒。无优化器更新、GPU、DEV 或 TEST。 
