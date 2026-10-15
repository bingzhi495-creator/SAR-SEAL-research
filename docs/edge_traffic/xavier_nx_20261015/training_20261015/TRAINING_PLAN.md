# YOLO11n路口微调首轮计划

日期：2026-10-15。执行状态：数据获取中，训练尚未开始。用户明确授权训练并看效果。

## 本轮目标及范围

用Intersection-Flow-5K官方v4的带框标注，检查YOLO11n从COCO预训练微调后的路口检测表现。不是新方法、SOTA或Xavier NX部署验收。电动自行车类别没有得到独立定义，暂不做五类已达标结论。

## 预先固定配置

- 数据：官方v4，原划分train5483 / val722 / test723；按实际完整性审计后的可用范围执行，任何排除必须在读指标前记录。
- 初始化：上一轮已核验的官方yolo11n.pt，SHA256 0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1。
- 原生8类全部保留：vehicle/bus/bicycle/pedestrian/engine/truck/tricycle/obstacle。engine不改名为电动自行车；车辆与骑行者框定义要抽查。
- 训练预算：30 epochs，640×640，batch16，seed42，单GPU；不做多模型/多seed挑选，patience设大于预算以执行固定周期。
- 优化器SGD，lr0=0.01、lrf=0.01、momentum=0.937、weight_decay=0.0005、warmup_epochs=3；默认标准增强，最后5轮关闭mosaic；禁用在线实验平台。完整实际参数落盘。
- best按val选择；test只作最终锁定模型评价，不据test选择模型、调置信度或加预算。保存last和best。
- 评价：原生8类mAP50/mAP50–95与每类指标；与COCO初始化比较仅限可映射的vehicle/car、bus、bicycle、pedestrian/person四类，同图、同输入、同阈值和相同COCO评估器。
- 常用运行点：conf0.25、IoU0.7，AP计算用conf0.001候选，max_det300。置信度和NMS参数不根据结果调整。
- 速度与视频：同一719帧Pexels路口视频，固定640、FP16、batch1，标清RTX3090Ti，保留罐车误报公交帧对照；不把视频视觉表现算作准确率。

## 诚信检查

检查图像/标签配对、坐标有效性、空标注、每类数量、跨划分精确重复和可推断的视频近邻。若官方划分不能证明按路口分隔，则只报告官方划分内性能，不宣传跨路口泛化。不得从文件名前缀自动认定相机身份。

30轮是首轮固定预算，并不代表充分收敛；单seed结果不能说明稳定性。短预算负结果不能说明方法总体无效。数据域与类别定义差异限制COCO初始化对照的解释。

## 获取与发布

官方入口：https://www.kaggle.com/datasets/starsw/intersection-flow-5k ，作者仓库：https://github.com/AstronZh/Intersection-Flow-5K 。v4元数据与真实ZIP目录已读取；完整JPEG、YOLO标签正在按ZIP CRC/字节范围核验下载，不另存6GB整ZIP。

原始数据、全量标注、权重和运行缓存仅本地。自有计划/源码/配置/经核验摘要及时同步。运行中不为同步读取科学载荷或改写其绑定源码。任何工程失败如实保留并在恢复前说明。
