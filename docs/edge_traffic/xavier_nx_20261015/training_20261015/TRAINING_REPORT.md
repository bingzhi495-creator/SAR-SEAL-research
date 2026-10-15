# YOLO11n 路口微调实测（2026-10-15）

**已实际完成30轮训练、128张保留测试评价和719帧公开路口视频推理。测试集四类AP改善，但行人、实际异域视频误检、电动自行车定义仍不足，尚未达到原定五类高精度和Xavier NX 30FPS验收。**

## 实验范围与实际预算

Intersection-Flow-5K官方v4原划分内固定抽样；因实际网络下载较慢，在读取任何精度前将首轮锁定为1024训练/128验证/128测试、30轮。两张训练图原始框坐标越界，在读指标前排除整图且未替补，实际**1022/128/128**；val/test标签未改。

全量不是本次执行范围。抽样seed20261015，训练seed42；跨划分SHA256完全相同图为0，未证明路口/视频近邻独立。结果仅适用于这份官方划分内的小规模单seed实验，不表明跨路口泛化或已充分收敛。

模型从官方COCO yolo11n.pt初始化，原生8类全部训练。640×640、batch16、SGD、lr0=0.01、lrf=0.01、momentum0.937、weight_decay0.0005、warmup3、最后5轮关闭mosaic、AMP；完整参数见training_config.json。训练30轮均执行；best由val选取，test不参与选模型和调阈值。

RTX3090Ti；results.csv末轮累计训练计时约205.7秒，不含下载、初次失败、加载及最终评价。第1次启动因OpenMP重复线程库在0 epoch退出，原日志保留；同一最小复现验证MKL_THREADING_LAYER=SEQUENTIAL后以恢复入口重启，训练数据、初始化和预算不变。未改共享DLL。

## 训练前后同协议对照

同128张测试图、真实官方GT、4个可映射类别、640方形输入、FP16；候选conf0.001、NMS IoU0.7、max_det300；AP用官方pycocotools，IoU0.50:0.95、101点、maxDets100。vehicle与COCO car、bicycle骑行者框等语义尚未完全一致审定，因此这反映数据集适配效果，不是统一训练下优于其他模型的结论。

| 类别 | GT框数 | 训练前AP50 | 训练后AP50 | 训练前AP50–95 | 训练后AP50–95 | 训练后P | 训练后R |
|---|---:|---:|---:|---:|---:|---:|---:|
| 车辆（vehicle） | 4188 | 43.81% | 73.61% | 27.22% | 52.69% | 75.08% | 71.63% |
| 公交车 | 60 | 17.11% | 58.87% | 13.59% | 46.12% | 76.74% | 55.00% |
| 自行车（数据集定义） | 556 | 0.87% | 43.66% | 0.41% | 20.93% | 62.63% | 42.81% |
| 行人 | 476 | 8.32% | 16.33% | 3.79% | 6.78% | 65.66% | 13.66% |
| 四类平均 | 5280 | 17.53% | 48.12% | 11.25% | 31.63% | — | — |

P/R为固定conf0.25、匹配IoU0.50的官方COCO匹配结果；AP不是单阈值准确率。

mAP50提高30.59个百分点，相对增加174.5%；mAP50–95提高20.38个百分点，相对增加181.1%。没有多seed，不报告稳定性或显著性。

## 原生8类评价（另一个评价口径）

Ultralytics 8.3.43，test128、max_det300；原生8类mAP50 **39.92%**，mAP50–95 **26.30%**。以下不能与上表混算：SDK的插值/最大检测数口径与官方COCO四类不同。

| 原生类别 | AP50 | AP50–95 |
|---|---:|---:|
| vehicle | 76.50% | 55.85% |
| bus | 62.23% | 47.94% |
| bicycle | 44.19% | 21.77% |
| pedestrian | 18.47% | 7.40% |
| engine | 10.79% | 7.33% |
| truck | 35.12% | 23.59% |
| tricycle | 55.67% | 40.89% |
| obstacle | 16.39% | 5.63% |

engine只有115个训练框、14个测试框，且作者说明没有将其独立定义为电动自行车。不能将engine改名后声称满足电动自行车检测；本轮没有该类可认可的单独精度。

## 公开路口视频速度与实际错误

同一Pexels视频，1920×1080、29.970FPS、719帧/23.99秒，逐帧处理无跳帧。固定640、FP16、batch1、conf0.25、NMS IoU0.7、预热30帧。以下为RTX3090Ti + PyTorch单次顺序测试，并非Xavier NX或TensorRT。训练前保留6个COCO交通类，训练后保留8个原生类；后处理类别不同且单次测量，不把约4%的速度差解释为模型显著加速。

| 模型 | SDK网络推理ms/帧 | 解码+预处理+推理+NMS+传框 FPS | 包含画框/JSON/存视频 FPS | P95管线延迟ms |
|---|---:|---:|---:|---:|
| COCO预训练 | 5.99 | 81.24 | 43.06 | 13.96 |
| 微调best | 5.97 | 84.77 | 44.15 | 13.52 |

计时不含加载和预热；不包含真实摄像头网络/显示延迟。视频没有逐帧GT，检测框数是累计框数，不能当目标数、正确数或准确率。

人工检查固定帧0/239/479/718，保留失败案例：

1. 第0帧：近处红色小汽车被微调模型误报bus约0.70；广告牌区域被误报truck。
2. 第239帧：左侧广告牌/楼体被误报大truck框，前方行人和部分两轮车漏检。
3. 第479帧：预训练模型将罐车和相邻小巴合并报bus0.52；微调后仍有左侧异常大truck框和近处车辆重复框，没有解决该场景问题。
4. 第718帧：左下方白色小汽车被微调模型高置信度误报bus0.99，右侧骑行者没有可靠检测。

因此保留测试集内适配收益与异域公开视频误检同时存在；不能宣传本权重已经适合直接部署。公开视频只作定性外部场景检查，未用于本轮训练或调参。

## 下一步的优先顺序

1. 首先明确自行车/电动自行车/摩托车与骑行人的框定义，补充实际目标路口的电动自行车标注，不能仅改标签名字。
2. 建立按摄像头或路口隔离的验证/测试集，加入当前视频暴露的广告牌、近景小车、两轮车和小行人难例；完整标注相关类别，避免只标电动车造成其他类别假负例。
3. 在预先固定的新预算内比较全量数据、较长训练和更高输入尺寸；640缩小远处行人是待验证因素，需与NX速度共同测量。单次短预算不能说明YOLO11n总体失效。
4. 准确率和类别定义过关后，在Xavier NX本机导出TensorRT FP16/校准INT8并测端到端≥30FPS；3090Ti速度不能换算成NX保证。

## 本地交付与复核

- `delivery/intersection_before_after.mp4`：训练前后同帧左右对比，H.264，完整719帧。
- `delivery/intersection_finetuned.mp4`：微调后1080p视频。
- `delivery/heldout_gt_before_after_local_only.jpg`：测试列表固定索引0/32/64/96的官方GT、训练前、训练后，未按好坏排序。只展示四个共同类别，不代表数据集全标注。
- `delivery/metrics_and_training.png`：测试AP与原生8类验证曲线。
- `runs/yolo11n_pilot302/weights/best.pt`：验证选择权重；不是Xavier TensorRT engine。
- `evaluation_summary.json`、`dataset_audit_summary.json`、`training_completion.json`、`training_config.json`与各视频summary：可复核摘要。
- 训练检查：完成30行、末epoch30；真实GT128图/5280四类框；跨集重复拒绝与类别映射完美框接口测试已通过，合成检查不是模型精度。

原始图/标签、完整逐帧预测、权重和第三方视频只在本地交付，不随持续GitHub同步公开。源数据CC BY-NC-SA4.0；商业使用需另行核验授权，不能从此训练推出可商用。

## 来源

- [作者仓库与类别说明](https://github.com/AstronZh/Intersection-Flow-5K)
- [Kaggle官方数据v4入口](https://www.kaggle.com/datasets/starsw/intersection-flow-5k)
- [Pexels测试视频](https://www.pexels.com/video/cars-waiting-in-a-traffic-light-5675618/)
- [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/)
- [官方COCO评价器实现](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py)
- [Intel支持的MKL线程层接口](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-1/mkl-set-threading-layer.html)
