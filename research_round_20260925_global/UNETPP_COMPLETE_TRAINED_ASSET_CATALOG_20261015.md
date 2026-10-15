# D:/SAR/UNet++ 全部权重文件与历史训练证据目录（2026-10-15）

上一份8组报告是优先抽查，不能代表全盘。本次扩展确认，项目还保存了此前遗漏的 Mamba-YOLO、MSCK-Net、YOLOv8/v12/26、多任务检测与分类的实际长训资产。用户指出“还有很多训练过及设计的模型”有事实依据；应充分利用这些记录，而不是继续把已有工作当成只有计划。

## 范围和准确计数

- 对 `D:/SAR/UNet++` 进行一次全树文件名/大小/修改时间枚举，排除虚拟环境、site-packages、wheelhouse、缓存和声明的原始数据目录，未遍历整个D/E盘；扫描无错误。完整排除清单在JSON的 `inventory.excluded_directories`。
- D内找到 **280个 `.pt/.pth` 扩展名文件**（242 pth、38 pt，stat大小合计93,296,312,150字节）。这不是280种方法，也不是280次独立训练。
- 其中 **20个 MSCK `eval/*.pth` 是bbox evaluator序列化结果，不能算模型权重**；其余260个为模型候选文件。
- 260个模型候选中，**222个能关联到已有数值训练记录**，36个仅有stat、2个仅有配置/评价关联。后38个的本地训练来源链仍未确认；不能把没有找到证据解释成从未训练。
- 沿D日志明确记载的 `E:/SAR_Benchmark_Data/runs/...` 精确路径，另外定位26个已存在训练/验证目录；其中14个训练目录有连续1—100轮CSV、28个best/last模型文件。未扫描其他E目录。
- 当前目录索引包含1087个结果文本候选、84份已解析train JSON、95份eval/val/metric-row JSON、122份根配置YAML及19份D内CSV/JSONL运行记录。**元数据候选的完整枚举不等于逐份内容全文审阅**；额外读取346个独立文本/限定字节窗口，并复用上一轮118文件的提取结果。
- 新增长训证据中，Mamba四集、MSCK四集与E下14个训练目录合计 **22组**有连续100或160轮数值历史。旧总表有23个DONE声明，额外那组YOLO26n/HRSID本次没有沿精确日志路径定位对应原始训练目录，保留为汇总声明，不补猜路径。

所有checkpoint只做stat，**没有打开、哈希或加载**。未导入Torch、运行模型、训练、测试、安装或修改D项目，未Git。每次文本读取及新确认E路径的stat都先检查原路径和resolve路径不在CPU159活跃绑定集合，快照SHA为 `092d3f60aeb3788ffd7a1e5ae4df485e2b65cbf2a12dc2093f842edc9367cee6`。

完整逐文件映射：`E:/SAR/seal/tmp/UNETPP_COMPLETE_TRAINED_ASSET_CATALOG_20261015_01.json`。`assets[*]`保留280个原路径、大小、mtime、角色、证据、架构候选、任务、数据/预算及指标协议；`external_explicit_paths`另列28个E权重及其100轮CSV。原8组报告未改写。

## 证据等级，不能混同“独立复现”

| 等级 | D内模型候选数 | 含义 |
| --- | ---: | --- |
| L3 | 77 | 数值训练记录，加上明确checkpoint评价引用 |
| L2 | 145 | 数值训练记录关联，缺完整对应评价引用或身份链 |
| L1 | 2 | 只有配置/评价引用，未连接到数值训练历史 |
| L0 | 36 | 仅文件stat，未建立训练链 |

另20个bbox evaluator文件不进入此模型数表。L2/L3仍不证明本次核验了原训练进程退出码、模型字节身份、冻结源码、指标实现或标签转换。注册输出的8个YOLO文件通过旧train JSON明确声明的run_dir与注册保存关系关联CSV，**没有据此声称注册副本与last权重的字节相等**。旧launch JSON中的running只是历史字段，不是当前活跃进程状态。

## 本次新增的主要模型和用途

| 家族/任务 | 直接训练证据 | 数据与预算 | 可借鉴内容与限制 |
| --- | --- | --- | --- |
| Mamba-YOLO-T，水平框检测 | D内5份CSV与args；4份连续1—100轮，另1份SSDD dry仅1轮；同目录best/last共10个文件 | SSDD/HRSID/SADD/SAR-AIRcraft-1.0；full配置batch16、640、SGD、seed0、100轮 | 可借鉴状态空间/局部特征设计及成熟检测基线接口；不是掩膜实例分割。args的pretrained=true只是配置开关，不证明实际加载了哪份预训练权重。 |
| MSCK-Net，水平框检测 | 四份log.txt均覆盖epoch0—159；每集4个模型checkpoint与5个evaluator文件 | SSDD/HRSID/SADD/SAR-AIRcraft-1.0，160轮；源码/配置连接HGNetv2_MSCK、HybridEncoder、DFINETransformer | 已有长训与COCO bbox指标，可研究多尺度/轻量编码；不能把bbox AP嫁接成mask AP。种子与初始化应按原运行配置另核验，不从当前YAML代替。 |
| YOLOv8n / YOLOv12n / YOLO26n，水平框检测 | D训练日志精确指向的E目录，11份CSV均连续1—100轮，22个best/last文件 | v8/v12各四集；26本次定位SSDD/SADD/SAR-AIR三集，100轮 | 是实训工程资产，可复用转换和运行接口；YOLO26/HRSID目前仅旧总表声明，不冒充已定位原件。 |
| YOLO26n-cls，分类 | E下MSTAR/FUSAR两份100轮CSV及4个权重；D日志有分类指标 | MSTAR、FUSAR-Ship，各100轮 | 可借鉴SAR特征/轻量化，不是船舶实例分割。MSTAR top1约95.42%、FUSAR约74.12%为历史记录，不能当mask AP。 |
| YOLO26n-obb，旋转框检测 | E下RSAR 100轮CSV、2个权重及D日志 | RSAR，100轮 | 可借鉴细长目标方向建模，但OBB仍不是像素实例分割。 |
| Mask R-CNN bounded4，实例掩膜诊断 | train JSON记录1轮、4个batch/4图，eval仅4图 | HRSID400 | 确有训练步骤与实例输出；只是极小诊断。该记录的AP50/AP75为0不能证明成熟Mask R-CNN无效，也不构成充分训练强基线。 |
| Coast-gate / metadata-CX / targeted coast / RABF / generic recall / HDSC 与generic geometry | 发现对应checkpoint、train/eval JSON及配置 | HRSID/SSDD，主要smoke或bounded记录，逐项预算在JSON | 这些是上轮8组外应保留的设计/短训资产；不把smoke通过或文件存在提升为完整效用、机制或创新证明。 |
| KD教师/学生、轻量UNet旧设计 | KD目录9个pth、对应设计源码由架构审阅独立列明 | 训练计数/数据/种子和逐权重源码身份尚未连接 | 值得研究蒸馏与计算成本设计；不能仅凭KD文件名认定知识蒸馏收益已获验证。 |

第三方Mamba库内另有终端为Segment的设计，但本次实训目录是Detect任务，不能由检测训练记录推断该分割设计已训练。HGTFN在旧汇总中标记 `EXCLUDED_HYPERSPECTRAL_TASK_MISMATCH`，不能把源码存在当SAR实例训练成果。DiffSARShipInst/Detectron2附带的PointRend、PointSup等也要区分库设计、权重来源与本地训练记录。

## 新增两组可核对的检测结果：保留任务和端点

Mamba以下是**CSV第100轮**数值（不是本次加载best权重后的重测），四份 `results.csv` 的第101行：

| 数据 | bbox mAP50 | bbox mAP50-95 |
| --- | ---: | ---: |
| SSDD | 97.695 | 68.646 |
| HRSID | 87.126 | 62.269 |
| SADD | 98.207 | 62.972 |
| SAR-AIRcraft-1.0 | 86.230 | 60.475 |

上述值直接来自CSV末行，完整数值保留在JSON `numeric_run_records[*].last_row`，不是本次重新测量。四集结果只能在各自数据及该检测协议内使用。

MSCK的原始JSONL可把**最后epoch159**和**历次记录最大bbox AP**分开，不继承旧总表混合final AP50/best AP50-95的写法：

| 数据 | 最后bbox AP | 最后bbox AP50 | 历次最大bbox AP | 最大值epoch（0起） |
| --- | ---: | ---: | ---: | ---: |
| SSDD | 71.238 | 93.602 | 71.459 | 157 |
| HRSID | 63.492 | 81.177 | 63.492 | 159 |
| SADD | 68.267 | 97.115 | 68.267 | 159 |
| SAR-AIRcraft-1.0 | 61.019 | 82.573 | 61.045 | 148 |

这不是跨方法总排行，也没有建立可比初始化、训练成本或数据转换，不能据此认定MSCK/Mamba或当前N20哪一个更好。MSCK原日志有测试/验证中逐轮指标，正式外测是否曾被用于选型须保留历史暴露审计。

## 需要纠正的历史表述

1. `results_comparison.csv` 的多行 `false_alarm_rate` 实际恰为 `1−precision`，即FP/(TP+FP)形式的FDR；它不是按背景或每图归一化的虚警率。原字段和值保留，今后另列正确名称及分母，不能把FDR写成已验证SAR虚警控制。
2. MSCK旧总表某些行合并final AP50和best AP50-95；应像上表分列端点。不能把这些不同权重/epoch的数值拼成一套最终结果。
3. 100/160轮检测资产补全了历史工程积累，但不能替代本项目COCO mask AP、重复种子和跨集实例证据。
4. 260个模型候选中38项仍没有数值训练链：36个L0包含KD9个、根YOLO预训练候选4个（含Mamba库内1个）、debug副本2个以及111和部分legacy；L1为 `weights/current/222.pth`、`weights/legacy/ALL_NEW_0.01_100_SSDD.pth`。有旧评价关联不等于已有公平的训练比较。

## 原文件与行号入口

- 总表：`D:/SAR/UNet++/research_ops_runs/sar_full_repro_benchmark_20260610/results_comparison.csv:1`；其DONE、TASK_INCOMPATIBLE与HGTFN排除状态均保留在JSON `benchmark_aggregate.rows`。
- Mamba：`D:/SAR/UNet++/research_ops_runs/sar_model_benchmark_comparison_20260608_1312/repos/Mamba-YOLO/output_dir/sar/mby_hrsid_full/results.csv:101`；其他三集对应 `mby_ssdd_full/mby_sadd_full/mby_sarair_full/results.csv:101`。同run `args.yaml:1`起为任务、模型、数据和预算；seed/optimizer等字段在JSON原样保留。
- MSCK：同一repos根下 `MSCK-Net/msck/msck-n-{hrsid,sadd,sarair}/log.txt:160`、`msck-n-ssdd/log.txt:163` 为最后epoch159。JSONL字段 `epoch/test_coco_eval_bbox`，数组索引0/1/2分别是该COCO bbox输出的AP/AP50/AP75；最佳值由全部记录索引0比较得到，未读取任何pth。
- evaluator角色：`MSCK-Net/engine/solver/det_solver.py:175`—184 保存 `coco_evaluator.coco_eval['bbox'].eval`，而106—112是模型state_dict保存；这是架构代理独立静态审阅结论，不是本次加载序列化文件所得。
- 模型角色：`MSCK-Net/configs/IS/msck-n-hrsid.yml:14`起的backbone/encoder/decoder；Mamba `ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T.yaml:38`终端Detect，另分割配置 `yolo-mamba-seg.yaml:47`终端Segment。引用架构代理静态审阅，不自动宣称已训练分割版本。
- E下已定位训练目录及CSV：JSON `external_explicit_paths`逐项保存D原日志声明、精确路径、args、1—100 epoch与权重stat，避免从目录名猜路径；权重均未打开。
- Mask R-CNN：`D:/SAR/UNet++/runs/logs/hrsid_maskrcnn400_bounded4_train.json:1` 的 `processed_batches/checkpoint_written`；对应metrics文件的 `num_images/publishable_scope`；不能省略bounded边界。

下一步应优先借鉴实训家族中的结构选择、强控制与可复用接口，保持当前N20固定评价不变。把检测成功直接改写成分割创新，或把旧权重统一当成熟模型，都不会增加论文可信度；后续候选必须按真实实例误差与公平协议单独验证。


发布说明：本文件是自写审阅与研究记录。逐文件私有JSON索引留存本地；未发布历史权重、原始数据、全量标注、运维配置或第三方论文全文。相关记录见[扩展总评](UNETPP_EXPANDED_MODEL_ASSET_REVIEW_AND_NEXT_DESIGN_20261015.md)。
