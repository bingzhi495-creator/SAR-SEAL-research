# D:/SAR/UNet++ 历史训练结果的只读证据审阅（2026-10-15）

用户指出这里有此前训练好的模型和方法。这个判断有实际文件支持：`runs/checkpoints` 顶层有 78 项，`runs/metrics` 顶层有 214 项；本次有针对性地检查了 118 个文本文件，找到 50 轮训练历史、三种子配对结果、方法对照及跨数据集诊断，不能把它们统称为“只有计划”。但不同任务和评价协议必须分开，权重文件存在本身不证明指标正确或方法创新。

本次没有运行 D 项目代码、加载模型、导入 Torch、重训或重新评估，也没有打开或哈希任何权重。D 目录未修改。本次仅读取不受当前运行绑定影响的历史文本。下列数字是历史 JSON/CSV 的记录，本次仅复算已有数值的均值、样本标准差与配对差值，不是独立复现结果。

机器可读证据索引留存在本地，含每组原文件、JSON 字段、指标行号、当前配置行号、训练历史长度与 batch 计数。以下所有 F1/IoU 均以百分数表示，差值为百分点。

## 证据与协议总边界

1. 多数 UNet++ 记录是二值**语义分割**，JSON 直接写有 `publishable_scope: semantic segmentation only`、`instance_ap_reported: false`。例如 `D:/SAR/UNet++/runs/metrics/hrsid_clean_unetpp400_full_eval.json:10` 的 `miou` 和第 14—15 行的任务边界。历史连通域 `AP50/AP75` 是代理实例评价，`CLAIMS_FROM_RESULTS.md:52` 明确说明不是 COCO API AP；不能移到当前 COCO mask AP 主表。
2. 指标名 `miou` 需要重新解释：本次36份评价记录的数字均与 `TP/(TP+FP+FN)` 一致（误差小于1e-12），直接证明的是**前景 IoU**。例如 clean seed42 的 `1200564/(1200564+532549+206759)=0.6188882565`。未核验评价实现前，不应称为两类平均 mIoU。
3. D 项目 HRSID 使用 `HRSID_JPG/HRSID_JPG/JPEGImages1` 与 `train2017/test2017`。记录的训练/测试规模为 3642/1962 图、11047/5922 标注；SSDD 使用 `cocossdd` 的 928/232 图划分。本次未读取或哈希原始数据，不能把这些旧记录等同于当前官方 PNG 包的身份核验。
4. JSON 内保存的旧权重/config SHA 是**历史记录字符串**，本次没有重新核验权重。当前若干 YAML 文件的大小与旧 eval 保存的 config 大小不同，说明配置有历史演变，不能用当前 YAML 自动替代当时源码/配置快照。即使大小相同，也不等于内容相同。
5. `history` 中连续 50 个 epoch、每轮 batch 数与 `resume_from_checkpoint=false` 支持“日志记录完成了指定计数”，不替代原进程退出码、未改写源码及训练权重来源的独立证明。初始化实现仍需与当时版本对应，不能从文件名猜测预训练来源。
6. 旧 `test2017/val2017` 被反复用于阈值扫选和方法分析，已不是未接触的最终留出集。今后正式评价必须保留这一使用历史。

## 1. 最有价值的基线资产：clean 与 boundary 的完整三种子

记录的共同配方：base32、GroupNorm UNet++，batch2、50 轮，AdamW 3e-4/weight_decay1e-4，seed42/43/44，阈值0.5；boundary 使用辅助边界监督，当前配置权重0.2。HRSID 每轮1821 batch、总91050；SSDD 每轮464、总23200。两臂均有对应训练 JSON、评价 JSON 和 checkpoint stat。

| 数据 | 方法 | F1：seed42 / 43 / 44 | 均值 ± 样本标准差 | 前景 IoU均值 |
| --- | --- | --- | --- | --- |
| HRSID | clean | 76.458 / 77.176 / 77.949 | 77.195 ± 0.746 | 62.863 |
| HRSID | boundary | 78.028 / 77.415 / 75.242 | 76.895 ± 1.464 | 62.478 |
| SSDD | clean | 84.114 / 86.461 / 85.027 | 85.201 ± 1.183 | 74.229 |
| SSDD | boundary | 85.572 / 86.292 / 86.813 | 86.226 ± 0.623 | 75.790 |

HRSID 配对 F1 差为 +1.570/+0.239/−2.707，均值 −0.299；SSDD 为 +1.458/−0.169/+1.787，均值 +1.025。边界监督在 SSDD 的平均信号值得肯定，但不能声称每种子或跨数据集稳定获益，更不能把 seed42 的 HRSID 正收益代表三种子结果。HRSID 平均召回还从81.793降至80.556。

可复用内容：成对种子、固定阈值、边界/FP/FN 诊断，以及“precision 提高但 recall 下降”的真实权衡。对当前实例研究的用途是设计**条件化、保召回的边界修复假设和对照**，不应直接把语义网络换名作为当前创新。

证据：`runs/analysis/hrsid_boundary_vs_clean_multiseed_summary.json` 与 `ssdd_boundary_vs_clean_multiseed_summary.json`，字段 `arms.*.per_seed`、`arms.*.summary`、`comparison.paired_delta_by_seed`。本次从各独立 eval JSON 重算均值/样本标准差，与原聚合一致。例配置 `configs/baseline_hrsid_clean_unetpp400.yaml:38`、boundary 配置`:39`；对应 `runs/logs/*_train.json` 的 `seed/epochs/history[*].num_batches`。

## 2. 频域残差与边界联合：已有正信号，也有明显数据集反转

全部 seed42、50轮，记录预算与上面的同数据集 clean/boundary 一致，HRSID 总91050 batch，SSDD 总23200。

| 数据 | clean F1 | boundary F1 | frequency F1 | frequency+boundary F1 |
| --- | ---: | ---: | ---: | ---: |
| HRSID | 76.458 | 78.028 | 76.282 | 78.859 |
| SSDD | 84.114 | 85.572 | 86.317 | 82.477 |

HRSID 联合版本比 clean +2.401、比 boundary +0.831；但 SSDD 联合版本比 clean −1.637、比 frequency −3.840。该组非常适合借鉴“互补特征并不一定可直接相加”的经验，不能根据 HRSID 一行宣布融合有效，也不能根据 SSDD 一行宣布频域信息无效。缺重复种子、机制/容量对照和当前实例 AP。

可复用方案：先问频域残差在何种图像/实例误差上有用，再设计由实例条件或观测可靠性控制的残差；以常规局部卷积/高通以及容量对照排除普通特征增强解释。只是候选方向，本次没有启动或认定新方法。

证据：`runs/metrics/{hrsid,ssdd}_frequency{,_boundary}_unetpp{400,256}_full_eval.json` 的 `f1/miou/recall`（通常第8—15行）；`runs/logs/` 同前缀 `_train.json`。模型类分别为 `ScatteringResidualUNetPlusPlus`、`ScatteringBoundaryUNetPlusPlus`，是历史配置记录，结构细节另行静态审阅。

## 3. SBRF 与 descriptor-free learned suppression：强对照阻止了过度解释

seed42、50轮，HRSID/SSDD 分别91050/23200 batch；同数据集候选和 generic 对照训练记录预算一致。

| 数据 | SBRF F1 | generic对照F1 | 差值 | SBRF / generic 每图FP像素 |
| --- | ---: | ---: | ---: | ---: |
| HRSID | 77.464 | 77.530 | −0.066 | 168.12 / 157.24 |
| SSDD | 81.107 | 86.317 | −5.210 | 216.73 / 105.67 |

这组没有支持“专用描述符优于普通学习抑制”。HRSID 的边界F1略高不足以抵消主要效用和FP控制；SSDD 的退化更明确。仍限于这套50轮配方、seed42及语义任务，不能声称所有 SBRF 设计总体失效。

可复用内容：保留 generic learned suppression 为强替代解释，而不是只比较 clean；这比再添加模块更有助于辨别真正技术贡献。新方法若依赖 SAR 专用特征，必须超过合理常规特征控制。

证据：`runs/analysis/sbrf_r123_hrsid_full_screen_decision.json` 与 `sbrf_r126_ssdd_full_screen_decision.json` 的 `sbrf_candidate_full_screen.metrics`、`generic_learned_suppression_full_screen.metrics`；直接 eval 与训练 JSON 均存在。报告中的 `ap50/ap75` 为历史连通域协议，不能当当前 COCO mask AP。

## 4. sampling-boundary：旧否决有不公平比较，不能继承为总体失败结论

候选只记录5轮×64 batch=320次，HRSID batch1、SSDD batch2；评价分别32图/64图。固定0.5阈值F1为43.067/63.811，HRSID扫到0.9可得80.992，显示严重工作点敏感，不能将扫选最优值替代固定结果。

旧 R143/R144 直接拿这些结果与50轮 full参考（HRSID1962图、SSDD232图）作门控。预算、batch和评价集合均不同，因此足以记录“旧流程选择停止扩展”，**不足以证明该采样机制劣于参考或总体失效**。反过来，32图上扫选的80.992也不能证明比1962图的 full参考更好。

可复用内容：采样实现与校准风险分析可能有价值，但仅在当前 N20 固定端点结束后，且新误差证据确实需要它时，才考虑最小同预算、同图集、同初始化的配对检验。不要因为这次审阅立即复活整个旧分支。

证据：`configs/baseline_hrsid_sampling_boundary_unetpp400_r142_bounded64.yaml:40`（batch、epoch44、max_batches45）；SSDD 对应配置`:46`。`runs/metrics/hrsid_sampling_boundary_r142_unetpp400_bounded64_eval.json` 的 `num_images/f1`；`runs/analysis/sampling_boundary_r143_bounded_evidence.json` 的 `datasets.*.candidate.metrics` 与 `controls.*.metrics.num_images`；R144 的 `datasets.*.candidate_metrics`。

## 5. 区域 Gamma 似然：损失层面的候选资产，特定物理解释仍未成立

clean 架构未改，Gamma/Gaussian 控制都记录50轮、batch2、HRSID91050次、region_weight0.1、seed42；另有 Gamma seed1，同预算。

| 组别 | F1 | 前景IoU | recall |
| --- | ---: | ---: | ---: |
| clean seed42 | 76.458 | 61.889 | 85.308 |
| Gamma seed42 | 77.910 | 63.814 | 79.765 |
| Gaussian seed42 | 77.528 | 63.303 | 83.183 |
| Gamma seed1 | 77.469 | 63.225 | 77.733 |

Gamma seed42 比 clean +1.452，比 Gaussian仅+0.382。每图FP像素从clean271.43降至179.29，但召回明显降低；seed1有正于seed42clean的记录，却不能替代**seed1同预算clean/Gauss**，不能叫配对稳定复验。seed1 的FP像素164.59、图像级任意FP比例97.30%；纯背景只有1图，命中FP率100%不能推断广泛纯背景风险的总体概率。Gamma seed42相同纯背景指标也只有1图。

可复用内容：区别“抑制背景带来的precision上升”与“修复轮廓”；Gamma 相对合理 Gaussian 的较小差距，要求物理分布假设有独立证据。JPEG/归一化强度也不能自动当标定 SAR 幅度或强度。先进行数据/幅度含义及失配机制检查，再考虑实例条件区域统计；本次没有给物理解释背书。

证据：`runs/metrics/hrsid_clean_unetpp400_full_region{,_gauss,_seed1_r382}_eval.json:8`；配置 `baseline_hrsid_clean_unetpp400_region.yaml:38`、损失第52—58行；`runs/analysis/region_likelihood_multi_seed_seed1_gamma_evaluation_review_r511.json` 的 `source_metric_summary`、`comparisons`、`gate`；`region_likelihood_false_alarm_triage_r377.json` 的 `gate`。

## 6. 已有跨集诊断：可直接利用来定位迁移问题，不能冒充当前正式跨集证明

| 记录方向 | clean F1 | boundary F1 | Gamma F1 |
| --- | ---: | ---: | ---: |
| HRSID权重 → SSDD256 | 65.256 | 65.665 | 67.848 |
| SSDD权重 → HRSID400 | 57.445 | 59.736 | 未见对应记录 |

这些目标分辨率诊断比同目标训练集重新训练的语义模型低很多，真实提示了可迁移性困难。Gamma在已记录 HRSID→SSDD 相对clean+2.592，但只是历史语义F1/一个方向；没有实例AP、成对重复、完整冻结与历史测试暴露审计。未核验历史是否曾用目标结果调参，不能直接称作严格零调参泛化。

可复用内容：跨数据集**失败条件与阈值/召回变化**，优先解释分辨率、成像/编码和监督口径差异，再建立固定跨集协议。HRSID JPG 与官方 PNG 的版本关系必须先核验。

证据：`runs/metrics/cross_hrsid_{clean,boundary}_to_ssdd256_eval.json`、`cross_ssdd_{clean,boundary}_to_hrsid400_eval.json`、`cross_hrsid_region_gamma_to_ssdd256_r378_eval.json` 的 `run_metadata.model.checkpoint`、`run_metadata.dataset`、`f1`；`runs/analysis/cross_dataset_transfer_summary.json` 的 `scope/gate` 自身也未允许最终泛化主张。

## 7. DeepLabV3+/SegFormer/“Mask2FormerBinary”：有运行记录，不能当成熟实例SOTA复现

三者均为10轮、seed42、batch2，HRSID每模型18210次、SSDD4640次。HRSID配置320分辨率，而旧 UNet++ 为400；SSDD为256。日志标为不恢复、loaded_checkpoint为null；是否另有模块内初始化仍需核查。

| 数据 | DeepLabV3+ F1 | SegFormer F1 | Mask2FormerBinary F1 |
| --- | ---: | ---: | ---: |
| HRSID | 66.632 | 63.198 | 63.585 |
| SSDD | 80.663 | 77.730 | 79.543 |

这些是语义分割输出；“Mask2FormerBinary”记录是 tiny、16 queries、hidden_dim32，不能仅凭名字等同原作者成熟 Mask2Former实例复现，更不能证明其弱于50轮的UNet++或当前N20。10轮/50轮、HRSID320/400及初始化配方不一致。

可复用内容：数据转换、工程接口和从零短预算行为。若将来需要强比较，应该使用正确任务、合理预训练与可比数据/预算，明确“结构适配”或“同协议实现”的身份。

证据：`runs/metrics/{hrsid,ssdd}_{deeplabv3plus,segformer,mask2former}_full_protocol_budget10_eval.json` 的 `publishable_scope/instance_ap_reported`、`run_metadata.model.kwargs`；同前缀训练 JSON；相应 configs 的 `training.epochs`、`image_size`、`model.kwargs`。

## 8. YOLO-seg：确有实例掩膜指标，可作现成工程基线入口

`yolo11n-seg.yaml`，pretrained=false、seed42、10轮、batch8、AMP；HRSID320、SSDD256。两份 CSV 都记录1—10轮及下降的损失，训练时 `val=false`，前9轮评价列为0，因此**不能把前9轮零指标解释为真实AP为0**。末轮内置评价与独立 val JSON 不完全相同，须选择预先声明的端点。

| 数据 | 独立val JSON mask mAP50 | 独立val mask mAP50-95 | CSV末轮mask mAP50-95 |
| --- | ---: | ---: | ---: |
| HRSID | 51.488 | 19.441 | 20.227 |
| SSDD | 61.033 | 25.040 | 25.687 |

这是历史 Ultralytics mask mAP，仍未核验与当前COCO API评价的完全等价性，不与语义Dice或连通域代理AP混排。单种子、10轮、随机初始化及标签转换细节限制其作为充分训练强参考。CSV累计时间508.322/99.0208秒只是该次日志值，没有统一硬件/预处理/同步测量，不能据此宣称当前模型速度优势。

可复用内容：已有实例标签导出及训练入口、现成权重和独立验证记录，有条件用于后续公平同协议基线，避免从零重写转换器。旧YOLO-OBB与Deformable DETR是旋转框/水平框检测，不是mask AP；例如后者R241整理记录为bbox AP0也不能当实例分割对比。

证据：`runs/ultralytics/{hrsid,ssdd}_yolo_seg_full_protocol_budget10/results.csv:11`；`args.yaml:1`（任务/模型/预算/分辨率、pretrained第18、seed21、val36）；`runs/metrics/{hrsid,ssdd}_yolo_seg_full_protocol_budget10_val.json:12`（mask指标第12—15行）。

## 旧大权重池与下一步取舍

`weights/current` 有 `111.pth` 2,670,229,875字节、`222.pth` 761,363,299字节；`weights/legacy` 有21个文件，包括loss对比、CB/CC/SCONV、ALL_NEW、UNet++和Mask R-CNN。这里只做了文件名/大小stat。`weights/WEIGHT_MANIFEST.md:9` 已明确111训练来源待核验，第16行指出预期SSDD权重缺失，第31—37行要求先补训练元数据才能用于论文表。不能从这些文件名反推已验证结构收益或完整预算。

优先借鉴：①完整三种子反例与强generic控制；②频域+边界的跨集反转；③Gamma/Gaussian的损失级对照与precision/recall权衡；④可复用的YOLO实例标签/基线接口。优先纠正：短预算对full参考、代理AP当COCO AP、只报最好种子/阈值，以及把当前配置当历史冻结版本。

本次审阅支持缩小后续假设空间，没有产生新训练或新的科研成功。当前 N20 仍按既定固定端点评价和停止规则执行；历史资产只有在该结果及真实误差证据需要时才进入下一条单一假设。不能据这些旧语义收益认定当前实例创新成立或已具备TGRS投稿依据。


[总评与有条件复用方案](UNETPP_HISTORICAL_ASSETS_SYNTHESIS_AND_REUSE_PLAN_20261015.md)。源文件位置均相对于用户指定的历史工作区；本次未复制原项目代码、权重或数据。


## 后续扩展与自主结构解释（2026-10-15）

本文件保留首轮八组优先审阅的历史快照，不能代表全目录。用户明确说明UNet++结构为团队自主设计，整体结构与基础算子来源分开；新增完整目录、自主前向还原、权重/长训证据及直接先例见[后续总评](UNETPP_EXPANDED_MODEL_ASSET_REVIEW_AND_NEXT_DESIGN_20261015.md)。旧full与clean并非同一结构或公平配方；未定位旧权重完整来源不等于未训练。BatchFormer可能有意跨样本计算，轴风险不能直接判成编码错误。

候选“邻船作为负类”也不是新的监督定义：原生逐实例mask损失已包含本实例外的负类目标；不确定性点采样支持由配置决定的随机补点。真正的技术差异需落在信息、路由、采样或目标的具体改变，并与强常规替代比较。新候选仍未验证，不以本目录审阅宣称创新成立。
