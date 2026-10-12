# SEAL：公共 SAR 船舶实例分割研究

论文、方法源码、协议和研究记录的工作快照，更新至 **2026-10-12**。研究仍在进行；当前没有可直接投稿的最终论文，也没有经过正式测试和跨数据集验证的 SOTA 结论。

## 从这里开始

- [当前研究目标与执行要求](RESEARCH_GOAL.md)：问题、公平比较、重复实验、机制验证和停止规则。
- [论文版本索引](docs/PAPER_INDEX.md)：各代正文、补充材料与历史稿的关系。
- 可阅读历史修订稿：[正文 PDF](paper_tgrs/revision_tgrs_20260905/main.pdf)、[补充 PDF](paper_tgrs/revision_tgrs_20260905/supplement.pdf)、[说明](paper_tgrs/revision_tgrs_20260905/README_CN.md)。它们属于旧融合主线，未重写成 10 月当前研究；PDF 与当前源码的对应关系未重新编译核验。
- [数据获取与版本说明](docs/DATA_ACCESS.md)、[上游数据索引](docs/upstream_data_index.json)：PSeg-SSDD 与 HRSID 的官方入口。
- [完整实验计划](refine-logs/EXPERIMENT_PLAN.md)、[实验跟踪](refine-logs/EXPERIMENT_TRACKER.md)、[下一研究批次](refine-logs/NEXT_RESEARCH_BATCH.md)。这些记录保留历史合同；有日期的最新状态优先。

## 当前证据和决定

所有下表结果均来自历史内部 **DEV187 / 380 GT**，不是官方 TEST。AP 使用百分制；“+1 AP 点”表示 67 到 68。没有把本项目的内部指标与外部论文数值直接排名。

| 固定比较 | mask AP | 能支持的结论 |
|---|---:|---|
| 预选强 E24 原生参考 | 67.071057 | 同协议固定参考 |
| A：原生模型 + 同样点监督，24 轮再追加 48 轮 | 64.686168 | 当前续训配方的端点 |
| B：联合残差读出，同样预算 | 66.797068 | B 高于 A，但低于强 E24；结束该配方扩展 |
| C：冻结 E24，粗 F/e/L 点头，两个下游种子均值 | 68.166136 | 相对 E24 +1.095079 AP 点的开发集效用 |
| R：相同点头加真实 JPEG 7×7 patch，两个下游种子均值 | 68.153211 | 与 C 几乎持平；未支持新增 patch 的作用 |

详见 [固定长预算 A/B 结果](docs/fixed_endpoint_joint_native_E24warm48_result_20261012.md)与 [N07 全部种子及固定端点评价](docs/N07_fixed8_fullDEV_point_refinement_result_20261012.md)。N07 每头固定 8 轮、13,288 更新；两个下游种子共享同一个教师初始化，不能替代多教师重复或充分收敛证据。R−C 配对均值为 −0.012924 AP 点，未满足结果前锁定的跟进规则；保留效用，停止当前 patch 配方扩展。

当前 N09 检验简单全局 logit 偏移能解释多少 C 收益。只在 FIT 缓存点上拟合、只在原生 |L|≤2 内应用、保持 query 和原生分数。FIT CPU 已正常结束，δ = −0.169106405372359；首个 DEV 评价在原生精确一致性检查处退出 1，没有产生 AP。失败及待查原因见 [N09 实际执行状态](docs/N09_global_offset_runtime_state_20261012.md)。它是已知替代控制，不是创新点，也不能由 FIT BCE 下降推断 AP 提高。

当前核心创新、正式 TEST、跨集收益和 TGRS 投稿依据仍未成立。失败结果和停止原因保留，不将局部提升包装为最终成功。

## 目录导航

| 目录 | 内容 |
|---|---|
| `paper/`、`paper_tgrs/` | 多代论文 LaTeX / BibTeX、作者说明、历史正文和补充 PDF |
| `carrier/`、`mne/`、`seeding/`、`losses/` | 历史模型扩展、配置与候选组件 |
| `protocol/`、`benchmark/`、`diagnostics/` | 数据转换、评价和误差诊断源码 |
| `research_round_20260925_global/` | 当前及历史训练、固定评价和独立核验 SOURCE |
| `docs/`、`refine-logs/` | 审阅、先例分析、实验计划和研究范围 |
| `autodl/`、`tools/` | 历史环境准备与辅助工具 |

最新 SOURCE 位于 `research_round_20260925_global/multimodal_data_pilot_20260928/native_geometry_paired_mask_readout_preparation_20261009/` 下：

- `N07_frozen_E24_original_detail_point_refinement_source_20261012_01/`：FIT 缓存、点头训练、GPU 评价、独立 CPU 核验。
- `N09_FIT_only_global_logit_offset_control_source_20261012_01/`：全局偏移数学、FIT 求解及待完成评价。
- `N06_fixed_E24_one_to_one_error_audit_source_20261012_01/`、`N01_uniform_point_supervision_gradient_probe_source_20261012_01/`：固定误差和监督诊断。

## 数据与复现范围

本仓库包含数据获取说明和转换/审计源码，**不包含原始数据、全量标注、权重及运行缓存**，不能称为完整数据镜像。官方 SSDD 的 PSeg 多边形与 BBox/RBox 检测框版本不同；内部 FIT741/DEV187 不等于官方 TEST232。HRSID 的 JPG/PNG、场景协议和纯背景补充版本也必须分别记录。数据及其许可入口见 [DATA_ACCESS.md](docs/DATA_ACCESS.md)。

历史环境记录为 PyTorch 2.0.1+cu118、torchvision 0.15.2、MMEngine 0.10.4、MMCV 2.1.0、MMDetection 3.3.0，并依赖 NumPy、SciPy、OpenCV、Pillow、pycocotools。最新本机资格环境与历史环境不可直接混用；没有声称全仓库在新机器完成重跑。

历史入口示例：`python protocol/pseg_to_coco.py`（需配置数据路径）及 `python protocol/seal_eval.py --preds ... --gt ...`。当前科学入口采用 `--spec ABS --owner-execute`，依赖本地冻结数据、初始化和来源资格链。本次保存源码，不包含全部绑定输入，**尚不能下载后一键复现最新实验**；不得删除检查以冒充已验证的可移植版本。

## 持续同步

本仓库是后续研究的持续记录入口。总结、综述、方法设计、研发源码和论文更新形成后，在当前工作任务内及时提交；实验成功、失败和未完成的验证均按实际状态保留。具体范围与流程见 [持续同步规则](docs/CONTINUOUS_SYNC.md)。

## 发布范围与许可

首次发布快照及完整性验证见 [publication_manifest_20261012.json](docs/publication_manifest_20261012.json)（限定于其中记录的提交；后续版本以 Git 提交历史为准），历史工作区盘点见 `docs/workspace_inventory_20261011.csv.gz`（元数据，非原始数据）。[发布范围](PUBLICATION_SCOPE.md)明确哪些内容未随仓库提供。

作者、单位、日期和其他提交元数据保留占位符。自有项目尚未指定统一开源许可；第三方实现使用原始来源链接与各自许可证，不把第三方代码改授为本项目许可。外部下载的文献全文、第三方参考仓库、凭据、运维配置、无关工程以及整个工作区 Git 历史不随本快照发布。
