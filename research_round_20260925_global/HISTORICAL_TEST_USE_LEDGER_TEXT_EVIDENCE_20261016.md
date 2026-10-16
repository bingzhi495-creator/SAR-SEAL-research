# 历史 TEST 用途台账：已发布文稿中的明确披露（2026-10-16）

本次补齐一项实际证据缺口：PSeg-SSDD 不仅有“结果曾被查看”的记录，历史修订实验稿还明确披露了具体模型的测试集选型。此结论来自已发布普通文稿；没有重新核验原日志、权重、数据或运行时间链，不构成完整历史用途台账。当前 N20 的科学载荷和资格链未被读取或改变。

## 来源与证据等级

根执行者与独立审阅者分别读取发布副本中的 `paper_tgrs/revision_tgrs_20260905/sec_exp.tex`；根执行者还核对了 GitHub 提交 `d2b38d430302153c55a2e5330cea296a9324dcdb` 中该文件的 Git blob，为 `dfcc50808134ebed5cd49e2fa08de5af40f894a7`，与本地普通文件一致。以下链接固定到该历史提交。

证据等级为**历史文稿直接披露，原始执行链尚未独立复核**。文稿的日期、行号与提交位置能够定位披露内容，不能代替实际运行发生时间、权重身份或事件先后关系。

| 数据 / 项目 | 文稿明确披露的用途 | 定位与适用范围 |
|---|---|---|
| HRSID 融合设置 | `τ=0.5`、`Smax=6`、`γ=0`、视图尺度和融合算子由 TEST sweep 选择 | [第21行](https://github.com/bingzhi495-creator/SAR-SEAL-research/blob/d2b38d430302153c55a2e5330cea296a9324dcdb/paper_tgrs/revision_tgrs_20260905/sec_exp.tex#L21)；主载体采用最终迭代，不能把该披露扩大为所有载体 checkpoint 均按 TEST 选择 |
| HRSID BPR 种子 | 报告的 s0 为检查三次运行后的最佳者，后续重排继承此选择条件 | [第21行](https://github.com/bingzhi495-creator/SAR-SEAL-research/blob/d2b38d430302153c55a2e5330cea296a9324dcdb/paper_tgrs/revision_tgrs_20260905/sec_exp.tex#L21)、[第99行](https://github.com/bingzhi495-creator/SAR-SEAL-research/blob/d2b38d430302153c55a2e5330cea296a9324dcdb/paper_tgrs/revision_tgrs_20260905/sec_exp.tex#L99)；与单次训练内部 checkpoint 的选择规则分别登记 |
| HRSID 固定候选诊断 | 1962 图历史前向重放、top-1 与门控比较，以及复用历史选定 BPR 的比较 | [第62行](https://github.com/bingzhi495-creator/SAR-SEAL-research/blob/d2b38d430302153c55a2e5330cea296a9324dcdb/paper_tgrs/revision_tgrs_20260905/sec_exp.tex#L62)、[第64行](https://github.com/bingzhi495-creator/SAR-SEAL-research/blob/d2b38d430302153c55a2e5330cea296a9324dcdb/paper_tgrs/revision_tgrs_20260905/sec_exp.tex#L64)；历史诊断不能写成未接触测试集上的独立确认 |
| HRSID 200 轮控制 | 采用 best-on-official-test 的 checkpoint | [第106行](https://github.com/bingzhi495-creator/SAR-SEAL-research/blob/d2b38d430302153c55a2e5330cea296a9324dcdb/paper_tgrs/revision_tgrs_20260905/sec_exp.tex#L106)；调度、选择与模型平均预算同时不同，不能据此仅归因于训练时长 |
| PSeg-SSDD 对比模型 | 该旧稿对比表的 competitor 行使用 best-on-test checkpoint，两项 SAR 模块本地复现按近岸子集选择 | [第203行](https://github.com/bingzhi495-creator/SAR-SEAL-research/blob/d2b38d430302153c55a2e5330cea296a9324dcdb/paper_tgrs/revision_tgrs_20260905/sec_exp.tex#L203)、[第209行](https://github.com/bingzhi495-creator/SAR-SEAL-research/blob/d2b38d430302153c55a2e5330cea296a9324dcdb/paper_tgrs/revision_tgrs_20260905/sec_exp.tex#L209)；仅指对应历史表格，不能推及全部 SSDD 实验或当前 N20 |
| 训练内选择规则 | BPR 每次训练以随机 patch holdout 的损失选 checkpoint；重排 α 由训练图像 ID 的 4:1 划分选择 | [第14行](https://github.com/bingzhi495-creator/SAR-SEAL-research/blob/d2b38d430302153c55a2e5330cea296a9324dcdb/paper_tgrs/revision_tgrs_20260905/sec_exp.tex#L14)；patch holdout 不是图像独立验证，载体见过训练图像也限制残差泛化解释，但不能把这两项规则直接记成 TEST 选型 |

## 对后续实验的实际影响

1. 更新 [N16 正式 TEST 与跨集准备页](N16_FORMAL_TEST_AND_CROSS_DATASET_EVIDENCE_READINESS_20261014.md)：SSDD 的具体模型选型已有可定位的历史文本，剩余缺口是完整接触清单、发生时间、实际参数/权重身份与原始产物绑定，不能继续把“只知道看过结果”作为全部已知事实。
2. HRSID 与 PSeg-SSDD 的这些历史结果均须保留选择条件。后续冻结方法后，在对应官方协议上进行的固定配方外部评价仍有报告价值；应披露历史接触，不能通过改名、重跑或重新划分就恢复其未接触测试身份。零调参直接迁移与目标训练集重新训练必须分别报告。
3. 当前 N20 的 FIT/DEV、重复种子、预算与停止规则保持原样。本次没有重新判定其 TEST 使用情况，也没有新增正式或跨集 AP。原始执行资格仍按原流程核验。
4. HRSID 的下一项数据入场工作仍是实际像素、真实实例掩膜与空间对齐核验；已有本地文件及元数据准备见 [JPG 核查记录](HRSID_LOCAL_JPG_METADATA_AND_VERSION_BOUNDARY_20261015.md)。这项工作由科学 owner 在执行条件允许后单独安排，本次未执行；无需为补这份文本台账重复下载 PNG。

本次取得的是更准确的历史用途证据，并未完成数据入场、独立确认性评价或论文效用证明。
