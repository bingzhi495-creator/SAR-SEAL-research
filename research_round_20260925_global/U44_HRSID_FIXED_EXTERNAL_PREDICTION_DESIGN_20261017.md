# U44：HRSID 固定外部自动实例推理的最小通路

2026-10-17。**PLANNED / NOT RUN。** 本页仅依据普通已发布源码与准备度记录；未读取实际状态、资格链、权重、图像或GT，未执行数值或实现runner。当前U40主训练、模型、50轮预算及停止规则不变。

当前直接缺口是：U43全人口像素/实例检查尚未实际接收；U41仅有单图函数，其真实无GT预处理、teacher构造/配置闭合及新图数值尚未执行；E50评价入口固定内部187图/缓存，不能只改人口常数用于HRSID。**只有固定E50/37050双臂实际接收、内部效用按原规则获准推进后，才启动本路线。** 首组至少1AP的原门槛及小正收益/重复规则沿主计划，不在外部TEST上重新决定阈值、端点或配方。

最小接口是原TEST顺序的图像清单→小窗口CPU载体→三路最终预测→一次全人口COCO。由已核原split派生仅含 `split/ordinal/image_id/image_ref/original_shape` 的1962图清单；预测阶段不准入annotations或matcher，不用旧SSDD缓存补身份。GT此前的U43用途是输入兼容检查；本路线的模型预测不读GT，最终指标阶段才读取原TEST标注。

1. **采集同一窗口。** 调用已发布 `U41.capture_native_image(torch,np,model,image_metadata, modules/source_refs/preprocessing/provenance/closed_ref/check=...)`。teacher由原调用方按既有构造/加载来源恢复；使用原800 Resize、实际mean/std/channel及training=True图像补边路径，不能套用新分支的RGB/255。保留最终mask_embed输出e，不能替换为h。U41校准原完整fusion掩膜/分数/类别后，将所需载体复制到CPU：`e[0]/M[0]/L/C[0]/identity/export_query_ids/native_predictions/S0_by_query`，自然轴均为0..99。F只在需要原强R精修时暂存，其他情况下保留摘要后释放，不送给新UNet分支。
2. **轮换GPU模型。** 固定小窗口B=1..4，按实际资源预锁；窗口不跨叶积累。采集后释放teacher的全部GPU别名、hooks与临时tensor，实际allocated/reserved均0，才让PLAIN上卡；该臂处理同一窗口并释放至双0后，再运行FULL。同一时刻只有一个大模型驻GPU。叶内可复用已加载且状态核验不变的CPU模型/状态，或按臂加载；**不要求每图从磁盘重读teacher**。CPU驻留与恢复峰也计入16GiB限制，不预先承诺可行或加速。
3. **复用原数值，替换数据接线。** 两臂使用实际Root接收的同一E50端点、原factory、strict模型/BN/inactive状态与初始化谱系检查，不创建优化器。完整checkpoint可能连带反序列化CPU Adam字典，及时释放，不能称未加载过该容器。新RGB分支仍沿原几何生成400输入，每图每臂一次encode、100查询按4个一组读出。CPU载体按真实导出qids索引自然e/M/L，再计算 `L_q + geometry.render_delta(delta_q)`，阈值仍`>0`；所有类别、S0、查询和非segmentation字段保持。逐查询显式零delta校准原L、RLE及整份native字典；不把训练head清零。全域残差不硬限Ω，不转连通域作实例。
4. **只封存最终结果。** 每片保留native/PLAIN/FULL最终RLE、原顺序与最少身份/摘要、来源和实际资源/成本记录；处理完释放CPU载体。不得保存全人口F/M/L/e/RGB400或校准bitmap。固定范围无重叠/缺项，原actual0与Root接收后才进入合并。某图失败不能跳过；期限/OOM/源不符则保留partial，不授该片完整性、不自动缩块重试。
5. **全人口统一评价。** 全部1962图、每方法196200个原查询齐全后，按原TEST顺序合并，再对原5922实例运行通用官方segm COCO（maxDets1/10/100、原crowd/ignore、全部12指标）。不平均shard AP、不筛空预测、不换merged16951版标注；仅TEST命名空间参与本次合并。保留原非segmentation字段和loadRes面积语义，不为涨点暗改bbox/area策略。

| 方法 | 外部来源与当前可用边界 |
| --- | --- |
| native | U41现场采集同一个冻结teacher后原fusion导出；没有现成1962图合格预测。 |
| PLAIN / FULL | 同一固定E50双臂状态；端点与内部效用尚待实际完成，不能拿中途点或旧语义best权重替代。 |
| 强R | 现有内部187图RLE不能搬到HRSID。若外部预锁要求该参考，须接齐原固定head、F采样/投影、Ω/K与精确原渲染，在同流式窗口中实际产生外部R预测；不能用R的二值RLE冒充logit基底。未接齐时明确缺项，不能看外部AP后决定删参考；本候选基底始终是native L。 |

内存不能按内部缓存方案放大：L是**原图**`[100,H_i,W_i]`，不是800补边画布。其全人口FP32大小为 `4×100×Σ(H_iW_i)` 字节。仅作同形示例，若1962图都为800×800，L即502,272,000,000字节（502.272GB，约467.78GiB）；不是已测人口占用。单图此例L约244.14MiB，加F约39.06MiB、M约15.26MiB、e约0.10MiB；B=4仅这些载体约1.17GiB，**尚未包含模型、恢复容器、RGB、query临时量及分配器峰**。不能据此直接宣布满足RSS门。RLE体积也依数据变化，按每片实际输出字节规划剩余磁盘与保留余量，不承诺压缩比例。

Root统一oneleaf：每叶1800秒、CPU0/1和2线程、4GiB allocator/8GiB整卡、RSS≤16GiB/系统可用RAM≥16GiB，保持FP32与原TF32/确定性设置。先锁一个短的**正式预测片**，其结果及资源记录就是全人口的一部分，不另加一套资格试点；据实际耗时/峰值安排其余固定范围，不猜1962图一叶可完成。记录teacher恢复/采集、CPU暂存、两臂恢复/encode/readout/render/RLE及整叶walltime，不能把仅UNet时间报成完整自动pipeline FPS。

本结果最多是源训练后固定参数、零外部调参的受限官方切片评价。既有HRSID TEST曾参与选择、编码重复图/异标注及母景独立未知必须披露；不能恢复盲测或直接声称未见场景泛化。teacher初始化已有官方端点多指纹关联，但完整远端SHA256与独立训练谱系仍未证明。U43即使通过也只证明输入可读/栅格兼容，不证明语义对齐或官方JPG与PNG像素相等。

依据：[U41接口与边界](code/U41_UNBOUND_NEW_IMAGE_NATIVE_CAPTURE_DRAFT_20261017_01/README.md)、[E50设计](U40_FIXED50_DEV_EVALUATION_DESIGN_20261017.md)、[预测数学/恢复源码](code/U40_FIXED50_DEV_PREDICTION_DRAFT_20261017_01/prediction_core.py)、[最薄内部入口](code/U40_FIXED50_DEV_ENTRY_DRAFT_20261017_01/README.md)、[跨集准备度及来源增量](N16_FORMAL_TEST_AND_CROSS_DATASET_EVIDENCE_READINESS_20261014.md)、[主计划及效用规则](../refine-logs/UNETPP_FULL_METHOD_20261017_01/EXPERIMENT_PLAN.md)。既有primary：[官方COCO实现](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py)、[官方模型索引](https://github.com/open-mmlab/mmdetection/blob/v3.1.0/configs/mask2former/metafile.yml)。未新增文献/权重下载。
