# N16 配对 scout 训练主体：普通 UNBOUND 接口合同

状态：普通研发草稿。仅真实运行过随机 CPU 合成 fixture；未执行 SAR 科学训练，未读取实际数据、权重、缓存、SOURCE/SPEC/Q/RAW或运行中的协议。训练入口及真实资格由 ROOT 绑定和执行。本文不修改任何冻结文件。

主体：[N16_PAIRED_SCOUT_TRAIN_BODY_UNBOUND_DRAFT_20261014_01.py](N16_PAIRED_SCOUT_TRAIN_BODY_UNBOUND_DRAFT_20261014_01.py)。真实测试文件：[N16_PAIRED_SCOUT_TRAIN_BODY_SYNTHETIC_TEST_20261014_01.py](N16_PAIRED_SCOUT_TRAIN_BODY_SYNTHETIC_TEST_20261014_01.py)。完整 GPU 普通入口：[N16_FAIR_PAIRED_SCOUT_TRAIN_GPU_ROOT_UNBOUND_DRAFT_20261014_01.py](N16_FAIR_PAIRED_SCOUT_TRAIN_GPU_ROOT_UNBOUND_DRAFT_20261014_01.py)。三者均未获得实际科学执行资格；ROOT 对实际绑定副本另行审阅。

## 固定数学与预算

两臂 `UTILITY`、`ERROR` 使用同一个正常初始化 seed `202610140161`，每臂四张量、16547参数，非仿射 LN(F256/e256, eps1e-5)，原始 L/8，Linear513→32/ReLU→3，无 dropout/AMP/梯度裁剪。两臂同一次缓存图读取、同采样槽、同次序、同 FP32 importance weights；utility/error 分别读取对应 INT64 三类标签。

AdamW 固定 lr=3e-4、weight_decay=1e-4、betas=(.9,.999)、eps=1e-8，恒定学习率。损失为 `mean(weight * CE(reduction='none'))`，不按 sum(weights) 归一，不追加类权重。每个非空缓存图每臂一步；空图两臂同跳，真实计数。24轮每轮私有 Torch CPU `randperm(image_count)`，seed=`202610140160001+epoch`，epoch 从1开始。每轮重新建立该私有 generator；完整图序及最后一次 generator before/after state进入快照，不影响外部 CPU/CUDA RNG。

固定端点8/24。端点损失重新遍历全缓存，在固定权重下计算；在线混合权重训练 loss 单独标识，不当端点结果。两端点不用于挑最佳轮次，更不是 AP。固定 refiner 曾接触 FIT，标签生成成本及缓存重复曝光必须披露。

## 主体 API

- `make_paired_state(torch, scout_module, device='cpu')`：构建两臂及独立 AdamW；`scout_module.make_scout(torch, seed)` 返回正常初始化模型。
- `train_one_epoch(torch, state, loader_callback, epoch, image_count=741, observer=None)`：连续执行指定一轮，返回纯元数据；修改传入 state 的模型、优化器、计数、图序和历史。
- `evaluate_fixed(torch, state, loader_callback, image_count=741, observer=None)`：固定当前权重，重算全部有效图的 weighted CE 与分层行动统计，无梯度；返回后恢复原模型 train/eval mode。
- `cpu_snapshot(torch, state)`：完整 CPU model/optimizer/history/orders/RNG/counts 快照，AdamW四参数 step必须有限且精确等于实际 updates；有更新时须有四组 moments，无更新时可为空。
- `train_until(torch, state, loader_callback, last_epoch=24, image_count=741, observer=None, checkpoint_callback=None)`：只允许固定8或24目标，保留原 state/AdamW接续。到8/24时重算端点并调用 `checkpoint_callback(epoch, endpoint, snapshot)`；检查点 IO由 ROOT callback负责，主体不写实际文件。

`observer(event)` 接收每图之前及 paired update 的纯元数据，可由 ROOT核 deadline/RSS、写实际 trace；端点读图前也有通知。

## 薄 packet loader 合同

`loader_callback(dataset_index)` 每次返回该图的已资格、当前绑定来源一致的 CPU 薄 packet，不需要加载完整100查询 natural-L carrier、教师、PR头或GT。ROOT wrapper必须校验 packet 真 ref、source/spec/admission/phase/identity/索引与同一个 qualified GPU report，并绑定 independent CPU Q及 ROOT实际终止/完整SOURCE闭合。

| 字段 | 类型/形状 |
|---|---|
| `sampled_F_unique` | CPU FP32 `[U,256]` |
| `sample_to_unique` | CPU INT64 `[S]`，合法索引到 U |
| `native_e0` | CPU FP32 `[1,100,256]` |
| `samples.query_ids` | CPU INT64 `[S]`，0..99 |
| `samples.native_L` | CPU FP32 `[S]` |
| `samples.importance_weights` | CPU FP32 `[S]`，非空时严格正 |
| `samples.utility_labels` | CPU INT64 `[S]`，help0/harm1/noop2 |
| `samples.error_labels` | CPU INT64 `[S]`，FP0/FN1/correct2 |
| `samples.stratum_ids` | CPU UINT8 `[S]`，matched0/background1 |
| `samples.delta` | CPU BOOL `[S]`，实际 flip `P!=A`，应等于 utility_labels!=2；不是有符号PR残差 |

S=0合法，两臂同跳。F按 unique位置共享，再由 sample_to_unique恢复同一采样槽；重复槽保留。原 background→empty-set 是共同新增代理监督，不是原 Mask2Former negative mask target；其 assignment与GT正确性由缓存资格证明，训练主体不以 labels或mask_weights伪造该证明。

## 端点统计与快照

主损失是各非空图 `mean(weight*CE)` 的均值，各图同等权重。分层统计 matched/background 中 importance加权的真实 help/harm/noop，以及 scout-positive选择量、actual-flip help召回、harm选择率和positive actual-flip help精度。不报 overall accuracy。这里的 sample-positive统计使用 CPU Torch FP64 softmax，仅为 FIT 诊断；不是部署 NumPy FP64排名、topK、全100预测或选点资格。

快照保存两臂完整参数及 AdamW、completed_epoch、updates、visited_images、empty_images、sampled_points、所有 epoch orders/history、最后私有 order RNG before/after、初始化/图序 seeds与优化器配置。ROOT负责8/24真实保存、精确双读回、科学 provenance、独立CPU数值资格、实际原始owner终止及最终全100 official COCO评价。

## 实际合成验证范围

独立物理测试文件已执行，最终输出 `SYNTHETIC_CPU_ONLY_OK`。覆盖：手算 `wCE.mean` 且反证 sumweights归一、共同初始化、同标签两臂更新后参数精确相同、不同标签两臂参数分开但 step数相同、私有图序、每图只加载一次、空图同跳、固定权重端点重算、完整 CPU快照与真实AdamW step、固定8/24 callback及全空24轮零更新、外部CPU RNG未改变且未初始化CUDA。主体及完整入口通过 AST静态解析。没有实际SAR数据/缓存训练、GPU训练、AP或科学终态验证；本次不重复运行已有测试，不执行Git。
