# U39 跨图资源函数普通草稿

状态：未绑定、未执行 Torch/NumPy/模型/真实输入，仅完成标准库 metadata 测试与 AST。没有 runner、binder、Root 资格、权重输出或 AP。六次更新是未来资源试点，正式训练预算记账仍为零。

## API 与明确依赖

`probe(torch,np,c,context,modules,source_refs,pre,report=None,image_control=None)` 返回只含 JSON 元数据的报告。调用者可传入空 `report={}` 保留部分进展。正常完成状态为 `COMPLETED_PENDING_ROOT_REVIEW`；运行中异常返回 `FAILED` 和实际已经返回的 optimizer 更新次数，未来 Root 入口必须拒绝把 FAILED 当作成功，保留原终态和报告，不自动重试。

modules/source_refs 恰有十二角色：U38 已审的 `shared_initialization/rgb_matched_reader/full_legacy/full_dsconv/full_head/full_feature/plain_legacy/plain_feature/instance_adapter/matched_reader/image_geometry`，再加显式 `u36_helpers`。每个模块都核 `c.closed`、绝对模块文件路径、字节数、固定公开 SHA；不得动态搜索旧路径或重新加载另一份代码。

额外依赖只使用公开 `code/U36_REAL400_CPU_SAVED_ACTIVATIONS_DRAFT_20261017_01/probe_body.py`，固定 SHA `1e726092a19d80a06f58d6a56f85f4c15bff2b744f0468a7e0a35fa2c85999e5`。只调用 `_observed_save_on_cpu/_require_actual_cpu_saves/_gradient_report/_optimizer_report/_head_buffers/_verify_buffers/_digest`，不调用其 run、单图选择、packet、两更新主流程或 tiny 契约。未来 wrapper 必须为这第十二模块显式准入并记录来源。

传给 U34.build_pair 的严格子集仍是其七角色；传给 U37.prepare 的 source_refs 仍恰为 matched_reader/image_geometry 两角色。context 仍使用原 FIT_context/N13_qualified/N13_report 三个 strict refs；image_control 如分开提供，必须由 Root 先独立准入。原 reader 的完整 741/1661 metadata、逐载入图 F/e/M/native-L/GT、原 GT/query 顺序和 RGB JPEG/几何校验全部保留，不重写 matcher，不要求全 FIT native100 M/S0。单次试点只实际加载提议中的图，不声称重新完成741图 tensor 扫描。

## 选图与预锁

`select_image_order(records)` 只读取原 G、原 H×W 与 dataset_index。small 取 `(G,H×W,index)` 最小；maxGarea 取 `(-G×H×W,index)` 最小；返回 `[small,maxGarea,small]`。没有 loss/AP 参与，也不要求人为凑足 shape/G 覆盖；如角色退化成同图，报告真实 distinct_probe_images，不能声称已跨不同图。索引必须唯一、0..740；G 是1..100的 Python int，原尺寸为正整数二元组。

预锁字段由 `validate_prelock` 与测试中的 `locked_pre()` 完整给出：schema `U39-full-method-cross-image-resource-prelock-v1`；image_order 为上述原 metadata 结果，arms `[PLAIN,FULL]`，updates_per_arm3/total_pilot_updates6；image_batch_size1/query_chunk1/branch_size400/canvas_size800/dtype `float32`；initialization_seed202610173101。AdamW 固定 lr3e-4/weight_decay1e-4/betas[.9,.999]/eps1e-8/foreachFalse/fusedFalse。training_encode_saved_tensor_mode=`save_on_cpu_pin_memory_true`，checkpointing/AMP/VJP/automatic_retry/main_training均False，no_AP/no_DEV/no_TEST均True。

Root 随后提供已终态并接收的 U38 元数据事实：small 为 dataset_index54、G1、原190×303；maxGarea 为710、G27、原327×489，故未来真实预锁应为 `[54,710,54]`。函数仍须从完整原 reader.records 重核规则与此列表一致，不读取或猜造 U38 report。训练循环使用实际 G，710的27个配对实例必须全部按1/27参与，不能沿用旧U36单图9实例常量。maxG709和maxarea132仍仅候选目录，不新增必须全部运行的门。以上是 Root 已核事实；U39 自身尚无数值结果。

## 实际更新路径，尚未执行

未来 wrapper 必须在本函数前完成库与 Source 准入、CPU0/1及两线程、FP32默认类型、cuDNN deterministic=True/benchmark=False，并保持 CUDA 冷。函数先实际执行 U34 共同104状态与相同 adapter 初始化；只有原回执的摘要进入报告，不额外保存整套 CPU state_dict 克隆或权重文件。另一臂的 pristine 模型暂留 CPU，当前臂由内部函数从字典 pop，避免调用参数栈残留强引用阻止 CUDA 清零。

CPU 准备完成后，本函数负责首次 CUDA 初始化，读取真实显存总量，并在任何模型/输入 GPU 转移前实际调用 `set_per_process_memory_fraction(4GiB/total_memory,device=0)`。future wrapper 仍须硬控单叶、CPU01/线程2、进程RSS≤16GiB、可用RAM≥16GiB、整卡观测≤8GiB与1800秒期限，原 admission/held/FIRST、连续资源观察、导入来源及报告保存均不在本函数伪造。

PLAIN 和 FULL 顺序执行，各自同一模型及同一个 AdamW 连续更新三次。每图仅训练 encode 一次；只在 encode 的 canonical `save_on_cpu(pin_memory=True)` 上包计数，不包查询头、损失或其他保存 hook，不用 checkpoint/VJP/AMP。原 reader 的 e/M/native-L/GT 已是 GT 配对顺序；每 query 单独将 e/M 与该 query 的 native-L、目标转移 GPU，损失为 `1/G × (BCEWithLogits逐像素均值 + 1-(2Σ(sigmoid(logit)GT)+1)/(Σsigmoid(logit)+ΣGT+1))`。GT 仅入损失，F 仅由旧 reader 校验而不送入新主干。共享 encode 图最后一个 query 才结束 retain_graph，AdamW 每图只 step 一次。

有效与失活参数在执行前按原 wrapper 声明划分；不改原 requires_grad，不优化失活参数。每次更新均要求所有有效参数有 finite 梯度，精确零合法并如实记录；失活梯度保持 None。第一步不强求上游非零，后续也不把数据相关的零梯度自动解释为失败。每步检查原 AdamW moments/形状与逐参数 step1/2/3，BN 摘要跨图相接；不重置模型、BN或优化器。原 semantic 四头调用均0，其 BN buffers保持，实际 feature-only backbone/CONV0_4各每步一次。

每图结束 zero_grad(set_to_none=True)，释放输入、原图目标及临时 GPU query/图，函数返回后才同步记录驻留 RSS/allocated/reserved；图间不 empty_cache，以保留真实 allocator/optimizer 驻留观察。每步另记 reset 后的 CUDA allocated/reserved peak，不能把最终一次 peak 当全过程峰。每臂释放模型、优化器和所有局部别名，GC/sync/clearCublas/empty_cache/sync 后必须 allocated=reserved=0；失败 traceback 仅留字符串并清 frame 引用，防止异常帧保留 GPU 张量。两臂和最终清理均保留结果，失败不继续下一臂。

计数是 save/unpack 累计事件，可能重复同一存储与多次恢复，不是唯一驻留大小、带宽或效率收益。图后 RSS 是边界样本，future wrapper 才负责连续 RSS 与整卡观察。单个小图返回观察不能证明无泄漏或全741稳态，六次更新不授主训练准入、收敛、效用、创新或投稿资格。

## 当前验证范围

纯标准库三条 metadata 公共行为测试依次执行 RED→GREEN：稳定选图与不改输入；非法尺寸/重复或越界索引/G拒绝；预锁顺序与固定配方修改拒绝。测试不导入 Torch/NumPy，不构造假 NN，不执行 probe。真实共同初始化、训练图、保存/恢复、梯度、优化器、BN、资源与释放均是未来 Root 实测事项。
