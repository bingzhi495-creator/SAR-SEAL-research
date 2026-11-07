# RF-DETR 下一次训练资格的最小接口草稿

2026-11-07。状态为 source-only 候选，执行仍由 Root 单独绑定来源、数据角色、依赖和资源。本轮只读普通官方源码和普通公开配置，没有导入 Torch、读取权重/图像/GT、访问活动 SOURCE、SSH 或训练。没有新增可自动运行的 worker。

复用已有 RFDETR_FULL_TRAIN_CLI_DEPENDENCY_READINESS、RFDETR_SAR_DATA_INTERFACE_REVIEW、当前 RFDETR_BASELINE_AND_DIAGNOSTIC_PLAN、单类初始化审查和 canonical_export.py。本草稿只补官方调用顺序及一次训练更新的观测契约。固定官方代码为 639eac4555ceb89ef399ccda3fde716bbcfe34cb，版本 1.12.0。

## 必须先满足的输入

- 独立完整 train+cli 依赖 inventory、离线 resolver、安装、pip check 和实际 CLI parser 回执。已有基础 CPU forward 不能替代这些回执。依赖候选和保护的 Torch/CUDA/NumPy 版本沿用现有依赖准备文档。
- 已核官方 COCO SegNano 本地权重和文件身份，禁止使用待下载文件名代替实际路径。新 worker 需要正常离线审计覆盖模型导入和官方 loader 的所有阶段。
- 已获资格的原始 FIT/DEV 图像列表、数据角色、掩膜规则和类别映射。YAML 中 /data/REQUIRES_QUALIFIED_SAR_COCO_SPLITS 是占位符，必须先拒绝，不能由 worker 自动搜索替换。正式数据和 GT 保持原样，读入后增强仅产生训练 target。
- 现公开 BASELINE_YAML 的 A 政策为 num_classes=1，官方 loader 最终保留原分类行 0/1。B 为 boat9/1 的候选初始化对照，仍未取得相同预算的效用证据。首个资格沿用 A，记录所有 decoder/encoder 分类头的实际行来源。slot1 的导出规则与该行的 COCO 语义分开记录。不能给后续方法组静默切换 B。

## 官方 API 顺序

1. **解析并固定配置。** 使用已有 YAML 的 ModelConfig/TrainConfig 子类和 trainer overrides。全局 seed_everything=5201 必须在模型构造前生效，train_config.seed=5201 在 RFDETRModelModule.on_fit_start 还会再次设种子。重复种子同时修改两处。CLI 的 --print_config 在实例化前退出的行为仍应由真实 parser 检查确认，LightningCLI(run=False) 会实例化类，不能当纯解析。
2. **构造训练模块。** RFDETRModelModule(model_config, train_config, load_encoder_weights=False)。该参数在模块构造时用 force_no_pretrain=True 阻止额外 encoder 预训练获取。mc.pretrain_weights 设为已核本地 COCO 权重。__init__ 先 build_model_from_config，再调用 load_pretrain_weights，最后 build_criterion_from_config。原生训练 loader 使用有意的部分加载及结构适配，不能报告为“原 COCO state 原样 strict 成功”。 [实际构造顺序](https://github.com/roboflow/rf-detr/blob/639eac4555ceb89ef399ccda3fde716bbcfe34cb/src/rfdetr/training/module_model.py#L400-L481)
3. **证明初始化适配范围。** 当前配置为 576、PE48、单类双 logit、queries100、group_detr13、mask downsample4、compile/cuda_graphs/fused_optimizer=False、关键点关闭。官方 loader 先临时对齐 COCO 分类头，插值 PE 后加载，再截回两行。PE 插值保留 class token，patch grid 用 float32 bicubic、align_corners=False，在 CPU/CUDA 上 antialias=True，结果转换回原 dtype，目标为 1×(48²+1)×D。应独立列出全部允许变化的分类头、PE 键与空关键点 schema，对其余 Tensor 验证等值和有限性，任何额外差异都保留为失败。首个 CPU312探针的完整等值证据不能覆盖这次不同头和 PE。 [PE 公式与实现](https://github.com/roboflow/rf-detr/blob/639eac4555ceb89ef399ccda3fde716bbcfe34cb/src/rfdetr/models/weights.py#L256-L312) · [加载后截头](https://github.com/roboflow/rf-detr/blob/639eac4555ceb89ef399ccda3fde716bbcfe34cb/src/rfdetr/models/weights.py#L642-L661)
4. **接入原数据模块。** RFDETRDataModule(mc, tc)，由 Trainer.fit(module, datamodule=dm) 正常完成 setup 和 batch transfer。setup("fit") 会构建 train 与 val，即使 limit_val_batches=0，来源资格也须覆盖两者。Roboflow 路径将原 category1 映射为训练 label0，保留 label2cat={0:1}。正常 batch 为 NestedTensor 和 target dict 列表，boxes 为当前训练画布上的归一化 cxcywh，mask 与当前画布对应，orig_size/image_id 保留原身份。size 与 orig_size 不可混用。不得以框填 mask，原 GT 不重写。
5. **保留原 matcher/loss/optimizer。** criterion 和 matcher 由同一 mc/tc 构造。默认 matcher 分类/L1/GIoU 权重 2/5/2，mask BCE/Dice 权重 5/5，mask_point_sample_ratio=16。训练权重为分类1、框5、GIoU2、mask BCE5、mask Dice5，辅助 decoder 层及 encoder loss 保留。ia_bce_loss=True、focal_alpha=.25、aux_loss=True、sum_group_losses=False，均需从实际对象及完整配置记录，不自行改成教程配方。AdamW 参数组通过 configure_optimizers→get_param_dict，保留 lr=1e-4、encoder lr=1.5e-4、weight_decay=1e-4、cosine min_factor=.05、warmup3，clip_max_norm 默认 .1。分割使用 Lightning 自动优化，不能再次手动除以累积4。 [原 criterion](https://github.com/roboflow/rf-detr/blob/639eac4555ceb89ef399ccda3fde716bbcfe34cb/src/rfdetr/models/lwdetr.py#L922-L990) · [原 matcher](https://github.com/roboflow/rf-detr/blob/639eac4555ceb89ef399ccda3fde716bbcfe34cb/src/rfdetr/models/matcher.py#L1130-L1164) · [自动优化路径](https://github.com/roboflow/rf-detr/blob/639eac4555ceb89ef399ccda3fde716bbcfe34cb/src/rfdetr/training/module_model.py#L894-L1002)

## 一次更新与原调度的区别

batch1×accumulate4 意味着一次完整 optimizer update 通常消耗四个真实 microbatch。资格必须实际记录消费数、增强后非空实例数、mask loss 是否被执行、backward 次数、AdamW step counter 和梯度有限性。全部微批次若均无匹配实例，只能覆盖背景分类等路径，不能声称 mask 训练已验证。

不建议以 max_steps=1 加速后再称原 100 轮调度前缀。configure_optimizers 根据 trainer.estimated_stepping_batches 确定总更新数，max_steps 或 limit_train_batches 会改变调度规模。若要求保持原配方，保留 max_epochs=100、完整训练集长度、默认无限 max_steps，以资格观察 callback 在预定 global_step 达到后令 trainer.should_stop=True，Root 外部另有时限。callback 自身停止行为需在其真实资格中核验，不能靠本草稿授予许可。

官方 warmup factor=current_step/warmup_steps，LambdaLR 初始化阶段学习率可能为零。首次 optimizer step 因此允许参数零变化，但必须证明非零有限梯度、优化器状态初始化、step counter 增加及 EMA 路径确实执行。若出口要求参数真实变化，最小检查需要第二次 optimizer update，即当前累积设置下至多八个预定微批次，且不得擅自取消 warmup。 [真实调度](https://github.com/roboflow/rf-detr/blob/639eac4555ceb89ef399ccda3fde716bbcfe34cb/src/rfdetr/training/module_model.py#L314-L355) · [总步数来源](https://github.com/roboflow/rf-detr/blob/639eac4555ceb89ef399ccda3fde716bbcfe34cb/src/rfdetr/training/module_model.py#L1660-L1759)

最小构造建议使用官方 build_trainer(train_config=tc, model_config=mc, include_training_callbacks=False, extra_callbacks=[官方RFDETREMACallback, 资格观察callback])，保持真实训练模块和优化器，但不自动写训练 checkpoint 或实例化训练日志服务。该模式仍会附加原 COCOEvalCallback，可通过 limit_val_batches=0、num_sanity_val_steps=0 暂不验证。include_training_callbacks=False 同时省去了默认 EMA，因此必须显式加入官方 EMA callback，使用 tc.ema_decay/.ema_tau/.ema_update_interval，当前默认 .993/100/1。不要保留 ModelCheckpoint callbacks 后再仅设 enable_checkpointing=False。此构造是短资格入口，正式 E100 使用完整官方训练 callback 栈。 [官方 Trainer 入口](https://github.com/roboflow/rf-detr/blob/639eac4555ceb89ef399ccda3fde716bbcfe34cb/src/rfdetr/training/trainer.py#L537-L588) · [默认 callbacks 与覆盖顺序](https://github.com/roboflow/rf-detr/blob/639eac4555ceb89ef399ccda3fde716bbcfe34cb/src/rfdetr/training/trainer.py#L930-L1006)

EMA 在 on_fit_start 建立 AveragedModel，在每个合格 optimizer step 后更新，on_train_end 会将 live 模块换成 EMA。因此 raw 参数变化、梯度和 optimizer 状态须在 on_train_batch_end 或更早的正确钩子保存摘要，再单列 EMA counter。仅在 fit 返回后比较参数会混淆这两种状态。 [EMA 生命周期](https://github.com/roboflow/rf-detr/blob/639eac4555ceb89ef399ccda3fde716bbcfe34cb/src/rfdetr/training/callbacks/ema.py#L267-L443)

## 真实 batch 的输出和精度契约

训练模式使用 group_detr13，queries 和辅助分支的成本远高于 CPU312 eval。LWDETR 在训练模式调用 segmentation_head.sparse_forward，pred_masks 是包含 spatial_features/query_features/bias 的结构，并由 criterion 做点计算。不能把 eval 的密集 [1,100,78,78] 掩膜形状检查直接复用到训练。576 下掩膜空间特征网格由 downsample4 对应 144×144，具体中间特征、query 数及辅助层键须按原对象记录，无需为记录额外物化所有 dense masks。 [训练稀疏分支](https://github.com/roboflow/rf-detr/blob/639eac4555ceb89ef399ccda3fde716bbcfe34cb/src/rfdetr/models/lwdetr.py#L503-L520) · [实际稀疏结构](https://github.com/roboflow/rf-detr/blob/639eac4555ceb89ef399ccda3fde716bbcfe34cb/src/rfdetr/models/heads/segmentation.py#L361-L405)

先按既有计划在独立 FP32 资格中验证真实加载、损失、backward、optimizer 与 EMA，再从同一 COCO 初值和相同随机种子另做 BF16 资格。官方 amp_dtype=bf16 在设备不支持时可能回落 fp16，必须记录实际 Trainer precision plugin、选中设备的 BF16 能力及实际 autocast dtype；出现回落应保留为未通过所要求 BF16。BF16 mixed 下参数、Adam 矩及 EMA 通常仍为 FP32，不能由 128GB 标称统一内存推导吞吐或可用训练容量。分阶段记录加载、EMA复制、首 forward/backward、首次 Adam 状态分配后的 allocated/reserved 峰值，以及 CUDA 同步后的耗时；主机 RSS 另外计数。两种精度数值不要求逐位相同，都需要 loss/梯度/参数有限。 [官方精度选择与回落](https://github.com/roboflow/rf-detr/blob/639eac4555ceb89ef399ccda3fde716bbcfe34cb/src/rfdetr/training/trainer.py#L634-L734)

## canonical 出口与最小回执

若未来另获单图导出资格，在记录模型为 raw 还是 EMA 后，用原生 PostProcess 对全部 query×class sigmoid 选 top100，score_threshold=None、target_sizes=原始H/W、upsample_masks_to_image_size=True。再交当前 canonical_export.py，显式 label2cat={0:1}，过滤 slot1、不补选，原尺寸 bool mask 由冻结 codec 编成 RLE 并做实际像素 round-trip。独立 canonical evaluator 使用原 image/category ID、crowd、area、原始掩膜和 maxDets100。在线 callback 的缩小再放大 GT 仅属于内部通道，不能覆盖 canonical 真值。

未来 worker 的回执最小包含输入来源及实际数据角色、初始化政策A及键变化、576/PE48、解析后完整配置和 matcher/criterion 摘要、seed状态、实际微批次和非空目标数量、逐项loss、有限梯度及范数、optimizer step与LR/参数组、raw参数变化、EMA计数、实际精度、阶段耗时与CPU/GPU峰值。一次训练资格与一次 canonical 导出资格分别记录，qualified 初始为 False，只有 Root 真实核验相应执行及资源闭合后才作出口判断。此草稿本身不产生训练成功或 AP 结论。
