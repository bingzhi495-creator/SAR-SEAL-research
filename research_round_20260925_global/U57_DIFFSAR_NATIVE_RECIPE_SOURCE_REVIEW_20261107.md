# DiffSAR原公开配方：有限真实训练健康阶段的源码依据

2026-11-07。状态：**仅源码审阅和实施建议，尚未锁定或执行32更新训练**。只读本机作者固定提交`bdc3362372213bc8c3a1f6d5b7b74dc224d1b871`归档；未SSH、导入科学库、读图像/GT/权重或运行训练。本页不认证论文实际初始化来源，也不产生AP或新方法结论。以下路径全部相对该作者提交；`train_net.py`特指仓库根目录，不是`projects/DeepLab/train_net.py`。

## 先纠正容易套错的默认值

SSDD YAML原本已经是**batch=1、BASE_LR=3.5e-5、WARMUP_FACTOR=1.0、WARMUP_ITERS=10**。不能套用其他Detectron2/COCO配置的batch16、0.1倍或1000步warmup。

| 实际配置 | 原值 | 来源 |
|---|---|---|
| 优化器 / LR / backbone倍率 | ADAMW / 3.5e-5 / 1.0 | `configs/DiffSARShipInst-SSDD-train.yaml:333–334,350` |
| 权重衰减 | 0.0001 | 同上`:359` |
| 全模型梯度裁剪 | enabled / full_model / 1.0；YAML norm_type=2.0 | 同上`:338–342` |
| batch / 参考worldsize | 1 / 0 | 同上`:344,351` |
| LR计划 | WarmupMultiStepLR，MAX_ITER=18560，GAMMA=.1，milestones14848/17632 | 同上`:343,345–355` |
| warmup | factor1.0 / 10更新 / linear，RESCALE_INTERVAL=false | 同上`:352,356–358` |
| loader | 4 workers、TrainingSampler、aspect grouping=true、FILTER_EMPTY=false | 同上`:2–7` |
| seed / AMP / EMA | 40244023 / false / false | 同上`:325,329,331–332` |
| 骨架 / 归一化 | R101，SCJE-FPN，FrozenBN，FREEZE_AT=0 | 同上`:53–55,130–150` |

`MAX_ITER=18560`数值恰为928×20，但这份YAML本身未声明实际数据人口，不能单凭这个等式认证作者实际训练图数，更不能把它当作741图FIT上的20轮。有限健康阶段保留该LR计划，只在外部循环到32次真实optimizer更新后停止，不把32当作作者完整训练预算。

## 原优化器有独立的分组规则

根`Trainer.build_optimizer`（`train_net.py:108–154`）遍历`model.named_parameters(recurse=True)`，跳过`requires_grad=false`并按参数对象去重；**每个可训练参数各为一组**。每组`lr=cfg.SOLVER.BASE_LR`、`weight_decay=cfg.SOLVER.WEIGHT_DECAY`，仅当名字包含`backbone`时再乘`BACKBONE_MULTIPLIER`（`:111–122`）。当前倍率为1.0，所以所有组起始LR相同。

它不调用通用Detectron2的`get_default_optimizer_params`。尽管YAML含`WEIGHT_DECAY_NORM=0`、`WEIGHT_DECAY_BIAS=null`和`BIAS_LR_FACTOR=1`，**该作者实现没有读取这些选项，也没有norm、bias或embedding特判**。不能为了“常规做法”自行增加零衰减分组。FrozenBN的不可训练buffer不是这段循环中的参数。

ADAMW分支直接调用经过包装的`torch.optim.AdamW(params, cfg.SOLVER.BASE_LR)`（`:146–149`）。betas、eps等未显式传入，取实际Torch运行时默认；执行时应记录optimizer.defaults，而不是在本次静态审阅中假定其数值或替换实现。也不要擅自合并每参数组、改foreach/fused或更换优化器以提速后称为相同原配方。

裁剪由作者内层`FullModelGradientClippingOptimizer.step`实现：先把全部group的params串联，调用`torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)`，再执行`super().step(closure=closure)`（`:124–139`）。当前full_model/正clip值开启，随后不会走通用per-parameter裁剪分支（`:152–153`）。该调用未显式传`norm_type`，使用Torch默认；YAML的2.0并没有直接传入本包装。应直接调用作者`Trainer.build_optimizer(cfg, model)`，避免手写AdamW遗漏此行为。

## LR调度与更新顺序

根Trainer继承`DefaultTrainer.build_lr_scheduler`，实际转调`detectron2.solver.build_lr_scheduler`（`detectron2/engine/defaults.py:534–540`）。字符串`WarmupMultiStepLR`在本版本走的是`MultiStepParamScheduler`→`WarmupParamScheduler`→`LRMultiplier`（`detectron2/solver/build.py:283–323`），不是同文件树中已标为deprecated的旧scheduler类。

warmup起点为factor×原scheduler(0)，终点为scheduler(warmup_length)；factor=1，首衰减远在14848，所以起终点均1（`detectron2/solver/lr_scheduler.py:22–58`）。`LRMultiplier`用`last_epoch/MAX_ITER`取倍率（`:95–125`）。因此保留原18560计划时，前32次更新LR都应为3.5e-5；执行应记录实际LR以核对，而不是只写“已warmup”。

不要为限定32更新而把cfg.SOLVER.MAX_ITER改成32：builder会过滤大于MAX_ITER的milestones并改变计划定义（`solver/build.py:289–301`）。有限退出条件和原scheduler全周期应分开。

原非AMP `SimpleTrainer.run_step`顺序是读取data→model(data)→sum(loss_dict)→`optimizer.zero_grad()`→`losses.backward()`→`after_backward()`→写metrics→`optimizer.step()`（`detectron2/engine/train_loop.py:288–339`）。作者full-model裁剪在最后的step内部发生。`LRScheduler.after_step`记录本次所用LR后再`scheduler.step()`（`detectron2/engine/hooks.py:356–359`），所以外部有限循环也应先optimizer再scheduler。梯度观察应标明裁剪前/后；不能把参数随AdamW衰减移动误判为非零数据梯度。

## loader、sampler、单GPU与seed

根`Trainer.build_train_loader`构造作者`DiffSARShipInstDatasetMapper(cfg,is_train=True)`，再调native`build_detection_train_loader`（`train_net.py:102–105`）。native builder接受已装载的dataset list、mapper、sampler、total_batch_size、aspect_ratio_grouping、num_workers，并自动包装DatasetFromList/MapDataset（`detectron2/data/build.py:469–530`）；不要重复映射两次。

默认cfg路径使用`TrainingSampler(len(dataset))`（`data/build.py:418–466`），而非顺序读取。有限健康适配若要求规范741顺序的前32图，可显式传`InferenceSampler(len(records))`，保留全741逻辑records，依照portable manifest的dataset_index重排并核对image_id，batch仍1、workers显式0、aspect grouping保留true。native loader只要batch1，每个宽/高桶收一项就可立即返回；应在运行记录逐步核实actual image_id，不能仅相信loader名字。此处InferenceSampler只决定顺序，**mapper仍is_train=True、模型仍train()，不意味着在做推理**。显式顺序和worker0是健康阶段适配，不冒充作者随机shuffle/4worker轨迹。

原TrainingSampler默认shuffle=true；seed未传时先用`comm.shared_random_seed()`，之后以独立Torch generator逐轮randperm生成无限流，再按rank/worldsize切片（`data/samplers/distributed_sampler.py:36–69`）。也可显式使用原`TrainingSampler(len(records),shuffle=False,seed=40244023)`得到逐轮arange；二者在单卡前32索引可一致，但应在预锁中选定一个，不在执行中换sampler。

`default_setup`调用`seed_all_rng(cfg.SEED+rank)`（`engine/defaults.py:218–220`）；该函数设置NumPy、Torch、Python random和PYTHONHASHSEED（`utils/env.py:27–45`）。原共享sampler种子由`np.random.randint(2**31)`产生后all_gather（`utils/comm.py:195–206`），所以不应把它直接记成40244023；其实际值受此前RNG调用顺序影响。原loader子worker以`torch.initial_seed()%2**31+worker_id`重设种子（`data/build.py:634–636`），worker0会取消这一进程分支。未初始化distributed时native comm返回worldsize1/rank0（`utils/comm.py:21–34`）。

`build_batch_data_loader`要求total_batch_size可被worldsize整除，每进程batch=total/worldsize（`data/build.py:308–319`）。作者REFERENCE_WORLD_SIZE=0，`DefaultTrainer.auto_scale_workers`直接返回原cfg（`engine/defaults.py:675–680`），单GPU不自动改LR/迭代数。未初始化分布式时应核验native comm的worldsize=1。固定SEED并不自动让worker0与原4worker增强序列相同；要记录三类RNG的设定时点及实际顺序，不能声称与此前单样例探针逐位相同。

## mapper与目标：保持作者实现

作者mapper必须收到`file_name`、`height`、`width`及训练`annotations`。标准native COCO loader会按image_id排序、设置bbox_mode=XYWH_ABS，并在给定dataset_name时将类别ID映射至0起始（`detectron2/data/datasets/coco.py:73–101,153–217`）。若不传dataset_name，则不能假定category_id=1会自动变0。它会移除无合法polygon的实例（`:180–192`）；运行时应记录解析前后实例数，不静默丢失GT后仍称1661全部参与。

mapper深拷贝record，以RGB读取/check_image_size（`diffsarshipinst/dataset_mapper.py:80–82`）；基础变换为RandomFlip与ResizeShortestEdge（`:19–35`）。crop启用时，以np.random.rand分支选择直接基础变换或flip→[400,500,600]短边resize→absolute_range crop→最终shortedge/max512（`:55–60,84–92`）。**输出不是强制512×512**，例如比例约束可生成400×512；网络之后还会自行pad。

变换后的HWC图像转为连续CHW tensor（`:94–99`）；annotations去keypoints，过滤iscrowd，按同一transforms更新（`:106–118`），硬编码`annotations_to_instances(...,mask_format='bitmask')`再`filter_empty_instances`（`:119–120`）。因此尽管YAML的INPUT.MASK_FORMAT写polygon，实际训练目标为BitMasks。应保留BitMasks对象，不能只替换为裸tensor。随机crop可能让真实实例变空，这属于原mapper行为；不要为了确保非空挑换下一张图而悄悄改变样本轨迹。

## 权重加载、检查点与最小调用边界

原YAML `MODEL.WEIGHTS:`是空值（null）而非明确的预训练地址（`:321`）。native `build_model`只构造/搬到device，不加载配置权重（`detectron2/modeling/meta_arch/build.py:16–24`）；作者主程序在Trainer构造后才`resume_or_load`（根`train_net.py:269–271`）。DefaultTrainer的加载规则是：resume且有last_checkpoint优先恢复全部可用状态，否则加载MODEL.WEIGHTS并从0开始（`engine/defaults.py:398–416`）。DetectionCheckpointer还包含模型键转换/匹配机制（`checkpoint/detection_checkpoint.py:116–143`）。

本轮健康阶段若不加载任何checkpoint，应显式写为**作者公开代码构造初值下的适配**；这不能认证作者论文正式实验使用随机初始化，也不能填补此前论文全文/权重来源缺口。应使用新输出目录且不读取历史last_checkpoint，不把健康阶段算作主PLAIN/FULL轨迹或作者完整效果复现。

建议直接使用根Trainer的model/optimizer/scheduler构造类方法、作者mapper与native loader，外部有限循环保留原forward/backward/step顺序。不要直接启动`Trainer(cfg).train()`来只跑32步：它自动挂载PeriodicCheckpointer、EvalHook与Writer（根`train_net.py:186–233`），而EvalHook即使period=0也可能在结束时评价（`engine/hooks.py:551–561`）。健康阶段无DEV/TEST/AP，不能由hook意外触发评价或额外checkpoint搜索。

## 可以支持的下一项实际检验

固定规范顺序前32个真实FIT逻辑样本、原seed/模型数学/随机初始化/优化器/LR/裁剪与mapper，只明确适配loader workers0和顺序sampler。记录全部32更新实际image_id、增强后shape/实例数、24项loss、所用LR、更新成功计数和资源；预定节点观察IAMR/controller参数梯度与共享输出M的grad是否None/有限/非零，区分optimizer更新与AdamW衰减。观察使用原反向，不额外追加forward/backward挑样本。

首图IAMR零梯度不是阻止真实训练的充分条件；若32步中其他样本或原更新后出现非零，能否定“在这些样本/这段轨迹一直无信号”，不能证明已经收敛或有检测效用。若仍全零，也只约束这32步/构造初值，不能宣称DiffSAR总体不可训练。损失非有限、原代码异常或资源中断应保持真实失败并区分原因，不能自动换配方重跑；是否增加单一初始化干预需另立配对设计。

原始来源统一为[作者固定提交](https://github.com/XiaowoXuSAR/DiffSARShipInst/tree/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871)。所有运行资格、完整训练/AP、初始化论文来源与后续强对照均未由本页完成。
