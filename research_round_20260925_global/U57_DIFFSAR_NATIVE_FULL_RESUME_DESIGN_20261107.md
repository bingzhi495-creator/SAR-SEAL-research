# DiffSAR完整训练：原生采样与分段恢复的最小设计

2026-11-07。只读固定作者ZIP及已结束健康阶段的普通源码副本；没有SSH、科学库导入、图像/标注/权重读取或实际续训。本页是待实际连续/分段比较检验的设计，不授予恢复或完整外部基线资格。原作者提交为 `bdc3362372213bc8c3a1f6d5b7b74dc224d1b871`。

## 已核实的源码边界

- `detectron2/data/samplers/distributed_sampler.py:36–69`：TrainingSampler将seed、rank和world_size存为属性；`_infinite_indices`每次新建独立torch.Generator并按seed重建无限randperm流。保留这个原实现，以已消费索引数跳过其前缀即可恢复次序，不必重写shuffle算法。sampler_seed单独固定并保存，不能未经核实将它与cfg.SEED混称；原seed=None会调用shared_random_seed，属于额外的全局NumPy随机消耗。
- `detectron2/data/build.py:512–529,318–328`：原loader链为AspectRatioGroupedDataset → PyTorch DataLoader → ToIterableDataset → MapDataset → DatasetFromList。workers=0、单GPU、batch1为本方案的锁定条件，不可按同一恢复协议静默换为多worker/多GPU。
- **`detectron2/data/common.py:290–301`在yield前已经清空bucket**：先复制data，再`del bucket[:]`，最后yield。故batch1的返回边界两桶应为空，不存在需要预读或保存的未发出batch。不能误称清空发生在yield之后。
- `common.py:59–103`的MapDataset另持有`random.Random(42)`与fallback候选集合，不属于全局Python随机状态。两者应保存/恢复并核验；原DiffSAR mapper在成功路径始终返回dict（`dataset_mapper.py:72–121`），裁剪后零实例也不等于返回None，不应挑换样本。
- mapper分支、flip、短边resize和crop使用全局NumPy状态（`dataset_mapper.py:84–92`；`augmentation.py:176–184`；`augmentation_impl.py:102–111,168–178,400–430`）。只保存seed不足以保留后续增强；必须保存完整随机状态。

## 单一建议路径：先创建内层iterator，再恢复随机状态

原AspectRatioGroupedDataset本身是lazy generator，`iter(grouped_loader)`并不能证明PyTorch内层iterator已经创建。不能仅在这句后恢复RNG，然后声称新增DataLoader的base_seed消耗已经抵消。

```python
native = TrainingSampler(741, shuffle=True, seed=saved_sampler_seed)
offset_sampler = SkipNativeSampler(native, consumed_indices)
# SkipNativeSampler仍是torch Sampler，只对iter(native)做islice前缀跳过。
grouped = build_detection_train_loader(
    dataset=canonical_records, mapper=original_train_mapper,
    sampler=offset_sampler, total_batch_size=1,
    aspect_ratio_grouping=True, num_workers=0,
)
dl = grouped.dataset                         # 原native PyTorch DataLoader
map_dataset = dl.dataset.dataset             # ToIterableDataset.dataset
dl_iterator = iter(dl)                       # 强制创建，不调用next、不映射旧图
iterator = iter(AspectRatioGroupedDataset(dl_iterator, batch_size=1))
restore_map_dataset_state(map_dataset, checkpoint)
restore_python_numpy_torchCPU_all_visibleCUDA_RNG(checkpoint)
# 之后才next(iterator)，再调用原model/反向/optimizer/scheduler。
```

fresh和resume均走同一路径。resume重建模型、优化器、scheduler及DataLoader时的额外随机消耗全部发生在恢复checkpoint随机状态之前；不得在恢复之后再次seed或抽新sampler_seed。内层iterator创建是否尚未调用mapper，需在实际连续/分段检验中记录，并将实际Torch DataLoader/fetch源码纳入选定来源；本页没有独立读取DGX上Torch2.10的完整DataLoader实现，不能把本设计当作该运行时已经验证。

跳过只消耗原独立sampler的索引，不调用loader来重放旧图，因此不重复执行旧增强、不消耗其NumPy状态。canonical顺序仍是741个逻辑records，不按730份不同JPEG字节去重。单GPUbatch1的已消费sampler索引数与已完成更新数对应；若步骤失败，不能把已取但未成功更新的样本写成已提交训练步。

## 原子边界与完整状态

在一次原`optimizer.step()`、随后`scheduler.step()`、CUDA同步全部完成后保存：模型全部state（保留扩散buffer的原float64等dtype）、native optimizer全部group/状态与逐参数step、native scheduler state、每个参数grad的None标记或精确tensor、Python/NumPy/TorchCPU/所有可见CUDA完整RNG、MapDataset独立_rng与fallback集合、sampler seed/消费cursor、绝对global_updates、配置/来源/输入身份与rank/world_size。EventStorage.iter恢复为绝对训练计数，不能每段归零；原完整37050计划不缩成分段长度。

Root已选择**保存/恢复param.grad**，不改变原zero_grad位于下一forward之后的顺序；state_dict本身不含grad。额外梯度载荷及约0.5–0.6GB量级的潜在I/O/RSS成本需实测记录，不能省略后仍称边界所有训练tensor相同。公开native LRMultiplier.state_dict只返回base_lrs和last_epoch（`solver/lr_scheduler.py:119–125`）；还需核对恢复后的实际optimizer group LR与原native计划、下一step顺序，不能只看一个epoch标签。EMA/AMP关闭时不另造EMA或GradScaler状态。

每段输出采用新文件名、原子完成、不可覆盖前一段checkpoint；失败保留上一已提交状态及未提交计算记录，不自动把观察超时当终止、重启或恢复依据。资格链和权重保留私有，公开仅安全训练计数、资源与结果范围。

## 一次有限且有意义的实际恢复检验

采用Root选定的两次调用：A从0连续至8，在4保存真实checkpoint后不中断继续；B从这个同一A4恢复到8。共12次实际更新，不做第三次独立重训。比较范围预锁为：

1. B加载起点与A4的模型/optimizer/scheduler/grad、全部RNG、MapDataset与sampler计数必须严格相等；输入和源码身份固定。
2. A/B的第5–8步实际dataset_index/image_id、映射图像、目标mask/box/class字节哈希必须严格相等；绝对step、LR、结束cursor与RNG亦严格核对。不能用相同shape代替增强相同。
3. 完整报告第5–8步loss和最终模型、optimizer浮点tensor的实测差异。Root预锁归一maxdiff `max_abs(A−B)/(1+max_abs(A)) ≤ 1e-4` 只作非确定CUDA环境下的sanity阈值，不是机器舍入理论界，也不授予逐位连续复现结论。超界保持失败并调查，不事后放宽；整数、结构、计数及加载起点不使用这个宽容。

原03曾出现同seed/相同loss但梯度非零计数不同，已有反例不支持预先承诺整个CUDA过程bitexact。以上实际比较尚未运行；通过也仅支持同平台、同来源、单GPU/worker0/batch1及已检验边界的状态恢复，不能代替长期端点、AP、收敛、强对照或TGRS依据。无需另加同样例梯度探针或无限精度测试。

原始来源：[作者固定源码](https://github.com/XiaowoXuSAR/DiffSARShipInst/tree/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871)。同协议50轮是完整外部适配设计；与含COCO教师上游成本的精修系统仍须分层比较，不能由相同更新次数推出初始化或总计算公平。
