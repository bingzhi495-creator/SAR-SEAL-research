# U40 continuous full-method training core — planned, unbound, not run

本目录只含可供未来 Root 单叶调用的薄训练核心和纯元数据测试。没有 runner、binder、正式 Source/执行锁或作业启动。本轮未导入 Torch/NumPy，未构造模型、读取实际 checkpoint/缓存/图像/GT，未运行 forward、backward 或 optimizer。这里的测试通过不等于恢复数值、资源门或完整训练已通过。

## 固定数值路径与范围

- 只实现首组完整 seed `202610173101`；PLAIN/FULL 各完整 FIT741/1661，50 轮、37,050 次图像更新。每次使用原 reader 的全部 G 个原匹配查询，没有重新 matcher、筛图、阈值选择或中途 AP。
- 直接调用公开 U39 `resource_probe.py::_update`，固定 SHA `378c955877674a65f6b2180d4c269e01355245881229ed0d2ec555b184ee7e5d`。传入全局 `step=ordinal+1`，不调用其硬三步 `_run_arm` 或 `probe`，不复制数值循环。
- 原 RGB400/FP32/B1、query chunk1、native L、原图渲染、每图 G 查询的均值 BCEWithLogits + softDice(smooth1)、一次共享 encode、canonical `save_on_cpu(pin_memory=True)`、重复查询 backward 原样保留。无 AMP/VJP/重算 checkpoint/新增 scheduler。
- AdamW 使用排序后的预声明 effective 参数；lr `3e-4`、wd `1e-4`、betas `(0.9,0.999)`、eps `1e-8`、foreach/fused 均 false。inactive 参数仍实际分配，梯度必须 None；effective 梯度有限，允许零。旧四语义 head 不调用；PLAIN 12 / FULL 18 个旧 head BN buffers 不变，其余 BN 按原训练行为持续更新。
- 上述是普通未绑定实现提案；尚不能据此声称 50 轮收敛、泛化、资源稳定或完整轨迹 bitexact。

## 调用合同

```python
report = train_segment(
    torch, np, c, context, modules, source_refs, segment,
    parent_checkpoint_refs, output_directory,
    report={}, image_control=None,
)
```

`torch/np/c` 由后续 Root entry 按原公共 ABI 注入。`c.closed(strict_ref)`、`c.check()`、`c.psutil` 沿既有控制器；本核心不自行构造或绕过准入。`context` 保持 U37 原始三引用。`image_control` 仅原 U37 支持的预准入 JPEG 引用控制，不改变 reader 验证。

`modules/source_refs` 必须是同一精确 13 角色集合：

```
u39_training_ops
shared_initialization rgb_matched_reader
full_legacy full_dsconv full_head full_feature
plain_legacy plain_feature instance_adapter
matched_reader image_geometry u36_helpers
```

其中原 12 角色的源 SHA 与模块路径由固定 U39 `_closed_sources` 核验；另核 U39 固定 SHA 和本 `train_core.py` 的 caller-admitted `core_source`。不修改任一公开旧源码。Root 未来还须闭合 entry/runtime、实际解释器依赖与所有输入 refs。

`segment` 至少显式给出：

```
seed: 202610173101
start: int                 # 已完成的共同更新数
stop: int                  # start < stop <= 37050
order_sha256: str
recipe: 精确等于 train_core.RECIPE
limits: 精确等于 train_core.LIMITS
core_source: {path, bytes, sha256}
order_ref: None 或首叶 ORDER.json 的 strict ref
parent_pair_ref: None 或已由 Root 接收的 PAIR_COMMIT.json strict ref
```

`parent_checkpoint_refs` 精确两键 PLAIN/FULL：首叶都 None；续段必须是对应 arm 的 strict checkpoint refs。Root 必须先实际接收父叶，再将其 pair/checkpoints/order/两份 trace refs 准入；核心验证字节和身份，不伪造 Root 接收资格。`output_directory` 为调用者预准入的全新绝对科学输出目录，父目录已存在；已有目录、checkpoint、partial 均拒覆。返回 report 由 caller 真实保存；caller 必须把 FAILED 当失败，保存 stderr/partial，不能仅依据函数正常返回判成功。

## 唯一图序与随机状态

生产路径要求 CPython 3.8、Torch 2.0.1+cu118、NumPy 1.23.5。第一叶以具名私有 `random.Random(202610173101)` 预生成 50 轮：每轮新建 `list(range(741))` 后 shuffle。保存完整 37,050 列表、Python 精确版本、生成器初末 state 及覆盖这些字段的摘要到 `ORDER.json`。纯元数据生产可由 Root 事先用相同 Python 得到预锁 `order_sha256`；这不是模型执行。

续段只读取封存列表和全局游标，逐轮核完整人口、版本和摘要，不再 shuffle。两臂共用该列表；checkpoint 记录其 strict ref。`cursor=741*k+r` 表示完成 k 个完整 epoch 和本轮前 r 图；段边界不是新 epoch。Python/Torch 用完整 seed，NumPy 明确 `%2**32`。私有图序 RNG 不与模型 RNG 共用；每臂自身 Python/NumPy/Torch CPU/CUDA RNG 独立保存、恢复。

## 连续状态与文件

每叶冷 CUDA 仅调用一次 U34 `build_pair`，得到 CPU PLAIN/FULL 同源初始化；104 个公共 state / adapter 检查与 original inactive 声明继续由其精确源码执行。续段这只是临时 CPU skeleton，随后严格载入各臂父 model；不会在载入训练状态后再次 common-copy 或 zero-head。

每臂 `_train_arm` 在 callee 内 `pop` 自己的模型，另一臂只留 CPU；同一段内每图不重建模型/BN/Adam/RNG。续段载入前后逐项核对：

- full model state（包括全部 BN buffers 和 inactive 参数）；module train/eval flags；effective/inactive 名单和排序。
- AdamW state_dict 的参数映射顺序、每参数 step/两个矩、完整 param_groups，以及 CPU step/CUDA moments 的实际设备合同；不能只恢复权重后新建空 Adam。
- 原初始化回执、source/recipe/context、原始 order ref、全局游标、已实现的请求序列链和实际输入身份链。
- 各臂完整 Python/NumPy/Torch CPU/CUDA RNG 在构造/加载后、下一图之前恢复，并立即逐值核验。

每臂实际写：`PLAIN.trace.jsonl` / `FULL.trace.jsonl` 和 `PLAIN.pth` / `FULL.pth`。trace 每图记录实际 image/index/GT-query 身份、完整原 image ref、loss、更新数、调用计数、offload 事件计数和资源采样。双方比较的是实际 reader 返回身份链，另保留请求顺序链，不能把计划当 actual。

checkpoint 包含上述完整 CPU state、父 checkpoint/pair refs 和身份。只为当前一臂生成 CPU snapshot，保存 `.pth.partial` 后释放该 snapshot，再以 CPU load 检验全树签名，成功才独占命名为 `.pth`；不同时保留双臂完整 CPU snapshot。文件字节 SHA 与 CPU tensor/state 签名分别记录，不假称 torch.save 字节可跨次复现。

两臂都完成相同 `[start,stop)`、保存成功、实际输入链相同、来源复核并清零 CUDA 后，重核两份 checkpoint/trace 字节，才独占写 `PAIR_COMMIT.json`，状态为 `PAIRED_CHECKPOINTS_WRITTEN_PENDING_ROOT_ACCEPTANCE`。report 返回 `SEGMENT_COMPLETED_PENDING_ROOT_ACCEPTANCE`；`paired_prefix_advanced=False` 有意保留，真正接收/推进属 Root。此文件不是科学资格自证。

第二臂失败时保留第一臂已写 checkpoint 和真实计数，但没有配对提交，后继不能使用单边新游标。失败不重跑，不自动回到旧前缀重复更新，也不抹去已发生的更新；任何恢复处置须另由 Root 明确决定。`optimizer_steps_entered`、`successful_optimizer_updates`、每臂已返回 step 和仍 active 的失败阶段分别保留。checkpoint/序列化错误不改称“训练未发生”。

## 生命周期与资源边界

原 4 GiB allocator / whole GPU 8 GiB / RSS 16 GiB / free RAM 16 GiB / CPU01 两线程 / 单叶 1800s 不变。核心在任何模型上卡前设 allocator cap，逐图读取 U39 实测采样；whole-GPU 连续采样、RSS/RAM 连续监督及硬 1800s owned process 边界仍必须由未来 Root wrapper 实现，不能由一个 Python `c.check` 宣称已硬保证。

每图 `_update` 释放 image/query/graph/grad 局部，保留真实 Adam/BN；不在每图 empty_cache。每臂所有模型/optimizer/param alias 和 hooks 在 callee 中释放，清 cublas workspace、empty_cache、同步后要求 allocated/reserved 双 0；函数最终再次核双 0。CPU/pinned allocator 的缓存高水位仍需实际连续段观察，累计 restore bytes 不是瞬时 RSS，也不是泄漏证明。

不默认一对完整 741 图可入 1800s。segment 大小由 Root 根据已准入 U39 实际时长和后续 checkpoint I/O 余量预锁；256 不是本实现常量。总 37,050 更新不变。实际文件 bytes 在每段记录；尚未实测 paired checkpoint 体积，不能把约 1.4GB 估计写成事实。必须在选段前核总保留量/目标盘余量；冻结原件不由本模块删除。权重仅科学输出，不能同步公开仓库。

## 本轮验证与后续缺口

`test_metadata.py` 先实际 MissingModule RED，再实现 GREEN，3 项 stdlib 测试覆盖私有 RNG 不污染全局、全 50 轮人口/预算、跨 epoch 顺序变化、游标/分段前缀，以及错误 seed/预算/重复身份/摘要/边界拒绝。没有 fake Torch，没有模型状态恢复或数值重演。

未来仍需 Root 审阅并实现 source/parent/输出准入的 entry/binder 和唯一 owned supervisor、实际预锁 segment 大小及完整 order SHA、checkpoint 可用磁盘规划。首个实际正式段内才能验证真实加载/序列化、第二叶才能验证真实断点恢复与持续资源；不另加资格叶，不声称与未分段训练已实证 bitexact。DEV 固定端点评价由原主计划另行接线，训练核心不读 DEV/TEST/AP。
