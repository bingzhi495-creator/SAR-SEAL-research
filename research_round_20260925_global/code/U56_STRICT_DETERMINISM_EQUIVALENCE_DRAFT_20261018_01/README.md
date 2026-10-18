# U56 严格确定性后端下的独立等价性诊断

**SOURCE-ONLY / UNEXECUTED。** 本轮只有三个微小标准库元数据测试、AST及普通源码差分；没有Torch/NumPy导入、GPU或张量计算、main/owner调用、实际文件/数据/权重/科学资格链读取或SSH。不能据此宣称原法可复现、candidate等价、速度收益或主训练可继续。

直接克隆已停止的普通U52 equivalence body/runtime。Root已报告旧U52原法BASE_A/BASE_B控制未bitexact、PLAIN在5次实际更新后停止、candidate未进入，且不是资源失败；本轮没有重读旧实际产物，也不改变其失败记录、源代码或判断门。U56是新的后端协议，不能覆盖旧尝试，也不能称作原U40路线的复现或跨环境续训。

## 唯一诊断变量：严格后端

原法与candidate共同使用新预锁的七项backend_policy：

```text
CUBLAS_WORKSPACE_CONFIG = ':4096:8'
deterministic_algorithms_enabled = True
deterministic_warn_only = False
cuda_matmul_allow_tf32 = False
cudnn_allow_tf32 = False
cudnn_benchmark = False
cudnn_deterministic = True
```

新Linux owner通过真实child环境，在首次导入Torch/CUDA之前传入workspace；不修改父进程环境或在父进程导入Torch。receipt记录 `CUBLAS_WORKSPACE_CONFIG_passed`，worker独立记录并要求 `CUBLAS_WORKSPACE_CONFIG_before_Torch_import` 恰为该值。

worker在首次CUDA之前保存调用者原 `are_deterministic_algorithms_enabled()` 与 `is_deterministic_algorithms_warn_only_enabled()`，调用 `torch.use_deterministic_algorithms(True,warn_only=False)`，并保留原禁TF32/cudnn设置。严格算法状态分别在body前、每臂结束和body后实际读取、记录和核验。worker的finally独立恢复原enabled与warnOnly两项，记录outside_determinism_before/after并要求完全相等；这一步不会因其它RNG清理失败而被跳过。原CUDA math flags和CPU/NumPy/Python/首次初始化后CUDA RNG的保存恢复仍保留。

严格模式没有支持的算子必须真实失败；不切warn_only、不关闭严格模式、不替换算子、不另设容差fallback。PyTorch2.10文档说明严格模式会使用可用的确定算法，否则抛出错误；该开关本身并不保证整个应用可复现。[官方API](https://docs.pytorch.org/docs/2.10/generated/torch.use_deterministic_algorithms.html)列出CUDA双线性插值反向可能报错的情况。因此是否由真实安装版本走确定性实现/分解、是否能够完成、是否仍出现baseline不一致，都须由本新任务实际报告，当前不预判结果。[warn-only状态API](https://docs.pytorch.org/docs/2.10/generated/torch.is_deterministic_algorithms_warn_only_enabled.html)用于精确保存与恢复，不能将只发警告冒称严格通过。

## 保留的比较、数值与预算

完整保留原common104初始化/相同adapter、同两图输入、原法warmup一步、BASE_A/BASE_B/CANDIDATE分别从同一真实warm模型/Adam/BN/RNG恢复的两步轨迹。两臂全完成仍为每臂7次、总14次实际disposable更新。query chunk1、完整原G、B1/RGB400、原FP32/BCE+softDice、save_on_cpu以及AdamW配方不变；candidate源码仍为原U52固定SHA，未修改或抄成新候选。

BASE_A/BASE_B仍要求全部有效/无效梯度、模型参数/buffer/BN、Adam step/两moment、参数组、loss和RNG逐位一致；控制失败仍停止，candidate未进入不能被解释为候选失败。candidate原六类rtol/atol不变，整数/RNG/groups仍exact。新后端不能替代这些实测检查，也不允许因控制失败而放宽门。

原资源限制保持：CPU0/1、两线程、CUDA0、allocator4GiB、RSS16GiB、可用RAM16GiB、deadline1800秒，无自动重试。新严格模式可能增加耗时或不被支持；资源失败和算子失败都如实保留，不能临时放宽限额。此任务没有性能基准、AP、创新、模型采用或正式训练预算，不修改U40 frontier。

## 接口与新产物

新spec schema `ROOT-U56-DGX-equivalence-spec-v1`；operation `GPU_U56_DGX_SHARED_ENCODER_EQUIVALENCE`。原13个spec字段、14个组件角色、三选定库、两私有U49包与独立scratch布局保持；源集合仍为这些精确依赖。body和entry/owner的新字节SHA应由Root后续真实冻结并闭合，不能借旧U52绑定直接启动。

body API保持 `run_equivalence(torch,np,c,*,context,source_refs,prelock,body_source_ref,report)`，新prelock schema `U56-shared-encoder-equivalence-prelock-v1`，除上述backend_policy与版本命名外所有原字段保持。body报告schema `U56-shared-encoder-equivalence-report-pending-v1`；私有warm/snapshot均新U56 schema、新owned scratch，不读旧失败权重。worker报告schema `U56-DGX-equivalence-worker-report-v1`；owner receipt为新U56命名。CLI仍为：

```text
<DGX-python> linux_owner.py --spec /absolute/new-spec.json --spec-sha SHA256
```

worker新增 `strict_backend_settings_before_body/after_body`、`outside_determinism_before/after` 和实际restoreflag。原worker十flags保留，额外 `caller_determinism_controls_restored` 必须True；新owner在原终态/回收/进程组/来源与14更新完成门之外核这些实际字段。控制或不支持算子失败保留partial body、stdout/stderr、worker与owner记录，不能自授Root资格或盲目重启。

## 本轮测试证据范围

两个worker/body标准库测试覆盖：新预锁拒绝workspace或warn-only漂移；调用者enabled和warnOnly两项均须按原状态还原，即使原调用者使用warning模式。一个owner测试覆盖实际child环境构造函数强制新workspace且不改传入环境。fixture只有小字典、字符串和布尔值，不使用fakeTorch，也不证明真正的API调用、CUDA、梯度或算子支持。

AST核对比较函数、warm恢复、optimizer、RNG与候选容差/预算字段保持；旧普通U52文件未改。实际科学执行仍待Root按新sources/预锁启动单一原任务并独立接收；本草稿没有数值等价或收益结果。
