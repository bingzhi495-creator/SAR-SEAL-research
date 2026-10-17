# Full / Plain 公共层同源初始化草稿

仅为普通、未绑定的 B1 初始化准备。当前没有导入或执行 Torch/NumPy、构造模型、读取权重/图像/GT；没有访问运行中的 Ops/H/SCI/资格链，也没有修改 U33。B0 尚不能由这份草稿追认为通过，正式训练仍须 Root 独立批准。

本次只读公开副本 `P/research_round_20260925_global/code/U33_REAL_IMAGE_400_PROBE_DRAFT_20261017_01/`，其中 `P=E:/SAR/seal/github_release_20261011`。`model_full.py` SHA 为 `c210b318be29aca7d26516e9f47c8849e2c6a76642d4049fe7b133726ee809b9`；`plain_model1.py` 为 `902e13e5c60b1579459854c6bf4393ad8f268092a73a1d851f0fa640c810f48d`。这是固定宽度 64/128/256/512/1024 的原 model1，不是历史 clean32 或 SMP 同名模型。

## 配对范围和边界

两原文件的 `ContinusParalleConv` 整个类，以及下列 16 个根模块的构造表达式，可逐项做标准库 AST 比较；代码要求完全一致，不使用模糊名称或尺寸搜索。

| 公共根 | 采用理由 | 精确叶子状态规则 |
| --- | --- | --- |
| `stage_0`、`stage_2` | 同名、同分辨率转换中的普通双卷积，声明及原局部计算相同 | `Conv_forward.0/.3` 为 Conv 的 weight/bias；`.1/.4` 为 BN 的 weight/bias/running_mean/running_var/num_batches_tracked |
| `CONV0_1` 至 `CONV0_4` | 同一最高分辨率嵌套节点的普通 pre-BN 双卷积 | `Conv_forward.0/.3` 为上述五项 BN 状态；`.2/.5` 为 Conv weight/bias |
| 十个 `upsample_*` | 同名嵌套图边；相同 ConvTranspose2d，kernel4/stride2/padding1 | weight 为 `[in_channels,out_channels,4,4]`，bias 为输出通道 |

由原 Sequential 源码导出的期望是 **68 个参数张量、36 个 BN buffer，共 104 个 state 条目**；不是完整模型参数数量。未来还必须核实际名称、shape、dtype、参数或 buffer 类型、完整子模块树和 Conv/BN/转置卷积属性，再原位复制。当前未进行实际张量验证。

`stage_2` 在 FULL 中是方向融合前的普通支路，在 PLAIN 中是普通编码路径；共享其局部算子初值，并不声称其全网角色、输入分布或激活一致。`stage_0`、上采样和最高层解码的输入也会因其余结构而不同。这种初始化只排除已匹配公共算子的随机初值差异，不能把整体差值归因于一个模块。

**明确不映射**：FULL 的 CBAM/SimAM/DSConv/GN/融合层到任何同尺寸 PLAIN 层；FULL 中 inactive 的 `CONV1_* / CONV2_* / CONV3_1` 到 PLAIN 的有效低层节点；四个旧语义头；六个 discarded `bn1` 参数及其 buffers；DSConv 未选方向。所有这些保持各自 seed202610173101 构造所得的状态、原模块与 inactive 边界。四个旧头都不参与新实例特征损失，第四头的运算也不同，故不以其偶合名称充数。

## 未来接口

`shared_initialization.build_pair(torch, modules, source_refs)` 接受已由 Root 闭合加载的七个组件：`full_legacy/full_dsconv/full_head/full_feature/plain_legacy/plain_feature/instance_adapter`。函数核模块路径与固定源码 SHA、原依赖对象、操作 AST；随后才在独立 `fork_rng(devices=[])` 内各以 CPU seed202610173101 建原 PLAIN/FULL。FULL 固定400并显式禁用、不分配两种旧 Transformer。

以 **PLAIN 的原始 seed202610173101 公共状态**为统一来源，逐条 `copy_` 到 FULL，覆盖对应权重、偏置和 BN buffers；不替换 Parameter/Buffer 对象，不修架构或 BN。未映射状态逐条摘要验证不变，模块身份/模式/requires_grad 不变，inactive 名单保持原 wrapper 的定义。然后再分别重置同一 CPU seed202610173101 建 64→32 的公共实例适配器，验证实际小头所有初始状态一致、末层为零。

返回两 CPU 适配器与实际初始化回执，不保存 checkpoint、不运行前向、不建 optimizer、不生成训练 runner。两个源模型不同构造顺序会消耗不同数量的随机数，因此“分别设置同一个 seed”本身不足以确保公共层相同；这里的显式映射正是为消除这一点。构造与复制的实际张量检查、B0 资源资格及后续公平训练尚未执行。
