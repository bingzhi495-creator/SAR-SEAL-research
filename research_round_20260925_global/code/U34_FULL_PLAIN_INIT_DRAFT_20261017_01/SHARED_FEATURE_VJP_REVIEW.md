# 共享特征单次 VJP：普通工程候选静态评议

结论：在下面的明确条件下，先累计共享特征梯度、再对编码器做一次向量–雅可比乘积（VJP），与现有分块损失对同一前向图逐次反传具有相同的一阶实数梯度。它只调整反向计算组织，不提供新的模型信息或监督；FP32 不保证逐位一致，速度与峰值内存收益尚未实测。本说明没有导入 Torch/NumPy、运行模型、读图像/GT/权重，也没有改变 U33/U35 或生成运行器。U35 仍只改变训练 query chunk；如后续需要检验此候选，应由 Root 另立明确版本。

## 对现有接口的准确分界

只读核对公开 `P/research_round_20260925_global/code/U33_REAL_IMAGE_400_PROBE_DRAFT_20261017_01/full_instance_adapter.py`，其中 `P=E:/SAR/seal/github_release_20261011`；SHA256 为 `9ad221170f69c028474f76700c991cb76dfecf19b11f5198767843b8e378d8b3`。

其 `encode(image)` 调用主干一次，通过原 `legacy.CONV0_4` 临时 hook 捕获并 clone 特征，再执行 `feature_projection`，返回 `[1,32,H,W]`，图仍完整。`predict(encoded,e_q,M200)` 只做 query projection、原 coarse-logit 插值及 mask projection、加法、ReLU 和末层卷积；没有头部 BN/dropout，也不再次调用主干。这正好提供可分割接口：

- 编码参数 θ：有效主干参数及 **feature_projection**。
- 读出参数 φ：query_projection、mask_projection、head。
- 原 inactive 参数保持原名单，梯度仍应为 None；不能为方便统计而纳入优化器。

实现前应按 Parameter 对象身份核两集合无交集、优化器无重复引用，并排除不同 Parameter 对象共享可写存储的别名。不能用参数名称或 shape 代替身份判断。数学上共享参数可另作完整链式求导，但本最小候选不默默处理这种情况。

## 一阶等价的条件与推导

令 `u=E_θ(x)`（已经包含 feature_projection），第 k 个查询块损失为 ℓ_k，块大小 n_k，总有效监督查询数 G：

\[
\mathcal L=\sum_k (n_k/G)\,\ell_k(H_\phi(u,e_k,M_k),y_k),\qquad
g_u=\sum_k(n_k/G)\,\partial\ell_k/\partial u.
\]

现有逐查询 BCE 与 smooth=1 的 Dice 先按每个查询计算，再取块均值；故 `n_k/G` 恢复原全 G 查询均值，包括最后不足整块的情况。固定 native logits 加法及原几何渲染仍在每个 ℓ_k 的计算图中。

在编码参数影响损失的路径全部经过 u、两参数集合分离、参数在整个更新期间不变时，

\[
\nabla_\theta\mathcal L=J_{E,\theta}^{\mathsf T}g_u,
\qquad
\nabla_\phi\mathcal L=\sum_k(n_k/G)\,\partial\ell_k/\partial\phi.
\]

因此可让每块头图消费一个共享的、值与 u 相同的独立叶子；各块 backward 累加该叶子的 `.grad` 与 φ 的 `.grad`，释放各块头图；最后把累计的 `g_u` 传给原 **U_orig.backward(g_u)**。编码器只做一次 VJP。`U_orig` 本身不能 detach，feature_projection 不能在这个分界之外丢失连接。仅把头所消费的特征换为叶子，不 detach 头参数、预测、渲染或损失，也不更换 target/assignment。当前原 GT、native logits、e/M 是固定自有输入，本来不要求梯度；这不等于允许切断未来可学习教师/输入的路径。

这是普通一阶反传等价，未覆盖 create_graph、高阶导数、共享可学习 e/M 的上游图、编码器辅助损失或依赖块次序的梯度变换。若后来加入这些路径，不能直接沿用本结论。每块梯度 hook 若执行非线性变换或裁剪，同样会破坏上述线性累加条件；单纯记录 hook 次数也会观察到预期的反传调用次数变化。

## 必须保持的执行语义

每次完整更新只 zero_grad 一次；叶子 `.grad` 在各块间累加，不逐块清空。整个 G 完成并做最后 VJP 后，才统一检查梯度、按原合同裁剪并执行一次 AdamW；中间不得更新、weight decay 或修改任一参数。每次优化器更新须重新 encode 并新建叶子，不能跨两步复用旧特征图。

`U_orig` 的完整前向图必须保留到最后 VJP；叶子 detach 通常共享数值存储，不应原位修改任一视图。日志仅保存 detached 标量或摘要，不为了记录 loss 保留头图。保持原 chunk 顺序、mode、输入、RNG 和几何渲染；主干不做 checkpoint 重算，也不重复编码，从而不额外更新 BN buffers。FULL 原有六个 discarded bn1 在 train 前向中的 buffer 更新仍按原行为发生，不能声称“全部 BN 不变”。

## 资源预期与最小验证

可减少的是每个 chunk 对共享编码图的重复 VJP，以及头图与编码器反向工作区的同时驻留。主干前向所保存激活仍在，不能据此保证解决 400 分辨率 OOM。`[1,32,400,400]` FP32 叶子梯度本身约 19.53 MiB；如不必要地 clone 还会增加存储。旧 `retain_graph=True` 也不意味着所有历史头图必然永久驻留：未被引用的块图可被回收。因此不能在未测量前声称节省某一显存比例。

最小有意义的后续合同是对实际 PLAIN/FULL 小图与实际 adapter，分别从相同完整参数/buffer/RNG 状态运行旧反传和叶子累计反传，采用完全相同 chunk 划分、非整除末块、固定监督与原渲染。比较总 loss、累计特征梯度、**每个有效参数**的梯度 None 状态/有限性/最大绝对差/范数相对差，同时保持 inactive 全 None。需要覆盖零末层的第一步及 AdamW 已有状态后的第二步，并有非零读出状态保证编码器梯度比较不是全零空验证；比较两步参数与 AdamW moments/step，核一次主干前向、旧四 head 零调用及预期 BN buffer 行为。FP32 容差须在实际对比前固定并逐张量报告，不能用“全局范数小”掩盖个别失败，也不要求 bitexact。

通过上述正确性对照后，才有理由在单独版本的 400 资源探针中测同步时间、每阶段 CUDA 峰值与 RSS；仍保留原资源门和无 OOM 重试规则。这些验证当前均未执行，也不等于正式训练、收敛或 mask AP 效用证明。
