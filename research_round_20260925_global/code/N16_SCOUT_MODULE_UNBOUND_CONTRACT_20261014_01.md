# UNBOUND_DRAFT — N16 scout module contract

未读取实际SCI/State/Source/Spec/Q/RAW、数据、权重或运行缓存；未Git、未科学训练。只在seal_gpu的随机CPU fixtures上测试，CUDA_VISIBLE_DEVICES空、Torch2.0.1+cu118、2threads且CUDA始终cold。此前公开方案的采样/监督范围仍由ROOT正式锁定；本模块只实现网络，不产生实验资格或创新结论。

ABI：make_scout(torch,seed)返回CPU FP32 nn.Module；forward(F,e,L)接dense同device FP32 [N,256]/[N,256]/[N]，输出finite FP32[N,3]。N0合法。F/e为非affine LN256 eps1e-5，L原值/8不clip；Linear513→32、ReLU、Linear32→3。四named tensors：hidden.weight(32,513)、hidden.bias(32)、output.weight(3,32)、output.bias(3)，16547 scalars，无buffers/dropout、正常Linear随机初态，不zero输出。ctor仅private CPU seed（int非bool、0≤seed<2**63），恢复global CPU RNG，不seed/query/init CUDA。正常forward可由ROOT移至已准入GPU；所有输入/参数及输出检查finite/shape/dtype/device，非法输入抛ValueError。checkpoint保存state_dict而非局部class对象。

independent_logits(torch,state,F,e,L)为CPU资格公式：显式mean/centered-square population variance/rsqrt实现LN，再matmul+bias、clamp_min0、matmul+bias；不调用nn.LayerNorm/nn.Linear。它与NN在实数算术等价，不宣称FP32逐bit或认证误差界。当前随机/常数/rawL超8/N0 fixture互核使用atol1e-6/rtol1e-5，只是synthetic测试标准，不是未来科学/排序资格协议。logits保持FP32；utility/error类解释与FP64 score/排序由独立helper/ROOT预锁，不以logit allclose代排名exact。

四项关键测试均经历RED→GREEN并最终通过：同seed/state/RNG/4 tensors；18个random fixtures、两不同3class target的8步真实AdamW梯度/参数/loss变化；NaN/Inf、形状、dtype和meta-device拒绝；NN与explicit公式随机/常数/空输入互核。随机fixture损失下降仅证明代码可更新，不证明真实数据可学。全部源/测试仍是ordinary UNBOUND，ROOT后续负责Source/admission与真实资格。
