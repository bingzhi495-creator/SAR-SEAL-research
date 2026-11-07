# E50最后186次更新：普通源码静态合同已核清

2026-11-07。独立代理和Root只读独立发布副本，不读取运行SOURCE或资格链。普通training_body.py的43423字节及SHA与普通entry.py中原BODY pin精确相符；validate_span的FORMAL分支只要求整数0≤start<stop≤37050，没有256或整除约束。因此36864→37050/186在该静态合同上允许，不需要修改数值训练算法或放宽验证。

训练循环range(start,stop)、两臂完成计数及预算均用实际stop−start，最后每臂186、两臂372。这里没有执行训练模块或重跑测试，没有建立真实最后父状态、实际prepare、冷恢复或资源资格。

Root仍须在接收36864前选用已有final186门控：旧入口上限36608会拒绝36864的接收；新候选允许普通256直到36864，仅最后一步186，next取min(37050)，真正37050后不再发训练。末段状态格式器也使用已有03候选，不拿普通256格式器冒充最终完成。最后真实prepare必须沿合格36864父状态和原来源闭合执行，不能提前制造父引用。

原tool、来源、状态恢复、进程后验、Root数值接受及固定AP政策均不因本次静态检查放宽。最终达到37050仍只是训练预算终点；四片完整预测、零残差校准、原native/R完整COCO12复现及统一端点AP仍待。旧A/B阶段已闭合，不因本次恢复旧实验。

普通代码入口：[training_body.py](code/U57_DGX_ORIGINAL_UPDATE_TRAINING_DRAFT_20261018_01/training_body.py)、[entry.py](code/U57_DGX_ORIGINAL_UPDATE_TRAINING_DRAFT_20261018_01/entry.py)。私有门控/运维不随本文发布；本页是静态范围说明，不是运行资格。
