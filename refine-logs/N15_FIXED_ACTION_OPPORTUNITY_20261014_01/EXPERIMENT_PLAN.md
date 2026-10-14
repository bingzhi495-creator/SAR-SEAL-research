# N15 固定行动机会诊断预登记

锁定UTC：2026-10-14T04:31:14.389Z。状态：预登记完成，真实N15尚未执行。配方文件 LOCKED_PROTOCOL_20261014T043114Z.json。该文件冻结，不因结果修改。

主问题：固定701 LOG E24精修器在较宽的无GT候选池中，相同K预算下能否发现额外的真实有益行动？先验来源是当前有限编辑域与实际残差行动，不以“更多错误”自动证明可修复。

数据为已复用内部DEV187图、380GT，原生.50固定匹配374与未匹配6。主统计仅374匹配条件群体；6例明确不可评估，不补零，不编造query。原自然100 query、S0和导出顺序固定。当前不是官方测试或跨集。

## 主张与反例

C1：C中存在旧Omega以外的实际有益修补，须同一固定PR动作、相同K、相同原生mask比较。反例是仅有理想GT纠正空间，固定PR不能修；或收益只有旧域拒绝有害操作；或仅极少数大船获益。目标只决定下一研究是否值得做，不是论文效用主张。

## 最小实验

1. 原生mask P=(GPU L>0)，Omega=(abs(GPU L)<=2)，K=|Omega|；C=Omega∪P∪Chebyshev8像素膨胀(P)，边界截断，空P时C=Omega，K0不编辑。只使用推理输入生成C。
2. 冻结强701 E24原PR40497参数，分别原Omega-only和完整C升序2048点执行。前者exact重建既有完整LOG结果；两路径重叠residual/combined容差1e-4且所有finalsign一致，否则数值资格失败，不据此否定假设。GO的Omega和C取同一C输出。
3. 精确记录a=FP→TN,b=FN→TP,c=TP→FN,d=TN→FP,z=真实noop。atmostK优先b,a；exactK优先b,a,z,d,c，没有虚拟noop。GT替换上界保持相同C和K，但它不是固定头实际行动。
4. 同实际GPU采样F/e/L回放两个独立CPU公式，严格核残差/符号/计数/oracle。Root实际终止后资格确认才读研究结果。

## 决策门

主Δ为atmostK行动oracle(C)−atmostK行动oracle(Omega)，不把拒绝有害行动冒充宽池效用。三条件同时成立才GO：374均值Δ>=1/200；至少20例Δ>=1/200；IoU>=9/10实例数增加至少4例。以有理数精确比较。另报Omega oracle−实际Omega、GT上界、exactK的强制有害动作及.95次指标和全分布，不事后改主门。

不达门仅停止此E24固定头/radius8配方，不泛化到所有采样或精修方法。GO也仅允许一条公平selector假设的下一登记，必须按原至少+1AP实用门、重复种子、机制和跨集再验证。

## 成本和停止

科学执行只有ROOT；GPU与CPU每步最多1800秒，CPU01/2线程，allocator4096MiB、全GPU观察8192MiB、RAM16GiB，不自动重试。候选C全量PR与额外Omega核对调用实际计费，不因输出K称总计算相等。捕获仅用于同输入CPU资格，不重渲染CPU F。没有训练、AP或TEST执行。

冻结前先例与范围：PointRend/MagNet/MaskTransfiner已有选择/细化，PromptMoE和SAM-Veteran已有实际质量或IoU收益监督；“效用预测”原语不声称首次。GLDet提供空间邻域和整网结合先例，当前任务没有导入其GIM或Mamba。下一步即便学到像素收益也须排除常规错误预测+已知实际翻转的解释。
