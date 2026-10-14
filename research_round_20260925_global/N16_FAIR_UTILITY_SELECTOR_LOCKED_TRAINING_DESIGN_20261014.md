# N16：固定强E24后的唯一公平选择器实验

状态：协议于 UTC 2026-10-14T06:02:44.021Z 锁定；真实新缓存已启动（当前单次修正版执行）；两个scout尚未训练。N15机会诊断已合格 GO，但它只说明374固定匹配域内的GT理想选择余量，没有新AP或可学性证明。

## 核查后明确的监督边界

Root核验已闭合原生Mask2Former实际目标源码：负查询分类label为background，mask_weights为0，mask_targets只保留正匹配。此前草案“原规则直接产生空mask”的表达不成立。本轮显式增加双方共同的 matcher-background→emptyset 代理监督，不冒充原始mask目标；original sampler negative、background分类和mask_weight0须同时成立，ignore不补空目标。保留正query→真实实例的原十阶段匹配及原私有随机种子。此扩展可能抑制重复但可在COCO中成为真阳性的query；记录背景对任一GT的最大IoU及损害，结果不佳则否定当前代理配方，不事后改变关联来救结果。没有新增人工标注。

## 相同学习条件与单一假设

固定701 LOG PR E24。C=Ω∪P∪Chebyshev半径8膨胀(P)，Ω=|原GPU L|≤2，P=L>0，K=|Ω|。训练/部署均保留自然全部100查询；部署候选、query及排序不读GT或改原S0/类别。唯一学习比较为16547参数的两个513→32→3 scout，F256/e256非仿射LN，原L/8不截断，ReLU；同正常初始化、同点及次序。utility预测help/harm/noop；error预测FP/FN/correct。动作标签来自同冻结PR真实GPU残差加原L后的strict0翻转。

每张FIT图至多4096点：matched与background活跃组各2048；只存在一组则4096。组内均匀活跃query，再均匀它的真实C点，允许重复；空C/K0不采。私有Torch CPU Generator以202610140160000+图序号初始化，matched先，单次query-vector随机抽样后逐draw点抽样。目标分布为**逐图均匀活跃query→均匀其C点**，不是全C像素均权；每样本权重=(组活跃query数/全活跃query数)/提议组概率。loss=mean(w×CE)，不再除batch权重和、不用额外类别权重，不由DEV调整抽样或概率。披露FIT支持量、加权三类数及范围，不以整体accuracy或此前97.12%noop外推all100可学性。

两arm新训练24轮、seed202610140161、同私有图序；AdamW lr3e-4/wd1e-4、betas(.9,.999)/eps1e-8、恒定学习率、FP32。每缓存图各一次更新，空图双方同跳；实际更新/曝光记数，不继承旧1661步/轮。E8只报预锁曲线，E24固定主端点；这不是旧PR的连续追加训练，也不是不同结构按同轮数自动同计算量。

## 评价和可否证门

canonical GPU scout logits为FP32，转CPU后统一稳定NumPy FP64 softmax/score。utility score=phelp−pharm；error=2(pFP+pFN)−1。主atmostK只选score>0，补充exactK只选真实C点；统一score降序、|L|升序、flat index升序，无rank容差。原S0/类别/query顺序不改。自然18700预测的正式COCO mask AP相对 max(同stage error AP,68.09938494745545) 至少+1点且大于配对error方能GO；非正停止当前配方，正而不足门仅按原goal最多一个追加seed规则。不能用中间epoch/exact赢家救主结论。后续仍需重复、针对性机制和跨公共集。

强对照保留原强E24、同训error、**全C先跑实际PR得到flip指示δ，再以δ×(2perror−1)排序，应用原固定残差**，以及适配二类MagNet稳定分数2σ(−|L|)|tanh((L+u)/2)|。δ是BOOL翻转指示，u是raw PR residual；不能混淆，也不改PR符号来偷换固定动作。两者全C PR成本单列，MagNet不是完整作者复现。行动效用有直接先例，不称新原语；参数匹配或固定K不等于成本匹配。

## 缓存与执行范围

全741 FIT图一次冻结教师loss forward，记录全100 target/valid/neg/ignore证明、实际GPU自然L、原matching/e/C一致性、样本F/e/L/u/combined/labels/weights、私有采样状态与更新可计数元数据。完整L科学核查载体与薄训练点文件分开，训练不每轮读大L。按实际尺寸ΣN=119780015、maxN=306257，L raw总47912006000B；raw已知总界约51322624484B，另计samplerstate/proof；每packet≤512MiB、总序列化≤56GiB，当前预锁是上限不是实测成本。CPU01/2threads、allocator4GiB/整GPU观察8GiB、单叶1800s且不自动重试。Root实际terminal/RAW先核验后才读/hash新科学载荷，独立CPU用同GPU sampledF/e/L回放原PR两公式，abs1e-4/rtol0、最终sign exact、选择indices exact。

此前CPU/GPU渲染不一致失败完整保留，不由此放宽标准。全L只是科学资格载体，本仓库不上传原GT/缓存/权重/SPEC/Q或运维配置。

## GLDet的位置

[GLDet原论文综述](GLDET_PRIMARY_PAPER_REVIEW_AND_SEGMENTATION_TRANSFER_20261014.md)已核全文方法和mask表格，属于真实TGRS实例分割先例。原文SSDD/HRSID mask AP71.3/58.6、整网500轮，与本项目内部DEV配方不直接判胜负；局部RepVGG邻域/方向pool需要真实空间载体，本轮点scout没有实现它。保留其局部空间信息作为下一候选，当前不同时叠加多个未知模块。

参考：[N15实际合格机会](N15_FIXED_E24_ACTION_OPPORTUNITY_QUALIFIED_RESULT_AND_NEXT_STEP_20261014.md)、[直接原始先例查新](N15_ACTION_UTILITY_TARGETED_PRIMARY_PRIOR_CHECK_20261014.md)、[旧UNBOUND设计及Root采样质疑](N15_SINGLE_FAIR_UTILITY_SELECTOR_MINIMAL_UNBOUND_DESIGN_20261014.md)。本文件纠正草案监督和dense分数歧义，保留旧记录提出时状态。

## 两项实现错误及原件保留

前两次真实缓存任务均以actual1结束：第一次配置漏region_common_source，在模型加载前退出；第二次在首图目标分配时，adapter错把原native分类label_weights要求为FP32。原源码、配置、输出和终止身份均保留为第10/11次失败。原native label_weights由int64 gt_labels.new_ones产生，应原样INT64；mask_weights与训练采样importance仍FP32。已通过缺失入口及native类型两项RED→GREEN回归检查，并以新Source/独立新执行修正，不自动重试，不改GT、模型、采样、24轮预算、数值容差或AP门。这些错误不支持方法有效/无效结论。

当前真实缓存执行中；本仓库只更新既有普通发布副本，不在运行中读取/复制/hash科学输入或产物用于上传。此前混合query采样批次生成PR labels；部署全C升序和selected-only批次仍需actual共同输入abs1e-4及strictsign资格，不能由合成测试替代。独立CPU资格主体、配对训练及部署评价代码已形成并有合成CPU检查，仍属UNBOUND，尚无新AP。
