# FIT实例级修补机会：数据可用性与下一项缺口

2026-10-17。**本文件保留采集前的可用性核查。后续N22完整采集及独立CPU复算已经完成，见[最新实际结果](N22_FIT_FULLC_QUALIFIED_DIAGNOSTIC_RESULT_20261017.md)；尚无新模型训练或AP。** N21已经拒绝所锁范围内“普遍共享四角”的几何前提。下一步先区分路由未利用的动作机会与固定PR自身的能力限制，不把未知原因当成新瓶颈。

## 为什么不能直接再换路由模块

[N18](N18_QUALIFIED_ROUTING_CAPABILITY_RESULT_AND_NEXT_HYPOTHESIS_20261015.md)观察到漏修和误改并存，但374配对按原生IoU≥0.5构成，不覆盖低质量候选、6个未匹配GT或其余自然query。其像素比例不能说明全量AP主要受哪一环节限制。

[N16](N16_FULLDEV_QUALIFIED_FIXED24_RESULT_AND_STOP_20261015.md)已经同输入、同容量、同24轮比较Utility与Error，主Utility未胜Error及强R，原STOP保持。实际DENSE_ERROR_FLIP对照还使用了`delta=(L>0)!=(L+u>0)`，评分为`delta*(2*p_error-1)`；其主AP67.877995也未胜强R68.099385。因此，“让实际翻转可见”或“只看动作效用”不能改名重启并称为新机制。强R/Omega本身是否同样使用delta，本轮有限追踪未确认，不把未追到当成没有。

## 已有底座及缺少的输入

本轮只追已完成实际绑定缓存的明确writer/评估角色，不遍历完整SourceMap；没有载入张量、权重、图像或GT全文。清单、writer声明及首末文件存在不等于本轮重新核验了741份载荷的全部内容。

| 项目 | 已核代码和元数据证据 | 尚需处理 |
| --- | --- | --- |
| 全自然query的原生L | 清单登记741图、每图100 query的独立carrier；writer `run.py:506–515`另存完整native_L | 本轮仅查首末文件存在/大小，后续选定输入仍需按新采集契约验证 |
| 实例embedding与类别 | `cache_body.py:292–294`保存全部e和原始分类logits | 最终类别与S0需沿原导出规则恢复，不自定softmax、类别或排序 |
| C/K | `:249–253、297`保存每图100项计数；完整C曾临时构建 | 全C坐标未保存；按原C_engine从完整L重建并核计数 |
| PR动作与选点 | `:273–301`只保存至多4096个分层有放回采样点的u、L+u、P/A及标签 | 缺少完整C的实际PR after/sign和当前router的S；不能将4096点外推为全C |
| GT与实例关系 | `:128–149、296、303–305`保留matcher对应raw GT身份及转换证明 | 未保存完整GT mask。每query的max-IoU仅为最大值标量，不是assigned-GT IoU，不含argmax或完整query–GT矩阵 |

全C所需F也未完整保留，不能凭现有稀疏F缓存假装无需新取数。负query的空集属于训练代理，不能当作其余自然query均为真实背景。

## 下一项工作的范围

先另行锁定FIT子集和实例对应规则，补采完整C的实际PR动作、当前router的S及锁定评分，恢复真实GT mask关系；尽量复用原生L和类别底座，不扩大为整轮模型训练。之后才有条件比较现有router、预算内最优固定PR动作、同预算理想原错误翻转的实例IoU及固定0.50～0.95阈值跨越，未匹配/重复候选另列。

该比较仅是固定人口的诊断，不是AP上界或唯一因果分解。PR可用动作与router之间的差距、PR与理想翻转之间的差距，必须实际测出后再选建模方向。当前尚未注册新采集协议、补齐数据或证明任何新方法有效；[N20停止决定](N20_TWO_SEED_MAIN_RESULTS_20261017.md)与[N21结果](N21_NATIVE_SUPPORT_GEOMETRY_RESULT_20261017.md)保持。

## 2026-10-17 接口补查与纯计数实现

后续定点核到原region_common.gpu_regions第439–454行：LOG_PR强R选区为abs(L)<=float32(2)，K为该选区像素数，此选区不读取实际翻转delta。这补足上文当时尚未确认的事实，不改变已有DENSE_ERROR_FLIP对照。全C实际动作需取PR工厂未门控residual并按原FP32运算计算L+u；工厂logits只在Omega内修改，不能误当全C动作。旧wrapper在K=0时跳过全C计算，新采集需明确覆盖。

已形成[精确计数内核与可移植合成测试](code/N22_INSTANCE_ACTION_ORACLE_SOURCE_ONLY_20261017_01/README.md)，8项测试实际退出0，包含2750个小穷举场景，仅UNIT_FIXTURE。比较固定真实实例配对、相同C与at-most K下的现有路由、最优固定PR有益动作、理想原错误翻转；依赖静态、独立且坐标唯一的翻转假设，不是AP上界或因果分解。空并集返回NA，负query空集监督代理不冒充真实GT。

真实FIT全C采集协议尚未注册，未启动新采集或训练，没有新增真实机会统计、AP或创新结论。下一步先锁独立选图与原matcher真实GT对应规则，再采集。N20停止决定保持。
