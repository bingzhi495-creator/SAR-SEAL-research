# N27：真匹配实例固定支持范围诊断源码

2026-10-17。状态：已冻结、静态审阅与纯标准库小例检查通过，真实数据尚未执行。本目录是源码发布副本，不包含执行资格或原数据。

对 FIT741图/1661个原真实 matcher 配对，以原 FP32 native_L 构造 P=L>0、Ω=abs(L)<=2，核对完整R及原GT身份、逐像素摘要、原K计数和N26 FP64目标。按原匹配保留所有实例，不改配对、不把负query当背景。

逐实例统计 native/R 的 TP/FP/FN 域内外数量，并将GT填入Ω内作为不可部署诊断，核对其条件最佳IoU=(GT面积−域外FN)/(GT面积+域外FP)。保留10个预锁阈值的完整覆盖及零错误、空mask、K0分组；这不是AP上界、学习能力或唯一因果证明。

- `run_support.py`：原单次CPU入口与资源约束，实际结束前只生成待接收产物。
- `support_body.py`：741/1661完整真实匹配诊断。
- `test_support_ceiling.py`：纯标准库穷举1至4像素非空GT、预测和支持的小例；既有3项检查共4340个情形通过，未以此冒充真实数据验证。
- `resource_runtime.py`：原有资源控制源码的精确副本，入口通过绑定的 runtime_source 路径加载Control；不执行该文件中的旧训练CLI。

[执行前计划](../../../refine-logs/N27_MATCHED_SUPPORT_CENSUS_20261017_01/EXPERIMENT_PLAN.md)。本轮无teacher/PR前向、训练、DEV、TEST、AP或权重。CPU0/1、线程2、空CUDA、单任务1800秒、RSS至多16GiB、可用RAM至少16GiB，无自动重试。真实任务结束后仍需原持有身份/actual0、新鲜RAW FIRST及Root接收，才能报告实测。

原packet/L载体、完整标注、AUDIT RLE、逐实例计数、RAW/SPEC/Q/准入和运维绑定均不发布。因此本目录不是可直接独立执行的完整数据镜像；数据获取说明见仓库入口。用户UNet++结构尚未在该诊断中训练。
