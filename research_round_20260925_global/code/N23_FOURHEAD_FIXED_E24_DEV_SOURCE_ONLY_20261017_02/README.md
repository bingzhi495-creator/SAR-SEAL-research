# N23 固定E24四头DEV：源码、合成检查与待执行接口

状态：执行前形成；核心坐标/掩膜契约4项合成检查已通过，实际缓存前向、逐query强R校准和完整COCO尚未执行。独立静态审阅发现整片K0时清理别名未定义，已在新02源码中初始化；01未执行副本保持原样。

四头A_COMMON/A_ACTIVE/B_C/冻结强R共用原FP32 native L、全部自然query的Omega、K和原score。Omega升序，以2048点及真实尾块单独前向canonical PR头，再做原设备FP32 L+u和严格>0掩膜。沿原自然query导出顺序，不按GT或score过滤。每图原native掩膜/RLE及重跑R的完整预测字典必须逐query精确重建；不符即停止诊断，不降低参考线。四个物理GPU分片只导出预测，全部187图/18700query合并后统一官方COCO；不平均局部AP。

`run_eval.py`是资源和绑定入口，`dev_body.py`执行GPU导出或CPU整集COCO；`N23_FOURHEAD_DEV_CORE_UNBOUND_20261017_01.py`限制原坐标Omega替换。`resource_control.py`是已使用资源控制模块的原字节副本；仅复用Control，不调用其中训练入口。`point_heads.py`为相同8张量/40497参数的既有PR factory，非新结构。

在本目录运行`python -B -m unittest discover -p 'test_*.py' -v`可检查非恒等union映射、域外不变、零阈值、缺F拒绝、K不符拒绝和K0。这是NumPy合成检查，不替代真实缓存、CUDA或AP资格。

实际入口需要已核缓存、固定端点、官方标注和私有绑定配置；未发布这些载荷或权重，完整一键复现尚未完成。已有共享F覆盖Omega，因此无需教师forward或F9邻域。成本仅记录缓存后处理，不宣称端到端FPS。

[执行前固定评价方案](../../../refine-logs/N23_FIXED_E24_FOURHEAD_DEV_20261017_01/EXPERIMENT_PLAN.md)、[实际训练阶段](../../N23_ANCHORED_TRAINING_STAGE_PROGRESS_20261017.md)。当前没有新AP、核心创新或TGRS就绪结论。
