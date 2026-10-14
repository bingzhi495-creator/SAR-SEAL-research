# N17 配对24轮：固定入口与实际任务启动

2026-10-15。真实全741 FIT八邻缓存已经原任务退出0并通过Root资格。新的LOCAL_ERROR/CENTER_ERROR配对24轮原任务现已退出0，原监督会话78143；Root已核实际结束、完整更新计数及端点文件绑定。独立CPU数值资格和新AP尚未完成。

两臂使用原合格FIT741图/1661实例/74100自然query及每图4096原样本，原标签、权重和顺序不变；LOCAL使用同query原网格3×3八邻clip[-8,8]/8，CENTER重复同中心值八次。保留旧513维输入及固定教师/PR，每臂521→32→3、16803参数、四张量。新增输入是待检验假设，不宣称SAR专属或新原语。

首初始化202610150171、图序种子202610140160001，预锁初始状态digest5b9f58a3dece9720f84b257d03ae6928a7129e6f364e218f6bd49475d5fb7aa7；两臂全状态相同且存储独立，原513先初始化后八列补零。AdamW lr3e-4、weight_decay1e-4、betas(0.9,0.999)、eps1e-8、固定24轮，无AMP/额外重采样/轮次选择。目标每臂17784更新、72843264点曝光，E8与E24完整保存model/optimizer/私有图序RNG/history。

GPU入口一次定向静态检查READY，最终源码SHA fbaff44bf3f8dfe54c0884647ca63732e8aaaed384b552e7a6c4f6d586cc8bf2。锁字段名接口问题在科学启动前修正，未修改预锁协议；普通发布副本两项小型合成测试原退出0（test0.079秒），仅验证锁字面ABI、真实合成NPY点序/严格引用和四参两臂接口，不替代真实训练或任务效果。

原RAW完整前缀120正常+17失败、137身份pin闭合后进入新Source；仅一个原任务，CPU01/两线程/GPU0，allocator4096MiB、整卡观察8192MiB、RAM16GiB、单任务1800秒、无自动重试。观察超时只等待78143原handle，不重启。运行中的Source、资格链、数据、权重和缓存不因同步而读取或发布，本页为独立普通摘要。

UPDATE_TRACE只保存17784 paired_image_update更新子流；CPU可核更新身份/图序/损失/history，不能宣称保存或复放了未落盘的before_image/endpoint事件。后续按[端点CPU接线方案](N17_PAIRED_ERROR_ENDPOINT_CPU_MINIMAL_WIRING_PLAN_UNBOUND_20261015_01.md)核两组E8/E24完整端点，再做固定187图/380GT/18700自然query的统一mask AP评价。正式测试、重复种子、机制及跨集均未完成。

主门保持[取数前锁定计划](N17_LOCAL_NATIVE_LOGIT_ERROR_SELECTOR_EXPERIMENT_PLAN_20261015.md)：固定E24 atmost，Delta=LOCAL_AP−max(CENTER_AP,68.09938494745545)。非正停止当前配方；正但不足1点最多补预定172，达到既定条件才补173；达到1点且配对优于CENTER才推进两个预定重复。单次结果不等于核心创新或TGRS水平。N16旧主门STOP保留，不通过新实验改写旧负结果。

[独立GPU入口](code/N17_LOCAL_CENTER_ERROR_PAIRED_TRAIN_GPU_ROOT_UNBOUND_DRAFT_20261015_01.py) · [入口合同](code/N17_LOCAL_CENTER_ERROR_PAIRED_TRAIN_GPU_ROOT_UNBOUND_DRAFT_20261015_01.md) · [可移植小型合成测试](code/test_N17_LOCAL_CENTER_ERROR_PAIRED_TRAIN_GPU_ENTRY_SYNTHETIC_ONLY_20261015_01.py) · [FIT缓存资格摘要](N17_FIT_NEIGHBOR_CACHE_PREPARATION_AND_STATUS_20261015.md)。


## 端点数值核验体已形成，尚未执行真实CPU资格

[独立CPU数值body](code/N17_PAIRED_ERROR_ENDPOINT_CPU_QUALIFIER_BODY_UNBOUND_DRAFT_20261015_01.py)及[合同](code/N17_PAIRED_ERROR_ENDPOINT_CPU_QUALIFIER_BODY_UNBOUND_DRAFT_20261015_01.md)已保存为UNBOUND草稿；[发布副本小型测试](code/test_N17_PAIRED_ERROR_ENDPOINT_CPU_QUALIFIER_SYNTHETIC_ONLY_20261015_01.py)一项原实际退出0（test0.382秒）。fixture的全量计数/图序元数据为显式合成，不是实际741训练结果。它核两CPU公式与完整snapshot/更新子流接口；真实CPU资格仍须原GPU任务结束、锁定Source/输入并实际运行。初始digest由外层核验；训练后的新8列及两臂不要求仍零、必非零或必分化。


## 原GPU120实际退出0，完整配对24轮元数据核验完成

原监督会话78143实际退出0，用时673.813秒；held0/reaped/原birth消失、abort/error为空且无观察错误，整卡观察峰值2652MiB、最小可用RAM31335907328B。纯原RAW121前缀（121正常+17失败、138身份pin）实际通过后，Root才首次读取/哈希新产物。

两臂各24轮、17784真实paired更新、72843264点曝光；完整17784条更新子流、24段history及图序、E8/E24完整snapshot和FIT端点文件哈希/绑定均核验。GPU START/REPORT中两臂实际初始digest及每张量摘要都吻合取数前的预锁初态，原教师/PR、合格邻域缓存、新锁和输入Source绑定保持。此为原实际结束及元数据证据；原pending REPORT不改写，独立CPU完整数值资格仍未执行。

GPU记录的固定FIT每图平均加权ERROR CE：E8 LOCAL0.0097649774/CENTER0.0062145345，E24 LOCAL0.0052634131/CENTER0.0023572680。数值独立复核尚待完成；LOCAL报告损失高于CENTER，不能按预期解释为有效，也不能由FIT损失判定部署AP或方法总体失效。必须继续固定端点CPU核验与完整DEV主评价。

[固定DEV最小接线方案](N17_FIXED_DEV_SELECTOR_MINIMAL_WIRING_PLAN_UNBOUND_20261015_01.md)仅依据普通源码确认缓存载体接口可能足够；实际复用资格尚未核验，不能由这份方案认定已复用或已完成DEV预测。下一项是CPU121端点资格，之后主E24 atmost统一全187图/18700自然query COCO；当前没有新AP、正效用、创新或TGRS结论。
