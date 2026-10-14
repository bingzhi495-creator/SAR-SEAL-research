# N17独立CPU首次执行失败与最小修复

2026-10-15。原监督30221真实退出1，held1/reaped/原birth消失；144.547秒，无error/abort/资源观察错误，GPU不可见、CPU01两线程。Root先核127正常+21失败纯原RAW（9ecab2），再读取原stderr和核原Source/spec/ad/phase/bootstrap；REPORT与结果目录不存在。原件保持冻结，不消费失败产物。

## 已确认原因

CPU02配置的CPU_adapter_source指向冻结解码器，但Root构造器没有把该strict3ref注册到sources。程序source_gate在实际数值前向及输出前报“Missing exact complete SOURCE reference”。这是任务构造错误，不是模型未涨点、收敛不足或方法失效；当前仍无新AP。

## 修复与证据

已补齐解码器和全部六项CPU必需来源的登记。新增[启动前结构检查](code/source_reference_closure.mjs)递归核配置中strict3ref是否在sources精确出现，拒绝遗漏、冲突和重复；它不读取文件、不计算SHA、不替代原科学资格。新检查对原实际失败spec准确拒绝缺失解码器，针对遗漏顶层/嵌套引用、冲突、重复和有效清单的[合成回归](code/test_source_reference_closure.mjs)实际退出0（a2b38f）。

CPU03已冻结，发布副本一项fake metadata回归实际退出0（95155c/test0.084秒），一次新增前门静态审阅READY；修正版首CPU任务已启动（原监督45412，owner127，[0,47)）。final03/post02已冻结，发布副本一项合并接口fixture实际退出0（fd027c/test0.042秒），一次接线静态审阅READY，尚未实际运行。版本变更仅为保留第21次失败、升级当前计数及闭合新执行来源的接口适配；旧GPU03/04、CPU02/final02/post01及失败原件不改。训练、初始化171、24轮预算、真实八邻与中心复制、固定PR/C/K/分数、数值容差1e-4、完整187图COCO及主E24-atmost/重复停止规则保持。

## 当前状态

五个GPU分区187图/18700query/10策略捕获与导出metadata均已核；首CPU没有产生独立数值报告。当前正常127/保留失败21/148实际终态身份，当前只有修正版首CPU原45412运行，无自动重试。不发布RAW、运维配置、数据、权重或载荷；可核验记录由Root保存在本地。


[统一全量COCO03](code/N17_FIXEDDEV_FULL187_CPU_COCO_FINAL_ROOT_UNBOUND_DRAFT_20261015_03.py)及[独立后置ROOT02](code/N17_FIXEDDEV_FULL187_POST_ACTUAL_ROOT_CPU_UNBOUND_DRAFT_20261015_02.py)仅适配新numeric-v3/current21和冻结final引用，30个typed前门与CPU03一致，COCO/主解释/合并算法保留。不以fixture或静态READY宣称真实数值、全量AP或创新成立。
