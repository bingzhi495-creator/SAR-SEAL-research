# 正式训练1024步与真实CPU恢复检查（2026-10-21）

DGX 的 PLAIN/FULL 已各完成并核验接收 **1024/37050 次正式更新**（完整一轮741图后再283图）。本机仍各2816次，记录独立，不拼接、不当作重复种子。本轮另实际通过独立 SMOKE2 的两臂 CPU 恢复技术检查。**尚无最终 COCO AP、精度收益、创新成立或 TGRS 就绪结论。**

## 本轮正式训练

两臂均从已接收768端点恢复，新增256次更新，各访问556个匹配实例；结尾游标均为 epoch1/within283/global1024，输入轨迹一致。模型/BN、Adam、模块模式、四路RNG和原顺序完整接续，未改网络、配方、初始化或原数值更新。

原任务实际退出0；owner/worker原进程已退出、组为空，771项来源及输出/日志字节核验后由Root接收。owner耗时998.905秒，组RSS峰值13075329024字节，可用RAM最低59518529536字节，未记录资源违规；两臂最终CUDA allocated/reserved均零。GB10没有Windows whole-device显存读数，不能据此认证该平台8GiB合同、独占设备或速度优势。

## 独立SMOKE2真实CPU恢复

此前U58只通过替身测试，无法证明实际网络能恢复。本次由专用U60入口在真实DGX环境运行原 `shared_initialization.build_pair`，得到两个不同的原模型对象，再分别调用固定的U58 `restore_smoke_arm`，不接收外部伪造的构造回执。

本次闭合26个直接来源：14份原代码、四份已接收SMOKE2的JSON、两臂CP、库入口及U58/U60代码等；不递归读取741份FIT包、训练manifest或DEV/GT。完整CP中优化器与RNG容器为验证完整签名被CPU加载，**没有创建/恢复优化器，也没有恢复训练RNG**。

实际两臂检查均通过：完整CP与独立model签名核对、CPU FP32/有限性、完整名称/shape/模块树、strict model load、训练头/BN/失活状态保留、eval/frozen及外部RNG检查。构造和恢复全程CUDA未初始化，未做forward、训练、matching或评价，不生成新模型checkpoint。模型对象在检查结束后释放。

原SSH/owner/worker真实返回0，owner/worker原birth不存在、组为空、26项来源及日志/报告字节已核验并由Root单独接收。owner耗时11.556秒、组RSS峰值2071130112字节、可用RAM最低74399010816字节，无资源违规。固定环境为Linux/aarch64、Python3.12.14、Torch2.10.0+cu130、NumPy1.26.4、psutil6.1.1，CPU0/1及两线程；并非跨版本本机恢复。

**该结果仅支持实际SMOKE2的CPU恢复路径。** 它不给正式训练额度，不证明E50权重恢复、完整预测人口、真实推理、零残差RLE校准、AP、泛化或方法效用。两步技术状态不能作为中途模型选型工具。正式配对前沿仍是1024，而不是1026。

## 源码状态与余下缺口

[U60源码与监督入口](code/U60_DGX_SMOKE_RESTORE_ENTRY_DRAFT_20261020_01/README.md)成稿时有7项标准库/替身测试、原监督函数AST核对和独立静态审阅；本次实际运行另有上述外部终态证据。目录README中的UNEXECUTED是成稿时状态，**当前执行状态以本记录为准**。原U44/U57训练源码保持原样，不伪装成旧741图数据校验任务。

[U58恢复辅助代码](code/U58_DGX_FIXED50_RESTORE_DRAFT_20261018_01/README.md)的SMOKE入口现有上述实际验证，生产E50入口仍待最终权重验证。[U59完整候选包](code/U59_DGX_DEV_FULL_QUERY_PACKAGE_DRAFT_20261018_01/README.md)仍未做真实导出/序列化/传输/重载及全187图零残差校准；matched训练包不能替代完整100候选输入。旧缓存读取可能接触GT关联容器，预测包白名单不等于整个上游流程从未接触GT。

下一步保持原预算从1024继续，并推进全候选输入的最小真实技术检查；不得计算FORMAL中途AP或据其挑选模型。固定E50完成后才按原完整人口、原候选/分数和COCO规则评价，再依预锁效用规则决定重复种子、容量/计算控制、机制和跨集验证。旧失败和资源中断继续保留。本机接续须有新鲜且整对达标的资源观察；本轮未新增本机训练。

[结构化标量结果](results/U57_U60_ACTUAL_SCALARS_20261021.json) · [前序训练及评估缺口](U57_TRAINING_PROGRESS_AND_EVALUATION_READINESS_20261020.md) · [直接先例查新](U57_NEAREST_PRIOR_NOVELTY_CHECK_20261018.md)
