# 主E50端点评价资源衔接：新候选已补齐，真实运行仍待

2026-11-07。状态：**GPU资源闭合、独立CPU后验及CPU资源闭合的三个新私有metadata候选已完成；有限合成测试和独立静态复核完成，尚未部署或真实运行。** 主PLAIN/FULL固定E50训练不变；原U66/U67数值数学、旧启动/收集/接受器及其冻结来源保持。没有新的模型AP或训练额度。

旧普通接口有两处具体断点：U80 GPU启动后保留active reservation，而U79提交要求idle且只写预测账户，U77也不清资源状态；CPU U81复用同一保留机制，旧U77又仅支持GPU，不能把GPU后验拿来冒充CPU。本次新增独立Root资源衔接及CPU专用后验，避免未来真实端点卡在这些接口之间。

| 分支 | 本次补齐的候选 | 已完成的有限验证 | 尚未取得 |
|---|---|---|---|
| GPU预测后 | 原U80/U77实际材料→独立Root资源范围观察→只释放原reservation→原U79接受/提交 | 12项新SYNTHETIC metadata测试，含临时CLI/CAS、重放拒收、unknown/身份/跨host引用边界；独立只读复核未见剩余具体接口阻点 | 真实37050部署、原工具/进程事实、完整4片预测与原结果接收 |
| CPU总体评价后 | 独立COLLECT→外部真实终态→CPU ENVELOPE→资源范围观察及闭合 | 7项post与5项closure合成测试统一12/12；跨host字节运输与临时CLI/CAS检查；有限独立静态复核 | 真实原namespace采集/完整COCO、Root实际数值与AP接收 |

候选要求同一原request/spec/source/reservation/history、原tool/SSH/owner严格整数0、child实际wait、同boot与出生身份/整组消失及资源事实闭合。另需独立Root范围观察，不凭单个组消失就清全局状态。UNKNOWN、非0、仅returning或metadata自身0均不能升级原科学失败。仅变更被证据覆盖的资源字段，保留尝试历史、训练对象、最终接收与预测账户；资源释放不授AP、结果qualified、训练credit或自动重跑资格。

Root持续同步可能改变authority中的GitHub metadata。新候选以当次fresh状态和范围观察做字节比较/CAS，保留当前全部无关字段，不要求活文件SHA回退到最初reservation；合作锁仍以Root单写者串行协调为前提，不是跨所有writer的通用数据库事务。替换后错误或部分持久化必须先核事实，不盲重试。

这些测试全部使用明确SYNTHETIC metadata/临时文件，不跑模型、COCO或旧科学suite。独立审阅采用继承模型，只读新候选和必要普通接口，没有SSH/实际Ops/运行SOURCE/标注/预测/权重读取，不冒称规定后端或完整跨模型审计。运维实现、真实执行引用和配置留私有，本说明只发布接缝、有限证据和未执行边界。

输入准备另取得真实数据层进展：[两个context与GT的原187顺序、380实例和三套18700预测角色已核验](MAIN_E50_REFERENCE_ORDER_ROLE_BRIDGE_STATUS_20261107.md)。主登记R与N26比较R解析内容完全相同但原byte refs不同，必须各保原引用、不相互替换。该数据桥没有导入科学库、解码mask、重现COCO12或赋予E50模型资格；查询完整身份仍由原数值合同检验。

实际顺序仍为：真正37050训练接收和资源闭合→原GPU启动及4片真实预测→每片原收集/外部终态/本次资源闭合/原接受提交→原CPU总体绑定与保存SPEC消费→真实CPU评价/独立后验/资源闭合→Root原数值/AP接受。启动普通候选说明见[U80](code/U80_ROOT_U74_ORIGINAL_LAUNCH_DRAFT_20261021_02/README.md)、[U81](code/U81_ROOT_U71_SAVED_SPEC_LAUNCH_DRAFT_20261021_01/README.md)；旧日期说明中的未实现项是当时快照，本说明不追溯声称当时已经执行。

最后186的Root门控及末尾格式器已有独立未部署metadata候选；新增普通源码静态复核确认原validate_span允许最后186，普通副本与entry原pin相符；真实prepare及其来源/父状态核验仍须按原资格完成，不绕过或改写旧资格。训练达到37050也仅表示固定预算终点。真实状态恢复、冷CUDA/库ABI、完整预测、零残差RLE校准、原native/R全部COCO12复现和Root接受依然待完成，不能用合成测试或本次数据桥代替。原晋级、重复、两GT次要分析及停止规则保持；核心创新、外部SOTA或TGRS就绪尚未成立。
