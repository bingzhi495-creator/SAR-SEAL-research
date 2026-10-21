# U75/U76：单分片外部接收与完整人口准入的有限源码进展

截至 2026-10-21，本备忘记录两份未来外部校验 helper 的源码合同、一次独立静态集成审阅，以及审阅中发现并在独立 U76 修订 02 修复的清单身份缺口。**这不是四个真实 E50 分片已经产生、Root 已实际接收或 CPU/AP 已运行的报告。** 原训练、模型数学、U73/U74 owner、U67/U70/U71 原件均未因此改写。

## 两层校验分别解决什么

U75 的 `accept_shard` 面向一个真实预测分片。它要求外部原始执行记录中的 Root 工具、SSH、owner OS 三层退出码均为严格整数 0，并另核 owner 保存的真实 child wait 结果、原始日志与产物字节、进程 birth/group 终止证据和既定资源记录。collector 自己成功返回或 owner 报告“等待 Root 审阅”，均不能替代原始执行终态。postcheck 还需 Root 独立观察 collector 实际结束后再封装；collector 不能以自身报告证明自己已经退出。

单片输入包括实际完整 U66 预测 JSON、U73 worker/U74 entry 报告、owner receipt/run-state 和外部证据。U75 比较完整预测与摘要，按固定分片范围和原清单联接 image/query 顺序，检查两个臂的候选、identity、原 native 字典非掩膜字段、逐图校准声明、成本及清理字段；校验原 FORMAL E50/37050 端点、两臂 checkpoint 和原恢复来源的整 ref 关系，保留 U73 外层 22 来源与 U66 原 21 来源投影。它流式核原始字节，不反序列化权重，不重新执行预测、RLE、校准或指标数学。报告中的数学与运行事实仍须来自可信的真实执行及 Root 外部观察，不能靠填写 JSON 产生。

U75 的纯 `validate_accepted_proof` 保留 11 份嵌入的原 JSON 字节及其 SHA 联接；完整预测另由调用者按实际原始字节闭合。其返回对象只表达单片外部接收合同，`qualified/AP` 仍为 False。`accept_shard` 本身不保存接收结果、不更新账户、不做 Ops CAS，也不发起后续作业。

U76 修订 02 面向四份已经由 U75 真正接收的单片证据与四份实际完整预测。它先闭合精确九项 raw 输入（固定 U75 源码、四证据、四预测），再调用该固定版本的原 `validate_accepted_proof`。四范围固定为 `[0,47)`、`[47,94)`、`[94,141)`、`[141,187)`；按 ordinal 覆盖 47/47/47/46 图、4700/4700/4700/4600 query，读取各片完整 identity 块的既有图像顺序，要求合计 187 个唯一 ID，而非从文件名猜测或排序重建。

跨片还要求相同完整端点、两臂 checkpoint、21 科学来源、外层 22 来源及所选库来源。未来 CPU 请求中的 population contract、mask API 及共享 pycocotools/extension 必须与预测采用的对应实际整 ref 相同；返回的证据 refs 和 CPU 请求内预测 refs 保持 ordinal 精确对应。U76 不执行模型、COCO 或 AP；返回状态只是未来 CPU 材料准入，不能代表 CPU 已运行或 AP 已取得资格。

## 发现的 01 缺口与 02 最小修复

原 U76 01 只共同比较端点、来源和库，并要求 187 个图像 ID 不重复，未共同比较四份 `evidence_refs.manifest`。U75 逐片证明“本片与自己的完整清单一致”，不自动证明“四片使用同一个实际完整清单”。因此可以保留各片选中记录及不重叠 ID，却改变其他未选中记录或清单的实际路径；四片各自通过单片门仍不足以闭合统一人口身份。后续 U70 的内容检查不能替代此处应有的准入约束。

作者自查、Root 与独立审阅均确认该反例。原 01 保持冻结，新独立 02 只增加两项生产检查：清单 ref 通过原 `strict_ref`；后三片的清单 path/bytes/SHA 整 ref 与首片相同。相同 bytes/SHA 的另一实际路径也被拒绝。不新增文件输入、不改变九项 raw ABI 或输出 schema、不改任何数学及训练预算。这是未执行源码门的修复，**不是一次科学实验失败，不增加训练失败计数。**

新增合成负例只替换一片清单路径，保留其 bytes/SHA 和选中 ID：作者记录 RED `7c0d88` → GREEN `2329cf`。修订 02 八项 U76 有限测试 `ec8130` 实际退出 0，保留固定原 U75 validator 的合成正向集成；AST/delta `42024e` 记录 01 三文件未变且生产改动只限清单门。U75 作者七项有限合成测试 `c19c66`、AST `7b24cf` 也实际退出 0。上述均为作者的 stdlib/合成字节、JSON 与 stub 证据；本次独立审阅只读普通源码、测试和说明及 01→02 最小差异，未重跑测试、未调用 helper、未读取实际运行链或科学载荷。审阅后未发现剩余具体集成阻断，不能据此保证真实库、数据及执行必然通过。

## 尚未完成的真实接线与证据

冻结 U71 只对四份 `external_shard_admission_refs` 做字节闭合，不自动调用 U76 的语义门。Root 仍须在真实 CPU prelaunch 前实际运行 U76，并将其返回的 exact refs 与同一 CPU 请求绑定；helper 完成、暂存或 README 说明均不等于此门已经安装。

原始 Root 执行工具终态、collector 工具终态，以及之后封装/保存工具终态是三个不同事件；任一后续工具成功都不能把原失败或未知状态改写为 0。可信观察的来历属于外部信任边界，纯 metadata helper 不能独立认证一份任意构造的证据。真实接收结果持久化、Root 资格/CAS、账户处理和后续 CPU 启动仍需独立完成并保留真实退出记录。

当前没有由本备忘证明的四个完整真实 E50 分片、全人口模型预测或新 AP；也没有新涨点、方法创新或 TGRS 投稿级成功结论。正式总预算仍为每臂 37050，固定协议不因这次源码审阅改变。

## 冻结源码导航与身份

- [U75 单分片外部接收源码说明](code/U75_ROOT_FIXED50_SHARD_ACCEPTANCE_DRAFT_20261021_01/README.md)：`acceptance.py` 34428 B，SHA256 `fd7ee03ded035ced8f3aea9c74afbf018b7e6ef73a25b5a463018d9c3b4f41c2`。
- [U76 完整人口准入修订 02](code/U76_ROOT_FIXED50_POPULATION_ADMISSION_DRAFT_20261021_02/README.md)：`admission.py` 9071 B，SHA256 `9b7ae0dbd2605598d9f83d5f31f8805aad01ff2c37006e427138af0f68d2f75b`。
- 原 01 `admission.py` 为 8829 B、SHA256 `f40eba49b4d6cc406b2459da0cd86131dd23a93f3a13072564a9d44b0d32d1c0`；记录其缺口，不将 02 结果追溯赋给 01。

这里只发布项目新源码合同及范围说明，不发布真实 proof/raw JSON、Ops、私有路径、数据、权重或第三方库全文。
