# U70：总体评价输入格式追溯与下一步最薄适配

日期：2026-10-21。新增观察截止2026-10-21 00:05:34 UTC。状态：**薄CPU I/O worker源码已完成并冻结，有限合成测试及静态审阅通过；仍未绑定、未运行。** 已有 U67 总体评价数值接线、U68 单片预测文件保存、U69 单片 GPU 监督入口；本说明保留格式事实、薄I/O接线及待验证依赖，不增加模型、匹配或指标数学；源码完成不等于真实库、AP、CPU owner或Root接收资格已经完成。

## 本次确认了什么

此前 Root 对原输入做了字节闭合、JSON 容器解码及项目 helper 的静态 AST 检查，形成三份 PUBLIC_SAFE 摘要：

- `U70_FOUR_ORIGINAL_REFERENCE_JSON_SHAPES_PUBLIC_SAFE_20261021_01.json`
- `U70_ORIGINAL_COCO_REFERENCE_PROJECTIONS_PUBLIC_SAFE_20261021_01.json`
- `U70_ORIGINAL_DEV_AND_COCO_HELPER_FORMAT_PUBLIC_SAFE_20261021_01.json`

| 角色 | 已确认的真实格式 | 后续读取约束 |
| --- | --- | --- |
| 原 native、原 R 预测 | 两个文件顶层均为直接 JSON 列表，各18700条 | 不增加 ref 包装解包层，不重建或删改原字典；全人口每条身份、分数和顺序仍由原合同核验 |
| 原参考结果 | 顶层字典直接包含 `control_results` | 按下面两条已确认投影读取完整 COCO 字典，不取其他模型结果代替 |
| 原 `legacy_context.DEV` | 指向直接 COCO JSON；顶层为 `images / annotations / categories` 列表，长度187/380/1 | 使用真正 COCO 库 loader 构造原对象；类别的具体 ID、图像顺序仍须核验 |
| 原 `legacy_context.image_ids` | 长度187的列表 | 保留权威原顺序，后续核完整类型、唯一性及与原预测、COCO 和四片的一致性 |
| 原 `legacy_context.COCO_source` | 49923字节的项目科学 helper，静态存在 `coco_metrics`；顶层 imports 为标准库 | 本机源码身份现已闭合；它不是 COCO 库 loader，也不能默认等于61452字节的 encoder，真实调用依赖及运行资格仍待验证 |

两份参考字典的确切投影为：

```python
reference_COCO = {
    'native_official': legacy_result['control_results']['native_official']['COCO'],
    'R': legacy_result['control_results']['strong_fixed_E24_Omega']['COCO'],
}
```

两者都包含原七键 `iouType / maxDets / metrics / official_COCO_crowd_ignore_semantics / original_GT_preserved / score_filter / stats`，其中 `stats` 长度12。必须保留整份原值及类型；只抄 AP、只抄 stats 或自行补元数据，都不能满足 U67 的精确复现检查。

本次追溯**已实际解码含 annotations 的原标注 JSON 容器**。没有执行科学源码、导入科学库、栅格化 GT 或计算 AP；没有公开原始记录。不能将这一范围写成整个历史流程从未接触 GT。格式追溯也没有重新审计或重新授予旧结果的完整运行资格。

## 新补记：明确的 context 桥与 helper 身份

Root 的安全摘要现明确：当前 `ROOT-N28-fixed24-dense-native-DEV-context-v1` 中的 `legacy_context` 是精确文件引用，已与字节闭合的原 `ROOT-N23-qualified-cache-DEV-context-v1` 对应；不是根据名称猜测旧上下文。两者的 native predictions、R predictions 与 legacy ROOT result 三个共同引用，其原文件 ref 及内容身份分别完全一致。两个 context 各自的 image IDs 列表长度均187，**尚未确认双列表完整类型、唯一性及全序相等**；实际迁移位置与未来绑定仍须取得明确对应证明。

原 `official` 角色的项目 helper 已在本机按原 context 引用闭合：**49923字节**，SHA-256 为 `2357929fcd38ffc3020c5b8d204ee371da8da3972ff2e2e03398c8225c55e916`。静态 AST 确认顶层存在 `coco_metrics`；该身份不是 COCO 库 loader，也不默认等于 encoder。没有执行源码、导入科学库或取得运行资格，旧结果没有因此重新获得资格（`legacy_result_requalified=False`）。

## 新补记：库静态差异与独立候选依赖

本次观察的当前本机 pycocotools 分发版本为2.0.7，DGX 为2.0.11；这不等于已经重审旧 legacy 运行当时的完整库来源。静态比较结果如下：

| 文件 | 已确认的静态事实 | 不能据此推出 |
| --- | --- | --- |
| `cocoeval.py` | CRLF规范化为LF后，两侧字节完全相同 | 真实模块导入、二进制ABI或完整旧COCO结果可复现 |
| `mask.py` | CRLF/LF规范化后字节完全相同 | 可以放宽原始字节pin，或 `_mask` 扩展已通过ABI验证 |
| `coco.py` loader | DGX 版只新增一行，将 dataset 的 `info` metadata 深复制到结果对象，缺失时使用空字典；无删除行 | 数值或ABI等价已经证明；metadata范围的预期只是推断 |

Root 另建并逐字节核验一份未来五文件依赖候选：`__init__.py / _mask.abi3.so / mask.py / coco.py / cocoeval.py`。其中 mask 保留原 **4693字节**及 SHA-256 `1281ce0b5436ffe924c96521b2ae1d589f6f0e2a7b7281b6bb284f7b2245545c`。这是新候选，原 U69 三文件 overlay 与已安装环境均未修改或替换；尚未绑定 spec、执行源码或导入科学库，没有取得ABI或完整二进制闭合资格。未来预测和 CPU 评价必须共用**同一实际 mask source ref**，不能只按相同文件名或版本号推定一致。第三方库全文、二进制及候选 overlay 不发布。

本次新增事实仅依据五份安全摘要：原 official helper 身份、原 context 角色桥、未来 loader/eval 源码身份、库静态兼容差异及五文件候选暂存；本说明没有读取实际链、原始记录、包或权重，没有运行测试、SSH 或科学库。

## U70 worker 源码收尾状态

[薄CPU总体指标worker源码说明](code/U70_DGX_FIXED50_POPULATION_METRICS_WORKER_DRAFT_20261021_01/README.md)的三份普通文件（worker、metadata测试、README）已完成并冻结。作者7项标准库/合成JSON及source/import边界替身测试、AST检查通过；独立协作者与Root的有限源码集成审阅均未发现剩余的特定阻断。测试没有真实数值运算、科学库导入或COCO评价，静态审阅没有执行源码。它实现I/O及原模块的接线，不新建指标数学；**当前仍未绑定、未运行，CPU owner尚未实现，真实库/ABI、实际AP及外部Root接收仍待完成**。

后续独立字节暂存步骤已真实退出0：7份未来CPU评价科学源码（5份原项目数学源码、U67及新U70）和6份原JSON已在DGX独立位置暂存并逐字节核验；六份JSON为context、legacy context、原结果、native、R及DEV，合计14004851字节。此次仅字节传输核验，未执行科学源码、导入科学库、解码JSON记录容器、加载模型/张量或栅格化GT，没有AP，未绑定spec、未取得运行资格，当前训练未改；此前已解码标注容器的历史事实照旧保留。本暂存摘要不发布原始输入JSON、旧数学源码副本或私有路径；新U70源码按独立项目范围同步。未来预测与CPU评价必须共用同一实际population_contract ref，mask亦须共用前述五文件候选中的实际ref；正式CPU owner仍未实现，四份真实E50分片尚未产生，后续真实导入/ABI、完整参考复现与AP资格均待完成。

未来真实报告区分三种AP执行状态：

- 尚未调用评价入口：`AP=False`。
- 已进入调用后异常：`AP=None`（unknown），不伪写为未调用或成功。
- 完整返回：`AP=True`，但仍为 `qualified=False`、待外部Root接收。

上述是源码约定；本次替身测试中的合成值不是真实AP，当前没有新的COCO复现、收益或创新证据。

## 薄I/O接线与后续真实执行顺序

1. **Root 先准入输入。** 独立确认四片原始真实终态和完整来源，按 `[0,47] / [47,94] / [94,141] / [141,187]` 固定排序。每片读取 U69 `prediction_ref` 指向、经 bytes/SHA 闭合的完整 U68 `U66_SHARD.json`。U69 的 `prediction_metadata` 只是身份投影，不能替代完整两臂、proofs、cost、identity 和 Omega。顶层与 arms 中的副本不得重复计入。
2. **只接明确的参考角色。** Root 沿已确认的 `context.legacy_context` 精确引用，提供 native、R、原参考结果、DEV 标注及 COCO helper 的显式引用和两份原 `image_ids`；双列表全序仍由运行核验。worker 逐份闭合后解析上表格式；不递归跟随旧 checkpoints、运行链或其他无关引用，不加载权重或包内张量，不用 U66 identity 自行生成独立 native 参考。
3. **闭合真实库和原模块。** 加载真实 NumPy、真正的 COCO loader，以及七角色 `population_contract / eval_body / official / cocoeval / mask_api / geometry_diagnostic / population_metrics`。`official` 提供原 `coco_metrics`；COCO loader 是另一个明确的库依赖。本机原 official helper 身份及候选库的静态差异已明确，但不能代替七角色完整来源和真实库 ABI 核验；完整装载与调用仍未完成。预测共用的 population/mask 来源也须与四片保持一致。
4. **单次调用已有数值入口。** 通过真正 loader 构造原 COCO 对象，核 `list(coco.imgs) == ids`、380标注及类别集合 `{1}` 后，调用原 `U67.evaluate_u66_population(np, c, shards=..., native=..., strong=..., reference_COCO=..., coco=..., ids=..., modules=...)` 一次。U67 继续检查四片完整同一 U57 FORMAL E50/37050端点与来源、187/18700人口、原查询排列/S0/非 segmentation 字段、两臂身份和每图证明，以及 native/R 两份完整 COCO 字典精确复现。不得平均四片 AP。
5. **分开保存结果。** 将返回的 `summary` 与 `private_pair_records` 分开独占持久化，保留真实 bytes/SHA 和失败诊断。配对明细保持私有；原几何诊断仍使用 native evaluation 固定匹配和 PLAIN 原 Omega，保留 selected/380 的条件适用范围。保留原逐片成本；原 -1 未定义项在差值中保持 `None`，不推导 FPS。无需重写 COCO、匹配或差值数学。

## 执行前仍必须补齐

- 原 IDs 的完整类型、唯一性、顺序与四片/native/COCO 的逐项一致性，以及唯一类别的实际 ID；容器长度并未证明这些条件。
- 七个原模块的完整真实来源、全部所需库 ABI 与实际部署对应。原 official helper 本机身份闭合、候选五文件暂存及静态比较，不等于七角色全部闭合、真实导入/COCO调用或数值复现；旧 legacy 库来源仍未重新取得资格。
- 独立 CPU owner 和明确资源准入。U69 硬锁 GPU operation/CUDA0，不能原样冒充 CPU 评价资格；本说明没有实现 CPU owner 或绑定器。
- 原始 Root 工具、SSH/执行链、owner OS 与 worker 的真实退出0，以及实际 wait/reap、原 birth 和 owned group 消失、资源/日志/输出闭合。pending JSON 或 owner 的 returning 自述都不能替代这些外证；未知或非零不得推断成功，不自动重试。

U67 成功返回仍是 `METRICS_COMPLETE_PENDING_ROOT_ACCEPTANCE`、`qualified=False`。后续真实总体评价应明确 AP/DEV scope、TEST=False、无新训练及零训练额度，不推进训练 frontier。当前已确认输入格式、部分来源身份与库静态差异，薄CPU I/O worker源码已冻结且通过有限合成测试/静态审阅，但未绑定、未运行。完整四分片真实预测及总体CPU COCO均未执行，没有解读旧 AP 数值来证明当前 E50 提升，也没有新的 AP、创新效用或 TGRS 投稿结论。

发布边界：本说明仅保存结构事实与下一步接口；项目科学源文件、标注全文、原始记录、私有配对及实际运行引用不随此说明发布。冻结 U67/U68/U69 和当前训练均未修改。
