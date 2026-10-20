# U70：总体评价输入格式追溯与下一步最薄适配

日期：2026-10-21。状态：**已完成有限真实格式追溯；U70 CPU 评价 worker 尚未实现、绑定或运行。** 已有 U67 总体评价数值接线、U68 单片预测文件保存、U69 单片 GPU 监督入口；本说明只界定下一步 I/O，不增加模型、匹配或指标算法。

## 本次确认了什么

Root 对原输入做了字节闭合、JSON 容器解码及项目 helper 的静态 AST 检查，形成三份 PUBLIC_SAFE 摘要：

- `U70_FOUR_ORIGINAL_REFERENCE_JSON_SHAPES_PUBLIC_SAFE_20261021_01.json`
- `U70_ORIGINAL_COCO_REFERENCE_PROJECTIONS_PUBLIC_SAFE_20261021_01.json`
- `U70_ORIGINAL_DEV_AND_COCO_HELPER_FORMAT_PUBLIC_SAFE_20261021_01.json`

| 角色 | 已确认的真实格式 | 后续读取约束 |
| --- | --- | --- |
| 原 native、原 R 预测 | 两个文件顶层均为直接 JSON 列表，各18700条 | 不增加 ref 包装解包层，不重建或删改原字典；全人口每条身份、分数和顺序仍由原合同核验 |
| 原参考结果 | 顶层字典直接包含 `control_results` | 按下面两条已确认投影读取完整 COCO 字典，不取其他模型结果代替 |
| 原 `legacy_context.DEV` | 指向直接 COCO JSON；顶层为 `images / annotations / categories` 列表，长度187/380/1 | 使用真正 COCO 库 loader 构造原对象；类别的具体 ID、图像顺序仍须核验 |
| 原 `legacy_context.image_ids` | 长度187的列表 | 保留权威原顺序，后续核完整类型、唯一性及与原预测、COCO 和四片的一致性 |
| 原 `legacy_context.COCO_source` | 49923字节的项目科学 helper，静态存在 `coco_metrics`；顶层 imports 为标准库 | 它不是 COCO 库 loader，也不能默认等于61452字节的 encoder；仍须明确来源闭合及真实调用依赖 |

两份参考字典的确切投影为：

```python
reference_COCO = {
    'native_official': legacy_result['control_results']['native_official']['COCO'],
    'R': legacy_result['control_results']['strong_fixed_E24_Omega']['COCO'],
}
```

两者都包含原七键 `iouType / maxDets / metrics / official_COCO_crowd_ignore_semantics / original_GT_preserved / score_filter / stats`，其中 `stats` 长度12。必须保留整份原值及类型；只抄 AP、只抄 stats 或自行补元数据，都不能满足 U67 的精确复现检查。

本次追溯**已实际解码含 annotations 的原标注 JSON 容器**。没有执行科学源码、导入科学库、栅格化 GT 或计算 AP；没有公开原始记录。不能将这一范围写成整个历史流程从未接触 GT。格式追溯也没有重新审计或重新授予旧结果的完整运行资格。

## 下一步只需新增的 I/O

1. **Root 先准入输入。** 独立确认四片原始真实终态和完整来源，按 `[0,47] / [47,94] / [94,141] / [141,187]` 固定排序。每片读取 U69 `prediction_ref` 指向、经 bytes/SHA 闭合的完整 U68 `U66_SHARD.json`。U69 的 `prediction_metadata` 只是身份投影，不能替代完整两臂、proofs、cost、identity 和 Omega。顶层与 arms 中的副本不得重复计入。
2. **只接明确的参考角色。** Root 从原 context 提供 native、R、原参考结果、DEV 标注及 COCO helper 的显式引用和原 `image_ids`。worker 逐份闭合后解析上表格式；不递归跟随旧 checkpoints、运行链或其他无关引用，不加载权重或包内张量，不用 U66 identity 自行生成独立 native 参考。
3. **闭合真实库和原模块。** 加载真实 NumPy、真正的 COCO loader，以及七角色 `population_contract / eval_body / official / cocoeval / mask_api / geometry_diagnostic / population_metrics`。`official` 提供原 `coco_metrics`；COCO loader 是另一个明确的库依赖。项目 helper 的字节数或函数名称不能替代来源和库 ABI 核验；这一步尚未完成。预测共用的 population/mask 来源也须与四片保持一致。
4. **单次调用已有数值入口。** 通过真正 loader 构造原 COCO 对象，核 `list(coco.imgs) == ids`、380标注及类别集合 `{1}` 后，调用原 `U67.evaluate_u66_population(np, c, shards=..., native=..., strong=..., reference_COCO=..., coco=..., ids=..., modules=...)` 一次。U67 继续检查四片完整同一 U57 FORMAL E50/37050端点与来源、187/18700人口、原查询排列/S0/非 segmentation 字段、两臂身份和每图证明，以及 native/R 两份完整 COCO 字典精确复现。不得平均四片 AP。
5. **分开保存结果。** 将返回的 `summary` 与 `private_pair_records` 分开独占持久化，保留真实 bytes/SHA 和失败诊断。配对明细保持私有；原几何诊断仍使用 native evaluation 固定匹配和 PLAIN 原 Omega，保留 selected/380 的条件适用范围。保留原逐片成本；原 -1 未定义项在差值中保持 `None`，不推导 FPS。无需重写 COCO、匹配或差值数学。

## 执行前仍必须补齐

- 原 IDs 的完整类型、唯一性、顺序与四片/native/COCO 的逐项一致性，以及唯一类别的实际 ID；容器长度并未证明这些条件。
- 七个原模块的真实来源、完整所需库 ABI 和实际部署。helper 的顶层 AST 检查不等于真实库调用或数值验证。
- 独立 CPU owner 和明确资源准入。U69 硬锁 GPU operation/CUDA0，不能原样冒充 CPU 评价资格；本说明没有实现 CPU owner 或绑定器。
- 原始 Root 工具、SSH/执行链、owner OS 与 worker 的真实退出0，以及实际 wait/reap、原 birth 和 owned group 消失、资源/日志/输出闭合。pending JSON 或 owner 的 returning 自述都不能替代这些外证；未知或非零不得推断成功，不自动重试。

U67 成功返回仍是 `METRICS_COMPLETE_PENDING_ROOT_ACCEPTANCE`、`qualified=False`。后续真实总体评价应明确 AP/DEV scope、TEST=False、无新训练及零训练额度，不推进训练 frontier。当前只确认了输入格式，没有解读旧 AP 数值来证明当前 E50 提升，也没有新的 AP、创新效用或 TGRS 投稿结论。

发布边界：本说明仅保存结构事实与下一步接口；项目科学源文件、标注全文、原始记录、私有配对及实际运行引用不随此说明发布。冻结 U67/U68/U69 和当前训练均未修改。
