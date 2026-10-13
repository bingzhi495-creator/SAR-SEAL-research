# 固定PR候选—原输出重建—统计连接层：合成检查

更新：2026-10-14。状态：**UNBOUND_DRAFT / SYNTHETIC_ONLY**。这份研发记录不报告真实SAR效用、科学执行资格或核心创新。

## 实现及必要检查

[连接层](code/N13_fixed_PR_candidate_statistics_pipeline_draft.py)将已有[原栅格适配器](code/N13_original_raster_fixed_PR_adapter_draft.py)、[统计模块](code/N13_fixed_PR_diagnostic_statistics_draft.py)及[纯布尔oracle](code/N13_fixed_candidate_action_oracle_draft.py)接到一个显式输入接口。调用者提供Torch、八张量权重、原栅格F/L、互斥old_LOG/new_halo、G、image/GT/query身份及原LOG二值预测参考；程序不读取文件、不加载模型或权重、不构造width2。

先严格核对old_LOG=abs(L)≤2及native A=L>0。适配器计算Ω=old_LOG∪new_halo中的固定C，未收到G、最终选集合或K；域外须保持A。重建时从A开始，只在old_LOG写入C，并与必需的原LOG预测参考逐像素精确比较；失败即拒绝统计。通过后才把G交给统计模块；S=old_LOG、K=|S|固定。这里的参考是原预测而非GT。

全图二值相等只证明显式提供的两个数组相等，不认证原权重、F/L/参考来源、实际配对、渲染器、科学运行时、float logit相等或class/score/order/空输出。候选生成API不接收G/S/K，也不等于已认证输入的候选独立性。真实来源与成本仍须另登记核验；Ω大小是逻辑候选点数，不能当作实际调用或稀疏K成本。

## 已执行的有限合成证据

首次六像素手算例的72项核对实际子进程退出0，3.495秒，stderr为空；包含G改变而C/重建不变、错误参考与错误旧域在统计前拒绝、输入保持。这一例的Ω覆盖全图且halo候选没有实际翻转，不能单独验证非空域外和新增域翻转。

独立审阅针对这一缺口编写[八像素补查](code/N13_fixed_PR_pipeline_halo_outside_hand_check.py)，仅新增一个手算例和一次错误参考拒绝。唯一一次实际子进程退出0，2.245秒，stderr为空，106项显式核对通过；未改或重跑原模块/72项检查。

八像素L=[[-1,0,1,3],[-3,-.25,-4,4]]，F全零，权重全零但final.bias=5。旧域flat={0,1,2,5}，新增域={4}，Ω=5、K=4；G={1,2,4}。native={2,3,7}，C={0,1,2,3,4,5,7}，原LOG重建={0,1,2,3,5,7}。这些期望来自独立字面值及手算：

| 分支 | IoU | 选点/写入/翻转 | 修复/伤害/不变 |
|---|---:|---:|---:|
| native | 1/5 | 0/0/0 | 0/0/0 |
| 原LOG | 2/7 | 4/4/3 | 1/2/1 |
| 固定C exact-K | 1/2 | 4/4/3 | 2/1/1 |
| GT同K | 3/5 | 4/4/2 | 2/0/2 |
| GT不限K | 3/5 | 5/5/2 | 2/0/3 |
| 固定C at-most-K | 3/5 | 2/2/2 | 2/0/0 |

新增域idx4在C中确实0→1，原LOG重建该位置仍为false。域外idx6若错误写入L+5会转正，实际C仍为native false；全部域外保持native。完整C冒充原LOG参考时，在统计前明确拒绝；统计仅调用一次；输入未改。exact-K的强制伤害1和at-most-K拒绝分别保留，不能混称同一预算/行动条件。

两次都是host Python3.10.13、Torch2.2.2+cpu、NumPy1.26.4、CPU亲和[0,1]、1/1线程、四线程环境值1、CUDA为空且未初始化、-B、30秒期限内。它们不取得科学Torch2.0.1+cu118/NumPy1.23.5、大栅格或真实SAR资格。原执行输出和隐藏辅助运行器本地保留；公开的是便携显式模块路径检查，不公开本机运维配置或执行资格载荷。

## 复现新增的合成补查

在仓库根目录运行，需Torch、NumPy、psutil和可用CPU0/1。以下环境设置只说明合成检查的要求；不执行科学数据读取或上传。

```powershell
$env:CUDA_VISIBLE_DEVICES = ""
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
python -B research_round_20260925_global/code/N13_fixed_PR_pipeline_halo_outside_hand_check.py --pipeline research_round_20260925_global/code/N13_fixed_PR_candidate_statistics_pipeline_draft.py --adapter research_round_20260925_global/code/N13_original_raster_fixed_PR_adapter_draft.py --statistics research_round_20260925_global/code/N13_fixed_PR_diagnostic_statistics_draft.py --oracle research_round_20260925_global/code/N13_fixed_candidate_action_oracle_draft.py
```

原件按字节复制，公开源码SHA256：连接层cc857878920e651b3aaa59859dd942e79dbf4c9d686b25a8653b9a7ec860adf6；新增检查9ca726759cd5a52d5cefb5a52e343d1b7041a0ae2a6d7135500a10b149647dde。只对这两份明确UNBOUND普通源码做发布核对，未读取运行中的科学载荷或绑定SOURCE。

## 真实研究下一步

[诊断方案](N13_FIXED_PR_COVERAGE_REPAIRABILITY_DIAGNOSTIC_DRAFT_20261014.md)及[接口文字核对](N13_FIXED_PR_ACTUAL_SOURCE_INTERFACE_REVIEW_20261014.md)继续有效。剩余预锁E24曲线完成后，再登记固定首种子701 LOG E48、374固定匹配与6未匹配GT的真实诊断；逐项核验exact L/F、实际width2、原LOG重建、身份、资源及独立CPU结果。当前没有执行这一真实诊断，不据此选择新结构或报告提升。

三个固定E48主比较均为负，当前GEO区域/监督联合配方停止决定保留；次要曲线或合成检查不替换主结果。未来假设以真实覆盖、候选修复能力和选择余量证据决定，不把已知点头连接或检查工具本身称为核心创新。
