# 固定PR诊断最小接口补齐：预计算人口行与距离分层透传

更新：2026-10-14。状态：**UNBOUND_DRAFT / SYNTHETIC_ONLY**。两项改动服务于已有诊断方案，不是新的SAR方法、实际加速测量或创新结论。

## 保留逐对JSON行，复用原人口汇总

旧[统计接口](code/N13_fixed_PR_diagnostic_statistics_draft.py)的人口入口只接收整批数组记录，并逐记录调用diagnose_instance。[连接层](code/N13_fixed_PR_candidate_statistics_pipeline_draft.py)已经生成逐对diagnostic；重新组装数组再汇总会重复统计，也需要保留整个人口的布尔栅格。

新增matched_diagnostics，与原matched_instances严格二选一，均接受显式list/tuple。新路径复制已生成的JSON行，检查JSON有限性、必要字段、身份及基本计数，随后复用同一段原人口汇总代码；不调用diagnose_instance。旧数组路径、身份去重、匹配/未匹配/总GT计数锁与距离带键一致性规则保留。调用者可以逐对装配域、生成C并重建原LOG，保留diagnostic后释放该对栅格，再汇总JSON行。此机制说明可避免接口要求保留全人口栅格，未测实际峰值内存或时耗。

```python
population = statistics.diagnose_population(
    oracle,
    matched_diagnostics=diagnostic_rows,
    unmatched_gt=unmatched_records,
    metadata=metadata,
    locked_population={"matched_count": 374,
                       "unmatched_gt_count": 6,
                       "total_gt_count": 380},
)
```

这里的374/6/380仅展示既定真实诊断人口锁；合成检查实际为2/1/3，真实人口未执行。预计算行验证**不重算原像素，也不认证数值或来源**；即使行结构和计数自洽，也不证明来自真实图像、GT、候选、配对或科学运行。来源与独立CPU数值资格须由ROOT另核验。输出仍保留科学资格、候选认证、AP上界和部署收益等字段为false，调用者不能将资格scope改成真。

[便携人口检查](code/N13_fixed_PR_precomputed_population_hand_check.py)使用一个2匹配+1未匹配的手算人口及两个显式假分层。最终GREEN实际子进程退出0并回收，0.221秒、stderr为空；59项核对、10项拒绝通过。新数组路径与预计算行路径的完整JSON相同；spy在预计算路径禁止diagnose_instance，实际调用数0。独立手算原输出宏IoU=17/24、像素IoU=5/7、TP/FP/FN=5/1/1，选点/写入/翻转均3，修复3、伤害0；配对均值分母2、总GT阈值分母3分别保留。

拒绝包括双路径、重复GT及image/query、缺分支、非有限值、人口锁不符、负计数、布尔身份、伪造资格和不一致距离带键。两条必要行为分别做RED→GREEN；预期未知新参数/schema拒绝测试失败独立保留，不计作科学失败。未重跑旧525检查、原自测或oracle穷举。

## 距离分层只在原LOG重建门后透传

连接层增加distance_bands=None，原样传给统计模块，调用发生在完整原LOG预测参考相等门通过之后。候选生成不接收这些分层或G，旧域/严格二值规则/重建及scope不变。这里不构造距离公式，也不认证分层定义和来源。

[便携透传检查](code/N13_fixed_PR_pipeline_distance_passthrough_hand_check.py)的一次GREEN实际子进程退出0并回收，1.721秒、stderr为空，35项核对通过。一个2×4例的分层A={0,1,4}、B={2,3,6}只是字面索引，不是真实几何距离。A像素3/候选域内3、原生TP/FP/FN=0/0/2；B像素3/域内1、原生1/1/0；两层剩余未覆盖{5,7}共2像素。候选与原LOG重建在传入/省略参数时一致，只有分层诊断字段改变；错误原预测参考仍在统计收到分层/G之前拒绝。RED准确报缺失新参数，属于预期研发测试失败。

## 复现与未验证范围

两项使用host Python3.10.13、NumPy1.26.4、CPU[0,1]、四线程环境值1、CUDA空、-B、30秒内；人口检查未导入Torch，透传检查使用Torch2.2.2+cpu/1/1线程且CUDA未初始化。需NumPy、psutil；透传检查还需Torch。仓库根目录：

```powershell
$env:CUDA_VISIBLE_DEVICES = ""
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
python -B research_round_20260925_global/code/N13_fixed_PR_precomputed_population_hand_check.py --statistics research_round_20260925_global/code/N13_fixed_PR_diagnostic_statistics_draft.py --oracle research_round_20260925_global/code/N13_fixed_candidate_action_oracle_draft.py
python -B research_round_20260925_global/code/N13_fixed_PR_pipeline_distance_passthrough_hand_check.py --pipeline research_round_20260925_global/code/N13_fixed_PR_candidate_statistics_pipeline_draft.py --adapter research_round_20260925_global/code/N13_original_raster_fixed_PR_adapter_draft.py --statistics research_round_20260925_global/code/N13_fixed_PR_diagnostic_statistics_draft.py --oracle research_round_20260925_global/code/N13_fixed_candidate_action_oracle_draft.py
```

两项分别针对新增接口执行；尚未联合运行真实逐对装配和整个人口，也没有测量大图、科学Torch2.0.1+cu118/NumPy1.23.5、CUDA、内存和部署成本。独立静态审阅未发现新增发布阻塞或人口规则破坏；有限小例不证明全部输入正确或真实SAR效果。

公开源码按原件字节复制，SHA256：统计acbe0f9bb1e0afbeb33616844168ed29691d1dff55478a01dbb6741823e803be；连接层36cda8f3cab7569b42d80cf45c65cc5a89fbbc2846c7a9626ae241f8e92a5cee；人口检查5c7ee9aaea4737264917c30afd5c18810c8ad8de432494a1b77d4442c23f3046；透传检查7ddefb6dba018963439ccb39bfffa8e62b2bf41d009b492d54e8847d97d56821。旧版通过Git历史保留，私有执行输出/辅助运行器与新UNBOUND原件本地保留；没有为发布读取实时科学载荷/SOURCE。

[真实诊断方案](N13_FIXED_PR_COVERAGE_REPAIRABILITY_DIAGNOSTIC_DRAFT_20261014.md)仍等待最后703 E24独立CPU及ROOT曲线资格后登记。701 LOG E48端点/八张量、exact L/F及renderer、原LOG参考、作者width2与独立整数计算、374+6身份、实际距离定义、科学运行时及资源均须按真实来源核验。三种子E48当前联合配方停止决定不变，尚无核心创新、正式TEST/跨集或TGRS依据。
