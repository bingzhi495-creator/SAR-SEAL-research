# N20 独立 CPU 训练端点数值 body（UNBOUND / NOT_EXECUTED）

普通源码与tiny测试已保存，未运行Torch、测试或AST；当前训练结果未读取，不称训练或端点已合格。无I/O/owner/admission/历史前门，无失败计数硬编码。ROOT在实际GPU终止、RAW/Source/锁/父cache资格闭合后，注入以下已解码对象。

外层所需gate/spec依赖：scout_module_source（create_head）、base_scout_source（旧513 factory）、本qualifier_body_source、完整N20锁和expected_N20_initialization_digest；qualified旧/邻域两report与真实packet refs；训练GPU source/spec/report/START、E8/E24 checkpoint/FIT endpoint/UPDATE_TRACE真实refs及actual0、CPU自己的entry/spec/ad。物理refs、20项绑定和Rootcache/初始化metadata闭合由front负责，body绝不自行load。

- snapshot 是 torch.load 得到的顶层dict，无额外 snapshot 包装。GPU_endpoint 是 FIT JSON['endpoint']内层。cache_rows 是旧qualified report.rows完整741，逐行4096slots。
- loader(index)只返回{packet,neighbor_packet:{extra_F,neighbor_to_combined}}；CPU同实际GPU输入，LOCAL严格cat(old unique F,extra)[map][原sample_to_unique]，中心列4为原unique行；CENTER显式重复center九份。不能CPU重渲染F/重匹配/改标签或读full nativeL。
- validate_report(torch,scout_module,base_factory,report,START,expected_digest)：核原pending状态、完整20项binding/24轮计数/E8E24固定ref键/初态metadata。
- state_digest(torch,state)->(whole_hex,tensor_metadata)：逐字沿N20tail sorted(name)、JSON排序紧凑UTF8头、8byte小端头长、FP32连续C-order原始字节。validate_initialization_evidence(torch,scout_module,base_factory,initial_states,expected_digest)重建首seed202610150201的共同4tensor+zero空间并比较保存的整个初态证明。
- validate_state_evidence(torch,scout_module,base_factory,snapshot,cache_rows,epoch,expected_initialization_digest=None)：最后参数必须显式传锁值；5tensor18851有限、原AdamW所有超参数/参数ID次序/完整五份moments、非负二阶矩、FP32 scalar step值精确5928或17784（不是int截断）。history/每epoch私有randperm741与最后generator状态/完整曝光逐值核。训练后空间权重不强制zero/非zero或两臂分化。
- validate_snapshot_report(torch,snapshot,report,epoch)与validate_snapshot_pair(torch,E8,E24)：保存的history/order prefix、原20项metadata及外部RNG记录一致。只核保存记录，不声称复放原外部事件。
- validate_trace(trace_records,snapshot,cache_rows)：E8传实际paired_image_update前5928条，E24传全部17784条；与history同顺序在线和/均值exact。没有落盘before/endpoint事件，不声称这些轨迹已回放。
- validate_endpoint(torch,scout_module,base_factory,snapshot,GPU_endpoint,loader,cache_rows,epoch,observer=None,expected_initialization_digest=None)：完整741图同4096slots，factory与独立空间加权/LN有偏方差/matmul/ReLU公式绝对logit误差≤1e-4、rtol0；两CPU路径各自的完整逐图均值CE与保存GPU最终FIT均值绝对误差≤1e-4。沿旧冻结值，不增容差；原loss为mean(weight*ERROR_CE)。

推荐外层顺序：validate_report；每端点validate_snapshot_report→validate_state_evidence/validate_trace→validate_endpoint；最后validate_snapshot_pair并核自己的资源/Source/RNG和原actual0后另出ROOT资格。validate_endpoint本身也会执行state_evidence，可避免外层重复单独调用。

轻量返回保留qualified=False/ROOT_actual_CPU0_verified=False；两CPU公式不是GPU逐点logit核验，GPU仅保存aggregate CE，故不声称排名/selected/mask/RLE/AP或AdamW训练轨迹重放。原START/report/snapshot中pending/False状态不可改写。

一个tiny合成测试先成文，覆盖非零空间权重作用、实际稀疏slot映射、两CPU公式、完整五参数moments，明确17784.0的FP32载体有效、17784.5拒绝；未执行，不代替真实741资格。
