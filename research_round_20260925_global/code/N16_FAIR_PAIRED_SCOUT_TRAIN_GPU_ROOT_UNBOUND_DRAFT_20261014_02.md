# N16 配对 scout 完整 GPU 入口 v02：UNBOUND 合同

状态：普通未执行草稿。未读取实际科研载荷、绑定SOURCE、状态、Q、GT或缓存；未运行科学任务、测试或Git。v01保留不改。

入口：[N16_FAIR_PAIRED_SCOUT_TRAIN_GPU_ROOT_UNBOUND_DRAFT_20261014_02.py](N16_FAIR_PAIRED_SCOUT_TRAIN_GPU_ROOT_UNBOUND_DRAFT_20261014_02.py)。其 `def run` 之前的整个纯RAW/Source资格门段取自 ROOT 普通 `N16_ALL100_FIT_CACHE_GPU_ROOT_UNBOUND_DRAFT_20261014_04.py`，仅将 `SCHEMA` 设为 `ROOT-N16-fair-paired-scout-training-GPU-spec-v1`，保留未使用的 ROOT CPU_SCHEMA。新增受控 resource failure12 原始记录资格门、原owner谱系连接和 `C_engine_source` SOURCE闭合均沿用 ROOT04。

自 `def tree_equal` 开始完整训练后段取自本入口v01。AST逐函数比对确认：ROOT04继承的gate functions改动数0；v01独有的 `tree_equal`、`qualified_cache_gate`、`run` 改动数0。原locked protocol字面内容继承ROOT04，无实际协议文件读取。训练方法、学习率、point/matching/order/init seeds、两个scout同初始化、weighted CE、私有图序、24轮、固定8/24以及checkpoint/readback/cleanup语义不变。

ROOT随后绑定的正常祖先至少107，当前预期107；12个失败原记录完整保留，当前总身份pins=119。failure12按ROOT04的特定资源终止原RAW门验证，不能当成通用科学失败门、不得抹除或伪造actual0。实际科学worker只有 ROOT 在完成完整输入谱系后才能运行。

缓存必须有独立CPU check证据、ROOT最终实际0/完整SOURCE闭合记录及缓存GPU0/CPU0 owner joins；CPU check自报pendingROOT不自动升级为资格。`ROOTcache_result`真实schema/status和keys仍由 ROOT 按真实最终记录适配，不能用预期字符串替代真实终止证据。

v01沿用字段：`scouter_source`、`training_body_source`；`cache_gpu_source`、`cache_gpu_spec`、`cache_GPU_report`、`cache_GPU_RAW`、`cache_GPU_admission_refs`（单真实ref字典）、`cachephase`；`cache_CPU_source`、`cache_CPU_spec`、`cache_CPU_RAW`、`cache_CPU_admission`、`cache_CPU_qualification`；`ROOTcache_result`、`ROOT_cache_result_schema`、`ROOT_cache_result_status`。继承ROOT04的原carrier、机会GO、fail9/10/11/12、完整SOURCE/RAW/admission字段继续由 ROOT SPEC提供。薄packet原字段、完整natural-L只ref不载、教师及固定PR权重不加载均保持v01。

主体API、薄packet形状、真实CPU合成测试及未验证范围见 [训练主体合同](N16_PAIRED_SCOUT_TRAIN_BODY_UNBOUND_DRAFT_20261014_01.md)。本v02仅做AST静态验证，没有重复合成测试，没有实际GPU训练、CPU资格、AP或科学成功结论。训练报告仍是固定24训练完成且待独立CPU资格，不是AP提升。
