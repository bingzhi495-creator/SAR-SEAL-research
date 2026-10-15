# N20 执行状态（2026-10-15）

**最新：独立CPU保存模型恢复原进程实际exit0，pure RAW141+failed24实际0后，Root完成来源、原六产物、完整输入遍历和数值证据核对。E8/E24保存模型资格通过；原GPU清理失败24与缺失REPORT不变。141正常/24失败/165终态，固定DEV首分片原任务已启动，没有N20 AP。**

- 全741图真实F邻域缓存已通过GPU/独立CPU/Root资格：旧样本3,035,136、unique中心1,786,820、额外位置2,159,732，新增序列化缓存2,360,944,527B。旧中心逐byte一致；CPU未独立重渲染额外F。
- 预锁初始化digest a13802fc0bf4f23a8ab368b11b15a24b60552baadf1ab8264daf242173854a68；LOCAL真实九邻域与CENTER重复中心，两臂5张量/18,851参数、相同四层MLP初态/采样/图序/AdamW/固定24轮。
- 原训练首试因输出父目录缺失实际exit1，失败22保留；第二次因旧当前计数门实际exit1，失败23保留；第三次保存完整E8/E24与17,784次成对更新后，在最后CUDA清理门实际exit1，失败24保留。原失败未重写，没有第四次训练，也未创建原GPU REPORT。旧成功脚本含cuBLAS workspace清理，属于可能诊断方向，未证明是唯一原因。
- 独立CPU恢复：原进程CPU01/两线程/coldCUDA，实际0，726.765秒；原终态纯前缀实际0先于任何结果读取。随后Root核原模型与20绑定、五AdamW状态、E8各5,928/E24各17,784更新、私有图序/RNG/历史/保存轨迹子流、1,482次完整canonical输入读取、Q/DETAILS/导入与全部引用闭包。独立Root恢复资格另存，原pending Q未改。
- E24保存GPU训练端点CE：LOCAL 0.005694717780、CENTER 0.005710665533；CPU两公式最大logit差3.05176e-5、与GPU保存均值CE最大差2.33460e-10（原abs1e-4/rtol0）。这是训练输入端点证据，不能推出AP收益、充分收敛、逐点GPUlogit或整轨迹重放。
- 后续DEV前门已适配恢复资格路径：首GPU owner141、failed24、pins165，四逻辑范围[0,47)、[47,94)、[94,141)、[141,187)。仅保存的E8/E24快照；不消费不存在的原GPU报告。N16旧四分片载体与N17五物理公共PR载体严格分开。新增prior carrier metadata hook已保存、一次source-only审阅未见确定阻断，尚未真实执行。
- 新DEV真实F邻域capture/writer和独立CPU解码器正在普通未绑定稿中接线。保留2048点块、8策略、全C FP64全局排序、K0；复用已核全C公共PR参考，但新选择子集仍fresh PR并核原容差/strictsign。开发草稿和metadata门不替代真实张量资格、全量COCO或AP。
- 下一步：完成必要接口检查，绑定并执行第一个固定DEV范围；原任务终态后先纯RAW前缀，再完整GPU/CPU数值与统一全量评价。
- 主预锁不变：Δ=AP_LOCAL−max(AP_CENTER,68.09938494745545)，E24-atmost主、E8/exact辅助。Δ≤0 STOP；正收益才按既有规则复验，不能事后补参数追分。N16/N17/N19旧STOP与所有失败保留。普通DW信息检验不是GLDet复现或核心创新；没有正式TEST、跨集、TGRS成功结论。

[详细恢复证据与范围](../../research_round_20260925_global/N20_SAVED_E8_E24_CPU_RECOVERY_QUALIFIED_STATUS_20261015.md)。

- **固定DEV首分片已启动**：RAW-only启动前检查actual0（0.228秒，无Torch/科学输入哈希），完整入口AST通过；一次独立source-only ABI审阅READY。原supervisor47836持有唯一新GPU任务，范围[0,47)，无自动重试。真实原栅格F9捕获/2048评分/保存GPUlogit+FP64分数及新CPU解码器已接线；CPU手算2×3邻域与反例拒绝1项actual0（2.006秒），只是合成行为检查。运行中仅同步独立TMP普通副本和此状态，不读取/哈希运行载荷或绑定SOURCE用于发布；本段不代表GPU实际数值完成或AP。
