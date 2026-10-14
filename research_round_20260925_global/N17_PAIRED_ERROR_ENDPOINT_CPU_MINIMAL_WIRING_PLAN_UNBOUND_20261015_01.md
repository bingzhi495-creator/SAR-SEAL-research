# N17配对ERROR端点CPU资格：最小接线方案（UNBOUND）

2026-10-15。一次定向静态检查：公开N16 endpoint CPU proof、许可N17 train body、公开N17 scout/init模块、初始化metadata和实验计划。未读科学载荷/实际SOURCE/Q/权重、未运行SAR或合成测试、未操作Git。ROOT告知741 FIT邻域缓存已合格，本稿不重复核验该事实。不写未知实际执行外层或预造其字段。

## 1. 模型与输入：不能直接调用旧proof

|旧N16硬编码|N17必要替换|
|---|---|
|UTILITY/ERROR、seed202610140161|LOCAL_ERROR/CENTER_ERROR、首组seed202610150171|
|hidden.weight=(32,513)、16547参数|hidden.weight=(32,521)、16803参数|
|make_scout(torch,seed)|N17 make_scout(torch,seed,ROOT绑定的旧base_factory)|
|forward(F,e,L)|forward(F,e,L,extra8)，两臂均用原ERROR labels|

四张量仍严格为hidden.weight、hidden.bias(32)、output.weight(3,32)、output.bias(3)，FP32、无buffers；dict/OrderedDict均接受，键集合和实际named_parameters顺序须精确。loader接口按已读train body返回且仅返回packet/neighbors8；沿旧CPU proof核F256/e256/raw L、query/unique映射、labels、importance，新增finite FP32[N,8]且逐点次序与已合格薄cache一致。LOCAL消费实际八邻数组；CENTER消费clip(raw L,-8,8)/8的八份复制，不造零输入，不将原513维中心raw L/8改成裁剪值。

CPU独立第二公式需新增extra8：分别计算F/e的256维非仿射LN（biased variance、eps1e-5），拼接LN(F),LN(e),raw L/8,extra8；用对应臂的四张量作Linear→ReLU→Linear。不得两次调用同一factory冒充独立公式，不重渲染F、不重抽点。原absolute logits1e-4和meanCE1e-4/rtol0保留。

## 2. 冷初始化与预锁digest

独立CPU用相同受绑定旧factory的私有seed171生成513列原初态，再补八零列；不是先随机生成521层，也不是载入N16已训练权重。实际GPU捕获初态需四张量完全等于这一CPU参照，两臂初态相等、各自optimizer初始state为空且groups等于参照。实际活体参数storage独立由GPU执行保留；CPU克隆快照本身不能补证GPU storage独立。

首组actual初态combined digest必须等于 **5b9f58a3dece9720f84b257d03ae6928a7129e6f364e218f6bd49475d5fb7aa7**，沿公开metadata的排序tensor name、little-endian uint64 header长度、sorted compact JSON(name/shape/dtype)、C-order native FP32 bytes合同；ROOT marker或预期digest字段不等于实际核算通过。其他种子只在另行实际绑定时消费其对应锁定metadata。

**新增八列仅初始化要求为零。** E8/E24加载完整训练态后不要求仍零，也不要求一定非零、两臂一定不同或继续相等；旧三张量同样可学习。两臂都严格用原error_labels和原mean(weight×CE)，原utility_labels仅保留原输入诊断，不参与N17端点loss。

## 3. E8/E24完整状态、optimizer、RNG、history

|端点|每臂updates|共享点曝光（每臂相同）|visited_images/history长度|
|---|---:|---:|---|
|E8|5928|24281088|5928/8|
|E24|17784|72843264|17784/24|

canonical cache_rows必须恰741，dataset_index顺序0..740且每行4096；empty_images=0，固定endpoint再读741图/3035136点。不能沿旧proof容许0点/少图的通用分支，不能把paired累计点数再乘2。actual checkpoints保持初始化、E8、E24及其真实输出完整记录，E8→E24不重建optimizer。

两个独立AdamW都保留完整四参数state及groups：lr3e-4、wd1e-4、betas(.9,.999)、eps1e-8、全部flags/param-ID顺序等于参照；moments首张量改(32,521)，其余形状不变，finite FP32、exp_avg_sq非负、step是FP32 scalar且原值严格等于updates，不经int截断。参数组/四参数覆盖和状态形状不因新增列而放宽。

ORDER_SEED仍202610140160001，不能改成新init seed。每epoch用私有CPU Generator.manual_seed(ORDER_SEED+epoch)生成randperm(741)，对照全部epoch_orders和history的epoch/order_seed/initialization_seed、visited/updated/empty/points及两臂finite在线CE。最后一轮before/after RNG需与独立重建exact一致。保留原完整observer trace资格；只替换arm名及固定计数：train body三种既知事件的完整子流为E8 12597条、最终E24 37050条（全图before/update及E8/E24各741条固定endpoint读取），无缺/额外记录，实际更新和逐轮CE加总与history一致；不把外层未知资源/来源日志混入该子流。不是重新训练或重放AdamW优化轨迹。

## 4. 资格边界与ROOT后续接线

逐臂CPU factory和独立公式都遍历同741 actual inputs，以原ERROR目标复算fixed endpoint的mean per-image mean(weight×CE)，分别对实际GPU aggregate meanCE按绝对1e-4核验。scope仍仅状态/moments/steps/orders/counts、同输入CPU两公式和GPU aggregate CE；GPU逐点logits、canonical FP64选点、selected indices、自然100完整mask/RLE/COCO AP、优化轨迹都不在本资格内。不能把N17 initializer的synthetic digest metadata当训练科学终止证据。

ROOT实际GPU原会话退出后，按真实checkpoint/trace/缓存refs和实际现有schema接入新独立CPU body，再闭合actual RAW0/SOURCE/资格记录；本稿只给必要差异，不预造实际RAW/GPU日志字段、checkpoint文件名、SOURCE计数或外层提交资格。无需扩展旧史审计。
