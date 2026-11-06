# 固定作者Detectron2扩展：实际构建成功

2026-11-06。固定DiffSARShipInst作者提交bdc3362372213bc8c3a1f6d5b7b74dc224d1b871的原setup.py build_ext --inplace已在DGX GB10完成，产生新的20097880字节_C二进制。原工具/SSH/setup进程均0，Root独立后核源码、产物及进程回收通过。**这只证明受限构建成功，不授予产物导入、D2算子数值、完整模型、外部AP或SOTA资格。**

源码归档约2.97MB，1121文件/7446109源字节逐个与固定Git提交的blob SHA一致，本机和DGX均核验。源码与完整第三方归档仅私有保留，未上传本仓库。归档SHA256为3b990faabb5ea14bb0d40b64cc11e5ae5af27eb5843241c405ffc4acc05bc33e。构建前_C不存在，完成时恰一个新非symlink二进制；SHA256为8b3323891f77ce4d2e9fdc330336cc600e821c0f77b90f398714f4d6dc6398af。

使用专用venv前缀，通过显式.pth共享已有Torch等base site-packages，因此不是完全独立依赖环境。没有在base安装/降级包。先构建原扩展，不预装整套运行依赖；setup顶层不导入D2/fvcore，正常D2._C包导入所需fvcore/PyYAML等仍待补齐。作者配置、构建入口与官方来源见[独立源码核查](U57_AUTHOR_D2_BUILD_ENTRY_REVIEW_20261106.md)。

明确CUDA_HOME对应CUDA13.0.88、GNU g++13.3、Torch2.10.0+cu130、架构12.1；FORCE_CUDA=1、MAX_JOBS=2、BUILD_NIGHTLY=0。GPU对构建子进程隐藏，CPU2/3与主训练CPU0/1分开。构建不执行模型、GT、数据读取、训练或数值GPU测试。原setup及18个csrc文件与控制来源共23个pin前后精确一致；没有因版本猜测预先改THC头或替换数学算子。setup正常生成model_zoo配置的打包副作用不等于核心源码修改。

预锁单次1800秒、组RSS8GiB、启动可用RAM48GiB、运行下限32GiB。实际parent约135.645秒，组RSS峰值730324992字节，可用RAM最低59444969472字节，违规为空。日志包含正常编译警告；GPU隐藏时Torch提示未发现CUDA runtime并使用指定CUDA_HOME，实际日志编译/链接CUDA源单元。未用这条提示推出CPU-only或数值CUDA成功。

Root后核23项pin与新产物摘要、同boot原父子出生身份均消失、所属组为空。构建独立记为1正常/0失败/1总计，不混入主训练计数或重复种子。选定来源核验不是传递二进制全闭合，产物尚未导入，不声称不存在后续ABI或数值问题。

同时主PLAIN/FULL三个正常段新增各768更新/1846匹配访问，严格至24064/37050（64.95%，32轮+352图），下一24064→24320已启动、尚未接收。一次新段登记在实际启动前抢跑并退出1；随后确认同原任务已启动后补登记，未重启科学计算。读就绪状态的保护已在下一正常段实际生效；该元数据失败未计成模型失败或新数值重试。

下一步应在专用前缀补齐最少运行依赖、正常导入作者包和扩展，再验证实际使用的DCN/RoIAlign/NMS前后向与完整单批模型，完成统一数据注册/原网格mask评价后才进入完整外部训练。当前FULL端点、重复、容量/计算、正式评价与跨集仍缺，核心目标未达成。

[完整安全构建记录](results/U57_AUTHOR_D2_BUILD_ACTUAL_20261106.json)；[23552实测训练摘要](results/U57_FORMAL23552_ACTUAL_SCALARS_20261021.json)；[23808实测训练摘要](results/U57_FORMAL23808_ACTUAL_SCALARS_20261021.json)。
