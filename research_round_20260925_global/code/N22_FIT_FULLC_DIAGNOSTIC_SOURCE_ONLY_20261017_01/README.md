# N22 全候选动作采集与独立实例诊断源码

本目录提供实际成功执行版本的通用源码；不包含数据、全量标注、权重、私有SPEC/资格/运维配置或原始科学输出。GPU采集与独立CPU复算均已真实结束并通过Root外部接收；这不是新模型、AP收益或投稿就绪证明。

- capture_body.py：冻结teacher/PR、原真实GT对应与单次forward、全部100自然query的完整C动作，包含K0。使用实际未门控u和FP32 L+u。
- verify_body.py：不加载teacher/PR/F，逐像素独立重建C/Omega、符号、真实实例计数及区域错误分层。
- analysis.py和oracle.py：固定真实配对、同C和at-most K的精确IoU计数；不构成AP上界或因果分解。
- run_owned.py：通用资源及阶段入口，需要调用方提供自己的完整合法输入清单；未包含本机配置实例。

本目录原始方法源码与实际运行Source逐字节一致。测试适配只将相对路径改为本目录，并将纯标准库模块的导入检查限定到该次模块加载，以允许与NumPy合成测试共同收集；未改计算实现或测试数据。

在本目录执行 python -B -m unittest discover -p "test_*.py" -v。测试全部为合成fixture，CPU流程测试使用假Torch与NumPy，不读取真实数据/资格链，也不加载模型；通过不等于真实方法有效。

真实执行还依赖原冻结模型构建、原渲染器、GT验证及分类导出的角色接口（body的SOURCE_ROLES），以及官方数据和固定模型/PR权重。必须自行提供并核验这些输入；此源码目录不承诺跨机器一条命令复现完整科研流程。正式独立复现入口仍待整理。

首次实际尝试因初始化构造基底取错训练阶段的schedule而在模型构造前退出1，原件保留；成功版本仅将construction基底context改为已核验train，所有recipe、state、GT及资源检查保持。另有纯metadata回归精确复现并修复此错误，原私有集成fixture未随本目录发布。

R是同一次fullC PR动作限制到Omega，不是旧S-only不同批次前向的逐位或AP复现。未分配query的空集训练代理不作为真实背景GT。当前只完成FIT32条件诊断，正式测试、重复新方法与跨数据集工作尚未完成。
