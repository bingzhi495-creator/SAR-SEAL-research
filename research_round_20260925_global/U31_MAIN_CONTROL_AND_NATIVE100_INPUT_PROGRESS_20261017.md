# 完整自主结构：主对照、真实输入与补采进展

2026-10-17。主线保留用户原full完整卷积主体及共享实例读出；此前full64精确等价和eval反向已实际通过。下述新几何/普通主干仅AST，未运行数值，不继承旧测试成功。

历史clean默认base32/GN/双线性上采样，而full为64起始/BN为主/独立转置卷积，直接差值有额外解释。主普通对照改为原archive/old_code/Unet++/model1.py的plain64，尺度、最高分辨率节点及上采样边与full更直接对应；不称历史clean或参数配平。[静态来源](U31_PLAIN_UNETPP_STATIC_INTERFACE_20261017.md)、[特征包装及唯一未执行测试](code/U31_PLAIN_FEATURE_DRAFT_20261017_01/README.md)。

[400输入/残差几何模块](code/U31_IMAGE_GEOMETRY_DRAFT_20261017_01/README.md)已实现，Root先执行无Torch的MissingModule RED后实现，仅AST。解析非方图测试避免用另一次插值镜像实现，仍须实际GREEN与真实影像方向/内容/原图渲染检查。

[缓存接口调查](U31_CACHE_READER_STATIC_INTERFACE_20261017.md)发现：N28/N13 FIT只保存matched_M200；N16/N22有全部100 e/L/C，N22另有原S0和export映射，但未保存M100。187DEV虽有M100，不混入FIT补足。下一[真实M100补采](code/U32_NATIVE100_M_CAPTURE_DRAFT_20261017_01/README.md)选择已核N22 FIT32中最低dataset_index且非方/至少3GT的691（image871/G9），以相同原teacher真实forward补存M/F，并严格核原N22锚点；不是从L倒推，尚未执行。

此前全FIT候选3/3GT只形成未绑定metadata；换至691是为了同时取得既有S0/export锚点，不依据AP或像素结果。Root已只读核691编码文件摘要及JPEG头，503×283、RGB、无非平凡EXIF方向；未解码像素，不能替代真实图像B0。

独立N28第12、13、14轮均原实际退出0并经Root接收，两臂各累计10374次更新，参数/原优化器/随机状态继续接续；仍固定24轮才完整DEV评价。没有新AP，不能从中途loss决定胜负。[逐轮实际摘要](results/N28_E12_TO_E14_CONTINUOUS_TRAINING_RESULT_20261017.json)。

完整主线尚无400真实影像训练或新AP。补采及实际B0通过后，锁定plain/full公共输入、共享头、有效优化器参数、初始化与相同50轮主训练；旧权重未排除留出泄漏前不用。资源试点拟每臂两次真实一次性更新，以覆盖持有AdamW状态时的下一前后向峰值；它不计作正式50轮主训练，也不构成效用或全741资源保证。该后续试点源码仍研发中。
