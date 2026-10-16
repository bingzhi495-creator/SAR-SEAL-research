# N26 固定R开发集特征捕获：执行前源码

状态：源码已冻结，入口静态审阅未见阻塞；真实执行尚未开始。质量模块7项合成检查与元数据2项检查已实际通过，不代表真实数据或整体流程已合格。

完整187图/18700个自然query，沿N25原预测ordinal，从原R mask与既有e0[query]生成FP32 1040维输入；不读GT标注、不生成target、不调用matcher/COCO，不训练、不计算新score/AP。原R文件与旧bundle关联R是不同文件/字节格式，按(image,query)核完整字典相等，不错误要求引用相等。所有原S0/RLE/class均参与校准但不作质量特征。

每图torch.load旧容器时会连带反序列化F/L；仅e0和身份参与特征，F/L不作为数值输入且不复制到输出。不宣称完全没加载F/L。mask仅按自身bbox裁剪，empty全0；GT不能进入特征接口。全部自然query保留，不按类别分数或匹配筛选。

输出仅私有DEV特征、身份及审计摘要；不发布18700个原预测或缓存。数据产物在真实退出和Root接收前均pending；即使成功也仅为数据捕获，独立数值验证与质量头训练另行执行。

两项纯metadata测试覆盖不同合法R文件引用、query顺序错位拒绝；7项质量模块合成检查见相邻代码。可用python -B -m unittest discover -p 'test_*.py' -v运行；真实入口需要私有原数据/绑定，不是一键复现包。CPU0/1、线程2、空CUDA、RSS与可用RAM各16GiB阈值；resource_control.py只复用Control，不执行其训练CLI。

[数据阶段计划](../../../refine-logs/N26_QUERY_MASK_QUALITY_BASELINE_20261017_01/DATA_CAPTURE_PLAN.md)、[原训练计划](../../../refine-logs/N26_QUERY_MASK_QUALITY_BASELINE_20261017_01/EXPERIMENT_PLAN.md)、[N25诊断](../../N25_FIXEDMASK_SCORE_ORACLE_RESULT_20261017.md)。
