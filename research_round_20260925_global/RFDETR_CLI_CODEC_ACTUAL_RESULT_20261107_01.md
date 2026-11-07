# RF-DETR 原生 CLI 首次实测及第二版修正

2026-11-07。使用已安装并通过依赖检查的 DGX 独立环境，在主实验已验收的35,840次更新后实际运行首个合成 CPU 探针。实际工具、SSH 和子进程均退出1，原生 CLI 解析返回2。子进程被回收，独立同启动检查确认原父进程和子进程组已消失。随后原主实验正常接续。

- CLI 拒绝配置中的 multi_scale 值 off，错误指出需要 OFF、PER_BATCH 或 PER_SAMPLE。固定源码中 OFF 的值仍为 off。沿给出的配置类核对后，第二版 CLI 配方用带引号的 OFF 和 TV，其中 TV 仍对应 torchvision。其余候选字段保持原值。第一版的原始配置、错误和退出证据全部保留。[RF-DETR 固定配置源码](https://github.com/roboflow/rf-detr/blob/639eac4555ceb89ef399ccda3fde716bbcfe34cb/src/rfdetr/config.py) · [jsonargparse 4.46.0 枚举规则](https://github.com/omni-us/jsonargparse/blob/v4.46.0/jsonargparse/_typehints.py#L768-L778)
- 隔离记录拒绝一次预期的 IPv6 回环探测，以及一次 Matplotlib 字体枚举的子进程启动。原调用栈指向实际安装的 Matplotlib 3.11.2 的 _get_fontconfig_fonts。已重新读取该安装的普通源码并与原调用栈哈希核对，确认 findSystemFonts 支持 MPL_IGNORE_SYSTEM_FONTS。第二版在导入前设置该官方开关，仍保留原网络与子进程隔离。[Matplotlib 3.11.2 固定源码](https://github.com/matplotlib/matplotlib/blob/v3.11.2/lib/matplotlib/font_manager.py)

首次原任务没有进入 JPEG、PNG、OpenCV、RLE 或 COCO 指标检查。第二版 [CLI 配方](configs/RFDETR_SEG_NANO_BASELINE_576_E100_CLI_ENUM_20261107_02.yaml) 与[自有探针](RFDETR_CLI_CODEC_QUALIFICATION_DRAFT_20261107_02/probe.py)已完成语法和接口核对，实际执行待资源空档。原生真实 SAR 数据入口、训练和 AP 仍待独立验证。
