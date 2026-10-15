# -*- coding: utf-8 -*-
"""Build a factual report, plots and local-only real-image comparisons.

This does not train, select checkpoints, retune thresholds or alter evaluations.
Third-party images/video and weights stay local; summary-only plots are publishable.
"""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
COMMON = ["vehicle", "bus", "bicycle", "pedestrian"]
ZH = ["车辆（vehicle）", "公交车", "自行车（数据集定义）", "行人"]
FONT = "C:/Windows/Fonts/arial.ttf"


def load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def save(path, value):
    (ROOT / path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def figures(result, rows):
    out = ROOT / "delivery"
    out.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    x = np.arange(4)
    before = [100 * result["before_common4"]["per_class"][c]["map50"] for c in COMMON]
    after = [100 * result["after_common4"]["per_class"][c]["map50"] for c in COMMON]
    axes[0].bar(x - .19, before, .38, label="COCO pretrained")
    axes[0].bar(x + .19, after, .38, label="30-epoch fine-tuned")
    axes[0].set(xticks=x, xticklabels=COMMON, ylim=(0, 100), ylabel="AP50 (%)",
                title="Same 128 test images / official COCO evaluator")
    axes[0].legend(fontsize=8)
    for values, offset in [(before, -.19), (after, .19)]:
        for i, y in enumerate(values):
            axes[0].text(i + offset, y + 1, f"{y:.1f}", ha="center", fontsize=8)
    epochs = [int(r["epoch"]) for r in rows]
    for key, label in [("metrics/mAP50(B)", "8-class val mAP50"),
                       ("metrics/mAP50-95(B)", "8-class val mAP50-95")]:
        axes[1].plot(epochs, [100 * float(r[key]) for r in rows], label=label)
    axes[1].set(xlabel="Epoch", ylabel="AP (%)", ylim=(0, 60),
                title="Validation curve (SDK; distinct from common-4 test)")
    axes[1].legend(fontsize=8)
    fig.suptitle("YOLO11n / 1022 train + 128 val + 128 test / single seed 42\n"
                 "Electric bicycle not independently defined; Xavier NX not tested", fontsize=11)
    fig.savefig(out / "metrics_and_training.png", dpi=180)
    plt.close(fig)


def local_gallery():
    data = load("prepared_dataset.json")["test"]
    gt = load("evaluation/after_common4/ground_truth_local_only.json")
    before = load("evaluation/before_common4/predictions_local_only.json")
    after = load("evaluation/after_common4/predictions_local_only.json")
    colors = ["#2878ef", "#ff9828", "#c94bdd", "#16b77b"]
    font = ImageFont.truetype(FONT, 15)
    titlefont = ImageFont.truetype(FONT, 19)
    indices = [0, 32, 64, 96]  # Fixed spacing; no scoring/ranking or best-case selection.
    canvas = Image.new("RGB", (2880, 4 * 580), "#151515")
    metadata = []
    for row, index in enumerate(indices):
        item = data[index]
        base = Image.open(item["path"]).convert("RGB")
        image_id = index + 1
        for col, (kind, predictions, mapping) in enumerate([
                ("Official GT (common 4 only)", None, None),
                ("COCO pretrained / conf >= .25", before, {2: 0, 5: 1, 1: 2, 0: 3}),
                ("Fine-tuned / conf >= .25", after, {i: i for i in range(4)})]):
            panel = base.resize((960, 540))
            draw = ImageDraw.Draw(panel)
            boxes = ([{"bbox": a["bbox"], "label": a["category_id"] - 1, "score": None}
                      for a in gt["annotations"] if a["image_id"] == image_id]
                     if predictions is None else
                     [{"bbox": p["bbox"], "label": mapping[p["class_id"]], "score": p["score"]}
                      for p in predictions if p["image_id"] == image_id
                      and p["class_id"] in mapping and p["score"] >= .25])
            for b in boxes:
                x, y, w, h = b["bbox"]
                x, y, w, h = [v / 2 for v in [x, y, w, h]]
                color = colors[b["label"]]
                draw.rectangle([x, y, x + w, y + h], outline=color, width=2)
                label = COMMON[b["label"]] + (f" {b['score']:.2f}" if b["score"] is not None else "")
                draw.text((max(0, x), max(0, y - 16)), label, font=font, fill=color, stroke_width=1,
                          stroke_fill="black")
            canvas.paste(panel, (col * 960, row * 580 + 40))
            ImageDraw.Draw(canvas).text((col * 960 + 10, row * 580 + 8),
                                        f"test index {index}: {kind}", font=titlefont, fill="white")
        metadata.append({"test_list_zero_based_index": index, "file": Path(item["path"]).name})
    canvas.save(ROOT / "delivery/heldout_gt_before_after_local_only.jpg", quality=93)
    save("delivery/gallery_selection_local_only.json", metadata)


def video(ffmpeg):
    header = Image.new("RGB", (1920, 42), "#111111")
    draw = ImageDraw.Draw(header)
    font = ImageFont.truetype(FONT, 23)
    draw.text((14, 7), "BEFORE: COCO pretrained YOLO11n", fill="white", font=font)
    draw.text((974, 7), "AFTER: 30 epochs / 1022 training images", fill="white", font=font)
    header.save(ROOT / "delivery/comparison_header.png")
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-threads", "4",
               "-i", str(ROOT / "video_before/annotated.mp4"),
               "-i", str(ROOT / "video_after/annotated.mp4"),
               "-loop", "1", "-framerate", "30000/1001", "-i", str(ROOT / "delivery/comparison_header.png"),
               "-filter_complex_threads", "2", "-filter_complex",
               "[0:v]scale=960:540,setsar=1[a];[1:v]scale=960:540,setsar=1[b];"
               "[a][b]hstack=inputs=2[pair];[2:v][pair]vstack=inputs=2[out]",
               "-map", "[out]", "-frames:v", "719", "-r", "30000/1001", "-an",
               "-c:v", "libx264", "-threads", "4", "-preset", "fast", "-crf", "21",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart",
               str(ROOT / "delivery/intersection_before_after.mp4")]
    subprocess.run(command, check=True)
    # Also retain a full-size single-model playable video.
    subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                    "-i", str(ROOT / "video_after/annotated.mp4"), "-an", "-c:v", "libx264",
                    "-threads", "4", "-preset", "fast", "-crf", "21", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(ROOT / "delivery/intersection_finetuned.mp4")], check=True)
    checked = {}
    for name in ["intersection_before_after.mp4", "intersection_finetuned.mp4"]:
        cap = cv2.VideoCapture(str(ROOT / "delivery" / name))
        dims = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = 0
        while cap.read()[0]:
            count += 1
        cap.release()
        if count != 719:
            raise RuntimeError(f"Incomplete video {name}: {count} frames")
        checked[name] = {"decoded_frames": count, "resolution": dims, "fps": fps, "status": "all_frames_decoded"}
    save("delivery/video_verification.json", checked)


def report(result, before_video, after_video, rows):
    lines = ["# YOLO11n 路口微调实测（2026-10-15）", "",
             "**已实际完成30轮训练、128张保留测试评价和719帧公开路口视频推理。测试集四类AP改善，但行人、实际异域视频误检、电动自行车定义仍不足，尚未达到原定五类高精度和Xavier NX 30FPS验收。**", "",
             "## 实验范围与实际预算", "",
             "Intersection-Flow-5K官方v4原划分内固定抽样；因实际网络下载较慢，在读取任何精度前将首轮锁定为1024训练/128验证/128测试、30轮。两张训练图原始框坐标越界，在读指标前排除整图且未替补，实际**1022/128/128**；val/test标签未改。", "",
             "全量不是本次执行范围。抽样seed20261015，训练seed42；跨划分SHA256完全相同图为0，未证明路口/视频近邻独立。结果仅适用于这份官方划分内的小规模单seed实验，不表明跨路口泛化或已充分收敛。", "",
             "模型从官方COCO yolo11n.pt初始化，原生8类全部训练。640×640、batch16、SGD、lr0=0.01、lrf=0.01、momentum0.937、weight_decay0.0005、warmup3、最后5轮关闭mosaic、AMP；完整参数见training_config.json。训练30轮均执行；best由val选取，test不参与选模型和调阈值。", "",
             f"RTX3090Ti；results.csv末轮累计训练计时约{float(rows[-1]['time']):.1f}秒，不含下载、初次失败、加载及最终评价。第1次启动因OpenMP重复线程库在0 epoch退出，原日志保留；同一最小复现验证MKL_THREADING_LAYER=SEQUENTIAL后以恢复入口重启，训练数据、初始化和预算不变。未改共享DLL。", "",
             "## 训练前后同协议对照", "",
             "同128张测试图、真实官方GT、4个可映射类别、640方形输入、FP16；候选conf0.001、NMS IoU0.7、max_det300；AP用官方pycocotools，IoU0.50:0.95、101点、maxDets100。vehicle与COCO car、bicycle骑行者框等语义尚未完全一致审定，因此这反映数据集适配效果，不是统一训练下优于其他模型的结论。", "",
             "| 类别 | GT框数 | 训练前AP50 | 训练后AP50 | 训练前AP50–95 | 训练后AP50–95 | 训练后P | 训练后R |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for cls, zh in zip(COMMON, ZH):
        a, b = result["before_common4"]["per_class"][cls], result["after_common4"]["per_class"][cls]
        lines.append(f"| {zh} | {b['gt_objects']} | {a['map50']*100:.2f}% | {b['map50']*100:.2f}% | "
                     f"{a['map50_95']*100:.2f}% | {b['map50_95']*100:.2f}% | "
                     f"{b['precision_conf025_iou05']*100:.2f}% | {b['recall_conf025_iou05']*100:.2f}% |")
    a, b = result["before_common4"], result["after_common4"]
    lines += [f"| 四类平均 | 5280 | {a['map50']*100:.2f}% | {b['map50']*100:.2f}% | "
              f"{a['map50_95']*100:.2f}% | {b['map50_95']*100:.2f}% | — | — |", "",
              "P/R为固定conf0.25、匹配IoU0.50的官方COCO匹配结果；AP不是单阈值准确率。", "",
              f"mAP50提高{(b['map50']-a['map50'])*100:.2f}个百分点，相对增加{(b['map50']/a['map50']-1)*100:.1f}%；"
              f"mAP50–95提高{(b['map50_95']-a['map50_95'])*100:.2f}个百分点，相对增加{(b['map50_95']/a['map50_95']-1)*100:.1f}%。"
              "没有多seed，不报告稳定性或显著性。", "",
              "## 原生8类评价（另一个评价口径）", "",
              "Ultralytics 8.3.43，test128、max_det300；原生8类mAP50 **39.92%**，mAP50–95 **26.30%**。以下不能与上表混算：SDK的插值/最大检测数口径与官方COCO四类不同。", "",
              "| 原生类别 | AP50 | AP50–95 |", "|---|---:|---:|"]
    for cls, values in result["native8_per_class"].items():
        lines.append(f"| {cls} | {values['map50']*100:.2f}% | {values['map50_95']*100:.2f}% |")
    lines += ["", "engine只有115个训练框、14个测试框，且作者说明没有将其独立定义为电动自行车。不能将engine改名后声称满足电动自行车检测；本轮没有该类可认可的单独精度。", "",
              "## 公开路口视频速度与实际错误", "",
              "同一Pexels视频，1920×1080、29.970FPS、719帧/23.99秒，逐帧处理无跳帧。固定640、FP16、batch1、conf0.25、NMS IoU0.7、预热30帧。以下为RTX3090Ti + PyTorch单次顺序测试，并非Xavier NX或TensorRT。训练前保留6个COCO交通类，训练后保留8个原生类；后处理类别不同且单次测量，不把约4%的速度差解释为模型显著加速。", "",
              "| 模型 | SDK网络推理ms/帧 | 解码+预处理+推理+NMS+传框 FPS | 包含画框/JSON/存视频 FPS | P95管线延迟ms |",
              "|---|---:|---:|---:|---:|"]
    for label, v in [("COCO预训练", before_video), ("微调best", after_video)]:
        lines.append(f"| {label} | {v['sdk_mean_ms']['inference']:.2f} | {v['pipeline_fps_without_encoding']:.2f} | "
                     f"{v['wall_fps_with_render_encoding']:.2f} | {v['pipeline_latency_ms_p95']:.2f} |")
    lines += ["", "计时不含加载和预热；不包含真实摄像头网络/显示延迟。视频没有逐帧GT，检测框数是累计框数，不能当目标数、正确数或准确率。", "",
              "人工检查固定帧0/239/479/718，保留失败案例：", "",
              "1. 第0帧：近处红色小汽车被微调模型误报bus约0.70；广告牌区域被误报truck。", "2. 第239帧：左侧广告牌/楼体被误报大truck框，前方行人和部分两轮车漏检。", "3. 第479帧：预训练模型将罐车和相邻小巴合并报bus0.52；微调后仍有左侧异常大truck框和近处车辆重复框，没有解决该场景问题。", "4. 第718帧：左下方白色小汽车被微调模型高置信度误报bus0.99，右侧骑行者没有可靠检测。", "",
              "因此保留测试集内适配收益与异域公开视频误检同时存在；不能宣传本权重已经适合直接部署。公开视频只作定性外部场景检查，未用于本轮训练或调参。", "",
              "## 下一步的优先顺序", "",
              "1. 首先明确自行车/电动自行车/摩托车与骑行人的框定义，补充实际目标路口的电动自行车标注，不能仅改标签名字。", "2. 建立按摄像头或路口隔离的验证/测试集，加入当前视频暴露的广告牌、近景小车、两轮车和小行人难例；完整标注相关类别，避免只标电动车造成其他类别假负例。", "3. 在预先固定的新预算内比较全量数据、较长训练和更高输入尺寸；640缩小远处行人是待验证因素，需与NX速度共同测量。单次短预算不能说明YOLO11n总体失效。", "4. 准确率和类别定义过关后，在Xavier NX本机导出TensorRT FP16/校准INT8并测端到端≥30FPS；3090Ti速度不能换算成NX保证。", "",
              "## 本地交付与复核", "",
              "- `delivery/intersection_before_after.mp4`：训练前后同帧左右对比，H.264，完整719帧。", "- `delivery/intersection_finetuned.mp4`：微调后1080p视频。", "- `delivery/heldout_gt_before_after_local_only.jpg`：测试列表固定索引0/32/64/96的官方GT、训练前、训练后，未按好坏排序。只展示四个共同类别，不代表数据集全标注。", "- `delivery/metrics_and_training.png`：测试AP与原生8类验证曲线。", "- `runs/yolo11n_pilot302/weights/best.pt`：验证选择权重；不是Xavier TensorRT engine。", "- `evaluation_summary.json`、`dataset_audit_summary.json`、`training_completion.json`、`training_config.json`与各视频summary：可复核摘要。", "- 训练检查：完成30行、末epoch30；真实GT128图/5280四类框；跨集重复拒绝与类别映射完美框接口测试已通过，合成检查不是模型精度。", "",
              "原始图/标签、完整逐帧预测、权重和第三方视频只在本地交付，不随持续GitHub同步公开。源数据CC BY-NC-SA4.0；商业使用需另行核验授权，不能从此训练推出可商用。", "",
              "## 来源", "",
              "- [作者仓库与类别说明](https://github.com/AstronZh/Intersection-Flow-5K)", "- [Kaggle官方数据v4入口](https://www.kaggle.com/datasets/starsw/intersection-flow-5k)", "- [Pexels测试视频](https://www.pexels.com/video/cars-waiting-in-a-traffic-light-5675618/)", "- [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/)", "- [官方COCO评价器实现](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py)", "- [Intel支持的MKL线程层接口](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-1/mkl-set-threading-layer.html)", ""]
    (ROOT / "TRAINING_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ffmpeg", default="C:/Users/houbingzhi/anaconda3/Library/bin/ffmpeg.exe")
    args = parser.parse_args()
    result = load("evaluation_summary.json")
    completion = load("training_completion.json")
    rows = list(csv.DictReader((ROOT / "runs/yolo11n_pilot302/results.csv").open()))
    assert completion["epoch_index"] == 29 and len(rows) == 30 and int(rows[-1]["epoch"]) == 30
    assert result["after_common4"]["images"] == 128 and result["after_common4"]["gt_objects"] == 5280
    before, after = load("video_before/summary.json"), load("video_after/summary.json")
    assert before["frames_inferred"] == after["frames_inferred"] == 719
    assert before["xavier_nx_tested"] is False and after["ebike_supported"] is False
    figures(result, rows)
    local_gallery()
    video(args.ffmpeg)
    report(result, before, after, rows)
    manifest = {"checkpoint": completion["best"],
                "sha256": hashlib.sha256(Path(completion["best"]).read_bytes()).hexdigest(),
                "classes": ["vehicle", "bus", "bicycle", "pedestrian", "engine", "truck", "tricycle", "obstacle"],
                "epochs_completed": 30, "electric_bicycle_verified": False,
                "xavier_nx_tested": False, "intended_status": "pilot_for_inspection_not_deployment_approved"}
    save("delivery/trained_checkpoint_manifest.json", manifest)
    # A small user bundle, not a publication of dataset/GT or a full run cache.
    with zipfile.ZipFile(ROOT / "delivery/YOLO11n_intersection_pilot30_results.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(completion["best"], "weights/yolo11n_intersection_pilot30_best.pt")
        for p in [ROOT / "TRAINING_REPORT.md", ROOT / "evaluation_summary.json", ROOT / "training_config.json",
                  ROOT / "training_completion.json", ROOT / "dataset_audit_summary.json"]:
            archive.write(p, p.name)
        for p in (ROOT / "delivery").iterdir():
            if p.is_file() and p.suffix != ".zip":
                archive.write(p, "delivery/" + p.name)
    print(json.dumps({"status": "delivery_built", "report": str(ROOT / "TRAINING_REPORT.md"),
                      "weight_sha256": manifest["sha256"]}))


if __name__ == "__main__":
    main()
