"""Read-only YOLO detection diagnostics using Ultralytics metrics and TIDE."""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from diagnosis import build_diagnosis, read_support
from tidecv import TIDE, Data

import ultralytics
from ultralytics import YOLO
from ultralytics.data.utils import IMG_FORMATS, img2label_paths
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics, box_iou


def digest(path):
    """Fingerprint a file without loading it all into memory."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_dataset(path, split):
    """Resolve YAML directories or image lists and audit detection labels without writing caches."""
    config = yaml.safe_load(path.read_text())
    raw_names = config["names"]
    names = dict(enumerate(raw_names)) if isinstance(raw_names, list) else {int(k): v for k, v in raw_names.items()}
    if sorted(names) != list(range(len(names))) or len(set(names.values())) != len(names):
        raise ValueError("类别 ID 必须从 0 连续递增，类别名称不能重复。")
    names = dict(sorted(names.items()))
    root = Path(config.get("path") or path.parent)
    if not root.is_absolute():
        root = path.parent / root
    root = root.resolve()
    groups, counts, issues, hashes, records = {}, {}, [], defaultdict(list), {}
    for part in ("train", "val", "test"):
        entries = config.get(part) or []
        entries = [entries] if isinstance(entries, str) else entries
        files = []
        for entry in entries:
            source = (root / entry).resolve()
            if source.is_dir():
                files.extend(p.resolve() for p in source.rglob("*") if p.suffix[1:].lower() in IMG_FORMATS)
            elif source.is_file() and source.suffix == ".txt":
                for line in source.read_text().splitlines():
                    if line.strip():
                        p = Path(line.strip())
                        files.append((source.parent / p).resolve() if not p.is_absolute() else p.resolve())
            else:
                raise FileNotFoundError(f"{part} 路径不存在或不是目录/图片列表: {source}")
        groups[part] = sorted(set(files))
        if len(files) != len(groups[part]):
            issues.append({"kind": "重复列表条目", "split": part, "count": len(files) - len(groups[part])})
        count = Counter()
        for image in groups[part]:
            if image not in records:
                label = Path(img2label_paths([str(image)])[0])
                rows = []
                if not label.is_file():
                    issues.append({"kind": "缺失标签，不能确认是否背景图", "path": str(label)})
                else:
                    for ln, line in enumerate(label.read_text().splitlines(), 1):
                        if not line.strip():
                            continue
                        try:
                            v = np.array([float(x) for x in line.split()])
                            if len(v) != 5 or not np.isfinite(v).all():
                                raise ValueError("仅支持检测格式 class x y w h，且数值必须有限")
                            c, x, y, w, h = v
                            if c != int(c) or int(c) not in names or min(w, h) <= 0:
                                raise ValueError("类别或框尺寸无效")
                            if min(x - w / 2, y - h / 2) < -1e-6 or max(x + w / 2, y + h / 2) > 1 + 1e-6:
                                raise ValueError("标注框超出图像")
                            row = v.tolist()
                            if row in rows:
                                raise ValueError("重复标注框")
                            rows.append(row)
                        except ValueError as exc:
                            issues.append({"kind": str(exc), "path": str(label), "line": ln})
                records[image] = {
                    "labels": rows,
                    "sha256": digest(image),
                    "label_sha256": digest(label) if label.exists() else None,
                }
            count.update(int(row[0]) for row in records[image]["labels"])
            hashes[records[image]["sha256"]].append(
                {
                    "split": part,
                    "path": str(image),
                    "normalized_labels": hashlib.sha256(
                        json.dumps(sorted(records[image]["labels"])).encode()
                    ).hexdigest(),
                }
            )
        counts[part] = [count[i] for i in names]
    if not groups[split]:
        raise ValueError(f"{split} 没有图片，请提供独立的带标签评估集。")
    if not sum(counts[split]):
        raise ValueError("评估集没有正例标注，不能计算目标检测 AP 和漏检率。")
    selected_labels = set(img2label_paths([str(p) for p in groups[split]]))
    fatal = [x for x in issues if x.get("path") in selected_labels]
    if fatal:
        raise ValueError(
            "评估集标签需要先确认，未跳过问题后继续评分：\n" + json.dumps(fatal[:20], ensure_ascii=False, indent=2)
        )
    duplicates = [v for v in hashes.values() if len(v) > 1]
    return (
        names,
        groups,
        records,
        {
            "root": str(root),
            "image_counts": {k: len(v) for k, v in groups.items()},
            "provided_splits": [k for k in groups if config.get(k)],
            "counts": counts,
            "issues": issues,
            "duplicates": duplicates,
            "conflicting_labels": [g for g in duplicates if len({x["normalized_labels"] for x in g}) > 1],
        },
    )


def prepare_samples(paths, records, output, imgsz):
    """Read image geometry, make report previews, and characterize each labeled object."""
    samples = []
    (output / "images").mkdir()
    for i, path in enumerate(paths):
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"无法读取图片: {path}")
        h, w = image.shape[:2]
        gt, features = [], []
        for c, x, y, bw, bh in records[path]["labels"]:
            box = np.array([x - bw / 2, y - bh / 2, x + bw / 2, y + bh / 2]) * [w, h, w, h]
            gt.append([int(c), *box.tolist()])
            x1, y1, x2, y2 = box.astype(int)
            roi = image[max(0, y1) : max(y1 + 1, min(h, y2)), max(0, x1) : max(x1 + 1, min(w, x2))]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            pixels = min(bw * w, bh * h) * imgsz / max(w, h)
            features.append(
                {
                    "size": "短边<32px" if pixels < 32 else "短边32–96px" if pixels < 96 else "短边≥96px",
                    "edge": "靠近边缘"
                    if min(x - bw / 2, y - bh / 2, 1 - x - bw / 2, 1 - y - bh / 2) < 0.03
                    else "非边缘",
                    "brightness": float(gray.mean()),
                    "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                }
            )
        scale = min(1, 1280 / max(w, h))
        preview = cv2.resize(image, (round(w * scale), round(h * scale)))
        if not cv2.imwrite(str(output / "images" / f"{i}.jpg"), preview, [cv2.IMWRITE_JPEG_QUALITY, 88]):
            raise OSError("报告预览图片写入失败")
        samples.append(
            {
                "id": i,
                "path": str(path),
                "image": f"images/{i}.jpg",
                "width": w,
                "height": h,
                "gt": gt,
                "features": features,
                "sha256": records[path]["sha256"],
                "label_sha256": records[path]["label_sha256"],
            }
        )
    return samples


def tensors(rows, prediction=False):
    """Convert stored class/xyxy/score rows to the native metric input."""
    a = torch.tensor(rows, dtype=torch.float32).reshape(-1, 6 if prediction else 5)
    result = {"cls": a[:, 0], "bboxes": a[:, 1:5]}
    if prediction:
        result["conf"] = a[:, 5]
    return result


def match_rows(match, key, score=False):
    """Serialize native confusion-matrix evidence."""
    item = match[key]
    return [
        [int(c), *b.tolist(), *([float(s)] if score else [])]
        for c, b, s in zip(
            item.get("cls", []), item.get("bboxes", []), item.get("conf", [0] * len(item.get("cls", [])))
        )
    ]


def count_metrics(matrix):
    """Compute fixed-threshold counts; absent denominators are explicitly null."""
    tp = matrix.diagonal()[:-1]
    fp = matrix.sum(1)[:-1] - tp
    fn = matrix.sum(0)[:-1] - tp
    return [
        {
            "tp": int(t),
            "fp": int(p),
            "fn": int(n),
            "precision": float(t / (t + p)) if t + p else None,
            "recall": float(t / (t + n)) if t + n else None,
        }
        for t, p, n in zip(tp, fp, fn)
    ]


def explain_misses(missed, predictions, conf):
    """Describe the most overlapping visible prediction, without overriding native one-to-one matches."""
    pred = tensors([r for r in predictions if r[5] > conf], True)
    overlap = box_iou(tensors(missed)["bboxes"], pred["bboxes"])
    low = tensors([r for r in predictions if r[5] <= conf], True)
    low_overlap = box_iou(tensors(missed)["bboxes"], low["bboxes"])
    contexts = []
    for row, ious, low_ious in zip(missed, overlap, low_overlap):
        candidate = None
        if len(ious):
            j = int(ious.argmax())
            if float(ious[j]) >= 0.1:
                candidate = {
                    "class": int(pred["cls"][j]),
                    "conf": float(pred["conf"][j]),
                    "box": pred["bboxes"][j].tolist(),
                    "iou": float(ious[j]),
                }
        low_candidate = None
        same = torch.where(low["cls"] == row[0])[0]
        if len(same):
            j = int(same[low_ious[same].argmax()])
            if float(low_ious[j]) >= 0.1:
                low_candidate = {
                    "class": int(low["cls"][j]),
                    "conf": float(low["conf"][j]),
                    "box": low["bboxes"][j].tolist(),
                    "iou": float(low_ious[j]),
                }
        contexts.append({"gt": row, "candidate": candidate, "low_conf_candidate": low_candidate})
    return contexts


def evaluate(model_path, samples, names, args, title):
    """Run one model once, then reuse raw predictions for AP, TIDE, and threshold analysis."""
    model = YOLO(str(model_path))
    if model.task != "detect":
        raise ValueError(f"目前仅支持检测模型，本模型 task={model.task}")
    if len(set(model.names.values())) != len(model.names) or set(model.names.values()) != set(names.values()):
        raise ValueError("模型与数据集类别名称不一致，不能按数字 ID 强行比较。")
    by_name = {v: k for k, v in names.items()}
    remap = {k: by_name[v] for k, v in model.names.items()}
    by_path = {}
    results = model.predict(
        str(args.output / "images.txt"),
        stream=True,
        batch=1,
        imgsz=args.imgsz,
        device=args.device,
        conf=0.001,
        iou=0.7,
        max_det=300,
        verbose=False,
        save=False,
    )
    for result in results:
        by_path[str(Path(result.path).resolve())] = [
            [remap[int(c)], *b, float(p)]
            for b, c, p in zip(
                result.boxes.xyxy.cpu().tolist(), result.boxes.cls.cpu().tolist(), result.boxes.conf.cpu().tolist()
            )
        ]
        print(f"进度 | {title} | {len(by_path)}/{len(samples)} 张图片", flush=True)
    if set(by_path) != {s["path"] for s in samples}:
        raise ValueError("模型未返回全部图片结果")
    predictions = [by_path[s["path"]] for s in samples]
    validator = DetectionValidator(args={"plots": False, "save": False, "project": str(args.output), "name": "metrics"})
    metrics = DetMetrics(names=names)
    thresholds = sorted({0.05, 0.1, 0.25, 0.5, 0.75, args.conf})
    matrices = {c: ConfusionMatrix(names=names, save_matches=c == args.conf) for c in thresholds}
    tide_gt, tide_pred = Data("ground_truth", max_dets=300), Data(title, max_dets=300)
    evidence, slices = [], defaultdict(lambda: [0, 0])
    for i, (sample, rows) in enumerate(zip(samples, predictions)):
        gt, pred = tensors(sample["gt"]), tensors(rows, True)
        metrics.update_stats(
            {
                **validator._process_batch(pred, gt),
                "conf": pred["conf"].numpy(),
                "pred_cls": pred["cls"].numpy(),
                "target_cls": gt["cls"].numpy(),
                "target_img": np.unique(gt["cls"].numpy()),
                "im_name": sample["path"],
            }
        )
        before = matrices[args.conf].matrix.copy()
        for conf, matrix in matrices.items():
            matrix.process_batch(pred, gt, conf=conf, iou_thres=args.match_iou)
        matches = matrices[args.conf].matches
        missed = match_rows(matches, "FN")
        ev = {"fn": missed, "fp": match_rows(matches, "FP", True), "tp": match_rows(matches, "TP", True)}
        delta = matrices[args.conf].matrix - before
        ev["confusions"] = [
            {"true": int(t), "pred": int(p), "count": int(delta[p, t])}
            for p, t in np.argwhere(delta[:-1, :-1] > 0)
            if p != t
        ]
        ev["fn_context"] = explain_misses(missed, rows, args.conf)
        evidence.append(ev)
        missed_keys = {(int(row[0]), *row[1:]) for row in missed}
        for c, box, feat in zip(gt["cls"].tolist(), gt["bboxes"].tolist(), sample["features"]):
            failed = (int(c), *box) in missed_keys
            for dimension in ("size", "edge"):
                stat = slices[f"{int(c)}|{dimension}|{feat[dimension]}"]
                stat[0] += 1
                stat[1] += int(failed)
        for data in (tide_gt, tide_pred):
            data.add_image(i, sample["path"])
        for c, x1, y1, x2, y2 in sample["gt"]:
            tide_gt.add_ground_truth(i, c, box=[x1, y1, x2 - x1, y2 - y1])
        for c, x1, y1, x2, y2, conf in rows:
            tide_pred.add_detection(i, c, conf, box=[x1, y1, x2 - x1, y2 - y1])
    metrics.process()
    tide = TIDE()
    tide.evaluate(tide_gt, tide_pred, mode=TIDE.BOX, pos_threshold=0.5)
    errors = tide.get_main_errors()[title]
    ap = {
        int(c): {"ap50": float(metrics.box.ap50[j]), "ap50_95": float(metrics.box.ap[j])}
        for j, c in enumerate(metrics.ap_class_index)
    }
    run_dir = (args.train_run if title == "当前模型" else args.baseline_run) or model_path.parent.parent
    curve_path = run_dir / "results.csv"
    args_path = run_dir / "args.yaml"
    history = list(csv.DictReader(curve_path.open())) if curve_path.exists() else []
    train_args = yaml.safe_load(args_path.read_text()) if args_path.exists() else model.ckpt.get("train_args", {})
    return {
        "title": title,
        "path": str(model_path),
        "sha256": digest(model_path),
        "class_remap": remap,
        "train_args": train_args,
        "history": history,
        "training_sources": {
            "results_csv": str(curve_path) if curve_path.exists() else None,
            "args_yaml": str(args_path) if args_path.exists() else None,
            "association": "用户指定或权重相邻目录，尚未验证是该权重的原始训练记录",
        },
        "predictions": predictions,
        "evidence": evidence,
        "ap": ap,
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "thresholds": {str(c): count_metrics(m.matrix) for c, m in matrices.items()},
        "confusion": matrices[args.conf].matrix.tolist(),
        "slices": dict(slices),
        "tide": errors,
    }


def main():
    """Build a self-contained local report directory."""
    parser = argparse.ArgumentParser(description="YOLO 本地问题分析：原生指标 + TIDE + 中文证据报告")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--baseline", type=Path, help="可选的旧模型；名称必须与数据集一致，允许类别顺序不同")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model-family", choices=["auto", "v8", "v9", "v10", "v11", "v12", "v26"], default="auto")
    parser.add_argument("--context", type=Path, help="补充资料 JSON")
    parser.add_argument("--metadata", type=Path, help="逐图场景 CSV")
    parser.add_argument("--train-run", type=Path, help="当前模型训练结果目录")
    parser.add_argument("--baseline-run", type=Path, help="对照模型训练结果目录")
    parser.add_argument("--output", type=Path, required=True, help="新的报告目录（不覆盖旧报告）")
    args = parser.parse_args()
    if not 0.001 < args.conf < 1 or not 0 < args.match_iou < 1 or args.imgsz <= 0:
        parser.error("要求 0.001 < conf < 1、0 < match-iou < 1、imgsz > 0")
    args.data, args.model, args.output = args.data.resolve(), args.model.resolve(), args.output.resolve()
    for model in (args.model, args.baseline):
        if model is not None and not model.is_file():
            parser.error(f"模型文件不存在: {model}")
    if args.baseline:
        args.baseline = args.baseline.resolve()
        if digest(args.model) == digest(args.baseline):
            parser.error("两个模型 SHA-256 相同，是同一模型的副本，不能作为新旧对照。")
    names, groups, records, audit = read_dataset(args.data, args.split)
    support = read_support(args.context, args.metadata, groups[args.split])
    args.output.mkdir(parents=True, exist_ok=False)
    samples = prepare_samples(groups[args.split], records, args.output, args.imgsz)
    (args.output / "images.txt").write_text("\n".join(s["path"] for s in samples), encoding="utf-8")
    print(f"评估 {len(samples)} 张 {args.split} 图片；数据和模型均只读。", flush=True)
    models = [evaluate(args.model, samples, names, args, "当前模型")]
    detected_family = str(models[0]["train_args"].get("model", "")) if models[0].get("train_args") else ""
    requested_family = getattr(args, "model_family", "auto")
    if requested_family != "auto" and requested_family not in detected_family.lower():
        print(
            f"提示：用户选择 {requested_family}，权重训练参数未显示同名系列；以权重实际 task 和结构为准。", flush=True
        )
    if args.baseline:
        models.append(evaluate(args.baseline, samples, names, args, "对照模型"))
    report = {
        "created": datetime.now().astimezone().isoformat(),
        "data": str(args.data),
        "data_sha256": digest(args.data),
        "names": list(names.values()),
        "settings": {
            "split": args.split,
            "imgsz": args.imgsz,
            "conf": args.conf,
            "match_iou": args.match_iou,
            "prediction_conf": 0.001,
            "nms_iou": 0.7,
            "max_det": 300,
            "device": args.device,
            "model_family_selected": args.model_family,
            "task": "detect",
            "ultralytics": ultralytics.__version__,
            "torch": torch.__version__,
        },
        "audit": audit,
        "samples": samples,
        "support": support,
        "models": models,
    }
    report["diagnosis"] = build_diagnosis(report)
    (args.output / "action-plan.json").write_text(
        json.dumps(report["diagnosis"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    payload = json.dumps(report, ensure_ascii=False, allow_nan=False, default=str)
    (args.output / "report.json").write_text(payload, encoding="utf-8")
    template = Path(__file__).resolve().parents[1].joinpath("ui/report.html").read_text()
    (args.output / "index.html").write_text(
        template.replace("__REPORT_DATA__", payload.replace("<", "\\u003c")), encoding="utf-8"
    )
    with (args.output / "review.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "image", "type", "class", "x1", "y1", "x2", "y2"])
        for model in models:
            for sample, ev in zip(samples, model["evidence"]):
                for kind in ("fn", "fp"):
                    for row in ev[kind]:
                        writer.writerow([model["title"], sample["path"], kind, names[row[0]], *row[1:5]])
    print(f"报告完成: {args.output / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
