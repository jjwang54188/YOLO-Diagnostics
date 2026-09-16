"""证据和建议规则。只解释已有测量，不替代原生匹配，也不把相关性升级为病因。"""

import csv
import hashlib
import json
from collections import defaultdict

# 这也是界面和报告共用的数据清单；支持范围必须与实际读取器一致。
DATA_GUIDE = [
    {
        "id": "evaluation",
        "name": "评估集 + YOLO 标签 + data.yaml（需核验独立性）",
        "priority": "必需",
        "format": "图片、labels/*.txt、data.yaml",
        "use": "量化分类、漏检、定位和误报；空 TXT 表示已确认背景图。",
        "support": "自动读取全部提供的 train/val/test；仅对选择的分组推理",
    },
    {
        "id": "baseline",
        "name": "当前模型与旧模型",
        "priority": "强烈建议",
        "format": ".pt 检测权重，类别名称一致",
        "use": "同图同参数逐目标对照；旧模型不作为真值。",
        "support": "当前模型必需，对照模型可选",
    },
    {
        "id": "training_data",
        "name": "两模型各自的原始训练数据版本",
        "priority": "强烈建议",
        "format": "保留完整图片、标签、划分、文件哈希和版本号",
        "use": "区分新增样本、类别覆盖和版本差异，核实评估是否见过。",
        "support": "当前 YAML 的各分组自动审计；权重与数据的对应关系在补充 JSON 填写，仍属用户陈述。第二套数据请单独建任务审计",
    },
    {
        "id": "training",
        "name": "两次训练的原始结果目录",
        "priority": "强烈建议",
        "format": "results.csv、args.yaml、best.pt/last.pt 及日志",
        "use": "读学习曲线、参数差异，形成过拟合/未收敛的待验证线索。",
        "support": "选择训练目录自动读取 results.csv 和 args.yaml；训练日志、优化器状态仅保留供人工复核，不自动解析",
    },
    {
        "id": "context",
        "name": "问题描述、类别含义和标注规范",
        "priority": "强烈建议",
        "format": "templates/context.json",
        "use": "明确错成什么、业务影响、容量/SKU真值、遮挡框口径。",
        "support": "JSON 自动接入报告；其内容标为用户提供，系统不会当成已验证事实",
    },
    {
        "id": "deployment",
        "name": "现场推理配置与预处理",
        "priority": "强烈建议",
        "format": "补充 JSON 中 deployment",
        "use": "比对 imgsz/conf/iou/max_det；记录裁剪、缩放、RGB/BGR、FP16/INT8、导出运行时。",
        "support": "可比数值自动对照；代码、ONNX/TensorRT 文件和硬件现场输出需要另做复现，当前不执行",
    },
    {
        "id": "field",
        "name": "现场失败图 + 同条件成功图 + 背景负样本",
        "priority": "强烈建议",
        "format": "加入带标注评估集，CSV 的 known_failure 字段注明",
        "use": "覆盖真正故障场景，避免只看训练同源测试图。",
        "support": "图片按评估集处理；没有标签不能量化漏检；失败标记为用户陈述",
    },
    {
        "id": "metadata",
        "name": "逐图拍摄与复核信息",
        "priority": "可选增强",
        "format": "templates/scenes.csv：image,batch,scene,lighting,occlusion,view,distance,known_failure,reviewed",
        "use": "按批次、光照、遮挡等分组观察失败率与独立图片数。",
        "support": "CSV 自动读取；文件名必须唯一或使用绝对路径；不自动从像素猜遮挡等级",
    },
    {
        "id": "experiments",
        "name": "单变量对照实验记录与验收目标",
        "priority": "可选增强",
        "format": "补充 JSON 中 experiments、acceptance",
        "use": "记录只改什么、控制了什么、随机种子、模型/数据哈希、验证集和最终独立测试结果。",
        "support": "记录进入报告和 AI 上下文；文本实验声明不会自动确诊因果，需复核原始产物",
    },
    {
        "id": "raw",
        "name": "原始视频、相机参数、采集与部署代码",
        "priority": "深入排查",
        "format": "原始视频/无损帧、曝光/白平衡/焦距、裁剪前后成对图、环境锁文件",
        "use": "排查运动模糊、域差异、预处理和导出精度。",
        "support": "本版不直接分析视频/执行代码；先按时间与批次抽帧标注，在 JSON 记录相机/预处理，相关源码供人工或你另行选择的 AI 审核",
    },
]

METHODS = [
    {
        "name": "固定阈值错误计数",
        "algorithm": "Ultralytics 8.4.54 ConfusionMatrix：保留 conf > 用户阈值的预测，以 IoU > 匹配阈值按重叠度去重成一对一匹配；类别正确为 TP，错类别同时贡献该真类 FN 与预测类 FP。P=TP/(TP+FP)，R=TP/(TP+FN)。",
        "limit": "相对于现有标签成立。此匹配先看几何再看类别，与 AP 的类别约束匹配不同；改变阈值可能改变配对，TP 不保证单调。",
    },
    {
        "name": "AP 与排序质量",
        "algorithm": "DetectionValidator._process_batch + DetMetrics：conf=0.001 保留的预测，在 IoU 0.50:0.05:0.95 下做原生同类匹配，形成 PR 曲线并积分。",
        "limit": "mAP 只平均有真值的类别；不是 conf=0.25 的单点正确率，也不能代表没有采样的现场场景。",
    },
    {
        "name": "错误类型影响",
        "algorithm": "TIDE @ IoU 0.50：分类、定位、两者兼有、重复、背景、遗漏六类 oracle 修正后的 AP50 变化。",
        "limit": "是假设性上限分析，各项不相加；不是重训能获得的收益，更不是训练根因。",
    },
    {
        "name": "逐目标回归和候选框",
        "algorithm": "在同一张图、同一真值框上比较两模型是否进入 FN。候选框取当前阈值下最大 IoU，IoU≥0.10 才展示。",
        "limit": "候选框只是可视化线索，不是额外匹配，不证明某个低 IoU 框就属于该物品。",
    },
    {
        "name": "数据与场景证据",
        "algorithm": "图片 SHA-256 查完全重复；排序后的标签数值摘要查同图异标；尺寸、边缘、用户 CSV 场景分组统计目标数、失败数与独立图片数。",
        "limit": "完全重复不等于近重复检测；图片数量不等于独立拍摄批次。亮度和拉普拉斯方差只描述像素，不用统一阈值确诊模糊。",
    },
    {
        "name": "病因与建议规则",
        "algorithm": "将混淆计数、未匹配目标、新旧回归、数据审计、配置差异和学习曲线变成有编号的证据，再生成所需补充数据、竞争解释与单变量实验。",
        "limit": "规则没有训练成病因分类模型，也没有经过概率校准。本版不输出伪精确的病因置信度；没有受控干预证据时，根因保持未确诊。",
    },
]


def read_support(context_path, metadata_path, image_paths):
    """只读取明确定义的 JSON/CSV；不追踪 JSON 内的路径或执行用户文本。"""
    context, metadata, sources = {}, {}, {}
    for path in (context_path, metadata_path):
        if path and path.stat().st_size > 5_000_000:
            raise ValueError(f"补充资料超过 5 MB，请拆分：{path}")
    if context_path:
        context = json.loads(context_path.read_text(encoding="utf-8-sig"))
        if not isinstance(context, dict):
            raise ValueError("补充资料 JSON 顶层必须是对象。")
        sources["context"] = {
            "file": str(context_path),
            "sha256": hashlib.sha256(context_path.read_bytes()).hexdigest(),
        }
    if metadata_path:
        paths = {str(p): str(p) for p in image_paths}
        basenames = defaultdict(list)
        for p in image_paths:
            basenames[p.name].append(str(p))
        with metadata_path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or "image" not in reader.fieldnames:
                raise ValueError("场景 CSV 必须包含 image 列。")
            for row in reader:
                name = (row.get("image") or "").strip()
                candidates = [paths[name]] if name in paths else basenames.get(name, [])
                if len(candidates) != 1:
                    raise ValueError(f"场景 CSV 图片不存在于选定评估组或重名：{name}")
                target = candidates[0]
                if target in metadata:
                    raise ValueError(f"场景 CSV 同一图片重复出现：{name}")
                metadata[target] = {k: str(v or "").strip() for k, v in row.items() if k and k != "image"}
        sources["metadata"] = {
            "file": str(metadata_path),
            "sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        }
    return {"context": context, "metadata": metadata, "sources": sources, "origin": "用户提供，未独立验证"}


def build_diagnosis(report):
    """所有确证均限定到提供的数据和运行条件；输出可追溯的证据、假设与实验。"""
    audit, models, samples = report["audit"], report["models"], report["samples"]
    support = report.get("support", {})
    context = support.get("context", {})
    current = models[0]
    counts = current["thresholds"][str(report["settings"]["conf"])]
    facts, hypotheses, actions = [], [], []

    def fact(title, evidence, scope, cases=None, category="表现"):
        identifier = f"E{len(facts) + 1:03}"
        facts.append(
            {
                "id": identifier,
                "title": title,
                "evidence": evidence,
                "scope": scope,
                "cases": cases or [],
                "category": category,
            }
        )
        return identifier

    def hypothesis(title, refs, missing, alternatives, experiment):
        identifier = f"H{len(hypotheses) + 1:03}"
        hypotheses.append(
            {
                "id": identifier,
                "title": title,
                "evidence_ids": refs,
                "status": "待验证假设",
                "missing": missing,
                "alternatives": alternatives,
                "experiment": experiment,
            }
        )
        return identifier

    def action(title, refs, steps, acceptance, kind="数据集", priority="优先"):
        actions.append(
            {
                "id": f"A{len(actions) + 1:03}",
                "title": title,
                "evidence_ids": refs,
                "kind": kind,
                "priority": priority,
                "steps": steps,
                "acceptance": acceptance,
                "status": "建议，尚未执行",
            }
        )

    overlap = [g for g in audit["duplicates"] if len({x["split"] for x in g}) > 1]
    if overlap:
        ref = fact(
            "所提供数据的分组存在完全重复图片",
            f"{len(overlap)} 组图片 SHA-256 相同且跨 train/val/test。",
            "这是数据划分问题；尚未证明这些图片进入过两模型的实际训练，不能据此确诊模型泄漏。",
            category="数据问题",
        )
        action(
            "建立按采集批次隔离的评估集",
            [ref],
            [
                "先保存原划分和文件哈希。",
                "将同一视频/拍摄批次放在同一分组，人工检查近重复。",
                "固定新验证集与测试集，重新评估两份模型。",
            ],
            "跨组完全重复为 0，并人工核验批次隔离；重新报告每类 P/R、FN/FP。",
        )
    for key, title in [("conflicting_labels", "相同图片存在不同标注"), ("issues", "提供的数据有标签/列表问题")]:
        if audit[key]:
            ref = fact(
                title,
                f"{len(audit[key])} 项，原始文件和行号见数据审计。",
                "只确认文件内容异常；正确标签版本仍需人工核实。",
                category="数据问题",
            )
            action(
                "逐项复核标注，另存修订版本",
                [ref],
                [
                    "核对类别定义、完整物体框和遮挡口径。",
                    "记录原标签、修订标签、复核人和理由；不要用预测自动覆盖真值。",
                ],
                "格式问题清零；争议样本完成复核；原版本可回溯。",
            )

    classes = []
    for c, name in enumerate(report["names"]):
        row = counts[c]
        cases = [s["id"] for s, ev in zip(samples, current["evidence"]) if any(r[0] == c for r in ev["fn"] + ev["fp"])]
        nimages = sum(any(int(g[0]) == c for g in s["gt"]) for s in samples)
        reg, improved = [], []
        if len(models) > 1:
            for s, a, b in zip(samples, current["evidence"], models[1]["evidence"]):
                ka = {tuple(x) for x in a["fn"] if x[0] == c}
                kb = {tuple(x) for x in b["fn"] if x[0] == c}
                reg.extend([s["id"]] * len(ka - kb))
                improved.extend([s["id"]] * len(kb - ka))
        classes.append(
            {
                "class": name,
                "instances": row["tp"] + row["fn"],
                "images": nimages,
                "regressed": len(reg),
                "improved": len(improved),
                "fn": row["fn"],
                "fp": row["fp"],
                "coverage": "未覆盖" if not nimages else "不足以外推" if nimages < 30 else "需核实采样代表性",
            }
        )
        if row["fn"] or row["fp"]:
            ref = fact(
                f"{name}：FN {row['fn']}，FP {row['fp']}",
                f"TP {row['tp']}；{nimages} 张图片、{row['tp'] + row['fn']} 个标注目标。较对照逐目标退步 {len(reg)}、改善 {len(improved)}。"
                if len(models) > 1
                else f"TP {row['tp']}；{nimages} 张图片、{row['tp'] + row['fn']} 个标注目标，未提供对照。",
                "仅对应当前评估集、标签和阈值；30 张只是样本提示，不是统计保证。",
                cases,
            )
            if row["fn"]:
                action(
                    f"复核 {name} 的难例并补充同条件正反样本",
                    [ref],
                    [
                        "打开证据图，区分错分类、定位不足、遮挡与真漏检。",
                        "按批次收集失败与成功场景；补充相似类别和纯背景负样本。",
                        "复核后加入训练候选集；原回归集与独立测试集继续保留。",
                    ],
                    "同参数回归集 FN 减少，同时检查 FP 是否增加；在另一个独立批次验证，不只看已复核图片。",
                )
        for p, pred_name in enumerate(report["names"]):
            n = int(current["confusion"][p][c])
            if p != c and n:
                pair_cases = [
                    s["id"]
                    for s, ev in zip(samples, current["evidence"])
                    if any(x["true"] == c and x["pred"] == p for x in ev.get("confusions", []))
                ]
                ref = fact(
                    f"{name} 被匹配为 {pred_name}",
                    f"原生混淆矩阵记录 {n} 次错分类。",
                    "相对于所给标签和几何匹配成立；不是视觉相似度证明。",
                    pair_cases,
                )
                hypothesis(
                    f"{name} / {pred_name} 的区分特征、样本覆盖或标注口径可能不足",
                    [ref],
                    ["两类原始训练样本及版本", "同视角同光照的难例与成功例", "经人工确认的类别规则和标注"],
                    ["目标尺寸/遮挡变化", "现场预处理与训练不同", "评估标签有争议"],
                    "先复核同场景两类目标；固定划分和推理条件。若仅补充这两类难例训练，需保留其余训练参数并比较多个随机种子。",
                )
    signals = defaultdict(lambda: {"count": 0, "images": set()})
    for sample, ev in zip(samples, current["evidence"]):
        for miss in ev.get("fn_context", []):
            candidate, low = miss.get("candidate"), miss.get("low_conf_candidate")
            name = report["names"][int(miss["gt"][0])]
            if candidate and candidate["class"] == miss["gt"][0]:
                signal = (
                    "同类候选框重合不足"
                    if candidate["iou"] <= report["settings"]["match_iou"]
                    else "同类框存在但匹配竞争"
                )
                item = signals[(name, signal)]
                item["count"] += 1
                item["images"].add(sample["id"])
            if low and low["iou"] > report["settings"]["match_iou"]:
                item = signals[(name, "阈值以下有同类重叠候选")]
                item["count"] += 1
                item["images"].add(sample["id"])
    for (name, signal), item in signals.items():
        ref = fact(
            f"{name}：{signal}",
            f"{item['count']} 个未匹配目标存在此线索，分布在 {len(item['images'])} 张图片。",
            "这是候选框几何与置信度事实；不替代一对一匹配，不保证降低阈值或放宽 IoU 后检出。",
            sorted(item["images"]),
            "候选线索",
        )
        if signal == "同类候选框重合不足":
            hypothesis(
                f"{name} 可能存在框口径、分辨率或定位学习问题",
                [ref],
                ["遮挡与完整物体框规范", "训练和现场目标像素尺寸", "同图不同尺寸的预测"],
                ["候选框实际属于相邻物体", "局部特征被识别但物体不完整", "标注有争议"],
                "先核对完整框/可见框口径；固定模型和图，只改变推理尺寸，比较同一目标 IoU 与 FN；标签口径统一后才决定重训。",
            )
        elif signal == "阈值以下有同类重叠候选":
            hypothesis(
                f"{name} 的显示阈值可能掩盖候选",
                [ref],
                ["现场真实 conf", "目标类别的误报容忍度"],
                ["更高重合的错类框仍占据匹配", "低置信预测本身不可靠"],
                "查看报告阈值表和逐框预测；在验证集逐档调整 conf，同时检查 FP 和 FN，不只追求召回。",
            )

    parameter_differences = []
    if len(models) > 1:
        for key in (
            "epochs",
            "imgsz",
            "batch",
            "optimizer",
            "lr0",
            "lrf",
            "weight_decay",
            "seed",
            "deterministic",
            "hsv_h",
            "hsv_s",
            "hsv_v",
            "mosaic",
            "mixup",
            "copy_paste",
            "close_mosaic",
            "degrees",
            "translate",
            "scale",
            "fliplr",
        ):
            a, b = current.get("train_args") or {}, models[1].get("train_args") or {}
            if key in a and key in b and a[key] != b[key]:
                parameter_differences.append({"parameter": key, "current": a[key], "baseline": b[key]})
        if parameter_differences:
            fact(
                "两模型提供的训练参数存在差异",
                "；".join(f"{x['parameter']}: {x['current']} / {x['baseline']}" for x in parameter_differences),
                "来自权重或所选 args.yaml；参数差异不证明导致错误，未核验训练日志与权重关联。",
                category="训练记录",
            )
    total = sum(x["tp"] + x["fn"] for x in counts)
    if total:
        action(
            "先做推理与数据对照，再决定是否重训",
            [],
            [
                "用现场 imgsz/conf 和预处理复现，保存两模型逐框输出。",
                "固定模型与图片，只改变 imgsz；再单独改变 conf，检查召回和误报取舍。",
                "确认标签和数据覆盖问题后，再单独比较数据修订或一个训练参数；保留模型、数据哈希和随机种子。",
            ],
            "预先填写业务最低 P/R 与误报容忍度；验证集选参数，独立测试集只做最终验收；不以 mAP 单独验收。",
            "实验/重训",
            "先执行",
        )

    deployment = context.get("deployment", {})
    if isinstance(deployment, dict):
        mismatches = []
        for key, setting in [("imgsz", "imgsz"), ("conf", "conf"), ("iou", "nms_iou"), ("max_det", "max_det")]:
            value = deployment.get(key)
            if value not in (None, ""):
                try:
                    if float(value) != float(report["settings"][setting]):
                        mismatches.append(f"{key}：用户填报现场 {value}，本报告 {report['settings'][setting]}")
                except (TypeError, ValueError):
                    mismatches.append(f"{key}：现场值无法数值比较")
        if mismatches:
            ref = fact(
                "填报的现场参数与本次分析不同",
                "；".join(mismatches),
                "现场值来自用户陈述，尚未核验现场程序。",
                category="配置差异",
            )
            hypothesis(
                "离线分析尚未覆盖现场推理条件",
                [ref],
                ["现场实际加载的配置与预处理输出"],
                ["模型本身变化", "场景分布变化"],
                "用同一张原图、相同预处理、相同推理版本重跑，再逐项改变参数定位影响。",
            )

    curves = []
    for m in models:
        history = [{k.strip(): v for k, v in row.items() if k} for row in m.get("history", [])]
        key = next((k for k in (history[0] if history else {}) if k.startswith("metrics/mAP50-95")), None)
        values = []
        if key:
            for row in history:
                try:
                    v = float(row[key])
                    if 0 <= v <= 1:
                        values.append(v)
                except (ValueError, KeyError, TypeError):
                    continue
        if values:
            peak, last = max(values), values[-1]
            curves.append(
                {
                    "model": m["title"],
                    "rows": len(history),
                    "valid_ap_rows": len(values),
                    "peak": peak,
                    "last": last,
                    "source": m.get("training_sources", {}),
                    "limit": "CSV 与权重的关系未独立验证；最后 epoch 不一定是 best.pt 对应 epoch。",
                }
            )
            if len(values) >= 10 and peak - last >= 0.03:
                ref = fact(
                    f"{m['title']}：提供的训练曲线后期 AP 低于峰值",
                    f"有效点 {len(values)}，峰值 {peak:.4f}，末点 {last:.4f}。",
                    "这是 CSV 中的数值变化；10 个点和 0.03 为提示阈值，不是过拟合检验。",
                    category="训练记录",
                )
                hypothesis(
                    "训练后期可能退化，尚不能确诊过拟合",
                    [ref],
                    ["完整损失曲线与日志", "best/last 权重及数据版本", "多个种子结果"],
                    ["验证波动", "数据/增强变化", "记录与权重不对应"],
                    "在同一独立验证集比较 best 与 last；联合训练/验证损失和重复种子判断，不直接加 epoch。",
                )

    scene_groups = defaultdict(lambda: {"images": set(), "instances": 0, "fn": 0})
    for s, ev in zip(samples, current["evidence"]):
        for dim, value in support.get("metadata", {}).get(s["path"], {}).items():
            if value and dim in {
                "batch",
                "scene",
                "lighting",
                "occlusion",
                "view",
                "distance",
                "known_failure",
                "reviewed",
            }:
                for c in {int(r[0]) for r in s["gt"]}:
                    stat = scene_groups[(dim, value, c)]
                    stat["images"].add(s["id"])
                    stat["instances"] += sum(int(r[0]) == c for r in s["gt"])
                    stat["fn"] += sum(int(r[0]) == c for r in ev["fn"])
    scene_slices = [
        {
            "dimension": d,
            "value": v,
            "class": report["names"][c],
            "images": len(x["images"]),
            "instances": x["instances"],
            "fn": x["fn"],
            "recall": 1 - x["fn"] / x["instances"],
        }
        for (d, v, c), x in sorted(scene_groups.items())
    ]
    present = {
        "evaluation": True,
        "baseline": len(models) > 1,
        "training_data": bool(audit.get("image_counts", {}).get("train")),
        "training": any(m.get("history") for m in models),
        "context": bool(context),
        "deployment": bool(deployment),
        "field": any(
            str(x.get("known_failure", "")).lower() in {"true", "1", "yes", "是"}
            for x in support.get("metadata", {}).values()
        ),
        "metadata": bool(support.get("metadata")),
        "experiments": bool(context.get("experiments")),
    }
    coverage = [
        {
            **item,
            "status": ("已提供，仍需核实来源" if present[item["id"]] else "未提供 / 未核实")
            if item["id"] in present
            else "需人工准备或复核",
        }
        for item in DATA_GUIDE
    ]
    return {
        "version": 2,
        "scope": "诊断基于当前标签与已提供数据；确证表现不等于确证训练根因。",
        "root_causes": [],
        "root_cause_status": "尚无经过受控实验与原始证据复核的已确诊根因。",
        "facts": facts,
        "hypotheses": hypotheses,
        "actions": actions,
        "coverage": coverage,
        "class_coverage": classes,
        "scene_slices": scene_slices,
        "training_curves": curves,
        "parameter_differences": parameter_differences,
        "methods": METHODS,
        "sample_count": len(samples),
        "limitations": [
            "补充资料越多可检验的解释越多，但矛盾或不相关资料不会提高准确性。",
            "完整性是资料覆盖情况，不是诊断准确率或病因概率。",
            "同图的多个目标具有相关性，本版未报告总体显著性或假设性的独立样本置信区间。",
            "AI 结论必须引用证据编号并人工核实，不能自动升级为确诊。",
        ],
    }
