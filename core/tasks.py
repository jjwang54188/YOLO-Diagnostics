"""任务管理：保存配置、启动分析子进程、记录日志、取消任务。

这里不计算指标；指标始终由 analyzer.py 负责。一次只运行一个分析，避免争抢显存。
"""

import json
import os
import re
import signal
import subprocess
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def save_json(path, value):
    """先写临时文件再替换，避免中途退出留下半份 JSON。"""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


class TaskManager:
    def __init__(self, root):
        self.root = root
        self.storage = root / "data" / "jobs"
        self.storage.mkdir(parents=True, exist_ok=True)
        (root / "reports").mkdir(exist_ok=True)
        self.lock = threading.RLock()
        self.process = None
        self.active = None
        self.jobs = {}
        for path in self.storage.glob("*.json"):
            job = json.loads(path.read_text(encoding="utf-8"))
            if job["status"] in {"running", "cancelling"}:
                job.update(status="interrupted", error="程序上次退出时任务未完成，请重新运行。", ended=now())
                save_json(path, job)
            self.jobs[job["id"]] = job

    def settings(self):
        path = self.root / "data" / "settings.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return {
            "name": "模型问题分析",
            "model": "",
            "baseline": "",
            "data": "",
            "split": "test",
            "imgsz": 1024,
            "conf": 0.25,
            "match_iou": 0.5,
            "device": "cpu",
        }

    def validate(self, config):
        """只接受界面实际支持的设置；文件路径使用参数列表传给 Python，不执行 shell。"""
        if not isinstance(config, dict):
            raise TypeError("任务配置必须是对象。")
        clean = {
            key: str(config.get(key, "")).strip() for key in ("name", "model", "baseline", "data", "split", "device")
        }
        clean["name"] = clean["name"][:100] or "未命名分析"
        for key, suffixes, title in (
            ("model", {".pt"}, "问题模型"),
            ("baseline", {".pt"}, "对照模型"),
            ("data", {".yaml", ".yml"}, "数据集配置"),
        ):
            if key == "baseline" and not clean[key]:
                continue
            if not clean[key]:
                raise ValueError(f"请先选择{title}。")
            path = Path(clean[key]).expanduser().resolve()
            if not path.is_file() or path.suffix.lower() not in suffixes:
                raise ValueError(f"{title}不存在或格式不正确：{path}")
            clean[key] = str(path)
        clean["model_family"] = str(config.get("model_family", "auto")).strip() or "auto"
        if clean["model_family"] not in {"auto", "v8", "v9", "v10", "v11", "v12", "v26"}:
            raise ValueError("YOLO 系列请选择自动识别、v8、v9、v10、v11、v12 或 v26。")
        clean["context_text"] = str(config.get("context_text", "")).strip()[:12000]
        for key in ("context", "metadata", "train_run", "baseline_run"):
            value = str(config.get(key, "")).strip()
            clean[key] = ""
            if value:
                path = Path(value).expanduser().resolve()
                is_dir = key in {"train_run", "baseline_run"}
                if not (path.is_dir() if is_dir else path.is_file()):
                    raise ValueError(f"补充资料路径不存在：{key} {path}")
                if not is_dir and path.suffix.lower() != (".json" if key == "context" else ".csv"):
                    raise ValueError("补充资料使用 JSON，场景元数据使用 CSV。")
                if is_dir and not (path / "results.csv").is_file():
                    raise ValueError("训练结果目录中没有 results.csv。")
                clean[key] = str(path)
        if clean["split"] not in {"test", "val"}:
            raise ValueError("请选择 test 或 val 分组。")
        if clean["device"] not in {"cpu", "0"}:
            raise ValueError("请选择 CPU 或第一个 GPU。")
        clean["imgsz"] = int(config["imgsz"])
        clean["conf"] = float(config["conf"])
        clean["match_iou"] = float(config["match_iou"])
        if not 32 <= clean["imgsz"] <= 4096 or clean["imgsz"] % 32:
            raise ValueError("推理尺寸应为 32 的整数倍，范围 32～4096。")
        if not 0.001 < clean["conf"] < 1 or not 0 < clean["match_iou"] < 1:
            raise ValueError("置信度应在 0.001～1 之间，匹配 IoU 应在 0～1 之间（均不含端点）。")
        return clean

    def start(self, config):
        config = self.validate(config)
        with self.lock:
            if self.active:
                raise ValueError("已有分析正在运行，请等待结束或先取消。")
            identifier = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
            output = self.root / "reports" / identifier
            command = [sys.executable, "-u", str(self.root / "core" / "analyzer.py")]
            context_text = config.pop("context_text", "")
            if context_text:
                context_path = self.root / "data" / "contexts" / f"{identifier}.json"
                context_path.parent.mkdir(parents=True, exist_ok=True)
                context = json.loads(Path(config["context"]).read_text(encoding="utf-8")) if config.get("context") else {}
                context["problem_description"] = context_text
                save_json(context_path, context)
                config["context_text"] = context_text
            for key in (
                "model_family",
                "model",
                "baseline",
                "data",
                "split",
                "imgsz",
                "conf",
                "match_iou",
                "device",
                "context",
                "metadata",
                "train_run",
                "baseline_run",
            ):
                if config.get(key, "") != "" and not (key == "context" and context_text):
                    command.extend(["--" + key.replace("_", "-"), str(config[key])])
            if context_text:
                command.extend(["--context", str(context_path)])
            command.extend(["--output", str(output)])
            job = {
                "id": identifier,
                "name": config["name"],
                "status": "running",
                "created": now(),
                "config": config,
                "progress": {"phase": "检查数据与准备图片", "done": 0, "total": 0},
                "report": None,
                "error": "",
                "output": str(output),
            }
            environment = os.environ.copy()
            environment["YOLO_CONFIG_DIR"] = str(self.root / "data" / "runtime")
            environment["YOLO_AUTOINSTALL"] = "false"
            (self.root / "data" / "runtime" / "Ultralytics").mkdir(parents=True, exist_ok=True)
            self.process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=environment,
                cwd=self.root,
                start_new_session=True,
            )
            self.active = identifier
            self.jobs[identifier] = job
            save_json(self.storage / f"{identifier}.json", job)
            save_json(self.root / "data" / "settings.json", config)
            threading.Thread(target=self._watch, args=(identifier, self.process), daemon=True).start()
            return identifier

    def _watch(self, identifier, process):
        """持续读 stdout，进度来自已处理图片数；没有拿计时器伪造百分比。"""
        with (self.storage / f"{identifier}.log").open("w", encoding="utf-8") as log:
            for line in process.stdout:
                log.write(line)
                log.flush()
                match = re.search(r"进度 \| (.+) \| (\d+)/(\d+) 张图片", line)
                if match:
                    with self.lock:
                        self.jobs[identifier]["progress"] = {
                            "phase": match[1],
                            "done": int(match[2]),
                            "total": int(match[3]),
                        }
            code = process.wait()
        with self.lock:
            job = self.jobs[identifier]
            if job["status"] == "cancelling":
                job["status"] = "cancelled"
            elif code == 0 and (Path(job["output"]) / "index.html").exists():
                job.update(status="completed", report=f"/reports/{identifier}/index.html")
            else:
                job.update(status="failed", error="分析未完成，请查看下方日志末尾的具体原因。")
            job.update(ended=now(), exit_code=code)
            save_json(self.storage / f"{identifier}.json", job)
            self.active = None
            self.process = None

    def cancel(self, identifier):
        with self.lock:
            if identifier != self.active or self.jobs[identifier]["status"] != "running":
                raise ValueError("该任务当前不在运行。")
            self.jobs[identifier]["status"] = "cancelling"
            process = self.process
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass  # 任务恰好在点击取消时结束。
            # 等待退出的线程不占用 HTTP 请求；只对这个分析进程组升级终止。
            threading.Thread(target=self._finish_cancel, args=(process,), daemon=True).start()

    @staticmethod
    def _finish_cancel(process):
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def snapshot(self, identifier=None):
        with self.lock:
            if identifier:
                job = json.loads(json.dumps(self.jobs[identifier]))
                path = self.storage / f"{identifier}.log"
                if path.exists():
                    with path.open("rb") as log:
                        log.seek(max(0, path.stat().st_size - 60000))
                        job["log"] = log.read().decode("utf-8", errors="replace")
                else:
                    job["log"] = ""
                return job
            return json.loads(
                json.dumps(
                    {
                        "active": self.active,
                        "jobs": sorted(self.jobs.values(), key=lambda x: x["created"], reverse=True),
                    }
                )
            )
