"""本地应用入口与 HTTP 接口。运行方式：.venv/bin/python main.py --open。

只监听 127.0.0.1；界面通过随机令牌访问本地文件选择和任务接口。
"""

import argparse
import hashlib
import json
import mimetypes
import secrets
import signal
import urllib.error
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

import yaml
from core.ai import AIService
from core.diagnosis import DATA_GUIDE, METHODS
from core.tasks import TaskManager

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="YOLO 诊断助手：本地图形界面")
    parser.add_argument("--port", type=int, default=8880)
    parser.add_argument("--open", action="store_true", help="自动打开浏览器")
    args = parser.parse_args()
    token = secrets.token_urlsafe(32)
    instance = hashlib.sha256(str(ROOT).encode()).hexdigest()[:16]
    base_url = f"http://127.0.0.1:{args.port}"
    manager = None
    ai = None

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *values):
            pass  # 任务日志另存；不把轮询请求刷满终端。

        def send(self, body, status=200, content_type="application/json; charset=utf-8"):
            if isinstance(body, (dict, list)):
                body = json.dumps(body, ensure_ascii=False).encode()
            if isinstance(body, str):
                body = body.encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.end_headers()
            self.wfile.write(body)

        def body(self):
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= 65536:
                raise ValueError("请求大小不正确。")
            value = json.loads(self.rfile.read(length))
            if not isinstance(value, dict):
                raise TypeError("请求必须为 JSON 对象。")
            return value

        def do_GET(self):
            self.route()

        def do_POST(self):
            self.route()

        def route(self):
            if self.headers.get("Host") not in {f"127.0.0.1:{args.port}", f"localhost:{args.port}"}:
                self.send({"error": "仅允许本机访问。"}, 403)
                return
            path = unquote(urlsplit(self.path).path)
            query = parse_qs(urlsplit(self.path).query)
            if path.startswith("/api/") and self.headers.get("X-Diagnostics-Token") != token:
                self.send({"error": "请在本机重新打开应用页面。"}, 403)
                return
            try:
                if path == "/health" and self.command == "GET":
                    self.send({"app": "yolo-diagnostics", "instance": instance})
                elif path == "/" and self.command == "GET":
                    self.send(
                        (ROOT / "ui" / "index.html").read_text().replace("__APP_TOKEN__", token),
                        content_type="text/html; charset=utf-8",
                    )
                elif path == "/api/state" and self.command == "GET":
                    demo_path = ROOT / "data" / "demo.json"
                    demo = json.loads(demo_path.read_text()) if demo_path.exists() else None
                    self.send(
                        {**manager.snapshot(), "settings": manager.settings(), "home": str(Path.home()), "demo": demo}
                    )
                elif path == "/api/browse" and self.command == "GET":
                    directory = Path(query.get("path", [str(Path.home())])[0]).expanduser().resolve()
                    if not directory.is_dir():
                        raise ValueError("请选择一个存在的文件夹。")
                    kind = query.get("kind", ["data"])[0]
                    suffixes = {
                        "model": {".pt"},
                        "data": {".yaml", ".yml"},
                        "context": {".json"},
                        "metadata": {".csv"},
                        "directory": set(),
                    }.get(kind, set())
                    entries = [
                        {"name": p.name, "path": str(p), "directory": p.is_dir()}
                        for p in directory.iterdir()
                        if not p.name.startswith(".") and (p.is_dir() or p.suffix.lower() in suffixes)
                    ]
                    self.send(
                        {
                            "path": str(directory),
                            "parent": str(directory.parent),
                            "entries": sorted(entries, key=lambda x: (not x["directory"], x["name"].lower())),
                        }
                    )
                elif path == "/api/dataset" and self.command == "GET":
                    source = Path(query.get("path", [""])[0]).expanduser()
                    if not source.is_file() or source.suffix.lower() not in {".yaml", ".yml"}:
                        raise ValueError("请选择数据集 YAML 文件。")
                    data = yaml.safe_load(source.read_text())
                    if not isinstance(data, dict) or "names" not in data:
                        raise ValueError("该 YAML 不是包含 names 类别列表的数据集配置。")
                    self.send({key: data.get(key) for key in ("names", "path", "train", "val", "test")})
                elif path == "/api/jobs" and self.command == "POST":
                    self.send({"id": manager.start(self.body())}, 201)
                elif path == "/api/guide" and self.command == "GET":
                    self.send({"data": DATA_GUIDE, "methods": METHODS})
                elif path == "/api/ai/settings":
                    self.send(ai.configure(self.body()) if self.command == "POST" else ai.settings())
                elif path == "/api/ai/preview" and self.command == "POST":
                    body = self.body()
                    self.send(ai.preview(manager.snapshot(body["job_id"]), body))
                elif path == "/api/ai/send" and self.command == "POST":
                    body = self.body()
                    self.send(ai.send(body["preview_id"], body.get("consent")))
                elif path == "/api/ai/history" and self.command == "GET":
                    job = manager.snapshot(query.get("job_id", [""])[0])
                    self.send(ai.history(job["id"]))
                elif path == "/api/ai/images" and self.command == "GET":
                    job = manager.snapshot(query.get("job_id", [""])[0])
                    if job["status"] != "completed":
                        raise ValueError("任务尚未完成。")
                    report = json.loads((Path(job["output"]) / "report.json").read_text())
                    self.send(
                        [
                            {"id": s["id"], "name": Path(s["path"]).name, "url": f"/reports/{job['id']}/{s['image']}"}
                            for s in report["samples"]
                        ]
                    )
                elif path.startswith("/api/jobs/"):
                    parts = path.split("/")
                    if len(parts) == 5 and parts[4] == "cancel" and self.command == "POST":
                        manager.cancel(parts[3])
                        self.send({"ok": True})
                    elif len(parts) == 4 and self.command == "GET":
                        self.send(manager.snapshot(parts[3]))
                    else:
                        self.send({"error": "接口不存在。"}, 404)
                elif self.command == "GET" and path.startswith(("/ui/", "/reports/", "/templates/")):
                    folder = ROOT / path.split("/")[1]
                    file = (ROOT / path.lstrip("/")).resolve()
                    if not file.is_relative_to(folder.resolve()):
                        self.send({"error": "文件不在可查看目录中。"}, 403)
                        return
                    if file.is_dir():
                        file = file / "index.html"
                    if not file.is_file():
                        raise FileNotFoundError("文件不存在。")
                    self.send(
                        file.read_bytes(), content_type=mimetypes.guess_type(file.name)[0] or "application/octet-stream"
                    )
                else:
                    self.send({"error": "页面不存在。"}, 404)
            except (ValueError, KeyError, TypeError, OSError, yaml.YAMLError) as exc:
                self.send({"error": str(exc)}, 400)

    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    except OSError:
        try:
            with urllib.request.urlopen(base_url + "/health", timeout=2) as response:
                existing = json.load(response)
        except (OSError, ValueError, urllib.error.URLError):
            existing = {}
        if existing.get("instance") == instance:
            print(f"应用已经启动：{base_url}")
            if args.open:
                webbrowser.open(base_url)
            return
        raise SystemExit(f"端口 {args.port} 已被其他程序占用。请使用 --port 8881 更换端口。")
    manager = TaskManager(ROOT)
    ai = AIService(ROOT)

    def stop_server(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop_server)
    print(f"YOLO 诊断助手已启动：{base_url}", flush=True)
    if args.open:
        webbrowser.open(base_url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        if manager.active:
            manager.cancel(manager.active)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
