# -*- coding: utf-8 -*-
"""ETF 策略仪表盘本地服务(零依赖, Python 标准库)
- GET  /            仪表盘页面
- GET  /api/status  当前数据状态(不触发刷新), 供页面轮询
- GET  /api/data    返回最新数据; 若今天尚未生成则自动刷新(拉数据→重建面板→生成JSON)
- POST /api/refresh 强制刷新
用法: python3 serve_dashboard.py [--port 8787] [--no-auto]
环境变量: ETF_REPORT_DIR 覆盖工作目录(默认 ~/ETF策略日报)
说明: 网页打开自动刷新只更新数据与JSON, 不做 git 提交; 每日 09:00 的 Codex 定时任务仍负责 README/git 同步"""
import os, sys, json, threading, subprocess, datetime as dt
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
SKILL = os.environ.get("ETF_SKILL_DIR", os.path.expanduser("~/.codex/skills/etf-dynamic-allocation"))
SCRIPTS = os.path.join(SKILL, "scripts")
WORK = os.environ.get("ETF_REPORT_DIR", os.path.expanduser("~/ETF策略日报"))
DATA_DIR = os.path.join(WORK, "data")
DASH_JSON = os.path.join(WORK, "dashboard.json")
GEN = os.path.join(HERE, "gen_dashboard.py")
PORT = 8787
AUTO_REFRESH = True

_lock = threading.Lock()
_state = {"refreshing": False, "last": "", "log": []}


def run(cmd, cwd=None, env=None, timeout=900):
    e = dict(os.environ)
    e["ETF_DATA_DIR"] = DATA_DIR
    e["ETF_SKILL_DIR"] = SKILL
    if env: e.update(env)
    r = subprocess.run(cmd, cwd=cwd, env=e, capture_output=True, text=True, timeout=timeout)
    return r.returncode, (r.stdout or "")[-800:], (r.stderr or "")[-800:]


def dashboard_meta():
    if not os.path.exists(DASH_JSON):
        return None
    try:
        d = json.load(open(DASH_JSON))
        return {"generated_at": d.get("generated_at", ""), "data_date": d.get("data_date", ""),
                "size": os.path.getsize(DASH_JSON)}
    except Exception:
        return None


def is_fresh(meta):
    if not meta:
        return False
    try:
        gd = dt.datetime.strptime(meta["generated_at"], "%Y-%m-%d %H:%M").date()
    except Exception:
        return False
    return gd == dt.date.today()


def refresh():
    with _lock:
        _state["refreshing"] = True
        _state["log"] = []
        steps = [
            ("拉取行情(ETF净值/指数/溢价)", [sys.executable, "-W", "ignore", os.path.join(SCRIPTS, "daily_fetch.py")], WORK),
            ("重建数据面板", [sys.executable, "-W", "ignore", "-c",
                          "import sys; sys.path.insert(0, r'" + SCRIPTS + "'); from data_prep import save_cache; save_cache()"], WORK),
            ("生成仪表盘数据", [sys.executable, "-W", "ignore", GEN, DASH_JSON], HERE),
        ]
        for name, cmd, cwd in steps:
            try:
                rc, out, err = run(cmd, cwd=cwd)
                ok = rc == 0
            except Exception as ex:
                ok, err = False, str(ex)[:200]
            _state["log"].append({"step": name, "ok": ok, "err": err[:200]})
            if not ok:
                _state["refreshing"] = False
                return {"ok": False, "log": _state["log"]}
        _state["refreshing"] = False
        return {"ok": True, "log": _state["log"]}


class Handler(BaseHTTPRequestHandler):
    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path, ctype):
        if not os.path.exists(path):
            self.send_error(404); return
        body = open(path, "rb").read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._file(os.path.join(HERE, "index.html"), "text/html; charset=utf-8")
        elif self.path == "/api/status":
            meta = dashboard_meta()
            self._json({"meta": meta, "fresh": is_fresh(meta), "refreshing": _state["refreshing"],
                        "log": _state["log"][-3:]})
        elif self.path.startswith("/api/data"):
            force = "force=1" in self.path
            meta = dashboard_meta()
            if AUTO_REFRESH and (force or not is_fresh(meta)) and not _state["refreshing"]:
                r = refresh()
                if not r["ok"]:
                    old = meta
                    if os.path.exists(DASH_JSON):
                        d = json.load(open(DASH_JSON))
                        self._json({"data": d, "warning": "刷新失败, 展示上次数据", "refresh": r})
                    else:
                        self._json({"error": "刷新失败且无历史数据", "refresh": r}, 500)
                    return
            try:
                d = json.load(open(DASH_JSON))
            except Exception:
                self._json({"error": "dashboard.json 缺失"}, 500); return
            self._json({"data": d, "refresh": _state["log"][-3:] if _state["log"] else []})
        elif self.path == "/favicon.ico":
            self.send_response(204); self.end_headers()
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/api/refresh":
            if _state["refreshing"]:
                self._json({"ok": False, "msg": "正在刷新中, 请稍候"}, 409); return
            r = refresh()
            self._json({"ok": r["ok"], "log": r["log"]})
        else:
            self.send_error(404)

    def log_message(self, *a):
        pass


def main():
    global PORT, AUTO_REFRESH
    args = sys.argv[1:]
    if "--no-auto" in args:
        AUTO_REFRESH = False
    for i, a in enumerate(args):
        if a == "--port" and i + 1 < len(args):
            PORT = int(args[i + 1])
    os.makedirs(DATA_DIR, exist_ok=True)
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"ETF 策略仪表盘: http://127.0.0.1:{PORT}")
    print(f"工作目录: {WORK} ｜ 自动刷新: {'开' if AUTO_REFRESH else '关'} ｜ Ctrl+C 停止")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")


if __name__ == "__main__":
    main()
