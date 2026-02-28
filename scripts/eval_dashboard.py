"""
eval_dashboard.py — Dashboard real-time per auto_eval (porta 8020).

Legge i JSONL in output/eval_sessions/ e li espone via API.
Serve ui/eval_dashboard.html.

Avvio:
  python scripts/eval_dashboard.py
  → http://localhost:8020/
"""

import glob
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT       = Path(__file__).resolve().parent.parent
UI_DIR     = ROOT / "ui"
SESSIONS_DIR = ROOT / "output" / "eval_sessions"
PORT       = 8020


def load_session(jsonl_path: Path) -> dict:
    """Legge un file JSONL di sessione e costruisce la struttura dati per la UI."""
    entries = []
    try:
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except FileNotFoundError:
        return {}

    if not entries:
        return {}

    session = {
        "session_id": "",
        "concept": "",
        "model": "",
        "available_layers": [],
        "best_layer": None,
        "max_probes": 0,
        "turns_per_block": 0,
        "status": "running",
        "turns": [],
        "probe_analyses": [],
        "report": None,
        "jsonl_file": jsonl_path.name,
    }

    for e in entries:
        t = e.get("type")

        if t == "session_start":
            session.update({
                "session_id":      e.get("session_id", ""),
                "concept":         e.get("concept", ""),
                "model":           e.get("model", ""),
                "available_layers": e.get("available_layers", []),
                "best_layer":      e.get("best_layer"),
                "max_probes":      e.get("max_probes", 0),
                "turns_per_block": e.get("turns_per_block", 0),
                "timestamp_start": e.get("timestamp", ""),
            })

        elif t == "turn":
            session["turns"].append({
                "probe":          e.get("probe"),
                "turn_global":    e.get("turn_global"),
                "turn_in_block":  e.get("turn_in_block"),
                "alpha_label":    e.get("alpha_label"),
                "alpha":          e.get("alpha"),
                "layer_configs":  e.get("layer_configs", []),
                "prompt":         e.get("prompt", ""),
                "response":       e.get("response", ""),
                "evaluation":     e.get("evaluation", {}),
                "timestamp":      e.get("timestamp", ""),
            })

        elif t == "probe_analysis":
            session["probe_analyses"].append({
                "probe":         e.get("probe"),
                "layer_configs": e.get("layer_configs", []),
                "analysis":      e.get("analysis", {}),
                "timestamp":     e.get("timestamp", ""),
            })

        elif t == "session_end":
            session["status"] = "completed"
            session["timestamp_end"] = e.get("timestamp", "")

    # Leggi il report markdown se esiste
    report_path = jsonl_path.with_name(jsonl_path.stem + "_report.md")
    if report_path.exists():
        session["report"] = report_path.read_text(encoding="utf-8")

    return session


def get_sessions() -> list[dict]:
    """Ritorna lista di sessioni ordinate per data (più recente prima)."""
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(
        SESSIONS_DIR.glob("session_*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    result = []
    for f in files[:20]:   # ultime 20 sessioni
        try:
            s = load_session(f)
            if s:
                result.append(s)
        except Exception:
            pass
    return result


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # silenzia i log HTTP

    def _send(self, code: int, content: bytes, content_type: str = "application/json"):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        parsed = urlparse(self.path)
        path   = parsed.path

        if path in ("/", "/index.html"):
            html_path = UI_DIR / "eval_dashboard.html"
            if html_path.exists():
                content = html_path.read_bytes()
                self._send(200, content, "text/html; charset=utf-8")
            else:
                self._send(404, b"eval_dashboard.html not found")
            return

        if path == "/api/sessions":
            data = get_sessions()
            self._send(200, json.dumps(data, ensure_ascii=False).encode())
            return

        if path == "/api/latest":
            sessions = get_sessions()
            data = sessions[0] if sessions else {}
            self._send(200, json.dumps(data, ensure_ascii=False).encode())
            return

        self._send(404, b"Not found")


def main():
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"Eval dashboard → http://localhost:{PORT}/")
    print(f"Sessioni dir   : {SESSIONS_DIR}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStop.")


if __name__ == "__main__":
    main()
