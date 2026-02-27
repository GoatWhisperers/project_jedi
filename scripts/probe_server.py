#!/usr/bin/env python3
"""Probe Dashboard Server — porta 8000

Gestisce l'estrazione di concept vectors via browser.
Non carica modelli ML: lancia probe_concept.py come sottoprocesso.

Avvio:
    project_jedi/.venv/bin/python project_jedi/scripts/probe_server.py
    PROBE_PORT=8000 project_jedi/.venv/bin/python project_jedi/scripts/probe_server.py
"""
import collections
import glob
import json
import os
import re
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT, "config", "settings.json")
OUTPUT_ROOT = os.path.join(ROOT, "output")
STATUS_PATH = os.path.join(OUTPUT_ROOT, "status.json")
VECTOR_LIB_ROOT = os.path.join(OUTPUT_ROOT, "vector_library")
CONCEPTS_DIR = os.path.join(ROOT, "config", "concepts")
PROBE_SCRIPT = os.path.join(ROOT, "scripts", "probe_concept.py")
VENV_PYTHON = os.path.join(ROOT, ".venv", "bin", "python")
UI_PATH = os.path.join(ROOT, "ui", "probe_dashboard.html")
MAX_LOG_LINES = 300

PROBE_PORT = int(os.environ.get("PROBE_PORT", "8000"))


# ---------------------------------------------------------------------------
# StatsCache — background thread, aggiorna ogni 2s
# ---------------------------------------------------------------------------

class StatsCache:
    data: dict = {}
    _lock = threading.Lock()
    _prev_cpu: tuple | None = None

    @classmethod
    def _read_cpu(cls):
        try:
            with open("/proc/stat") as f:
                line = f.readline()
            # cpu user nice system idle iowait irq softirq ...
            fields = list(map(int, line.split()[1:8]))
            total = sum(fields)
            idle = fields[3]  # 4th field = idle
            return total, idle
        except Exception:
            return None, None

    @classmethod
    def _read_ram(cls):
        try:
            info = {}
            with open("/proc/meminfo") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        info[parts[0].rstrip(":")] = int(parts[1])
            total_mb = info.get("MemTotal", 0) / 1024
            avail_mb = info.get("MemAvailable", 0) / 1024
            used_mb = total_mb - avail_mb
            pct = used_mb / total_mb * 100 if total_mb > 0 else 0
            return {
                "total_mb": round(total_mb),
                "used_mb": round(used_mb),
                "pct": round(pct, 1),
            }
        except Exception:
            return {"total_mb": 0, "used_mb": 0, "pct": 0}

    @classmethod
    def _read_gpu(cls):
        try:
            res = subprocess.run(
                ["rocm-smi", "--showtemp", "--showuse", "--showmemuse", "--showmeminfo", "vram", "--json"],
                capture_output=True, text=True, timeout=5,
            )
            data = json.loads(res.stdout)
            card = data.get("card0", {})
            edge     = float(card.get("Temperature (Sensor edge) (C)", 0) or 0)
            junction = float(card.get("Temperature (Sensor junction) (C)", 0) or 0)
            gpu_use  = float(card.get("GPU use (%)", 0) or 0)
            mem_pct  = float(card.get("GPU memory use (%)", 0) or 0)
            vram_total_b = int(card.get("VRAM Total Memory (B)", 0) or 0)
            vram_used_b  = int(card.get("VRAM Total Used Memory (B)", 0) or 0)
            vram_total_gb = round(vram_total_b / 1024**3, 1)
            vram_used_gb  = round(vram_used_b  / 1024**3, 1)
            return {
                "ok": True,
                "edge_c": edge,
                "junction_c": junction,
                "gpu_use_pct": gpu_use,
                "mem_pct": mem_pct,
                "vram_used_gb": vram_used_gb,
                "vram_total_gb": vram_total_gb,
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @classmethod
    def update(cls):
        total, idle = cls._read_cpu()
        cpu_pct = 0.0
        if total is not None and cls._prev_cpu is not None:
            prev_total, prev_idle = cls._prev_cpu
            d_total = total - prev_total
            d_idle = idle - prev_idle
            if d_total > 0:
                cpu_pct = (d_total - d_idle) / d_total * 100
        cls._prev_cpu = (total, idle) if total is not None else cls._prev_cpu

        ram = cls._read_ram()
        gpu = cls._read_gpu()

        with cls._lock:
            cls.data = {
                "cpu_pct": round(cpu_pct, 1),
                "ram": ram,
                "gpu": gpu,
                "ts": time.time(),
            }

    @classmethod
    def get(cls):
        with cls._lock:
            return dict(cls.data)


def _stats_loop():
    while True:
        StatsCache.update()
        time.sleep(2)


# ---------------------------------------------------------------------------
# ProbeState — stato del sottoprocesso di probing
# ---------------------------------------------------------------------------

class ProbeState:
    proc = None
    log = collections.deque(maxlen=MAX_LOG_LINES)
    status = "idle"       # idle | running | done | error
    returncode = None
    selected_model = None
    selected_concept = None
    _lock = threading.Lock()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_settings():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def list_concepts():
    """Glob config/concepts/*.json → lista di dicts con metadati."""
    results = []
    for path in sorted(glob.glob(os.path.join(CONCEPTS_DIR, "*.json"))):
        try:
            with open(path) as f:
                data = json.load(f)
            results.append({
                "name": data.get("concept", os.path.splitext(os.path.basename(path))[0]),
                "category": data.get("category", ""),
                "n_pos": len(data.get("positive", [])),
                "n_neg": len(data.get("negative", [])),
                "path": path,
                "filename": os.path.basename(path),
            })
        except Exception:
            continue
    return results


def get_library_overview():
    """Scan vector_library/*/*/meta.json → summary per ogni concept."""
    entries = []
    for meta_path in sorted(glob.glob(os.path.join(VECTOR_LIB_ROOT, "*/*/*/meta.json"))):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except Exception:
            continue
        dir_path = os.path.dirname(meta_path)
        eval_data = {}
        eval_path = os.path.join(dir_path, "eval.json")
        if os.path.exists(eval_path):
            try:
                with open(eval_path) as f:
                    eval_data = json.load(f)
            except Exception:
                pass
        entries.append({
            "concept":    meta.get("concept", ""),
            "category":   meta.get("category", ""),
            "model":      meta.get("model_name", ""),
            "n_pairs":    meta.get("n_pairs", 0),
            "best_layer": eval_data.get("best_layer"),
            "best_snr":   eval_data.get("best_snr"),
        })
    return entries


def _read_proc_output(proc):
    """Thread: legge stdout dal processo riga per riga, aggiorna ProbeState."""
    try:
        for line in proc.stdout:
            line = line.rstrip("\n")
            with ProbeState._lock:
                ProbeState.log.append(line)
        proc.wait()
        with ProbeState._lock:
            ProbeState.returncode = proc.returncode
            ProbeState.status = "done" if proc.returncode == 0 else "error"
    except Exception as e:
        with ProbeState._lock:
            ProbeState.log.append(f"[reader error] {e}")
            ProbeState.status = "error"


def start_probe(concept_path, model_name=None, run_eval=False):
    """Lancia probe_concept.py come sottoprocesso. Ritorna (ok, msg)."""
    with ProbeState._lock:
        if ProbeState.status == "running":
            return False, "already running"

    cmd = [VENV_PYTHON, PROBE_SCRIPT, "--concept", concept_path]
    if model_name:
        cmd += ["--model", model_name]
    if run_eval:
        cmd.append("--eval")

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=ROOT,
        )
    except Exception as e:
        return False, str(e)

    with ProbeState._lock:
        ProbeState.proc = proc
        ProbeState.log.clear()
        ProbeState.status = "running"
        ProbeState.returncode = None
        ProbeState.selected_concept = os.path.basename(concept_path)

    t = threading.Thread(target=_read_proc_output, args=(proc,), daemon=True)
    t.start()
    return True, "ok"


def stop_probe():
    """Termina il probe corrente. Ritorna (ok, msg)."""
    with ProbeState._lock:
        proc = ProbeState.proc
        if proc is None or ProbeState.status != "running":
            return False, "not running"
    try:
        proc.terminate()
    except Exception:
        pass
    with ProbeState._lock:
        ProbeState.status = "idle"
    return True, "stopped"


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def json_response(handler, code, payload):
    data = json.dumps(payload).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(data)


def text_response(handler, code, content, content_type="text/html"):
    data = content.encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def read_body(handler):
    length = int(handler.headers.get("Content-Length", "0"))
    if length == 0:
        return b""
    return handler.rfile.read(length)


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # sopprimi log HTTP per non intasare stdout del probe

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        if path == "/":
            try:
                with open(UI_PATH) as f:
                    html = f.read()
                return text_response(self, 200, html)
            except FileNotFoundError:
                return text_response(self, 404, "<h1>UI not found — build probe_dashboard.html</h1>")

        if path == "/api/status":
            status_data = {}
            if os.path.exists(STATUS_PATH):
                try:
                    with open(STATUS_PATH) as f:
                        status_data = json.load(f)
                except Exception:
                    pass
            with ProbeState._lock:
                probe_status = ProbeState.status
                selected = ProbeState.selected_concept
            return json_response(self, 200, {
                "status": status_data,
                "probe_status": probe_status,
                "concept": selected,
            })

        if path == "/api/concepts":
            return json_response(self, 200, {"concepts": list_concepts()})

        if path == "/api/models":
            try:
                settings = load_settings()
            except Exception:
                settings = {}
            models = settings.get("models", [])
            with ProbeState._lock:
                active = ProbeState.selected_model
            if active is None:
                mp = settings.get("model_path", "")
                for m in models:
                    if m.get("path") == mp:
                        active = m.get("name")
                        break
            return json_response(self, 200, {"models": models, "active": active})

        if path == "/api/stats":
            return json_response(self, 200, StatsCache.get())

        if path == "/api/log":
            n = int((qs.get("n") or ["100"])[0])
            with ProbeState._lock:
                lines = list(ProbeState.log)[-n:]
                status = ProbeState.status
                rc = ProbeState.returncode
            return json_response(self, 200, {"lines": lines, "status": status, "returncode": rc})

        if path == "/api/library":
            return json_response(self, 200, {"library": get_library_overview()})

        if path == "/api/stop":
            ok, msg = stop_probe()
            return json_response(self, 200, {"ok": ok, "msg": msg})

        return self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/probe":
            body = read_body(self)
            try:
                payload = json.loads(body.decode("utf-8"))
            except Exception:
                return json_response(self, 400, {"error": "invalid_json"})
            concept_path = payload.get("concept_path", "")
            model_name = payload.get("model") or None
            run_eval = bool(payload.get("eval", False))
            if not concept_path:
                return json_response(self, 400, {"error": "concept_path required"})
            ok, msg = start_probe(concept_path, model_name, run_eval)
            code = 200 if ok else 409
            return json_response(self, code, {"ok": ok, "msg": msg})

        if path == "/api/upload_concept":
            body = read_body(self)
            try:
                data = json.loads(body.decode("utf-8"))
            except Exception:
                return json_response(self, 400, {"error": "invalid_json"})
            concept_name = data.get("concept", "")
            if not concept_name:
                return json_response(self, 400, {"error": "campo 'concept' richiesto"})
            safe = re.sub(r"[^a-z0-9_\-]", "", concept_name.lower().replace(" ", "_"))
            if not safe:
                return json_response(self, 400, {"error": "nome concept non valido"})
            os.makedirs(CONCEPTS_DIR, exist_ok=True)
            out_path = os.path.join(CONCEPTS_DIR, f"{safe}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return json_response(self, 200, {
                "ok": True,
                "path": out_path,
                "filename": f"{safe}.json",
            })

        if path == "/api/set_model":
            body = read_body(self)
            try:
                payload = json.loads(body.decode("utf-8"))
            except Exception:
                return json_response(self, 400, {"error": "invalid_json"})
            name = payload.get("name", "")
            with ProbeState._lock:
                ProbeState.selected_model = name
            return json_response(self, 200, {"ok": True, "active": name})

        return self.send_error(404)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    os.makedirs(CONCEPTS_DIR, exist_ok=True)

    # Inizializza modello attivo da settings
    try:
        settings = load_settings()
        mp = settings.get("model_path", "")
        for m in settings.get("models", []):
            if m.get("path") == mp:
                ProbeState.selected_model = m.get("name")
                break
    except Exception:
        pass

    # Avvia thread statistiche
    t = threading.Thread(target=_stats_loop, daemon=True)
    t.start()

    server = HTTPServer(("0.0.0.0", PROBE_PORT), Handler)
    print(f"Probe dashboard server on http://0.0.0.0:{PROBE_PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
