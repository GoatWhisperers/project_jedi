#!/usr/bin/env python3
"""
steering_server.py — Presentation/routing layer (porta 8010).

Espone la stessa interfaccia HTTP agli utenti e alla UI, ma delega tutte
le operazioni GPU a mi50_manager (porta 8020).

Non gestisce più modelli HuggingFace direttamente.
"""
import glob
import json
import os
import re
import threading
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
from urllib.request import urlopen, Request
from urllib.error import URLError

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT, "config", "settings.json")
OUTPUT_ROOT = os.path.join(ROOT, "output")
CATALOG_PATH = os.path.join(OUTPUT_ROOT, "catalog.json")
VECTOR_LIB_ROOT = os.path.join(OUTPUT_ROOT, "vector_library")
UI_PATH = os.path.join(ROOT, "ui", "steering.html")
LOG_PATH = os.path.join(OUTPUT_ROOT, "steering_log.jsonl")

# mi50_manager URL — unico owner GPU MI50
MI50_URL = os.environ.get("MI50_URL", "http://localhost:8020")


# ── mi50_manager HTTP helpers ──────────────────────────────────────────────────

def mi50_post(path: str, payload: dict, timeout: int = 300) -> dict:
    """POST a mi50_manager. Ritorna dict JSON o {"error": ...} in caso di fallimento."""
    url = f"{MI50_URL}{path}"
    data = json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"error": str(e)}


def mi50_get(path: str, timeout: int = 30) -> dict:
    """GET a mi50_manager. Ritorna dict JSON o {} in caso di fallimento."""
    url = f"{MI50_URL}{path}"
    try:
        with urlopen(url, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return {}


def mi50_stream_post(path: str, payload: dict, timeout: int = 300):
    """
    POST a mi50_manager e ritorna il file-like object SSE aperto.
    Il chiamante è responsabile di chiuderlo.
    """
    url = f"{MI50_URL}{path}"
    data = json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    return urlopen(req, timeout=timeout)


# ── State (solo catalog e modelli disponibili, niente GPU) ────────────────────

class State:
    catalog          = []
    available_models = []   # list of {"name": ..., "path": ...}
    lock             = threading.Lock()


def load_settings():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def load_catalog():
    if os.path.exists(CATALOG_PATH):
        with open(CATALOG_PATH, "r") as f:
            return json.load(f)
    return []


def _get_active_model_name() -> str:
    """Chiede a mi50_manager qual è il modello attivo."""
    st = mi50_get("/api/status")
    return st.get("model", "") or ""


# ── Vector library lookup (rimane qui per catalog/routing) ────────────────────

def _resolve_lib_dir(concept_name: str, model_slug: str):
    """
    Restituisce il path della directory dei vettori per un concept nel vector_library.
    Supporta sia Gd0 ("hot_vs_cold") che Gd1 ("hot_vs_cold/thermal_intensity").
    Ritorna il primo match trovato, o None.
    """
    if not os.path.isdir(VECTOR_LIB_ROOT):
        return None
    if "/" in concept_name:
        parent, slug = concept_name.split("/", 1)
        for cat in os.listdir(VECTOR_LIB_ROOT):
            lib_dir = os.path.join(VECTOR_LIB_ROOT, cat, parent, "sub", slug, model_slug)
            if os.path.isdir(lib_dir):
                return lib_dir
    else:
        for cat in os.listdir(VECTOR_LIB_ROOT):
            lib_dir = os.path.join(VECTOR_LIB_ROOT, cat, concept_name, model_slug)
            if os.path.isdir(lib_dir):
                return lib_dir
    return None


def get_available_layers(concept_name: str, model_name: str):
    """Scan vector_library e ritorna (sorted layers list, best_layer) per concept+model."""
    if not model_name:
        return [], None
    slug = model_name.lower().replace(" ", "-").replace("_", "-")
    lib_dir = _resolve_lib_dir(concept_name, slug)
    if lib_dir is None:
        return [], None
    layers = []
    for f in glob.glob(os.path.join(lib_dir, "layer_*.npy")):
        name = os.path.basename(f)
        if "_pca" not in name:
            m = re.match(r"layer_(\d+)\.npy", name)
            if m:
                layers.append(int(m.group(1)))
    layers.sort()
    best_layer = layers[-1] if layers else None
    summary_path = os.path.join(lib_dir, "summary.json")
    if os.path.exists(summary_path):
        try:
            with open(summary_path) as f:
                summary = json.load(f)
            results = summary.get("results", {})
            if results:
                best_key = max(results, key=lambda k: results[k].get("sep_snr", -99))
                best_layer = int(best_key)
        except Exception:
            pass
    return layers, best_layer


def _find_vector_path(concept_name: str, model_name: str, layer_idx: int) -> str | None:
    """Ritorna il path .npy del vettore concept per un layer specifico, o None."""
    if not model_name or not os.path.isdir(VECTOR_LIB_ROOT):
        return None
    slug = model_name.lower().replace(" ", "-").replace("_", "-")
    lib_dir = _resolve_lib_dir(concept_name, slug)
    if lib_dir is None:
        return None
    vec_path = os.path.join(lib_dir, f"layer_{layer_idx}.npy")
    return vec_path if os.path.exists(vec_path) else None


def pick_latest_concept_entry(concept_name: str, active_model: str):
    matches = [e for e in State.catalog if e.get("concept") == concept_name]
    if not matches:
        return None
    matches.sort(key=lambda x: x.get("timestamp", ""))
    if active_model:
        model_matches = [e for e in matches if e.get("model_name", "") == active_model]
        if model_matches:
            return model_matches[-1]
    return matches[-1]


# ── Logging ───────────────────────────────────────────────────────────────────

def append_log(entry):
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ── Library overview ──────────────────────────────────────────────────────────

def get_library_overview():
    """Scan vector_library/*/*/meta.json e ritorna sommario per ogni concept."""
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


# ── GPU stats (rocm-smi, augmenta lo status di mi50_manager) ─────────────────

_prev_cpu = None

def _read_cpu_pct():
    global _prev_cpu
    try:
        with open("/proc/stat") as f:
            line = f.readline()
        vals = list(map(int, line.split()[1:]))
        idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
        total = sum(vals)
        if _prev_cpu is None:
            _prev_cpu = (total, idle)
            return 0.0
        d_total = total - _prev_cpu[0]
        d_idle  = idle  - _prev_cpu[1]
        _prev_cpu = (total, idle)
        return round((d_total - d_idle) / d_total * 100, 1) if d_total > 0 else 0.0
    except Exception:
        return 0.0


def _read_ram():
    try:
        info = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":")
                info[k.strip()] = int(v.split()[0])
        total_mb = info["MemTotal"] / 1024
        avail_mb = info["MemAvailable"] / 1024
        used_mb  = total_mb - avail_mb
        return {"used_mb": round(used_mb), "total_mb": round(total_mb),
                "pct": round(used_mb / total_mb * 100, 1)}
    except Exception:
        return {}


def get_gpu_stats():
    """Combina rocm-smi + mi50_manager /api/status per il dashboard."""
    result = {"ok": False}
    try:
        res = subprocess.run([
            "rocm-smi", "--showtemp", "--showuse", "--showmeminfo", "vram", "--json"
        ], capture_output=True, text=True, check=True)
        data = json.loads(res.stdout)
        card = data.get("card0", {})
        edge     = float(card.get("Temperature (Sensor edge) (C)", "0") or 0)
        junction = float(card.get("Temperature (Sensor junction) (C)", "0") or 0)
        gpu_use  = float(card.get("GPU use (%)", "0") or 0)
        vram_used  = float(card.get("VRAM Total Used Memory (B)", "0") or 0) / 1024**3
        vram_total = float(card.get("VRAM Total Memory (B)", "0") or 0) / 1024**3
        vram_pct   = round(vram_used / vram_total * 100, 1) if vram_total > 0 else 0
        result = {
            "ok": True,
            "edge_c": edge,
            "junction_c": junction,
            "gpu_use_pct": gpu_use,
            "vram_used_gb": round(vram_used, 1),
            "vram_total_gb": round(vram_total, 1),
            "vram_pct": vram_pct,
        }
    except Exception as e:
        result = {"ok": False, "error": str(e)}

    # Arricchisce con stato mi50_manager
    mi50_st = mi50_get("/api/status")
    if mi50_st:
        result["mi50_busy"]       = mi50_st.get("busy", False)
        result["mi50_busy_owner"] = mi50_st.get("busy_owner")
        result["mi50_model"]      = mi50_st.get("model")

    result["cpu_pct"] = _read_cpu_pct()
    result["ram"] = _read_ram()
    return result


# ── HTTP response helpers ─────────────────────────────────────────────────────

def json_response(handler, code, payload):
    data = json.dumps(payload).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(data)))
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


def _build_chat_prompt_local(messages: list, active_model: str) -> str:
    """
    Costruisce un prompt testuale da messages senza tokenizer locale.
    Usato solo come fallback; steering_server ora passa i messaggi raw a mi50_manager.
    """
    parts = []
    for m in messages:
        role    = m.get("role", "user")
        content = m.get("content", "")
        parts.append(f"{role.upper()}: {content}")
    parts.append("ASSISTANT:")
    return "\n".join(parts)


# ── Handler HTTP ──────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/":
            with open(UI_PATH, "r") as f:
                html = f.read()
            return text_response(self, 200, html)

        if parsed.path == "/api/concepts":
            active = _get_active_model_name()
            concepts = set()
            sub_concepts = set()
            if active and os.path.isdir(VECTOR_LIB_ROOT):
                slug = active.lower().replace(" ", "-").replace("_", "-")
                for cat in os.listdir(VECTOR_LIB_ROOT):
                    cat_path = os.path.join(VECTOR_LIB_ROOT, cat)
                    if not os.path.isdir(cat_path):
                        continue
                    for concept in os.listdir(cat_path):
                        model_dir = os.path.join(cat_path, concept, slug)
                        if os.path.isdir(model_dir) and glob.glob(os.path.join(model_dir, "layer_*.npy")):
                            concepts.add(concept)
                        sub_root = os.path.join(cat_path, concept, "sub")
                        if os.path.isdir(sub_root):
                            for sub_slug in os.listdir(sub_root):
                                sub_model_dir = os.path.join(sub_root, sub_slug, slug)
                                if os.path.isdir(sub_model_dir) and glob.glob(os.path.join(sub_model_dir, "layer_*.npy")):
                                    sub_concepts.add(f"{concept}/{sub_slug}")
            if not concepts and not sub_concepts:
                eligible = [e for e in State.catalog if e.get("model_name") == active] if active else State.catalog
                for e in eligible:
                    if e.get("concept"):
                        (sub_concepts if e.get("is_sub_concept") else concepts).add(e["concept"])

            mi50_st = mi50_get("/api/status")
            return json_response(self, 200, {
                "concepts":     sorted(concepts),
                "sub_concepts": sorted(sub_concepts),
                "device": mi50_st.get("device", ""),
                "model": active,
            })

        if parsed.path == "/api/concept_layers":
            qs = parse_qs(urlparse(self.path).query)
            concept = (qs.get("concept") or [None])[0]
            active  = _get_active_model_name()
            model   = (qs.get("model") or [active])[0] or active
            if not concept:
                return json_response(self, 200, {"layers": [], "best_layer": None, "model": model})
            layers, best_layer = get_available_layers(concept, model)
            return json_response(self, 200, {"layers": layers, "best_layer": best_layer, "model": model})

        if parsed.path == "/api/model_info":
            st = mi50_get("/api/status")
            return json_response(self, 200, {
                "num_layers": st.get("num_layers"),
                "device":     st.get("device", ""),
                "model":      st.get("model"),
            })

        if parsed.path == "/api/gpu":
            return json_response(self, 200, get_gpu_stats())

        if parsed.path == "/api/models":
            settings = load_settings()
            available = settings.get("models", [])
            active = _get_active_model_name()
            return json_response(self, 200, {
                "models": available,
                "active": active,
            })

        if parsed.path == "/api/library":
            return json_response(self, 200, {"library": get_library_overview()})

        if parsed.path == "/api/reload_catalog":
            State.catalog = load_catalog()
            return json_response(self, 200, {"ok": True, "entries": len(State.catalog)})

        if parsed.path == "/api/stop":
            result = mi50_post("/api/stop", {})
            return json_response(self, 200, result)

        if parsed.path == "/api/unload_model":
            result = mi50_post("/api/unload_model", {})
            code = 200 if result.get("ok") else 503
            return json_response(self, code, result)

        return self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)

        # ── /api/load_model → proxy a mi50_manager ─────────────────────────
        if parsed.path == "/api/load_model":
            body = read_body(self)
            try:
                payload = json.loads(body.decode("utf-8"))
            except Exception:
                return json_response(self, 400, {"error": "invalid_json"})

            result = mi50_post("/api/load_model", payload)
            code = 200 if result.get("ok") or result.get("noop") else (
                404 if "not found" in result.get("error", "").lower() else 500
            )
            return json_response(self, code, result)

        # ── /api/unload_model → proxy a mi50_manager ───────────────────────
        if parsed.path == "/api/unload_model":
            result = mi50_post("/api/unload_model", {})
            code = 200 if result.get("ok") else 503
            return json_response(self, code, result)

        # ── /api/stop → proxy a mi50_manager ───────────────────────────────
        if parsed.path == "/api/stop":
            result = mi50_post("/api/stop", {})
            return json_response(self, 200, result)

        # ── /api/generate → risolve vettore localmente, poi proxy ──────────
        if parsed.path == "/api/generate":
            body = read_body(self)
            try:
                payload = json.loads(body.decode("utf-8"))
            except Exception:
                return json_response(self, 400, {"error": "invalid_json"})

            messages       = payload.get("messages", [])
            prompt         = payload.get("prompt", "")
            concept        = payload.get("concept", "hot_vs_cold")
            vector_layer   = int(payload.get("vector_layer", 0))
            inject_layer   = int(payload.get("inject_layer", 0))
            alpha          = float(payload.get("alpha", 0.0))
            gain           = float(payload.get("gain", 1.0))
            max_new_tokens = int(payload.get("max_new_tokens", 128))
            mode           = payload.get("mode", "inject")
            multi          = bool(payload.get("multi", False))
            layer_configs  = payload.get("layer_configs", None)
            vector_path    = payload.get("vector_path", None)

            # Costruisce prompt da messages se necessario
            if messages and not prompt:
                prompt = _build_chat_prompt_local(messages, _get_active_model_name())

            if not prompt:
                return json_response(self, 400, {"error": "empty_prompt"})

            # Risolve vector_path dal vector_library se non passato direttamente
            if not vector_path and mode != "baseline" and alpha != 0.0:
                active = _get_active_model_name()
                vp = _find_vector_path(concept, active, vector_layer)
                if vp:
                    vector_path = vp

            # Valida layer disponibile (solo single-layer senza layer_configs)
            if mode != "baseline" and alpha != 0.0 and not layer_configs and not multi:
                active = _get_active_model_name()
                avail_layers, _ = get_available_layers(concept, active)
                if avail_layers and vector_layer not in avail_layers and not vector_path:
                    return json_response(self, 400, {
                        "error": f"Layer {vector_layer} not available for '{concept}' / '{active}'. "
                                 f"Available: {avail_layers}"
                    })

            mi50_payload = {
                "prompt":        prompt,
                "concept":       concept,
                "vector_layer":  vector_layer,
                "inject_layer":  inject_layer,
                "alpha":         alpha,
                "gain":          gain,
                "max_new_tokens": max_new_tokens,
                "mode":          mode,
                "multi":         multi,
                "layer_configs": layer_configs,
                "vector_path":   vector_path,
            }

            result = mi50_post("/api/generate", mi50_payload, timeout=300)
            if "error" in result and "text" not in result:
                return json_response(self, 500, result)

            text = result.get("text", "")
            # Logging
            used_layers = layer_configs if layer_configs else (
                [] if mode == "baseline" or alpha == 0.0
                else [{"layer": inject_layer, "gain": gain}]
            )
            append_log({
                "prompt":        prompt,
                "concept":       concept,
                "alpha":         alpha,
                "mode":          mode,
                "output":        text,
                "layer_configs": used_layers,
            })
            return json_response(self, 200, {"text": text, "formatted_prompt": prompt})

        # ── /api/generate_stream → proxy SSE a mi50_manager ────────────────
        if parsed.path == "/api/generate_stream":
            body = read_body(self)
            try:
                payload = json.loads(body.decode("utf-8"))
            except Exception:
                return json_response(self, 400, {"error": "invalid_json"})

            messages       = payload.get("messages", [])
            prompt         = payload.get("prompt", "")
            concept        = payload.get("concept", "hot_vs_cold")
            vector_layer   = int(payload.get("vector_layer", 0))
            inject_layer   = int(payload.get("inject_layer", 0))
            alpha          = float(payload.get("alpha", 0.0))
            gain           = float(payload.get("gain", 1.0))
            max_new_tokens = int(payload.get("max_new_tokens", 128))
            mode           = payload.get("mode", "inject")
            multi          = bool(payload.get("multi", False))
            layer_configs  = payload.get("layer_configs", None)
            vector_path    = payload.get("vector_path", None)

            if messages and not prompt:
                prompt = _build_chat_prompt_local(messages, _get_active_model_name())
            if not prompt:
                return json_response(self, 400, {"error": "empty_prompt"})

            # Risolve vector_path
            if not vector_path and mode != "baseline" and alpha != 0.0:
                active = _get_active_model_name()
                vp = _find_vector_path(concept, active, vector_layer)
                if vp:
                    vector_path = vp

            mi50_payload = {
                "prompt":        prompt,
                "concept":       concept,
                "vector_layer":  vector_layer,
                "inject_layer":  inject_layer,
                "alpha":         alpha,
                "gain":          gain,
                "max_new_tokens": max_new_tokens,
                "mode":          mode,
                "multi":         multi,
                "layer_configs": layer_configs,
                "vector_path":   vector_path,
            }

            # SSE headers verso il client
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()

            # Invia il formatted_prompt come primo evento (compatibilità UI)
            fp_data = json.dumps({"type": "prompt", "text": prompt})
            self.wfile.write(f"data: {fp_data}\n\n".encode("utf-8"))
            self.wfile.flush()

            full_text = []
            try:
                r = mi50_stream_post("/api/generate_stream", mi50_payload, timeout=300)
                for raw_line in r:
                    line = raw_line.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    try:
                        ev = json.loads(data_str)
                    except Exception:
                        continue

                    if ev.get("done"):
                        break
                    if "error" in ev:
                        err_data = json.dumps({"type": "error", "text": ev["error"]})
                        self.wfile.write(f"data: {err_data}\n\n".encode("utf-8"))
                        self.wfile.flush()
                        break
                    if "token" in ev:
                        token = ev["token"]
                        full_text.append(token)
                        out_data = json.dumps({"type": "token", "text": token})
                        self.wfile.write(f"data: {out_data}\n\n".encode("utf-8"))
                        self.wfile.flush()
                r.close()
            except Exception as e:
                err_data = json.dumps({"type": "error", "text": str(e)})
                self.wfile.write(f"data: {err_data}\n\n".encode("utf-8"))
                self.wfile.flush()
            finally:
                done_data = json.dumps({"type": "done"})
                self.wfile.write(f"data: {done_data}\n\n".encode("utf-8"))
                self.wfile.flush()
                append_log({
                    "prompt": prompt, "concept": concept,
                    "alpha": alpha, "mode": mode,
                    "output": "".join(full_text),
                    "layer_configs": layer_configs or [],
                })
            return

        return self.send_error(404)


def main():
    # Carica catalog e lista modelli da settings
    State.catalog = load_catalog()
    settings = load_settings()
    State.available_models = settings.get("models", [])

    port = int(os.environ.get("JEDI_PORT", "8010"))
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"Jedi steering server (proxy) listening on http://0.0.0.0:{port}")
    print(f"  → delegating GPU ops to mi50_manager at {MI50_URL}")
    server.serve_forever()


if __name__ == "__main__":
    main()
