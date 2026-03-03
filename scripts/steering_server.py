#!/usr/bin/env python3
import gc
import glob
import json
import os
import re
import threading
import subprocess
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
from transformers import TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT, "config", "settings.json")
OUTPUT_ROOT = os.path.join(ROOT, "output")
CATALOG_PATH = os.path.join(OUTPUT_ROOT, "catalog.json")
VECTOR_LIB_ROOT = os.path.join(OUTPUT_ROOT, "vector_library")
UI_PATH = os.path.join(ROOT, "ui", "steering.html")
LOG_PATH = os.path.join(OUTPUT_ROOT, "steering_log.jsonl")


class AbortFlag(StoppingCriteria):
    """StoppingCriteria that checks State.abort_flag to interrupt generation."""
    def __call__(self, input_ids, scores, **kwargs):
        return State.abort_flag


class State:
    model = None
    tokenizer = None
    layers = None
    device = None
    catalog = []
    num_layers = 0
    lock = threading.Lock()
    active_model_name = ""
    available_models = []  # list of {"name": ..., "path": ...}
    abort_flag = False


def load_settings():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def get_transformer_layers(model):
    for attr_path in [
        "model.layers",
        "model.model.layers",
        "transformer.h",
        "gpt_neox.layers",
        "transformer.blocks",
    ]:
        obj = model
        try:
            for attr in attr_path.split("."):
                obj = getattr(obj, attr)
            return list(obj)
        except AttributeError:
            continue
    raise ValueError(f"Cannot find transformer layers for {type(model).__name__}")


def load_catalog():
    if os.path.exists(CATALOG_PATH):
        with open(CATALOG_PATH, "r") as f:
            return json.load(f)
    return []


def _resolve_lib_dir(concept_name: str, model_slug: str):
    """
    Restituisce il path della directory dei vettori per un concept nel vector_library.
    Supporta sia Gd0 ("hot_vs_cold") che Gd1 ("hot_vs_cold/thermal_intensity").
    Ritorna il primo match trovato, o None.
    """
    if not os.path.isdir(VECTOR_LIB_ROOT):
        return None
    if "/" in concept_name:
        # Sub-concept: "parent/slug" → {cat}/{parent}/sub/{slug}/{model_slug}/
        parent, slug = concept_name.split("/", 1)
        for cat in os.listdir(VECTOR_LIB_ROOT):
            lib_dir = os.path.join(VECTOR_LIB_ROOT, cat, parent, "sub", slug, model_slug)
            if os.path.isdir(lib_dir):
                return lib_dir
    else:
        # Gd0: {cat}/{concept}/{model_slug}/
        for cat in os.listdir(VECTOR_LIB_ROOT):
            lib_dir = os.path.join(VECTOR_LIB_ROOT, cat, concept_name, model_slug)
            if os.path.isdir(lib_dir):
                return lib_dir
    return None


def get_available_layers(concept_name: str, model_name: str):
    """Scan vector_library and return (sorted layers list, best_layer) for concept+model.
    This is the ground truth — independent of the catalog.
    Supports both Gd0 ('hot_vs_cold') and Gd1 ('hot_vs_cold/thermal_intensity')."""
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


def _load_vec_for_layer(concept_name: str, model_name: str, layer_idx: int):
    """Load concept vector .npy from vector_library for a specific layer.
    Supports Gd0 and Gd1 ('parent/slug'). Returns numpy array or None."""
    if not model_name or not os.path.isdir(VECTOR_LIB_ROOT):
        return None
    slug = model_name.lower().replace(" ", "-").replace("_", "-")
    lib_dir = _resolve_lib_dir(concept_name, slug)
    if lib_dir is None:
        return None
    vec_path = os.path.join(lib_dir, f"layer_{layer_idx}.npy")
    if os.path.exists(vec_path):
        return np.load(vec_path)
    return None


def pick_latest_concept_entry(concept_name: str):
    matches = [e for e in State.catalog if e.get("concept") == concept_name]
    if not matches:
        return None
    matches.sort(key=lambda x: x.get("timestamp", ""))
    # Prefer entries matching the currently loaded model
    active = State.active_model_name
    if active:
        model_matches = [e for e in matches if e.get("model_name", "") == active]
        if model_matches:
            return model_matches[-1]
    return matches[-1]


def load_vector(path: str):
    v = np.load(path)
    return v


def make_steering_hook(steer_tensor, apply_to="all", prompt_len=None):
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        modified = hidden.clone()
        if apply_to == "all":
            modified += steer_tensor
        elif apply_to == "last":
            modified[:, -1, :] += steer_tensor
        elif apply_to == "new":
            # Only generated tokens beyond the original prompt
            if hidden.shape[1] == 1:
                modified[:, -1, :] += steer_tensor
            elif prompt_len is not None and hidden.shape[1] > prompt_len:
                modified[:, prompt_len:, :] += steer_tensor
        if isinstance(output, tuple):
            return (modified,) + output[1:]
        return modified
    return hook_fn


def build_chat_prompt(messages):
    # Prefer tokenizer chat template if present
    tok = State.tokenizer
    if hasattr(tok, "apply_chat_template"):
        try:
            return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass

    # Fallback
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        parts.append(f"{role.upper()}: {content}")
    parts.append("ASSISTANT:")
    return "\n".join(parts)


def generate_text(prompt, max_new_tokens=128):
    inputs = State.tokenizer(prompt, return_tensors="pt").to(State.device)
    with torch.no_grad():
        out = State.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
        )
    return State.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def stream_generate(prompt, max_new_tokens=128, hooks_fn=None):
    """Generator that yields text tokens one by one via TextIteratorStreamer.
    hooks_fn: callable(streamer) that registers forward hooks before generation
              and returns a list of hooks to remove when done.
    """
    streamer = TextIteratorStreamer(
        State.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=60.0
    )
    inputs = State.tokenizer(prompt, return_tensors="pt").to(State.device)

    hooks = hooks_fn(streamer) if hooks_fn else []

    State.abort_flag = False
    gen_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        do_sample=True,
        stopping_criteria=StoppingCriteriaList([AbortFlag()]),
    )

    def _run():
        try:
            with torch.no_grad():
                State.model.generate(**gen_kwargs)
        except Exception as e:
            streamer.text_queue.put(f"\n[ERROR: {e}]")
        finally:
            for h in hooks:
                h.remove()

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    for token in streamer:
        yield token


def stream_with_injection(prompt, concept, vector_layer_idx, inject_layer_idx, alpha,
                          gain=1.0, max_new_tokens=128, apply_to="new",
                          multi_layers=None, layer_configs=None):
    """Streaming version of generate_with_injection."""
    available_layers, _ = get_available_layers(concept, State.active_model_name)
    if not available_layers:
        yield f"[ERROR: No vectors for '{concept}' / '{State.active_model_name}']"
        return

    prompt_len = len(State.tokenizer.encode(prompt))
    dtype = next(State.model.parameters()).dtype

    def hooks_fn(streamer):
        hooks = []
        def _make_hook(layer_idx, layer_gain):
            vec = _load_vec_for_layer(concept, State.active_model_name, layer_idx)
            if vec is None:
                return None
            steer = torch.tensor(vec, dtype=dtype, device=State.device) * float(alpha) * float(layer_gain)
            return State.layers[layer_idx].register_forward_hook(
                make_steering_hook(steer, apply_to=apply_to, prompt_len=prompt_len)
            )

        if layer_configs:
            for cfg in layer_configs:
                h = _make_hook(int(cfg["layer"]), float(cfg["gain"]))
                if h: hooks.append(h)
        elif multi_layers:
            for li in multi_layers:
                h = _make_hook(li, gain)
                if h: hooks.append(h)
        else:
            if vector_layer_idx not in available_layers:
                return []
            h = _make_hook(vector_layer_idx, gain)
            if h: hooks.append(h)
        return hooks

    yield from stream_generate(prompt, max_new_tokens=max_new_tokens, hooks_fn=hooks_fn)


def generate_with_injection(prompt, concept, vector_layer_idx, inject_layer_idx, alpha, gain=1.0, max_new_tokens=128, apply_to="new", multi_layers=None, layer_configs=None, preloaded_vec=None):
    """
    layer_configs:   list of {"layer": int, "gain": float} — per-layer control.
                     When provided, overrides multi_layers and vector/inject_layer_idx.
    multi_layers:    list of layer indices — legacy multi-layer mode (same gain for all).
    preloaded_vec:   numpy array — bypass vector_library lookup (for sub-concept vectors).
    Default:         single layer, vector_layer_idx → inject_layer_idx.
    Vectors are loaded directly from vector_library (ground truth, not catalog).
    """
    if preloaded_vec is None:
        available_layers, _ = get_available_layers(concept, State.active_model_name)
        if not available_layers:
            raise RuntimeError(
                f"No vectors found for concept '{concept}' / model '{State.active_model_name}'. "
                f"Run probe first."
            )
    else:
        available_layers = [vector_layer_idx]  # not used for lookup when preloaded

    hooks = []
    prompt_len = len(State.tokenizer.encode(prompt))
    dtype = next(State.model.parameters()).dtype

    def _make_hook_for_layer(layer_idx, layer_gain, custom_vec=None):
        vec = custom_vec if custom_vec is not None else _load_vec_for_layer(concept, State.active_model_name, layer_idx)
        if vec is None:
            return None
        steer = torch.tensor(vec, dtype=dtype, device=State.device) * float(alpha) * float(layer_gain)
        return State.layers[layer_idx].register_forward_hook(
            make_steering_hook(steer, apply_to=apply_to, prompt_len=prompt_len)
        )

    if layer_configs:
        for cfg in layer_configs:
            hook = _make_hook_for_layer(int(cfg["layer"]), float(cfg["gain"]))
            if hook is not None:
                hooks.append(hook)
    elif multi_layers:
        for layer_idx in multi_layers:
            hook = _make_hook_for_layer(layer_idx, gain)
            if hook is not None:
                hooks.append(hook)
    else:
        if preloaded_vec is None and vector_layer_idx not in available_layers:
            raise RuntimeError(
                f"Layer {vector_layer_idx} not available for '{concept}' / '{State.active_model_name}'. "
                f"Available: {available_layers}"
            )
        vec = preloaded_vec if preloaded_vec is not None else _load_vec_for_layer(concept, State.active_model_name, vector_layer_idx)
        if vec is None:
            raise RuntimeError(f"Vector file missing for layer {vector_layer_idx}")
        steer = torch.tensor(vec, dtype=dtype, device=State.device) * float(alpha) * float(gain)
        hooks.append(State.layers[inject_layer_idx].register_forward_hook(
            make_steering_hook(steer, apply_to=apply_to, prompt_len=prompt_len)
        ))

    try:
        return generate_text(prompt, max_new_tokens=max_new_tokens)
    finally:
        for h in hooks:
            h.remove()


def load_model(model_path=None, model_name=None):
    settings = load_settings()
    State.available_models = settings.get("models", [])

    if model_path is None:
        model_path = settings.get("model_path", "")
    if not model_path:
        raise SystemExit("model_path vuoto in config/settings.json")

    if model_name is None:
        # Derive name from available_models list
        for m in State.available_models:
            if m.get("path") == model_path:
                model_name = m.get("name", "")
                break
        else:
            model_name = os.path.basename(model_path)

    # Free previous model before loading new one
    if State.model is not None:
        del State.model
        State.model = None
        State.tokenizer = None
        State.layers = []
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    dtype_str = settings.get("dtype", "bfloat16")
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map.get(dtype_str, torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=bool(settings.get("trust_remote_code", False)), use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=bool(settings.get("trust_remote_code", False)),
        dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    model.eval()
    pref = settings.get("device", "cuda")
    if pref == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    State.model = model
    State.tokenizer = tokenizer
    State.layers = get_transformer_layers(model)
    State.device = device
    State.catalog = load_catalog()
    State.num_layers = model.config.num_hidden_layers
    State.active_model_name = model_name


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


_prev_cpu = None

def _read_cpu_pct():
    """Read CPU usage % from /proc/stat (delta between two calls)."""
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
    """Read RAM usage from /proc/meminfo."""
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

    if torch.cuda.is_available() and State.model is not None:
        try:
            allocated_mb = torch.cuda.memory_allocated() / 1024**2
            total_mb     = torch.cuda.get_device_properties(0).total_memory / 1024**2
            result["vram_allocated_mb"] = round(allocated_mb, 1)
            result["vram_total_mb"]     = round(total_mb, 1)
            result["vram_used_pct"]     = round(allocated_mb / total_mb * 100, 1)
        except Exception:
            pass

    result["cpu_pct"] = _read_cpu_pct()
    result["ram"] = _read_ram()
    return result


def append_log(entry):
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def get_library_overview():
    """Scan vector_library/*/*/meta.json and return summary for each concept."""
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


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            with open(UI_PATH, "r") as f:
                html = f.read()
            return text_response(self, 200, html)
        if parsed.path == "/api/concepts":
            active = State.active_model_name
            # Source of truth: vector_library on disk
            concepts = set()
            sub_concepts = set()
            if active and os.path.isdir(VECTOR_LIB_ROOT):
                slug = active.lower().replace(" ", "-").replace("_", "-")
                for cat in os.listdir(VECTOR_LIB_ROOT):
                    cat_path = os.path.join(VECTOR_LIB_ROOT, cat)
                    if not os.path.isdir(cat_path):
                        continue
                    for concept in os.listdir(cat_path):
                        # Gd0
                        model_dir = os.path.join(cat_path, concept, slug)
                        if os.path.isdir(model_dir) and glob.glob(os.path.join(model_dir, "layer_*.npy")):
                            concepts.add(concept)
                        # Gd1: {cat}/{concept}/sub/{slug}/{model}/
                        sub_root = os.path.join(cat_path, concept, "sub")
                        if os.path.isdir(sub_root):
                            for sub_slug in os.listdir(sub_root):
                                sub_model_dir = os.path.join(sub_root, sub_slug, slug)
                                if os.path.isdir(sub_model_dir) and glob.glob(os.path.join(sub_model_dir, "layer_*.npy")):
                                    sub_concepts.add(f"{concept}/{sub_slug}")
            if not concepts and not sub_concepts:
                # Fallback to catalog
                eligible = [e for e in State.catalog if e.get("model_name") == active] if active else State.catalog
                for e in eligible:
                    if e.get("concept"):
                        (sub_concepts if e.get("is_sub_concept") else concepts).add(e["concept"])
            return json_response(self, 200, {
                "concepts":     sorted(concepts),
                "sub_concepts": sorted(sub_concepts),
                "device": str(State.device),
                "model": active,
            })
        if parsed.path == "/api/concept_layers":
            qs = parse_qs(parsed.query)
            concept = (qs.get("concept") or [None])[0]
            model = (qs.get("model") or [State.active_model_name])[0] or State.active_model_name
            if not concept:
                return json_response(self, 200, {"layers": [], "best_layer": None, "model": model})
            layers, best_layer = get_available_layers(concept, model)
            return json_response(self, 200, {"layers": layers, "best_layer": best_layer, "model": model})
        if parsed.path == "/api/model_info":
            return json_response(self, 200, {"num_layers": State.num_layers, "device": str(State.device)})
        if parsed.path == "/api/gpu":
            return json_response(self, 200, get_gpu_stats())
        if parsed.path == "/api/models":
            return json_response(self, 200, {
                "models": State.available_models,
                "active": State.active_model_name,
            })
        if parsed.path == "/api/library":
            return json_response(self, 200, {"library": get_library_overview()})
        if parsed.path == "/api/reload_catalog":
            State.catalog = load_catalog()
            return json_response(self, 200, {"ok": True, "entries": len(State.catalog)})
        if parsed.path == "/api/stop":
            State.abort_flag = True
            return json_response(self, 200, {"ok": True, "aborted": True})
        if parsed.path == "/api/unload_model":
            # Aspetta al massimo 120s che il lock sia libero (evita blocco su load_model in corso)
            acquired = State.lock.acquire(timeout=120)
            if acquired:
                try:
                    if State.model is not None:
                        del State.model
                        State.model = None
                        State.tokenizer = None
                        State.layers = []
                        if torch.cuda.is_available():
                            gc.collect()
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                        State.num_layers = 0
                        State.active_model_name = ""
                finally:
                    State.lock.release()
                return json_response(self, 200, {"ok": True, "unloaded": True})
            else:
                return json_response(self, 503, {"ok": False, "error": "lock_timeout"})
        return self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/generate":
            body = read_body(self)
            try:
                payload = json.loads(body.decode("utf-8"))
            except Exception:
                return json_response(self, 400, {"error": "invalid_json"})

            messages = payload.get("messages", [])
            prompt = payload.get("prompt", "")
            concept = payload.get("concept", "hot_vs_cold")
            vector_layer = int(payload.get("vector_layer", 0))
            inject_layer = int(payload.get("inject_layer", 0))
            alpha = float(payload.get("alpha", 0.0))
            gain = float(payload.get("gain", 1.0))
            max_new_tokens = int(payload.get("max_new_tokens", 128))
            mode = payload.get("mode", "inject")
            multi = bool(payload.get("multi", False))
            layer_configs = payload.get("layer_configs", None)  # [{layer, gain}, ...]
            vector_path = payload.get("vector_path", None)      # direct .npy path (sub-concepts)

            # Load preloaded_vec if vector_path provided (bypasses catalog lookup)
            preloaded_vec = None
            if vector_path:
                try:
                    preloaded_vec = np.load(vector_path)
                except Exception as e:
                    return json_response(self, 400, {"error": f"Cannot load vector_path '{vector_path}': {e}"})

            if messages:
                formatted_prompt = build_chat_prompt(messages)
            else:
                formatted_prompt = prompt

            if not formatted_prompt:
                return json_response(self, 400, {"error": "empty_prompt"})

            # Resolve multi_layers from vector_library before acquiring lock
            if preloaded_vec is None:
                avail_layers, best_layer = get_available_layers(concept, State.active_model_name)
            else:
                avail_layers = [vector_layer]  # not used for lookup when preloaded
            if multi and not layer_configs:
                multi_layers_resolved = avail_layers
            else:
                multi_layers_resolved = None

            # T5: Validate requested layers against available (skip if preloaded)
            if preloaded_vec is None and mode != "baseline" and alpha != 0.0 and not layer_configs and not multi:
                if avail_layers and vector_layer not in avail_layers:
                    return json_response(self, 400, {
                        "error": f"Layer {vector_layer} not available for '{concept}' / '{State.active_model_name}'. "
                                 f"Available: {avail_layers}"
                    })

            with State.lock:
                try:
                    if mode == "baseline" or alpha == 0.0:
                        text = generate_text(formatted_prompt, max_new_tokens=max_new_tokens)
                    else:
                        text = generate_with_injection(
                            formatted_prompt,
                            concept,
                            vector_layer,
                            inject_layer,
                            alpha,
                            gain=gain,
                            max_new_tokens=max_new_tokens,
                            apply_to="new",
                            multi_layers=multi_layers_resolved,
                            layer_configs=layer_configs,
                            preloaded_vec=preloaded_vec,
                        )
                except Exception as e:
                    return json_response(self, 500, {"error": str(e)})

            used_layers = layer_configs if layer_configs else (
                [{"layer": l, "gain": gain} for l in (multi_layers_resolved or [])] if multi
                else [{"layer": inject_layer, "gain": gain}]
            )
            log_entry = {
                "prompt": formatted_prompt,
                "concept": concept,
                "alpha": alpha,
                "mode": mode,
                "output": text,
                "layer_configs": used_layers,
            }
            append_log(log_entry)

            return json_response(self, 200, {"text": text, "formatted_prompt": formatted_prompt})

        if parsed.path == "/api/generate_stream":
            if State.model is None:
                return json_response(self, 503, {"error": "Model not loaded — click Load first."})
            body = read_body(self)
            try:
                payload = json.loads(body.decode("utf-8"))
            except Exception:
                return json_response(self, 400, {"error": "invalid_json"})

            messages = payload.get("messages", [])
            prompt   = payload.get("prompt", "")
            concept  = payload.get("concept", "hot_vs_cold")
            vector_layer  = int(payload.get("vector_layer", 0))
            inject_layer  = int(payload.get("inject_layer", 0))
            alpha         = float(payload.get("alpha", 0.0))
            gain          = float(payload.get("gain", 1.0))
            max_new_tokens = int(payload.get("max_new_tokens", 128))
            mode          = payload.get("mode", "inject")
            multi         = bool(payload.get("multi", False))
            layer_configs = payload.get("layer_configs", None)

            if messages:
                formatted_prompt = build_chat_prompt(messages)
            else:
                formatted_prompt = prompt
            if not formatted_prompt:
                return json_response(self, 400, {"error": "empty_prompt"})

            avail_layers, _ = get_available_layers(concept, State.active_model_name)
            if multi and not layer_configs:
                multi_layers_resolved = avail_layers
            else:
                multi_layers_resolved = None

            # SSE headers
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()

            # Send formatted prompt as first SSE event
            fp_data = json.dumps({"type": "prompt", "text": formatted_prompt})
            self.wfile.write(f"data: {fp_data}\n\n".encode("utf-8"))
            self.wfile.flush()

            full_text = []
            try:
                with State.lock:
                    if mode == "baseline" or alpha == 0.0:
                        token_gen = stream_generate(formatted_prompt, max_new_tokens=max_new_tokens)
                    else:
                        token_gen = stream_with_injection(
                            formatted_prompt, concept,
                            vector_layer, inject_layer, alpha,
                            gain=gain, max_new_tokens=max_new_tokens,
                            apply_to="new",
                            multi_layers=multi_layers_resolved,
                            layer_configs=layer_configs,
                        )
                    for token in token_gen:
                        if State.abort_flag:
                            break
                        full_text.append(token)
                        data = json.dumps({"type": "token", "text": token})
                        self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
                        self.wfile.flush()
            except Exception as e:
                err_data = json.dumps({"type": "error", "text": str(e)})
                self.wfile.write(f"data: {err_data}\n\n".encode("utf-8"))
                self.wfile.flush()
            finally:
                done_data = json.dumps({"type": "done"})
                self.wfile.write(f"data: {done_data}\n\n".encode("utf-8"))
                self.wfile.flush()
                append_log({
                    "prompt": formatted_prompt, "concept": concept,
                    "alpha": alpha, "mode": mode,
                    "output": "".join(full_text),
                    "layer_configs": layer_configs or [],
                })
            return

        if parsed.path == "/api/load_model":
            body = read_body(self)
            try:
                payload = json.loads(body.decode("utf-8"))
            except Exception:
                return json_response(self, 400, {"error": "invalid_json"})

            requested_name = payload.get("name", "")
            target = None
            for m in State.available_models:
                if m.get("name") == requested_name:
                    target = m
                    break
            if target is None:
                return json_response(self, 404, {"error": f"Model not found: {requested_name}"})

            with State.lock:
                try:
                    del State.model
                    State.model = None
                    torch.cuda.empty_cache()
                    load_model(model_path=target["path"], model_name=target["name"])
                except Exception as e:
                    return json_response(self, 500, {"error": str(e)})

            return json_response(self, 200, {
                "ok": True,
                "name": State.active_model_name,
                "num_layers": State.num_layers,
                "device": str(State.device),
            })

        return self.send_error(404)


def main():
    load_model()
    port = int(os.environ.get("JEDI_PORT", "8010"))
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"Jedi steering server listening on http://0.0.0.0:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
