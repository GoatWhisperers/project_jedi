#!/usr/bin/env python3
"""
mi50_manager.py — Server HTTP dedicato (porta 8020) che è l'UNICO processo
autorizzato a caricare modelli HuggingFace sulla GPU MI50 (ROCm).

Tutti gli altri script (steering_server, probe_concept) diventano client HTTP.

Endpoints:
  GET  /api/status              → stato GPU e modello caricato
  POST /api/load_model          → carica un modello (noop se già caricato)
  POST /api/unload_model        → scarica modello e libera VRAM
  POST /api/generate            → generazione testo (con/senza steering)
  POST /api/generate_stream     → generazione streaming SSE
  POST /api/stop                → interrompe generazione in corso
  POST /api/extract_activations → estrae hidden states dai layer richiesti
"""

import gc
import glob
import json
import os
import re
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    StoppingCriteria,
    StoppingCriteriaList,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT            = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH     = os.path.join(ROOT, "config", "settings.json")
OUTPUT_ROOT     = os.path.join(ROOT, "output")
VECTOR_LIB_ROOT = os.path.join(OUTPUT_ROOT, "vector_library")

PORT = int(os.environ.get("MI50_PORT", "8020"))

# Max frasi per batch in extract_activations (evita OOM)
EXTRACT_BATCH_SIZE = 8


# ── Stato globale thread-safe ──────────────────────────────────────────────────

class AbortFlag(StoppingCriteria):
    """StoppingCriteria che controlla State.abort_flag per interrompere la generazione."""
    def __call__(self, input_ids, scores, **kwargs):
        return State.abort_flag


class State:
    model        = None
    tokenizer    = None
    model_name   = None    # es. "Gemma3-1B-IT"
    model_path   = None    # es. "/mnt/raid0/gemma-3-1b-it"
    num_layers   = None
    layers       = None    # lista dei moduli transformer layer
    device       = None
    lock         = threading.Lock()   # un'operazione GPU alla volta
    busy         = False              # True durante generate/extract
    busy_owner   = None              # stringa descrittiva di chi usa la GPU
    abort_flag   = False


# ── Settings ──────────────────────────────────────────────────────────────────

def load_settings() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


# ── Utility modello ───────────────────────────────────────────────────────────

def get_transformer_layers(model):
    """Trova la lista dei layer transformer del modello."""
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


# ── Vector library lookup ─────────────────────────────────────────────────────

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


def _load_vec_for_layer(concept_name: str, model_name: str, layer_idx: int):
    """
    Carica il vettore concept .npy dalla vector_library per un layer specifico.
    Supporta Gd0 e Gd1. Ritorna numpy array o None.
    """
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


# ── Operazioni GPU ────────────────────────────────────────────────────────────

def _do_load_model(model_path: str, model_name: str, settings: dict):
    """Carica il modello in State. Deve essere chiamato con State.lock acquisito."""
    # Scarica modello precedente se presente
    if State.model is not None:
        prev = State.model
        State.model = None
        State.tokenizer = None
        State.layers = None
        del prev
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            time.sleep(2)

    dtype_str = settings.get("dtype", "bfloat16")
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16":  torch.float16,
        "float32":  torch.float32,
    }
    torch_dtype = dtype_map.get(dtype_str, torch.bfloat16)

    trust = bool(settings.get("trust_remote_code", False))

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=trust, use_fast=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=trust,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    model.eval()

    pref = settings.get("device", "cuda")
    if pref == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    State.model      = model
    State.tokenizer  = tokenizer
    State.model_name = model_name
    State.model_path = model_path
    State.num_layers = model.config.num_hidden_layers
    State.layers     = get_transformer_layers(model)
    State.device     = device


def _do_unload_model():
    """Scarica il modello e libera VRAM. Deve essere chiamato con State.lock acquisito."""
    if State.model is not None:
        prev = State.model
        State.model     = None
        State.tokenizer = None
        State.layers    = None
        del prev
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    State.model_name = None
    State.model_path = None
    State.num_layers = None
    State.device     = None


def _get_vram_stats():
    """Legge VRAM usata/totale dalla GPU. Ritorna (used_gb, total_gb)."""
    try:
        res = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--json"],
            capture_output=True, text=True, check=True
        )
        data = json.loads(res.stdout)
        card = data.get("card0", {})
        used  = float(card.get("VRAM Total Used Memory (B)", "0") or 0) / 1024**3
        total = float(card.get("VRAM Total Memory (B)", "0") or 0) / 1024**3
        return round(used, 2), round(total, 2)
    except Exception:
        pass
    # Fallback: torch
    if torch.cuda.is_available():
        try:
            props  = torch.cuda.get_device_properties(0)
            used   = torch.cuda.memory_allocated(0) / 1024**3
            total  = props.total_memory / 1024**3
            return round(used, 2), round(total, 2)
        except Exception:
            pass
    return 0.0, 0.0


# ── Generazione ───────────────────────────────────────────────────────────────

def make_steering_hook(steer_tensor, apply_to="new", prompt_len=None):
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
            if hidden.shape[1] == 1:
                modified[:, -1, :] += steer_tensor
            elif prompt_len is not None and hidden.shape[1] > prompt_len:
                modified[:, prompt_len:, :] += steer_tensor
        if isinstance(output, tuple):
            return (modified,) + output[1:]
        return modified
    return hook_fn


def _generate_plain(prompt: str, max_new_tokens: int = 128) -> str:
    """Generazione senza steering (baseline)."""
    inputs = State.tokenizer(prompt, return_tensors="pt").to(State.device)
    with torch.no_grad():
        out = State.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
        )
    return State.tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )


def _generate_steered(
    prompt: str,
    concept: str,
    vector_layer: int,
    inject_layer: int,
    alpha: float,
    gain: float,
    max_new_tokens: int,
    apply_to: str,
    multi_layers=None,
    layer_configs=None,
    vector_path=None,
) -> str:
    """Generazione con activation steering."""
    hooks = []
    prompt_len = len(State.tokenizer.encode(prompt))
    dtype = next(State.model.parameters()).dtype

    def _make_hook(layer_idx: int, layer_gain: float, custom_vec=None):
        if custom_vec is not None:
            vec = custom_vec
        else:
            vec = _load_vec_for_layer(concept, State.model_name, layer_idx)
        if vec is None:
            return None
        steer = torch.tensor(vec, dtype=dtype, device=State.device) * float(alpha) * float(layer_gain)
        return State.layers[layer_idx].register_forward_hook(
            make_steering_hook(steer, apply_to=apply_to, prompt_len=prompt_len)
        )

    # Carica vettore preloaded da path se fornito
    preloaded_vec = None
    if vector_path:
        preloaded_vec = np.load(vector_path)

    if layer_configs:
        for cfg in layer_configs:
            h = _make_hook(int(cfg["layer"]), float(cfg["gain"]))
            if h:
                hooks.append(h)
    elif multi_layers:
        for li in multi_layers:
            h = _make_hook(li, gain)
            if h:
                hooks.append(h)
    else:
        h = _make_hook(vector_layer, gain, custom_vec=preloaded_vec)
        if h:
            hooks.append(h)

    try:
        return _generate_plain(prompt, max_new_tokens=max_new_tokens)
    finally:
        for h in hooks:
            h.remove()


def _stream_generate(prompt: str, max_new_tokens: int = 128, hooks_fn=None):
    """Generator che yield token uno alla volta via TextIteratorStreamer."""
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


def _stream_steered(
    prompt: str,
    concept: str,
    vector_layer: int,
    inject_layer: int,
    alpha: float,
    gain: float,
    max_new_tokens: int,
    apply_to: str,
    multi_layers=None,
    layer_configs=None,
    vector_path: str = None,
):
    """Generazione streaming con steering."""
    prompt_len = len(State.tokenizer.encode(prompt))
    dtype = next(State.model.parameters()).dtype

    preloaded_vec = None
    if vector_path:
        preloaded_vec = np.load(vector_path)

    def hooks_fn(streamer):
        hooks = []

        def _make_hook(layer_idx: int, layer_gain: float, custom_vec=None):
            if custom_vec is not None:
                vec = custom_vec
            else:
                vec = _load_vec_for_layer(concept, State.model_name, layer_idx)
            if vec is None:
                return None
            steer = torch.tensor(vec, dtype=dtype, device=State.device) * float(alpha) * float(layer_gain)
            return State.layers[layer_idx].register_forward_hook(
                make_steering_hook(steer, apply_to=apply_to, prompt_len=prompt_len)
            )

        if layer_configs:
            for cfg in layer_configs:
                h = _make_hook(int(cfg["layer"]), float(cfg["gain"]))
                if h:
                    hooks.append(h)
        elif multi_layers:
            for li in multi_layers:
                h = _make_hook(li, gain)
                if h:
                    hooks.append(h)
        else:
            h = _make_hook(vector_layer, gain, custom_vec=preloaded_vec)
            if h:
                hooks.append(h)
        return hooks

    yield from _stream_generate(prompt, max_new_tokens=max_new_tokens, hooks_fn=hooks_fn)


# ── Extract activations ───────────────────────────────────────────────────────

def _pool_hidden(hidden: torch.Tensor, attention_mask: torch.Tensor, token_position: str) -> torch.Tensor:
    """Pooling degli hidden states: 'mean' o 'last'."""
    if token_position == "last":
        lengths = attention_mask.sum(dim=1) - 1
        batch_size = hidden.shape[0]
        return torch.stack([hidden[i, lengths[i], :] for i in range(batch_size)])
    else:  # mean
        mask = attention_mask.unsqueeze(-1).float()
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-8)


def extract_activations(
    sentences_pos: list,
    sentences_neg: list,
    layers: list,
    token_position: str = "mean",
) -> dict:
    """
    Estrae hidden states dai layer richiesti per frasi pos e neg.
    Processa in batch da EXTRACT_BATCH_SIZE per evitare OOM.
    Ritorna: {"pos": {layer_str: [[...], ...]}, "neg": {layer_str: [[...], ...]}}
    """
    def _extract_sentences(sentences: list) -> dict:
        accum = {l: [] for l in layers}
        for i in range(0, len(sentences), EXTRACT_BATCH_SIZE):
            batch = sentences[i: i + EXTRACT_BATCH_SIZE]
            inputs = State.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            inputs = {k: v.to(State.device) for k, v in inputs.items()}
            attention_mask = inputs["attention_mask"]

            with torch.no_grad():
                outputs = State.model(**inputs, output_hidden_states=True)

            for layer_idx in layers:
                hidden = outputs.hidden_states[layer_idx + 1]  # 0 = embedding
                vecs = _pool_hidden(hidden, attention_mask, token_position)
                accum[layer_idx].append(vecs.cpu().float().numpy())

            del outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return {l: np.vstack(accum[l]).tolist() for l in layers}

    pos_result = _extract_sentences(sentences_pos)
    neg_result = _extract_sentences(sentences_neg)

    # Chiave stringa per JSON serialization
    return {
        "pos": {str(l): pos_result[l] for l in layers},
        "neg": {str(l): neg_result[l] for l in layers},
    }


# ── HTTP helpers ───────────────────────────────────────────────────────────────

def json_response(handler, code: int, payload: dict):
    data = json.dumps(payload).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def read_body(handler) -> bytes:
    length = int(handler.headers.get("Content-Length", "0"))
    if length == 0:
        return b""
    return handler.rfile.read(length)


# ── Handler HTTP ──────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        # Silenzia i log HTTP di default (troppo verbosi)
        pass

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/api/status":
            vram_used, vram_total = _get_vram_stats()
            return json_response(self, 200, {
                "model":        State.model_name,
                "num_layers":   State.num_layers,
                "busy":         State.busy,
                "busy_owner":   State.busy_owner,
                "vram_used_gb": vram_used,
                "vram_total_gb": vram_total,
            })

        return self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)

        # ── /api/load_model ─────────────────────────────────────────────────
        if parsed.path == "/api/load_model":
            body = read_body(self)
            try:
                payload = json.loads(body.decode("utf-8"))
            except Exception:
                return json_response(self, 400, {"error": "invalid_json"})

            requested_name = payload.get("name", "")
            settings = load_settings()
            available = settings.get("models", [])

            target = None
            for m in available:
                if m.get("name") == requested_name:
                    target = m
                    break
            if target is None:
                return json_response(self, 404, {"error": f"Model not found: {requested_name}"})

            # Noop se già caricato
            if State.model_name == requested_name and State.model is not None:
                return json_response(self, 200, {
                    "ok": True,
                    "name": State.model_name,
                    "num_layers": State.num_layers,
                    "device": str(State.device),
                    "noop": True,
                })

            if State.busy:
                return json_response(self, 503, {
                    "error": "GPU busy",
                    "busy_owner": State.busy_owner,
                })

            acquired = State.lock.acquire(timeout=30)
            if not acquired:
                return json_response(self, 503, {"error": "lock_timeout"})
            try:
                _do_load_model(target["path"], target["name"], settings)
            except Exception as e:
                return json_response(self, 500, {"error": str(e)})
            finally:
                State.lock.release()

            return json_response(self, 200, {
                "ok": True,
                "name": State.model_name,
                "num_layers": State.num_layers,
                "device": str(State.device),
            })

        # ── /api/unload_model ───────────────────────────────────────────────
        if parsed.path == "/api/unload_model":
            if State.busy:
                return json_response(self, 503, {
                    "error": "GPU busy",
                    "busy_owner": State.busy_owner,
                })

            acquired = State.lock.acquire(timeout=120)
            if not acquired:
                return json_response(self, 503, {"error": "lock_timeout"})
            try:
                _do_unload_model()
            finally:
                State.lock.release()

            return json_response(self, 200, {"ok": True})

        # ── /api/stop ───────────────────────────────────────────────────────
        if parsed.path == "/api/stop":
            State.abort_flag = True
            return json_response(self, 200, {"ok": True, "aborted": True})

        # ── /api/generate ───────────────────────────────────────────────────
        if parsed.path == "/api/generate":
            if State.model is None:
                return json_response(self, 503, {"error": "Model not loaded"})

            body = read_body(self)
            try:
                payload = json.loads(body.decode("utf-8"))
            except Exception:
                return json_response(self, 400, {"error": "invalid_json"})

            prompt        = payload.get("prompt", "")
            concept       = payload.get("concept", "hot_vs_cold")
            vector_layer  = int(payload.get("vector_layer", 0))
            inject_layer  = int(payload.get("inject_layer", 0))
            alpha         = float(payload.get("alpha", 0.0))
            gain          = float(payload.get("gain", 1.0))
            max_new_tokens = int(payload.get("max_new_tokens", 128))
            mode          = payload.get("mode", "inject")
            multi         = bool(payload.get("multi", False))
            layer_configs = payload.get("layer_configs", None)
            vector_path   = payload.get("vector_path", None)

            if not prompt:
                return json_response(self, 400, {"error": "empty_prompt"})

            # Risolvi multi_layers se richiesto
            multi_layers_resolved = None
            if multi and not layer_configs:
                # Richiede di usare tutti i layer disponibili per il concept
                slug = State.model_name.lower().replace(" ", "-").replace("_", "-")
                lib_dir = _resolve_lib_dir(concept, slug)
                if lib_dir:
                    ml = []
                    for f in glob.glob(os.path.join(lib_dir, "layer_*.npy")):
                        m = re.match(r"layer_(\d+)\.npy", os.path.basename(f))
                        if m and "_pca" not in f:
                            ml.append(int(m.group(1)))
                    multi_layers_resolved = sorted(ml)

            with State.lock:
                State.busy = True
                State.busy_owner = "generate"
                try:
                    if mode == "baseline" or alpha == 0.0:
                        text = _generate_plain(prompt, max_new_tokens=max_new_tokens)
                    else:
                        text = _generate_steered(
                            prompt=prompt,
                            concept=concept,
                            vector_layer=vector_layer,
                            inject_layer=inject_layer,
                            alpha=alpha,
                            gain=gain,
                            max_new_tokens=max_new_tokens,
                            apply_to="new",
                            multi_layers=multi_layers_resolved,
                            layer_configs=layer_configs,
                            vector_path=vector_path,
                        )
                except Exception as e:
                    return json_response(self, 500, {"error": str(e)})
                finally:
                    State.busy = False
                    State.busy_owner = None

            tokens = len(State.tokenizer.encode(text)) if State.tokenizer else 0
            return json_response(self, 200, {"text": text, "tokens": tokens})

        # ── /api/generate_stream ────────────────────────────────────────────
        if parsed.path == "/api/generate_stream":
            if State.model is None:
                return json_response(self, 503, {"error": "Model not loaded"})

            body = read_body(self)
            try:
                payload = json.loads(body.decode("utf-8"))
            except Exception:
                return json_response(self, 400, {"error": "invalid_json"})

            prompt        = payload.get("prompt", "")
            concept       = payload.get("concept", "hot_vs_cold")
            vector_layer  = int(payload.get("vector_layer", 0))
            inject_layer  = int(payload.get("inject_layer", 0))
            alpha         = float(payload.get("alpha", 0.0))
            gain          = float(payload.get("gain", 1.0))
            max_new_tokens = int(payload.get("max_new_tokens", 128))
            mode          = payload.get("mode", "inject")
            multi         = bool(payload.get("multi", False))
            layer_configs = payload.get("layer_configs", None)
            vector_path   = payload.get("vector_path", None)

            if not prompt:
                return json_response(self, 400, {"error": "empty_prompt"})

            multi_layers_resolved = None
            if multi and not layer_configs:
                slug = State.model_name.lower().replace(" ", "-").replace("_", "-") if State.model_name else ""
                lib_dir = _resolve_lib_dir(concept, slug)
                if lib_dir:
                    ml = []
                    for f in glob.glob(os.path.join(lib_dir, "layer_*.npy")):
                        m = re.match(r"layer_(\d+)\.npy", os.path.basename(f))
                        if m and "_pca" not in f:
                            ml.append(int(m.group(1)))
                    multi_layers_resolved = sorted(ml)

            # SSE headers
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()

            try:
                with State.lock:
                    State.busy = True
                    State.busy_owner = "generate_stream"
                    try:
                        if mode == "baseline" or alpha == 0.0:
                            token_gen = _stream_generate(prompt, max_new_tokens=max_new_tokens)
                        else:
                            token_gen = _stream_steered(
                                prompt=prompt,
                                concept=concept,
                                vector_layer=vector_layer,
                                inject_layer=inject_layer,
                                alpha=alpha,
                                gain=gain,
                                max_new_tokens=max_new_tokens,
                                apply_to="new",
                                multi_layers=multi_layers_resolved,
                                layer_configs=layer_configs,
                                vector_path=vector_path,
                            )
                        for token in token_gen:
                            if State.abort_flag:
                                break
                            data = json.dumps({"token": token})
                            self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
                            self.wfile.flush()
                    finally:
                        State.busy = False
                        State.busy_owner = None
            except Exception as e:
                err_data = json.dumps({"error": str(e)})
                self.wfile.write(f"data: {err_data}\n\n".encode("utf-8"))
                self.wfile.flush()
            finally:
                done_data = json.dumps({"done": True})
                self.wfile.write(f"data: {done_data}\n\n".encode("utf-8"))
                self.wfile.flush()
            return

        # ── /api/extract_activations ────────────────────────────────────────
        if parsed.path == "/api/extract_activations":
            if State.model is None:
                return json_response(self, 503, {"error": "Model not loaded"})

            body = read_body(self)
            try:
                payload = json.loads(body.decode("utf-8"))
            except Exception:
                return json_response(self, 400, {"error": "invalid_json"})

            sentences_pos  = payload.get("sentences_pos", [])
            sentences_neg  = payload.get("sentences_neg", [])
            layers         = payload.get("layers", [])
            token_position = payload.get("token_position", "mean")

            if not sentences_pos or not sentences_neg:
                return json_response(self, 400, {"error": "sentences_pos/neg cannot be empty"})
            if not layers:
                return json_response(self, 400, {"error": "layers cannot be empty"})

            with State.lock:
                State.busy = True
                State.busy_owner = "extract_activations"
                try:
                    result = extract_activations(
                        sentences_pos=sentences_pos,
                        sentences_neg=sentences_neg,
                        layers=layers,
                        token_position=token_position,
                    )
                except Exception as e:
                    return json_response(self, 500, {"error": str(e)})
                finally:
                    State.busy = False
                    State.busy_owner = None

            return json_response(self, 200, result)

        return self.send_error(404)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"[mi50_manager] Avvio su porta {PORT}")
    settings = load_settings()

    # Carica modello di default
    default_path = settings.get("model_path", "")
    default_name = None
    for m in settings.get("models", []):
        if m.get("path") == default_path:
            default_name = m.get("name")
            break
    if not default_name and default_path:
        default_name = os.path.basename(default_path)

    if default_path and os.path.isdir(default_path):
        print(f"[mi50_manager] Carico modello di default: {default_name} ({default_path})")
        try:
            _do_load_model(default_path, default_name, settings)
            print(f"[mi50_manager] Modello pronto: {default_name}  "
                  f"layers={State.num_layers}  device={State.device}")
        except Exception as e:
            print(f"[mi50_manager] WARN: impossibile caricare modello di default: {e}")
    else:
        print(f"[mi50_manager] Nessun modello caricato (model_path non trovato)")

    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"[mi50_manager] In ascolto su http://0.0.0.0:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
