#!/usr/bin/env python3
"""
Generic concept vector extraction.

Reads a concept JSON from config/concepts/ and extracts concept vectors
for all deep layers, saving results to output/vector_library/{category}/{concept}/{model_name}/.

Usage:
    python scripts/probe_concept.py --concept config/concepts/luce_vs_buio.json
    python scripts/probe_concept.py --concept config/concepts/luce_vs_buio.json --eval
    python scripts/probe_concept.py --concept config/concepts/luce_vs_buio.json --model "Gemma2-Uncensored"

Model loading is delegated to mi50_manager (porta 8020).
"""
import argparse
import json
import os
import re
import sys
import time
from typing import Dict, List, Optional, Tuple
from urllib.request import urlopen, Request

import numpy as np

# Reuse functions from probe_hot_cold in the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_hot_cold import (
    compute_mean_diff,
    compute_pca_diff,
    convergence_report,
    diff_coherence,
    get_deep_layers,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT, "config", "settings.json")
OUTPUT_ROOT = os.path.join(ROOT, "output")
STATUS_PATH = os.path.join(OUTPUT_ROOT, "status.json")
VECTOR_LIB_ROOT = os.path.join(OUTPUT_ROOT, "vector_library")

MI50_URL = os.environ.get("MI50_URL", "http://localhost:8020")

# Number of sentences from each side held out for evaluation
EVAL_HOLDOUT = 5


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def normalize_name_for_path(name: str) -> str:
    """Lowercase, spaces→hyphens, strip chars that are invalid in paths."""
    name = name.lower().replace(" ", "-")
    name = re.sub(r"[^a-z0-9_\-]", "", name)
    return name


def write_status_extended(payload: dict) -> None:
    payload["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    tmp = STATUS_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, STATUS_PATH)


def load_settings_raw() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def resolve_model(settings: dict, model_name_override: Optional[str]) -> Tuple[str, str]:
    """
    Returns (model_path, model_name).
    If --model is given, look it up in settings["models"] list; otherwise use settings["model_path"].
    """
    models_list = settings.get("models", [])
    if model_name_override:
        for m in models_list:
            if m.get("name") == model_name_override:
                return m["path"], m["name"]
        return model_name_override, os.path.basename(model_name_override)
    model_path = settings.get("model_path", "")
    for m in models_list:
        if m.get("path") == model_path:
            return model_path, m["name"]
    return model_path, os.path.basename(model_path)


# ---------------------------------------------------------------------------
# mi50_manager HTTP helpers
# ---------------------------------------------------------------------------

def _mi50_post(path: str, payload: dict, timeout: int = 1800) -> dict:
    """POST a mi50_manager. Ritorna dict JSON o {"error": ...}."""
    url = f"{MI50_URL}{path}"
    data = json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"error": str(e)}


def mi50_load_model(model_name: str) -> bool:
    """
    Chiede a mi50_manager di caricare il modello.
    Noop se già caricato. Ritorna True se ok.
    """
    print(f"  [mi50] Richiedo caricamento modello: {model_name}")
    result = _mi50_post("/api/load_model", {"name": model_name})
    if result.get("ok") or result.get("noop"):
        noop = result.get("noop", False)
        print(f"  [mi50] Modello pronto: {result.get('name', model_name)}"
              f"  layers={result.get('num_layers', '?')}"
              f"  {'(già caricato)' if noop else ''}")
        return True
    else:
        print(f"  [mi50] ERRORE load_model: {result.get('error', 'unknown')}")
        return False


def mi50_extract_activations(
    sentences_pos: List[str],
    sentences_neg: List[str],
    layers: List[int],
    token_position: str = "mean",
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    Chiede a mi50_manager di estrarre gli hidden states.
    Ritorna (pos_reps, neg_reps): dict {layer_idx: float32 array[n_texts, hidden_dim]}.
    """
    payload = {
        "sentences_pos":  sentences_pos,
        "sentences_neg":  sentences_neg,
        "layers":         layers,
        "token_position": token_position,
    }
    result = _mi50_post("/api/extract_activations", payload, timeout=1800)
    if "error" in result and "pos" not in result:
        raise RuntimeError(f"mi50_manager extract_activations error: {result['error']}")

    pos_reps = {int(k): np.array(v, dtype=np.float32) for k, v in result["pos"].items()}
    neg_reps = {int(k): np.array(v, dtype=np.float32) for k, v in result["neg"].items()}
    return pos_reps, neg_reps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generic concept vector extraction")
    parser.add_argument("--concept", required=True,
                        help="Path to concept JSON (e.g. config/concepts/luce_vs_buio.json)")
    parser.add_argument("--model", default=None,
                        help="Model name from settings.json models list (e.g. 'Gemma3-1B-IT')")
    parser.add_argument("--eval", action="store_true",
                        help="Evaluate on held-out sentences and save eval.json")
    parser.add_argument("--output-root", default=None,
                        help="Override output root directory (default: output/vector_library)")
    args = parser.parse_args()

    # --- Load concept ---
    concept_path = args.concept
    if not os.path.isabs(concept_path):
        cwd_path  = os.path.join(os.getcwd(), concept_path)
        root_path = os.path.join(ROOT, concept_path)
        concept_path = cwd_path if os.path.exists(cwd_path) else root_path
    with open(concept_path, "r") as f:
        concept_data = json.load(f)

    concept_name = concept_data["concept"]
    category     = concept_data.get("category", "uncategorized")
    pos_sents    = concept_data["positive"]
    neg_sents    = concept_data["negative"]

    n_total = min(len(pos_sents), len(neg_sents))
    pos_sents = pos_sents[:n_total]
    neg_sents = neg_sents[:n_total]

    # Split train / held-out
    n_holdout = EVAL_HOLDOUT if args.eval and n_total > EVAL_HOLDOUT * 2 else 0
    n_train   = n_total - n_holdout
    pos_train, neg_train = pos_sents[:n_train], neg_sents[:n_train]
    pos_eval,  neg_eval  = pos_sents[n_train:], neg_sents[n_train:]

    print(f"Concept   : {concept_name}  [{category}]")
    print(f"Pairs     : {n_train} train" + (f"  +  {len(pos_eval)} held-out" if n_holdout else ""))

    # --- Load settings ---
    settings = load_settings_raw()
    model_path, model_name = resolve_model(settings, args.model)
    if not model_path:
        raise SystemExit("model_path non configurato in config/settings.json")

    token_position = str(settings.get("token_position", "mean"))
    deep_range     = tuple(settings.get("deep_range", [0.70, 0.90]))

    # --- Output dir ---
    concept_key = normalize_name_for_path(concept_name)
    model_key   = normalize_name_for_path(model_name)
    lib_root    = args.output_root if args.output_root else VECTOR_LIB_ROOT
    out_dir     = os.path.join(lib_root, category, concept_key, model_key)
    os.makedirs(out_dir, exist_ok=True)
    run_id = f"{concept_key}_{model_key}_{time.strftime('%Y%m%d_%H%M%S')}"
    print(f"Output    : {out_dir}")

    # --- Carica modello su mi50_manager ---
    write_status_extended({
        "run_id": run_id, "phase": "load_model",
        "concept": concept_name, "category": category, "model": model_name,
        "query_current": 0, "query_total": n_train, "query_pct": 0.0,
        "layer_current": 0, "layer_total": 0,
        "elapsed_s": 0.0, "eta_s": None, "throughput_q_per_s": 0.0,
        "model_path": model_path, "device": "mi50_manager", "progress_percent": 5,
        "notes": f"Loading model {model_name} via mi50_manager…",
    })

    ok = mi50_load_model(model_name)
    if not ok:
        print(f"\n[ERRORE] Impossibile caricare {model_name} su mi50_manager ({MI50_URL})")
        print("  Verificare che mi50_manager sia in esecuzione.")
        sys.exit(2)

    # Legge num_layers dal manager
    try:
        with urlopen(f"{MI50_URL}/api/status", timeout=10) as r:
            st = json.loads(r.read())
        num_layers = st.get("num_layers") or 0
    except Exception:
        num_layers = 0

    deep_layers = get_deep_layers(num_layers, deep_range) if num_layers else []
    if not deep_layers:
        # Fallback: stima dai settings
        for m in settings.get("models", []):
            if m.get("name") == model_name:
                nl = m.get("num_layers", 0)
                dr = m.get("deep_range", list(deep_range))
                deep_layers = get_deep_layers(nl, tuple(dr))
                num_layers  = nl
                break

    n_layers = len(deep_layers)
    print(f"Model     : {num_layers} layers total  |  deep layers: {deep_layers}")
    print(f"Pooling   : {token_position}")

    # ---- Extract POS + NEG representations ----
    t_start = time.time()

    def _update_status(phase_label, q_done, q_total):
        elapsed = time.time() - t_start
        q_pct = q_done / q_total * 100 if q_total > 0 else 0
        thr = q_done / elapsed if elapsed > 0 else 0
        eta = (q_total - q_done) / thr if thr > 0 else None
        write_status_extended({
            "run_id": run_id, "phase": phase_label,
            "concept": concept_name, "category": category, "model": model_name,
            "query_current": q_done, "query_total": q_total, "query_pct": round(q_pct, 1),
            "layer_current": 0, "layer_total": n_layers,
            "elapsed_s": round(elapsed, 1),
            "eta_s": round(eta, 1) if eta is not None else None,
            "throughput_q_per_s": round(thr, 3),
            "model_path": model_path, "device": "mi50_manager",
            "progress_percent": max(10, min(90, int(q_pct * 0.7))),
            "notes": f"{phase_label}…",
        })

    print(f"\n[1/4] Extracting POS reps ({n_train} sentences)…")
    _update_status("extract_pos", 0, n_train * 2)

    pos_reps, neg_reps = mi50_extract_activations(
        pos_train, neg_train, deep_layers, token_position
    )

    print(f"      pos shape: {next(iter(pos_reps.values())).shape}")
    print(f"[2/4] NEG already extracted (batched with POS).")
    _update_status("extract_neg", n_train * 2, n_train * 2)

    write_status_extended({
        "run_id": run_id, "phase": "compute_vectors",
        "concept": concept_name, "category": category, "model": model_name,
        "query_current": n_train * 2, "query_total": n_train * 2, "query_pct": 100.0,
        "layer_current": 0, "layer_total": n_layers,
        "elapsed_s": round(time.time() - t_start, 1), "eta_s": None,
        "throughput_q_per_s": round(n_train * 2 / max(time.time() - t_start, 1e-3), 3),
        "model_path": model_path, "device": "mi50_manager", "progress_percent": 70,
        "notes": "Computing concept vectors…",
    })

    # ---- Compute vectors per layer ----
    print(f"\n[3/4] Computing concept vectors…\n")
    hdr = (f"{'Layer':>5}  {'coherence':>9}  {'cos(µ+,µ-)':>10}  "
           f"{'boot(mean)':>10}  {'boot(min)':>9}  {'pca≈mean':>8}  {'converged':>9}")
    print(hdr)
    print("-" * len(hdr))

    results = {}
    for li, layer_idx in enumerate(deep_layers):
        h = pos_reps[layer_idx]
        c = neg_reps[layer_idx]

        unit_mean = compute_mean_diff(h, c)
        unit_pca  = compute_pca_diff(h, c)
        cos_pca_vs_mean = float(np.dot(unit_pca, unit_mean))

        proj_pos  = h @ unit_pca
        proj_neg  = c @ unit_pca
        sep   = float(proj_pos.mean() - proj_neg.mean())
        noise = float((proj_pos.std() + proj_neg.std()) / 2 + 1e-8)
        snr   = sep / noise

        cos_means = float(
            np.dot(h.mean(0), c.mean(0)) /
            (np.linalg.norm(h.mean(0)) * np.linalg.norm(c.mean(0)) + 1e-8)
        )
        coherence = diff_coherence(h, c)
        conv      = convergence_report(h, c, method="pca")

        print(f"{layer_idx:>5}  {coherence:>9.5f}  {cos_means:>10.5f}  "
              f"{conv['bootstrap_cos_mean']:>10.5f}  {conv['bootstrap_cos_min']:>9.5f}  "
              f"{cos_pca_vs_mean:>8.5f}  {'YES' if conv['converged'] else 'NO':>9}")

        results[str(layer_idx)] = {
            "layer_idx":        layer_idx,
            "coherence":        round(coherence, 5),
            "cos_means":        round(cos_means, 5),
            "sep_snr":          round(snr, 4),
            "cos_pca_vs_mean":  round(cos_pca_vs_mean, 5),
            "convergence_pca":  conv,
            "convergence_mean": convergence_report(h, c, method="mean"),
        }

        # Save vectors
        np.save(os.path.join(out_dir, f"layer_{layer_idx}.npy"), unit_mean)
        np.save(os.path.join(out_dir, f"layer_{layer_idx}_pca.npy"), unit_pca)

        write_status_extended({
            "run_id": run_id, "phase": "compute_vectors",
            "concept": concept_name, "category": category, "model": model_name,
            "query_current": n_train * 2, "query_total": n_train * 2, "query_pct": 100.0,
            "layer_current": li + 1, "layer_total": n_layers,
            "elapsed_s": round(time.time() - t_start, 1), "eta_s": None,
            "throughput_q_per_s": 0.0,
            "model_path": model_path, "device": "mi50_manager",
            "progress_percent": 70 + int((li + 1) / n_layers * 20),
            "notes": f"Layer {layer_idx} done (snr={snr:.3f})",
        })

    print("-" * len(hdr))

    best_layer = max(results, key=lambda l: results[l]["convergence_pca"]["bootstrap_cos_min"])
    best = results[best_layer]
    print(f"\nBest layer by stability: {best_layer}  "
          f"(boot_min={best['convergence_pca']['bootstrap_cos_min']:.5f})")

    # ---- Eval on held-out (if requested) ----
    eval_data = None
    if args.eval and pos_eval:
        print(f"\n[3b] Evaluating on {len(pos_eval)} held-out pairs…")
        pos_eval_reps, neg_eval_reps = mi50_extract_activations(
            pos_eval, neg_eval, deep_layers, token_position
        )

        eval_layers = {}
        for layer_idx in deep_layers:
            he = pos_eval_reps[layer_idx]
            ce = neg_eval_reps[layer_idx]
            vec     = np.load(os.path.join(out_dir, f"layer_{layer_idx}.npy"))
            proj_pos = he @ vec
            proj_neg = ce @ vec
            sep   = float(proj_pos.mean() - proj_neg.mean())
            noise = float((proj_pos.std() + proj_neg.std()) / 2 + 1e-8)
            snr   = sep / noise
            eval_layers[str(layer_idx)] = {
                "snr":      round(snr, 4),
                "sep":      round(sep, 5),
                "noise":    round(noise, 5),
                "pos_mean": round(float(proj_pos.mean()), 5),
                "neg_mean": round(float(proj_neg.mean()), 5),
            }
            print(f"  layer {layer_idx:>3}: held-out SNR = {snr:+.3f}")

        best_eval_layer = max(eval_layers, key=lambda l: eval_layers[l]["snr"])
        eval_data = {
            "n_held_out": len(pos_eval),
            "layers":     eval_layers,
            "best_layer": int(best_eval_layer),
            "best_snr":   eval_layers[best_eval_layer]["snr"],
        }
        with open(os.path.join(out_dir, "eval.json"), "w") as f:
            json.dump(eval_data, f, indent=2)
        print(f"\n  Best held-out layer: {best_eval_layer}  "
              f"(SNR={eval_layers[best_eval_layer]['snr']:+.3f})")

    # ---- Save meta.json ----
    meta = {
        "concept":        concept_name,
        "category":       category,
        "model_path":     model_path,
        "model_name":     model_name,
        "n_pairs":        n_train,
        "n_held_out":     len(pos_eval) if pos_eval else 0,
        "token_position": token_position,
        "deep_range":     list(deep_range),
        "layers":         deep_layers,
        "vector_method":  "mean_diff",
        "date":           time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_id":         run_id,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # ---- Save full results summary ----
    summary = {
        "concept":        concept_name,
        "category":       category,
        "model_path":     model_path,
        "model_name":     model_name,
        "timestamp":      time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_layers":     num_layers,
        "deep_layers":    deep_layers,
        "n_pairs_train":  n_train,
        "token_position": token_position,
        "vector_method":  "mean_diff",
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    write_status_extended({
        "run_id": run_id, "phase": "done",
        "concept": concept_name, "category": category, "model": model_name,
        "query_current": n_train * 2, "query_total": n_train * 2, "query_pct": 100.0,
        "layer_current": n_layers, "layer_total": n_layers,
        "elapsed_s": round(time.time() - t_start, 1), "eta_s": 0.0,
        "throughput_q_per_s": round(n_train * 2 / max(time.time() - t_start, 1e-3), 3),
        "model_path": model_path, "device": "mi50_manager", "progress_percent": 100,
        "notes": f"Done: {out_dir}",
    })

    print(f"\n[4/4] Done.")
    print(f"  Output   : {out_dir}")
    print(f"  Layers   : {deep_layers}")
    print(f"  Best     : layer {best_layer}  (stability)")
    if eval_data:
        print(f"  Best SNR : layer {eval_data['best_layer']}  (held-out SNR={eval_data['best_snr']:+.3f})")


if __name__ == "__main__":
    main()
