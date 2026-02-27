#!/usr/bin/env python3
"""
Concept vector extraction with convergence check.

Extraction strategy:
  - Single forward pass per batch extracts hidden states for ALL deep layers at once.
  - Two extraction methods compared per layer:
      (A) mean-diff:  mean(hot_reps) - mean(cold_reps)           [classic]
      (B) pca-diff:   first singular vector of the matrix         [more stable]
                      {hot[i] - cold[i]}  (paired diffs)
  - Token pooling: configurable via settings.json (last / mean / first).
    "mean" pooling is recommended — averages semantics across all tokens.
  - Convergence check:
      * Bootstrap: N random 50%-subsets → cos-sim vs full vector
      * Incremental: vector at k pairs → cos-sim vs prev and vs full
"""
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT, "config", "settings.json")
OUTPUT_ROOT = os.path.join(ROOT, "output")
STATUS_PATH = os.path.join(OUTPUT_ROOT, "status.json")
LATEST_PATH = os.path.join(OUTPUT_ROOT, "latest.json")


@dataclass
class Settings:
    model_path: str
    trust_remote_code: bool
    max_length: int
    batch_size: int
    token_position: str
    deep_range: Tuple[float, float]
    dtype: str
    concept_name: str


def load_settings() -> Settings:
    with open(CONFIG_PATH, "r") as f:
        data = json.load(f)
    return Settings(
        model_path=data.get("model_path", ""),
        trust_remote_code=bool(data.get("trust_remote_code", False)),
        max_length=int(data.get("max_length", 256)),
        batch_size=int(data.get("batch_size", 4)),
        token_position=str(data.get("token_position", "mean")),
        deep_range=tuple(data.get("deep_range", [0.70, 0.90])),
        dtype=str(data.get("dtype", "auto")),
        concept_name=str(data.get("concept_name", "hot_vs_cold")),
    )


def write_status(payload: dict) -> None:
    payload["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    tmp = STATUS_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, STATUS_PATH)


def write_latest(run_id: str, run_dir: str) -> None:
    payload = {
        "run_id": run_id,
        "run_dir": run_dir,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    tmp = LATEST_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, LATEST_PATH)


def ensure_output_dir() -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(OUTPUT_ROOT, f"run_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def load_queries_from_config():
    hot_path = os.path.join(ROOT, "config", "queries_hot.json")
    cold_path = os.path.join(ROOT, "config", "queries_cold.json")
    if os.path.exists(hot_path) and os.path.exists(cold_path):
        with open(hot_path, "r") as f:
            hot = json.load(f)
        with open(cold_path, "r") as f:
            cold = json.load(f)
        if hot and cold:
            return hot, cold
    return None


def get_deep_layers(num_layers: int, deep_range: Tuple[float, float]) -> List[int]:
    low, high = deep_range
    low_idx = max(0, int(round(num_layers * low)))
    high_idx = min(num_layers - 1, int(round(num_layers * high)))
    if high_idx < low_idx:
        high_idx = low_idx
    return list(range(low_idx, high_idx + 1))


def _pool(hidden: torch.Tensor, attention_mask: torch.Tensor, token_position: str) -> torch.Tensor:
    """Pool [B, T, D] → [B, D]."""
    if token_position == "last":
        seq_lengths = attention_mask.sum(dim=1) - 1
        return torch.stack([hidden[b, seq_lengths[b], :] for b in range(hidden.size(0))])
    elif token_position == "mean":
        mask_f = attention_mask.unsqueeze(-1).float()
        return (hidden * mask_f).sum(1) / mask_f.sum(1)
    elif token_position == "first":
        return hidden[:, 0, :]
    else:
        raise ValueError(f"Unknown token_position: {token_position}")


def extract_deep_layers(
    model, tokenizer, texts: List[str],
    deep_layers: List[int], token_position: str,
    batch_size: int, max_length: int,
) -> Dict[int, np.ndarray]:
    """
    Single forward pass per batch — extracts all deep layers at once.
    Returns {layer_idx: float32 array[n_texts, hidden_dim]}.
    """
    device = next(model.parameters()).device
    layer_accum: Dict[int, list] = {l: [] for l in deep_layers}

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        attention_mask = inputs["attention_mask"]

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        for layer_idx in deep_layers:
            hidden = outputs.hidden_states[layer_idx + 1]   # 0 = embedding
            vecs = _pool(hidden, attention_mask, token_position)
            layer_accum[layer_idx].append(vecs.cpu().float().numpy())

        del outputs
        torch.cuda.empty_cache()

    return {l: np.vstack(layer_accum[l]) for l in deep_layers}


# ---------------------------------------------------------------------------
# Concept vector computation
# ---------------------------------------------------------------------------

def compute_mean_diff(hot: np.ndarray, cold: np.ndarray) -> np.ndarray:
    """Classic: mean(hot) - mean(cold), unit-normalised."""
    vec = hot.mean(0) - cold.mean(0)
    return vec / (np.linalg.norm(vec) + 1e-8)


def compute_pca_diff(hot: np.ndarray, cold: np.ndarray) -> np.ndarray:
    """
    PCA on paired differences.

    diffs[i] = hot[i] - cold[i]  cancels topic-specific signal, isolating
    only the temperature-relevant direction.  The first right singular vector
    of the centred diff matrix is the direction of maximum consistent variance.

    Sign convention: aligned with mean-diff (positive dot product).
    """
    diffs = hot - cold                              # [n, D]
    diffs_c = diffs - diffs.mean(0)                 # centre rows
    _, _, Vt = np.linalg.svd(diffs_c, full_matrices=False)
    direction = Vt[0]                               # first principal direction
    # Ensure same sign as mean-diff
    mean_d = hot.mean(0) - cold.mean(0)
    if np.dot(direction, mean_d) < 0:
        direction = -direction
    return direction / (np.linalg.norm(direction) + 1e-8)


def diff_coherence(hot: np.ndarray, cold: np.ndarray) -> float:
    """
    Mean cosine similarity between all normalised pairwise diffs.
    = 1.0 when every diff points in exactly the same direction (perfect signal).
    Near 0 when diffs are random (no consistent signal).
    """
    diffs = hot - cold
    norms = np.linalg.norm(diffs, axis=1, keepdims=True) + 1e-8
    unit_diffs = diffs / norms
    # Gram matrix of cosines
    G = unit_diffs @ unit_diffs.T
    n = len(diffs)
    # Mean of off-diagonal elements
    off_diag = (G.sum() - np.trace(G)) / (n * (n - 1))
    return float(off_diag)


# ---------------------------------------------------------------------------
# Convergence check
# ---------------------------------------------------------------------------

def convergence_report(
    hot: np.ndarray,
    cold: np.ndarray,
    method: str = "pca",        # "pca" or "mean"
    step: int = 5,
    n_bootstrap: int = 30,
    seed: int = 42,
) -> dict:
    """
    Bootstrap stability + incremental convergence for a concept vector.

    method="pca"  → uses compute_pca_diff  on each subset
    method="mean" → uses compute_mean_diff on each subset
    """
    n = len(hot)
    assert n == len(cold)

    compute = compute_pca_diff if method == "pca" else compute_mean_diff
    full_unit = compute(hot, cold)

    # Incremental convergence
    incremental = []
    prev_unit = None
    for k in range(step, n + 1, step):
        unit = compute(hot[:k], cold[:k])
        cos_vs_full = float(np.dot(unit, full_unit))
        cos_vs_prev = float(np.dot(unit, prev_unit)) if prev_unit is not None else None
        incremental.append({
            "n_pairs": k,
            "cos_vs_full": round(cos_vs_full, 5),
            "cos_vs_prev": round(cos_vs_prev, 5) if cos_vs_prev is not None else None,
        })
        prev_unit = unit

    # Bootstrap
    rng = np.random.default_rng(seed)
    boot_sims = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n // 2, replace=False)
        unit = compute(hot[idx], cold[idx])
        boot_sims.append(float(np.dot(unit, full_unit)))

    boot_arr = np.array(boot_sims)
    return {
        "n_pairs": n,
        "method": method,
        "bootstrap_cos_mean": round(float(boot_arr.mean()), 5),
        "bootstrap_cos_min":  round(float(boot_arr.min()),  5),
        "bootstrap_cos_std":  round(float(boot_arr.std()),  5),
        "converged": bool(boot_arr.min() > 0.995),
        "incremental": incremental,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    settings = load_settings()
    if not settings.model_path:
        raise SystemExit("model_path vuoto in config/settings.json")

    out_dir = ensure_output_dir()
    run_id = os.path.basename(out_dir)
    write_latest(run_id, out_dir)

    with open(os.path.join(out_dir, "settings_used.json"), "w") as f:
        json.dump(settings.__dict__, f, indent=2)

    loaded = load_queries_from_config()
    if not loaded:
        raise SystemExit("Query files not found in config/")
    hot, cold = loaded

    n_pairs = min(len(hot), len(cold))
    hot, cold = hot[:n_pairs], cold[:n_pairs]
    print(f"Sentences: {n_pairs} pairs  |  token_position: {settings.token_position}")

    with open(os.path.join(out_dir, "queries_hot.json"),  "w") as f:
        json.dump(hot,  f, indent=2)
    with open(os.path.join(out_dir, "queries_cold.json"), "w") as f:
        json.dump(cold, f, indent=2)

    # ---- load model ----
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                 "float32": torch.float32, "auto": None}
    torch_dtype = dtype_map.get(settings.dtype, None)
    model_kwargs = {"trust_remote_code": settings.trust_remote_code, "low_cpu_mem_usage": True}
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype

    write_status({"run_id": run_id, "phase": "load_model", "layer": None,
                  "progress_percent": 5, "model_path": settings.model_path,
                  "device": "pending", "notes": f"{n_pairs} pairs, pos={settings.token_position}"})

    tokenizer = AutoTokenizer.from_pretrained(
        settings.model_path, trust_remote_code=settings.trust_remote_code, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(settings.model_path, **model_kwargs)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    num_layers  = model.config.num_hidden_layers
    deep_layers = get_deep_layers(num_layers, settings.deep_range)
    print(f"Model: {num_layers} layers  |  deep layers: {deep_layers}")

    # ---- extract representations ----
    write_status({"run_id": run_id, "phase": "extract_hot", "layer": None,
                  "progress_percent": 15, "model_path": settings.model_path,
                  "device": str(device), "notes": "Extracting HOT reps..."})

    print(f"\n[1/4] HOT representations (single pass, all layers)...")
    t0 = time.time()
    hot_reps = extract_deep_layers(model, tokenizer, hot, deep_layers,
                                   settings.token_position, settings.batch_size, settings.max_length)
    print(f"      {time.time()-t0:.1f}s  shape: {next(iter(hot_reps.values())).shape}")

    write_status({"run_id": run_id, "phase": "extract_cold", "layer": None,
                  "progress_percent": 40, "model_path": settings.model_path,
                  "device": str(device), "notes": "Extracting COLD reps..."})

    print(f"[2/4] COLD representations...")
    t0 = time.time()
    cold_reps = extract_deep_layers(model, tokenizer, cold, deep_layers,
                                    settings.token_position, settings.batch_size, settings.max_length)
    print(f"      {time.time()-t0:.1f}s")

    write_status({"run_id": run_id, "phase": "compute_vectors", "layer": None,
                  "progress_percent": 65, "model_path": settings.model_path,
                  "device": str(device), "notes": "Computing vectors + convergence..."})

    # ---- compute + report ----
    print(f"\n[3/4] Computing concept vectors + convergence...\n")

    hdr = (f"{'Layer':>5}  {'coherence':>9}  {'cos(µ+,µ-)':>10}  "
           f"{'boot(mean)':>10}  {'boot(min)':>9}  "
           f"{'pca≈mean':>8}  {'converged':>9}")
    print(hdr)
    print("-" * len(hdr))

    results = {}
    for layer_idx in deep_layers:
        h = hot_reps[layer_idx]
        c = cold_reps[layer_idx]

        # Both directions
        unit_mean = compute_mean_diff(h, c)
        unit_pca  = compute_pca_diff(h, c)
        cos_pca_vs_mean = float(np.dot(unit_pca, unit_mean))

        # Quality metrics on PCA vector (primary)
        proj_hot  = h @ unit_pca
        proj_cold = c @ unit_pca
        sep   = float(proj_hot.mean()  - proj_cold.mean())
        noise = float((proj_hot.std()  + proj_cold.std()) / 2 + 1e-8)
        snr   = sep / noise

        cos_means = float(
            np.dot(h.mean(0), c.mean(0)) /
            (np.linalg.norm(h.mean(0)) * np.linalg.norm(c.mean(0)) + 1e-8)
        )
        coherence = diff_coherence(h, c)

        # Convergence on PCA method
        conv = convergence_report(h, c, method="pca")

        print(f"{layer_idx:>5}  {coherence:>9.5f}  {cos_means:>10.5f}  "
              f"{conv['bootstrap_cos_mean']:>10.5f}  {conv['bootstrap_cos_min']:>9.5f}  "
              f"{cos_pca_vs_mean:>8.5f}  {'YES' if conv['converged'] else 'NO':>9}")

        results[str(layer_idx)] = {
            "layer_idx": layer_idx,
            "coherence": round(coherence, 5),
            "cos_means": round(cos_means, 5),
            "sep_snr": round(snr, 4),
            "cos_pca_vs_mean": round(cos_pca_vs_mean, 5),
            "convergence_pca": conv,
            "convergence_mean": convergence_report(h, c, method="mean"),
        }

        # Save mean-diff as the primary concept vector (generalises better at deep layers)
        np.save(
            os.path.join(out_dir, f"concept_{settings.concept_name}_layer_{layer_idx}.npy"),
            unit_mean,
        )
        # Also save PCA vector for reference / comparison
        np.save(
            os.path.join(out_dir, f"concept_{settings.concept_name}_pca_layer_{layer_idx}.npy"),
            unit_pca,
        )

    print("-" * len(hdr))

    # Best layer by bootstrap_cos_min (most stable)
    best_layer = max(results, key=lambda l: results[l]["convergence_pca"]["bootstrap_cos_min"])
    best = results[best_layer]
    print(f"\nBest layer by stability: {best_layer}  "
          f"(boot_min={best['convergence_pca']['bootstrap_cos_min']:.5f}, "
          f"coherence={best['coherence']:.5f})")

    # Incremental table for best layer
    print(f"\nIncremental convergence — layer {best_layer} (PCA method):")
    print(f"  {'pairs':>7}  {'cos_vs_prev':>12}  {'cos_vs_full':>12}")
    for e in best["convergence_pca"]["incremental"]:
        cprev = f"{e['cos_vs_prev']:.5f}" if e["cos_vs_prev"] is not None else "       —"
        print(f"  {e['n_pairs']:>7}  {cprev:>12}  {e['cos_vs_full']:>12.5f}")

    # ---- save summary ----
    write_status({"run_id": run_id, "phase": "finalize", "layer": None,
                  "progress_percent": 95, "model_path": settings.model_path,
                  "device": str(device), "notes": "Writing results..."})

    summary = {
        "concept": settings.concept_name,
        "model_path": settings.model_path,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_layers": num_layers,
        "deep_layers": deep_layers,
        "num_hot": len(hot),
        "num_cold": len(cold),
        "n_pairs": n_pairs,
        "token_position": settings.token_position,
        "vector_method": "pca_diff",
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    write_status({"run_id": run_id, "phase": "done", "layer": None,
                  "progress_percent": 100, "model_path": settings.model_path,
                  "device": str(device), "notes": f"Done: {out_dir}"})

    print(f"\n[4/4] Done. Output: {out_dir}")


if __name__ == "__main__":
    main()
