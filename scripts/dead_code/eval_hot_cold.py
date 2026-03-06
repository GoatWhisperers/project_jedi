#!/usr/bin/env python3
import json
import os
import time
from typing import List, Dict

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT, "config", "settings.json")
OUTPUT_ROOT = os.path.join(ROOT, "output")


def load_settings():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def find_latest_run() -> str:
    runs = [d for d in os.listdir(OUTPUT_ROOT) if d.startswith("run_")]
    if not runs:
        raise SystemExit("Nessun run trovato in output/")
    runs.sort()
    return os.path.join(OUTPUT_ROOT, runs[-1])


def load_vectors(run_dir: str) -> Dict[int, np.ndarray]:
    vectors = {}
    for name in os.listdir(run_dir):
        if name.startswith("concept_") and name.endswith(".npy"):
            try:
                layer = int(name.split("_")[-1].replace(".npy", ""))
            except ValueError:
                continue
            vectors[layer] = np.load(os.path.join(run_dir, name))
    if not vectors:
        raise SystemExit("Nessun vettore trovato nel run")
    return dict(sorted(vectors.items()))


def build_eval_queries():
    # 10 held-out hot/cold sentences — NOT in queries_hot/cold.json training set
    hot = [
        "The iron skillet was still hot from breakfast.",
        "She burned her tongue on the hot soup.",
        "The exhaust pipe glowed from the engine heat.",
        "He sweated through his shirt in the midday sun.",
        "The boiling water filled the thermos with steam.",
        "The summer air inside the parked car was stifling.",
        "The fire crackled and gave off tremendous heat.",
        "The fresh waffles steamed on the plate.",
        "He couldn't hold the pot without oven gloves.",
        "The metal was still warm hours after the forge.",
    ]
    cold = [
        "The mountain stream was bracingly cold.",
        "She shivered and pulled her coat tighter.",
        "The frozen ground crunched under his boots.",
        "He couldn't feel his toes after the winter hike.",
        "The morning air nipped sharply at her cheeks.",
        "Frost patterns covered the window pane at dawn.",
        "The icy handshake startled him.",
        "The lake was too cold for swimming.",
        "The refrigerator held everything just above freezing.",
        "Her breath misted in the cold morning air.",
    ]
    neutral = [
        "The book lay on the table.",
        "She read the letter twice.",
        "The cat jumped onto the chair.",
        "He opened the window in the morning.",
        "The train arrived on time.",
    ]
    return {"hot": hot, "cold": cold, "neutral": neutral}


def extract_all_layers(model, tokenizer, texts, max_length=256, batch_size=4,
                       token_position="mean"):
    """
    Single forward pass returning hidden states for all layers.
    token_position: "mean" (default, matches probe_hot_cold.py), "last", or "first".
    """
    device = next(model.parameters()).device
    all_layer_vecs: Dict[int, list] = {}
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
        attention_mask = inputs.get("attention_mask")
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        # hidden_states[0] = embedding, hidden_states[l+1] = layer l output
        for layer_idx in range(len(outputs.hidden_states) - 1):
            hidden = outputs.hidden_states[layer_idx + 1]
            if token_position == "last":
                seq_lengths = attention_mask.sum(dim=1) - 1
                vecs = torch.stack([hidden[b, seq_lengths[b], :] for b in range(hidden.size(0))])
            elif token_position == "mean":
                mask_f = attention_mask.unsqueeze(-1).float()
                vecs = (hidden * mask_f).sum(1) / mask_f.sum(1)
            elif token_position == "first":
                vecs = hidden[:, 0, :]
            else:
                raise ValueError(f"Unknown token_position: {token_position}")
            if layer_idx not in all_layer_vecs:
                all_layer_vecs[layer_idx] = []
            all_layer_vecs[layer_idx].append(vecs.cpu().numpy())
        del outputs
        torch.cuda.empty_cache()
    return {layer_idx: np.vstack(batches) for layer_idx, batches in all_layer_vecs.items()}


def main():
    settings = load_settings()
    model_path = settings.get("model_path", "")
    if not model_path:
        raise SystemExit("model_path vuoto in config/settings.json")

    run_dir = find_latest_run()
    vectors = load_vectors(run_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=bool(settings.get("trust_remote_code", False)), use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=bool(settings.get("trust_remote_code", False)),
        low_cpu_mem_usage=True,
    )
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    queries = build_eval_queries()
    results = {
        "run_dir": run_dir,
        "model_path": model_path,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "projections": {},
        "ranked_layers": []
    }

    all_texts = sum(queries.values(), [])
    token_position = settings.get("token_position", "mean")
    # One forward pass: extract hidden states for every layer
    all_reps = extract_all_layers(model, tokenizer, all_texts,
                                  max_length=int(settings.get("max_length", 256)),
                                  batch_size=int(settings.get("batch_size", 4)),
                                  token_position=token_position)

    idx = 0
    group_slices = {}
    for k, lst in queries.items():
        group_slices[k] = slice(idx, idx + len(lst))
        idx += len(lst)

    ranking = []

    for layer, concept_vec in vectors.items():
        # Use the layer-specific hidden states for this concept vector
        reps = all_reps.get(layer)
        if reps is None:
            continue
        concept_unit = concept_vec / (np.linalg.norm(concept_vec) + 1e-8)
        proj = reps @ concept_unit
        layer_res = {}
        for k, sl in group_slices.items():
            layer_res[k] = {
                "mean": float(np.mean(proj[sl])),
                "std": float(np.std(proj[sl])),
                "samples": [float(x) for x in proj[sl]],
            }
        results["projections"][str(layer)] = layer_res

        score = layer_res["hot"]["mean"] - layer_res["cold"]["mean"]
        ranking.append({"layer": layer, "score": float(score)})

    ranking.sort(key=lambda x: x["score"], reverse=True)
    results["ranked_layers"] = ranking

    out_path = os.path.join(run_dir, "projections_eval.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("Eval complete")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
