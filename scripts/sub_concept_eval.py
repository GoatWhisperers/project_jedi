"""
sub_concept_eval.py — Valutazione comparativa di sub-vettori concept.

Parte del loop di auto-analisi (Fase 2 — Sub-Concept Decomposition).

Per ogni coppia di sub-vettori estratti:
  1. Stesso prompt neutro → steered con sub_A vs sub_B vs broad
  2. M40 giudica: gli output sono fenomenologicamente distinti?
  3. Produce un verdict con score di distinzione e feedback per Step 1

Output:
  output/sub_concept_evals/{parent}/{model_slug}/eval_v{N}.json

Uso:
  python scripts/sub_concept_eval.py --concept hot_vs_cold --version 1
  python scripts/sub_concept_eval.py --concept hot_vs_cold --version 1 --model Gemma2-Uncensored
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Optional

import requests

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT             = Path(__file__).resolve().parent.parent
SUB_CONCEPTS_DIR = ROOT / "config" / "sub_concepts"
VECTOR_LIB_ROOT  = ROOT / "output" / "vector_library"
EVAL_OUTPUT_DIR  = ROOT / "output" / "sub_concept_evals"
CATALOG_PATH     = ROOT / "output" / "catalog.json"

# ── Defaults ───────────────────────────────────────────────────────────────────
DEFAULT_STEERING_URL    = "http://localhost:8010"
DEFAULT_M40_URL         = "http://localhost:11435"
DEFAULT_MAX_TOKENS_GEN  = 120
DEFAULT_MAX_TOKENS_EVAL = 500
DEFAULT_TEMPERATURE     = 0.2

# Prompt neutri per il test di distinzione (non nominano nessun concetto)
NEUTRAL_PROMPTS = [
    "Describe what happens in your body right now.",
    "Write about a physical sensation you are experiencing at this moment.",
    "Describe the quality of your immediate physical environment.",
    "Write about what the inside of your mouth feels like right now.",
    "Describe the sensation in your hands at this moment.",
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_json(raw: str, fallback: dict) -> dict:
    """Estrae il primo JSON valido dalla stringa. Stesso pattern di auto_eval.py."""
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    print(f"    [WARN] JSON parse fallito. Preview: {raw[:150]!r}")
    return fallback


def normalize_slug(name: str) -> str:
    return name.lower().replace(" ", "-").replace("_", "-")


def find_best_layer(concept_slug: str, model_slug: str, parent: Optional[str] = None) -> Optional[int]:
    """
    Trova il layer migliore per un (sub)concetto dato.
    Cerca in: vector_library/.../sub/{concept_slug}/{model_slug}/ se parent è specificato,
    altrimenti in vector_library/{category}/{concept_slug}/{model_slug}/.
    Prima prova eval.json (SNR), poi summary.json (boot_min).
    """
    # Cerca il path nella vector library
    if parent:
        # Sub-concetto: path = vector_library/*/{parent}/sub/{slug}/{model_slug}/
        candidates = list(VECTOR_LIB_ROOT.glob(f"*/{parent}/sub/{concept_slug}/{model_slug}"))
    else:
        candidates = list(VECTOR_LIB_ROOT.glob(f"*/{concept_slug}/{model_slug}"))

    if not candidates:
        return None
    vec_dir = candidates[0]

    # 1. Prova eval.json (held-out SNR — più affidabile)
    eval_path = vec_dir / "eval.json"
    if eval_path.exists():
        with open(eval_path) as f:
            data = json.load(f)
        best = data.get("best_layer")
        if best is not None:
            return int(best)

    # 2. Fallback: summary.json (boot_min)
    summary_path = vec_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            data = json.load(f)
        results = data.get("results", {})
        if results:
            best = max(results,
                       key=lambda l: results[l]["convergence_pca"]["bootstrap_cos_min"])
            return int(best)

    # 3. Ultimo fallback: primo .npy disponibile
    npy_files = [f for f in vec_dir.glob("layer_*.npy") if "_pca" not in f.name]
    if npy_files:
        layers = [int(re.search(r'layer_(\d+)', f.name).group(1)) for f in npy_files]
        return max(layers)

    return None


def load_vector_path(concept_slug: str, model_slug: str,
                     layer: int, parent: Optional[str] = None) -> Optional[Path]:
    """Ritorna il path al .npy per un dato layer."""
    if parent:
        candidates = list(VECTOR_LIB_ROOT.glob(f"*/{parent}/sub/{concept_slug}/{model_slug}"))
    else:
        candidates = list(VECTOR_LIB_ROOT.glob(f"*/{concept_slug}/{model_slug}"))
    if not candidates:
        return None
    return candidates[0] / f"layer_{layer}.npy"


# ── Steering Client ─────────────────────────────────────────────────────────────

class SteeringClient:
    """Client HTTP per steering_server.py (porta 8010, MI50). Identico a auto_eval.py."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def get_active_model(self) -> str:
        r = requests.get(f"{self.base_url}/api/models", timeout=10)
        r.raise_for_status()
        return r.json().get("active", "")

    def load_sub_vector(self, npy_path: Path, concept_alias: str) -> bool:
        """
        Carica un sub-vettore custom nel server via /api/load_vector.
        Il server deve supportare questo endpoint (da aggiungere in steering_server.py).
        """
        r = requests.post(f"{self.base_url}/api/load_vector", json={
            "path": str(npy_path),
            "alias": concept_alias,
        }, timeout=10)
        return r.ok

    def generate_with_vector(
        self,
        prompt: str,
        concept: str,
        layer: int,
        gain: int,
        alpha: float,
        max_new_tokens: int = DEFAULT_MAX_TOKENS_GEN,
        npy_path: Optional[Path] = None,
    ) -> str:
        """Genera testo con iniezione del vettore concept al layer dato.
        Se npy_path è fornito, il vettore viene caricato direttamente dal file
        (bypassa il catalog lookup — necessario per sub-concept vectors)."""
        payload = {
            "prompt": prompt,
            "concept": concept,
            "alpha": alpha,
            "gain": gain,
            "vector_layer": layer,
            "inject_layer": layer,
            "max_new_tokens": max_new_tokens,
            "mode": "inject",
            "multi": False,
        }
        if npy_path is not None:
            payload["vector_path"] = str(npy_path)
        r = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=120)
        r.raise_for_status()
        return r.json()["text"]

    def generate_baseline(
        self,
        prompt: str,
        max_new_tokens: int = DEFAULT_MAX_TOKENS_GEN,
    ) -> str:
        """Genera testo senza iniezione (baseline)."""
        payload = {
            "prompt": prompt,
            "mode": "baseline",
            "max_new_tokens": max_new_tokens,
        }
        r = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=120)
        r.raise_for_status()
        return r.json()["text"]


# ── M40 Client ─────────────────────────────────────────────────────────────────

class M40Client:
    """Client per M40 llama-server (porta 11435, API OpenAI-compatible). Stateless."""

    SYSTEM_PROMPT = """Sei un analista di rappresentazioni semantiche nei transformer.
Il tuo compito è valutare se due testi prodotti da iniezione di vettori concept diversi
sono fenomenologicamente distinguibili — ovvero se catturano aspetti sensoriali distinti.
Rispondi SEMPRE e SOLO con JSON valido. Nessun testo fuori dal JSON."""

    def __init__(self, base_url: str = DEFAULT_M40_URL):
        self.base_url = base_url.rstrip("/")

    def _call(self, user_msg: str, max_tokens: int = DEFAULT_MAX_TOKENS_EVAL) -> str:
        r = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": "gemma3-4b",
                "messages": [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                "max_tokens": max_tokens,
                "temperature": DEFAULT_TEMPERATURE,
                "stream": False,
            },
            timeout=180,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def judge_pair(
        self,
        concept_a: str, text_a: str, desc_a: str,
        concept_b: str, text_b: str, desc_b: str,
        prompt: str,
    ) -> dict:
        """
        M40 giudica se due output sono fenomenologicamente distinti.
        Ritorna dict con distinction_score, is_distinct, dominant_a, dominant_b, feedback.
        """
        msg = (
            f"PROMPT NEUTRO USATO:\n{prompt}\n\n"
            f"TESTO A (steered con '{concept_a}' — {desc_a}):\n{text_a}\n\n"
            f"TESTO B (steered con '{concept_b}' — {desc_b}):\n{text_b}\n\n"
            f"DOMANDE:\n"
            f"1. I due testi descrivono aspetti sensoriali DISTINTI? (es. A=febbre, B=mani fredde)\n"
            f"2. O invece si sovrappongono sulla stessa dimensione?\n"
            f"3. Qual è il tema fenomenologico dominante in A? E in B?\n"
            f"4. Se i vettori NON sono distinti, cosa bisogna cambiare nella definizione dei sub-concetti?\n\n"
            f"Rispondi SOLO con JSON:\n"
            f'{{"distinction_score": <1-5>, '
            f'"is_distinct": <true|false>, '
            f'"dominant_theme_a": "...", '
            f'"dominant_theme_b": "...", '
            f'"overlap_description": "...", '
            f'"feedback_for_refinement": "..."}}'
        )
        raw = self._call(msg)
        return _parse_json(raw, {
            "distinction_score": 1,
            "is_distinct": False,
            "dominant_theme_a": "unknown",
            "dominant_theme_b": "unknown",
            "overlap_description": raw[:200],
            "feedback_for_refinement": "JSON parsing fallito",
        })

    def generate_session_verdict(
        self,
        parent_concept: str,
        sub_concepts: list[dict],
        pair_results: list[dict],
        version: int,
    ) -> dict:
        """
        Verdetto finale della sessione di valutazione.
        Decide se i sub-vettori sono validati o se serve raffinamento (con suggerimenti).
        """
        # Riassunto compatto per non saturare il contesto M40
        summary = {
            "parent": parent_concept,
            "version": version,
            "sub_concepts": [
                {"slug": s["slug"], "description": s["description"]}
                for s in sub_concepts
            ],
            "pairs_evaluated": len(pair_results),
            "pairs_distinct": sum(1 for p in pair_results if p["judgment"].get("is_distinct")),
            "avg_distinction_score": round(
                sum(p["judgment"].get("distinction_score", 1) for p in pair_results)
                / max(len(pair_results), 1), 2
            ),
            "failed_pairs": [
                {
                    "a": p["concept_a"], "b": p["concept_b"],
                    "score": p["judgment"].get("distinction_score"),
                    "feedback": p["judgment"].get("feedback_for_refinement", "")[:100],
                }
                for p in pair_results if not p["judgment"].get("is_distinct")
            ],
        }

        msg = (
            f"FINE SESSIONE — Sub-concept eval per '{parent_concept}' (v{version})\n\n"
            f"RISULTATI:\n{json.dumps(summary, indent=2, ensure_ascii=False)}\n\n"
            f"Sulla base di questi risultati:\n"
            f"1. I sub-vettori sono VALIDATI (tutte le coppie distinte)?\n"
            f"2. Quali sub-concetti NON sono sufficientemente distinti?\n"
            f"3. Come andrebbero ridefiniti per il prossimo ciclo del loop?\n\n"
            f"Rispondi SOLO con JSON:\n"
            f'{{"all_validated": <true|false>, '
            f'"validated_concepts": ["..."], '
            f'"needs_refinement": ["..."], '
            f'"refinement_suggestions": {{"slug": "suggerimento", ...}}, '
            f'"overall_assessment": "..."}}'
        )
        raw = self._call(msg, max_tokens=800)
        return _parse_json(raw, {
            "all_validated": False,
            "validated_concepts": [],
            "needs_refinement": [s["slug"] for s in sub_concepts],
            "refinement_suggestions": {},
            "overall_assessment": raw[:300],
        })


# ── Session Runner ──────────────────────────────────────────────────────────────

def run_eval(
    parent_concept: str,
    version: int,
    model_name: str,
    steering_url: str,
    m40_url: str,
    gain: int,
    alpha: float,
    max_tokens: int,
    n_prompts: int,
) -> dict:

    model_slug = normalize_slug(model_name)
    meta_path  = SUB_CONCEPTS_DIR / parent_concept / f"_meta_v{version}.json"

    if not meta_path.exists():
        print(f"ERRORE: {meta_path} non trovato. Esegui prima concept_expander.py --step 1")
        sys.exit(1)

    with open(meta_path) as f:
        meta = json.load(f)

    sub_concepts = meta.get("sub_concepts", [])
    if not sub_concepts:
        print("ERRORE: _meta.json non contiene sub_concepts")
        sys.exit(1)

    steering = SteeringClient(steering_url)
    m40      = M40Client(m40_url)

    # Verifica server
    print("Connessione steering server...", end=" ", flush=True)
    active = steering.get_active_model()
    if not active:
        print("ERRORE: nessun modello caricato")
        sys.exit(1)
    print(f"OK — {active}")

    print("Connessione M40...", end=" ", flush=True)
    try:
        r = requests.get(f"{m40_url}/health", timeout=10)
        r.raise_for_status()
        print("OK")
    except requests.RequestException as e:
        print(f"ERRORE: {e}")
        sys.exit(1)

    # Carica layer migliore per ogni sub-concetto
    print(f"\nSub-concetti (v{version}):")
    sub_info = []
    for sub in sub_concepts:
        slug  = sub["slug"]
        layer = find_best_layer(slug, model_slug, parent=parent_concept)
        if layer is None:
            print(f"  ⚠ {slug} — nessun vettore trovato. Esegui probe prima. SKIP.")
            continue
        npy_path = load_vector_path(slug, model_slug, layer, parent=parent_concept)
        sub_info.append({**sub, "layer": layer, "npy_path": str(npy_path) if npy_path else None})
        print(f"  ✓ {slug} → L{layer} ({npy_path})")

    if len(sub_info) < 2:
        print("ERRORE: servono almeno 2 sub-concetti con vettori estratti")
        sys.exit(1)

    # Trova layer del broad concept (per confronto)
    broad_layer = find_best_layer(parent_concept, model_slug)
    print(f"  ✓ {parent_concept} (broad) → L{broad_layer}")

    prompts = NEUTRAL_PROMPTS[:n_prompts]
    pairs   = list(combinations(range(len(sub_info)), 2))

    print(f"\nCoppie da valutare: {len(pairs)}")
    print(f"Prompt neutri: {n_prompts}")
    print(f"Gain: {gain}, Alpha: {alpha:+.1f}")
    print(f"{'='*62}\n")

    pair_results = []

    for i, (idx_a, idx_b) in enumerate(pairs):
        sub_a = sub_info[idx_a]
        sub_b = sub_info[idx_b]
        print(f"[{i+1}/{len(pairs)}] {sub_a['slug']} vs {sub_b['slug']}")

        prompt_judgments = []
        for p_idx, prompt in enumerate(prompts):
            print(f"  Prompt {p_idx+1}/{n_prompts}...", end=" ", flush=True)

            # Genera con sub_A (usa il vettore del sub-concept, non il broad)
            try:
                text_a = steering.generate_with_vector(
                    prompt, sub_a["slug"], sub_a["layer"], gain, alpha, max_tokens,
                    npy_path=sub_a.get("npy_path"),
                )
            except requests.RequestException as e:
                print(f"[ERR A: {e}]")
                text_a = "[errore]"

            # Genera con sub_B
            try:
                text_b = steering.generate_with_vector(
                    prompt, sub_b["slug"], sub_b["layer"], gain, alpha, max_tokens,
                    npy_path=sub_b.get("npy_path"),
                )
            except requests.RequestException as e:
                print(f"[ERR B: {e}]")
                text_b = "[errore]"

            # M40 giudica
            judgment = m40.judge_pair(
                sub_a["slug"], text_a, sub_a.get("description", ""),
                sub_b["slug"], text_b, sub_b.get("description", ""),
                prompt,
            )
            score = judgment.get("distinction_score", "?")
            distinct = "✓" if judgment.get("is_distinct") else "✗"
            print(f"score={score} {distinct}")

            prompt_judgments.append({
                "prompt": prompt,
                "text_a": text_a,
                "text_b": text_b,
                "judgment": judgment,
            })
            time.sleep(0.3)

        # Aggrega judgment per questa coppia
        scores = [j["judgment"].get("distinction_score", 1) for j in prompt_judgments]
        avg_score = sum(scores) / len(scores) if scores else 0
        is_distinct = avg_score >= 3.0

        pair_result = {
            "concept_a": sub_a["slug"],
            "concept_b": sub_b["slug"],
            "layer_a": sub_a["layer"],
            "layer_b": sub_b["layer"],
            "avg_distinction_score": round(avg_score, 2),
            "is_distinct": is_distinct,
            "judgment": {
                "distinction_score": round(avg_score, 2),
                "is_distinct": is_distinct,
                "dominant_theme_a": prompt_judgments[-1]["judgment"].get("dominant_theme_a", ""),
                "dominant_theme_b": prompt_judgments[-1]["judgment"].get("dominant_theme_b", ""),
                "feedback_for_refinement": prompt_judgments[-1]["judgment"].get("feedback_for_refinement", ""),
            },
            "prompt_details": prompt_judgments,
        }
        pair_results.append(pair_result)
        status = "DISTINTI ✓" if is_distinct else "SOVRAPPOSTI ✗"
        print(f"  → Avg score: {avg_score:.1f} — {status}\n")

    # Verdetto finale
    print("Verdetto finale M40...", end=" ", flush=True)
    verdict = m40.generate_session_verdict(
        parent_concept, sub_info, pair_results, version
    )
    validated   = verdict.get("all_validated", False)
    needs_ref   = verdict.get("needs_refinement", [])
    print("OK")
    print(f"\n{'='*62}")
    print(f"  VERDETTO: {'VALIDATI ✓' if validated else 'RICHIEDE RAFFINAMENTO ✗'}")
    if needs_ref:
        print(f"  Da raffinare: {', '.join(needs_ref)}")
    print(f"  {verdict.get('overall_assessment', '')[:120]}")
    print(f"{'='*62}\n")

    # Salva risultato
    result = {
        "type": "sub_concept_eval",
        "parent_concept": parent_concept,
        "model": model_name,
        "version": version,
        "timestamp": datetime.now().isoformat(),
        "sub_concepts": sub_info,
        "pair_results": pair_results,
        "verdict": verdict,
        "config": {"gain": gain, "alpha": alpha, "n_prompts": n_prompts},
    }

    out_dir = EVAL_OUTPUT_DIR / parent_concept / model_slug
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"eval_v{version}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Risultato salvato: {out_path}")

    return result


# ── Entry point ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Valutazione comparativa sub-vettori concept (Fase 2)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--concept",      required=True,
                        help="Concept parent (es. hot_vs_cold)")
    parser.add_argument("--version",      type=int, default=1,
                        help="Versione _meta da usare (iterazione del loop)")
    parser.add_argument("--model",        default="Gemma2-Uncensored",
                        help="Nome modello (per trovare i vettori estratti)")
    parser.add_argument("--steering-url", default=DEFAULT_STEERING_URL)
    parser.add_argument("--m40-url",      default=DEFAULT_M40_URL)
    parser.add_argument("--gain",         type=int, default=200,
                        help="Gain per lo steering")
    parser.add_argument("--alpha",        type=float, default=1.0,
                        help="Alpha per lo steering")
    parser.add_argument("--max-tokens",   type=int, default=DEFAULT_MAX_TOKENS_GEN)
    parser.add_argument("--n-prompts",    type=int, default=3,
                        help="Numero di prompt neutri per coppia")

    args = parser.parse_args()

    run_eval(
        parent_concept = args.concept,
        version        = args.version,
        model_name     = args.model,
        steering_url   = args.steering_url,
        m40_url        = args.m40_url,
        gain           = args.gain,
        alpha          = args.alpha,
        max_tokens     = args.max_tokens,
        n_prompts      = args.n_prompts,
    )


if __name__ == "__main__":
    main()
