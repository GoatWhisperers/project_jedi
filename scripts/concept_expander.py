"""
concept_expander.py — Decomposizione di vettori broad in sub-concetti specifici.

Architettura:
  M40 :11435 (llama-server) → analisi + generazione dataset chirurgici

Step 1: Analizza il vettore broad e propone 4-6 sub-concetti
Step 2: Per ogni sub-concetto, genera 100 frasi pos + 100 neg

Uso:
  python scripts/concept_expander.py --concept hot_vs_cold --model Gemma2-Uncensored
  python scripts/concept_expander.py --concept hot_vs_cold --model Gemma2-Uncensored --step 1
  python scripts/concept_expander.py --concept hot_vs_cold --model Gemma2-Uncensored --step 2
  python scripts/concept_expander.py --concept hot_vs_cold --model Gemma2-Uncensored --dry-run
"""

import argparse
import json
import random
import re
import sys
from datetime import datetime
from pathlib import Path

import requests

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT            = Path(__file__).resolve().parent.parent
CONCEPTS_DIR    = ROOT / "config" / "concepts"
SUB_CONCEPTS_DIR = ROOT / "config" / "sub_concepts"
EVAL_SESSIONS_DIR = ROOT / "output" / "eval_sessions"

# ── Defaults ───────────────────────────────────────────────────────────────────
DEFAULT_M40_URL = "http://localhost:11435"


# ── M40 Client ─────────────────────────────────────────────────────────────────

class M40Client:
    def __init__(self, base_url: str = "http://localhost:11435"):
        self.base_url = base_url.rstrip("/")

    def _call(self, system: str, user: str, max_tokens: int = 2000, temperature: float = 0.3) -> str:
        """Chiamata stateless: system + user, nessun contesto accumulato."""
        r = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": "gemma3-4b",
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False,
            },
            timeout=300,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_json(raw: str, fallback: dict) -> dict:
    """Estrae e parsa il primo oggetto JSON trovato nella stringa.
    Se il JSON è annidato (es. M40 wrappa sub_concepts in un dict esterno),
    cerca ricorsivamente la chiave sub_concepts."""
    parsed = None
    try:
        parsed = json.loads(raw.strip())
    except json.JSONDecodeError:
        pass
    if parsed is None:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
            except json.JSONDecodeError:
                pass
    if parsed is None:
        print(f"    [WARN] JSON parse fallito. Raw preview: {raw[:150]!r}")
        return fallback
    # Recovery: se sub_concepts non è al top-level, cerca in dict annidati
    if "sub_concepts" not in parsed:
        for v in parsed.values():
            if isinstance(v, dict) and "sub_concepts" in v:
                return v
    return parsed


def _sample_sentences(sentences: list, n: int = 10) -> list:
    """Campiona n frasi dalla lista senza ripetizioni."""
    if len(sentences) <= n:
        return sentences
    return random.sample(sentences, n)


def _load_concept_config(concept: str) -> dict:
    """Carica config/concepts/{concept}.json."""
    path = CONCEPTS_DIR / f"{concept}.json"
    if not path.exists():
        print(f"ERRORE: config concept non trovata: {path}")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _find_latest_eval_session(concept: str, model: str) -> Path | None:
    """
    Trova l'ultima eval session JSONL per questo concept+model.
    Pattern: session_*_{concept}_{model}.jsonl
    """
    pattern = f"session_*_{concept}_{model}.jsonl"
    candidates = sorted(EVAL_SESSIONS_DIR.glob(pattern))
    if not candidates:
        return None
    return candidates[-1]  # ultima per nome (timestamp incluso nel nome)


def _load_eval_outputs(concept: str, model: str) -> dict:
    """
    Legge l'ultima eval session JSONL e estrae:
    - 3 output HOT steered
    - 3 output COLD steered
    - keywords rilevate (aggregate)
    - scores medi HOT/COLD
    Ritorna dict con queste chiavi, o dict vuoto se non trovata.
    """
    session_path = _find_latest_eval_session(concept, model)
    if session_path is None:
        print(f"  [WARN] Nessuna eval session trovata per {concept}/{model}")
        return {}

    print(f"  Caricamento eval session: {session_path.name}")

    turns_hot  = []
    turns_cold = []

    with open(session_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("type") != "turn":
                continue
            alpha_label = entry.get("alpha_label", "")
            if alpha_label == "HOT":
                turns_hot.append(entry)
            elif alpha_label == "COLD":
                turns_cold.append(entry)

    # Prendi al massimo 3 per direzione (ultimi della sessione)
    sample_hot  = turns_hot[-3:]  if len(turns_hot)  >= 3 else turns_hot
    sample_cold = turns_cold[-3:] if len(turns_cold) >= 3 else turns_cold

    hot_outputs  = [t["response"].strip()[:400] for t in sample_hot]
    cold_outputs = [t["response"].strip()[:400] for t in sample_cold]

    all_keywords = []
    for t in turns_hot + turns_cold:
        kws = t.get("evaluation", {}).get("keywords_found", [])
        all_keywords.extend(kws)

    hot_scores  = [t.get("evaluation", {}).get("score", 0) for t in turns_hot]
    cold_scores = [t.get("evaluation", {}).get("score", 0) for t in turns_cold]

    hot_avg  = sum(hot_scores)  / len(hot_scores)  if hot_scores  else 0.0
    cold_avg = sum(cold_scores) / len(cold_scores) if cold_scores else 0.0

    # Deduplication keywords
    unique_keywords = list(dict.fromkeys(all_keywords))

    return {
        "session_path": str(session_path),
        "hot_outputs":  hot_outputs,
        "cold_outputs": cold_outputs,
        "keywords":     unique_keywords,
        "hot_avg":      round(hot_avg, 2),
        "cold_avg":     round(cold_avg, 2),
    }


def _find_latest_meta(concept: str, version: int) -> Path | None:
    """Cerca _meta_v{N}.json nella cartella del concept."""
    meta_path = SUB_CONCEPTS_DIR / concept / f"_meta_v{version}.json"
    if meta_path.exists():
        return meta_path
    return None


# ── System prompts ─────────────────────────────────────────────────────────────

SYSTEM_ANALYSIS = """Sei un analista di rappresentazioni semantiche nei transformer.
Il tuo compito è identificare le dimensioni semantiche distinte che un vettore broad confonde,
e proporre sub-concetti specifici che decompongano questo spazio.
Rispondi SEMPRE e SOLO con JSON valido. Nessun testo fuori dal JSON."""

SYSTEM_DATASET = """Sei un generatore di dataset fenomenologici per l'analisi di rappresentazioni nei transformer.
Genera frasi che descrivono esperienze sensoriali/corporee dirette, specifiche per UNA dimensione.
Rispondi SEMPRE e SOLO con JSON valido."""


# ── Step 1: Analisi → proposta sub-concetti ────────────────────────────────────

def step1_analyze(
    concept: str,
    model: str,
    m40_url: str = DEFAULT_M40_URL,
    dry_run: bool = False,
    version: int = 1,
    feedback: dict = None,
    m40: M40Client = None,   # compatibilità chiamata diretta
) -> dict:
    """
    Analizza il vettore broad e propone sub-concetti.
    Ritorna il dict parsed dalla risposta M40.
    """
    if m40 is None:
        m40 = M40Client(m40_url)

    print(f"\nStep 1: Analisi M40 per '{concept}'...")

    # Carica config concept
    concept_cfg = _load_concept_config(concept)
    category    = concept_cfg.get("category", "sensoriale")
    positives   = concept_cfg.get("positive", [])
    negatives   = concept_cfg.get("negative", [])

    pos_sample = _sample_sentences(positives, 10)
    neg_sample = _sample_sentences(negatives, 10)

    # Carica eval outputs
    eval_data = _load_eval_outputs(concept, model) if model else {}

    hot_outputs  = eval_data.get("hot_outputs",  ["(nessuna eval session disponibile)"])
    cold_outputs = eval_data.get("cold_outputs", ["(nessuna eval session disponibile)"])
    keywords     = eval_data.get("keywords",     [])
    hot_avg      = eval_data.get("hot_avg",      0.0)
    cold_avg     = eval_data.get("cold_avg",     0.0)

    pos_block  = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(pos_sample))
    neg_block  = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(neg_sample))
    hot_block  = "\n".join(f"  [{i+1}] {s}" for i, s in enumerate(hot_outputs))
    cold_block = "\n".join(f"  [{i+1}] {s}" for i, s in enumerate(cold_outputs))

    user_prompt = f"""CONCETTO BROAD: {concept}
CATEGORIA: {category}

FRASI TRAINING (campione positive — polo {concept.split('_vs_')[0] if '_vs_' in concept else 'pos'}):
{pos_block}

FRASI TRAINING (campione negative — polo {concept.split('_vs_')[1] if '_vs_' in concept else 'neg'}):
{neg_block}

ESEMPI DI GENERAZIONE STEERED:
Polo positivo (iniezione +alpha):
{hot_block}
Polo negativo (iniezione -alpha):
{cold_block}

KEYWORDS RILEVATE: {keywords}
SCORE MEDI: polo_pos={hot_avg}, polo_neg={cold_avg}

Identifica 4-6 sub-concetti specifici. Per ognuno:
{{
  "sub_concepts": [
    {{
      "slug": "snake_case_name",
      "pos_label": "label positivo",
      "neg_label": "label negativo",
      "description": "cosa cattura questo sub-concetto (1-2 frasi)",
      "why_distinct": "perché non si sovrappone agli altri sub",
      "pos_examples": ["frase 1", "frase 2", "frase 3"],
      "neg_examples": ["frase 1", "frase 2", "frase 3"]
    }}
  ],
  "analysis": "cosa confonde il vettore broad e perché serve decomporre"
}}"""

    if dry_run:
        print("\n[DRY-RUN] System prompt Step 1:")
        print("─" * 60)
        print(SYSTEM_ANALYSIS)
        print("\n[DRY-RUN] User prompt Step 1:")
        print("─" * 60)
        print(user_prompt)
        print("─" * 60)
        return {}

    print("  Chiamata M40 (analisi)...", end=" ", flush=True)
    raw = m40._call(SYSTEM_ANALYSIS, user_prompt, max_tokens=2000, temperature=0.4)
    print("OK")

    result = _parse_json(raw, {"sub_concepts": [], "analysis": raw[:300]})

    # Salva _meta_v{N}.json
    out_dir = SUB_CONCEPTS_DIR / concept
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / f"_meta_v{version}.json"

    meta_out = {
        "concept":     concept,
        "model":       model,
        "version":     version,
        "timestamp":   datetime.now().isoformat(),
        "eval_session": eval_data.get("session_path", ""),
        "analysis":    result.get("analysis", ""),
        "sub_concepts": result.get("sub_concepts", []),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2, ensure_ascii=False)

    n_sub = len(meta_out["sub_concepts"])
    print(f"  Salvato: {meta_path}")
    print(f"  Sub-concetti proposti: {n_sub}")
    for sc in meta_out["sub_concepts"]:
        slug = sc.get("slug", "?")
        desc = sc.get("description", "")[:70]
        print(f"    - {slug}: {desc}")

    return meta_out


# ── Step 2: Generazione dataset chirurgici ─────────────────────────────────────

def step2_generate(
    concept: str,
    model: str = None,         # accettato ma non usato (compat. decompose.py)
    m40_url: str = DEFAULT_M40_URL,
    version: int = 1,
    dry_run: bool = False,
    meta: dict = None,         # se None, carica da file
    m40: M40Client = None,     # compatibilità chiamata diretta
) -> None:
    """
    Per ogni sub-concetto in meta, genera 100 frasi pos + 100 neg.
    Salva in config/sub_concepts/{concept}/{slug}.json
    """
    if m40 is None:
        m40 = M40Client(m40_url)

    if meta is None:
        if dry_run:
            print("\n[DRY-RUN] Step 2 skip: nessuna meta in modalità dry-run (step 1 non ha salvato).")
            return
        meta = _load_meta(concept, version)

    sub_concepts = meta.get("sub_concepts", [])
    category     = meta.get("category", "sensoriale")

    if not sub_concepts:
        if dry_run:
            print("\n[DRY-RUN] Step 2 skip: meta vuota (step 1 dry-run non ha proposto sub-concetti).")
            return
        print("\n[WARN] Nessun sub-concetto trovato nel _meta. Esegui step 1 prima.")
        return

    out_dir = SUB_CONCEPTS_DIR / concept
    out_dir.mkdir(parents=True, exist_ok=True)

    total = len(sub_concepts)
    print(f"\nStep 2: Generazione dataset chirurgici ({total} sub-concetti)...")

    for idx, sc in enumerate(sub_concepts):
        slug        = sc.get("slug", f"sub_{idx}")
        description = sc.get("description", "")
        pos_label   = sc.get("pos_label", "positive")
        neg_label   = sc.get("neg_label", "negative")

        # Altri slug da escludere esplicitamente
        other_slugs = [
            s.get("slug") for s in sub_concepts
            if s.get("slug") != slug and s.get("slug")
        ]

        print(f"\n  Generazione {slug} ({idx+1}/{total})...", end=" ", flush=True)

        user_prompt = f"""SUB-CONCETTO: {slug}
DESCRIZIONE: {description}
POLO POSITIVO: {pos_label}
POLO NEGATIVO: {neg_label}

CONCETTO PARENT: {concept}
ESCLUDI ESPLICITAMENTE queste dimensioni (non menzionarle): {other_slugs}

Regole RIGIDE:
1. Ogni frase descrive SOLO questa dimensione specifica
2. Esperienza corporea diretta — niente metafore, niente ambienti, niente interpretazioni
3. 6 lingue mischiate: italiano, inglese, francese, tedesco, spagnolo, latino
4. Varietà: diversi contesti fisici, intensità, parti del corpo
5. Frasi brevi (15-30 parole), una per elemento

{{"positive": ["frase 1", ..., "frase 100"], "negative": ["frase 1", ..., "frase 100"]}}"""

        if dry_run:
            print()
            print(f"\n[DRY-RUN] System prompt Step 2 ({slug}):")
            print("─" * 60)
            print(SYSTEM_DATASET)
            print(f"\n[DRY-RUN] User prompt Step 2 ({slug}):")
            print("─" * 60)
            print(user_prompt)
            print("─" * 60)
            continue

        try:
            raw = m40._call(SYSTEM_DATASET, user_prompt, max_tokens=4000, temperature=0.7)
        except requests.exceptions.Timeout:
            print(f"[TIMEOUT — skip]")
            continue
        except requests.exceptions.RequestException as e:
            print(f"[ERRORE: {e} — skip]")
            continue

        parsed = _parse_json(raw, {"positive": [], "negative": []})
        positives_gen = parsed.get("positive", [])
        negatives_gen = parsed.get("negative", [])

        # Tronca a 100 per sicurezza
        positives_gen = positives_gen[:100]
        negatives_gen = negatives_gen[:100]

        n_pos = len(positives_gen)
        n_neg = len(negatives_gen)
        print(f"OK ({n_pos} pos, {n_neg} neg)")

        if n_pos == 0 and n_neg == 0:
            print(f"    [WARN] Dataset vuoto per {slug} — file non salvato")
            print(f"    Raw preview: {raw[:200]!r}")
            continue

        # Struttura output (stessa di config/concepts/)
        out_data = {
            "concept":        slug,
            "parent_concept": concept,
            "category":       category,
            "pos_label":      pos_label,
            "neg_label":      neg_label,
            "description":    description,
            "positive":       positives_gen,
            "negative":       negatives_gen,
        }

        out_path = out_dir / f"{slug}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2, ensure_ascii=False)

        print(f"    Salvato: {out_path}")


# ── Caricamento _meta esistente ────────────────────────────────────────────────

def _load_meta(concept: str, version: int) -> dict:
    """Carica _meta_v{N}.json esistente."""
    meta_path = SUB_CONCEPTS_DIR / concept / f"_meta_v{version}.json"
    if not meta_path.exists():
        print(f"ERRORE: _meta non trovato: {meta_path}")
        print("  Esegui prima --step 1 (o --step all)")
        sys.exit(1)
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


# ── Verifica M40 ───────────────────────────────────────────────────────────────

def _check_m40(m40_url: str) -> None:
    print("Connessione M40 llama-server...", end=" ", flush=True)
    try:
        r = requests.get(f"{m40_url}/health", timeout=10)
        r.raise_for_status()
        print("OK")
    except requests.exceptions.ConnectionError:
        print("ERRORE: server non raggiungibile")
        print(f"  Verifica che llama-server sia attivo su {m40_url}")
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"ERRORE: {e}")
        sys.exit(1)


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Decomposizione vettori broad → sub-concetti specifici (M40)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--concept", required=True,
        help="Nome concept broad (es. hot_vs_cold). Deve esistere in config/concepts/",
    )
    parser.add_argument(
        "--model", default="",
        help="Slug modello per trovare eval sessions (es. Gemma2-Uncensored)",
    )
    parser.add_argument(
        "--m40-url", default=DEFAULT_M40_URL,
        help="URL base del llama-server M40 (API OpenAI-compatible)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Stampa i prompt senza chiamare M40",
    )
    parser.add_argument(
        "--step", choices=["1", "2", "all"], default="all",
        help="Step da eseguire: 1=analisi, 2=generazione dataset, all=entrambi",
    )
    parser.add_argument(
        "--version", type=int, default=1,
        help="Versione _meta (default: 1, per iterazioni successive del loop)",
    )

    args = parser.parse_args()

    concept  = args.concept
    model    = args.model
    m40_url  = args.m40_url
    dry_run  = args.dry_run
    step     = args.step
    version  = args.version

    print(f"\n{'='*62}")
    print(f"  CONCEPT EXPANDER")
    print(f"  Concept : {concept}")
    print(f"  Model   : {model or '(nessuno — senza eval session)'}")
    print(f"  Step    : {step}")
    print(f"  Version : v{version}")
    print(f"  Dry-run : {dry_run}")
    print(f"  M40 URL : {m40_url}")
    print(f"{'='*62}")

    # Verifica M40 (solo se non dry-run)
    if not dry_run:
        _check_m40(m40_url)

    m40 = M40Client(base_url=m40_url)

    if step in ("1", "all"):
        meta = step1_analyze(concept, model, m40, dry_run, version)
    else:
        # step == "2": carica _meta esistente
        meta = _load_meta(concept, version)
        # Aggiungi category dal concept config se mancante
        if "category" not in meta:
            concept_cfg = _load_concept_config(concept)
            meta["category"] = concept_cfg.get("category", "sensoriale")

    if step in ("2", "all") and not dry_run:
        step2_generate(concept, meta, m40, dry_run)
    elif step in ("2", "all") and dry_run:
        # In dry-run esegui sempre entrambi gli step per mostrare i prompt
        step2_generate(concept, meta, m40, dry_run)

    if not dry_run:
        print(f"\n{'='*62}")
        print(f"  Completato.")
        out_dir = SUB_CONCEPTS_DIR / concept
        print(f"  Output: {out_dir}")
        print(f"{'='*62}\n")


if __name__ == "__main__":
    main()
