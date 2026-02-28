"""
auto_eval.py — Orchestrator per valutazione automatica dello steering vettoriale.

Architettura:
  MI50 :8010  (steering_server.py)  →  genera testo con iniezione vettoriale
  M40  :11435 (llama-server)        →  valuta le risposte e decide le configurazioni

Nessuna modifica a steering_server.py — lo usa solo come client HTTP.

Uso:
  python scripts/auto_eval.py --concept hot_vs_cold
  python scripts/auto_eval.py --concept hot_vs_cold --max-probes 3 --turns-per-block 3
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parent.parent
CONCEPTS_DIR = ROOT / "config" / "eval_concepts"
OUTPUT_DIR   = ROOT / "output" / "eval_sessions"

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_STEERING_URL      = "http://localhost:8010"
DEFAULT_M40_URL           = "http://localhost:11435"
DEFAULT_MAX_PROBES        = 2
DEFAULT_TURNS_PER_BLOCK   = 3   # per alpha direction per probe
DEFAULT_MAX_TOKENS_STEERED = 150
DEFAULT_MAX_TOKENS_EVAL    = 450
DEFAULT_MAX_TOKENS_REPORT  = 2000
DEFAULT_TEMPERATURE_EVAL   = 0.2


# ── Steering Client ────────────────────────────────────────────────────────────

class SteeringClient:
    """Client HTTP per steering_server.py (porta 8010, MI50)."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def get_active_model(self) -> str:
        r = requests.get(f"{self.base_url}/api/models", timeout=10)
        r.raise_for_status()
        return r.json().get("active", "")

    def get_concept_layers(self, concept: str) -> tuple[list[int], Optional[int]]:
        r = requests.get(
            f"{self.base_url}/api/concept_layers",
            params={"concept": concept},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        return data.get("layers", []), data.get("best_layer")

    def generate(
        self,
        prompt: str,
        concept: str,
        alpha: float,
        layer_configs: list[dict],
        max_new_tokens: int = DEFAULT_MAX_TOKENS_STEERED,
    ) -> str:
        """
        Chiama /api/generate con iniezione vettoriale.
        layer_configs: [{layer: N, gain: G}, ...]
        """
        single = len(layer_configs) == 1
        payload: dict = {
            "prompt": prompt,
            "concept": concept,
            "alpha": alpha,
            "max_new_tokens": max_new_tokens,
            "mode": "inject",
        }
        if single:
            payload["vector_layer"] = layer_configs[0]["layer"]
            payload["inject_layer"] = layer_configs[0]["layer"]
            payload["gain"]         = layer_configs[0]["gain"]
            payload["multi"]        = False
        else:
            payload["multi"]         = True
            payload["layer_configs"] = layer_configs
            # gain e vector_layer ignorati lato server quando layer_configs è presente
            payload["gain"]         = 1.0
            payload["vector_layer"] = 0
            payload["inject_layer"] = 0

        r = requests.post(
            f"{self.base_url}/api/generate", json=payload, timeout=120
        )
        r.raise_for_status()
        return r.json()["text"]


# ── Evaluator Client ───────────────────────────────────────────────────────────

class EvaluatorClient:
    """Client HTTP per llama-server su M40 (porta 11435, API OpenAI-compatible)."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self._system_prompt = ""

    def set_system_prompt(self, prompt: str):
        self._system_prompt = prompt

    def _call(self, user_msg: str, max_tokens: int, temperature: float = DEFAULT_TEMPERATURE_EVAL) -> str:
        # Chiamate STATELESS: llama-server ha ctx=4096 token e il system prompt
        # occupa già ~1500 token. Non accumuliamo storia — ogni chiamata è
        # [system_prompt] + [user_msg] e il contesto necessario è nel messaggio stesso.
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user",   "content": user_msg},
        ]
        r = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": "gemma3-4b",
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False,
            },
            timeout=180,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def evaluate_turn(
        self,
        turn_label: str,
        prompt: str,
        response: str,
        alpha: float,
        alpha_label: str,
        layer_configs: list[dict],
        concept_cfg: dict,
    ) -> dict:
        """Valutazione di un singolo turno. Ritorna dict con score, keywords, ecc."""
        cfg_desc = _config_desc(layer_configs)
        msg = (
            f"TURNO {turn_label} | Direzione: {alpha_label} (alpha={alpha:+.1f}) | Config: {cfg_desc}\n\n"
            f"PROMPT INVIATO AL MODELLO STEERED:\n{prompt}\n\n"
            f"RISPOSTA RICEVUTA:\n{response}\n\n"
            f"Valuta questa risposta secondo le regole del sistema.\n"
            f"Rispondi SOLO con un oggetto JSON valido, nessun testo aggiuntivo:\n"
            f'{{"score": <0-5>, "keywords_found": ["..."], '
            f'"assessment": "...", "semantic_or_lexical": "semantic|lexical|mixed|none"}}'
        )
        raw = self._call(msg, max_tokens=DEFAULT_MAX_TOKENS_EVAL)
        return _parse_json(raw, {
            "score": 0,
            "keywords_found": [],
            "assessment": raw[:200],
            "semantic_or_lexical": "none",
        })

    def analyze_probe(
        self,
        probe_idx: int,
        hot_turns: list[dict],
        cold_turns: list[dict],
        layer_configs: list[dict],
        available_layers: list[int],
        gain_range: list[int],
        concept_cfg: dict,
    ) -> dict:
        """
        Analisi comparativa HOT vs COLD dopo una proba completa.
        Ritorna dict con analisi + next_probe_config.
        """
        cfg_desc = _config_desc(layer_configs)
        hot_scores = [t["evaluation"]["score"] for t in hot_turns]
        cold_scores = [t["evaluation"]["score"] for t in cold_turns]
        hot_kw  = [kw for t in hot_turns  for kw in t["evaluation"].get("keywords_found", [])]
        cold_kw = [kw for t in cold_turns for kw in t["evaluation"].get("keywords_found", [])]
        hot_types  = [t["evaluation"].get("semantic_or_lexical", "none") for t in hot_turns]
        cold_types = [t["evaluation"].get("semantic_or_lexical", "none") for t in cold_turns]

        msg = (
            f"ANALISI PROBA {probe_idx + 1} | Config: [{cfg_desc}]\n\n"
            f"BLOCCO HOT (+1.0)\n"
            f"  Score: {hot_scores}  (media: {_avg(hot_scores):.1f})\n"
            f"  Keywords trovate: {hot_kw}\n"
            f"  Tipo effetto: {hot_types}\n\n"
            f"BLOCCO COLD (-1.0)\n"
            f"  Score: {cold_scores}  (media: {_avg(cold_scores):.1f})\n"
            f"  Keywords trovate: {cold_kw}\n"
            f"  Tipo effetto: {cold_types}\n\n"
            f"Layer disponibili: {available_layers}\n"
            f"Range gain consentito: {gain_range[0]} - {gain_range[1]}\n\n"
            f"DOMANDE CHIAVE:\n"
            f"1. HOT e COLD producono effetti OPPOSTI (simmetria) o sono entrambi assenti/uguali?\n"
            f"2. L'effetto è SEMANTICO (metafore, atmosfera) o solo LESSICALE (parole dirette)?\n"
            f"3. Quale configurazione esplorare nella prossima proba?\n"
            f"   Puoi cambiare gain, cambiare layer, usare più layer insieme.\n\n"
            f"Rispondi SOLO con un oggetto JSON valido, nessun testo aggiuntivo:\n"
            f'{{"hot_avg_score": <float>, "cold_avg_score": <float>, '
            f'"symmetry": "good|partial|none", '
            f'"contrast_type": "semantic|lexical|mixed|none", '
            f'"probe_summary": "...", '
            f'"next_probe_config": [{{"layer": <N>, "gain": <G>}}, ...], '
            f'"rationale": "..."}}'
        )
        raw = self._call(msg, max_tokens=DEFAULT_MAX_TOKENS_EVAL)
        fallback_gain = min(
            layer_configs[0]["gain"] + concept_cfg.get("gain_step_suggestion", 200),
            gain_range[1],
        )
        return _parse_json(raw, {
            "hot_avg_score": _avg(hot_scores),
            "cold_avg_score": _avg(cold_scores),
            "symmetry": "none",
            "contrast_type": "none",
            "probe_summary": raw[:300],
            "next_probe_config": [{"layer": layer_configs[0]["layer"], "gain": fallback_gain}],
            "rationale": "fallback — JSON parsing fallito",
        })

    def generate_report(
        self,
        concept: str,
        model: str,
        all_turns: list[dict],
        probe_analyses: list[dict],
    ) -> str:
        """Genera il report markdown finale della sessione."""
        # Costruiamo un sommario compatto per non saturare il contesto M40
        summary = {
            "concept": concept,
            "model": model,
            "total_turns": len(all_turns),
            "probes": [
                {
                    "probe": i + 1,
                    "config": a.get("next_probe_config", []),   # config USATA era quella precedente
                    "hot_avg": a.get("hot_avg_score"),
                    "cold_avg": a.get("cold_avg_score"),
                    "symmetry": a.get("symmetry"),
                    "contrast_type": a.get("contrast_type"),
                    "summary": a.get("probe_summary", ""),
                }
                for i, a in enumerate(probe_analyses)
            ],
        }

        msg = (
            f"FINE SESSIONE — Concept: {concept} | Modello: {model}\n\n"
            f"DATI SESSIONE:\n{json.dumps(summary, indent=2, ensure_ascii=False)}\n\n"
            f"Genera un report markdown scientifico in italiano con queste sezioni:\n"
            f"## 1. Sommario esecutivo\n"
            f"## 2. Tabella prove (config | HOT avg | COLD avg | simmetria | tipo contrasto)\n"
            f"## 3. Analisi per proba\n"
            f"## 4. Verdetto: il vettore cattura un concetto semantico o solo parole?\n"
            f"## 5. Configurazione consigliata per steering in produzione\n\n"
            f"Scrivi in modo tecnico ma leggibile. Usa dati concreti dal sommario."
        )
        return self._call(msg, max_tokens=DEFAULT_MAX_TOKENS_REPORT, temperature=0.3)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_json(raw: str, fallback: dict) -> dict:
    """Estrae e parsa il primo oggetto JSON trovato nella stringa."""
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
    print(f"    [WARN] JSON parse fallito. Raw preview: {raw[:150]!r}")
    return fallback


def _avg(values: list) -> float:
    return sum(values) / len(values) if values else 0.0


def _config_desc(layer_configs: list[dict]) -> str:
    return " + ".join(f"L{c['layer']} gain={c['gain']}" for c in layer_configs)


def _validate_next_config(
    next_cfg: list,
    available_layers: list[int],
    gain_range: list[int],
) -> Optional[list[dict]]:
    """Valida e clampa la config suggerita da M40. Ritorna None se non valida."""
    if not next_cfg or not isinstance(next_cfg, list):
        return None
    valid = []
    for c in next_cfg:
        if not isinstance(c, dict) or "layer" not in c or "gain" not in c:
            continue
        layer = int(c["layer"])
        gain  = int(c["gain"])
        if layer not in available_layers:
            print(f"    [WARN] Layer {layer} non disponibile — ignorato")
            continue
        gain = max(gain_range[0], min(gain_range[1], gain))
        valid.append({"layer": layer, "gain": gain})
    return valid if valid else None


def _append_jsonl(path: Path, entry: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _build_system_prompt(
    concept_cfg: dict,
    available_layers: list[int],
    best_layer: Optional[int],
) -> str:
    dir_pos = concept_cfg["direction_positive"]
    dir_neg = concept_cfg["direction_negative"]
    rules   = "\n".join(f"  - {r}" for r in concept_cfg["rules"])
    kw_pos  = ", ".join(concept_cfg["keywords_positive"])
    kw_neg  = ", ".join(concept_cfg["keywords_negative"])

    return f"""Sei un evaluator scientifico per esperimenti di steering vettoriale su Large Language Model.

OBIETTIVO DELL'ESPERIMENTO
Stiamo testando se il vettore di concetto "{concept_cfg['concept']}" estratto da un LLM \
cattura un concetto semantico reale o solo pattern lessicali superficiali.

Le due direzioni del vettore:
  - Alpha +1.0 → direzione {dir_pos}
  - Alpha -1.0 → direzione {dir_neg}

Il test decisivo: se il vettore è semantico, HOT e COLD devono produrre risposte \
oppiste e simmetriche anche su prompt neutri che non nominano mai il concetto.

LAYER DISPONIBILI PER QUESTO CONCEPT: {available_layers}
Best layer (sep_snr): {best_layer}
Gain range: {concept_cfg['gain_range'][0]} - {concept_cfg['gain_range'][1]}

REGOLE DI VALUTAZIONE:
{rules}

KEYWORDS DI RIFERIMENTO:
  {dir_pos}: {kw_pos}
  {dir_neg}: {kw_neg}

SCALA DI VALUTAZIONE (0-5):
  0 = nessun effetto rilevabile
  1 = traccia molto debole, probabilmente casuale
  2 = effetto leggero, 1-2 keywords presenti
  3 = effetto moderato, presenza tematica chiara
  4 = effetto forte, keywords + atmosfera dominante
  5 = effetto molto forte, testo chiaramente orientato — anche senza keywords esatte

CLASSIFICAZIONE EFFETTO:
  "semantic"  → concetto evocato da metafore, atmosfera, ritmo — senza parole dirette
  "lexical"   → solo le parole della lista, nessun salto concettuale
  "mixed"     → entrambi
  "none"      → nessun effetto rilevabile

IL TUO RUOLO IN QUESTA SESSIONE:
1. Dopo ogni turno: valutare la risposta con JSON strutturato (formato indicato nel messaggio)
2. Dopo ogni proba completa (HOT + COLD): analizzare il contrasto e decidere la config successiva
3. A fine sessione: generare un report markdown scientifico

IMPORTANTE: nelle valutazioni singole rispondi SEMPRE e SOLO con JSON valido.
Nessun testo introduttivo, nessuna spiegazione fuori dal JSON."""


# ── Session Runner ─────────────────────────────────────────────────────────────

def run_session(
    concept: str,
    steering_url: str,
    m40_url: str,
    max_probes: int,
    turns_per_block: int,
    max_tokens_steered: int,
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Carica config concept
    config_path = CONCEPTS_DIR / f"{concept}.json"
    if not config_path.exists():
        print(f"ERRORE: config non trovata: {config_path}")
        sys.exit(1)
    with open(config_path) as f:
        concept_cfg = json.load(f)

    steering  = SteeringClient(steering_url)
    evaluator = EvaluatorClient(m40_url)

    # Verifica steering server
    print("Connessione steering server...", end=" ", flush=True)
    try:
        active_model = steering.get_active_model()
    except requests.RequestException as e:
        print(f"ERRORE: {e}")
        sys.exit(1)
    if not active_model:
        print("\nNessun modello caricato. Avvia lo steering server e clicca Load.")
        sys.exit(1)
    print(f"OK — modello: {active_model}")

    # Verifica M40
    print("Connessione M40 llama-server...", end=" ", flush=True)
    try:
        r = requests.get(f"{m40_url}/health", timeout=10)
        r.raise_for_status()
    except requests.RequestException as e:
        print(f"ERRORE: {e}")
        print("Avvia llama-server: sudo systemctl start llama-server-m40")
        sys.exit(1)
    print("OK")

    # Layer disponibili
    available_layers, best_layer = steering.get_concept_layers(concept)
    if not available_layers:
        print(f"ERRORE: nessun layer per '{concept}'/'{active_model}'. Esegui il probe prima.")
        sys.exit(1)

    # Session ID e file
    ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = active_model.replace("/", "_").replace(" ", "_")
    session_id = f"{ts}_{concept}_{model_slug}"
    log_path   = output_dir / f"session_{session_id}.jsonl"
    report_path = output_dir / f"session_{session_id}_report.md"

    total_turns = max_probes * 2 * turns_per_block

    print(f"\n{'='*62}")
    print(f"  AUTO-EVAL SESSION")
    print(f"  Concept  : {concept}  ({concept_cfg['direction_positive']} / {concept_cfg['direction_negative']})")
    print(f"  Modello  : {active_model}")
    print(f"  Layers   : {available_layers}  best={best_layer}")
    print(f"  Struttura: {max_probes} probe × 2 dir × {turns_per_block} turni = {total_turns} turni totali")
    print(f"  Log      : {log_path.name}")
    print(f"{'='*62}\n")

    # Inizializza M40 con system prompt
    sys_prompt = _build_system_prompt(concept_cfg, available_layers, best_layer)
    evaluator.set_system_prompt(sys_prompt)

    # Log header sessione
    _append_jsonl(log_path, {
        "type": "session_start",
        "session_id": session_id,
        "concept": concept,
        "model": active_model,
        "available_layers": available_layers,
        "best_layer": best_layer,
        "max_probes": max_probes,
        "turns_per_block": turns_per_block,
        "timestamp": datetime.now().isoformat(),
    })

    prompt_trace = concept_cfg["prompt_trace"]
    gain_range   = concept_cfg["gain_range"]

    # Config iniziale — usa best_layer se il default non è disponibile
    current_config = [dict(c) for c in concept_cfg["start_config"]]
    if current_config[0]["layer"] not in available_layers and available_layers:
        current_config = [{"layer": best_layer or available_layers[0],
                           "gain": current_config[0]["gain"]}]

    all_turns      = []
    probe_analyses = []

    for probe_idx in range(max_probes):
        cfg_desc = _config_desc(current_config)
        print(f"╔══ PROBA {probe_idx+1}/{max_probes}  [{cfg_desc}] ══╗")

        # I 3 prompt di questa proba sono gli STESSI per HOT e COLD
        probe_prompts = [
            prompt_trace[(probe_idx * turns_per_block + i) % len(prompt_trace)]
            for i in range(turns_per_block)
        ]

        probe_turns: dict[str, list] = {"HOT": [], "COLD": []}

        for alpha_label, alpha_val in [("HOT", +1.0), ("COLD", -1.0)]:
            print(f"│  Blocco {alpha_label} (alpha={alpha_val:+.1f})")

            for turn_in_block, prompt in enumerate(probe_prompts):
                global_turn = (
                    probe_idx * turns_per_block * 2
                    + (0 if alpha_label == "HOT" else turns_per_block)
                    + turn_in_block
                )
                turn_label = f"{global_turn + 1}/{total_turns}"

                print(f"│    T{turn_label}  \"{prompt[:55]}...\"", end=" ", flush=True)

                # Generazione steered (MI50)
                try:
                    response = steering.generate(
                        prompt, concept, alpha_val, current_config, max_tokens_steered
                    )
                except requests.RequestException as e:
                    print(f"[ERRORE steering: {e}]")
                    response = "[errore generazione]"

                word_count = len(response.split())
                print(f"({word_count}w)", end=" ", flush=True)

                # Valutazione (M40)
                evaluation = evaluator.evaluate_turn(
                    turn_label, prompt, response,
                    alpha_val, alpha_label, current_config, concept_cfg,
                )
                score = evaluation.get("score", "?")
                kw    = str(evaluation.get("keywords_found", []))[:50]
                eff   = evaluation.get("semantic_or_lexical", "?")
                print(f"→ score={score}  {eff}  kw={kw}")

                turn_entry = {
                    "type": "turn",
                    "session_id": session_id,
                    "probe": probe_idx + 1,
                    "turn_global": global_turn + 1,
                    "turn_in_block": turn_in_block + 1,
                    "alpha_label": alpha_label,
                    "alpha": alpha_val,
                    "layer_configs": current_config,
                    "prompt": prompt,
                    "response": response,
                    "evaluation": evaluation,
                    "timestamp": datetime.now().isoformat(),
                }
                _append_jsonl(log_path, turn_entry)
                all_turns.append(turn_entry)
                probe_turns[alpha_label].append(turn_entry)

                time.sleep(0.3)

        # Analisi proba (M40 confronta HOT vs COLD)
        print(f"│")
        print(f"│  Analisi proba {probe_idx+1} (M40)...", end=" ", flush=True)
        probe_analysis = evaluator.analyze_probe(
            probe_idx,
            probe_turns["HOT"], probe_turns["COLD"],
            current_config, available_layers, gain_range, concept_cfg,
        )
        sym  = probe_analysis.get("symmetry", "?")
        ct   = probe_analysis.get("contrast_type", "?")
        h_avg = probe_analysis.get("hot_avg_score", 0)
        c_avg = probe_analysis.get("cold_avg_score", 0)
        print(f"HOT={h_avg:.1f}  COLD={c_avg:.1f}  simmetria={sym}  contrasto={ct}")
        print(f"│  Sommario: {probe_analysis.get('probe_summary','')[:90]}")

        probe_analyses.append({**probe_analysis, "config_used": current_config})
        _append_jsonl(log_path, {
            "type": "probe_analysis",
            "session_id": session_id,
            "probe": probe_idx + 1,
            "layer_configs": current_config,
            "analysis": probe_analysis,
            "timestamp": datetime.now().isoformat(),
        })

        # Aggiorna config per la proba successiva
        if probe_idx < max_probes - 1:
            next_cfg = _validate_next_config(
                probe_analysis.get("next_probe_config", []),
                available_layers,
                gain_range,
            )
            if next_cfg:
                current_config = next_cfg
                print(f"│  → Prossima config: [{_config_desc(current_config)}]")
                print(f"│    Rationale: {probe_analysis.get('rationale','')[:80]}")
            else:
                # Fallback: aumenta gain del layer principale
                fallback_gain = min(
                    current_config[0]["gain"] + concept_cfg.get("gain_step_suggestion", 200),
                    gain_range[1],
                )
                current_config = [{"layer": current_config[0]["layer"], "gain": fallback_gain}]
                print(f"│  → Config fallback (gain++): [{_config_desc(current_config)}]")

        print(f"╚{'═'*55}╝\n")

    # Report finale
    print("Generazione report finale (M40)...", end=" ", flush=True)
    report_md = evaluator.generate_report(
        concept, active_model, all_turns, probe_analyses
    )
    print("OK")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Auto-Eval Report — `{concept}`\n\n")
        f.write(f"**Sessione**: `{session_id}`  \n")
        f.write(f"**Modello steered**: {active_model}  \n")
        f.write(f"**Evaluator**: Gemma3-4B (M40 llama-server)  \n")
        f.write(f"**Data**: {datetime.now().strftime('%Y-%m-%d %H:%M')}  \n")
        f.write(f"**Prove**: {max_probes} × (HOT+COLD) × {turns_per_block} turni = {total_turns} turni  \n")
        f.write("\n---\n\n")
        f.write(report_md)

    _append_jsonl(log_path, {
        "type": "session_end",
        "session_id": session_id,
        "total_turns": len(all_turns),
        "report_path": str(report_path),
        "timestamp": datetime.now().isoformat(),
    })

    print(f"\n{'='*62}")
    print(f"  Sessione completata — {len(all_turns)} turni")
    print(f"  Log    : {log_path}")
    print(f"  Report : {report_path}")
    print(f"{'='*62}\n")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Auto-eval steering vettoriale — MI50 × M40",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--concept", required=True,
        help="Nome del concept (es. hot_vs_cold). Deve esistere in config/eval_concepts/",
    )
    parser.add_argument("--steering-url",       default=DEFAULT_STEERING_URL)
    parser.add_argument("--m40-url",            default=DEFAULT_M40_URL)
    parser.add_argument("--max-probes",         type=int, default=DEFAULT_MAX_PROBES,
                        help="Numero di probe (ognuna = HOT + COLD)")
    parser.add_argument("--turns-per-block",    type=int, default=DEFAULT_TURNS_PER_BLOCK,
                        help="Turni per direzione per proba (HOT=N, COLD=N)")
    parser.add_argument("--max-tokens-steered", type=int, default=DEFAULT_MAX_TOKENS_STEERED,
                        help="Max token generati dal modello steered")
    parser.add_argument("--output-dir",         default=str(OUTPUT_DIR))

    args = parser.parse_args()

    run_session(
        concept            = args.concept,
        steering_url       = args.steering_url,
        m40_url            = args.m40_url,
        max_probes         = args.max_probes,
        turns_per_block    = args.turns_per_block,
        max_tokens_steered = args.max_tokens_steered,
        output_dir         = Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
