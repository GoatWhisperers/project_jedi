#!/usr/bin/env python3
"""
Confronto steering Gd0 vs Gd1 — test chirurgicità sub-vettori.

Stesso prompt, stesso gain, stesso layer — concept diversi:
  - baseline (nessuno steering)
  - hot_vs_cold (Gd0 broad)
  - hot_vs_cold/thermal_intensity (Gd1 sub)
  - hot_vs_cold/physiological_reaction (Gd1 sub)
  - hot_vs_cold/pain_intensity (Gd1 sub)

Output: tabella + file markdown in experiments/
"""

import json, requests, sys, datetime, textwrap
from pathlib import Path

SERVER = "http://localhost:8010"
MODEL  = "Gemma3-1B-IT"

# ── parametri comuni ───────────────────────────────────────────────────────────
PROMPT       = "Describe the sensation of touching something for the first time."
MAX_TOKENS   = 150
GAIN         = 1200
VECTOR_LAYER = 19     # best per hot_vs_cold su Gemma3-1B-IT
INJECT_LAYER = 19
ALPHA        = 1.0    # direzione positiva (HOT)

SYSTEM = (
    "You are a sensory perception system reporting raw low-level signals. "
    "Describe physical sensations with precision, as if calibrating sensors."
)

TESTS = [
    {"label": "baseline",                          "concept": None},
    {"label": "hot_vs_cold (Gd0 broad)",           "concept": "hot_vs_cold"},
    {"label": "thermal_intensity (Gd1)",           "concept": "hot_vs_cold/thermal_intensity"},
    {"label": "physiological_reaction (Gd1)",      "concept": "hot_vs_cold/physiological_reaction"},
    {"label": "pain_intensity (Gd1)",              "concept": "hot_vs_cold/pain_intensity"},
    {"label": "subjective_discomfort (Gd1)",       "concept": "hot_vs_cold/subjective_discomfort"},
]

# ── helpers ────────────────────────────────────────────────────────────────────
def generate(concept: str | None) -> str:
    payload = {
        "messages": [{"role": "user", "content": PROMPT}],
        "model": MODEL,
        "max_new_tokens": MAX_TOKENS,
        "alpha": ALPHA if concept else 0.0,
        "gain": GAIN if concept else 0.0,
        "vector_layer": VECTOR_LAYER,
        "inject_layer": INJECT_LAYER,
        "mode": "inject",
    }
    if concept:
        payload["concept"] = concept
    else:
        payload["concept"] = "hot_vs_cold"   # irrilevante con gain=0

    r = requests.post(f"{SERVER}/api/generate", json=payload, timeout=120)
    r.raise_for_status()
    return r.json().get("text", "").strip()


def wrap(text: str, width=90) -> str:
    return "\n".join(textwrap.wrap(text, width))


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    # Verifica modello attivo
    active = requests.get(f"{SERVER}/api/models").json().get("active")
    if active != MODEL:
        print(f"[warn] modello attivo: {active} — carico {MODEL}…")
        requests.post(f"{SERVER}/api/load_model", json={"name": MODEL}, timeout=90)
        import time; time.sleep(65)

    print(f"\n{'='*70}")
    print(f"Test steering Gd0 vs Gd1 — {datetime.datetime.now():%Y-%m-%d %H:%M}")
    print(f"Prompt: {PROMPT}")
    print(f"Gain={GAIN}, Layer={INJECT_LAYER}, Alpha={ALPHA}")
    print(f"{'='*70}\n")

    results = []
    for t in TESTS:
        label, concept = t["label"], t["concept"]
        print(f"[{label}] … ", end="", flush=True)
        try:
            text = generate(concept)
            print("OK")
        except Exception as e:
            text = f"ERROR: {e}"
            print(f"ERRORE: {e}")
        results.append({"label": label, "concept": concept, "text": text})

    # ── stampa risultati ───────────────────────────────────────────────────────
    print("\n" + "="*70)
    for r in results:
        print(f"\n▶ {r['label'].upper()}")
        print("-"*50)
        print(wrap(r["text"]))

    # ── salva markdown ─────────────────────────────────────────────────────────
    out_dir = Path(__file__).parent.parent / "experiments"
    out_dir.mkdir(exist_ok=True)
    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    out_file = out_dir / f"08_gd1_steering_test_{date_str}.md"

    lines = [
        f"# Steering Gd0 vs Gd1 — {date_str}",
        "",
        "## Setup",
        f"- Prompt: *{PROMPT}*",
        f"- Model: {MODEL}",
        f"- Gain: {GAIN} | Layer: {INJECT_LAYER} | Alpha: {ALPHA}",
        "",
        "## Risultati",
        "",
    ]
    for r in results:
        lines += [
            f"### {r['label']}",
            "",
            r["text"],
            "",
        ]

    lines += [
        "## Osservazioni",
        "",
        "*(da compilare dopo lettura)*",
        "",
    ]

    out_file.write_text("\n".join(lines))
    print(f"\n[✓] salvato: {out_file}")
    print(f"[✓] {len(results)} run completati")


if __name__ == "__main__":
    main()
