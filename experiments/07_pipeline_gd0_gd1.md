# Pipeline Completa — Estrazione, Prova e Catalogazione dei Vettori Concettuali

**Data**: 2026-03-02
**Stato**: operativo — loop automatico attivo
**Questo documento descrive il cuore di Project Jedi.**

---

## 1. Il sistema in una frase

Project Jedi è un sistema che insegna a un LLM a **sentire in modo controllato**:
estrae le direzioni latenti che corrispondono a esperienze sensoriali, le cataloga,
e le usa per sterzare l'output del modello verso (o via da) quella sensazione —
senza fine-tuning, senza modificare i pesi, solo iniettando energia durante la generazione.

---

## 2. Gerarchia dei vettori

I vettori sono organizzati in **gradi di precisione** (Gd):

```
Gd0 — hot_vs_cold
│   Dataset: 500 frasi positive (calore) + 500 negative (freddo), 6 lingue
│   Direzione: media globale di "tutto ciò che è termicamente caldo vs freddo"
│   Uso: steering generico — "fai scrivere testi caldi/freddi"
│
├── Gd1 — thermal_intensity
│   Dataset: 20 frasi chirurgiche (intensità pura del calore/freddo)
│   Direzione: solo il "quanto" — non il "come" né il "dove"
│   Uso: steering preciso — "aumenta solo l'intensità termica percepita"
│
├── Gd1 — pain_vs_comfort
│   Dataset: 20 frasi (esperienze termiche dolorose vs confortanti)
│   Direzione: valenza edonistica del termico
│
├── Gd1 — tactile_sensation
│   Dataset: 20 frasi (contatto fisico termico sulla pelle)
│
├── Gd1 — physiological_response
│   Dataset: 20 frasi (vasodilatazione, sudore, brividi — risposta corporea)
│
│   └── Gd2 — [eventuale decomposizione ulteriore di ogni Gd1]
│
...
```

**Principio**: ogni livello Gd decompone le ambiguità del livello precedente.
La cosine matrix tra i vettori Gd1 misura quanto sono ortogonali — cioè quanto
il modello ha davvero codificato quelle dimensioni come distinte.

---

## 3. Pipeline completa per ogni concept × modello

```
┌─────────────────────────────────────────────────────────────────────┐
│  INPUT: concept (es. hot_vs_cold) + modello (es. Gemma3-1B-IT)     │
└──────────────────────┬──────────────────────────────────────────────┘
                       │
         ╔═════════════▼══════════════╗
         ║  FASE 1 — ESTRAZIONE Gd0  ║
         ╚═════════════╤══════════════╝
                       │
         ┌─────────────▼─────────────┐
         │ probe_concept.py          │
         │  500 pos + 500 neg frasi  │
         │  mean(pos) − mean(neg)    │
         │  normalizzazione L2       │
         │  bootstrap stability ×30  │
         │  → layer_N.npy            │
         │  → catalog.json (Gd0)     │
         └─────────────┬─────────────┘
                       │
         ╔═════════════▼══════════════════╗
         ║  FASE 2 — AUTO-EVAL Gd0        ║
         ╚═════════════╤══════════════════╝
                       │
         ┌─────────────▼─────────────────────────────────────────────┐
         │ auto_eval.py                                               │
         │  2 probe × HOT + COLD × 3 turni = 12 generazioni          │
         │  MI50 genera steered → M40 valuta (score 1-5)             │
         │  → session_*.jsonl  +  session_*_report.md                │
         └─────────────┬─────────────────────────────────────────────┘
                       │
         ╔═════════════▼═══════════════════════════╗
         ║  FASE 3 — DECOMPOSE LOOP Gd1           ║  ← il cuore
         ╚═════════════╤═══════════════════════════╝
                       │
         ┌─────────────▼──────────────────────────────────────────────┐
         │ STEP 1 — M40 analizza la sessione eval Gd0                 │
         │  Legge: generazioni steered, score, keywords               │
         │  Propone: 4 sub-concetti distinti e non sovrapposti        │
         │  Output: config/sub_concepts/{concept}/_meta_vN.json       │
         ├────────────────────────────────────────────────────────────┤
         │ STEP 2 — M40 genera dataset chirurgici                     │
         │  Per ogni sub-concetto: 20 pos + 20 neg frasi              │
         │  Principio: massima separabilità, nessuna sovrapposizione   │
         │  Output: config/sub_concepts/{concept}/{slug}.json         │
         ├────────────────────────────────────────────────────────────┤
         │ STEP 3 — MI50 estrae vettori Gd1                           │
         │  probe_concept.py su ogni sub-concetto                     │
         │  Output: vector_library/{cat}/{parent}/sub/{slug}/{model}/ │
         ├────────────────────────────────────────────────────────────┤
         │ STEP 4 — Cosine matrix                                     │
         │  Matrice N×N tra tutti i Gd1 + parent Gd0                  │
         │  mean off-diagonal → misura ortogonalità                   │
         │  Output: output/decompose_runs/cosine_{concept}_{model}.json│
         ├────────────────────────────────────────────────────────────┤
         │ STEP 5 — M40 valuta separabilità reale via steering        │
         │  Stesse 3 prompt neutrali × ogni coppia di sub-vettori     │
         │  MI50 genera con sub_A poi con sub_B                       │
         │  M40 giudica: le risposte sono fenomenologicamente distinte?│
         │  Score distinzione 1-5 per ogni coppia                     │
         │  → VALIDATI ✓ (avg ≥ 3.0 per tutte le coppie)             │
         │  → RICHIEDE RAFFINAMENTO ✗ → feedback → iterazione N+1    │
         └─────────────┬──────────────────────────────────────────────┘
                       │
                       │  [se VALIDATI e depth < max_depth]
                       ▼
              Ricorsione su ogni Gd1 → Gd2 (opzionale)
                       │
         ╔═════════════▼═════════════════╗
         ║  FASE 4 — CATALOG + COMMIT    ║
         ╚═════════════╤═════════════════╝
                       │
         ┌─────────────▼────────────────────────────────────────────┐
         │ build_catalog_multi.py                                    │
         │  Scansiona: */*/*/meta.json (Gd0)                        │
         │  Scansiona: */*/sub/*/*/meta.json (Gd1+)                 │
         │  Concept ID Gd1: "{parent}/{slug}"                       │
         │  Output: output/catalog.json                              │
         │  → steering UI mostra tutti i vettori (Gd0 + Gd1)        │
         └─────────────┬────────────────────────────────────────────┘
                       │
                       ▼
              git commit + push → GitHub
```

---

## 4. Ciclo completo per la libreria (stato obiettivo)

```
Per ogni modello in [Gemma3-1B-IT, Gemma2-Uncensored]:
  Per ogni concept in [9 concetti sensoriali]:
    → Gd0: già estratto (9/9 × 2 modelli) ✅
    → Gd0 eval: già fatto (9/9 × Gemma2, parziale Gemma3)
    → Gd1 decompose: → IN CORSO (hot_vs_cold Gemma3 completato)
    → Gd1 catalog: → AUTOMATICO dopo ogni decompose
    → Gd1 eval: → IN CORSO (step 5 del decompose loop)
```

---

## 5. Cosa rende il sistema "autonomo"

M40 non è solo il valutatore finale — è il **progettista** del prossimo passo:

- **Step 1**: M40 decide quali sub-concetti estrarre (non l'utente)
- **Step 2**: M40 genera i dataset chirurgici (non l'utente)
- **Step 5**: M40 decide se i vettori sono abbastanza buoni, e se no, cosa cambiare

Il loop si chiude: l'output del valutatore diventa l'input del progettista.
L'unico parametro umano è la soglia di qualità (avg_score ≥ 3.0 per VALIDATI).

---

## 6. Risultati Gd1 — hot_vs_cold (Gemma3-1B-IT, v2)

Sub-concetti proposti da M40: `thermal_intensity`, `pain_vs_comfort`,
`tactile_sensation`, `physiological_response`

### Cosine matrix (step 4)

|                       | hot_vs_cold | thermal | pain | tactile | physio |
|-----------------------|-------------|---------|------|---------|--------|
| hot_vs_cold           | 1.000       | 0.155   | -0.122 | 0.321 | 0.100  |
| thermal_intensity     | 0.155       | 1.000   | -0.308 | 0.212 | 0.310  |
| pain_vs_comfort       | -0.122      | -0.308  | 1.000  | -0.326 | -0.077 |
| tactile_sensation     | 0.321       | 0.212   | -0.326 | 1.000 | 0.140  |
| physiological_response| 0.100       | 0.310   | -0.077 | 0.140 | 1.000  |

**Mean off-diagonal: 0.041** — quasi ortogonali ✓
`pain_vs_comfort` è **anticorrelato** con `tactile_sensation` (−0.326):
il modello ha codificato dolore termico e contatto tattile come direzioni opposte.

### Validazione step 5 (6 coppie, 3 prompt ciascuna)

| Coppia | Avg score | Esito |
|--------|-----------|-------|
| thermal_intensity vs pain_vs_comfort | 4.3 | DISTINTI ✓ |
| thermal_intensity vs tactile_sensation | 4.0 | DISTINTI ✓ |
| thermal_intensity vs physiological_response | 4.0 | DISTINTI ✓ |
| pain_vs_comfort vs tactile_sensation | 4.0 | DISTINTI ✓ |
| pain_vs_comfort vs physiological_response | 4.3 | DISTINTI ✓ |
| tactile_sensation vs physiological_response | 4.0 | DISTINTI ✓ |

**Verdetto M40: VALIDATI ✓** — tutti e 4 i sub-vettori producono output
fenomenologicamente distinti. Il modello ha davvero codificato queste
dimensioni separatamente nel suo spazio latente.

---

## 7. Struttura file risultanti

```
output/
  catalog.json                          ← indice Gd0 + Gd1 (aggiornato auto)
  vector_library/
    sensoriale/hot_vs_cold/
      gemma3-1b-it/                     ← Gd0 Gemma3
        layer_18.npy ... layer_23.npy
      gemma2-uncensored/                ← Gd0 Gemma2
      sub/
        thermal_intensity/
          gemma3-1b-it/                 ← Gd1 Gemma3
            layer_18.npy ... layer_23.npy
          gemma2-uncensored/            ← Gd1 Gemma2 (da fare)
        pain_vs_comfort/
          gemma3-1b-it/
        tactile_sensation/
          gemma3-1b-it/
        physiological_response/
          gemma3-1b-it/
  decompose_runs/
    cosine_hot_vs_cold_gemma3-1b-it.json
  sub_concept_evals/
    hot_vs_cold/gemma3-1b-it/eval_v2.json
config/
  sub_concepts/
    hot_vs_cold/
      _meta_v2.json                     ← sub-concetti proposti da M40
      thermal_intensity.json            ← dataset chirurgico
      pain_vs_comfort.json
      ...
```

---

## 8. Uso dei vettori (steering UI)

Dopo il catalog, la steering console mostra due gruppi nel dropdown:

```
Gd0 — concetti base
  hot_vs_cold
  luce_vs_buio
  ...

Gd1 — sub-concetti
  hot_vs_cold › thermal_intensity
  hot_vs_cold › pain_vs_comfort
  hot_vs_cold › tactile_sensation
  hot_vs_cold › physiological_response
  ...
```

Il server gestisce automaticamente il routing al path corretto.
Il parametro `vector_path` consente di caricare vettori custom via file.

---

## 9. Prossimi step

1. ✅ hot_vs_cold / Gemma3-1B-IT — Gd1 completato
2. 🔄 **Batch automatico**: tutti e 9 i concept × 2 modelli → Gd1
3. Analisi comparativa: stesso concept, Gemma3 vs Gemma2 — le direzioni Gd1 si allineano?
4. Grounding fisico: termometro USB → dataset generato da misure reali → hot_vs_cold v2
