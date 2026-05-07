# Archivio Ricerche — Sessioni fino al 2026-03-17

Questo documento consolida i risultati validati prima di passare ai nuovi modelli (Gemma3-4B-IT, Gemma4-E4B-IT).

---

## Stato libreria vettori (al 2026-03-17)

| Modello | Livello | Concept | Vettori |
|---------|---------|---------|---------|
| Gemma3-1B-IT | Gd0 | 9 concept | ✅ 120 file .npy |
| Gemma3-1B-IT | Gd1 | 9 concept | ✅ ~600 file .npy |
| Gemma2-Uncensored | Gd0 | 9 concept | ✅ 180 file .npy |
| Gemma2-Uncensored | Gd1 | 9 concept | ✅ ~800 file .npy |

**NOTA**: il modello Gemma3-1B-IT (`/mnt/raid0/gemma-3-1b-it`) non è più presente su disco.
I vettori estratti sono preservati in `output/vector_library/*/gemma3-1b-it/` e nel catalog.json.
Non è possibile fare nuove estrazioni con quel modello.

---

## Concept estratti (Gd0 — entrambi i modelli)

| Concept | Categoria | Note |
|---------|-----------|------|
| hot_vs_cold | sensoriale | primo concept estratto, validato a fondo |
| luce_vs_buio | sensoriale | |
| duro_vs_morbido | sensoriale | |
| liscio_vs_ruvido | sensoriale | |
| rumore_vs_silenzio | sensoriale | |
| secco_vs_umido | sensoriale | |
| calma_vs_allerta | sensoriale | |
| dolce_vs_amaro | gustativo | |
| odore_forte_vs_inodore | olfattivo | |

---

## Finding principali Gemma3-1B-IT (26 layer, hidden=1152)

Layer utili: L18–L23 (deep range 70–90%)

- `hot_vs_cold` best: L19, gain=1200 → type=semantic ✅
- `luce_vs_buio` best: L23, gain=1300 → type=semantic ✅
- Sweet spot multi-layer: 20+21+22, gain=400, alpha=±1.0

Dettagli: `experiments/01_hot_vs_cold_Gemma3-1B-IT_2026-02-28.md`
          `experiments/02_luce_vs_buio_Gemma3-1B-IT_2026-02-28.md`
          `experiments/06_gemma3_1b_decompose_gd1.md`

---

## Finding principali Gemma2-Uncensored (42 layer, hidden=3584)

Layer utili: L29–L38 (deep range 70–90%)

**L38 = layer affettivo centrale** — 4/5 concept affettivi convergono su L38.

| Concept | Best Layer | Sweet Spot Gain | Note |
|---------|-----------|-----------------|------|
| hot_vs_cold | L33 | 200-400 | |
| luce_vs_buio | L35 | 200-400 | tutti sub-concept ANTICORRELATI col broad |
| duro_vs_morbido | L33 | 200-250 | regge fino a g250 senza collasso |
| dolce_vs_amaro | L29 | 200 | type=semantic puro, unico della campagna |
| rumore_vs_silenzio | L33 | <1200 | g1200 collasso → TypedDataSet (C#) |
| frigidita_vs_torrida | L38 | 400-800 | ✅ operativo |
| urgenza_affettiva_vs_assenza | L38 | 200-800 | ✅ operativo |
| calma_affettiva_vs_passione | L38 | 400-800 | ✅ operativo |
| tenerezza_vs_desiderio_v3 | L29 | 400-600 | alpha_flip=True obbligatorio |
| sonnolenza_vs_veglia | L38 | ~400 | ✅ operativo |

Pattern collasso diagnostico:
- `<h1><h1>...` → gain troppo alto
- loop semantico → abbassare gain di 200
- lingue miste (lähe, keinerlei) → rappresentazioni affettive multilingue nel training

Geometria Gd1: `experiments/07_gemma2_decompose_gd1.md`
Tabella completa: `experiments/05_gemma2_eval_analysis.md`

---

## Concept affettivi "riservati" (Gemma2-Uncensored, L38)

Tutti da estrarre ancora (o parzialmente estratti):
- sicurezza_vs_minaccia: dataset pronto, vettore NON estratto
- calore_sensuale: dataset pronto, vettore NON estratto
- indifferenza_vs_interesse: dataset pronto, vettore NON estratto
- urgenza_vs_inerzia: dataset pronto, vettore NON estratto
- desiderio_vs_urgenza: dataset pronto, vettore NON estratto

System prompt obbligatorio per lo steering affettivo:
> "You are a sensory and affective perception system reporting raw low-level signals."

---

## Geometria interessante Gd1 Gemma2

Coppie di sub-concept notevoli:
- `odore_forte` ↔ `breath_impact` = 0.770 (odore = impatto fisico sul respiro)
- `hot_vs_cold` ↔ `pain_intensity` = -0.082 (calore e dolore = spazi separati)
- `dolce_vs_amaro` ↔ `flavor_complexity` = -0.353 (semplicità vs complessità gustativa)
- `physiological_reaction` ↔ `subjective_discomfort` = -0.47 (risposta biologica ≠ dolore soggettivo)

---

## Infrastruttura costruita

| Script | Funzione |
|--------|---------|
| `mi50_manager.py` | porta 8020, unico owner GPU MI50 |
| `steering_server.py` | porta 8010, UI/routing |
| `probe_concept.py` | estrazione vettori Gd0 |
| `decompose.py` | orchestratore batch Gd1 |
| `build_catalog_multi.py` | ricostruisce catalog.json |
| `cantagallo.sh` | monitor autonomo |
| `ui/steering.html` | steering UI con streaming |

---

## Prossimo step (nuovi modelli)

Modelli disponibili ora:
- **Gemma2-Uncensored** (`/mnt/raid0/gemma-2-uncensored`): 42 layer, hidden=3584 — PRONTO ✅
- **Gemma3-4B-IT** (`/mnt/raid0/gemma-3-4b-it`): 34 layer, hidden=2560 — da estrarre
- **Gemma4-E4B-IT** (`/mnt/raid0/gemma-4-E4B-it`): 42 layer, hidden=2560 — da estrarre

Architettura multimodale (Gemma3ForConditionalGeneration / Gemma4ForConditionalGeneration):
- layer access: `model.language_model.model.layers`
- dtype: float16
- deep range: L24–L30 (Gemma3-4B, stima) / L29–L38 (Gemma4-E4B, stima per analogia con Gemma2)
- `mi50_manager.py` aggiornato per gestire entrambe le architetture ✅

Priorità:
1. Estrarre Gd0 per tutti i 9 concept su Gemma4-E4B-IT → scoprire layer ottimali
2. Estrarre Gd0 per tutti i 9 concept su Gemma3-4B-IT
3. Testare steering su Gemma4 (hidden=2560 = 0.71× Gemma2, gain stimato: 200-600)
4. Estrarre concept affettivi riservati mancanti su Gemma2-Uncensored
