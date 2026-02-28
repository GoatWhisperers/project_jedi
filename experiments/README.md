# Experiments — Project Jedi

Questo diario documenta gli esperimenti condotti su concept vectors e activation steering.
Ogni file descrive una sessione sperimentale: configurazione, risultati, generazioni, analisi.

---

## Indice

| # | File | Concetto | Modello | Data | Risultato |
|---|------|----------|---------|------|-----------|
| 00 | [architettura.md](00_architettura.md) | — | — | 2026-02-28 | Descrizione del sistema completo |
| 01 | [hot_vs_cold](01_hot_vs_cold_Gemma3-1B-IT_2026-02-28.md) | hot_vs_cold | Gemma3-1B-IT | 2026-02-28 | ✅ Semantico, score 4.0/5 |
| 02 | [luce_vs_buio](02_luce_vs_buio_Gemma3-1B-IT_2026-02-28.md) | luce_vs_buio | Gemma3-1B-IT | 2026-02-28 | ✅ Semantico, score 4.0/3.7 |

---

## Come leggere i risultati

Ogni sessione è una **auto-eval**: il modello Gemma3-1B-IT (MI50) genera testo con il vettore iniettato,
e Gemma3-4B (M40) valuta ciascuna generazione in modo cieco, senza sapere quale direzione è stata iniettata.

Il valutatore assegna:
- **score** da 1 a 5 (quanto forte è l'effetto del concetto nella generazione)
- **semantic** / **lexical** (effetto semantico evocativo o solo parole letterali)
- **keywords** trovate (parole che rivelano la direzione)

Poi aggrega:
- **HOT avg** / **COLD avg** — media score per direzione
- **symmetry** — se i due poli sono egualmente chiari e opposti
- **contrast** — se l'effetto è puramente semantico, solo lessicale, o misto

---

## Setup hardware

| Ruolo | Hardware | Modello |
|-------|----------|---------|
| Steering (MI50) | AMD Radeon MI50, 32 GB VRAM, ROCm | Gemma3-1B-IT (HF/Transformers) |
| Evaluator (M40) | NVIDIA Tesla M40, 12 GB VRAM, CUDA 11.8 | Gemma3-4B Q4_K_M (llama.cpp) |

Il M40 usa un binario `llama-server` compilato con sm_52 (Maxwell) via Docker,
con il flag `--allow-shlib-undefined` per risolvere i simboli CUDA a runtime.

---

## Domanda scientifica centrale

> **Il vettore concetto cattura una direzione semantica o solo pattern lessicali superficiali?**

Un vettore lessicale produce generazioni che contengono letteralmente la parola (es. "hot", "cold").
Un vettore semantico produce generazioni che *evocano* il concetto attraverso metafore, atmosfera,
imagery — senza necessariamente usare la parola diretta.

I nostri esperimenti indicano che i layer profondi (70–90% del modello) catturano direzioni
prevalentemente **semantiche**, con alcune interferenze lessicali residue.
