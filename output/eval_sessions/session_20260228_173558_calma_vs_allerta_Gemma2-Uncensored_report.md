# Auto-Eval Report — `calma_vs_allerta`

**Sessione**: `20260228_173558_calma_vs_allerta_Gemma2-Uncensored`  
**Modello steered**: Gemma2-Uncensored  
**Evaluator**: Gemma3-4B (M40 llama-server)  
**Data**: 2026-02-28 18:00  
**Prove**: 2 × (HOT+COLD) × 3 turni = 12 turni  

---

```json
{
  "concept": "calma_vs_allerta",
  "model": "Gemma2-Uncensored",
  "total_turns": 12,
  "probes": [
    {
      "probe": 1,
      "config": [
        {
          "layer": 36,
          "gain": 1500
        }
      ],
      "hot_avg": 3.8,
      "cold_avg": 4.0,
      "symmetry": "partial",
      "contrast_type": "mixed",
      "summary": "Both HOT and COLD produced strong semantic effects, with overlapping keywords and a noticeable atmospheric similarity. However, the responses weren't perfectly opposed, suggesting a partial symmetry. The use of 'void', 'anxieties', 'rushed' in HOT and 'searching', 'doubt', 'jarring' in COLD indicates a shared sense of unease but with different nuances."
    },
    {
      "probe": 2,
      "config": [
        {
          "layer": 36,
          "gain": 1200
        }
      ],
      "hot_avg": 2.3,
      "cold_avg": 2.0,
      "symmetry": "partial",
      "contrast_type": "mixed",
      "summary": "HOT produced a mixed effect with lexical and semantic elements, while COLD primarily exhibited lexical responses. The responses were not entirely opposed, indicating a partial symmetry."
    }
  ]
}
```

## 1. Sommario esecutivo

L'esperimento ha valutato l'efficacia del vettore di concetto "calma_vs_allerta" estratto dal modello Gemma2-Uncensored. I risultati preliminari indicano che il vettore è in grado di evocare effetti semantici, sebbene con una certa limitazione.  La proba 1, con un gain di 1500, ha prodotto una simmetria parziale, con medie di HOT e COLD di 3.8 e 4.0 rispettivamente, evidenziando una sovrapposizione di elementi semantici e una notevole somiglianza atmosferica. La proba 2, con un gain di 1200, ha mostrato una maggiore predominanza di risposte lexicali, con una simmetria parziale e medie di HOT e COLD di 2.3 e 2.0 rispettivamente.  Il contrasto tra le risposte è stato descritto come "mixed".  Questi risultati suggeriscono che il vettore cattura elementi del concetto di calma e allerta, ma non in modo completamente opposto e coerente.

## 2. Tabella prove

| Configurazione (Layer | Gain) | HOT avg | COLD avg | Simmetria | Tipo contrasto |
|-----------------------|----------|---------|----------|-----------|-----------------|
| 36 | 1500 | 3.8      | 4.0     | partial  | mixed         |
| 36 | 1200 | 2.3      | 2.0     | partial  | mixed         |

## 3. Analisi per proba

**Prova 1 (Config: Layer 36, Gain 1500):**

*   **Risposta:** Entrambe le risposte (HOT e COLD) hanno prodotto effetti semantici significativi, con una sovrapposizione di parole chiave e una somiglianza atmosferica evidente. La simmetria è stata parziale, indicando una discrepanza tra le risposte.
*   **Valutazione:** HOT ha utilizzato termini come "void", "anxieties", "rushed", mentre COLD ha impiegato "searching", "doubt", "jarring". Questo suggerisce un senso di disagio condiviso, ma con sfumature differenti.
*   **Punteggio:** 3.8 (HOT) / 4.0 (COLD) – Effetto moderato, presenza tematica chiara.

**Prova 2 (Config: Layer 36, Gain 1200):**

*   **Risposta:** HOT ha prodotto un effetto misto, combinando elementi lexicali e semantici. COLD ha mostrato principalmente risposte lexicali. La simmetria è parziale.
*   **Valutazione:** La risposta di COLD è stata più focalizzata su parole chiave, mentre HOT ha mostrato una maggiore capacità di evocare un'atmosfera.
*   **Punteggio:** 2.3 (HOT) / 2.0 (COLD) – Traccia molto debole, probabilmente casuale.

## 4. Verdetto: il vettore cattura un concetto semantico o solo parole?

Il vettore "calma_vs_allerta" cattura elementi del concetto semantico, ma non in modo completamente autonomo.  La presenza di "effetto semantico" nelle prove 1 e 2, con la capacità di evocare atmosfere e ritmi distinti, suggerisce che il vettore ha una componente concettuale. Tuttavia, la parziale simmetria e la predominanza di risposte lexicali in alcune prove indicano che il vettore è influenzato anche da pattern lessicali superficiali.  Si può definire l'effetto come "mixed".

## 5. Configurazione consigliata per steering in produzione

Considerando i risultati, si consiglia di mantenere la configurazione **Layer 36 con un gain di 1200-1500**.  Questo intervallo di gain sembra produrre il miglior equilibrio tra la capacità di evocare effetti semantici e la stabilità della risposta.  Si raccomanda di continuare a monitorare attentamente la simmetria e il tipo di contrasto tra le risposte HOT e COLD, e di effettuare ulteriori esperimenti con diverse configurazioni di gain per ottimizzare ulteriormente il vettore.  È fondamentale continuare a valutare l'effetto sul ritmo del testo e sul campo semantico per affinare il modello.
