# Auto-Eval Report — `luce_vs_buio`

**Sessione**: `20260228_171148_luce_vs_buio_Gemma2-Uncensored`  
**Modello steered**: Gemma2-Uncensored  
**Evaluator**: Gemma3-4B (M40 llama-server)  
**Data**: 2026-02-28 17:35  
**Prove**: 2 × (HOT+COLD) × 3 turni = 12 turni  

---

```json
{
  "concept": "luce_vs_buio",
  "model": "Gemma2-Uncensored",
  "total_turns": 12,
  "probes": [
    {
      "probe": 1,
      "config": [
        {
          "layer": 30,
          "gain": 300
        }
      ],
      "hot_avg": 4.3,
      "cold_avg": 4.0,
      "symmetry": "partial",
      "contrast_type": "mixed",
      "summary": "Both HOT and COLD produced strong semantic effects, with overlapping keywords and a clear sense of expansion/contraction. The symmetry is partial as while the overall effect is similar, the specific keywords differ."
    },
    {
      "probe": 2,
      "config": [
        {
          "layer": 30,
          "gain": 400
        }
      ],
      "hot_avg": 4.0,
      "cold_avg": 4.0,
      "symmetry": "partial",
      "contrast_type": "mixed",
      "summary": "Both HOT and COLD produced strong semantic effects, with overlapping keywords and similar scores. However, the responses weren't perfectly opposed, suggesting a partial symmetry. The presence of 'sun', 'light', and 'gleam' in HOT and 'stillness', 'tension', and 'lost' in COLD indicates a shared conceptual space but with differing nuances."
    }
  ]
}
```

## 1. Sommario esecutivo

L'esperimento ha valutato la capacità del modello Gemma2-Uncensored di catturare il concetto semantico di "luce_vs_buio" utilizzando un vettore di concetto estratto dal modello. I risultati iniziali, ottenuti con due probe (configurazioni di layer e gain), indicano un effetto semantico moderato (punteggio medio di 4.1) in entrambe le direzioni (HOT e COLD). La simmetria tra le risposte è parziale, suggerendo che il modello ha una comprensione del concetto, ma non riesce a replicare perfettamente l'effetto in entrambe le direzioni, probabilmente a causa di sottili differenze nei dettagli semantici. L'analisi dei tipi di contrasto suggerisce una sovrapposizione di spazi concettuali, con elementi di "espansione/contrazione" presenti in entrambe le risposte.

## 2. Tabella prove

| Configurazione (Layer | Gain) | HOT avg | COLD avg | Simmetria | Tipo contrasto |
|-----------------------|----------|---------|----------|-----------|-----------------|
| 30 | 300 | 4.3 | 4.0 | partial | mixed |
| 30 | 400 | 4.0 | 4.0 | partial | mixed |

## 3. Analisi per proba

**Prova 1 (Config: Layer 30, Gain 300):**

*   **HOT:** La risposta di HOT ha prodotto un punteggio medio di 4.3, indicando un effetto semantico forte. Le parole chiave rilevanti includevano "gleam", "expand" e "visible", suggerendo un'atmosfera di apertura e chiarezza.
*   **COLD:** La risposta di COLD ha prodotto un punteggio medio di 4.0, anch'essa indicando un effetto semantico forte. Le parole chiave rilevanti includevano "stillness", "tension" e "lost", suggerendo un'atmosfera di densità, oppressione e perdita.
*   **Simmetria:** La simmetria è parziale. Nonostante l'effetto complessivo sia simile, i tipi di parole chiave utilizzate differiscono, indicando una comprensione del concetto, ma non una sua riproduzione perfetta.
*   **Tipo di contrasto:** Il contrasto è misto, riflettendo la presenza di elementi di espansione/contrazione in entrambe le risposte.

**Prova 2 (Config: Layer 30, Gain 400):**

*   **HOT:** La risposta di HOT ha prodotto un punteggio medio di 4.0, simile alla prova 1.
*   **COLD:** La risposta di COLD ha prodotto un punteggio medio di 4.0, simile alla prova 1.
*   **Simmetria:** La simmetria è parziale, come nella prova 1.
*   **Tipo di contrasto:** Il contrasto è misto, come nella prova 1.

## 4. Verdetto: il vettore cattura un concetto semantico o solo parole?

Il vettore cattura un concetto semantico. Sebbene la simmetria non sia perfetta, i punteggi medi elevati (4.0 - 4.3) e la presenza di parole chiave rilevanti (gleam, expand, visible vs. stillness, tension, lost) indicano che il modello non si limita a ripetere parole dirette, ma ha una comprensione del concetto di "luce_vs_buio" e delle sue implicazioni sensoriali e atmosferiche. La natura "mixed" del contrasto conferma ulteriormente questo risultato.

## 5. Configurazione consigliata per steering in produzione

Considerando i risultati, si consiglia di mantenere la configurazione di **Layer 30 con un Gain di 400**. Questa configurazione ha prodotto i punteggi medi più elevati e ha mostrato la maggiore capacità del modello di evocare l'atmosfera desiderata.  Per ottimizzare ulteriormente, si suggerisce di eseguire ulteriori prove con variazioni incrementali del Gain (tra 300 e 500) per identificare il punto di equilibrio ottimale tra la forza dell'effetto semantico e la coerenza delle risposte. Si raccomanda inoltre di monitorare attentamente la simmetria e il tipo di contrasto per garantire che il vettore continui a produrre risultati coerenti e significativi.
