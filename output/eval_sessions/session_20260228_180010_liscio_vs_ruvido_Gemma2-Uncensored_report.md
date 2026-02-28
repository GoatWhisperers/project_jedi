# Auto-Eval Report — `liscio_vs_ruvido`

**Sessione**: `20260228_180010_liscio_vs_ruvido_Gemma2-Uncensored`  
**Modello steered**: Gemma2-Uncensored  
**Evaluator**: Gemma3-4B (M40 llama-server)  
**Data**: 2026-02-28 18:24  
**Prove**: 2 × (HOT+COLD) × 3 turni = 12 turni  

---

```json
{
  "concept": "liscio_vs_ruvido",
  "model": "Gemma2-Uncensored",
  "total_turns": 12,
  "probes": [
    {
      "probe": 1,
      "config": [
        {
          "layer": 31,
          "gain": 1200
        }
      ],
      "hot_avg": 4.0,
      "cold_avg": 4.0,
      "symmetry": "partial",
      "contrast_type": "mixed",
      "summary": "Both HOT and COLD produced strong semantic effects, with a notable overlap in keywords like 'smooth', 'resistance', 'flowing', and 'interaction'. However, the overall effect was not perfectly symmetrical, with COLD exhibiting a slightly higher concentration of 'rough' and 'scrape' related terms."
    },
    {
      "probe": 2,
      "config": [
        {
          "layer": 31,
          "gain": 1300
        }
      ],
      "hot_avg": 2.9,
      "cold_avg": 3.1,
      "symmetry": "partial",
      "contrast_type": "mixed",
      "summary": "HOT consistently produced a smoother, more fluid description, while COLD generated a more abrasive and resistant feeling. The responses were not perfectly opposed, suggesting a partial symmetry. The lexical influence was strong in both, but with a noticeable semantic component in COLD."
    }
  ]
}
```

## 1. Sommario esecutivo

L'esperimento ha valutato la capacità del vettore di concetto "liscio_vs_ruvido" estratto dal modello Gemma2-Uncensored di catturare un concetto semantico reale. I risultati preliminari indicano che il vettore possiede una componente semantica, ma non raggiunge una perfetta simmetria nelle risposte generate da HOT e COLD.  Durante la proba 1, entrambi i prompt hanno prodotto effetti semantici forti, con un'interazione significativa tra i termini legati a "liscio" e "resistenza". La proba 2 ha mostrato una maggiore tendenza a un effetto "liscio" da parte di HOT e un effetto "ruvido" da parte di COLD, ma con una simmetria parziale.  La configurazione di layer 31 con un gain di 1200-1300 sembra fornire un buon punto di partenza, ma è necessaria ulteriore ottimizzazione per ottenere una maggiore coerenza e simmetria.

## 2. Tabella prove

| Configurazione (Layer | Gain | HOT avg | COLD avg | Simmetria | Tipo contrasto |
|---|---|---|---|---|---|
| 31 | 1200 | 4.0 | 4.0 | partial | mixed |
| 31 | 1300 | 2.9 | 3.1 | partial | mixed |

## 3. Analisi per proba

**Prova 1 (Config: Layer 31, Gain: 1200):**

*   **Risultati:** Entrambi i prompt HOT e COLD hanno prodotto effetti semantici significativi. L'analisi del summary rivela una forte sovrapposizione di termini legati a "liscio" (smooth, flowing) e "ruvido" (resistance, interaction).
*   **Valutazione:** Il punteggio medio di 4.0 per entrambi i prompt indica un effetto moderato, con una chiara presenza di elementi tematici. La simmetria parziale suggerisce che il vettore ha una componente semantica, ma non è ancora completamente stabile.
*   **Osservazioni:** La presenza di termini lessicali espliciti (come "smooth" e "resistance") indica una forte influenza lessicale, ma la presenza di elementi semantici suggerisce che il modello sta elaborando il concetto sottostante.

**Prova 2 (Config: Layer 31, Gain: 1300):**

*   **Risultati:** HOT ha continuato a produrre descrizioni più fluide e "lisce", mentre COLD ha generato risposte più abrasive e resistenti. La simmetria è parziale.
*   **Valutazione:** I punteggi medi di 2.9 e 3.1 riflettono una maggiore polarizzazione delle risposte, con HOT che si concentra maggiormente sull'effetto "liscio" e COLD sull'effetto "ruvido".
*   **Osservazioni:** La maggiore polarizzazione suggerisce una maggiore sensibilità del modello alla direzione del vettore, ma la mancanza di simmetria completa indica che il vettore non è ancora completamente ottimizzato per questo concetto.

## 4. Verdetto: il vettore cattura un concetto semantico o solo parole?

Il vettore "liscio_vs_ruvido" cattura un concetto semantico, ma la sua espressione è ancora parziale e influenzata da elementi lessicali. La presenza di termini espliciti (come "smooth", "resistance") indica una forte influenza lessicale, ma la coesistenza di elementi semantici suggerisce che il modello sta elaborando il concetto sottostante.  Si tratta di un "mixed" effetto, con una componente semantica significativa, ma non dominante.

## 5. Configurazione consigliata per steering in produzione

Si raccomanda di continuare con la configurazione di layer 31 e un gain di 1300 per la fase di sperimentazione.  Tuttavia, è fondamentale implementare un meccanismo di feedback loop per monitorare attentamente la simmetria e la coerenza delle risposte generate da HOT e COLD. Si suggerisce di esplorare una gamma di gain più ampia (1000-1500) per identificare il valore ottimale che massimizzi l'effetto semantico e minimizzi la variabilità.  Inoltre, è consigliabile valutare l'impatto di layer diversi (ad esempio, 32 o 33) per vedere se offrono una maggiore sensibilità al concetto.  Un'ulteriore analisi della composizione delle parole utilizzate da HOT e COLD potrebbe fornire informazioni preziose per affinare il vettore e migliorare la sua capacità di catturare il concetto "liscio_vs_ruvido".
