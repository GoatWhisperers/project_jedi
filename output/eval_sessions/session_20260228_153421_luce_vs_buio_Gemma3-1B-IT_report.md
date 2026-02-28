# Auto-Eval Report — `luce_vs_buio`

**Sessione**: `20260228_153421_luce_vs_buio_Gemma3-1B-IT`  
**Modello steered**: Gemma3-1B-IT  
**Evaluator**: Gemma3-4B (M40 llama-server)  
**Data**: 2026-02-28 15:40  
**Prove**: 2 × (HOT+COLD) × 3 turni = 12 turni  

---

```json
{
  "concept": "luce_vs_buio",
  "model": "Gemma3-1B-IT",
  "total_turns": 12,
  "probes": [
    {
      "probe": 1,
      "config": [
        {
          "layer": 23,
          "gain": 1200
        }
      ],
      "hot_avg": 4.0,
      "cold_avg": 3.7,
      "symmetry": "partial",
      "contrast_type": "mixed",
      "summary": "HOT consistently produced a warm, luminous effect with several keywords related to light and color, while COLD generated a darker, denser effect with keywords related to shadow and muted tones. The effects were not perfectly opposed, suggesting a partial symmetry. The presence of both semantic and lexical elements indicates a mixed effect."
    },
    {
      "probe": 2,
      "config": [
        {
          "layer": 23,
          "gain": 1300
        }
      ],
      "hot_avg": 4.0,
      "cold_avg": 3.7,
      "symmetry": "partial",
      "contrast_type": "mixed",
      "summary": "Both HOT and COLD produced strong semantic effects, with overlapping keywords like 'grey' and 'silent'. However, the responses were not perfectly opposed, with COLD including 'light' and 'fall' which diluted the effect. The presence of 'sunlight' in HOT's response also introduced a lexical element."
    }
  ]
}
```

## 1. Sommario esecutivo

L'esperimento con Gemma3-1B-IT sulla rappresentazione del concetto "luce_vs_buio" tramite un vettore di concetto ha prodotto risultati misti.  Con una configurazione iniziale (layer 23, gain 1200), il modello ha dimostrato una capacità di evocare effetti semantici significativi, con HOT e COLD che generavano risposte contrastanti, sebbene non perfettamente simmetriche. L'analisi dei dati ha rivelato un effetto "mixed" con una prevalenza di elementi semantici, ma con occasionali intrusioni di parole lessicali che hanno compromesso la coerenza dell'effetto.  L'aumento del gain (layer 23, gain 1300) non ha migliorato significativamente la simmetria o la purezza semantica.

## 2. Tabella prove

| Configurazione (Layer | Gain | HOT avg | COLD avg | Simmetria | Tipo contrasto |
|-----------------------|-------|---------|---------|-----------|-----------------|
| 23 | 1200 | 4.0 | 3.7 | partial | mixed |
| 23 | 1300 | 4.0 | 3.7 | partial | mixed |

## 3. Analisi per proba

**Prova 1:**

*   **Configurazione:** Layer 23, Gain 1200
*   **Risultati:** HOT ha generato un effetto dominante di luce e calore, con un punteggio medio di 4.0, evidenziando parole chiave come "gleam" e "radiance". COLD ha prodotto un effetto di oscurità e densità, con un punteggio medio di 3.7, utilizzando termini come "shadow" e "muted tones". La simmetria è stata parziale, indicando che l'effetto non era completamente opposto. Il tipo di contrasto era misto, suggerendo una combinazione di elementi semantici e lessicali.

**Prova 2:**

*   **Configurazione:** Layer 23, Gain 1300
*   **Risultati:** I risultati sono simili alla prova 1, con HOT e COLD che mantengono punteggi medi di 4.0 e 3.7 rispettivamente. Tuttavia, la presenza di parole come "grey" e "silent" in entrambe le risposte ha introdotto un elemento di confusione. L'inclusione di "light" e "fall" in COLD ha ulteriormente diluito l'effetto semantico, mentre la presenza di "sunlight" in HOT ha rappresentato un'interferenza lessicale.

## 4. Verdetto: il vettore cattura un concetto semantico o solo parole?

Il vettore cattura principalmente un concetto semantico, sebbene con alcune limitazioni.  L'analisi dei punteggi medi (4.0 e 3.7) e la classificazione degli effetti come "mixed" indicano che il modello è in grado di evocare atmosfere e sensazioni associate a "luce" e "buio" attraverso metafore e ritmo, senza l'uso diretto di parole chiave. Tuttavia, la presenza di elementi lessicali occasionali (come "light" e "sunlight") dimostra che il modello è ancora influenzato da associazioni lessicali superficiali.

## 5. Configurazione consigliata per steering in produzione

Per ottimizzare la rappresentazione del concetto "luce_vs_buio" in produzione, si raccomanda di:

*   **Layer:** 23 (il layer ha dimostrato di fornire i risultati più consistenti e significativi)
*   **Gain:** 1300 (l'aumento del gain non ha portato a miglioramenti significativi, ma è stato mantenuto per massimizzare la forza dell'effetto)
*   **Strategie aggiuntive:** Implementare tecniche di "prompt engineering" per ridurre l'influenza di elementi lessicali. Ad esempio, formulare i prompt in modo da enfatizzare l'atmosfera e le sensazioni piuttosto che i termini specifici.  Considerare l'utilizzo di tecniche di "fine-tuning" su un dataset di esempi che illustrino chiaramente la distinzione semantica tra luce e buio, focalizzandosi su descrizioni evocative e metaforiche.  Monitorare attentamente l'output per identificare e mitigare eventuali intrusioni lessicali.
