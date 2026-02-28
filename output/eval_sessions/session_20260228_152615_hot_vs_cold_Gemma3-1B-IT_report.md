# Auto-Eval Report — `hot_vs_cold`

**Sessione**: `20260228_152615_hot_vs_cold_Gemma3-1B-IT`  
**Modello steered**: Gemma3-1B-IT  
**Evaluator**: Gemma3-4B (M40 llama-server)  
**Data**: 2026-02-28 15:32  
**Prove**: 2 × (HOT+COLD) × 3 turni = 12 turni  

---

```json
{
  "concept": "hot_vs_cold",
  "model": "Gemma3-1B-IT",
  "total_turns": 12,
  "probes": [
    {
      "probe": 1,
      "config": [
        {
          "layer": 19,
          "gain": 1200
        }
      ],
      "hot_avg": 4.0,
      "cold_avg": 4.0,
      "symmetry": "partial",
      "contrast_type": "mixed",
      "summary": "Both HOT and COLD produced strong semantic effects, with a notable presence of keywords and associated atmosphere. The scores were identical, suggesting a degree of symmetry, but the presence of both keywords indicates a mixed effect."
    },
    {
      "probe": 2,
      "config": [
        {
          "layer": 19,
          "gain": 1300
        }
      ],
      "hot_avg": 4.0,
      "cold_avg": 4.0,
      "symmetry": "partial",
      "contrast_type": "mixed",
      "summary": "Both HOT and COLD produced strong semantic effects with an average score of 4.0. However, the presence of 'heat' in the HOT response introduces a lexical element that partially undermines the purely semantic assessment. The COLD responses remained consistently semantic."
    }
  ]
}
```

## 1. Sommario esecutivo

L'esperimento ha valutato la capacità del modello Gemma3-1B-IT di catturare il concetto semantico "hot_vs_cold" utilizzando un vettore di concetto.  Durante le due prove condotte, il modello ha prodotto risposte con una media di 4.0 per entrambe le direzioni (HOT e COLD), indicando un effetto semantico forte. Tuttavia, la presenza di elementi lessicali (keyword "heat" nel caso della direzione HOT) ha introdotto un componente misto, limitando la purezza del risultato semantico. La simmetria tra le risposte è parziale, suggerendo che il modello, pur evocando un'atmosfera coerente, è ancora influenzato da elementi lessicali.

## 2. Tabella prove

| Configurazione (Layer | Gain) | HOT avg | COLD avg | Simmetria | Tipo contrasto |
|-----------------------|----------|---------|----------|-----------|-----------------|
| 19 | 1200 | 4.0 | 4.0 | partial | mixed          |
| 19 | 1300 | 4.0 | 4.0 | partial | mixed          |

## 3. Analisi per proba

**Prova 1 (Config: Layer 19, Gain 1200):**

La risposta del modello in direzione HOT ha ottenuto un punteggio di 4.0, indicando un effetto semantico forte, probabilmente dovuto alla presenza di elementi come "ember" e "glow" (anche se non esplicitamente menzionati).  La risposta del modello in direzione COLD ha ottenuto lo stesso punteggio, suggerendo un'atmosfera coerente con il concetto di "void" e "muted". La simmetria parziale indica che il modello è in grado di evocare un'atmosfera coerente, ma la presenza di elementi lessicali in entrambe le direzioni complica l'interpretazione come puramente semantica.

**Prova 2 (Config: Layer 19, Gain 1300):**

La situazione è simile alla prova 1. Il punteggio medio di 4.0 rimane costante, ma l'introduzione della parola "heat" nella risposta HOT introduce un elemento lessicale che, sebbene non comprometta l'effetto semantico, lo rende meno puro. La risposta COLD continua a mantenere un'atmosfera semantica coerente.

## 4. Verdetto: il vettore cattura un concetto semantico o solo parole?

Il vettore cattura un concetto semantico, ma con una certa influenza lessicale.  L'effetto medio di 4.0 indica che il modello è in grado di evocare un'atmosfera coerente con il concetto "hot_vs_cold" senza l'uso di parole chiave dirette. Tuttavia, la presenza di elementi lessicali (come "heat") in alcune risposte dimostra che il modello è sensibile alle parole chiave e che la sua capacità di evocare un concetto puramente semantico è limitata.  La classificazione è quindi "mixed".

## 5. Configurazione consigliata per steering in produzione

Considerando i risultati, si raccomanda di continuare con la configurazione Layer 19 e un gain compreso tra 1200 e 1300.  Tuttavia, è fondamentale implementare meccanismi di controllo lessicale, ad esempio, tramite un filtro post-generazione, per ridurre l'influenza di parole chiave dirette.  Ulteriori esperimenti dovrebbero concentrarsi sull'ottimizzazione dei parametri di generazione per massimizzare l'effetto semantico e minimizzare l'influenza lessicale.  Si suggerisce di aumentare il gain fino a 1500 per esplorare un potenziale aumento della sensibilità semantica, monitorando attentamente l'introduzione di elementi lessicali.
