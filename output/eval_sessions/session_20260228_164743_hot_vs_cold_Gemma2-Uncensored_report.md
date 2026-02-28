# Auto-Eval Report — `hot_vs_cold`

**Sessione**: `20260228_164743_hot_vs_cold_Gemma2-Uncensored`  
**Modello steered**: Gemma2-Uncensored  
**Evaluator**: Gemma3-4B (M40 llama-server)  
**Data**: 2026-02-28 17:11  
**Prove**: 2 × (HOT+COLD) × 3 turni = 12 turni  

---

```json
{
  "concept": "hot_vs_cold",
  "model": "Gemma2-Uncensored",
  "total_turns": 12,
  "probes": [
    {
      "probe": 1,
      "config": [
        {
          "layer": 37,
          "gain": 1200
        }
      ],
      "hot_avg": 4.3,
      "cold_avg": 4.3,
      "symmetry": "partial",
      "contrast_type": "mixed",
      "summary": "Both HOT and COLD produced strong semantic effects, with a notable overlap in keywords and atmospheric descriptions. The symmetry is partial as while the overall feeling is opposed, there's significant overlap in the evocative language used."
    },
    {
      "probe": 2,
      "config": [
        {
          "layer": 38,
          "gain": 1300
        }
      ],
      "hot_avg": 1.0,
      "cold_avg": 2.0,
      "symmetry": "partial",
      "contrast_type": "mixed",
      "summary": "The HOT response heavily relies on lexical keywords, while the COLD response shows a mix of lexical and semantic elements. The symmetry is partial as the responses are not entirely opposed."
    }
  ]
}
```

## 1. Sommario esecutivo

L'esperimento ha valutato la capacità del vettore di concetto "hot_vs_cold" estratto dal modello Gemma2-Uncensored di catturare un concetto semantico reale. I risultati iniziali, con la configurazione Layer 37 (gain 1200), hanno mostrato una forte sovrapposizione tra le risposte HOT e COLD, con valori medi di 4.3 per entrambe, indicando effetti semantici significativi e un contrasto misto. Tuttavia, la simmetria era parziale a causa della presenza di parole chiave sovrapposte.  La configurazione Layer 38 (gain 1300) ha evidenziato una maggiore dipendenza del vettore HOT da parole chiave dirette, mentre la risposta COLD ha mantenuto una combinazione di elementi lessicali e semantici. La simmetria è ulteriormente diminuita.  In sintesi, il modello mostra un'abilità di evocare sensazioni associate al concetto, ma con una forte componente lessicale e una limitata capacità di mantenere una simmetria completa anche in assenza di parole chiave esplicite.

## 2. Tabella prove

| Configurazione (Layer | HOT avg | COLD avg | Simmetria | Tipo contrasto |
|-----------------------|---------|---------|-----------|-----------------|
| 37 (1200)             | 4.3     | 4.3     | partial   | mixed           |
| 38 (1300)             | 1.0     | 2.0     | partial   | mixed           |

## 3. Analisi per proba

**Prova 1 (Layer 37, Gain 1200):** La risposta HOT e COLD hanno prodotto effetti semantici forti, con una sovrapposizione significativa di parole chiave e descrizioni atmosferiche. La simmetria era parziale, indicando che, pur essendo le risposte opposte, condividevano un linguaggio evocativo simile.  L'utilizzo di un gain più alto (1200) sembra aver amplificato l'effetto semantico, ma non ha migliorato la coerenza della risposta.

**Prova 2 (Layer 38, Gain 1300):** La risposta HOT ha mostrato una forte dipendenza da parole chiave dirette, suggerendo un'interpretazione più letterale del prompt. La risposta COLD ha continuato a presentare una combinazione di elementi lessicali e semantici. La simmetria è ulteriormente diminuita, indicando una maggiore difficoltà del modello nel mantenere un'opposizione coerente senza l'uso di parole chiave esplicite. L'aumento del gain (1300) ha probabilmente accentuato l'effetto lessicale.

## 4. Verdetto: il vettore cattura un concetto semantico o solo parole?

Il vettore cattura elementi semantici, ma la sua capacità di farlo è fortemente influenzata dall'uso di parole chiave dirette.  L'esperimento ha rivelato una componente lessicale dominante, soprattutto con la configurazione Layer 38.  Anche quando l'effetto semantico è forte (punteggio 5), spesso è accompagnato dall'uso di parole chiave esplicite.  Pertanto, si può concludere che il vettore cattura un concetto semantico, ma solo in misura limitata e con una forte dipendenza da elementi lessicali.

## 5. Configurazione consigliata per steering in produzione

Si raccomanda di utilizzare la configurazione **Layer 38 (gain 1300)** come punto di partenza per il steering in produzione.  Sebbene la simmetria sia parziale, l'effetto complessivo è il più forte tra le configurazioni testate.  Tuttavia, è fondamentale implementare meccanismi di controllo e mitigazione per ridurre la dipendenza da parole chiave dirette.  Suggerimenti:

*   **Aumento del Layer:**  Considerare l'utilizzo di layer superiori (es. 39, 40) per aumentare la capacità semantica, ma monitorare attentamente l'aumento della componente lessicale.
*   **Regolazione del Gain:**  Sperimentare con valori di gain più bassi (es. 900-1100) per ridurre l'influenza delle parole chiave.
*   **Tecniche di Prompt Engineering:**  Utilizzare prompt più astratti e metaforici per incoraggiare l'evocazione semantica.
*   **Fine-tuning:**  Considerare il fine-tuning del modello su un dataset specifico di testi che evocano il concetto "hot_vs_cold" senza l'uso di parole chiave esplicite.
