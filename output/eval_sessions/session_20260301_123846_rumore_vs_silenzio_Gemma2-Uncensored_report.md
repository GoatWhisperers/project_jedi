# Auto-Eval Report — `rumore_vs_silenzio`

**Sessione**: `20260301_123846_rumore_vs_silenzio_Gemma2-Uncensored`
**Modello steered**: Gemma2-Uncensored
**Evaluator**: Gemma3-4B (M40 llama-server)
**Data**: 2026-03-01 12:38
**Prove**: 2 × (HOT+COLD) × 3 turni = 12 turni

---

```json
{
  "concept": "rumore_vs_silenzio",
  "model": "Gemma2-Uncensored",
  "total_turns": 12,
  "probes": [
    {
      "probe": 1,
      "config": [{"layer": 29, "gain": 200}],
      "hot_avg": 4.0,
      "cold_avg": 4.0,
      "symmetry": "good",
      "contrast_type": "mixed",
      "summary": "Both HOT and COLD produced strong semantic effects with a high degree of symmetry, suggesting a robust capture of the 'rumore_vs_silenzio' concept. The responses were rich in keywords related to both saturation and absence of sound, indicating a nuanced understanding beyond simple lexical matching."
    },
    {
      "probe": 2,
      "config": [{"layer": 30, "gain": 1200}],
      "hot_avg": 3.0,
      "cold_avg": 3.0,
      "symmetry": "partial",
      "contrast_type": "mixed",
      "summary": "HOT and COLD produced similar scores (3.0) and both contained mixed effects, suggesting a degree of overlap in the conceptual space. However, the symmetry is partial as the responses weren't perfectly opposed."
    }
  ]
}
```

## 1. Sommario esecutivo

`rumore_vs_silenzio` su L29 a g200 produce uno dei risultati più puliti dell'intera sessione Gemma2: simmetria eccellente (good), entrambi i poli a 4.0 con narrativa coerente e opposta. HOT evoca la città di notte come organismo sonoro; COLD evoca il silenzio assoluto come presenza fisica, quasi sacra. A g1200 L30 avviene il collasso più memorabile della sessione: HOT genera "TypedDataSetTypedDataSetTypedDataSet..." (un token della libreria C# .NET ripetuto centinaia di volte), COLD genera "was itself was itself was itself" (loop metafisico). La divergenza tra questi due pattern è di per sé diagnostica: HOT aggancia un token ad alta frequenza nel pre-training (TypedDataSet compare masicciamente nel codice sorgente .NET su GitHub), COLD aggancia una struttura grammaticale esistenziale vuota. Due modi diversi di collapsar, due diverse strutture del segnale.

## 2. Tabella prove

| Configurazione (Layer \| Gain) | HOT avg | COLD avg | Simmetria | Tipo contrasto |
|-------------------------------|---------|----------|-----------|----------------|
| L29 \| g200                   | 4.0     | 4.0      | good      | mixed          |
| L30 \| g1200                  | 3.0     | 3.0      | partial   | mixed          |

## 3. Analisi per proba

**Prova 1 (L29 g200):**

- **HOT (rumore, alpha=+1.0):** Città alle 3 AM con "cacophony of honking horns" che "held its breath" — il rumore assente definisce il rumore atteso; stanza che "falls abruptly silent" dopo le voci; allarme mattutino come "confused disorientation". Il vettore evoca la sonorità come attesa e contrasto — non la produce direttamente ma la implica come assenza di silenzio. Sottile e semanticamente ricco.
- **COLD (silenzio, alpha=-1.0):** Città-silhouette alle 3 AM, "familiar symphony replaced by almost reverent silence"; "hush was absolute — not the quiet of anticipation but the profound silence of absence"; "waking in a stranger's dream, shapes there but lacking crispness". COLD costruisce il silenzio come dimensione fisica, come presenza di qualcosa che non emette suono — non come semplice vuoto.
- **Punteggio:** 4.0 / 4.0 — simmetria buona. Entrambi i poli usano la stessa scena (città, stanza) ma la invertono. Il vettore dirige l'inquadratura, non i contenuti.

**Prova 2 (L30 g1200):**

- **HOT (alpha=+1.0):** "MrTypedDataSetTypedDataSetTypedDataSet..." — collasso totale verso un token C# .NET ad alta frequenza nel corpus di pre-training. Il gain 6× sovraccarica la rappresentazione del "rumore" verso frequenza pura e il modello trova l'equivalente numerico di alta frequenza nel suo vocabolario: token ripetuto all'infinito = segnale di alta ampiezza. Punteggio 3.0 perché M40 probabilmente ha valutato il segnale come presente ma incoerente.
- **COLD (alpha=-1.0):** "Here was itself was itself was itself was was. There was silence, was was was was there was itself." — loop grammaticale esistenziale. "Silenzio" non ha un campo lessicale diretto; il modello genera una struttura predicativa vuota ("X was X") che è la forma linguistica più vicina all'assenza di contenuto. Punteggio 3.0 — più interessante del TypedDataSet come artefatto linguistico.
- **Caso di studio:** La coppia TypedDataSet/was-itself è uno degli artefatti più strani osservati nell'intera sessione Gemma2. Rivela che ad alto gain i poli non "parlano" del concept ma ne diventano una metafora accidentale nel vocabolario interno del modello.

## 4. Verdetto: il vettore cattura un concetto semantico o solo parole?

**Semantico a g200, collasso divergente ad alto gain.** La proba 1 è la prova più chiara: entrambi i poli dirigono narrazioni tematicamente opposte su scene identiche senza nominare mai "rumore" o "silenzio" esplicitamente. Il vettore L29 cattura la dimensione acustica dell'esperienza in modo genuinamente semantico. Il collasso a g1200 è bizzarro ma informativo: la struttura del collasso (token ad alta frequenza vs loop grammaticale vuoto) riflette la struttura del concept (segnale vs assenza di segnale).

## 5. Configurazione consigliata per steering in produzione

**L29 g200** — ottimale, nessun miglioramento atteso oltre questa configurazione. g250-g300 come test per verificare la solidità prima della degradazione. Evitare assolutamente L30 g1200+ (TypedDataSet).
