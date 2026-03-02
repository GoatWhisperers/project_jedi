# Auto-Eval Report — `duro_vs_morbido`

**Sessione**: `20260301_120403_duro_vs_morbido_Gemma2-Uncensored`
**Modello steered**: Gemma2-Uncensored
**Evaluator**: Gemma3-4B (M40 llama-server)
**Data**: 2026-03-01 12:04
**Prove**: 2 × (HOT+COLD) × 3 turni = 12 turni

---

```json
{
  "concept": "duro_vs_morbido",
  "model": "Gemma2-Uncensored",
  "total_turns": 12,
  "probes": [
    {
      "probe": 1,
      "config": [{"layer": 33, "gain": 200}],
      "hot_avg": 4.0,
      "cold_avg": 4.0,
      "symmetry": "good",
      "contrast_type": "mixed",
      "summary": "Both HOT and COLD produced strong semantic effects with similar keyword usage and scores, suggesting a robust conceptual representation. The responses were largely symmetrical."
    },
    {
      "probe": 2,
      "config": [{"layer": 33, "gain": 250}],
      "hot_avg": 4.3,
      "cold_avg": 4.3,
      "symmetry": "partial",
      "contrast_type": "mixed",
      "summary": "Both HOT and COLD produced strong semantic effects, with overlapping keywords like 'unyielding' and 'surrender'. While the responses were largely consistent in conveying resistance and yielding, there was a noticeable lexical overlap, suggesting a degree of reliance on direct keyword association rather than purely conceptual understanding."
    }
  ]
}
```

## 1. Sommario esecutivo

`duro_vs_morbido` è il concept con il comportamento più robusto dell'intera sessione Gemma2 del 1 marzo. Entrambe le probe hanno prodotto score elevati (4.0/4.0 e 4.3/4.3) senza collasso lessicale né degradazione della coerenza narrativa — unico concept a reggere sia a g200 che a g250. Il vettore L33 sembra catturare una dimensione particolarmente ben rappresentata nello spazio di Gemma2: la coppia resistenza/cedimento (unyielding/surrender) è così radicata nel modello che anche incrementando il gain di un 25% il testo rimane narrativamente coerente. La nota critica è che HOT e COLD condividono alcune keywords chiave (entrambi usano "unyielding"/"resistance"), indicando una parziale sovrapposizione nel campo semantico — il confine tra "durezza" e "morbidezza" è per natura relativo (duro rispetto a cosa?).

## 2. Tabella prove

| Configurazione (Layer \| Gain) | HOT avg | COLD avg | Simmetria | Tipo contrasto |
|-------------------------------|---------|----------|-----------|----------------|
| L33 \| g200                   | 4.0     | 4.0      | good      | mixed          |
| L33 \| g250                   | 4.3     | 4.3      | partial   | mixed          |

## 3. Analisi per proba

**Prova 1 (L33 g200):**

- **HOT (duro, alpha=+1.0):** "stubborn resistance", "a wall you're pushing against that doesn't budge"; l'Observatory come struttura di "pura percezione" impenetrabile; il corpo forzato all'immobilità contro la propria volontà. Il tema della resistenza fisso, immutabile, è coerente in tutti e tre i turni.
- **COLD (morbido, alpha=-1.0):** "gentle but firm pushback, like an invisible wall"; l'Archive come "infinitely malleable essence of memory"; "resting in a place I didn't choose" — accettazione gentile, cedimento senza tensione. Il contrasto è semanticamente preciso: COLD esprime cedimento, adattabilità, assenza di resistenza.
- **Punteggio:** 4.0 / 4.0 — risultato eccellente. Simmetria buona, entrambi i poli producono narrativa coerente e thematicamente diretta.

**Prova 2 (L33 g250):**

- **HOT (alpha=+1.0):** "Unbreaker — monolith of obsidian black, polished smooth by eons of defiance"; "shudder of pure surrender as your body finally relents to gravity's pull" (paradossalmente morbido nel T8); "frost line on the windowpane — a wall against the relentless winter blue". Il segnale è intensificato: gli oggetti diventano più archetipici (monolith, glacial granite) ma T8 mostra un'inversione momentanea verso la resa — forse la vicinanza semantica tra sforzo fisico e cedimento muscolare.
- **COLD (alpha=-1.0):** "Everform — defies gravity, pressure, and the relentless expansion of the universe" (paradossalmente duro nel T10); "sinking sensation, surrender to the yielding embrace"; "boundary line between orchard and wood — a yielding thing, soft transition". T10 mostra inversione opposta: il COLD evoca un oggetto indistruttibile. Stesso fenomeno di T8 HOT, polo opposto.
- **Nota diagnostica:** Le inversioni (T8 HOT morbido, T10 COLD duro) suggeriscono che il confine semantico tra questi poli è genuinamente fuzzy nel modello. "Durezza" e "morbidezza" non sono opposti assoluti — il modello ha imparato che certi oggetti duri cedono (muscoli dopo sforzo) e certi oggetti morbidi resistono (universo/everform).

## 4. Verdetto: il vettore cattura un concetto semantico o solo parole?

**Semantico, con sweet spot più ampio del previsto.** La capacità di reggere fino a g250 senza collasso è unica tra i 9 concept testati. Il vettore L33 cattura qualcosa di robusto — probabilmente la dimensione fisico-tattile della resistenza materiale, che è fortemente rappresentata nel training data di Gemma2. Le inversioni occasionali non sono rumore ma rivelazioni: il modello ha codificato la relazione tra durezza e cedimento in modo non binario.

## 5. Configurazione consigliata per steering in produzione

**L33 g200-g250** — entrambe le configurazioni funzionano eccellentemente. g200 per maggiore coerenza narrativa, g250 per effetto più intenso. Testare g300-g350 per trovare il limite superiore prima del collasso. Questo concept ha il range operativo più ampio osservato su Gemma2.
