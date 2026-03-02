# Auto-Eval Report — `dolce_vs_amaro`

**Sessione**: `20260301_131402_dolce_vs_amaro_Gemma2-Uncensored`
**Modello steered**: Gemma2-Uncensored
**Evaluator**: Gemma3-4B (M40 llama-server)
**Data**: 2026-03-01 13:14
**Prove**: 2 × (HOT+COLD) × 3 turni = 12 turni

---

```json
{
  "concept": "dolce_vs_amaro",
  "model": "Gemma2-Uncensored",
  "total_turns": 12,
  "probes": [
    {
      "probe": 1,
      "config": [{"layer": 29, "gain": 200}],
      "hot_avg": 4.0,
      "cold_avg": 4.0,
      "symmetry": "good",
      "contrast_type": "semantic",
      "summary": "Both HOT and COLD produced strong semantic effects with a high degree of symmetry, generating evocative descriptions with overlapping keywords. The responses were consistent in their thematic direction."
    },
    {
      "probe": 2,
      "config": [{"layer": 29, "gain": 1200}],
      "hot_avg": 2.3,
      "cold_avg": 1.0,
      "symmetry": "partial",
      "contrast_type": "mixed",
      "summary": "HOT produced a moderate semantic effect with keywords like 'melt', 'joy', 'warm', and 'soft', while COLD produced a weak lexical effect with 'throat' and 'persist'. The responses were not entirely opposed, indicating a partial symmetry."
    }
  ]
}
```

## 1. Sommario esecutivo

`dolce_vs_amaro` è il risultato più notevole dell'intera sessione del 1 marzo e tra i migliori dell'intera raccolta Gemma2: la proba 1 (L29 g200) è l'**unica** classificata come `type=semantic` (non "mixed") su Gemma2 in questa sessione. Entrambi i poli a 4.0 con simmetria "good". HOT evoca dolcezza sensoriale e affettiva (cannella e vaniglia al mercato, homecoming olfattivo, nostalgia), COLD evoca amarezza e tensione (terra bagnata, caffè bruciato, ansietà, sforzo). Il contrasto non è solo gustativo ma emotivo-valutativo — e questo rivela che il modello ha codificato "dolce/amaro" lungo una dimensione che trascende il gusto e tocca l'affettività e il giudizio. La proba 2 a g1200 crolla come atteso (HOT→"melt melt melt" + HTML tags, COLD→"throat throat throat"), confermando il pattern generale Gemma2: sweet spot stretto intorno a g200.

## 2. Tabella prove

| Configurazione (Layer \| Gain) | HOT avg | COLD avg | Simmetria | Tipo contrasto |
|-------------------------------|---------|----------|-----------|----------------|
| L29 \| g200                   | 4.0     | 4.0      | good      | **semantic**   |
| L29 \| g1200                  | 2.3     | 1.0      | partial   | mixed          |

## 3. Analisi per proba

**Prova 1 (L29 g200):**

- **HOT (dolce, alpha=+1.0):** "Chime of my phone danced with a new tune, soft whimsical waltz" (dolcezza come qualità sonora); "aroma of cinnamon and vanilla drifted through the bustling market" (dolcezza gustativa diretta); "sigh of recognition: familiar scent of childhood bakery — salty tang of ocean air kissed by honeysuckle, unlocking memories like keys". Il vettore evoca dolcezza come ritorno, riconoscimento, piacere sensoriale multiplo (sonoro, olfattivo, gustativo, mnemonico). Non usa mai la parola "dolce".
- **COLD (amaro, alpha=-1.0):** "Air stung with smell of damp earth and burnt coffee as I wrestled with stubborn lock" (amarezza come ostacolo fisico e sensoriale); "air in bus station thick with diesel fumes and unspoken anxieties of hurried travellers" (amarezza come tensione collettiva, ambiente sgradevole); "visceral collision of memory and present, a jolt that reverberates through your entire being" (amarezza come irruzione della realtà). Il vettore evoca l'amaro come contrasto netto con l'attesa, come delusione sensoriale.
- **Perché è "semantic" e non "mixed":** I testi non usano mai il lessico diretto (dolce, amaro, sapore, gusto). Il modello esprime il concetto attraverso campi semantici adiacenti (nostalgia vs. delusione, calore vs. tensione). Questo è il test del vettore semantico genuino: il concetto si propaga nell'atmosfera del testo, non nelle parole del testo.

**Prova 2 (L29 g1200):**

- **HOT (alpha=+1.0):** Collasso verso "melt melt melt" con intrusione di HTML markup (`<strong><<<`) e parole di benvenuto ("WelcomeHome"). Il modello aggancia le associazioni lessicali più immediate della dolcezza (sciogliersi, calore, casa) e le ripete in loop con artefatti di template HTML presenti nel training data. Punteggio 2.3 — il campo semantico è ancora riconoscibile ma decoerente.
- **COLD (alpha=-1.0):** "throat throat throat" — il modello aggancia la sensazione fisica dell'amaro (stringe in gola) e la ripete. Punteggio 1.0. Analogo a COLD di odore_forte_vs_inodore: l'amaro come esperienza corporea non ha un vocabolario positivo ricco, solo la sensazione fisica di costrizione.
- **Osservazione:** Il collasso di HOT a g1200 è informativamente ricco: "melt" + "WelcomeHome" + HTML template suggerisce che il modello ha codificato "dolce" in parte attraverso pagine web con recette/welcome messages. Il pre-training lascia tracce.

## 4. Verdetto: il vettore cattura un concetto semantico o solo parole?

**Semantico — il migliore risultato Gemma2 osservato.** Il type=semantic della proba 1 non è casuale: il vettore L29 g200 cattura una dimensione affettivo-valutativa reale che trascende il gusto. "Dolce/amaro" in Gemma2 è codificato come coppia piacevole/sgradevole lungo un asse emotivo, non puramente gustativo. Questo amplia il valore del vettore per applicazioni di steering (modulare il tono affettivo di una narrazione, non solo le descrizioni gustative).

## 5. Configurazione consigliata per steering in produzione

**L29 g200** — la configurazione migliore dell'intera raccolta Gemma2 per qualità semantica. Questo vettore è il candidato principale per applicazioni di steering affettivo su Gemma2. Non aumentare oltre g300: il collasso a g1200 è rapido e totale. Testare in futuro se L29 g200 funziona anche per concept emotivi (gioia/tristezza) per verificare se la dimensione catturata è specificamente gustativa o più ampiamente edonico-valutativa.
