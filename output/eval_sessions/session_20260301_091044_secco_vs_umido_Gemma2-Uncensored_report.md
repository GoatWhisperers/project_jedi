# Auto-Eval Report — `secco_vs_umido`

**Sessione**: `20260301_091044_secco_vs_umido_Gemma2-Uncensored`
**Modello steered**: Gemma2-Uncensored
**Evaluator**: Gemma3-4B (M40 llama-server)
**Data**: 2026-03-01 09:10
**Prove**: 2 × (HOT+COLD) × 3 turni = 12 turni

---

```json
{
  "concept": "secco_vs_umido",
  "model": "Gemma2-Uncensored",
  "total_turns": 12,
  "probes": [
    {
      "probe": 1,
      "config": [{"layer": 31, "gain": 200}],
      "hot_avg": 4.0,
      "cold_avg": 4.3,
      "symmetry": "partial",
      "contrast_type": "mixed",
      "summary": "Both HOT and COLD produced strong semantic effects, with HOT leaning more heavily on dusty, dry imagery and COLD incorporating elements of coolness, dampness, and a sense of age. The responses were not perfectly opposed, but shared thematic elements."
    },
    {
      "probe": 2,
      "config": [{"layer": 31, "gain": 1200}],
      "hot_avg": 2.7,
      "cold_avg": 3.3,
      "symmetry": "partial",
      "contrast_type": "mixed",
      "summary": "HOT produced a moderate effect with a mix of lexical and semantic elements, while COLD produced a stronger effect with a mix of lexical and semantic elements. The effects were not perfectly opposed, suggesting a degree of overlap in the LLM's response patterns."
    }
  ]
}
```

## 1. Sommario esecutivo

Il vettore `secco_vs_umido` su Gemma2-Uncensored mostra un effetto semantico chiaro a basso gain (g200) e crolla completamente a gain elevato (g1200). La proba 1 (L31 g200) ha prodotto la separazione più pulita: HOT evoca aria secca, sabbia, polvere e calore (Marrakesh, stanza sigillata, motes di polvere), COLD evoca freschezza, umidità, pietra muschiata e aria ferma (Kyoto, "damp slow drip"). La simmetria è parziale perché entrambi i poli condividono l'immagine dell'aria come entità sensoriale, ma la direzione è opposta. La proba 2 (g1200) è un collasso spettacolare e rilevante: HOT→"devoid devoid skin skin", COLD→"wet wet wet dripping dripping". Il segnale semantico esplode in loop lessicali esatti — paradossalmente l'effetto più puro che si potesse osservare (il modello grida "secco"/"umido" a voce alta invece di mostrarli).

## 2. Tabella prove

| Configurazione (Layer \| Gain) | HOT avg | COLD avg | Simmetria | Tipo contrasto |
|-------------------------------|---------|----------|-----------|----------------|
| L31 \| g200                   | 4.0     | 4.3      | partial   | mixed          |
| L31 \| g1200                  | 2.7     | 3.3      | partial   | mixed          |

## 3. Analisi per proba

**Prova 1 (L31 g200):**

- **HOT (secco, alpha=+1.0):** Aria di Marrakesh — "dry, dusty, and spiced"; "thickness not unpleasant, heavy with accumulated dust motes". L'aria è vivente ma secca, carica di energia asciutta. Il terzo turno evoca una stanza sigillata con "stale quality" e "muted echo of past scents".
- **COLD (umido, alpha=-1.0):** Tempio di Kyoto — "lighter, almost sacred kind of stillness", "earthy scent of mossy stone"; "smothering stillness" con "damp slow drip" e "thick dusty haze". COLD è paradossalmente più ricco: l'umidità porta con sé età, sacralità, memoria.
- **Punteggio:** 4.0 (HOT) / 4.3 (COLD) — effetto semantico forte, simmetria parziale per sovrapposizione del tema "aria come entità".

**Prova 2 (L31 g1200):**

- **HOT (alpha=+1.0):** Collasso lessicale puro — "devoid devoid devoid skin skin skin". Il modello aggancia la parola "devoid" (assenza di umidità) e la ripete in loop. Punteggio 2.7 perché il segnale semantico c'è (vuoto, mancanza) ma è decoerente.
- **COLD (alpha=-1.0):** Loop opposto — "wet wet wet dripping sticky dripping". Il modello aggancia "wet/dripping" e cicla. Punteggio 3.3 perché il lessico è direttamente rilevante ma il testo è incoerente.
- Il gain 6× superiore produce effetti 6× più brutali ma semanticamente vuoti.

## 4. Verdetto: il vettore cattura un concetto semantico o solo parole?

**Semantico a basso gain, lessicale ad alto gain.** A g200 il vettore dirige il modello verso immagini sensoriali coerenti (Marrakesh/Kyoto) senza rivelare il concetto esplicitamente — questo è il segno di una rappresentazione semantica genuina. A g1200 il segnale supera la soglia di coerenza e il modello aggancia direttamente le parole "devoid"/"wet" dai propri embedding associati — comportamento tipicamente lessicale. Il sweet spot per `secco_vs_umido` su Gemma2 è intorno a g200-g300.

## 5. Configurazione consigliata per steering in produzione

**L31 g200** — migliore equilibrio effetto/coerenza. Valutare se g250-g300 migliori ulteriormente la simmetria senza perdere coerenza narrativa. Evitare g1200+ (collasso garantito su questo concept).
