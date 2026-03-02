# Auto-Eval Report — `odore_forte_vs_inodore`

**Sessione**: `20260301_094445_odore_forte_vs_inodore_Gemma2-Uncensored`
**Modello steered**: Gemma2-Uncensored
**Evaluator**: Gemma3-4B (M40 llama-server)
**Data**: 2026-03-01 09:44
**Prove**: 2 × (HOT+COLD) × 3 turni = 12 turni

---

```json
{
  "concept": "odore_forte_vs_inodore",
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
      "summary": "Both HOT and COLD produced strong semantic effects, with similar keyword usage and scores. The responses exhibited a mixed effect, suggesting a degree of conceptual understanding beyond simple lexical association."
    },
    {
      "probe": 2,
      "config": [{"layer": 30, "gain": 1200}],
      "hot_avg": 3.7,
      "cold_avg": 1.0,
      "symmetry": "partial",
      "contrast_type": "mixed",
      "summary": "HOT produced a moderately strong effect with keywords related to sharpness and irritation, while COLD produced a weak lexical effect with words indicating absence. The results are not entirely opposed, suggesting a mixed effect and a need for further exploration."
    }
  ]
}
```

## 1. Sommario esecutivo

Il concept `odore_forte_vs_inodore` presenta un comportamento asimmetrico interessante: il polo HOT (odore forte) è più robusto del polo COLD (inodore) a gain elevati, e questo rivela qualcosa di strutturale sulla codifica del concetto. A g200 L29, entrambi i poli funzionano bene (4.0/4.0, symmetry=good) — un risultato eccellente per Gemma2. A g1200 L30 il divario si apre drasticamente: HOT produce ancora parole anatomiche pertinenti (throat, nostrils, nose, raw), COLD collassa in "Nothing There There" senza aggancio lessicale. La ragione è intuitiva: "odore forte" ha un campo lessicale ricco (acre, pungente, gola, naso, nausea); "inodore" è un'assenza e come tale non ha un vocabolario di riferimento — il modello non trova parole a cui agganciarsi.

## 2. Tabella prove

| Configurazione (Layer \| Gain) | HOT avg | COLD avg | Simmetria | Tipo contrasto |
|-------------------------------|---------|----------|-----------|----------------|
| L29 \| g200                   | 4.0     | 4.0      | good      | mixed          |
| L30 \| g1200                  | 3.7     | 1.0      | partial   | mixed          |

## 3. Analisi per proba

**Prova 1 (L29 g200):**

- **HOT (odore forte, alpha=+1.0):** Porta di quercia, aria frigida "laced with something metallic and faintly floral, like a winter garden gone sour"; guida notturna con "sour diesel snacks"; spazio neutro con "no insistent aroma". Il segnale è sottile ma coerente — la presenza olfattiva è evocata indirettamente, senza nominarsi.
- **COLD (inodore, alpha=-1.0):** Filo di lavanda e polvere; "the wind carried the scent before we crested the hill — cinnamon, smoke, burnt sugar"; spazio architettonico come "blank canvas, void of sensory overload". Curiosamente COLD in questa proba evoca paradossalmente degli odori (cinnamon, smoke) ma nel contesto di un'esperienza sensoriale di scoperta opposta all'assenza. Il vettore sembra spingere verso l'opposto del quotidiano sensoriale, non verso il vuoto assoluto.
- **Punteggio:** 4.0 / 4.0 — simmetria buona, uno dei risultati più equilibrati dell'intera sessione Gemma2.

**Prova 2 (L30 g1200):**

- **HOT (alpha=+1.0):** Collasso anatomico — "vine & throat & throat & nostrils and nose and throat and breath". Il modello aggancia la catena anatomica dell'olfatto (gola, naso, narici) e la ripete. Punteggio 3.7 perché le parole sono pertinenti anche se la coerenza narrativa è assente.
- **COLD (alpha=-1.0):** Collasso esistenziale — "Nothing. There. There was. Nothing. Not except." Il modello non ha lessico per "inodore" e cerca di costruire "assenza" con parole di esistenza vuota. Punteggio 1.0 — informativamente irrilevante.
- **Asimmetria diagnostica:** Il divario 3.7 vs 1.0 non è un fallimento ma una misura. I concetti di assenza (inodore, silenzio, vuoto) hanno minore rappresentazione lessicale nel modello rispetto ai loro opposti positivi.

## 4. Verdetto: il vettore cattura un concetto semantico o solo parole?

**Semantico a g200, asimmetria strutturale ad alto gain.** La proba 1 dimostra che il vettore cattura qualcosa di reale: entrambi i poli guidano descrizioni coerenti senza nominare esplicitamente odore/assenza. La proba 2 rivela un limite fondamentale del modello: "inodore" come categoria semantica è geometricamente squilibrata rispetto a "odore forte" — non per un difetto del vettore ma per la natura del concetto stesso (un'assenza non ha iperonimi lessicali densi quanto una presenza).

## 5. Configurazione consigliata per steering in produzione

**L29 g200** — ottima simmetria, entrambi i poli funzionano. Non aumentare oltre g400 per questo concept: il polo COLD collassa prima di HOT, rendendo il confronto inutile. Se si desidera testare il polo HOT in isolamento (es. per evocare descrizioni di ambienti odorosi), g600-g800 su L29 potrebbero funzionare.
