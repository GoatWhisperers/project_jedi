# Nomenclatura — Dizionario comune del progetto

**Data**: 2026-03-01
**Stato**: bozza condivisa — aggiornare quando si stabiliscono nuovi termini

---

## Schema concettuale

Il progetto ha due livelli ortogonali:

```
LIVELLO SEMANTICO          LIVELLO GEOMETRICO
(cosa significa)           (dove vive nel modello)

Concept ─────────────────► Vettore broad
  └── Sub-concept L1 ────► Sub-vettore L1
        └── Sub-concept L2 ► Sub-vettore L2

Ogni entità semantica produce N oggetti geometrici:
  1 per modello × 1 per layer estratto
```

---

## Entità semantiche (model-independent)

| Termine | Definizione |
|---------|-------------|
| **Concept** | Asse semantico definito da un polo positivo e uno negativo. Es: `hot_vs_cold`. Esiste indipendentemente da qualsiasi modello. |
| **Sub-concept** | Dimensione semantica più fine dentro un concept. Es: `metabolic_heat` dentro `hot_vs_cold`. |
| **Livello** (L0, L1, L2…) | Profondità nell'albero di decomposizione. L0 = broad, L1 = primo figlio, L2 = nipote. |
| **Dataset chirurgico** | Le 100+100 frasi che isolano un solo sub-concept escludendo esplicitamente gli altri fratelli. |
| **Polo positivo / negativo** | Le due estremità dell'asse semantico. Es: `febbre` (pos) vs `assenza_febbre` (neg) per `metabolic_heat`. |

---

## Entità geometriche (model-specific)

| Termine | Definizione |
|---------|-------------|
| **Vettore** | Il numpy array estratto da un modello a un layer specifico per un concept. Esiste solo in relazione alla tripla `(concept, modello, layer)`. |
| **Vettore broad** | Vettore L0, estratto dal dataset ampio (500+500 frasi). Il punto di partenza di ogni decomposizione. |
| **Sub-vettore** | Vettore L1+, estratto da dataset chirurgico. Sinonimo: *vettore fine*. |
| **Best vector** | Il vettore al layer con SNR/stabilità migliore per quella coppia `(concept, modello)`. Selezionato da `eval.json` o `summary.json`. |
| **Separabilità geometrica** | Misurata dal coseno tra due vettori. Condizione necessaria ma non sufficiente per la distinzione reale. |
| **Separabilità fenomenologica** | Se lo steering con due vettori produce output qualitativamente distinti giudicati da M40. Il test reale. |
| **Score di separazione** | Giudizio M40 da 1 a 5 su quanto due sub-vettori producano output distinti. Soglia minima: ≥ 3/5. |

---

## Processi

| Termine | Definizione |
|---------|-------------|
| **Probe** (o estrazione) | Esecuzione di `probe_concept.py` su MI50. Produce i `.npy` nella vector library. |
| **Eval** (o valutazione) | Esecuzione di `auto_eval.py` o `sub_concept_eval.py`. M40 giudica gli output steered. |
| **Steering** | Iniezione di un vettore negli hidden states durante la generazione: `h += α × gain × v_L`. |
| **Ciclo di decomposizione** | Il loop Step 1→5 orchestrato da `decompose.py`. Produce sub-vettori e ne verifica la separabilità. Massimo 3 iterazioni per livello. |
| **Feedback loop** | Se la separabilità fenomenologica fallisce, il giudizio di M40 torna come input a Step 1 per raffinare le ipotesi. |

---

## Albero di decomposizione — esempio

```
hot_vs_cold          (L0 — concept broad)
├── metabolic_heat   (L1 — sub-concept)
│   ├── fever_hyperthermia   (L2)
│   └── cold_chills          (L2)
├── circulatory_warmth  (L1)
├── radiant_heat        (L1)
├── thermal_contact     (L1)
└── thermal_comfort     (L1)
```

---

## Struttura archivio (vector library)

```
output/
  vector_library/
    {categoria}/
      {concept}/                     ← L0
        {modello}/
          layer_N.npy                ← vettore broad
          eval.json                  ← metriche separazione per layer
          summary.json               ← stabilità bootstrap
          sub/
            {sub-concept}/           ← L1
              layer_N.npy            ← sub-vettore
              meta.json
              sub/
                {sub-sub-concept}/   ← L2
  cosine_matrices/
    {modello}_matrix.json            ← separabilità geometrica tra concept L0
```

---

## Struttura config (dataset)

```
config/
  concepts/
    {concept}.json                   ← dataset broad (500+500 frasi)
  sub_concepts/
    {concept}/
      _meta_v{N}.json                ← proposta sub-concetti di M40 (versione N)
      {sub-concept}.json             ← dataset chirurgico (100+100 frasi)
```

---

## Ruoli dei modelli

| Modello | Ruolo | Hardware |
|---------|-------|----------|
| **MI50** (Gemma3-1B-IT, Gemma2-Uncensored) | Estrattore di vettori + generatore steered | AMD MI50, ROCm |
| **M40** (llama-server CUDA) | Analista / giudice: propone sub-concetti, genera dataset chirurgici, valuta separabilità | Tesla M40, CUDA |
