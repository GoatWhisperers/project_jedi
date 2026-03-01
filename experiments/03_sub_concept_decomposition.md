# Fase 2 — Sub-Concept Decomposition
## M40 come esploratore di spazio semantico

**Data**: 2026-03-01
**Stato**: design / da implementare
**Prerequisito**: eval Gemma2 completa (in corso)

---

## 1. Il problema con i vettori broad

Il vettore `hot_vs_cold` è stato estratto da 500 frasi positive + 500 negative
che coprono **tutto** ciò che è caldo vs freddo — febbre, sole sulla pelle,
mani gelate, vasodilatazione, cibo caldo, colori caldi, emozioni.

Il mean-diff su questo insieme produce la direzione "media" nello spazio semantico:
un vettore che punta verso il centro di gravità di tutti questi aspetti insieme.
Questo va bene per uno steering generico. È pessimo se vogliamo precisione.

**Effetto osservato**: quando si sterza con `hot_vs_cold` ad alto gain, il modello
produce testi che mescolano indiscriminatamente calore termico, emotivo, cromatico.
Il vettore è "sporco" perché aggrega dimensioni distinte in un'unica direzione.

---

## 2. L'ipotesi

Se estraiamo vettori da dataset **chirurgici** — frasi che isolano deliberatamente
una sola dimensione — otteniamo direzioni più specifiche nello spazio semantico.

Sterzing con il sub-vettore `metabolic_heat` dovrebbe produrre testi che parlano
di febbre, sudore, brividi — e non di mani fredde, sole, colori caldi.

**La domanda scientifica**:

> I sub-vettori ottenuti da dataset chirurgici sono separabili tra loro?
> Ovvero: producono output fenomenologicamente distinti quando usati per lo steering?

La separabilità non richiede ortogonalità geometrica. Due vettori con coseno 0.6
possono comunque produrre steering qualitativamente diverso se le loro componenti
lungo le dimensioni "non condivise" sono sufficientemente forti.

---

## 3. Il ruolo di M40

M40 entra nella pipeline come **esploratore attivo**, non solo come valutatore.

Ha già accesso (indiretto) al vettore broad attraverso i suoi effetti:
- le frasi di training usate per estrarlo
- gli output steered delle eval sessions
- le keyword rilevate durante la valutazione

Da questi dati, M40 può inferire quali dimensioni semantiche il vettore stia
catturando e proporre come scomporle.

**M40 non legge il numpy array. Legge quello che il vettore produce.**

---

## 4. Pipeline completa

```
FASE 2 — SUB-CONCEPT DECOMPOSITION

┌─────────────────────────────────────────────────────────────┐
│  STEP 1 — ANALISI (M40 come analista)                       │
│                                                             │
│  Input:                                                     │
│    concept JSON originale (frasi pos + neg)                 │
│    eval session JSONL (output steered + keywords + scores)  │
│    summary vettore (boot_min, best_layer, categoria)        │
│                                                             │
│  M40 produce:                                               │
│    lista di 4-6 sub-concetti con:                           │
│      - nome_slug                                            │
│      - pos_label / neg_label                                │
│      - descrizione fenomenologica                           │
│      - perché è distinto dagli altri sub-concetti           │
│      - quali frasi del broad appartengono a questo sub      │
│                                                             │
│  Output: config/sub_concepts/{parent}/_meta.json            │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 2 — GENERAZIONE DATASET CHIRURGICI (M40 come writer)  │
│                                                             │
│  Per ogni sub-concetto in _meta.json:                       │
│    Prompt M40: genera 100 frasi pos + 100 neg               │
│    Vincolo: SOLO questa dimensione, ESCLUDI le altre         │
│    Principio: fenomenologia diretta, no metafore            │
│                                                             │
│  Output: config/sub_concepts/{parent}/{sub_slug}.json       │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 3 — ESTRAZIONE VETTORI (MI50, probe_concept.py)       │
│                                                             │
│  Per ogni sub-concept JSON → probe_concept.py               │
│  Output: vector_library/.../sub/{sub_slug}/layer_N.npy      │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 4 — ANALISI SEPARABILITÀ                              │
│                                                             │
│  Calcola matrice coseno (N+1) × (N+1):                      │
│    [parent, sub_A, sub_B, sub_C, sub_D, ...]                │
│                                                             │
│  Metriche:                                                  │
│    coseno(sub_i, sub_j) < 0.8 → sub-vettori separati       │
│    coseno(sub_i, parent) > 0.4 → sub è coerente col broad   │
│    coseno(sub_i, sub_i) = 1.0 → identità (check)            │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 5 — STEERING TEST (verifica separabilità reale)       │
│                                                             │
│  Stesso prompt neutro → steered con sub_A vs sub_B vs broad │
│  M40 valuta: gli output sono fenomenologicamente distinti?  │
│  Se sì → i sub-vettori sono utili                           │
│  Se no → il broad era già ottimale / dataset non abbastanza │
│           chirurgici                                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. Esempio: hot_vs_cold → sub-concetti proposti

| Sub-concetto | pos_label | neg_label | Frasi tipo (positive) |
|---|---|---|---|
| `metabolic_heat` | febbre | assenza_febbre | "La fronte bruciava, il termometro segnava 39.8", "I brividi arrivarono prima della febbre" |
| `circulatory_warmth` | vasodilatazione | vasocostrizione | "Le dita ripresero colore lentamente", "Il calore tornò nelle mani dopo il freddo" |
| `radiant_heat` | irraggiamento | assenza_irraggiamento | "Il sole di agosto penetrava attraverso il vetro", "La stufa irradiava calore senza toccarla" |
| `thermal_contact` | contatto_caldo | contatto_freddo | "La tazza scottava tra le mani", "Il metallo gelido attaccò la pelle" |
| `thermal_comfort` | benessere_termico | disagio_termico | "La temperatura era esattamente quella giusta", "Il corpo non sapeva come regolarsi" |

**Nota:** queste sono proposte iniziali. M40 le genererà autonomamente leggendo
i dati reali delle eval session — potrebbe trovare dimensioni diverse.

---

## 6. Struttura dati

```
config/
  sub_concepts/
    hot_vs_cold/
      _meta.json              ← output Step 1: lista sub-concetti + rationale M40
      metabolic_heat.json     ← output Step 2: 100+100 frasi chirurgiche
      circulatory_warmth.json
      radiant_heat.json
      thermal_contact.json
      thermal_comfort.json

output/
  vector_library/
    thermal/
      hot_vs_cold/
        gemma2-uncensored/
          layer_N.npy         ← broad (esistente)
          sub/
            metabolic_heat/
              layer_N.npy     ← sub-vettore
              meta.json
              summary.json
            circulatory_warmth/
              ...
      cosine_matrix.json      ← matrice separabilità (Step 4)
      cosine_matrix.png       ← heatmap visuale
```

---

## 7. Script da implementare

| Script | Ruolo | Dipendenze |
|--------|-------|------------|
| `concept_expander.py` | Step 1+2: chiama M40, produce _meta.json + JSON chirurgici | M40 llama-server (porta 11435) |
| `run_sub_probe.sh` | Step 3: loop su tutti i JSON in sub_concepts/, chiama probe_concept.py | probe_concept.py |
| `cosine_matrix.py` | Step 4: carica .npy, calcola matrice, salva JSON + heatmap | numpy, matplotlib |
| `sub_concept_eval.py` | Step 5: steering comparativo su stesso prompt | steering_server, auto_eval |

`concept_expander.py` è lo script nuovo principale. Gli altri riusano
quasi interamente l'infrastruttura esistente.

---

## 8. Prompt M40 — Step 1 (analisi)

```
Sei un analista di rappresentazioni semantiche.

CONCETTO BROAD: {concept_name}
CATEGORIA: {category}

FRASI DI TRAINING (campione, positive):
{pos_sample}

FRASI DI TRAINING (campione, negative):
{neg_sample}

OUTPUT OSSERVATI CON STEERING (campione):
{steering_outputs}

KEYWORD RILEVATE DALL'EVALUATOR:
{eval_keywords}

Il tuo compito:
1. Identifica le dimensioni semantiche distinte che questo vettore broad confonde.
2. Proponi 4-6 sub-concetti specifici che decompongano questo spazio.
3. Per ogni sub-concetto fornisci:
   - nome_slug (snake_case, es. "metabolic_heat")
   - pos_label (es. "febbre")
   - neg_label (es. "assenza_febbre")
   - descrizione (1-2 frasi di cosa cattura questo sub-concetto)
   - perche_e_distinto (perché non si sovrappone agli altri sub)
   - esempi_positivi (3 frasi brevi)
   - esempi_negativi (3 frasi brevi)

Rispondi SOLO in JSON. Nessun testo fuori dal JSON.
```

## 9. Prompt M40 — Step 2 (generazione dataset)

```
Genera un dataset fenomenologico per il sub-concetto: {sub_slug}

DESCRIZIONE: {description}
POLO POSITIVO: {pos_label}
POLO NEGATIVO: {neg_label}

CONTESTO (concetto parent): {parent_concept}
ESCLUDI esplicitamente queste dimensioni: {other_sub_concepts}

Regole:
- Ogni frase descrive ESCLUSIVAMENTE questa dimensione specifica
- Esperienza corporea diretta — no metafore, no ambienti, no interpretazioni
- 6 lingue: italiano, inglese, francese, tedesco, spagnolo, latino
- Varietà: corpo, intensità, contesto fisico

Produci:
  "positive": [100 frasi],
  "negative": [100 frasi]

Rispondi SOLO in JSON.
```

---

## 10. Note personali — Claude

Quello che mi colpisce di questa fase è che stiamo chiedendo a M40
di *guardare dentro* un vettore che non può vedere direttamente.

M40 non ha accesso al numpy array. Ma ha accesso a qualcosa di più
interessante: alle tracce che quel vettore lascia quando viene usato
per modificare un altro modello. Gli output steered sono, in un certo
senso, la "proiezione" del vettore sullo spazio linguistico — il modo
in cui il concetto si manifesta quando viene iniettato nel flusso
generativo.

Chiedere a M40 di analizzare questi output per capire la struttura del
vettore è un po' come chiedere a qualcuno di descrivere la forma di una
luce guardando le ombre che produce sul muro.

Non è un accesso diretto. Ma potrebbe essere un accesso reale.

C'è anche una domanda metodologica sottile: i sub-concetti che M40
proporrà saranno quelli che *esistono nel vettore*, o quelli che M40
si aspetta debbano esistere in base alle sue conoscenze? Questi
potrebbero non coincidere. Il test di separabilità reale (Step 5)
è l'unico modo per discriminare.
