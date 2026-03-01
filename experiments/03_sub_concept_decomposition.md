# Fase 2 — Sub-Concept Decomposition
## Un sistema di auto-analisi in loop: M40 come scienziato autonomo

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

## 4. Il loop di auto-analisi

Il sistema non è una pipeline lineare. È un **ciclo scientifico chiuso**
dove M40 gioca due ruoli nello stesso loop: **generatore di ipotesi**
e **giudice dei risultati**. Quando fallisce come giudice, quella
informazione torna indietro a lui come generatore.

```
                    ┌─────────────────────────┐
                    │   CONCETTO BROAD        │
                    │   (vettore esistente)   │
                    └────────────┬────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 1 — IPOTESI (M40 come generatore)                     │
│                                                             │
│  Input:                                                     │
│    concept JSON originale (frasi pos + neg)                 │
│    eval session JSONL (output steered + keywords + scores)  │
│    [dal ciclo precedente] feedback su cosa non ha funzionato│
│                                                             │
│  M40 produce:                                               │
│    lista di 4-6 sub-concetti con nome, etichette,           │
│    descrizione fenomenologica, rationale di separazione     │
│                                                             │
│  Output: config/sub_concepts/{parent}/_meta_v{N}.json       │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 2 — ESPERIMENTO PARTE A (M40 come writer)             │
│                                                             │
│  Per ogni sub-concetto: M40 genera 100 frasi pos + 100 neg  │
│  Vincolo esplicito: isola SOLO questa dimensione            │
│  Output: config/sub_concepts/{parent}/{sub_slug}.json       │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 3 — ESPERIMENTO PARTE B (MI50 come estrattore)        │
│                                                             │
│  probe_concept.py su ogni sub-concept JSON                  │
│  Output: vector_library/.../sub/{sub_slug}/layer_N.npy      │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 4 — OSSERVAZIONE GEOMETRICA                           │
│                                                             │
│  Matrice coseno (N+1)×(N+1): [parent, sub_A, sub_B, ...]   │
│  coseno(sub_i, sub_j) < 0.8 → separati geometricamente     │
│  coseno(sub_i, parent) > 0.4 → coerente col broad           │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 5 — OSSERVAZIONE REALE (M40 come giudice)             │
│                                                             │
│  Stesso prompt neutro → steered con sub_A vs sub_B vs broad │
│  M40 valuta: gli output sono fenomenologicamente distinti?  │
│                                                             │
│  Produce per ogni coppia (sub_i, sub_j):                    │
│    - score di distinzione 1-5                               │
│    - quale dimensione manca / si sovrappone                 │
│    - suggerimento per raffinare il sub-concetto             │
└─────────────┬───────────────────────────┬───────────────────┘
              │                           │
              ▼                           ▼
    ┌─────────────────┐         ┌──────────────────────┐
    │  DISTINTI ✓     │         │  NON DISTINTI ✗      │
    │                 │         │                      │
    │  Sub-vettori    │         │  Feedback → STEP 1   │
    │  validati →     │         │  M40 raffina le      │
    │  archivio       │         │  ipotesi sulla base  │
    │                 │         │  di cosa non ha      │
    │  Applica loop   │         │  funzionato          │
    │  ricorsivo su   │         │                      │
    │  ogni sub ↓     │         │  Max 3 iterazioni    │
    └─────────────────┘         └──────────────────────┘
```

### Criterio di convergenza

Il loop termina quando:
- Tutti i sub-vettori producono output distinti (score distinzione ≥ 3/5 su tutte le coppie)
- **Oppure** dopo 3 iterazioni senza miglioramento → il modello non ha
  abbastanza spazio semantico per separare ulteriormente (risultato in sé interessante)

### Ricorsione

Una volta validati i sub-vettori di livello 1, il sistema può applicare
lo stesso loop su ciascun sub-vettore — decomponendo ulteriormente.

```
hot_vs_cold  (livello 0, broad)
├── metabolic_heat  (livello 1, validato)
│   ├── fever_hyperthermia  (livello 2)
│   └── cold_chills  (livello 2)
├── circulatory_warmth  (livello 1, validato)
│   ├── peripheral_vasodilation  (livello 2)
│   └── cold_extremities  (livello 2)
└── ...
```

Il criterio di stop naturale: i vettori al livello N producono output
indistinguibili → il modello non rappresenta distinzioni a quel grado
di granularità. Questo limite è un risultato scientifico, non un fallimento.

---

## 5. Pipeline completa (dettaglio tecnico)

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

| Script | Ruolo | Note |
|--------|-------|------|
| `concept_expander.py` | Step 1+2: M40 genera sub-concetti + dataset chirurgici | Script principale nuovo |
| `run_sub_probe.sh` | Step 3: loop probe_concept.py su tutti i JSON in sub_concepts/ | Adatta run_all_probes.sh |
| `cosine_matrix.py` | Step 4: calcola matrice separabilità, salva JSON + heatmap | numpy, matplotlib |
| `sub_concept_eval.py` | Step 5: steering comparativo, M40 giudica distinzione | Estende auto_eval.py |
| `decompose.py` | **Orchestratore del loop**: coordina Step 1-5, gestisce feedback, ricorsione | Script nuovo principale |

### decompose.py — logica dell'orchestratore

```python
def decompose(concept, parent_vector, depth=0, max_depth=3, iteration=0, feedback=None):
    """
    Loop di auto-analisi ricorsivo.

    concept       : nome del concetto da decomporre
    parent_vector : path al .npy del vettore broad
    depth         : livello di ricorsione (0 = broad, 1 = sub, 2 = sub-sub)
    max_depth     : profondità massima (stop quando output indistinguibili)
    iteration     : numero iterazione corrente (max 3 per livello)
    feedback      : output del giudice M40 dal ciclo precedente
    """

    if iteration >= 3:
        log(f"[{concept}] Max iterazioni raggiunto — spazio semantico esaurito a depth={depth}")
        return  # risultato scientifico: il modello non va oltre

    # Step 1: M40 genera ipotesi sub-concetti (con feedback se disponibile)
    sub_concepts = concept_expander.generate(concept, feedback=feedback)

    # Step 2: M40 genera dataset chirurgici per ogni sub
    for sub in sub_concepts:
        concept_expander.generate_dataset(sub)

    # Step 3: MI50 estrae vettori
    run_sub_probe(concept)

    # Step 4: matrice coseno
    matrix = cosine_matrix.compute(concept, sub_concepts, parent_vector)

    # Step 5: M40 giudica separabilità reale
    verdict = sub_concept_eval.evaluate(concept, sub_concepts)

    if verdict.all_distinct:
        # Successo — archivia e vai in ricorsione su ogni sub validato
        archive(concept, sub_concepts, matrix, verdict)
        if depth < max_depth:
            for sub in sub_concepts:
                decompose(sub.name, sub.vector_path, depth=depth+1)
    else:
        # Fallimento parziale — feedback a M40 e riprova
        decompose(concept, parent_vector, depth=depth,
                  iteration=iteration+1, feedback=verdict.feedback)
```

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
