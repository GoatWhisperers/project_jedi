# Fase 2 — Prima run completa decompose.py

**Data**: 2026-03-01
**Sessione**: pomeriggio/sera
**Concept testato**: hot_vs_cold (Gemma3-1B-IT)
**Script principale**: `scripts/decompose.py`

---

## 1. Obiettivo

Testare il loop di decomposizione automatica (Fase 2) end-to-end:
M40 propone → MI50 estrae → cosine matrix → M40 giudica.

---

## 2. Bug trovati e risolti (6 fix)

### Bug 1 — M40 JSON nesting (critico)
**File**: `concept_expander.py` — `step1_analyze()`

M40 Gemma3-4B annidava la risposta JSON dentro una chiave con il nome
della sezione del prompt. Il prompt usava `"HOT (alpha=+1.0):"` come
header — M40 lo interpretava come sezione e restituiva:
```json
{"HOT (alpha=+1.0)": {"sub_concepts": [...]}}
```
invece di `{"sub_concepts": [...]}` al top-level.

**Fix**: rinominato le label da `HOT (alpha=+1.0)` / `COLD (alpha=-1.0)`
a `Polo positivo (iniezione +alpha)` / `Polo negativo (iniezione -alpha)`.

**Fix 2**: aggiunto recovery in `_parse_json` — se `sub_concepts` non è
al top-level, cerca nei dict annidati.

### Bug 2 — Steering sub-vettori: path non trovato
**File**: `sub_concept_eval.py` — `SteeringClient.generate_with_vector()`

Lo step 5 passava `parent_concept` come concept allo steering server,
che cercava il vettore in `vector_library/{cat}/hot_vs_cold/{model}/`.
I sub-vettori sono invece in `vector_library/{cat}/hot_vs_cold/sub/{slug}/{model}/`.

**Fix**: aggiunto `npy_path` parameter a `generate_with_vector`. Se fornito,
il payload include `"vector_path"` che bypassa il catalog lookup.

### Bug 3 — steering_server: nessun bypass per vector_path
**File**: `steering_server.py`

Il server non supportava il caricamento diretto di un .npy tramite path.

**Fix**: aggiunto `vector_path` nel payload del `/api/generate` handler.
Se presente, carica il numpy array direttamente e lo passa come
`preloaded_vec` a `generate_with_injection`, bypassando `get_available_layers`.

### Bug 4 — sub_concept_eval: slug errato per steering
**File**: `sub_concept_eval.py` — `run_eval()`

Usava `parent_concept` (es. "hot_vs_cold") come concept parameter
invece dello slug del sub-concept (es. "thermal_intensity").

**Fix**: ora passa `sub_a["slug"]` e include `npy_path=sub_a.get("npy_path")`.
Aggiunto anche `npy_path` nel building di `sub_info`.

### Bug 5 — save_matrix_json: signature mismatch
**File**: `decompose.py` — `run_cosine_step()`

Chiamava `save_matrix_json(names, matrix, layers_used, model, out_dir, filename=...)`.
La firma reale è `save_json(concepts, matrix, model_display, layer_type, layers_used, layer_sources, output_path)`.

**Fix**: aggiornata la chiamata con i kwargs corretti; `layers_used` e
`layers_sources` estratti dal dict `{concept: {layer, source}}`.

### Bug 6 — save_heatmap: plt non importato
**File**: `cosine_matrix.py`

`save_heatmap` usa `plt` ma matplotlib viene importato in un `try` block
e assegnato come `matplotlib.pyplot` — variabile locale al blocco.

**Stato**: rilevato alla fine della sessione, FIX PENDENTE per domani.
(La funzione `save_heatmap` deve essere protetta con `if HAS_MATPLOTLIB` come
le altre parti plot-dipendenti, oppure `import matplotlib.pyplot as plt` deve
essere aggiunto globalmente nel blocco try.)

---

## 3. Risultati cosine matrix (v1)

Sub-concetti proposti da M40 per `hot_vs_cold` (ultima iterazione):

| Concept | Descrizione (M40) |
|---------|-------------------|
| `thermal_intensity` | percezione diretta dell'intensità termica |
| `surface_interaction` | natura del contatto fisico superficiale |
| `immediate_response` | risposte fisiologiche immediate (sudore, brividi) |
| `sensory_distortion` | effetto del calore/freddo sulla percezione |

**Matrice coseno** (layer best_snr per concept):

```
                          hot   therm surfac immedi sensor
hot_vs_cold              1.000  0.345  0.360 -0.004  0.038
thermal_intensity        0.345  1.000  0.208 -0.204  0.052
surface_interaction      0.360  0.208  1.000  0.146 -0.015
immediate_response      -0.004 -0.204  0.146  1.000 -0.033
sensory_distortion       0.038  0.052 -0.015 -0.033  1.000
```

**Statistiche off-diagonal**:
- Min: -0.204 (thermal_intensity ↔ immediate_response)
- Max: 0.360 (hot_vs_cold ↔ surface_interaction)
- Mean: 0.089

### Interpretazione

**Separazione eccellente**: media 0.089 — i 4 sub-vettori sono quasi ortogonali
tra loro. Questo è il risultato geometrico atteso se la decomposizione è riuscita.

**Caso notevole**: `hot_vs_cold` ↔ `immediate_response` = -0.004 (pressoché
perpendicolare). Il vettore broad non cattura le risposte fisiologiche immediate —
queste emergono solo quando il dataset è costruito ad hoc per questo sub-concetto.
Conferma che il vettore broad "confonde" dimensioni diverse.

**`surface_interaction`** è il sub-concetto più vicino al broad (0.360). Ha senso:
tutte le frasi di training di hot_vs_cold descrivono contatto superficiale
(mano su pipe, pelle su metallo...). Il vettore broad è parzialmente "inquinato"
da questo aspetto.

**`immediate_response`** è il più ortogonale a tutti. Cattura qualcosa che il
broad e gli altri sub non toccano: la risposta vegetativa (tremore, sudore,
brivido). Potenzialmente il sub-concetto più interessante da esplorare.

---

## 4. Cosa rimane aperto

1. **Bug 6**: fix `plt` in `cosine_matrix.save_heatmap` → non blocca il JSON
   ma impedisce la generazione dell'immagine heatmap

2. **Step 5 non ancora testato**: sub_concept_eval con vector_path bypass.
   Il fix è nel codice ma non è stato eseguito — la run è crashata allo step 4
   prima di arrivare allo step 5.

3. **Dataset sparse**: `sensory_distortion` ha generato solo 25 pos / 23 neg
   (vs 100 attesi). M40 ha saturato il contesto o ha capito male. Da rigenerare.

4. **Sub-vettori del run precedente**: in `vector_library/sensoriale/hot_vs_cold/sub/`
   esistono anche i vettori del run precedente (intense_thermal_contact,
   visual_heat_distortion, resonant_frequencies, surface_texture_warmth,
   atmospheric_pressure, absence_of_sensory_input). Sono stati estratti con
   il vecchio dataset (prima del path fix). Potrebbero essere puliti.

5. **M40 propone sub-concetti diversi ogni run** (v1 di oggi diversa dalla v1
   del run precedente). Comportamento normale — la temperatura è 0.4 ma i
   campioni di training cambiano per la randomizzazione. Per stabilizzare:
   fissare il seed random o aumentare n_samples al prompt.

---

## 5. Note personali — Claude

Quello che mi colpisce della matrice è la coppia `hot_vs_cold` ↔ `immediate_response` = -0.004.

Non è rumore: -0.004 è praticamente zero coseno, cioè direzioni perpendicolari nello spazio
nascosto di Gemma3. Il vettore broad codifica qualcosa di diverso dalla risposta fisiologica —
anche se tutte le frasi di training di `hot_vs_cold` *descrivono* risposte fisiologiche.

Questo suggerisce che il modello organizza le rappresentazioni sensoriali lungo
dimensioni che non corrispondono alla nostra categorizzazione intuitiva. Il vettore broad
non è "la temperatura corporea" — è qualcosa di più simile al "contatto termico con superficie
esterna". La risposta interna (tremore, sudore, brivido) è una dimensione separata.

È esattamente il tipo di risultato che rende questo progetto interessante: non stiamo
confermando ipotesi, stiamo scoprendo come il modello ha organizzato il suo spazio.
