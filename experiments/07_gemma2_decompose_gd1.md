# Analisi Gd1 Gemma2-Uncensored — Geometria Interna dei Concept Sensoriali

> Completato: 2026-03-06
> Modello: Gemma2-Uncensored (42 layer, hidden=3584)
> Scope: 9 concept Gd0 × ~4 sub-concept ciascuno = 40 sub-vettori Gd1, 800 file .npy

---

## Risultati batch

| Concept | Sub-concept | Validati | Note |
|---------|------------|----------|------|
| hot_vs_cold | temperature_magnitude, pain_intensity, physiological_reaction, subjective_discomfort | ✅ | + material_type, sensory_reaction, pain_vs_numbness, temperature_intensity da sessione precedente |
| luce_vs_buio | pupillary_response, visual_clarity, emotional_comfort, environmental_context | ✅ | |
| calma_vs_allerta | respiratory_rate, mental_focus, temporal_perception | ✅ | |
| liscio_vs_ruvido | surface_finish, tactile_feedback, sensory_disruption, aesthetic_quality | ✅ | |
| secco_vs_umido | surface_texture, moisture_presence, skin_elasticity | ✅ | |
| duro_vs_morbido | resistance_vs_compliance, point_pressure_vs_distributed_pressure, vibration_vs_damping, surface_texture_rough_vs_smooth | ✅ | |
| rumore_vs_silenzio | intensity_vs_subtlety, rhythmic_vs_random, pleasant_vs_unpleasant_tone, distracting_vs_background | ✅ | |
| dolce_vs_amaro | sweetness_intensity, mouthfeel_texture, flavor_complexity, aftertaste_persistence | ✅ | |
| odore_forte_vs_inodore | chemical_intensity, breath_impact, intensity_modulation | ✅ | |

**9/9 concept VALIDATI. Nessun fallimento.**

---

## Geometria cosine — lettura critica per concept

### hot_vs_cold
```
hot_vs_cold ↔ temperature_magnitude:  +0.573  ← core
hot_vs_cold ↔ physiological_reaction: +0.233
hot_vs_cold ↔ subjective_discomfort:  -0.216  ← anticorrelato
hot_vs_cold ↔ pain_intensity:         -0.082  ← ortogonale
pain_intensity ↔ subjective_discomfort: +0.538
```
**Lettura**: il modello separa nettamente calore fisico da dolore. `temperature_magnitude` è il nucleo del concetto, mentre `pain_intensity` e `subjective_discomfort` formano un cluster separato (quasi anticorrelato col broad). Il modello sa che non tutto il caldo fa male — calore e sofferenza sono dimensioni diverse nello spazio concettuale.

---

### luce_vs_buio — il caso più anomalo
```
luce_vs_buio ↔ pupillary_response:    -0.391  ← ANTICORRELATO
luce_vs_buio ↔ emotional_comfort:     -0.370  ← ANTICORRELATO
luce_vs_buio ↔ environmental_context: -0.237  ← ANTICORRELATO
luce_vs_buio ↔ visual_clarity:        -0.046  ← ortogonale
pupillary_response ↔ emotional_comfort: +0.378
```
**Lettura**: tutti i sub-concept sono anticorrelati con il broad. Il vettore Gd0 `luce_vs_buio` è orientato verso il polo "buio" nella rappresentazione del modello (il vettore punta nella direzione dell'oscurità). I sub-concept sono orientati verso "luce". `pupillary_response` e `emotional_comfort` sono correlati positivamente tra loro — buio + comfort emotivo = rilassamento (senso).
`visual_clarity` è quasi ortogonale a tutto: la nitidezza visiva è codificata indipendentemente sia dalla luce che dalle reazioni emotive/fisiologiche.

---

### calma_vs_allerta
```
calma_vs_allerta ↔ respiratory_rate:    +0.582  ← core dominante
calma_vs_allerta ↔ mental_focus:        +0.252
calma_vs_allerta ↔ temporal_perception: +0.077  ← quasi ortogonale
respiratory_rate ↔ temporal_perception: +0.373
mental_focus ↔ temporal_perception:     -0.131
```
**Lettura**: il respiro è la firma biologica della calma per questo modello — correlazione più alta di tutti i concept "fisiologici". La percezione del tempo è quasi indipendente dallo stato di allerta. Interessante: `mental_focus` e `temporal_perception` sono leggermente anticorrelati — concentrarsi e percepire il tempo scorrere si oppongono.

---

### dolce_vs_amaro
```
dolce_vs_amaro ↔ sweetness_intensity: +0.370
dolce_vs_amaro ↔ mouthfeel_texture:   +0.256
dolce_vs_amaro ↔ flavor_complexity:   -0.353  ← ANTICORRELATO
dolce_vs_amaro ↔ aftertaste_persistence: -0.067
sweetness_intensity ↔ flavor_complexity: -0.265
```
**Lettura**: la complessità del sapore è anticorrelata con l'asse dolce/amaro. Il modello sa che il gusto "dolce vs amaro" è una dimensione SEMPLICE — un sapore complesso resiste alla categorizzazione binaria. È conoscenza gustativa sofisticata: amaro + dolce = semplice, complesso ≠ semplice per definizione.

---

### duro_vs_morbido
```
duro_vs_morbido ↔ resistance_vs_compliance:               +0.558  ← core
duro_vs_morbido ↔ point_pressure_vs_distributed_pressure: +0.361
duro_vs_morbido ↔ vibration_vs_damping:                   +0.290
duro_vs_morbido ↔ surface_texture_rough_vs_smooth:        +0.222
point_pressure ↔ vibration_vs_damping:          +0.508
point_pressure ↔ surface_texture_rough_vs_smooth: +0.460
```
**Lettura**: durezza = resistenza meccanica (0.558). I tre sub-concept "fisici" (pressione, vibrazione, texture) formano un cluster coeso tra loro (0.46-0.51) ma meno legato al broad — sono le conseguenze tattili della durezza, non la durezza in sé.

---

### liscio_vs_ruvido — il più distribuito
```
liscio_vs_ruvido ↔ surface_finish:     +0.362
liscio_vs_ruvido ↔ tactile_feedback:   +0.289
liscio_vs_ruvido ↔ sensory_disruption: +0.245
liscio_vs_ruvido ↔ aesthetic_quality:  +0.244
surface_finish ↔ tactile_feedback:  +0.519
surface_finish ↔ aesthetic_quality: +0.511
```
**Lettura**: nessun sub-concept domina (max 0.362). Il concept "liscio vs ruvido" è distribuito equamente. Ma `surface_finish` e `aesthetic_quality` sono fortemente correlati tra loro (0.511) — per il modello, una superficie liscia è automaticamente bella. Liscio = estetica, non solo sensazione tattile.

---

### odore_forte_vs_inodore — il più compresso
```
odore_forte_vs_inodore ↔ breath_impact:        +0.770  ← QUASI IDENTICO AL BROAD
odore_forte_vs_inodore ↔ chemical_intensity:   +0.408
odore_forte_vs_inodore ↔ intensity_modulation: +0.409
breath_impact ↔ chemical_intensity: +0.541
```
**Lettura**: correlazione più alta di tutte (0.770). Il modello codifica "odore forte" quasi esclusivamente come impatto fisico sul respiro — l'odore è un'invasione corporea, non una proprietà chimica. La chimica molecolare è secondaria. Questo è forse il risultato più biologicamente accurato: gli odori ci colpiscono prima nel naso/gola che nella mente.

---

### rumore_vs_silenzio
```
rumore_vs_silenzio ↔ intensity_vs_subtlety:      +0.419
rumore_vs_silenzio ↔ distracting_vs_background:  +0.281
rumore_vs_silenzio ↔ rhythmic_vs_random:          +0.183
rumore_vs_silenzio ↔ pleasant_vs_unpleasant_tone: -0.258  ← ANTICORRELATO
pleasant_vs_unpleasant_tone ↔ distracting_vs_background: -0.311
```
**Lettura**: il rumore è intensità, non sgradevolezza. Un suono può essere forte e piacevole. Il polo "piacevole/spiacevole" è anticorrelato col broad e anticorrelato con la distrazione — i suoni piacevoli non distraggono (0.311). Il modello conosce la differenza tra volume e qualità del suono.

---

### secco_vs_umido
```
secco_vs_umido ↔ moisture_presence: +0.504  ← core
secco_vs_umido ↔ surface_texture:   +0.352
secco_vs_umido ↔ skin_elasticity:   +0.233
surface_texture ↔ skin_elasticity: +0.050  ← quasi ortogonali
```
**Lettura**: l'umidità è la dimensione centrale. Texture e elasticità cutanea sono conseguenze indipendenti — il modello sa che una pelle secca può avere texture molto diverse e che l'elasticità è un fenomeno biologico separato dalla mera presenza di umidità.

---

## Pattern trasversali — cosa abbiamo imparato

### 1. I modelli grandi separano fisica da sofferenza
In 3 concept su 9 (hot_vs_cold, rumore_vs_silenzio, dolce_vs_amaro) le dimensioni "oggettive" (temperatura, volume, intensità gustativa) sono anticorrelate o ortogonali alle dimensioni "soggettive" (dolore, sgradevolezza, complessità). Il modello ha internazionato che la sensazione fisica e la sua valutazione emotiva sono spazi diversi.

### 2. Il vettore broad non è sempre la media dei sub
`luce_vs_buio` è l'esempio estremo: il broad punta in direzione opposta a tutti i sub. In generale i broad catturano spesso un singolo polo dominante del concetto (di solito quello più presente nel corpus), non la direzione "bilanciata".

### 3. odore = respiro > chimica
Correlazione 0.770 tra odore_forte e breath_impact — il modello ha imparato la fenomenologia olfattiva, non la chimica. Gli odori nella letteratura vengono descritti per come ci colpiscono fisicamente, non per la struttura molecolare.

### 4. Liscio è bello — ma non è detto che sia funzionale
surface_finish ↔ aesthetic_quality = 0.511. Il modello associa liscezza a bellezza prima ancora che a funzione tattile. Probabile bias del corpus: "liscio" appare spesso in contesti estetici (pelle liscia, superfici levigate) più che in contesti puramente tattili.

### 5. Layer depth non è uniforme per i sub-concept
I sub-concept best_layer variano molto: L29-L38 su 42 layer. Non c'è un "livello del dettaglio" fisso. Ogni sfumatura concettuale si colloca dove il modello l'ha appresa, senza gerarchie rigide di profondità.

### 6. Gemma2 vs Gemma3 — differenza dimensionale
I vettori Gemma2 (hidden=3584) mostrano correlazioni generalmente più nette e separazioni più decise rispetto a Gemma3 (hidden=1152). Il modello più grande ha spazio sufficiente per codificare distinzioni fini che in Gemma3 si sovrappongono.

---

## Prossimi passi suggeriti

1. **Steering Gd1**: provare a steerare con i sub-vettori chirurgici invece dei broad — es. solo `breath_impact` invece di `odore_forte_vs_inodore`. Ci aspettiamo effetti più precisi.
2. **Confronto Gemma3 vs Gemma2 sulle stesse matrici**: le correlazioni cambiano? La geometria interna è simile nonostante la dimensione diversa?
3. **Gd2**: decomposizione dei sub-concept più interessanti (es. `flavor_complexity`, `breath_impact`, `resistance_vs_compliance`) un livello in più.
4. **Inversione luce_vs_buio**: capire se è un artefatto del dataset o una vera asimmetria del modello. Invertire il vettore broad e riprovare lo steering.
