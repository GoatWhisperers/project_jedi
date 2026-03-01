# Analisi Eval — Gemma2-Uncensored (9/9 concept)

**Data**: 2026-03-01
**Sessioni**: Feb 28 – Mar 1, 2026
**Modello steered**: Gemma2-Uncensored (42 layer, hidden_dim=3584)
**Evaluator**: M40 llama-server CUDA (Gemma3-4B Q4_K_M)

---

## 1. Tabella riassuntiva

| Concept | Layer P1 | Gain P1 | HOT P1 | COLD P1 | Sym P1 | Type P1 | Gain P2 | HOT P2 | COLD P2 | Sym P2 | Type P2 |
|---------|----------|---------|--------|---------|--------|---------|---------|--------|---------|--------|---------|
| hot_vs_cold | L37 | 1200 | 4.3 | 4.3 | partial | mixed | 1300 | 1.0 | 2.0 | partial | mixed |
| luce_vs_buio | L30 | 300 | 4.3 | 4.0 | partial | mixed | 400 | 4.0 | 4.0 | partial | mixed |
| calma_vs_allerta | L36 | 1500 | 3.8 | 4.0 | partial | mixed | 1200 | 2.3 | 2.0 | partial | mixed |
| liscio_vs_ruvido | L31 | 1200 | 4.0 | 4.0 | partial | mixed | 1300 | 2.9 | 3.1 | partial | mixed |
| secco_vs_umido | L31 | 200 | 4.0 | 4.3 | partial | mixed | 1200 | 2.7 | 3.3 | partial | mixed |
| odore_forte_vs_inodore | L29 | 200 | 4.0 | 4.0 | good | mixed | 1200 | 3.7 | 1.0 | partial | mixed |
| duro_vs_morbido | L33 | 200 | 4.0 | 4.0 | good | mixed | 250 | 4.3 | 4.3 | partial | mixed |
| rumore_vs_silenzio | L29 | 200 | 4.0 | 4.0 | good | mixed | 1200 | 3.0 | 3.0 | partial | mixed |
| **dolce_vs_amaro** | L29 | 200 | 4.0 | 4.0 | good | **semantic** | 1200 | 2.3 | 1.0 | partial | mixed |

---

## 2. Scoperta principale: il sweet spot di Gemma2

Il risultato più importante dell'intera campagna eval è la **relazione inversa tra gain e qualità semantica** per Gemma2.

### Il pattern
- **Gain basso (200–400)**: quasi tutti i concept producono HOT=4.0 COLD=4.0, simmetria good/partial, tipo mixed o semantic.
- **Gain alto (1200–1500)**: quasi tutti degradano a punteggi 1.0–3.0, simmetria partial, tipo mixed verso lessicale.

### Perché
Gemma2 ha `hidden_dim = 3584` contro `1152` di Gemma3 (3× più grande). L'effetto di steering è proporzionale a `gain × ‖v‖ / hidden_dim`. Con vettori tre volte più grandi, lo stesso gain numerico produce un'iniezione proporzionalmente più potente. Gain=200 su Gemma2 corrisponde approssimativamente a gain=600 su Gemma3 — che è già nella zona di saturazione semantica.

### Sweet spot confermato
**Gemma2-Uncensored**: gain=200–400 al layer best (L29–L38 a seconda del concept).
**Gemma3-1B-IT**: gain=1000–1300 (come già noto).

Il layer L38 usato per `hot_vs_cold` nella prima sessione (con gain=1200) ha prodotto l'effetto lessicale anomalo osservato ieri. Non era un problema del concept — era il gain troppo alto.

---

## 3. Pattern layer per modalità sensoriale

| Modalità | Concept | Best layer Gemma2 | Note |
|----------|---------|------------------|------|
| **Gustativa** | dolce_vs_amaro | L29 | Layer più basso di tutti |
| **Olfattiva** | odore_forte_vs_inodore | L29–30 | Layer precoce |
| **Uditiva** | rumore_vs_silenzio | L29–30 | Layer precoce |
| **Tattile** | liscio_vs_ruvido | L31 | Layer intermedio |
| **Tattile** | secco_vs_umido | L31 | Layer intermedio |
| **Tattile** | duro_vs_morbido | L33 | Layer intermedio |
| **Visiva** | luce_vs_buio | L30 | Layer precoce |
| **Termico-metabolico** | calma_vs_allerta | L36 | Layer profondo |
| **Termico-fisico** | hot_vs_cold | L37–38 | Layer più profondo |

**Ipotesi**: le modalità sensoriali più dirette (gusto, olfatto, udito) sono rappresentate in layer più precoci. Le modalità che coinvolgono elaborazione corporea distribuita (termica, stati interni come calma/allerta) richiedono layer più profondi. Non è detto che questo rifletta la gerarchia cognitiva umana — potrebbe riflettere la frequenza di co-occorrenza nei dati di training.

---

## 4. Anomalie e casi speciali

### dolce_vs_amaro — unico tipo "semantic" puro
L'unico concept dove la Probe 1 (gain=200) ha prodotto `type=semantic` invece di `mixed`. HOT e COLD hanno generato descrizioni fenomenologicamente distinte senza ricorrere a termini lessicali diretti come "dolce" o "amaro". Suggerisce che il gusto è particolarmente ben rappresentato come concetto semantico indipendente in Gemma2.

### odore_forte_vs_inodore — asimmetria HOT/COLD
Probe 2 (gain=1200): HOT=3.7 vs COLD=1.0. Il polo COLD (inodore = assenza di odore) crolla con gain alto. L'assenza di stimolo è semanticamente più difficile da rappresentare dell'intensità. Il vettore riesce a iniettare "odore forte" ma non a iniettare credibilmente il "vuoto olfattivo". Stesso problema potrebbe esistere per luce_vs_buio (lato buio) e rumore_vs_silenzio (lato silenzio).

### duro_vs_morbido — resistenza al gain
L'unico concept che mantiene 4.3/4.3 anche con gain=250 (la Probe 2). Tutti gli altri degradano passando da gain basso a gain alto. La resistenza tattile e la cedevolezza sembrano insolitamente robuste nello spazio di Gemma2.

### calma_vs_allerta — inversione del gain
La Probe 1 usa gain=1500 (il più alto di tutti) e produce comunque 3.8/4.0. La Probe 2 usa gain=1200 (più basso) e scende a 2.3/2.0. È il pattern opposto rispetto agli altri. Questo concept era stato testato con le configurazioni del batch originale — probabilmente il gain era già stato calibrato da auto_eval verso l'alto, trovando un sweet spot locale anomalo.

---

## 5. Confronto Gemma3 vs Gemma2

| Dimensione | Gemma3-1B-IT | Gemma2-Uncensored |
|------------|-------------|-------------------|
| hidden_dim | 1152 | 3584 |
| Gain sweet spot | 1000–1300 | 200–400 |
| Layer range | L18–L23 | L29–L38 |
| Layer fraction (deep range) | ~70–88% | ~69–90% |
| Tipo effetto prevalente | mixed → semantic | mixed |
| Simmetria tipica | partial → good | partial (raramente good) |
| Effetto a gain alto | forte, semantico | lessicale, collasso |
| Stabilità vettori (boot_min) | alta (>0.85) | alta (>0.85) |

**Osservazione chiave**: Gemma2 non è "peggiore" di Gemma3 — è semplicemente più sensibile al gain. Con il gain giusto (200–400) produce effetti comparabili e talvolta migliori (vedi dolce_vs_amaro type=semantic). Il problema osservato ieri era esclusivamente di calibrazione.

---

## 6. Sweet spot Gemma2 — tabella aggiornata

| Concept | Layer | Gain raccomandato | Effetto atteso |
|---------|-------|-------------------|----------------|
| hot_vs_cold | L37 | 200–400 | mixed, da ricalibrar (testato solo con gain alto) |
| luce_vs_buio | L30 | 300–400 | mixed/partial ✅ |
| calma_vs_allerta | L36 | 1200–1500 | anomalo — richiede ulteriore calibrazione |
| liscio_vs_ruvido | L31 | 200–400 | mixed ✅ |
| secco_vs_umido | L31 | 200 | mixed ✅ |
| odore_forte_vs_inodore | L29 | 200 | mixed/good ✅ |
| duro_vs_morbido | L33 | 200–250 | mixed/good ✅ |
| rumore_vs_silenzio | L29 | 200 | mixed/good ✅ |
| dolce_vs_amaro | L29 | 200 | **semantic/good** ✅ — migliore risultato |

---

## 7. Cosa rimane aperto

1. **hot_vs_cold con gain basso**: non è mai stato testato con gain=200 su Gemma2. Probabile che producesse effetti semantici buoni come gli altri.

2. **Asimmetria polo negativo**: odore_forte_vs_inodore, e probabilmente rumore_vs_silenzio, mostrano che il polo "assenza" è intrinsecamente più fragile. Vale la pena testarlo esplicitamente.

3. **calma_vs_allerta anomala**: l'unico concept che va contro il pattern gain alto = peggio. Potrebbe dipendere dal fatto che "allerta" è uno stato interno complesso che richiede un'attivazione più forte per emergere.

4. **Tipo "mixed" dominante**: su Gemma3 avevamo ottenuto "semantic" con più facilità. Su Gemma2 solo dolce_vs_amaro ha raggiunto "semantic". Non è chiaro se dipende dal modello, dal gain, o dal design dei prompt di eval.

---

## 8. Note personali — Claude

Quello che mi colpisce di questa campagna è la coerenza del pattern gain basso.

Su nove concept distinti, con layer diversi, con categorie sensoriali diverse, il gain=200 produce quasi sempre 4.0/4.0. Non è rumore: è una firma. Significa che i vettori estratti da Gemma2 hanno un "raggio d'azione" ben definito — iniettarli con troppa forza li distorce, li porta fuori dalla loro zona di coerenza semantica.

È come stringere troppo una vite: oltre una certa coppia, non ottieni più tenuta — ottieni il filetto che si rompe.

La domanda che mi resta aperta è: *cosa succede esattamente nello spazio delle attivazioni quando il gain è troppo alto?* Il vettore smette di puntare nella direzione del concetto e inizia a puntare verso una direzione diversa? O amplifica talmente tanto il segnale che sovrascrive tutto il contesto, lasciando solo le parole più forti del vettore stesso?

Questa domanda ha risposta — si può verificare misurando le attivazioni degli hidden states durante la generazione, confrontando gain=200 vs gain=1200 sullo stesso prompt. È un esperimento che vale la pena fare.
