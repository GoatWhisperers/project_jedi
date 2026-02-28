# Project Jedi — TODO & Roadmap

**Ultimo aggiornamento**: 2026-02-28

---

## IMMEDIATO (prossima sessione)

### 1. Analisi risultati Gemma2-Uncensored
- [ ] Leggere tutti i 9 report generati dal batch `run_all_eval.sh --model Gemma2-Uncensored`
- [ ] **Anomalia da investigare**: hot_vs_cold con Gemma2 produce risultati *lessicali* (score 0-2, scrive "hot" direttamente) invece di semantici come Gemma3. Possibili cause:
  - Gain troppo alto per Gemma2 (hidden_dim=3584, vettore più grande → effetto più brutale)
  - Best layer errato (L38 potrebbe essere troppo profondo per Gemma2)
  - I vettori Gemma2 sono qualitativamente diversi (meno "semantici")
- [ ] Scrivere diario esperimento per ogni concept Gemma2 in `experiments/`
- [ ] Aggiungere note personali a ogni diario Gemma2

### 2. Confronto sistematico Gemma3 vs Gemma2
- [ ] Tabella comparativa: stesso concept, entrambi i modelli, HOT avg / COLD avg / tipo effetto
- [ ] Ipotesi da testare: Gemma2 ha vettori più potenti ma meno controllabili?
- [ ] Aggiungere file `experiments/03_confronto_gemma3_vs_gemma2.md`

### 3. Calibrazione gain per Gemma2
- [ ] Ripetere hot_vs_cold con gain molto più basso (es. 20-50 invece di 200)
- [ ] Trovare il "sweet spot" per Gemma2 equivalente a quello trovato per Gemma3 (L21 gain=1000)

---

## MEDIO TERMINE

### 4. Scomposizione gerarchica dei concept
Idea: ogni concept attuale è *misto*. Scomporlo in sotto-dimensioni più precise.

Esempio per `hot_vs_cold`:
- `hot_fisico`: temperature misurabili (termometro, ebollizione, febbre)
- `hot_metaforico`: calore di carattere ("accoglienza calorosa", "freddo come persona")
- `hot_tattile`: sensazione diretta sulla pelle (scottatura, gelo sulle dita)

Per ogni sotto-concept:
- [ ] Generare dataset chirurgico (~200 frasi per polo, solo quel registro)
- [ ] Estrarre vettore separato
- [ ] Misurare cosine similarity tra sotto-vettori → sono la stessa direzione o divergono?
- [ ] Testare se lo steering con vettore chirurgico è più preciso

Priority order:
- [ ] hot_fisico vs hot_metaforico (facile, dataset chiaro)
- [ ] luce_visiva vs luce_metaforica ("luce della ragione", "buio dell'ignoranza")
- [ ] rumore_fisico vs rumore_emotivo (caos mentale vs suono fisico)

### 5. Auto-eval migliorato
- [ ] Aggiungere `--gain-range` a `run_all_eval.sh` per testare automaticamente più gain
- [ ] Aggiungere modalità "calibration run" che trova il gain ottimale per un modello prima della sessione completa
- [ ] Score 0 in T8 Gemma2: il valutatore M40 dovrebbe dire anche *perché* il testo è solo lessicale

### 6. Eval dashboard
- [ ] Aggiungere confronto fianco a fianco Gemma3 vs Gemma2 per stesso concept
- [ ] Visualizzare la distribuzione dei tipi (semantic/lexical/mixed) per sessione

---

## AMBIZIOSO (lungo termine)

### 7. Grounding con sensori fisici
Obiettivo: collegare misure fisiche reali ai vettori concept.

**Architettura ipotizzata**:
```
sensore fisico → lettura → normalizzazione → alpha/gain → iniezione vettore
```

Sensori candidati (in ordine di semplicità tecnica):
- [ ] **Termometro USB/I2C** → hot_vs_cold (priorità 1 — più diretto)
- [ ] **Fotoresistenza o sensore lux** → luce_vs_buio
- [ ] **Microfono + RMS level** → rumore_vs_silenzio
- [ ] **Sensore umidità DHT22** → secco_vs_umido

Per ogni sensore:
- [ ] Connettere lettura in tempo reale all'API di steering
- [ ] Testare: il modello genera testo "più accurato" sul calore quando il termometro legge davvero 38°C?
- [ ] Misurare: c'è differenza qualitativa tra steering fisicamente ancorato vs steering con alpha fisso?

**Domanda scientifica**: se il grounding fisico migliora la qualità dell'output, significa che il vettore semantico e la misura fisica puntano nella stessa direzione nello spazio di rappresentazione?

### 8. Multi-concept simultaneo
- [ ] Iniettare due vettori contemporaneamente (es. hot + rumore → "stanza calda e rumorosa")
- [ ] Misurare interferenza / indipendenza tra vettori diversi
- [ ] I vettori sensoriali sono ortogonali tra loro? (test: cosine similarity tra tutti i 9 vettori)

### 9. Vettori su altri modelli
- [ ] Llama 3 / Mistral: stesse estrazioni, stesso protocollo
- [ ] I vettori sono trasferibili tra modelli della stessa famiglia?
- [ ] Esiste una "direzione universale del calore" indipendente dal modello?

---

## NOTE OPERATIVE

### Batch Gemma2 in corso
```bash
# Stato:
tail -f /tmp/eval_batch_gemma2.log
tail -f /tmp/eval_logs/eval_<concept>.log

# PID:
cat /tmp/eval_batch_gemma2.pid   # → 62974
```
Avviato 2026-02-28 16:47 — stimato completamento in ~3-4 ore.
Output in: `output/eval_sessions/session_*_Gemma2-Uncensored.*`

### Sweet spot confermati
| Modello | Concept | Layer | Gain | Effetto |
|---------|---------|-------|------|---------|
| Gemma3-1B-IT | hot_vs_cold | L19 | 1200 | semantico ✅ |
| Gemma3-1B-IT | luce_vs_buio | L23 | 1300 | semantico ✅ |
| Gemma2-Uncensored | hot_vs_cold | ? | ? | lessicale con gain=200 ⚠️ — da calibrare |

### Avvio server (reminder)
```bash
cd /home/lele/codex-openai
# Steering server (MI50):
nohup project_jedi/.venv/bin/python project_jedi/scripts/steering_server.py > /tmp/steering_server.log 2>&1 &
# M40 evaluator:
/mnt/raid0/llama-cpp-m40/start_cuda.sh &
# Batch eval (tutti i concept):
bash project_jedi/scripts/run_all_eval.sh --model Gemma2-Uncensored
```
