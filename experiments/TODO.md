# Project Jedi — TODO & Roadmap

**Ultimo aggiornamento**: 2026-03-02

---

## IN CORSO ORA

### Batch decompose Gd1 — automatico
```bash
bash scripts/monitor_decompose.sh
tail -f /tmp/decompose_gd1_*/batch_main.log
```

| Concept | Modello | Stato |
|---------|---------|-------|
| hot_vs_cold | Gemma3-1B-IT | ✅ VALIDATI |
| luce_vs_buio | Gemma3-1B-IT | 🔄 step 5 |
| calma_vs_allerta … (7 altri) | Gemma3-1B-IT | ⏳ |
| tutti e 9 | Gemma2-Uncensored | ⏳ |

Il batch: `run_decompose_gd1_all.sh` → 18 run (9 concept × 2 modelli)
Cantagallo in background: `tmux display-message` + `/tmp/cantagallo_pending.txt`

---

## IMMEDIATO (dopo il batch)

### A. Analisi risultati Gd1
- [ ] Leggere tutti i report decompose (18 run)
- [ ] Aggiornare `experiments/07_pipeline_gd0_gd1.md` con risultati completi
- [ ] Tabella cosine matrix per ogni concept (mean off-diagonal = separabilità)
- [ ] Confronto Gemma3 vs Gemma2: stessi sub-concepts? stesse direzioni?

### B. Catalog e steering
- [ ] Verificare catalog dopo batch: `python scripts/build_catalog_multi.py`
- [ ] Test steering UI con sub-concepts Gd1 (dropdown `Gd1 — parent › slug`)

### C. Commit finale batch
Il batch fa il commit automatico, verificare con `git log --oneline -3`.

---

## MEDIO TERMINE

### 1. Confronto sistematico Gemma3 vs Gemma2 su Gd1
- [ ] Per ogni concept: cosine similarity tra stessi sub-concepts su modelli diversi
- [ ] Se coseno alto → direzione "universale"; se basso → codificazioni qualitatiamente diverse
- [ ] Scrivere `experiments/08_confronto_gd1_gemma3_gemma2.md`

### 2. Grounding con sensori fisici
- [ ] Termometro USB/I2C → `hot_vs_cold` v2 con dataset da misure reali
- [ ] Fotoresistenza o sensore lux → `luce_vs_buio` v2
- [ ] Architettura: `sensore → lettura → normalizzazione → alpha → steering API`
- [ ] Domanda scientifica: il grounding migliora la qualità dell'output?

### 3. Decompose Gd2 (opzionale, dopo Gd1 consolidato)
- [ ] Decomposizione ricorsiva di un Gd1 ben validato (es. `thermal_intensity`)
- [ ] `--max-depth 1` in decompose.py (già implementato, non usato)

### 4. Multi-concept simultaneo
- [ ] Iniettare due vettori insieme (es. hot + rumore)
- [ ] Cosine matrix 9×9 dei vettori Gd0: sono ortogonali tra loro?

### 5. Auto-eval migliorato
- [ ] `--gain-range` per testare automaticamente più gain in un run
- [ ] Modalità "calibration run" per trovare gain ottimale

### 6. Vettori su altri modelli
- [ ] Llama 3 / Mistral: stesso protocollo
- [ ] Trasferibilità dei vettori tra modelli della stessa famiglia?

---

## COMPLETATO ✅

### Pipeline Gd0 (2026-02-28 — 2026-03-01)
- ✅ Estrazione vettori Gd0: 9/9 concept × Gemma3-1B-IT (L18-23)
- ✅ Estrazione vettori Gd0: 9/9 concept × Gemma2-Uncensored (L29-38)
- ✅ Auto-eval Gd0: 9/9 × Gemma2-Uncensored (report MD completi)
- ✅ Sweet spot Gemma3: L21 gain=1000 (base), L19/L23 per concept specifici
- ✅ Sweet spot Gemma2: L37 gain=1200 (semantico), L29 gain=200 (migliore per dolce)

### Infrastruttura decompose Gd1 (2026-03-01 — 2026-03-02)
- ✅ `decompose.py` — loop completo steps 1-5
- ✅ `concept_expander.py` — M40 analizza + genera dataset chirurgici
- ✅ `cosine_matrix.py` — matrice N×N + save_heatmap (fix HAS_MATPLOTLIB)
- ✅ `sub_concept_eval.py` — steering test coppie + giudizio M40 (fix PosixPath)
- ✅ `build_catalog_multi.py` — indicizza Gd0 + Gd1
- ✅ `steering_server.py` — routing Gd0/Gd1, `/api/concepts` con sub_concepts[]
- ✅ `steering.html` — dropdown optgroup Gd0/Gd1
- ✅ `run_decompose_gd1_all.sh` — batch script 18 run
- ✅ `monitor_decompose.sh` — monitor stato batch
- ✅ `cantagallo.sh` — monitor senza tmux send-keys (fix disruption)
- ✅ M40 Gemma3-12B Q4_K_M CUDA — upgrade evaluator ~33 tok/s
- ✅ `experiments/07_pipeline_gd0_gd1.md` — documentazione cuore del progetto

---

## NOTE OPERATIVE

### Avvio server (reminder)
```bash
cd /home/lele/codex-openai
nohup project_jedi/.venv/bin/python project_jedi/scripts/steering_server.py > /tmp/steering_server.log 2>&1 &
/mnt/raid0/llama-cpp-m40/start_cuda.sh
```

### Cantagallo
```bash
nohup bash project_jedi/scripts/cantagallo.sh >> /tmp/jedi_cantagallo.log 2>&1 &
cat /tmp/cantagallo_pending.txt   # messaggi per Claude
```

### Sweet spot confermati
| Modello | Layer | Gain | Effetto |
|---------|-------|------|---------|
| Gemma3-1B-IT | L19-23 (concept-dep.) | 1000-1300 | semantico ✅ |
| Gemma2-Uncensored | L29-37 (concept-dep.) | 200-1200 | semantico ✅ |
