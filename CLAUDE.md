# Project Jedi — Istruzioni per Claude

## 🔴 FINE SESSIONE — obbligatorio

Quando Lele dice che per oggi si finisce, Claude DEVE (senza aspettare istruzioni):
1. `bash scripts/update_stato.sh "fine sessione"` — aggiorna STATO.md
2. Scrivere `experiments/RIPRESA_YYYYMMDD.md` — tutto ciò che serve per riprendere
3. Committare e pushare tutto su GitHub
4. Aggiornare il diario se c'è qualcosa da dire

---

## ⚡ INIZIO SESSIONE — obbligatorio

All'inizio di ogni sessione Claude DEVE:
1. Leggere `/home/lele/codex-openai/project_jedi/STATO.md`
2. Leggere `/tmp/cantagallo_pending.txt` (messaggi pendenti)
3. Verificare server: `curl -s http://localhost:8010/api/models` + `curl -s http://localhost:11435/health`

Fatto questo, riassumere lo stato a Lele e proporre cosa fare.

---

## Contesto

Project Jedi è un sistema di ricerca su activation steering in LLM.
Il cuore è un loop automatico che estrae, prova e cataloga vettori concettuali
da modelli transformer, a livelli gerarchici crescenti (Gd0 → Gd1 → Gd2...).

**Claude ha autorizzazione piena per operare in autonomia su questo progetto.**

---

## Permessi espliciti

Claude può eseguire senza chiedere conferma:

### Script e processi
- `project_jedi/.venv/bin/python project_jedi/scripts/*.py` — qualsiasi script del progetto
- `bash project_jedi/scripts/*.sh` — qualsiasi script bash del progetto
- `nohup ... &` — avvio processi in background (decompose, probe, eval)
- `kill <PID>` — terminazione processi del progetto
- `tail / cat` su log in `/tmp/decompose_*.log`, `/tmp/steering_server.log`, ecc.

### Git (repo project_jedi)
- `git add`, `git commit`, `git push origin main` — commit e push di routine
- Commit message standard: descrivere cosa è stato fatto, `Co-Authored-By: Claude`
- NON fare force push, NON modificare branch diversi da main

### File di progetto
- Leggere e modificare qualsiasi file in `project_jedi/`
- Creare nuovi script in `project_jedi/scripts/`
- Scrivere log e risultati in `project_jedi/output/`
- Aggiornare `MEMORY.md` in `.claude/projects/...`

### Monitoraggio
- Leggere log di sistema, `/tmp/*.log`, `output/decompose_runs/*.jsonl`
- Controllare processi con `ps aux | grep ...`
- Verificare stato GPU con `nvidia-smi`, `rocm-smi`

---

## Operazioni che richiedono conferma

- Eliminare file o directory
- Modificare `config/settings.json` (cambia il modello di default)
- Riavviare i server (`steering_server.py`, `llama-server`)
- Qualsiasi operazione fuori da `project_jedi/`

---

## Loop automatico — comportamento atteso

Quando si lancia `run_decompose_gd1_all.sh` (o simili), Claude deve:
1. Avviare il batch in background (`nohup`)
2. Controllare il progresso periodicamente (ogni 10-15 minuti) leggendo i log
3. Se un concept fallisce: loggare l'errore, continuare con il successivo
4. Al termine di ogni modello: ricostruire il catalog (`build_catalog_multi.py`)
5. Al termine del batch: committare e pushare i risultati su GitHub
6. Scrivere un report finale in `experiments/`

---

## Stack hardware (non modificare senza chiedere)

- **MI50** (ROCm): steering server porta 8010, Gemma3-1B-IT o Gemma2-Uncensored
- **M40** (CUDA): llama-server porta 11435, Gemma3-12B Q4_K_M
- Cambio modello MI50: `POST http://localhost:8010/api/load_model {"name": "..."}` + 60s wait
- Riavvio M40: `/mnt/raid0/llama-cpp-m40/start_cuda.sh`

---

## Nomenclatura

- **Gd0**: vettore broad (es. `hot_vs_cold`) — dataset 500+500 frasi
- **Gd1**: sub-vettori chirurgici (es. `hot_vs_cold/thermal_intensity`) — 20+20 frasi
- **boot_min > 0.85**: vettore stabile e usabile
- **type=semantic**: il vettore propaga il concept senza usare il lessico diretto (ottimo)
