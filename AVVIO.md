# Project Jedi — Avvio rapido

## Avvio server (in ordine)

```bash
cd /home/lele/codex-openai

# 1. Steering server (MI50 — Gemma3-1B-IT o Gemma2)
nohup project_jedi/.venv/bin/python project_jedi/scripts/steering_server.py \
    > /tmp/steering_server.log 2>&1 &
# Aspetta ~25s per il caricamento del modello
# Per caricare Gemma2: usa la UI (Load button) oppure:
# curl -s -X POST http://localhost:8010/api/load_model -H "Content-Type: application/json" \
#      -d '{"name":"Gemma2-Uncensored"}'

# 2. M40 evaluator (Tesla M40 — Gemma3-12B Q4_K_M)
/mnt/raid0/llama-cpp-m40/start_cuda.sh
# Riavvio se necessario:
# kill $(cat /tmp/llama_server_m40.pid) 2>/dev/null; sleep 2
# /mnt/raid0/llama-cpp-m40/start_cuda.sh

# 3. Verifica che tutto sia su
curl -s http://localhost:8010/api/models | python3 -c "import sys,json; d=json.load(sys.stdin); print('Steering OK — modello:', d.get('active'))"
curl -s http://localhost:11435/health && echo " ← M40 OK"
```

---

## Batch decompose Gd1 (9 concept × 2 modelli)

```bash
cd /home/lele/codex-openai/project_jedi

# Avvio (una volta sola — gira per ~36 ore)
nohup bash scripts/run_decompose_gd1_all.sh \
    > /tmp/decompose_batch_main.log 2>&1 &
echo "Batch PID: $!"

# Monitor leggibile
bash scripts/monitor_decompose.sh

# Log in diretta
tail -f /tmp/decompose_gd1_*/batch_main.log

# Dialoghi scienziato/cavia in diretta (M40 e Gemma)
bash scripts/watch_dialoghi.sh
```

---

## Cantagallo (monitor autonomo)

```bash
# Avvio (in background — si ferma da solo quando il batch finisce)
nohup bash project_jedi/scripts/cantagallo.sh >> /tmp/jedi_cantagallo.log 2>&1 &

# Messaggi pendenti per Claude
cat /tmp/cantagallo_pending.txt

# Log cantagallo
tail /tmp/jedi_cantagallo.log
```

> Il cantagallo usa `tmux display-message` per notifiche visive + file pending.
> NON usa `tmux send-keys` — non interrompe mai la sessione Claude.

---

## Vedi cosa stanno facendo i modelli

```bash
bash scripts/watch_dialoghi.sh          # dialoghi M40 + output Gemma in tempo reale
bash scripts/monitor_decompose.sh       # stato batch (progress + GPU)
watch -n 30 bash scripts/monitor_decompose.sh  # aggiornamento automatico ogni 30s
```

---

## Steering manuale (UI)

```bash
# Apri nel browser:
# http://localhost:8010/ui/steering.html  (oppure servito da probe_server)

# Probe dashboard:
nohup project_jedi/.venv/bin/python project_jedi/scripts/probe_server.py \
    > /tmp/probe_server.log 2>&1 &
# → http://localhost:8000
```

---

## Kill server

```bash
# Steering (MI50):
ps aux | grep steering_server | grep python | awk '{print $2}' | xargs kill -9

# Probe:
ps aux | grep probe_server | grep python | awk '{print $2}' | xargs kill -9

# M40:
kill $(cat /tmp/llama_server_m40.pid) 2>/dev/null

# Batch decompose (ferma tutto):
pkill -f run_decompose_gd1_all.sh
pkill -f decompose.py

# Cantagallo:
pkill -f cantagallo.sh
```

---

## Ricostruzione catalog

```bash
cd /home/lele/codex-openai/project_jedi
.venv/bin/python scripts/build_catalog_multi.py

# Verifica:
python3 -c "
import json
d = json.load(open('output/catalog.json'))
gd0 = [e for e in d['concepts'] if not e.get('is_sub_concept')]
gd1 = [e for e in d['concepts'] if e.get('is_sub_concept')]
print(f'Gd0: {len(gd0)}  |  Gd1: {len(gd1)}')
"
```

---

## Dove sono i risultati

| Cosa | Dove |
|------|------|
| Vettori Gd0 estratti | `output/vector_library/{cat}/{concept}/{model}/` |
| Vettori Gd1 estratti | `output/vector_library/{cat}/{concept}/sub/{slug}/{model}/` |
| Catalog (indice) | `output/catalog.json` |
| Dialoghi M40 completi | `output/m40_dialogues/{concept}/{model}/` |
| Eval sub-concepts | `output/sub_concept_evals/{concept}/{model}/eval_v*.json` |
| Cosine matrix | `output/decompose_runs/cosine_{concept}_{model}.json` |
| Run events | `output/decompose_runs/{timestamp}_{concept}_{model}_d0.jsonl` |
| Log batch (temp) | `/tmp/decompose_gd1_*/batch_main.log` ← perso al reboot |
| Log steering | `output/steering_log.jsonl` |
