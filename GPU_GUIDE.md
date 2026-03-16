# GPU Guide — Project Jedi

## Architettura GPU (regola fondamentale)

**Una GPU = un solo modello in VRAM = un solo processo owner.**

Ogni GPU ha un manager dedicato. Nessun script carica modelli direttamente.

```
MI50 (ROCm)  →  mi50_manager.py  (porta 8020)  — unico owner GPU MI50
M40  (CUDA)  →  llama-server     (porta 11435) — unico owner GPU M40
```

---

## MI50 — mi50_manager.py

### Avvio
```bash
# Via systemd (automatico al boot):
systemctl is-active mi50-manager   # verifica stato

# Manuale (sviluppo):
cd /home/lele/codex-openai
nohup project_jedi/.venv/bin/python project_jedi/scripts/mi50_manager.py > /tmp/mi50_manager.log 2>&1 &
```

### Endpoints

| Metodo | Path | Descrizione |
|--------|------|-------------|
| GET | `/api/status` | Modello attivo, VRAM usata, busy flag |
| POST | `/api/load_model` | Carica modello (`{"name": "Gemma3-1B-IT"}`) |
| POST | `/api/unload_model` | Scarica modello, libera VRAM |
| POST | `/api/generate` | Generazione testo con/senza steering |
| POST | `/api/generate_stream` | Generazione streaming SSE |
| POST | `/api/stop` | Interrompe generazione in corso |
| POST | `/api/extract_activations` | Forward pass + hidden states (per probe) |

### Cambio modello
```bash
# Carica Gemma2-Uncensored (scarica automaticamente quello corrente)
curl -X POST http://localhost:8020/api/load_model \
  -H "Content-Type: application/json" \
  -d '{"name": "Gemma2-Uncensored"}'
sleep 65   # attendi caricamento
```

### Regole
- Se `busy: true` → rifiuta load/unload con HTTP 503
- Se stesso modello già caricato → load è noop (non scarica/ricarica)
- Se modello diverso → scarica prima, poi carica
- `extract_activations` processa in batch da 8 frasi (evita OOM)

### Kill e riavvio pulito
```bash
# Kill
ps aux | grep mi50_manager | grep python | awk '{print $2}' | xargs kill -9
# Riavvio via systemd
echo 'pippopippo33$$' | sudo -S systemctl restart mi50-manager.service
```

---

## MI50 — steering_server.py (porta 8010)

Layer di presentazione sopra mi50_manager. Gestisce:
- Serving `steering.html`
- Catalog vettori e vector library lookup
- Routing concept → layer → vector path
- Log generazioni in `output/steering_log.jsonl`
- Proxy di tutte le operazioni GPU verso mi50_manager

**Non carica modelli.** Delega tutto a `http://localhost:8020`.

```bash
# Avvio manuale
nohup project_jedi/.venv/bin/python project_jedi/scripts/steering_server.py > /tmp/steering_server.log 2>&1 &
```

---

## M40 — llama-server (porta 11435)

Modello: Gemma3-12B Q4_K_M (`/mnt/raid0/models-gguf/gemma-3-12b-it-Q4_K_M.gguf`)
Binario CUDA: `/mnt/raid0/llama-cpp-m40/build_cuda/bin/llama-server`

```bash
# Verifica stato
curl -s http://localhost:11435/health

# Riavvio
/mnt/raid0/llama-cpp-m40/start_cuda.sh
# oppure via systemd:
echo 'pippopippo33$$' | sudo -S systemctl restart llama-server-m40.service
```

**Nessun script del progetto carica modelli sulla M40 direttamente.**
Tutti usano l'API HTTP OpenAI-compatible su porta 11435.

---

## Systemd services

| Servizio | GPU | Porta | Avvio automatico |
|----------|-----|-------|-----------------|
| `mi50-manager.service` | MI50 | 8020 | ✅ |
| `steering-server.service` | — (client) | 8010 | ✅ |
| `llama-server-m40.service` | M40 | 11435 | ✅ |

```bash
# Verifica tutti
systemctl is-active mi50-manager steering-server llama-server-m40

# Riinstalla servizi (dopo modifiche a setup_services.sh)
echo 'pippopippo33$$' | sudo -S bash project_jedi/scripts/setup_services.sh
```

---

## Verifica VRAM

```bash
# MI50
rocm-smi --showmeminfo vram

# M40
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
```

---

## Flusso probe_concept.py (nuovo)

```
probe_concept.py
  → POST /api/load_model  (mi50_manager:8020)  — noop se modello già giusto
  → POST /api/extract_activations              — forward pass + hidden states
  ← riceve pos_reps/neg_reps come JSON
  → calcola PCA, mean_diff, salva .npy
```

## Flusso decompose.py (nuovo)

```
decompose.py
  → gpu_utils.gpu_prepare_for_probe()
      → POST /api/unload_model (mi50_manager:8020)
  → subprocess probe_concept.py
      → POST /api/load_model + /api/extract_activations
  → gpu_utils.gpu_restore_after_probe()
      → POST /api/load_model (mi50_manager:8020)
  → sub_concept_eval.py
      → POST /api/generate (steering_server:8010 → mi50_manager:8020)
```
