# Project Jedi — Documentazione Completa

Data: 2026-02-26

Questa guida descrive **l’intero progetto**, come riprendere domani e dove mettere mano.
Tutto è contenuto dentro `project_jedi/`.

---

## 1) Obiettivo
- Estrarre vettori concettuali (direzioni latenti) da un LLM.
- Usarli per **steering/iniezione** durante inferenza chat.
- Valutare l’effetto del concetto su output.

Concetto attuale: **hot_vs_cold** (caldo ↔ freddo).

---

## 2) Hardware e Modello
- GPU: **MI50** (ROCm)
- Modello default: **Gemma3‑1B-IT** (HF/Transformers)

### Modelli disponibili

| Nome | Path | Layer | Hidden Size |
|------|------|-------|-------------|
| Gemma3-1B-IT | `/mnt/raid0/gemma-3-1b-it` | 26 | 1152 |
| Gemma2-Uncensored | `/mnt/raid0/gemma-2-uncensored` | 42 | 3584 |

> **Nota:** i vettori del catalogo sono specifici per il modello (hidden_size diverso).
> Vettori estratti con un modello **non sono intercambiabili** con un altro.

---

## 3) Ambiente isolato
Venv isolato in `project_jedi/.venv` con pacchetti copiati da `/home/lele/venv-rocm311`.

---

## 4) Struttura cartelle
```
project_jedi/
  config/
    settings.json
    queries_hot.json
    queries_cold.json
  output/
    run_YYYYMMDD_HHMMSS/
    dashboard.html
    status.json
    latest.json
    catalog.json
    steering_log.jsonl
  scripts/
    probe_hot_cold.py
    eval_hot_cold.py
    build_catalog.py
    build_catalog_multi.py
    steering_server.py
  ui/
    steering.html
  swarm/
    WORKFLOW.md
  DOCS.md
  PROJECT.md
  README.md
```

---

## 5) Configurazione (`config/settings.json`)
Chiavi principali:
- `model_path`: path HF del modello default
- `models`: lista modelli disponibili per hot-swap dalla UI (name + path)
- `deep_range`: frazione layer profondi (es. 0.70–0.90)
- `batch_size`, `max_length`
- `concept_name`: nome concetto
- `device`: `cuda` o `cpu`

---

## 6) Estrazione vettori (probing)
Script: `scripts/probe_hot_cold.py`

Flusso:
1. Carica query da `config/queries_hot.json` e `queries_cold.json`.
2. Estrae hidden states nei layer profondi.
3. Calcola vettore concetto: mean(pos) − mean(neg).
4. Salva un `.npy` per layer.

Output tipico:
```
output/run_.../
  concept_hot_vs_cold_layer_18.npy
  ...
  summary.json
  queries_hot.json
  queries_cold.json
```

---

## 7) Valutazione
Script: `scripts/eval_hot_cold.py`

Genera `projections_eval.json` con:
- proiezioni su frasi di test
- ranking layer (separazione hot‑cold)

**Fix B1 applicato:** la valutazione ora usa `extract_all_layers()` che fa **una sola forward pass**
e restituisce hidden states per ogni layer. Per ciascun layer, le proiezioni vengono calcolate
usando le rappresentazioni dello **stesso layer** (non più sempre l'ultimo), garantendo
coerenza metodologica tra vettore concettuale e rappresentazione.

---

## 8) Catalogo concetti
- `build_catalog.py`: aggiorna `output/catalog.json` dall’ultimo run.
- `build_catalog_multi.py`: ricostruisce catalogo da tutti i run.

Catalogo contiene:
- concetto
- layer
- paths ai vettori (scoperti via glob dai filename reali)
- query usate
- metriche

**Fix B2 applicato:** `build_catalog.py` deduplica per `(run_dir, concept)`.
Se un’entry per la stessa coppia esiste, viene rimpiazzata; altrimenti aggiunta.

**Fix B3 applicato:** `build_catalog_multi.py` usa `glob("concept_*.npy")` e regex
`_layer_(\d+).npy` per costruire il dict layer→path. Nessun path costruito a mano
(che era errato quando il naming variava, es. `hot_cold` vs `hot_vs_cold`).

---

## 9) Dashboard (probing)
File: `output/dashboard.html`
Server:
```bash
cd /home/lele/codex-openai/project_jedi/output
python3 -m http.server 8001
```

La dashboard include una **card GPU** che legge `/api/gpu` dal steering server
(porta 8010) con gestione errore graceful (se offline → N/A). Mostra:
- Temperatura Edge e Junction (°C) da rocm-smi
- VRAM usata % (PyTorch)
- MB allocati / totale

---

## 10) Steering Console (Chat + Iniezione)

### Server
```bash
cd /home/lele/codex-openai
./project_jedi/.venv/bin/python project_jedi/scripts/steering_server.py
```

Porta default: `8010`

### UI
Apri: `http://<server>:8010/`

### Funzioni
- Chat con contesto (system + history)
- Prompt formattato via `apply_chat_template` se disponibile
- Injection con:
  - **Vector Layer (source)**: layer da cui proviene il vettore
  - **Inject Layer (target)**: layer dove iniettare
  - **Alpha**: verso (+ caldo / − freddo)
  - **Gain**: amplificazione
- Modalità **Multi-layer**: inietta ogni vettore nel proprio layer
- **Clear Context**: reset conversazione
- **Widget GPU** nell’header: temp edge, junction, VRAM% (aggiornato ogni 5s)
- **Model selector**: dropdown + pulsante "Load" per hot-swap modello a runtime

### Quando avviene l’iniezione
- Solo sui **token nuovi** generati dal modello.

### API Endpoints

| Metodo | Path | Descrizione |
|--------|------|-------------|
| GET | `/` | UI steering (HTML) |
| GET | `/api/concepts` | Lista concetti disponibili |
| GET | `/api/concept_layers` | Layer disponibili per il primo concetto |
| GET | `/api/model_info` | num_layers e device del modello attivo |
| GET | `/api/gpu` | Stats GPU: temp edge/junction (rocm-smi) + VRAM (PyTorch) |
| GET | `/api/models` | Lista modelli configurati + modello attivo |
| POST | `/api/generate` | Genera testo con o senza injection |
| POST | `/api/load_model` | Hot-swap modello `{"name":"..."}` |

### Selezione modello dalla UI
1. Aprire `http://<server>:8010/`
2. Nel widget **Model** nell’header, selezionare il modello dal dropdown
3. Premere **Load** — spinner visibile durante il caricamento
4. Dopo il caricamento, le dropdown layer si aggiornano automaticamente

> **Attenzione:** dopo il cambio modello, i vettori del catalogo precedente potrebbero
> avere `hidden_size` incompatibile. Rigenerare i vettori con il nuovo modello prima di
> usare l’injection.

---

## 11) Multi-layer injection
Se spunti **Multi-layer**, il server:
- prende i vettori di tutti i layer nel catalogo
- inietta ogni vettore nel **proprio layer**
- usa lo stesso `alpha` (segno scelto dall’utente)

---

## 12) Log
File: `output/steering_log.jsonl`
Contiene prompt, alpha, layer e output per ogni test.

---

## 13) Comandi rapidi
Probing:
```bash
./project_jedi/.venv/bin/python project_jedi/scripts/probe_hot_cold.py
./project_jedi/.venv/bin/python project_jedi/scripts/eval_hot_cold.py
./project_jedi/.venv/bin/python project_jedi/scripts/build_catalog.py
```

Steering:
```bash
./project_jedi/.venv/bin/python project_jedi/scripts/steering_server.py
```

---

## 14) Note operative
- Se la MI50 è piena, il server non parte (OOM).
- Iniezioni troppo forti (gain alto) possono rompere il testo.
- Per test visibili: `alpha` 0.5–1.0, `gain` 100–400.

---

## 15) GPU Monitor
Il server espone `/api/gpu` con:
- `edge_c`, `junction_c`: temperature da rocm-smi
- `mem_use_pct`, `mem_activity`: uso memoria GPU (ROCm)
- `vram_allocated_mb`, `vram_reserved_mb`, `vram_total_mb`, `vram_used_pct`: VRAM PyTorch
  (disponibili solo quando il modello è caricato)

Il widget GPU nella steering UI usa semaforo colore:
- verde: < 75°C
- giallo: 75–90°C
- rosso: ≥ 90°C

---

## 16) Prossimi passi
- Aggiungere concetti multipli con un runner generalizzato.
- A/B test automatico e confronto output.
- UI: confronto affiancato baseline vs inject.
- Nota modelli: rigenerare catalogo quando si cambia modello (hidden_size incompatibile).
