# Project Jedi — Documentazione Operativa (Run Caldo/Freddo)

Data: 2026-02-26

Questa documentazione descrive **quanto realizzato** finora in `project_jedi/`, con focus su:
- probing di concetto **caldo ↔ freddo**
- workflow su MI50 (ROCm) con modello **Gemma3‑1B** in formato HF/Transformers
- catalogazione dei vettori concettuali e dashboard locale

Tutto il lavoro è **isolato** nella cartella `project_jedi/`.

---

## 1) Obiettivo
Estrarre e catalogare **vettori concettuali** dai layer profondi di un LLM, usando un set contrastivo di frasi **calde** vs **fredde**. Il risultato è una **direzione** nello spazio latente per ciascun layer selezionato:

- proiezione **positiva** → più “caldo”
- proiezione **negativa** → più “freddo”

---

## 2) Modello e Hardware
- GPU: **MI50** (ROCm)
- Modello default: **Gemma3‑1B-IT** (HF/Transformers)

### Modelli disponibili

| Nome | Path | Layer | Hidden Size |
|------|------|-------|-------------|
| Gemma3-1B-IT | `/mnt/raid0/gemma-3-1b-it` | 26 | 1152 |
| Gemma2-Uncensored | `/mnt/raid0/gemma-2-uncensored` | 42 | 3584 |

> I vettori nel catalogo sono **model-specific** (hidden_size diverso). Non intercambiabili tra modelli.

Nota: la M40 non è adatta a questo workflow perché il backend GGUF/llama.cpp non espone `hidden_states`.

---

## 3) Ambiente isolato
Venv creato e **copiato** da un ambiente ROCm funzionante:

- `project_jedi/.venv`
- Pacchetti copiati da `/home/lele/venv-rocm311`

Questo evita download esterni e mantiene l’isolamento del progetto.

---

## 4) Struttura file
- `project_jedi/config/settings.json`
  - parametri runtime (model_path, **models list**, batch_size, deep_range, concept_name)
- `project_jedi/config/queries_hot.json`
- `project_jedi/config/queries_cold.json`
  - set di frasi multilingua (latino + lingue meno comuni)
- `project_jedi/scripts/probe_hot_cold.py`
  - estrae vettori concettuali per layer
- `project_jedi/scripts/eval_hot_cold.py`
  - calcola proiezioni su frasi nuove e ranking layer
- `project_jedi/scripts/build_catalog.py`
  - aggiorna catalogo concetti dall’ultimo run
- `project_jedi/scripts/build_catalog_multi.py`
  - ricostruisce catalogo da tutti i run
- `project_jedi/output/`
  - risultati delle run, dashboard, status

---

## 5) Run eseguita (multilingua)
Ultimo run:
- `project_jedi/output/run_20260226_182542/`

Contenuto:
- `concept_hot_vs_cold_layer_18.npy`
- `concept_hot_vs_cold_layer_19.npy`
- `concept_hot_vs_cold_layer_20.npy`
- `concept_hot_vs_cold_layer_21.npy`
- `concept_hot_vs_cold_layer_22.npy`
- `concept_hot_vs_cold_layer_23.npy`
- `summary.json`
- `queries_hot.json`
- `queries_cold.json`
- `projections_eval.json`

**Numero vettori per file:** 1 vettore (shape `(1152,)`) per layer.

---

## 6) Risultati sintetici
Layer profondi selezionati (70–90%): **18–23**.

Ranking (separazione hot‑cold) dal file `projections_eval.json`:
- layer 23 (score ≈ 23.143)
- layer 21 (score ≈ 22.946)
- layer 22 (score ≈ 22.757)

I layer **non vengono scartati**: il ranking serve solo come evidenza visiva.

---

## 7) Dashboard locale
File:
- `project_jedi/output/dashboard.html`

Mostra:
- stato run (`status.json`)
- grafico separazione per layer
- top layer evidenziati
- **card GPU**: temp edge/junction e VRAM% (richiede steering server su :8010 attivo)

Avvio server locale:
```bash
cd /home/lele/codex-openai/project_jedi/output
python3 -m http.server 8001
```

Apri:
```
http://<server>:8001/dashboard.html
```

---

## 8) Catalogo concetti
Catalogo cumulativo:
- `project_jedi/output/catalog.json`

Contiene:
- nome concetto
- run dir
- modello
- layer
- path ai vettori (scoperti via glob per naming corretto)
- query usate
- metriche

**Fix B2:** `build_catalog.py` deduplica per `(run_dir, concept)` — rimpiazza se esiste.
**Fix B3:** `build_catalog_multi.py` usa glob + regex sui filename reali invece di path costruiti.

---

## 9) Comandi usati
Esecuzione run:
```bash
cd /home/lele/codex-openai
./project_jedi/.venv/bin/python project_jedi/scripts/probe_hot_cold.py
```

Valutazione proiezioni:
```bash
./project_jedi/.venv/bin/python project_jedi/scripts/eval_hot_cold.py
```

Aggiorna catalogo:
```bash
./project_jedi/.venv/bin/python project_jedi/scripts/build_catalog.py
```

---

## 10) Prossimi passi (domani)
1. **Prova inversa**: usare il vettore con segno opposto per confermare l’effetto.
2. **Iniezione concetto durante inferenza**:
   - inserire il vettore in un layer selezionato
   - osservare effetti sull’output
   - confrontare con baseline

Per l’iniezione servirà un runner dedicato (hook sui layer), da mantenere dentro `project_jedi/`.

---


## 11) Steering Console (Chat + Iniezione)

Console web per testare l’influenza dei vettori durante una **chat** con prompt formattato.
La console mostra il **prompt finale** (con template del tokenizer se disponibile) e l’output.

- Server: `project_jedi/scripts/steering_server.py`
- UI: `project_jedi/ui/steering.html`
- Log: `project_jedi/output/steering_log.jsonl`

Avvio:
```bash
cd /home/lele/codex-openai
./project_jedi/.venv/bin/python project_jedi/scripts/steering_server.py
```

Apri:
```
http://<server>:8010/
```

Funzioni:
- **Baseline**: chat normale senza iniezione
- **Inject**: iniezione vettore concettuale
- **Vector Layer (source)**: layer da cui proviene il vettore concettuale
- **Inject Layer (target)**: layer del modello dove iniettare (anche basso)
- **Alpha**: intensità iniezione
- **Widget GPU** (header): temp edge, junction, VRAM% — aggiornato ogni 5s
- **Model selector** (header): dropdown modelli + pulsante Load per hot-swap

### API Endpoints

| Metodo | Path | Descrizione |
|--------|------|-------------|
| GET | `/api/gpu` | Stats GPU (rocm-smi + PyTorch VRAM) |
| GET | `/api/models` | Lista modelli + modello attivo |
| POST | `/api/load_model` | Hot-swap modello `{"name":"..."}` |

### Risposta /api/gpu (con modello caricato)
```json
{
  "ok": true,
  "edge_c": 72.0,
  "junction_c": 75.0,
  "mem_use_pct": 45.0,
  "vram_allocated_mb": 2048.0,
  "vram_reserved_mb": 2200.0,
  "vram_total_mb": 16384.0,
  "vram_used_pct": 12.5
}
```

### Risposta /api/models
```json
{
  "models": [
    {"name": "Gemma3-1B-IT", "path": "/mnt/raid0/gemma-3-1b-it"},
    {"name": "Gemma2-Uncensored", "path": "/mnt/raid0/gemma-2-uncensored"}
  ],
  "active": "Gemma3-1B-IT"
}
```

### Fix B1: eval_hot_cold.py
La valutazione usa ora `extract_all_layers()`: una sola forward pass restituisce
hidden states di tutti i layer. Per ogni layer viene usata la rappresentazione
del **medesimo layer** — non più sempre l’ultimo.

