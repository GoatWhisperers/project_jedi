# Project Jedi — Probing Workflow (MI50)

Questo progetto esegue probing su hidden states profondi di un LLM usando PyTorch/Transformers
su MI50. Tutto resta dentro `project_jedi/`.

## Struttura
- `scripts/probe_hot_cold.py` — prova concetto caldo vs freddo su layer profondi
- `scripts/eval_hot_cold.py` — verifica con proiezioni layer-specifiche su query nuove
- `scripts/build_catalog.py` — aggiorna catalogo dall'ultimo run (con deduplication)
- `scripts/build_catalog_multi.py` — ricostruisce il catalogo da tutti i run (glob-based)
- `scripts/steering_server.py` — server steering con GPU monitor e model hot-swap
- `config/settings.json` — percorso modello, lista modelli e parametri runtime
- `output/` — risultati delle run
- `output/dashboard.html` — dashboard locale (legge `output/status.json`, card GPU)
- `ui/steering.html` — console steering con widget GPU e selezione modello

## Ambiente isolato
Venv isolato in `project_jedi/.venv` con pacchetti copiati da `/home/lele/venv-rocm311`.

## Uso
1) Imposta `model_path` e `concept_name` in `project_jedi/config/settings.json`.
2) Avvia il run:

```bash
cd /home/lele/codex-openai
./project_jedi/.venv/bin/python project_jedi/scripts/probe_hot_cold.py
```

3) Valuta proiezioni:

```bash
./project_jedi/.venv/bin/python project_jedi/scripts/eval_hot_cold.py
```

4) Catalogo:

```bash
./project_jedi/.venv/bin/python project_jedi/scripts/build_catalog.py
# oppure ricostruisci tutto
./project_jedi/.venv/bin/python project_jedi/scripts/build_catalog_multi.py
```

## Dashboard

```bash
cd /home/lele/codex-openai/project_jedi/output
python3 -m http.server 8001
```

Apri: `http://<server>:8001/dashboard.html`

## Output
I risultati finiscono in `project_jedi/output/run_YYYYMMDD_HHMMSS/`:
- `summary.json`
- `settings_used.json`
- `queries_hot.json`
- `queries_cold.json`
- `concept_<concept>_layer_*.npy`
- `projections_eval.json`

Il progresso live è in `project_jedi/output/status.json`.


## Modelli disponibili

| Nome | Path | Layer | Hidden Size |
|------|------|-------|-------------|
| Gemma3-1B-IT | `/mnt/raid0/gemma-3-1b-it` | 26 | 1152 |
| Gemma2-Uncensored | `/mnt/raid0/gemma-2-uncensored` | 42 | 3584 |

> I vettori estratti sono model-specific: hidden_size diverso → non intercambiabili.

## Steering Console (Chat + Injection)

Avvio server:
```bash
cd /home/lele/codex-openai
./project_jedi/.venv/bin/python project_jedi/scripts/steering_server.py
```

Apri: `http://<server>:8010/`

La UI mostra il prompt formattato e consente injection su layer bassi usando un vettore estratto.
L'header include un **widget GPU** (temp + VRAM%) e un **selector modello** per hot-swap a runtime.

### API principali

| Endpoint | Metodo | Descrizione |
|----------|--------|-------------|
| `/api/gpu` | GET | Stats GPU: temp rocm-smi + VRAM PyTorch |
| `/api/models` | GET | Lista modelli + modello attivo |
| `/api/load_model` | POST | Hot-swap modello `{"name":"..."}` |
| `/api/generate` | POST | Genera testo con/senza injection |

### Selezione modello dalla UI
1. Aprire `http://<server>:8010/`
2. Header → dropdown **Model** → selezionare modello
3. Premere **Load** (spinner durante caricamento)
4. Layer dropdown si aggiornano automaticamente

## Bug fix applicati
- **B1** `eval_hot_cold.py`: usa `extract_all_layers()` — una forward pass, rappresentazioni layer-specifiche
- **B2** `build_catalog.py`: deduplicazione per `(run_dir, concept)` prima di aggiungere
- **B3** `build_catalog_multi.py`: glob + regex per path `.npy` reali invece di path costruiti

