# Project Jedi

**Concept vector extraction and activation steering for transformer language models.**

Extract linear concept directions from hidden states, then inject them during inference to steer model outputs toward specific sensory or semantic concepts — without fine-tuning.

---

## Table of Contents

1. [Theory](#theory)
2. [Project Structure](#project-structure)
3. [Quick Start](#quick-start)
4. [Concept Datasets](#concept-datasets)
5. [Probe Dashboard (port 8000)](#probe-dashboard)
6. [Steering Console (port 8010)](#steering-console)
7. [Configuration](#configuration)
8. [API Reference](#api-reference)
9. [Results](#results)
10. [Hardware & Requirements](#hardware--requirements)
11. [License](#license)

---

## Theory

### What are concept vectors?

A large language model, while processing text, builds internal representations in its hidden layers. These representations are high-dimensional vectors — one per token, per layer. At deep layers, these vectors encode rich semantic information: meaning, syntax, sentiment, and sensory qualities.

**Concept vectors** are directions in this hidden-state space that correspond to specific concepts. If such a direction exists and is linear, then:

- Sentences about *heat* cluster on one side of that direction
- Sentences about *cold* cluster on the other side
- The model's representation of any text can be decomposed into its projection onto this direction

### How we extract them: mean-diff

Given a set of *positive* sentences `P` (e.g., "her fingers burned on the hot stove") and *negative* sentences `N` (e.g., "she numbed her hands packing snow"):

1. Run both sets through the model (forward pass only — no generation)
2. At each layer `L`, collect the hidden state for each sentence, pooled over all tokens: `h_i ∈ ℝ^d`
3. Compute the **mean-diff** vector:

```
v_L = mean({h_i : i ∈ P}) − mean({h_j : j ∈ N})
v_L = v_L / ‖v_L‖         # unit-normalize
```

This direction `v_L` is the concept vector for layer `L`. It points from the "negative" cluster toward the "positive" cluster in the model's internal representation space.

### Why deep layers?

- **Early layers** (L0–L10): Encode surface form — tokenization, syntax. Very stable (bootstrap cosine ≈ 0.999) but capture spurious correlations, not true semantics. Concept direction is unstable on held-out data.
- **Mid layers** (L10–L18): Transition zone. Semantic signal begins to emerge.
- **Deep layers** (L18–L26 for Gemma3, L29–L38 for Gemma2): Encode rich semantic content. Less perfectly stable but semantically correct. **These are the layers we use.**

The `deep_range` parameter (default `[0.70, 0.90]`) automatically selects 70–90% of the total layers, which corresponds to the empirically best zone.

### Convergence validation

We don't just compute the vector — we validate it is **stable** and **reproducible**:

**Bootstrap stability:** Repeat 30 times — randomly select half the training sentences, compute the concept vector on that subset, measure cosine similarity to the full-data vector. The minimum across 30 runs (`boot_min`) tells us the worst-case stability.

| boot_min | Meaning |
|----------|---------|
| > 0.95 | Excellent — vector stable across any random subset |
| 0.85–0.95 | Good — usable for steering |
| 0.70–0.85 | Borderline |
| < 0 | Sign-unstable — vector is noise |

**Held-out SNR:** The real test. Project 5–10 held-out sentences (not in training) onto the concept vector. If positive sentences score higher than negative, the direction is semantically correct. If inverted: wrong direction (early-layer trap).

### Activation steering

Once we have a concept vector `v` for layer `L`, we can steer the model during generation by injecting it into the forward pass via a hook:

```
h_new_tokens += α × gain × v
```

Where:
- `α ∈ [-2, +2]`: direction (positive = toward concept, negative = away)
- `gain ∈ [1, 2000]`: amplification factor
- Applied only to newly generated tokens (`apply_to="new"`)

This causes the model to generate text biased toward (or away from) the concept, without any fine-tuning or prompt modification.

---

## Project Structure

```
project_jedi/
│
├── config/
│   ├── settings.json              # Model paths, runtime parameters
│   └── concepts/                  # Concept datasets (500+500 sentences each)
│       ├── hot_vs_cold.json
│       ├── luce_vs_buio.json
│       ├── calma_vs_allerta.json
│       ├── liscio_vs_ruvido.json
│       ├── secco_vs_umido.json
│       ├── duro_vs_morbido.json
│       ├── rumore_vs_silenzio.json
│       ├── dolce_vs_amaro.json
│       └── odore_forte_vs_inodore.json
│
├── scripts/
│   ├── probe_hot_cold.py          # Core extraction (convergence, mean-diff, PCA)
│   ├── probe_concept.py           # Generic extractor → vector_library/
│   ├── eval_hot_cold.py           # Evaluate vectors on held-out sentences
│   ├── build_catalog.py           # Incremental catalog update
│   ├── build_catalog_multi.py     # Full catalog rebuild (all runs + library)
│   ├── probe_server.py            # HTTP server port 8000 (probe dashboard)
│   └── steering_server.py         # HTTP server port 8010 (steering console)
│
├── ui/
│   ├── probe_dashboard.html       # Probe control UI
│   └── steering.html              # Steering chat UI
│
├── output/
│   ├── catalog.json               # Index of all extracted vectors
│   ├── vector_library/            # Organized concept vectors
│   │   └── {category}/
│   │       └── {concept}/
│   │           └── {model}/
│   │               ├── layer_N.npy       # Mean-diff vector (primary)
│   │               ├── layer_N_pca.npy   # PCA vector (backup)
│   │               ├── meta.json
│   │               ├── summary.json      # Convergence stats per layer
│   │               └── eval.json         # Held-out SNR (if --eval)
│   └── run_YYYYMMDD_HHMMSS/       # Legacy run directories
│
├── concept_generation_prompt.md   # Prompt template for dataset generation
└── LLM_Internal_Probing_Reference.md  # Technical reference
```

---

## Quick Start

### Prerequisites

```bash
# GPU: AMD MI50 with ROCm
# Python: 3.11 in project_jedi/.venv
# Models: /mnt/raid0/gemma-3-1b-it and /mnt/raid0/gemma-2-uncensored
```

### 1. Extract concept vectors

```bash
cd /home/lele/codex-openai

# Extract a concept (e.g., hot_vs_cold) with Gemma3-1B
project_jedi/.venv/bin/python project_jedi/scripts/probe_concept.py \
  --concept project_jedi/config/concepts/hot_vs_cold.json

# With evaluation on held-out sentences
project_jedi/.venv/bin/python project_jedi/scripts/probe_concept.py \
  --concept project_jedi/config/concepts/hot_vs_cold.json \
  --eval

# With a specific model
project_jedi/.venv/bin/python project_jedi/scripts/probe_concept.py \
  --concept project_jedi/config/concepts/hot_vs_cold.json \
  --model "Gemma2-Uncensored" \
  --eval
```

Output goes to `output/vector_library/{category}/{concept}/{model}/`

### 2. Update the catalog

```bash
# After extraction, rebuild the catalog
project_jedi/.venv/bin/python project_jedi/scripts/build_catalog_multi.py
```

### 3. Start the servers

```bash
# Probe dashboard (concept extraction control)
project_jedi/.venv/bin/python project_jedi/scripts/probe_server.py &

# Steering console (chat + injection)
project_jedi/.venv/bin/python project_jedi/scripts/steering_server.py &
```

Then open:
- `http://<server-ip>:8000` — Probe Dashboard
- `http://<server-ip>:8010` — Steering Console

---

## Concept Datasets

Each concept JSON in `config/concepts/` contains 500 positive + 500 negative sentences in 6 languages (Italian, English, French, German, Spanish, Latin), generated with Claude Opus following strict perceptual/phenomenological guidelines.

**Key principle:** Sentences must describe **direct bodily/perceptual experience**, not environments or metaphors.

| Works ✓ | Fails ✗ |
|---------|---------|
| "Le pupille si strinsero istantaneamente uscendo dal tunnel alla luce piena." | "La stanza era avvolta da luce dorata al mattino." |
| "He held his hand an inch from his face and could not see it at all." | "The future looked bright." |
| "Adeo connivebat propter lucem ut capitis dolor ei oreretur." | "Sein Herz war kalt wie Eis." |

### Available concepts

| Concept | Category | Positive | Negative |
|---------|----------|----------|----------|
| hot_vs_cold | thermal | Heat, burning, sweat, vasodilation | Cold, numbness, shivers, vasoconstriction |
| luce_vs_buio | visual | Glare, pupils contracting, afterimage | Total darkness, pupils dilated, visual void |
| calma_vs_allerta | autonomic | Slow breathing, low heart rate, relaxed muscles | Racing heart, short breath, hypervigilance |
| liscio_vs_ruvido | tactile | Frictionless surfaces, silk, glass | Friction, abrasion, micro-pain, rough edges |
| secco_vs_umido | tactile | Dry skin, cracked lips, arid air | Condensation, wet clothes, saturated air |
| duro_vs_morbido | tactile | Rigid surfaces, bone, metal, stone | Soft yielding, cushion, relaxed muscle |
| rumore_vs_silenzio | auditory | Noise overload, ear pain, tinnitus | Total silence, pressure, absence of sound |
| dolce_vs_amaro | gustatory | Salivation, sweetness on tongue tip | Contraction, grimace, persistent bitter aftertaste |
| odore_forte_vs_inodore | olfactory | Burning nostrils, receptor saturation | Neutral air, no olfactory response |

---

## Probe Dashboard

**URL:** `http://<server>:8000`

The probe dashboard lets you control concept extraction from the browser.

### Features

- **Concept selector** — dropdown with color-coded status:
  - 🟢 Not yet extracted (todo)
  - 🟡 Extracted, good quality (boot_min ≥ 0.85)
  - 🔴 Extracted but poor quality (boot_min < 0.85, needs better dataset)
- **Model selector** — choose between available models
- **Eval checkbox** — run held-out evaluation after extraction
- **Live log** — real-time stdout from probe_concept.py
- **Progress bars** — Phase / Query / Layer with ETA and throughput
- **Hardware stats** — GPU temp, GPU load %, VRAM, CPU %, RAM (updated every 2s)
- **Vector library** — table of all extracted concepts with boot_min scores

### Workflow

1. Select a concept (🟢 = not done yet)
2. Select model (Gemma3-1B-IT or Gemma2-Uncensored)
3. Check "eval" if you want held-out SNR
4. Click ▶ Start Probe
5. Watch the log — look for `boot_min` values per layer
6. When done, the concept turns 🟡 if boot_min ≥ 0.85

### Reading the log

```
[1/4] Extracting POS representations (500 sentences)...
      shape: (500, 1152)   ← all sentences processed, hidden_dim=1152

[2/4] Extracting NEG representations (500 sentences)...

[3/4] Computing concept vectors...
  Layer 18: boot(mean)=0.9134  boot(min)=0.8821  pca≈mean=0.312  converged=NO
  Layer 21: boot(mean)=0.9712  boot(min)=0.9506  pca≈mean=0.951  converged=NO
  Layer 23: boot(mean)=0.9501  boot(min)=0.8974  pca≈mean=0.884  converged=NO

[4/4] Done.
  Best: layer 21 (boot_min=0.9506)
  Output: output/vector_library/thermal/hot_vs_cold/gemma3-1b-it/
```

---

## Steering Console

**URL:** `http://<server>:8010`

The steering console lets you chat with the model while injecting concept vectors.

### Features

- **Hardware stats** — 5 cards: GPU Edge temp, GPU Load %, VRAM (GB used/total), CPU %, RAM %
- **Model selector** — hot-swap between models (Gemma3-1B-IT / Gemma2-Uncensored)
- **Concept dropdown** — shows concepts available for the loaded model
- **↺ Reload catalog** — refresh concept list after new extraction
- **Alpha slider** — direction and intensity (`+1.0` = toward concept, `-1.0` = away)
- **Layer selector** — choose which layer to inject
- **Multi-layer** — inject on all available layers simultaneously
- **Baseline button** — generate without injection (control)
- **Inject button** — generate with concept vector injected

### Steering parameters

| Parameter | Range | Effect |
|-----------|-------|--------|
| alpha | -2.0 to +2.0 | Direction × intensity. `+1.0` = toward concept, `-1.0` = away |
| gain | 1 to 2000 | Amplification. Below 400 = too weak. 400–1000 = good range |
| layer | model-dependent | Which transformer layer to inject. Use best layer from probe |
| multi | checkbox | Use all available layers at once. Often strongest effect |
| apply_to | "new" | Applied only to generated tokens (not the prompt) |

### Tested sweet spots (Gemma3-1B-IT, hot_vs_cold)

| Configuration | Effect |
|---------------|--------|
| L21, gain=1000, alpha=+1.0 | Strong heat — "warm ember", "scorched earth" |
| L21, gain=1000, alpha=-1.0 | Cold — "shards of ice", "grey walls", "absence of light" |
| Multi L20+21+22, gain=400, alpha=+1.0 | Diffuse warmth across output register |

### Important: model-specific vectors

Vectors extracted from Gemma3-1B (hidden_dim=1152) **cannot** be used with Gemma2-Uncensored (hidden_dim=3584). The steering console automatically shows only concepts available for the currently loaded model.

---

## Configuration

### config/settings.json

```json
{
  "model_path": "/mnt/raid0/gemma-3-1b-it",
  "models": [
    {"name": "Gemma3-1B-IT",      "path": "/mnt/raid0/gemma-3-1b-it"},
    {"name": "Gemma2-Uncensored", "path": "/mnt/raid0/gemma-2-uncensored"}
  ],
  "trust_remote_code": false,
  "max_length": 128,
  "batch_size": 1,
  "token_position": "mean",
  "deep_range": [0.70, 0.90],
  "dtype": "bfloat16",
  "device": "cuda"
}
```

| Parameter | Values | Notes |
|-----------|--------|-------|
| `token_position` | `"mean"` / `"last"` / `"first"` | How to pool hidden states. **mean = recommended** |
| `deep_range` | `[0.0, 1.0]` × 2 | Fraction of layers to scan. `[0.70, 0.90]` = deep 20% |
| `dtype` | `"bfloat16"` / `"float16"` / `"float32"` | **bfloat16 required for Gemma2** (halves VRAM) |
| `batch_size` | 1 | Sentences per forward pass. Keep at 1 for MI50 |
| `max_length` | 128 | Max tokens per sentence during extraction |

### Concept JSON format

```json
{
  "concept": "hot_vs_cold",
  "category": "thermal",
  "description": "Direct physical experience of heat vs cold...",
  "pos_label": "hot",
  "neg_label": "cold",
  "positive": ["sentence 1", "sentence 2", ...],
  "negative": ["sentence 1", "sentence 2", ...]
}
```

---

## API Reference

### Probe Server (port 8000)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Probe dashboard UI |
| GET | `/api/concepts` | List concepts with status (todo/good/bad), boot_min |
| GET | `/api/models` | Available models + active model |
| GET | `/api/stats` | CPU %, RAM, GPU temp/load/VRAM |
| GET | `/api/log?n=100` | Last N log lines from running probe |
| GET | `/api/library` | Vector library overview |
| GET | `/api/stop` | Terminate running probe |
| POST | `/api/probe` | Start extraction: `{"concept_path": "...", "model": "...", "eval": true}` |
| POST | `/api/upload_concept` | Upload new concept JSON |
| POST | `/api/set_model` | Set active model |

### Steering Server (port 8010)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Steering console UI |
| GET | `/api/concepts` | Concepts available for loaded model |
| GET | `/api/concept_layers?concept=X` | Layer indices for concept X |
| GET | `/api/models` | Available models + active |
| GET | `/api/model_info` | num_layers, device |
| GET | `/api/gpu` | GPU edge/junction temp, load %, VRAM, CPU %, RAM |
| GET | `/api/library` | Vector library overview |
| GET | `/api/reload_catalog` | Reload catalog.json at runtime |
| POST | `/api/generate` | Generate text with/without injection |
| POST | `/api/load_model` | Hot-swap model: `{"name": "Gemma2-Uncensored"}` |

#### POST /api/generate

```json
{
  "messages": [{"role": "user", "content": "Describe the desert at noon."}],
  "concept": "hot_vs_cold",
  "vector_layer": 21,
  "inject_layer": 21,
  "alpha": 1.0,
  "gain": 400,
  "max_new_tokens": 128,
  "mode": "inject",
  "multi": false
}
```

Response:
```json
{
  "text": "The sand burned underfoot...",
  "formatted_prompt": "USER: Describe the desert at noon.\nASSISTANT:"
}
```

---

## Results

### Best concept vectors (boot_min)

| Concept | Model | boot_min | Notes |
|---------|-------|----------|-------|
| calma_vs_allerta | Gemma2-Uncensored | **0.983** ★ | Near-perfect stability |
| liscio_vs_ruvido | Gemma2-Uncensored | **0.981** ★ | |
| luce_vs_buio | Gemma2-Uncensored | **0.949** | |
| hot_vs_cold | Gemma2-Uncensored | 0.895 | 101 sentences; will improve with 500 |
| luce_vs_buio | Gemma3-1B-IT | 0.865 | |

Gemma2-Uncensored (42 layers, 3584-dim) consistently outperforms Gemma3-1B-IT for sensory concepts.

### Key finding: stability ≠ correctness

Early layers (L5–L13) show `boot_min > 0.995` — seemingly perfect. But held-out SNR is **negative** — the direction is wrong. They capture corpus-level statistical patterns, not semantic content.

Deep layers (L18–L23 for Gemma3, L29–L38 for Gemma2) show `boot_min ≈ 0.89–0.98` — slightly less stable, but semantically correct on held-out data.

**Always validate with `--eval` before using vectors for steering.**

---

## Hardware & Requirements

| Component | Spec |
|-----------|------|
| GPU | AMD MI50, 32 GB VRAM |
| ROCm | Required for PyTorch GPU support |
| Python | 3.11 (venv at `project_jedi/.venv`) |
| Gemma3-1B-IT | ~2.5 GB VRAM in bfloat16 |
| Gemma2-Uncensored | ~14 GB VRAM in bfloat16 |
| Extraction time | ~5 min for 500 sentences × 10 layers |
| Generation latency | ~2–3s per response |

> **Note:** `dtype: "bfloat16"` is required in settings.json. float32 uses ~28 GB for Gemma2 and causes OOM.

---

## License

Project Jedi uses a dual-license model:

| What | License |
|------|---------|
| Source code (`scripts/`, `ui/`) | [Apache 2.0](LICENSE) — use freely, including commercially |
| Datasets, vectors, protocol, research content | [CC BY-NC-SA 4.0](LICENSE-DATA) — research & teaching with citation; no commercial use |

**Academic use:** Cite as:
> GoatWhisperers, "Project Jedi: Phenomenological Concept Vectors for Transformer Steering," 2026.
> https://github.com/GoatWhisperers/project_jedi

**Commercial use of research content:** Contact us via [GitHub Issues](https://github.com/GoatWhisperers/project_jedi/issues).

See [LICENSING.md](LICENSING.md) for the full breakdown and use-case guide.

---

## GitHub

Repository: [https://github.com/GoatWhisperers/project_jedi](https://github.com/GoatWhisperers/project_jedi)
