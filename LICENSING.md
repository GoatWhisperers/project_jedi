# Licensing

Project Jedi uses a **dual-license model** that reflects its dual nature:
open science infrastructure paired with proprietary research content.

---

## TL;DR

| What | License | You can... |
|------|---------|-----------|
| Source code | Apache 2.0 | Use, modify, fork, embed — freely, including commercially |
| Datasets, vectors, protocol | CC BY-NC-SA 4.0 | Research & teaching with citation — not for commercial use |

---

## Source Code — Apache 2.0

**Applies to:** `scripts/` and `ui/`

```
scripts/probe_concept.py
scripts/probe_server.py
scripts/steering_server.py
scripts/auto_eval.py
scripts/build_catalog_multi.py
scripts/run_all_eval.sh
scripts/run_all_probes.sh
ui/probe_dashboard.html
ui/steering.html
```

The infrastructure is open — you can run it on your own models, fork it,
integrate it into other projects, or build commercial tools on top of it.
Attribution appreciated, not required.

Full text: [LICENSE](LICENSE)

---

## Research Content — CC BY-NC-SA 4.0

**Applies to:** datasets, vectors, evaluation protocol, and findings

```
config/concepts/              phenomenological sentence datasets (9 concepts × 1000 sentences)
config/eval_concepts/         blind evaluation protocol definitions
output/vector_library/        extracted concept vectors (.npy)
output/eval_sessions/         blind evaluation session results
experiments/                  research diary and architectural notes
concept_generation_prompt.md  generation protocol for phenomenological datasets
LLM_Internal_Probing_Reference.md
*.md documentation
```

These are the intellectual core of the project. The phenomenological datasets
took significant effort to design and generate correctly. The evaluation protocol
and the concept vectors represent original research contributions.

Full text: [LICENSE-DATA](LICENSE-DATA)

---

## What this means in practice

### Academic researcher
You want to use the concept vectors in a paper comparing steering techniques
across architectures.

**Permitted.** Use the vectors, cite the repo:

> GoatWhisperers, "Project Jedi: Phenomenological Concept Vectors for
> Transformer Steering," 2026. https://github.com/GoatWhisperers/project_jedi

### University course
You want to use the datasets and probe scripts as a teaching example of
linear representation in transformers.

**Permitted.** No need to ask, just attribute.

### Personal/hobby project
You want to run the steering console on a different model and experiment with
the concept vectors.

**Permitted.** Have fun.

### Company building a product
You want to integrate the phenomenological datasets into a commercial
feature (e.g., emotional tone analysis in a SaaS product).

**Not permitted under CC BY-NC-SA.** Contact us to discuss a commercial license.

### Company using the code
You want to build a commercial product using the probe server infrastructure.

**Permitted under Apache 2.0** (code only — datasets still CC BY-NC-SA).

---

## Contact

For commercial licensing inquiries: open an issue on
[GitHub](https://github.com/GoatWhisperers/project_jedi) or contact the
GoatWhisperers organization directly.
