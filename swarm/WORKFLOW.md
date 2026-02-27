# Swarm Workflow — Concept Steering (Jedi)

## Ruoli
- **Orchestrator**: coordina run, concetti, layer target.
- **Extractor**: esegue probing e genera vettori per layer.
- **Evaluator**: calcola proiezioni e ranking layer.
- **Cataloger**: aggiorna catalogo JSON con query e vettori.
- **UI Operator**: gestisce dashboard e console di steering.

## Sequenza
1) Extractor → `scripts/probe_hot_cold.py`
2) Evaluator → `scripts/eval_hot_cold.py`
3) Cataloger → `scripts/build_catalog.py`
4) UI Operator → `scripts/steering_server.py`

## Note
- Tutto resta in `project_jedi/`.
- Nessun file esterno modificato.
