#!/bin/bash
# probe_and_eval_missing.sh — Estrae i vettori Gemma2 mancanti poi lancia l'eval.
# Pensato per completare i 3 concept (duro, rumore, dolce) dopo l'interruzione.
#
# Uso: bash scripts/probe_and_eval_missing.sh

set -euo pipefail

ROOT=/home/lele/codex-openai/project_jedi
cd /home/lele/codex-openai

PYTHON=$ROOT/.venv/bin/python
LOG_PROBE=/tmp/probe_batch_gemma2.log
LOG_EVAL=/tmp/eval_batch_gemma2_final.log
PID_FILE_PROBE=/tmp/probe_batch_gemma2.pid
PID_FILE_EVAL=/tmp/eval_batch_gemma2_resume.pid
MISSING=(duro_vs_morbido rumore_vs_silenzio dolce_vs_amaro)

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "=== FASE 1: Probe vettori mancanti (Gemma2-Uncensored) ==="
log "Concepts: ${MISSING[*]}"
log "Log: $LOG_PROBE"

{
    echo "=== PROBE BATCH $(date) ==="
    # run_all_probes.sh skippa automaticamente i già estratti
    bash "$ROOT/scripts/run_all_probes.sh" --eval
    echo "=== PROBE COMPLETATO $(date) ==="
} > "$LOG_PROBE" 2>&1 &

PROBE_PID=$!
echo $PROBE_PID > "$PID_FILE_PROBE"
log "Probe avviato — PID: $PROBE_PID"

# Attende completamento probe
log "In attesa del completamento del probe..."
wait $PROBE_PID
PROBE_EXIT=$?

if [ $PROBE_EXIT -ne 0 ]; then
    log "ATTENZIONE: probe terminato con exit code $PROBE_EXIT — continuo comunque con eval"
fi

log ""
log "=== FASE 2: Eval dei 3 concept ora disponibili ==="

{
    echo "=== EVAL BATCH FINALE $(date) ==="
    for concept in "${MISSING[@]}"; do
        echo ""
        echo ">>> Avvio: $concept ($(date '+%H:%M:%S'))"
        bash "$ROOT/scripts/run_all_eval.sh" --concept "$concept" 2>&1
        echo "<<< Fine: $concept ($(date '+%H:%M:%S'))"
    done
    echo ""
    echo "=== EVAL COMPLETATO $(date) ==="
} > "$LOG_EVAL" 2>&1 &

EVAL_PID=$!
echo $EVAL_PID > "$PID_FILE_EVAL"
log "Eval avviato — PID: $EVAL_PID"
log "Log eval: $LOG_EVAL"
