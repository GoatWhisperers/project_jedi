#!/bin/bash
# Run all missing concept extractions sequentially.
# Usage: bash scripts/run_all_probes.sh [--eval]
#
# This script runs probe_concept.py for every (concept, model) combination
# that doesn't yet have a vector in the vector library.
# It runs sequentially to avoid VRAM conflicts.
# Before running, it unloads the steering server model (port 8010) to free VRAM,
# then reloads it after all probes are done.

cd /home/lele/codex-openai

PYTHON=project_jedi/.venv/bin/python
PROBE=project_jedi/scripts/probe_concept.py
CONCEPTS_DIR=project_jedi/config/concepts
LIB=project_jedi/output/vector_library
EVAL_FLAG=${1:-""}   # pass --eval to enable held-out evaluation
STEERING_PORT=${STEERING_PORT:-8010}

log() { echo "[$(date '+%H:%M:%S')] $*"; }

ERRORS=0

# ── Unload steering server model to free VRAM ─────────────────────────────
unload_steering() {
    if curl -sf "http://localhost:$STEERING_PORT/api/model_info" > /dev/null 2>&1; then
        log "Unloading steering server model (freeing VRAM)..."
        curl -sf "http://localhost:$STEERING_PORT/api/unload_model" > /dev/null 2>&1 || true
        sleep 10   # wait for HIP/ROCm to actually release VRAM
    fi
}

# ── Reload steering server model after all probes ─────────────────────────
reload_steering() {
    if curl -sf "http://localhost:$STEERING_PORT/api/model_info" > /dev/null 2>&1; then
        local active_model
        active_model=$(curl -sf "http://localhost:$STEERING_PORT/api/models" 2>/dev/null \
            | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('active',''))" 2>/dev/null || true)
        if [ -n "$active_model" ] && [ "$active_model" != "None" ]; then
            log "Reloading steering server model: $active_model"
            curl -sf -X POST "http://localhost:$STEERING_PORT/api/load_model" \
                -H 'Content-Type: application/json' \
                -d "{\"name\":\"$active_model\"}" > /dev/null 2>&1 || true
        fi
    fi
}

run_probe() {
    local concept=$1
    local model=$2
    local json="$CONCEPTS_DIR/${concept}.json"

    if [ ! -f "$json" ]; then
        log "SKIP $concept — JSON not found"
        return
    fi

    # Derive model slug for path check
    local slug
    slug=$(echo "$model" | tr '[:upper:]' '[:lower:]' | tr ' ' '-' | tr '_' '-')
    local cat_dir
    cat_dir=$(python3 -c "import json; d=json.load(open('$json')); print(d.get('category','sensoriale'))")
    local lib_dir="$LIB/$cat_dir/$concept/$slug"

    if [ -d "$lib_dir" ] && ls "$lib_dir"/layer_*.npy 2>/dev/null | head -1 | grep -q .; then
        log "SKIP $concept / $model — already extracted ($lib_dir)"
        return
    fi

    log "START $concept / $model"
    if $PYTHON $PROBE --concept "$json" --model "$model" $EVAL_FLAG; then
        log "DONE  $concept / $model"
    else
        log "ERROR $concept / $model — probe failed (exit $?)"
        ERRORS=$((ERRORS + 1))
    fi
    sleep 5   # let GPU cool / free memory
}

log "=== Project Jedi — Batch Probe Run ==="
log "Models: Gemma3-1B-IT, Gemma2-Uncensored"
log "Eval: ${EVAL_FLAG:-disabled}"
log ""

# Free VRAM from steering server before starting
unload_steering

# --- Gemma3-1B-IT ---
log "--- Gemma3-1B-IT ---"
for concept in hot_vs_cold luce_vs_buio calma_vs_allerta liscio_vs_ruvido \
               secco_vs_umido duro_vs_morbido rumore_vs_silenzio dolce_vs_amaro \
               odore_forte_vs_inodore; do
    run_probe "$concept" "Gemma3-1B-IT"
done

# --- Gemma2-Uncensored ---
log "--- Gemma2-Uncensored ---"
for concept in hot_vs_cold luce_vs_buio calma_vs_allerta liscio_vs_ruvido \
               secco_vs_umido duro_vs_morbido rumore_vs_silenzio dolce_vs_amaro \
               odore_forte_vs_inodore; do
    run_probe "$concept" "Gemma2-Uncensored"
done

log ""
log "=== All probes done. Errors: $ERRORS. Rebuilding catalog... ==="
$PYTHON project_jedi/scripts/build_catalog_multi.py
log "=== Catalog rebuilt. ==="

# Reload steering server model
reload_steering
log "=== Done. ==="
