#!/bin/bash
# run_decompose_gd1_gemma2_ripresa.sh
#
# Ripresa batch Gd1 — solo Gemma2-Uncensored.
# Gemma3-1B-IT è già completo (9/9 concept).
#
# Uso:
#   nohup bash scripts/run_decompose_gd1_gemma2_ripresa.sh > /tmp/gemma2_ripresa4.log 2>&1 &
#   tail -f /tmp/gemma2_ripresa4.log

set -uo pipefail

PYTHON="/home/lele/codex-openai/project_jedi/.venv/bin/python"
SCRIPTS="/home/lele/codex-openai/project_jedi/scripts"
ROOT="/home/lele/codex-openai/project_jedi"
LOG_DIR="/tmp/decompose_gd1_gemma2_$(date +%Y%m%d_%H%M%S)"
MAIN_LOG="$LOG_DIR/batch_main.log"
STEERING_URL="http://localhost:8010"
M40_URL="http://localhost:11435"

mkdir -p "$LOG_DIR"

CONCEPTS=(
    "hot_vs_cold:sensoriale"
    "luce_vs_buio:sensoriale"
    "calma_vs_allerta:sensoriale"
    "liscio_vs_ruvido:sensoriale"
    "secco_vs_umido:sensoriale"
    "duro_vs_morbido:sensoriale"
    "rumore_vs_silenzio:uditivo"
    "dolce_vs_amaro:gustativo"
    "odore_forte_vs_inodore:olfattivo"
)

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MAIN_LOG"
}

check_server() {
    local url="$1" name="$2"
    if curl -sf "${url}/health" > /dev/null 2>&1 || \
       curl -sf "${url}/api/models" > /dev/null 2>&1; then
        log "  ✓ $name OK ($url)"
        return 0
    fi
    log "  ✗ $name non raggiungibile ($url)"
    return 1
}

rebuild_catalog() {
    log "Ricostruzione catalog.json..."
    "$PYTHON" "$SCRIPTS/build_catalog_multi.py" >> "$MAIN_LOG" 2>&1
    log "  ✓ Catalog aggiornato"
}

TOTAL=0
SUCCEEDED=0
FAILED=0
declare -a FAILED_LIST=()

log "============================================================"
log "  BATCH DECOMPOSE Gd1 — Gemma2-Uncensored (ripresa)"
log "  Log dir: $LOG_DIR"
log "============================================================"

check_server "$STEERING_URL" "Steering server" || { log "ABORT: avvia prima steering_server.py"; exit 1; }
check_server "$M40_URL"      "M40 llama-server" || { log "ABORT: avvia prima llama-server M40"; exit 1; }

# Carica Gemma2-Uncensored
log "Caricamento Gemma2-Uncensored..."
curl -sf -X POST "${STEERING_URL}/api/load_model" \
    -H "Content-Type: application/json" \
    -d '{"name": "Gemma2-Uncensored"}' > /dev/null 2>&1 || true

# Poll fino a caricamento (max 3 min)
waited=0
while [ $waited -lt 180 ]; do
    sleep 10
    waited=$((waited + 10))
    active=$(curl -sf "${STEERING_URL}/api/models" 2>/dev/null | \
             python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('active',''))" 2>/dev/null || echo "")
    if [ "$active" = "Gemma2-Uncensored" ]; then
        log "  ✓ Gemma2-Uncensored caricato (${waited}s)"
        break
    fi
done

sleep 5

for CONCEPT_CAT in "${CONCEPTS[@]}"; do
    CONCEPT="${CONCEPT_CAT%%:*}"
    CATEGORY="${CONCEPT_CAT##*:}"
    TOTAL=$((TOTAL + 1))

    log ""
    log "  ── Gemma2-Uncensored / ${CONCEPT} ──"
    LOG_FILE="${LOG_DIR}/Gemma2-Uncensored_${CONCEPT}.log"

    "$PYTHON" "$SCRIPTS/decompose.py" \
        --concept    "$CONCEPT" \
        --model      "Gemma2-Uncensored" \
        --category   "$CATEGORY" \
        --max-depth  0 \
        --max-iter   2 \
        --steering-url "$STEERING_URL" \
        --m40-url    "$M40_URL" \
        > "$LOG_FILE" 2>&1

    EXIT=$?
    if [ $EXIT -eq 0 ]; then
        SUCCEEDED=$((SUCCEEDED + 1))
        VERDICT=$(grep "VERDETTO:" "$LOG_FILE" 2>/dev/null | tail -1 || echo "")
        log "  ✓ OK  $VERDICT"
    else
        FAILED=$((FAILED + 1))
        FAILED_LIST+=("Gemma2-Uncensored/${CONCEPT}")
        LAST_ERR=$(tail -3 "$LOG_FILE" 2>/dev/null || echo "")
        log "  ✗ FAILED — ${LAST_ERR}"
    fi
done

rebuild_catalog

log ""
log "════════════════════════════════════════"
log "  BATCH COMPLETATO"
log "  Totale: $TOTAL  ✓ $SUCCEEDED  ✗ $FAILED"
if [ ${#FAILED_LIST[@]} -gt 0 ]; then
    log "  Falliti:"
    for f in "${FAILED_LIST[@]}"; do log "    - $f"; done
fi
log "  Log: $LOG_DIR/"
log "════════════════════════════════════════"

log "Git commit risultati..."
cd "$ROOT"
git add output/vector_library output/decompose_runs output/sub_concept_evals \
        output/catalog.json config/sub_concepts \
        2>/dev/null || true
git diff --cached --quiet && log "  (nessuna modifica)" || \
git commit -m "data: decompose Gd1 Gemma2 ripresa — ${SUCCEEDED}/${TOTAL} concept

Vettori Gd1 Gemma2-Uncensored estratti e validati.
$([ ${#FAILED_LIST[@]} -gt 0 ] && echo "Falliti: ${FAILED_LIST[*]}" || true)

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>" && \
git push origin main && log "  ✓ Pushato su GitHub" || log "  ⚠ Push fallito"
