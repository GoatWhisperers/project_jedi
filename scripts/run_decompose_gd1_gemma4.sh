#!/bin/bash
# run_decompose_gd1_gemma4.sh
#
# Esegue il decompose loop (Gd1) per tutti i 9 concept su Gemma4-E4B-IT.
# Stop a Gd1 (--max-depth 0): estrae e valida i sub-vettori.
#
# Prerequisiti:
#   - MI50 con Gemma4-E4B-IT caricato (porta 8020/8010)
#   - M40 llama-server attivo su porta 11435 (Gemma4-E4B-IT Q4_K_M)
#
# Uso:
#   nohup bash scripts/run_decompose_gd1_gemma4.sh > /tmp/decompose_gemma4_gd1.log 2>&1 &
#   tail -f /tmp/decompose_gemma4_gd1.log

set -uo pipefail

PYTHON="/home/lele/codex-openai/project_jedi/.venv/bin/python"
SCRIPTS="/home/lele/codex-openai/project_jedi/scripts"
ROOT="/home/lele/codex-openai/project_jedi"
LOG_DIR="/tmp/decompose_gemma4_gd1_$(date +%Y%m%d_%H%M%S)"
MAIN_LOG="$LOG_DIR/batch_main.log"
STEERING_URL="http://localhost:8010"
MANAGER_URL="http://localhost:8020"
M40_URL="http://localhost:11435"
MODEL="Gemma4-E4B-IT"

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
log "  BATCH DECOMPOSE Gd1 — Gemma4-E4B-IT — 9 concept"
log "  Log dir: $LOG_DIR"
log "============================================================"

# Verifica server
curl -sf "${STEERING_URL}/api/models" > /dev/null 2>&1 || { log "ABORT: steering server non raggiungibile (${STEERING_URL})"; exit 1; }
curl -sf "${MANAGER_URL}/api/status"  > /dev/null 2>&1 || { log "ABORT: MI50 manager non raggiungibile (${MANAGER_URL})"; exit 1; }
curl -sf "${M40_URL}/health"          > /dev/null 2>&1 || { log "ABORT: M40 llama-server non raggiungibile"; exit 1; }

# Verifica modello caricato (MI50 manager: /api/status → campo "model")
ACTIVE=$(curl -sf "${MANAGER_URL}/api/status" 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('model',''))" 2>/dev/null || echo "")
log "Modello attivo su MI50 manager: '$ACTIVE'"
if [ "$ACTIVE" != "$MODEL" ]; then
    log "  → Cambio modello a $MODEL..."
    curl -sf -X POST "${MANAGER_URL}/api/load_model" \
        -H "Content-Type: application/json" \
        -d "{\"name\": \"${MODEL}\"}" > /dev/null 2>&1 || true
    sleep 70
    log "  ✓ Modello cambiato"
fi

log ""
log "════════════════════════════════════════"
log "  MODELLO: $MODEL"
log "════════════════════════════════════════"

for CONCEPT_CAT in "${CONCEPTS[@]}"; do
    CONCEPT="${CONCEPT_CAT%%:*}"
    CATEGORY="${CONCEPT_CAT##*:}"
    TOTAL=$((TOTAL + 1))

    log ""
    log "  ── ${MODEL} / ${CONCEPT} ──"
    LOG_FILE="${LOG_DIR}/${CONCEPT}.log"

    "$PYTHON" "$SCRIPTS/decompose.py" \
        --concept    "$CONCEPT" \
        --model      "$MODEL" \
        --category   "$CATEGORY" \
        --max-depth  0 \
        --max-iter   2 \
        --steering-url "$STEERING_URL" \
        --manager-url  "$MANAGER_URL" \
        --m40-url    "$M40_URL" \
        > "$LOG_FILE" 2>&1

    EXIT=$?
    if [ $EXIT -eq 0 ]; then
        SUCCEEDED=$((SUCCEEDED + 1))
        VERDICT=$(grep "VERDETTO:" "$LOG_FILE" 2>/dev/null | tail -1 || echo "")
        log "  ✓ OK  $VERDICT"
    else
        FAILED=$((FAILED + 1))
        FAILED_LIST+=("${CONCEPT}")
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
log "  Log completi: $LOG_DIR/"
log "════════════════════════════════════════"

log ""
log "Git commit risultati..."
cd "$ROOT"
git add output/vector_library output/decompose_runs output/sub_concept_evals \
        output/catalog.json config/sub_concepts \
        2>/dev/null || true
git diff --cached --quiet && log "  (nessuna modifica da committare)" || \
git commit -m "data: decompose Gd1 Gemma4-E4B-IT — ${SUCCEEDED}/${TOTAL} concept completati

Vettori Gd1 estratti e validati per Gemma4-E4B-IT.
$([ ${#FAILED_LIST[@]} -gt 0 ] && echo "Falliti: ${FAILED_LIST[*]}" || true)

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>" && \
git push origin main && log "  ✓ Pushato su GitHub" || log "  ⚠ Push fallito"
