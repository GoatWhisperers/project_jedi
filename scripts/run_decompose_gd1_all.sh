#!/bin/bash
# run_decompose_gd1_all.sh
#
# Esegue il decompose loop (Gd1) per tutti i 9 concept × 2 modelli, in sequenza.
# Stop a Gd1 (--max-depth 0): estrae e valida i sub-vettori senza scendere a Gd2.
#
# Prerequisiti:
#   - Steering server attivo su porta 8010 (gestisce cambio modello via API)
#   - M40 llama-server attivo su porta 11435 (Gemma3-12B Q4_K_M)
#
# Uso:
#   nohup bash scripts/run_decompose_gd1_all.sh > /tmp/decompose_batch.log 2>&1 &
#   tail -f /tmp/decompose_batch.log
#   bash scripts/monitor_decompose.sh           # monitoraggio separato

set -uo pipefail

PYTHON="/home/lele/codex-openai/project_jedi/.venv/bin/python"
SCRIPTS="/home/lele/codex-openai/project_jedi/scripts"
ROOT="/home/lele/codex-openai/project_jedi"
LOG_DIR="/tmp/decompose_gd1_$(date +%Y%m%d_%H%M%S)"
MAIN_LOG="$LOG_DIR/batch_main.log"
STEERING_URL="http://localhost:8010"
M40_URL="http://localhost:11435"

mkdir -p "$LOG_DIR"

# ── Concept list: "slug:category" ────────────────────────────────────────────
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

# ── Helpers ───────────────────────────────────────────────────────────────────
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

load_model_steering() {
    local model="$1"
    log "  → Caricamento modello '$model' su steering server..."
    curl -sf -X POST "${STEERING_URL}/api/load_model" \
        -H "Content-Type: application/json" \
        -d "{\"name\": \"${model}\"}" > /dev/null 2>&1 || true

    # Attende che il modello sia caricato (poll /api/model_info ogni 10s, max 3 min)
    local waited=0
    while [ $waited -lt 180 ]; do
        sleep 10
        waited=$((waited + 10))
        local active
        active=$(curl -sf "${STEERING_URL}/api/models" 2>/dev/null | \
                 python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('active',''))" 2>/dev/null || echo "")
        if [ "$active" = "$model" ]; then
            log "  ✓ Modello '$model' caricato (${waited}s)"
            return 0
        fi
    done
    log "  ⚠ Timeout caricamento '$model' — continuo comunque"
}

rebuild_catalog() {
    log "Ricostruzione catalog.json..."
    "$PYTHON" "$SCRIPTS/build_catalog_multi.py" >> "$MAIN_LOG" 2>&1
    log "  ✓ Catalog aggiornato"
}

# ── Contatori risultati ───────────────────────────────────────────────────────
TOTAL=0
SUCCEEDED=0
FAILED=0
declare -a FAILED_LIST=()

# ── Verifiche iniziali ────────────────────────────────────────────────────────
log "============================================================"
log "  BATCH DECOMPOSE Gd1 — tutti i concept × 2 modelli"
log "  Log dir: $LOG_DIR"
log "============================================================"

check_server "$STEERING_URL" "Steering server" || { log "ABORT: avvia prima steering_server.py"; exit 1; }
check_server "$M40_URL"      "M40 llama-server" || { log "ABORT: avvia prima llama-server M40"; exit 1; }

# ── Loop modelli ──────────────────────────────────────────────────────────────
for MODEL in "Gemma3-1B-IT" "Gemma2-Uncensored"; do
    log ""
    log "════════════════════════════════════════"
    log "  MODELLO: $MODEL"
    log "════════════════════════════════════════"

    load_model_steering "$MODEL"
    sleep 5  # pausa dopo cambio modello

    for CONCEPT_CAT in "${CONCEPTS[@]}"; do
        CONCEPT="${CONCEPT_CAT%%:*}"
        CATEGORY="${CONCEPT_CAT##*:}"
        TOTAL=$((TOTAL + 1))

        log ""
        log "  ── ${MODEL} / ${CONCEPT} ──"
        LOG_FILE="${LOG_DIR}/${MODEL}_${CONCEPT}.log"

        "$PYTHON" "$SCRIPTS/decompose.py" \
            --concept    "$CONCEPT" \
            --model      "$MODEL" \
            --category   "$CATEGORY" \
            --max-depth  0 \
            --max-iter   2 \
            --steering-url "$STEERING_URL" \
            --m40-url    "$M40_URL" \
            > "$LOG_FILE" 2>&1

        EXIT=$?
        if [ $EXIT -eq 0 ]; then
            SUCCEEDED=$((SUCCEEDED + 1))
            # Estrai verdetto dall'ultima riga del log
            VERDICT=$(grep "VERDETTO:" "$LOG_FILE" 2>/dev/null | tail -1 || echo "")
            log "  ✓ OK  $VERDICT"
        else
            FAILED=$((FAILED + 1))
            FAILED_LIST+=("${MODEL}/${CONCEPT}")
            LAST_ERR=$(tail -3 "$LOG_FILE" 2>/dev/null || echo "")
            log "  ✗ FAILED — ${LAST_ERR}"
        fi
    done

    # Ricostruisce catalog dopo ogni modello
    rebuild_catalog
    log ""
done

# ── Report finale ─────────────────────────────────────────────────────────────
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

# Commit e push risultati
log ""
log "Git commit risultati..."
cd "$ROOT"
git add output/vector_library output/decompose_runs output/sub_concept_evals \
        output/catalog.json config/sub_concepts \
        2>/dev/null || true
git diff --cached --quiet && log "  (nessuna modifica da committare)" || \
git commit -m "data: decompose Gd1 batch — ${SUCCEEDED}/${TOTAL} concept completati

Vettori Gd1 estratti e validati per ${SUCCEEDED} concept × modello.
$([ ${#FAILED_LIST[@]} -gt 0 ] && echo "Falliti: ${FAILED_LIST[*]}" || true)

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>" && \
git push origin main && log "  ✓ Pushato su GitHub" || log "  ⚠ Push fallito"
