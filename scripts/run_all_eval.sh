#!/usr/bin/env bash
# run_all_eval.sh — Lancia auto_eval su tutti i concetti per un modello dato.
#
# Uso:
#   ./scripts/run_all_eval.sh                          # usa modello attivo sul server
#   ./scripts/run_all_eval.sh --model Gemma2-Uncensored
#   ./scripts/run_all_eval.sh --model Gemma3-1B-IT
#   ./scripts/run_all_eval.sh --concept hot_vs_cold    # solo un concept
#
# Il modello viene caricato via API prima di partire.
# I log singoli vanno in /tmp/eval_<concept>.log
# Output finale in output/eval_sessions/

set -euo pipefail

# ── Configurazione ─────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR/.."
PYTHON="$ROOT/.venv/bin/python"
AUTO_EVAL="$SCRIPT_DIR/auto_eval.py"
STEERING_URL="http://localhost:8010"
LOG_DIR="/tmp/eval_logs"

ALL_CONCEPTS=(
    hot_vs_cold
    luce_vs_buio
    calma_vs_allerta
    liscio_vs_ruvido
    secco_vs_umido
    duro_vs_morbido
    rumore_vs_silenzio
    dolce_vs_amaro
    odore_forte_vs_inodore
)

# ── Parsing argomenti ──────────────────────────────────────────────────────────
MODEL=""
SINGLE_CONCEPT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)   MODEL="$2";          shift 2 ;;
        --concept) SINGLE_CONCEPT="$2"; shift 2 ;;
        *) echo "Argomento sconosciuto: $1"; exit 1 ;;
    esac
done

# ── Utility ────────────────────────────────────────────────────────────────────
hr() { printf '%0.s─' {1..62}; echo; }

check_server() {
    if ! curl -sf "$STEERING_URL/api/models" > /dev/null 2>&1; then
        echo "ERRORE: steering server non risponde su $STEERING_URL"
        echo "Avvialo con: nohup $PYTHON $SCRIPT_DIR/steering_server.py > /tmp/steering_server.log 2>&1 &"
        exit 1
    fi
}

get_active_model() {
    curl -sf "$STEERING_URL/api/models" | python3 -c "import sys,json; print(json.load(sys.stdin)['active'])"
}

load_model() {
    local model="$1"
    echo "Caricamento modello: $model ..."
    curl -sf -X POST "$STEERING_URL/api/load_model" \
        -H "Content-Type: application/json" \
        -d "{\"name\": \"$model\"}" > /dev/null
    echo -n "Attesa completamento caricamento"
    for i in $(seq 1 60); do
        sleep 2
        current=$(get_active_model)
        if [[ "$current" == "$model" ]]; then
            echo " OK ($((i*2))s)"
            return 0
        fi
        echo -n "."
    done
    echo ""
    echo "ERRORE: timeout nel caricamento del modello ($model)"
    exit 1
}

# ── Main ───────────────────────────────────────────────────────────────────────
mkdir -p "$LOG_DIR"

echo ""
hr
echo "  AUTO-EVAL BATCH"
echo "  Script : run_all_eval.sh"
echo "  Data   : $(date '+%Y-%m-%d %H:%M')"
hr

# Controlla server
check_server

# Carica il modello richiesto (se specificato)
if [[ -n "$MODEL" ]]; then
    current=$(get_active_model)
    if [[ "$current" != "$MODEL" ]]; then
        load_model "$MODEL"
    else
        echo "Modello già caricato: $MODEL"
    fi
fi

# Modello attivo finale
ACTIVE_MODEL=$(get_active_model)
echo ""
echo "Modello steered : $ACTIVE_MODEL"
echo "Evaluator       : Gemma3-4B (M40, porta 11435)"
echo ""

# Lista dei concetti da eseguire
if [[ -n "$SINGLE_CONCEPT" ]]; then
    CONCEPTS=("$SINGLE_CONCEPT")
else
    CONCEPTS=("${ALL_CONCEPTS[@]}")
fi

TOTAL=${#CONCEPTS[@]}
echo "Concetti da valutare ($TOTAL):"
for c in "${CONCEPTS[@]}"; do
    echo "  · $c"
done
echo ""
hr

# Ciclo principale
SUCCEEDED=()
FAILED=()

for i in "${!CONCEPTS[@]}"; do
    concept="${CONCEPTS[$i]}"
    n=$((i + 1))
    log_file="$LOG_DIR/eval_${concept}.log"

    echo ""
    echo "[$n/$TOTAL] $concept"
    echo "  Log: $log_file"
    echo "  Avvio: $(date '+%H:%M:%S')"

    # Verifica che esista il file eval_concepts
    cfg_file="$ROOT/config/eval_concepts/${concept}.json"
    if [[ ! -f "$cfg_file" ]]; then
        echo "  SKIP: config non trovata ($cfg_file)"
        FAILED+=("$concept (config mancante)")
        continue
    fi

    # Lancia auto_eval
    if "$PYTHON" "$AUTO_EVAL" --concept "$concept" > "$log_file" 2>&1; then
        echo "  Fine:  $(date '+%H:%M:%S') ✓"
        SUCCEEDED+=("$concept")
    else
        echo "  Fine:  $(date '+%H:%M:%S') ✗ — vedi $log_file"
        FAILED+=("$concept")
    fi
done

# Riepilogo finale
echo ""
hr
echo "  RIEPILOGO"
hr
echo "  OK    (${#SUCCEEDED[@]}): ${SUCCEEDED[*]:-nessuno}"
echo "  FAIL  (${#FAILED[@]}):   ${FAILED[*]:-nessuno}"
echo "  Output: $ROOT/output/eval_sessions/"
echo "  Log:    $LOG_DIR/"
hr
echo ""
