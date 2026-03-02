#!/bin/bash
# monitor_decompose.sh — Cantagallo del batch decompose
#
# Mostra lo stato corrente del batch in modo leggibile.
# Uso: bash scripts/monitor_decompose.sh
#      watch -n 60 bash scripts/monitor_decompose.sh   # aggiornamento ogni minuto

BATCH_LOG=$(ls -t /tmp/decompose_gd1_*/batch_main.log 2>/dev/null | head -1)
STEERING_URL="http://localhost:8010"
M40_URL="http://localhost:11435"

echo "══════════════════════════════════════════════════════"
echo "  MONITOR DECOMPOSE Gd1 — $(date '+%Y-%m-%d %H:%M:%S')"
echo "══════════════════════════════════════════════════════"

# ── Stato server ──────────────────────────────────────────
echo ""
echo "SERVER:"
if active=$(curl -sf "${STEERING_URL}/api/models" 2>/dev/null | \
            python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('active','?'))" 2>/dev/null); then
    echo "  ✓ Steering server — modello: $active"
else
    echo "  ✗ Steering server NON raggiungibile"
fi

if curl -sf "${M40_URL}/health" > /dev/null 2>&1; then
    echo "  ✓ M40 llama-server OK"
else
    echo "  ✗ M40 llama-server NON raggiungibile"
fi

# ── Processo batch ────────────────────────────────────────
echo ""
echo "PROCESSO BATCH:"
BATCH_PID=$(pgrep -f "run_decompose_gd1_all.sh" 2>/dev/null || echo "")
DECOMPOSE_PID=$(pgrep -f "decompose.py" 2>/dev/null || echo "")

if [ -n "$BATCH_PID" ]; then
    echo "  ✓ Batch in esecuzione (PID $BATCH_PID)"
else
    echo "  ─ Batch non attivo"
fi
if [ -n "$DECOMPOSE_PID" ]; then
    echo "  ✓ decompose.py attivo (PID $DECOMPOSE_PID)"
fi

# ── Log corrente ──────────────────────────────────────────
if [ -z "$BATCH_LOG" ]; then
    echo ""
    echo "  (nessun batch log trovato in /tmp/decompose_gd1_*/)"
    exit 0
fi

LOG_DIR=$(dirname "$BATCH_LOG")
echo ""
echo "LOG: $BATCH_LOG"
echo ""
echo "PROGRESSO:"
grep -E "MODELLO:|── .*/|✓ OK|✗ FAILED|BATCH COMPLETATO|Ricostruzione|Pushato" \
    "$BATCH_LOG" 2>/dev/null | tail -30 | while IFS= read -r line; do
    echo "  $line"
done

# ── Contatori ────────────────────────────────────────────
DONE=$(grep -c "✓ OK\|✗ FAILED" "$BATCH_LOG" 2>/dev/null || echo 0)
OK=$(grep -c "✓ OK" "$BATCH_LOG" 2>/dev/null || echo 0)
FAIL=$(grep -c "✗ FAILED" "$BATCH_LOG" 2>/dev/null || echo 0)
echo ""
echo "CONTEGGIO: $DONE/18 concept  (✓ $OK  ✗ $FAIL)"

# ── Ultimo log individuale ────────────────────────────────
LAST_LOG=$(ls -t "$LOG_DIR"/*.log 2>/dev/null | grep -v batch_main | head -1)
if [ -n "$LAST_LOG" ]; then
    echo ""
    echo "ULTIMO CONCEPT ($(basename "$LAST_LOG")):"
    tail -8 "$LAST_LOG" | while IFS= read -r line; do
        echo "  $line"
    done
fi

# ── GPU ───────────────────────────────────────────────────
echo ""
echo "GPU:"
nvidia-smi --query-gpu=name,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader,nounits 2>/dev/null | \
    while IFS=',' read -r name used free util; do
        echo "  M40  VRAM: ${used}/${$(( used + free ))} MB  GPU: ${util}%"
    done || true

rocm-smi --showmeminfo vram --csv 2>/dev/null | tail -1 | \
    awk -F',' '{printf "  MI50 VRAM: %s / %s MB\n", $2, $3}' || true

echo ""
echo "══════════════════════════════════════════════════════"
