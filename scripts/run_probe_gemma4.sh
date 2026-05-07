#!/usr/bin/env bash
# Estrae vettori Gd0 per tutti i 9 concept standard su Gemma4-E4B-IT.
# Usa mi50_manager (porta 8020) come da architettura progetto.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
VENV="$ROOT/.venv"
LOGDIR="/tmp/probe_gemma4_$(date +%Y%m%d_%H%M%S)"
MODEL="Gemma4-E4B-IT"

CONCEPTS=(
    hot_vs_cold
    luce_vs_buio
    duro_vs_morbido
    liscio_vs_ruvido
    rumore_vs_silenzio
    secco_vs_umido
    calma_vs_allerta
    dolce_vs_amaro
    odore_forte_vs_inodore
)

mkdir -p "$LOGDIR"
echo "[run_probe_gemma4] avvio — log in $LOGDIR"
echo "[run_probe_gemma4] modello: $MODEL"
echo "[run_probe_gemma4] $(date)"

# Verifica manager attivo
if ! curl -s http://localhost:8020/api/status | grep -q '"model"'; then
    echo "[ERRORE] mi50_manager non risponde su porta 8020"
    exit 1
fi

ok=0
fail=0

for concept in "${CONCEPTS[@]}"; do
    CONCEPT_FILE="$ROOT/config/concepts/${concept}.json"
    if [ ! -f "$CONCEPT_FILE" ]; then
        echo "[SKIP] $concept — file non trovato: $CONCEPT_FILE"
        continue
    fi

    LOG="$LOGDIR/${concept}.log"
    echo -n "[$(date +%H:%M:%S)] $concept ... "

    if source "$VENV/bin/activate" && \
       HSA_OVERRIDE_GFX_VERSION=9.0.6 HSA_ENABLE_SDMA=0 HIP_VISIBLE_DEVICES=0 \
       HF_HOME=/mnt/raid0/hf_cache TORCH_BLAS_PREFER_HIPBLASLT=0 \
       python "$SCRIPT_DIR/probe_concept.py" \
           --concept "$CONCEPT_FILE" \
           --model "$MODEL" \
           --eval \
           >> "$LOG" 2>&1; then
        echo "OK"
        ok=$((ok+1))
    else
        echo "FALLITO (vedi $LOG)"
        fail=$((fail+1))
    fi
done

echo ""
echo "[run_probe_gemma4] completato — OK=$ok  FALLITI=$fail"
echo "[run_probe_gemma4] log: $LOGDIR"
