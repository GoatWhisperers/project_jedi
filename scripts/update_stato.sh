#!/bin/bash
# update_stato.sh — aggiorna STATO.md con lo stato attuale del progetto
# Uso: bash scripts/update_stato.sh "messaggio opzionale"

ROOT="/home/lele/codex-openai/project_jedi"
STATO="$ROOT/STATO.md"
PYTHON="$ROOT/.venv/bin/python"
NOTE="${1:-aggiornamento automatico}"

# Conta vettori estratti
G3_GD0=$(find "$ROOT/output/vector_library" -path "*/gemma3-1b-it/layer_*.npy" ! -path "*/sub/*" | wc -l)
G2_GD0=$(find "$ROOT/output/vector_library" -path "*/gemma2-uncensored/layer_*.npy" ! -path "*/sub/*" | wc -l)
G3_GD1=$(find "$ROOT/output/vector_library" -path "*/sub/*/gemma3-1b-it/layer_*.npy" | wc -l)
G2_GD1=$(find "$ROOT/output/vector_library" -path "*/sub/*/gemma2-uncensored/layer_*.npy" | wc -l)

# Server status
STEERING=$(curl -s http://localhost:8010/api/models 2>/dev/null | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('active','—'))" 2>/dev/null || echo "—")
M40=$(curl -s http://localhost:11435/health 2>/dev/null | grep -q "ok" && echo "✅" || echo "❌")

# Batch in corso?
BATCH_LOG=$(ls -t /tmp/decompose_gemma2_ripresa*.log /tmp/gemma2_ripresa*.log 2>/dev/null | head -1)
if [ -n "$BATCH_LOG" ]; then
    BATCH_OK=$(grep -c "✓ OK" "$BATCH_LOG" 2>/dev/null | tr -d '[:space:]')
    BATCH_FAIL=$(grep -c "✗ FAILED" "$BATCH_LOG" 2>/dev/null | tr -d '[:space:]')
    BATCH_DONE=$(grep -q "BATCH COMPLETATO" "$BATCH_LOG" 2>/dev/null && echo "COMPLETATO" || echo "IN CORSO")
    BATCH_INFO="$BATCH_DONE (✓${BATCH_OK} ✗${BATCH_FAIL}) — $(basename $BATCH_LOG)"
else
    BATCH_INFO="nessun batch attivo"
fi

cat > "$STATO" << EOF
# STATO — Project Jedi

> Questo file va letto SUBITO all'inizio di ogni sessione Claude.
> Viene aggiornato automaticamente da cantagallo e dai batch script.
> Ultima modifica: $(date '+%Y-%m-%d %H:%M') — $NOTE

---

## Server

| Servizio | Porta | Stato |
|----------|-------|-------|
| Steering server MI50 | 8010 | active: $STEERING |
| M40 llama-server CUDA | 11435 | $M40 |

\`\`\`bash
curl -s http://localhost:8010/api/models
curl -s http://localhost:11435/health
\`\`\`

---

## Libreria Vettori

| Livello | Gemma3-1B-IT | Gemma2-Uncensored |
|---------|-------------|------------------|
| Gd0 (broad) | ${G3_GD0} layer files | ${G2_GD0} layer files |
| Gd1 (sub) | ${G3_GD1} layer files | ${G2_GD1} layer files |

Gd0: 9/9 concept × 6 layer × 2 modelli = attesi 108 file per modello
Gd1: variabile (dipende dai sub-concept estratti)

---

## Batch

$BATCH_INFO

\`\`\`bash
# Log batch più recente:
tail -f $BATCH_LOG
\`\`\`

---

## Prossima sessione — checklist

\`\`\`
1. Leggi questo file (STATO.md)
2. cat /tmp/cantagallo_pending.txt
3. Verifica server (vedi sopra)
4. Se Gd1 Gemma2 incompleto: rilanciare batch ripresa
5. Quando Gd1 completo: scrivere experiments/07_gemma2_decompose_gd1.md
6. Poi: avviare ricerche riservate
\`\`\`

---

## Avvio rapido server

\`\`\`bash
cd /home/lele/codex-openai
nohup project_jedi/.venv/bin/python project_jedi/scripts/steering_server.py > /tmp/steering_server.log 2>&1 &
/mnt/raid0/llama-cpp-m40/start_cuda.sh
\`\`\`

Vedi anche: \`AVVIO.md\` per dettagli completi.
EOF

echo "STATO.md aggiornato ($(date '+%H:%M'))"
echo "  Gd0: G3=${G3_GD0} G2=${G2_GD0} | Gd1: G3=${G3_GD1} G2=${G2_GD1}"
echo "  Batch: $BATCH_INFO"
