#!/bin/bash
# watch_dialoghi.sh — Finestra live sui dialoghi tra M40 (scienziato) e Gemma (cavia)
#
# Mostra in tempo reale:
#   - Cosa sta chiedendo M40 a Gemma (prompt di steering)
#   - Cosa risponde Gemma (output steered)
#   - Cosa giudica M40 (valutazione fenomenologica)
#   - Stato generale del batch
#
# Uso:
#   bash scripts/watch_dialoghi.sh           # refresh automatico ogni 8s
#   bash scripts/watch_dialoghi.sh --once    # stampa e esci (utile per pipe/log)

INTERVAL=8
ONE_SHOT=false
[ "$1" = "--once" ] && ONE_SHOT=true

RESET="\033[0m"
BOLD="\033[1m"
DIM="\033[2m"
CYAN="\033[36m"
YELLOW="\033[33m"
GREEN="\033[32m"
RED="\033[31m"
BLUE="\033[34m"
MAGENTA="\033[35m"
WHITE="\033[37m"

# Larghezza terminale
COLS=$(tput cols 2>/dev/null || echo 100)
SEP=$(printf '─%.0s' $(seq 1 $COLS))

print_header() {
    echo -e "${BOLD}${CYAN}"
    echo "  PROJECT JEDI — Dialoghi scienziato / cavia   $(date '+%H:%M:%S')"
    echo -e "  M40 = scienziato (Gemma3-12B)   │   Gemma3/2 = cavia${RESET}"
    echo -e "${DIM}${SEP}${RESET}"
}

# ── Stato batch ────────────────────────────────────────────────────────────────
print_batch_status() {
    local BATCH_LOG
    BATCH_LOG=$(ls -t /tmp/decompose_gd1_*/batch_main.log 2>/dev/null | head -1)

    if [ -z "$BATCH_LOG" ]; then
        echo -e "${RED}  ✗ Nessun batch attivo${RESET}"
        return
    fi

    local DONE OK FAIL RUNNING
    # grep -c stampa sempre il conteggio (anche 0) — il || echo 0 creerebbe doppioni
    DONE=$(grep -c "✓ OK\|✗ FAILED" "$BATCH_LOG" 2>/dev/null); DONE=${DONE:-0}
    OK=$(grep -c "✓ OK" "$BATCH_LOG" 2>/dev/null); OK=${OK:-0}
    FAIL=$(grep -c "✗ FAILED" "$BATCH_LOG" 2>/dev/null); FAIL=${FAIL:-0}
    pgrep -f "run_decompose_gd1_all.sh" > /dev/null 2>&1 && RUNNING="${GREEN}●${RESET}" || RUNNING="${RED}●${RESET}"

    # Concept corrente (ultima riga con "── ... ──")
    local CURRENT_CONCEPT
    CURRENT_CONCEPT=$(grep "── .*\/\|── Gemma" "$BATCH_LOG" 2>/dev/null | tail -1 | sed 's/.*──  *//;s/  *──.*//')

    echo -e "  ${RUNNING} Batch  ${BOLD}${DONE}/18${RESET} concept  (✓${GREEN}${OK}${RESET}  ✗${RED}${FAIL}${RESET})"
    [ -n "$CURRENT_CONCEPT" ] && echo -e "  ${DIM}In corso: ${RESET}${BOLD}${CURRENT_CONCEPT}${RESET}"
}

# ── Step corrente dal log del concept in lavorazione ──────────────────────────
print_current_step() {
    local BATCH_LOG LOG_DIR LAST_LOG
    BATCH_LOG=$(ls -t /tmp/decompose_gd1_*/batch_main.log 2>/dev/null | head -1)
    [ -z "$BATCH_LOG" ] && return

    LOG_DIR=$(dirname "$BATCH_LOG")
    LAST_LOG=$(ls -t "$LOG_DIR"/*.log 2>/dev/null | grep -v batch_main | head -1)
    [ -z "$LAST_LOG" ] && return

    local CONCEPT_NAME
    CONCEPT_NAME=$(basename "$LAST_LOG" .log | sed 's/Gemma[0-9]*-[^_]*_//')

    echo ""
    echo -e "${DIM}${SEP}${RESET}"
    echo -e "  ${BOLD}${YELLOW}STEP CORRENTE${RESET} — ${BOLD}${CONCEPT_NAME}${RESET}"
    echo -e "${DIM}${SEP}${RESET}"

    # Mostra le ultime righe del log concept (progresso step)
    tail -12 "$LAST_LOG" | while IFS= read -r line; do
        # Colora le righe per tipo
        if echo "$line" | grep -q "STEP\|Step [0-9]"; then
            echo -e "  ${BOLD}${BLUE}${line}${RESET}"
        elif echo "$line" | grep -q "✓\|VALIDATI\|DISTINTI\|OK"; then
            echo -e "  ${GREEN}${line}${RESET}"
        elif echo "$line" | grep -q "✗\|FAILED\|ERRORE\|WARN"; then
            echo -e "  ${RED}${line}${RESET}"
        elif echo "$line" | grep -q "score=\|Avg score"; then
            echo -e "  ${MAGENTA}${line}${RESET}"
        elif echo "$line" | grep -q "Proposti\|sub-concetti"; then
            echo -e "  ${CYAN}${line}${RESET}"
        else
            echo -e "  ${DIM}${line}${RESET}"
        fi
    done
}

# ── Ultimo dialogo M40 salvato ─────────────────────────────────────────────────
print_last_m40_dialogue() {
    local DIALOGUE_DIR="/home/lele/codex-openai/project_jedi/output/m40_dialogues"
    local LAST_JSONL
    LAST_JSONL=$(find "$DIALOGUE_DIR" -name "*.jsonl" -newer /tmp 2>/dev/null | \
                 xargs ls -t 2>/dev/null | head -1)
    # fallback: semplicemente il più recente
    if [ -z "$LAST_JSONL" ]; then
        LAST_JSONL=$(find "$DIALOGUE_DIR" -name "*.jsonl" 2>/dev/null | \
                     xargs ls -t 2>/dev/null | head -1)
    fi
    [ -z "$LAST_JSONL" ] && return

    local LAST_ENTRY
    LAST_ENTRY=$(tail -1 "$LAST_JSONL" 2>/dev/null)
    [ -z "$LAST_ENTRY" ] && return

    local STEP TS USER_EXCERPT RESP_EXCERPT
    STEP=$(echo "$LAST_ENTRY" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('step','?'))" 2>/dev/null)
    TS=$(echo "$LAST_ENTRY" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('timestamp','')[:19])" 2>/dev/null)
    USER_EXCERPT=$(echo "$LAST_ENTRY" | python3 -c "
import sys,json
d=json.load(sys.stdin)
u=d.get('user','')
# Mostra le prime 300 chars del prompt utente (cosa M40 aveva davanti)
print(u[:300])
" 2>/dev/null)
    RESP_EXCERPT=$(echo "$LAST_ENTRY" | python3 -c "
import sys,json
d=json.load(sys.stdin)
r=d.get('raw_response','')
# Mostra le prime 500 chars della risposta grezza
print(r[:500])
" 2>/dev/null)

    echo ""
    echo -e "${DIM}${SEP}${RESET}"
    echo -e "  ${BOLD}${YELLOW}ULTIMO DIALOGO M40${RESET}  ${DIM}$(basename "$LAST_JSONL") — ${TS}${RESET}"
    echo -e "  ${DIM}Step: ${RESET}${BOLD}${STEP}${RESET}"
    echo -e "${DIM}${SEP}${RESET}"

    echo -e "  ${BOLD}${BLUE}── SCIENZIATO (M40 chiede / presenta):${RESET}"
    echo "$USER_EXCERPT" | fold -s -w $((COLS - 4)) | while IFS= read -r line; do
        echo -e "  ${DIM}${line}${RESET}"
    done

    echo ""
    echo -e "  ${BOLD}${GREEN}── RISPOSTA M40 (ragionamento / proposta):${RESET}"
    echo "$RESP_EXCERPT" | fold -s -w $((COLS - 4)) | while IFS= read -r line; do
        echo -e "  ${line}"
    done
    echo -e "  ${DIM}[… continua nel file JSONL]${RESET}"
}

# ── Ultime generazioni Gemma (steering_log) ────────────────────────────────────
print_last_gemma_output() {
    local STEERING_LOG="/home/lele/codex-openai/project_jedi/output/steering_log.jsonl"
    [ ! -f "$STEERING_LOG" ] && return

    local N_LINES
    N_LINES=$(wc -l < "$STEERING_LOG" 2>/dev/null || echo 0)
    [ "$N_LINES" -eq 0 ] && return

    # Ultima generazione
    local LAST
    LAST=$(tail -1 "$STEERING_LOG")
    local PROMPT CONCEPT ALPHA OUTPUT
    PROMPT=$(echo "$LAST" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('prompt','')[:80])" 2>/dev/null)
    CONCEPT=$(echo "$LAST" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('concept','?'))" 2>/dev/null)
    ALPHA=$(echo "$LAST" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('alpha','?'))" 2>/dev/null)
    OUTPUT=$(echo "$LAST" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('output','')[:400])" 2>/dev/null)

    echo ""
    echo -e "${DIM}${SEP}${RESET}"
    echo -e "  ${BOLD}${YELLOW}ULTIMA GENERAZIONE GEMMA (cavia steered)${RESET}  ${DIM}(${N_LINES} totali)${RESET}"
    echo -e "  ${DIM}Concept: ${RESET}${BOLD}${CONCEPT}${RESET}  ${DIM}│  alpha: ${RESET}${BOLD}${ALPHA}${RESET}"
    echo -e "${DIM}${SEP}${RESET}"
    echo -e "  ${BOLD}${BLUE}── Prompt (cosa gli è stato chiesto):${RESET}"
    echo -e "  ${DIM}${PROMPT}${RESET}"
    echo ""
    echo -e "  ${BOLD}${GREEN}── Output Gemma (cavia risponde):${RESET}"
    echo "$OUTPUT" | fold -s -w $((COLS - 4)) | while IFS= read -r line; do
        echo -e "  ${line}"
    done
    echo -e "  ${DIM}[… continua in steering_log.jsonl]${RESET}"
}

# ── GPU ────────────────────────────────────────────────────────────────────────
print_gpu() {
    local MI50 M40_GPU
    MI50=$(rocm-smi --showmeminfo vram --csv 2>/dev/null | tail -1 | \
           awk -F',' '{printf "MI50 VRAM: %s/%s MB", $2, $3}' 2>/dev/null || echo "MI50: n/d")
    M40_GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.free \
              --format=csv,noheader,nounits 2>/dev/null | \
              awk -F',' '{printf "M40 GPU: %s%% VRAM: %s/%s MB", $1, $2, $2+$3}' 2>/dev/null || echo "M40: n/d")

    echo ""
    echo -e "${DIM}${SEP}${RESET}"
    echo -e "  ${DIM}${MI50}   │   ${M40_GPU}${RESET}"
}

# ── Loop principale ────────────────────────────────────────────────────────────
run_once() {
    clear
    print_header
    print_batch_status
    print_current_step
    print_last_m40_dialogue
    print_last_gemma_output
    print_gpu
    echo ""
    echo -e "${DIM}  [q per uscire  │  refresh ogni ${INTERVAL}s  │  --once per stampa singola]${RESET}"
}

if $ONE_SHOT; then
    run_once
    exit 0
fi

# Loop con refresh
while true; do
    run_once
    # Leggi input con timeout (q per uscire)
    if read -t "$INTERVAL" -n 1 key 2>/dev/null; then
        [ "$key" = "q" ] || [ "$key" = "Q" ] && echo "" && exit 0
    fi
done
