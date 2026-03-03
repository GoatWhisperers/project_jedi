#!/bin/bash
# cantagallo.sh — Monitor autonomo del batch decompose Gd1
#
# NON usa tmux send-keys (causa disruption nella sessione Claude Code).
#
# Meccanismi di notifica (non invasivi):
#   1. Log continuo su /tmp/jedi_cantagallo.log
#   2. tmux display-message → popup visivo all'utente (sparisce da solo)
#   3. File /tmp/cantagallo_pending.txt → messaggio per Claude alla prossima interazione
#
# Uso:
#   tmux new-window -n cantagallo "bash /home/lele/codex-openai/project_jedi/scripts/cantagallo.sh"
#   oppure in background:
#   nohup bash scripts/cantagallo.sh > /tmp/jedi_cantagallo.log 2>&1 &

INTERVAL=900                       # 15 minuti
CANTAGALLO_LOG="/tmp/jedi_cantagallo.log"
PENDING_FILE="/tmp/cantagallo_pending.txt"
PID_FILE="/tmp/jedi_cantagallo.pid"
CLAUDE_PANE="0:0.0"               # pane tmux dove gira Claude Code
STATO_FILE="/home/lele/codex-openai/project_jedi/STATO.md"

echo $$ > "$PID_FILE"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$CANTAGALLO_LOG"
}

# Notifica visiva non invasiva all'utente via tmux display-message
# Il popup appare nell'angolo di tmux e sparisce dopo ~4 secondi
notify_user() {
    local msg="$1"
    tmux display-message -d 4000 "$msg" 2>/dev/null || true
    echo ">>> CANTAGALLO: $msg" >&2
}

# Controlla se Claude sta elaborando (spinner "Esc to interrupt" visibile nel pane)
claude_is_busy() {
    tmux capture-pane -pt "$CLAUDE_PANE" -p 2>/dev/null | grep -q "Esc to interrupt"
}

# Scrive una PROPOSTA nel file pending che Claude legge alla prossima interazione.
# Se Claude è occupato: solo popup visivo, niente scrittura (non lo interrompiamo).
# Se Claude è idle: popup + pending.
notify_claude() {
    local msg="$1"
    if claude_is_busy; then
        log "→ Claude occupato — solo popup, niente pending"
        notify_user "CANTAGALLO (in attesa): $msg"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] $msg" >> "$PENDING_FILE"
        log "→ Proposta per Claude: ${msg:0:100}..."
        notify_user "CANTAGALLO: $msg"
    fi
}

get_batch_log() {
    ls -t /tmp/decompose_gd1_*/batch_main.log 2>/dev/null | head -1
}

batch_running() {
    pgrep -f "run_decompose_gd1_all.sh" > /dev/null 2>&1
}

count_done() {
    local batch_log; batch_log="$(get_batch_log)"
    [ -n "$batch_log" ] && grep -c "✓ OK\|✗ FAILED" "$batch_log" 2>/dev/null || echo 0
}

count_ok() {
    local batch_log; batch_log="$(get_batch_log)"
    [ -n "$batch_log" ] && grep -c "✓ OK" "$batch_log" 2>/dev/null || echo 0
}

count_fail() {
    local batch_log; batch_log="$(get_batch_log)"
    [ -n "$batch_log" ] && grep -c "✗ FAILED" "$batch_log" 2>/dev/null || echo 0
}

check_and_report() {
    local done ok fail batch_active
    done=$(count_done)
    ok=$(count_ok)
    fail=$(count_fail)
    batch_running && batch_active="✓ ATTIVO" || batch_active="✗ FERMO"

    log "=== Check — batch:$batch_active  done:$done/18  ok:$ok  fail:$fail ==="

    if [ "$done" -ge 18 ]; then
        # ── BATCH COMPLETATO ──────────────────────────────────────────────────
        log "=== BATCH COMPLETATO ==="
        notify_user "CANTAGALLO: batch decompose Gd1 COMPLETATO (${ok}/18 ✓  ${fail} ✗)"
        notify_claude "Batch decompose Gd1 completato (${ok}/18 ✓, ${fail} ✗). Quando vuoi posso leggere il report, aggiornare il catalog e fare il commit — dimmi tu."
        update_stato "batch COMPLETATO ${ok}/18"
        log "Cantagallo in standby — batch finito."
        exit 0

    elif ! batch_running && [ "$done" -lt 18 ]; then
        # ── BATCH FERMO PRIMA DEL TERMINE ────────────────────────────────────
        log "⚠ Batch FERMO con $done/18 concept completati"
        notify_user "CANTAGALLO ⚠: batch FERMO a ${done}/18 concept (✓${ok} ✗${fail}) — controlla!"
        notify_claude "Il batch decompose Gd1 sembra fermo a ${done}/18 concept (✓ ${ok}  ✗ ${fail}). Vuoi che guardo i log per capire cosa è successo?"

    else
        # ── BATCH IN CORSO — aggiornamento di routine ─────────────────────────
        notify_user "CANTAGALLO: decompose Gd1 in corso — ${done}/18 (✓${ok} ✗${fail})"
        # Ogni 6 concept (~3 ore) scrive anche il pending per Claude
        if [ "$done" -gt 0 ] && [ $(( done % 6 )) -eq 0 ]; then
            notify_claude "Aggiornamento batch: ${done}/18 concept completati (✓ ${ok}  ✗ ${fail}). Tutto ok per ora."
        fi
    fi
}

# Aggiorna il campo "Ultima modifica" in STATO.md
update_stato() {
    local note="$1"
    if [ -f "$STATO_FILE" ]; then
        sed -i "s/> Ultima modifica:.*/> Ultima modifica: $(date '+%Y-%m-%d %H:%M') — $note/" "$STATO_FILE"
    fi
}

# ── Main ─────────────────────────────────────────────────────────────────────
log "=== Cantagallo avviato (PID $$) — intervallo ${INTERVAL}s ==="
log "    Notifiche: tmux display-message + $PENDING_FILE (NO tmux send-keys)"

# Prima chiamata dopo 60s (lascia tempo al batch di avviarsi se appena lanciato)
sleep 60
check_and_report

while true; do
    sleep "$INTERVAL"
    check_and_report
done
