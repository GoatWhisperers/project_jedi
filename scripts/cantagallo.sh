#!/bin/bash
# cantagallo.sh — Sveglia automatica per Claude Code via tmux.
# Controlla ogni 30 minuti lo stato di probe+eval Gemma2,
# notifica l'utente e inietta il messaggio direttamente nella sessione Claude.
# Si auto-resuscita via cron watchdog se muore inaspettatamente.
#
# Uso generico: bash scripts/cantagallo.sh &
# Vedi: ~/claude/trucchetti/sveglia_tmux.md
#
# Notifiche:
#   - tmux send-keys → Claude Code nella sessione corrente (sveglia automatica!)
#   - notify-send (desktop notification)
#   - wall (broadcast a tutti i terminali)
#   - /tmp/gemma_monitor.log (log persistente)

# Sessione tmux dove gira Claude Code
TMUX_TARGET="0:%0"    # sessione 0, pane %0

LOG=/tmp/gemma_monitor.log
DONE_FILE=/tmp/gemma_eval_done
PID_FILE=/tmp/gemma_monitor.pid
PID_PROBE=/tmp/probe_batch_gemma2.pid
PID_EVAL=/tmp/eval_batch_gemma2_resume.pid
EVAL_SESSIONS=/home/lele/codex-openai/project_jedi/output/eval_sessions
SCRIPT_PATH=/home/lele/codex-openai/project_jedi/scripts/cantagallo.sh
INTERVAL=1800   # 30 minuti

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

wake_claude() {
    local msg="$1"
    if tmux list-panes -t "$TMUX_TARGET" &>/dev/null; then
        tmux send-keys -t "$TMUX_TARGET" "$msg" Enter
        log "→ Inviato a Claude: $msg"
    else
        log "WARN: pane tmux $TMUX_TARGET non trovato"
    fi
}

notify() {
    local title="$1"
    local msg="$2"
    notify-send "$title" "$msg" --urgency=normal 2>/dev/null || true
    wall "$(printf '\n🐐 CANTAGALLO — %s\n%s\n' "$title" "$msg")" 2>/dev/null || true
}

count_sessions() {
    ls "$EVAL_SESSIONS"/session_*_Gemma2-Uncensored.jsonl 2>/dev/null | wc -l
}

probe_running() {
    local pid
    pid=$(cat "$PID_PROBE" 2>/dev/null) || return 1
    kill -0 "$pid" 2>/dev/null
}

eval_running() {
    local pid
    pid=$(cat "$PID_EVAL" 2>/dev/null) || return 1
    kill -0 "$pid" 2>/dev/null
}

any_running() {
    probe_running || eval_running
}

status_report() {
    local sessions
    sessions=$(count_sessions)
    local probe_status="fermo"
    local eval_status="fermo"
    probe_running && probe_status="IN CORSO"
    eval_running  && eval_status="IN CORSO"
    local last
    last=$(ls -t "$EVAL_SESSIONS"/session_*_Gemma2-Uncensored.jsonl 2>/dev/null | head -1)
    last=$(basename "$last" 2>/dev/null || echo "nessuna")
    echo "Probe: $probe_status | Eval: $eval_status | Sessioni: $sessions/9 | Ultima: $last"
}

# ── Watchdog cron: si assicura che cantagallo riesca sempre ───────────────────
install_watchdog() {
    local cron_line="*/31 * * * * [ -f $DONE_FILE ] || (pgrep -f cantagallo.sh > /dev/null || bash $SCRIPT_PATH >> /tmp/gemma_monitor_stdout.log 2>&1 &)"
    # Aggiunge solo se non già presente
    if ! crontab -l 2>/dev/null | grep -q "cantagallo"; then
        ( crontab -l 2>/dev/null; echo "$cron_line" ) | crontab -
        log "Watchdog cron installato (ogni 31 min)"
    fi
}

remove_watchdog() {
    crontab -l 2>/dev/null | grep -v "cantagallo" | crontab - 2>/dev/null || true
    log "Watchdog cron rimosso"
}

# ── Main ──────────────────────────────────────────────────────────────────────
rm -f "$DONE_FILE"
echo $$ > "$PID_FILE"
log "=== Cantagallo avviato — check ogni $((INTERVAL/60)) minuti — PID: $$ ==="

install_watchdog

check_count=0
while true; do
    sleep "$INTERVAL"
    check_count=$((check_count + 1))

    report=$(status_report)
    log "--- Check #$check_count — $report ---"

    if any_running; then
        # Processi ancora in corso — status update
        notify "Cantagallo #$check_count" "$report"
        wake_claude "cosa stanno combinando i gemma? (cantagallo check #$check_count)"

    else
        sessions=$(count_sessions)

        if [ "$sessions" -ge 9 ]; then
            # ── COMPLETATO ───────────────────────────────────────────────────
            log "=== COMPLETATO — tutte 9 sessioni Gemma2 presenti ==="
            notify "Cantagallo — COMPLETATO ✓" "Tutte 9 sessioni pronte."
            touch "$DONE_FILE"
            echo "COMPLETATO alle $(date)" >> "$DONE_FILE"
            remove_watchdog

            wake_claude "$(cat <<'MSG'
i gemma hanno finito l'eval. leggi tutti i report in output/eval_sessions/ per Gemma2-Uncensored, analizza i risultati (pattern sui gain, confronto tra concept, anomalie), scrivi una nuova entry nel diario degli esperimenti in experiments/, poi committa e pusha tutto su github.
MSG
)"
            break

        else
            # ── PROCESSI FERMI MA INCOMPLETO ─────────────────────────────────
            log "Processi fermi — sessioni: $sessions/9 — rilancio pipeline"
            notify "Cantagallo — rilancio" "Solo $sessions/9 sessioni. Riavvio pipeline."

            # Riavvia probe+eval per i mancanti
            cd /home/lele/codex-openai
            bash project_jedi/scripts/probe_and_eval_missing.sh >> /tmp/pipeline_launcher.log 2>&1 &
            log "Pipeline rilanciata — PID: $!"

            wake_claude "attenzione: i gemma si erano fermati con solo $sessions sessioni su 9. ho rilanciato la pipeline automaticamente. continuo a monitorare."
        fi
    fi
done

log "=== Cantagallo terminato ==="
