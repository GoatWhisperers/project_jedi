# STATO — Project Jedi

> Questo file va letto SUBITO all'inizio di ogni sessione Claude.
> Viene aggiornato automaticamente da cantagallo e dai batch script.
> Ultima modifica: 2026-03-06 18:09 — fine sessione

---

## Server

| Servizio | Porta | Stato |
|----------|-------|-------|
| Steering server MI50 | 8010 | active: Gemma2-Uncensored |
| M40 llama-server CUDA | 11435 | ✅ |

```bash
curl -s http://localhost:8010/api/models
curl -s http://localhost:11435/health
```

---

## Libreria Vettori

| Livello | Gemma3-1B-IT | Gemma2-Uncensored |
|---------|-------------|------------------|
| Gd0 (broad) | 120 layer files | 180 layer files |
| Gd1 (sub) | 600 layer files | 800 layer files |

Gd0: 9/9 concept × 6 layer × 2 modelli = attesi 108 file per modello
Gd1: variabile (dipende dai sub-concept estratti)

---

## Batch

COMPLETATO (✓0 ✗0) — gemma2_ripresa5.log

```bash
# Log batch più recente:
tail -f /tmp/gemma2_ripresa5.log
```

---

## Libreria completata — 2026-03-06

Gd0 + Gd1 completi per entrambi i modelli. 800 file Gd1 Gemma2.
Analisi geometria interna: `experiments/07_gemma2_decompose_gd1.md`

## Prossima sessione — checklist

```
1. Leggi questo file (STATO.md)
2. cat /tmp/cantagallo_pending.txt
3. Verifica server (vedi sopra)
4. Leggi RIPRESA_20260307.md
5. Prossimi esperimenti: steering Gd1 + confronto Gemma3 vs Gemma2 + ricerche riservate
```

---

## Avvio rapido server

```bash
cd /home/lele/codex-openai
nohup project_jedi/.venv/bin/python project_jedi/scripts/steering_server.py > /tmp/steering_server.log 2>&1 &
/mnt/raid0/llama-cpp-m40/start_cuda.sh
```

Vedi anche: `AVVIO.md` per dettagli completi.
