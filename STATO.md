# STATO — Project Jedi

> Questo file va letto SUBITO all'inizio di ogni sessione Claude.
> Viene aggiornato automaticamente da cantagallo e dai batch script.
> Ultima modifica: 2026-03-03 18:39 — fine sessione — batch ripresa3 in background PID 57948

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
| Gd1 (sub) | 600 layer files | 80 layer files |

Gd0: 9/9 concept × 6 layer × 2 modelli = attesi 108 file per modello
Gd1: variabile (dipende dai sub-concept estratti)

---

## Batch

IN CORSO (✓0 ✗0) — gemma2_ripresa3.log

```bash
# Log batch più recente:
tail -f /tmp/gemma2_ripresa3.log
```

---

## Prossima sessione — checklist

```
1. Leggi questo file (STATO.md)
2. cat /tmp/cantagallo_pending.txt
3. Verifica server (vedi sopra)
4. Se Gd1 Gemma2 incompleto: rilanciare batch ripresa
5. Quando Gd1 completo: scrivere experiments/07_gemma2_decompose_gd1.md
6. Poi: avviare ricerche riservate
```

---

## Avvio rapido server

```bash
cd /home/lele/codex-openai
nohup project_jedi/.venv/bin/python project_jedi/scripts/steering_server.py > /tmp/steering_server.log 2>&1 &
/mnt/raid0/llama-cpp-m40/start_cuda.sh
```

Vedi anche: `AVVIO.md` per dettagli completi.
