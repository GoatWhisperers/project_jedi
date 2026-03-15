# STATO — Project Jedi

> Questo file va letto SUBITO all'inizio di ogni sessione Claude.
> Viene aggiornato automaticamente da cantagallo e dai batch script.
> Ultima modifica: 2026-03-15 — sessione riservata + flag --output-root

---

## Server

| Servizio | Porta | Stato |
|----------|-------|-------|
| Steering server MI50 | 8010 | active: Gemma3-1B-IT |
| M40 llama-server CUDA | 11435 | ✅ |

```bash
curl -s http://localhost:8010/api/models
curl -s http://localhost:11435/health
```

---

## Libreria Vettori (pubblica)

| Livello | Gemma3-1B-IT | Gemma2-Uncensored |
|---------|-------------|------------------|
| Gd0 (broad) | 9/9 ✅ | 9/9 ✅ |
| Gd1 (sub) | 9/9 ✅ ~600 file | 9/9 ✅ ~800 file |

Analisi geometria Gd1: `experiments/07_gemma2_decompose_gd1.md`

---

## Batch

Nessun batch pubblico in corso.

Esiste una sessione di estrazione riservata separata — vedi
`/home/lele/ricerche_riservate/` (fuori repo).

```bash
# Log estrazione riservata (se in corso):
tail -f /tmp/riservati_extraction.log
```

---

## Modifiche sessione 2026-03-15

- `scripts/probe_concept.py` — aggiunto flag `--output-root` per reindirizzare
  l'output fuori dalla vector_library pubblica

---

## Prossima sessione — checklist

```
1. Leggi questo file (STATO.md)
2. cat /tmp/cantagallo_pending.txt
3. Verifica server (vedi sopra)
4. Leggi RIPRESA_20260315.md
5. Prossimi esperimenti: steering Gd1 + confronto Gemma3 vs Gemma2
```

---

## Avvio rapido server

```bash
cd /home/lele/codex-openai
nohup project_jedi/.venv/bin/python project_jedi/scripts/steering_server.py > /tmp/steering_server.log 2>&1 &
/mnt/raid0/llama-cpp-m40/start_cuda.sh
```

Vedi anche: `AVVIO.md` per dettagli completi.
