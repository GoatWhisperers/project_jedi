# STATO — Project Jedi

> Questo file va letto SUBITO all'inizio di ogni sessione Claude.
> Ultima modifica: 2026-03-15 — sessione vettori affettivi riservati

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

---

## Libreria Vettori (riservata — `/home/lele/ricerche_riservate/vector_library/`)

8 assi affettivi estratti su Gemma3-1B-IT + Gemma2-Uncensored.

| Concept | Gemma3 SNR | Gemma2 SNR | Stato |
|---------|-----------|-----------|-------|
| sonnolenza_vs_veglia | +6.56 L19 | +4.14 L38 | ✅ pronto steering |
| desiderio_vs_urgenza | +2.17 L20 | +1.13 L38 | ✅ usabile |
| sicurezza_vs_minaccia | +0.83 L23 | +1.26 L38 | 🟡 debole |
| calore_sensuale | +0.12 | +0.69 L38 | 🟡 solo Gemma2 |
| passione_vs_torrida | -0.73 | +0.59 L33 | 🟡 solo Gemma2 |
| tenerezza_vs_desiderio | -0.35 | +0.10 | ❌ da rigenerare |
| urgenza_vs_inerzia | -0.42 | +0.01 | ❌ overlap calma/allerta |
| indifferenza_vs_interesse | -1.65 | -0.12 | ❌ da rigenerare |

**Problema chiave emerso:** assi con poli non opposti (intensità crescente)
non separabili con mean-diff. Dataset di passione/torrida riformulato
con polo negativo genuino = frigidità affettiva.

---

## Batch in corso

**Generazione dataset riservati v2** — Gemma2-Uncensored genera frasi
per 4 nuovi concetti riformulati con poli correttamente opposti.

```bash
# Stato:
tail -f /tmp/riservati_generation.log
ps aux | grep generate_dataset | grep python
# Concepts generati:
ls /home/lele/ricerche_riservate/concepts/
# Logs raw Gemma2:
ls /home/lele/ricerche_riservate/generation_logs/
```

Dopo la generazione, ri-estrarre con:
```bash
# Per ogni nuovo concept:
cd /home/lele/codex-openai && project_jedi/.venv/bin/python \
  project_jedi/scripts/probe_concept.py \
  --concept /home/lele/ricerche_riservate/concepts/SLUG.json \
  --model Gemma2-Uncensored \
  --output-root /home/lele/ricerche_riservate/vector_library \
  --eval
```

---

## Modifiche sessione 2026-03-15

- `scripts/probe_concept.py` — flag `--output-root` aggiunto
- `ricerche_riservate/generate_dataset_riservati.py` — script generazione dataset
  con Gemma2-Uncensored via transformers (4 nuovi concetti affettivi v2)
- `ricerche_riservate/run_extraction.sh` — batch estrazione riservata

---

## Prossima sessione — checklist

```
1. Leggi questo file (STATO.md)
2. cat /tmp/cantagallo_pending.txt
3. Verifica server (vedi sopra)
4. Leggi RIPRESA_20260315.md
5. Controlla se generazione è completa:
   tail /tmp/riservati_generation.log
   ls /home/lele/ricerche_riservate/concepts/
6. Se completa: ri-estrarre i 4 concept v2 e confrontare SNR
7. Steering sonnolenza_vs_veglia (pronto)
```

---

## Avvio rapido server

```bash
cd /home/lele/codex-openai
nohup project_jedi/.venv/bin/python project_jedi/scripts/steering_server.py > /tmp/steering_server.log 2>&1 &
/mnt/raid0/llama-cpp-m40/start_cuda.sh
```
