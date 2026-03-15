# STATO — Project Jedi

> Questo file va letto SUBITO all'inizio di ogni sessione Claude.
> Ultima modifica: 2026-03-15 — fine sessione

---

## Server

| Servizio | Porta | Stato |
|----------|-------|-------|
| Steering server MI50 | 8010 | spento (sera) |
| M40 llama-server CUDA | 11435 | spento (sera) |

```bash
curl -s http://localhost:8010/api/models
curl -s http://localhost:11435/health
```

---

## Libreria Vettori (pubblica)

| Livello | Gemma3-1B-IT | Gemma2-Uncensored |
|---------|-------------|------------------|
| Gd0 (broad) | 9/9 ✅ | 9/9 ✅ |
| Gd1 (sub) | 9/9 ✅ | 9/9 ✅ |

---

## Libreria Vettori (riservata)

`/home/lele/ricerche_riservate/vector_library/affettivo/`

| Concept | Gemma3 SNR | Gemma2 SNR | Stato |
|---------|-----------|-----------|-------|
| sonnolenza_vs_veglia | +6.56 L19 | +4.14 L38 | ✅ pronto steering |
| desiderio_vs_urgenza | +2.17 L20 | +1.13 L38 | ✅ usabile |
| sicurezza_vs_minaccia | +0.83 L23 | +1.26 L38 | 🟡 debole |
| calore_sensuale | +0.12 | +0.69 L38 | 🟡 solo Gemma2 |
| passione_vs_torrida | -0.73 | +0.59 L33 | ❌ riformulato |
| tenerezza_vs_desiderio | -0.35 | +0.10 | ❌ riformulato |
| urgenza_vs_inerzia | -0.42 | +0.01 | ❌ overlap |
| indifferenza_vs_interesse | -1.65 | -0.12 | ❌ riformulato |

**4 nuovi concept v2 generati da Gemma2, pronti per estrazione:**
- `frigidita_vs_torrida` (57+51 frasi)
- `urgenza_affettiva_vs_assenza` (57+49 frasi)
- `tenerezza_vs_desiderio_v2` (53+48 frasi)
- `calma_affettiva_vs_passione` (59+53 frasi)

---

## Batch

Nessun batch in corso. Server spenti per la notte.

---

## Prossima sessione — checklist

```
1. Leggi STATO.md + cantagallo
2. Avvia server (aspetta il sole)
3. Leggi RIPRESA_20260315.md
4. Estrai i 4 nuovi concept v2 su Gemma2-Uncensored:
   bash /home/lele/ricerche_riservate/run_extraction_v2.sh
5. Confronta SNR con i vecchi — verifica ipotesi poli opposti
6. Steering sonnolenza_vs_veglia
```

---

## Avvio rapido server

```bash
cd /home/lele/codex-openai
nohup project_jedi/.venv/bin/python project_jedi/scripts/steering_server.py > /tmp/steering_server.log 2>&1 &
/mnt/raid0/llama-cpp-m40/start_cuda.sh
```
