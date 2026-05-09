# STATO — Project Jedi

> Questo file va letto SUBITO all'inizio di ogni sessione Claude.
> Ultima modifica: 2026-05-10 — sessione 2026-05-09/10

---

## Server

| Servizio | Porta | Stato |
|----------|-------|-------|
| MI50 manager | 8020 | ✅ UP — Gemma4-E4B-IT caricato (15.3 GB) |
| Steering server | 8010 | ✅ UP |
| M40 llama-server | 11435 | ✅ UP — Gemma4-E4B-IT Q4_K_M |

```bash
curl -s http://localhost:8020/api/status
curl -s http://localhost:8010/api/models
curl -s http://localhost:11435/health
```

---

## Modelli disponibili

| Nome | Path | Layer | Hidden | Stato |
|------|------|-------|--------|-------|
| Gemma2-Uncensored | /mnt/raid0/gemma-2-uncensored | 42 | 3584 | ✅ vettori completi Gd0+Gd1 |
| Gemma3-4B-IT | /mnt/raid0/gemma-3-4b-it | 34 | 2560 | 🟡 nessuna estrazione |
| Gemma4-E4B-IT | /mnt/raid0/gemma-4-E4B-it | 42 | 2560 | 🔄 minimal pairs in corso |
| ~~Gemma3-1B-IT~~ | rimosso dal disco | 26 | 1152 | 📦 vettori conservati |

---

## Libreria Vettori

| Livello | Gemma2-Uncensored | Gemma4-E4B-IT |
|---------|------------------|---------------|
| Gd0 (500+500 frasi) | ✅ 9/9 L29-38 | ✅ 9/9 — instabili (boot_min≈-1, cos≈0.9999) |
| Gd1 | ✅ 9/9 ~800 file | ✅ 9/9 — instabili per stesso motivo |
| **Minimal pairs** (50 cop.) | — | 🔄 in estrazione (PID 622983) |

---

## Batch attivo

```
probe_minimal_pairs_gemma4.py — PID 622983
Log: /tmp/probe_minimal_pairs.log
Output: output/vector_library_minimal/
Stima: ~50 minuti totali (9 concept × ~5 min)
Avviato: 2026-05-10 ~00:30
```

---

## Finding chiave sessione 2026-05-09/10

### Problema Gemma4 con dataset generico (500+500 frasi)
- `cos(µ+,µ-)≈0.9999` su tutti i layer → vettore mean_diff ≈ rumore → boot_min≈-1
- `coherence≈0.003` → i 500 vettori differenza non concordano sulla direzione
- Stesso problema con last-token pooling e con range L35-L41

### Soluzione: coppie minimali (50 coppie per concept)
- Stessa frase, SOLO la parola chiave cambia ("The water is hot." vs "The water is cold.")
- **hot_vs_cold test**: coherence=+0.66, SNR=+9.17 @ L33 — tutti 1225/1225 coseni positivi!
- Vs dataset generico: coherence era 0.003, SNR 2.89

### Steering test Gemma4 (vettori minimali hot_vs_cold L33)
- Gain utile: 50-80 (strettissimo vs Gemma2 200-800)
- COLD g80: produce "cold" spontaneamente su prompt neutro ✓
- HOT g120+: collasso immediato
- Template Gemma4 fixato: `<start_of_turn>user\n...<end_of_turn>` in steering_server.py

### Fix pipeline
- `decompose.py`: split `--steering-url` (8010) / `--manager-url` (8020)
- `concept_expander.py` + `sub_concept_eval.py`: JSON parse robusto + max_tokens aumentati
- `probe_concept.py`: `--deep-range`, `--token-position`, `--vector-method` CLI
- `steering_server.py`: template Gemma3/4 corretto

---

## Prossima sessione — checklist

```
1. Leggi STATO.md + cantagallo_pending.txt
2. Verifica server: 8020 + 8010 + 11435
3. Controlla risultati probe minimal pairs:
   tail /tmp/probe_minimal_pairs.log
   ls output/vector_library_minimal/
4. Se probe completato → testare steering tutti 9 concept
   → trovare sweet spot gain per ciascun concept
   → script: probe_minimal_pairs_gemma4.py (già fatto)
   → steering via vector_path=<abs>/layer_NN.npy in /api/generate
5. Se steering funziona → creare minimal pairs per Gd1 sub-concept
```

---

## Avvio rapido server (se down)

```bash
# MI50:
echo 'pippopippo33$$' | sudo -S systemctl restart mi50-manager
sleep 70
echo 'pippopippo33$$' | sudo -S systemctl restart steering-server

# M40:
bash /mnt/raid0/llama-cpp-m40/start_cuda.sh
```
