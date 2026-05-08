# STATO — Project Jedi

> Questo file va letto SUBITO all'inizio di ogni sessione Claude.
> Ultima modifica: 2026-05-08 23:18 — sessione 2026-05-08

---

## Server

| Servizio | Porta | Stato |
|----------|-------|-------|
| MI50 manager | 8020 | systemd — Gemma4-E4B-IT caricato (15 GB) |
| Steering server | 8010 | systemd |
| M40 llama-server | 11435 | ✅ UP — Gemma4-E4B-IT Q4_K_M (nuovo binario compilato oggi) |

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
| Gemma4-E4B-IT | /mnt/raid0/gemma-4-E4B-it | 42 | 2560 | ✅ Gd0 9/9 completo |
| ~~Gemma3-1B-IT~~ | rimosso dal disco | 26 | 1152 | 📦 vettori conservati |

---

## Libreria Vettori

| Livello | Gemma2-Uncensored | Gemma4-E4B-IT | Gemma3-4B-IT |
|---------|------------------|---------------|--------------|
| Gd0 | ✅ 9/9 L29-38 | ✅ 9/9 L29-38 | 🔴 da fare |
| Gd1 | ✅ 9/9 ~800 file | 🔴 da fare | 🔴 da fare |

---

## Batch

Nessun batch attivo.

Ultimo batch completato oggi: `run_probe_gemma4.sh` — 9/9 OK, pushato su GitHub (commit 5b8b986).

---

## M40 — llama.cpp ricompilato per Gemma4 ✅

Binario ricompilato il 2026-05-08 con Docker nvidia/cuda:11.8.0-devel-ubuntu22.04.
Flag: `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=52 -DGGML_CUDA_NO_VMM=ON -DGGML_CUDA_GRAPHS=OFF -DGGML_CUDA_NCCL=OFF`
Modello: `/mnt/raid0/models-gguf/gemma-4-e4b-it-Q4_K_M.gguf` (5 GB)

Per riavviare M40: `bash /mnt/raid0/llama-cpp-m40/start_cuda.sh`

---

## Prossima sessione — checklist

```
1. Leggi STATO.md + cantagallo_pending.txt
2. Verifica server: 8020 + 8010 + 11435
3. AZIONE IMMEDIATA: verificare se build llama.cpp è completo
   → ls -lh /mnt/raid0/llama-cpp-m40/build_cuda/bin/llama-server (data odierna?)
   → ldd ... | grep nccl  (non deve esserci)
   → Se OK: bash /mnt/raid0/llama-cpp-m40/start_cuda.sh
   → Se non OK: rilanciare build Docker (vedi sopra)
4. Dopo M40 up: avviare Gd0 su Gemma3-4B-IT
   → bash scripts/run_probe_gemma3.sh  (da creare)
5. Analisi SNR Gemma4: verificare se full_attention (L29,L35,L41) > sliding
6. Poi: Gd1 su Gemma4 (richiede M40 funzionante)
```

---

## Finding chiave sessione 2026-05-08

- **Gd0 Gemma4-E4B-IT completo** — 9/9 concept estratti L29-38, pushati
- **M40 llama-server**: binario vecchio non supporta Gemma4 (`unknown model architecture: 'gemma4'`)
- **Build llama.cpp**: ricompilazione Docker con flag NO_VMM + NO_GRAPHS + NO_NCCL
  - Problema 1: GCC 14 incompatibile con CUDA 11.8 → Docker Ubuntu 22.04
  - Problema 2: cuMemMap undefined → GGML_CUDA_NO_VMM=ON
  - Problema 3: libnccl.so.2 not found → GGML_CUDA_NCCL=OFF (flag corretto, non GGML_NCCL)
- **Gemma3-12B GGUF rimosso** — sostituito da gemma-4-e4b-it-Q4_K_M.gguf (5 GB)

---

## Avvio rapido server (se down)

```bash
# MI50 (se manager down):
echo 'pippopippo33$$' | sudo -S systemctl restart mi50-manager
sleep 70
echo 'pippopippo33$$' | sudo -S systemctl restart steering-server

# M40:
bash /mnt/raid0/llama-cpp-m40/start_cuda.sh
```
