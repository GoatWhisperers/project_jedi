# STATO — Project Jedi

> Questo file va letto SUBITO all'inizio di ogni sessione Claude.
> Ultima modifica: 2026-03-06 08:45 — fix permanenti + avvio batch Gd1 Gemma2

---

## Server

| Servizio | Porta | Stato |
|----------|-------|-------|
| Steering server MI50 | 8010 | systemd managed — autostart al boot |
| M40 llama-server CUDA | 11435 | systemd managed — autostart al boot |

```bash
systemctl is-active llama-server-m40 steering-server
curl -s http://localhost:8010/api/models
curl -s http://localhost:11435/health
```

**NOTA**: i server ora sono gestiti da systemd. Non servono più avvii manuali.
Se down: `echo 'pippopippo33$$' | sudo -S systemctl restart <nome>.service`

---

## Fix permanenti applicati (2026-03-06)

- `llama-server-m40.service`: ora usa build_cuda + 12B + --n-gpu-layers 99
- `steering-server.service`: autostart MI50 steering_server.py
- OOM fixes: GC+sync corretto in load_model(), torch_dtype fix, VRAM check in probe
- GPU verification: check_m40_on_gpu() blocca decompose se M40 è su CPU
- Dead code: eval_hot_cold.py + build_catalog.py → scripts/dead_code/

---

## Libreria Vettori

| Livello | Gemma3-1B-IT | Gemma2-Uncensored |
|---------|-------------|------------------|
| Gd0 (broad) | 9/9 ✅ | 9/9 ✅ |
| Gd1 (sub) | 9/9 ✅ completo | 1/9 parziale (solo hot_vs_cold) |

---

## Batch

**IN CORSO**: `run_decompose_gd1_gemma2_ripresa.sh` — Gd1 Gemma2 (9 concept)

```bash
tail -f /tmp/gemma2_ripresa4.log
# oppure per concept singolo:
ls /tmp/decompose_gd1_gemma2_*/
```

---

## Prossima sessione — checklist

```
1. Leggi questo file (STATO.md)
2. cat /tmp/cantagallo_pending.txt
3. systemctl is-active llama-server-m40 steering-server
4. Verifica batch: tail /tmp/gemma2_ripresa4.log
5. Se Gd1 Gemma2 completo: scrivere experiments/07_gemma2_decompose_gd1.md
6. Poi: avviare ricerche riservate
```
