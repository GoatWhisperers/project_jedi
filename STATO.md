# STATO — Project Jedi

> Questo file va letto SUBITO all'inizio di ogni sessione Claude.
> Ultima modifica: 2026-05-07 23:00 — fine sessione

---

## Server

| Servizio | Porta | Stato |
|----------|-------|-------|
| MI50 manager | 8020 | systemd (Gemma4-E4B-IT caricato) |
| Steering server | 8010 | systemd |
| M40 llama-server | 11435 | systemd (Gemma3-12B GGUF) |

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
| Gemma4-E4B-IT | /mnt/raid0/gemma-4-E4B-it | 42 | 2560 | 🟡 Gd0 parziale (4/9) |
| ~~Gemma3-1B-IT~~ | rimosso dal disco | 26 | 1152 | 📦 vettori conservati |

---

## Libreria Vettori

| Livello | Gemma2-Uncensored | Gemma4-E4B-IT |
|---------|------------------|---------------|
| Gd0 | ✅ 9/9 | 🟡 4/9 estratti |
| Gd1 | ✅ 9/9 ~800 file | 🔴 da fare |

**Gemma4 Gd0 estratti** (L29-38, hidden=2560):
- ✅ hot_vs_cold
- ✅ luce_vs_buio
- ✅ duro_vs_morbido
- ✅ liscio_vs_ruvido
- 🔴 rumore_vs_silenzio
- 🔴 secco_vs_umido
- 🔴 calma_vs_allerta
- 🔴 dolce_vs_amaro
- 🔴 odore_forte_vs_inodore

---

## Batch

Nessun batch attivo. Ultimo batch interrotto per batteria (dati salvati e pushati).

---

## Prossima sessione — checklist

```
1. Leggi STATO.md + cantagallo_pending.txt
2. Verifica server: 8020 + 8010 + 11435
3. AZIONE IMMEDIATA: completare Gd0 Gemma4 (5 concept mancanti)
   → bash scripts/run_probe_gemma4.sh
   NB: i 4 già estratti vengono rieseguiti (nessun problema, sovrascrive)
4. Dopo il batch: analisi SNR per layer → capire se full_attention (L29,L35,L41)
   sono sistematicamente migliori
5. Aggiornare catalog: python scripts/build_catalog_multi.py
6. Commit vettori completi
7. Poi: Gd0 su Gemma3-4B-IT
8. Poi: concept affettivi riservati mancanti su Gemma2-Uncensored
```

---

## Finding chiave sessione 2026-05-07

**Infrastruttura aggiornata:**
- transformers 4.56 → 5.5.0
- mi50_manager: path layer `model.language_model.layers` per Gemma3/4
- mi50_manager: num_layers da text_config, dtype float16 per-modello, forward hooks
- settings.json: Gemma3-4B-IT e Gemma4-E4B-IT aggiunti

**Gemma4 geometria (preliminare, da hot_vs_cold):**
- cos(µ+,µ-) ≈ 0.9998 → separazione media quasi zero (diverso da Gemma2)
- boot_min ≈ -1 ovunque → vettori instabili con mean pooling
- held-out SNR positivo: best **L35 +2.912** (full_attention)
- Ipotesi da verificare: full_attention layers (L29, L35, L41) > sliding_attention

---

## Avvio rapido server (se down)

```bash
echo 'pippopippo33$$' | sudo -S systemctl restart mi50-manager
sleep 70
echo 'pippopippo33$$' | sudo -S systemctl restart steering-server
```
- usa la memoria condivisa in /mnt/raid0/memoria_ai
