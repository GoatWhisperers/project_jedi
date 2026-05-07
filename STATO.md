# STATO — Project Jedi

> Questo file va letto SUBITO all'inizio di ogni sessione Claude.
> Ultima modifica: 2026-05-07 22:47 — batch Gemma4 Gd0 in corso

---

## Server (aggiornato 2026-05-07)

| Servizio | Porta | Stato |
|----------|-------|-------|
| MI50 manager | 8020 | ✅ attivo — Gemma4-E4B-IT caricato (15.3 GB) |
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
| Gemma4-E4B-IT | /mnt/raid0/gemma-4-E4B-it | 42 | 2560 | 🔄 Gd0 in estrazione |
| ~~Gemma3-1B-IT~~ | rimosso dal disco | 26 | 1152 | 📦 vettori conservati |

Note architettura Gemma4:
- 35 layer `sliding_attention` + 7 layer `full_attention` (ogni 6: L5,L11,L17,L23,L29,L35,L41)
- layer path: `model.language_model.layers`
- dtype: float16

---

## Libreria Vettori

| Livello | Gemma3-1B-IT | Gemma2-Uncensored | Gemma4-E4B-IT |
|---------|-------------|------------------|---------------|
| Gd0 (broad) | 📦 ~120 file | ✅ 9/9 L29-38 | 🔄 3/9 estratti (hot_vs_cold, luce_vs_buio, duro_vs_morbido) |
| Gd1 (sub) | 📦 ~600 file | ✅ 9/9 ~800 file | 🔴 da fare |

---

## Batch attivo

```
run_probe_gemma4.sh — Gd0 tutti i 9 concept su Gemma4-E4B-IT
Log: /tmp/probe_gemma4_batch.log
PID: 409407

[22:06] hot_vs_cold      ✅
[22:19] luce_vs_buio     ✅
[22:33] duro_vs_morbido  ✅
[22:47] liscio_vs_ruvido 🔄 in corso...
        rumore_vs_silenzio
        secco_vs_umido
        calma_vs_allerta
        dolce_vs_amaro
        odore_forte_vs_inodore

Fine stimata: ~00:30
```

---

## Prime osservazioni Gemma4-E4B-IT (da hot_vs_cold)

- `cos(µ+,µ-)` ≈ 0.9998 in tutti i layer deep → separazione media quasi zero
- `boot_min` ≈ -1 ovunque → vettori instabili col metodo attuale (mean pooling)
- **MA** held-out SNR positivo: best **L35 +2.912** (full_attention layer)
- Ipotesi: i full_attention layer (L29, L35, L41) sono i più utili
- Da testare dopo il batch: `token_position=last` e range ristretto ai soli full_attention

---

## Prossima sessione — checklist

```
1. Leggi STATO.md + cantagallo_pending.txt
2. Verifica server 8020 + 8010 + 11435
3. Controlla se batch completato: cat /tmp/probe_gemma4_batch.log
4. Analizza risultati 9 concept: confronta SNR per layer, identifica pattern full vs sliding
5. Aggiorna catalog: python scripts/build_catalog_multi.py
6. Commit vettori Gemma4 + results
7. Prossimi step:
   - Testare token_position=last su Gemma4
   - Estrarre Gd0 su Gemma3-4B-IT
   - Concept affettivi riservati mancanti su Gemma2-Uncensored
   - Test steering su Gemma4 (gain stimato 200-600, hidden=2560)
```

---

## Finding chiave sessione 2026-05-07

**Architettura mi50_manager aggiornata** per Gemma3/4ForConditionalGeneration:
- transformers 4.56 → 5.5.0
- layer path: `model.language_model.layers` (AutoModel da text_config → TextModel direttamente)
- `num_layers` da `text_config.num_hidden_layers` per modelli multimodali
- estrazione hidden states via forward hooks (non `output_hidden_states=True`)
- dtype float16 per-modello in settings.json

---

## Avvio rapido server

```bash
echo 'pippopippo33$$' | sudo -S systemctl restart mi50-manager
sleep 70
echo 'pippopippo33$$' | sudo -S systemctl restart steering-server
```
- usa la memoria condivisa in /mnt/raid0/memoria_ai
